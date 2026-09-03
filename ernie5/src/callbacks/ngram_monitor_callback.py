#!/usr/bin/env python3
"""Trainer callback driving the N-gram embedding usage / collision monitor.

Responsibilities:

* tell the monitor which global step is running (this is what makes the analysis
  cadence identical on every rank, which in turn keeps the collectives inside
  ``analyze()`` in lockstep);
* restore the cumulative counters when the job restarts from a checkpoint
  (``bash longjob/restart.sh <step>``), and warn loudly when it cannot, so that a
  cold start is never silently mistaken for a continued history;
* flush and persist the counters into every checkpoint directory, which gives
  rollback-to-an-older-step the correct semantics for free: the state file lives
  inside ``checkpoint-<step>/`` and is deleted together with it.
"""

import json
import os

import numpy as np
import paddle.distributed as dist
from paddleformers.trainer.trainer import PREFIX_CHECKPOINT_DIR
from paddleformers.trainer.trainer_callback import TrainerCallback
from paddleformers.utils.log import logger

__all__ = ["NgramMonitorCallback"]

STATE_FILE_NAME = "ngram_monitor_state.npz"
METRIC_PREFIX = "ngram/"


def find_ngram_monitor(model):
    """Locate the monitor instance hanging off the NgramEmbedding sub-layer."""
    if model is None:
        return None
    candidates = [model]
    if hasattr(model, "named_sublayers"):
        candidates += [layer for _, layer in model.named_sublayers()]
    for layer in candidates:
        monitor = getattr(layer, "monitor", None)
        if monitor is not None and hasattr(monitor, "begin_step"):
            return monitor
    return None


class NgramMonitorCallback(TrainerCallback):
    """Drives NgramUsageMonitor: step clock, analysis trigger, persistence."""

    def __init__(self, save_state=True, save_full_state=True, jsonl=True):
        self.monitor = None
        self.save_state = save_state
        self.save_full_state = save_full_state
        self.jsonl = jsonl
        self.output_dir = None
        self._pending = {}

    # ------------------------------------------------------------------ helpers
    def _state_path(self, step):
        return os.path.join(
            self.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{step}", STATE_FILE_NAME
        )

    @staticmethod
    def _is_writer():
        return not dist.is_initialized() or dist.get_rank() == 0

    def _write_jsonl(self, metrics, step):
        if not (self.jsonl and self._is_writer()):
            return
        path = os.path.join(self.output_dir, "ngram_monitor", "metrics.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        record = {"step": int(step)}
        record.update({k: round(float(v), 8) for k, v in metrics.items()})
        with open(path, "a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    # ------------------------------------------------------------------- hooks
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.monitor = find_ngram_monitor(model)
        if self.monitor is None:
            logger.warning(
                "[ngram_monitor] enabled, but no NgramEmbedding monitor was found "
                "on this rank; monitoring is inactive."
            )
            return
        self.output_dir = args.output_dir
        if self.save_state and not self.save_full_state:
            logger.warning(
                "[ngram_monitor] save_full_state=False: the per-row hit counts "
                "will not be checkpointed, so cumulative metrics cannot survive a "
                "restart and a resumed run will start them from zero."
            )
        step = int(getattr(state, "global_step", 0) or 0)
        if step <= 0:
            logger.info("[ngram_monitor] cold start, counters begin at zero.")
            return
        path = self._state_path(step)
        if not os.path.isfile(path):
            logger.warning(
                f"[ngram_monitor] resuming at step {step} but {path} is missing; "
                "cumulative counters restart from zero (windowed metrics stay valid)."
            )
            return
        with np.load(path) as f:
            restored = self.monitor.load_saved_state(f)
        if not restored:
            logger.warning(
                f"[ngram_monitor] {path} is not compatible with the current table "
                "layout (or carries no per-row counts); counters restart from zero."
            )
            return
        logger.info(
            f"[ngram_monitor] restored counters from {path} "
            f"(window_id={self.monitor.window_id})."
        )
        # Collective: catches a non-shared output_dir, which would otherwise
        # leave every rank continuing from a different history.
        if not self.monitor.verify_loaded_state():
            logger.warning(
                "[ngram_monitor] restored state differs across ranks; is "
                f"{self.output_dir} shared storage? Cumulative metrics are "
                "unreliable for this run."
            )

    def on_step_begin(self, args, state, control, **kwargs):
        if self.monitor is not None:
            self.monitor.begin_step(state.global_step)

    def on_step_end(self, args, state, control, **kwargs):
        if self.monitor is None:
            return
        metrics = self.monitor.analyze()
        if not metrics:
            return
        self._pending.update(metrics)
        self._write_jsonl(metrics, state.global_step)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or not self._pending:
            return
        for key, value in self._pending.items():
            logs[METRIC_PREFIX + key] = value
        self._pending = {}

    def on_save(self, args, state, control, **kwargs):
        if self.monitor is None or not self.save_state:
            return
        # Collective: every rank must take part before rank 0 writes the file.
        self.monitor.sync_pending()
        if not self._is_writer():
            return
        path = self._state_path(state.global_step)
        directory = os.path.dirname(path)
        if not os.path.isdir(directory):
            logger.warning(
                f"[ngram_monitor] {directory} does not exist, skipping state save."
            )
            return
        state_dict = self.monitor.state_dict_for_save(full=self.save_full_state)
        with open(path, "wb") as f:
            np.savez_compressed(f, **state_dict)
        logger.info(f"[ngram_monitor] wrote {path}")
