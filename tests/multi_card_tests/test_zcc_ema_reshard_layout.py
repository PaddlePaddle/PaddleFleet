# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Multi-card regression test for ZCC EMA resume across a 2D shard layout change.

Resume used to be gated on ``check_same_strategy``, which compares nothing but
the pp/mp/sharding/ep/moe_sharding degrees in ``model_meta.json``.
``machine_balanced_2d_partition`` moves which rank owns which 2D param without
touching any of those, so the gate reported "same strategy" and the ZCC
subprocess read shards laid out for the previous owner map.

FC-format EMA resume is now gated on ``Trainer.ema_weight_reshard``, which
reuses paddle's ``check_resumable_locally`` -- the same per-rank shard-alignment
check ``dist.load_state_dict`` runs for its own fast path. The EMA reshard fires
only when a rank's shard does not line up with the checkpoint, so an unchanged
layout skips the ~30-40s collective entirely and the subprocess reads its shard
straight from file.

Pinned here:
  * ``Trainer._load_flex_checkpoint`` reshards an FC-format EMA resume only when
    the layout moved (the call site this PR changed, trainer.py:1677-1699) -- an
    identity resume leaves ``_ema_reshard_result`` None, a rotated one sets it --
    and when it does reshard, each rank still ends up with its own block;
  * a real ZCC worker consumes the resulting shared memory -- which now happens
    in ``_maybe_prepare_ema`` on the offload after the first UPDATE, not in the
    LOAD_EMA handler -- without hanging;
  * the two paths the mocked-worker single-card units never reach are covered
    directly on bypass-constructed instances: ``_release_ema_shm`` (release once
    all workers consume) and ``_maybe_prepare_ema`` Path B (subprocess loads EMA
    from file when no reshard happened).

A rank-rotated ownership map stands in for the flag -- same global tensors, same
degrees, different ``global_offset``, which is what the change looks like to
``check_resumable_locally`` -- so the test also runs on paddle wheels that
predate the flag.

Run with:
  python -m paddle.distributed.launch --gpus 0,1,2,3 \
      tests/multi_card_tests/test_zcc_ema_reshard_layout.py
"""

import multiprocessing
import os
import shutil
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
)

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.flex_checkpoint.dcp.sharded_weight import ShardedWeight

from paddlefleet.trainer.trainer import Trainer
from paddlefleet.trainer.utils.zero_cost_checkpoint import (
    ZCCTaskType,
    ZCCWorkerStatus,
    ZeroCostCheckpointManager,
    ZeroCostCheckpointWorker,
    ZeroCostCheckpointWorkerFcBased,
    worker_loop,
)
from paddlefleet.utils.env import (
    EMA_STATE_DIC,
    MASTER_WEIGHT_DIC,
    MODEL_STATE_DIC,
    OPTIMIZER_STATE_DIC,
)

ROWS_PER_RANK = 4
COLS = 8
WORKER_TIMEOUT = 180

# ``_load_ema_with_reshard`` takes master weights (".w_0") from the optimizer
# state dict and fp32 params from the model's, so both portions are covered. The
# two decoys must be filtered out by those same rules -- if they leak into the
# load target, dist.load_state_dict fails on a key the checkpoint lacks.
MASTER_KEY = "ema_linear_0.w_0"
MODEL_KEY = "ema_layer.weight"
OPT_MOMENT_KEY = "ema_linear_0.moment1_0"
MODEL_BF16_KEY = "ema_layer.bf16_weight"

# The worker keys master weights by their original name, reached by reversing
# ``unified_name_mapping``, so the two names must differ for that to mean
# anything.
WORKER_ORIG_MASTER = "layer_0.linear.w_0"


def _global_reference(key):
    """Deterministic global EMA content, distinct per key."""
    rows = ROWS_PER_RANK * dist.get_world_size()
    flat = np.arange(rows * COLS, dtype="float32")
    return (flat.reshape(rows, COLS) + len(key) * 1000.0).astype("float32")


def _owned_block(rank, layout):
    """Which global row block ``rank`` owns: its own, or its neighbour's."""
    if layout == "identity":
        return rank
    if layout == "rotated":
        return (rank + 1) % dist.get_world_size()
    raise ValueError(f"unknown layout: {layout}")


def _rows_of(block):
    return slice(block * ROWS_PER_RANK, (block + 1) * ROWS_PER_RANK)


def _sharded_weight(key, layout, dtype="float32", fill_reference=False):
    """One rank's ShardedWeight for ``key`` under ``layout``."""
    block = _owned_block(dist.get_rank(), layout)
    if fill_reference:
        local = paddle.to_tensor(
            _global_reference(key)[_rows_of(block), :]
        ).astype(dtype)
    else:
        local = paddle.zeros([ROWS_PER_RANK, COLS], dtype=dtype)
    return ShardedWeight(
        key=key,
        local_tensor=local,
        local_shape=(ROWS_PER_RANK, COLS),
        global_shape=(ROWS_PER_RANK * dist.get_world_size(), COLS),
        global_offset=(block * ROWS_PER_RANK, 0),
    )


class _Stub:
    """Stands in for both the model and the optimizer.

    Holds a factory so every ``sharded_state_dict`` call returns fresh tensors:
    ``_load_flex_checkpoint`` asks more than once (load target, then again inside
    the reshard), and shared tensors would let one load's writes leak into
    another's target. The unbound production methods run as-is on this stub, so
    no TrainingArguments/dataloader/ZCC stack is needed.
    """

    def __init__(self, factory):
        self._factory = factory

    def sharded_state_dict(self, *_):
        return self._factory()


def _stub_trainer():
    trainer = SimpleNamespace(
        model=_Stub(
            lambda: {
                MODEL_KEY: _sharded_weight(MODEL_KEY, "identity"),
                MODEL_BF16_KEY: _sharded_weight(
                    MODEL_BF16_KEY, "identity", dtype="bfloat16"
                ),
            }
        ),
        optimizer=_Stub(
            lambda: {
                MASTER_KEY: _sharded_weight(MASTER_KEY, "identity"),
                OPT_MOMENT_KEY: _sharded_weight(OPT_MOMENT_KEY, "identity"),
            }
        ),
        args=SimpleNamespace(aoa_config=None, load_via_cpu=False),
    )
    # _load_ema_with_reshard now assembles its load target through
    # _build_ema_target, so the stub has to carry that method too.
    trainer._build_ema_target = Trainer._build_ema_target.__get__(trainer)
    return trainer


def _read_shared_memory(reshard_result):
    """Reopen the shm handles the way the ZCC worker does.

    ``_load_ema_with_reshard`` clears the GPU tensors and hands the subprocess
    nothing but these metas, so this is the only view of what a worker consumes.
    """
    out = {}
    for key, info in reshard_result.items():
        lod = paddle.base.core.LoDTensor._new_shared_filename(
            info["shared_meta"]
        )
        out[key] = paddle.to_tensor(lod).reshape(info["shape"]).numpy().copy()
    return out


def _ipc(tensor):
    return tensor.value().get_tensor()._share_cuda()


def _save_dir(root, sub, state_dict):
    directory = os.path.join(root, sub)
    os.makedirs(directory, exist_ok=True)
    dist.save_state_dict(state_dict, directory)


def _build_flex_checkpoint(root, ema_layout="identity"):
    """Write the four flex-checkpoint subdirs a resume reads.

    model_state, optimizer_state and master_weight always use the identity
    layout: optimizer_state is only scanned for its ``.metadata`` (opt load is
    skipped via ignore_load_lr_and_optim); model_state and master_weight are
    really loaded by ``_load_flex_checkpoint`` and must match their identity load
    targets key-for-key -- including the bf16 decoy in the model state.

    Only ema_state varies: saving it under ``ema_layout="rotated"`` gives it a
    different ``global_offset`` than the identity EMA target, which is what makes
    ``ema_weight_reshard`` report the shards misaligned and trigger a reshard.
    """
    _save_dir(
        root,
        MODEL_STATE_DIC,
        {
            MODEL_KEY: _sharded_weight(
                MODEL_KEY, "identity", fill_reference=True
            ),
            MODEL_BF16_KEY: _sharded_weight(
                MODEL_BF16_KEY,
                "identity",
                dtype="bfloat16",
                fill_reference=True,
            ),
        },
    )
    _save_dir(
        root,
        OPTIMIZER_STATE_DIC,
        {
            OPT_MOMENT_KEY: _sharded_weight(
                OPT_MOMENT_KEY, "identity", fill_reference=True
            )
        },
    )
    _save_dir(
        root,
        MASTER_WEIGHT_DIC,
        {
            MASTER_KEY: _sharded_weight(
                MASTER_KEY, "identity", fill_reference=True
            )
        },
    )
    _save_dir(
        root,
        EMA_STATE_DIC,
        {
            MASTER_KEY: _sharded_weight(
                MASTER_KEY, ema_layout, fill_reference=True
            ),
            MODEL_KEY: _sharded_weight(
                MODEL_KEY, ema_layout, fill_reference=True
            ),
        },
    )


def _flex_harness():
    """A stub carrying the real ``_load_flex_checkpoint`` and its args knobs.

    The args take the shortest successful path to the reshard block: no HF load,
    no EMA-sourced model, optimizer/scheduler load skipped, bf16 off (so the bf16
    branch short-circuits before touching ``_inner_opt``), ZCC on with an EMA
    coefficient. ``init_optimizer`` is patched out by the caller.
    """
    harness = SimpleNamespace(
        model=_Stub(
            lambda: {
                MODEL_KEY: _sharded_weight(MODEL_KEY, "identity"),
                MODEL_BF16_KEY: _sharded_weight(
                    MODEL_BF16_KEY, "identity", dtype="bfloat16"
                ),
            }
        ),
        optimizer=_Stub(
            lambda: {
                MASTER_KEY: _sharded_weight(MASTER_KEY, "identity"),
                OPT_MOMENT_KEY: _sharded_weight(OPT_MOMENT_KEY, "identity"),
            }
        ),
        args=SimpleNamespace(
            flex_ckpt_comm_method="broadcast",
            load_from_hf=False,
            sharded_model_from_ema=False,
            ignore_load_lr_and_optim=True,
            bf16=False,
            tensorwise_offload_optimizer=False,
            enable_zero_cost_checkpoint=True,
            zcc_save_ema_coef=0.99,
            aoa_config=None,
            load_via_cpu=False,
        ),
    )
    harness._load_flex_checkpoint = Trainer._load_flex_checkpoint.__get__(
        harness
    )
    harness._is_fc_format_ema = Trainer._is_fc_format_ema.__get__(harness)
    harness._load_ema_with_reshard = Trainer._load_ema_with_reshard.__get__(
        harness
    )
    # The reshard is now gated: _load_flex_checkpoint asks ema_weight_reshard
    # first, and both it and _load_ema_with_reshard build their target via
    # _build_ema_target -- so the harness needs all three bound.
    harness.ema_weight_reshard = Trainer.ema_weight_reshard.__get__(harness)
    harness._build_ema_target = Trainer._build_ema_target.__get__(harness)
    return harness


class _Consumed:
    """Minimal stand-in for a worker's ``multiprocessing.Event``."""

    def __init__(self, done):
        self._done = done

    def set(self):
        self._done = True

    def is_set(self):
        return self._done


class TestZCCEMAReshardLayout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # fleet.init (not just init_parallel_env) so _load_flex_checkpoint can
        # fetch a hybrid communicate group; sharding-only keeps global tensors
        # split one-block-per-rank, matching the hand-built shards.
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": 1,
            "sharding_degree": dist.get_world_size(),
        }
        fleet.init(is_collective=True, strategy=strategy)
        cls.rank = dist.get_rank()
        assert dist.get_world_size() > 1, "needs more than one rank"
        master = os.environ.get("PADDLE_MASTER", "local").replace(":", "_")
        cls.tmp_root = os.path.join(
            os.environ.get("TMPDIR", "/tmp"), f"zcc_ema_reshard_{master}"
        )

    def _reshard_from(self, save_layout):
        """Save EMA under ``save_layout``, resume it under the identity layout.

        Returns the stub too: it owns the references keeping the shared memory
        alive, so callers must hold it while reading.
        """
        path = os.path.join(self.tmp_root, save_layout)
        if self.rank == 0:
            shutil.rmtree(path, ignore_errors=True)
            os.makedirs(path, exist_ok=True)
        dist.barrier()
        dist.save_state_dict(
            {
                key: _sharded_weight(key, save_layout, fill_reference=True)
                for key in (MASTER_KEY, MODEL_KEY)
            },
            path,
        )
        dist.barrier()

        trainer = _stub_trainer()
        result = Trainer._load_ema_with_reshard(
            trainer, path, "broadcast", None
        )
        self.assertEqual(sorted(result.keys()), sorted([MASTER_KEY, MODEL_KEY]))
        dist.barrier()
        return trainer, result

    def _update_payload(self):
        """Smallest UPDATE a worker accepts before it can consume EMA.

        Only what the handoff reads is real: the fused master-weight buffer
        (whose offsets drive the re-padding) and one fp32 param buffer (so
        ``ema_buffer_model_params`` is non-empty). The rest belongs to the save
        path -- the point is the UPDATE -> LOAD_EMA boundary, not rebuilding the
        callback.
        """
        numel = ROWS_PER_RANK * COLS
        # Kept on the instance: the child opens these over CUDA IPC.
        self._opt_buffer = paddle.zeros([numel], dtype="float32")
        self._param_buffer = paddle.zeros([numel], dtype="float32")
        meta = {"start": 0, "end": numel, "shape": [ROWS_PER_RANK, COLS]}
        dynamic = dict.fromkeys(
            [
                "optimizer_states_name_path",
                "model_states_name_path",
                "distcp_file_name",
                "model_ckpt_meta",
                "opt_ckpt_meta",
                "master_weight_ckpt_meta",
            ]
        )
        dynamic.update(
            {
                key: {}
                for key in (
                    "model_state_filter",
                    "opt_state_filter",
                    "master_weights_filter",
                    "grouped_gemm_params",
                    "param_slice_info",
                )
            }
        )
        dynamic.update(
            {
                "optimizer_states_meta": (
                    {},
                    {WORKER_ORIG_MASTER: {**meta, "name": WORKER_ORIG_MASTER}},
                    {},
                    _ipc(self._opt_buffer),
                ),
                "model_states_meta": (
                    {
                        MODEL_KEY: {
                            **meta,
                            "buffer_index": "b0",
                            "name": MODEL_KEY,
                        }
                    },
                    {"b0": _ipc(self._param_buffer)},
                ),
                "unified_name_mapping": {WORKER_ORIG_MASTER: MASTER_KEY},
            }
        )
        static = dict.fromkeys(
            ["model_config", "training_args", "model_meta", "user_file"]
        )
        # The offload that drives the consume runs ema_accumulate, which reads
        # training_args.zcc_ema_loss_threshold; the high loss the PREPARE carries
        # keeps it on the skip branch so no real accumulation math is exercised.
        static["training_args"] = SimpleNamespace(zcc_ema_loss_threshold=0.0)
        return dynamic, static

    def test_worker_consumes_shared_memory_after_first_update(self):
        """Real worker, real handoff: consumed on the offload after UPDATE, no hang."""
        trainer, reshard_result = self._reshard_from("rotated")
        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue()
        version = ctx.Value("i", 0)
        consumed = ctx.Event()
        worker = ZeroCostCheckpointWorkerFcBased(
            0,
            int(os.environ.get("FLAGS_selected_gpus", "0")),
            self.rank,
            1,
            queue,
            ctx.Value("i", ZCCWorkerStatus.IDLE.value),
            ctx.Value("i", 0),
            version,
            False,
            0,
            0,
            0,
            self.rank,
            0.99,
            -1,
            consumed,
        )
        process = ctx.Process(target=worker_loop, args=(worker,))
        process.start()
        try:
            # UPDATE only arms pending_ema_rebuild; the EMA processor is actually
            # built -- and the shared memory consumed -- inside _maybe_prepare_ema
            # on the offload below. The manager still defers LOAD_EMA until the
            # UPDATE version has landed, which is why we wait on it here.
            queue.put((ZCCTaskType.UPDATE, [1, *self._update_payload()]))
            deadline = time.time() + WORKER_TIMEOUT
            while (
                version.value != 1
                and process.is_alive()
                and time.time() < deadline
            ):
                time.sleep(0.1)
            self.assertEqual(
                version.value,
                1,
                f"first UPDATE never landed, exitcode={process.exitcode}",
            )

            # LOAD_EMA now only stashes the metas; the consume is deferred to
            # _maybe_prepare_ema, which runs on the last offload chunk after the
            # global_step bump. So drive one full PREPARE + OFFLOAD cycle to reach
            # it. PREPARE carries the trainer_state (a high loss, so ema_accumulate
            # takes its skip branch) and null save dirs (so the dump is a no-op);
            # the single-chunk worker offloads everything in one OFFLOAD.
            queue.put((ZCCTaskType.LOAD_EMA_FROM_SHARED_MEM, reshard_result))
            queue.put(
                (
                    ZCCTaskType.PREPARE,
                    (
                        (None, None),
                        (
                            None,
                            SimpleNamespace(global_step=1, loss=1.0e9),
                            None,
                        ),
                    ),
                )
            )
            queue.put((ZCCTaskType.OFFLOAD, 1))
            # set() runs after _load_ema_from_shared_memory returns, so this also
            # fails if unpacking raised: the worker dies quietly and the
            # manager's own wait() has no timeout.
            self.assertTrue(
                consumed.wait(WORKER_TIMEOUT),
                "EMA shared memory never consumed -- would hang the trainer "
                f"(alive={process.is_alive()}, exitcode={process.exitcode})",
            )

            queue.put((ZCCTaskType.FINISH, None))
            process.join(WORKER_TIMEOUT)
            self.assertEqual(process.exitcode, 0, "worker did not exit cleanly")
        finally:
            if process.is_alive():
                process.terminate()
                process.join(30)
        dist.barrier()

    def test_resume_gates_reshard_and_lands_right_shard(self):
        """Drive the real _load_flex_checkpoint end to end.

        Folds two concerns into one drive of the call site this PR changed
        (trainer.py:1677-1699): the gate (Trainer.ema_weight_reshard, reusing
        paddle's check_resumable_locally) reshards only when a rank's shard moved,
        and when it does the reshard has to land each rank's own block. So an
        identity resume leaves _ema_reshard_result None; a rotated one sets it and
        the resulting shared memory must carry the right values.
        """
        for ema_layout, expect_reshard in (
            ("identity", False),
            ("rotated", True),
        ):
            ckpt = os.path.join(self.tmp_root, f"resume_{ema_layout}")
            if self.rank == 0:
                shutil.rmtree(ckpt, ignore_errors=True)
            dist.barrier()
            _build_flex_checkpoint(ckpt, ema_layout=ema_layout)
            dist.barrier()

            harness = _flex_harness()
            # init_optimizer would need a real sharding optimizer to build
            # accumulators; the reshard block under test does not depend on it.
            with patch("paddlefleet.trainer.trainer.init_optimizer"):
                harness._load_flex_checkpoint(ckpt)

            if expect_reshard:
                self.assertIsNotNone(
                    harness._ema_reshard_result,
                    f"{ema_layout} EMA resume should have resharded but "
                    "_ema_reshard_result is None",
                )
                self.assertEqual(
                    sorted(harness._ema_reshard_result.keys()),
                    sorted([MASTER_KEY, MODEL_KEY]),
                )
                # A reshard must also land each rank's own block -- this is the
                # value check the standalone reshard test used to own.
                loaded = _read_shared_memory(harness._ema_reshard_result)
                for key in (MASTER_KEY, MODEL_KEY):
                    np.testing.assert_array_equal(
                        loaded[key],
                        _global_reference(key)[_rows_of(self.rank), :],
                        err_msg=(
                            f"rank {self.rank} got the wrong resharded EMA "
                            f"shard for {key}"
                        ),
                    )
            else:
                self.assertIsNone(
                    harness._ema_reshard_result,
                    f"{ema_layout} EMA resume should have skipped the reshard "
                    "but _ema_reshard_result is set",
                )
            dist.barrier()

    def test_release_ema_shm_and_maybe_prepare_ema_path_b(self):
        """Cover two paths the mocked-worker single-card units never reach.

        Both run only in states those units never build, so they are driven
        directly on bypass-constructed instances -- no real worker subprocess
        needed. (Path A of _maybe_prepare_ema is already covered by the
        worker-consume test above.)
        """
        # --- ZeroCostCheckpointManager._release_ema_shm body ----------------
        # Runs only after an EMA-reshard resume, once every worker has consumed
        # the shm; drive the still-consuming wait, then the leak and no-leak ends.
        fd, leaked_path = tempfile.mkstemp(prefix="zcc_ema_shm_")
        os.close(fd)
        self.addCleanup(
            lambda: os.path.exists(leaked_path) and os.remove(leaked_path)
        )
        gone_path = leaked_path + "_absent"

        m = ZeroCostCheckpointManager.__new__(ZeroCostCheckpointManager)
        done, pending = _Consumed(True), _Consumed(False)
        m.workers = [
            SimpleNamespace(ema_shm_consumed=done),
            SimpleNamespace(ema_shm_consumed=pending),
        ]
        m._ema_shm_release_pending = True
        m._ema_tensor_refs = {"ema": object()}
        m._ema_shm_filenames = [leaked_path]

        m._release_ema_shm()  # a worker still consuming -> early return
        self.assertTrue(m._ema_shm_release_pending)
        self.assertIsNotNone(m._ema_tensor_refs)

        pending.set()
        m._release_ema_shm()  # all consumed, file still there -> leak branch
        self.assertFalse(m._ema_shm_release_pending)
        self.assertIsNone(m._ema_tensor_refs)
        self.assertEqual(m._ema_shm_filenames, [])

        m._ema_shm_release_pending = True
        m._ema_tensor_refs = {"ema": object()}
        m._ema_shm_filenames = [gone_path]
        m._release_ema_shm()  # files already gone -> no-leak branch
        self.assertFalse(m._ema_shm_release_pending)
        self.assertEqual(m._ema_shm_filenames, [])

        # --- ZeroCostCheckpointWorker._maybe_prepare_ema Path B -------------
        # No reshard: the subprocess (re)builds the processor, then loads EMA
        # straight from file.
        fd, ckpt = tempfile.mkstemp(suffix=".pdparams", prefix="zcc_ema_")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(ckpt) and os.remove(ckpt))
        paddle.save({"master_weights": {}}, ckpt)

        worker = ZeroCostCheckpointWorker.__new__(ZeroCostCheckpointWorker)
        worker.ema_coef = 0.99
        worker.pending_ema_rebuild = True  # exercise the processor (re)build
        worker.pending_ema_shared_metas = None  # skip Path A
        worker.pending_ema_ckpt_path = ckpt  # take Path B
        worker.optimizer_fusion_storage_helper = object()
        worker.param_fusion_storage_helper = object()
        worker.unified_name_mapping = (
            None  # _reverse_unified_name_for_ema no-op
        )
        worker.use_expert_parallel = False
        worker.dp_rank = 0
        worker.zcc_ema_processor = None

        with patch(
            "paddlefleet.trainer.utils.zero_cost_checkpoint."
            "ZeroCostCheckpointEMAProcessor"
        ) as mock_proc_cls:
            mock_proc_cls.return_value = MagicMock()
            worker._maybe_prepare_ema()

        self.assertFalse(worker.pending_ema_rebuild)
        self.assertIsNone(worker.pending_ema_ckpt_path)  # consumed
        worker.zcc_ema_processor.load_ema_state_dict.assert_called_once()
        dist.barrier()

    @classmethod
    def tearDownClass(cls):
        dist.barrier()
        if cls.rank == 0:
            shutil.rmtree(cls.tmp_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
