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

"""Multi-card regression test for ZCC EMA resume across a 2D shard layout change.

FC-format EMA resume is gated on ``Trainer.ema_weight_reshard`` (reuses paddle's
``check_resumable_locally``): it reshards only when a rank's shard no longer lines
up with the checkpoint. A rank-rotated ownership map (same tensors and degrees,
different ``global_offset``) stands in for a real layout change.

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

# Master weights (".w_0") come from the optimizer dict, fp32 params from the
# model's; OPT_MOMENT/bf16 are decoys that must be filtered out by those rules.
MASTER_KEY = "ema_linear_0.w_0"
MODEL_KEY = "ema_layer.weight"
OPT_MOMENT_KEY = "ema_linear_0.moment1_0"
MODEL_BF16_KEY = "ema_layer.bf16_weight"

# The worker re-keys master weights by reversing ``unified_name_mapping``, so the
# original name must differ from the unified one.
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
    """Model/optimizer stand-in; the factory returns fresh tensors each call
    because ``_load_flex_checkpoint`` asks more than once."""

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
    trainer._build_ema_target = Trainer._build_ema_target.__get__(trainer)
    return trainer


def _read_shared_memory(reshard_result):
    """Reopen the shm handles the way the ZCC worker does."""
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
    """Write the four subdirs a resume reads. model/optimizer/master stay
    identity (they must match the identity load targets); only ema_state uses
    ``ema_layout``, so a "rotated" EMA trips ``ema_weight_reshard``.
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
    """Stub carrying the real ``_load_flex_checkpoint``, with args set to the
    shortest path to the reshard block. ``init_optimizer`` is patched by the caller.
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
    # _load_flex_checkpoint gates on ema_weight_reshard; both it and the reshard
    # build their target via _build_ema_target -- bind all three.
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
        # fleet.init (not init_parallel_env) so _load_flex_checkpoint can fetch a
        # hybrid group; sharding-only keeps one block per rank.
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
        """Save EMA under ``save_layout`` and reshard it to identity; returns the
        stub too (it holds the refs keeping the shm alive)."""
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
        """Smallest UPDATE a worker accepts before consuming EMA: only the fused
        master-weight buffer and one fp32 param buffer are real."""
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
        # ema_accumulate reads training_args.zcc_ema_loss_threshold; the high
        # PREPARE loss keeps it on the skip branch (no real accumulation).
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
            # UPDATE only arms pending_ema_rebuild; the consume happens in
            # _maybe_prepare_ema on the offload below. Wait for the version first.
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

            # LOAD_EMA only stashes; the consume runs in _maybe_prepare_ema on the
            # offload. Drive PREPARE (high-loss state, null save dirs) + OFFLOAD.
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
            # set() runs only after the consume returns, so this also catches a
            # crash mid-consume.
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
        """Drive the real _load_flex_checkpoint: reshard only when the layout
        moved (identity -> _ema_reshard_result None; rotated -> set), and a
        reshard must land each rank's own block."""
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
                # A reshard must also land each rank's own block.
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
        """Cover two paths the mocked-worker single-card units never reach,
        driven directly on bypass-constructed instances (Path A is covered
        above)."""
        # _release_ema_shm body: only after a reshard resume, once workers consume.
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

        # _maybe_prepare_ema Path B: no reshard -> (re)build processor, load EMA
        # from file.
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
