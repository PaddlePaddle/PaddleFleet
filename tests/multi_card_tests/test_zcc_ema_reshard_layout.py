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
Multi-card regression test for ZCC EMA resume after a 2D shard layout change.

Resume used to be gated on ``check_same_strategy``, which compares nothing but
the pp/mp/sharding/ep/moe_sharding degrees in ``model_meta.json``.
``machine_balanced_2d_partition`` moves which rank owns which 2D param without
touching any of those, so the gate reported "same strategy" and the ZCC
subprocess read shards laid out for the previous owner map. FC-format EMA is
therefore now always routed through ``Trainer._load_ema_with_reshard``.

Three things are pinned:
  * resharding lands the right values whether or not the layout moved;
  * a real ZCC worker consumes the resulting shared memory after its first
    UPDATE, without hanging;
  * ``Trainer._load_flex_checkpoint`` actually routes an FC-format EMA resume
    through the reshard (the call site this PR changed, trainer.py:1677-1684) --
    the other two exercise ``_load_ema_with_reshard`` directly and so never
    reach that call site.

A rank-rotated ownership map stands in for the flag -- same global tensors, same
degrees, different ``global_offset``, which is what the change looks like to
``dist.load_state_dict`` -- so the test also runs on paddle wheels that predate
the flag.

Run with:
  python -m paddle.distributed.launch --gpus 0,1,2,3 \
      tests/multi_card_tests/test_zcc_ema_reshard_layout.py
"""

import multiprocessing
import os
import shutil
import sys
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

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
    return SimpleNamespace(
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


def _build_flex_checkpoint(root):
    """Write the four flex-checkpoint subdirs a resume reads (identity layout).

    optimizer_state is only scanned for its ``.metadata`` (opt load is skipped
    via ignore_load_lr_and_optim); model_state, master_weight and ema_state are
    really loaded and must match their load targets key-for-key -- including the
    bf16 decoy in the model state.
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
                MASTER_KEY, "identity", fill_reference=True
            ),
            MODEL_KEY: _sharded_weight(
                MODEL_KEY, "identity", fill_reference=True
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
    return harness


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

    def test_reshard_lands_the_right_shard(self):
        """Layout moved or not, a rank must end up with its own block."""
        for layout in ("rotated", "identity"):
            trainer, result = self._reshard_from(layout)
            loaded = _read_shared_memory(result)
            for key in (MASTER_KEY, MODEL_KEY):
                np.testing.assert_array_equal(
                    loaded[key],
                    _global_reference(key)[_rows_of(self.rank), :],
                    err_msg=(
                        f"rank {self.rank} got the wrong EMA shard for {key} "
                        f"from a {layout} checkpoint"
                    ),
                )

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
        return dynamic, static

    def test_worker_consumes_shared_memory_after_first_update(self):
        """Real worker, real handoff: consumed after the first UPDATE, no hang."""
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
            # The EMA processor is built after the first UPDATE, which is why the
            # manager defers LOAD_EMA until the version has landed.
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

            queue.put((ZCCTaskType.LOAD_EMA_FROM_SHARED_MEM, reshard_result))
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

    def test_resume_routes_fc_ema_through_reshard(self):
        """Drive the real _load_flex_checkpoint: FC-format EMA resume must reshard.

        This is the call site the PR changed (trainer.py:1677-1684). On the
        parent commit a same-degree resume took the file-read path and left
        ``_ema_reshard_result`` None; the fix reshards unconditionally.
        """
        ckpt = os.path.join(self.tmp_root, "resume")
        if self.rank == 0:
            shutil.rmtree(ckpt, ignore_errors=True)
        dist.barrier()
        _build_flex_checkpoint(ckpt)
        dist.barrier()

        harness = _flex_harness()
        # init_optimizer would need a real sharding optimizer to build
        # accumulators; the reshard block under test does not depend on it.
        with patch("paddlefleet.trainer.trainer.init_optimizer"):
            harness._load_flex_checkpoint(ckpt)

        self.assertIsNotNone(
            harness._ema_reshard_result,
            "FC-format EMA resume did not go through _load_ema_with_reshard",
        )
        self.assertEqual(
            sorted(harness._ema_reshard_result.keys()),
            sorted([MASTER_KEY, MODEL_KEY]),
        )
        dist.barrier()

    @classmethod
    def tearDownClass(cls):
        dist.barrier()
        if cls.rank == 0:
            shutil.rmtree(cls.tmp_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
