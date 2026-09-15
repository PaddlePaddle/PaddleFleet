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
Multi-card regression test for ZCC EMA resume when only the 2D shard layout
changed.

The resume path used to be gated on ``DistInfoCollectorValidator.check_same_strategy``,
which compares nothing but the pp/mp/sharding/ep/moe_sharding degrees recorded in
``model_meta.json``. ``machine_balanced_2d_partition`` changes which rank owns which
2D parameter without changing any of those degrees, so the gate reported "same
strategy" and let the ZCC subprocess read its own rank file straight from disk --
shards laid out for the *previous* owner mapping. FC-format EMA is therefore now
always routed through ``Trainer._load_ema_with_reshard``.

What is pinned here is that resharding is layout-invariant: the EMA a rank ends up
with must depend only on the descriptors it asks for, never on how the checkpoint
happened to be split. A rank-rotated ownership map stands in for the layout the
flag produces, because that is exactly what the change looks like to
``dist.load_state_dict`` -- same global tensors, same degrees, different
``global_offset`` per rank. Building it from descriptors instead of from
``machine_balanced_2d_partition`` also keeps the test runnable on paddle wheels
that predate the flag.

Run with:
  python -m paddle.distributed.launch --gpus 0,1,2,3 \
      tests/multi_card_tests/test_zcc_ema_reshard_layout.py
"""

import os
import shutil
import sys
import unittest

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
)

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed.flex_checkpoint.dcp.sharded_weight import ShardedWeight

from paddlefleet.trainer.trainer import Trainer

# Rows per rank is what makes the shards distinguishable: rank r owning block r
# and rank r owning block (r + 1) % world are different bytes on disk.
ROWS_PER_RANK = 4
COLS = 8

# The master-weight portion is keyed ".w_0" and the model-param portion is not;
# ``_load_ema_with_reshard`` picks them out of two different state dicts, so both
# are covered. The two extra keys must be dropped by the same filters -- if they
# leak into the load target, ``dist.load_state_dict`` fails on a key the
# checkpoint does not have.
MASTER_KEY = "ema_linear_0.w_0"
MODEL_KEY = "ema_layer.weight"
OPT_MOMENT_KEY = "ema_linear_0.moment1_0"
MODEL_BF16_KEY = "ema_layer.bf16_weight"


def _global_reference(key):
    """Deterministic global EMA content, distinct per key."""
    world = dist.get_world_size()
    rows = ROWS_PER_RANK * world
    base = float(len(key))
    flat = np.arange(rows * COLS, dtype="float32")
    return (flat.reshape(rows, COLS) + base * 1000.0).astype("float32")


def _owned_block(rank, layout):
    """Which global row block ``rank`` owns under ``layout``."""
    world = dist.get_world_size()
    if layout == "identity":
        return rank
    if layout == "rotated":
        return (rank + 1) % world
    raise ValueError(f"unknown layout: {layout}")


def _sharded_weight(key, layout, dtype="float32", fill_reference=False):
    """One rank's ShardedWeight for ``key`` under ``layout``."""
    world = dist.get_world_size()
    block = _owned_block(dist.get_rank(), layout)
    rows = slice(block * ROWS_PER_RANK, (block + 1) * ROWS_PER_RANK)
    if fill_reference:
        local = paddle.to_tensor(_global_reference(key)[rows, :]).astype(dtype)
    else:
        local = paddle.zeros([ROWS_PER_RANK, COLS], dtype=dtype)
    return ShardedWeight(
        key=key,
        local_tensor=local,
        local_shape=(ROWS_PER_RANK, COLS),
        global_shape=(ROWS_PER_RANK * world, COLS),
        global_offset=(block * ROWS_PER_RANK, 0),
    )


def _save_ema_checkpoint(path, layout):
    """Write an FC-format EMA checkpoint whose shards follow ``layout``."""
    state_dict = {
        MASTER_KEY: _sharded_weight(MASTER_KEY, layout, fill_reference=True),
        MODEL_KEY: _sharded_weight(MODEL_KEY, layout, fill_reference=True),
    }
    dist.save_state_dict(state_dict, path)
    dist.barrier()


class _StubModel:
    """Only ``sharded_state_dict`` is reached by the code under test."""

    def __init__(self, state_dict):
        self._state_dict = state_dict

    def sharded_state_dict(self):
        return self._state_dict


class _StubOptimizer:
    def __init__(self, state_dict):
        self._state_dict = state_dict

    def sharded_state_dict(self, model_sharded_state_dict):
        return self._state_dict


class _StubTrainer:
    """Carries just the attributes ``_load_ema_with_reshard`` touches.

    The method is called unbound so the production implementation runs as-is,
    without dragging in TrainingArguments, a dataloader or the ZCC workers.
    """

    def __init__(self):
        self.model = _StubModel(
            {
                MODEL_KEY: _sharded_weight(MODEL_KEY, "identity"),
                MODEL_BF16_KEY: _sharded_weight(
                    MODEL_BF16_KEY, "identity", dtype="bfloat16"
                ),
            }
        )
        self.optimizer = _StubOptimizer(
            {
                MASTER_KEY: _sharded_weight(MASTER_KEY, "identity"),
                OPT_MOMENT_KEY: _sharded_weight(OPT_MOMENT_KEY, "identity"),
            }
        )
        self.args = type(
            "_Args", (), {"aoa_config": None, "load_via_cpu": False}
        )()


def _read_shared_memory(reshard_result):
    """Reopen the shm handles the way the ZCC worker does.

    ``_load_ema_with_reshard`` clears the GPU tensors and hands the subprocess
    nothing but these metas, so reading them back here is the only way to see
    what a worker would actually consume -- and it fails if the handoff produced
    unusable handles.
    """
    out = {}
    for key, info in reshard_result.items():
        lod = paddle.base.core.LoDTensor._new_shared_filename(
            info["shared_meta"]
        )
        out[key] = paddle.to_tensor(lod).reshape(info["shape"]).numpy().copy()
    return out


class TestZCCEMAReshardLayout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dist.init_parallel_env()
        cls.rank = dist.get_rank()
        cls.world = dist.get_world_size()
        assert cls.world > 1, "this regression needs more than one rank"
        cls.tmp_root = os.path.join(
            os.environ.get("TMPDIR", "/tmp"),
            f"zcc_ema_reshard_layout_{os.environ.get('PADDLE_MASTER', 'local').replace(':', '_')}",
        )

    def _reshard_from(self, save_layout):
        """Save EMA under ``save_layout``, resume it under the identity layout."""
        path = os.path.join(self.tmp_root, save_layout)
        if self.rank == 0:
            shutil.rmtree(path, ignore_errors=True)
            os.makedirs(path, exist_ok=True)
        dist.barrier()

        _save_ema_checkpoint(path, save_layout)

        trainer = _StubTrainer()
        reshard_result = Trainer._load_ema_with_reshard(
            trainer, path, "broadcast", None
        )
        self.assertEqual(
            sorted(reshard_result.keys()), sorted([MASTER_KEY, MODEL_KEY])
        )
        loaded = _read_shared_memory(reshard_result)
        dist.barrier()
        return loaded

    def _expected_local(self, key):
        block = _owned_block(self.rank, "identity")
        rows = slice(block * ROWS_PER_RANK, (block + 1) * ROWS_PER_RANK)
        return _global_reference(key)[rows, :]

    def test_resume_from_changed_2d_layout(self):
        """Old layout on disk, new layout in the live run: EMA must still land."""
        loaded = self._reshard_from("rotated")
        for key in (MASTER_KEY, MODEL_KEY):
            np.testing.assert_array_equal(
                loaded[key],
                self._expected_local(key),
                err_msg=f"rank {self.rank} got the wrong EMA shard for {key}",
            )

    def test_layout_unchanged_matches_changed_layout(self):
        """Resharding is layout-invariant, so both routes give the same EMA."""
        unchanged = self._reshard_from("identity")
        changed = self._reshard_from("rotated")
        for key in (MASTER_KEY, MODEL_KEY):
            np.testing.assert_array_equal(
                unchanged[key],
                self._expected_local(key),
                err_msg=f"rank {self.rank} got the wrong EMA shard for {key}",
            )
            np.testing.assert_array_equal(
                unchanged[key],
                changed[key],
                err_msg=(
                    f"rank {self.rank} EMA for {key} depends on the checkpoint "
                    "layout"
                ),
            )

    @classmethod
    def tearDownClass(cls):
        dist.barrier()
        if cls.rank == 0:
            shutil.rmtree(cls.tmp_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
