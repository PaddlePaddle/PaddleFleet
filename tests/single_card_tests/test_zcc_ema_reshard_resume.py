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
Regression test that drives ``Trainer._load_flex_checkpoint`` through the ZCC
EMA reshard block (trainer.py:1677-1684), the lines this PR changed.

The layout test (tests/multi_card_tests/test_zcc_ema_reshard_layout.py)
exercises ``_load_ema_with_reshard`` directly, so it never reaches the call
site inside ``_load_flex_checkpoint`` and stays green on the parent commit.
This test reaches it: it asserts the resume routes an FC-format EMA through the
reshard (``_ema_reshard_result`` set), which on the parent commit only happened
when a degree actually changed -- a same-degree, layout-only resume there took
the file-read path and left the attribute None.

Kept single-rank and stub-driven on purpose. ``_load_flex_checkpoint`` reaches
``self.model``/``self.optimizer`` only through ``sharded_state_dict``, and with
``sharded_model_from_ema`` off + ``ignore_load_lr_and_optim`` on the only real
loads left are the master weight and the EMA reshard; ``init_optimizer`` is
patched out. So a real sharding optimizer and a real model are unnecessary --
matching hand-built ShardedWeight fixtures are enough, and the test runs on
paddle wheels without ``machine_balanced_2d_partition``.

Run with:
  pytest -s tests/single_card_tests/test_zcc_ema_reshard_resume.py
"""

import os
import shutil
import sys
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
from paddlefleet.utils.env import (
    EMA_STATE_DIC,
    MASTER_WEIGHT_DIC,
    MODEL_STATE_DIC,
    OPTIMIZER_STATE_DIC,
)

ROWS = 4
COLS = 8

# ``_load_ema_with_reshard`` pulls master weights ("*.w_0") from the optimizer
# state dict and fp32 params from the model's, so both portions are covered.
MASTER_KEY = "layer_0.linear.w_0"
MODEL_KEY = "layer_0.linear.weight"
OPT_MOMENT_KEY = "layer_0.linear.moment1_0"
MODEL_BF16_KEY = "layer_0.norm.weight"


def _reference(key):
    flat = np.arange(ROWS * COLS, dtype="float32")
    return (flat.reshape(ROWS, COLS) + len(key) * 1000.0).astype("float32")


def _sw(key, dtype="float32", fill=False):
    """A whole-tensor (single-rank) ShardedWeight for ``key``."""
    if fill:
        local = paddle.to_tensor(_reference(key)).astype(dtype)
    else:
        local = paddle.zeros([ROWS, COLS], dtype=dtype)
    return ShardedWeight(
        key=key,
        local_tensor=local,
        local_shape=(ROWS, COLS),
        global_shape=(ROWS, COLS),
        global_offset=(0, 0),
    )


class _Stub:
    """Returns freshly zeroed load targets each call, like a real module would.

    ``_load_flex_checkpoint`` and ``_load_ema_with_reshard`` each ask for the
    sharded state dict independently; handing back the same tensors twice would
    let one load's writes leak into the other's target.
    """

    def __init__(self, factory):
        self._factory = factory

    def sharded_state_dict(self, *_):
        return self._factory()


def _model_sd():
    return {
        MODEL_KEY: _sw(MODEL_KEY),
        MODEL_BF16_KEY: _sw(MODEL_BF16_KEY, dtype="bfloat16"),
    }


def _optimizer_sd():
    return {MASTER_KEY: _sw(MASTER_KEY), OPT_MOMENT_KEY: _sw(OPT_MOMENT_KEY)}


class _FlexHarness:
    """Runs the real ``_load_flex_checkpoint`` without a full Trainer."""

    _load_flex_checkpoint = Trainer._load_flex_checkpoint
    _is_fc_format_ema = Trainer._is_fc_format_ema
    _load_ema_with_reshard = Trainer._load_ema_with_reshard

    def __init__(self):
        self.model = _Stub(_model_sd)
        self.optimizer = _Stub(_optimizer_sd)
        self.args = SimpleNamespace(
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
        )


def _save(directory, state_dict):
    os.makedirs(directory, exist_ok=True)
    dist.save_state_dict(state_dict, directory)


def _build_checkpoint(root):
    """Write the four flex-checkpoint subdirs a resume reads.

    optimizer_state is only scanned for its ``.metadata`` (opt load is skipped
    via ignore_load_lr_and_optim), so its content is irrelevant; model_state,
    master_weight and ema_state are really loaded and must match their load
    targets key-for-key -- including the bf16 decoy in the model state.
    """
    _save(
        os.path.join(root, MODEL_STATE_DIC),
        {
            MODEL_KEY: _sw(MODEL_KEY, fill=True),
            MODEL_BF16_KEY: _sw(MODEL_BF16_KEY, dtype="bfloat16", fill=True),
        },
    )
    _save(
        os.path.join(root, OPTIMIZER_STATE_DIC),
        {OPT_MOMENT_KEY: _sw(OPT_MOMENT_KEY, fill=True)},
    )
    _save(
        os.path.join(root, MASTER_WEIGHT_DIC),
        {MASTER_KEY: _sw(MASTER_KEY, fill=True)},
    )
    _save(
        os.path.join(root, EMA_STATE_DIC),
        {
            MASTER_KEY: _sw(MASTER_KEY, fill=True),
            MODEL_KEY: _sw(MODEL_KEY, fill=True),
        },
    )


class TestZCCEMAReshardResume(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": 1,
            "sharding_degree": 1,
        }
        fleet.init(is_collective=True, strategy=strategy)
        master = os.environ.get("PADDLE_MASTER", "local").replace(":", "_")
        cls.tmp_root = os.path.join(
            os.environ.get("TMPDIR", "/tmp"), f"zcc_ema_resume_{master}"
        )

    def test_resume_routes_fc_ema_through_reshard(self):
        ckpt = os.path.join(self.tmp_root, "ckpt")
        shutil.rmtree(ckpt, ignore_errors=True)
        _build_checkpoint(ckpt)

        harness = _FlexHarness()
        # init_optimizer would need a real sharding optimizer to build
        # accumulators; the reshard block under test does not depend on it.
        with patch("paddlefleet.trainer.trainer.init_optimizer"):
            harness._load_flex_checkpoint(ckpt)

        # Parent commit: a same-degree resume skipped the reshard and left this
        # None. The fix reshards unconditionally for FC-format EMA.
        self.assertIsNotNone(
            harness._ema_reshard_result,
            "FC-format EMA resume did not go through _load_ema_with_reshard",
        )
        self.assertEqual(
            sorted(harness._ema_reshard_result.keys()),
            sorted([MASTER_KEY, MODEL_KEY]),
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
