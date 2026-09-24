# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
"""FP8 expert-weight callbacks must stand down under paddle-native FSDP.

``fully_shard`` owns parameter storage: outside the owning unit's forward a
``.weight`` has no storage, and the optimizer is never wrapped into a sharding
optimizer, so it has no ``clear_param_storage``. Both pre-quantization callbacks
used to run anyway and died with "Tensor not initialized yet" /
"Tensor holds no memory" on the first step.

Run with:
    python -m pytest tests/single_card_tests/fp8/test_fp8_fsdp_callbacks.py -v
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from paddlefleet.trainer.trainer_callback import (
    FP8QuantWeightCallback,
    SonicMoELayoutSwitchCallback,
)
from paddlefleet.trainer.trainer_utils import ShardingOption


class _ExplodingModel:
    """Any weight access outside forward is a bug under FSDP."""

    def fp8_quant_weight(self, *args, **kwargs):
        raise AssertionError("fp8_quant_weight must not run under fsdp")

    def apply(self, fn):
        raise AssertionError("model.apply must not run under fsdp")


class _ExplodingOptimizer:
    def clear_param_storage(self, color):
        raise AssertionError("clear_param_storage must not run under fsdp")


def _args(sharding, using_sonic_moe=False, fp8="e4m3"):
    return SimpleNamespace(
        sharding=sharding,
        sharding_parallel_size=8,
        using_sonic_moe=using_sonic_moe,
        fp8=fp8,
        offload_fp8_expert_master_weight=True,
    )


class TestFP8QuantWeightCallbackUnderFSDP(unittest.TestCase):
    def setUp(self):
        self.callback = FP8QuantWeightCallback()
        self.kwargs = {
            "model": _ExplodingModel(),
            "optimizer": _ExplodingOptimizer(),
        }

    def test_on_step_begin_skips_under_fsdp(self):
        self.callback.on_step_begin(
            _args([ShardingOption.FSDP]), None, None, **self.kwargs
        )

    def test_on_optimizer_begin_skips_under_fsdp(self):
        self.callback.on_optimizer_begin(
            _args([ShardingOption.FSDP]), None, None, **self.kwargs
        )

    def test_stage1_still_quantizes(self):
        """Negative control: the stage1 path must keep pre-quantizing."""
        with self.assertRaises(AssertionError):
            self.callback.on_step_begin(
                _args([ShardingOption.SHARD_OP]), None, None, **self.kwargs
            )


class TestSonicMoECallbackUnderFSDP(unittest.TestCase):
    def setUp(self):
        self.callback = SonicMoELayoutSwitchCallback()
        self.kwargs = {
            "model": _ExplodingModel(),
            "optimizer": _ExplodingOptimizer(),
        }

    def test_on_step_begin_rejects_fsdp(self):
        args = _args([ShardingOption.FSDP], using_sonic_moe=True)
        with self.assertRaises(NotImplementedError):
            self.callback.on_step_begin(args, None, None, **self.kwargs)

    def test_on_optimizer_begin_rejects_fsdp(self):
        args = _args([ShardingOption.FSDP], using_sonic_moe=True)
        with self.assertRaises(NotImplementedError):
            self.callback.on_optimizer_begin(args, None, None, **self.kwargs)

    def test_no_op_when_sonic_moe_disabled(self):
        args = _args([ShardingOption.FSDP], using_sonic_moe=False)
        self.callback.on_step_begin(args, None, None, **self.kwargs)
        self.callback.on_optimizer_begin(args, None, None, **self.kwargs)


if __name__ == "__main__":
    unittest.main()
