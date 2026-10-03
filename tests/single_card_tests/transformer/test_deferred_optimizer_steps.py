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

"""Deferred gradients must update weights and clear between optimizer steps."""

import unittest
from unittest.mock import patch

import numpy as np
import paddle
from paddle.distributed.fleet.meta_parallel.zero_bubble_utils import (
    WeightGradStore,
)
from paddle.distributed.fleet.utils.mix_precision_utils import (
    MixPrecisionLayer,
    MixPrecisionOptimizer,
)

from paddlefleet.transformer.dw_overlap import DeferredWeightGradLinear


class TestDeferredOptimizerSteps(unittest.TestCase):
    def run_steps(self, deferred, enabled):
        model = paddle.nn.Linear(4, 2, bias_attr=False)
        model.weight.set_value(np.arange(8, dtype="float32").reshape(4, 2) / 16)
        MixPrecisionLayer(model, dtype="bfloat16")
        optimizer = MixPrecisionOptimizer(
            paddle.optimizer.SGD(
                learning_rate=1 / 128, parameters=model.parameters()
            )
        )
        observed = []
        pending = []
        previous_enabled = WeightGradStore.enabled
        self.addCleanup(setattr, WeightGradStore, "enabled", previous_enabled)
        WeightGradStore.enabled = enabled
        with patch.object(WeightGradStore, "put", side_effect=pending.append):
            for step in range(4):
                outputs, inputs = [], []
                previous_weight = model.weight.numpy().copy()
                for micro in range(3):
                    x = paddle.to_tensor(
                        np.arange(24, dtype="float32").reshape(2, 3, 4) / 32
                        + step / 8
                        + micro / 16,
                        stop_gradient=False,
                    )
                    output = (
                        DeferredWeightGradLinear.apply(x, model.weight)
                        if deferred
                        else model(x)
                    )
                    (output.square().sum()).backward()
                    self.assertEqual(WeightGradStore.enabled, enabled)
                    outputs.append(output.numpy().copy())
                    inputs.append(x.grad.numpy().copy())
                for finish in pending:
                    finish()
                pending.clear()
                gradient = model.weight.main_grad.numpy().copy()
                optimizer.step()
                current_weight = model.weight.numpy().copy()
                self.assertTrue(
                    np.any(current_weight != previous_weight),
                    "An accepted gradient must update the weight",
                )
                optimizer.clear_grad()
                np.testing.assert_array_equal(
                    model.weight.main_grad.numpy(),
                    np.zeros((4, 2), dtype="float32"),
                )
                observed.append((outputs, inputs, gradient, current_weight))
        return observed

    def test_deferred_steps_match_inline_with_either_outer_queue_state(self):
        for enabled in [False, True]:
            with self.subTest(enabled=enabled):
                reference = self.run_steps(False, enabled)
                actual = self.run_steps(True, enabled)
                for left, right in zip(reference, actual, strict=True):
                    for inline, deferred in zip(left, right, strict=True):
                        np.testing.assert_array_equal(inline, deferred)


if __name__ == "__main__":
    paddle.set_device("gpu:0")
    unittest.main()
