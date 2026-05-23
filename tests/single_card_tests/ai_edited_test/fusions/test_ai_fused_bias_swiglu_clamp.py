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
"""End-to-end PyLayer apply+backward coverage for WeightedSwiGLUFunction
and ClampedWeightedSwiGLUFunction.

Note: forward/backward shape coverage of standalone clamped_swiglu/
clamped_swiglu_back/clamped_weighted_swiglu/clamped_weighted_swiglu_back
is exercised in test_ai_fused_bias_swiglu.py. This file focuses on the
PyLayer apply paths for both clamp and non-clamp variants.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

import paddle

from paddlefleet.fusions.fused_bias_swiglu import (
    ClampedWeightedSwiGLUFunction,
    WeightedSwiGLUFunction,
)


class TestWeightedSwiGLUFunctionClampPyLayer(unittest.TestCase):
    """Forward/backward through PyLayer for clamp and non-clamp branches."""

    def test_backward_with_clamp(self):
        """ClampedWeightedSwiGLUFunction forward + clamp backward path."""
        x = paddle.randn([4, 16]).astype("float32")
        x.stop_gradient = False
        weights = paddle.ones([4, 1]).astype("float32")
        weights.stop_gradient = False
        result = ClampedWeightedSwiGLUFunction.apply(x, weights, False, 1.0)
        self.assertEqual(result.shape, [4, 8])
        result.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(weights.grad)

    def test_backward_without_clamp(self):
        """Non-clamp backward dispatches to weighted_swiglu_back."""
        x = paddle.randn([4, 16]).astype("float32")
        x.stop_gradient = False
        weights = paddle.ones([4, 1]).astype("float32")
        weights.stop_gradient = False
        mock_back = MagicMock(
            return_value=(paddle.randn([4, 16]), paddle.randn([4, 1]))
        )
        with patch(
            "paddlefleet.fusions.fused_bias_swiglu.weighted_swiglu_back",
            mock_back,
        ):
            result = WeightedSwiGLUFunction.apply(x, weights, False)
            self.assertEqual(result.shape, [4, 8])
            result.sum().backward()
            mock_back.assert_called_once()


if __name__ == "__main__":
    unittest.main()
