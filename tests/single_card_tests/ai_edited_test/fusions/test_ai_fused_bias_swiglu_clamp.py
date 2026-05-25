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
import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

import paddle


class TestClampedSwiglu(unittest.TestCase):
    """Cover clamped_swiglu (lines 136-141)."""

    def test_clamped_swiglu_basic(self):
        from paddlefleet.fusions.fused_bias_swiglu import clamped_swiglu

        y = paddle.randn([4, 16])
        result = clamped_swiglu(y, clamp_value=1.0)
        self.assertEqual(result.shape, [4, 8])

    def test_clamped_swiglu_saturated(self):
        """All values exceed clamp -> output should be bounded."""
        from paddlefleet.fusions.fused_bias_swiglu import clamped_swiglu

        y = paddle.full([2, 8], 100.0)
        result = clamped_swiglu(y, clamp_value=1.0)
        self.assertEqual(result.shape, [2, 4])


class TestClampedSwigluBack(unittest.TestCase):
    """Cover clamped_swiglu_back (lines 158-174)."""

    def test_clamped_swiglu_back_basic(self):
        from paddlefleet.fusions.fused_bias_swiglu import clamped_swiglu_back

        g = paddle.randn([4, 8])
        y = paddle.randn([4, 16])
        result = clamped_swiglu_back(g, y, clamp_value=1.0)
        self.assertEqual(result.shape, [4, 16])

    def test_clamped_swiglu_back_saturated(self):
        """Saturated inputs -> gradients should be zero in clamped regions."""
        from paddlefleet.fusions.fused_bias_swiglu import clamped_swiglu_back

        g = paddle.ones([2, 4])
        y = paddle.full([2, 8], 100.0)
        result = clamped_swiglu_back(g, y, clamp_value=1.0)
        self.assertEqual(result.shape, [2, 8])
        self.assertTrue(bool((result.abs().sum() == 0).item()))


class TestClampedWeightedSwiglu(unittest.TestCase):
    """Cover clamped_weighted_swiglu (lines 189-191)."""

    def test_clamped_weighted_swiglu_basic(self):
        from paddlefleet.fusions.fused_bias_swiglu import (
            clamped_weighted_swiglu,
        )

        y = paddle.randn([4, 16])
        weights = paddle.ones([4, 1])
        result = clamped_weighted_swiglu(y, weights, clamp_value=1.0)
        self.assertEqual(result.shape, [4, 8])


class TestClampedWeightedSwigluBack(unittest.TestCase):
    """Cover clamped_weighted_swiglu_back (lines 207-212)."""

    def test_clamped_weighted_swiglu_back_basic(self):
        from paddlefleet.fusions.fused_bias_swiglu import (
            clamped_weighted_swiglu_back,
        )

        g = paddle.randn([4, 8])
        y = paddle.randn([4, 16])
        weights = paddle.ones([4, 1])
        input_grad, weights_grad = clamped_weighted_swiglu_back(
            g, y, weights, clamp_value=1.0
        )
        self.assertEqual(input_grad.shape, [4, 16])
        self.assertEqual(weights_grad.shape, [4, 1])


class TestWeightedSwiGLUFunctionClamp(unittest.TestCase):
    """Cover WeightedSwiGLUFunction forward/backward with clamp_value
    (lines 321-323, 330-337)."""

    def test_forward_with_clamp(self):
        """Lines 321-323: forward with clamp_value."""
        from paddlefleet.fusions.fused_bias_swiglu import (
            WeightedSwiGLUFunction,
        )

        x = paddle.randn([4, 16])
        weights = paddle.ones([4, 1])
        result = WeightedSwiGLUFunction.apply(
            x, weights, False, clamp_value=1.0
        )
        self.assertEqual(result.shape, [4, 8])

    def test_forward_without_clamp(self):
        """Line 324: forward without clamp_value."""
        from paddlefleet.fusions.fused_bias_swiglu import (
            WeightedSwiGLUFunction,
        )

        x = paddle.randn([4, 16])
        weights = paddle.ones([4, 1])
        result = WeightedSwiGLUFunction.apply(x, weights, False)
        self.assertEqual(result.shape, [4, 8])

    def test_backward_with_clamp(self):
        """Lines 330-334: backward with clamp_value."""
        from paddlefleet.fusions.fused_bias_swiglu import (
            WeightedSwiGLUFunction,
        )

        x = paddle.randn([4, 16]).astype("float32")
        x.stop_gradient = False
        weights = paddle.ones([4, 1]).astype("float32")
        weights.stop_gradient = False
        result = WeightedSwiGLUFunction.apply(
            x, weights, False, clamp_value=1.0
        )
        loss = result.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(weights.grad)

    def test_backward_without_clamp(self):
        """Line 336: backward without clamp_value calls weighted_swiglu_back."""
        from unittest.mock import MagicMock, patch

        import paddle

        from paddlefleet.fusions.fused_bias_swiglu import (
            WeightedSwiGLUFunction,
        )

        x = paddle.randn([4, 16]).astype("float32")
        x.stop_gradient = False
        weights = paddle.ones([4, 1]).astype("float32")
        weights.stop_gradient = False

        mock_back = MagicMock(
            return_value=(
                paddle.randn([4, 16]),
                paddle.randn([4, 1]),
            )
        )
        with patch(
            "paddlefleet.fusions.fused_bias_swiglu.weighted_swiglu_back",
            mock_back,
        ):
            result = WeightedSwiGLUFunction.apply(x, weights, False)
            loss = result.sum()
            loss.backward()
            mock_back.assert_called_once()


class TestWeightedBiasSwigluImplClamp(unittest.TestCase):
    """Cover weighted_bias_swiglu_impl with clamp_value (line 399)."""

    def test_no_bias_with_clamp(self):
        """Line 399: no bias -> calls WeightedSwiGLUFunction.apply with clamp."""
        from paddlefleet.fusions.fused_bias_swiglu import (
            weighted_bias_swiglu_impl,
        )

        x = paddle.randn([8, 16])
        weights = paddle.ones([8, 1])
        result = weighted_bias_swiglu_impl(
            x, None, weights, clamp_value=1.0, fp8_input_store=False
        )
        self.assertEqual(result.shape, [8, 8])

    def test_with_bias_raises(self):
        """Lines 394-396: bias not None -> raises NotImplementedError."""
        from paddlefleet.fusions.fused_bias_swiglu import (
            weighted_bias_swiglu_impl,
        )

        x = paddle.randn([8, 16])
        bias = paddle.randn([16])
        weights = paddle.ones([8, 1])
        with self.assertRaises(NotImplementedError):
            weighted_bias_swiglu_impl(
                x, bias, weights, clamp_value=1.0, fp8_input_store=False
            )


if __name__ == "__main__":
    unittest.main()
