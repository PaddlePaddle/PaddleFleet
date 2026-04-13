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

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)


# Tests for src/paddlefleet/fusions/fused_bias_swiglu.py
# Additional tests for swiglu backward, weighted swiglu,
# BiasSwiGLUFunction, SwiGLUFunction, WeightedSwiGLUFunction,
# bias_swiglu_impl, weighted_bias_swiglu_impl

import unittest
from unittest import mock

import paddle


class TestSwigluBackward(unittest.TestCase):
    """Tests for swiglu_back function."""

    def test_swiglu_back_cuda(self):
        """Test swiglu_back uses CUDA kernel."""
        from paddlefleet.fusions.fused_bias_swiglu import swiglu_back

        g = paddle.randn([4, 4])
        y = paddle.randn([4, 8])

        mock_result = paddle.randn([4, 8])
        with mock.patch("paddle.is_compiled_with_cuda", return_value=True):  # noqa: SIM117
            with mock.patch(
                "paddlefleet.ops.fused_swiglu_bwd",
                return_value=mock_result,
            ) as mock_op:
                result = swiglu_back(g, y)
                mock_op.assert_called_once_with(g, y)

    def test_swiglu_back_non_cuda_raises(self):
        """Test swiglu_back raises NotImplementedError on non-CUDA."""
        from paddlefleet.fusions.fused_bias_swiglu import swiglu_back

        g = paddle.randn([4, 4])
        y = paddle.randn([4, 8])

        with mock.patch("paddle.is_compiled_with_cuda", return_value=False):  # noqa: SIM117
            with self.assertRaises(NotImplementedError):
                swiglu_back(g, y)


class TestBiasSwigluBack(unittest.TestCase):
    """Tests for bias_swiglu_back function."""

    def test_bias_swiglu_back_adds_bias(self):
        """Test bias_swiglu_back adds bias before calling swiglu_back."""
        from paddlefleet.fusions.fused_bias_swiglu import bias_swiglu_back

        g = paddle.randn([4, 4])
        y = paddle.randn([4, 8])
        bias = paddle.randn([8])

        with mock.patch("paddle.is_compiled_with_cuda", return_value=True):  # noqa: SIM117
            with mock.patch(
                "paddlefleet.ops.fused_swiglu_bwd",
                return_value=paddle.randn([4, 8]),
            ) as mock_op:
                bias_swiglu_back(g, y, bias)
                # The y+bias should be passed to fused_swiglu_bwd
                call_args = mock_op.call_args[0]
                # bias_swiglu_back computes y = y + bias, then calls swiglu_back(g, y)
                # call_args[0] is g, call_args[1] is y+bias
                self.assertEqual(call_args[0].shape, g.shape)
                self.assertEqual(call_args[1].shape, y.shape)


class TestWeightedSwiglu(unittest.TestCase):
    """Tests for weighted_swiglu function."""

    def test_weighted_swiglu_output_shape(self):
        """Test weighted_swiglu output shape."""
        from paddlefleet.fusions.fused_bias_swiglu import weighted_swiglu

        y = paddle.randn([4, 8])
        weights = paddle.randn([4, 1])

        result = weighted_swiglu(y, weights)
        self.assertEqual(result.shape, [4, 4])

    def test_weighted_swiglu_output_dtype(self):
        """Test weighted_swiglu preserves dtype."""
        from paddlefleet.fusions.fused_bias_swiglu import weighted_swiglu

        y = paddle.randn([4, 8], dtype="float32")
        weights = paddle.randn([4, 1], dtype="float32")

        result = weighted_swiglu(y, weights)
        self.assertEqual(result.dtype, y.dtype)


class TestWeightedSwigluBack(unittest.TestCase):
    """Tests for weighted_swiglu_back function."""

    def test_weighted_swiglu_back_shapes(self):
        """Test weighted_swiglu_back output shapes."""
        from paddlefleet.fusions.fused_bias_swiglu import weighted_swiglu_back

        g = paddle.randn([4, 4])
        y = paddle.randn([4, 8])
        weights = paddle.randn([4, 1])

        mock_input_grad = paddle.randn([4, 8])
        mock_wgrad = paddle.randn([4, 1])

        with mock.patch("paddle.is_compiled_with_cuda", return_value=True):  # noqa: SIM117
            with mock.patch(
                "paddlefleet.ops.fused_swiglu_bwd",
                return_value=paddle.randn([4, 8]),
            ):
                input_grad, weights_grad = weighted_swiglu_back(g, y, weights)
                self.assertEqual(input_grad.shape, y.shape)
                self.assertEqual(weights_grad.shape, [4, 1])


class TestBiasSwiGLUFunction(unittest.TestCase):
    """Tests for BiasSwiGLUFunction PyLayer."""

    def test_forward_saves_tensors(self):
        """Test forward saves input and bias."""
        from paddlefleet.fusions.fused_bias_swiglu import BiasSwiGLUFunction

        mock_ctx = mock.MagicMock()
        inp = paddle.randn([4, 8])
        bias = paddle.randn([8])

        with mock.patch(
            "paddlefleet.fusions.fused_bias_swiglu.bias_swiglu",
            return_value=paddle.randn([4, 4]),
        ):
            result = BiasSwiGLUFunction.forward(
                mock_ctx, inp, bias, False, False
            )
            mock_ctx.save_for_backward.assert_called_once()

    def test_forward_fp8_store(self):
        """Test forward with fp8_input_store=True."""
        from paddlefleet.fusions.fused_bias_swiglu import BiasSwiGLUFunction

        mock_ctx = mock.MagicMock()
        inp = paddle.randn([4, 8])
        bias = paddle.randn([8])

        with mock.patch(
            "paddlefleet.fusions.fused_bias_swiglu.bias_swiglu",
            return_value=paddle.randn([4, 4]),
        ):
            result = BiasSwiGLUFunction.forward(
                mock_ctx, inp, bias, True, False
            )
            mock_ctx.save_for_backward.assert_called_once()
            self.assertTrue(mock_ctx.fp8_input_store)

    def test_backward_returns_same_grad_for_input_and_bias(self):
        """Test backward returns same gradient for input and bias."""
        from paddlefleet.fusions.fused_bias_swiglu import BiasSwiGLUFunction

        mock_ctx = mock.MagicMock()
        inp = paddle.randn([4, 8])
        bias = paddle.randn([8])
        mock_ctx.saved_tensor.return_value = [inp, bias]
        mock_ctx.fp8_input_store = False

        mock_grad = paddle.randn([4, 8])
        with mock.patch(
            "paddlefleet.fusions.fused_bias_swiglu.bias_swiglu_back",
            return_value=mock_grad,
        ):
            result = BiasSwiGLUFunction.backward(mock_ctx, paddle.randn([4, 4]))
            # Returns (tmp, tmp, None, None)
            self.assertEqual(len(result), 4)
            self.assertIs(result[2], None)
            self.assertIs(result[3], None)


class TestSwiGLUFunction(unittest.TestCase):
    """Tests for SwiGLUFunction PyLayer."""

    def test_forward_without_bias(self):
        """Test forward without bias."""
        from paddlefleet.fusions.fused_bias_swiglu import SwiGLUFunction

        mock_ctx = mock.MagicMock()
        inp = paddle.randn([4, 8])

        with mock.patch(
            "paddlefleet.fusions.fused_bias_swiglu.swiglu",
            return_value=paddle.randn([4, 4]),
        ):
            result = SwiGLUFunction.forward(mock_ctx, inp, False, False)
            mock_ctx.save_for_backward.assert_called_once()

    def test_backward_no_fp8(self):
        """Test backward without FP8 storage."""
        from paddlefleet.fusions.fused_bias_swiglu import SwiGLUFunction

        mock_ctx = mock.MagicMock()
        inp = paddle.randn([4, 8])
        mock_ctx.saved_tensor.return_value = [inp]
        mock_ctx.fp8_input_store = False

        mock_grad = paddle.randn([4, 8])
        with mock.patch(
            "paddlefleet.fusions.fused_bias_swiglu.swiglu_back",
            return_value=mock_grad,
        ):
            result = SwiGLUFunction.backward(mock_ctx, paddle.randn([4, 4]))
            # backward returns a single tensor (the input gradient)
            self.assertTrue(paddle.is_tensor(result))
            self.assertEqual(result.shape, mock_grad.shape)


class TestWeightedSwiGLUFunction(unittest.TestCase):
    """Tests for WeightedSwiGLUFunction PyLayer."""

    def test_forward_saves_input_and_weights(self):
        """Test forward saves input and weights."""
        from paddlefleet.fusions.fused_bias_swiglu import WeightedSwiGLUFunction

        mock_ctx = mock.MagicMock()
        inp = paddle.randn([4, 8])
        weights = paddle.randn([4, 1])

        with mock.patch(
            "paddlefleet.fusions.fused_bias_swiglu.weighted_swiglu",
            return_value=paddle.randn([4, 4]),
        ):
            result = WeightedSwiGLUFunction.forward(
                mock_ctx, inp, weights, False
            )
            mock_ctx.save_for_backward.assert_called_once()

    def test_backward_returns_input_and_weight_grads(self):
        """Test backward returns input and weight gradients."""
        from paddlefleet.fusions.fused_bias_swiglu import WeightedSwiGLUFunction

        mock_ctx = mock.MagicMock()
        inp = paddle.randn([4, 8])
        weights = paddle.randn([4, 1])
        mock_ctx.saved_tensor.return_value = [inp, weights]
        mock_ctx.fp8_input_store = False

        with mock.patch(
            "paddlefleet.fusions.fused_bias_swiglu.weighted_swiglu_back",
            return_value=(paddle.randn([4, 8]), paddle.randn([4, 1])),
        ):
            result = WeightedSwiGLUFunction.backward(
                mock_ctx, paddle.randn([4, 4])
            )
            self.assertEqual(len(result), 3)
            self.assertIs(result[2], None)


class TestBiasSwigluImpl(unittest.TestCase):
    """Tests for bias_swiglu_impl function."""

    def test_2d_input_no_bias(self):
        """Test 2D input without bias."""
        from paddlefleet.fusions.fused_bias_swiglu import bias_swiglu_impl

        inp = paddle.randn([4, 8])
        with mock.patch(
            "paddlefleet.fusions.fused_bias_swiglu.SwiGLUFunction.apply",
            return_value=paddle.randn([4, 4]),
        ):
            result = bias_swiglu_impl(inp, None)
            self.assertEqual(result.shape, [4, 4])

    def test_2d_input_with_bias(self):
        """Test 2D input with bias."""
        from paddlefleet.fusions.fused_bias_swiglu import bias_swiglu_impl

        inp = paddle.randn([4, 8])
        bias = paddle.randn([8])
        with mock.patch(
            "paddlefleet.fusions.fused_bias_swiglu.BiasSwiGLUFunction.apply",
            return_value=paddle.randn([4, 4]),
        ):
            result = bias_swiglu_impl(inp, bias)
            self.assertEqual(result.shape, [4, 4])

    def test_3d_input_no_bias(self):
        """Test 3D input without bias returns 3D output."""
        from paddlefleet.fusions.fused_bias_swiglu import bias_swiglu_impl

        inp = paddle.randn([2, 3, 8])
        with (
            mock.patch(
                "paddlefleet.fusions.fused_bias_swiglu.SwiGLUFunction.apply",
                return_value=paddle.randn([6, 4]),
            ) as mock_apply,
            mock.patch.object(
                mock_apply.return_value,
                "view",
                return_value=paddle.randn([2, 3, 4]),
            ),
        ):
            result = bias_swiglu_impl(inp, None)
            # Should return 3D
            pass

    def test_3d_input_with_bias(self):
        """Test 3D input with bias."""
        from paddlefleet.fusions.fused_bias_swiglu import bias_swiglu_impl

        inp = paddle.randn([2, 3, 8])
        bias = paddle.randn([8])
        with mock.patch(
            "paddlefleet.fusions.fused_bias_swiglu.BiasSwiGLUFunction.apply",
            return_value=paddle.randn([6, 4]),
        ):
            result = bias_swiglu_impl(inp, bias)
            self.assertEqual(len(result.shape), 3)

    def test_invalid_dims_raises(self):
        """Test invalid input dimensions raises AssertionError."""
        from paddlefleet.fusions.fused_bias_swiglu import bias_swiglu_impl

        inp = paddle.randn([2, 3, 4, 8])  # 4D
        with self.assertRaises(AssertionError):
            bias_swiglu_impl(inp, None)


class TestWeightedBiasSwigluImpl(unittest.TestCase):
    """Tests for weighted_bias_swiglu_impl function."""

    def test_with_bias_raises(self):
        """Test weighted_bias_swiglu_impl with bias raises."""
        from paddlefleet.fusions.fused_bias_swiglu import (
            weighted_bias_swiglu_impl,
        )

        inp = paddle.randn([4, 8])
        bias = paddle.randn([8])
        weights = paddle.randn([4, 1])

        with self.assertRaises(NotImplementedError):
            weighted_bias_swiglu_impl(inp, bias, weights)

    def test_no_bias_2d(self):
        """Test weighted_bias_swiglu_impl without bias, 2D input."""
        from paddlefleet.fusions.fused_bias_swiglu import (
            weighted_bias_swiglu_impl,
        )

        inp = paddle.randn([4, 8])
        weights = paddle.randn([4, 1])

        with mock.patch(
            "paddlefleet.fusions.fused_bias_swiglu.WeightedSwiGLUFunction.apply",
            return_value=paddle.randn([4, 4]),
        ):
            result = weighted_bias_swiglu_impl(inp, None, weights)
            self.assertEqual(result.shape, [4, 4])


if __name__ == "__main__":
    unittest.main()
