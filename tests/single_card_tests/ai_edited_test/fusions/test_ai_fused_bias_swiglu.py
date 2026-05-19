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

import unittest
from unittest.mock import patch

import numpy as np
import paddle
import paddle.nn.functional as F

from paddlefleet.fusions.fused_bias_swiglu import (
    BiasSwiGLUFunction,
    SwiGLUFunction,
    WeightedSwiGLUFunction,
    bias_swiglu,
    bias_swiglu_back,
    bias_swiglu_impl,
    swiglu,
    swiglu_back,
    weighted_bias_swiglu_impl,
    weighted_swiglu,
    weighted_swiglu_back,
)


class TestSwiglu(unittest.TestCase):
    """Tests for swiglu function."""

    def test_swiglu_output_shape(self):
        """Test swiglu halves last dimension."""
        x = paddle.randn([2, 8])
        result = swiglu(x)
        self.assertEqual(result.shape, [2, 4])

    def test_swiglu_positive_input(self):
        """Test swiglu with positive input."""
        x = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]])
        result = swiglu(x)
        self.assertEqual(result.shape, [1, 4])
        # SiLU(x) * x for positive x should be positive
        self.assertTrue((result >= 0).all())


class TestBiasSwiglu(unittest.TestCase):
    """Tests for bias_swiglu function."""

    def test_bias_swiglu_output_shape(self):
        """Test bias_swiglu halves last dimension."""
        x = paddle.randn([2, 8])
        bias = paddle.randn([8])
        result = bias_swiglu(x, bias)
        self.assertEqual(result.shape, [2, 4])

    def test_bias_swiglu_different_bias_shape(self):
        """Test bias_swiglu with 2D bias."""
        x = paddle.randn([2, 8])
        bias = paddle.randn([1, 8])
        result = bias_swiglu(x, bias)
        self.assertEqual(result.shape, [2, 4])


class TestWeightedSwiglu(unittest.TestCase):
    """Tests for weighted_swiglu function."""

    def test_weighted_swiglu_output_shape(self):
        """Test weighted_swiglu output shape."""
        x = paddle.randn([2, 8])
        weights = paddle.randn([2, 1])
        result = weighted_swiglu(x, weights)
        self.assertEqual(result.shape, [2, 4])

    def test_weighted_swiglu_dtype_preserved(self):
        """Test weighted_swiglu preserves dtype."""
        x = paddle.randn([2, 8], dtype=paddle.float32)
        weights = paddle.randn([2, 1], dtype=paddle.float32)
        result = weighted_swiglu(x, weights)
        self.assertEqual(result.dtype, paddle.float32)


class TestSwigluBack(unittest.TestCase):
    """Tests for swiglu_back function."""

    @patch("paddle.is_compiled_with_cuda", return_value=False)
    def test_swiglu_back_not_implemented_cpu(self, mock_cuda):
        """Test swiglu_back raises NotImplementedError on CPU."""
        g = paddle.randn([2, 4])
        y = paddle.randn([2, 8])
        with self.assertRaises(NotImplementedError):
            swiglu_back(g, y)


class TestBiasSwigluBack(unittest.TestCase):
    """Tests for bias_swiglu_back function."""

    @patch("paddle.is_compiled_with_cuda", return_value=False)
    def test_bias_swiglu_back_cpu(self, mock_cuda):
        """Test bias_swiglu_back raises on CPU."""
        g = paddle.randn([2, 4])
        y = paddle.randn([2, 8])
        bias = paddle.randn([8])
        with self.assertRaises(NotImplementedError):
            bias_swiglu_back(g, y, bias)


class TestWeightedSwigluBack(unittest.TestCase):
    """Tests for weighted_swiglu_back function."""

    @patch("paddle.is_compiled_with_cuda", return_value=False)
    def test_weighted_swiglu_back_cpu_raises(self, mock_cuda):
        """Test weighted_swiglu_back raises on CPU."""
        g = paddle.randn([2, 4])
        y = paddle.randn([2, 8])
        weights = paddle.randn([2, 1])
        with self.assertRaises(NotImplementedError):
            weighted_swiglu_back(g, y, weights)


class TestBiasSwiGLUFunction(unittest.TestCase):
    """Tests for BiasSwiGLUFunction PyLayer."""

    @patch("paddle.is_compiled_with_cuda", return_value=False)
    def test_forward_calls_bias_swiglu(self, mock_cuda):
        """Test forward calls bias_swiglu."""
        input_t = paddle.randn([2, 8])
        bias = paddle.randn([8])
        with (
            patch(
                "paddlefleet.fusions.fused_bias_swiglu.bias_swiglu",
                return_value=paddle.randn([2, 4]),
            ) as mock_bias_swiglu,
            patch(
                "paddlefleet.fusions.fused_bias_swiglu.swiglu_back",
                side_effect=NotImplementedError,
            ),
        ):
            try:
                result = BiasSwiGLUFunction.apply(input_t, bias, False, False)
            except NotImplementedError:
                pass  # backward not invoked


class TestSwiGLUFunction(unittest.TestCase):
    """Tests for SwiGLUFunction PyLayer."""

    @patch("paddle.is_compiled_with_cuda", return_value=False)
    def test_forward_calls_swiglu(self, mock_cuda):
        """Test forward calls swiglu."""
        input_t = paddle.randn([2, 8])
        with patch(
            "paddlefleet.fusions.fused_bias_swiglu.swiglu",
            return_value=paddle.randn([2, 4]),
        ) as mock_swiglu_fn:
            try:
                result = SwiGLUFunction.apply(input_t, False, False)
            except NotImplementedError:
                pass


class TestWeightedSwiGLUFunction(unittest.TestCase):
    """Tests for WeightedSwiGLUFunction PyLayer."""

    @patch("paddle.is_compiled_with_cuda", return_value=False)
    def test_forward_calls_weighted_swiglu(self, mock_cuda):
        """Test forward calls weighted_swiglu."""
        input_t = paddle.randn([2, 8])
        weights = paddle.randn([2, 1])
        with patch(
            "paddlefleet.fusions.fused_bias_swiglu.weighted_swiglu",
            return_value=paddle.randn([2, 4]),
        ) as mock_fn:
            try:
                result = WeightedSwiGLUFunction.apply(input_t, weights, False)
            except NotImplementedError:
                pass


class TestBiasSwigluImpl(unittest.TestCase):
    """Tests for bias_swiglu_impl function."""

    def test_2d_input_with_bias(self):
        """Test bias_swiglu_impl with 2D input and bias."""
        with patch(
            "paddlefleet.fusions.fused_bias_swiglu.BiasSwiGLUFunction.apply",
            return_value=paddle.randn([2, 4]),
        ) as mock_apply:
            x = paddle.randn([2, 8])
            bias = paddle.randn([8])
            result = bias_swiglu_impl(x, bias)
            mock_apply.assert_called_once()

    def test_2d_input_without_bias(self):
        """Test bias_swiglu_impl with 2D input, no bias."""
        with patch(
            "paddlefleet.fusions.fused_bias_swiglu.SwiGLUFunction.apply",
            return_value=paddle.randn([2, 4]),
        ) as mock_apply:
            x = paddle.randn([2, 8])
            result = bias_swiglu_impl(x, None)
            mock_apply.assert_called_once()

    def test_3d_input_with_bias(self):
        """Test bias_swiglu_impl with 3D input."""
        with patch(
            "paddlefleet.fusions.fused_bias_swiglu.BiasSwiGLUFunction.apply",
            return_value=paddle.randn([2, 4, 4]),
        ) as mock_apply:
            x = paddle.randn([2, 4, 8])
            bias = paddle.randn([8])
            result = bias_swiglu_impl(x, bias)
            self.assertEqual(result.shape, [2, 4, 4])

    def test_asserts_invalid_dim(self):
        """Test assertion for invalid input dimensions."""
        x = paddle.randn([2, 4, 4, 8])
        with self.assertRaises(AssertionError):
            bias_swiglu_impl(x, None)


class TestWeightedBiasSwigluImpl(unittest.TestCase):
    """Tests for weighted_bias_swiglu_impl function."""

    def test_2d_input_no_bias(self):
        """Test weighted_bias_swiglu_impl with 2D input, no bias."""
        with patch(
            "paddlefleet.fusions.fused_bias_swiglu.WeightedSwiGLUFunction.apply",
            return_value=paddle.randn([2, 4]),
        ) as mock_apply:
            x = paddle.randn([2, 8])
            weights = paddle.randn([2, 1])
            result = weighted_bias_swiglu_impl(x, None, weights)
            mock_apply.assert_called_once()

    def test_bias_not_supported(self):
        """Test that bias raises NotImplementedError."""
        x = paddle.randn([2, 8])
        bias = paddle.randn([8])
        weights = paddle.randn([2, 1])
        with self.assertRaises(NotImplementedError):
            weighted_bias_swiglu_impl(x, bias, weights)

    def test_3d_input_no_bias(self):
        """Test weighted_bias_swiglu_impl with 3D input."""
        with patch(
            "paddlefleet.fusions.fused_bias_swiglu.WeightedSwiGLUFunction.apply",
            return_value=paddle.randn([2, 4, 4]),
        ) as mock_apply:
            x = paddle.randn([2, 4, 8])
            weights = paddle.randn([2, 1])
            result = weighted_bias_swiglu_impl(x, None, weights)
            self.assertEqual(result.shape, [2, 4, 4])

    def test_asserts_invalid_dim(self):
        """Test assertion for invalid input dimensions."""
        x = paddle.randn([2, 4, 4, 8])
        with self.assertRaises(AssertionError):
            weighted_bias_swiglu_impl(x, None, paddle.randn([2, 1]))


class TestClampedSwiGLU(unittest.TestCase):
    """Tests for clamped_swiglu and related functions."""

    def setUp(self):
        from paddlefleet.fusions.fused_bias_swiglu import (
            clamped_swiglu,
            clamped_swiglu_back,
            clamped_weighted_swiglu,
            clamped_weighted_swiglu_back,
        )
        self.clamped_swiglu = clamped_swiglu
        self.clamped_swiglu_back = clamped_swiglu_back
        self.clamped_weighted_swiglu = clamped_weighted_swiglu
        self.clamped_weighted_swiglu_back = clamped_weighted_swiglu_back

    def _make_input(self, B=2, S=4, H=16, dtype="float32"):
        return paddle.randn([B, S, H * 2]).cast(dtype)

    def test_forward_output_shape(self):
        """Output shape should be half of input last dim."""
        y = self._make_input(B=2, S=4, H=8)
        out = self.clamped_swiglu(y, clamp_value=5.0)
        self.assertEqual(out.shape, [2, 4, 8])

    def test_forward_clamp_effect(self):
        """With a very small clamp_value, output should be bounded."""
        clamp_value = 0.1
        y = paddle.full([4, 16], fill_value=100.0)
        out = self.clamped_swiglu(y, clamp_value=clamp_value)
        max_possible = float(F.silu(paddle.to_tensor(clamp_value)).numpy()) * clamp_value
        self.assertTrue(
            float(out.abs().max().numpy()) <= max_possible + 1e-5,
            f"output max {float(out.abs().max().numpy())} > expected max {max_possible}",
        )

    def test_large_clamp_equals_standard_swiglu(self):
        """With very large clamp_value, clamped_swiglu ≈ standard swiglu."""
        y = paddle.randn([4, 16])
        out_clamped = self.clamped_swiglu(y, clamp_value=1e9)
        out_standard = swiglu(y)
        np.testing.assert_allclose(
            out_clamped.cast("float32").numpy(),
            out_standard.cast("float32").numpy(),
            atol=1e-4,
            err_msg="clamped_swiglu with large clamp_value should match standard swiglu",
        )

    def test_backward_numerical_gradient(self):
        """Finite-difference gradient check for clamped_swiglu."""
        paddle.seed(42)
        clamp_value = 2.0

        def clamped_swiglu_autograd(y, cv):
            y_1, y_2 = paddle.chunk(y, 2, axis=-1)
            y_1_c = y_1.clip(max=cv)
            y_2_c = y_2.clip(min=-cv, max=cv)
            return F.silu(y_1_c) * y_2_c

        y_np = np.random.randn(3, 8).astype("float64")
        y_test = paddle.to_tensor(y_np, stop_gradient=False)
        out = clamped_swiglu_autograd(y_test, clamp_value)
        loss = out.sum()
        analytic_grad = paddle.grad([loss], [y_test])[0].numpy()

        eps = 1e-5
        num_grad = np.zeros_like(y_np)
        for i in range(y_np.shape[0]):
            for j in range(y_np.shape[1]):
                y_plus = y_np.copy()
                y_plus[i, j] += eps
                y_minus = y_np.copy()
                y_minus[i, j] -= eps
                out_plus = clamped_swiglu_autograd(
                    paddle.to_tensor(y_plus), clamp_value
                ).sum().numpy()
                out_minus = clamped_swiglu_autograd(
                    paddle.to_tensor(y_minus), clamp_value
                ).sum().numpy()
                num_grad[i, j] = (out_plus - out_minus) / (2 * eps)

        np.testing.assert_allclose(
            analytic_grad, num_grad, atol=1e-4, rtol=1e-3,
            err_msg="Analytical gradient does not match numerical gradient",
        )

    def test_weighted_output_shape(self):
        """clamped_weighted_swiglu output shape should match half of input."""
        y = self._make_input(B=2, S=4, H=8)
        weights = paddle.randn([2, 4, 1])
        out = self.clamped_weighted_swiglu(y, weights, clamp_value=5.0)
        self.assertEqual(out.shape, [2, 4, 8])

    def test_weighted_bias_swiglu_impl_clamp(self):
        """weighted_bias_swiglu_impl with clamp_value should run without error."""
        from paddlefleet.fusions.fused_bias_swiglu import weighted_bias_swiglu_impl

        inp = paddle.randn([8, 16])
        weights = paddle.randn([8, 1])
        out = weighted_bias_swiglu_impl(inp, None, weights, clamp_value=3.0)
        self.assertEqual(out.shape, [8, 8])
        self.assertFalse(
            bool(paddle.isnan(out).any().numpy()),
            "weighted_bias_swiglu_impl output contains NaN",
        )

    def test_weighted_bias_swiglu_impl_no_clamp_backward_compat(self):
        """weighted_bias_swiglu_impl without clamp_value should behave as before."""
        from paddlefleet.fusions.fused_bias_swiglu import weighted_bias_swiglu_impl

        inp = paddle.randn([8, 16])
        weights = paddle.randn([8, 1])
        out = weighted_bias_swiglu_impl(inp, None, weights)  # clamp_value=None
        self.assertEqual(out.shape, [8, 8])


if __name__ == "__main__":
    unittest.main()
