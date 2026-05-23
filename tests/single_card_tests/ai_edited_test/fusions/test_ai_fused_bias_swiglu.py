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
from unittest.mock import MagicMock, patch

import numpy as np
import paddle
import paddle.nn.functional as F

from paddlefleet.fusions.fused_bias_swiglu import (
    BiasSwiGLUFunction,
    ClampedWeightedSwiGLUFunction,
    SwiGLUFunction,
    WeightedSwiGLUFunction,
    bias_swiglu,
    bias_swiglu_back,
    bias_swiglu_impl,
    clamped_swiglu,
    clamped_swiglu_back,
    clamped_weighted_swiglu,
    clamped_weighted_swiglu_back,
    swiglu,
    swiglu_back,
    weighted_bias_swiglu_impl,
    weighted_swiglu,
    weighted_swiglu_back,
)


class TestSwiGLUForward(unittest.TestCase):
    """Forward computations for swiglu / bias_swiglu / weighted_swiglu."""

    def test_swiglu_forward(self):
        out = swiglu(paddle.randn([4, 16]))
        self.assertEqual(out.shape, [4, 8])

    def test_bias_swiglu_forward(self):
        out = bias_swiglu(paddle.randn([4, 16]), paddle.randn([16]))
        self.assertEqual(out.shape, [4, 8])

    def test_weighted_swiglu_dtype_preserved(self):
        x = paddle.randn([2, 8], dtype=paddle.float32)
        w = paddle.randn([2, 1], dtype=paddle.float32)
        out = weighted_swiglu(x, w)
        self.assertEqual(out.shape, [2, 4])
        self.assertEqual(out.dtype, paddle.float32)


class TestSwiGLUBackwardCPU(unittest.TestCase):
    """All *_back functions must raise NotImplementedError on non-CUDA backends."""

    @patch("paddle.is_compiled_with_cuda", return_value=False)
    def test_swiglu_back_cpu_raises(self, mock_cuda):
        with self.assertRaises(NotImplementedError):
            swiglu_back(paddle.randn([2, 4]), paddle.randn([2, 8]))

    @patch("paddle.is_compiled_with_cuda", return_value=False)
    def test_bias_swiglu_back_cpu_raises(self, mock_cuda):
        with self.assertRaises(NotImplementedError):
            bias_swiglu_back(
                paddle.randn([2, 4]), paddle.randn([2, 8]), paddle.randn([8])
            )

    @patch("paddle.is_compiled_with_cuda", return_value=False)
    def test_weighted_swiglu_back_cpu_raises(self, mock_cuda):
        with self.assertRaises(NotImplementedError):
            weighted_swiglu_back(
                paddle.randn([2, 4]), paddle.randn([2, 8]), paddle.randn([2, 1])
            )


class TestSwigluBackGPUPath(unittest.TestCase):
    """Exercise GPU branches of swiglu_back / weighted_swiglu_back via mocking."""

    def test_swiglu_back_cuda_branch(self):
        """When CUDA is available, swiglu_back must dispatch to fused_swiglu_bwd."""
        fake_grad = paddle.randn([2, 8])
        with patch("paddle.is_compiled_with_cuda", return_value=True):
            mock_mod = type(sys)("paddlefleet_ops")
            mock_mod.fused_swiglu_bwd = lambda g, y: fake_grad
            with patch.dict(sys.modules, {"paddlefleet_ops": mock_mod}):
                out = swiglu_back(paddle.randn([2, 4]), paddle.randn([2, 8]))
        np.testing.assert_array_equal(out.numpy(), fake_grad.numpy())

    def test_weighted_swiglu_back_cuda_branch(self):
        """weighted_swiglu_back GPU path returns (input_grad, weights_grad)."""
        with patch("paddle.is_compiled_with_cuda", return_value=True):
            mock_mod = type(sys)("paddlefleet_ops")
            mock_mod.fused_swiglu_bwd = lambda g, y: paddle.zeros_like(y)
            with patch.dict(sys.modules, {"paddlefleet_ops": mock_mod}):
                ig, wg = weighted_swiglu_back(
                    paddle.randn([2, 4]),
                    paddle.randn([2, 8]),
                    paddle.randn([2, 1]),
                )
        self.assertEqual(ig.shape, [2, 8])
        self.assertEqual(wg.shape, [2, 1])


class TestPyLayerForward(unittest.TestCase):
    """Forward dispatch through PyLayer.apply for each variant."""

    @patch("paddle.is_compiled_with_cuda", return_value=False)
    def test_bias_swiglu_function_apply(self, mock_cuda):
        with patch(
            "paddlefleet.fusions.fused_bias_swiglu.bias_swiglu",
            return_value=paddle.randn([2, 4]),
        ):
            try:
                BiasSwiGLUFunction.apply(
                    paddle.randn([2, 8]), paddle.randn([8]), False, False
                )
            except NotImplementedError:
                pass

    @patch("paddle.is_compiled_with_cuda", return_value=False)
    def test_swiglu_function_apply(self, mock_cuda):
        with patch(
            "paddlefleet.fusions.fused_bias_swiglu.swiglu",
            return_value=paddle.randn([2, 4]),
        ):
            try:
                SwiGLUFunction.apply(paddle.randn([2, 8]), False, False)
            except NotImplementedError:
                pass

    @patch("paddle.is_compiled_with_cuda", return_value=False)
    def test_weighted_swiglu_function_apply(self, mock_cuda):
        with patch(
            "paddlefleet.fusions.fused_bias_swiglu.weighted_swiglu",
            return_value=paddle.randn([2, 4]),
        ):
            try:
                WeightedSwiGLUFunction.apply(
                    paddle.randn([2, 8]), paddle.randn([2, 1]), False
                )
            except NotImplementedError:
                pass


class TestImplShapes(unittest.TestCase):
    """bias_swiglu_impl / weighted_bias_swiglu_impl shape & branch coverage."""

    def test_bias_swiglu_impl_2d_with_bias(self):
        out = bias_swiglu_impl(paddle.randn([4, 16]), paddle.randn([16]))
        self.assertEqual(out.shape, [4, 8])

    def test_bias_swiglu_impl_3d_no_bias(self):
        # Covers both 3D-reshape path and bias-None branch.
        out = bias_swiglu_impl(paddle.randn([2, 4, 16]), None)
        self.assertEqual(out.shape, [2, 4, 8])

    def test_weighted_bias_swiglu_impl_no_bias(self):
        out = weighted_bias_swiglu_impl(
            paddle.randn([4, 16]), None, paddle.randn([4, 1])
        )
        self.assertEqual(out.shape, [4, 8])

    def test_weighted_bias_swiglu_impl_with_bias_raises(self):
        with self.assertRaises(NotImplementedError):
            weighted_bias_swiglu_impl(
                paddle.randn([4, 16]),
                paddle.randn([16]),
                paddle.randn([4, 1]),
            )


class TestClampedSwiGLU(unittest.TestCase):
    """Tests for clamped_swiglu / clamped_weighted_swiglu and their backwards."""

    def test_forward_clamp_effect(self):
        """Output is bounded when clamp_value is small AND shape is correct."""
        clamp_value = 0.1
        y = paddle.full([4, 16], fill_value=100.0)
        out = clamped_swiglu(y, clamp_value=clamp_value)
        self.assertEqual(out.shape, [4, 8])
        max_possible = (
            float(F.silu(paddle.to_tensor(clamp_value)).numpy()) * clamp_value
        )
        self.assertTrue(
            float(out.abs().max().numpy()) <= max_possible + 1e-5,
        )

    def test_large_clamp_equals_standard_swiglu(self):
        """With huge clamp_value, clamped_swiglu must match standard swiglu."""
        y = paddle.randn([4, 16])
        np.testing.assert_allclose(
            clamped_swiglu(y, clamp_value=1e9).cast("float32").numpy(),
            swiglu(y).cast("float32").numpy(),
            atol=1e-4,
        )

    def test_clamped_swiglu_back_zero_at_saturation(self):
        """Backward shape matches AND saturated inputs produce zero grad."""
        y = paddle.full([2, 8], fill_value=100.0)
        g = paddle.ones([2, 4])
        grad = clamped_swiglu_back(g, y, clamp_value=1.0)
        self.assertEqual(grad.shape, [2, 8])
        np.testing.assert_allclose(
            grad.numpy(), np.zeros_like(grad.numpy()), atol=1e-6
        )

    def test_clamped_weighted_swiglu_fwd_bwd(self):
        """clamped_weighted_swiglu forward + backward shape coverage."""
        y = paddle.randn([4, 16])
        w = paddle.randn([4, 1])
        out = clamped_weighted_swiglu(y, w, clamp_value=2.0)
        self.assertEqual(out.shape, [4, 8])
        grad_y, grad_w = clamped_weighted_swiglu_back(
            paddle.randn([4, 8]), y, w, clamp_value=2.0
        )
        self.assertEqual(grad_y.shape, [4, 16])
        self.assertEqual(grad_w.shape, [4, 1])

    def test_weighted_bias_swiglu_impl_clamp_e2e(self):
        """End-to-end fwd+bwd through clamped_weighted_bias_swiglu_impl."""
        from paddlefleet.fusions.fused_bias_swiglu import (
            clamped_weighted_bias_swiglu_impl,
        )

        inp = paddle.randn([4, 16])
        inp.stop_gradient = False
        w = paddle.randn([4, 1])
        w.stop_gradient = False
        out = clamped_weighted_bias_swiglu_impl(inp, None, w, clamp_value=2.0)
        self.assertEqual(out.shape, [4, 8])
        grads = paddle.grad([out.sum()], [inp, w])
        self.assertEqual(grads[0].shape, [4, 16])
        self.assertEqual(grads[1].shape, [4, 1])
        self.assertFalse(bool(paddle.isnan(out).any().numpy()))

    def test_clamped_weighted_bias_swiglu_impl_bias_raises(self):
        """Line 438: clamped_weighted_bias_swiglu_impl with non-None bias
        raises NotImplementedError."""
        from paddlefleet.fusions.fused_bias_swiglu import (
            clamped_weighted_bias_swiglu_impl,
        )

        inp = paddle.randn([4, 16])
        w = paddle.randn([4, 1])
        bias = paddle.randn([4, 16])
        with self.assertRaises(NotImplementedError):
            clamped_weighted_bias_swiglu_impl(inp, bias, w, clamp_value=2.0)

    def test_clamped_weighted_pylayer_fwd_bwd(self):
        """ClampedWeightedSwiGLUFunction PyLayer apply + backward."""
        x = paddle.randn([4, 16]).astype("float32")
        x.stop_gradient = False
        w = paddle.ones([4, 1]).astype("float32")
        w.stop_gradient = False
        result = ClampedWeightedSwiGLUFunction.apply(x, w, False, 1.0)
        self.assertEqual(result.shape, [4, 8])
        result.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(w.grad)

    def test_weighted_pylayer_without_clamp(self):
        """Non-clamp PyLayer dispatches to weighted_swiglu_back."""
        x = paddle.randn([4, 16]).astype("float32")
        x.stop_gradient = False
        w = paddle.ones([4, 1]).astype("float32")
        w.stop_gradient = False
        mock_back = MagicMock(
            return_value=(paddle.randn([4, 16]), paddle.randn([4, 1]))
        )
        with patch(
            "paddlefleet.fusions.fused_bias_swiglu.weighted_swiglu_back",
            mock_back,
        ):
            result = WeightedSwiGLUFunction.apply(x, w, False)
            self.assertEqual(result.shape, [4, 8])
            result.sum().backward()
            mock_back.assert_called_once()


if __name__ == "__main__":
    unittest.main()
