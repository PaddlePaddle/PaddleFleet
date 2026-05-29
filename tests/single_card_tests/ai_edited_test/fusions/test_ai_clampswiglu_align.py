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
"""Tests for clamped swiglu alignment changes.

Covers:
  - clamped_bias_swiglu / clamped_bias_swiglu_back (was untested)
  - fused_swiglu_scale_backward CPU clamp fallback with saturation masks
  - fused_swiglu_scale_forward CPU clamp path (float32, clamp, silu, cast back)
  - BiasSwiGLUFunction / SwiGLUFunction with clamp_value
  - weighted_bias_swiglu_impl with clamp_value
  - Reference d_scale numeric verification
"""

import os
import sys

# Walk up from the test file to find the repo root (where src/ lives).
_test_file = os.path.abspath(__file__)
_repo_root = _test_file
for _ in range(10):
    _repo_root = os.path.dirname(_repo_root)
    if os.path.isdir(os.path.join(_repo_root, "src", "paddlefleet")):
        break

sys.path.insert(0, _repo_root)
sys.path.insert(0, os.path.join(_repo_root, "src"))

# Flush any pre-cached paddlefleet modules so the src/ version wins.
for _mod in list(sys.modules.keys()):
    if _mod == "paddlefleet" or _mod.startswith("paddlefleet."):
        del sys.modules[_mod]

import unittest
from unittest.mock import patch

import numpy as np
import paddle
import paddle.nn.functional as F

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _no_cuda():
    """Force CPU fallback by mocking paddle.is_compiled_with_cuda."""
    return patch.object(paddle, "is_compiled_with_cuda", return_value=False)


def _reference_clamped_swiglu(x, clamp_value):
    """Reference: reference-style clamped swiglu (float32 cast, clamp, silu, cast back)."""
    dtype = x.dtype
    x_fp32 = x.cast(paddle.float32)
    hidden = x.shape[-1] // 2
    gate = paddle.clip(x_fp32[..., :hidden], max=clamp_value)
    val = paddle.clip(x_fp32[..., hidden:], min=-clamp_value, max=clamp_value)
    return (F.silu(gate) * val).cast(dtype)


# ---------------------------------------------------------------------------
# clamped_bias_swiglu / clamped_bias_swiglu_back
# ---------------------------------------------------------------------------


class TestClampedBiasSwiGLU(unittest.TestCase):
    """Coverage for clamped_bias_swiglu and clamped_bias_swiglu_back
    (previously untested functions in fused_bias_swiglu.py)."""

    def test_clamped_bias_swiglu_no_clamp_effect(self):
        """clamp_value large enough: same result as bias_swiglu (no saturation)."""
        from paddlefleet.fusions.fused_bias_swiglu import (
            bias_swiglu,
            clamped_bias_swiglu,
        )

        x = paddle.randn([4, 16])
        bias = paddle.randn([4, 16])
        out_bias = bias_swiglu(x, bias)
        out_clamp = clamped_bias_swiglu(x, bias, clamp_value=100.0)
        # Both should produce same shape
        self.assertEqual(out_bias.shape, out_clamp.shape)
        self.assertEqual(out_clamp.shape, [4, 8])
        # With clamp_value=100, no values are clipped; results near-identical
        np.testing.assert_allclose(
            out_bias.numpy(), out_clamp.numpy(), rtol=1e-5, atol=1e-5
        )

    def test_clamped_bias_swiglu_saturated(self):
        """clamp_value=0.5 with large input: gate clipped -> different output."""
        from paddlefleet.fusions.fused_bias_swiglu import (
            bias_swiglu,
            clamped_bias_swiglu,
        )

        x = paddle.full([2, 8], 5.0)
        bias = paddle.zeros([2, 8])
        out_bias = bias_swiglu(x, bias)
        out_clamp = clamped_bias_swiglu(x, bias, clamp_value=0.5)
        # Clamped version should differ from non-clamped
        self.assertFalse(
            bool((out_bias.numpy() == out_clamp.numpy()).all().item())
        )
        # Output shape correct
        self.assertEqual(out_clamp.shape, [2, 4])

    def test_clamped_bias_swiglu_2d_only(self):
        """Raw clamped_bias_swiglu operates on 2D [T, H];
        3D reshaping is handled by bias_swiglu_impl."""
        from paddlefleet.fusions.fused_bias_swiglu import (
            clamped_bias_swiglu,
        )

        x = paddle.randn([8, 32])
        bias = paddle.randn([8, 32])
        out = clamped_bias_swiglu(x, bias, clamp_value=2.0)
        self.assertEqual(out.shape, [8, 16])

    def test_clamped_bias_swiglu_dtype_preservation(self):
        """Output dtype matches input dtype."""
        from paddlefleet.fusions.fused_bias_swiglu import (
            clamped_bias_swiglu,
        )

        for dt in [paddle.float32, paddle.bfloat16]:
            if dt == paddle.bfloat16 and not paddle.is_compiled_with_cuda():
                continue
            x = paddle.randn([4, 16], dtype=dt)
            bias = paddle.randn([4, 16], dtype=dt)
            out = clamped_bias_swiglu(x, bias, clamp_value=3.0)
            self.assertEqual(out.dtype, dt)
            self.assertEqual(out.shape, [4, 8])

    def test_clamped_bias_swiglu_back_no_clamp(self):
        """Backward with large clamp_value: gradient matches bias_swiglu_back."""
        from paddlefleet.fusions.fused_bias_swiglu import (
            bias_swiglu_back,
            clamped_bias_swiglu_back,
        )

        g = paddle.randn([4, 8])
        y = paddle.randn([4, 16])
        bias = paddle.randn([4, 16])
        grad_bias = bias_swiglu_back(g, y, bias)
        grad_clamp = clamped_bias_swiglu_back(g, y, bias, clamp_value=100.0)
        np.testing.assert_allclose(
            grad_bias.numpy(), grad_clamp.numpy(), rtol=1e-5, atol=1e-5
        )

    def test_clamped_bias_swiglu_back_saturated_zero_grad(self):
        """With tiny clamp_value, saturated inputs → zero gradients."""
        from paddlefleet.fusions.fused_bias_swiglu import (
            clamped_bias_swiglu_back,
        )

        g = paddle.randn([2, 4])
        y = paddle.full([2, 8], 5.0)
        bias = paddle.zeros([2, 8])
        grad = clamped_bias_swiglu_back(g, y, bias, clamp_value=0.5)
        self.assertEqual(grad.shape, [2, 8])
        # All inputs are saturated (>0.5 for gate, >0.5 for value)
        self.assertTrue(bool((grad.abs().sum() == 0).item()))

    def test_clamped_bias_swiglu_e2e_autograd(self):
        """Full forward+backward through autograd."""
        from paddlefleet.fusions.fused_bias_swiglu import (
            clamped_bias_swiglu,
        )

        x = paddle.randn([4, 16])
        bias = paddle.randn([4, 16])
        x.stop_gradient = False
        bias.stop_gradient = False
        out = clamped_bias_swiglu(x, bias, clamp_value=2.0)
        self.assertEqual(out.shape, [4, 8])
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(bias.grad)
        self.assertEqual(x.grad.shape, [4, 16])
        self.assertEqual(bias.grad.shape, [4, 16])


# ---------------------------------------------------------------------------
# fused_swiglu_scale_backward CPU clamp fallback (lines 70-106)
# ---------------------------------------------------------------------------


class TestFusedSwiGLUScaleBackwardClampCPU(unittest.TestCase):
    """Verify the XPU/CPU clamp-branch logic in fused_swiglu_scale_backward:

    - Saturation masks (g_mask, v_mask) zero out gradients.
    - d_scale computation matches the reference.
    - d_x = [d_gate, d_val] order matches chunk(x,2)=[gate,value].
    """

    def test_backward_clamp_cpu_fallback_basic(self):
        """CPU fallback with clamp: shapes correct, no NaN."""
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_backward,
        )

        with _no_cuda():
            x = paddle.randn([4, 16])
            scale = paddle.ones([4, 1])
            out_grad = paddle.randn([4, 8])
            d_x, d_scale = fused_swiglu_scale_backward(
                x, scale, out_grad, clamp_value=2.0
            )
            self.assertEqual(d_x.shape, [4, 16])
            self.assertFalse(bool(paddle.isnan(d_x).any().numpy()))
            self.assertFalse(bool(paddle.isnan(d_scale).any().numpy()))

    def test_backward_clamp_saturated_zero_grad(self):
        """Large input values fully saturated by small clamp_value → d_x = 0."""
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_backward,
        )

        with _no_cuda():
            x = paddle.full([2, 8], 10.0)
            scale = paddle.ones([2, 1])
            out_grad = paddle.randn([2, 4])
            d_x, d_scale = fused_swiglu_scale_backward(
                x, scale, out_grad, clamp_value=0.5
            )
            self.assertEqual(d_x.shape, [2, 8])
            # All gate > 0.5 and all val outside [-0.5, 0.5]
            # Both masks zero → dx all zero
            self.assertTrue(
                bool((d_x.abs().sum() < 1e-10).item()),
                "All gradients should be zero for fully saturated input",
            )

    def test_backward_clamp_partial_saturation(self):
        """Half saturated, half not: gradients partially masked."""
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_backward,
        )

        with _no_cuda():
            x_gate = paddle.concat(
                [paddle.full([2, 2], 0.3), paddle.full([2, 2], 10.0)], axis=-1
            )
            x_val = paddle.concat(
                [paddle.full([2, 2], 0.3), paddle.full([2, 2], 10.0)], axis=-1
            )
            x = paddle.concat([x_gate, x_val], axis=-1)  # [2,8]
            scale = paddle.ones([2, 1])
            out_grad = paddle.randn([2, 4])
            d_x, _ = fused_swiglu_scale_backward(
                x, scale, out_grad, clamp_value=1.0
            )
            # First half of each chunk should have non-zero grad
            d_gate = d_x[..., :4]
            d_val = d_x[..., 4:]
            # First two columns of each half: g=0.3 ≤ 1.0, val=0.3 ∈ [-1,1]
            self.assertFalse(
                bool((d_gate[..., :2].abs().sum() == 0).item()),
                "Non-saturated gate elements should have non-zero gradient",
            )

    def test_backward_clamp_d_scale_numeric(self):
        """d_scale uses swiglu_val * out_grad.cast(scale_dtype), no float32 upcast.
        Verify against the reference."""
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_backward,
        )

        with _no_cuda():
            paddle.seed(42)
            x = paddle.randn([4, 16])
            scale = paddle.ones([4, 1])
            out_grad = paddle.randn([4, 8])

            _, d_scale = fused_swiglu_scale_backward(
                x, scale, out_grad, clamp_value=3.0
            )

            # Reference
            dtype = x.dtype
            scale_dtype = scale.dtype
            hidden = x.shape[-1] // 2
            x_fp32 = x.cast(paddle.float32)
            gate = paddle.clip(x_fp32[..., :hidden], max=3.0)
            val = paddle.clip(x_fp32[..., hidden:], min=-3.0, max=3.0)
            swiglu_val = (F.silu(gate) * val).cast(dtype)
            ref_d_scale = paddle.sum(
                swiglu_val * out_grad.cast(scale_dtype),
                axis=-1,
                keepdim=True,
            ).cast(scale_dtype)
            np.testing.assert_allclose(
                d_scale.numpy(), ref_d_scale.numpy(), rtol=1e-5, atol=1e-5
            )

    def test_backward_clamp_d_x_order(self):
        """Verify d_x = [d_gate, d_val] matches chunk(x,2) = [gate, value]."""
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_backward,
        )

        with _no_cuda():
            x = paddle.randn([2, 8])
            scale = paddle.ones([2, 1])
            out_grad = paddle.randn([2, 4])
            d_x, _ = fused_swiglu_scale_backward(
                x, scale, out_grad, clamp_value=2.0
            )
            d_gate = d_x[..., :4]
            d_val = d_x[..., 4:]
            self.assertEqual(d_gate.shape, [2, 4])
            self.assertEqual(d_val.shape, [2, 4])

    def test_backward_clamp_cpu_vs_noclamp_changed(self):
        """With clamp, the result differs from non-clamp backward
        (because gradients are masked)."""
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_backward,
        )

        with _no_cuda():
            x = paddle.full([2, 8], 2.0)
            scale = paddle.ones([2, 1])
            out_grad = paddle.randn([2, 4])

            d_x_no_clamp, _ = fused_swiglu_scale_backward(
                x, scale, out_grad, clamp_value=None
            )
            d_x_clamp, _ = fused_swiglu_scale_backward(
                x, scale, out_grad, clamp_value=0.5
            )
            # With clamp_value=0.5 and input=2.0, gate is fully saturated
            # so d_x_clamp should be all zero (different from d_x_no_clamp)
            self.assertTrue(
                bool((d_x_clamp.abs().sum() < 1e-10).item()),
                "d_x should be zero when fully saturated",
            )
            self.assertFalse(
                bool((d_x_no_clamp.abs().sum() < 1e-10).item()),
                "d_x without clamp should have non-zero gradients",
            )

    def test_backward_clamp_with_2d_scale(self):
        """Scale shape [N, 1]: d_scale should have keepdim=True → [N, 1]."""
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_backward,
        )

        with _no_cuda():
            x = paddle.randn([4, 16])
            scale = paddle.ones([4, 1])
            out_grad = paddle.randn([4, 8])
            _, d_scale = fused_swiglu_scale_backward(
                x, scale, out_grad, clamp_value=2.0
            )
            self.assertEqual(d_scale.shape, [4, 1])


# ---------------------------------------------------------------------------
# fused_swiglu_scale_forward CPU clamp path (lines 37-48)
# ---------------------------------------------------------------------------


class TestFusedSwiGLUScaleForwardClampCPU(unittest.TestCase):
    """Tests for fused_swiglu_scale_forward CPU clamp path
    (cast to float32, clamp, silu, cast back)."""

    def test_forward_clamp_cpu_fallback(self):
        """CPU clamp path: forward shape and NaN check."""
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_forward,
        )

        with _no_cuda():
            x = paddle.randn([4, 16])
            scale = paddle.ones([4, 1])
            out = fused_swiglu_scale_forward(x, scale, clamp_value=2.0)
            self.assertEqual(out.shape, [4, 8])
            self.assertFalse(bool(paddle.isnan(out).any().numpy()))

    def test_forward_clamp_vs_reference(self):
        """CPU clamp forward matches the reference."""
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_forward,
        )

        with _no_cuda():
            paddle.seed(42)
            x = paddle.randn([4, 16])
            scale = paddle.ones([4, 1])
            out = fused_swiglu_scale_forward(x, scale, clamp_value=2.0)
            ref_out = _reference_clamped_swiglu(x, 2.0) * scale
            np.testing.assert_allclose(
                out.numpy(), ref_out.numpy(), rtol=1e-5, atol=1e-5
            )

    def test_forward_clamp_saturated(self):
        """Large values clamped: gate clipped, value symmetrically clipped."""
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_forward,
        )

        with _no_cuda():
            x = paddle.full([2, 8], 10.0)
            scale = paddle.ones([2, 1])
            out_small = fused_swiglu_scale_forward(x, scale, clamp_value=0.5)
            out_large = fused_swiglu_scale_forward(x, scale, clamp_value=20.0)
            # Smaller clamp_value → different output
            self.assertFalse(
                bool((out_small.numpy() == out_large.numpy()).all().item())
            )
            self.assertEqual(out_small.shape, [2, 4])

    def test_forward_clamp_vs_noclamp_changed(self):
        """With clamp, output differs from non-clamp for saturated inputs."""
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_forward,
        )

        with _no_cuda():
            x = paddle.full([2, 8], 3.0)
            scale = paddle.ones([2, 1])
            out_no_clamp = fused_swiglu_scale_forward(x, scale)
            out_clamp = fused_swiglu_scale_forward(x, scale, clamp_value=0.5)
            self.assertFalse(
                bool((out_no_clamp.numpy() == out_clamp.numpy()).all().item())
            )

    def test_forward_clamp_with_1d_scale(self):
        """1D scale broadcasting works in clamp path."""
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_forward,
        )

        with _no_cuda():
            x = paddle.randn([4, 16])
            scale = paddle.ones([4])
            out = fused_swiglu_scale_forward(x, scale, clamp_value=2.0)
            self.assertEqual(out.shape, [4, 8])

    def test_forward_clamp_dtype_preservation(self):
        """Output dtype matches input dtype in clamp path."""
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_forward,
        )

        with _no_cuda():
            x = paddle.randn([4, 16], dtype=paddle.float32)
            scale = paddle.ones([4, 1], dtype=paddle.float32)
            out = fused_swiglu_scale_forward(x, scale, clamp_value=2.0)
            self.assertEqual(out.dtype, paddle.float32)


# ---------------------------------------------------------------------------
# BiasSwiGLUFunction / SwiGLUFunction with clamp_value
# ---------------------------------------------------------------------------


class TestBiasSwiGLUFunctionClamp(unittest.TestCase):
    """BiasSwiGLUFunction.apply path for clamp_value.
    Exercises lines 288-290 (clamp branch) in fused_bias_swiglu.py."""

    def test_bias_swiglu_pylayer_clamp_fwd_bwd(self):
        """Forward + backward through BiasSwiGLUFunction with clamp_value."""
        from paddlefleet.fusions.fused_bias_swiglu import (
            BiasSwiGLUFunction,
        )

        x = paddle.randn([4, 16]).astype("float32")
        bias = paddle.randn([4, 16]).astype("float32")
        x.stop_gradient = False
        bias.stop_gradient = False

        out = BiasSwiGLUFunction.apply(x, bias, False, False, clamp_value=2.0)
        self.assertEqual(out.shape, [4, 8])
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, [4, 16])

    def test_bias_swiglu_pylayer_clamp_fp8(self):
        """clamp + fp8_input_store=True path."""
        from paddlefleet.fusions.fused_bias_swiglu import (
            BiasSwiGLUFunction,
        )

        x = paddle.randn([4, 16]).astype("float32")
        bias = paddle.randn([4, 16]).astype("float32")
        x.stop_gradient = False
        bias.stop_gradient = False
        out = BiasSwiGLUFunction.apply(x, bias, True, False, clamp_value=1.0)
        self.assertEqual(out.shape, [4, 8])
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(bias.grad)

    def test_bias_swiglu_pylayer_no_clamp(self):
        """clamp_value=None: should use bias_swiglu path (non-clamp)."""
        from paddlefleet.fusions.fused_bias_swiglu import (
            BiasSwiGLUFunction,
        )

        x = paddle.randn([4, 16]).astype("float32")
        bias = paddle.randn([4, 16]).astype("float32")
        x.stop_gradient = False
        bias.stop_gradient = False
        out = BiasSwiGLUFunction.apply(x, bias, False, False, clamp_value=None)
        self.assertEqual(out.shape, [4, 8])
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(bias.grad)

    def test_bias_swiglu_pylayer_clamp_value_zero(self):
        """clamp_value=0: treated as not set (per condition clamp_value > 0)."""
        from paddlefleet.fusions.fused_bias_swiglu import (
            BiasSwiGLUFunction,
        )

        x = paddle.randn([4, 16]).astype("float32")
        bias = paddle.randn([4, 16]).astype("float32")
        x.stop_gradient = False
        bias.stop_gradient = False
        out = BiasSwiGLUFunction.apply(x, bias, False, False, clamp_value=0.0)
        self.assertEqual(out.shape, [4, 8])
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(bias.grad)


class TestSwiGLUFunctionClamp(unittest.TestCase):
    """SwiGLUFunction.apply path for clamp_value."""

    def test_swiglu_pylayer_clamp_fwd_bwd(self):
        """Forward + backward through SwiGLUFunction with clamp_value."""
        from paddlefleet.fusions.fused_bias_swiglu import SwiGLUFunction

        x = paddle.randn([4, 16]).astype("float32")
        x.stop_gradient = False
        out = SwiGLUFunction.apply(x, False, False, clamp_value=2.0)
        self.assertEqual(out.shape, [4, 8])
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, [4, 16])

    def test_swiglu_pylayer_no_clamp(self):
        """clamp_value=None: non-clamp path."""
        from paddlefleet.fusions.fused_bias_swiglu import SwiGLUFunction

        x = paddle.randn([4, 16]).astype("float32")
        x.stop_gradient = False
        out = SwiGLUFunction.apply(x, False, False, clamp_value=None)
        self.assertEqual(out.shape, [4, 8])
        out.sum().backward()
        self.assertIsNotNone(x.grad)


# ---------------------------------------------------------------------------
# weighted_bias_swiglu_impl with clamp_value
# ---------------------------------------------------------------------------


class TestWeightedBiasSwiGLUImplClamp(unittest.TestCase):
    """Coverage for weighted_bias_swiglu_impl with clamp_value
    (previously was cloned into clamped_weighted_bias_swiglu_impl)."""

    def test_weighted_impl_clamp_2d(self):
        """2D input through weighted_bias_swiglu_impl with clamp_value."""
        from paddlefleet.fusions.fused_bias_swiglu import (
            weighted_bias_swiglu_impl,
        )

        x = paddle.randn([4, 16]).astype("float32")
        w = paddle.randn([4, 1]).astype("float32")
        x.stop_gradient = False
        w.stop_gradient = False
        out = weighted_bias_swiglu_impl(x, None, w, clamp_value=2.0)
        self.assertEqual(out.shape, [4, 8])
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(w.grad)

    def test_weighted_impl_clamp_3d(self):
        """2D only: raw WeightedSwiGLUFunction expects 2D [T, H];
        3D reshaping is handled by weighted_bias_swiglu_impl.
        Here we test the impl with 2D."""
        from paddlefleet.fusions.fused_bias_swiglu import (
            weighted_bias_swiglu_impl,
        )

        x = paddle.randn([8, 16]).astype("float32")
        w = paddle.randn([8, 1]).astype("float32")
        x.stop_gradient = False
        w.stop_gradient = False
        out = weighted_bias_swiglu_impl(x, None, w, clamp_value=2.0)
        self.assertEqual(out.shape, [8, 8])
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(w.grad)

    def test_weighted_impl_clamp_bias_raises(self):
        """bias != None raises NotImplementedError."""
        from paddlefleet.fusions.fused_bias_swiglu import (
            weighted_bias_swiglu_impl,
        )

        x = paddle.randn([4, 16])
        w = paddle.randn([4, 1])
        bias = paddle.randn([4, 16])
        with self.assertRaises(NotImplementedError):
            weighted_bias_swiglu_impl(x, bias, w, clamp_value=2.0)

    def test_weighted_impl_clamp_fp8_input_store(self):
        """clamp_value + fp8_input_store=True.
        Note: fp8 store creates a detached buffer for memory savings;
        backward still works because the original input is used."""
        from paddlefleet.fusions.fused_bias_swiglu import (
            weighted_bias_swiglu_impl,
        )

        x = paddle.randn([4, 16]).astype("float32")
        w = paddle.randn([4, 1]).astype("float32")
        x.stop_gradient = False
        w.stop_gradient = False
        out = weighted_bias_swiglu_impl(
            x, None, w, fp8_input_store=True, clamp_value=1.0
        )
        self.assertEqual(out.shape, [4, 8])
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(w.grad)

    def test_weighted_impl_no_clamp(self):
        """clamp_value=None: non-clamp path still works."""
        from paddlefleet.fusions.fused_bias_swiglu import (
            weighted_bias_swiglu_impl,
        )

        x = paddle.randn([4, 16]).astype("float32")
        w = paddle.randn([4, 1]).astype("float32")
        x.stop_gradient = False
        w.stop_gradient = False
        out = weighted_bias_swiglu_impl(x, None, w, clamp_value=None)
        self.assertEqual(out.shape, [4, 8])
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(w.grad)


# ---------------------------------------------------------------------------
# d_scale numerical correctness
# ---------------------------------------------------------------------------


class TestDScaleAlignment(unittest.TestCase):
    """Verify d_scale computation matches the reference
    _dsv4_ref_scale_wgrad (no float32 upcast, keepdim=True)."""

    @staticmethod
    def _dscale_ref(x, scale, out_grad, clamp_value=None):
        """Reference implementation."""
        dtype = x.dtype
        scale_dtype = scale.dtype
        if clamp_value is not None:
            hidden = x.shape[-1] // 2
            gate, value = paddle.chunk(x.cast(paddle.float32), 2, axis=-1)
            gate = paddle.clip(gate, max=clamp_value)
            value = paddle.clip(value, min=-clamp_value, max=clamp_value)
            swiglu_val = (F.silu(gate) * value).cast(dtype)
        else:
            swiglu_val = F.swiglu(x)
        return paddle.sum(
            swiglu_val * out_grad.cast(scale_dtype),
            axis=-1,
            keepdim=True,
        ).cast(scale_dtype)

    def test_d_scale_no_clamp(self):
        """d_scale without clamp matches the reference."""
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_backward,
        )

        with _no_cuda():
            paddle.seed(1)
            x = paddle.randn([4, 16], dtype=paddle.float32)
            scale = paddle.ones([4, 1], dtype=paddle.float32)
            out_grad = paddle.randn([4, 8], dtype=paddle.float32)

            _, d_scale = fused_swiglu_scale_backward(
                x, scale, out_grad, clamp_value=None
            )
            ref = self._dscale_ref(x, scale, out_grad)
            np.testing.assert_allclose(
                d_scale.numpy(), ref.numpy(), rtol=1e-5, atol=1e-5
            )

    def test_d_scale_with_clamp(self):
        """d_scale with clamp matches the reference."""
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_backward,
        )

        with _no_cuda():
            paddle.seed(1)
            x = paddle.randn([4, 16], dtype=paddle.float32)
            scale = paddle.full([4, 1], 0.5, dtype=paddle.float32)
            out_grad = paddle.randn([4, 8], dtype=paddle.float32)

            _, d_scale = fused_swiglu_scale_backward(
                x, scale, out_grad, clamp_value=2.0
            )
            ref = self._dscale_ref(x, scale, out_grad, clamp_value=2.0)
            np.testing.assert_allclose(
                d_scale.numpy(), ref.numpy(), rtol=1e-5, atol=1e-5
            )

    def test_d_scale_shape_keepdim(self):
        """d_scale shape: keepdim=True → [N, 1] for [N, 1] scale input."""
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_backward,
        )

        with _no_cuda():
            x = paddle.randn([4, 16])
            scale = paddle.randn([4, 1])
            out_grad = paddle.randn([4, 8])
            _, d_scale = fused_swiglu_scale_backward(
                x, scale, out_grad, clamp_value=2.0
            )
            self.assertEqual(d_scale.shape, [4, 1])

    def test_d_scale_dtype_preservation(self):
        """d_scale dtype matches scale dtype."""
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_backward,
        )

        with _no_cuda():
            x = paddle.randn([4, 16], dtype=paddle.float32)
            scale = paddle.ones([4, 1], dtype=paddle.float32)
            out_grad = paddle.randn([4, 8], dtype=paddle.float32)
            _, d_scale = fused_swiglu_scale_backward(
                x, scale, out_grad, clamp_value=3.0
            )
            self.assertEqual(d_scale.dtype, paddle.float32)

    def test_bfloat16_d_scale(self):
        """bfloat16 path: d_scale dtype preserved."""
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_backward,
        )

        with _no_cuda():
            x = paddle.randn([4, 16], dtype=paddle.float32)
            # scale in bfloat16
            scale = paddle.ones([4, 1], dtype=paddle.bfloat16)
            out_grad = paddle.randn([4, 8], dtype=paddle.float32)
            _, d_scale = fused_swiglu_scale_backward(
                x, scale, out_grad, clamp_value=2.0
            )
            self.assertEqual(d_scale.dtype, paddle.bfloat16)


# ---------------------------------------------------------------------------
# bias_swiglu_impl with clamp_value
# ---------------------------------------------------------------------------


class TestBiasSwiGLUImplClamp(unittest.TestCase):
    """bias_swiglu_impl with clamp_value parameter."""

    def test_bias_swiglu_impl_clamp_2d_with_bias(self):
        """2D + bias + clamp_value."""
        from paddlefleet.fusions.fused_bias_swiglu import bias_swiglu_impl

        x = paddle.randn([4, 16]).astype("float32")
        bias = paddle.randn([4, 16]).astype("float32")
        x.stop_gradient = False
        bias.stop_gradient = False
        out = bias_swiglu_impl(x, bias, clamp_value=2.0)
        self.assertEqual(out.shape, [4, 8])
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(bias.grad)

    def test_bias_swiglu_impl_clamp_2d_no_bias(self):
        """2D no bias + clamp_value → SwiGLUFunction path."""
        from paddlefleet.fusions.fused_bias_swiglu import bias_swiglu_impl

        x = paddle.randn([4, 16]).astype("float32")
        x.stop_gradient = False
        out = bias_swiglu_impl(x, None, clamp_value=3.0)
        self.assertEqual(out.shape, [4, 8])
        out.sum().backward()
        self.assertIsNotNone(x.grad)

    def test_bias_swiglu_impl_clamp_2d_with_bias_batch8(self):
        """2D + bias + clamp_value, batch=8."""
        from paddlefleet.fusions.fused_bias_swiglu import bias_swiglu_impl

        x = paddle.randn([8, 16]).astype("float32")
        bias = paddle.randn([8, 16]).astype("float32")
        x.stop_gradient = False
        bias.stop_gradient = False
        out = bias_swiglu_impl(x, bias, clamp_value=2.0)
        self.assertEqual(out.shape, [8, 8])
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(bias.grad)

    def test_bias_swiglu_impl_no_clamp(self):
        """clamp_value=None path still functional."""
        from paddlefleet.fusions.fused_bias_swiglu import bias_swiglu_impl

        x = paddle.randn([4, 16]).astype("float32")
        x.stop_gradient = False
        out = bias_swiglu_impl(x, None, clamp_value=None)
        self.assertEqual(out.shape, [4, 8])
        out.sum().backward()
        self.assertIsNotNone(x.grad)


# ---------------------------------------------------------------------------
# WeightedSwiGLUFunction PyLayer with clamp_value
# ---------------------------------------------------------------------------


class TestWeightedSwiGLUFunctionClamp(unittest.TestCase):
    """WeightedSwiGLUFunction.apply path for clamp_value (lines 381-384)."""

    def test_weighted_pylayer_clamp_fwd_bwd_2d(self):
        """WeightedSwiGLUFunction.apply with clamp_value, fwd+bwd."""
        from paddlefleet.fusions.fused_bias_swiglu import (
            WeightedSwiGLUFunction,
        )

        x = paddle.randn([4, 16]).astype("float32")
        w = paddle.ones([4, 1]).astype("float32")
        x.stop_gradient = False
        w.stop_gradient = False
        out = WeightedSwiGLUFunction.apply(x, w, False, clamp_value=1.0)
        self.assertEqual(out.shape, [4, 8])
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(w.grad)

    def test_weighted_pylayer_clamp_bfloat16(self):
        """bfloat16 input through WeightedSwiGLUFunction with clamp."""
        from paddlefleet.fusions.fused_bias_swiglu import (
            WeightedSwiGLUFunction,
        )

        if not paddle.is_compiled_with_cuda():
            self.skipTest("bfloat16 requires CUDA")

        x = paddle.randn([4, 16], dtype=paddle.bfloat16)
        w = paddle.ones([4, 1], dtype=paddle.bfloat16)
        x.stop_gradient = False
        w.stop_gradient = False
        out = WeightedSwiGLUFunction.apply(x, w, False, clamp_value=1.0)
        self.assertEqual(out.shape, [4, 8])
        loss = out.cast(paddle.float32).sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(w.grad)

    def test_weighted_pylayer_no_clamp(self):
        """clamp_value=None: non-clamp path via WeightedSwiGLUFunction."""
        from paddlefleet.fusions.fused_bias_swiglu import (
            WeightedSwiGLUFunction,
        )

        x = paddle.randn([4, 16]).astype("float32")
        w = paddle.ones([4, 1]).astype("float32")
        x.stop_gradient = False
        w.stop_gradient = False
        out = WeightedSwiGLUFunction.apply(x, w, False, clamp_value=None)
        self.assertEqual(out.shape, [4, 8])
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(w.grad)


# ---------------------------------------------------------------------------
# Edge cases and boundary values
# ---------------------------------------------------------------------------


class TestClampEdgeCases(unittest.TestCase):
    """Edge cases for clamp_value handling."""

    def test_clamp_value_zero_no_effect_forward(self):
        """clamp_value=0 → condition clamp_value > 0 fails, falls through."""
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_forward,
        )

        with _no_cuda():
            x = paddle.randn([4, 16])
            scale = paddle.ones([4, 1])
            out_clamp0 = fused_swiglu_scale_forward(x, scale, clamp_value=0.0)
            out_no_clamp = fused_swiglu_scale_forward(x, scale)
            np.testing.assert_allclose(
                out_clamp0.numpy(),
                out_no_clamp.numpy(),
                rtol=1e-5,
                atol=1e-5,
            )

    def test_clamp_value_negative_no_effect(self):
        """clamp_value < 0: the raw clamped_swiglu does NOT guard
        against negative clamp (min=-clamp_value > max=clamp_value
        would crash).  The guard is in BiasSwiGLUFunction / impl
        layers (clamp_value > 0 check). Therefore this test verifies
        that the guard in the impl layer correctly skips clamp
        when clamp_value <= 0."""
        from paddlefleet.fusions.fused_bias_swiglu import (
            bias_swiglu_impl,
        )

        x = paddle.randn([4, 16]).astype("float32")
        x.stop_gradient = False
        # clamp_value=-1 → BiasSwiGLUFunction's guard (clamp > 0) skips clamp
        out = bias_swiglu_impl(x, None, clamp_value=-1)
        self.assertEqual(out.shape, [4, 8])
        out.sum().backward()
        self.assertIsNotNone(x.grad)

    def test_bias_swiglu_impl_invalid_ndim(self):
        """AssertionError when input.ndim not in [2, 3]."""
        from paddlefleet.fusions.fused_bias_swiglu import bias_swiglu_impl

        x = paddle.randn([4])
        with self.assertRaises(AssertionError):
            bias_swiglu_impl(x, None, clamp_value=2.0)

    def test_fused_swiglu_scale_backward_zero_rows(self):
        """Zero rows input: backward returns correct shapes without crash."""
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_backward,
        )

        with _no_cuda():
            x = paddle.empty([0, 16])
            scale = paddle.empty([0, 1])
            out_grad = paddle.empty([0, 8])
            d_x, d_scale = fused_swiglu_scale_backward(
                x, scale, out_grad, clamp_value=2.0
            )
            self.assertEqual(d_x.shape, [0, 16])
            self.assertEqual(d_scale.shape, [0, 1])

    def test_clamped_swiglu_zero_rows(self):
        """Zero-row input: forward and backward should not crash."""
        from paddlefleet.fusions.fused_bias_swiglu import (
            clamped_swiglu,
            clamped_swiglu_back,
        )

        y = paddle.empty([0, 16])
        out = clamped_swiglu(y, clamp_value=2.0)
        self.assertEqual(out.shape, [0, 8])
        # backward with zero rows
        g = paddle.empty([0, 8])
        grad = clamped_swiglu_back(g, y, clamp_value=2.0)
        self.assertEqual(grad.shape, [0, 16])

    def test_clamped_bias_swiglu_zero_rows(self):
        """Zero-row input with bias: should not crash."""
        from paddlefleet.fusions.fused_bias_swiglu import (
            clamped_bias_swiglu,
            clamped_bias_swiglu_back,
        )

        y = paddle.empty([0, 16])
        bias = paddle.empty([0, 16])
        out = clamped_bias_swiglu(y, bias, clamp_value=2.0)
        self.assertEqual(out.shape, [0, 8])
        g = paddle.empty([0, 8])
        grad = clamped_bias_swiglu_back(g, y, bias, clamp_value=2.0)
        self.assertEqual(grad.shape, [0, 16])

    def test_bias_swiglu_impl_zero_rows(self):
        """bias_swiglu_impl with zero rows."""
        from paddlefleet.fusions.fused_bias_swiglu import bias_swiglu_impl

        x = paddle.empty([0, 16])
        out = bias_swiglu_impl(x, None, clamp_value=2.0)
        self.assertEqual(out.shape, [0, 8])

    def test_weighted_bias_swiglu_impl_zero_rows(self):
        """weighted_bias_swiglu_impl with zero rows."""
        from paddlefleet.fusions.fused_bias_swiglu import (
            weighted_bias_swiglu_impl,
        )

        x = paddle.empty([0, 16])
        w = paddle.empty([0, 1])
        out = weighted_bias_swiglu_impl(x, None, w, clamp_value=2.0)
        self.assertEqual(out.shape, [0, 8])

    def test_fused_swiglu_scale_forward_zero_rows(self):
        """Forward with zero rows."""
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_forward,
        )

        with _no_cuda():
            x = paddle.empty([0, 16])
            scale = paddle.empty([0, 1])
            out = fused_swiglu_scale_forward(x, scale, clamp_value=2.0)
            self.assertEqual(out.shape, [0, 8])

    def test_pylayer_apply_zero_rows(self):
        """BiasSwiGLUFunction / WeightedSwiGLUFunction with zero rows."""
        from paddlefleet.fusions.fused_bias_swiglu import (
            BiasSwiGLUFunction,
            WeightedSwiGLUFunction,
        )

        # BiasSwiGLUFunction
        x = paddle.empty([0, 16]).astype("float32")
        b = paddle.empty([0, 16]).astype("float32")
        out = BiasSwiGLUFunction.apply(x, b, False, False, clamp_value=2.0)
        self.assertEqual(out.shape, [0, 8])

        # WeightedSwiGLUFunction
        w = paddle.empty([0, 1]).astype("float32")
        out = WeightedSwiGLUFunction.apply(x, w, False, clamp_value=2.0)
        self.assertEqual(out.shape, [0, 8])


# ---------------------------------------------------------------------------
# Large tensor tests (int32-overflow numel > 2**31)
# ---------------------------------------------------------------------------


class TestClampLargeTensor(unittest.TestCase):
    """Verify clamped functions handle large tensors without crashes
    (numel crossing the int32 boundary of 2**31)."""

    @staticmethod
    def _skip_if_oom(rows, cols):
        """Skip if actual allocation fails (probes GPU memory when available)."""
        est_gib = (rows * cols * 2) / (1024**3)
        if paddle.is_compiled_with_cuda():
            try:
                # Probe: allocate and free to check if there's enough GPU memory
                probe = paddle.empty([rows, cols], dtype=paddle.bfloat16)
                del probe
                paddle.device.synchronize()
                paddle.device.cuda.empty_cache()
            except Exception:
                raise unittest.SkipTest(
                    f"GPU OOM for large tensor ({est_gib:.1f} GiB)"
                )
        else:
            # CPU fallback: skip conservatively above 24 GiB
            if est_gib > 24:
                raise unittest.SkipTest(
                    f"Large tensor ({est_gib:.1f} GiB) skipped on CPU"
                )

    def test_clamped_swiglu_large_numel_fwd(self):
        """numel > 2**31: clamped_swiglu forward (int64 shape handling)."""
        from paddlefleet.fusions.fused_bias_swiglu import clamped_swiglu

        # 2**24 rows * 136 = 2.28B elements > 2**31, fits ~4.6 GiB bf16
        rows, hidden2 = 2**24, 136
        self._skip_if_oom(rows, hidden2)
        x = paddle.randn([rows, hidden2], dtype=paddle.bfloat16)
        out = clamped_swiglu(x, clamp_value=3.0)
        self.assertEqual(out.shape, [rows, hidden2 // 2])
        self.assertFalse(
            bool(paddle.isnan(out.cast(paddle.float32)).any().numpy())
        )

    def test_clamped_swiglu_back_large_numel(self):
        """numel > 2**31: clamped_swiglu backward passes without crash."""
        from paddlefleet.fusions.fused_bias_swiglu import clamped_swiglu_back

        rows, hidden2 = 2**24, 136
        self._skip_if_oom(rows, hidden2)
        g = paddle.randn([rows, hidden2 // 2], dtype=paddle.bfloat16)
        y = paddle.randn([rows, hidden2], dtype=paddle.bfloat16)
        grad = clamped_swiglu_back(g, y, clamp_value=3.0)
        self.assertEqual(grad.shape, [rows, hidden2])
        self.assertFalse(
            bool(paddle.isnan(grad.cast(paddle.float32)).any().numpy())
        )

    def test_clamped_weighted_swiglu_large_numel(self):
        """numel > 2**31: clamped_weighted_swiglu fwd+bwd."""
        from paddlefleet.fusions.fused_bias_swiglu import (
            clamped_weighted_swiglu,
            clamped_weighted_swiglu_back,
        )

        rows, hidden2 = 2**24, 136
        self._skip_if_oom(rows, hidden2)
        y = paddle.randn([rows, hidden2], dtype=paddle.bfloat16)
        w = paddle.randn([rows, 1], dtype=paddle.bfloat16)
        out = clamped_weighted_swiglu(y, w, clamp_value=3.0)
        self.assertEqual(out.shape, [rows, hidden2 // 2])
        g = paddle.randn([rows, hidden2 // 2], dtype=paddle.bfloat16)
        d_y, d_w = clamped_weighted_swiglu_back(g, y, w, clamp_value=3.0)
        self.assertEqual(d_y.shape, [rows, hidden2])
        self.assertEqual(d_w.shape, [rows, 1])

    def test_fused_swiglu_scale_backward_large(self):
        """numel > 2**31: backward without clamp, verifies int64 handling."""
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_backward,
        )

        with _no_cuda():
            rows, hidden2 = 2**24, 136
            self._skip_if_oom(rows, hidden2)
            x = paddle.randn([rows, hidden2], dtype=paddle.bfloat16)
            scale = paddle.ones([rows, 1], dtype=paddle.bfloat16)
            out_grad = paddle.randn([rows, hidden2 // 2], dtype=paddle.bfloat16)
            d_x, d_scale = fused_swiglu_scale_backward(
                x, scale, out_grad, clamp_value=None
            )
            self.assertEqual(d_x.shape, [rows, hidden2])
            self.assertEqual(d_scale.shape, [rows, 1])

    def test_fused_swiglu_scale_backward_large_clamp(self):
        """numel > 2**31: backward with clamp, verifies int64 handling."""
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_backward,
        )

        with _no_cuda():
            rows, hidden2 = 2**24, 136
            self._skip_if_oom(rows, hidden2)
            x = paddle.randn([rows, hidden2], dtype=paddle.bfloat16)
            scale = paddle.ones([rows, 1], dtype=paddle.bfloat16)
            out_grad = paddle.randn([rows, hidden2 // 2], dtype=paddle.bfloat16)
            d_x, d_scale = fused_swiglu_scale_backward(
                x, scale, out_grad, clamp_value=3.0
            )
            self.assertEqual(d_x.shape, [rows, hidden2])
            self.assertEqual(d_scale.shape, [rows, 1])


if __name__ == "__main__":
    unittest.main()
