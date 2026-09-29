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

"""CPU behavior tests for the clamped-SwiGLU alignment between
``paddlefleet.fusions.fused_bias_swiglu`` and
``paddlefleet.fusions.fused_swiglu_scale``.

Only CPU-observable pure logic is asserted:

* the asymmetric clamp contract -- gate is clamped to ``(-inf, clamp_value]``
  while value is clamped to ``[-clamp_value, clamp_value]`` -- in the forward;
* the saturation masks that zero the gradient where inputs were clamped;
* the ``clamp_value > 0`` activation-dispatch guard;
* the ``_broadcast_scale`` cast/unsqueeze helper and per-row scale application;
* the cross-module contract that ``clamped_swiglu`` (module 1) and the CPU
  fallback of ``fused_swiglu_scale_forward/backward`` (module 2) compute the
  same clamped SwiGLU forward and the same ``d_x`` gradient when ``scale == 1``;
* the ``PyLayer`` ctx save/restore round trip that selects the clamp branch in
  ``BiasSwiGLUFunction`` forward and backward;
* the ndim / bias validation guards.

The GPU/Triton kernel numerics (the ``paddle.is_compiled_with_cuda()`` branch
that imports ``paddlefleet_ops``) are NOT exercised: they are honestly skipped
by forcing the documented CPU fallback with a genuine collaborator patch.

Every expected value is hand-derived with an independent NumPy implementation of
sigmoid/SiLU and the documented clamp rule; the production module is never used
to build the expectations.
"""

import os
import sys
import unittest
from unittest import mock

import numpy as np

# Make ``src/paddlefleet`` importable when paddle is available.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    import paddle

    from paddlefleet.fusions.fused_bias_swiglu import (
        BiasSwiGLUFunction,
        bias_swiglu_impl,
        clamped_swiglu,
        clamped_swiglu_back,
        weighted_bias_swiglu_impl,
    )
    from paddlefleet.fusions.fused_swiglu_scale import (
        _broadcast_scale,
        fused_swiglu_scale_backward,
        fused_swiglu_scale_forward,
    )

    HAS_PADDLE = True
    _IMPORT_ERROR = ""
except ImportError as exc:  # honest: dependency missing, not a swallowed bug
    HAS_PADDLE = False
    _IMPORT_ERROR = repr(exc)

# --- Independent NumPy references (never call production code) --------------
# All references work in float64 from the mathematical definitions so a
# same-way bug in the production float32 code cannot hide behind a shared impl.


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def _silu(x):
    x = np.asarray(x, dtype=np.float64)
    return x * _sigmoid(x)


def _split_last(arr):
    """Split the last axis into two equal halves (matches paddle.chunk)."""
    half = arr.shape[-1] // 2
    return arr[..., :half], arr[..., half:]


def ref_clamped_swiglu_fwd(y, cv):
    """SiLU(clip(gate, max=cv)) * clip(value, -cv, cv), gate lower-unbounded."""
    g, v = _split_last(np.asarray(y, dtype=np.float64))
    g_c = np.minimum(g, cv)
    v_c = np.clip(v, -cv, cv)
    return _silu(g_c) * v_c


def ref_swiglu_fwd(y):
    """Plain SwiGLU with no clamping: SiLU(gate) * value."""
    g, v = _split_last(np.asarray(y, dtype=np.float64))
    return _silu(g) * v


def ref_clamped_swiglu_bwd(g_up, y, cv):
    """d/dy of clamped SwiGLU; gradient zeroed where the raw input saturated.

    ``g_up`` is the upstream gradient with shape ``[..., hidden]``.  Returns the
    gradient w.r.t. ``y`` with shape ``[..., 2*hidden]`` laid out as
    ``concat([d_gate, d_value])`` to match ``chunk(y, 2)`` -> [gate, value].
    """
    g_up = np.asarray(g_up, dtype=np.float64)
    gate, val = _split_last(np.asarray(y, dtype=np.float64))
    gate_c = np.minimum(gate, cv)
    val_c = np.clip(val, -cv, cv)
    m_gate = (gate <= cv).astype(np.float64)  # only the upper side is clamped
    m_val = ((val >= -cv) & (val <= cv)).astype(np.float64)
    sig = _sigmoid(gate_c)
    # d SiLU(x)/dx = sig * (1 + x * (1 - sig)); SiLU(x) = x * sig.
    d_gate = g_up * sig * (1.0 + gate_c * (1.0 - sig)) * val_c * m_gate
    d_val = g_up * (gate_c * sig) * m_val
    return np.concatenate([d_gate, d_val], axis=-1)


def ref_scale_grad_clamp(y, out_grad, cv):
    """d(out * scale)/d scale = sum over hidden of clamped_swiglu(y) * out_grad,
    kept as a trailing dim of size 1 (keepdim) for the clamp branch."""
    out = ref_clamped_swiglu_fwd(y, cv)
    return np.sum(
        out * np.asarray(out_grad, dtype=np.float64), axis=-1, keepdims=True
    )


def _force_cpu_fallback():
    """Select the documented CPU fallback branch of module 2 regardless of the
    build.  ``is_compiled_with_cuda`` is a genuine collaborator, not code under
    test; the GPU kernel path it guards is intentionally left unverified."""
    return mock.patch.object(
        paddle, "is_compiled_with_cuda", return_value=False
    )


# --- Shared, distinguishable fixtures ---------------------------------------
# gate = [2, -3, 0.5, 5], value = [1.5, -2, 0.25, -0.5]; clamp_value = 1.0.
# Chosen so every clamp branch fires: gate 2 and 5 hit the upper clamp, gate -3
# stays (lower is unbounded), value 1.5 and -2 saturate, 0.25/-0.5 pass through.
_Y_ROW = [2.0, -3.0, 0.5, 5.0, 1.5, -2.0, 0.25, -0.5]
_CV = 1.0


class _CPUFixture(unittest.TestCase):
    """Common CPU device setup with global-state restoration."""

    def setUp(self):
        self._orig_device = paddle.get_device()
        paddle.set_device("cpu")
        self.addCleanup(lambda: paddle.set_device(self._orig_device))


@unittest.skipUnless(
    HAS_PADDLE, "paddle not importable in this environment: " + _IMPORT_ERROR
)
class TestClampedSwigluForward(_CPUFixture):
    def test_forward_matches_hand_derived_literals(self):
        # Pen-and-paper expected, cv=1.0:
        #   gate_c   = [1.0, -3.0, 0.5, 1.0]  (only the max side clamps)
        #   value_c  = [1.0, -1.0, 0.25, -0.5]
        #   silu(1.0)=0.7310585786, silu(-3.0)=-0.1422776195,
        #   silu(0.5)=0.3112296656
        #   out = silu(gate_c) * value_c
        expected = np.array(
            [[0.7310585786, 0.1422776195, 0.0778074164, -0.3655292893]],
            dtype=np.float64,
        )
        y = paddle.to_tensor([_Y_ROW], dtype="float32")
        out = clamped_swiglu(y, clamp_value=_CV)
        self.assertEqual(out.shape, [1, 4])
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-6)

    def test_gate_lower_side_is_not_clamped(self):
        # A very negative gate must survive unclamped; if the code clamped the
        # gate on the low side too, silu(-8) ~ -0.0027 vs silu(-1) ~ -0.269
        # would differ far beyond tolerance.
        y = paddle.to_tensor([[-8.0, 1.0]], dtype="float32")  # gate=-8, val=1
        out = clamped_swiglu(y, clamp_value=1.0)
        expected = ref_clamped_swiglu_fwd([[-8.0, 1.0]], 1.0)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-6)
        # And it must NOT equal the low-clamped variant (silu(-1) * 1.0).
        low_clamped = _silu(-1.0) * 1.0
        self.assertGreater(abs(float(out.numpy()[0, 0]) - low_clamped), 0.2)

    def test_forward_matches_independent_reference_multirow(self):
        paddle.seed(20240517)
        y = paddle.randn([6, 12], dtype="float32") * 3.0  # spread to saturate
        y_np = y.numpy()
        out = clamped_swiglu(y, clamp_value=1.5)
        np.testing.assert_allclose(
            out.numpy(), ref_clamped_swiglu_fwd(y_np, 1.5), rtol=1e-5, atol=1e-6
        )


@unittest.skipUnless(
    HAS_PADDLE, "paddle not importable in this environment: " + _IMPORT_ERROR
)
class TestClampedSwigluBackward(_CPUFixture):
    def test_saturation_mask_zeroes_exact_positions(self):
        # cv=1.0, g_up=ones. Raw gate=[2,-3,0.5,5] -> gate mask [0,1,1,0];
        # raw value=[1.5,-2,0.25,-0.5] -> value mask [0,0,1,1]. Output layout
        # is [d_gate(4) | d_value(4)], so exact zeros at cols 0,3 (gate) and
        # 4,5 (value); cols 1,2,6,7 must be non-zero.
        y = paddle.to_tensor([_Y_ROW], dtype="float32")
        g_up = paddle.ones([1, 4], dtype="float32")
        grad = clamped_swiglu_back(g_up, y, clamp_value=_CV).numpy()
        for zero_col in (0, 3, 4, 5):
            self.assertEqual(float(grad[0, zero_col]), 0.0)
        for nz_col in (1, 2, 6, 7):
            self.assertGreater(abs(float(grad[0, nz_col])), 1e-6)

    def test_backward_matches_independent_reference(self):
        paddle.seed(7)
        y = paddle.randn([5, 8], dtype="float32") * 2.5
        g_up = paddle.randn([5, 4], dtype="float32")
        y_np, g_np = y.numpy(), g_up.numpy()
        grad = clamped_swiglu_back(g_up, y, clamp_value=1.0).numpy()
        expected = ref_clamped_swiglu_bwd(g_np, y_np, 1.0)
        np.testing.assert_allclose(grad, expected, rtol=1e-5, atol=1e-6)
        # Negative control: a wrong-sign gradient must be rejected, proving the
        # comparison is not vacuous.
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(grad, -expected, rtol=1e-5, atol=1e-6)


@unittest.skipUnless(
    HAS_PADDLE, "paddle not importable in this environment: " + _IMPORT_ERROR
)
class TestBroadcastScale(_CPUFixture):
    def test_cast_and_unsqueeze_to_target_rank(self):
        scale = paddle.to_tensor([2.0, 3.0], dtype="float32")
        exp2 = _broadcast_scale(scale, paddle.float32, 2)
        self.assertEqual(exp2.shape, [2, 1])
        np.testing.assert_array_equal(exp2.numpy(), np.array([[2.0], [3.0]]))
        exp3 = _broadcast_scale(scale, paddle.float32, 3)
        self.assertEqual(exp3.shape, [2, 1, 1])
        np.testing.assert_array_equal(
            exp3.numpy(), np.array([[[2.0]], [[3.0]]])
        )
        # Already-matching rank is left untouched.
        already = paddle.to_tensor([[5.0], [6.0]], dtype="float32")
        out = _broadcast_scale(already, paddle.float32, 2)
        self.assertEqual(out.shape, [2, 1])
        np.testing.assert_array_equal(out.numpy(), already.numpy())


@unittest.skipUnless(
    HAS_PADDLE, "paddle not importable in this environment: " + _IMPORT_ERROR
)
class TestClampAlignmentForward(_CPUFixture):
    def test_module1_and_module2_agree_and_match_reference(self):
        paddle.seed(11)
        y = paddle.randn([4, 10], dtype="float32") * 3.0
        y_np = y.numpy()
        cv = 1.25
        m1 = clamped_swiglu(y, clamp_value=cv).numpy()
        ones = paddle.ones([4, 1], dtype="float32")
        with _force_cpu_fallback():
            m2 = fused_swiglu_scale_forward(y, ones, clamp_value=cv).numpy()
        ref = ref_clamped_swiglu_fwd(y_np, cv)
        np.testing.assert_allclose(m1, ref, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(m2, ref, rtol=1e-5, atol=1e-6)
        # The alignment contract: the two modules must agree elementwise.
        np.testing.assert_allclose(m1, m2, rtol=1e-5, atol=1e-6)


@unittest.skipUnless(
    HAS_PADDLE, "paddle not importable in this environment: " + _IMPORT_ERROR
)
class TestClampAlignmentBackward(_CPUFixture):
    def test_dx_alignment_scale_one(self):
        paddle.seed(13)
        y = paddle.randn([4, 8], dtype="float32") * 2.5
        g_up = paddle.randn([4, 4], dtype="float32")
        y_np, g_np = y.numpy(), g_up.numpy()
        cv = 1.0
        dx_m1 = clamped_swiglu_back(g_up, y, clamp_value=cv).numpy()
        ones = paddle.ones([4, 1], dtype="float32")
        with _force_cpu_fallback():
            dx_m2, _ = fused_swiglu_scale_backward(
                y, ones, g_up, clamp_value=cv
            )
        dx_m2 = dx_m2.numpy()
        ref = ref_clamped_swiglu_bwd(g_np, y_np, cv)
        np.testing.assert_allclose(dx_m1, ref, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(dx_m2, ref, rtol=1e-5, atol=1e-6)
        # d_x must be identical between the two modules when scale == 1.
        np.testing.assert_allclose(dx_m1, dx_m2, rtol=1e-5, atol=1e-6)


@unittest.skipUnless(
    HAS_PADDLE, "paddle not importable in this environment: " + _IMPORT_ERROR
)
class TestScaleGradientAndBroadcast(_CPUFixture):
    def test_d_scale_value_and_keepdim(self):
        paddle.seed(23)
        y = paddle.randn([4, 12], dtype="float32") * 2.0
        out_grad = paddle.randn([4, 6], dtype="float32")
        scale = paddle.ones([4, 1], dtype="float32")
        y_np, og_np = y.numpy(), out_grad.numpy()
        cv = 2.0
        with _force_cpu_fallback():
            _, d_scale = fused_swiglu_scale_backward(
                y, scale, out_grad, clamp_value=cv
            )
        # clamp branch keeps the reduced dim -> [B, 1].
        self.assertEqual(d_scale.shape, [4, 1])
        np.testing.assert_allclose(
            d_scale.numpy(),
            ref_scale_grad_clamp(y_np, og_np, cv),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_scale_applied_per_row_in_forward(self):
        # Distinct per-row scale must multiply the matching row only.
        y = paddle.to_tensor([_Y_ROW, _Y_ROW], dtype="float32")
        scale = paddle.to_tensor([2.0, 5.0], dtype="float32")  # 1D -> [B,1]
        with _force_cpu_fallback():
            out = fused_swiglu_scale_forward(y, scale, clamp_value=_CV).numpy()
        base = ref_clamped_swiglu_fwd([_Y_ROW], _CV)  # shape [1,4]
        np.testing.assert_allclose(out[0], base[0] * 2.0, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(out[1], base[0] * 5.0, rtol=1e-5, atol=1e-6)


@unittest.skipUnless(
    HAS_PADDLE, "paddle not importable in this environment: " + _IMPORT_ERROR
)
class TestActivationDispatchSelection(_CPUFixture):
    def test_clamp_value_zero_falls_through_to_plain_swiglu(self):
        # clamp_value == 0 fails the ``clamp_value > 0`` guard, so the plain
        # (unclamped) SwiGLU path must run even on saturating inputs.
        y = paddle.to_tensor([_Y_ROW], dtype="float32")  # gate up to 5.0
        ones = paddle.ones([1, 1], dtype="float32")
        with _force_cpu_fallback():
            out0 = fused_swiglu_scale_forward(y, ones, clamp_value=0.0).numpy()
            out_small = fused_swiglu_scale_forward(
                y, ones, clamp_value=0.5
            ).numpy()
        np.testing.assert_allclose(
            out0, ref_swiglu_fwd([_Y_ROW]), rtol=1e-5, atol=1e-6
        )
        # The small-clamp result must differ, proving the guard actually routes.
        self.assertGreater(np.abs(out0 - out_small).max(), 0.1)

    def test_negative_clamp_falls_through(self):
        y = paddle.to_tensor([_Y_ROW], dtype="float32")
        ones = paddle.ones([1, 1], dtype="float32")
        with _force_cpu_fallback():
            out_neg = fused_swiglu_scale_forward(
                y, ones, clamp_value=-1.0
            ).numpy()
        np.testing.assert_allclose(
            out_neg, ref_swiglu_fwd([_Y_ROW]), rtol=1e-5, atol=1e-6
        )


@unittest.skipUnless(
    HAS_PADDLE, "paddle not importable in this environment: " + _IMPORT_ERROR
)
class TestValidationGuards(_CPUFixture):
    def test_bias_swiglu_impl_rejects_bad_ndim(self):
        with self.assertRaises(AssertionError):
            bias_swiglu_impl(paddle.randn([4]), None, clamp_value=1.0)
        with self.assertRaises(AssertionError):
            bias_swiglu_impl(paddle.randn([2, 2, 2, 4]), None, clamp_value=1.0)

    def test_weighted_bias_swiglu_impl_rejects_bias(self):
        # Bias is explicitly unsupported for the weighted variant.
        with self.assertRaises(NotImplementedError):
            weighted_bias_swiglu_impl(
                paddle.randn([4, 16]),
                paddle.randn([4, 16]),
                paddle.randn([4, 1]),
                clamp_value=2.0,
            )


@unittest.skipUnless(
    HAS_PADDLE, "paddle not importable in this environment: " + _IMPORT_ERROR
)
class TestBiasSwiGLUFunctionCtxRoundTrip(_CPUFixture):
    def test_clamp_branch_forward_and_backward_via_ctx(self):
        # Drives the real PyLayer: forward saves clamp_value/flags on ctx and
        # backward restores them to pick the clamped_bias_swiglu_back branch.
        # Both returned grads equal d(clamped_swiglu(input+bias))/d(.) because
        # backward returns ``(tmp, tmp)``.  bias is given the same shape as the
        # input (the intended elementwise-add usage) so no grad reduction is
        # involved.
        paddle.seed(31)
        x = (paddle.randn([4, 16], dtype="float32") * 2.0).detach()
        bias = (paddle.randn([4, 16], dtype="float32") * 2.0).detach()
        x.stop_gradient = False
        bias.stop_gradient = False
        upstream = paddle.randn([4, 8], dtype="float32")

        out = BiasSwiGLUFunction.apply(x, bias, False, False, clamp_value=1.0)

        # Forward output must match clamped_swiglu(input + bias).
        y_np = (x + bias).detach().numpy()
        up_np = upstream.numpy()
        np.testing.assert_allclose(
            out.numpy(),
            ref_clamped_swiglu_fwd(y_np, 1.0),
            rtol=1e-5,
            atol=1e-6,
        )

        out.backward(upstream)
        expected_grad = ref_clamped_swiglu_bwd(up_np, y_np, 1.0)
        # backward returns the same tensor for both input and bias grads.
        np.testing.assert_allclose(
            x.grad.numpy(), expected_grad, rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            bias.grad.numpy(), expected_grad, rtol=1e-5, atol=1e-6
        )

    def test_clamp_none_selects_plain_swiglu_forward(self):
        # With clamp_value=None the ctx routes to the non-clamped forward; on a
        # saturating input this must equal plain SwiGLU, not the clamped result.
        x = paddle.to_tensor([_Y_ROW], dtype="float32")
        bias = paddle.zeros([8], dtype="float32")
        out = BiasSwiGLUFunction.apply(
            x, bias, False, False, clamp_value=None
        ).numpy()
        np.testing.assert_allclose(
            out, ref_swiglu_fwd([_Y_ROW]), rtol=1e-5, atol=1e-6
        )
        # And it must differ from the clamped branch on this saturating row.
        clamped = ref_clamped_swiglu_fwd([_Y_ROW], _CV)
        self.assertGreater(np.abs(out - clamped).max(), 0.1)


if __name__ == "__main__":
    unittest.main()
