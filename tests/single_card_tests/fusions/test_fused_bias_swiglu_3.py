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

"""CPU behaviour tests for the *plain* (non-clamped, non-accuracy-compatible)
SwiGLU slice of ``paddlefleet.fusions.fused_bias_swiglu``.

Slice under test (matching the plain-path helper names):
``swiglu``, ``bias_swiglu``, ``weighted_swiglu``, the ``SwiGLUFunction`` /
``BiasSwiGLUFunction`` / ``WeightedSwiGLUFunction`` autograd PyLayers (their
``clamp_value=None`` branch), and the ``bias_swiglu_impl`` /
``weighted_bias_swiglu_impl`` shape-handling wrappers.  The clamped and the
``use_accuracy_compatible`` branches are covered by sibling files and are not
re-tested here.

``jit_fuser`` is an identity no-op (``jit_fuser = lambda fn: fn`` in
``paddlefleet/jit.py``), so every function is plain paddle tensor arithmetic.
``F.swiglu`` and ``paddle._C_ops.swiglu_grad`` have CPU kernels, so both the
forward values and the autograd backward run on CPU; nothing here claims to
verify a GPU/Triton kernel.

Every expected value comes from an INDEPENDENT reference:

* forward values from a NumPy ``SiLU(gate) * value`` implementation written from
  the mathematical definition (never calling ``F.swiglu``);
* backward gradients from ``paddle.grad`` of an INDEPENDENTLY written paddle
  forward (``F.silu(y1) * y2``), never from the analytic ``swiglu_back`` /
  ``swiglu_grad`` code the PyLayers actually run.
"""

import os
import sys
import unittest

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    import numpy as np
    import paddle
    import paddle.nn.functional as F

    from paddlefleet.fusions.fused_bias_swiglu import (
        BiasSwiGLUFunction,
        SwiGLUFunction,
        WeightedSwiGLUFunction,
        bias_swiglu,
        bias_swiglu_impl,
        swiglu,
        weighted_bias_swiglu_impl,
        weighted_swiglu,
    )

    HAS_PADDLE = True
    _IMPORT_ERROR = ""
except ImportError as exc:  # honest: dependency missing, not a swallowed bug
    HAS_PADDLE = False
    _IMPORT_ERROR = repr(exc)

_SKIP_REASON = "paddle/paddlefleet not importable: " + _IMPORT_ERROR


# --- Independent references (never call the production activation) -----------


def _distinct(shape):
    """Deterministic, non-degenerate array (mixed sign, no all-zeros)."""
    n = int(np.prod(shape))
    vals = 2.0 * np.sin(np.arange(1, n + 1, dtype=np.float64))
    return vals.reshape(shape).astype(np.float32)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def _silu(x):
    x = np.asarray(x, dtype=np.float64)
    return x * _sigmoid(x)


def _split_last(arr):
    """Split the last axis into two equal halves (matches paddle.chunk)."""
    half = arr.shape[-1] // 2
    return arr[..., :half], arr[..., half:]


def ref_swiglu(y):
    """Plain SwiGLU: SiLU(gate) * value, gate/value = first/second half."""
    g, v = _split_last(np.asarray(y, dtype=np.float64))
    return _silu(g) * v


def _ref_swiglu_paddle(y):
    """Independent paddle forward, used ONLY through autodiff for gradients."""
    y1, y2 = paddle.chunk(y, 2, axis=-1)
    return F.silu(y1) * y2


def _leaf(np_arr):
    t = paddle.to_tensor(np_arr, dtype="float32")
    t.stop_gradient = False
    return t


class _CpuBase(unittest.TestCase):
    """Pin execution to CPU and restore the global device afterwards."""

    def setUp(self):
        self._orig_device = paddle.get_device()
        paddle.set_device("cpu")
        self.addCleanup(paddle.set_device, self._orig_device)


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestPlainForwardValues(_CpuBase):
    def test_swiglu_matches_independent_silu_value(self):
        y_np = _distinct([3, 8])
        out = swiglu(paddle.to_tensor(y_np)).numpy()
        self.assertEqual(list(out.shape), [3, 4])
        np.testing.assert_allclose(out, ref_swiglu(y_np), rtol=1e-5, atol=1e-6)
        # Non-vacuity: the reference must reject a wrong (halves-swapped) result.
        g, v = _split_last(y_np)
        swapped = _silu(v) * g
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(out, swapped, rtol=1e-5, atol=1e-6)

    def test_bias_swiglu_adds_bias_then_swiglu(self):
        y_np = _distinct([4, 8])
        bias_np = _distinct([8]) * 0.5
        out = bias_swiglu(
            paddle.to_tensor(y_np), paddle.to_tensor(bias_np)
        ).numpy()
        # bias_swiglu(y, bias) == swiglu(y + bias); bias broadcasts over rows.
        expected = ref_swiglu(y_np + bias_np.reshape([1, 8]))
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)
        # The bias must actually move the result away from the no-bias case.
        no_bias = ref_swiglu(y_np)
        self.assertGreater(np.abs(out - no_bias).max(), 1e-3)

    def test_weighted_swiglu_scales_per_token(self):
        y_np = _distinct([4, 8])
        # Distinct per-row weights so a mis-broadcast would show up.
        w_np = np.array([[1.5], [2.0], [0.5], [3.0]], dtype=np.float32)
        out = weighted_swiglu(
            paddle.to_tensor(y_np), paddle.to_tensor(w_np)
        ).numpy()
        expected = ref_swiglu(y_np) * w_np
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)
        # weights == 1 must reproduce plain swiglu; a different weight must not.
        out1 = weighted_swiglu(
            paddle.to_tensor(y_np), paddle.ones([4, 1], dtype="float32")
        ).numpy()
        np.testing.assert_allclose(out1, ref_swiglu(y_np), rtol=1e-5, atol=1e-6)
        self.assertGreater(np.abs(out - out1).max(), 1e-3)

    def test_weighted_swiglu_preserves_input_dtype(self):
        y_np = _distinct([2, 8])
        w_np = np.array([[1.25], [0.75]], dtype=np.float32)
        out = weighted_swiglu(
            paddle.to_tensor(y_np, dtype="float32"),
            paddle.to_tensor(w_np, dtype="float32"),
        )
        self.assertEqual(out.dtype, paddle.float32)
        np.testing.assert_allclose(
            out.numpy(), ref_swiglu(y_np) * w_np, rtol=1e-5, atol=1e-6
        )


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestPyLayerForwardBackward(_CpuBase):
    """Drive the real PyLayers' plain (clamp_value=None) forward AND their
    custom backward (``swiglu_back`` -> ``paddle._C_ops.swiglu_grad``)."""

    def test_swiglu_function_forward_and_backward(self):
        x_np = _distinct([3, 8])
        g_np = _distinct([3, 4]) * 0.7  # upstream cotangent, shape [N, H]
        g = paddle.to_tensor(g_np)

        x = _leaf(x_np)
        out = SwiGLUFunction.apply(x, False, False)
        np.testing.assert_allclose(
            out.numpy(), ref_swiglu(x_np), rtol=1e-5, atol=1e-6
        )

        (grad,) = paddle.grad((out * g).sum(), [x])
        x_ref = _leaf(x_np)
        ref_out = _ref_swiglu_paddle(x_ref)
        (grad_ref,) = paddle.grad((ref_out * g).sum(), [x_ref])
        self.assertEqual(list(grad.shape), [3, 8])
        np.testing.assert_allclose(
            grad.numpy(), grad_ref.numpy(), rtol=2e-4, atol=1e-6
        )
        # Non-vacuity: a wrong-sign gradient must be rejected.
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(
                grad.numpy(), -grad_ref.numpy(), rtol=2e-4, atol=1e-6
            )

    def test_bias_swiglu_function_forward_and_backward(self):
        # bias given the same shape as input (elementwise-add usage) so the
        # returned (tmp, tmp) grads need no reduction.
        x_np = _distinct([3, 8])
        bias_np = _distinct([3, 8]) * 0.4
        g_np = _distinct([3, 4]) * 0.6
        g = paddle.to_tensor(g_np)

        x = _leaf(x_np)
        bias = _leaf(bias_np)
        out = BiasSwiGLUFunction.apply(x, bias, False, False)
        np.testing.assert_allclose(
            out.numpy(), ref_swiglu(x_np + bias_np), rtol=1e-5, atol=1e-6
        )

        x_grad, bias_grad = paddle.grad((out * g).sum(), [x, bias])
        x_ref = _leaf(x_np)
        bias_ref = _leaf(bias_np)
        ref_out = _ref_swiglu_paddle(x_ref + bias_ref)
        gx_ref, gb_ref = paddle.grad((ref_out * g).sum(), [x_ref, bias_ref])
        np.testing.assert_allclose(
            x_grad.numpy(), gx_ref.numpy(), rtol=2e-4, atol=1e-6
        )
        np.testing.assert_allclose(
            bias_grad.numpy(), gb_ref.numpy(), rtol=2e-4, atol=1e-6
        )
        # backward returns the same tensor for both grads (production contract).
        np.testing.assert_allclose(
            x_grad.numpy(), bias_grad.numpy(), rtol=1e-6, atol=1e-7
        )

    def test_weighted_swiglu_function_forward_and_backward(self):
        x_np = _distinct([4, 8])
        w_np = np.array([[1.3], [0.6], [2.1], [0.9]], dtype=np.float32)
        g_np = _distinct([4, 4]) * 0.5
        g = paddle.to_tensor(g_np)

        x = _leaf(x_np)
        w = _leaf(w_np)
        out = WeightedSwiGLUFunction.apply(x, w, False)
        np.testing.assert_allclose(
            out.numpy(), ref_swiglu(x_np) * w_np, rtol=1e-5, atol=1e-6
        )

        x_grad, w_grad = paddle.grad((out * g).sum(), [x, w])
        x_ref = _leaf(x_np)
        w_ref = _leaf(w_np)
        ref_out = _ref_swiglu_paddle(x_ref) * w_ref
        gx_ref, gw_ref = paddle.grad((ref_out * g).sum(), [x_ref, w_ref])
        np.testing.assert_allclose(
            x_grad.numpy(), gx_ref.numpy(), rtol=2e-4, atol=1e-6
        )
        # weights broadcast over the feature dim -> grad summed to [N, 1].
        self.assertEqual(list(w_grad.shape), [4, 1])
        np.testing.assert_allclose(
            w_grad.numpy(), gw_ref.numpy(), rtol=2e-4, atol=1e-6
        )


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestBiasSwigluImplControlFlow(_CpuBase):
    def test_impl_2d_with_bias_no_reshape(self):
        x_np = _distinct([5, 8])
        bias_np = _distinct([8]) * 0.4
        out = bias_swiglu_impl(
            paddle.to_tensor(x_np), paddle.to_tensor(bias_np)
        )
        self.assertEqual(list(out.shape), [5, 4])
        np.testing.assert_allclose(
            out.numpy(),
            ref_swiglu(x_np + bias_np.reshape([1, 8])),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_impl_2d_without_bias_uses_swiglu_branch(self):
        x_np = _distinct([5, 8])
        out = bias_swiglu_impl(paddle.to_tensor(x_np), None)
        self.assertEqual(list(out.shape), [5, 4])
        np.testing.assert_allclose(
            out.numpy(), ref_swiglu(x_np), rtol=1e-5, atol=1e-6
        )

    def test_impl_3d_reshapes_and_preserves_content(self):
        # 3D path: view to [B*S, 2H], apply, view back to [B, S, H].
        x_np = _distinct([2, 3, 8])
        bias_np = _distinct([8]) * 0.4
        out = bias_swiglu_impl(
            paddle.to_tensor(x_np), paddle.to_tensor(bias_np)
        ).numpy()
        self.assertEqual(list(out.shape), [2, 3, 4])
        expected = ref_swiglu(x_np + bias_np.reshape([1, 1, 8]))
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)

    def test_impl_rejects_invalid_ndim(self):
        # assert len(shape) in [2, 3] fires before any activation runs.
        for bad_shape in ([8], [2, 2, 2, 8]):
            with self.assertRaises(AssertionError):
                bias_swiglu_impl(paddle.to_tensor(_distinct(bad_shape)), None)


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestWeightedBiasSwigluImplControlFlow(_CpuBase):
    def test_impl_2d_no_bias(self):
        x_np = _distinct([4, 8])
        w_np = np.array([[1.5], [2.0], [0.5], [3.0]], dtype=np.float32)
        out = weighted_bias_swiglu_impl(
            paddle.to_tensor(x_np), None, paddle.to_tensor(w_np)
        )
        self.assertEqual(list(out.shape), [4, 4])
        np.testing.assert_allclose(
            out.numpy(), ref_swiglu(x_np) * w_np, rtol=1e-5, atol=1e-6
        )

    def test_impl_3d_reshapes_weights_and_content(self):
        # 3D path: input view to [B*S, 2H], weights view to [B*S, 1],
        # output view back to [B, S, H].
        x_np = _distinct([2, 3, 8])
        w_np = (
            (1.0 + 0.25 * np.arange(1, 7)).reshape([2, 3, 1]).astype(np.float32)
        )
        out = weighted_bias_swiglu_impl(
            paddle.to_tensor(x_np), None, paddle.to_tensor(w_np)
        ).numpy()
        self.assertEqual(list(out.shape), [2, 3, 4])
        expected = ref_swiglu(x_np) * w_np  # [2,3,1] broadcasts over hidden
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)

    def test_impl_rejects_bias(self):
        # Bias is explicitly unsupported for the weighted variant.
        with self.assertRaises(NotImplementedError) as ctx:
            weighted_bias_swiglu_impl(
                paddle.to_tensor(_distinct([4, 8])),
                paddle.to_tensor(_distinct([8])),
                paddle.to_tensor(np.ones([4, 1], dtype=np.float32)),
            )
        self.assertIn(
            "Bias is not supported for weighted swiglu fusion",
            str(ctx.exception),
        )

    def test_impl_rejects_invalid_ndim(self):
        w = paddle.to_tensor(np.ones([1, 1], dtype=np.float32))
        with self.assertRaises(AssertionError):
            weighted_bias_swiglu_impl(
                paddle.to_tensor(_distinct([2, 2, 2, 8])), None, w
            )


if __name__ == "__main__":
    unittest.main()
