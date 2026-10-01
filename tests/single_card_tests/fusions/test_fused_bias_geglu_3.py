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

"""CPU behavior tests for ``paddlefleet.fusions.fused_bias_geglu``.

Slice under test (the gradient/``*_back`` family and the shape-handling
``*_impl`` wrappers, plus the forward values they depend on):
``geglu``, ``bias_geglu``, ``quick_gelu``, ``quick_geglu``,
``weighted_quick_geglu``, ``weighted_bias_quick_geglu``,
``geglu_back``, ``bias_geglu_back``, ``quick_geglu_back``,
``weighted_quick_geglu_back``, ``weighted_bias_quick_geglu_back``,
``GeGLUFunction``/``BiasGeGLUFunction`` autograd, ``bias_geglu_impl`` and
``weighted_bias_quick_geglu_impl``.

The production ``jit_fuser`` is an identity no-op (``jit_fuser = lambda fn: fn``),
so every function here is plain paddle tensor arithmetic that runs on CPU.

Expected forward values come from an INDEPENDENT NumPy implementation of the
canonical tanh-approx GELU and sigmoid-approx quick-GELU. Expected gradients
come from autodiff (``paddle.grad``) of an INDEPENDENTLY written paddle forward
-- never from the hand-written analytic ``*_back`` code under test. This keeps
the reference and the tested backward formula genuinely different sources.
"""

import math

# Make ``src/paddlefleet`` importable when paddle is available.
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

    from paddlefleet.fusions.fused_bias_geglu import (
        BiasGeGLUFunction,
        GeGLUFunction,
        bias_geglu,
        bias_geglu_back,
        bias_geglu_impl,
        geglu,
        geglu_back,
        quick_geglu,
        quick_geglu_back,
        quick_gelu,
        weighted_bias_quick_geglu,
        weighted_bias_quick_geglu_back,
        weighted_bias_quick_geglu_impl,
        weighted_quick_geglu,
        weighted_quick_geglu_back,
    )

    HAS_PADDLE = True
    _IMPORT_ERROR = ""
except ImportError as exc:  # honest: dependency missing, not a swallowed bug
    HAS_PADDLE = False
    _IMPORT_ERROR = str(exc)

_SKIP_REASON = f"paddle/paddlefleet not importable: {_IMPORT_ERROR}"

# sqrt(2/pi) derived independently rather than copying the production literal.
_GELU_C = math.sqrt(2.0 / math.pi)


def _distinct(shape):
    """Deterministic, non-degenerate array (mixed sign, no all-zeros)."""
    n = int(np.prod(shape))
    vals = 2.0 * np.sin(np.arange(1, n + 1, dtype=np.float64))
    return vals.reshape(shape).astype(np.float32)


# ---- independent NumPy forward references -------------------------------


def _np_gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(_GELU_C * (x + 0.044715 * x**3)))


def _np_quick_gelu(x):
    return x * (1.0 / (1.0 + np.exp(-1.702 * x)))


def _np_split_half(arr):
    h = arr.shape[-1] // 2
    return arr[..., :h], arr[..., h:]


def _np_geglu(y):
    y1, y2 = _np_split_half(y)
    return _np_gelu_tanh(y1) * y2


def _np_quick_geglu(y, offset):
    y1, y2 = _np_split_half(y)
    return _np_quick_gelu(y1) * (y2 + offset)


# ---- independent paddle forwards, used only via autodiff for gradients --
# These are re-expressed from the canonical activation definitions; the
# gradients are obtained by paddle.grad, NOT from the analytic *_back code.


def _ref_geglu_paddle(y):
    y1, y2 = paddle.chunk(y, 2, axis=-1)
    gelu = 0.5 * y1 * (1.0 + paddle.tanh(_GELU_C * (y1 + 0.044715 * y1**3)))
    return gelu * y2


def _ref_quick_geglu_paddle(y, offset):
    y1, y2 = paddle.chunk(y, 2, axis=-1)
    qgelu = y1 * paddle.nn.functional.sigmoid(1.702 * y1)
    return qgelu * (y2 + offset)


def _leaf(np_arr):
    t = paddle.to_tensor(np_arr, dtype="float32")
    t.stop_gradient = False
    return t


def _vjp(forward_callable, leaves, cotangent):
    """Return grads of ``sum(forward*cotangent)`` w.r.t. ``leaves``."""
    out = forward_callable()
    loss = (out * cotangent).sum()
    grads = paddle.grad(loss, leaves)
    return out, grads


class _CpuBase(unittest.TestCase):
    """Pin execution to CPU and restore the global device afterwards."""

    def setUp(self):
        self._orig_device = paddle.get_device()
        paddle.set_device("cpu")
        self.addCleanup(paddle.set_device, self._orig_device)


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestForwardValues(_CpuBase):
    def test_geglu_matches_independent_gelu_tanh(self):
        y_np = _distinct([2, 8])
        out = geglu(paddle.to_tensor(y_np)).numpy()
        np.testing.assert_allclose(out, _np_geglu(y_np), rtol=1e-5, atol=1e-6)

    def test_bias_geglu_applies_bias_then_geglu(self):
        y_np = _distinct([3, 6])
        bias_np = _distinct([6]) * 0.5
        out = bias_geglu(
            paddle.to_tensor(bias_np), paddle.to_tensor(y_np)
        ).numpy()
        # bias_geglu(bias, y) computes geglu(y + bias).
        expected = _np_geglu(y_np + bias_np)
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)

    def test_quick_gelu_matches_sigmoid_reference(self):
        x_np = _distinct([4, 5])
        out = quick_gelu(paddle.to_tensor(x_np)).numpy()
        np.testing.assert_allclose(
            out, _np_quick_gelu(x_np), rtol=1e-5, atol=1e-6
        )

    def test_quick_geglu_consumes_offset(self):
        y_np = _distinct([2, 8])
        offset = 0.75
        out = quick_geglu(paddle.to_tensor(y_np), linear_offset=offset).numpy()
        np.testing.assert_allclose(
            out, _np_quick_geglu(y_np, offset), rtol=1e-5, atol=1e-6
        )
        # The offset must actually move the result away from the zero-offset
        # case; a dropped offset would make these equal.
        out0 = quick_geglu(paddle.to_tensor(y_np), linear_offset=0.0).numpy()
        self.assertFalse(np.allclose(out, out0))

    def test_weighted_quick_geglu_scales_per_token(self):
        y_np = _distinct([4, 6])
        w_np = (
            (1.5 + 0.5 * np.cos(np.arange(4)))
            .reshape([4, 1])
            .astype(np.float32)
        )
        out = weighted_quick_geglu(
            paddle.to_tensor(y_np), paddle.to_tensor(w_np), linear_offset=0.25
        ).numpy()
        expected = _np_quick_geglu(y_np, 0.25) * w_np
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)

    def test_weighted_bias_quick_geglu_matches_reference(self):
        y_np = _distinct([4, 6])
        bias_np = _distinct([4, 6]) * 0.3
        w_np = (
            (1.2 + 0.4 * np.sin(np.arange(4)))
            .reshape([4, 1])
            .astype(np.float32)
        )
        out = weighted_bias_quick_geglu(
            paddle.to_tensor(y_np),
            paddle.to_tensor(bias_np),
            paddle.to_tensor(w_np),
            linear_offset=-0.5,
        ).numpy()
        expected = _np_quick_geglu(y_np + bias_np, -0.5) * w_np
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestBackwardAgainstAutodiff(_CpuBase):
    """``*_back`` functions must equal the VJP of an independent forward."""

    def test_geglu_back_matches_autodiff_vjp(self):
        y_np = _distinct([3, 8])
        g_np = _distinct([3, 4]) * 0.7  # upstream cotangent, shape [N, H]
        g = paddle.to_tensor(g_np)

        got = geglu_back(g, paddle.to_tensor(y_np)).numpy()

        y_ref = _leaf(y_np)
        _, (grad_ref,) = _vjp(lambda: _ref_geglu_paddle(y_ref), [y_ref], g)
        np.testing.assert_allclose(got, grad_ref.numpy(), rtol=2e-4, atol=1e-6)

    def test_bias_geglu_back_matches_autodiff_vjp(self):
        y_np = _distinct([3, 8])
        bias_np = _distinct([8]) * 0.4
        g_np = _distinct([3, 4]) * 0.7
        g = paddle.to_tensor(g_np)

        got = bias_geglu_back(
            g, paddle.to_tensor(y_np), paddle.to_tensor(bias_np)
        ).numpy()

        # d/dy geglu(y + bias) equals the VJP taken w.r.t. the shifted leaf.
        shifted = _leaf(y_np + bias_np)
        _, (grad_ref,) = _vjp(lambda: _ref_geglu_paddle(shifted), [shifted], g)
        np.testing.assert_allclose(got, grad_ref.numpy(), rtol=2e-4, atol=1e-6)

    def test_quick_geglu_back_matches_autodiff_vjp(self):
        y_np = _distinct([3, 8])
        g_np = _distinct([3, 4]) * 0.6
        g = paddle.to_tensor(g_np)
        offset = 0.75

        got = quick_geglu_back(
            g, paddle.to_tensor(y_np), linear_offset=offset
        ).numpy()

        y_ref = _leaf(y_np)
        _, (grad_ref,) = _vjp(
            lambda: _ref_quick_geglu_paddle(y_ref, offset), [y_ref], g
        )
        np.testing.assert_allclose(got, grad_ref.numpy(), rtol=2e-4, atol=1e-6)

    def test_weighted_quick_geglu_back_matches_autodiff_vjp(self):
        y_np = _distinct([4, 6])
        w_np = (
            (1.3 + 0.3 * np.cos(np.arange(4)))
            .reshape([4, 1])
            .astype(np.float32)
        )
        g_np = _distinct([4, 3]) * 0.5
        g = paddle.to_tensor(g_np)
        offset = 0.2

        input_grad, weights_grad = weighted_quick_geglu_back(
            g,
            paddle.to_tensor(y_np),
            paddle.to_tensor(w_np),
            linear_offset=offset,
        )

        y_ref = _leaf(y_np)
        w_ref = _leaf(w_np)
        _, (gy, gw) = _vjp(
            lambda: _ref_quick_geglu_paddle(y_ref, offset) * w_ref,
            [y_ref, w_ref],
            g,
        )
        np.testing.assert_allclose(
            input_grad.numpy(), gy.numpy(), rtol=2e-4, atol=1e-6
        )
        # weights broadcast over the feature dim -> grad is summed to [N, 1].
        np.testing.assert_allclose(
            weights_grad.numpy(), gw.numpy(), rtol=2e-4, atol=1e-6
        )
        self.assertEqual(list(weights_grad.shape), [4, 1])

    def test_weighted_bias_quick_geglu_back_matches_autodiff_vjp(self):
        y_np = _distinct([4, 6])
        bias_np = _distinct([4, 6]) * 0.3
        w_np = (
            (1.1 + 0.2 * np.sin(np.arange(4)))
            .reshape([4, 1])
            .astype(np.float32)
        )
        g_np = _distinct([4, 3]) * 0.5
        g = paddle.to_tensor(g_np)
        offset = -0.4

        input_grad, bias_grad, weights_grad = weighted_bias_quick_geglu_back(
            g,
            paddle.to_tensor(y_np),
            paddle.to_tensor(bias_np),
            paddle.to_tensor(w_np),
            linear_offset=offset,
        )

        y_ref = _leaf(y_np)
        bias_ref = _leaf(bias_np)
        w_ref = _leaf(w_np)
        _, (gy, gb, gw) = _vjp(
            lambda: _ref_quick_geglu_paddle(y_ref + bias_ref, offset) * w_ref,
            [y_ref, bias_ref, w_ref],
            g,
        )
        np.testing.assert_allclose(
            input_grad.numpy(), gy.numpy(), rtol=2e-4, atol=1e-6
        )
        np.testing.assert_allclose(
            bias_grad.numpy(), gb.numpy(), rtol=2e-4, atol=1e-6
        )
        np.testing.assert_allclose(
            weights_grad.numpy(), gw.numpy(), rtol=2e-4, atol=1e-6
        )
        # Forward depends on y, bias only through (y + bias): the two input
        # gradients are identical, which production also relies on.
        np.testing.assert_allclose(
            input_grad.numpy(), bias_grad.numpy(), rtol=1e-6, atol=1e-7
        )


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestPyLayerAutograd(_CpuBase):
    """Drive the custom PyLayer forward AND its custom backward."""

    def test_geglu_function_forward_and_backward(self):
        x_np = _distinct([3, 8])
        g_np = _distinct([3, 4]) * 0.6
        g = paddle.to_tensor(g_np)

        x = _leaf(x_np)
        out = GeGLUFunction.apply(x)
        # Forward value: independent NumPy reference.
        np.testing.assert_allclose(
            out.numpy(), _np_geglu(x_np), rtol=1e-5, atol=1e-6
        )
        (grad,) = paddle.grad((out * g).sum(), [x])

        x_ref = _leaf(x_np)
        _, (grad_ref,) = _vjp(lambda: _ref_geglu_paddle(x_ref), [x_ref], g)
        np.testing.assert_allclose(
            grad.numpy(), grad_ref.numpy(), rtol=2e-4, atol=1e-6
        )

    def test_bias_geglu_function_forward_and_backward(self):
        x_np = _distinct([3, 8])
        bias_np = _distinct([8]) * 0.4
        g_np = _distinct([3, 4]) * 0.6
        g = paddle.to_tensor(g_np)

        x = _leaf(x_np)
        bias = _leaf(bias_np)
        out = BiasGeGLUFunction.apply(x, bias)
        np.testing.assert_allclose(
            out.numpy(), _np_geglu(x_np + bias_np), rtol=1e-5, atol=1e-6
        )
        x_grad, bias_grad = paddle.grad((out * g).sum(), [x, bias])

        # Reference: independent forward with a broadcast bias leaf.
        x_ref = _leaf(x_np)
        bias_ref = _leaf(bias_np)
        _, (gx, gb) = _vjp(
            lambda: _ref_geglu_paddle(x_ref + bias_ref), [x_ref, bias_ref], g
        )
        np.testing.assert_allclose(
            x_grad.numpy(), gx.numpy(), rtol=2e-4, atol=1e-6
        )
        # bias broadcasts over the batch, so its grad is the batch-summed
        # input grad (production computes this via ``reduce_as``).
        self.assertEqual(list(bias_grad.shape), [8])
        np.testing.assert_allclose(
            bias_grad.numpy(), gb.numpy(), rtol=2e-4, atol=1e-6
        )


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestBiasGeGLUImplControlFlow(_CpuBase):
    def test_impl_2d_with_bias_no_reshape(self):
        x_np = _distinct([5, 8])
        bias_np = _distinct([8]) * 0.4
        out = bias_geglu_impl(paddle.to_tensor(x_np), paddle.to_tensor(bias_np))
        self.assertEqual(list(out.shape), [5, 4])
        np.testing.assert_allclose(
            out.numpy(), _np_geglu(x_np + bias_np), rtol=1e-5, atol=1e-6
        )

    def test_impl_2d_no_bias_uses_geglu_branch(self):
        x_np = _distinct([5, 8])
        out = bias_geglu_impl(paddle.to_tensor(x_np), None)
        self.assertEqual(list(out.shape), [5, 4])
        np.testing.assert_allclose(
            out.numpy(), _np_geglu(x_np), rtol=1e-5, atol=1e-6
        )

    def test_impl_3d_reshapes_and_preserves_content(self):
        # 3D path: view to [B*S, 2H], apply, view back to [B, S, H].
        x_np = _distinct([2, 3, 8])
        bias_np = _distinct([8]) * 0.4
        out = bias_geglu_impl(
            paddle.to_tensor(x_np), paddle.to_tensor(bias_np)
        ).numpy()
        self.assertEqual(list(out.shape), [2, 3, 4])
        # Independent expectation: same math applied per row, kept in 3D.
        expected = _np_geglu(x_np + bias_np.reshape([1, 1, 8]))
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)

    def test_impl_rejects_invalid_ndim(self):
        # Only 2D/3D are accepted (assert len(shape) in [2, 3]).
        for bad_shape in ([16], [2, 2, 2, 8]):
            with self.assertRaises(AssertionError):
                bias_geglu_impl(paddle.to_tensor(_distinct(bad_shape)), None)


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestWeightedBiasQuickGeGLUImpl(_CpuBase):
    def test_impl_rejects_invalid_ndim(self):
        # The dimensionality assert fires before the offset tensor is built.
        w = paddle.to_tensor(_distinct([1, 1]))
        with self.assertRaises(AssertionError):
            weighted_bias_quick_geglu_impl(
                paddle.to_tensor(_distinct([16])), None, w
            )

    def test_impl_valid_input_builds_offset_and_matches_reference(self):
        """A valid 2D call returns the weighted-bias quick-GEGLU value.

        ``weighted_bias_quick_geglu_impl`` builds the linear offset with
        ``paddle.tensor(linear_offset, dtype=..., device=input.device)``
        (fused_bias_geglu.py line 482). Under the paddlefleet_ops torch-compat
        layer active in this runtime, ``paddle.tensor`` is a callable factory
        that accepts the ``device=`` keyword, so the offset tensor is built and
        the impl produces the correct weighted-bias quick-GEGLU output. The
        result is checked against an INDEPENDENT NumPy reference.
        """
        y_np = _distinct([4, 6])
        bias_np = _distinct([4, 6]) * 0.3
        w_np = np.full([4, 1], 1.25, dtype=np.float32)
        out = weighted_bias_quick_geglu_impl(
            paddle.to_tensor(y_np),
            paddle.to_tensor(bias_np),
            paddle.to_tensor(w_np),
            linear_offset=0.5,
        ).numpy()
        expected = _np_quick_geglu(y_np + bias_np, 0.5) * w_np
        self.assertEqual(list(out.shape), [4, 3])
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
