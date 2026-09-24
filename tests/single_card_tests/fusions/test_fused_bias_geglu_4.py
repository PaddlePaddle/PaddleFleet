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

"""CPU behavior tests for the tanh-GEGLU slice of
``paddlefleet.fusions.fused_bias_geglu``.

Slice under test (matches the helper names of this file's coverage source,
kept mostly disjoint from sibling ``_2`` which only covers Function forwards):
``geglu``, ``bias_geglu``, ``geglu_back``, ``bias_geglu_back``,
``bias_geglu_impl`` plus ``GeGLUFunction`` / ``BiasGeGLUFunction`` /
``WeightedQuickGeGLUFunction`` forward.

The production ``jit_fuser`` decorator is an identity no-op
(``jit_fuser = lambda fn: fn``), so these are plain paddle tensor ops that
execute on CPU. Every expected value is hand-derived from an independent NumPy
implementation of the tanh-approx GELU / sigmoid-approx quick GELU, or from the
autograd engine's vector-Jacobian product of the *forward* op (independent of
the hand-written backward formulas). The production module is never used to
build expectations.
"""

import os
import sys
import unittest

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

    from paddlefleet.fusions.fused_bias_geglu import (
        BiasGeGLUFunction,
        GeGLUFunction,
        WeightedQuickGeGLUFunction,
        bias_geglu,
        bias_geglu_back,
        bias_geglu_impl,
        geglu,
        geglu_back,
    )

    HAS_PADDLE = True
    _IMPORT_ERROR = ""
except ImportError as exc:  # honest: dependency missing, not a swallowed bug
    HAS_PADDLE = False
    _IMPORT_ERROR = repr(exc)


# --- Independent NumPy references (do NOT call the production module) -------

# Standard GELU-tanh constants, derived here rather than copied from source:
#   sqrt(2/pi) == 0.7978845608...  (production literal 0.79788456)
_GELU_C = float(np.sqrt(2.0 / np.pi))
_GELU_A = 0.044715


def _np_gelu_tanh(x):
    return x * 0.5 * (1.0 + np.tanh(_GELU_C * x * (1.0 + _GELU_A * x * x)))


def _np_geglu(y):
    """GEGLU: GELU(y1) * y2 with y split on the last axis."""
    half = y.shape[-1] // 2
    y1 = y[..., :half]
    y2 = y[..., half:]
    return _np_gelu_tanh(y1) * y2


def _np_quick_gelu(x):
    return x * (1.0 / (1.0 + np.exp(-1.702 * x)))


def _np_quick_geglu(y, offset):
    """Quick-GEGLU: quick_gelu(y1) * (y2 + offset)."""
    half = y.shape[-1] // 2
    y1 = y[..., :half]
    y2 = y[..., half:]
    return _np_quick_gelu(y1) * (y2 + offset)


# Fixed, distinguishable, non-degenerate inputs (distinct nonzero values,
# both signs) so that swapped halves / dropped bias / sign flips are visible.
_Y_2x4 = np.array(
    [[-1.5, 0.7, 2.0, -0.3], [0.4, -2.1, 1.1, 0.9]], dtype=np.float32
)
_BIAS_4 = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
_G_2x2 = np.array([[1.3, -0.6], [0.5, 2.2]], dtype=np.float32)

_X_2x6 = np.array(
    [[-1.0, 0.5, 1.5, -0.7, 2.1, 0.3], [0.8, -1.2, 0.4, 1.9, -0.5, 1.1]],
    dtype=np.float32,
)
_BIAS_6 = np.array([0.2, -0.1, 0.4, -0.3, 0.15, -0.25], dtype=np.float32)

_RTOL = 1e-4
_ATOL = 1e-5


@unittest.skipUnless(
    HAS_PADDLE,
    "paddle/paddlefleet not importable (dependency missing): " + _IMPORT_ERROR,
)
class _GeGLUBase(unittest.TestCase):
    """Force CPU execution and restore the global device afterwards."""

    def setUp(self):
        self._orig_device = paddle.get_device()
        paddle.set_device("cpu")

    def tearDown(self):
        paddle.set_device(self._orig_device)

    def _t(self, arr):
        return paddle.to_tensor(arr, dtype="float32")


class TestGeGLUForward(_GeGLUBase):
    def test_geglu_matches_independent_reference(self):
        out = geglu(self._t(_Y_2x4))
        expected = _np_geglu(_Y_2x4.astype(np.float64))
        self.assertEqual(out.shape, [2, 2])  # last dim halved
        np.testing.assert_allclose(
            out.numpy(), expected, rtol=_RTOL, atol=_ATOL
        )


class TestBiasGeGLUForward(_GeGLUBase):
    def test_bias_geglu_adds_bias_then_gates(self):
        # bias_geglu(bias, y) computes geglu(y + bias).
        out = bias_geglu(self._t(_BIAS_4), self._t(_Y_2x4))
        expected = _np_geglu((_Y_2x4 + _BIAS_4).astype(np.float64))
        np.testing.assert_allclose(
            out.numpy(), expected, rtol=_RTOL, atol=_ATOL
        )
        # The bias must actually shift the input: result differs from no-bias.
        no_bias = _np_geglu(_Y_2x4.astype(np.float64))
        self.assertTrue(np.abs(expected - no_bias).max() > 1e-3)


class TestGeGLUBack(_GeGLUBase):
    def test_geglu_back_matches_autograd(self):
        # Independent reference: VJP of the forward geglu via the autograd
        # engine, which never touches the hand-written geglu_back formula.
        y_ad = self._t(_Y_2x4)
        y_ad.stop_gradient = False
        g = self._t(_G_2x2)
        out = geglu(y_ad)
        ref_grad = paddle.grad(outputs=out, inputs=y_ad, grad_outputs=g)[0]

        actual = geglu_back(g, self._t(_Y_2x4))
        self.assertEqual(actual.shape, [2, 4])
        np.testing.assert_allclose(
            actual.numpy(), ref_grad.numpy(), rtol=_RTOL, atol=_ATOL
        )


class TestBiasGeGLUBack(_GeGLUBase):
    def test_bias_geglu_back_adds_bias_before_grad(self):
        # bias_geglu_back(g, y, bias) == d/dy geglu(y + bias); since
        # d(y+bias)/dy == 1 this equals the VJP of geglu evaluated at y+bias.
        yb = self._t(_Y_2x4 + _BIAS_4)
        yb.stop_gradient = False
        g = self._t(_G_2x2)
        out = geglu(yb)
        ref_grad = paddle.grad(outputs=out, inputs=yb, grad_outputs=g)[0]

        actual = bias_geglu_back(g, self._t(_Y_2x4), self._t(_BIAS_4))
        self.assertEqual(actual.shape, [2, 4])
        np.testing.assert_allclose(
            actual.numpy(), ref_grad.numpy(), rtol=_RTOL, atol=_ATOL
        )


class TestBiasGeGLUFunction(_GeGLUBase):
    def test_forward_matches_reference(self):
        # forward calls bias_geglu(input, bias) -> geglu(input + bias).
        out = BiasGeGLUFunction.apply(self._t(_Y_2x4), self._t(_BIAS_4))
        expected = _np_geglu((_Y_2x4 + _BIAS_4).astype(np.float64))
        self.assertEqual(out.shape, [2, 2])
        np.testing.assert_allclose(
            out.numpy(), expected, rtol=_RTOL, atol=_ATOL
        )


class TestGeGLUFunction(_GeGLUBase):
    def test_forward_matches_reference(self):
        out = GeGLUFunction.apply(self._t(_Y_2x4))
        expected = _np_geglu(_Y_2x4.astype(np.float64))
        self.assertEqual(out.shape, [2, 2])
        np.testing.assert_allclose(
            out.numpy(), expected, rtol=_RTOL, atol=_ATOL
        )

    def test_backward_drives_pylayer_and_matches_autograd(self):
        # Exercise the PyLayer save/restore + backward wiring end to end, and
        # compare against the independent autograd VJP of the plain forward.
        g = self._t(_G_2x2)

        x = self._t(_Y_2x4)
        x.stop_gradient = False
        out = GeGLUFunction.apply(x)
        pylayer_grad = paddle.grad(outputs=out, inputs=x, grad_outputs=g)[0]

        x_ref = self._t(_Y_2x4)
        x_ref.stop_gradient = False
        out_ref = geglu(x_ref)
        ref_grad = paddle.grad(outputs=out_ref, inputs=x_ref, grad_outputs=g)[0]

        self.assertEqual(pylayer_grad.shape, [2, 4])
        np.testing.assert_allclose(
            pylayer_grad.numpy(), ref_grad.numpy(), rtol=_RTOL, atol=_ATOL
        )


class TestBiasGeGLUImpl(_GeGLUBase):
    def test_2d_with_bias(self):
        out = bias_geglu_impl(self._t(_X_2x6), self._t(_BIAS_6))
        expected = _np_geglu((_X_2x6 + _BIAS_6).astype(np.float64))
        self.assertEqual(out.shape, [2, 3])
        np.testing.assert_allclose(
            out.numpy(), expected, rtol=_RTOL, atol=_ATOL
        )

    def test_2d_without_bias(self):
        out = bias_geglu_impl(self._t(_X_2x6), None)
        expected = _np_geglu(_X_2x6.astype(np.float64))
        self.assertEqual(out.shape, [2, 3])
        np.testing.assert_allclose(
            out.numpy(), expected, rtol=_RTOL, atol=_ATOL
        )

    def test_3d_with_bias_reshapes_and_preserves_content(self):
        x3d = _X_2x6.reshape(2, 1, 6)
        out = bias_geglu_impl(self._t(x3d), self._t(_BIAS_6))
        # impl flattens to [2, 6], applies, then restores [2, 1, 3].
        expected = _np_geglu(
            (x3d.reshape(-1, 6) + _BIAS_6).astype(np.float64)
        ).reshape(2, 1, 3)
        self.assertEqual(out.shape, [2, 1, 3])
        np.testing.assert_allclose(
            out.numpy(), expected, rtol=_RTOL, atol=_ATOL
        )

    def test_3d_without_bias_reshapes_and_preserves_content(self):
        x3d = _X_2x6.reshape(2, 1, 6)
        out = bias_geglu_impl(self._t(x3d), None)
        expected = _np_geglu(x3d.reshape(-1, 6).astype(np.float64)).reshape(
            2, 1, 3
        )
        self.assertEqual(out.shape, [2, 1, 3])
        np.testing.assert_allclose(
            out.numpy(), expected, rtol=_RTOL, atol=_ATOL
        )

    def test_invalid_rank_raises_assertion(self):
        bad = paddle.to_tensor(
            np.zeros((2, 1, 1, 6), dtype=np.float32), dtype="float32"
        )
        with self.assertRaises(AssertionError):
            bias_geglu_impl(bad, None)


class TestWeightedQuickGeGLUFunction(_GeGLUBase):
    def test_forward_consumes_weights_and_offset(self):
        weights = np.array([[2.0], [-1.5]], dtype=np.float32)
        offset = 0.5
        out = WeightedQuickGeGLUFunction.apply(
            self._t(_X_2x6),
            self._t(weights),
            False,  # fp8_input_store: keep the CPU-representable path
            paddle.to_tensor(offset, dtype="float32"),
        )
        # weighted_quick_geglu = quick_geglu(input, offset) * weights.
        expected = _np_quick_geglu(
            _X_2x6.astype(np.float64), offset
        ) * weights.astype(np.float64)
        self.assertEqual(out.shape, [2, 3])
        np.testing.assert_allclose(
            out.numpy(), expected, rtol=_RTOL, atol=_ATOL
        )
        # A nonzero offset genuinely changes the output (offset consumed).
        no_offset = _np_quick_geglu(
            _X_2x6.astype(np.float64), 0.0
        ) * weights.astype(np.float64)
        self.assertTrue(np.abs(expected - no_offset).max() > 1e-3)


if __name__ == "__main__":
    unittest.main()
