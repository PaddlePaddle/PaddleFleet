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

"""CPU behavior tests for the functional + backward slice of
``paddlefleet.fusions.fused_bias_geglu``.

Slice under test (kept disjoint from sibling ``test_fused_bias_geglu_2.py``,
which covers the PyLayer *forward* path):
  * functional forwards: ``geglu``, ``bias_geglu``, ``quick_geglu``,
    ``weighted_quick_geglu``
  * backward helpers: ``geglu_back``, ``bias_geglu_back``,
    ``quick_geglu_back``, ``weighted_quick_geglu_back``
  * the shape-dispatching wrapper ``bias_geglu_impl``
  * the autograd *backward* of ``GeGLUFunction`` / ``BiasGeGLUFunction``

The production ``jit_fuser`` decorator is an identity no-op (``lambda fn: fn``),
so these are plain paddle tensor ops that run on CPU. Every expected value is
derived from an independent NumPy reimplementation of the documented math; the
analytic gradients shipped by the module are cross-checked against a NumPy
*finite-difference* derivative of that same independent forward, so a typo in a
hand-written gradient formula would be caught. The production module is never
used to build an expectation, and no code under test is mocked.
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
        bias_geglu,
        bias_geglu_back,
        bias_geglu_impl,
        geglu,
        geglu_back,
        quick_geglu,
        quick_geglu_back,
        weighted_quick_geglu,
        weighted_quick_geglu_back,
    )

    HAS_PADDLE = True
    _IMPORT_ERROR = ""
except ImportError as exc:  # honest: dependency missing, not a swallowed bug
    HAS_PADDLE = False
    _IMPORT_ERROR = repr(exc)

# --- Independent NumPy references (never call production code) ---------------
# tanh-approx GELU coefficients: sqrt(2/pi) = 0.79788456, cubic = 0.044715.
# quick-gelu slope = 1.702. All reference math runs in float64.


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _gelu_tanh_ref(a):
    """tanh approximation of GELU, elementwise."""
    inner = 0.79788456 * a * (1.0 + 0.044715 * a * a)
    return a * 0.5 * (1.0 + np.tanh(inner))


def _quick_gelu_ref(y):
    """quick_gelu(y) = y * sigmoid(1.702 * y)."""
    return y * _sigmoid(1.702 * y)


def _split_last(arr):
    """Split the last axis into two equal halves (matches paddle.chunk)."""
    half = arr.shape[-1] // 2
    return arr[..., :half], arr[..., half:]


def _fd(fn, x, h=1e-4):
    """Central finite-difference derivative of an elementwise ``fn``.

    Independent of the analytic gradient formulas shipped in production.
    """
    return (fn(x + h) - fn(x - h)) / (2.0 * h)


def _geglu_ref(y):
    y1, y2 = _split_last(y)
    return _gelu_tanh_ref(y1) * y2


def _geglu_back_ref(g, y):
    """d/dy of GELU(y1)*y2 for upstream g, concatenated as [dy1, dy2]."""
    y1, y2 = _split_last(y)
    dy1 = g * y2 * _fd(_gelu_tanh_ref, y1)
    dy2 = g * _gelu_tanh_ref(y1)
    return np.concatenate([dy1, dy2], axis=-1)


def _quick_geglu_ref(y, offset=0.0):
    y1, y2 = _split_last(y)
    return _quick_gelu_ref(y1) * (y2 + offset)


def _quick_geglu_back_ref(g, y, offset=0.0):
    y1, y2 = _split_last(y)
    dy1 = g * (y2 + offset) * _fd(_quick_gelu_ref, y1)
    dy2 = g * _quick_gelu_ref(y1)
    return np.concatenate([dy1, dy2], axis=-1)


def _weighted_quick_geglu_back_ref(g, y, weights, offset=0.0):
    input_grad = _quick_geglu_back_ref(g * weights, y, offset)
    weights_grad = (_quick_geglu_ref(y, offset) * g).sum(axis=-1, keepdims=True)
    return input_grad, weights_grad


def _t(arr):
    """float32 paddle tensor from a NumPy array."""
    return paddle.to_tensor(np.asarray(arr, dtype="float32"))


class _CPUFixture(unittest.TestCase):
    """Force CPU execution and restore the global device afterwards."""

    def setUp(self):
        self._orig_device = paddle.get_device()
        paddle.set_device("cpu")
        self.addCleanup(lambda: paddle.set_device(self._orig_device))


# Fixed, non-degenerate inputs whose two halves differ, so swapping halves or
# dropping the gate would change the result. Rows differ so per-row mixups show.
_Y = np.array([[1.0, -1.0, 2.0, 3.0], [0.5, -0.5, 1.5, -1.5]], dtype="float32")
_G = np.array(
    [[0.7, -0.3], [1.2, 0.4]], dtype="float32"
)  # upstream grad, matches [N, H]


@unittest.skipUnless(
    HAS_PADDLE, "paddle not importable in this environment: " + _IMPORT_ERROR
)
class TestGegluForward(_CPUFixture):
    def test_matches_independent_reference(self):
        out = geglu(_t(_Y))
        self.assertEqual(out.shape, [2, 2])
        np.testing.assert_allclose(
            out.numpy(), _geglu_ref(_Y), rtol=1e-5, atol=1e-6
        )

    def test_scalar_anchor(self):
        # Hand-computed: gelu_tanh(1)=0.8411919903841787; *2 = 1.6823839807...
        out = geglu(_t([[1.0, 2.0]]))
        self.assertAlmostEqual(
            float(out.numpy()[0, 0]), 1.6823839807683574, places=5
        )


@unittest.skipUnless(
    HAS_PADDLE, "paddle not importable in this environment: " + _IMPORT_ERROR
)
class TestBiasGegluForward(_CPUFixture):
    def test_bias_added_before_geglu(self):
        bias = np.array([0.5, -0.5, 1.0, -1.0], dtype="float32")
        # Production signature is bias_geglu(bias, y): computes y + bias.
        out = bias_geglu(_t(bias), _t(_Y))
        expected = _geglu_ref(_Y + bias)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-6)
        # Bias genuinely shifts the input: differs from the un-biased path.
        self.assertFalse(
            np.allclose(expected, _geglu_ref(_Y), rtol=1e-5, atol=1e-6)
        )


@unittest.skipUnless(
    HAS_PADDLE, "paddle not importable in this environment: " + _IMPORT_ERROR
)
class TestQuickGegluForward(_CPUFixture):
    def test_matches_independent_reference(self):
        out = quick_geglu(_t(_Y))
        self.assertEqual(out.shape, [2, 2])
        np.testing.assert_allclose(
            out.numpy(), _quick_geglu_ref(_Y, 0.0), rtol=1e-5, atol=1e-6
        )

    def test_scalar_anchor(self):
        # Hand-computed: quick_gelu(1)=0.8457957659328212; *2 = 1.6915915318...
        out = quick_geglu(_t([[1.0, 2.0]]))
        self.assertAlmostEqual(
            float(out.numpy()[0, 0]), 1.6915915318656425, places=5
        )

    def test_linear_offset_is_consumed(self):
        out0 = quick_geglu(_t(_Y), linear_offset=0.0)
        out_off = quick_geglu(_t(_Y), linear_offset=0.5)
        exp0 = _quick_geglu_ref(_Y, 0.0)
        exp_off = _quick_geglu_ref(_Y, 0.5)
        np.testing.assert_allclose(out0.numpy(), exp0, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(
            out_off.numpy(), exp_off, rtol=1e-5, atol=1e-6
        )
        self.assertFalse(np.allclose(exp0, exp_off, rtol=1e-5, atol=1e-6))


@unittest.skipUnless(
    HAS_PADDLE, "paddle not importable in this environment: " + _IMPORT_ERROR
)
class TestWeightedQuickGegluForward(_CPUFixture):
    def test_per_token_weight_applied(self):
        weights = np.array([[2.0], [3.0]], dtype="float32")
        out = weighted_quick_geglu(_t(_Y), _t(weights))
        self.assertEqual(out.shape, [2, 2])
        expected = _quick_geglu_ref(_Y, 0.0) * weights
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-6)
        # Guard against per-token weights being swapped between rows.
        swapped = _quick_geglu_ref(_Y, 0.0) * weights[::-1]
        self.assertFalse(np.allclose(expected, swapped, rtol=1e-5, atol=1e-6))


@unittest.skipUnless(
    HAS_PADDLE, "paddle not importable in this environment: " + _IMPORT_ERROR
)
class TestGegluBackward(_CPUFixture):
    def test_matches_finite_difference_reference(self):
        out = geglu_back(_t(_G), _t(_Y))
        self.assertEqual(out.shape, [2, 4])
        np.testing.assert_allclose(
            out.numpy(),
            _geglu_back_ref(_G.astype("float64"), _Y.astype("float64")),
            rtol=1e-3,
            atol=1e-4,
        )


@unittest.skipUnless(
    HAS_PADDLE, "paddle not importable in this environment: " + _IMPORT_ERROR
)
class TestBiasGegluBackward(_CPUFixture):
    def test_gradient_uses_biased_input(self):
        bias = np.array([0.5, -0.5, 1.0, -1.0], dtype="float32")
        out = bias_geglu_back(_t(_G), _t(_Y), _t(bias))
        expected = _geglu_back_ref(
            _G.astype("float64"), (_Y + bias).astype("float64")
        )
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-3, atol=1e-4)
        # Bias shifts the operating point: differs from the un-biased gradient.
        unbiased = _geglu_back_ref(_G.astype("float64"), _Y.astype("float64"))
        self.assertFalse(np.allclose(expected, unbiased, rtol=1e-3, atol=1e-4))


@unittest.skipUnless(
    HAS_PADDLE, "paddle not importable in this environment: " + _IMPORT_ERROR
)
class TestQuickGegluBackward(_CPUFixture):
    def test_matches_finite_difference_reference(self):
        out = quick_geglu_back(_t(_G), _t(_Y))
        self.assertEqual(out.shape, [2, 4])
        np.testing.assert_allclose(
            out.numpy(),
            _quick_geglu_back_ref(_G.astype("float64"), _Y.astype("float64")),
            rtol=1e-3,
            atol=1e-4,
        )

    def test_offset_shifts_gate_gradient(self):
        out0 = quick_geglu_back(_t(_G), _t(_Y), linear_offset=0.0)
        out_off = quick_geglu_back(_t(_G), _t(_Y), linear_offset=0.5)
        ref_off = _quick_geglu_back_ref(
            _G.astype("float64"), _Y.astype("float64"), 0.5
        )
        np.testing.assert_allclose(
            out_off.numpy(), ref_off, rtol=1e-3, atol=1e-4
        )
        self.assertFalse(
            np.allclose(out0.numpy(), out_off.numpy(), rtol=1e-3, atol=1e-4)
        )


@unittest.skipUnless(
    HAS_PADDLE, "paddle not importable in this environment: " + _IMPORT_ERROR
)
class TestWeightedQuickGegluBackward(_CPUFixture):
    def test_input_and_weight_gradients(self):
        weights = np.array([[2.0], [3.0]], dtype="float32")
        input_grad, weights_grad = weighted_quick_geglu_back(
            _t(_G), _t(_Y), _t(weights)
        )
        ig_ref, wg_ref = _weighted_quick_geglu_back_ref(
            _G.astype("float64"),
            _Y.astype("float64"),
            weights.astype("float64"),
            0.0,
        )
        self.assertEqual(input_grad.shape, [2, 4])
        self.assertEqual(weights_grad.shape, [2, 1])
        np.testing.assert_allclose(
            input_grad.numpy(), ig_ref, rtol=1e-3, atol=1e-4
        )
        np.testing.assert_allclose(
            weights_grad.numpy(), wg_ref, rtol=1e-4, atol=1e-5
        )


@unittest.skipUnless(
    HAS_PADDLE, "paddle not importable in this environment: " + _IMPORT_ERROR
)
class TestBiasGegluImpl(_CPUFixture):
    def test_2d_no_bias_matches_geglu(self):
        out = bias_geglu_impl(_t(_Y), None)
        self.assertEqual(out.shape, [2, 2])
        np.testing.assert_allclose(
            out.numpy(), _geglu_ref(_Y), rtol=1e-5, atol=1e-6
        )

    def test_2d_with_bias(self):
        bias = np.array([0.5, -0.5, 1.0, -1.0], dtype="float32")
        out = bias_geglu_impl(_t(_Y), _t(bias))
        np.testing.assert_allclose(
            out.numpy(), _geglu_ref(_Y + bias), rtol=1e-5, atol=1e-6
        )

    def test_3d_preserves_leading_shape_and_content(self):
        y3d = _Y.reshape(1, 2, 4)
        out = bias_geglu_impl(_t(y3d), None)
        self.assertEqual(out.shape, [1, 2, 2])
        np.testing.assert_allclose(
            out.numpy(), _geglu_ref(y3d), rtol=1e-5, atol=1e-6
        )

    def test_1d_input_rejected(self):
        with self.assertRaises(AssertionError):
            bias_geglu_impl(_t([1.0, 2.0, 3.0, 4.0]), None)

    def test_4d_input_rejected(self):
        with self.assertRaises(AssertionError):
            bias_geglu_impl(_t(_Y.reshape(1, 1, 2, 4)), None)


@unittest.skipUnless(
    HAS_PADDLE, "paddle not importable in this environment: " + _IMPORT_ERROR
)
class TestPyLayerBackward(_CPUFixture):
    def test_geglu_function_backward(self):
        x = _t(_Y)
        x.stop_gradient = False
        out = GeGLUFunction.apply(x)
        np.testing.assert_allclose(
            out.numpy(), _geglu_ref(_Y), rtol=1e-5, atol=1e-6
        )
        out.backward(_t(_G))
        self.assertIsNotNone(x.grad)
        expected = _geglu_back_ref(_G.astype("float64"), _Y.astype("float64"))
        np.testing.assert_allclose(
            x.grad.numpy(), expected, rtol=1e-3, atol=1e-4
        )

    def test_bias_geglu_function_backward(self):
        bias_np = np.array([0.5, -0.5, 1.0, -1.0], dtype="float32")
        x = _t(_Y)
        bias = _t(bias_np)
        x.stop_gradient = False
        bias.stop_gradient = False
        out = BiasGeGLUFunction.apply(x, bias)
        np.testing.assert_allclose(
            out.numpy(), _geglu_ref(_Y + bias_np), rtol=1e-5, atol=1e-6
        )
        out.backward(_t(_G))
        # input grad == geglu_back at the biased input.
        tmp = _geglu_back_ref(
            _G.astype("float64"), (_Y + bias_np).astype("float64")
        )
        self.assertIsNotNone(x.grad)
        np.testing.assert_allclose(x.grad.numpy(), tmp, rtol=1e-3, atol=1e-4)
        # bias grad == input grad reduced (summed) onto the bias shape [4].
        self.assertIsNotNone(bias.grad)
        self.assertEqual(bias.grad.shape, [4])
        np.testing.assert_allclose(
            bias.grad.numpy(), tmp.sum(axis=0), rtol=1e-3, atol=1e-4
        )


if __name__ == "__main__":
    unittest.main()
