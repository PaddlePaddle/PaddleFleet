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

"""CPU behavior tests for the GEGLU/Quick-GEGLU forward slice of
``paddlefleet.fusions.fused_bias_geglu``.

Slice under test (kept disjoint from sibling coverage files):
``quick_gelu``, ``GeGLUFunction``, ``BiasGeGLUFunction``,
``WeightedQuickGeGLUFunction`` and ``WeightedBiasQuickGeGLUFunction`` forward.

The production ``jit_fuser`` decorator is an identity no-op, so these are plain
paddle tensor ops that execute on CPU. Expected values are hand-derived with an
independent NumPy implementation of the tanh-approx GELU and the sigmoid-approx
quick GELU; the production module is never used to build the expectations.
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
        WeightedBiasQuickGeGLUFunction,
        WeightedQuickGeGLUFunction,
        quick_gelu,
    )

    HAS_PADDLE = True
    _IMPORT_ERROR = ""
except ImportError as exc:  # honest: dependency missing, not a swallowed bug
    HAS_PADDLE = False
    _IMPORT_ERROR = repr(exc)

# --- Independent NumPy references (do NOT call production code) --------------
# Constants are the well-known tanh-approx GELU coefficients:
#   sqrt(2/pi) = 0.79788456, cubic term = 0.044715, quick-gelu slope = 1.702.


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _quick_gelu_ref(y):
    """quick_gelu(y) = y * sigmoid(1.702 * y)."""
    return y * _sigmoid(1.702 * y)


def _gelu_tanh_ref(a):
    """tanh approximation of GELU applied elementwise."""
    inner = 0.79788456 * a * (1.0 + 0.044715 * a * a)
    return a * 0.5 * (1.0 + np.tanh(inner))


def _split_last(arr):
    """Split the last axis into two equal halves (matches paddle.chunk)."""
    half = arr.shape[-1] // 2
    return arr[..., :half], arr[..., half:]


def _geglu_ref(y):
    y1, y2 = _split_last(y)
    return _gelu_tanh_ref(y1) * y2


def _quick_geglu_ref(y, offset=0.0):
    y1, y2 = _split_last(y)
    return _quick_gelu_ref(y1) * (y2 + offset)


class _CPUFixture(unittest.TestCase):
    """Common CPU device setup with global-state restoration."""

    def setUp(self):
        self._orig_device = paddle.get_device()
        paddle.set_device("cpu")
        self.addCleanup(lambda: paddle.set_device(self._orig_device))


@unittest.skipUnless(
    HAS_PADDLE, "paddle not importable in this environment: " + _IMPORT_ERROR
)
class TestQuickGelu(_CPUFixture):
    def test_matches_independent_sigmoid_reference(self):
        x = np.array([[-2.0, -0.5, 0.0, 1.0, 3.0]], dtype="float32")
        out = quick_gelu(paddle.to_tensor(x))
        np.testing.assert_allclose(
            out.numpy(), _quick_gelu_ref(x), rtol=1e-5, atol=1e-6
        )

    def test_zero_is_exact_zero(self):
        # sigmoid(0) = 0.5, so quick_gelu(0) = 0 * 0.5 = 0 exactly.
        out = quick_gelu(paddle.zeros([3], dtype="float32"))
        np.testing.assert_array_equal(
            out.numpy(), np.zeros([3], dtype="float32")
        )


@unittest.skipUnless(
    HAS_PADDLE, "paddle not importable in this environment: " + _IMPORT_ERROR
)
class TestGeGLUFunction(_CPUFixture):
    def test_forward_matches_reference(self):
        x = np.array([[1.0, -1.0, 2.0, 3.0]], dtype="float32")
        out = GeGLUFunction.apply(paddle.to_tensor(x))
        self.assertEqual(out.shape, [1, 2])
        np.testing.assert_allclose(
            out.numpy(), _geglu_ref(x), rtol=1e-5, atol=1e-6
        )


@unittest.skipUnless(
    HAS_PADDLE, "paddle not importable in this environment: " + _IMPORT_ERROR
)
class TestBiasGeGLUFunction(_CPUFixture):
    def test_bias_is_added_before_geglu(self):
        x = np.array([[1.0, -1.0, 2.0, 3.0]], dtype="float32")
        bias = np.array([0.5, -0.5, 1.0, -1.0], dtype="float32")
        out = BiasGeGLUFunction.apply(
            paddle.to_tensor(x), paddle.to_tensor(bias)
        )
        expected = _geglu_ref(x + bias)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-6)
        # Bias must actually be consumed: result differs from the no-bias path.
        self.assertFalse(
            np.allclose(expected, _geglu_ref(x), rtol=1e-5, atol=1e-6)
        )


@unittest.skipUnless(
    HAS_PADDLE, "paddle not importable in this environment: " + _IMPORT_ERROR
)
class TestWeightedQuickGeGLUFunction(_CPUFixture):
    def _run(self, x, weights, offset):
        return WeightedQuickGeGLUFunction.apply(
            paddle.to_tensor(x),
            paddle.to_tensor(weights),
            False,
            paddle.to_tensor(offset),
        )

    def test_forward_applies_per_token_weight(self):
        x = np.array(
            [[1.0, -1.0, 2.0, 3.0], [0.5, -0.5, 1.5, -1.5]], dtype="float32"
        )
        weights = np.array([[2.0], [3.0]], dtype="float32")
        out = self._run(x, weights, 0.0)
        self.assertEqual(out.shape, [2, 2])
        expected = _quick_geglu_ref(x, 0.0) * weights
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-6)

    def test_linear_offset_is_consumed(self):
        x = np.array(
            [[1.0, -1.0, 2.0, 3.0], [0.5, -0.5, 1.5, -1.5]], dtype="float32"
        )
        weights = np.array([[2.0], [3.0]], dtype="float32")
        out0 = self._run(x, weights, 0.0)
        out_off = self._run(x, weights, 0.5)
        exp0 = _quick_geglu_ref(x, 0.0) * weights
        exp_off = _quick_geglu_ref(x, 0.5) * weights
        np.testing.assert_allclose(out0.numpy(), exp0, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(
            out_off.numpy(), exp_off, rtol=1e-5, atol=1e-6
        )
        # Offset changes the output; guards against it being ignored.
        self.assertFalse(np.allclose(exp0, exp_off, rtol=1e-5, atol=1e-6))


@unittest.skipUnless(
    HAS_PADDLE, "paddle not importable in this environment: " + _IMPORT_ERROR
)
class TestWeightedBiasQuickGeGLUFunction(_CPUFixture):
    def test_forward_bias_weight_and_offset(self):
        x = np.array(
            [[1.0, -1.0, 2.0, 3.0], [0.5, -0.5, 1.5, -1.5]], dtype="float32"
        )
        bias = np.array([0.5, 0.5, -1.0, 1.0], dtype="float32")
        weights = np.array([[2.0], [3.0]], dtype="float32")
        out = WeightedBiasQuickGeGLUFunction.apply(
            paddle.to_tensor(x),
            paddle.to_tensor(bias),
            paddle.to_tensor(weights),
            False,
            paddle.to_tensor(0.5),
        )
        self.assertEqual(out.shape, [2, 2])
        expected = _quick_geglu_ref(x + bias, 0.5) * weights
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-6)
        # Bias must be added before activation: differs from the no-bias path.
        self.assertFalse(
            np.allclose(
                expected,
                _quick_geglu_ref(x, 0.5) * weights,
                rtol=1e-5,
                atol=1e-6,
            )
        )


if __name__ == "__main__":
    unittest.main()
