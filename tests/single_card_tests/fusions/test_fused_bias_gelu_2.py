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

"""CPU behavior tests for the backward-gradient slice of
``paddlefleet.fusions.fused_bias_gelu``.

Slice under test (kept disjoint from the sibling forward/ctx/dispatch file):
``bias_gelu_back`` -- the analytic derivative of the tanh approximation of GELU,
including how the upstream gradient ``g`` and the ``x = bias + y`` combination
are consumed.

The production ``jit_fuser`` decorator is an identity no-op, so ``bias_gelu_back``
is plain elementwise paddle arithmetic that executes on CPU. Expected values are
built WITHOUT the production module:

* a central finite-difference gradient of an *independently* defined NumPy
  tanh-approx GELU (basic-operation reference; makes no assumption about the
  analytic form of the production backward, so a wrong coefficient is caught);
* exact hand-derived anchors that hold for any reasonable GELU approximation:
  gelu'(0) = 0.5, and the saturation limits gelu'(+inf) = 1, gelu'(-inf) = 0.
"""

import math
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

    from paddlefleet.fusions.fused_bias_gelu import bias_gelu_back

    HAS_PADDLE = True
    _IMPORT_ERROR = ""
except ImportError as exc:  # honest: dependency missing, not a swallowed bug
    HAS_PADDLE = False
    _IMPORT_ERROR = repr(exc)


# --- Independent NumPy reference (does NOT call production code) ------------
# tanh-approx GELU:  g(a) = 0.5 * a * (1 + tanh(sqrt(2/pi) * (a + 0.044715 a^3)))
# The mathematically exact sqrt(2/pi) is used; production rounds it to
# 0.79788456, a ~1e-8 relative difference well inside the tolerances below.
_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


def _gelu_tanh_ref(a):
    """Independent tanh-approx GELU forward, evaluated in float64."""
    a = np.asarray(a, dtype=np.float64)
    inner = _SQRT_2_OVER_PI * (a + 0.044715 * a**3)
    return 0.5 * a * (1.0 + np.tanh(inner))


def _gelu_tanh_grad_numeric(a, h=1e-4):
    """Central finite-difference derivative of the independent forward.

    Central differences give O(h^2) truncation (~1e-9 here) and negligible
    float64 round-off, so this is an independent anchor accurate far below the
    float32 precision of the production kernel.
    """
    a = np.asarray(a, dtype=np.float64)
    return (_gelu_tanh_ref(a + h) - _gelu_tanh_ref(a - h)) / (2.0 * h)


class _CPUFixture(unittest.TestCase):
    """Common CPU device setup with global-state restoration."""

    def setUp(self):
        self._orig_device = paddle.get_device()
        paddle.set_device("cpu")
        self.addCleanup(lambda: paddle.set_device(self._orig_device))


@unittest.skipUnless(
    HAS_PADDLE, "paddle not importable in this environment: " + _IMPORT_ERROR
)
class TestBiasGeluBack(_CPUFixture):
    # Fixed, distinguishable inputs spanning negative / zero / small / large.
    X = np.array(
        [[-3.0, -1.0, -0.25, 0.0, 0.5, 1.0, 2.5, 4.0]], dtype="float32"
    )

    def test_matches_independent_numerical_gradient(self):
        # x = bias + y; use ones upstream so the output equals d/dx GELU(x).
        bias = np.array(
            [-0.5, -0.25, 0.75, 0.0, -0.5, 0.4, 1.5, 1.0], dtype="float32"
        )
        y = (self.X - bias).astype("float32")
        g = np.ones_like(self.X, dtype="float32")

        out = bias_gelu_back(
            paddle.to_tensor(g),
            paddle.to_tensor(bias),
            paddle.to_tensor(y),
        )
        expected = _gelu_tanh_grad_numeric(self.X)
        np.testing.assert_allclose(out.numpy(), expected, rtol=2e-3, atol=1e-4)

    def test_gradient_at_zero_is_one_half(self):
        # At x = 0: tanh(0) = 0, so ff = 0.5 * (1 + 0) = 0.5 exactly, and this
        # holds for ANY choice of the approximation constants.
        bias = np.array([0.7, -1.5, 3.0], dtype="float32")
        y = (-bias).astype("float32")  # bias + y == 0 elementwise
        g = np.ones([3], dtype="float32")
        out = bias_gelu_back(
            paddle.to_tensor(g),
            paddle.to_tensor(bias),
            paddle.to_tensor(y),
        )
        np.testing.assert_allclose(
            out.numpy(), np.full([3], 0.5, dtype="float64"), atol=1e-6
        )

    def test_saturation_limits(self):
        # For |x| large, tanh(arg) saturates to +/-1, so (1 - tanh^2) -> 0 and
        # ff -> 0.5 * (1 + tanh) which is 1.0 for x >> 0 and 0.0 for x << 0.
        # These limits are independent of the exact constants.
        bias = np.array([10.0, -10.0], dtype="float32")
        y = np.array([20.0, -20.0], dtype="float32")  # x = 30, -30
        g = np.ones([2], dtype="float32")
        out = bias_gelu_back(
            paddle.to_tensor(g),
            paddle.to_tensor(bias),
            paddle.to_tensor(y),
        ).numpy()
        np.testing.assert_allclose(out, [1.0, 0.0], atol=1e-6)

    def test_upstream_gradient_multiplied_elementwise(self):
        # bias_gelu_back returns ff * g; a non-uniform g must scale each element
        # of the derivative independently.
        bias = np.array(
            [0.2, -0.3, 1.0, 0.0, -0.5, 0.6, -1.2, 0.9], dtype="float32"
        )
        y = (self.X - bias).astype("float32")
        g = np.array(
            [[1.5, -2.0, 0.5, 3.0, -1.0, 2.5, -0.5, 4.0]], dtype="float32"
        )

        ff_ref = _gelu_tanh_grad_numeric(self.X)  # derivative with g == 1
        expected = ff_ref * g.astype("float64")
        out = bias_gelu_back(
            paddle.to_tensor(g),
            paddle.to_tensor(bias),
            paddle.to_tensor(y),
        )
        np.testing.assert_allclose(out.numpy(), expected, rtol=2e-3, atol=1e-4)
        # Guard: g is genuinely consumed (every |g_i| != 1), so the scaled
        # output must differ from the bare derivative.
        self.assertFalse(np.allclose(out.numpy(), ff_ref, rtol=2e-3, atol=1e-4))

    def test_depends_only_on_bias_plus_sum(self):
        # The kernel uses x = bias + y; two different decompositions of the same
        # sum must give identical gradients, and both must match the independent
        # numerical derivative of that sum.
        g = np.ones_like(self.X, dtype="float32")
        bias_a = np.array(
            [-1.0, 0.5, -0.25, 2.0, 0.0, -0.6, 1.0, -1.0], dtype="float32"
        )
        y_a = (self.X - bias_a).astype("float32")
        bias_b = np.array(
            [0.3, -0.7, 1.1, -0.4, 0.5, 0.2, -0.5, 2.0], dtype="float32"
        )
        y_b = (self.X - bias_b).astype("float32")

        out_a = bias_gelu_back(
            paddle.to_tensor(g),
            paddle.to_tensor(bias_a),
            paddle.to_tensor(y_a),
        ).numpy()
        out_b = bias_gelu_back(
            paddle.to_tensor(g),
            paddle.to_tensor(bias_b),
            paddle.to_tensor(y_b),
        ).numpy()

        np.testing.assert_allclose(out_a, out_b, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(
            out_a, _gelu_tanh_grad_numeric(self.X), rtol=2e-3, atol=1e-4
        )


if __name__ == "__main__":
    unittest.main()
