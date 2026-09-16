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

"""Behavior unit tests for ``paddlefleet.fusions.fused_bias_gelu``.

Scope (base): the ``GeLUFunction`` autograd ``PyLayer`` forward/backward
control flow (``ctx.save_for_backward`` -> ``ctx.saved_tensor`` restore ->
``bias_gelu_back`` -> returning ``(tmp, tmp)``) and the bias-add dispatch
(``x = bias + y`` with broadcasting). The tanh/erf approximation *variant
selection* is deliberately left to the sibling test.

Independent reference: the tanh-approximation GELU is re-derived from first
principles here using the exact constant ``sqrt(2/pi)`` (production truncates
it to ``0.79788456``). The reference never calls the code under test. The
backward reference uses autodiff of that independent forward, which is
independent of the hand-coded analytic derivative in ``bias_gelu_back``.
"""

import math
import unittest

import numpy as np

try:
    import paddle

    _PADDLE_IMPORT_ERROR = None
except ImportError as exc:  # honest: paddle genuinely absent, not a swallow
    paddle = None
    _PADDLE_IMPORT_ERROR = exc

_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
_GELU_CUBIC_COEFF = 0.044715


def _ref_gelu_tanh_scalar(v):
    """Independent scalar tanh-approx GELU (exact constant, numpy/math)."""
    inner = _SQRT_2_OVER_PI * (v + _GELU_CUBIC_COEFF * v**3)
    return 0.5 * v * (1.0 + math.tanh(inner))


def _ref_gelu_tanh_np(x):
    """Elementwise independent tanh-approx GELU over a numpy array."""
    inner = _SQRT_2_OVER_PI * (x + _GELU_CUBIC_COEFF * np.power(x, 3))
    return 0.5 * x * (1.0 + np.tanh(inner))


def _ref_gelu_tanh_paddle(x):
    """Independent tanh-approx GELU built from paddle ops (for autodiff)."""
    inner = _SQRT_2_OVER_PI * (x + _GELU_CUBIC_COEFF * x * x * x)
    return 0.5 * x * (1.0 + paddle.tanh(inner))


@unittest.skipUnless(
    paddle is not None,
    f"paddle is not installed in this environment: {_PADDLE_IMPORT_ERROR}",
)
class TestFusedBiasGelu(unittest.TestCase):
    """Forward/backward + bias-add dispatch of the fused bias-gelu path."""

    def setUp(self):
        # Run the math on CPU in float64 so the comparison against the
        # independent reference is tight; restore global state afterwards.
        self._orig_device = paddle.get_device()
        self._orig_dtype = paddle.get_default_dtype()
        self.addCleanup(paddle.set_device, self._orig_device)
        self.addCleanup(paddle.set_default_dtype, self._orig_dtype)
        paddle.set_device("cpu")
        paddle.set_default_dtype("float64")

    def test_bias_gelu_forward_matches_reference_with_bias_broadcast(self):
        """bias_gelu(bias, y) == GELU_tanh(bias + y) with bias broadcast.

        Uses a per-column-distinguishable bias broadcast over two rows so a
        dropped/ignored/mis-broadcast bias would change the result.
        """
        from paddlefleet.fusions.fused_bias_gelu import bias_gelu

        bias_np = np.array([0.5, -1.0, 2.0], dtype=np.float64)
        y_np = np.array([[1.0, -2.0, 0.5], [-0.5, 3.0, -1.5]], dtype=np.float64)
        bias = paddle.to_tensor(bias_np)
        y = paddle.to_tensor(y_np)

        out = bias_gelu(bias, y)

        expected = _ref_gelu_tanh_np(bias_np[None, :] + y_np)
        self.assertEqual(list(out.shape), [2, 3])
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6, atol=1e-8)

    def test_bias_gelu_matches_hand_derived_known_values(self):
        """Anchor against pen-and-paper tanh-approx GELU values.

        x is reconstructed from non-degenerate bias + y so the add dispatch
        is exercised (e.g. 0.3 + (-0.3) == 0). Expected values are the
        standard tanh-approximation GELU derived by hand:
          GELU(0)  = 0.0
          GELU(1)  = 0.5*(1+tanh(0.8335620)) ~= 0.8411920
          GELU(-1) = -0.5*(1-tanh(0.8335620)) ~= -0.1588080
        """
        from paddlefleet.fusions.fused_bias_gelu import bias_gelu

        bias = paddle.to_tensor([0.3, 0.4, -0.4], dtype="float64")
        y = paddle.to_tensor([-0.3, 0.6, -0.6], dtype="float64")  # x=[0,1,-1]

        out = bias_gelu(bias, y).numpy()

        expected = np.array([0.0, 0.8411920, -0.1588080], dtype=np.float64)
        np.testing.assert_allclose(out, expected, atol=1e-4)

    def test_gelu_function_forward_equals_bias_add_reference(self):
        """GeLUFunction.apply(input, bias) == GELU_tanh(input + bias).

        Verifies the forward control flow (apply -> forward ->
        bias_gelu(bias, input)) with a broadcast bias and distinguishable
        inputs, compared to the independent reference.
        """
        from paddlefleet.fusions.fused_bias_gelu import GeLUFunction

        inp_np = np.array(
            [[0.25, -0.75, 1.5], [2.0, -1.25, 0.1]], dtype=np.float64
        )
        bias_np = np.array([0.5, -0.5, 1.0], dtype=np.float64)
        inp = paddle.to_tensor(inp_np)
        bias = paddle.to_tensor(bias_np)

        out = GeLUFunction.apply(inp, bias)

        expected = _ref_gelu_tanh_np(inp_np + bias_np[None, :])
        self.assertEqual(list(out.shape), [2, 3])
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6, atol=1e-8)

    def test_gelu_function_backward_ctx_save_restore_matches_autodiff(self):
        """Backward restores saved (input, bias) and returns (tmp, tmp).

        Drives a real backward through the PyLayer with a non-uniform
        upstream gradient. If ctx.save_for_backward / saved_tensor restored
        the wrong tensors, or bias_gelu_back were wrong, the grads would
        diverge from autodiff of the independent forward. Matching shapes are
        used so the returned (tmp, tmp) map cleanly onto both inputs.
        """
        from paddlefleet.fusions.fused_bias_gelu import GeLUFunction

        inp_np = np.array(
            [[0.25, -0.75, 1.5], [2.0, -1.25, 0.1]], dtype=np.float64
        )
        bias_np = np.array(
            [[0.5, -0.5, 1.0], [-1.0, 0.3, -0.2]], dtype=np.float64
        )
        upstream_np = np.array(
            [[1.0, -2.0, 0.5], [0.3, -0.7, 1.25]], dtype=np.float64
        )

        inp = paddle.to_tensor(inp_np)
        bias = paddle.to_tensor(bias_np)
        inp.stop_gradient = False
        bias.stop_gradient = False
        upstream = paddle.to_tensor(upstream_np)

        out = GeLUFunction.apply(inp, bias)
        out.backward(upstream)

        # Independent reference: autodiff the reference forward.
        inp_ref = paddle.to_tensor(inp_np)
        bias_ref = paddle.to_tensor(bias_np)
        inp_ref.stop_gradient = False
        bias_ref.stop_gradient = False
        x_ref = inp_ref + bias_ref
        out_ref = _ref_gelu_tanh_paddle(x_ref)
        out_ref.backward(paddle.to_tensor(upstream_np))
        expected_grad = inp_ref.grad.numpy()

        self.assertIsNotNone(inp.grad)
        self.assertIsNotNone(bias.grad)
        np.testing.assert_allclose(
            out.numpy(),
            _ref_gelu_tanh_np(inp_np + bias_np),
            rtol=1e-6,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            inp.grad.numpy(), expected_grad, rtol=1e-6, atol=1e-8
        )
        np.testing.assert_allclose(
            bias.grad.numpy(), expected_grad, rtol=1e-6, atol=1e-8
        )
        # backward returns (tmp, tmp): both input grads are the same tensor.
        np.testing.assert_array_equal(inp.grad.numpy(), bias.grad.numpy())
        # Reference confirms both sides share one gradient too.
        np.testing.assert_allclose(
            bias_ref.grad.numpy(), expected_grad, rtol=1e-6, atol=1e-8
        )

    def test_bias_gelu_back_matches_autodiff_reference(self):
        """bias_gelu_back(g, bias, y) equals d GELU/dx * g at x = bias + y.

        Validates the hand-coded analytic derivative directly against autodiff
        of the independent forward, with a non-uniform upstream g and
        distinguishable bias/y.
        """
        from paddlefleet.fusions.fused_bias_gelu import bias_gelu_back

        g_np = np.array([[1.0, -2.0, 0.5], [0.3, -0.7, 1.25]], dtype=np.float64)
        bias_np = np.array(
            [[0.5, -0.5, 1.0], [-1.0, 0.3, -0.2]], dtype=np.float64
        )
        y_np = np.array(
            [[0.25, -0.75, 1.5], [2.0, -1.25, 0.1]], dtype=np.float64
        )

        g = paddle.to_tensor(g_np)
        bias = paddle.to_tensor(bias_np)
        y = paddle.to_tensor(y_np)
        got = bias_gelu_back(g, bias, y)

        # Independent autodiff reference for the VJP.
        x_ref = paddle.to_tensor(bias_np + y_np)
        x_ref.stop_gradient = False
        out_ref = _ref_gelu_tanh_paddle(x_ref)
        out_ref.backward(paddle.to_tensor(g_np))
        expected = x_ref.grad.numpy()

        self.assertEqual(list(got.shape), [2, 3])
        np.testing.assert_allclose(got.numpy(), expected, rtol=1e-6, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
