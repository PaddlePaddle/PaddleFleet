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

"""CPU behavior tests for the Quick-GEGLU *backward* slice of
``paddlefleet.fusions.fused_bias_geglu``.

Slice under test (kept disjoint from the forward-focused sibling
``test_fused_bias_geglu_2.py``):

* raw forward helpers ``quick_geglu`` / ``weighted_quick_geglu`` /
  ``weighted_bias_quick_geglu`` (validated against an independent NumPy
  reference),
* the hand-written analytic gradients ``quick_geglu_back`` /
  ``weighted_quick_geglu_back`` / ``weighted_bias_quick_geglu_back`` (validated
  against paddle autograd of an *independent* paddle reforward, so the analytic
  gradient code is never used to produce its own expectation),
* the ``weighted_bias_quick_geglu_impl`` dispatcher control flow.

``jit_fuser`` is an identity no-op, so these are plain paddle tensor ops that
run on CPU. No GPU-only path (e.g. fp8 input storage) is exercised here.
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
        quick_geglu,
        quick_geglu_back,
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
    _IMPORT_ERROR = repr(exc)


# --- Independent NumPy references (never call production code) ---------------
# quick-gelu slope 1.702 is the sigmoid-approx constant.


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _quick_gelu_ref(y):
    return y * _sigmoid(1.702 * y)


def _split_last(arr):
    half = arr.shape[-1] // 2
    return arr[..., :half], arr[..., half:]


def _quick_geglu_ref(y, offset=0.0):
    y1, y2 = _split_last(y)
    return _quick_gelu_ref(y1) * (y2 + offset)


# --- Independent paddle reforward, used only to drive autograd references ----
# These mirror the mathematical definition with paddle ops that differ from the
# production implementation (``paddle.split`` instead of ``paddle.chunk``), so
# differentiating them yields a reference independent of the analytic backward.


def _pd_quick_geglu(y, offset):
    y1, y2 = paddle.split(y, num_or_sections=2, axis=-1)
    return (y1 * paddle.nn.functional.sigmoid(1.702 * y1)) * (y2 + offset)


# Fixed, distinguishable fixtures (no degenerate all-equal / all-zero inputs).
_Y = np.array([[-1.0, 0.5, 2.0, -0.5], [0.3, -2.0, 1.5, 1.0]], dtype="float32")
_BIAS = np.array(
    [[0.1, -0.2, 0.3, 0.4], [-0.5, 0.6, -0.7, 0.8]], dtype="float32"
)
_W = np.array([[0.5], [2.0]], dtype="float32")
_G = np.array([[1.0, -2.0], [0.5, 3.0]], dtype="float32")  # upstream grad [N,H]


class _CPUFixture(unittest.TestCase):
    """Force CPU so numerics are CPU-observable; restore global device."""

    def setUp(self):
        self._orig_device = paddle.get_device()
        paddle.set_device("cpu")
        self.addCleanup(lambda: paddle.set_device(self._orig_device))


@unittest.skipUnless(
    HAS_PADDLE, "paddle not importable in this environment: " + _IMPORT_ERROR
)
class TestQuickGegluForward(_CPUFixture):
    def test_matches_numpy_reference(self):
        out = quick_geglu(paddle.to_tensor(_Y))
        self.assertEqual(out.shape, [2, 2])
        np.testing.assert_allclose(
            out.numpy(), _quick_geglu_ref(_Y, 0.0), rtol=1e-5, atol=1e-6
        )

    def test_linear_offset_is_consumed(self):
        off0 = quick_geglu(paddle.to_tensor(_Y), linear_offset=0.0).numpy()
        off = quick_geglu(paddle.to_tensor(_Y), linear_offset=0.75).numpy()
        np.testing.assert_allclose(
            off0, _quick_geglu_ref(_Y, 0.0), rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            off, _quick_geglu_ref(_Y, 0.75), rtol=1e-5, atol=1e-6
        )
        # Offset must actually reach the gate: y2 + offset shifts every element.
        self.assertFalse(np.allclose(off0, off, rtol=1e-5, atol=1e-6))


@unittest.skipUnless(
    HAS_PADDLE, "paddle not importable in this environment: " + _IMPORT_ERROR
)
class TestWeightedForward(_CPUFixture):
    def test_weighted_quick_geglu_applies_per_token_weight(self):
        out = weighted_quick_geglu(paddle.to_tensor(_Y), paddle.to_tensor(_W))
        expected = _quick_geglu_ref(_Y, 0.0) * _W
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-6)

    def test_weighted_bias_quick_geglu_adds_bias_first(self):
        out = weighted_bias_quick_geglu(
            paddle.to_tensor(_Y),
            paddle.to_tensor(_BIAS),
            paddle.to_tensor(_W),
            linear_offset=0.5,
        )
        expected = _quick_geglu_ref(_Y + _BIAS, 0.5) * _W
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-6)
        # Bias enters before the activation: differs from the no-bias path.
        self.assertFalse(
            np.allclose(
                expected, _quick_geglu_ref(_Y, 0.5) * _W, rtol=1e-5, atol=1e-6
            )
        )


@unittest.skipUnless(
    HAS_PADDLE, "paddle not importable in this environment: " + _IMPORT_ERROR
)
class TestQuickGegluBack(_CPUFixture):
    def _autograd_input_grad(self, offset):
        y = paddle.to_tensor(_Y, stop_gradient=False)
        out = _pd_quick_geglu(y, offset)
        out.backward(grad_tensor=paddle.to_tensor(_G))
        return y.grad.numpy()

    def test_matches_autograd_no_offset(self):
        got = quick_geglu_back(paddle.to_tensor(_G), paddle.to_tensor(_Y))
        np.testing.assert_allclose(
            got.numpy(), self._autograd_input_grad(0.0), rtol=1e-5, atol=1e-6
        )

    def test_matches_autograd_with_offset(self):
        got = quick_geglu_back(
            paddle.to_tensor(_G), paddle.to_tensor(_Y), linear_offset=0.75
        )
        np.testing.assert_allclose(
            got.numpy(), self._autograd_input_grad(0.75), rtol=1e-5, atol=1e-6
        )


@unittest.skipUnless(
    HAS_PADDLE, "paddle not importable in this environment: " + _IMPORT_ERROR
)
class TestWeightedQuickGegluBack(_CPUFixture):
    def test_input_and_weight_grads_match_autograd(self):
        offset = 0.5
        y = paddle.to_tensor(_Y, stop_gradient=False)
        w = paddle.to_tensor(_W, stop_gradient=False)
        out = _pd_quick_geglu(y, offset) * w
        out.backward(grad_tensor=paddle.to_tensor(_G))
        ref_in, ref_w = y.grad.numpy(), w.grad.numpy()

        input_grad, weights_grad = weighted_quick_geglu_back(
            paddle.to_tensor(_G),
            paddle.to_tensor(_Y),
            paddle.to_tensor(_W),
            offset,
        )
        self.assertEqual(input_grad.shape, [2, 4])
        self.assertEqual(weights_grad.shape, [2, 1])
        np.testing.assert_allclose(
            input_grad.numpy(), ref_in, rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            weights_grad.numpy(), ref_w, rtol=1e-5, atol=1e-6
        )


@unittest.skipUnless(
    HAS_PADDLE, "paddle not importable in this environment: " + _IMPORT_ERROR
)
class TestWeightedBiasQuickGegluBack(_CPUFixture):
    def test_all_grads_match_autograd_and_bias_equals_input(self):
        offset = 0.5
        # Non-broadcasting bias (same shape as y) so autograd's bias grad is not
        # reduced, matching production's ``bias_grad = input_grad`` contract.
        y = paddle.to_tensor(_Y, stop_gradient=False)
        bias = paddle.to_tensor(_BIAS, stop_gradient=False)
        w = paddle.to_tensor(_W, stop_gradient=False)
        out = _pd_quick_geglu(y + bias, offset) * w
        out.backward(grad_tensor=paddle.to_tensor(_G))
        ref_in, ref_bias, ref_w = (
            y.grad.numpy(),
            bias.grad.numpy(),
            w.grad.numpy(),
        )

        input_grad, bias_grad, weights_grad = weighted_bias_quick_geglu_back(
            paddle.to_tensor(_G),
            paddle.to_tensor(_Y),
            paddle.to_tensor(_BIAS),
            paddle.to_tensor(_W),
            offset,
        )
        np.testing.assert_allclose(
            input_grad.numpy(), ref_in, rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            bias_grad.numpy(), ref_bias, rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            weights_grad.numpy(), ref_w, rtol=1e-5, atol=1e-6
        )
        # Documented contract: bias gradient is identical to the input gradient.
        np.testing.assert_array_equal(bias_grad.numpy(), input_grad.numpy())


@unittest.skipUnless(
    HAS_PADDLE, "paddle not importable in this environment: " + _IMPORT_ERROR
)
class TestWeightedBiasQuickGegluImpl(_CPUFixture):
    def test_rejects_rank_4_input(self):
        # ``assert len(ori_shape) in [2, 3]`` guards the entry; a 4-D tensor is
        # rejected before any activation work happens.
        x = paddle.to_tensor(np.zeros((2, 2, 2, 4), dtype="float32"))
        w = paddle.to_tensor(np.ones((4, 1), dtype="float32"))
        with self.assertRaises(AssertionError):
            weighted_bias_quick_geglu_impl(x, None, w)

    def test_offset_tensor_construction_builds_and_matches_reference(self):
        """A valid no-bias 2D call returns the weighted quick-GEGLU value.

        ``weighted_bias_quick_geglu_impl`` builds the linear offset with
        ``paddle.tensor(linear_offset, dtype=..., device=input.device)``. Under
        the paddlefleet_ops torch-compat layer active in this runtime,
        ``paddle.tensor`` is a callable factory that accepts the ``device=``
        keyword, so the offset tensor is constructed and (with ``bias=None``)
        the impl routes to the weighted quick-GEGLU path and returns the
        per-token-weighted value. Checked against an INDEPENDENT NumPy
        reference at the default ``linear_offset=0.0``.
        """
        x_np = np.array(
            [[1.0, -1.0, 2.0, 3.0], [0.5, -0.5, 1.5, -1.5]],
            dtype="float32",
        )
        w_np = np.array([[2.0], [3.0]], dtype="float32")
        out = weighted_bias_quick_geglu_impl(
            paddle.to_tensor(x_np), None, paddle.to_tensor(w_np)
        ).numpy()
        expected = _quick_geglu_ref(x_np, 0.0) * w_np
        self.assertEqual(list(out.shape), [2, 2])
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
