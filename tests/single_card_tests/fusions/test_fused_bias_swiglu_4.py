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

"""CPU behavior tests for the (non-clamped) SwiGLU backward-wrapper slice of
``paddlefleet.fusions.fused_bias_swiglu``.

Slice under test (the helper names of this file's coverage source, kept
disjoint from the clamp-focused sibling ``test_clampswiglu_align.py`` and the
apply/shape-focused sibling coverage):

* ``swiglu_back``            -- native ``swiglu_grad`` kernel wrapper;
* ``bias_swiglu_back``       -- the ``y = y + bias`` wiring before backward;
* ``weighted_swiglu_back``   -- per-token-weighted input grad + weight grad;
* the ``cpu_offload_input=True`` branch of ``BiasSwiGLUFunction`` /
  ``SwiGLUFunction`` ``forward`` (the CPU-observable ``activation_offloading``
  flag plumbing + ctx save round trip).

Every expected value is hand-derived from an independent NumPy implementation
of sigmoid / SiLU / SwiGLU and its analytic vector-Jacobian product.  The
production module -- including ``paddle._C_ops.swiglu_grad`` and the autograd
VJP of ``F.swiglu`` (which would call that same kernel) -- is NEVER used to
build an expectation, so a bug inside the fused kernel or the wrappers cannot
hide behind a shared implementation.  Negative controls prove each comparison
is non-vacuous.

These wrappers are plain tensor ops (``jit_fuser`` is an identity no-op) and
run on CPU; no GPU/Triton-only numerics are exercised.  When paddle is not
importable the whole module is skipped with an honest dependency reason.
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
        SwiGLUFunction,
        bias_swiglu_back,
        swiglu_back,
        weighted_swiglu_back,
    )

    HAS_PADDLE = True
    _IMPORT_ERROR = ""
except ImportError as exc:  # honest: dependency missing, not a swallowed bug
    HAS_PADDLE = False
    _IMPORT_ERROR = repr(exc)


# --- Independent NumPy references (never call the production module) --------
# Work in float64 from the mathematical definitions so a same-way bug in the
# float32 production code cannot be masked by a shared implementation.


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def _silu(x):
    x = np.asarray(x, dtype=np.float64)
    return x * _sigmoid(x)


def _split_last(arr):
    """Split the last axis into two equal halves (matches paddle.chunk)."""
    arr = np.asarray(arr, dtype=np.float64)
    half = arr.shape[-1] // 2
    return arr[..., :half], arr[..., half:]


def ref_swiglu_fwd(y):
    """SwiGLU forward: SiLU(gate) * value, gate = first half, value = second."""
    gate, value = _split_last(y)
    return _silu(gate) * value


def ref_swiglu_back(g_up, y):
    """Analytic VJP of SwiGLU w.r.t. ``y`` for upstream grad ``g_up``.

    ``d SiLU(x)/dx = sig(x) * (1 + x * (1 - sig(x)))`` and ``SiLU(x)=x*sig(x)``.
    Output layout is ``concat([d_gate, d_value])`` to match the ``chunk`` order.
    """
    g_up = np.asarray(g_up, dtype=np.float64)
    gate, value = _split_last(y)
    sig = _sigmoid(gate)
    d_gate = g_up * value * sig * (1.0 + gate * (1.0 - sig))
    d_value = g_up * _silu(gate)
    return np.concatenate([d_gate, d_value], axis=-1)


# --- Fixed, distinguishable, non-degenerate fixtures ------------------------
# Distinct nonzero values of both signs so swapped halves / dropped bias /
# ignored weights / sign flips are all visible.
_Y_4x8 = np.array(
    [
        [-1.5, 0.7, 2.0, -0.3, 1.1, -0.8, 0.4, -2.2],
        [0.9, -1.2, 0.5, 1.8, -0.6, 2.1, -0.9, 0.3],
        [0.2, -0.4, 1.6, -1.9, 0.8, -0.5, 1.3, -0.7],
        [-2.1, 1.0, -0.6, 0.6, -1.4, 0.9, 2.3, -0.2],
    ],
    dtype=np.float32,
)
_G_4x4 = np.array(
    [
        [1.3, -0.6, 0.8, -1.1],
        [0.5, 2.2, -0.7, 1.4],
        [-1.0, 0.4, 1.7, -0.9],
        [0.6, -1.5, 0.3, 2.0],
    ],
    dtype=np.float32,
)
_BIAS_8 = np.array(
    [0.1, -0.2, 0.3, -0.4, 0.15, -0.25, 0.35, -0.45], dtype=np.float32
)
_W_4x1 = np.array([[2.0], [-1.5], [0.5], [-0.8]], dtype=np.float32)

_RTOL = 1e-5
_ATOL = 1e-6


@unittest.skipUnless(
    HAS_PADDLE, "paddle not importable in this environment: " + _IMPORT_ERROR
)
class _CPUFixture(unittest.TestCase):
    """Force CPU execution and restore the global device afterwards."""

    def setUp(self):
        self._orig_device = paddle.get_device()
        paddle.set_device("cpu")
        self.addCleanup(lambda: paddle.set_device(self._orig_device))

    def _t(self, arr):
        return paddle.to_tensor(arr, dtype="float32")


class TestSwigluBack(_CPUFixture):
    def test_matches_independent_analytic_vjp(self):
        out = swiglu_back(self._t(_G_4x4), self._t(_Y_4x8))
        expected = ref_swiglu_back(_G_4x4, _Y_4x8)
        # last dim of the grad equals the (doubled) input dim, not the upstream.
        self.assertEqual(out.shape, [4, 8])
        np.testing.assert_allclose(
            out.numpy(), expected, rtol=_RTOL, atol=_ATOL
        )
        # Negative control: a sign-flipped reference must be rejected, proving
        # the comparison is not vacuous.
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(
                out.numpy(), -expected, rtol=_RTOL, atol=_ATOL
            )

    def test_gate_and_value_halves_are_not_swapped(self):
        # gate = [1.0], value = [2.0], upstream = [1.0].
        #   d_value = g * silu(gate) = silu(1.0)
        #   d_gate  = g * value * silu'(1.0)
        # Swapping the two halves would put silu(2.0)*... in the wrong slot, so
        # comparing against the correctly-ordered reference detects the swap.
        y = self._t([[1.0, 2.0]])
        g = self._t([[1.0]])
        out = swiglu_back(g, y).numpy()
        expected = ref_swiglu_back([[1.0]], [[1.0, 2.0]])
        np.testing.assert_allclose(out, expected, rtol=_RTOL, atol=_ATOL)
        # d_value slot must be silu(1.0), clearly different from silu(2.0).
        self.assertAlmostEqual(float(out[0, 1]), float(_silu(1.0)), places=6)
        self.assertNotAlmostEqual(float(out[0, 1]), float(_silu(2.0)), places=4)


class TestBiasSwigluBack(_CPUFixture):
    def test_adds_bias_before_backward(self):
        # bias_swiglu_back(g, y, bias) == d/dy swiglu(y + bias); because
        # d(y+bias)/dy == 1 this equals the VJP evaluated at (y + bias).
        out = bias_swiglu_back(
            self._t(_G_4x4), self._t(_Y_4x8), self._t(_BIAS_8)
        )
        expected = ref_swiglu_back(_G_4x4, _Y_4x8 + _BIAS_8)
        self.assertEqual(out.shape, [4, 8])
        np.testing.assert_allclose(
            out.numpy(), expected, rtol=_RTOL, atol=_ATOL
        )
        # The bias must genuinely shift the input: the result differs from the
        # no-bias backward beyond tolerance.
        no_bias = ref_swiglu_back(_G_4x4, _Y_4x8)
        self.assertGreater(np.abs(expected - no_bias).max(), 1e-2)
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(
                out.numpy(), no_bias, rtol=_RTOL, atol=_ATOL
            )


class TestWeightedSwigluBack(_CPUFixture):
    def test_input_and_weight_grads_match_reference(self):
        g, y, w = self._t(_G_4x4), self._t(_Y_4x8), self._t(_W_4x1)
        input_grad, weights_grad = weighted_swiglu_back(g, y, w)

        # input_grad == swiglu_back(g * weights, y)
        expected_input = ref_swiglu_back(_G_4x4 * _W_4x1, _Y_4x8)
        # weights_grad == sum(swiglu(y) * g, axis=-1, keepdim=True)
        expected_weights = np.sum(
            ref_swiglu_fwd(_Y_4x8) * _G_4x4.astype(np.float64),
            axis=-1,
            keepdims=True,
        )

        self.assertEqual(input_grad.shape, [4, 8])
        self.assertEqual(weights_grad.shape, [4, 1])
        np.testing.assert_allclose(
            input_grad.numpy(), expected_input, rtol=_RTOL, atol=_ATOL
        )
        np.testing.assert_allclose(
            weights_grad.numpy(), expected_weights, rtol=_RTOL, atol=_ATOL
        )

    def test_weights_actually_scale_the_input_grad(self):
        # With weights == 1 the input grad collapses to the plain swiglu_back;
        # the real per-row weights must produce a different, row-scaled grad.
        g, y = self._t(_G_4x4), self._t(_Y_4x8)
        weighted, _ = weighted_swiglu_back(g, y, self._t(_W_4x1))
        unit, _ = weighted_swiglu_back(
            g, y, self._t(np.ones((4, 1), np.float32))
        )
        # Row 0 uses weight 2.0 -> its input grad is 2x the unit-weight grad.
        np.testing.assert_allclose(
            weighted.numpy()[0], 2.0 * unit.numpy()[0], rtol=_RTOL, atol=_ATOL
        )
        self.assertGreater(np.abs(weighted.numpy() - unit.numpy()).max(), 1e-2)


class TestCpuOffloadInputForwardBranch(_CPUFixture):
    """The ``cpu_offload_input=True`` branch tags tensors for offloading and
    still returns the real SwiGLU forward.  ``ctx`` is the autograd-framework
    collaborator (not code under test) and is substituted by a mock so the
    stored state can be observed; the production ``bias_swiglu`` / ``swiglu``
    are executed for real."""

    def test_bias_variant_sets_flags_saves_and_computes(self):
        ctx = mock.MagicMock()
        inp = self._t(_Y_4x8)
        bias = self._t(_BIAS_8)

        out = BiasSwiGLUFunction.forward(ctx, inp, bias, False, True)

        # CPU-observable control flow: both operands tagged for offloading.
        self.assertTrue(inp.activation_offloading)
        self.assertTrue(bias.activation_offloading)
        # Real forward numerics: bias_swiglu == swiglu(input + bias).
        np.testing.assert_allclose(
            out.numpy(),
            ref_swiglu_fwd(_Y_4x8 + _BIAS_8),
            rtol=_RTOL,
            atol=_ATOL,
        )
        # ctx must receive the un-fp8 input + bias (identity, no copy) and the
        # book-keeping the backward pass depends on.
        ctx.save_for_backward.assert_called_once()
        saved_args, _ = ctx.save_for_backward.call_args
        self.assertIs(saved_args[0], inp)
        self.assertIs(saved_args[1], bias)
        self.assertEqual(ctx.ori_input_dtype, inp.dtype)
        self.assertFalse(ctx.fp8_input_store)
        self.assertIsNone(ctx.clamp_value)
        self.assertFalse(ctx.use_accuracy_compatible)

    def test_no_bias_variant_sets_flag_saves_and_computes(self):
        ctx = mock.MagicMock()
        inp = self._t(_Y_4x8)

        out = SwiGLUFunction.forward(ctx, inp, False, True)

        self.assertTrue(inp.activation_offloading)
        np.testing.assert_allclose(
            out.numpy(), ref_swiglu_fwd(_Y_4x8), rtol=_RTOL, atol=_ATOL
        )
        ctx.save_for_backward.assert_called_once()
        saved_args, _ = ctx.save_for_backward.call_args
        self.assertIs(saved_args[0], inp)
        self.assertEqual(ctx.ori_input_dtype, inp.dtype)

    def test_offload_flag_is_actually_gated(self):
        # With cpu_offload_input=False the branch must NOT tag the input, so the
        # flag routing is real and not unconditional.
        ctx = mock.MagicMock()
        inp = self._t(_Y_4x8)
        SwiGLUFunction.forward(ctx, inp, False, False)
        self.assertNotEqual(getattr(inp, "activation_offloading", False), True)


if __name__ == "__main__":
    unittest.main()
