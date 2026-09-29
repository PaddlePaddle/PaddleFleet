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

"""CPU behavior tests for the *default* (unclamped, non-accuracy-compatible)
forward dispatch of the three ``paddle.autograd.PyLayer`` wrappers in
``paddlefleet.fusions.fused_bias_swiglu``:

* ``SwiGLUFunction``        -> ``swiglu(input)``
* ``BiasSwiGLUFunction``    -> ``bias_swiglu(input, bias)`` (bias added first)
* ``WeightedSwiGLUFunction``-> ``weighted_swiglu(input, weights)``

Slice boundary: this file only exercises the plain fused SwiGLU forward branch
(``clamp_value=None``, ``use_accuracy_compatible=False``).  The clamped branch,
the backward pass, the accuracy-compatible/eager branch, the ``*_impl`` shape
wrappers and their validation guards are covered by sibling files and are NOT
re-tested here.

The production ``jit_fuser`` decorator is an identity no-op and ``F.swiglu`` has
a CPU kernel, so the default forward is plain CPU tensor math.  Every expected
value is hand-derived with an independent NumPy implementation of
``SiLU(gate) * value``; the production module is never used to build an
expectation.  The Triton/GPU kernel numerics that ``F.swiglu`` may dispatch to
on device are not asserted here -- only the CPU-observable forward contract is.
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

    from paddlefleet.fusions.fused_bias_swiglu import (
        BiasSwiGLUFunction,
        SwiGLUFunction,
        WeightedSwiGLUFunction,
    )

    HAS_PADDLE = True
    _IMPORT_ERROR = ""
except ImportError as exc:  # honest: dependency missing, not a swallowed bug
    HAS_PADDLE = False
    _IMPORT_ERROR = repr(exc)


# --- Independent NumPy references (never call production code) --------------
# Work in float64 from the mathematical definition so a same-way bug in the
# production float32 path cannot hide behind a shared implementation.


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def _silu(x):
    x = np.asarray(x, dtype=np.float64)
    return x * _sigmoid(x)


def _split_last(arr):
    """Split the last axis into two equal halves (matches paddle.chunk)."""
    half = arr.shape[-1] // 2
    return arr[..., :half], arr[..., half:]


def ref_swiglu_fwd(y):
    """Plain SwiGLU: SiLU(gate) * value, gate/value being the last-axis halves."""
    g, v = _split_last(np.asarray(y, dtype=np.float64))
    return _silu(g) * v


def ref_clamped_swiglu_fwd(y, cv):
    """Clamped SwiGLU used ONLY as a negative control to prove the default
    branch does not clamp: gate clipped to (-inf, cv], value to [-cv, cv]."""
    g, v = _split_last(np.asarray(y, dtype=np.float64))
    return _silu(np.minimum(g, cv)) * np.clip(v, -cv, cv)


# Saturating fixture: gate=[2,-3,5,0.5], value=[1.5,-2,0.25,-0.5].  Chosen so a
# clamp_value=1.0 would visibly change several elements, letting the tests prove
# the default branch leaves the input unclamped.
_Y_SAT_ROW = [2.0, -3.0, 5.0, 0.5, 1.5, -2.0, 0.25, -0.5]
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
class TestSwiGLUFunctionForward(_CPUFixture):
    def test_forward_matches_hand_derived_literals(self):
        # gate=[1.0,-1.0], value=[2.0,0.5]; pen-and-paper:
        #   silu(1.0)  = 1.0 * sigmoid(1.0)  =  0.7310585786
        #   silu(-1.0) = -1.0 * sigmoid(-1.0)= -0.2689414214
        #   out = [silu(1.0)*2.0, silu(-1.0)*0.5]
        #       = [1.4621171572, -0.1344707107]
        x = paddle.to_tensor([[1.0, -1.0, 2.0, 0.5]], dtype="float32")
        out = SwiGLUFunction.apply(x, False, False)
        self.assertEqual(out.shape, [1, 2])
        expected = np.array([[1.4621171572, -0.1344707107]], dtype=np.float64)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-6)

    def test_forward_matches_independent_reference_multirow(self):
        paddle.seed(20250517)
        x = paddle.randn([6, 12], dtype="float32") * 2.0
        x_np = x.numpy()
        out = SwiGLUFunction.apply(x, False, False)
        self.assertEqual(out.shape, [6, 6])
        np.testing.assert_allclose(
            out.numpy(), ref_swiglu_fwd(x_np), rtol=1e-5, atol=1e-6
        )
        # Negative control: a wrong-sign output would be rejected.
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(
                out.numpy(), -ref_swiglu_fwd(x_np), rtol=1e-5, atol=1e-6
            )

    def test_default_branch_is_unclamped(self):
        # clamp_value defaults to None, so a saturating input must produce the
        # plain (unclamped) SwiGLU and differ from the clamped reference.
        x = paddle.to_tensor([_Y_SAT_ROW], dtype="float32")
        out = SwiGLUFunction.apply(x, False, False).numpy()
        np.testing.assert_allclose(
            out, ref_swiglu_fwd([_Y_SAT_ROW]), rtol=1e-5, atol=1e-6
        )
        clamped = ref_clamped_swiglu_fwd([_Y_SAT_ROW], _CV)
        self.assertGreater(np.abs(out - clamped).max(), 0.1)


@unittest.skipUnless(
    HAS_PADDLE, "paddle not importable in this environment: " + _IMPORT_ERROR
)
class TestBiasSwiGLUFunctionForward(_CPUFixture):
    def test_bias_is_added_before_swiglu(self):
        # Non-zero bias so the add is observable; expected = swiglu(x + bias).
        x = np.array([[1.0, -1.0, 2.0, 0.5]], dtype="float32")
        bias = np.array([0.5, -0.5, 1.0, -1.0], dtype="float32")
        out = BiasSwiGLUFunction.apply(
            paddle.to_tensor(x), paddle.to_tensor(bias), False, False
        )
        self.assertEqual(out.shape, [1, 2])
        expected = ref_swiglu_fwd(x + bias)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-6)
        # Bias must be consumed: result differs from the no-bias forward.
        self.assertFalse(
            np.allclose(expected, ref_swiglu_fwd(x), rtol=1e-5, atol=1e-6)
        )

    def test_forward_matches_independent_reference_multirow(self):
        paddle.seed(4242)
        x = paddle.randn([5, 16], dtype="float32") * 2.0
        bias = paddle.randn([16], dtype="float32")
        x_np, bias_np = x.numpy(), bias.numpy()
        out = BiasSwiGLUFunction.apply(x, bias, False, False)
        self.assertEqual(out.shape, [5, 8])
        # Bias broadcasts over the leading (token) dimension.
        np.testing.assert_allclose(
            out.numpy(), ref_swiglu_fwd(x_np + bias_np), rtol=1e-5, atol=1e-6
        )


@unittest.skipUnless(
    HAS_PADDLE, "paddle not importable in this environment: " + _IMPORT_ERROR
)
class TestWeightedSwiGLUFunctionForward(_CPUFixture):
    def test_weight_applied_per_row(self):
        # Distinct per-row weights must scale only the matching row.
        x = paddle.to_tensor(
            [[1.0, -1.0, 2.0, 0.5], [1.0, -1.0, 2.0, 0.5]], dtype="float32"
        )
        weights = paddle.to_tensor([[2.0], [5.0]], dtype="float32")
        out = WeightedSwiGLUFunction.apply(x, weights, False)
        self.assertEqual(out.shape, [2, 2])
        base = ref_swiglu_fwd([[1.0, -1.0, 2.0, 0.5]])  # shape [1, 2]
        np.testing.assert_allclose(
            out.numpy()[0], base[0] * 2.0, rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            out.numpy()[1], base[0] * 5.0, rtol=1e-5, atol=1e-6
        )

    def test_forward_matches_independent_reference_multirow(self):
        paddle.seed(909)
        x = paddle.randn([4, 12], dtype="float32") * 2.0
        weights = paddle.randn([4, 1], dtype="float32")
        x_np, w_np = x.numpy(), weights.numpy()
        out = WeightedSwiGLUFunction.apply(x, weights, False)
        self.assertEqual(out.shape, [4, 6])
        # weighted_swiglu = swiglu(x) * weights, weights broadcasting over hidden.
        np.testing.assert_allclose(
            out.numpy(),
            ref_swiglu_fwd(x_np) * w_np,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_weight_is_consumed(self):
        # A non-unit weight must change the output relative to weight == 1.
        paddle.seed(17)
        x = paddle.randn([3, 8], dtype="float32") * 2.0
        ones = paddle.ones([3, 1], dtype="float32")
        w = paddle.to_tensor([[2.0], [0.5], [-1.0]], dtype="float32")
        out_ones = WeightedSwiGLUFunction.apply(x, ones, False).numpy()
        out_w = WeightedSwiGLUFunction.apply(x, w, False).numpy()
        x_np = x.numpy()
        np.testing.assert_allclose(
            out_ones, ref_swiglu_fwd(x_np), rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            out_w,
            ref_swiglu_fwd(x_np) * w.numpy(),
            rtol=1e-5,
            atol=1e-6,
        )
        self.assertGreater(np.abs(out_ones - out_w).max(), 1e-3)


if __name__ == "__main__":
    unittest.main()
