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

"""Behavior unit tests for paddlefleet.fusions.fused_bias_dropout (part _2).

Scope of this file (the sibling test_fused_bias_dropout.py concurrently covers
branch/dispatch selection). Here we pin, on CPU:

* the prob=0 fast path: dropout becomes identity, so the observable result is
  pure elementwise arithmetic ``x + bias + residual`` (or ``x + residual`` when
  bias is None), hand-derived below;
* the residual-add / bias-broadcast arithmetic on that fast path;
* get_bias_dropout_add as a factory: each call yields a fresh callable, the
  captured ``training`` flag genuinely drives the in-place eval branch, and the
  ``fused`` argument is (in the current implementation) not consulted.

Random dropout (prob>0) is GPU/RNG dependent and is intentionally NOT asserted
here; that is out of this file's scope and cannot be reproduced deterministically
as pure CPU arithmetic without pinning RNG.
"""

import os
import sys
import unittest

import numpy as np

# Reach src/ so that ``paddlefleet`` is importable when running the file directly.
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
        ),
        "src",
    ),
)

try:
    import paddle

    from paddlefleet.fusions.fused_bias_dropout import (
        _bias_dropout_add_func,
        bias_dropout_add_unfused,
        get_bias_dropout_add,
    )

    HAS_PADDLE = True
    _IMPORT_ERROR = None
except ImportError as exc:  # honest: only a genuinely missing dep skips.
    HAS_PADDLE = False
    _IMPORT_ERROR = repr(exc)

_SKIP_REASON = f"paddle / paddlefleet not importable: {_IMPORT_ERROR}"


# --- Fixed, distinguishable fixtures (no degenerate/all-equal inputs). ---
# Rows differ, columns differ, and bias differs per column so a broadcast bug,
# a dropped residual, or a row/column transposition all change the result.
_X = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
_BIAS = [10.0, 20.0, 30.0]
_RESIDUAL = [[100.0, 200.0, 300.0], [400.0, 500.0, 600.0]]
# Hand-derived x + broadcast(bias) + residual:
#   [1+10+100, 2+20+200, 3+30+300] = [111, 222, 333]
#   [4+10+400, 5+20+500, 6+30+600] = [414, 525, 636]
_EXPECTED_WITH_BIAS = [[111.0, 222.0, 333.0], [414.0, 525.0, 636.0]]
# Hand-derived x + residual (bias is None):
#   [1+100, 2+200, 3+300] = [101, 202, 303]
#   [4+400, 5+500, 6+600] = [404, 505, 606]
_EXPECTED_NO_BIAS = [[101.0, 202.0, 303.0], [404.0, 505.0, 606.0]]


def _t(values, stop_gradient=True):
    """Fresh float32 CPU tensor with an explicit stop_gradient flag."""
    out = paddle.to_tensor(values, dtype="float32")
    out.stop_gradient = stop_gradient
    return out


def _inplace_eligible_inputs():
    """Inputs that make ``_bias_dropout_add_func`` take its in-place branch.

    The production predicate is:
        not training and x.stop_gradient and not residual.stop_gradient
        and (bias is None or bias.stop_gradient)
    """
    x = _t(_X, stop_gradient=True)
    bias = _t(_BIAS, stop_gradient=True)
    residual = _t(_RESIDUAL, stop_gradient=False)
    return x, bias, residual


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestProbZeroArithmetic(unittest.TestCase):
    """prob=0 makes dropout an identity; result is exact CPU arithmetic."""

    def test_with_bias_equals_x_plus_bias_plus_residual(self):
        x, bias, residual = _t(_X), _t(_BIAS), _t(_RESIDUAL)
        # training=True forces the non-in-place branch, so inputs are untouched.
        out = _bias_dropout_add_func((x, bias), residual, 0.0, True)
        np.testing.assert_array_equal(
            out.numpy(), np.array(_EXPECTED_WITH_BIAS)
        )
        # Non-in-place path must not mutate the caller's x tensor.
        np.testing.assert_array_equal(x.numpy(), np.array(_X))

    def test_without_bias_equals_x_plus_residual(self):
        x, residual = _t(_X), _t(_RESIDUAL)
        out = _bias_dropout_add_func((x, None), residual, 0.0, True)
        np.testing.assert_array_equal(out.numpy(), np.array(_EXPECTED_NO_BIAS))
        np.testing.assert_array_equal(x.numpy(), np.array(_X))

    def test_bias_contribution_is_broadcast_bias(self):
        # Isolate bias: (with bias) - (without bias) must equal broadcast(bias),
        # which also proves bias is broadcast across rows rather than dropped.
        with_bias = _bias_dropout_add_func(
            (_t(_X), _t(_BIAS)), _t(_RESIDUAL), 0.0, True
        )
        without_bias = _bias_dropout_add_func(
            (_t(_X), None), _t(_RESIDUAL), 0.0, True
        )
        diff = with_bias.numpy() - without_bias.numpy()
        np.testing.assert_array_equal(diff, np.array([_BIAS, _BIAS]))

    def test_residual_is_added_not_ignored(self):
        # Two residuals differing by a known delta must shift the output by the
        # same delta, proving residual is genuinely summed in.
        res_a = _t(_RESIDUAL)
        res_b = _t(
            [[100.0, 200.0, 300.0], [400.0, 500.0, 601.0]]
        )  # +1 at [1,2]
        out_a = _bias_dropout_add_func((_t(_X), _t(_BIAS)), res_a, 0.0, True)
        out_b = _bias_dropout_add_func((_t(_X), _t(_BIAS)), res_b, 0.0, True)
        delta = out_b.numpy() - out_a.numpy()
        np.testing.assert_array_equal(
            delta, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        )


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestEvalInplaceBranch(unittest.TestCase):
    """Eval mode with the right stop_gradient config takes the in-place branch."""

    def test_eval_inplace_returns_sum_and_mutates_x(self):
        x, bias, residual = _inplace_eligible_inputs()
        x_before = x.numpy().copy()
        out = _bias_dropout_add_func((x, bias), residual, 0.0, False)
        # The returned value is the full sum regardless of aliasing internals.
        np.testing.assert_array_equal(
            out.numpy(), np.array(_EXPECTED_WITH_BIAS)
        )
        # Defining behavior of the in-place branch: the caller's x is mutated
        # (at minimum x += bias runs). We assert mutation occurred rather than a
        # specific intermediate, because whether dropout(p=0, eval) aliases x is
        # a paddle internal we do not verify offline.
        self.assertFalse(
            np.array_equal(x.numpy(), x_before),
            "eval in-place branch must mutate the input x tensor",
        )

    def test_training_true_does_not_mutate_x(self):
        # Same stop_gradient config, but training=True disables in-place, so the
        # caller's x tensor must be preserved.
        x, bias, residual = _inplace_eligible_inputs()
        out = _bias_dropout_add_func((x, bias), residual, 0.0, True)
        np.testing.assert_array_equal(
            out.numpy(), np.array(_EXPECTED_WITH_BIAS)
        )
        np.testing.assert_array_equal(x.numpy(), np.array(_X))


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestBiasDropoutAddUnfused(unittest.TestCase):
    """The closure returned by bias_dropout_add_unfused binds `training`."""

    def test_closure_applies_prob_zero_arithmetic(self):
        fn = bias_dropout_add_unfused(True)
        out = fn((_t(_X), _t(_BIAS)), _t(_RESIDUAL), 0.0)
        np.testing.assert_array_equal(
            out.numpy(), np.array(_EXPECTED_WITH_BIAS)
        )

    def test_closure_signature_is_three_positional_args(self):
        # The unfused wrapper hides `training`; callers pass (x_with_bias,
        # residual, prob) only.
        fn = bias_dropout_add_unfused(False)
        out = fn((_t(_X), None), _t(_RESIDUAL), 0.0)
        np.testing.assert_array_equal(out.numpy(), np.array(_EXPECTED_NO_BIAS))


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestGetBiasDropoutAddFactory(unittest.TestCase):
    """get_bias_dropout_add(training, fused) as a callable factory."""

    def test_each_call_returns_a_fresh_callable(self):
        f1 = get_bias_dropout_add(True, fused=True)
        f2 = get_bias_dropout_add(True, fused=True)
        self.assertTrue(callable(f1) and callable(f2))
        self.assertIsNot(f1, f2)  # a new closure object per call

    def test_all_training_fused_combos_are_distinct_objects(self):
        combos = [
            get_bias_dropout_add(True, fused=True),
            get_bias_dropout_add(True, fused=False),
            get_bias_dropout_add(False, fused=True),
            get_bias_dropout_add(False, fused=False),
        ]
        self.assertEqual(len({id(fn) for fn in combos}), 4)

    def test_captured_training_flag_drives_inplace_branch(self):
        # This is what makes the callables genuinely (not just nominally)
        # different: the eval callable mutates its x in place; the training
        # callable does not. Both still return the same prob=0 sum.
        fn_eval = get_bias_dropout_add(False, fused=False)
        fn_train = get_bias_dropout_add(True, fused=False)

        xe, bias_e, res_e = _inplace_eligible_inputs()
        xe_before = xe.numpy().copy()
        out_eval = fn_eval((xe, bias_e), res_e, 0.0)
        np.testing.assert_array_equal(
            out_eval.numpy(), np.array(_EXPECTED_WITH_BIAS)
        )
        self.assertFalse(
            np.array_equal(xe.numpy(), xe_before),
            "eval callable must take the in-place branch and mutate x",
        )

        xt, bias_t, res_t = _inplace_eligible_inputs()
        out_train = fn_train((xt, bias_t), res_t, 0.0)
        np.testing.assert_array_equal(
            out_train.numpy(), np.array(_EXPECTED_WITH_BIAS)
        )
        np.testing.assert_array_equal(
            xt.numpy(), np.array(_X)
        )  # training callable leaves x untouched

    def test_fused_flag_is_currently_not_consulted(self):
        # Observed contract of the current implementation: fused=True and
        # fused=False produce behaviorally identical callables (both route to the
        # unfused path). Documented, not asserted as a defect.
        fn_fused = get_bias_dropout_add(True, fused=True)
        fn_unfused = get_bias_dropout_add(True, fused=False)
        out_fused = fn_fused((_t(_X), _t(_BIAS)), _t(_RESIDUAL), 0.0)
        out_unfused = fn_unfused((_t(_X), _t(_BIAS)), _t(_RESIDUAL), 0.0)
        np.testing.assert_array_equal(out_fused.numpy(), out_unfused.numpy())
        np.testing.assert_array_equal(
            out_fused.numpy(), np.array(_EXPECTED_WITH_BIAS)
        )


if __name__ == "__main__":
    unittest.main()
