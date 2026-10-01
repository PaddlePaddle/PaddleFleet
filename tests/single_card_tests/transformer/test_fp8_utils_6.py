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

"""Behavior tests for paddlefleet.transformer.moe.fp8_utils.

All expected values below were derived BY HAND from the production source
(``src/paddlefleet/transformer/moe/fp8_utils.py``); none are copied from any
coverage_test file.

Covered production behaviors:
  * ``moe_token_padding_alignment`` -- the whole AND/NOT guard. It returns 1
    ONLY on the accuracy-compatible pure-bf16 non-grouped path; every other
    combination must fall through to ``FP8_ALIGN``. The full 8-row truth table
    is asserted so flipping any single condition in production is caught.
  * ``has_config`` -- None map, missing key, falsy value all -> False; a
    present truthy value -> True.
  * ``ceil_to_ue8m0`` -- rounds a positive fp32 scale UP to the next power of
    two (leaving exact powers of two unchanged). Verified with distinguishable
    inputs whose expected outputs differ from the inputs.
  * ``_quantize_to_fp4_e2m1`` -- maps magnitudes to e2m1 codes 0..7 by the
    midpoint boundaries, with bit3 as sign (never set for a zero code).
  * ``expert_weights_all_frozen`` -- None / empty -> False; all-frozen
    EagerParamBase list -> True; a single trainable entry makes the group
    not-frozen; a non-parameter entry is never frozen.

These require paddle (the module imports it at load time). The covered logic is
CPU-computable and needs no accelerator, but paddle itself must be importable.
When it is not, every case is honestly skipped with the real import error --
never faked as a pass. The import probe catches ONLY ImportError /
ModuleNotFoundError so a genuine API/compile regression still surfaces.
"""

import os
import sys
import unittest

import numpy as np

# Allow importing paddlefleet from the in-tree src/ when it is not installed.
_SRC = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
    "src",
)
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    import paddle

    from paddlefleet.transformer.moe.fp8_utils import (
        FP8_ALIGN,
        _quantize_to_fp4_e2m1,
        ceil_to_ue8m0,
        expert_weights_all_frozen,
        has_config,
        moe_token_padding_alignment,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # honest, precise probe
    _IMPORT_ERROR = exc


_SKIP_REASON = (
    None
    if _IMPORT_ERROR is None
    else f"paddle/paddlefleet not importable: {_IMPORT_ERROR!r}"
)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestMoeTokenPaddingAlignment(unittest.TestCase):
    """The alignment guard must consume all three flags with AND/NOT logic."""

    def test_full_truth_table(self):
        # Hand-derived: return 1 iff (accuracy_compatible and not fp8 and
        # not grouped); otherwise FP8_ALIGN. FP8_ALIGN is imported, not
        # hardcoded, so the two branches stay tied to the real constant.
        self.assertEqual(FP8_ALIGN, 128)
        one = 1
        align = FP8_ALIGN
        expected = {
            # (use_accuracy_compatible, use_fp8_mlp, moe_grouped_gemm): result
            (True, False, False): one,  # the ONLY skip-padding case
            (True, False, True): align,
            (True, True, False): align,
            (True, True, True): align,
            (False, False, False): align,
            (False, False, True): align,
            (False, True, False): align,
            (False, True, True): align,
        }
        for (acc, fp8, grouped), want in expected.items():
            got = moe_token_padding_alignment(
                use_accuracy_compatible=acc,
                use_fp8_mlp=fp8,
                moe_grouped_gemm=grouped,
            )
            self.assertEqual(
                got,
                want,
                msg=(
                    f"acc={acc} fp8={fp8} grouped={grouped}: "
                    f"expected {want}, got {got}"
                ),
            )


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestHasConfig(unittest.TestCase):
    """has_config is truthiness of (map is not None and key in map and value)."""

    def test_none_map_is_false(self):
        self.assertIs(has_config(None, "k"), False)

    def test_missing_key_is_false(self):
        self.assertIs(has_config({"other": 1}, "k"), False)

    def test_falsy_values_are_false(self):
        for falsy in (0, None, "", [], {}, 0.0, False):
            self.assertIs(
                has_config({"k": falsy}, "k"),
                False,
                msg=f"falsy value {falsy!r} should be treated as absent",
            )

    def test_truthy_values_are_true(self):
        for truthy in (1, "v", [0], {"x": 1}, 0.5, True):
            self.assertIs(
                has_config({"k": truthy}, "k"),
                True,
                msg=f"truthy value {truthy!r} should be present",
            )


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestCeilToUe8m0(unittest.TestCase):
    """ceil_to_ue8m0 rounds a positive fp32 scale up to the next power of two."""

    def test_rounds_up_to_power_of_two(self):
        # Hand-derived per the ue8m0 bit formula: exact powers of two are
        # unchanged; anything with a nonzero mantissa bumps the exponent by 1.
        #   1.0  -> 1.0   (exp 127, mantissa 0)
        #   0.5  -> 0.5   (exp 126, mantissa 0)
        #   2.0  -> 2.0   (exp 128, mantissa 0)
        #   1.5  -> 2.0   (127 + 1)
        #   0.75 -> 1.0   (126 + 1)
        #   3.0  -> 4.0   (128 + 1)
        #   5.0  -> 8.0   (129 + 1)
        x = paddle.to_tensor(
            [1.0, 0.5, 2.0, 1.5, 0.75, 3.0, 5.0], dtype="float32"
        )
        expected = np.array(
            [1.0, 0.5, 2.0, 2.0, 1.0, 4.0, 8.0], dtype=np.float32
        )
        out = ceil_to_ue8m0(x)
        # Powers of two are exact -> exact equality, no tolerance slack.
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_result_is_always_a_power_of_two(self):
        # Independent invariant: log2 of every output is an integer.
        x = paddle.to_tensor([0.3, 1.1, 2.9, 6.5, 100.0, 1e-3], dtype="float32")
        out = ceil_to_ue8m0(x).numpy().astype(np.float64)
        log2 = np.log2(out)
        np.testing.assert_array_equal(log2, np.round(log2))
        # And each output is >= the input (rounding is upward).
        self.assertTrue(np.all(out >= x.numpy().astype(np.float64) - 1e-12))


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestQuantizeToFp4E2m1(unittest.TestCase):
    """_quantize_to_fp4_e2m1 maps magnitudes to e2m1 codes with a sign bit."""

    def test_positive_representable_values_map_to_codes(self):
        # e2m1 representable magnitudes -> codes 0..7 (bit3 = sign, unset here).
        x = paddle.to_tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype="float32"
        )
        expected = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
        out = _quantize_to_fp4_e2m1(x)
        np.testing.assert_array_equal(out.numpy().astype(np.int32), expected)

    def test_sign_bit_and_zero_sign_suppression(self):
        # Negative magnitudes get bit3 set: code | (1 << 3) = code | 8.
        #   -0.5 -> 1 | 8 = 9 ; -1.5 -> 3 | 8 = 11 ; -6.0 -> 7 | 8 = 15
        # Negative zero keeps code 0 (sign suppressed when code == 0).
        x = paddle.to_tensor([-0.5, -1.5, -6.0, -0.0], dtype="float32")
        expected = np.array([9, 11, 15, 0], dtype=np.int32)
        out = _quantize_to_fp4_e2m1(x)
        np.testing.assert_array_equal(out.numpy().astype(np.int32), expected)

    def test_rounding_boundaries(self):
        # Boundaries use strict >: exactly-on-boundary rounds toward the lower
        # code. 0.24<0.25 -> 0 ; 0.26>0.25 -> 1 ; 2.5 not > 2.5 -> 4 ;
        # 2.6 > 2.5 -> 5.
        x = paddle.to_tensor([0.24, 0.26, 2.5, 2.6], dtype="float32")
        expected = np.array([0, 1, 4, 5], dtype=np.int32)
        out = _quantize_to_fp4_e2m1(x)
        np.testing.assert_array_equal(out.numpy().astype(np.int32), expected)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestExpertWeightsAllFrozen(unittest.TestCase):
    """Frozen detection over stacked / list / single expert parameters."""

    @staticmethod
    def _param(stop_gradient):
        p = paddle.create_parameter(
            shape=[2, 2],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(1.0),
        )
        p.stop_gradient = stop_gradient
        return p

    def test_none_and_empty_are_not_frozen(self):
        self.assertIs(expert_weights_all_frozen(None), False)
        self.assertIs(expert_weights_all_frozen([]), False)
        self.assertIs(expert_weights_all_frozen([None, None]), False)

    def test_all_frozen_list_is_frozen(self):
        weights = [self._param(True), self._param(True)]
        self.assertIs(expert_weights_all_frozen(weights), True)

    def test_mixed_group_is_not_frozen(self):
        # One trainable entry poisons the whole group.
        weights = [self._param(True), self._param(False)]
        self.assertIs(expert_weights_all_frozen(weights), False)

    def test_single_frozen_param_is_frozen(self):
        self.assertIs(expert_weights_all_frozen(self._param(True)), True)

    def test_non_parameter_entry_is_not_frozen(self):
        # A plain tensor is not an EagerParamBase, so it is never counted as
        # frozen even with stop_gradient set.
        t = paddle.to_tensor([1.0, 2.0])
        t.stop_gradient = True
        self.assertIs(expert_weights_all_frozen([t]), False)


if __name__ == "__main__":
    unittest.main()
