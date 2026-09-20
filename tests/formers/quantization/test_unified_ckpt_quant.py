# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

"""Behavior tests for the unified-checkpoint optimizer quant orchestration.

Target production surface:
    paddlefleet.quantization.unified_checkpoint_quantization
        - quant_unified_optimizer
        - dequant_unified_optimizer

These are the *orchestration* functions: they decide which state-dict keys are
quantized, which scale keys are produced, which key is replaced by an int8/uint8
payload and which pass through untouched, and they wire the moment1 / moment2
routing together. The low-level numeric helpers (qdq_weight, group_wise_...,
merge/split int4, cal_*) are behavior-tested independently in
``test_ckpt_quant_utils.py``; this file deliberately does NOT re-test that math
and instead pins the key/index routing and the end-to-end reconstruction.

Every expected value is hand-derived from small, fully-known numpy tensors. The
inputs are chosen so the symmetric int8 scale, the asymmetric uint8 (min, max)
scales and the round(x/scale*bnt) mapping all land on exact integers, and so the
1/(sqrt(m2)+eps) ratio transform round-trips exactly. The functions under test
are never used to build their own reference.

Environment: the O0 / O1 paths are numpy-based; ``dequant_unified_optimizer``
only calls ``paddle.distributed.get_world_size()`` (returns 1 without an
initialized process group), so these cases run 无卡 (CPU) and need no GPU. The
O2 path (bw-int4, group-wise, ``split_int8`` building paddle tensors) is not
exercised here; a real orchestration bug on that path is documented below.
"""

import unittest

import numpy as np

from paddlefleet.quantization.unified_checkpoint_quantization import (
    dequant_unified_optimizer,
    quant_unified_optimizer,
)
from paddlefleet.utils.env import (
    ASYMMETRY_QUANT_SCALE_MAX,
    ASYMMETRY_QUANT_SCALE_MIN,
    MOMENT1_KEYNAME,
    MOMENT2_KEYNAME,
    SYMMETRY_QUANT_SCALE,
)

M1_KEY = "p/" + MOMENT1_KEYNAME
M2_KEY = "p/" + MOMENT2_KEYNAME

# moment1: per-column abs-max is 127 for both columns, so the symmetric int8
# scale is [127, 127] and round(x/127*127) == x exactly (all entries integral).
M1 = np.array([[127.0, 100.0], [-64.0, -127.0]], dtype=np.float32)
# moment2 (variance, > 0): sqrt = [[1, 2], [4, 10]] so the adam ratio
# 1/(sqrt(m2)+eps) is approx [[1.0, 0.5], [0.25, 0.1]]. With exactly two rows,
# every column value is either the column min or max, so the uint8 asymmetric
# quant maps them to {0, 255} and dequant reconstructs them exactly.
M2 = np.array([[1.0, 4.0], [16.0, 100.0]], dtype=np.float32)


class TestQuantUnifiedOptimizerGating(unittest.TestCase):
    """O0 and non-optimizer state dicts must NOT be quantized at all."""

    def test_o0_returns_state_dict_untouched(self):
        state_dict = {M1_KEY: M1.copy(), M2_KEY: M2.copy()}
        result = quant_unified_optimizer(state_dict, "optimizer_weight", "O0")
        # Same object, same keys, no scale keys, values still float32.
        self.assertIs(result, state_dict)
        self.assertEqual(set(result), {M1_KEY, M2_KEY})
        self.assertEqual(result[M1_KEY].dtype, np.float32)
        np.testing.assert_array_equal(result[M1_KEY], M1)
        np.testing.assert_array_equal(result[M2_KEY], M2)

    def test_model_weight_is_not_quantized_even_in_o1(self):
        # quant is enabled (O1) but the gate requires "optimizer_weight";
        # model_weight must fall through with no int8 payload / scale keys.
        state_dict = {M1_KEY: M1.copy(), M2_KEY: M2.copy()}
        result = quant_unified_optimizer(state_dict, "model_weight", "O1")
        self.assertEqual(set(result), {M1_KEY, M2_KEY})
        self.assertEqual(result[M1_KEY].dtype, np.float32)
        np.testing.assert_array_equal(result[M1_KEY], M1)
        self.assertNotIn(M1_KEY + SYMMETRY_QUANT_SCALE, result)


class TestQuantUnifiedOptimizerO1(unittest.TestCase):
    """O1 optimizer_weight: moment1 -> symmetric int8, moment2 -> asymmetric
    uint8 of the adam ratio, plus three hand-derived scale keys."""

    def test_moment1_replaced_by_symmetric_int8(self):
        state_dict = {M1_KEY: M1.copy(), M2_KEY: M2.copy()}
        result = quant_unified_optimizer(state_dict, "optimizer_weight", "O1")
        q1 = result[M1_KEY]
        self.assertEqual(q1.dtype, np.int8)
        # scale = [127, 127], round(M1 / scale * 127) == M1 (already integral).
        np.testing.assert_array_equal(q1, [[127, 100], [-64, -127]])

    def test_moment2_replaced_by_asymmetric_uint8_of_ratio(self):
        state_dict = {M1_KEY: M1.copy(), M2_KEY: M2.copy()}
        result = quant_unified_optimizer(state_dict, "optimizer_weight", "O1")
        q2 = result[M2_KEY]
        self.assertEqual(q2.dtype, np.uint8)
        # Per column the ratio's max maps to 255 and its min maps to 0.
        # ratio col0 = [1.0, 0.25] -> [255, 0]; col1 = [0.5, 0.1] -> [255, 0].
        np.testing.assert_array_equal(q2, [[255, 255], [0, 0]])

    def test_scale_keys_added_with_hand_derived_values(self):
        state_dict = {M1_KEY: M1.copy(), M2_KEY: M2.copy()}
        result = quant_unified_optimizer(state_dict, "optimizer_weight", "O1")
        sym_key = M1_KEY + SYMMETRY_QUANT_SCALE
        min_key = M2_KEY + ASYMMETRY_QUANT_SCALE_MIN
        max_key = M2_KEY + ASYMMETRY_QUANT_SCALE_MAX
        for key in (sym_key, min_key, max_key):
            self.assertIn(key, result)
        # symmetric scale = per-column abs-max of moment1.
        np.testing.assert_allclose(
            result[sym_key], [127.0, 127.0], rtol=0, atol=1e-5
        )
        # asymmetric (min, max) = per-column (min, max) of the adam ratio.
        np.testing.assert_allclose(
            result[min_key], [0.25, 0.1], rtol=0, atol=1e-5
        )
        np.testing.assert_allclose(
            result[max_key], [1.0, 0.5], rtol=0, atol=1e-5
        )

    def test_non_moment_key_passes_through_unchanged(self):
        # beta accumulators end with neither moment1_0 nor moment2_0: they are
        # copied through as-is and gain no scale key.
        beta_key = "p/beta1_pow_acc_0"
        state_dict = {
            M1_KEY: M1.copy(),
            M2_KEY: M2.copy(),
            beta_key: np.array([0.9], dtype=np.float32),
        }
        result = quant_unified_optimizer(state_dict, "optimizer_weight", "O1")
        self.assertEqual(result[beta_key].dtype, np.float32)
        np.testing.assert_array_equal(result[beta_key], [0.9])
        self.assertNotIn(beta_key + SYMMETRY_QUANT_SCALE, result)


class TestDequantUnifiedOptimizerGating(unittest.TestCase):
    """O0 dequant is a no-op regardless of scales."""

    def test_o0_returns_state_dict_untouched(self):
        state_dict = {M1_KEY: np.array([[1.0, 2.0]], dtype=np.float32)}
        result = dequant_unified_optimizer(state_dict, "O0", {})
        self.assertIs(result, state_dict)
        np.testing.assert_array_equal(result[M1_KEY], [[1.0, 2.0]])


class TestQuantDequantRoundTripO1(unittest.TestCase):
    """quant(O1) then dequant(O1) reconstructs the original moments.

    The reconstruction target is the original hand-known M1/M2, not a value
    recomputed from the functions under test.
    """

    def test_o1_roundtrip_reconstructs_moments(self):
        state_dict = {M1_KEY: M1.copy(), M2_KEY: M2.copy()}
        quantized = quant_unified_optimizer(
            state_dict, "optimizer_weight", "O1"
        )
        # Split scales out into scale_dict, exactly as the real load path does;
        # dequant must only iterate the moment payload keys.
        scale_dict = {}
        for key in list(quantized.keys()):
            if key.endswith(
                (
                    SYMMETRY_QUANT_SCALE,
                    ASYMMETRY_QUANT_SCALE_MIN,
                    ASYMMETRY_QUANT_SCALE_MAX,
                )
            ):
                scale_dict[key] = quantized.pop(key)
        self.assertEqual(set(quantized), {M1_KEY, M2_KEY})

        restored = dequant_unified_optimizer(quantized, "O1", scale_dict)
        # moment1: exact symmetric reconstruction.
        np.testing.assert_allclose(restored[M1_KEY], M1, rtol=1e-4, atol=1e-4)
        # moment2: ratio dequant then square(1/ratio - eps) recovers variance.
        np.testing.assert_allclose(restored[M2_KEY], M2, rtol=1e-4, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
