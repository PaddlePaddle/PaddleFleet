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

"""Behavior tests for paddlefleet.quantization.checkpoint_quantization_utils.

Every expected value below is hand-derived from a small, fully-known tensor:
the quantization scale, block layout, int4 bit-packing, rounding/saturation
and the dequant reconstruction are all computed by hand and asserted exactly.
The function under test is never used to build its own reference.

Environment: all functions are numpy-based (split_int8 additionally builds a
paddle.Tensor on the default CPU device), so these cases run 无卡 (CPU) and do
not require a GPU.
"""

import unittest

import numpy as np

from paddlefleet.quantization.checkpoint_quantization_utils import (
    asymmetry_qdq_weight,
    cal_abs_max_channel,
    cal_abs_min_max_channel,
    cal_ratio,
    group_wise_quant_dequant,
    merge_int4,
    qdq_weight,
    split_int8,
)


class TestCalRatio(unittest.TestCase):
    """cal_ratio returns 1 / (sqrt(v) + eps); the `m` argument is unused."""

    def test_exact_reciprocal_of_sqrt_plus_eps(self):
        v = np.array([4.0, 9.0, 16.0], dtype=np.float64)
        eps = 1e-8
        # Hand-derived: 1/(2+eps), 1/(3+eps), 1/(4+eps)
        expected = np.array(
            [1.0 / (2.0 + eps), 1.0 / (3.0 + eps), 1.0 / (4.0 + eps)]
        )
        result = cal_ratio(np.array([1.0, 2.0, 3.0]), v, eps=eps)
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-15)

    def test_m_is_not_consumed(self):
        # The production formula ignores `m`; two different `m` with the same
        # `v` must produce identical output. This pins the current contract.
        v = np.array([1.0, 100.0], dtype=np.float64)
        r1 = cal_ratio(np.array([5.0, 5.0]), v)
        r2 = cal_ratio(np.array([-3.0, 42.0]), v)
        np.testing.assert_array_equal(r1, r2)
        np.testing.assert_allclose(
            r1, 1.0 / (np.sqrt(v) + 1e-8), rtol=0, atol=1e-15
        )


class TestGroupWiseSymmetricQuantDequant(unittest.TestCase):
    """group-wise symmetric int4 quant: per-group per-column abs-max scale,
    quant = round(x / scale * bnt) clipped to [-bnt-1, bnt], bnt = 7."""

    def _inputs(self):
        # shape [4, 2], group_size=2 -> 2 groups (rows 0-1, rows 2-3).
        # Values chosen so x/scale*7 lands on exact integers (no .5 rounding).
        # group0 col0 abs-max=7, col1 abs-max=14
        # group1 col0 abs-max=14, col1 abs-max=7
        return np.array(
            [
                [7.0, 14.0],
                [-3.0, 4.0],
                [-14.0, 7.0],
                [6.0, -5.0],
            ],
            dtype=np.float32,
        )

    def test_quant_scale_and_content(self):
        inputs = self._inputs()
        quant_tensor, scales = group_wise_quant_dequant(
            inputs, quant_bits=4, group_size=2, quant=True, symmetry=True
        )
        # scales = per-group per-column abs max, layout [num_groups, ncols]
        np.testing.assert_array_equal(scales, [[7.0, 14.0], [14.0, 7.0]])
        # bnt = 7. Hand-derived round(x/scale*7):
        # row0: 7/7*7=7,   14/14*7=7
        # row1: -3/7*7=-3, 4/14*7=2
        # row2: -14/14*7=-7, 7/7*7=7
        # row3: 6/14*7=3,  -5/7*7=-5
        expected = np.array([[7, 7], [-3, 2], [-7, 7], [3, -5]], dtype=np.int8)
        self.assertEqual(quant_tensor.dtype, np.int8)
        np.testing.assert_array_equal(quant_tensor, expected)

    def test_dequant_reconstructs_original(self):
        inputs = self._inputs()
        quant_tensor, scales = group_wise_quant_dequant(
            inputs, quant_bits=4, group_size=2, quant=True, symmetry=True
        )
        # symmetric dequant passes scales via the `mins` argument.
        dequant = group_wise_quant_dequant(
            quant_tensor,
            mins=scales,
            quant_bits=4,
            group_size=2,
            quant=False,
            symmetry=True,
        )
        # dequant = quant * scale / bnt; inputs were exactly representable,
        # so reconstruction is exact.
        np.testing.assert_allclose(
            dequant, inputs.astype(np.float32), rtol=0, atol=1e-5
        )


class TestGroupWiseAsymmetricQuantDequant(unittest.TestCase):
    """group-wise asymmetric uint4 quant: per-group per-column (min, max),
    quant = round((x-min)/(max-min) * qmax) clipped to [0, qmax], qmax = 15."""

    def _inputs(self):
        # shape [8, 2], group_size=4 -> 2 groups (rows 0-3, rows 4-7).
        # Every column has range (max-min)=15 so (x-min) maps to itself.
        return np.array(
            [
                [0.0, -15.0],
                [5.0, -10.0],
                [10.0, -5.0],
                [15.0, 0.0],
                [3.0, 10.0],
                [6.0, 13.0],
                [9.0, 16.0],
                [18.0, 25.0],
            ],
            dtype=np.float32,
        )

    def test_quant_min_max_and_content(self):
        inputs = self._inputs()
        quant_tensor, mins, maxs = group_wise_quant_dequant(
            inputs, quant_bits=4, group_size=4, quant=True, symmetry=False
        )
        np.testing.assert_array_equal(mins, [[0.0, -15.0], [3.0, 10.0]])
        np.testing.assert_array_equal(maxs, [[15.0, 0.0], [18.0, 25.0]])
        # qmax=15. Hand-derived round((x-min)/15*15) = (x-min):
        expected = np.array(
            [
                [0, 0],
                [5, 5],
                [10, 10],
                [15, 15],
                [0, 0],
                [3, 3],
                [6, 6],
                [15, 15],
            ],
            dtype=np.uint8,
        )
        self.assertEqual(quant_tensor.dtype, np.uint8)
        np.testing.assert_array_equal(quant_tensor, expected)

    def test_dequant_reconstructs_original(self):
        inputs = self._inputs()
        quant_tensor, mins, maxs = group_wise_quant_dequant(
            inputs, quant_bits=4, group_size=4, quant=True, symmetry=False
        )
        dequant = group_wise_quant_dequant(
            quant_tensor,
            mins=mins,
            maxs=maxs,
            quant_bits=4,
            group_size=4,
            quant=False,
            symmetry=False,
        )
        # dequant = quant/qmax*(max-min) + min; exact reconstruction.
        np.testing.assert_allclose(
            dequant, inputs.astype(np.float32), rtol=0, atol=1e-5
        )


class TestMergeAndSplitInt4(unittest.TestCase):
    """merge_int4 packs two signed int4 into one int8 (high nibble = x,
    low nibble = y & 0x0F); split_int8 recovers the high nibble as a signed
    int4 and the low nibble as an unsigned nibble."""

    def test_merge_bit_packing_exact(self):
        x = np.array([1, -2, 7, -8], dtype=np.int8)
        y = np.array([3, 5, 0, 6], dtype=np.int8)  # all in [0, 7]
        # Hand-derived bytes:
        #  1<<4|3   = 0x13 =  19
        # -2<<4|5   = 0xE5 = -27
        #  7<<4|0   = 0x70 = 112
        # -8<<4|6   = 0x86 = -122
        merged = merge_int4(x, y)
        self.assertEqual(merged.dtype, np.int8)
        np.testing.assert_array_equal(merged, [19, -27, 112, -122])

    def test_merge_split_roundtrip_signed_high_and_low(self):
        x = np.array([1, -2, 7, -8], dtype=np.int8)
        y = np.array([3, 5, 0, 6], dtype=np.int8)
        merged = merge_int4(x, y)
        high, low = split_int8(merged)
        # high nibble is sign-extended back to the original signed int4 x.
        np.testing.assert_array_equal(high.numpy(), x)
        # low nibble equals y for y in [0, 7].
        np.testing.assert_array_equal(low.numpy(), y)

    def test_split_low_nibble_is_unsigned(self):
        # For a negative y the low nibble is masked (y & 0x0F), not sign
        # extended: y=-1 -> 0x0F = 15. This documents the actual contract.
        x = np.array([2], dtype=np.int8)
        y = np.array([-1], dtype=np.int8)
        merged = merge_int4(x, y)  # 2<<4 | 0x0F = 0x2F = 47
        np.testing.assert_array_equal(merged, [47])
        high, low = split_int8(merged)
        np.testing.assert_array_equal(high.numpy(), [2])
        np.testing.assert_array_equal(low.numpy(), [15])


class TestCalAbsMinMaxChannel(unittest.TestCase):
    """cal_abs_min_max_channel reduces over every axis except quant_axis and
    returns plain (max, min) per channel -- despite the name it does NOT take
    absolute values -- with exact zeros replaced by 1e-8."""

    def test_axis1_plain_min_max_per_column(self):
        inputs = np.array([[1.0, -2.0, 3.0], [4.0, -5.0, -6.0]], np.float32)
        maxs, mins = cal_abs_min_max_channel(inputs, quant_axis=1)
        # per column max/min (reduce over rows); col1 max is -2 (no abs).
        np.testing.assert_array_equal(maxs, [4.0, -2.0, 3.0])
        np.testing.assert_array_equal(mins, [1.0, -5.0, -6.0])

    def test_axis0_reduces_over_columns(self):
        inputs = np.array([[1.0, -2.0, 3.0], [4.0, -5.0, -6.0]], np.float32)
        maxs, mins = cal_abs_min_max_channel(inputs, quant_axis=0)
        np.testing.assert_array_equal(maxs, [3.0, 4.0])
        np.testing.assert_array_equal(mins, [-2.0, -6.0])

    def test_zero_channel_replaced_with_eps(self):
        inputs = np.array([[0.0, 2.0], [0.0, 5.0]], np.float32)
        maxs, mins = cal_abs_min_max_channel(inputs, quant_axis=1)
        # column 0 is all zeros -> both max and min become eps=1e-8. The output
        # is float32, so compare against a float32 reference: 1e-8 rounded to
        # float32 differs from the float64 literal by ~6e-17, which an exact
        # (atol=0) comparison against a Python float would spuriously reject.
        np.testing.assert_allclose(
            maxs, np.array([1e-8, 5.0], dtype=np.float32), rtol=0, atol=0
        )
        np.testing.assert_allclose(
            mins, np.array([1e-8, 2.0], dtype=np.float32), rtol=0, atol=0
        )


class TestCalAbsMaxChannel(unittest.TestCase):
    """cal_abs_max_channel returns max(|x|) per channel with zeros -> eps."""

    def test_axis1_abs_max_per_column(self):
        inputs = np.array([[1.0, -2.0, 3.0], [-4.0, -5.0, 6.0]], np.float32)
        result = cal_abs_max_channel(inputs, quant_axis=1)
        np.testing.assert_array_equal(result, [4.0, 5.0, 6.0])

    def test_zero_channel_replaced_with_eps(self):
        inputs = np.array([[0.0, -7.0], [0.0, 3.0]], np.float32)
        result = cal_abs_max_channel(inputs, quant_axis=1)
        # column 0 is all zeros -> abs max becomes eps=1e-8. Compare against a
        # float32 reference so the float32 rounding of 1e-8 is not rejected.
        np.testing.assert_allclose(
            result, np.array([1e-8, 7.0], dtype=np.float32), rtol=0, atol=0
        )


class TestQdqWeightSymmetric(unittest.TestCase):
    """qdq_weight symmetric int8: scale = per-column abs-max, bnt = 127,
    quant = round(x/scale*bnt) clipped to [-128, 127]; dequant = q/bnt*scale."""

    def _x(self):
        # per-column abs-max: col0 = 127, col1 = 20 (exact-integer mapping).
        return np.array([[127.0, 20.0], [-30.0, -20.0]], dtype=np.float32)

    def test_quant_scale_and_content(self):
        quant_x, scales = qdq_weight(self._x(), quant_bit=8)
        np.testing.assert_array_equal(scales, [127.0, 20.0])
        # row0: 127/127*127=127, 20/20*127=127
        # row1: -30/127*127=-30, -20/20*127=-127
        expected = np.array([[127, 127], [-30, -127]], dtype=np.int8)
        self.assertEqual(quant_x.dtype, np.int8)
        np.testing.assert_array_equal(quant_x, expected)

    def test_dequant_reconstructs_original(self):
        quant_x, scales = qdq_weight(self._x(), quant_bit=8)
        dequant_x, out_scales = qdq_weight(
            quant_x, quant_bit=8, scales=scales, dequant=True
        )
        self.assertEqual(dequant_x.dtype, np.float32)
        np.testing.assert_allclose(dequant_x, self._x(), rtol=0, atol=1e-5)
        np.testing.assert_array_equal(out_scales, scales)


class TestAsymmetryQdqWeight(unittest.TestCase):
    """asymmetry_qdq_weight uint8: scale = per-column (max-min), bnt = 255,
    quant = round((x-min)/scale*bnt) clipped to [0, 255]."""

    def _x(self):
        # shape [3, 2]; each column range = 255 for exact-integer mapping.
        return np.array(
            [[0.0, -255.0], [100.0, -155.0], [255.0, 0.0]], dtype=np.float32
        )

    def test_quant_min_max_and_content(self):
        quant_x, mins, maxs = asymmetry_qdq_weight(self._x(), quant_bit=8)
        # asymmetry_qdq_weight derives (max, min) via cal_abs_min_max_channel,
        # which replaces an exactly-zero channel extreme with eps=1e-8. col0's
        # min (0.0) and col1's max (0.0) are therefore reported as 1e-8, not 0.
        np.testing.assert_allclose(
            mins, np.array([1e-8, -255.0], dtype=np.float32), rtol=0, atol=0
        )
        np.testing.assert_allclose(
            maxs, np.array([255.0, 1e-8], dtype=np.float32), rtol=0, atol=0
        )
        # (x-min)/255*255 = (x-min); the ~1e-8 eps offset rounds away:
        expected = np.array([[0, 0], [100, 100], [255, 255]], dtype=np.uint8)
        self.assertEqual(quant_x.dtype, np.uint8)
        np.testing.assert_array_equal(quant_x, expected)

    def test_dequant_reconstructs_original(self):
        quant_x, mins, maxs = asymmetry_qdq_weight(self._x(), quant_bit=8)
        dequant_x, scales = asymmetry_qdq_weight(
            x=quant_x, quant_bit=8, mins=mins, maxs=maxs, dequant=True
        )
        self.assertEqual(dequant_x.dtype, np.float32)
        np.testing.assert_allclose(dequant_x, self._x(), rtol=0, atol=1e-5)
        np.testing.assert_array_equal(scales, [255.0, 255.0])


if __name__ == "__main__":
    unittest.main()
