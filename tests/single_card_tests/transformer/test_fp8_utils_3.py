# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
"""Behaviour tests for CPU-computable MoE FP8 quantization helpers.

Targets pure-numeric helpers in
``paddlefleet.transformer.moe.fp8_utils`` whose expected outputs can be
derived by hand without a GPU, DeepGEMM, or the ``paddlefleet_ops`` custom
kernels:

* ``moe_token_padding_alignment`` -- per-expert token padding policy.
* ``ceil_to_ue8m0``              -- round a scale up to a power of two.
* ``_quantize_to_fp4_e2m1``      -- fp32 -> e2m1 4-bit code (with sign).
* ``quant_blockwize`` (fp4)      -- blockwise fp4 quant + nibble packing.
* ``fuse_stack_fp8_quant_python`` / transpose variant -- stacked layout.
* ``_weighted_swiglu_fp32``      -- fp32 weighted swiglu activation.

Every expected value below is an independent hand anchor (literal constants
or an inlined numpy formula), never produced by calling the function under
test. All computation is fp32/int only, so no ``float8_e4m3fn`` runtime
support is required.
"""

import math
import os
import sys
import unittest

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

# Honest capability probe: only a genuinely missing dependency (paddle or the
# paddlefleet package tree) is a legitimate skip. Any other error -- compile
# failure, API change, GPU-only import blowing up with RuntimeError -- must
# surface as a real failure rather than be swallowed into a green run.
try:
    import numpy as np
    import paddle

    from paddlefleet.transformer.moe import fp8_utils

    IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    np = None
    paddle = None
    fp8_utils = None
    IMPORT_ERROR = repr(exc)


@unittest.skipIf(
    fp8_utils is None,
    f"paddle / paddlefleet not importable in this environment: {IMPORT_ERROR}",
)
class _CpuFp8TestBase(unittest.TestCase):
    """Runs the numeric helpers on CPU and restores the process device.

    Setting the paddle device is process-global mutation, so the original
    device is captured and restored (even on failure) via ``addCleanup``.
    """

    def setUp(self):
        original_device = paddle.device.get_device()
        self.addCleanup(paddle.set_device, original_device)
        paddle.set_device("cpu")


def _silu_ref(x):
    """Independent numpy SiLU: x * sigmoid(x), computed in float64."""
    x = np.asarray(x, dtype=np.float64)
    return x / (1.0 + np.exp(-x))


class TestMoeTokenPaddingAlignment(_CpuFp8TestBase):
    """``moe_token_padding_alignment`` returns 1 only on the single
    accuracy-compatible pure-bf16 non-grouped path; every other flag
    combination keeps the FP8_ALIGN (128) padding."""

    def test_only_accuracy_compatible_bf16_nongrouped_skips_padding(self):
        self.assertEqual(fp8_utils.FP8_ALIGN, 128)
        # The one combination that disables padding.
        self.assertEqual(
            fp8_utils.moe_token_padding_alignment(
                use_fp8_mlp=False,
                moe_grouped_gemm=False,
                use_accuracy_compatible=True,
            ),
            1,
        )
        # Flip each contributing flag in isolation -> back to 128, proving
        # every flag is actually consumed by the branch.
        self.assertEqual(
            fp8_utils.moe_token_padding_alignment(
                use_fp8_mlp=True,
                moe_grouped_gemm=False,
                use_accuracy_compatible=True,
            ),
            128,
        )
        self.assertEqual(
            fp8_utils.moe_token_padding_alignment(
                use_fp8_mlp=False,
                moe_grouped_gemm=True,
                use_accuracy_compatible=True,
            ),
            128,
        )
        self.assertEqual(
            fp8_utils.moe_token_padding_alignment(
                use_fp8_mlp=False,
                moe_grouped_gemm=False,
                use_accuracy_compatible=False,
            ),
            128,
        )


class TestCeilToUe8m0(_CpuFp8TestBase):
    """``ceil_to_ue8m0`` rounds |x| UP to the nearest power of two.

    Hand anchors: exact powers of two map to themselves; anything in
    between snaps to the next power up; the magnitude is taken so sign is
    irrelevant.
    """

    def test_rounds_up_to_power_of_two(self):
        # (input, expected) pairs derived by hand from IEEE-754 exponents.
        pairs = [
            (1.0, 1.0),  # 2**0 exact
            (0.5, 0.5),  # 2**-1 exact
            (2.0, 2.0),  # 2**1 exact
            (4.0, 4.0),  # 2**2 exact
            (1.5, 2.0),  # (1, 2] -> 2
            (3.0, 4.0),  # (2, 4] -> 4
            (3.9, 4.0),  # still (2, 4] -> 4
            (0.1, 0.125),  # 2**-4 < 0.1 < 2**-3 -> 2**-3
            (-1.5, 2.0),  # magnitude only
            (-3.0, 4.0),
        ]
        src = paddle.to_tensor([p[0] for p in pairs], dtype="float32")
        out = fp8_utils.ceil_to_ue8m0(src).numpy().astype(np.float64)
        expected = np.array([p[1] for p in pairs], dtype=np.float64)
        np.testing.assert_array_equal(out, expected)

    def test_cross_check_against_log2_formula(self):
        # A second, independent reference: smallest power of two >= |x|.
        vals = [0.7, 1.1, 2.3, 5.0, 6.5, 0.26, 100.0]
        src = paddle.to_tensor(vals, dtype="float32")
        out = fp8_utils.ceil_to_ue8m0(src).numpy().astype(np.float64)
        ref = np.array(
            [2.0 ** math.ceil(math.log2(v)) for v in vals], dtype=np.float64
        )
        np.testing.assert_array_equal(out, ref)


class TestQuantizeToFp4E2m1(_CpuFp8TestBase):
    """``_quantize_to_fp4_e2m1`` maps fp32 to a 4-bit e2m1 code.

    Representable magnitudes {0,.5,1,1.5,2,3,4,6} -> codes {0..7}; bit3 is
    the sign, but only for a non-zero magnitude code (no negative zero).
    """

    def test_code_and_sign_boundaries(self):
        # (value, expected 4-bit code) hand-derived from the midpoint rule.
        pairs = [
            (0.0, 0),
            (0.5, 1),
            (1.0, 2),
            (1.5, 3),
            (2.0, 4),
            (3.0, 5),
            (4.0, 6),
            (6.0, 7),
            (-1.0, 0b1010),  # code 2 with sign bit -> 10
            (-6.0, 0b1111),  # code 7 with sign bit -> 15
            (-0.1, 0),  # rounds to magnitude 0 -> sign suppressed
        ]
        src = paddle.to_tensor([p[0] for p in pairs], dtype="float32")
        out = fp8_utils._quantize_to_fp4_e2m1(src).numpy().tolist()
        self.assertEqual(out, [p[1] for p in pairs])

    def test_rounding_midpoints_snap_down_on_tie(self):
        # Exactly on a midpoint the strict ``>`` keeps the lower code.
        src = paddle.to_tensor([0.25, 0.75, 2.5, 5.0], dtype="float32")
        out = fp8_utils._quantize_to_fp4_e2m1(src).numpy().tolist()
        # 0.25 -> code 0 ; 0.75 -> code 1 ; 2.5 -> code 4 ; 5.0 -> code 6
        self.assertEqual(out, [0, 1, 4, 6])


class TestQuantBlockwizeFp4(_CpuFp8TestBase):
    """``quant_blockwize`` fp4 path: per-row (per-token) scale = amax/6,
    e2m1 codes, then two nibbles packed into one int8 with the FIRST
    element in the LOW nibble.
    """

    def test_per_row_scale_and_nibble_packing(self):
        # Row0 amax=6 -> sf=1.0 ; Row1 amax=3 -> sf=0.5. After scaling both
        # rows become [6, 3] -> codes [7, 5] -> pack 7 | (5<<4) = 87.
        x = paddle.to_tensor([[6.0, 3.0], [3.0, 1.5]], dtype="float32")
        q, sf = fp8_utils.quant_blockwize(
            x,
            quant_method="1x2",
            quant_dtype="fp4",
            using_ue8m0_scale=False,
        )
        self.assertEqual(q.dtype, paddle.int8)
        self.assertEqual(q.numpy().tolist(), [[87], [87]])
        np.testing.assert_allclose(
            sf.numpy().astype(np.float64), [[1.0], [0.5]], atol=1e-6
        )

    def test_sign_lands_in_high_nibble_and_wraps_to_int8(self):
        # x=[[6, 3, -6, 0]], group size 2, sf per group = amax/6 = 1.0.
        # Codes: 6->7, 3->5, -6->15, 0->0.
        # pack g0: 7 | (5<<4) = 87 ; pack g1: 15 | (0<<4) = 15.
        x = paddle.to_tensor([[6.0, 3.0, -6.0, 0.0]], dtype="float32")
        q, sf = fp8_utils.quant_blockwize(
            x,
            quant_method="1x2",
            quant_dtype="fp4",
            using_ue8m0_scale=False,
        )
        self.assertEqual(q.numpy().tolist(), [[87, 15]])
        np.testing.assert_allclose(
            sf.numpy().astype(np.float64), [[1.0, 1.0]], atol=1e-6
        )

    def test_negative_high_nibble_two_complement(self):
        # x=[[6, -6]] one group: codes 7 and 15 -> 7 | (15<<4) = 247 which
        # wraps to int8 -9.
        x = paddle.to_tensor([[6.0, -6.0]], dtype="float32")
        q, _ = fp8_utils.quant_blockwize(
            x,
            quant_method="1x2",
            quant_dtype="fp4",
            using_ue8m0_scale=False,
        )
        self.assertEqual(q.numpy().tolist(), [[-9]])

    def test_odd_columns_rejected(self):
        # fp4 packs pairs, so an odd contraction dim must assert.
        bad = paddle.to_tensor([[1.0, 2.0, 3.0]], dtype="float32")
        with self.assertRaises(AssertionError):
            fp8_utils.quant_blockwize(
                bad, quant_method="1x3", quant_dtype="fp4"
            )


class TestFuseStackFp8QuantPython(_CpuFp8TestBase):
    """Stacked-expert fp4 quant helpers preserve the [E, R, C] layout and
    the transpose variant swaps H0/H1 BEFORE quantizing (so it is not the
    same as the non-transpose path)."""

    def test_non_transpose_layout_and_values(self):
        # w[0] = [[6, 3], [-6, 3]] -> rows quantize independently.
        # row [6,3]:  sf=1, codes [7,5] -> 7|(5<<4)=87
        # row [-6,3]: sf=1, codes [15,5] -> 15|(5<<4)=95
        w = paddle.to_tensor([[[6.0, 3.0], [-6.0, 3.0]]], dtype="float32")
        q, sf = fp8_utils.fuse_stack_fp8_quant_python(
            w, quant_method="1x2", quant_dtype="fp4", using_ue8m0_scale=False
        )
        self.assertEqual(list(q.shape), [1, 2, 1])
        self.assertEqual(q.numpy().tolist(), [[[87], [95]]])
        self.assertEqual(list(sf.shape), [1, 2, 1])
        np.testing.assert_allclose(
            sf.numpy().astype(np.float64), [[[1.0], [1.0]]], atol=1e-6
        )

    def test_transpose_variant_transposes_before_quant(self):
        # Same w, but transposed to [[6,-6],[3,3]] first.
        # row [6,-6]: sf=1,   codes [7,15] -> 7|(15<<4)=247 -> int8 -9
        # row [3,3]:  sf=0.5, codes [7,7]  -> 7|(7<<4)=119
        w = paddle.to_tensor([[[6.0, 3.0], [-6.0, 3.0]]], dtype="float32")
        q, sf = fp8_utils.fuse_stack_transpose_fp8_quant_python(
            w, quant_method="1x2", quant_dtype="fp4", using_ue8m0_scale=False
        )
        self.assertEqual(list(q.shape), [1, 2, 1])
        self.assertEqual(q.numpy().tolist(), [[[-9], [119]]])
        np.testing.assert_allclose(
            sf.numpy().astype(np.float64), [[[1.0], [0.5]]], atol=1e-6
        )

        # And confirm the transpose actually changed the packed output.
        q_plain, _ = fp8_utils.fuse_stack_fp8_quant_python(
            w, quant_method="1x2", quant_dtype="fp4", using_ue8m0_scale=False
        )
        self.assertNotEqual(q.numpy().tolist(), q_plain.numpy().tolist())


class TestWeightedSwigluFp32(_CpuFp8TestBase):
    """``_weighted_swiglu_fp32`` computes silu(gate) * up * probs, with an
    optional gate/up clamp, matching an independent numpy reference."""

    def test_matches_independent_reference(self):
        o1 = paddle.to_tensor([[1.0, -2.0, 3.0, 4.0]], dtype="float32")
        probs = paddle.to_tensor([[0.5]], dtype="float32")
        out = (
            fp8_utils._weighted_swiglu_fp32(o1, probs)
            .numpy()
            .astype(np.float64)
        )
        gate = np.array([[1.0, -2.0]])
        up = np.array([[3.0, 4.0]])
        expected = _silu_ref(gate) * up * 0.5
        np.testing.assert_allclose(out, expected, atol=1e-5, rtol=1e-5)

    def test_clamp_is_consumed(self):
        o1 = paddle.to_tensor([[10.0, -10.0, 2.0, 2.0]], dtype="float32")
        probs = paddle.to_tensor([[0.5]], dtype="float32")

        clamped = (
            fp8_utils._weighted_swiglu_fp32(o1, probs, clamp_value=1.0)
            .numpy()
            .astype(np.float64)
        )
        unclamped = (
            fp8_utils._weighted_swiglu_fp32(o1, probs)
            .numpy()
            .astype(np.float64)
        )

        gate = np.array([[10.0, -10.0]])
        up = np.array([[2.0, 2.0]])
        # Clamp: gate upper-bounded at c; up symmetric [-c, c].
        ref_clamped = (
            _silu_ref(np.minimum(gate, 1.0)) * np.clip(up, -1.0, 1.0) * 0.5
        )
        ref_unclamped = _silu_ref(gate) * up * 0.5

        np.testing.assert_allclose(clamped, ref_clamped, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(
            unclamped, ref_unclamped, atol=1e-5, rtol=1e-5
        )
        # The clamp must actually change the result for this input.
        self.assertFalse(np.allclose(ref_clamped, ref_unclamped))
        self.assertFalse(np.allclose(clamped, unclamped, atol=1e-3))

    def test_one_dim_probs_broadcast_per_row(self):
        o1 = paddle.to_tensor(
            [[1.0, 3.0], [2.0, 4.0]], dtype="float32"
        )  # h == 1, gate=[[1],[2]], up=[[3],[4]]
        probs = paddle.to_tensor([0.5, 2.0], dtype="float32")  # 1-D [M]
        out = (
            fp8_utils._weighted_swiglu_fp32(o1, probs)
            .numpy()
            .astype(np.float64)
        )
        gate = np.array([[1.0], [2.0]])
        up = np.array([[3.0], [4.0]])
        expected = _silu_ref(gate) * up * np.array([[0.5], [2.0]])
        np.testing.assert_allclose(out, expected, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
