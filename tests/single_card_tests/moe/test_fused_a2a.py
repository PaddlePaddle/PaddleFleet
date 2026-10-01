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
"""Single-process behavior tests for ``transformer/moe/fused_a2a``.

``fused_a2a`` is the DeepEP all-to-all bridge for expert parallelism. The
dispatch/combine paths are genuine cross-rank collectives, so their numerical
correctness (token routing, expert homing, cross-rank reduction) can only be
proven with a real multi-rank process group and are deliberately NOT asserted
here (faking ``world_size`` + mocking the collective would only re-assert the
test's own scaffolding).

What this file pins down instead is the single-process-observable layout logic
that runs on every rank *before* the collective is issued and *after* it
returns: hidden-byte sizing, the FP8 scale transpose/trim/validation feeding
DeepEP, and the SonicMoE E8M0 scale byte-pack/unpack round trip. Every expected
value is derived by hand (numpy transpose/slice, little-endian byte math,
signature introspection) without calling the function under test.

paddle / paddlefleet may be absent in a CPU-only checkout; imports are guarded
and the suite skips with an honest reason rather than reporting a false pass.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)
# The package lives under ``src/`` in a source checkout.
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

_SKIP_REASON = None
try:
    import paddle

    from paddlefleet.transformer.moe.fused_a2a import (
        _normalize_fp8_scale_for_deepep,
        _pack_sonic_fp8_scale_for_deepep,
        _supports_sonic_scale_word_packing,
        _unpack_sonic_fp8_scale_from_deepep,
        get_hidden_bytes,
    )
except ImportError as exc:  # pragma: no cover - env-dependent
    paddle = None
    _SKIP_REASON = f"paddle/paddlefleet import failed: {exc}"


def _le_pack(row_bytes):
    """Independent little-endian uint8[4] -> int32 reference (no paddle)."""
    value = 0
    for i, byte in enumerate(row_bytes):
        value |= int(byte) << (8 * i)
    return value


@unittest.skipUnless(_SKIP_REASON is None, _SKIP_REASON or "deps present")
class TestGetHiddenBytes(unittest.TestCase):
    """get_hidden_bytes = hidden_dim * max(element_size, 2)."""

    def test_fp32_uses_element_size_four(self):
        # hidden=64, element_size(float32)=4 -> max(4, 2)=4 -> 64*4=256.
        x = paddle.zeros([4, 64], dtype=paddle.float32)
        self.assertEqual(get_hidden_bytes(x), 256)

    def test_fp16_uses_element_size_two(self):
        # element_size(float16)=2 -> max(2, 2)=2 -> 64*2=128.
        x = paddle.zeros([4, 64], dtype=paddle.float16)
        self.assertEqual(get_hidden_bytes(x), 128)

    def test_fp8_floored_to_two_bytes(self):
        # element_size(fp8)=1 -> max(1, 2)=2 -> 64*2=128 (the 2-byte floor).
        x = paddle.zeros([4, 64], dtype=paddle.float8_e4m3fn)
        self.assertEqual(get_hidden_bytes(x), 128)

    def test_reads_hidden_axis_not_token_axis(self):
        # Distinguish shape[1] from shape[0]: 8 tokens must not leak into the
        # byte count; hidden=10, fp32 -> 10*4=40 regardless of 8 rows.
        x = paddle.zeros([8, 10], dtype=paddle.float32)
        self.assertEqual(get_hidden_bytes(x), 40)


@unittest.skipUnless(_SKIP_REASON is None, _SKIP_REASON or "deps present")
class TestNormalizeFp8ScaleForDeepEP(unittest.TestCase):
    """FP8 scale is coerced to token-major [num_tokens, num_scales]."""

    def test_transposes_scale_group_major_to_token_major(self):
        # x_fp8 hidden=256 -> num_scales=256//128=2; scale rows==num_scales(2)
        # so it is transposed. Use distinguishable content to catch a no-op or
        # wrong-direction transpose that a shape-only check would miss.
        x_fp8 = paddle.zeros([4, 256], dtype=paddle.float8_e4m3fn)
        src = np.arange(8, dtype=np.float32).reshape(2, 4)
        scale = paddle.to_tensor(src)
        out = _normalize_fp8_scale_for_deepep(x_fp8, scale)
        self.assertEqual(out.shape, [4, 2])
        np.testing.assert_array_equal(out.numpy(), src.T)

    def test_trims_padded_leading_tokens_keeping_first_rows(self):
        # x_fp8 has 3 real tokens, hidden=256 -> num_scales=2. Scale already
        # token-major with 4 rows (padded) must drop the LAST row, keeping
        # rows 0..2 in order.
        x_fp8 = paddle.zeros([3, 256], dtype=paddle.float8_e4m3fn)
        src = np.arange(8, dtype=np.float32).reshape(4, 2)
        scale = paddle.to_tensor(src)
        out = _normalize_fp8_scale_for_deepep(x_fp8, scale)
        self.assertEqual(out.shape, [3, 2])
        np.testing.assert_array_equal(out.numpy(), src[:3])

    def test_ue8m0_quarters_scale_width_then_transposes(self):
        # use_ue8m0 packs 4 blocks per word: hidden=512 -> 512//128//4=1 scale.
        x_fp8 = paddle.zeros([4, 512], dtype=paddle.float8_e4m3fn)
        src = np.arange(4, dtype=np.int32).reshape(1, 4)
        scale = paddle.to_tensor(src)
        out = _normalize_fp8_scale_for_deepep(x_fp8, scale, use_ue8m0=True)
        self.assertEqual(out.shape, [4, 1])
        np.testing.assert_array_equal(out.numpy(), src.T)

    def test_rejects_scale_that_matches_neither_layout(self):
        # hidden=256 -> num_scales=2; a [4, 3] scale cannot be coerced to
        # [4, 2] and must raise before any DeepEP dispatch.
        x_fp8 = paddle.zeros([4, 256], dtype=paddle.float8_e4m3fn)
        scale = paddle.to_tensor(np.arange(12, dtype=np.float32).reshape(4, 3))
        with self.assertRaisesRegex(RuntimeError, "Invalid FP8 scale shape"):
            _normalize_fp8_scale_for_deepep(x_fp8, scale)


# PLACEHOLDER_SONIC


@unittest.skipUnless(_SKIP_REASON is None, _SKIP_REASON or "deps present")
class TestSonicScaleWordPacking(unittest.TestCase):
    """E8M0 uint8 scale bytes carried across DeepEP as int32 words."""

    def test_pack_aligned_views_four_bytes_into_one_word(self):
        # hidden=128 -> num_groups=(128+31)//32=4 (multiple of 4), so 4 uint8
        # bytes are reinterpreted as one little-endian int32 word: (2,4)->(2,1).
        x_fp8 = paddle.zeros([2, 128], dtype=paddle.float8_e4m3fn)
        rows = [[1, 2, 3, 4], [5, 6, 7, 8]]
        scale = paddle.to_tensor(np.array(rows, dtype=np.uint8))
        out = _pack_sonic_fp8_scale_for_deepep(x_fp8, scale)
        self.assertEqual(out.shape, [2, 1])
        self.assertEqual(out.dtype, paddle.int32)
        expected = np.array(
            [[_le_pack(rows[0])], [_le_pack(rows[1])]], np.int32
        )
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_pack_already_packed_int32_returned_unchanged(self):
        # An int32 [num_tokens, packed_groups]=[2,1] scale is recognised as
        # already-packed and returned as the same object, untouched.
        x_fp8 = paddle.zeros([2, 128], dtype=paddle.float8_e4m3fn)
        scale = paddle.to_tensor(np.array([[10], [20]], dtype=np.int32))
        out = _pack_sonic_fp8_scale_for_deepep(x_fp8, scale)
        self.assertIs(out, scale)
        np.testing.assert_array_equal(out.numpy(), [[10], [20]])

    def test_pack_unaligned_groups_casts_values_not_bytes(self):
        # hidden=96 -> num_groups=3 (not a multiple of 4): the fallback casts
        # uint8 values to int32 element-wise (shape preserved), NOT a byte view.
        x_fp8 = paddle.zeros([2, 96], dtype=paddle.float8_e4m3fn)
        src = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
        scale = paddle.to_tensor(src)
        out = _pack_sonic_fp8_scale_for_deepep(x_fp8, scale)
        self.assertEqual(out.shape, [2, 3])
        self.assertEqual(out.dtype, paddle.int32)
        np.testing.assert_array_equal(out.numpy(), src.astype(np.int32))

    def test_pack_rejects_wrong_group_count(self):
        # hidden=128 -> num_groups=4; a [2,5] carrier is neither raw nor packed.
        x_fp8 = paddle.zeros([2, 128], dtype=paddle.float8_e4m3fn)
        scale = paddle.to_tensor(np.zeros((2, 5), dtype=np.uint8))
        with self.assertRaisesRegex(RuntimeError, "Invalid Sonic FP8 scale"):
            _pack_sonic_fp8_scale_for_deepep(x_fp8, scale)

    def test_pack_rejects_non_uint8_raw_carrier(self):
        # Correct [2,4] shape but float32 dtype: the raw carrier must be uint8.
        x_fp8 = paddle.zeros([2, 128], dtype=paddle.float8_e4m3fn)
        scale = paddle.to_tensor(np.zeros((2, 4), dtype=np.float32))
        with self.assertRaisesRegex(TypeError, "uint8"):
            _pack_sonic_fp8_scale_for_deepep(x_fp8, scale)

    def test_unpack_inverts_aligned_pack_round_trip(self):
        # Pack (2,4) uint8 -> (2,1) int32, then unpack must recover the exact
        # original bytes and shape. Shape changes both ways guard against a
        # no-op passing the round trip.
        x_fp8 = paddle.zeros([2, 128], dtype=paddle.float8_e4m3fn)
        src = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.uint8)
        packed = _pack_sonic_fp8_scale_for_deepep(x_fp8, paddle.to_tensor(src))
        self.assertEqual(packed.shape, [2, 1])
        recovered = _unpack_sonic_fp8_scale_from_deepep(x_fp8, packed)
        self.assertEqual(recovered.shape, [2, 4])
        self.assertEqual(recovered.dtype, paddle.uint8)
        np.testing.assert_array_equal(recovered.numpy(), src)

    def test_unpack_inverts_unaligned_pack_round_trip(self):
        # hidden=96 -> num_groups=3 cast path: pack casts to int32, unpack casts
        # back to uint8, preserving values and shape.
        x_fp8 = paddle.zeros([2, 96], dtype=paddle.float8_e4m3fn)
        src = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
        packed = _pack_sonic_fp8_scale_for_deepep(x_fp8, paddle.to_tensor(src))
        self.assertEqual(packed.shape, [2, 3])
        recovered = _unpack_sonic_fp8_scale_from_deepep(x_fp8, packed)
        self.assertEqual(recovered.shape, [2, 3])
        self.assertEqual(recovered.dtype, paddle.uint8)
        np.testing.assert_array_equal(recovered.numpy(), src)

    def test_unpack_rejects_carrier_matching_no_layout(self):
        # A uint8 [2,4] tensor is a valid raw carrier but never a valid packed
        # carrier for unpack (which expects int32); it must raise.
        x_fp8 = paddle.zeros([2, 128], dtype=paddle.float8_e4m3fn)
        scale = paddle.to_tensor(np.zeros((2, 4), dtype=np.uint8))
        with self.assertRaisesRegex(RuntimeError, "Invalid packed Sonic FP8"):
            _unpack_sonic_fp8_scale_from_deepep(x_fp8, scale)


@unittest.skipUnless(_SKIP_REASON is None, _SKIP_REASON or "deps present")
class TestSupportsSonicScaleWordPacking(unittest.TestCase):
    """Feature probe keys off a 'pack_scale_words' parameter in the signature."""

    def test_none_quantizer_unsupported(self):
        self.assertFalse(_supports_sonic_scale_word_packing(None))

    def test_signature_with_pack_scale_words_supported(self):
        def quantizer(x, pack_scale_words=False):
            return x

        self.assertTrue(_supports_sonic_scale_word_packing(quantizer))

    def test_signature_without_flag_unsupported(self):
        def quantizer(x, scale_dtype=None):
            return x

        self.assertFalse(_supports_sonic_scale_word_packing(quantizer))

    def test_uninspectable_object_unsupported(self):
        # inspect.signature(int_instance) raises TypeError -> caught -> False,
        # never propagated as an error.
        self.assertFalse(_supports_sonic_scale_word_packing(5))


if __name__ == "__main__":
    unittest.main()
