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

"""Tests for paddlefleet.models.multimodal.context_parallel.

Targets get_padding (pure-integer SP/CP/TP-overlap/FP8 padding rule) and
get_packed_seq_params (builds PackedSeqParams / cu_seqlens for TE attention).
Expected values are hand-derived from the production branch logic, never by
re-calling the function under test.

No accelerator is required: get_padding is integer arithmetic and
get_packed_seq_params only calls paddle.arange / shape ops, so tokens are
placed on CPU explicitly. Paddle is still needed to import the module (its
top-level `import paddle`), so every test honestly skips when Paddle is
absent rather than faking a pass.
"""

import unittest

import numpy as np

try:
    import paddle

    from paddlefleet.models.multimodal.context_parallel import (
        get_packed_seq_params,
        get_padding,
    )
    from paddlefleet.packed_seq_params import PackedSeqParams

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle (or the module) is unavailable locally
    paddle = None
    _IMPORT_ERROR = exc

_SKIP_REASON = (
    None
    if _IMPORT_ERROR is None
    else f"paddle / context_parallel import failed: {_IMPORT_ERROR!r}"
)


@unittest.skipUnless(
    _IMPORT_ERROR is None, _SKIP_REASON or "paddle unavailable"
)
class TestGetPadding(unittest.TestCase):
    """get_padding returns the minimal non-negative padding that rounds
    seq_len up to the factor selected by the (priority-ordered) branch."""

    def test_already_aligned_needs_no_padding(self):
        # factor 1 (nothing enabled): 128 is already a multiple -> 0.
        self.assertEqual(
            get_padding(seq_len=128, cp_size=1, tp_size=1, has_sp=False), 0
        )

    def test_sp_only_pads_to_tp_multiple(self):
        # has_sp, cp<=1 -> factor = tp_size = 4; ceil(10/4)*4 = 12 -> 2.
        self.assertEqual(
            get_padding(seq_len=10, cp_size=1, tp_size=4, has_sp=True), 2
        )
        # 13 -> ceil to 16 with tp=8 -> 3.
        self.assertEqual(
            get_padding(seq_len=13, cp_size=1, tp_size=8, has_sp=True), 3
        )

    def test_cp_only_pads_to_two_cp_multiple(self):
        # cp>1, no sp -> factor = cp_size*2 = 8; ceil(10/8)*8 = 16 -> 6.
        self.assertEqual(
            get_padding(seq_len=10, cp_size=4, tp_size=1, has_sp=False), 6
        )
        # cp=3 -> factor 6; ceil(17/6)*6 = 18 -> 1.
        self.assertEqual(
            get_padding(seq_len=17, cp_size=3, tp_size=1, has_sp=False), 1
        )

    def test_sp_and_cp_pads_to_tp_cp_two_multiple(self):
        # has_sp and cp>1 -> factor = tp*cp*2 = 2*2*2 = 8 -> pad 6.
        self.assertEqual(
            get_padding(seq_len=10, cp_size=2, tp_size=2, has_sp=True), 6
        )

    def test_fp8_mxfp8_uses_factor_32(self):
        # fp8 only, mxfp8 -> factor 32; ceil(10/32)*32 = 32 -> 22.
        self.assertEqual(
            get_padding(
                seq_len=10,
                cp_size=1,
                tp_size=1,
                has_sp=False,
                fp8_enabled=True,
                fp8_recipe="mxfp8",
            ),
            22,
        )

    def test_fp8_non_mxfp8_uses_factor_16(self):
        # fp8 only, non-mxfp8 -> factor 16; ceil(10/16)*16 = 16 -> 6.
        self.assertEqual(
            get_padding(
                seq_len=10,
                cp_size=1,
                tp_size=1,
                has_sp=False,
                fp8_enabled=True,
                fp8_recipe="e4m3",
            ),
            6,
        )
        # 48 is already a multiple of 16 -> 0.
        self.assertEqual(
            get_padding(
                seq_len=48,
                cp_size=1,
                tp_size=1,
                has_sp=False,
                fp8_enabled=True,
                fp8_recipe="e4m3",
            ),
            0,
        )

    def test_fp8_disabled_gives_factor_one(self):
        self.assertEqual(
            get_padding(
                seq_len=64,
                cp_size=1,
                tp_size=1,
                has_sp=False,
                fp8_enabled=False,
                fp8_recipe="mxfp8",
            ),
            0,
        )

    def test_branch_priority_sp_cp_over_fp8(self):
        # With fp8 also on, SP+CP must still win: factor 8 -> 6, not 22.
        self.assertEqual(
            get_padding(
                seq_len=10,
                cp_size=2,
                tp_size=2,
                has_sp=True,
                fp8_enabled=True,
                fp8_recipe="mxfp8",
            ),
            6,
        )

    def test_branch_priority_sp_over_fp8(self):
        # has_sp (cp<=1) wins over fp8: factor tp=4 -> 2, not 22.
        self.assertEqual(
            get_padding(
                seq_len=10,
                cp_size=1,
                tp_size=4,
                has_sp=True,
                fp8_enabled=True,
                fp8_recipe="mxfp8",
            ),
            2,
        )

    def test_branch_priority_cp_over_fp8(self):
        # cp>1 (no sp) wins over fp8: factor cp*2=8 -> 6, not 22.
        self.assertEqual(
            get_padding(
                seq_len=10,
                cp_size=4,
                tp_size=1,
                has_sp=False,
                fp8_enabled=True,
                fp8_recipe="mxfp8",
            ),
            6,
        )

    def test_decoder_tp_comm_overlap_returns_seq_len_delta(self):
        # has_sp and overlap -> early return decoder_seq_len - seq_len.
        self.assertEqual(
            get_padding(
                seq_len=100,
                cp_size=1,
                tp_size=1,
                has_sp=True,
                decoder_tp_comm_overlap=True,
                decoder_seq_len=256,
            ),
            156,
        )

    def test_decoder_tp_comm_overlap_does_no_clamping(self):
        # Current contract: raw subtraction, so a smaller decoder length
        # yields a negative padding (no max(0, ...) guard in production).
        self.assertEqual(
            get_padding(
                seq_len=200,
                cp_size=1,
                tp_size=1,
                has_sp=True,
                decoder_tp_comm_overlap=True,
                decoder_seq_len=128,
            ),
            -72,
        )

    def test_decoder_overlap_ignored_without_sp(self):
        # overlap flag only fires together with has_sp; without sp it falls
        # through to factor 1 -> 0 (NOT the 156 delta).
        self.assertEqual(
            get_padding(
                seq_len=100,
                cp_size=1,
                tp_size=1,
                has_sp=False,
                decoder_tp_comm_overlap=True,
                decoder_seq_len=256,
            ),
            0,
        )

    def test_decoder_overlap_requires_decoder_seq_len(self):
        with self.assertRaises(AssertionError):
            get_padding(
                seq_len=128,
                cp_size=1,
                tp_size=1,
                has_sp=True,
                decoder_tp_comm_overlap=True,
                decoder_seq_len=None,
            )


def _cpu_tokens(batch_size, seq_len):
    """Distinguishable int64 tokens on CPU (function only reads .shape)."""
    arr = np.arange(batch_size * seq_len, dtype="int64").reshape(
        [batch_size, seq_len]
    )
    return paddle.to_tensor(arr, place=paddle.CPUPlace())


@unittest.skipUnless(
    _IMPORT_ERROR is None, _SKIP_REASON or "paddle unavailable"
)
class TestGetPackedSeqParams(unittest.TestCase):
    """get_packed_seq_params builds cu_seqlens offsets and selects the qkv
    format. valid = text + img - padding; padded = text + img. THD is chosen
    only when cp_size > 1 AND (padding_needed > 0 OR use_packed_sequence)."""

    def test_no_cp_uses_sbhd_and_valid_offsets(self):
        # batch=2, seq=10, img=576, pad=0 -> valid=padded=586.
        params = get_packed_seq_params(
            tokens=_cpu_tokens(2, 10),
            img_seq_len=576,
            padding_needed=0,
            cp_size=1,
            use_packed_sequence=False,
        )
        self.assertIsInstance(params, PackedSeqParams)
        self.assertEqual(params.qkv_format, "sbhd")
        # cu_seqlens = [0, 586, 1172] (step = valid seqlen, batch+1 entries).
        self.assertEqual(params.cu_seqlens_q.tolist(), [0, 586, 1172])
        self.assertEqual(params.cu_seqlens_kv.tolist(), [0, 586, 1172])
        self.assertEqual(params.cu_seqlens_q.dtype, paddle.int32)
        # No CP -> no padded offsets provided.
        self.assertIsNone(params.cu_seqlens_q_padded)
        self.assertIsNone(params.cu_seqlens_kv_padded)
        self.assertEqual(params.max_seqlen_q, 586)
        self.assertEqual(params.max_seqlen_kv, 586)
        self.assertEqual(params.total_seqlen_q, 1172)
        self.assertEqual(params.total_seqlen_kv, 1172)

    def test_cp_with_padding_uses_thd_and_separate_padded_offsets(self):
        # batch=2, seq=10, img=576, pad=64 -> valid=522, padded=586.
        params = get_packed_seq_params(
            tokens=_cpu_tokens(2, 10),
            img_seq_len=576,
            padding_needed=64,
            cp_size=2,
            use_packed_sequence=False,
        )
        self.assertEqual(params.qkv_format, "thd")
        # Unpadded offsets step by the *valid* length 522.
        self.assertEqual(params.cu_seqlens_q.tolist(), [0, 522, 1044])
        self.assertEqual(params.cu_seqlens_kv.tolist(), [0, 522, 1044])
        # Padded offsets step by the *padded* length 586.
        self.assertEqual(params.cu_seqlens_q_padded.tolist(), [0, 586, 1172])
        self.assertEqual(params.cu_seqlens_kv_padded.tolist(), [0, 586, 1172])
        self.assertEqual(params.cu_seqlens_q_padded.dtype, paddle.int32)
        self.assertEqual(params.max_seqlen_q, 586)
        self.assertEqual(params.total_seqlen_q, 1044)
        self.assertEqual(params.total_seqlen_kv, 1044)

    def test_cp_with_packed_sequence_uses_thd_even_without_padding(self):
        # pad=0 but use_packed_sequence=True still triggers THD + padded off.
        params = get_packed_seq_params(
            tokens=_cpu_tokens(2, 10),
            img_seq_len=576,
            padding_needed=0,
            cp_size=2,
            use_packed_sequence=True,
        )
        self.assertEqual(params.qkv_format, "thd")
        # valid == padded == 586 here, so both offset sets match.
        self.assertEqual(params.cu_seqlens_q.tolist(), [0, 586, 1172])
        self.assertEqual(params.cu_seqlens_q_padded.tolist(), [0, 586, 1172])
        self.assertEqual(params.total_seqlen_q, 1172)

    def test_cp_without_padding_or_packing_stays_sbhd(self):
        # cp>1 alone is NOT enough: (pad>0 or packed) is False -> sbhd, None.
        params = get_packed_seq_params(
            tokens=_cpu_tokens(2, 10),
            img_seq_len=576,
            padding_needed=0,
            cp_size=2,
            use_packed_sequence=False,
        )
        self.assertEqual(params.qkv_format, "sbhd")
        self.assertIsNone(params.cu_seqlens_q_padded)
        self.assertIsNone(params.cu_seqlens_kv_padded)
        self.assertEqual(params.cu_seqlens_q.tolist(), [0, 586, 1172])

    def test_padding_without_cp_stays_sbhd(self):
        # cp_size==1 gates out THD even when padding/packing are requested.
        params = get_packed_seq_params(
            tokens=_cpu_tokens(2, 10),
            img_seq_len=576,
            padding_needed=64,
            cp_size=1,
            use_packed_sequence=True,
        )
        self.assertEqual(params.qkv_format, "sbhd")
        self.assertIsNone(params.cu_seqlens_q_padded)
        # valid = 10 + 576 - 64 = 522.
        self.assertEqual(params.cu_seqlens_q.tolist(), [0, 522, 1044])
        self.assertEqual(params.total_seqlen_q, 1044)

    def test_offsets_track_batch_size(self):
        # batch=3, seq=20, img=100, pad=0 -> valid=120; batch+1=4 offsets.
        params = get_packed_seq_params(
            tokens=_cpu_tokens(3, 20),
            img_seq_len=100,
            padding_needed=0,
            cp_size=1,
        )
        self.assertEqual(params.cu_seqlens_q.tolist(), [0, 120, 240, 360])
        self.assertEqual(params.cu_seqlens_q.shape[0], 4)
        self.assertEqual(params.max_seqlen_q, 120)
        self.assertEqual(params.total_seqlen_q, 360)


if __name__ == "__main__":
    unittest.main()
