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

"""Behavior tests for get_packed_seq_params in the multimodal context
parallel helper.

Scope: this file deliberately covers ONLY ``get_packed_seq_params`` (the
PackedSeqParams builder). The sibling ``test_context_parallel.py`` owns the
disjoint ``get_padding`` arithmetic slice, so the two files do not overlap.

The helper only reads ``tokens.shape`` and ``tokens.device``; token content
never reaches the output. Every expected value below (cu_seqlens contents,
qkv_format branch, max/total seqlen) is derived by hand from the documented
formula, not by re-running the production function.
"""

import unittest

# get_packed_seq_params lives in a module that imports paddle at top level and
# builds real paddle int32 index tensors via paddle.arange, so importing it (and
# calling it) requires a working paddle install. When paddle is absent we skip
# with an honest reason rather than faking a pass. Only precise import failures
# are treated as "missing dependency"; other errors must surface.
try:
    import paddle

    from paddlefleet.models.multimodal.context_parallel import (
        get_packed_seq_params,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # honest: dependency absent
    paddle = None
    get_packed_seq_params = None
    _IMPORT_ERROR = exc

_SKIP_REASON = (
    "paddle / paddlefleet.models.multimodal.context_parallel not importable "
    f"in this environment: {_IMPORT_ERROR}"
)


def _tokens(batch, seq):
    """Build a [batch, seq] int32 tensor on CPU.

    Only the shape is consumed by get_packed_seq_params; values are irrelevant,
    so zeros are used. Kept as a helper so every case shares one construction.
    """
    return paddle.zeros([batch, seq], dtype="int32")


@unittest.skipUnless(get_packed_seq_params is not None, _SKIP_REASON)
class TestGetPackedSeqParams(unittest.TestCase):
    """Hand-derived checks of the PackedSeqParams builder.

    combined_valid_seqlen  = seq + img - padding_needed
    combined_padded_seqlen = seq + img
    cu_seqlens             = arange(0, (B+1)*valid,  step=valid)  -> B+1 entries
    cu_seqlens_padded      = arange(0, (B+1)*padded, step=padded) when
                             cp_size > 1 and (padding_needed > 0 or packed)
    qkv_format             = "thd" on that same branch, else "sbhd"
    """

    def test_no_cp_no_padding_builds_sbhd_and_valid_cu_seqlens(self):
        # B=3, seq=20, img=100, pad=0, cp=1
        # valid = 20 + 100 - 0 = 120 ; padded = 120
        # cu_seqlens = arange(0, 4*120=480, 120) = [0,120,240,360]
        result = get_packed_seq_params(
            tokens=_tokens(3, 20),
            img_seq_len=100,
            padding_needed=0,
            cp_size=1,
        )
        self.assertEqual(result.qkv_format, "sbhd")
        self.assertEqual(result.cu_seqlens_q.tolist(), [0, 120, 240, 360])
        # q and kv share the exact same tensor object here.
        self.assertIs(result.cu_seqlens_q, result.cu_seqlens_kv)
        self.assertEqual(str(result.cu_seqlens_q.dtype), "paddle.int32")
        # cp_size == 1 -> padded branch not taken.
        self.assertIsNone(result.cu_seqlens_q_padded)
        self.assertIsNone(result.cu_seqlens_kv_padded)
        # max = padded seqlen = 120 ; total = B * valid = 3*120 = 360
        self.assertEqual(result.max_seqlen_q, 120)
        self.assertEqual(result.max_seqlen_kv, 120)
        self.assertEqual(result.total_seqlen_q, 360)
        self.assertEqual(result.total_seqlen_kv, 360)

    def test_cp_with_padding_builds_thd_with_distinct_valid_and_padded(self):
        # B=2, seq=10, img=576, pad=16, cp=2
        # valid  = 10 + 576 - 16 = 570 ; padded = 586
        # cu_seqlens        = arange(0, 3*570=1710, 570) = [0,570,1140]
        # cu_seqlens_padded = arange(0, 3*586=1758, 586) = [0,586,1172]
        result = get_packed_seq_params(
            tokens=_tokens(2, 10),
            img_seq_len=576,
            padding_needed=16,
            cp_size=2,
        )
        self.assertEqual(result.qkv_format, "thd")
        self.assertEqual(result.cu_seqlens_q.tolist(), [0, 570, 1140])
        self.assertIs(result.cu_seqlens_q, result.cu_seqlens_kv)
        # padded stream is present and numerically distinct from valid.
        self.assertIsNotNone(result.cu_seqlens_q_padded)
        self.assertEqual(result.cu_seqlens_q_padded.tolist(), [0, 586, 1172])
        self.assertIs(result.cu_seqlens_q_padded, result.cu_seqlens_kv_padded)
        # max = padded = 586 ; total = B * valid = 2*570 = 1140
        self.assertEqual(result.max_seqlen_q, 586)
        self.assertEqual(result.total_seqlen_q, 1140)
        self.assertEqual(result.total_seqlen_kv, 1140)

    def test_cp_use_packed_sequence_forces_thd_without_padding(self):
        # B=2, seq=10, img=576, pad=0, cp=2, use_packed_sequence=True
        # branch cond: cp>1 and (0>0 or True) -> True
        # valid = padded = 586 ; both streams = arange(0, 3*586=1758, 586)
        result = get_packed_seq_params(
            tokens=_tokens(2, 10),
            img_seq_len=576,
            padding_needed=0,
            cp_size=2,
            use_packed_sequence=True,
        )
        self.assertEqual(result.qkv_format, "thd")
        self.assertEqual(result.cu_seqlens_q.tolist(), [0, 586, 1172])
        self.assertIsNotNone(result.cu_seqlens_q_padded)
        self.assertEqual(result.cu_seqlens_q_padded.tolist(), [0, 586, 1172])
        # valid and padded arange calls produce equal values here (pad=0) but
        # are separate tensor objects; distinct construction, not aliased.
        self.assertIsNot(result.cu_seqlens_q, result.cu_seqlens_q_padded)
        self.assertEqual(result.max_seqlen_q, 586)
        self.assertEqual(result.total_seqlen_q, 1172)

    def test_cp_without_padding_or_packed_stays_sbhd(self):
        # B=2, seq=10, img=576, pad=0, cp=2, packed=False
        # branch cond: cp>1 and (0>0 or False) -> False -> sbhd, no padded.
        result = get_packed_seq_params(
            tokens=_tokens(2, 10),
            img_seq_len=576,
            padding_needed=0,
            cp_size=2,
        )
        self.assertEqual(result.qkv_format, "sbhd")
        self.assertIsNone(result.cu_seqlens_q_padded)
        self.assertIsNone(result.cu_seqlens_kv_padded)
        self.assertEqual(result.cu_seqlens_q.tolist(), [0, 586, 1172])
        self.assertEqual(result.max_seqlen_q, 586)
        self.assertEqual(result.total_seqlen_q, 1172)

    def test_cp1_use_packed_sequence_stays_sbhd(self):
        # cp_size == 1 short-circuits the branch even with use_packed_sequence.
        result = get_packed_seq_params(
            tokens=_tokens(2, 10),
            img_seq_len=576,
            padding_needed=0,
            cp_size=1,
            use_packed_sequence=True,
        )
        self.assertEqual(result.qkv_format, "sbhd")
        self.assertIsNone(result.cu_seqlens_q_padded)
        self.assertEqual(result.cu_seqlens_q.tolist(), [0, 586, 1172])

    def test_cp1_with_padding_stays_sbhd_but_shifts_valid(self):
        # cp_size == 1 -> no padded stream, sbhd; padding still lowers valid.
        # B=1, seq=8, img=4, pad=2, cp=1
        # valid = 8 + 4 - 2 = 10 ; padded = 12
        # cu_seqlens = arange(0, 2*10=20, 10) = [0, 10]
        result = get_packed_seq_params(
            tokens=_tokens(1, 8),
            img_seq_len=4,
            padding_needed=2,
            cp_size=1,
        )
        self.assertEqual(result.qkv_format, "sbhd")
        self.assertIsNone(result.cu_seqlens_q_padded)
        self.assertEqual(result.cu_seqlens_q.tolist(), [0, 10])
        # max tracks the padded length (12), total tracks the valid length (10).
        self.assertEqual(result.max_seqlen_q, 12)
        self.assertEqual(result.max_seqlen_kv, 12)
        self.assertEqual(result.total_seqlen_q, 10)
        self.assertEqual(result.total_seqlen_kv, 10)


if __name__ == "__main__":
    unittest.main()
