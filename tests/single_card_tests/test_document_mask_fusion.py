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

"""
Unit tests for ``paddlefleet.triton_ops.document_mask_fusion``.

Every triton kernel is validated against the *original* (pre-fusion)
implementation imported from ``csa_attention`` — the fusion replaced those
functions, so they are the ground-truth reference.
"""

import unittest

import numpy as np
import paddle

from paddlefleet.transformer.csa_attention import (
    CSADocMaskMetadata,
    _build_compress_topk_idxs_from_valid_range,
    _build_window_topk_idxs_from_doc_bounds,
    compact_kv_score_cutoff,
)
from paddlefleet.transformer.kimi_delta_attention import build_cu_seqlens
from paddlefleet.triton_ops.document_mask_fusion import (
    compressed_doc_start_triton,
    compressed_topk_idxs_triton,
    cu_seqlens_triton,
    cutoff_compact_triton,
    document_mask_triton,
    window_topk_idxs_triton,
)


def _cu_seqlens_paddle_ref(startend, batch, seq_len, keep_single_segment=False):
    """Independent pure-paddle reference (the pre-fusion algorithm).

    This is the ground truth for ``cu_seqlens_triton`` and the triton-dispatched
    ``build_cu_seqlens``; it must NOT go through the triton path itself.
    """
    total = batch * seq_len
    if startend is None:
        if not keep_single_segment:
            return None
        return paddle.to_tensor([0, total], dtype="int64")
    ends = startend[:, 0, :, 0].astype("int64")
    doc_edges = ends[:, 1:] != ends[:, :-1]
    head = paddle.ones([batch, 1], dtype="bool")
    is_start = paddle.concat([head, doc_edges], axis=1) if seq_len > 1 else head
    starts = paddle.nonzero(is_start.reshape([-1])).flatten()
    cu = paddle.concat([starts, paddle.full([1], total, dtype=starts.dtype)])
    if cu.shape[0] <= 2 and not keep_single_segment:
        return None
    return cu


class TestDocumentMaskFusion(unittest.TestCase):
    def setUp(self):
        self.ratio = 4
        self.window_size = 4
        self.batch_size = 1
        doc_lens = [5, 14, 3, 8]
        pad = 2

        startend_row_indices = []
        cum = 0
        for length in doc_lens:
            cum += length
            startend_row_indices += [cum] * length
        # padding continues the last doc's end value
        startend_row_indices += [cum] * pad

        self.seqlen = sum(doc_lens) + pad  # 32
        self.startend_row_indices = paddle.to_tensor(
            startend_row_indices, dtype="int32"
        ).reshape([1, 1, self.seqlen, 1])

        # --- reference doc boundaries (original implementation) ---
        self.meta_ref = CSADocMaskMetadata.build(
            self.ratio,
            self.batch_size,
            self.seqlen,
            self.startend_row_indices,
            dense_mode=False,
        )
        positions = paddle.arange(self.seqlen, dtype="int64")
        self.pos_in_doc_ref = positions - self.meta_ref.doc_start_per_pos

    def test_document_mask_kernel(self):
        """document_mask_fwd_kernel -> (doc_start, doc_len, pos_in_doc)."""
        doc_start, doc_len, pos_in_doc = document_mask_triton(
            self.startend_row_indices.flatten()
        )
        np.testing.assert_array_equal(
            doc_start, self.meta_ref.doc_start_per_pos
        )
        np.testing.assert_array_equal(doc_len, self.meta_ref.doc_len_per_pos)
        np.testing.assert_array_equal(pos_in_doc, self.pos_in_doc_ref)

    def test_cutoff_compact_kernel(self):
        """cutoff_compact_kernel -> gather idx / pos / n / is_first / comp_pos."""
        ratio = self.ratio
        seqlen = self.seqlen
        total_cutoff = self.meta_ref.doc_lens_cutoff.sum().item()
        actual_n_compressed = total_cutoff // ratio

        # ---- reference from original functions ----
        identity = paddle.arange(seqlen, dtype="int64").reshape([1, seqlen, 1])
        src_idx, _ = compact_kv_score_cutoff(
            self.meta_ref.doc_starts,
            self.meta_ref.doc_lens_cutoff,
            self.meta_ref.doc_starts_cutoff,
            total_cutoff,
            identity,
            identity,
        )
        gather_idx_ref = src_idx.flatten()  # [total_cutoff]
        cutoff_pos_ref = paddle.gather(self.pos_in_doc_ref, gather_idx_ref)

        # ---- kernel output ----
        (
            gather_idx,
            cutoff_pos,
            n_cutoff,
            is_first,
            compressed_pos,
        ) = cutoff_compact_triton(
            self.pos_in_doc_ref, self.meta_ref.doc_len_per_pos, ratio
        )

        self.assertEqual(n_cutoff.item(), total_cutoff)
        np.testing.assert_array_equal(gather_idx[:n_cutoff], gather_idx_ref)
        np.testing.assert_array_equal(cutoff_pos[:n_cutoff], cutoff_pos_ref)
        np.testing.assert_array_equal(
            is_first[:actual_n_compressed],
            self.meta_ref.get_is_first_compressed_group(),
        )

        # compressed_pos_in_doc has no original reference
        compressed_pos_ref = paddle.concat(
            [
                paddle.arange(0, doc_len, ratio)
                for doc_len in self.meta_ref.doc_lens_cutoff
            ],
        )
        np.testing.assert_array_equal(
            compressed_pos[:actual_n_compressed], compressed_pos_ref
        )

    @staticmethod
    def _rand_mask(batch, seq_len, avg_doc, seed):
        """[b,1,s,1] int32 mask of per-row-relative exclusive document ends."""
        rng = np.random.default_rng(seed)
        ends = np.empty((batch, seq_len), dtype=np.int64)
        for r in range(batch):
            col, cum = [], 0
            while len(col) < seq_len:
                length = int(rng.integers(max(1, avg_doc // 2), avg_doc * 2))
                cum += length
                col += [cum] * length
            ends[r] = np.array(col[:seq_len], dtype=np.int64)
        return paddle.to_tensor(
            ends.reshape(batch, 1, seq_len, 1), dtype="int32"
        )

    def _check_cu_seqlens(self, startend, batch, seq_len):
        flat = startend[:, 0, :, 0].flatten()
        for keep in (False, True):
            ref = _cu_seqlens_paddle_ref(startend, batch, seq_len, keep)
            out = cu_seqlens_triton(flat, seq_len, keep_single_segment=keep)
            msg = f"{batch}x{seq_len} keep={keep}"
            if ref is None:
                self.assertIsNone(out, msg)
            else:
                np.testing.assert_array_equal(out, ref, msg)

    def test_cu_seqlens_kernel(self):
        """cu_seqlens_fwd_kernel -> packed cu_seqlens for [b, s] flattening.

        Validated against an independent pure-paddle reference. Covers the
        row-seam corner (adjacent rows sharing an end value), seq_len == 1,
        single-segment with/without keep_single_segment, and multi-block
        streaming (total > BLOCK_N == 4096, exercising the cross-block
        running-count carry and the previous-position load at block edges).
        """
        fixed = [
            (1, 8, [[3, 3, 3, 5, 5, 8, 8, 8]]),
            # two rows both ending at s -> seam must stay a boundary
            (2, 4, [[4, 4, 4, 4], [4, 4, 4, 4]]),
            (2, 5, [[2, 2, 5, 5, 5], [3, 3, 3, 5, 5]]),
            # single global segment -> None unless keep_single_segment
            (1, 6, [[6, 6, 6, 6, 6, 6]]),
            # seq_len == 1: every position is a start
            (1, 1, [[1]]),
            (3, 1, [[1], [1], [1]]),
        ]
        for batch, seq_len, ends in fixed:
            startend = paddle.to_tensor(ends, dtype="int32").reshape(
                [batch, 1, seq_len, 1]
            )
            self._check_cu_seqlens(startend, batch, seq_len)

        # multi-block streaming (total > 4096) with real boundaries, plus a
        # large single document (single segment spanning >1 block).
        for batch, seq_len, avg in [(1, 5000, 512), (2, 8192, 700)]:
            mask = self._rand_mask(batch, seq_len, avg, seed=batch * seq_len)
            self._check_cu_seqlens(mask, batch, seq_len)
        big_single = paddle.full([1, 1, 8192, 1], 8192, dtype="int32")
        self._check_cu_seqlens(big_single, 1, 8192)

    def test_build_cu_seqlens_dispatch(self):
        """Public build_cu_seqlens (triton dispatch) matches the paddle ref.

        Covers the integration code in kimi_delta_attention: None input,
        keep_single_segment, value correctness through the dispatch, and the
        [b,1,s,1] / head-dim==1 shape validation.
        """
        # None input is handled before dispatch.
        self.assertIsNone(build_cu_seqlens(None, 2, 8))
        np.testing.assert_array_equal(
            build_cu_seqlens(None, 2, 8, keep_single_segment=True), [0, 16]
        )
        # value cases through the public API vs the independent reference.
        for batch, seq_len, ends in [
            (2, 4, [[4, 4, 4, 4], [4, 4, 4, 4]]),
            (1, 6, [[2, 2, 5, 5, 6, 6]]),
            (1, 6, [[6, 6, 6, 6, 6, 6]]),
        ]:
            startend = paddle.to_tensor(ends, dtype="int32").reshape(
                [batch, 1, seq_len, 1]
            )
            for keep in (False, True):
                ref = _cu_seqlens_paddle_ref(startend, batch, seq_len, keep)
                out = build_cu_seqlens(
                    startend, batch, seq_len, keep_single_segment=keep
                )
                if ref is None:
                    self.assertIsNone(out)
                else:
                    np.testing.assert_array_equal(out, ref)
        # head dim must be 1 (a linear recurrence is single-segment layout).
        with self.assertRaises(ValueError):
            build_cu_seqlens(paddle.ones([2, 2, 4, 1], dtype="int32"), 2, 4)

    def test_build_cu_seqlens_cpu_fallback(self):
        """On a CPU device / CPU tensor build_cu_seqlens must NOT launch the
        CUDA Triton kernel; it has to take the pure-paddle fallback.

        Guards the dispatch condition: a CUDA-compiled Paddle keeps
        ``is_compiled_with_cuda()`` True even on ``set_device('cpu')``, so the
        gate has to consult the tensor's place, not just torch-compat. Without
        that check this call would try to run a GPU kernel on a CPU tensor and
        crash.
        """
        prev = paddle.device.get_device()
        try:
            paddle.device.set_device("cpu")
            startend = paddle.to_tensor(
                [[2, 2, 5, 5, 6, 6]], dtype="int32"
            ).reshape([1, 1, 6, 1])
            self.assertTrue(startend.place.is_cpu_place())
            out = build_cu_seqlens(startend, 1, 6)
            # result stays on CPU -> the paddle path ran, not the GPU kernel.
            self.assertTrue(out.place.is_cpu_place())
            ref = _cu_seqlens_paddle_ref(startend, 1, 6)
            np.testing.assert_array_equal(out, ref)
        finally:
            paddle.device.set_device(prev)

    def test_build_cu_seqlens_triton_branch(self):
        """Force build_cu_seqlens through the fused Triton branch.

        The other build_cu_seqlens tests build tensors on the *default* device,
        so on a CPU test run the ``is_gpu_place()`` guard short-circuits and the
        triton dispatch branch is never executed. This test puts the mask
        explicitly on a GPU place so the branch is covered; it is skipped when
        Triton is unavailable.
        """
        from paddlefleet.triton_ops.utils import is_triton_available

        if not is_triton_available():
            self.skipTest("Triton not available")
        startend = paddle.to_tensor(
            [[2, 2, 5, 5, 6, 6]], dtype="int32", place=paddle.CUDAPlace(0)
        ).reshape([1, 1, 6, 1])
        self.assertTrue(startend.place.is_gpu_place())
        # is_triton_available() and is_gpu_place() -> the triton branch runs.
        out = build_cu_seqlens(startend, 1, 6)
        ref = _cu_seqlens_paddle_ref(startend, 1, 6)
        np.testing.assert_array_equal(out, ref)

    def test_window_topk_idxs_kernel(self):
        """window_topk_idxs_kernel -> [1, seqlen, window_size]."""
        window_ref = _build_window_topk_idxs_from_doc_bounds(
            self.batch_size,
            self.seqlen,
            self.window_size,
            self.meta_ref.doc_start_per_pos,
            self.meta_ref.is_valid,
        )
        window_out = window_topk_idxs_triton(
            self.meta_ref.doc_start_per_pos,
            self.meta_ref.doc_len_per_pos,
            self.window_size,
        )
        np.testing.assert_array_equal(window_out, window_ref)

    def test_compressed_topk_idxs_kernel(self):
        """compressed_topk_idxs_kernel -> [1, seqlen, seqlen // ratio]."""
        n_compressed = self.seqlen // self.ratio
        valid_range = self.meta_ref.valid_range

        compressed_doc_start = compressed_doc_start_triton(
            self.startend_row_indices.flatten(),
            self.meta_ref.doc_start_per_pos,
            self.ratio,
        )

        for offset in [0, 1000]:
            ref = _build_compress_topk_idxs_from_valid_range(
                self.batch_size, self.seqlen, n_compressed, offset, valid_range
            )
            out = compressed_topk_idxs_triton(
                compressed_doc_start,
                self.pos_in_doc_ref,
                self.meta_ref.doc_len_per_pos,
                self.ratio,
                offset,
            )
            np.testing.assert_array_equal(out, ref)


if __name__ == "__main__":
    unittest.main()
