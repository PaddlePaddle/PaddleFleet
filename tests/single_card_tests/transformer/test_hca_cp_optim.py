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

"""Rank-local unit tests for the HCA context-parallel optimizations.

The optimizations replace whole-sequence work with per-rank work. Every such
replacement rests on a local invariant that can be checked without any
collective:

  * ``compressed_topk_idxs_triton`` row window == the same rows of the full
    table, and the ``cp_size`` windows partition it exactly
  * ``CSADocMaskMetadata.get_compress_topk_idxs`` caches on the row window too,
    so a rank cannot be handed another rank's rows
  * ``prepend_prev_window`` single-process path, and its one-hop range guard
  * a sliding-window query reaches back at most ``window_size - 1`` rows, which
    is what makes rebasing the column ids onto the short ``kv_full`` exact
  * every ``ratio``-slot group of ``cutoff_gather_indices`` is a contiguous run
    inside one document, which is what makes group-index sharding exact
"""

import unittest

import numpy as np
import paddle

from paddlefleet.transformer.cp_utils import (
    get_window_topk_idxs_cp,
    prepend_prev_window,
)
from paddlefleet.transformer.csa_attention import CSADocMaskMetadata
from paddlefleet.triton_ops.document_mask_fusion import (
    compressed_doc_start_triton,
    compressed_topk_idxs_triton,
)

# All regimes pack to seqlen 32: exact multiples of the ratio, remainders on
# either side of a group boundary, single-token documents, trailing padding.
REGIMES = [
    ([32], 0),
    ([16, 16], 0),
    ([13, 19], 0),
    ([5, 14, 3, 8], 2),
    ([1, 1, 30], 0),
    ([31], 1),
]


def _startend(doc_lens, pad):
    """``[1, 1, seqlen, 1]`` int32 exclusive ends; padding repeats the last."""
    rows, cum = [], 0
    for length in doc_lens:
        cum += length
        rows += [cum] * length
    rows += [cum] * pad
    return paddle.to_tensor(rows, dtype="int32").reshape([1, 1, len(rows), 1])


def _meta(doc_lens, pad, ratio):
    startend = _startend(doc_lens, pad)
    return CSADocMaskMetadata.build(
        ratio, 1, startend.shape[2], startend, dense_mode=False
    )


def _topk_rows(meta, ratio, offset, row_start, row_count):
    cds = compressed_doc_start_triton(
        meta.startend_row_indices.flatten(), meta.doc_start_per_pos, ratio
    )
    return compressed_topk_idxs_triton(
        cds,
        meta.pos_in_doc,
        meta.doc_len_per_pos,
        ratio,
        offset,
        row_start=row_start,
        row_count=row_count,
    )


class TestCompressedTopkIdxsRowWindow(unittest.TestCase):
    """Row windows of the compressed-topk table."""

    def test_window_equals_full_slice(self):
        for doc_lens, pad in REGIMES:
            for ratio in (4, 8):
                meta = _meta(doc_lens, pad, ratio)
                s = meta.seqlen
                full = _topk_rows(meta, ratio, 100, 0, None)
                self.assertEqual(full.shape, [1, s, s // ratio])
                windows = [(0, s), (0, 1), (s - 1, 1), (3, 5), (s // 2, s // 2)]
                for row_start, row_count in windows:
                    win = _topk_rows(meta, ratio, 100, row_start, row_count)
                    np.testing.assert_array_equal(
                        win.numpy(),
                        full[:, row_start : row_start + row_count].numpy(),
                        err_msg=(
                            f"docs={doc_lens} pad={pad} ratio={ratio} "
                            f"rows=[{row_start}, {row_start + row_count})"
                        ),
                    )

    def test_cp_shards_partition_the_full_table(self):
        for doc_lens, pad in REGIMES:
            meta = _meta(doc_lens, pad, 4)
            full = _topk_rows(meta, 4, 0, 0, None)
            for cp_size in (2, 4, 8):
                sq = meta.seqlen // cp_size
                shards = [
                    _topk_rows(meta, 4, 0, rank * sq, sq)
                    for rank in range(cp_size)
                ]
                np.testing.assert_array_equal(
                    paddle.concat(shards, axis=1).numpy(),
                    full.numpy(),
                    err_msg=f"docs={doc_lens} pad={pad} cp_size={cp_size}",
                )

    def test_row_range_guard(self):
        meta = _meta([32], 0, 4)
        for row_start, row_count in ((-1, 4), (0, 0), (30, 4), (32, 1)):
            with self.assertRaises(ValueError):
                _topk_rows(meta, 4, 0, row_start, row_count)


class TestCompressTopkIdxsCache(unittest.TestCase):
    """The metadata cache must key on the row window, not only the offset."""

    def test_cache_key_covers_offset_and_row_window(self):
        meta = _meta([13, 19], 0, 4)
        n_comp = meta.seqlen // 4
        full = meta.get_compress_topk_idxs(0).clone()

        win = meta.get_compress_topk_idxs(0, row_start=8, row_count=8)
        self.assertEqual(win.shape, [1, 8, n_comp])
        np.testing.assert_array_equal(win.numpy(), full[:, 8:16].numpy())

        # same rows, different offset: the valid slots must shift, -1 stays -1
        shifted = meta.get_compress_topk_idxs(1000, row_start=8, row_count=8)
        np.testing.assert_array_equal(
            paddle.where(win >= 0, win + 1000, win).numpy(), shifted.numpy()
        )
        # identical key hits the cache
        self.assertIs(shifted, meta.get_compress_topk_idxs(1000, 8, 8))
        # same offset, other rows: a cache hit here would hand this rank
        # another rank's queries
        other = meta.get_compress_topk_idxs(1000, 0, 8)
        np.testing.assert_array_equal(
            other.numpy(),
            paddle.where(full >= 0, full + 1000, full)[:, 0:8].numpy(),
        )


class TestPrependPrevWindowLocal(unittest.TestCase):
    """Single-process behaviour and the one-hop range guard."""

    def test_zeros_prefix_and_gradient(self):
        x = paddle.randn([2, 8, 4], dtype="float32")
        x.stop_gradient = False
        out = prepend_prev_window(x, 3, None)
        self.assertEqual(out.shape, [2, 11, 4])
        np.testing.assert_array_equal(
            out[:, :3].numpy(), np.zeros([2, 3, 4], "float32")
        )
        np.testing.assert_array_equal(out[:, 3:].numpy(), x.numpy())
        (out * 2).sum().backward()
        np.testing.assert_array_equal(
            x.grad.numpy(), np.full([2, 8, 4], 2.0, "float32")
        )

    def test_zero_window_is_identity(self):
        x = paddle.randn([1, 4, 2], dtype="float32")
        self.assertIs(prepend_prev_window(x, 0, None), x)

    def test_window_beyond_one_shard_raises(self):
        x = paddle.randn([1, 4, 2], dtype="float32")
        with self.assertRaises(ValueError):
            prepend_prev_window(x, 5, None)


class TestWindowReachAndRebase(unittest.TestCase):
    """``kv_full`` may start at ``position_offset - window_size``."""

    def _assert_reachable(self, idxs, positions, window, offset, sq):
        idxs = idxs.astype("int64")
        valid = idxs >= 0
        pos = positions.reshape([1, -1, 1])
        in_reach = (idxs >= pos - (window - 1)) & (idxs <= pos)
        self.assertTrue(bool((in_reach | ~valid).all()))

        rebased = paddle.where(valid, idxs - (offset - window), idxs)
        in_kv_full = (rebased >= 0) & (rebased < window + sq)
        self.assertTrue(bool((in_kv_full | (rebased == -1)).all()))

    def test_causal_window(self):
        sq_global, window = 64, 16
        for cp_size in (2, 4):
            sq = sq_global // cp_size
            for rank in range(cp_size):
                offset = rank * sq
                positions = paddle.arange(offset, offset + sq, dtype="int64")
                self._assert_reachable(
                    get_window_topk_idxs_cp(positions, window, 1, sq_global),
                    positions,
                    window,
                    offset,
                    sq,
                )

    def test_document_window(self):
        window, cp_size = 8, 4
        for doc_lens, pad in REGIMES:
            meta = _meta(doc_lens, pad, 4)
            full = meta.get_window_topk_idxs(window)
            sq = meta.seqlen // cp_size
            for rank in range(cp_size):
                offset = rank * sq
                positions = paddle.arange(offset, offset + sq, dtype="int64")
                self._assert_reachable(
                    full[:, offset : offset + sq],
                    positions,
                    window,
                    offset,
                    sq,
                )


class TestCutoffGroupGeometry(unittest.TestCase):
    """Group-index sharding of ``cutoff_gather_indices``."""

    def test_groups_are_contiguous_runs_inside_one_document(self):
        for doc_lens, pad in REGIMES:
            for ratio in (4, 8):
                meta = _meta(doc_lens, pad, ratio)
                idxs = meta.cutoff_gather_indices.astype("int64")
                n = meta.actual_n_compressed
                self.assertEqual(idxs.shape[0], n * ratio)
                self.assertTrue(bool((idxs[1:] > idxs[:-1]).all()))

                groups = idxs.reshape([n, ratio]).numpy()
                doc_start = meta.doc_start_per_pos.numpy()
                for group in groups:
                    np.testing.assert_array_equal(
                        group, np.arange(group[0], group[0] + ratio)
                    )
                    self.assertEqual(len(set(doc_start[group].tolist())), 1)

    def test_group_shards_reassemble_in_dense_order(self):
        ratio = 4
        for doc_lens, pad in REGIMES:
            meta = _meta(doc_lens, pad, ratio)
            idxs = meta.cutoff_gather_indices
            n = meta.actual_n_compressed
            for cp_size in (2, 4, 8):
                n_shard = (n + cp_size - 1) // cp_size
                shards = []
                for rank in range(cp_size):
                    start = rank * n_shard * ratio
                    shard = idxs[start : start + n_shard * ratio]
                    pad_len = n_shard * ratio - shard.shape[0]
                    if pad_len > 0:
                        shard = paddle.concat(
                            [
                                shard,
                                paddle.zeros([pad_len], dtype=shard.dtype),
                            ]
                        )
                    shards.append(shard)
                np.testing.assert_array_equal(
                    paddle.concat(shards)[: n * ratio].numpy(),
                    idxs.numpy(),
                    err_msg=f"docs={doc_lens} pad={pad} cp_size={cp_size}",
                )


if __name__ == "__main__":
    unittest.main()
