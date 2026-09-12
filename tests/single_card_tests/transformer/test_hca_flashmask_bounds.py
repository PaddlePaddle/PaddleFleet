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

"""``build_hca_flashmask_bounds`` describes the same mask as the HCA index table.

``csa_hca_use_flashmask`` swaps the HCA layers' explicit
``concat([window_topk_idxs, compress_topk_idxs])`` column table for a FlashMask
``[LTS, UTE]`` row-bound pair per column. The two must select exactly the same
``(query, key)`` pairs, so these tests expand both back to a boolean mask and
compare, over the document layouts that stress the compressed cutoff (exact
multiples of the ratio, remainders either side of a group boundary,
single-token documents, trailing padding) and over the CP row/column
localisation.

Pure index math -- no attention kernel, so it runs anywhere.

Run with:
    python -m pytest tests/single_card_tests/transformer/test_hca_flashmask_bounds.py
"""

import unittest

import numpy as np
import paddle

from paddlefleet.transformer.cp_utils import (
    get_compress_topk_idxs_cp,
    get_window_topk_idxs_cp,
)
from paddlefleet.transformer.csa_attention import (
    CSADocMaskMetadata,
    build_hca_flashmask_bounds,
    get_compress_topk_idxs,
    get_window_topk_idxs,
)

# ``(doc_lens, padding)`` packings, all summing to seqlen 32.
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


def _mask_from_idxs(topk_idxs, n_kv):
    """``[sq, n_kv]`` bool: column ``j`` listed on row ``i``."""
    idxs = topk_idxs[0].numpy()
    mask = np.zeros([idxs.shape[0], n_kv], dtype=bool)
    for i in range(idxs.shape[0]):
        for j in idxs[i]:
            if j >= 0:
                mask[i, j] = True
    return mask


def _mask_from_bounds(bounds, sq):
    """``[sq, n_kv]`` bool: row ``i`` kept by column ``j``'s ``[UTE, LTS)``."""
    lts = bounds[0, 0, :, 0].numpy()
    ute = bounds[0, 0, :, 1].numpy()
    rows = np.arange(sq)[:, None]
    return (rows < lts[None, :]) & (rows >= ute[None, :])


def _reference_mask(ratio, window_size, seqlen, meta, n_compressed):
    """Mask of ``concat([window_topk_idxs, compress_topk_idxs])``."""
    window = get_window_topk_idxs(window_size, 1, seqlen, docmask_meta=meta)
    n_kv = seqlen + n_compressed
    if n_compressed == 0:
        return _mask_from_idxs(window.astype("int32"), n_kv)
    compressed = get_compress_topk_idxs(
        ratio, 1, seqlen, seqlen, docmask_meta=meta
    )
    return _mask_from_idxs(
        paddle.concat(
            [window.astype("int32"), compressed.astype("int32")], axis=-1
        ),
        n_kv,
    )


class TestHCAFlashmaskBoundsNoCP(unittest.TestCase):
    """cp_size == 1: bounds match the index table the HCA branch builds."""

    def test_docmask_regimes(self):
        seqlen = 32
        for doc_lens, pad in REGIMES:
            for ratio in (4, 8, 16):
                for window_size in (1, 4, 8, 32):
                    startend = _startend(doc_lens, pad)
                    meta = CSADocMaskMetadata.build(
                        ratio, 1, seqlen, startend, dense_mode=True
                    )
                    n_compressed = seqlen // ratio
                    expected = _reference_mask(
                        ratio, window_size, seqlen, meta, n_compressed
                    )
                    bounds = build_hca_flashmask_bounds(
                        seqlen,
                        n_compressed,
                        ratio,
                        window_size,
                        seqlen_global=seqlen,
                        seqlen_local=seqlen,
                        position_offset=0,
                        batch_size=1,
                        docmask_meta=meta,
                    )
                    np.testing.assert_array_equal(
                        _mask_from_bounds(bounds, seqlen),
                        expected,
                        err_msg=(
                            f"docs={doc_lens} pad={pad} ratio={ratio} "
                            f"window={window_size}"
                        ),
                    )

    def test_causal_only(self):
        """``docmask_meta=None``: plain causal window + compressed prefix."""
        seqlen = 32
        for ratio in (4, 8, 16):
            for window_size in (1, 4, 8, 32):
                n_compressed = seqlen // ratio
                expected = _mask_from_idxs(
                    paddle.concat(
                        [
                            get_window_topk_idxs(window_size, 1, seqlen).astype(
                                "int32"
                            ),
                            get_compress_topk_idxs(
                                ratio, 1, seqlen, seqlen
                            ).astype("int32"),
                        ],
                        axis=-1,
                    ),
                    seqlen + n_compressed,
                )
                bounds = build_hca_flashmask_bounds(
                    seqlen,
                    n_compressed,
                    ratio,
                    window_size,
                    seqlen_global=seqlen,
                    seqlen_local=seqlen,
                    position_offset=0,
                    batch_size=1,
                )
                np.testing.assert_array_equal(
                    _mask_from_bounds(bounds, seqlen),
                    expected,
                    err_msg=f"ratio={ratio} window={window_size}",
                )

    def test_zero_padded_compressed_tail_is_masked(self):
        """Compressor slots past ``actual_n_compressed`` hold zeros, not KV."""
        seqlen, ratio, window_size = 32, 4, 8
        meta = CSADocMaskMetadata.build(
            ratio, 1, seqlen, _startend([5, 14, 3, 8], 2), dense_mode=True
        )
        n_compressed = seqlen // ratio
        self.assertLess(meta.actual_n_compressed, n_compressed)
        bounds = build_hca_flashmask_bounds(
            seqlen,
            n_compressed,
            ratio,
            window_size,
            seqlen_global=seqlen,
            seqlen_local=seqlen,
            position_offset=0,
            batch_size=1,
            docmask_meta=meta,
        )
        mask = _mask_from_bounds(bounds, seqlen)
        tail = mask[:, seqlen + meta.actual_n_compressed :]
        self.assertFalse(tail.any())


class TestHCAFlashmaskBoundsCP(unittest.TestCase):
    """Context parallel: rows localised, raw columns rebased onto kv_reach."""

    def _cp_reference_mask(
        self, meta, ratio, window_size, seqlen, cp_size, cp_rank, n_compressed
    ):
        """Mask of this rank's rows over ``[kv_reach | compressed_global]``."""
        sq_local = seqlen // cp_size
        position_offset = cp_rank * sq_local
        kv_base = position_offset - window_size
        n_raw = window_size + sq_local
        q_positions = paddle.arange(
            position_offset, position_offset + sq_local, dtype="int64"
        )
        if meta is None:
            window = get_window_topk_idxs_cp(
                q_positions, window_size, 1, seqlen
            )
        else:
            window = get_window_topk_idxs(
                window_size, 1, seqlen, docmask_meta=meta
            )[:, position_offset : position_offset + sq_local]
        window = paddle.where(window >= 0, window - kv_base, window)
        parts = [window.astype("int32")]
        if n_compressed > 0:
            if meta is None:
                compressed = get_compress_topk_idxs_cp(
                    q_positions, ratio, 1, n_raw, n_compressed
                )
            else:
                compressed = meta.get_compress_topk_idxs(
                    n_raw, row_start=position_offset, row_count=sq_local
                )
            parts.append(compressed.astype("int32"))
        return _mask_from_idxs(
            paddle.concat(parts, axis=-1), n_raw + n_compressed
        )

    def test_docmask_regimes(self):
        seqlen = 32
        for doc_lens, pad in REGIMES:
            for ratio in (4, 8):
                for cp_size in (2, 4):
                    sq_local = seqlen // cp_size
                    for window_size in (1, 4, sq_local):
                        meta = CSADocMaskMetadata.build(
                            ratio,
                            1,
                            seqlen,
                            _startend(doc_lens, pad),
                            dense_mode=True,
                        )
                        n_compressed = (sq_local // ratio) * cp_size
                        for cp_rank in range(cp_size):
                            expected = self._cp_reference_mask(
                                meta,
                                ratio,
                                window_size,
                                seqlen,
                                cp_size,
                                cp_rank,
                                n_compressed,
                            )
                            bounds = build_hca_flashmask_bounds(
                                window_size + sq_local,
                                n_compressed,
                                ratio,
                                window_size,
                                seqlen_global=seqlen,
                                seqlen_local=sq_local,
                                position_offset=cp_rank * sq_local,
                                batch_size=1,
                                docmask_meta=meta,
                            )
                            np.testing.assert_array_equal(
                                _mask_from_bounds(bounds, sq_local),
                                expected,
                                err_msg=(
                                    f"docs={doc_lens} pad={pad} ratio={ratio} "
                                    f"cp={cp_rank}/{cp_size} "
                                    f"window={window_size}"
                                ),
                            )

    def test_causal_only(self):
        seqlen = 32
        for ratio in (4, 8):
            for cp_size in (2, 4):
                sq_local = seqlen // cp_size
                for window_size in (1, 4, sq_local):
                    n_compressed = (sq_local // ratio) * cp_size
                    for cp_rank in range(cp_size):
                        expected = self._cp_reference_mask(
                            None,
                            ratio,
                            window_size,
                            seqlen,
                            cp_size,
                            cp_rank,
                            n_compressed,
                        )
                        bounds = build_hca_flashmask_bounds(
                            window_size + sq_local,
                            n_compressed,
                            ratio,
                            window_size,
                            seqlen_global=seqlen,
                            seqlen_local=sq_local,
                            position_offset=cp_rank * sq_local,
                            batch_size=1,
                        )
                        np.testing.assert_array_equal(
                            _mask_from_bounds(bounds, sq_local),
                            expected,
                            err_msg=(
                                f"ratio={ratio} cp={cp_rank}/{cp_size} "
                                f"window={window_size}"
                            ),
                        )

    def test_rank0_window_prefix_is_unreadable(self):
        """Rank 0's ``prepend_prev_window`` prefix is zeros, not KV."""
        seqlen, ratio, window_size, cp_size = 32, 4, 8, 2
        sq_local = seqlen // cp_size
        bounds = build_hca_flashmask_bounds(
            window_size + sq_local,
            (sq_local // ratio) * cp_size,
            ratio,
            window_size,
            seqlen_global=seqlen,
            seqlen_local=sq_local,
            position_offset=0,
            batch_size=1,
        )
        mask = _mask_from_bounds(bounds, sq_local)
        self.assertFalse(mask[:, :window_size].any())


class TestHCAFlashmaskBoundsCache(unittest.TestCase):
    """``get_hca_flashmask_bounds`` caches without changing the table."""

    def _meta(self, seqlen=32, ratio=4):
        return CSADocMaskMetadata.build(
            ratio, 1, seqlen, _startend([5, 14, 3, 8], 2), dense_mode=True
        )

    def test_matches_uncached_builder(self):
        seqlen, ratio, window_size = 32, 4, 8
        meta = self._meta(seqlen, ratio)
        geometry = (seqlen, seqlen // ratio)
        cached = meta.get_hca_flashmask_bounds(
            *geometry, window_size, seqlen, seqlen, 0, 1
        )
        direct = build_hca_flashmask_bounds(
            *geometry,
            ratio,
            window_size,
            seqlen_global=seqlen,
            seqlen_local=seqlen,
            position_offset=0,
            batch_size=1,
            docmask_meta=self._meta(seqlen, ratio),
        )
        np.testing.assert_array_equal(cached.numpy(), direct.numpy())

    def test_same_geometry_hits(self):
        seqlen, ratio, window_size = 32, 4, 8
        meta = self._meta(seqlen, ratio)
        args = (seqlen, seqlen // ratio, window_size, seqlen, seqlen, 0, 1)
        first = meta.get_hca_flashmask_bounds(*args)
        # Same tensor object, not just an equal one: the point of the cache is
        # that all HCA layers of a micro-batch share one table.
        self.assertIs(meta.get_hca_flashmask_bounds(*args), first)

    def test_changed_geometry_rebuilds(self):
        """A CP rank shift must not read the previous rank's rows."""
        seqlen, ratio, window_size, cp_size = 32, 4, 8, 2
        sq_local = seqlen // cp_size
        meta = self._meta(seqlen, ratio)
        n_raw, n_compressed = window_size + sq_local, seqlen // ratio
        rank0 = meta.get_hca_flashmask_bounds(
            n_raw, n_compressed, window_size, seqlen, sq_local, 0, 1
        )
        rank0 = rank0.clone()
        rank1 = meta.get_hca_flashmask_bounds(
            n_raw, n_compressed, window_size, seqlen, sq_local, sq_local, 1
        )
        self.assertFalse(np.array_equal(rank0.numpy(), rank1.numpy()))
        np.testing.assert_array_equal(
            rank1.numpy(),
            build_hca_flashmask_bounds(
                n_raw,
                n_compressed,
                ratio,
                window_size,
                seqlen_global=seqlen,
                seqlen_local=sq_local,
                position_offset=sq_local,
                batch_size=1,
                docmask_meta=self._meta(seqlen, ratio),
            ).numpy(),
        )

    def test_global_halves_are_shared_across_cp_ranks(self):
        """The warmed, geometry-free halves serve every rank unchanged.

        This is what makes ``DocMaskMetaRegistry._warm`` worth doing: warming
        cannot know the CP geometry, so it may only build a table that every
        rank then slices.
        """
        seqlen, ratio, window_size, cp_size = 32, 4, 8, 2
        sq_local = seqlen // cp_size
        meta = self._meta(seqlen, ratio)
        warmed = meta.get_hca_global_row_bounds(window_size)
        for cp_rank in range(cp_size):
            meta.get_hca_flashmask_bounds(
                window_size + sq_local,
                seqlen // ratio,
                window_size,
                seqlen,
                sq_local,
                cp_rank * sq_local,
                1,
            )
            after = meta.get_hca_global_row_bounds(window_size)
            self.assertIs(after[0], warmed[0])
            self.assertIs(after[1], warmed[1])

    def test_wrong_seqlen_raises(self):
        """The global halves are indexed by global row, so seqlen must match."""
        seqlen, ratio, window_size = 32, 4, 8
        meta = self._meta(seqlen, ratio)
        with self.assertRaises(ValueError):
            meta.get_hca_flashmask_bounds(
                seqlen, seqlen // ratio, window_size, 2 * seqlen, seqlen, 0, 1
            )


if __name__ == "__main__":
    unittest.main()
