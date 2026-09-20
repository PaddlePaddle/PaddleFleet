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

"""Behavior tests for the CSA sparse-attn pure-Python utility helpers.

Module under test: ``paddlefleet.fusions.csa_sparse_attn`` -- the
index/geometry/rounding/validation helpers that surround the "cudnn" backend,
NOT the FlashMLA / cuDNN attention kernels themselves. In the repository module
map these belong to the "计算优化 / Fused Ops" boundary.

Disjoint scope. The head-tile math ``score_target_qheads`` and the tensor
padding helper ``pad_score_target_heads`` are already covered by
``tests/single_card_tests/ops/test_score_target_head_pad.py``; this file
deliberately covers the OTHER utilities to stay disjoint:
  * ``_dsa_head_tile``      -- smallest 64/128 attention tile, ValueError above.
  * ``_dsa_latent_dim``     -- the fixed 512 latent width, ValueError above.
  * ``_pad_latent_dim``     -- zero-pad the last axis (with identity fast path).
  * ``_pad_query_heads``    -- zero query rows + -1e30 sink for the pad heads.
  * ``_real_rows``          -- row ids of the real data in a padded kernel view.
  * ``_drop_padded_rows``   -- fused gather that undoes head + latent padding.
  * ``_csa_compute_topk_length`` -- trailing (last-valid+1) per-row bound.
  * ``_csa_compact_topk_idxs``   -- order-preserving densify + exact counts.

Scope and environment. Every one of these helpers is pure Python or pure Paddle
tensor algebra that runs on CPU; no GPU kernel and no cuDNN op is invoked, so
the device is pinned to CPU and NO GPU numerics are claimed. Every expected
value below is hand-derived from the helper's own arithmetic and written as a
literal -- never read back from the function under test. Paddle is required
(even CPU-only, since most helpers build real tensors); when it cannot be
imported the whole module skips with an honest reason instead of passing.
"""

import unittest

import numpy as np

try:
    import paddle

    if not paddle.is_compiled_with_cuda():
        paddle.set_device("cpu")
    else:
        # These helpers are device-agnostic; keep them on CPU so the suite
        # neither requires a card nor claims to have validated GPU numerics.
        paddle.set_device("cpu")

    from paddlefleet.fusions.csa_sparse_attn import (
        _DSA_HEAD_TILES,
        _DSA_LATENT_DIM,
        _NEG_SINK,
        _csa_compact_topk_idxs,
        _csa_compute_topk_length,
        _drop_padded_rows,
        _dsa_head_tile,
        _dsa_latent_dim,
        _pad_latent_dim,
        _pad_query_heads,
        _real_rows,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # honest probe: only a genuinely missing dependency
    paddle = None
    _IMPORT_ERROR = exc

_SKIP_REASON = (
    f"paddle / paddlefleet.fusions.csa_sparse_attn unavailable: {_IMPORT_ERROR}"
)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestDsaHeadTile(unittest.TestCase):
    """``_dsa_head_tile``: smallest tile in (64, 128) that fits num_heads."""

    def test_selects_smallest_fitting_tile(self):
        # Hand-derived from ``for tile in (64, 128): if h <= tile: return tile``.
        # The 64/65 boundary pins the step from the 64-tile to the 128-tile.
        self.assertEqual(_DSA_HEAD_TILES, (64, 128))
        cases = {
            1: 64,
            24: 64,
            63: 64,
            64: 64,  # fits the first tile exactly
            65: 128,  # just over -> next tile
            100: 128,
            128: 128,  # fits the last tile exactly
        }
        got = {h: _dsa_head_tile(h) for h in cases}
        self.assertEqual(got, cases)

    def test_raises_above_the_largest_tile(self):
        # 129 exceeds every tile -> ValueError naming the 128 cap and the count.
        with self.assertRaisesRegex(ValueError, "at most 128 query heads"):
            _dsa_head_tile(129)
        with self.assertRaisesRegex(ValueError, "got 200"):
            _dsa_head_tile(200)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestDsaLatentDim(unittest.TestCase):
    """``_dsa_latent_dim``: always 512 up to 512, ValueError beyond."""

    def test_always_returns_512_up_to_the_cap(self):
        self.assertEqual(_DSA_LATENT_DIM, 512)
        for hn in (1, 64, 256, 511, 512):
            self.assertEqual(_dsa_latent_dim(hn), 512)

    def test_raises_above_the_cap(self):
        with self.assertRaisesRegex(ValueError, "at most 512 latent dims"):
            _dsa_latent_dim(513)
        with self.assertRaisesRegex(ValueError, "got 576"):
            _dsa_latent_dim(576)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestPadLatentDim(unittest.TestCase):
    """``_pad_latent_dim``: zero-pad the last axis up to ``latent_dim``."""

    def test_passthrough_returns_same_object_when_already_wide(self):
        # pad == 0 -> the input is handed straight back (no copy, no concat).
        x = paddle.arange(6, dtype="float32").reshape([2, 3])
        out = _pad_latent_dim(x, 3)
        self.assertIs(out, x)

    def test_appends_zeros_and_preserves_real_columns(self):
        # [[1,2,3]] padded to width 5 -> real prefix untouched, 2 zero columns.
        x = paddle.to_tensor([[1.0, 2.0, 3.0]], dtype="float32")
        out = _pad_latent_dim(x, 5)
        self.assertEqual(out.shape, [1, 5])
        self.assertEqual(out.numpy().tolist(), [[1.0, 2.0, 3.0, 0.0, 0.0]])

    def test_pads_only_the_last_axis_of_a_3d_tensor(self):
        # Leading dims are preserved; only the trailing (latent) axis grows.
        x = paddle.to_tensor(
            [[[1.0, 2.0], [3.0, 4.0]]], dtype="float32"
        )  # shape [1, 2, 2]
        out = _pad_latent_dim(x, 4)
        self.assertEqual(out.shape, [1, 2, 4])
        self.assertEqual(
            out.numpy().tolist(),
            [[[1.0, 2.0, 0.0, 0.0], [3.0, 4.0, 0.0, 0.0]]],
        )


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestPadQueryHeads(unittest.TestCase):
    """``_pad_query_heads``: zero query rows and -1e30 sink for pad heads."""

    def test_zero_query_rows_and_neg_sink_for_pad_heads(self):
        # query [1, 1, 2, 3] widened to head_tile=4: real 2 heads kept, 2 pad
        # heads appended as zeros; sink [s0, s1] -> [s0, s1, -1e30, -1e30].
        query = paddle.arange(6, dtype="float32").reshape([1, 1, 2, 3])
        sink = paddle.to_tensor([7.0, 8.0], dtype="float32")
        q_out, sink_out = _pad_query_heads(query, sink, head_tile=4)

        self.assertEqual(q_out.shape, [1, 1, 4, 3])
        self.assertEqual(
            q_out.numpy().tolist(),
            [
                [
                    [
                        [0.0, 1.0, 2.0],
                        [3.0, 4.0, 5.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ]
                ]
            ],
        )
        self.assertEqual(sink_out.shape, [4])
        self.assertEqual(sink_out.dtype, paddle.float32)
        # Real sink prefix is untouched; the two pad heads carry the -1e30
        # sentinel. Expected is hand-built in the SAME float32 the helper stores
        # (a plain -1e30 python literal would differ after the fp32 round-trip).
        expected_sink = np.array(
            [7.0, 8.0, _NEG_SINK, _NEG_SINK], dtype="float32"
        )
        np.testing.assert_array_equal(sink_out.numpy(), expected_sink)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestRealRows(unittest.TestCase):
    """``_real_rows``: row ids of real data in a padded kernel view.

    Layout is ``token * (head_tile*total) + head * total + chunk`` with the
    first ``num_heads`` heads and first ``keep`` chunks being real.
    """

    def test_row_ids_two_tokens_two_heads_single_chunk(self):
        # num_tokens=2, num_heads=2, head_tile=3, keep=1, total=2.
        # head_rows = token*6 + head*2 = [[0,2],[6,8]]; keep=1 -> flatten.
        out = _real_rows(2, 2, 3, 1, 2)
        self.assertEqual(out.numpy().tolist(), [0, 2, 6, 8])

    def test_row_ids_single_token_multi_chunk(self):
        # num_tokens=1, num_heads=2, head_tile=2, keep=2, total=3.
        # head_rows = head*3 = [0,3]; + chunk in {0,1} -> [[0,1],[3,4]].
        out = _real_rows(1, 2, 2, 2, 3)
        self.assertEqual(out.numpy().tolist(), [0, 1, 3, 4])


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestDropPaddedRows(unittest.TestCase):
    """``_drop_padded_rows``: fused gather undoing head + latent padding."""

    def test_identity_passthrough_when_nothing_padded(self):
        # num_heads == head_tile and hn == kernel_hn -> input returned as-is.
        x = paddle.arange(6, dtype="float32").reshape([1, 2, 3])
        out = _drop_padded_rows(x, num_heads=2, head_tile=2, hn=3)
        self.assertIs(out, x)

    def test_drops_padded_head_rows_only(self):
        # kernel_hn == hn == 3, head_tile 2 -> 1 real head: keep head 0, drop 1.
        x = paddle.arange(6, dtype="float32").reshape([1, 2, 3])
        out = _drop_padded_rows(x, num_heads=1, head_tile=2, hn=3)
        self.assertEqual(out.shape, [1, 1, 3])
        self.assertEqual(out.numpy().tolist(), [[[0.0, 1.0, 2.0]]])

    def test_drops_head_rows_and_latent_columns_together(self):
        # x [1, head_tile=2, kernel_hn=4] -> num_heads=1, hn=2: keep head 0's
        # first two latent columns only. head0=[0,1,2,3] -> [0,1].
        x = paddle.arange(8, dtype="float32").reshape([1, 2, 4])
        out = _drop_padded_rows(x, num_heads=1, head_tile=2, hn=2)
        self.assertEqual(out.shape, [1, 1, 2])
        self.assertEqual(out.numpy().tolist(), [[[0.0, 1.0]]])

    def test_latent_drop_across_multiple_tokens(self):
        # Two leading tokens, head_tile=2, kernel_hn=4 -> num_heads=1, hn=2.
        # token0 head0 = [0,1,2,3] -> [0,1]; token1 head0 = [8,9,10,11] -> [8,9].
        x = paddle.arange(16, dtype="float32").reshape([2, 2, 4])
        out = _drop_padded_rows(x, num_heads=1, head_tile=2, hn=2)
        self.assertEqual(out.shape, [2, 1, 2])
        self.assertEqual(out.numpy().tolist(), [[[0.0, 1.0]], [[8.0, 9.0]]])


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestCsaComputeTopkLength(unittest.TestCase):
    """``_csa_compute_topk_length``: trailing (last-valid index + 1) bound."""

    def test_trailing_bound_counts_through_interior_holes(self):
        # Rows chosen so the trailing bound differs from sum(valid): row 4 has
        # two leading -1 then valid at cols 2,3 -> bound 4 (NOT 2). Row 3 is all
        # -1 -> clamped up to 1 so the kernel still writes its dq row.
        idxs = paddle.to_tensor(
            [
                [0, 1, 2, -1],  # last valid at col 2 -> 3
                [3, 4, 5, -1],  # last valid at col 2 -> 3
                [0, 5, -1, -1],  # last valid at col 1 -> 2
                [-1, -1, -1, -1],  # none valid -> clamped to 1
                [-1, -1, 0, 5],  # interior holes; last valid at col 3 -> 4
                [2, 2, 2, 2],  # all valid -> 4
            ],
            dtype="int32",
        )
        out = _csa_compute_topk_length(idxs)
        self.assertEqual(out.shape, [6])
        self.assertEqual(out.dtype, paddle.int32)
        self.assertEqual(out.numpy().tolist(), [3, 3, 2, 1, 4, 4])

    def test_trailing_bound_exceeds_valid_count_for_leading_holes(self):
        # Distinguishes trailing-bound from sum(valid): here sum(valid)==2 but
        # the safe loop bound must be 4 because a valid entry sits at col 3.
        row = paddle.to_tensor([[-1, -1, 0, 5]], dtype="int32")
        valid_count = int((row >= 0).astype("int32").sum().item())
        self.assertEqual(valid_count, 2)
        self.assertEqual(_csa_compute_topk_length(row).numpy().tolist(), [4])


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestCsaCompactTopkIdxs(unittest.TestCase):
    """``_csa_compact_topk_idxs``: order-preserving densify + exact counts."""

    def test_order_preserving_densify_and_lengths(self):
        # Valid (>=0) entries keep their original left-to-right order in a
        # contiguous prefix; -1 holes are pushed to the trailing region.
        #   [0,-1,5,-1]     -> [0, 5, -1, -1], count 2
        #   [-1,3,-1,7]     -> [3, 7, -1, -1], count 2 (3 stays before 7)
        #   [-1,-1,-1,-1]   -> [-1,-1,-1,-1],  count 0 (empty-row fast path)
        #   [5,6,7,8]       -> [5, 6, 7, 8],   count 4 (unchanged)
        idxs = paddle.to_tensor(
            [
                [0, -1, 5, -1],
                [-1, 3, -1, 7],
                [-1, -1, -1, -1],
                [5, 6, 7, 8],
            ],
            dtype="int32",
        )
        compact, lengths = _csa_compact_topk_idxs(idxs)
        self.assertEqual(
            compact.numpy().tolist(),
            [
                [0, 5, -1, -1],
                [3, 7, -1, -1],
                [-1, -1, -1, -1],
                [5, 6, 7, 8],
            ],
        )
        self.assertEqual(lengths.dtype, paddle.int32)
        self.assertEqual(lengths.numpy().tolist(), [2, 2, 0, 4])

    def test_lengths_are_the_exact_valid_count_not_a_trailing_bound(self):
        # Interior-hole row: compaction reports the TRUE count (2), unlike the
        # trailing bound (4) that ``_csa_compute_topk_length`` would give.
        row = paddle.to_tensor([[-1, 9, -1, 4]], dtype="int32")
        compact, lengths = _csa_compact_topk_idxs(row)
        self.assertEqual(compact.numpy().tolist(), [[9, 4, -1, -1]])
        self.assertEqual(lengths.numpy().tolist(), [2])


if __name__ == "__main__":
    unittest.main()
