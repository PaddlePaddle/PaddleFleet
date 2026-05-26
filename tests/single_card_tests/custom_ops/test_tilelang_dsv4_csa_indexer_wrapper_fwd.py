# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Forward correctness tests for the V4 CSA Indexer TileLang kernel
(``tilelang_csa_compressed_indexer_topk_paddle``).

The TileLang kernel selects, per query position ``t``, the top-k compressed
block ids whose causal validity is ``t < (t + 1) // ratio``. This file
verifies the kernel against the canonical Paddle reference path that DSv4
uses elsewhere — ``fused_qk_topk_naive`` from
``paddlefleet.transformer.dsa_attention`` — for the compressed-indexer
shape contract that ``CSAIndexer.forward_before_topk`` produces (i.e.
``q [b, sq, h_i, d_i]``, ``k [b, sk, d_i]``, ``weights [b, sq, h_i]``).

Import-order requirement
------------------------
``paddle.enable_compat(scope={'tilelang'})`` must run BEFORE any module
that transitively does ``import torch`` inside tilelang, otherwise the
kernel allocates real Torch output buffers and the strict
``tilelang_csa_compressed_indexer_topk_paddle`` wrapper rejects them.

These tests require CUDA + the TileLang stack and skip otherwise.
"""

import os
import sys
import unittest

# Ensure the local PaddleFleet source is loaded instead of any stale install.
_LOCAL_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")
)
if _LOCAL_SRC not in sys.path:
    sys.path.insert(0, _LOCAL_SRC)
for _m in [
    m for m in list(sys.modules)
    if m == "paddlefleet" or m.startswith("paddlefleet.")
]:
    _mod = sys.modules.get(_m)
    _f = getattr(_mod, "__file__", "") or ""
    if _LOCAL_SRC not in _f:
        sys.modules.pop(_m, None)

# CRITICAL ORDER: paddle -> enable_compat({'tilelang'}) -> tilelang.
import paddle
paddle.enable_compat(scope={"tilelang"}, silent=True)


def _cuda_or_skip(testcase):
    if not paddle.device.is_compiled_with_cuda():
        testcase.skipTest("CUDA build of Paddle is required")
    if paddle.device.cuda.device_count() == 0:
        testcase.skipTest("No CUDA device available")


def _paddle_ref_csa_indexer_topk(q, k, weights, ratio, topk_effective):
    """Paddle reference matching the TileLang kernel contract.

    Built on top of ``fused_qk_topk_naive`` (8.3): we feed it a causal
    mask that marks positions with no compressed block as ``-inf``, and
    request ``topk = topk_effective``. We then post-process the selected
    rows to reproduce the kernel's invalid-slot ``-1``/``0`` convention
    and softmax-over-selected-set output.
    """
    from paddlefleet.transformer.dsa_attention import fused_qk_topk_naive

    b, sq, h_i, d_i = q.shape
    sk = k.shape[1]
    sm_scale = d_i ** -0.5

    comp_ids = paddle.arange(sk, dtype="int64").reshape([1, 1, sk])
    valid_end = (
        paddle.arange(1, sq + 1, dtype="int64").reshape([1, sq, 1]) // ratio
    )
    valid_mask = (comp_ids < valid_end).expand([b, sq, sk])

    neg_inf = paddle.full([b, sq, sk], float("-inf"), dtype="float32")
    causal_mask = paddle.where(
        valid_mask, paddle.zeros_like(neg_inf), neg_inf
    )

    actual_topk = min(int(topk_effective), int(sk))
    index_scores, ref_topk_indices = fused_qk_topk_naive(
        q, k, weights, index_topk=actual_topk, mask=causal_mask
    )
    index_scores_scaled = index_scores * sm_scale

    masked_scaled = index_scores_scaled + causal_mask
    topk_scores_raw, topk_indices = paddle.topk(
        masked_scaled, k=actual_topk, axis=-1
    )
    topk_indices = paddle.clip(topk_indices, min=0, max=sk - 1)

    valid_topk = paddle.take_along_axis(
        valid_mask.cast("int32"), topk_indices, axis=-1
    ).cast("bool")
    topk_indices = paddle.where(
        valid_topk,
        topk_indices.cast("int32"),
        paddle.full_like(topk_indices, -1, dtype="int32"),
    )
    topk_scores_raw = paddle.where(
        valid_topk,
        topk_scores_raw,
        paddle.full_like(topk_scores_raw, float("-inf")),
    )

    row_has_valid = valid_topk.any(axis=-1, keepdim=True)
    safe_scores = paddle.where(
        row_has_valid,
        topk_scores_raw,
        paddle.zeros_like(topk_scores_raw),
    )
    topk_probs = paddle.nn.functional.softmax(
        safe_scores.cast("float32"), axis=-1
    )
    topk_probs = paddle.where(
        row_has_valid, topk_probs, paddle.zeros_like(topk_probs)
    )
    topk_probs = paddle.where(
        valid_topk, topk_probs, paddle.zeros_like(topk_probs)
    )

    if int(topk_effective) > actual_topk:
        pad = int(topk_effective) - actual_topk
        topk_indices = paddle.concat(
            [
                topk_indices,
                paddle.full([b, sq, pad], -1, dtype="int32"),
            ],
            axis=-1,
        )
        topk_probs = paddle.concat(
            [
                topk_probs,
                paddle.zeros([b, sq, pad], dtype="float32"),
            ],
            axis=-1,
        )

    return topk_indices, topk_probs


def _make_inputs(b, sq, sk, h_i, d_i, dtype="bfloat16", seed=2026):
    """Return ``(q, k, weights)`` matching the
    ``CSAIndexer.forward_before_topk`` output contract."""
    paddle.seed(seed)
    q = paddle.randn([b, sq, h_i, d_i]).astype(dtype)
    k = paddle.randn([b, sk, d_i]).astype(dtype)
    weights = paddle.randn([b, sq, h_i]).astype("float32")
    return q, k, weights


def _all_equal(tensor, value):
    """Return Python bool: whether all elements of ``tensor`` equal ``value``."""
    return bool((tensor == value).all().item())


def _sorted_compare_indices(out_indices, ref_indices):
    """Compare per-row index sets ignoring intra-row ordering (ties may
    flip between the kernel and the Paddle reference)."""
    out_sorted = paddle.sort(out_indices, axis=-1)
    ref_sorted = paddle.sort(ref_indices, axis=-1)
    return bool((out_sorted == ref_sorted).all().item())


@unittest.skipUnless(
    paddle.device.is_compiled_with_cuda()
    and paddle.device.cuda.device_count() > 0,
    "TileLang CSA Indexer kernel requires CUDA",
)
class TestTileLangDSV4CSAIndexerForward(unittest.TestCase):
    """Covers Tasks 8.3 / 8.4 / 8.5 / 8.6."""

    def setUp(self):
        from paddlefleet.tilelang_ops import (
            tilelang_csa_compressed_indexer_topk_paddle,
        )

        self._kernel = tilelang_csa_compressed_indexer_topk_paddle

    # -- 8.3 / 8.5 ------------------------------------------------------

    def test_phase3_selected_topk_matches_reference(self):
        """``topk_effective = dsa_indexer_topk`` (Phase 3 selected-topk)."""
        b, sq, sk, h_i, d_i, ratio = 1, 16, 4, 64, 128, 4
        topk_effective = 2
        q, k, w = _make_inputs(b, sq, sk, h_i, d_i)
        out_idx, out_prob = self._kernel(
            q, k, w, ratio=ratio, topk_effective=topk_effective
        )
        ref_idx, ref_prob = _paddle_ref_csa_indexer_topk(
            q, k, w, ratio, topk_effective
        )
        self.assertEqual(list(out_idx.shape), [b, sq, topk_effective])
        self.assertTrue(_sorted_compare_indices(out_idx, ref_idx))
        valid = ref_idx >= 0
        out_valid = paddle.masked_select(out_prob, valid)
        ref_valid = paddle.masked_select(ref_prob, valid)
        self.assertTrue(
            paddle.allclose(out_valid, ref_valid, rtol=8e-2, atol=3e-2).item()
        )

    def test_phase2_full_candidate_matches_reference(self):
        """``topk_effective = n_compressed`` (Phase 2 full range)."""
        b, sq, sk, h_i, d_i, ratio = 1, 16, 4, 64, 128, 4
        topk_effective = sk
        q, k, w = _make_inputs(b, sq, sk, h_i, d_i)
        out_idx, out_prob = self._kernel(
            q, k, w, ratio=ratio, topk_effective=topk_effective
        )
        ref_idx, ref_prob = _paddle_ref_csa_indexer_topk(
            q, k, w, ratio, topk_effective
        )
        self.assertTrue(_sorted_compare_indices(out_idx, ref_idx))
        valid = ref_idx >= 0
        out_valid = paddle.masked_select(out_prob, valid)
        ref_valid = paddle.masked_select(ref_prob, valid)
        self.assertTrue(
            paddle.allclose(out_valid, ref_valid, rtol=8e-2, atol=3e-2).item()
        )

    def test_padded_n_compressed_matches_reference(self):
        """``topk_effective = padded_n_compressed`` (8.5/8.6: topk > S_comp).

        When the requested topk exceeds the available compressed range,
        the kernel must pad with ``-1`` indices and ``0`` probabilities.
        """
        b, sq, sk, h_i, d_i, ratio = 1, 16, 4, 64, 128, 4
        topk_effective = 6
        q, k, w = _make_inputs(b, sq, sk, h_i, d_i)
        out_idx, out_prob = self._kernel(
            q, k, w, ratio=ratio, topk_effective=topk_effective
        )
        ref_idx, ref_prob = _paddle_ref_csa_indexer_topk(
            q, k, w, ratio, topk_effective
        )
        self.assertEqual(list(out_idx.shape), [b, sq, topk_effective])
        self.assertTrue(_all_equal(out_idx[:, :, sk:], -1))
        self.assertTrue(_all_equal(out_prob[:, :, sk:], 0))
        self.assertTrue(_sorted_compare_indices(out_idx, ref_idx))

    # -- 8.4 ------------------------------------------------------------

    def test_causal_t0_t1_t2_have_no_compressed_block(self):
        """For ratio=4, query positions 0,1,2 have ``valid_end=0``."""
        b, sq, sk, h_i, d_i, ratio = 1, 16, 4, 64, 128, 4
        topk_effective = 4
        q, k, w = _make_inputs(b, sq, sk, h_i, d_i)
        out_idx, out_prob = self._kernel(
            q, k, w, ratio=ratio, topk_effective=topk_effective
        )
        self.assertTrue(_all_equal(out_idx[:, :3, :], -1))
        self.assertTrue(_all_equal(out_prob[:, :3, :], 0))

    def test_causal_t3_only_block_zero_visible(self):
        """At ``t=3``: ``valid_end = 4 // 4 = 1``, only block 0 is valid."""
        b, sq, sk, h_i, d_i, ratio = 1, 16, 4, 64, 128, 4
        topk_effective = 4
        q, k, w = _make_inputs(b, sq, sk, h_i, d_i)
        out_idx, _ = self._kernel(
            q, k, w, ratio=ratio, topk_effective=topk_effective
        )
        row = out_idx[0, 3].numpy().tolist()
        self.assertIn(0, row)
        self.assertEqual(sum(int(x == -1) for x in row), topk_effective - 1)

    def test_causal_t7_blocks_zero_and_one_visible(self):
        """At ``t=7``: ``valid_end = 8 // 4 = 2``, blocks 0,1 are valid."""
        b, sq, sk, h_i, d_i, ratio = 1, 16, 4, 64, 128, 4
        topk_effective = 4
        q, k, w = _make_inputs(b, sq, sk, h_i, d_i)
        out_idx, _ = self._kernel(
            q, k, w, ratio=ratio, topk_effective=topk_effective
        )
        row = sorted(out_idx[0, 7].numpy().tolist())
        valid = [x for x in row if x != -1]
        self.assertEqual(sorted(valid), [0, 1])

    # -- 8.6 ------------------------------------------------------------

    def test_short_sequence_with_valid_end_less_than_topk(self):
        """``valid_end < topk_effective`` forces extra padding to ``-1``.

        Here ``sq=8, ratio=4`` so ``valid_end`` ramps 0,0,0,1,1,1,1,2 —
        even though ``topk_effective=4``, no row can produce 4 valid
        compressed ids until ``valid_end >= 4`` (which never happens).
        """
        b, sq, sk, h_i, d_i, ratio = 1, 8, 4, 64, 128, 4
        topk_effective = 4
        q, k, w = _make_inputs(b, sq, sk, h_i, d_i)
        out_idx, out_prob = self._kernel(
            q, k, w, ratio=ratio, topk_effective=topk_effective
        )
        ref_idx, ref_prob = _paddle_ref_csa_indexer_topk(
            q, k, w, ratio, topk_effective
        )
        for t in range(sq):
            valid_end = (t + 1) // ratio
            row = out_idx[0, t].numpy().tolist()
            n_valid = sum(int(x >= 0) for x in row)
            self.assertEqual(
                n_valid,
                min(valid_end, topk_effective),
                msg=f"row t={t}: expected {min(valid_end, topk_effective)} valid, got {n_valid} (row={row})",
            )
        self.assertTrue(_sorted_compare_indices(out_idx, ref_idx))


if __name__ == "__main__":
    unittest.main()
