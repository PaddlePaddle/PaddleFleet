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

"""Unit tests for cuDNN DSA indexer via DLPack bridge."""

import unittest

import paddle


# =========================================================================
# Helpers
# =========================================================================


def _cuda_or_skip(testcase):
    if not paddle.device.is_compiled_with_cuda():
        testcase.skipTest("CUDA build of Paddle is required")
    if paddle.device.cuda.device_count() == 0:
        testcase.skipTest("No CUDA device available")


def _make_indexer_inputs(b, sq, sk, h_i, d_i, dtype="bfloat16", seed=2026):
    paddle.seed(seed)
    q = paddle.randn([b, sq, h_i, d_i]).astype(dtype)
    k = paddle.randn([b, sk, d_i]).astype(dtype)
    w = paddle.randn([b, sq, h_i]).astype(dtype)
    return q, k, w


def _all_equal(tensor, value):
    return bool((tensor == value).all().item())


def _sorted_compare_indices(out_indices, ref_indices):
    out_sorted = paddle.sort(out_indices, axis=-1)
    ref_sorted = paddle.sort(ref_indices, axis=-1)
    return bool((out_sorted == ref_sorted).all().item())


def _build_causal_mask(batch, seq_len, seq_len_comp, ratio):
    comp_ids = paddle.arange(seq_len_comp, dtype="int64").reshape(
        [1, 1, seq_len_comp]
    )
    valid_end = (
        paddle.arange(1, seq_len + 1, dtype="int64").reshape(
            [1, seq_len, 1]
        )
        // int(ratio)
    )
    valid = (comp_ids < valid_end).expand([batch, seq_len, seq_len_comp])
    neg_inf = paddle.full(
        [batch, seq_len, seq_len_comp], float("-inf"), dtype="float32"
    )
    return paddle.where(valid, paddle.zeros_like(neg_inf), neg_inf)


def _paddle_indexer_scores_and_topk(index_q, index_k_comp, weights, ratio, topk):
    from paddlefleet.transformer.dsa_attention import fused_qk_topk_naive

    scores, indices = fused_qk_topk_naive(
        index_q,
        index_k_comp,
        weights.cast("float32"),
        index_topk=min(int(topk), int(index_k_comp.shape[1])),
        mask=_build_causal_mask(
            int(index_q.shape[0]),
            int(index_q.shape[1]),
            int(index_k_comp.shape[1]),
            ratio,
        ),
    )
    if indices.shape[-1] < int(topk):
        padding = paddle.full(
            [
                int(index_q.shape[0]),
                int(index_q.shape[1]),
                int(topk) - int(indices.shape[-1]),
            ],
            -1,
            dtype=indices.dtype,
        )
        indices = paddle.concat([indices, padding], axis=-1)
    return scores, indices.cast("int32")


# =========================================================================
# Test cases
# =========================================================================


class TestCudnnIndexerForward(unittest.TestCase):
    """Tests for cudnn_indexer_forward (score computation)."""

    def setUp(self):
        _cuda_or_skip(self)
        paddle.set_device("gpu:0")
        from paddlefleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn import (
            cudnn_indexer_forward,
        )

        self.cudnn_indexer_forward = cudnn_indexer_forward

    def test_output_shape_and_dtype(self):
        B, S_q, H_i, D_i, ratio = 2, 128, 64, 128, 4
        S_k = S_q // ratio
        q, k, w = _make_indexer_inputs(B, S_q, S_k, H_i, D_i)
        scores = self.cudnn_indexer_forward(q, k, w, ratio=ratio)
        self.assertEqual(list(scores.shape), [B, S_q, S_k])
        self.assertEqual(scores.dtype, paddle.float32)

    def test_masked_positions_are_neginf(self):
        B, S_q, H_i, D_i, ratio = 1, 32, 64, 128, 4
        S_k = S_q // ratio
        q, k, w = _make_indexer_inputs(B, S_q, S_k, H_i, D_i)
        scores = self.cudnn_indexer_forward(q, k, w, ratio=ratio)
        # Position 0: (0+1)//4 = 0 valid KV -> all scores should be -inf
        row0 = scores[0, 0, :].numpy()
        self.assertTrue(
            all(v == float("-inf") for v in row0),
            f"Position 0 should be all -inf, got {row0}",
        )

    def test_scores_match_paddle_reference(self):
        B, S_q, H_i, D_i, ratio = 1, 64, 64, 128, 4
        S_k = S_q // ratio
        q, k, w = _make_indexer_inputs(B, S_q, S_k, H_i, D_i, seed=2028)
        scores = self.cudnn_indexer_forward(q, k, w, ratio=ratio)
        ref_scores, _ = _paddle_indexer_scores_and_topk(
            q, k, w, ratio=ratio, topk=S_k
        )
        valid = paddle.isfinite(ref_scores)
        max_abs_diff = (scores[valid] - ref_scores[valid]).abs().max().item()
        self.assertTrue(
            paddle.allclose(scores[valid], ref_scores[valid], rtol=1e-2, atol=1e-2).item(),
            f"cuDNN indexer scores mismatch with Paddle reference, max_abs_diff={max_abs_diff}",
        )


class TestCudnnIndexerTopkFwd(unittest.TestCase):
    """Tests for cudnn_indexer_topk_fwd (combined score + top-K)."""

    def setUp(self):
        _cuda_or_skip(self)
        paddle.set_device("gpu:0")
        from paddlefleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn import (
            cudnn_indexer_topk_fwd,
        )

        self.cudnn_indexer_topk_fwd = cudnn_indexer_topk_fwd

    def test_output_shape_and_dtype(self):
        B, S_q, H_i, D_i, ratio, topk = 2, 128, 64, 128, 4, 8
        S_k = S_q // ratio
        q, k, w = _make_indexer_inputs(B, S_q, S_k, H_i, D_i)
        indices, lengths = self.cudnn_indexer_topk_fwd(
            q, k, w, ratio=ratio, topk_effective=topk
        )
        self.assertEqual(list(indices.shape), [B, S_q, topk])
        self.assertEqual(indices.dtype, paddle.int32)
        self.assertEqual(list(lengths.shape), [B, S_q])
        self.assertEqual(lengths.dtype, paddle.int32)

    def test_early_positions_all_invalid(self):
        """Positions 0..ratio-2 have (s+1)//ratio==0 -> all indices -1."""
        B, S_q, H_i, D_i, ratio, topk = 1, 64, 64, 128, 4, 8
        S_k = S_q // ratio
        q, k, w = _make_indexer_inputs(B, S_q, S_k, H_i, D_i)
        indices, _ = self.cudnn_indexer_topk_fwd(
            q, k, w, ratio=ratio, topk_effective=topk
        )
        for s in range(ratio - 1):
            self.assertTrue(
                _all_equal(indices[0, s, :], -1),
                f"Position {s} should be all -1",
            )

    def test_valid_indices_in_causal_range(self):
        """All non-negative indices satisfy idx < (s+1)//ratio."""
        B, S_q, H_i, D_i, ratio, topk = 2, 128, 64, 128, 4, 8
        S_k = S_q // ratio
        q, k, w = _make_indexer_inputs(B, S_q, S_k, H_i, D_i)
        indices, _ = self.cudnn_indexer_topk_fwd(
            q, k, w, ratio=ratio, topk_effective=topk
        )
        for s in range(S_q):
            max_valid = (s + 1) // ratio
            row = indices[0, s, :].numpy()
            for idx in row:
                if idx >= 0:
                    self.assertLess(
                        idx, max_valid,
                        f"pos {s}: index {idx} >= max_valid {max_valid}",
                    )

    def test_topk_length_matches_valid_count(self):
        B, S_q, H_i, D_i, ratio, topk = 1, 64, 32, 128, 4, 4
        S_k = S_q // ratio
        q, k, w = _make_indexer_inputs(B, S_q, S_k, H_i, D_i)
        indices, lengths = self.cudnn_indexer_topk_fwd(
            q, k, w, ratio=ratio, topk_effective=topk
        )
        expected = (indices >= 0).sum(axis=-1).cast("int32")
        self.assertTrue(
            (lengths == expected).all().item(),
            "topk_length mismatch with actual valid count",
        )

    def test_h32_support(self):
        """Verify H_i=32 (qhead_per_kv_head=32) also works."""
        B, S_q, H_i, D_i, ratio, topk = 1, 64, 32, 128, 4, 4
        S_k = S_q // ratio
        q, k, w = _make_indexer_inputs(B, S_q, S_k, H_i, D_i)
        indices, _ = self.cudnn_indexer_topk_fwd(
            q, k, w, ratio=ratio, topk_effective=topk
        )
        self.assertEqual(list(indices.shape), [B, S_q, topk])

    def test_index_sets_match_paddle_reference(self):
        B, S_q, H_i, D_i, ratio, topk = 1, 64, 64, 128, 4, 8
        S_k = S_q // ratio
        q, k, w = _make_indexer_inputs(B, S_q, S_k, H_i, D_i, seed=2029)
        indices, _ = self.cudnn_indexer_topk_fwd(
            q, k, w, ratio=ratio, topk_effective=topk
        )
        _, ref_indices = _paddle_indexer_scores_and_topk(
            q, k, w, ratio=ratio, topk=topk
        )
        self.assertTrue(
            _sorted_compare_indices(indices, ref_indices),
            "cuDNN and Paddle indexer top-k sets should match",
        )


class TestCudnnVsTileLangCrossValidation(unittest.TestCase):
    """Cross-validate cuDNN and TileLang indexer backends produce same sets."""

    def setUp(self):
        _cuda_or_skip(self)
        paddle.set_device("gpu:0")
        try:
            paddle.enable_compat(scope={"tilelang"}, silent=True)
            from paddlefleet.tilelang_ops import csa_indexer_topk_fwd

            self.csa_indexer_topk_fwd = csa_indexer_topk_fwd
        except Exception:
            self.skipTest("TileLang CSA indexer not available")
        from paddlefleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn import (
            cudnn_indexer_topk_fwd,
        )

        self.cudnn_indexer_topk_fwd = cudnn_indexer_topk_fwd

    def test_index_sets_match(self):
        B, S_q, H_i, D_i, ratio, topk = 1, 64, 64, 128, 4, 8
        S_k = S_q // ratio
        q, k, w = _make_indexer_inputs(B, S_q, S_k, H_i, D_i, seed=42)
        cudnn_indices, _ = self.cudnn_indexer_topk_fwd(
            q, k, w, ratio=ratio, topk_effective=topk
        )
        tl_indices, _ = self.csa_indexer_topk_fwd(
            q, k, w, ratio=ratio, topk_effective=topk
        )
        self.assertTrue(
            _sorted_compare_indices(cudnn_indices, tl_indices),
            "cuDNN and TileLang indexer top-k sets should match",
        )


if __name__ == "__main__":
    unittest.main()
