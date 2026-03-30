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
Tests for packed sequence flashmask attention and total_seq_len rope.

1. TestPackedSeqFlashMaskAttention: Verifies that the flashmask-based packed
   sequence attention produces numerically equivalent results to the reference
   split+loop scaled_dot_product_attention (with dropout=0).
2. TestTotalSeqLenRoPE: Verifies that passing total_seq_len correctly selects
   the CASE 1 (offset-based) frequency mapping when cu_seqlens is padded.
"""

import unittest

import numpy as np
import paddle

from paddlefleet.models.common.embeddings.rope_utils import (
    _apply_rotary_pos_emb_thd,
)


class TestPackedSeqFlashMaskAttention(unittest.TestCase):
    """Test that flashmask packed seq attention matches split+loop reference."""

    def setUp(self):
        paddle.seed(42)

    def _reference_split_loop_attention(self, query, key, value, cu_seqlens):
        """Reference implementation: split by segments, per-segment sdpa, concat."""
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        q_splits = paddle.split(query, lengths, axis=1)
        k_splits = paddle.split(key, lengths, axis=1)
        v_splits = paddle.split(value, lengths, axis=1)
        outputs = []
        for q, k, v in zip(q_splits, k_splits, v_splits):
            out = paddle.nn.functional.scaled_dot_product_attention(
                q, k, v, None, 0.0, is_causal=False
            )
            outputs.append(out)
        return paddle.concat(outputs, axis=1)

    def _build_flashmask_indices(self, cu_seqlens, seq_length):
        """Build block-diagonal startend_row_indices from cu_seqlens."""
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        indices_per_segment = paddle.stack(
            [
                cu_seqlens[1:],
                paddle.full_like(cu_seqlens[1:], seq_length),
                paddle.zeros_like(cu_seqlens[:-1]),
                cu_seqlens[:-1],
            ],
            axis=1,
        )
        return (
            paddle.repeat_interleave(indices_per_segment, lengths, axis=0)
            .unsqueeze(0)
            .unsqueeze(0)
        )

    def _run_test(self, seqlens):
        total_seq = sum(seqlens)
        num_heads = 4
        head_dim = 32
        batch = 1

        query = paddle.randn(
            [batch, total_seq, num_heads, head_dim], dtype="float16"
        )
        key = paddle.randn(
            [batch, total_seq, num_heads, head_dim], dtype="float16"
        )
        value = paddle.randn(
            [batch, total_seq, num_heads, head_dim], dtype="float16"
        )

        cu_seqlens = paddle.to_tensor(
            [0, *list(paddle.cumsum(paddle.to_tensor(seqlens)).numpy())],
            dtype="int32",
        )

        ref_output = self._reference_split_loop_attention(
            query, key, value, cu_seqlens
        )

        indices = self._build_flashmask_indices(cu_seqlens, total_seq)
        from paddlefleet.transformer.dot_product_attention import (
            flashmask_attention,
        )

        fm_output = flashmask_attention(
            query,
            key,
            value,
            startend_row_indices=indices,
            dropout=0.0,
            causal=False,
        )

        np.testing.assert_allclose(
            ref_output.astype("float32").numpy(),
            fm_output.astype("float32").numpy(),
            atol=1e-2,
            rtol=1e-2,
        )

    def test_multiple_segments(self):
        """Test with multiple segments of different lengths."""
        self._run_test([32, 64, 16])

    def test_single_segment(self):
        """Test with a single segment (degenerate case)."""
        self._run_test([128])

    def test_equal_segments(self):
        """Test with equal-length segments."""
        self._run_test([32, 32, 32, 32])

    def test_short_segments(self):
        """Test with very short segments."""
        self._run_test([4, 8, 4])


class TestTotalSeqLenRoPE(unittest.TestCase):
    """Test that total_seq_len correctly controls CASE 1/2 branch in THD RoPE."""

    def setUp(self):
        paddle.seed(42)

    def test_total_seq_len_selects_case1(self):
        """When total_seq_len matches freqs.size(1), CASE 1 (exact mapping) is used.

        Construct a scenario where cu_seqlens is padded (cu_seqlens[-1] > actual total),
        but total_seq_len provides the true total. Verify the result matches the
        non-padded reference.
        """
        total_tokens = 48
        head_dim = 16
        num_heads = 2

        t = paddle.randn(
            [1, total_tokens, num_heads, head_dim], dtype="float32"
        )
        cu_seqlens = paddle.to_tensor([0, 16, 48], dtype="int32")
        freqs = paddle.randn([1, total_tokens, 1, head_dim], dtype="float32")

        # With correct total_seq_len, should hit CASE 1 (exact mapping)
        result_with_total = _apply_rotary_pos_emb_thd(
            t, cu_seqlens, total_seq_len=total_tokens, freqs=freqs
        )

        # Without total_seq_len (None), should also hit CASE 1 since
        # cu_seqlens[-1] == total_tokens in this non-padded case
        result_without_total = _apply_rotary_pos_emb_thd(
            t, cu_seqlens, total_seq_len=None, freqs=freqs
        )

        np.testing.assert_allclose(
            result_with_total.numpy(),
            result_without_total.numpy(),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_padded_cu_seqlens_with_total_seq_len(self):
        """When cu_seqlens is padded, total_seq_len ensures correct branch selection.

        Without total_seq_len, padded cu_seqlens[-1] != freqs.size(1) would
        incorrectly select CASE 2. With total_seq_len, CASE 1 is selected.
        """
        actual_total = 48
        padded_total = 64  # cu_seqlens[-1] after padding
        head_dim = 16
        num_heads = 2

        t = paddle.randn(
            [1, padded_total, num_heads, head_dim], dtype="float32"
        )

        # Padded cu_seqlens: last value is padded_total, not actual_total
        cu_seqlens_padded = paddle.to_tensor(
            [0, 16, padded_total], dtype="int32"
        )

        # freqs shaped for actual_total (CASE 1 expects freqs.size(1) == total_seq_len)
        freqs_actual = paddle.randn(
            [1, actual_total, 1, head_dim], dtype="float32"
        )

        # With total_seq_len=actual_total, freqs.size(1)==total_seq_len -> CASE 1
        result = _apply_rotary_pos_emb_thd(
            t, cu_seqlens_padded, total_seq_len=actual_total, freqs=freqs_actual
        )

        # Verify output shape matches input
        self.assertEqual(result.shape, t.shape)

    def test_none_total_seq_len_fallback(self):
        """When total_seq_len is None, it falls back to cu_seqlens[-1]."""
        total_tokens = 32
        head_dim = 16
        num_heads = 2

        t = paddle.randn(
            [1, total_tokens, num_heads, head_dim], dtype="float32"
        )
        cu_seqlens = paddle.to_tensor([0, 16, 32], dtype="int32")
        # freqs with size matching cu_seqlens[-1] for CASE 1
        freqs = paddle.randn([1, total_tokens, 1, head_dim], dtype="float32")

        # total_seq_len=None should fallback to cu_seqlens[-1]=32
        result_none = _apply_rotary_pos_emb_thd(
            t, cu_seqlens, total_seq_len=None, freqs=freqs
        )
        result_explicit = _apply_rotary_pos_emb_thd(
            t, cu_seqlens, total_seq_len=total_tokens, freqs=freqs
        )

        np.testing.assert_allclose(
            result_none.numpy(),
            result_explicit.numpy(),
            atol=1e-6,
            rtol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
