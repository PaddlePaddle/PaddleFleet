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
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)


# Tests for src/paddlefleet/context_parallel_utils.py
# Test FlashMaskContextParallel, flashmask_attention_cp, preprocess_index

import unittest
from unittest import mock

import paddle


class TestFlashMaskContextParallelForward(unittest.TestCase):
    """Tests for FlashMaskContextParallel forward pass."""

    def test_dropout_not_supported(self):
        """Test that dropout > 0 raises NotImplementedError."""
        from paddlefleet.context_parallel_utils import FlashMaskContextParallel

        mock_ctx = mock.MagicMock()
        mock_config = mock.MagicMock()
        query = paddle.randn([2, 8, 4, 16])
        key = paddle.randn([2, 8, 4, 16])
        value = paddle.randn([2, 8, 4, 16])
        mask_indices = paddle.randint(0, 100, [100, 2])

        with self.assertRaises(NotImplementedError) as ctx:
            FlashMaskContextParallel.forward(
                mock_ctx,
                mock_config,
                query,
                key,
                value,
                mask_indices,
                dropout=0.5,
            )
        self.assertIn("Dropout", str(ctx.exception))

    def test_causal_not_supported(self):
        """Test that causal=True raises NotImplementedError."""
        from paddlefleet.context_parallel_utils import FlashMaskContextParallel

        mock_ctx = mock.MagicMock()
        mock_config = mock.MagicMock()
        query = paddle.randn([2, 8, 4, 16])
        key = paddle.randn([2, 8, 4, 16])
        value = paddle.randn([2, 8, 4, 16])
        mask_indices = paddle.randint(0, 100, [100, 2])

        with self.assertRaises(NotImplementedError) as ctx:
            FlashMaskContextParallel.forward(
                mock_ctx,
                mock_config,
                query,
                key,
                value,
                mask_indices,
                causal=True,
            )
        self.assertIn("causal", str(ctx.exception))

    def test_fixed_seed_offset_not_supported(self):
        """Test that fixed_seed_offset raises NotImplementedError."""
        from paddlefleet.context_parallel_utils import FlashMaskContextParallel

        mock_ctx = mock.MagicMock()
        mock_config = mock.MagicMock()
        query = paddle.randn([2, 8, 4, 16])
        key = paddle.randn([2, 8, 4, 16])
        value = paddle.randn([2, 8, 4, 16])
        mask_indices = paddle.randint(0, 100, [100, 2])
        seed = paddle.zeros([1], dtype="int64")

        with self.assertRaises(NotImplementedError):
            FlashMaskContextParallel.forward(
                mock_ctx,
                mock_config,
                query,
                key,
                value,
                mask_indices,
                fixed_seed_offset=seed,
            )

    def test_query_seq_len_must_be_even(self):
        """Test assertion on odd query seq_len."""
        from paddlefleet.context_parallel_utils import FlashMaskContextParallel

        mock_ctx = mock.MagicMock()
        mock_config = mock.MagicMock()
        query = paddle.randn([2, 7, 4, 16])  # 7 is odd
        key = paddle.randn([2, 7, 4, 16])
        value = paddle.randn([2, 7, 4, 16])
        mask_indices = paddle.randint(0, 100, [100, 2])

        mock_hcg = mock.MagicMock()
        mock_group = mock.MagicMock()
        mock_group.rank = 0
        mock_group.world_size = 2
        mock_hcg.get_context_parallel_group.return_value = mock_group

        with mock.patch(
            "paddle.distributed.fleet.get_hybrid_communicate_group",
            return_value=mock_hcg,
        ):
            with self.assertRaises(AssertionError) as ctx:
                FlashMaskContextParallel.forward(
                    mock_ctx,
                    mock_config,
                    query,
                    key,
                    value,
                    mask_indices,
                )
            self.assertIn("divisible by 2", str(ctx.exception))

    def test_forward_saves_context(self):
        """Test forward saves tensors and config to context."""
        from paddlefleet.context_parallel_utils import FlashMaskContextParallel

        mock_ctx = mock.MagicMock()
        mock_config = mock.MagicMock()
        query = paddle.randn([2, 8, 4, 16])
        key = paddle.randn([2, 8, 4, 16])
        value = paddle.randn([2, 8, 4, 16])
        mask_indices = paddle.randint(0, 100, [100, 2])

        mock_hcg = mock.MagicMock()
        mock_group = mock.MagicMock()
        mock_group.rank = 0
        mock_group.world_size = 2
        mock_hcg.get_context_parallel_group.return_value = mock_group

        mock_output = paddle.randn([2, 8, 4, 16])
        mock_lse = paddle.randn([2, 4, 8])
        mock_processed_indices = paddle.randint(0, 100, [100, 2])

        with mock.patch(  # noqa: SIM117
            "paddle.distributed.fleet.get_hybrid_communicate_group",
            return_value=mock_hcg,
        ):
            with mock.patch(
                "paddlefleet.context_parallel_utils.cp_flashmask_allgatherkv_balance_forward",
                return_value=(
                    mock_output,
                    mock_lse,
                    mock_processed_indices,
                ),
            ):
                result = FlashMaskContextParallel.forward(
                    mock_ctx,
                    mock_config,
                    query,
                    key,
                    value,
                    mask_indices,
                )
                mock_ctx.save_for_backward.assert_called_once()
                self.assertEqual(mock_ctx.group, mock_group)
                self.assertEqual(mock_ctx.config, mock_config)


class TestFlashMaskContextParallelBackward(unittest.TestCase):
    """Tests for FlashMaskContextParallel backward pass."""

    def test_backward_retrieves_saved_tensors(self):
        """Test backward retrieves saved tensors and calls backward function."""
        from paddlefleet.context_parallel_utils import FlashMaskContextParallel

        mock_ctx = mock.MagicMock()
        query = paddle.randn([2, 8, 4, 16])
        key = paddle.randn([2, 8, 4, 16])
        value = paddle.randn([2, 8, 4, 16])
        output = paddle.randn([2, 8, 4, 16])
        lse = paddle.randn([2, 4, 8])
        indices = paddle.randint(0, 100, [100, 2])
        mock_ctx.saved_tensor.return_value = (
            query,
            key,
            value,
            output,
            lse,
            indices,
        )
        mock_ctx.group = mock.MagicMock()
        mock_ctx.causal = False
        mock_ctx.config = mock.MagicMock()

        grad = paddle.randn([2, 8, 4, 16])

        with mock.patch(
            "paddlefleet.context_parallel_utils.cp_flashmask_allgatherkv_balance_backward",
            return_value=(grad, grad, grad),
        ) as mock_bwd:
            result = FlashMaskContextParallel.backward(mock_ctx, grad)
            mock_bwd.assert_called_once()
            call_args = mock_bwd.call_args[0]
            # Check first few args
            self.assertIs(call_args[0], mock_ctx.config)
            self.assertIs(call_args[1], query)
            self.assertIs(call_args[2], key)


class TestFlashmaskAttentionCP(unittest.TestCase):
    """Tests for flashmask_attention_cp public API."""

    def test_calls_flash_mask_context_parallel_apply(self):
        """Test flashmask_attention_cp calls FlashMaskContextParallel.apply."""
        from paddlefleet.context_parallel_utils import (
            FlashMaskContextParallel,
            flashmask_attention_cp,
        )

        mock_config = mock.MagicMock()
        query = paddle.randn([2, 8, 4, 16])
        key = paddle.randn([2, 8, 4, 16])
        value = paddle.randn([2, 8, 4, 16])
        mask_indices = paddle.randint(0, 100, [100, 2])

        with mock.patch.object(
            FlashMaskContextParallel,
            "apply",
            return_value=paddle.randn([2, 8, 4, 16]),
        ) as mock_apply:
            result = flashmask_attention_cp(
                mock_config, query, key, value, mask_indices
            )
            mock_apply.assert_called_once()


class TestPreprocessIndex(unittest.TestCase):
    """Tests for preprocess_index function."""

    def test_basic_preprocess(self):
        """Test basic preprocess_index functionality."""
        from paddlefleet.context_parallel_utils import preprocess_index

        indices = paddle.to_tensor([[10, 20], [30, 40]], dtype="int32")
        result = preprocess_index(
            indices, chunk_id=1, seq_blocksize=8, max_seqlen_q=16
        )
        expected = indices - 8  # rows_min = 1 * 8
        # After clip to [0, 16]
        self.assertEqual(result.shape, [2, 2])

    def test_chunk_id_zero(self):
        """Test preprocess_index with chunk_id=0."""
        from paddlefleet.context_parallel_utils import preprocess_index

        indices = paddle.to_tensor([[5, 10]], dtype="int32")
        result = preprocess_index(
            indices, chunk_id=0, seq_blocksize=8, max_seqlen_q=16
        )
        # rows_min = 0, so no adjustment
        expected = paddle.clip(indices, min=0, max=16)
        self.assertTrue(paddle.allclose(result, expected))

    def test_negative_after_subtraction_clipped_to_zero(self):
        """Test that negative values are clipped to zero."""
        from paddlefleet.context_parallel_utils import preprocess_index

        indices = paddle.to_tensor([[3, 5]], dtype="int32")
        result = preprocess_index(
            indices, chunk_id=1, seq_blocksize=8, max_seqlen_q=16
        )
        # 3 - 8 = -5 -> clipped to 0
        # 5 - 8 = -3 -> clipped to 0
        self.assertTrue(paddle.all(result >= 0))


class TestPreprocessIndexDualChunks(unittest.TestCase):
    """Tests for preprocess_index_dual_chunks function."""

    def test_basic_dual_chunks(self):
        """Test basic preprocess_index_dual_chunks."""
        from paddlefleet.context_parallel_utils import (
            preprocess_index_dual_chunks,
        )

        indices = paddle.to_tensor([[20, 30]], dtype="int32")
        result = preprocess_index_dual_chunks(
            indices,
            chunk_id_first=1,
            chunk_id_second=2,
            seq_blocksize=8,
            max_seqlen_q=16,
        )
        self.assertEqual(result.shape, [1, 2])

    def test_same_chunk_ids(self):
        """Test with same chunk_ids."""
        from paddlefleet.context_parallel_utils import (
            preprocess_index_dual_chunks,
        )

        indices = paddle.to_tensor([[10, 20]], dtype="int32")
        result = preprocess_index_dual_chunks(
            indices,
            chunk_id_first=0,
            chunk_id_second=0,
            seq_blocksize=8,
            max_seqlen_q=16,
        )
        # Both offsets are 0, so indices remain the same after clip
        self.assertEqual(result.shape, [1, 2])

    def test_output_shape_matches_input(self):
        """Test output shape matches input shape."""
        from paddlefleet.context_parallel_utils import (
            preprocess_index_dual_chunks,
        )

        indices = paddle.to_tensor([[5, 10], [15, 20]], dtype="int32")
        result = preprocess_index_dual_chunks(
            indices,
            chunk_id_first=1,
            chunk_id_second=3,
            seq_blocksize=8,
            max_seqlen_q=16,
        )
        self.assertEqual(result.shape, indices.shape)


if __name__ == "__main__":
    unittest.main()
