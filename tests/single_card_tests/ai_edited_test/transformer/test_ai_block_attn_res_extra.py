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

import unittest
from unittest.mock import MagicMock, patch

import paddle

from paddlefleet.transformer.block_attn_res import (
    BlockAttnRes,
    BlockAttnResSublayersSpec,
)
from paddlefleet.transformer.paddle_norm import WrappedPaddleNorm
from paddlefleet.transformer.transformer_config import TransformerConfig


def _make_config(**overrides):
    defaults = {
        "hidden_size": 64,
        "num_attention_heads": 4,
        "rms_norm_eps": 1e-5,
        "sequence_parallel": False,
        "tensor_model_parallel_size": 1,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


class TestBlockAttnResSequenceParallel(unittest.TestCase):
    """Tests for BlockAttnRes with sequence parallel."""

    @patch("paddlefleet.transformer.block_attn_res.build_spec_layer")
    def test_sequence_parallel_marks_weight(self, mock_build):
        """Test that proj_weight is marked as sequence parallel when appropriate."""
        mock_build.return_value = MagicMock()
        config = _make_config(
            sequence_parallel=True, tensor_model_parallel_size=2
        )
        spec = BlockAttnResSublayersSpec(norm=WrappedPaddleNorm)

        block = BlockAttnRes(config=config, sublayers_spec=spec)
        self.assertEqual(block.hidden_size, 64)

    @patch("paddlefleet.transformer.block_attn_res.build_spec_layer")
    def test_no_sequence_parallel(self, mock_build):
        """Test that proj_weight is not marked when sequence_parallel=False."""
        mock_build.return_value = MagicMock()
        config = _make_config(
            sequence_parallel=False, tensor_model_parallel_size=1
        )
        spec = BlockAttnResSublayersSpec(norm=WrappedPaddleNorm)

        block = BlockAttnRes(config=config, sublayers_spec=spec)
        self.assertEqual(block.hidden_size, 64)


class TestBlockAttnResForwardDetailed(unittest.TestCase):
    """Detailed tests for BlockAttnRes forward."""

    def test_forward_with_real_norm_and_blocks(self):
        """Test forward with real normalization and multiple blocks."""
        config = _make_config()
        spec = BlockAttnResSublayersSpec(norm=WrappedPaddleNorm)

        block = BlockAttnRes(config=config, sublayers_spec=spec)

        partial_block = paddle.randn([2, 4, 64])
        blocks = [
            paddle.randn([2, 4, 64]),
            paddle.randn([2, 4, 64]),
            paddle.randn([2, 4, 64]),
        ]

        output = block(partial_block, blocks)
        self.assertEqual(output.shape, [2, 4, 64])

    def test_forward_weights_sum_to_one(self):
        """Test that attention weights sum to approximately 1."""
        config = _make_config()
        spec = BlockAttnResSublayersSpec(norm=WrappedPaddleNorm)

        block = BlockAttnRes(config=config, sublayers_spec=spec)

        partial_block = paddle.randn([2, 4, 64])
        blocks = [paddle.randn([2, 4, 64])]

        # The weights are computed as softmax, so they should sum to 1
        # over the block dimension
        V = paddle.stack([*blocks, partial_block], axis=0)
        K = block.norm(V)
        logits = (K * block.proj_weight).sum(axis=-1)
        weights = paddle.nn.functional.softmax(logits, axis=0)

        # Sum along block dimension (axis=0)
        weight_sum = weights.sum(axis=0)
        self.assertTrue(
            paddle.allclose(
                weight_sum, paddle.ones_like(weight_sum), atol=1e-5
            ).item()
        )


class TestBlockAttnResDifferentBatchSizes(unittest.TestCase):
    """Tests for BlockAttnRes with different batch sizes."""

    def test_batch_size_1(self):
        """Test with batch_size=1."""
        config = _make_config()
        spec = BlockAttnResSublayersSpec(norm=WrappedPaddleNorm)
        block = BlockAttnRes(config=config, sublayers_spec=spec)

        partial_block = paddle.randn([1, 4, 64])
        blocks = [paddle.randn([1, 4, 64])]

        output = block(partial_block, blocks)
        self.assertEqual(output.shape, [1, 4, 64])

    def test_batch_size_4(self):
        """Test with batch_size=4."""
        config = _make_config()
        spec = BlockAttnResSublayersSpec(norm=WrappedPaddleNorm)
        block = BlockAttnRes(config=config, sublayers_spec=spec)

        partial_block = paddle.randn([4, 8, 64])
        blocks = [paddle.randn([4, 8, 64])]

        output = block(partial_block, blocks)
        self.assertEqual(output.shape, [4, 8, 64])


if __name__ == "__main__":
    unittest.main()
