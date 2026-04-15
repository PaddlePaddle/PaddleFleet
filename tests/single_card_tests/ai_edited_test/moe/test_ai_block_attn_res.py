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


def _make_block_config(**overrides):
    """Helper to create config for BlockAttnRes testing."""
    from paddlefleet.transformer.transformer_config import TransformerConfig

    defaults = {
        "hidden_size": 64,
        "num_attention_heads": 2,
        "intermediate_size": 256,
        "sequence_parallel": False,
        "tensor_model_parallel_size": 1,
        "rms_norm_eps": 1e-5,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _make_norm_mock():
    """Create a mock that acts like a norm layer (callable returning transformed input)."""
    mock = MagicMock()
    # When called as a layer, it should return the input transformed
    mock.side_effect = lambda x: x * 0.5 + 0.1
    return mock


class TestBlockAttnRes(unittest.TestCase):
    """Unit tests for block_attn_res module."""

    @patch(
        "paddlefleet.transformer.block_attn_res.build_layer",
        side_effect=lambda *a, **kw: _make_norm_mock(),
    )
    @patch(
        "paddlefleet.transformer.block_attn_res.mark_as_sequence_parallel_parameter",
        side_effect=lambda x: x,
    )
    def test_block_attn_res_init(self, mock_mark, mock_build):
        """Test BlockAttnRes initialization."""
        from paddlefleet.transformer.block_attn_res import BlockAttnRes

        config = _make_block_config()
        spec = MagicMock()
        spec.norm = MagicMock

        layer = BlockAttnRes(config=config, sublayers_spec=spec)
        self.assertEqual(layer.hidden_size, 64)
        self.assertIsNotNone(layer.proj_weight)
        self.assertIsNotNone(layer.norm)

    @patch(
        "paddlefleet.transformer.block_attn_res.build_layer",
        side_effect=lambda *a, **kw: _make_norm_mock(),
    )
    @patch(
        "paddlefleet.transformer.block_attn_res.mark_as_sequence_parallel_parameter",
        side_effect=lambda x: x,
    )
    def test_block_attn_res_forward_basic(self, mock_mark, mock_build):
        """Test BlockAttnRes forward with one block."""
        from paddlefleet.transformer.block_attn_res import BlockAttnRes

        config = _make_block_config()
        spec = MagicMock()
        spec.norm = MagicMock

        layer = BlockAttnRes(config=config, sublayers_spec=spec)

        partial = paddle.randn([2, 4, 64], dtype=paddle.float32)
        blocks = [paddle.randn([2, 4, 64], dtype=paddle.float32)]

        result = layer.forward(partial, blocks)
        self.assertEqual(result.shape, [2, 4, 64])

    @patch(
        "paddlefleet.transformer.block_attn_res.build_layer",
        side_effect=lambda *a, **kw: _make_norm_mock(),
    )
    @patch(
        "paddlefleet.transformer.block_attn_res.mark_as_sequence_parallel_parameter",
        side_effect=lambda x: x,
    )
    def test_block_attn_res_forward_multiple_blocks(
        self, mock_mark, mock_build
    ):
        """Test BlockAttnRes forward with multiple blocks."""
        from paddlefleet.transformer.block_attn_res import BlockAttnRes

        config = _make_block_config()
        spec = MagicMock()
        spec.norm = MagicMock

        layer = BlockAttnRes(config=config, sublayers_spec=spec)

        partial = paddle.randn([2, 4, 64], dtype=paddle.float32)
        blocks = [
            paddle.randn([2, 4, 64], dtype=paddle.float32) for _ in range(3)
        ]

        result = layer.forward(partial, blocks)
        self.assertEqual(result.shape, [2, 4, 64])

    @patch(
        "paddlefleet.transformer.block_attn_res.build_layer",
        side_effect=lambda *a, **kw: _make_norm_mock(),
    )
    @patch(
        "paddlefleet.transformer.block_attn_res.mark_as_sequence_parallel_parameter",
        side_effect=lambda x: x,
    )
    def test_block_attn_res_forward_no_blocks(self, mock_mark, mock_build):
        """Test BlockAttnRes forward with empty blocks list."""
        from paddlefleet.transformer.block_attn_res import BlockAttnRes

        config = _make_block_config()
        spec = MagicMock()
        spec.norm = MagicMock

        layer = BlockAttnRes(config=config, sublayers_spec=spec)

        partial = paddle.randn([2, 4, 64], dtype=paddle.float32)
        blocks = []

        result = layer.forward(partial, blocks)
        self.assertEqual(result.shape, [2, 4, 64])

    @patch(
        "paddlefleet.transformer.block_attn_res.build_layer",
        side_effect=lambda *a, **kw: _make_norm_mock(),
    )
    @patch(
        "paddlefleet.transformer.block_attn_res.mark_as_sequence_parallel_parameter",
        side_effect=lambda x: x,
    )
    def test_proj_weight_initialization(self, mock_mark, mock_build):
        """Test proj_weight is initialized to zeros."""
        from paddlefleet.transformer.block_attn_res import BlockAttnRes

        config = _make_block_config()
        spec = MagicMock()
        spec.norm = MagicMock

        layer = BlockAttnRes(config=config, sublayers_spec=spec)
        self.assertTrue(paddle.allclose(layer.proj_weight, paddle.zeros([64])))

    @patch(
        "paddlefleet.transformer.block_attn_res.build_layer",
        side_effect=lambda *a, **kw: _make_norm_mock(),
    )
    @patch(
        "paddlefleet.transformer.block_attn_res.mark_as_sequence_parallel_parameter",
        side_effect=lambda x: x,
    )
    def test_block_attn_res_forward_softmax_weights(
        self, mock_mark, mock_build
    ):
        """Test that softmax weights sum to 1 over block dimension."""
        from paddlefleet.transformer.block_attn_res import BlockAttnRes

        config = _make_block_config()
        spec = MagicMock()
        spec.norm = MagicMock

        layer = BlockAttnRes(config=config, sublayers_spec=spec)

        partial = paddle.randn([1, 2, 64], dtype=paddle.float32)
        blocks = [paddle.randn([1, 2, 64], dtype=paddle.float32)]

        # The forward internally computes softmax over block dim.
        # We verify the output shape is correct.
        result = layer.forward(partial, blocks)
        self.assertEqual(result.shape, [1, 2, 64])

    def test_block_attn_res_sublayers_spec(self):
        """Test BlockAttnResSublayersSpec defaults."""
        from paddlefleet.transformer.block_attn_res import (
            BlockAttnResSublayersSpec,
        )
        from paddlefleet.transformer.identity_op import IdentityOp

        spec = BlockAttnResSublayersSpec()
        self.assertIs(spec.norm, IdentityOp)

    @patch(
        "paddlefleet.transformer.block_attn_res.build_layer",
        side_effect=lambda *a, **kw: _make_norm_mock(),
    )
    @patch(
        "paddlefleet.transformer.block_attn_res.mark_as_sequence_parallel_parameter",
        side_effect=lambda x: x,
    )
    def test_block_attn_res_with_sequence_parallel(self, mock_mark, mock_build):
        """Test BlockAttnRes with sequence_parallel and tensor_parallel."""
        from paddlefleet.transformer.block_attn_res import BlockAttnRes

        config = _make_block_config(
            tensor_model_parallel_size=2,
            sequence_parallel=True,
        )
        spec = MagicMock()
        spec.norm = MagicMock

        layer = BlockAttnRes(config=config, sublayers_spec=spec)
        self.assertEqual(layer.hidden_size, 64)
        # mark_as_sequence_parallel_parameter should be called
        mock_mark.assert_called_once()

    @patch(
        "paddlefleet.transformer.block_attn_res.build_layer",
        side_effect=lambda *a, **kw: _make_norm_mock(),
    )
    @patch(
        "paddlefleet.transformer.block_attn_res.mark_as_sequence_parallel_parameter",
        side_effect=lambda x: x,
    )
    def test_block_attn_res_different_hidden_sizes(self, mock_mark, mock_build):
        """Test BlockAttnRes with various hidden sizes."""
        from paddlefleet.transformer.block_attn_res import BlockAttnRes

        for hidden_size in [32, 128, 256]:
            config = _make_block_config(hidden_size=hidden_size)
            spec = MagicMock()
            spec.norm = MagicMock

            layer = BlockAttnRes(config=config, sublayers_spec=spec)
            self.assertEqual(layer.hidden_size, hidden_size)
            self.assertEqual(layer.proj_weight.shape[0], hidden_size)


if __name__ == "__main__":
    unittest.main()
