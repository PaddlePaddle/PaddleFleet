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
from unittest.mock import patch

import paddle

from paddlefleet.transformer.block_attn_res import (
    BlockAttnRes,
    BlockAttnResSublayersSpec,
)
from paddlefleet.transformer.transformer_config import TransformerConfig


class FakeNorm(paddle.nn.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        hidden_size = kwargs.get(
            "normalized_shape", kwargs.get("hidden_size", 64)
        )
        if hidden_size is None:
            hidden_size = 64
        self.w = paddle.create_parameter(
            shape=[hidden_size],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(1.0),
        )

    def forward(self, x):
        return x * self.w


def _make_config(**overrides):
    defaults = {
        "hidden_size": 64,
        "num_attention_heads": 2,
        "rms_norm_eps": 1e-5,
        "tensor_model_parallel_size": 1,
        "sequence_parallel": False,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


class TestBlockAttnResConstruction(unittest.TestCase):
    """Test BlockAttnRes construction."""

    @patch("paddlefleet.transformer.block_attn_res.build_layer")
    def test_basic_construction(self, mock_build):
        mock_build.return_value = FakeNorm()
        config = _make_config()
        spec = BlockAttnResSublayersSpec(norm=FakeNorm)
        bar = BlockAttnRes(config, spec)
        self.assertEqual(bar.hidden_size, 64)
        self.assertIsNotNone(bar.proj_weight)
        self.assertIsNotNone(bar.norm)

    @patch("paddlefleet.transformer.block_attn_res.build_layer")
    def test_proj_weight_shape(self, mock_build):
        mock_build.return_value = FakeNorm()
        config = _make_config(hidden_size=128)
        spec = BlockAttnResSublayersSpec(norm=FakeNorm)
        bar = BlockAttnRes(config, spec)
        self.assertEqual(bar.proj_weight.shape, [128])

    @patch("paddlefleet.transformer.block_attn_res.build_layer")
    @patch(
        "paddlefleet.transformer.block_attn_res.mark_as_sequence_parallel_parameter"
    )
    def test_sequence_parallel_marks_param(self, mock_mark, mock_build):
        mock_build.return_value = FakeNorm()
        config = _make_config(
            tensor_model_parallel_size=2,
            sequence_parallel=True,
        )
        spec = BlockAttnResSublayersSpec(norm=FakeNorm)
        BlockAttnRes(config, spec)
        mock_mark.assert_called_once()


class TestBlockAttnResForward(unittest.TestCase):
    """Test BlockAttnRes forward."""

    @patch("paddlefleet.transformer.block_attn_res.build_layer")
    def test_forward_single_block(self, mock_build):
        mock_build.return_value = FakeNorm()
        config = _make_config()
        spec = BlockAttnResSublayersSpec(norm=FakeNorm)
        bar = BlockAttnRes(config, spec)

        partial = paddle.randn([2, 4, 64], dtype="float32")
        blocks = [paddle.randn([2, 4, 64], dtype="float32")]
        out = bar(partial, blocks)
        self.assertEqual(out.shape, [2, 4, 64])

    @patch("paddlefleet.transformer.block_attn_res.build_layer")
    def test_forward_multiple_blocks(self, mock_build):
        mock_build.return_value = FakeNorm()
        config = _make_config()
        spec = BlockAttnResSublayersSpec(norm=FakeNorm)
        bar = BlockAttnRes(config, spec)

        partial = paddle.randn([2, 4, 64], dtype="float32")
        blocks = [
            paddle.randn([2, 4, 64], dtype="float32"),
            paddle.randn([2, 4, 64], dtype="float32"),
            paddle.randn([2, 4, 64], dtype="float32"),
        ]
        out = bar(partial, blocks)
        self.assertEqual(out.shape, [2, 4, 64])

    @patch("paddlefleet.transformer.block_attn_res.build_layer")
    def test_forward_empty_blocks_list(self, mock_build):
        mock_build.return_value = FakeNorm()
        config = _make_config()
        spec = BlockAttnResSublayersSpec(norm=FakeNorm)
        bar = BlockAttnRes(config, spec)

        partial = paddle.randn([2, 4, 64], dtype="float32")
        blocks = []
        out = bar(partial, blocks)
        self.assertEqual(out.shape, [2, 4, 64])

    @patch("paddlefleet.transformer.block_attn_res.build_layer")
    def test_forward_output_not_nan(self, mock_build):
        mock_build.return_value = FakeNorm()
        config = _make_config()
        spec = BlockAttnResSublayersSpec(norm=FakeNorm)
        bar = BlockAttnRes(config, spec)

        paddle.seed(42)
        partial = paddle.randn([1, 2, 64], dtype="float32")
        blocks = [paddle.randn([1, 2, 64], dtype="float32")]
        out = bar(partial, blocks)
        self.assertFalse(paddle.isnan(out).any())

    @patch("paddlefleet.transformer.block_attn_res.build_layer")
    def test_forward_dtype_preserved(self, mock_build):
        mock_build.return_value = FakeNorm()
        config = _make_config()
        spec = BlockAttnResSublayersSpec(norm=FakeNorm)
        bar = BlockAttnRes(config, spec)

        partial = paddle.randn([2, 4, 64], dtype="float32")
        blocks = [paddle.randn([2, 4, 64], dtype="float32")]
        out = bar(partial, blocks)
        self.assertEqual(out.dtype, paddle.float32)

    @patch("paddlefleet.transformer.block_attn_res.build_layer")
    def test_forward_different_batch_sizes(self, mock_build):
        mock_build.return_value = FakeNorm()
        config = _make_config()
        spec = BlockAttnResSublayersSpec(norm=FakeNorm)
        bar = BlockAttnRes(config, spec)

        partial = paddle.randn([1, 8, 64], dtype="float32")
        blocks = [paddle.randn([1, 8, 64], dtype="float32")]
        out = bar(partial, blocks)
        self.assertEqual(out.shape, [1, 8, 64])

    @patch("paddlefleet.transformer.block_attn_res.build_layer")
    def test_forward_small_hidden(self, mock_build):
        def build_side_effect(*a, **kw):
            hidden_size = kw.get("hidden_size", kw.get("normalized_shape", 64))
            return FakeNorm(hidden_size=hidden_size)

        mock_build.side_effect = build_side_effect
        config = _make_config(hidden_size=32)
        spec = BlockAttnResSublayersSpec(norm=FakeNorm)
        bar = BlockAttnRes(config, spec)

        partial = paddle.randn([2, 4, 32], dtype="float32")
        blocks = [paddle.randn([2, 4, 32], dtype="float32")]
        out = bar(partial, blocks)
        self.assertEqual(out.shape, [2, 4, 32])


class TestBlockAttnResSublayersSpec(unittest.TestCase):
    """Test BlockAttnResSublayersSpec dataclass."""

    def test_default_norm(self):
        from paddlefleet.transformer.identity_op import IdentityOp

        spec = BlockAttnResSublayersSpec()
        self.assertEqual(spec.norm, IdentityOp)


class TestBlockAttnResGetNormExtraArgs(unittest.TestCase):
    """Test get_norm_extra_args integration."""

    @patch("paddlefleet.transformer.block_attn_res.build_layer")
    def test_norm_extra_args_called(self, mock_build):
        mock_build.return_value = FakeNorm()
        config = _make_config()
        spec = BlockAttnResSublayersSpec(norm=FakeNorm)
        # Construction should call build_layer with extra args
        BlockAttnRes(config, spec)
        mock_build.assert_called_once()
        call_kwargs = mock_build.call_args[1] if mock_build.call_args else {}
        self.assertIn("config", call_kwargs)
        self.assertIn("input_is_parallel", call_kwargs)


if __name__ == "__main__":
    unittest.main()
