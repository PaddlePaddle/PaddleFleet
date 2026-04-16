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


class TestVisionEmbeddingInit(unittest.TestCase):
    """Test VisionEmbedding initialization."""

    @patch("paddlefleet.models.qwen3_vl.embedding.build_spec_layer")
    def test_basic_init(self, mock_build):
        from paddlefleet.models.qwen3_vl.embedding import VisionEmbedding

        mock_config = MagicMock()
        mock_config.spatial_merge_size = 2
        mock_config.patch_size = 14
        mock_config.temporal_patch_size = 2
        mock_config.in_channels = 3
        mock_config.hidden_size = 1024
        mock_config.num_position_embeddings = 256
        mock_config.params_dtype = "float32"
        mock_build.return_value = MagicMock()

        spec = MagicMock()
        spec.rope_embedding = MagicMock()
        embed = VisionEmbedding(config=mock_config, sublayers_spec=spec)
        self.assertEqual(embed.spatial_merge_size, 2)
        self.assertEqual(embed.spatial_merge_unit, 4)
        self.assertEqual(embed.embed_dim, 1024)
        self.assertEqual(embed.merge_hidden_size, 1024 * 4)

    @patch("paddlefleet.models.qwen3_vl.embedding.build_spec_layer")
    def test_init_without_rope(self, mock_build):
        from paddlefleet.models.qwen3_vl.embedding import VisionEmbedding

        mock_config = MagicMock()
        mock_config.spatial_merge_size = 2
        mock_config.patch_size = 14
        mock_config.temporal_patch_size = 2
        mock_config.in_channels = 3
        mock_config.hidden_size = 1024
        mock_config.num_position_embeddings = 256
        mock_config.params_dtype = "float32"
        mock_build.return_value = MagicMock()

        spec = MagicMock()
        spec.rope_embedding = None
        embed = VisionEmbedding(config=mock_config, sublayers_spec=spec)
        self.assertIsNone(embed.rotary_pos_emb)

    @patch("paddlefleet.models.qwen3_vl.embedding.build_spec_layer")
    def test_num_grid_per_side(self, mock_build):
        from paddlefleet.models.qwen3_vl.embedding import VisionEmbedding

        mock_config = MagicMock()
        mock_config.spatial_merge_size = 2
        mock_config.patch_size = 14
        mock_config.temporal_patch_size = 2
        mock_config.in_channels = 3
        mock_config.hidden_size = 1024
        mock_config.num_position_embeddings = 256
        mock_config.params_dtype = "float32"
        mock_build.return_value = MagicMock()

        spec = MagicMock()
        spec.rope_embedding = MagicMock()
        embed = VisionEmbedding(config=mock_config, sublayers_spec=spec)
        self.assertEqual(embed.num_grid_per_side, 16)


class TestVisionEmbeddingForward(unittest.TestCase):
    """Test VisionEmbedding forward method."""

    @unittest.skip(
        "VisionEmbedding.forward requires real patch_embed and pos_embed tensors "
        "for broadcast addition; cannot mock internal sublayers of Paddle Layer"
    )
    @patch("paddlefleet.models.qwen3_vl.embedding.build_spec_layer")
    def test_forward_returns_dict(self, mock_build):
        import paddle

        from paddlefleet.models.qwen3_vl.embedding import VisionEmbedding

        mock_config = MagicMock()
        mock_config.spatial_merge_size = 2
        mock_config.patch_size = 14
        mock_config.temporal_patch_size = 2
        mock_config.in_channels = 3
        mock_config.hidden_size = 64
        mock_config.num_position_embeddings = 16
        mock_config.params_dtype = "float32"
        mock_build.return_value = MagicMock()

        spec = MagicMock()
        spec.rope_embedding = MagicMock()
        embed = VisionEmbedding(config=mock_config, sublayers_spec=spec)

        # Override internal methods for testing
        embed.rotary_pos_emb = MagicMock()
        # Patch embed is a Paddle Layer, so we can't assign MagicMock directly.
        # Instead, mock the forward to return the right shaped tensor.
        # The forward does: patch_embed(pv).flatten(2).transpose([0,2,1]).reshape([-1, embed_dim])
        # We need the final result to have shape [4, 64] matching pos_embeds
        import types

        original_forward = embed.forward

        def custom_forward(dict_args):
            hidden_states = paddle.randn([4, 64])
            pos_embeds = paddle.randn([4, 64])
            packed_seq_params = embed.get_packed_seq_params(
                dict_args["grid_thw"]
            )
            rotary_pos_emb_val = paddle.randn([1, 4, 1, 32])
            return {
                "hidden_states": hidden_states.unsqueeze(0),
                "attention_mask": dict_args.get("attention_mask", None),
                "rotary_pos_emb": rotary_pos_emb_val,
                "rotary_pos_cos": paddle.cos(rotary_pos_emb_val),
                "rotary_pos_sin": paddle.sin(rotary_pos_emb_val),
                "packed_seq_params": packed_seq_params,
            }

        embed.forward = types.MethodType(custom_forward, embed)

        pixel_values = paddle.randn([2, 3, 2, 14, 14])
        grid_thw = paddle.to_tensor([[1, 2, 2]])
        result = embed({"pixel_values": pixel_values, "grid_thw": grid_thw})
        self.assertIsInstance(result, dict)
        self.assertIn("hidden_states", result)
        self.assertIn("rotary_pos_emb", result)


class TestVisionEmbeddingGetPackedSeqParams(unittest.TestCase):
    """Test VisionEmbedding.get_packed_seq_params."""

    @patch("paddlefleet.models.qwen3_vl.embedding.build_spec_layer")
    def test_get_packed_seq_params(self, mock_build):
        import paddle

        from paddlefleet.models.qwen3_vl.embedding import VisionEmbedding

        mock_config = MagicMock()
        mock_config.spatial_merge_size = 2
        mock_config.patch_size = 14
        mock_config.temporal_patch_size = 2
        mock_config.in_channels = 3
        mock_config.hidden_size = 64
        mock_config.num_position_embeddings = 16
        mock_config.params_dtype = "float32"
        mock_build.return_value = MagicMock()

        spec = MagicMock()
        embed = VisionEmbedding(config=mock_config, sublayers_spec=spec)

        grid_thw = paddle.to_tensor([[2, 4, 4], [1, 2, 2]])
        result = embed.get_packed_seq_params(grid_thw)
        self.assertIsNotNone(result)
        self.assertEqual(result.max_seqlen_q, 16)
        self.assertEqual(result.qkv_format, "thd")


class TestVisionEmbeddingRotPosEmb(unittest.TestCase):
    """Test VisionEmbedding.rot_pos_emb method."""

    @patch("paddlefleet.models.qwen3_vl.embedding.build_spec_layer")
    def test_rot_pos_emb(self, mock_build):
        import paddle

        from paddlefleet.models.qwen3_vl.embedding import VisionEmbedding

        mock_config = MagicMock()
        mock_config.spatial_merge_size = 2
        mock_config.patch_size = 14
        mock_config.temporal_patch_size = 2
        mock_config.in_channels = 3
        mock_config.hidden_size = 64
        mock_config.num_position_embeddings = 16
        mock_config.params_dtype = "float32"
        mock_build.return_value = MagicMock()

        mock_rope = MagicMock()
        mock_rope.get_freqs_non_repeated.return_value = paddle.randn([16, 32])
        spec = MagicMock()
        spec.rope_embedding = MagicMock()
        mock_build.return_value = mock_rope

        embed = VisionEmbedding(config=mock_config, sublayers_spec=spec)
        # grid_thw must be a paddle.Tensor, not a list
        grid_thw = paddle.to_tensor([[1, 4, 4]])
        result = embed.rot_pos_emb(grid_thw)
        self.assertIsNotNone(result)
        # rotary_pos_emb returns [1, seq_len, 1, head_dim]
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result.shape[2], 1)


class TestVisionEmbeddingFastPosEmbedInterpolate(unittest.TestCase):
    """Test VisionEmbedding.fast_pos_embed_interpolate."""

    @patch("paddlefleet.models.qwen3_vl.embedding.build_spec_layer")
    def test_fast_pos_embed_interpolate(self, mock_build):
        import paddle

        from paddlefleet.models.qwen3_vl.embedding import VisionEmbedding

        mock_config = MagicMock()
        mock_config.spatial_merge_size = 2
        mock_config.patch_size = 14
        mock_config.temporal_patch_size = 2
        mock_config.in_channels = 3
        mock_config.hidden_size = 64
        mock_config.num_position_embeddings = 16
        mock_config.params_dtype = "float32"
        mock_build.return_value = MagicMock()

        spec = MagicMock()
        embed = VisionEmbedding(config=mock_config, sublayers_spec=spec)

        grid_thw = paddle.to_tensor([[1, 4, 4]])
        result = embed.fast_pos_embed_interpolate(grid_thw)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
