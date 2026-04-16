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


class TestGPTEmbeddingSpec(unittest.TestCase):
    """Test GPTEmbeddingSpec dataclass."""

    def test_defaults(self):
        from paddlefleet.models.gpt.gpt_embedding import GPTEmbeddingSpec

        spec = GPTEmbeddingSpec(
            language_embedding=MagicMock(), rope_embedding=None
        )
        self.assertIsNone(spec.rope_embedding)

    def test_with_rope_embedding(self):
        from paddlefleet.models.gpt.gpt_embedding import GPTEmbeddingSpec

        mock_rope = MagicMock()
        spec = GPTEmbeddingSpec(
            language_embedding=MagicMock(), rope_embedding=mock_rope
        )
        self.assertEqual(spec.rope_embedding, mock_rope)


class TestGPTEmbeddingInit(unittest.TestCase):
    """Test GPTEmbedding initialization."""

    @patch("paddlefleet.models.gpt.gpt_embedding.build_spec_layer")
    def test_basic_init(self, mock_build):
        from paddlefleet.models.gpt.gpt_embedding import (
            GPTEmbedding,
            GPTEmbeddingSpec,
        )

        mock_config = MagicMock()
        mock_config.sequence_parallel = False
        mock_config.multimodal_embedding = False
        mock_config.num_nextn_predict_layers = None
        mock_config.mtp_load_weight_only = False
        mock_config.head_dim = 64
        mock_config.rotary_interleaved = False
        mock_build.return_value = MagicMock()

        spec = GPTEmbeddingSpec(
            language_embedding=MagicMock(), rope_embedding=None
        )
        emb = GPTEmbedding(
            sublayers_spec=spec,
            config=mock_config,
            vocab_size=1024,
            max_sequence_length=64,
            position_embedding_type="rope",
        )
        self.assertFalse(emb.sequence_parallel)
        self.assertIsNone(emb.rotary_pos_emb)

    @patch("paddlefleet.models.gpt.gpt_embedding.build_spec_layer")
    def test_with_rope_embedding(self, mock_build):
        from paddlefleet.models.gpt.gpt_embedding import (
            GPTEmbedding,
            GPTEmbeddingSpec,
        )

        mock_config = MagicMock()
        mock_config.sequence_parallel = False
        mock_config.multimodal_embedding = False
        mock_config.num_nextn_predict_layers = None
        mock_config.mtp_load_weight_only = False
        mock_config.head_dim = 64
        mock_config.rotary_interleaved = False
        mock_build.return_value = MagicMock()

        mock_rope_spec = MagicMock()
        spec = GPTEmbeddingSpec(
            language_embedding=MagicMock(), rope_embedding=mock_rope_spec
        )
        emb = GPTEmbedding(
            sublayers_spec=spec,
            config=mock_config,
            vocab_size=1024,
            max_sequence_length=64,
            position_embedding_type="rope",
        )
        self.assertIsNotNone(emb.rotary_pos_emb)
        self.assertEqual(mock_build.call_count, 2)


class TestGPTEmbeddingEmbeddingWeight(unittest.TestCase):
    """Test GPTEmbedding.embedding_weight property."""

    @patch("paddlefleet.models.gpt.gpt_embedding.build_spec_layer")
    def test_embedding_weight_delegates(self, mock_build):
        from paddlefleet.models.gpt.gpt_embedding import (
            GPTEmbedding,
            GPTEmbeddingSpec,
        )

        mock_config = MagicMock()
        mock_config.sequence_parallel = False
        mock_config.multimodal_embedding = False
        mock_config.num_nextn_predict_layers = None
        mock_config.mtp_load_weight_only = False
        mock_config.head_dim = 64
        mock_config.rotary_interleaved = False
        mock_embedding = MagicMock()
        mock_embedding.embedding_weight = MagicMock()
        mock_build.return_value = mock_embedding

        spec = GPTEmbeddingSpec(
            language_embedding=MagicMock(), rope_embedding=None
        )
        emb = GPTEmbedding(
            sublayers_spec=spec,
            config=mock_config,
            vocab_size=1024,
            max_sequence_length=64,
            position_embedding_type="learned_absolute",
        )
        self.assertEqual(emb.embedding_weight, mock_embedding.embedding_weight)


class TestGPTEmbeddingBuildScheduleNode(unittest.TestCase):
    """Test GPTEmbedding.build_schedule_node method."""

    @patch("paddlefleet.models.gpt.gpt_embedding.build_spec_layer")
    def test_returns_schedule_node(self, mock_build):
        from paddlefleet.models.gpt.gpt_embedding import (
            GPTEmbedding,
            GPTEmbeddingSpec,
        )

        mock_config = MagicMock()
        mock_config.sequence_parallel = False
        mock_config.multimodal_embedding = False
        mock_config.num_nextn_predict_layers = None
        mock_config.mtp_load_weight_only = False
        mock_config.head_dim = 64
        mock_config.rotary_interleaved = False
        mock_build.return_value = MagicMock()

        spec = GPTEmbeddingSpec(
            language_embedding=MagicMock(), rope_embedding=None
        )
        emb = GPTEmbedding(
            sublayers_spec=spec,
            config=mock_config,
            vocab_size=1024,
            max_sequence_length=64,
            position_embedding_type="learned_absolute",
        )
        node = emb.build_schedule_node()
        self.assertIsNotNone(node)


class TestGPTEmbeddingGetPlaceholderMask(unittest.TestCase):
    """Test GPTEmbedding.get_placeholder_mask method."""

    @patch("paddlefleet.models.gpt.gpt_embedding.build_spec_layer")
    def test_placeholder_mask_with_input_ids(self, mock_build):
        import paddle

        from paddlefleet.models.gpt.gpt_embedding import (
            GPTEmbedding,
            GPTEmbeddingSpec,
        )

        mock_config = MagicMock()
        mock_config.sequence_parallel = False
        mock_config.multimodal_embedding = False
        mock_config.num_nextn_predict_layers = None
        mock_config.mtp_load_weight_only = False
        mock_config.head_dim = 64
        mock_config.rotary_interleaved = False
        mock_config.image_token_id = -200
        mock_config.video_token_id = -201
        mock_build.return_value = MagicMock()

        spec = GPTEmbeddingSpec(
            language_embedding=MagicMock(), rope_embedding=None
        )
        emb = GPTEmbedding.__new__(GPTEmbedding)
        emb.config = mock_config
        emb.embedding = MagicMock()
        emb.sequence_parallel = False

        input_ids = paddle.to_tensor([[1, -200, 3, -201, 5]])
        inputs_embeds = paddle.randn([1, 5, 64])
        image_mask, video_mask = emb.get_placeholder_mask(
            input_ids, inputs_embeds
        )
        self.assertEqual(image_mask.shape, inputs_embeds.shape)
        self.assertEqual(video_mask.shape, inputs_embeds.shape)

    @patch("paddlefleet.models.gpt.gpt_embedding.build_spec_layer")
    def test_placeholder_mask_image_size_mismatch_raises(self, mock_build):
        import paddle

        from paddlefleet.models.gpt.gpt_embedding import (
            GPTEmbedding,
            GPTEmbeddingSpec,
        )

        mock_config = MagicMock()
        mock_config.image_token_id = -200
        mock_config.video_token_id = -201
        mock_build.return_value = MagicMock()

        spec = GPTEmbeddingSpec(
            language_embedding=MagicMock(), rope_embedding=None
        )
        emb = GPTEmbedding.__new__(GPTEmbedding)
        emb.config = mock_config
        emb.embedding = MagicMock()
        emb.sequence_parallel = False

        input_ids = paddle.to_tensor([[1, -200, 3]])
        inputs_embeds = paddle.randn([1, 3, 64])
        # 1 image token * 64 hidden dim != 20 features
        with self.assertRaises(ValueError):
            emb.get_placeholder_mask(
                input_ids, inputs_embeds, image_features=paddle.randn([20])
            )


class TestGPTEmbeddingSequenceParallelInit(unittest.TestCase):
    """Test GPTEmbedding initialization with sequence parallel."""

    @patch("paddlefleet.models.gpt.gpt_embedding.build_spec_layer")
    def test_sp_with_multimodal_disables_scatter(self, mock_build):
        from paddlefleet.models.gpt.gpt_embedding import (
            GPTEmbedding,
            GPTEmbeddingSpec,
        )

        mock_config = MagicMock()
        mock_config.sequence_parallel = True
        mock_config.multimodal_embedding = True
        mock_config.num_nextn_predict_layers = None
        mock_config.mtp_load_weight_only = False
        mock_config.head_dim = 64
        mock_config.rotary_interleaved = False
        mock_embedding = MagicMock()
        mock_build.return_value = mock_embedding

        spec = GPTEmbeddingSpec(
            language_embedding=MagicMock(), rope_embedding=None
        )
        emb = GPTEmbedding(
            sublayers_spec=spec,
            config=mock_config,
            vocab_size=1024,
            max_sequence_length=64,
            position_embedding_type="rope",
        )
        self.assertTrue(emb.sequence_parallel)
        self.assertFalse(emb.embedding.reduce_scatter_embeddings)

    @patch("paddlefleet.models.gpt.gpt_embedding.build_spec_layer")
    def test_sp_with_mtp_disables_scatter(self, mock_build):
        from paddlefleet.models.gpt.gpt_embedding import (
            GPTEmbedding,
            GPTEmbeddingSpec,
        )

        mock_config = MagicMock()
        mock_config.sequence_parallel = True
        mock_config.multimodal_embedding = False
        mock_config.num_nextn_predict_layers = 2
        mock_config.mtp_load_weight_only = False
        mock_config.head_dim = 64
        mock_config.rotary_interleaved = False
        mock_embedding = MagicMock()
        mock_build.return_value = mock_embedding

        spec = GPTEmbeddingSpec(
            language_embedding=MagicMock(), rope_embedding=None
        )
        emb = GPTEmbedding(
            sublayers_spec=spec,
            config=mock_config,
            vocab_size=1024,
            max_sequence_length=64,
            position_embedding_type="learned_absolute",
        )
        self.assertTrue(emb.sequence_parallel)
        self.assertFalse(emb.embedding.reduce_scatter_embeddings)


if __name__ == "__main__":
    unittest.main()
