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
from unittest.mock import MagicMock

import paddle

from paddlefleet.models.gpt.gpt_embedding import GPTEmbedding, GPTEmbeddingSpec


class TestGPTEmbeddingSpec(unittest.TestCase):
    """Test GPTEmbeddingSpec dataclass."""

    def test_spec_with_both_fields(self):
        """Test spec with both language and rope embedding."""
        mock_lang = MagicMock()
        mock_rope = MagicMock()
        spec = GPTEmbeddingSpec(
            language_embedding=mock_lang,
            rope_embedding=mock_rope,
        )
        self.assertEqual(spec.language_embedding, mock_lang)
        self.assertEqual(spec.rope_embedding, mock_rope)

    def test_spec_with_no_rope(self):
        """Test spec with None rope embedding."""
        mock_lang = MagicMock()
        spec = GPTEmbeddingSpec(
            language_embedding=mock_lang,
            rope_embedding=None,
        )
        self.assertEqual(spec.language_embedding, mock_lang)
        self.assertIsNone(spec.rope_embedding)


class TestGPTEmbeddingGetPlaceholderMask(unittest.TestCase):
    """Test GPTEmbedding.get_placeholder_mask method."""

    def _make_embedding(self):
        """Create a GPTEmbedding-like object with get_placeholder_mask."""
        emb = GPTEmbedding.__new__(GPTEmbedding)
        # Initialize Paddle Layer internals
        emb.__dict__.setdefault("_parameters", {})
        emb.__dict__.setdefault("_buffers", {})
        emb.__dict__.setdefault("_sub_layers", {})
        emb.__dict__.setdefault("_loaddict_holder", {})
        emb.__dict__.setdefault("_non_persistable_buffers", set())
        emb.__dict__.setdefault("_non_persistable_buffer_names_set", set())
        return emb

    def test_image_mask_with_matching_tokens(self):
        """Test get_placeholder_mask when image tokens match features."""
        emb = self._make_embedding()
        # Mock the config
        emb.config = MagicMock()
        emb.config.image_token_id = 1
        emb.config.video_token_id = 2
        # Mock the embedding sublayer
        emb.embedding = MagicMock()

        input_ids = paddle.to_tensor([[1, 1, 0, 0]])
        inputs_embeds = paddle.randn([1, 4, 8])
        image_features = paddle.randn([2, 8])  # 2 image tokens, hidden=8

        image_mask, video_mask = emb.get_placeholder_mask(
            input_ids,
            inputs_embeds,
            image_features=image_features,
        )
        self.assertIsNotNone(image_mask)
        self.assertIsNotNone(video_mask)
        # Image mask should be True at positions 0, 1
        self.assertTrue(image_mask.shape == inputs_embeds.shape)

    def test_video_mask_with_matching_tokens(self):
        """Test get_placeholder_mask when video tokens match features."""
        emb = self._make_embedding()
        emb.config = MagicMock()
        emb.config.image_token_id = 1
        emb.config.video_token_id = 2

        input_ids = paddle.to_tensor([[2, 2, 2, 0]])
        inputs_embeds = paddle.randn([1, 4, 8])
        video_features = paddle.randn([3, 8])

        image_mask, video_mask = emb.get_placeholder_mask(
            input_ids,
            inputs_embeds,
            video_features=video_features,
        )
        self.assertIsNotNone(image_mask)
        self.assertIsNotNone(video_mask)

    def test_mismatched_image_tokens_raises(self):
        """Test ValueError when image tokens don't match features."""
        emb = self._make_embedding()
        emb.config = MagicMock()
        emb.config.image_token_id = 1
        emb.config.video_token_id = 2

        input_ids = paddle.to_tensor([[1, 1, 0, 0]])
        inputs_embeds = paddle.randn([1, 4, 8])
        # 2 image tokens but only 1 feature vector
        image_features = paddle.randn([1, 8])

        with self.assertRaises(ValueError):
            emb.get_placeholder_mask(
                input_ids,
                inputs_embeds,
                image_features=image_features,
            )

    def test_mismatched_video_tokens_raises(self):
        """Test ValueError when video tokens don't match features."""
        emb = self._make_embedding()
        emb.config = MagicMock()
        emb.config.image_token_id = 1
        emb.config.video_token_id = 2

        input_ids = paddle.to_tensor([[2, 2, 0, 0]])
        inputs_embeds = paddle.randn([1, 4, 8])
        # 2 video tokens but only 1 feature vector
        video_features = paddle.randn([1, 8])

        with self.assertRaises(ValueError):
            emb.get_placeholder_mask(
                input_ids,
                inputs_embeds,
                video_features=video_features,
            )

    def test_no_image_or_video_features(self):
        """Test get_placeholder_mask with no features (just masks)."""
        emb = self._make_embedding()
        emb.config = MagicMock()
        emb.config.image_token_id = 1
        emb.config.video_token_id = 2

        input_ids = paddle.to_tensor([[1, 2, 0, 0]])
        inputs_embeds = paddle.randn([1, 4, 8])

        image_mask, video_mask = emb.get_placeholder_mask(
            input_ids,
            inputs_embeds,
        )
        self.assertIsNotNone(image_mask)
        self.assertIsNotNone(video_mask)


class TestGPTEmbeddingForwardPaths(unittest.TestCase):
    """Test GPTEmbedding forward method paths."""

    def test_forward_with_decoder_input(self):
        """Test forward with decoder_input provided."""
        emb = GPTEmbedding.__new__(GPTEmbedding)
        emb.__dict__.setdefault("_parameters", {})
        emb.__dict__.setdefault("_buffers", {})
        emb.__dict__.setdefault("_sub_layers", {})
        emb.__dict__.setdefault("_loaddict_holder", {})
        emb.__dict__.setdefault("_non_persistable_buffers", set())
        emb.__dict__.setdefault("_non_persistable_buffer_names_set", set())
        emb.config = MagicMock()
        emb.config.sequence_parallel = False
        emb.config.multimodal_embedding = False
        emb.config.expert_model_parallel_size = 1
        emb.config.num_nextn_predict_layers = 0
        emb.config.apply_rope_fusion = False
        emb.position_embedding_type = "none"
        emb.rotary_pos_emb = None
        emb.mrope_section = None
        emb.sequence_parallel = False

        mock_decoder_input = paddle.randn([2, 8, 64])
        result = emb.forward(
            dict_args={"input_ids": None},
            decoder_input=mock_decoder_input,
        )
        self.assertIn("hidden_states", result)
        self.assertTrue(
            paddle.allclose(result["hidden_states"], mock_decoder_input)
        )

    def test_forward_removes_none_values(self):
        """Test that forward removes None values from output dict."""
        emb = GPTEmbedding.__new__(GPTEmbedding)
        emb.__dict__.setdefault("_parameters", {})
        emb.__dict__.setdefault("_buffers", {})
        emb.__dict__.setdefault("_sub_layers", {})
        emb.__dict__.setdefault("_loaddict_holder", {})
        emb.__dict__.setdefault("_non_persistable_buffers", set())
        emb.__dict__.setdefault("_non_persistable_buffer_names_set", set())
        emb.config = MagicMock()
        emb.config.sequence_parallel = False
        emb.config.multimodal_embedding = False
        emb.config.expert_model_parallel_size = 1
        emb.config.num_nextn_predict_layers = 0
        emb.config.apply_rope_fusion = False
        emb.position_embedding_type = "none"
        emb.rotary_pos_emb = None
        emb.mrope_section = None
        emb.sequence_parallel = False

        mock_decoder_input = paddle.randn([2, 8, 64])
        result = emb.forward(
            dict_args={"input_ids": None, "attention_mask": None},
            decoder_input=mock_decoder_input,
        )
        # None values should be removed
        self.assertNotIn("attention_mask", result)
        self.assertIn("hidden_states", result)

    def test_forward_mtp_assertion(self):
        """Test forward raises when mtp params are inconsistent."""
        emb = GPTEmbedding.__new__(GPTEmbedding)
        emb.__dict__.setdefault("_parameters", {})
        emb.__dict__.setdefault("_buffers", {})
        emb.__dict__.setdefault("_sub_layers", {})
        emb.__dict__.setdefault("_loaddict_holder", {})
        emb.__dict__.setdefault("_non_persistable_buffers", set())
        emb.__dict__.setdefault("_non_persistable_buffer_names_set", set())
        emb.config = MagicMock()
        emb.config.sequence_parallel = False
        emb.config.multimodal_embedding = False
        emb.config.expert_model_parallel_size = 1
        emb.config.num_nextn_predict_layers = 0
        emb.config.apply_rope_fusion = False
        emb.position_embedding_type = "none"
        emb.rotary_pos_emb = None
        emb.mrope_section = None
        emb.sequence_parallel = False

        mock_decoder_input = paddle.randn([2, 8, 64])
        # Only one of the two mtp params is set - should raise
        with self.assertRaises(AssertionError):
            emb.forward(
                dict_args={
                    "input_ids": None,
                    "mtp_startend_row_indices_all": paddle.randn([2, 8]),
                },
                decoder_input=mock_decoder_input,
            )


class TestGPTEmbeddingBuildScheduleNode(unittest.TestCase):
    """Test GPTEmbedding.build_schedule_node method."""

    def test_build_schedule_node(self):
        """Test build_schedule_node returns ScheduleNode."""
        from paddle.distributed.fleet.meta_parallel import ScheduleNode

        emb = GPTEmbedding.__new__(GPTEmbedding)
        node = emb.build_schedule_node()
        self.assertIsInstance(node, ScheduleNode)


if __name__ == "__main__":
    unittest.main()
