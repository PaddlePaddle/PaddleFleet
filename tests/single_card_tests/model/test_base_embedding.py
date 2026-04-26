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

import unittest
from unittest.mock import MagicMock

import paddle
from paddle.distributed import fleet

from paddlefleet.models.common.embeddings import LanguageModelEmbedding
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.utils import init_method_normal


class TestBaseEmbedding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fleet.init(is_collective=True)

    def setUp(self):
        config = TransformerConfig(
            num_hidden_layers=2, hidden_size=12, num_attention_heads=4
        )
        config.perform_initialization = True
        config.embedding_init_method = init_method_normal(1.0)
        config.hidden_dropout_prob = False
        config.fp32_residual_connection = False
        config.sequence_parallel = False
        self.base_embedding = LanguageModelEmbedding(
            config=config,
            vocab_size=100,
            max_sequence_length=4,
            position_embedding_type="learned_absolute",
        )

    def test_constructor(self):
        assert isinstance(self.base_embedding, LanguageModelEmbedding)
        num_weights = sum([p.numel() for p in self.base_embedding.parameters()])
        assert num_weights == 1248

    def test_zero_parameters(self):
        sum_weights = sum([p.sum() for p in self.base_embedding.parameters()])
        assert sum_weights != 0
        self.base_embedding.zero_parameters()
        sum_weights = sum([p.sum() for p in self.base_embedding.parameters()])
        assert sum_weights == 0

    def test_forward(self):
        input_ids = paddle.arange(8).reshape((2, 4))
        position_ids = paddle.arange(4).repeat((2, 1))
        embeddings = self.base_embedding(input_ids, position_ids)
        assert embeddings.place.is_gpu_place()
        assert embeddings.shape[0] == input_ids.shape[0]
        assert embeddings.shape[1] == self.base_embedding.max_sequence_length
        assert embeddings.shape[2] == self.base_embedding.config.hidden_size


class TestGPTEmbeddingFillFeatureBranch(unittest.TestCase):
    """Test fill_feature branch (lines 153-162) in GPTEmbedding.forward."""

    def test_forward_zeros_padding_and_sets_moe_mask(self):
        """Cover: fill_feature zeroes padding embeddings, input_ids_for_moe_mask is set."""
        from paddlefleet.models.gpt.gpt_embedding import GPTEmbedding

        B, S, H = 2, 4, 8
        # Bypass __init__ to avoid heavy distributed dependencies
        emb = object.__new__(GPTEmbedding)
        emb.sequence_parallel = False
        emb.multimodal_embedding = False
        emb.position_embedding_type = "none"
        emb.rotary_pos_emb = None
        emb.mrope_section = None
        emb.embedding = MagicMock(
            return_value=paddle.ones([B, S, H], dtype="float32")
        )
        cfg = MagicMock()
        cfg.expert_model_parallel_size = 2  # > 1 triggers the branch
        cfg.tensor_model_parallel_size = 1  # < 2 satisfies second condition
        cfg.num_nextn_predict_layers = None
        cfg.mtp_load_weight_only = False
        cfg.sequence_parallel = False
        emb.config = cfg

        input_ids = paddle.to_tensor([[1, 2, 0, 0], [3, 0, 5, 0]])
        result = emb.forward({"input_ids": input_ids})

        hidden = result["hidden_states"]
        self.assertEqual(hidden.shape, [B, S, H])
        # Padding positions (input_ids==0) should be zeroed
        self.assertAlmostEqual(hidden[0, 2, 0].item(), 0.0)
        self.assertAlmostEqual(hidden[0, 3, 0].item(), 0.0)
        self.assertAlmostEqual(hidden[1, 1, 0].item(), 0.0)
        self.assertAlmostEqual(hidden[1, 3, 0].item(), 0.0)
        # Valid positions should remain 1.0
        self.assertAlmostEqual(hidden[0, 0, 0].item(), 1.0)
        self.assertAlmostEqual(hidden[1, 2, 0].item(), 1.0)
        # input_ids_for_moe_mask is passed through as "input_ids"
        self.assertIn("input_ids", result)
        self.assertTrue(paddle.equal_all(result["input_ids"], input_ids).item())


if __name__ == "__main__":
    unittest.main()
