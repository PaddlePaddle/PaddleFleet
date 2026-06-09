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

"""Tests for SWA on MTP layers - model builds correctly with SWA configs."""

import functools
import random
import unittest

import numpy as np
import paddle
from paddle.distributed import fleet

import paddlefleet.parallel_state as ps
from paddlefleet.gpt_builders import gpt_builder
from paddlefleet.models.gpt import GPTConfig


class TestMTPSWA(unittest.TestCase):
    """Tests for sliding window attention on MTP layers."""

    @classmethod
    def setUpClass(cls):
        seed = 50
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": 1,
            "sharding_degree": 1,
            "sep_degree": 1,
            "cp_degree": 1,
            "ep_degree": 1,
            "moe_sharding_degree": 1,
            "order": [
                "sharding",
                "moe_sharding",
                "pp",
                "sep",
                "cp",
                "dp",
                "ep",
                "mp",
            ],
        }
        fleet.init(is_collective=True, strategy=strategy)
        hcg = fleet.get_hybrid_communicate_group()
        ps.initialize_model_parallel(hcg)
        cls.strategy = strategy

    def test_mtp_window_size_model_builds(self):
        """Model should build successfully with mtp_window_size configured."""
        config = GPTConfig(
            num_hidden_layers=2,
            hidden_size=256,
            vocab_size=64,
            max_sequence_length=32,
            num_attention_heads=4,
            intermediate_size=512,
            normalization="RMSNorm",
            hidden_dropout_prob=0.0,
            attention_dropout=0.0,
            use_bias=False,
            rotary_percent=1.0,
            rotary_base=10000,
            rope_scaling=1.0,
            init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            output_layer_init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            tie_word_embeddings=True,
            num_nextn_predict_layers=1,
            use_dense_mtp=True,
            mtp_window_size=16,
        )
        model = gpt_builder(config, num_stages=1)
        self.assertIsNotNone(model)

    def test_backbone_sliding_window_model_builds(self):
        """Model should build successfully with sliding_window configured."""
        config = GPTConfig(
            num_hidden_layers=2,
            hidden_size=256,
            vocab_size=64,
            max_sequence_length=32,
            num_attention_heads=4,
            intermediate_size=512,
            normalization="RMSNorm",
            hidden_dropout_prob=0.0,
            attention_dropout=0.0,
            use_bias=False,
            rotary_percent=1.0,
            rotary_base=10000,
            rope_scaling=1.0,
            init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            output_layer_init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            tie_word_embeddings=True,
            sliding_window=(32, 0),
        )
        model = gpt_builder(config, num_stages=1)
        self.assertIsNotNone(model)

    def test_mtp_window_size_mtp_layer_exists(self):
        """MTP layer should exist when num_nextn_predict_layers > 0."""
        from paddlefleet.transformer.multi_token_prediction import (
            MultiTokenPredictionLayer,
        )
        
        config = GPTConfig(
            num_hidden_layers=2,
            hidden_size=256,
            vocab_size=64,
            max_sequence_length=32,
            num_attention_heads=4,
            intermediate_size=512,
            normalization="RMSNorm",
            hidden_dropout_prob=0.0,
            attention_dropout=0.0,
            use_bias=False,
            rotary_percent=1.0,
            rotary_base=10000,
            rope_scaling=1.0,
            init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            output_layer_init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            tie_word_embeddings=True,
            num_nextn_predict_layers=1,
            use_dense_mtp=True,
            mtp_window_size=16,
        )
        model = gpt_builder(config, num_stages=1)

        mtp_layer = next(
            (l for l in model.run_function if isinstance(l, MultiTokenPredictionLayer)),
            None,
        )
        self.assertIsNotNone(mtp_layer, "MTP layer should exist")


if __name__ == "__main__":
    unittest.main()
