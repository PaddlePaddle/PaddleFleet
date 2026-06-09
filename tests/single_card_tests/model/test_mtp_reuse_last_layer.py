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

"""Tests for MTP parameter reuse (mtp_reuse_last_layer) feature."""

import functools
import random
import unittest

import numpy as np
import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import NoPipelineParallel

import paddlefleet.parallel_state as ps
from paddlefleet.gpt_builders import gpt_builder
from paddlefleet.models.gpt import GPTConfig


class TestMTPReuseLastLayer(unittest.TestCase):
    """Tests for _alias_mtp_to_last_backbone_layer."""

    @classmethod
    def setUpClass(cls):
        seed = 47
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

        # Config with MTP and mtp_reuse_last_layer
        cls.config_reuse = GPTConfig(
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
            mtp_reuse_last_layer=True,
        )
        cls.model_reuse = gpt_builder(cls.config_reuse, num_stages=1)

    def _find_transformer_layers(self, model):
        from paddlefleet.transformer.transformer_layer import TransformerLayer
        layers = []
        for layer in model.run_function:
            if isinstance(layer, TransformerLayer):
                layers.append(layer)
        return layers

    def _find_mtp_layer(self, model):
        from paddlefleet.transformer.multi_token_prediction import (
            MultiTokenPredictionLayer,
        )
        for layer in model.run_function:
            if isinstance(layer, MultiTokenPredictionLayer):
                return layer
        return None

    def test_mtp_reuse_aliases_parameters(self):
        """When mtp_reuse_last_layer=True, MTP transformer params should alias backbone last layer."""
        from paddlefleet.transformer.transformer_layer import TransformerLayer
        from paddlefleet.transformer.multi_token_prediction import (
            MultiTokenPredictionLayer,
        )

        backbone_layers = self._find_transformer_layers(self.model_reuse)
        self.assertGreater(len(backbone_layers), 0)
        last_backbone = backbone_layers[-1]

        mtp_layer = self._find_mtp_layer(self.model_reuse)
        self.assertIsNotNone(mtp_layer, "Model should have MTP layer")

        # Get parameters
        backbone_params = dict(last_backbone.named_parameters())
        mtp_params = dict(mtp_layer.transformer_layer.named_parameters())

        # Check that at least some parameters are aliased (same tensor object)
        aliased_count = 0
        for name, mtp_param in mtp_params.items():
            if name in backbone_params:
                backbone_param = backbone_params[name]
                # Aliased means same underlying tensor
                if mtp_param is backbone_param:
                    aliased_count += 1

        self.assertGreater(
            aliased_count,
            0,
            "At least some MTP params should be aliased to backbone last layer",
        )

    def test_mtp_reuse_incompatible_moe_dense_mtp_raises(self):
        """MoE backbone + use_dense_mtp=True should raise ValueError."""
        config = GPTConfig(
            num_hidden_layers=2,
            hidden_size=256,
            vocab_size=64,
            max_sequence_length=32,
            num_attention_heads=4,
            moe_expert_fusion=False,
            intermediate_size=512,
            normalization="RMSNorm",
            hidden_dropout_prob=0.0,
            attention_dropout=0.0,
            n_routed_experts=4,
            moe_intermediate_size=512,
            moe_token_dispatcher_type="alltoall",
            n_shared_experts=1,
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
            use_dense_mtp=True,  # Dense MTP with MoE backbone
            mtp_reuse_last_layer=True,
        )
        with self.assertRaises(ValueError) as ctx:
            gpt_builder(config, num_stages=1)
        self.assertIn("Incompatible configuration", str(ctx.exception))

    def test_mtp_reuse_disabled_no_aliasing(self):
        """When mtp_reuse_last_layer=False, params should NOT be aliased."""
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
            mtp_reuse_last_layer=False,
        )
        model = gpt_builder(config, num_stages=1)

        from paddlefleet.transformer.transformer_layer import TransformerLayer
        from paddlefleet.transformer.multi_token_prediction import (
            MultiTokenPredictionLayer,
        )

        backbone_layers = [
            l for l in model.run_function
            if isinstance(l, TransformerLayer)
        ]
        mtp_layer = next(
            (l for l in model.run_function if isinstance(l, MultiTokenPredictionLayer)),
            None,
        )

        self.assertIsNotNone(mtp_layer)
        backbone_params = dict(backbone_layers[-1].named_parameters())
        mtp_params = dict(mtp_layer.transformer_layer.named_parameters())

        # Check that parameters are NOT aliased
        for name, mtp_param in mtp_params.items():
            if name in backbone_params:
                self.assertIsNot(
                    mtp_param,
                    backbone_params[name],
                    f"Param {name} should NOT be aliased when mtp_reuse_last_layer=False",
                )


if __name__ == "__main__":
    unittest.main()
