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

import functools
import random
import unittest
from unittest import mock

import numpy as np
import paddle
from paddle import nn
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import (
    LayerDesc,
    PipelineLayer,
    SharedLayerDesc,
)

# from tests.unit_tests.test_utilities import Utils
# from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddlefleet.gpt_builders import gpt_builder
from paddlefleet.models.gpt import GPTConfig, GPTModel


class TestGPTModelStateDict(unittest.TestCase):
    """Test cases for GPTModel state_dict and sharded_state_dict methods."""

    def setUp(self):
        seed = 46
        random.seed(seed)
        np.random.seed(seed)
        paddle.manual_seed(seed)
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
        # ps.initialize_model_parallel(hcg)
        self.strategy = strategy

        config = GPTConfig(
            num_hidden_layers=2,
            hidden_size=512,
            rotary_base=10000,
            vocab_size=100,
            rotary_percent=1.0,
            rope_scaling=1.0,
            position_embedding_type="rope",
            num_attention_heads=4,
            intermediate_size=1024,
            max_sequence_length=64,
            normalization="RMSNorm",
            hidden_dropout_prob=0.0,
            attention_dropout=0.0,
            init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            output_layer_init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            tie_word_embeddings=True,
            use_qk_norm=True,
        )
        self.gpt_model = gpt_builder(config, num_stages=1)
        self.config = config

    def test_state_dict_structure(self):
        """Test that state_dict returns parameters with correct naming structure."""
        # Create model instance
        self.gpt_model._set_pipeline_name_mapping()
        # Call state_dict directly without mocking
        state_dict = self.gpt_model.state_dict()
        print("state_dict: ", state_dict)
        # Verify all keys start with expected prefixes
        valid_prefixes = ("model.", "model.layers.")
        for key in state_dict.keys():
            self.assertTrue(
                key.startswith(valid_prefixes),
                f"Key '{key}' does not start with any of {valid_prefixes}",
            )

    def test_sharded_state_dict_structure(self):
        """Test that sharded_state_dict remaps parameter keys correctly."""
        self.gpt_model._set_pipeline_name_mapping()

        sharded_state_dict = self.gpt_model.sharded_state_dict()
        print("sharded_state_dict: ", sharded_state_dict)
        # Verify all keys start with expected prefixes
        valid_prefixes = ("model.", "model.layers.")
        for key in sharded_state_dict.keys():
            self.assertTrue(
                key.startswith(valid_prefixes),
                f"Key '{key}' does not start with any of {valid_prefixes}",
            )

    def test_check_shared_model_state(self):
        """Test _check_shared_model_state method."""
        self.gpt_model._check_shared_model_state()


class TestVirtualPipelineNameMapping(unittest.TestCase):
    """Test _set_pipeline_name_mapping for virtual pipeline parallelism.

    Under VPP, layers directly added to the PipelineLayer (e.g. lm_head) are
    named `{global_idx}.rest` while layers inside a chunk are named
    `{chunk_start}.{local_idx}.rest`. Both forms must map back to their single
    card names without collision.
    """

    # 4 layers split into 2 chunks of 2 layers, lm_head is the last layer.
    PREFIXES = {
        "0": "model.embedding",
        "1": "model.layers.0",
        "2": "model.layers.1",
        "3": "model.lm_head",
    }

    def _build_mapping(
        self,
        pp_keys,
        layers_desc=(),
        stage_id=0,
        index_to_stage=None,
        num_virtual_pipeline_stages=2,
    ):
        model = GPTModel.__new__(GPTModel)
        model.layers = list(layers_desc)
        model._stage_id = stage_id
        model._num_virtual_pipeline_stages = num_virtual_pipeline_stages
        model._use_dualpipev = False
        model.get_stage_from_index = lambda idx: (index_to_stage or {}).get(
            idx, stage_id
        )
        with (
            mock.patch.object(
                PipelineLayer,
                "state_dict",
                return_value=dict.fromkeys(pp_keys),
            ),
            mock.patch.object(
                GPTModel,
                "get_sequential_name_prefixes",
                return_value=self.PREFIXES,
            ),
        ):
            model._set_pipeline_name_mapping()
        return model._pp_to_single_mapping

    def test_directly_added_layer_keys_do_not_collide(self):
        pp_keys = [
            # chunk 0: `{chunk_start}.{local_idx}.rest`
            "0.0.word_embeddings.weight",
            "0.1.self_attn.o_proj.weight",
            # chunk 1 and the directly added lm_head, whose composite params
            # all share the `3.` prefix.
            "2.0.self_attn.o_proj.weight",
            "3.weight",
            "3.norm.weight",
            "3.enorm.weight",
            "3.transformer_layer.self_attn.o_proj.weight",
        ]
        mapping = self._build_mapping(pp_keys)

        self.assertEqual(
            mapping,
            {
                "0.0.word_embeddings.weight": "model.embedding.word_embeddings.weight",
                "0.1.self_attn.o_proj.weight": "model.layers.0.self_attn.o_proj.weight",
                "2.0.self_attn.o_proj.weight": "model.layers.1.self_attn.o_proj.weight",
                "3.weight": "model.lm_head.weight",
                "3.norm.weight": "model.lm_head.norm.weight",
                "3.enorm.weight": "model.lm_head.enorm.weight",
                "3.transformer_layer.self_attn.o_proj.weight": "model.lm_head.transformer_layer.self_attn.o_proj.weight",
            },
        )
        # every pipeline key keeps a unique single card name
        self.assertEqual(len(set(mapping.values())), len(pp_keys))

    def test_chunk_shared_layer_keys_follow_shared_layer_rule(self):
        # A SharedLayerDesc with `forward_func` is registered on the chunk
        # itself under VPP, so the same parameter shows up both as
        # `shared_layers.{name}.rest` and as `{chunk_start}.{name}.rest`. Both
        # aliases must resolve to the same single card name.
        layers_desc = [
            SharedLayerDesc(
                "embed_weight_share", nn.Linear, shared_weight_attr="weight"
            ),
            LayerDesc(nn.Linear),
            LayerDesc(nn.Linear),
            SharedLayerDesc(
                "embed_weight_share",
                nn.Linear,
                forward_func=lambda layer, x: x,
                shared_weight_attr="weight",
            ),
        ]
        pp_keys = [
            "shared_layers.embed_weight_share.weight",
            "1.0.self_attn.o_proj.weight",
            "3.embed_weight_share.weight",
        ]
        # stage 1 of a pp=2, vpp=2 run owns virtual stages 1 and 3
        mapping = self._build_mapping(
            pp_keys,
            layers_desc=layers_desc,
            stage_id=1,
            index_to_stage={0: 0, 1: 1, 2: 0, 3: 1},
        )

        self.assertEqual(
            mapping,
            {
                "shared_layers.embed_weight_share.weight": "model.lm_head.weight",
                "1.0.self_attn.o_proj.weight": "model.layers.0.self_attn.o_proj.weight",
                "3.embed_weight_share.weight": "model.lm_head.weight",
            },
        )

    def test_mapping_when_shared_key_comes_first(self):
        # If the first chunk registers the SharedLayerDesc with `forward_func`,
        # the first non `shared_layers` key is `{chunk_start}.{name}.rest`,
        # whose second segment is not a digit. The
        # `{chunk_start}.{local_idx}.rest` keys of the other chunks must still
        # be resolved as chunk keys.
        layers_desc = [
            SharedLayerDesc(
                "embed_weight_share",
                nn.Linear,
                forward_func=lambda layer, x: x,
                shared_weight_attr="weight",
            ),
            LayerDesc(nn.Linear),
            LayerDesc(nn.Linear),
            SharedLayerDesc(
                "embed_weight_share",
                nn.Linear,
                forward_func=lambda layer, x: x,
                shared_weight_attr="weight",
            ),
        ]
        pp_keys = [
            "shared_layers.embed_weight_share.weight",
            "0.embed_weight_share.weight",
            "2.0.self_attn.o_proj.weight",
        ]
        # stage 0 of a pp=2, vpp=2 run owns virtual stages 0 and 2
        mapping = self._build_mapping(
            pp_keys,
            layers_desc=layers_desc,
            stage_id=0,
            index_to_stage={0: 0, 1: 1, 2: 0, 3: 1},
        )

        self.assertEqual(
            mapping,
            {
                "shared_layers.embed_weight_share.weight": "model.embedding.weight",
                "0.embed_weight_share.weight": "model.embedding.weight",
                "2.0.self_attn.o_proj.weight": "model.layers.1.self_attn.o_proj.weight",
            },
        )

    def test_ordinary_pp_keeps_numeric_sublayer_names(self):
        # Without chunking, `LayerDesc(nn.Sequential, ...)` yields
        # `{global_idx}.{sublayer_idx}.rest`, which looks exactly like a chunk
        # key. The numeric sublayer name has to survive, so the chunked form
        # must not be inferred from the key shape.
        pp_keys = ["1.0.weight", "1.1.bias", "2.self_attn.o_proj.weight"]
        mapping = self._build_mapping(pp_keys, num_virtual_pipeline_stages=1)

        self.assertEqual(
            mapping,
            {
                "1.0.weight": "model.layers.0.0.weight",
                "1.1.bias": "model.layers.0.1.bias",
                "2.self_attn.o_proj.weight": "model.layers.1.self_attn.o_proj.weight",
            },
        )


if __name__ == "__main__":
    unittest.main()
