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

import random
from copy import deepcopy

import numpy as np
import paddle
from paddle.distributed import fleet

import paddlefleet.parallel_state as ps
from paddlefleet.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
)
from paddlefleet.transformer.moe.moe_layer import MoELayer
from paddlefleet.transformer.transformer_config import TransformerConfig


class TestMoELayerInit:
    def setup_method(self, method):
        print("00000000")
        # pass

    def test_moe_single_card_fusion(self):
        moe_num_experts = 4
        hidden_size = 30
        transformer_config_moe_use_fusion_node_true = TransformerConfig(
            hidden_size=hidden_size,
            num_attention_heads=4,
            moe_num_experts=moe_num_experts,
            use_cpu_initialization=True,
            num_experts_per_tok=2,
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
            sequence_parallel=False,
            bf16=True,
            params_dtype=paddle.bfloat16,
            moe_intermediate_size=19,
            moe_use_fusion_node=True,
        )
        transformer_config_moe_use_fusion_node_false = deepcopy(
            transformer_config_moe_use_fusion_node_true
        )
        transformer_config_moe_use_fusion_node_false.moe_use_fusion_node = False

        transformer_layer_spec = get_gpt_layer_local_spec(
            num_experts=moe_num_experts
        )
        print("aaaaaaaaaa")
        seed = 123
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)

        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": 1,
            "sep_degree": 1,
            "cp_degree": 1,
            "ep_degree": 1,
            "moe_sharding_degree": 1,
            "order": [
                "moe_sharding",
                "pp",
                "sep",
                "cp",
                "dp",
                "ep",
                "mp",
            ],
        }
        print("bbbbbbbbbbb")
        fleet.init(is_collective=True, strategy=strategy)
        hcg = fleet.get_hybrid_communicate_group()
        ps.initialize_model_parallel(hcg)

        print("cccccccccccc")

        moe_layer_moe_use_fusion_node_true = MoELayer(
            transformer_config_moe_use_fusion_node_true,
            transformer_layer_spec.sublayers_spec.mlp.extra_kwargs["sublayers"],
        ).cuda()
        moe_layer_moe_use_fusion_node_false = MoELayer(
            transformer_config_moe_use_fusion_node_false,
            transformer_layer_spec.sublayers_spec.mlp.extra_kwargs["sublayers"],
        ).cuda()

        input_data = paddle.randn(16, 4, hidden_size, dtype=paddle.bfloat16)
        print("dddddddddd")
        output_moe_use_fusion_node_true = moe_layer_moe_use_fusion_node_true(
            input_data
        )
        output_moe_use_fusion_node_false = moe_layer_moe_use_fusion_node_false(
            input_data
        )

        assert paddle.allclose(
            output_moe_use_fusion_node_true,
            output_moe_use_fusion_node_false,
            rtol=1e-2,
            atol=1e-2,
        )

    def teardown_method(self, method):
        pass
