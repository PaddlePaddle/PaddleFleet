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

from paddle.distributed import fleet

# from tests.unit_tests.test_utilities import Utils
import fleet.core.parallel_state as ps

# from fleet.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from fleet.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from fleet.core.models.gpt.gpt_model import GPTModel
from fleet.core.transformer.transformer_config import TransformerConfig


class TestGPTModel(unittest.TestCase):
    def setUp(self):
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
        config = TransformerConfig(
            num_layers=2, hidden_size=12, num_attention_heads=4
        )
        transformer_layer_spec = get_gpt_layer_local_spec(
            num_experts=None,
            moe_grouped_gemm=False,
            qk_layernorm=True,
            multi_latent_attention=False,
            normalization=False,
        )
        pre_process = True
        post_process = True
        mtp_block_spec = None
        vp_stage = None
        self.model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=100,
            max_sequence_length=64,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=False,
            parallel_output=True,
            share_embeddings_and_output_weights=True,
            position_embedding_type="rope",
            rotary_percent=1.0,
            rotary_base=10000,
            rope_scaling=1.0,
            mtp_block_spec=mtp_block_spec,
            vp_stage=vp_stage,
        )

    def test_gpt_model(self):
        print(self.model)


if __name__ == "__main__":
    unittest.main()
