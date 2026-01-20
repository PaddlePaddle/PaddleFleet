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

import numpy as np
import paddle
from paddle.distributed import fleet

import paddlefleet
from paddlefleet.distributed.model import distributed_model
from paddlefleet.gpt_builders import gpt_builder
from paddlefleet.models.gpt import GPTConfig
from paddlefleet.training.initialize import initialize_fleet

PP_DEGREE = 4


def _set_random_seed(
    seed_: int,
    data_parallel_random_init: bool = False,
    te_rng_tracker: bool = False,
    inference_rng_tracker: bool = False,
    use_cudagraphable_rng: bool = False,
):
    """Set random seed for reproducibility."""
    if seed_ is not None and seed_ > 0:
        # Ensure that different pipeline MP stages get different seeds.
        seed = seed_ + (
            100 * paddlefleet.parallel_state.get_pipeline_model_parallel_rank()
        )
        # Ensure different data parallel ranks get different seeds
        if data_parallel_random_init:
            seed = seed + (
                10 * paddlefleet.parallel_state.get_data_parallel_rank()
            )
        random.seed(seed)
        np.random.seed(seed)
        paddle.manual_seed(seed)

        if (
            paddle.distributed.is_initialized()
            and paddle.cuda.device_count() > 0
        ):
            paddlefleet.tensor_parallel.model_parallel_cuda_manual_seed(
                seed,
                te_rng_tracker,
                inference_rng_tracker,
                use_cudagraphable_rng,
            )
    else:
        raise ValueError(f"Seed ({seed_}) should be a positive integer.")


def run_pp(
    seed,
    batch_size,
    seq_len,
    vocab_size,
    config,
    forward_backward_overlap_scheduler=False,
):
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": config.tensor_model_parallel_size,
        "pp_degree": config.pipeline_model_parallel_size,
        "sharding_degree": 1,
        "sep_degree": 1,
        "cp_degree": 1,
        "ep_degree": config.tensor_model_parallel_size,
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
        "pp_configs": {
            "forward_backward_overlap_scheduler": forward_backward_overlap_scheduler
        },
    }
    micro_batch_size = 1
    num_acc = batch_size // micro_batch_size
    strategy.pipeline_configs = {
        "accumulate_steps": num_acc,
        "micro_batch_size": micro_batch_size,
    }
    initialize_fleet(strategy)

    _set_random_seed(seed)

    gpt_model = gpt_builder(
        config,
        num_stages=config.pipeline_model_parallel_size,
        seg_method="layer:TransformerLayer|EmptyLayer",
    )
    gpt_model = paddle.amp.decorate(
        models=gpt_model, optimizers=None, level="O2", dtype="bfloat16"
    )

    gpt_pipe_model = distributed_model(gpt_model)

    data = paddle.randint(
        low=0, high=vocab_size, shape=(micro_batch_size, seq_len + 1)
    )
    input_ids = data[:, :-1]
    labels = data[:, 1:]
    position_ids = paddle.to_tensor(data, dtype=paddle.int64).repeat(
        (micro_batch_size, 1)
    )

    inputs = (
        {
            "input_ids": [input_ids] * num_acc,
            "position_ids": [position_ids] * num_acc,
        },
        [labels] * num_acc,
    )

    loss = gpt_pipe_model.forward_backward_pipeline(inputs, None)
    return loss, gpt_pipe_model


class TestPP(unittest.TestCase):
    def setUp(self):
        self.seed = 46
        self.batch_size = 12
        self.seq_len = 128
        self.vocab_size = 1024

    def test_pp(self):
        if (
            not paddle.device.current_device_is_cpu
            and paddle.device.get_device_capability()[0] < 9
        ):
            return
        config = GPTConfig(
            vocab_size=self.vocab_size,
            max_sequence_length=self.seq_len,
            num_hidden_layers=11,
            hidden_size=512,
            num_attention_heads=4,
            intermediate_size=1024,
            normalization="RMSNorm",
            hidden_dropout_prob=0.0,
            attention_dropout=0.0,
            use_cpu_initialization=True,
            parallel_output=True,
            tie_word_embeddings=True,
            position_embedding_type="rope",
            rotary_percent=1.0,
            rotary_base=10000,
            rope_scaling=1.0,
            init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            output_layer_init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            use_qk_norm=True,
            num_empty_layers_add_in_head=2,
            num_empty_layers_add_in_tail=3,
            pipeline_model_parallel_size=PP_DEGREE,
            virtual_pipeline_model_parallel_size=2,
            tensor_model_parallel_size=2,
            expert_model_parallel_size=2,
            sequence_parallel=True,
            n_shared_experts=1,
            n_routed_experts=8,
            moe_intermediate_size=1024,
            bf16=True,
            moe_token_dispatcher_type="deepep",
            gated_linear_unit=True,
            bias_activation_fusion=True,
        )
        config.moe_grouped_gemm = True
        config.moe_deep_gemm = True

        overlap_loss, overlap_gpt_model = run_pp(
            self.seed,
            self.batch_size,
            self.seq_len,
            self.vocab_size,
            config,
            forward_backward_overlap_scheduler=True,
        )

        assert overlap_loss._md5sum() == "6961acbcfafaca51949b9a6eba287d37"

        if paddle.distributed.get_rank() == 0:
            baseline = {
                "_layers.shared_layers.embed.embedding.embed_tokens.weight": "4f89fe1034b7c71906f1b943087e7143",
                "_layers.9.0.input_layernorm.weight": "acd6807e9e8dd4ead7362f569825fc8e",
                "_layers.9.0.self_attn.o_proj.weight": "c15ae05a0f67b67afd89692d517210eb",
                "_layers.9.0.self_attn.qkv_proj.weight": "868817a8321166bb18efbd47d8965ecb",
                "_layers.9.0.self_attn.q_layernorm.weight": "5249b6b02ee71429729c9a3fbb4da4a3",
                "_layers.9.0.self_attn.k_layernorm.weight": "2e3d27bcb3191478f710fab2fbbd1d03",
                "_layers.9.0.post_attention_layernorm.weight": "f8f95512ba1ad3cd19b7eebc99408a7a",
                "_layers.9.0.mlp.gate.weight": "69d8a7f2e71d34bdc097bc10f7435c8f",
                "_layers.9.0.mlp.grouped_gemm_experts.weight1": "0eb9d63340cb07d357d25fa5d013a06d",
                "_layers.9.0.mlp.grouped_gemm_experts.weight2": "5af58ec3a15be20f7989aaed5421f893",
                "_layers.9.0.mlp.shared_experts.up_gate_proj.weight": "cc1c84c5c1b110047f9936f5cd137710",
                "_layers.9.0.mlp.shared_experts.down_proj.weight": "bd4de642cc76354a51feff3ad6dfde8f",
                "_layers.9.1.input_layernorm.weight": "a8dc1207a5b7ede308d6493dfb698379",
                "_layers.9.1.self_attn.o_proj.weight": "14e2f5a8f2fe74b28f1402d49cacef79",
                "_layers.9.1.self_attn.qkv_proj.weight": "573985996338bc73ba3f78d47098b218",
                "_layers.9.1.self_attn.q_layernorm.weight": "1f4f6b99ad5d7ac0ae82658ee976030d",
                "_layers.9.1.self_attn.k_layernorm.weight": "839b18e440f44c7442529ab6d86ccd74",
                "_layers.9.1.post_attention_layernorm.weight": "4aa87a276c1aa2f10585b32fec952812",
                "_layers.9.1.mlp.gate.weight": "e55f1baf60d91c66373362f31f721e70",
                "_layers.9.1.mlp.grouped_gemm_experts.weight1": "d5974217aad9ac3186b6ab852f915b84",
                "_layers.9.1.mlp.grouped_gemm_experts.weight2": "3a094fe77ffbafb7e1940090bc634eea",
                "_layers.9.1.mlp.shared_experts.up_gate_proj.weight": "aaebb2a61cdd711d6fb341a50eb1cc1a",
                "_layers.9.1.mlp.shared_experts.down_proj.weight": "7f793a83f4958cfc85f82bb4254295d8",
            }

            for name, param in overlap_gpt_model.named_parameters():
                assert param.grad._md5sum() == baseline[name]


if __name__ == "__main__":
    unittest.main()
