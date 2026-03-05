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
            norm_topk_prob=False,
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
                "_layers.shared_layers.embed.embedding.embed_tokens.weight": "9ca08aca1b0d0c9baa4c7dcb5a3b09cc",
                "_layers.9.0.input_layernorm.weight": "89d6b6cd451ac90c449356bb49dfedbc",
                "_layers.9.0.self_attn.o_proj.weight": "7b1dc9804f644806c67b18c894187564",
                "_layers.9.0.self_attn.qkv_proj.weight": "f7d4fa68c188284d99554b4d7610a399",
                "_layers.9.0.self_attn.q_norm.weight": "29503b412d6025b5a158428c8b967a2b",
                "_layers.9.0.self_attn.k_norm.weight": "aae6fad3db7782a4d986a23c77632a4e",
                "_layers.9.0.post_attention_layernorm.weight": "b082d508c73df6960eb188dfeec9d60d",
                "_layers.9.0.mlp.gate.weight": "476406d990c412cd29ad67ab70c86ade",
                "_layers.9.0.mlp.grouped_gemm_experts.weight1": "b2186b44f3b2511122579c36c4bd663b",
                "_layers.9.0.mlp.grouped_gemm_experts.weight2": "20d7d3225c9f52c52b52a982006e8481",
                "_layers.9.0.mlp.shared_experts.up_gate_proj.weight": "0e2842ab12ab2e1a503fd7337a922222",
                "_layers.9.0.mlp.shared_experts.down_proj.weight": "1f09eb3ded670fe733eff65786270514",
                "_layers.9.1.input_layernorm.weight": "1497a2b44f6f6806d66a6ab9b8ec6cde",
                "_layers.9.1.self_attn.o_proj.weight": "c365438f1e7839c60f106e5c9ecac09c",
                "_layers.9.1.self_attn.qkv_proj.weight": "7e4ab11d95765fedfe476de90bc66657",
                "_layers.9.1.self_attn.q_norm.weight": "1519c0ce2dce5ba491319368c91836f8",
                "_layers.9.1.self_attn.k_norm.weight": "1cac550f934da31f8f4d8d728870eb47",
                "_layers.9.1.post_attention_layernorm.weight": "32aad93fa62bf97ad006bce2069fc29d",
                "_layers.9.1.mlp.gate.weight": "df3af6f5943b7f3888994ed1c52a0d8b",
                "_layers.9.1.mlp.grouped_gemm_experts.weight1": "ca83c67d1cba6c4b80cecd46ccb70d2b",
                "_layers.9.1.mlp.grouped_gemm_experts.weight2": "048e09c1fa7038638ed0163e926a233f",
                "_layers.9.1.mlp.shared_experts.up_gate_proj.weight": "0a706afa7618c07b2b83aeba38bdd286",
                "_layers.9.1.mlp.shared_experts.down_proj.weight": "3b60fd25f6872b06cdef9dd13df5c4d1",
            }

            for name, param in overlap_gpt_model.named_parameters():
                assert param.grad._md5sum() == baseline[name]


if __name__ == "__main__":
    unittest.main()
