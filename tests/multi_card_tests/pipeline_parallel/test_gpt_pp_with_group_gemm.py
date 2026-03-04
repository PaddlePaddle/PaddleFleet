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
                "_layers.shared_layers.embed.embedding.embed_tokens.weight": "04dd13ff80c88c9e392c4243e739aae9",
                "_layers.9.0.input_layernorm.weight": "53d1efada1bc761755439941579145d1",
                "_layers.9.0.self_attn.o_proj.weight": "8d987521e86dc47c4c7c051c20a02775",
                "_layers.9.0.self_attn.qkv_proj.weight": "91d00572d9d1849fe5a94404c5511a89",
                "_layers.9.0.self_attn.q_norm.weight": "eec0339313cbbc97a86cf95466b7884b",
                "_layers.9.0.self_attn.k_norm.weight": "4d5ad8473731e43d3221c645aa76fd41",
                "_layers.9.0.post_attention_layernorm.weight": "3d1784e697f9f448d8255f68ec01430b",
                "_layers.9.0.mlp.gate.weight": "5d5a58a6285835ac8dc1115c68abe169",
                "_layers.9.0.mlp.grouped_gemm_experts.weight1": "0373982c1e1d2942938c7df509d002a4",
                "_layers.9.0.mlp.grouped_gemm_experts.weight2": "15973c679531e24c482a72368674d31c",
                "_layers.9.0.mlp.shared_experts.up_gate_proj.weight": "e1ef74c7239c6e9eb6afa8a4d885a8b1",
                "_layers.9.0.mlp.shared_experts.down_proj.weight": "6529321301e32861b19a440da81fc935",
                "_layers.9.1.input_layernorm.weight": "7d8e851581056cd47928991c127b253f",
                "_layers.9.1.self_attn.o_proj.weight": "9c43a7efe610481bec980365b2a5ac0f",
                "_layers.9.1.self_attn.qkv_proj.weight": "b039440b8f080c55f273e1b90377d22b",
                "_layers.9.1.self_attn.q_norm.weight": "2d6423ddaacd244fb7473e5604eb3d91",
                "_layers.9.1.self_attn.k_norm.weight": "9ba75143f25a424907d58d691ec7bed5",
                "_layers.9.1.post_attention_layernorm.weight": "c08892a2c5baeaa0d9e5161301a1f745",
                "_layers.9.1.mlp.gate.weight": "87c694afccc8bbd1e311a55c755460ba",
                "_layers.9.1.mlp.grouped_gemm_experts.weight1": "d0f758e336b86a9ce9138ad91617bdb9",
                "_layers.9.1.mlp.grouped_gemm_experts.weight2": "29eadba750e8bfbbbf1309526b966f90",
                "_layers.9.1.mlp.shared_experts.up_gate_proj.weight": "814abfbf9fc86f00350965bc4d0dd581",
                "_layers.9.1.mlp.shared_experts.down_proj.weight": "40b5b721c07d19f081e94b10179edc0a",
            }

            for name, param in overlap_gpt_model.named_parameters():
                assert param.grad._md5sum() == baseline[name]


if __name__ == "__main__":
    unittest.main()
