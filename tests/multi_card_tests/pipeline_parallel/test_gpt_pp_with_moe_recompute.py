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
        config.recompute_granularity = "full"
        config.recompute_method = "uniform"
        config.recompute_num_layers = 1

        overlap_loss, overlap_gpt_model = run_pp(
            self.seed,
            self.batch_size,
            self.seq_len,
            self.vocab_size,
            config,
            forward_backward_overlap_scheduler=True,
        )

        assert overlap_loss._md5sum() == "22b2ebefac4ef6fe74088a2370958f86"

        if paddle.distributed.get_rank() == 0:
            baseline = {
                "_layers.shared_layers.embed.embedding.embed_tokens.weight": "27c56dd638a75c582b001f24c53385b9",
                "_layers.9.0.input_layernorm.weight": "9dffb6719e9b674d4e13ac0b148e94a4",
                "_layers.9.0.self_attn.o_proj.weight": "7b24efe38704c22c9fb8558a6a2fdb9b",
                "_layers.9.0.self_attn.qkv_proj.weight": "06d9916ff6dd9b5f0d70d036648232d7",
                "_layers.9.0.self_attn.q_norm.weight": "f53733aad415aaa97cdc8b420a110260",
                "_layers.9.0.self_attn.k_norm.weight": "7ccc0060d0ad4a51de53ac0a0d313061",
                "_layers.9.0.post_attention_layernorm.weight": "f8a1d9531e48494ffc3752fb3f011ed0",
                "_layers.9.0.mlp.gate.weight": "24dcc48f00fad245d2cdf92b8c0b5061",
                "_layers.9.0.mlp.experts.0.up_gate_proj.weight": "ff9a9e8d424293b2086028947c5a54b5",
                "_layers.9.0.mlp.experts.0.down_proj.weight": "7742f0f7359243bd78d1a8625606e593",
                "_layers.9.0.mlp.experts.1.up_gate_proj.weight": "b30442632c86f0762b7e2f97d97f0f79",
                "_layers.9.0.mlp.experts.1.down_proj.weight": "b5ac31be541553048a86f7f492a88a54",
                "_layers.9.0.mlp.experts.2.up_gate_proj.weight": "04b679d96dd750c5df0f8008fc14e414",
                "_layers.9.0.mlp.experts.2.down_proj.weight": "d942a2e008047c659a55a2da53104fa4",
                "_layers.9.0.mlp.experts.3.up_gate_proj.weight": "59e12000858f7216ff4a08639404efcb",
                "_layers.9.0.mlp.experts.3.down_proj.weight": "17bc945d749eba4b276291f1adb29edc",
                "_layers.9.0.mlp.shared_experts.up_gate_proj.weight": "5a16ce3f7565c603a969e4c47b772a97",
                "_layers.9.0.mlp.shared_experts.down_proj.weight": "502c930aabfc7511ca97e0ce0453d22a",
                "_layers.9.1.input_layernorm.weight": "c26f29f0884f5774defc463c87a94fdc",
                "_layers.9.1.self_attn.o_proj.weight": "55d44e0d9d4dea2f4cc5808a5ac870b8",
                "_layers.9.1.self_attn.qkv_proj.weight": "ea72e4c7048bf039a9fb7a384aa0783d",
                "_layers.9.1.self_attn.q_norm.weight": "59db4499e68df12fe7686a7b53f2e4ef",
                "_layers.9.1.self_attn.k_norm.weight": "9b8e1bd27f9faf9c389d3dde966a3d41",
                "_layers.9.1.post_attention_layernorm.weight": "aa9dc7d3249390393f7c797bcb993346",
                "_layers.9.1.mlp.gate.weight": "0b916bff64e51233fa4a6fbbc4b2adc3",
                "_layers.9.1.mlp.experts.0.up_gate_proj.weight": "d045f3d4451561384d61635e2a668122",
                "_layers.9.1.mlp.experts.0.down_proj.weight": "aed496956da8878ea86c1fb481f57afb",
                "_layers.9.1.mlp.experts.1.up_gate_proj.weight": "97bfd7b391588a5202d15891acdc0563",
                "_layers.9.1.mlp.experts.1.down_proj.weight": "65694627203373d2187a64c0cbef6476",
                "_layers.9.1.mlp.experts.2.up_gate_proj.weight": "b5cfa9d6c8febd618f91ac2843d50a1c",
                "_layers.9.1.mlp.experts.2.down_proj.weight": "b2d1236c286a3c0704224fe4105eca49",
                "_layers.9.1.mlp.experts.3.up_gate_proj.weight": "cd795348867d528de8f16e8e1227e22c",
                "_layers.9.1.mlp.experts.3.down_proj.weight": "48beffe78fa38c6dc28871fe43c60ea5",
                "_layers.9.1.mlp.shared_experts.up_gate_proj.weight": "1b60d1cf8e0de5b2969e3cabf848e8a9",
                "_layers.9.1.mlp.shared_experts.down_proj.weight": "ed051da3d3b8a706dca6002d1e136239",
            }

            for name, param in overlap_gpt_model.named_parameters():
                assert param.grad._md5sum() == baseline[name]


if __name__ == "__main__":
    unittest.main()
