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
                "_layers.shared_layers.embed.embedding.embed_tokens.weight": "4430cf80b32c7928bc7358069fb08b88",
                "_layers.9.0.input_layernorm.weight": "6d2a9f7bf89f2a2db8d2be1f53c3c79a",
                "_layers.9.0.self_attn.o_proj.weight": "db1dd8e47571351eaaa90977a7125234",
                "_layers.9.0.self_attn.qkv_proj.weight": "1b379eadb3993c98647c162fff2d5979",
                "_layers.9.0.self_attn.q_norm.weight": "9b89cc2aec2afb49c038f32e888cbb30",
                "_layers.9.0.self_attn.k_norm.weight": "216886fe35d543704b3c6f1cb40331d8",
                "_layers.9.0.post_attention_layernorm.weight": "ae6a62eccb209803772c7c7169d447b3",
                "_layers.9.0.mlp.gate.weight": "b145d40a1a0f11871011476c67a4b076",
                "_layers.9.0.mlp.experts.0.up_gate_proj.weight": "467f6502fad70e6b8e5218b3860d1bcb",
                "_layers.9.0.mlp.experts.0.down_proj.weight": "2051f28ae03fb73707342d36d97189b6",
                "_layers.9.0.mlp.experts.1.up_gate_proj.weight": "fa6c71910c38e766334564ad0d8b4bfc",
                "_layers.9.0.mlp.experts.1.down_proj.weight": "d853e4d02bf8130e3cd5b8a9ee4713fa",
                "_layers.9.0.mlp.experts.2.up_gate_proj.weight": "b4df456a4fcaf500a3e9ef5433f8513c",
                "_layers.9.0.mlp.experts.2.down_proj.weight": "da31bde0705e3dea25d28a4631f91582",
                "_layers.9.0.mlp.experts.3.up_gate_proj.weight": "f6fba40952795f012d0a0f9ceb1c0e8d",
                "_layers.9.0.mlp.experts.3.down_proj.weight": "c26fd386ae330a0c77e0e92a857af06c",
                "_layers.9.0.mlp.shared_experts.up_gate_proj.weight": "66df4deafe638c52a270f3b69302d9a1",
                "_layers.9.0.mlp.shared_experts.down_proj.weight": "dedc825a026bd32628a8ca34b33a5e36",
                "_layers.9.1.input_layernorm.weight": "beae5ac67244843e6ccf9ffb2e04225c",
                "_layers.9.1.self_attn.o_proj.weight": "df89facf3dec4d3a3741df356017aff0",
                "_layers.9.1.self_attn.qkv_proj.weight": "0a24b11495cbb3f0ec702c481ef55f7b",
                "_layers.9.1.self_attn.q_norm.weight": "c20f48a7176f865884d1d2740134deb8",
                "_layers.9.1.self_attn.k_norm.weight": "a533052b59a550ad71d53e8b5f0d922c",
                "_layers.9.1.post_attention_layernorm.weight": "3a71a6683cc256b181e4806c774e0037",
                "_layers.9.1.mlp.gate.weight": "3a8bcaed7fb5d892a0d555215c4739a1",
                "_layers.9.1.mlp.experts.0.up_gate_proj.weight": "c138e408ac9ce33829a2da1935bab54a",
                "_layers.9.1.mlp.experts.0.down_proj.weight": "c13aa29a2b0ee2c6ebbf911ba47900dc",
                "_layers.9.1.mlp.experts.1.up_gate_proj.weight": "b85d1d998940f2371aec4a2d2a0b923e",
                "_layers.9.1.mlp.experts.1.down_proj.weight": "39c3ca996471d1ea69a76996c7a4ef1a",
                "_layers.9.1.mlp.experts.2.up_gate_proj.weight": "b5cfa9d6c8febd618f91ac2843d50a1c",
                "_layers.9.1.mlp.experts.2.down_proj.weight": "b2d1236c286a3c0704224fe4105eca49",
                "_layers.9.1.mlp.experts.3.up_gate_proj.weight": "18108c4fb3bb6105ada98e9c364a0806",
                "_layers.9.1.mlp.experts.3.down_proj.weight": "d1ca2402d435256b37dd70396a94efaa",
                "_layers.9.1.mlp.shared_experts.up_gate_proj.weight": "80a4c577e33c525c1167900e1fd6ef06",
                "_layers.9.1.mlp.shared_experts.down_proj.weight": "85d83703a127e9391ca61da9abe2a09d",
            }

            for name, param in overlap_gpt_model.named_parameters():
                assert param.grad._md5sum() == baseline[name]


if __name__ == "__main__":
    unittest.main()
