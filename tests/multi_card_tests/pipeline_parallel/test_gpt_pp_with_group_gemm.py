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
                "_layers.shared_layers.embed.embedding.embed_tokens.weight": "c2624748a2b9c90aa0c746689f4f1dc9",
                "_layers.9.0.input_layernorm.weight": "de84892a3ba9e7f036bf73aaa6956445",
                "_layers.9.0.self_attn.o_proj.weight": "3dd32b53f27903a5efa586ee4cca14fe",
                "_layers.9.0.self_attn.qkv_proj.weight": "d833de1eac8ab32d83d2e1007a6981f8",
                "_layers.9.0.self_attn.q_norm.weight": "6332c209f2b128e1e311ac7ee8f1b15a",
                "_layers.9.0.self_attn.k_norm.weight": "3b3ceb6b1cb3a7b6ed09bc9ec6cc9afe",
                "_layers.9.0.post_attention_layernorm.weight": "fa7691f9f55938ad0c6b8709adf3f247",
                "_layers.9.0.mlp.gate.weight": "282b2546f26129ccb334a157f58c2e92",
                "_layers.9.0.mlp.grouped_gemm_experts.weight1": "2b1780c6225d01031ed035ca4ccc2ee0",
                "_layers.9.0.mlp.grouped_gemm_experts.weight2": "6c6525dc1b79e0123af0d1935a03ecb9",
                "_layers.9.0.mlp.shared_experts.up_gate_proj.weight": "2078c29e6997e2326ab6c559146dd6ca",
                "_layers.9.0.mlp.shared_experts.down_proj.weight": "7f6179556802dd85f9adcc085fc69568",
                "_layers.9.1.input_layernorm.weight": "60cf18460fb7f62588dc515516caa66f",
                "_layers.9.1.self_attn.o_proj.weight": "b713e8f5d1dfdb3b36ea9503b56453d9",
                "_layers.9.1.self_attn.qkv_proj.weight": "673b45c5b3ef13685c6a8549a9343c1e",
                "_layers.9.1.self_attn.q_norm.weight": "750d890d59f7295a0743e77e3315c6b2",
                "_layers.9.1.self_attn.k_norm.weight": "2ba31e0a79bb97d9cfd5ac9d8a9e7ac1",
                "_layers.9.1.post_attention_layernorm.weight": "5e4976d3407b83630cf2438062ab0264",
                "_layers.9.1.mlp.gate.weight": "3aae740e2f075fe875582a7c1a014b65",
                "_layers.9.1.mlp.grouped_gemm_experts.weight1": "93639fe4d8250226fab0e4f1f1da2b07",
                "_layers.9.1.mlp.grouped_gemm_experts.weight2": "ce91b4bea0f069b8b87a04f3de38bb28",
                "_layers.9.1.mlp.shared_experts.up_gate_proj.weight": "ad39ec9879bca0afedc9a5b61c5971e2",
                "_layers.9.1.mlp.shared_experts.down_proj.weight": "48f23d143744eccd47ead088e4d48173",
            }

            for name, param in overlap_gpt_model.named_parameters():
                assert param.grad._md5sum() == baseline[name]


if __name__ == "__main__":
    unittest.main()
