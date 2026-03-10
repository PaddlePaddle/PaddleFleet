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
MTP_DEGREE = 3


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
            "forward_backward_overlap_scheduler": forward_backward_overlap_scheduler,
            "overlap_p2p_comm": True,
            "enable_dynamic_shape": True,
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
        low=0,
        high=vocab_size,
        shape=(micro_batch_size, seq_len + MTP_DEGREE + 1),
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
            gated_linear_unit=True,
            bias_activation_fusion=True,
            num_nextn_predict_layers=MTP_DEGREE,
        )

        overlap_loss, overlap_gpt_model = run_pp(
            self.seed,
            self.batch_size,
            self.seq_len,
            self.vocab_size,
            config,
        )

        print(overlap_loss._md5sum())

        rst = {}
        for name, param in overlap_gpt_model.named_parameters():
            if param.grad is not None:
                rst[name] = param.grad._md5sum()

        print(rst)

        assert overlap_loss._md5sum() == "bef8aebcd0e33875e5bfb418e70bc6a1"

        if paddle.distributed.get_rank() == 0:
            baseline = {
                "_layers.shared_layers.embed.embedding.embed_tokens.weight": "768904869d2bae8541aa275a2cf366e5",
                "_layers.9.0.input_layernorm.weight": "07978a523af73b2558fb1d049e9f8c75",
                "_layers.9.0.self_attn.o_proj.weight": "13154cdbb1e13a1a0778b3b96a8bc289",
                "_layers.9.0.self_attn.qkv_proj.weight": "de886dfd465778194148ea73d3c1eb8c",
                "_layers.9.0.self_attn.q_norm.weight": "b00146d06887a578800b0c0c2eb1a05f",
                "_layers.9.0.self_attn.k_norm.weight": "2f3e0941991da20e088edfd404b864c6",
                "_layers.9.0.post_attention_layernorm.weight": "1761ea24f06616e4b94cd9073a809c59",
                "_layers.9.0.mlp.gate.weight": "9f2d8c778a48f48d1f2ed0316ea975cf",
                "_layers.9.0.mlp.experts.0.up_gate_proj.weight": "f5c4d0fec9e4e6a52dda2dcfa59ac707",
                "_layers.9.0.mlp.experts.0.down_proj.weight": "d2dcd1a22d94ed4b8eca9f4d9706d885",
                "_layers.9.0.mlp.experts.1.up_gate_proj.weight": "7015e46aad675e854dc342a5f135acef",
                "_layers.9.0.mlp.experts.1.down_proj.weight": "c2c52ab8648be361d78e55e871d1fa10",
                "_layers.9.0.mlp.experts.2.up_gate_proj.weight": "2bd05ba8e0af2a54b05ec3041510ca36",
                "_layers.9.0.mlp.experts.2.down_proj.weight": "92b701b0d4a850c477464d8f6e9bb7ae",
                "_layers.9.0.mlp.experts.3.up_gate_proj.weight": "99a04253c09f9362146ce9adac7e4def",
                "_layers.9.0.mlp.experts.3.down_proj.weight": "38f1c41ba59908483ee42c54c8f3388c",
                "_layers.9.0.mlp.shared_experts.up_gate_proj.weight": "33010b26f7ef9f9ca17cd45c5f387631",
                "_layers.9.0.mlp.shared_experts.down_proj.weight": "63a4ad06e46c8aa4f5d0b973e9e5cd93",
                "_layers.9.1.input_layernorm.weight": "4e3b54c14058a1721b5ff2daa8861562",
                "_layers.9.1.self_attn.o_proj.weight": "9332447f4549dba1c56d4e0630980eef",
                "_layers.9.1.self_attn.qkv_proj.weight": "9d8baf0f7a5a92fb305cfac2254a51f7",
                "_layers.9.1.self_attn.q_norm.weight": "4b7de5d9738a3e30fb97625dc4fb16b0",
                "_layers.9.1.self_attn.k_norm.weight": "a4dc9712d3ac232269380adb7cd43368",
                "_layers.9.1.post_attention_layernorm.weight": "c0bbb12135d945016e37f1daed956e5a",
                "_layers.9.1.mlp.gate.weight": "7789a0e4b6fed339b7e0484404ec74db",
                "_layers.9.1.mlp.experts.0.up_gate_proj.weight": "d6c5e0240a3ef67906046fa7f89d85aa",
                "_layers.9.1.mlp.experts.0.down_proj.weight": "ba18edda2107750acd85e7e1a5019408",
                "_layers.9.1.mlp.experts.1.up_gate_proj.weight": "061ea7f1c1b875e70ba0719958d8d4f5",
                "_layers.9.1.mlp.experts.1.down_proj.weight": "58deece95a0ce59690042f123d1abd5e",
                "_layers.9.1.mlp.experts.2.up_gate_proj.weight": "4cc180f5e35d9438f1de1b07b79633be",
                "_layers.9.1.mlp.experts.2.down_proj.weight": "d1ff2bc6e14a7d620e88d753bde65ca7",
                "_layers.9.1.mlp.experts.3.up_gate_proj.weight": "96d335ead56b00b3f7f2e3023e2d8205",
                "_layers.9.1.mlp.experts.3.down_proj.weight": "888547a7c6b2e8b58635711240585d6d",
                "_layers.9.1.mlp.shared_experts.up_gate_proj.weight": "87b635b2353e5cbd94f362a9bdfd0df9",
                "_layers.9.1.mlp.shared_experts.down_proj.weight": "c27c801635f3bf200afd59e6d93f8a54",
            }

            assert rst == baseline


if __name__ == "__main__":
    unittest.main()
