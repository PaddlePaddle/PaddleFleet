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

        assert overlap_loss._md5sum() == "8754358fe1c39633fa49e91eaa3b34da"

        if paddle.distributed.get_rank() == 0:
            baseline = {'_layers.shared_layers.embed.embedding.embed_tokens.weight': 'ce0633bcee0941a2c1977f4035b7292b', '_layers.9.0.input_layernorm.weight': '461674e41d2836cc73e5831f666ab9f5', '_layers.9.0.self_attn.o_proj.weight': '59b5595fde968e4e52c81b41749f4696', '_layers.9.0.self_attn.qkv_proj.weight': 'd9761897132c2e7ce2c8be4ff7221fa8', '_layers.9.0.self_attn.q_norm.weight': 'b94afb136bf440b337ae94010c9e4c17', '_layers.9.0.self_attn.k_norm.weight': '300d085cac222d53cf9bb92be27189af', '_layers.9.0.post_attention_layernorm.weight': '196a243f3ed01cd25c99a4b7d5e7dcd6', '_layers.9.0.mlp.gate.weight': '678e72fa625c1a1c09fc05ccca068648', '_layers.9.0.mlp.experts.0.up_gate_proj.weight': '78d26d064b88431101c1cbfac14541d6', '_layers.9.0.mlp.experts.0.down_proj.weight': 'f112d158c20544df428bfff8352c748d', '_layers.9.0.mlp.experts.1.up_gate_proj.weight': 'ab57c494486f609ab08ee45971861cc9', '_layers.9.0.mlp.experts.1.down_proj.weight': '87cf26c2c82a5a7d49c9e97d6e5efba8', '_layers.9.0.mlp.experts.2.up_gate_proj.weight': 'c90322445a61a7312527bf66c094d201', '_layers.9.0.mlp.experts.2.down_proj.weight': '50facb409710679e9934855bfcb8ea44', '_layers.9.0.mlp.experts.3.up_gate_proj.weight': '4abc089b00b8db3afd067874b5cdd0d4', '_layers.9.0.mlp.experts.3.down_proj.weight': '343e214a069204bc86adb9d9cb06b6d1', '_layers.9.0.mlp.shared_experts.up_gate_proj.weight': '8314b680ba97d723a942aca6391855f8', '_layers.9.0.mlp.shared_experts.down_proj.weight': 'ad69b07d0074115b2725e7590cff0b43', '_layers.9.1.input_layernorm.weight': 'e92ff13502c8e70af5a65cc923d90b34', '_layers.9.1.self_attn.o_proj.weight': '40cfd3888bea600770613555a3267a84', '_layers.9.1.self_attn.qkv_proj.weight': 'e2f896440d1928a72789e63f1cb87c14', '_layers.9.1.self_attn.q_norm.weight': '05e5b9a7e71699833013ab8db7835473', '_layers.9.1.self_attn.k_norm.weight': 'cc2442b1e266ad7d1d9d1b39527fc471', '_layers.9.1.post_attention_layernorm.weight': 'dc00fe16047cb84cfebc42662a8d8bb0', '_layers.9.1.mlp.gate.weight': 'd428dd107048be5db2d8432d38cbff30', '_layers.9.1.mlp.experts.0.up_gate_proj.weight': '53a1e5a8cf7d90c4fd0c73604e2bbc8d', '_layers.9.1.mlp.experts.0.down_proj.weight': '49bde2326fdadf67df3e1834dc1dc09a', '_layers.9.1.mlp.experts.1.up_gate_proj.weight': 'fb706cc9500a3ee4542bfcced88a4cd0', '_layers.9.1.mlp.experts.1.down_proj.weight': '7c88534136ffe7767d76b3c6bfeb2e75', '_layers.9.1.mlp.experts.2.up_gate_proj.weight': '3fee1850ae5cc3fdff09c2d50479d6fc', '_layers.9.1.mlp.experts.2.down_proj.weight': 'ef1a77fcb4cac9de38a7f7e874c0a199', '_layers.9.1.mlp.experts.3.up_gate_proj.weight': '97e9e22f7542fd3e132ad5e9c3885123', '_layers.9.1.mlp.experts.3.down_proj.weight': '924cd4162180eacff56e936da3d62539', '_layers.9.1.mlp.shared_experts.up_gate_proj.weight': '4dee5ee86a9bdbfbf0597c832aa03782', '_layers.9.1.mlp.shared_experts.down_proj.weight': 'b5a313887c62c1f27fa9ffe11da69cb2'}
            assert rst == baseline


if __name__ == "__main__":
    unittest.main()
