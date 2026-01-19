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

        assert overlap_loss._md5sum() == "6f0258cb6e53f336c0b31580c71d247a"

        if paddle.distributed.get_rank() == 0:
            baseline = {
                "_layers.shared_layers.embed.embedding.embed_tokens.weight": "835b53051447ceafe3587e07582f99a4",
                "_layers.9.0.input_layernorm.weight": "e547bbb55776fde8f3887a56a8398cfe",
                "_layers.9.0.self_attn.o_proj.weight": "13e48bd7cfe60f01b2e42c75d59910b6",
                "_layers.9.0.self_attn.qkv_proj.weight": "34aa85982747a5ff1f5c9f3c4f1e9338",
                "_layers.9.0.self_attn.q_layernorm.weight": "3e47d7c9821c4c5d8cc39bc7cd6e7afc",
                "_layers.9.0.self_attn.k_layernorm.weight": "86343e18e662b91946283f1e14a81924",
                "_layers.9.0.post_attention_layernorm.weight": "d4fc14b34cbc144e572d18178def8f67",
                "_layers.9.0.mlp.gate.weight": "13f372e34496aef8564a3a93c1a47663",
                "_layers.9.0.mlp.experts.0.up_gate_proj.weight": "5a22769db4cd9d73fa83627bcbeac0b3",
                "_layers.9.0.mlp.experts.0.down_proj.weight": "6d77d67d5cc3e2e64c405109bb85b79b",
                "_layers.9.0.mlp.experts.1.up_gate_proj.weight": "4142b55f373c040e322d8af50a74f977",
                "_layers.9.0.mlp.experts.1.down_proj.weight": "4d14ecce46621445366d2de97b9366e8",
                "_layers.9.0.mlp.experts.2.up_gate_proj.weight": "4b647ff21250e51a036f0c3710ab04df",
                "_layers.9.0.mlp.experts.2.down_proj.weight": "cdf1d64a8391057294cef6e27840377b",
                "_layers.9.0.mlp.experts.3.up_gate_proj.weight": "d6cc7c5930b43e6624a1008ea48c085f",
                "_layers.9.0.mlp.experts.3.down_proj.weight": "3a7a8088149c77b505919bc3073d8179",
                "_layers.9.0.mlp.shared_experts.up_gate_proj.weight": "622356ef204729bb3494442aabac8211",
                "_layers.9.0.mlp.shared_experts.down_proj.weight": "6e091b548884bbca82317e7d05a399f4",
                "_layers.9.1.input_layernorm.weight": "6eaca38c0d9fc067f754cc9c3f348966",
                "_layers.9.1.self_attn.o_proj.weight": "6fca6cca29bc4b5f7ae9d9a860a6d508",
                "_layers.9.1.self_attn.qkv_proj.weight": "968a8cfd3e3f49c78ffbff668cf5cbff",
                "_layers.9.1.self_attn.q_layernorm.weight": "ba15941d14339d4e7e56fcc20def0670",
                "_layers.9.1.self_attn.k_layernorm.weight": "6a1db726d41073b174938867aba7c046",
                "_layers.9.1.post_attention_layernorm.weight": "0bdd76206a27276351041381e5868dc9",
                "_layers.9.1.mlp.gate.weight": "fe71d21c57f2db7771d2a5eeeade6005",
                "_layers.9.1.mlp.experts.0.up_gate_proj.weight": "338f1ebc2e27ae3f5e67d300f7a5b293",
                "_layers.9.1.mlp.experts.0.down_proj.weight": "55e4155d15f5b2414bf090af749b5d91",
                "_layers.9.1.mlp.experts.1.up_gate_proj.weight": "91d8f19f626a35178b3506580ba8e175",
                "_layers.9.1.mlp.experts.1.down_proj.weight": "92522067dbcd224a2663f1fc128855ce",
                "_layers.9.1.mlp.experts.2.up_gate_proj.weight": "b5cfa9d6c8febd618f91ac2843d50a1c",
                "_layers.9.1.mlp.experts.2.down_proj.weight": "b2d1236c286a3c0704224fe4105eca49",
                "_layers.9.1.mlp.experts.3.up_gate_proj.weight": "aead4e0aaafbc57353bbb7e5aea17ac9",
                "_layers.9.1.mlp.experts.3.down_proj.weight": "0e55433f6d9031e10fa76f59fe0a9024",
                "_layers.9.1.mlp.shared_experts.up_gate_proj.weight": "ba55ae43fb845186d52f532aaf607144",
                "_layers.9.1.mlp.shared_experts.down_proj.weight": "c743cbd84941bf87fea253c1ef8a8c57",
            }

            for name, param in overlap_gpt_model.named_parameters():
                assert param.grad._md5sum() == baseline[name]


if __name__ == "__main__":
    unittest.main()
