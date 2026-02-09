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

        assert overlap_loss._md5sum() == "5c76535a368b7145e13c391b8b85fa7b"

        if paddle.distributed.get_rank() == 0:
            baseline = {
                "_layers.shared_layers.embed.embedding.embed_tokens.weight": "bb3821fc881d12b218d9aef67b1702a4",
                "_layers.9.0.input_layernorm.weight": "04a61d23996e44d532c8e5898d78b76e",
                "_layers.9.0.self_attn.o_proj.weight": "791dfdbdb64f8fd1ae654ce75b812975",
                "_layers.9.0.self_attn.qkv_proj.weight": "dc65f81c89348aa022be026fc586844d",
                "_layers.9.0.self_attn.q_norm.weight": "b6c24d27a71f637680aa3a765404186a",
                "_layers.9.0.self_attn.k_norm.weight": "9d68710b26f19d3eccc224d3ae21a28f",
                "_layers.9.0.post_attention_layernorm.weight": "5a58de658380e49ca47501055f8c8263",
                "_layers.9.0.mlp.gate.weight": "02c006bea575c5dd31cacd2f7a358c85",
                "_layers.9.0.mlp.experts.0.up_gate_proj.weight": "c2c8e89c08a01439f508424c4a11c6f0",
                "_layers.9.0.mlp.experts.0.down_proj.weight": "f5f676ba681cfe0a35603b508aba922d",
                "_layers.9.0.mlp.experts.1.up_gate_proj.weight": "39a78aa947f2c3be1a7e695f5ca6965a",
                "_layers.9.0.mlp.experts.1.down_proj.weight": "73fe21465b29f0c3f3729ec9f0f9edbf",
                "_layers.9.0.mlp.experts.2.up_gate_proj.weight": "6711089925168b3b26f8e9141322291d",
                "_layers.9.0.mlp.experts.2.down_proj.weight": "3a71fb088d29eeaf8c5a4af206e6c6fc",
                "_layers.9.0.mlp.experts.3.up_gate_proj.weight": "18d1f254681c36a9c5c1263d877ccbcb",
                "_layers.9.0.mlp.experts.3.down_proj.weight": "e065bcd80d5f98c1af9fb033ef95e52d",
                "_layers.9.0.mlp.shared_experts.up_gate_proj.weight": "039eb7e8960684d18b3601df97535f77",
                "_layers.9.0.mlp.shared_experts.down_proj.weight": "2162cdb6223a28a102c4ec7923fe4cc4",
                "_layers.9.1.input_layernorm.weight": "b0bd7fdc3f2b17b51e05dd164de7db87",
                "_layers.9.1.self_attn.o_proj.weight": "40cc0fdf3c0303563dd38e4ba0cff485",
                "_layers.9.1.self_attn.qkv_proj.weight": "9cf203c969bbb9111e2fff75bc803b1f",
                "_layers.9.1.self_attn.q_norm.weight": "a95bd8d2e91484e297eededae5198423",
                "_layers.9.1.self_attn.k_norm.weight": "e89df9e3ad37d6897dd5eac48fa665c4",
                "_layers.9.1.post_attention_layernorm.weight": "23322f802772a039562cc556b832d8e0",
                "_layers.9.1.mlp.gate.weight": "468c43985a9b17b21372744b6a206b74",
                "_layers.9.1.mlp.experts.0.up_gate_proj.weight": "5495c66a47c139573e53fd342e7a0aff",
                "_layers.9.1.mlp.experts.0.down_proj.weight": "996532bcd091d3a562292380e4ee8012",
                "_layers.9.1.mlp.experts.1.up_gate_proj.weight": "02066817764998faa1727893ef236854",
                "_layers.9.1.mlp.experts.1.down_proj.weight": "f4b9e8653f52547cfac923dc22f52593",
                "_layers.9.1.mlp.experts.2.up_gate_proj.weight": "bd19cbe8defac9bd89baa230f5a5030f",
                "_layers.9.1.mlp.experts.2.down_proj.weight": "965d429cf8750d46590a4008ab554526",
                "_layers.9.1.mlp.experts.3.up_gate_proj.weight": "c1e3f70e1bdaebc7c19029faae0bef42",
                "_layers.9.1.mlp.experts.3.down_proj.weight": "1e505947b75e279e40d1fb688f73387d",
                "_layers.9.1.mlp.shared_experts.up_gate_proj.weight": "097e3572904ab9b1fc80258ec9d67fa6",
                "_layers.9.1.mlp.shared_experts.down_proj.weight": "da23902bb1f5c709ac5c2aaefa950725",
            }

            for name, param in overlap_gpt_model.named_parameters():
                assert param.grad._md5sum() == baseline[name]


if __name__ == "__main__":
    unittest.main()
