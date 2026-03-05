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
                "_layers.shared_layers.embed.embedding.embed_tokens.weight": "6c94eec7c9fd2c012999b96fb40bb776",
                "_layers.9.0.input_layernorm.weight": "6976bdb72002c64043098b34102f332b",
                "_layers.9.0.self_attn.o_proj.weight": "54204a160ebb96c7df28bab35faebe97",
                "_layers.9.0.self_attn.qkv_proj.weight": "2e0c79cdc179ae83fd6ce4d6d0c76226",
                "_layers.9.0.self_attn.q_norm.weight": "71b81f49df17ae42fdd472f940468aaf",
                "_layers.9.0.self_attn.k_norm.weight": "330652f2596561c9e0015912508c22db",
                "_layers.9.0.post_attention_layernorm.weight": "c008226a46ec019e8823e2de6c71703c",
                "_layers.9.0.mlp.gate.weight": "fcb76704a1aeff66a9e9e89f3ea66b60",
                "_layers.9.0.mlp.experts.0.up_gate_proj.weight": "d31d422eefa2e41a5672ed342ed40d21",
                "_layers.9.0.mlp.experts.0.down_proj.weight": "223bf769dc86005e135354e6899f3224",
                "_layers.9.0.mlp.experts.1.up_gate_proj.weight": "3e75a51535dd06b3abf78d49b92e585c",
                "_layers.9.0.mlp.experts.1.down_proj.weight": "fcef2608a3fb79a5eb7dcbeb044b5795",
                "_layers.9.0.mlp.experts.2.up_gate_proj.weight": "ab2733628a23e3ef70bb5a245ed7f5d3",
                "_layers.9.0.mlp.experts.2.down_proj.weight": "496f745b47ff0a28e4530ac85a6a0e55",
                "_layers.9.0.mlp.experts.3.up_gate_proj.weight": "d9206f72be75d337bc1c67ea4e31a471",
                "_layers.9.0.mlp.experts.3.down_proj.weight": "ec4d135c0de794b0c3f809cd5003d238",
                "_layers.9.0.mlp.shared_experts.up_gate_proj.weight": "57f8cee9dc5fb831f50558d5a56583de",
                "_layers.9.0.mlp.shared_experts.down_proj.weight": "d37618391440663b6c3d2d744906825b",
                "_layers.9.1.input_layernorm.weight": "9ffb1882e97534e75fb55544f4ab664f",
                "_layers.9.1.self_attn.o_proj.weight": "f481670219c44861df964532e19c8ac3",
                "_layers.9.1.self_attn.qkv_proj.weight": "a0f187fcc1573aee1443c4212b59bdeb",
                "_layers.9.1.self_attn.q_norm.weight": "c3b600b4dae9a15ee28d15355b5ec673",
                "_layers.9.1.self_attn.k_norm.weight": "79c533d7364254fb942011ba28f5fdd2",
                "_layers.9.1.post_attention_layernorm.weight": "2821b9f026d9b1c71a351da9d7554627",
                "_layers.9.1.mlp.gate.weight": "82ffac490e191f46cf33dd926dbc7db7",
                "_layers.9.1.mlp.experts.0.up_gate_proj.weight": "e88475ddaaf7d2f01256fe937a238152",
                "_layers.9.1.mlp.experts.0.down_proj.weight": "1f675cc941e93c79f2bf9371185cb859",
                "_layers.9.1.mlp.experts.1.up_gate_proj.weight": "58032c2c345c72361d17f84ac3b7ff3f",
                "_layers.9.1.mlp.experts.1.down_proj.weight": "a3904ff31f9f0fd7881c95c7b6f2a436",
                "_layers.9.1.mlp.experts.2.up_gate_proj.weight": "b5cfa9d6c8febd618f91ac2843d50a1c",
                "_layers.9.1.mlp.experts.2.down_proj.weight": "b2d1236c286a3c0704224fe4105eca49",
                "_layers.9.1.mlp.experts.3.up_gate_proj.weight": "52b052eb36712553acd5ff26715883ba",
                "_layers.9.1.mlp.experts.3.down_proj.weight": "2581f280f4c851bf7e3057de1929d441",
                "_layers.9.1.mlp.shared_experts.up_gate_proj.weight": "24975a3e8f4cf807618cf83ba0ecc095",
                "_layers.9.1.mlp.shared_experts.down_proj.weight": "ed4d1fc143e5dbc4ea28a450e5ee3f1a",
            }

            for name, param in overlap_gpt_model.named_parameters():
                assert param.grad._md5sum() == baseline[name]


if __name__ == "__main__":
    unittest.main()
