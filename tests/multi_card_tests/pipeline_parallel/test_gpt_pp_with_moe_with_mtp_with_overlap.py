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
            num_nextn_predict_layers=MTP_DEGREE,
        )

        overlap_loss, overlap_gpt_model = run_pp(
            self.seed,
            self.batch_size,
            self.seq_len,
            self.vocab_size,
            config,
            forward_backward_overlap_scheduler=True,
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
                "_layers.shared_layers.embed.embedding.embed_tokens.weight": "0fd18a7c329aa573dee527e5711327d9",
                "_layers.9.0.input_layernorm.weight": "c13503b5abe8632378c28282d10ec5f3",
                "_layers.9.0.self_attn.o_proj.weight": "615f97f4a497d4147d857bc96b94a6cd",
                "_layers.9.0.self_attn.qkv_proj.weight": "aec3cbde1a4062879c9c04fde4762734",
                "_layers.9.0.self_attn.q_layernorm.weight": "1b626027d68ce87f22d3bc565a029b45",
                "_layers.9.0.self_attn.k_layernorm.weight": "0048e85979759200eb1d1fea0e812213",
                "_layers.9.0.post_attention_layernorm.weight": "5cbef10d6043b8a21a087b8cce94332e",
                "_layers.9.0.mlp.gate.weight": "07074355d6f8c05e24f00ebd01f3b1a6",
                "_layers.9.0.mlp.experts.0.up_gate_proj.weight": "8a5f5e0e0dd0e8c2f4e1c1298f6e4a8b",
                "_layers.9.0.mlp.experts.0.down_proj.weight": "a3a3baa2e9ec50479900a8aa2c3afb0c",
                "_layers.9.0.mlp.experts.1.up_gate_proj.weight": "4e77262d9865d4cae333beedd0980bd2",
                "_layers.9.0.mlp.experts.1.down_proj.weight": "49062fc5705734c349bdb758dd52baf3",
                "_layers.9.0.mlp.experts.2.up_gate_proj.weight": "30d7c68127b6e48a57459e2f79655349",
                "_layers.9.0.mlp.experts.2.down_proj.weight": "3ac11382888aa35629cdf742a3c36a64",
                "_layers.9.0.mlp.experts.3.up_gate_proj.weight": "09255c8e9cceaa016786f0346b3eb88d",
                "_layers.9.0.mlp.experts.3.down_proj.weight": "074bc6b5e925479e248375b1f7c61819",
                "_layers.9.0.mlp.shared_experts.up_gate_proj.weight": "ec6bbc7eb6713c3934304e89c1997d5f",
                "_layers.9.0.mlp.shared_experts.down_proj.weight": "0a342daea3b42539ceb3c5ab74a00b0a",
                "_layers.9.1.input_layernorm.weight": "1b512c0828ec37ee305e8138729f9e17",
                "_layers.9.1.self_attn.o_proj.weight": "338e71c080bfb0636f9c8bfbe6671f05",
                "_layers.9.1.self_attn.qkv_proj.weight": "4b7781ddd1da28efb903259bdb5230c0",
                "_layers.9.1.self_attn.q_layernorm.weight": "0fa57cf85bcafe155992b19342c0c48a",
                "_layers.9.1.self_attn.k_layernorm.weight": "1da475565ece11eff68be7f4c7bc4952",
                "_layers.9.1.post_attention_layernorm.weight": "ecc3d8f930d81cb6dc836eecd03aa840",
                "_layers.9.1.mlp.gate.weight": "970849406933efcf79d5bff9164a3bcb",
                "_layers.9.1.mlp.experts.0.up_gate_proj.weight": "53c1386af9eb654109e6b26acf4c5f5a",
                "_layers.9.1.mlp.experts.0.down_proj.weight": "e058eb34a6f524b224bca4d7085505ed",
                "_layers.9.1.mlp.experts.1.up_gate_proj.weight": "7b869c7af380f3c248f32b3f17519c75",
                "_layers.9.1.mlp.experts.1.down_proj.weight": "39f9551985528f0707f9782630e14a60",
                "_layers.9.1.mlp.experts.2.up_gate_proj.weight": "bcb9e116006601fe0b69fe448d79a392",
                "_layers.9.1.mlp.experts.2.down_proj.weight": "7b5981e3a4e93b734640bfb57f8eed89",
                "_layers.9.1.mlp.experts.3.up_gate_proj.weight": "5eb2f609bab0f99528a3686f0405c15c",
                "_layers.9.1.mlp.experts.3.down_proj.weight": "8919c90ad630f64787ec963f08020e40",
                "_layers.9.1.mlp.shared_experts.up_gate_proj.weight": "8c8cb93e6e0bc3dfe26b124b1dd8c1d0",
                "_layers.9.1.mlp.shared_experts.down_proj.weight": "8d7d1ea5506a6c3f3811390f568ae8c0",
            }

            for name, param in overlap_gpt_model.named_parameters():
                assert param.grad._md5sum() == baseline[name]


if __name__ == "__main__":
    unittest.main()
