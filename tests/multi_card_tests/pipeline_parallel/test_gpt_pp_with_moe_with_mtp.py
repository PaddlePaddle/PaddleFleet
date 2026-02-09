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
                "_layers.shared_layers.embed.embedding.embed_tokens.weight": "0da55f957c3eccc6803049ee6766be3b",
                "_layers.9.0.input_layernorm.weight": "219ce4996f356ef59e4b92b019e9b445",
                "_layers.9.0.self_attn.o_proj.weight": "abf199a4a8fb7c8ff42afb8740c83c06",
                "_layers.9.0.self_attn.qkv_proj.weight": "16dc0966272f5976df5328a54101bf4c",
                "_layers.9.0.self_attn.q_norm.weight": "b221e3d0d22950e5aefa8aab6e857f4f",
                "_layers.9.0.self_attn.k_norm.weight": "cc204a9e27ada35e03c643c69b860c16",
                "_layers.9.0.post_attention_layernorm.weight": "221526295e4ee920eaa53d2194aaa35b",
                "_layers.9.0.mlp.gate.weight": "941beae126522ec3b9a23e554a5ab3f3",
                "_layers.9.0.mlp.experts.0.up_gate_proj.weight": "cc6a459e28629ff63c8470058decb227",
                "_layers.9.0.mlp.experts.0.down_proj.weight": "9780c1f41049ec70192a0bd2d099f06e",
                "_layers.9.0.mlp.experts.1.up_gate_proj.weight": "3c2a60184d31fdb1d0f8822567e50d19",
                "_layers.9.0.mlp.experts.1.down_proj.weight": "0f7d1edf9251067f6de7db1c0f7ff4ca",
                "_layers.9.0.mlp.experts.2.up_gate_proj.weight": "12b9a42a459d8f3690e3b13765a24896",
                "_layers.9.0.mlp.experts.2.down_proj.weight": "768def433a43c2e8e6cdbcb1669085c8",
                "_layers.9.0.mlp.experts.3.up_gate_proj.weight": "8bfe633cb67bcf131937369d9199d5d7",
                "_layers.9.0.mlp.experts.3.down_proj.weight": "3f06721e769990c14dff32532dab6563",
                "_layers.9.0.mlp.shared_experts.up_gate_proj.weight": "a6a8b4540b6fea03329bb77ececa17d6",
                "_layers.9.0.mlp.shared_experts.down_proj.weight": "c05db68c62e2dfb08bcc52469b645539",
                "_layers.9.1.input_layernorm.weight": "1635d4f0f15672ba024f8b92015497f4",
                "_layers.9.1.self_attn.o_proj.weight": "75a3d6f11b4f1778239f2092b8d600e0",
                "_layers.9.1.self_attn.qkv_proj.weight": "98c6f3119701ffb56f2591468c69fa77",
                "_layers.9.1.self_attn.q_norm.weight": "35105779dc7416bc187bf17b78e941f7",
                "_layers.9.1.self_attn.k_norm.weight": "679393e6e1e699e43cc65a4313415f30",
                "_layers.9.1.post_attention_layernorm.weight": "c03e68a413b265e30090d0fcc95b4e6c",
                "_layers.9.1.mlp.gate.weight": "93c04931d89c20c27e0b1a9ba0321163",
                "_layers.9.1.mlp.experts.0.up_gate_proj.weight": "a8f53f2175838c707de4cef4c91741e5",
                "_layers.9.1.mlp.experts.0.down_proj.weight": "9e2b3cb1cbae0499da738bf1e53035c1",
                "_layers.9.1.mlp.experts.1.up_gate_proj.weight": "31c5c5109513020dfd3b2535f112c2b3",
                "_layers.9.1.mlp.experts.1.down_proj.weight": "d9393ea3723e741944ed4a811961c505",
                "_layers.9.1.mlp.experts.2.up_gate_proj.weight": "df7849a319d69d9950f458db047d3c5a",
                "_layers.9.1.mlp.experts.2.down_proj.weight": "1b5b01ccf91173489cc83991b8ed0c3a",
                "_layers.9.1.mlp.experts.3.up_gate_proj.weight": "0195812e7aae1b9038f7e9d22637ffa6",
                "_layers.9.1.mlp.experts.3.down_proj.weight": "0d3fc08d458e77e7ddea9014f745e1d5",
                "_layers.9.1.mlp.shared_experts.up_gate_proj.weight": "ba80c2baf97191dc54e99dbb072e427d",
                "_layers.9.1.mlp.shared_experts.down_proj.weight": "e7f5c24b18cd925e3936f3b0fa667558",
            }

            for name, param in overlap_gpt_model.named_parameters():
                assert param.grad._md5sum() == baseline[name]


if __name__ == "__main__":
    unittest.main()
