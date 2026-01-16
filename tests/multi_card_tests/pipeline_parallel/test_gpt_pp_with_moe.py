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

        overlap_loss, overlap_gpt_model = run_pp(
            self.seed,
            self.batch_size,
            self.seq_len,
            self.vocab_size,
            config,
            forward_backward_overlap_scheduler=True,
        )

        assert overlap_loss._md5sum() == "bef8aebcd0e33875e5bfb418e70bc6a1"

        if paddle.distributed.get_rank() == 0:
            baseline = {
                "_layers.shared_layers.embed.embedding.embed_tokens.weight": "1811c78c4265d2006a0394818341afbb",
                "_layers.9.0.input_layernorm.weight": "ab20c42b563b8bdf3305d91f04c010cb",
                "_layers.9.0.self_attn.o_proj.weight": "d5a6b3ca8856c4cb0578fba4d90a2392",
                "_layers.9.0.self_attn.qkv_proj.weight": "c170edab2c713ecd51652a60c07345e2",
                "_layers.9.0.self_attn.q_layernorm.weight": "8c293721e96e3dd2699363499067c142",
                "_layers.9.0.self_attn.k_layernorm.weight": "8c42d77cbcd0b2c9dc2cdb80119b23e2",
                "_layers.9.0.post_attention_layernorm.weight": "69f1320de4f8d0de93e07e8cb8b7c974",
                "_layers.9.0.mlp.gate.weight": "eeee1783f338c36487d860e104ddcad1",
                "_layers.9.0.mlp.experts.0.up_gate_proj.weight": "dd08760f66d71a7f059b5042e3ffd242",
                "_layers.9.0.mlp.experts.0.down_proj.weight": "79d6f53d36a9f6bd07a266075fd3f275",
                "_layers.9.0.mlp.experts.1.up_gate_proj.weight": "9291654cfbd8f5f18b589dfa9164cc3e",
                "_layers.9.0.mlp.experts.1.down_proj.weight": "d97f64ed4beb2ce1239abc85fff62524",
                "_layers.9.0.mlp.experts.2.up_gate_proj.weight": "b4dc4befb24ed1c4ed370f90fddbe103",
                "_layers.9.0.mlp.experts.2.down_proj.weight": "55e02fe89779d454974c020a024e579e",
                "_layers.9.0.mlp.experts.3.up_gate_proj.weight": "1fdb29b80455b83511fd2b101442f5b7",
                "_layers.9.0.mlp.experts.3.down_proj.weight": "a83b16829f7a5d7ca197bbae73d15603",
                "_layers.9.0.mlp.shared_experts.up_gate_proj.weight": "eb983bc78e9fe3a4af1f443b949359ff",
                "_layers.9.0.mlp.shared_experts.down_proj.weight": "14a4de251d0992175a6b4f742038fe1c",
                "_layers.9.1.input_layernorm.weight": "0903547f48ec93abdf16c20ace6985eb",
                "_layers.9.1.self_attn.o_proj.weight": "a553b22e692ad1465fdb01353d99cc99",
                "_layers.9.1.self_attn.qkv_proj.weight": "079e1bc75cde97c2fce25f0ce508d93c",
                "_layers.9.1.self_attn.q_layernorm.weight": "f013cf24c67a5450c3e8236b00a30612",
                "_layers.9.1.self_attn.k_layernorm.weight": "9b5d15731d5b254fedcbcb3327c9afda",
                "_layers.9.1.post_attention_layernorm.weight": "12aeec5f769d3dc18e40d70d07941379",
                "_layers.9.1.mlp.gate.weight": "a6e890e5e2fbfd81c30b99bacd9f679c",
                "_layers.9.1.mlp.experts.0.up_gate_proj.weight": "84f35708184105428735d534ea4bc16e",
                "_layers.9.1.mlp.experts.0.down_proj.weight": "32e1d52c32409150a0dfe8b6a09d5c9a",
                "_layers.9.1.mlp.experts.1.up_gate_proj.weight": "4e056f1e9fc33c3228af210ccb7e85d8",
                "_layers.9.1.mlp.experts.1.down_proj.weight": "165c11578cb2521e3a6a108ca3d9e7ad",
                "_layers.9.1.mlp.experts.2.up_gate_proj.weight": "57568e107e2142210bcfe913891cc09c",
                "_layers.9.1.mlp.experts.2.down_proj.weight": "11f0ea48085081008dce20f9785562e6",
                "_layers.9.1.mlp.experts.3.up_gate_proj.weight": "03fbf054ddf55756a617d5e807cf8594",
                "_layers.9.1.mlp.experts.3.down_proj.weight": "4301bb013f0a2d329f4a986c20077ddc",
                "_layers.9.1.mlp.shared_experts.up_gate_proj.weight": "f96587fde8da22ea1003deb7961e82c2",
                "_layers.9.1.mlp.shared_experts.down_proj.weight": "c7c50729a8177b7ec882f727daf63616",
            }

            for name, param in overlap_gpt_model.named_parameters():
                assert param.grad._md5sum() == baseline[name]


if __name__ == "__main__":
    unittest.main()
