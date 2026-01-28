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
import os
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
REPO_FLAG = os.getenv("repo_flag")
BRANCH = os.getenv("BRANCH")
SKIP_TESTS = (REPO_FLAG != "paddlefleet") and (BRANCH == "develop")


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


@unittest.skipIf(
    SKIP_TESTS,
    f"Skipping tests: repo_flag={REPO_FLAG} (not 'paddlefleet') and branch '{BRANCH}' is 'develop'",
)
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

        overlap_loss, overlap_gpt_model = run_pp(
            self.seed,
            self.batch_size,
            self.seq_len,
            self.vocab_size,
            config,
            forward_backward_overlap_scheduler=False,
        )

        assert overlap_loss._md5sum() == "864e194f213e7cc5e825e847c91a557d"

        if paddle.distributed.get_rank() == 0:
            baseline = {
                "_layers.shared_layers.embed.embedding.embed_tokens.weight": "440891fc3db4d1c59c2f202b18e52143",
                "_layers.9.0.input_layernorm.weight": "18703ff23d92f274c5876c510a34b3b7",
                "_layers.9.0.self_attn.o_proj.weight": "e7340fb090482fb317372286d605c6ab",
                "_layers.9.0.self_attn.qkv_proj.weight": "5ed1219159d856efd64873e01e63057f",
                "_layers.9.0.self_attn.q_layernorm.weight": "b4090f76dee2fa7bc1e32e7f5e94983f",
                "_layers.9.0.self_attn.k_layernorm.weight": "30357faa73750d88c7a667eef4b23aee",
                "_layers.9.0.post_attention_layernorm.weight": "c97ef7bdfecb00cae35decaba029e570",
                "_layers.9.0.mlp.gate.weight": "e21b95a266d6d6c886c22fd2885b4f0d",
                "_layers.9.0.mlp.experts.0.up_gate_proj.weight": "f5f3a77a6894bacf5f37872cf3e9b1bc",
                "_layers.9.0.mlp.experts.0.down_proj.weight": "d5bdf61a07af74513dbcb88357bb4843",
                "_layers.9.0.mlp.experts.1.up_gate_proj.weight": "9a9d939312b69d03f6296c2f72b95b99",
                "_layers.9.0.mlp.experts.1.down_proj.weight": "654e853ab432ec5d3ad610a65c9c68df",
                "_layers.9.0.mlp.experts.2.up_gate_proj.weight": "f6b3f2b31d5e7c48a445eae73d26347b",
                "_layers.9.0.mlp.experts.2.down_proj.weight": "977a22323572578d913cd1fc06740c2e",
                "_layers.9.0.mlp.experts.3.up_gate_proj.weight": "33a09c7df8b11a17eed961efadbca7d6",
                "_layers.9.0.mlp.experts.3.down_proj.weight": "a0677c19c82fc7ccc07e893c7c6e6d55",
                "_layers.9.0.mlp.shared_experts.up_gate_proj.weight": "1adf951aba32b17ee508182913a5cc6c",
                "_layers.9.0.mlp.shared_experts.down_proj.weight": "223b14f8310d80518c75ec331c4c135e",
                "_layers.9.1.input_layernorm.weight": "6f1ab95e6184a69d7c11e75303d6184b",
                "_layers.9.1.self_attn.o_proj.weight": "7d22f8ad6dc6d005c14aafed94914a5f",
                "_layers.9.1.self_attn.qkv_proj.weight": "6400cd60b62ba5886c985d339028eb43",
                "_layers.9.1.self_attn.q_layernorm.weight": "c16a0a0eab44a99259553a0329f207e1",
                "_layers.9.1.self_attn.k_layernorm.weight": "e3e3a48d52a6d915ac99fcdc5b448f10",
                "_layers.9.1.post_attention_layernorm.weight": "a54559687a119c4fe634607eb569a51b",
                "_layers.9.1.mlp.gate.weight": "43783a795c7b3c8f42f3a613965b891f",
                "_layers.9.1.mlp.experts.0.up_gate_proj.weight": "7ac193a0fbbaa0de8373bfb92fa15ea2",
                "_layers.9.1.mlp.experts.0.down_proj.weight": "cb2f5d8f8e175e1e01dc5ed835b7ab46",
                "_layers.9.1.mlp.experts.1.up_gate_proj.weight": "b3450a32aa8c023a0852367224c62baa",
                "_layers.9.1.mlp.experts.1.down_proj.weight": "d6c777c758bc8e74157a7f8c5e31df37",
                "_layers.9.1.mlp.experts.2.up_gate_proj.weight": "0dbe679f15bc2468099e5541a749a509",
                "_layers.9.1.mlp.experts.2.down_proj.weight": "741d6ae336d339395ec26114278ad440",
                "_layers.9.1.mlp.experts.3.up_gate_proj.weight": "11309d52f5aae2be6b307ce98b828af3",
                "_layers.9.1.mlp.experts.3.down_proj.weight": "6aa557d21a2417c3be613c7b51c3228c",
                "_layers.9.1.mlp.shared_experts.up_gate_proj.weight": "93e9399e44e93a402758b6a9f3014150",
                "_layers.9.1.mlp.shared_experts.down_proj.weight": "5a1cfaca5e995d8838e01932351e3a8d",
            }

            for name, param in overlap_gpt_model.named_parameters():
                assert param.grad._md5sum() == baseline[name]


if __name__ == "__main__":
    unittest.main()
