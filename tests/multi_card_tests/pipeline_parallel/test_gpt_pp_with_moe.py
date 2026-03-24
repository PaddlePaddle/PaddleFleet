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
import pprint
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

        print("Overlap PP loss MD5:", overlap_loss._md5sum())
        rst = {}
        for name, param in overlap_gpt_model.named_parameters():
            if param.grad is not None:
                rst[name] = param.grad._md5sum()

        pp = pprint.PrettyPrinter(depth=None, width=200, compact=False)
        pp.pprint(rst)

        assert overlap_loss._md5sum() == "d40256a233f15b6f3ea06dabc1d5b3ff"

        if paddle.distributed.get_rank() == 0:
            baseline = {
                "_layers.9.0.input_layernorm.weight": "1521496de2de6dd9fd7e5a295a76ae42",
                "_layers.9.0.mlp.experts.0.down_proj.weight": "a6123a0af26a4c668404225c855bf3b3",
                "_layers.9.0.mlp.experts.0.up_gate_proj.weight": "34a7079cff9d4ec8a9af7c72d53b1d27",
                "_layers.9.0.mlp.experts.1.down_proj.weight": "d855fce78b2ac796bd12d30087e9902f",
                "_layers.9.0.mlp.experts.1.up_gate_proj.weight": "4e7d9feb126e36d5953e143d8332318d",
                "_layers.9.0.mlp.experts.2.down_proj.weight": "1fa39f9998cf581b44e94c95f6f43472",
                "_layers.9.0.mlp.experts.2.up_gate_proj.weight": "83f310bc55d3cfa339c3ae6802584fd2",
                "_layers.9.0.mlp.experts.3.down_proj.weight": "31b86a15d14d9bf1612bcab7bc7ad4e5",
                "_layers.9.0.mlp.experts.3.up_gate_proj.weight": "f765b46624e7f978550423013dc79606",
                "_layers.9.0.mlp.gate.weight": "b0787c78abc120b815bfca50a51d5ce1",
                "_layers.9.0.mlp.shared_experts.down_proj.weight": "e0ac9dd447627770900685362dac0c4d",
                "_layers.9.0.mlp.shared_experts.up_gate_proj.weight": "22ee486f3315e623072bb4acb69faf8c",
                "_layers.9.0.post_attention_layernorm.weight": "56684b809b03d50749d664804e5a957f",
                "_layers.9.0.self_attn.k_norm.weight": "cd27eb47958d5250f9beb762c659e318",
                "_layers.9.0.self_attn.o_proj.weight": "9d9b5bc3dc820a311f36f13eef97dcda",
                "_layers.9.0.self_attn.q_norm.weight": "e7891f278a066f02f1abc5489c2550ab",
                "_layers.9.0.self_attn.qkv_proj.weight": "4bd9b32a69360c7e852619b399d85f06",
                "_layers.9.1.input_layernorm.weight": "0d4b5f01e28521ac69d5fa5028c6a8e5",
                "_layers.9.1.mlp.experts.0.down_proj.weight": "441387b73a62f35f0afb73879df497ab",
                "_layers.9.1.mlp.experts.0.up_gate_proj.weight": "13031b7e6a6dc46399db6f4147c84e58",
                "_layers.9.1.mlp.experts.1.down_proj.weight": "d1123e39ea83f91e3eba4a4312b17efe",
                "_layers.9.1.mlp.experts.1.up_gate_proj.weight": "06ba2f4e3e890911a999172011187c9e",
                "_layers.9.1.mlp.experts.2.down_proj.weight": "747705c30bf7b2cf7ae8f84fc707117b",
                "_layers.9.1.mlp.experts.2.up_gate_proj.weight": "05b4d9bb577c281d5d253aa1c612d2f1",
                "_layers.9.1.mlp.experts.3.down_proj.weight": "8a4909cf67e0c2ab7f6c9c33bdc907b3",
                "_layers.9.1.mlp.experts.3.up_gate_proj.weight": "25fcdde1a75c774b0ecd09bcbe1f2c2d",
                "_layers.9.1.mlp.gate.weight": "6abb9e8c91647585e075cae09c7eba84",
                "_layers.9.1.mlp.shared_experts.down_proj.weight": "3767e4ab5530caf3604f40b04a910d04",
                "_layers.9.1.mlp.shared_experts.up_gate_proj.weight": "b5d3863d422f20ed583cb8c6a875e5b4",
                "_layers.9.1.post_attention_layernorm.weight": "1847d4e593538ea5cb6ffa2cc2a5e0b1",
                "_layers.9.1.self_attn.k_norm.weight": "c6a2e5f74c87dbee8ecc146567864749",
                "_layers.9.1.self_attn.o_proj.weight": "f6fb6dad5191f8080037353c344ed748",
                "_layers.9.1.self_attn.q_norm.weight": "2b0dc33d8f4ce457eaf942b0272037f1",
                "_layers.9.1.self_attn.qkv_proj.weight": "a8d450e511c1b07bd286d50573ac5f69",
                "_layers.shared_layers.embed.embedding.embed_tokens.weight": "b649d8ace5b11ef1fe38a288a3b7b3bf",
            }

            for name, param in overlap_gpt_model.named_parameters():
                assert param.grad._md5sum() == baseline[name]


if __name__ == "__main__":
    unittest.main()
