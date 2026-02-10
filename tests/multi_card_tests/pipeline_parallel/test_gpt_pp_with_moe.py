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
from paddle.distributed.fleet import distributed_model

import paddlefleet
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

        assert overlap_loss._md5sum() == "29f8b7fa9402bbff4b94d361af58b6e2"

        if paddle.distributed.get_rank() == 0:
            baseline = {
                "_layers.9.0.input_layernorm.weight": "0692a0567389a437fdf55a82140f1895",
                "_layers.9.0.mlp.experts.0.down_proj.weight": "dfbec352f966837904140b4260fd6589",
                "_layers.9.0.mlp.experts.0.up_gate_proj.weight": "85536680873d0ee6ad445ce7554e1d62",
                "_layers.9.0.mlp.experts.1.down_proj.weight": "a4b0d27b218a7ef2241c3e006066ac4c",
                "_layers.9.0.mlp.experts.1.up_gate_proj.weight": "32d0903f0ee92a8cef89d63614ce2295",
                "_layers.9.0.mlp.experts.2.down_proj.weight": "62c3f662fe2b89ad1a1bbf67baa5a13d",
                "_layers.9.0.mlp.experts.2.up_gate_proj.weight": "29b681d39894e9c9c4f25c27fce7797e",
                "_layers.9.0.mlp.experts.3.down_proj.weight": "880aa8574cc4730f05b5f25115322fbc",
                "_layers.9.0.mlp.experts.3.up_gate_proj.weight": "09e9f524f8af3603c95d9744474819ab",
                "_layers.9.0.mlp.gate.weight": "57257f41bf4480a7eae4cd9101fd9394",
                "_layers.9.0.mlp.shared_experts.down_proj.weight": "22a0ec90d91bbecfdab73dc740cdb010",
                "_layers.9.0.mlp.shared_experts.up_gate_proj.weight": "e9018d95bd5cd8913c24b2648f64c31d",
                "_layers.9.0.post_attention_layernorm.weight": "a1b16c0d3ab2a8f7731cc23d73ef305e",
                "_layers.9.0.self_attn.k_norm.weight": "d27b09a3d828882baf059d747fb23583",
                "_layers.9.0.self_attn.o_proj.weight": "eb740c48002e36b82f065b6ae90e3674",
                "_layers.9.0.self_attn.q_norm.weight": "7a4edac3f974161a6c08e604f43bf5b1",
                "_layers.9.0.self_attn.qkv_proj.weight": "e6e0da5289eaeec65ac1073e6429f3de",
                "_layers.9.1.input_layernorm.weight": "2c2100b62607a0903aa24230e92dc99f",
                "_layers.9.1.mlp.experts.0.down_proj.weight": "6fb6e1655bb81a0065d1bc9b0dfaf867",
                "_layers.9.1.mlp.experts.0.up_gate_proj.weight": "a3150a6010fc1b1595fffe3a41daeca3",
                "_layers.9.1.mlp.experts.1.down_proj.weight": "38d97b8e28883d82dd1f82072c9e56bf",
                "_layers.9.1.mlp.experts.1.up_gate_proj.weight": "df825efd27e969c38840cd7c97a4ce38",
                "_layers.9.1.mlp.experts.2.down_proj.weight": "69331a69d7b7d3b3e5b283951ea6e38b",
                "_layers.9.1.mlp.experts.2.up_gate_proj.weight": "fc4291de267252c9d3b42a6846595277",
                "_layers.9.1.mlp.experts.3.down_proj.weight": "78b855751d6c2c0483c51040eaddb100",
                "_layers.9.1.mlp.experts.3.up_gate_proj.weight": "ff4568eaf479dc62b1517b6d9cd755d9",
                "_layers.9.1.mlp.gate.weight": "33333aa4afd220fb1caa6f85a3c74eb0",
                "_layers.9.1.mlp.shared_experts.down_proj.weight": "c0b4f7fa48224597856827fcbb8afa1a",
                "_layers.9.1.mlp.shared_experts.up_gate_proj.weight": "1726a880f8efa5fdada7bc330584d282",
                "_layers.9.1.post_attention_layernorm.weight": "e1bb9cba1bc5b54f2161761631a92e9e",
                "_layers.9.1.self_attn.k_norm.weight": "2233204a017ae9911388cbc8acd34528",
                "_layers.9.1.self_attn.o_proj.weight": "96e1e61abeac18bb00b10bdc4e8f4262",
                "_layers.9.1.self_attn.q_norm.weight": "34d93a0270a60d493766e53cea1e71a9",
                "_layers.9.1.self_attn.qkv_proj.weight": "6f00e9d9db3651416fbb64fc4a08c43a",
                "_layers.shared_layers.embed.embedding.embed_tokens.weight": "fd41a4621d2c89790fff2d7f55282444",
            }

            for name, param in overlap_gpt_model.named_parameters():
                assert param.grad._md5sum() == baseline[name]


if __name__ == "__main__":
    unittest.main()
