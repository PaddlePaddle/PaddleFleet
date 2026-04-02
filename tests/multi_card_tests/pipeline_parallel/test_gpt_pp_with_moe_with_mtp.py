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
            baseline = {
                "_layers.shared_layers.embed.embedding.embed_tokens.weight": "019b311b522a37d5a5b13942f04ecd16",
                "_layers.9.0.input_layernorm.weight": "a96acfc13b0b82803986430cdd521795",
                "_layers.9.0.self_attn.o_proj.weight": "407fa9dc5f8d91d6b26af9ad9cbe9978",
                "_layers.9.0.self_attn.qkv_proj.weight": "b247a462df7ec156f10fa50657085a59",
                "_layers.9.0.self_attn.q_norm.weight": "37fd50e07874520e4d1565b6fdb9a391",
                "_layers.9.0.self_attn.k_norm.weight": "4906bb49f7de0ab4748d312f44eea258",
                "_layers.9.0.post_attention_layernorm.weight": "495509211e1a7b091349a3d56c99d3ce",
                "_layers.9.0.mlp.gate.weight": "c367e4a087d382b3e615ee3674bcb232",
                "_layers.9.0.mlp.experts.0.up_gate_proj.weight": "1e0e2a4991208784f31d1612ce604a3b",
                "_layers.9.0.mlp.experts.0.down_proj.weight": "16d99b61ba01672499b4258a62ffdc5e",
                "_layers.9.0.mlp.experts.1.up_gate_proj.weight": "1b679016c931ba113353ec818f87c6b1",
                "_layers.9.0.mlp.experts.1.down_proj.weight": "0d10d214cf17bef9c0635e7401ff0de6",
                "_layers.9.0.mlp.experts.2.up_gate_proj.weight": "21606fde32ecde6eab77576ea7318227",
                "_layers.9.0.mlp.experts.2.down_proj.weight": "c4ba1d360bc53405fa3623934c6677cf",
                "_layers.9.0.mlp.experts.3.up_gate_proj.weight": "332e7b22acefe86e6e844a0cb4d38fa4",
                "_layers.9.0.mlp.experts.3.down_proj.weight": "65fde00f05ea736c52d405bc1590ffec",
                "_layers.9.0.mlp.shared_experts.up_gate_proj.weight": "c3847e9c7648570fbc3d4ef93d6563d4",
                "_layers.9.0.mlp.shared_experts.down_proj.weight": "08d20198efecd6efe90f284d64fe6544",
                "_layers.9.1.input_layernorm.weight": "7695a358028279d09c9cdc4ed6a47712",
                "_layers.9.1.self_attn.o_proj.weight": "b31a26a02d98c2bdfc187058a106df0d",
                "_layers.9.1.self_attn.qkv_proj.weight": "e59274049ac585624ab9e3c4a4719d16",
                "_layers.9.1.self_attn.q_norm.weight": "c6f651a6a508436fe0fcdb41907b3305",
                "_layers.9.1.self_attn.k_norm.weight": "2821f9cf37580e5ece88e0d6e2d20088",
                "_layers.9.1.post_attention_layernorm.weight": "607cc0548006ac910a4994846b19f460",
                "_layers.9.1.mlp.gate.weight": "ec6bfb2f8135188d1915457dc70e1936",
                "_layers.9.1.mlp.experts.0.up_gate_proj.weight": "f977cdf2ed82e9575e9a63f8263e4085",
                "_layers.9.1.mlp.experts.0.down_proj.weight": "18631b59aa698f37dee05b1cc85291d5",
                "_layers.9.1.mlp.experts.1.up_gate_proj.weight": "07fb4342ab28dcad95fcd3bb2941b0a0",
                "_layers.9.1.mlp.experts.1.down_proj.weight": "19d9b566427ab7b66803386ffac0585e",
                "_layers.9.1.mlp.experts.2.up_gate_proj.weight": "4e6e2b791c85869bffbacf8a03efe011",
                "_layers.9.1.mlp.experts.2.down_proj.weight": "8dabb197c0c82135a9b360d926c542fb",
                "_layers.9.1.mlp.experts.3.up_gate_proj.weight": "321991727fe41f0c9bd28e3a04fb8d70",
                "_layers.9.1.mlp.experts.3.down_proj.weight": "d253106f7ce86f059d8bb120f7f8f403",
                "_layers.9.1.mlp.shared_experts.up_gate_proj.weight": "15e8cba735e9fdb96e9406bf7b06f801",
                "_layers.9.1.mlp.shared_experts.down_proj.weight": "869f09749562cc176aa039fd8b4cae05",
            }
            assert rst == baseline


if __name__ == "__main__":
    unittest.main()
