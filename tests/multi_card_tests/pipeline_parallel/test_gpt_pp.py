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
        "mp_degree": 1,
        "pp_degree": config.pipeline_model_parallel_size,
        "sharding_degree": 1,
        "sep_degree": 1,
        "cp_degree": 1,
        "ep_degree": 1,
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
        )

        overlap_loss, overlap_gpt_model = run_pp(
            self.seed,
            self.batch_size,
            self.seq_len,
            self.vocab_size,
            config,
        )

        repo_name = os.environ.get("repo_flag")
        if repo_name == "paddlefleet":
            assert overlap_loss._md5sum() == "bce3fed95247f1b7a165e32b33d6fca7"
            if paddle.distributed.get_rank() == 0:
                baseline = {
                    "_layers.shared_layers.embed.embedding.embed_tokens.weight": "a05bf38eaa5f28b6e79f3df45941b396",
                    "_layers.9.0.input_layernorm.weight": "7b4816fdae7d3df0e0c3f7261550aaf0",
                    "_layers.9.0.self_attn.o_proj.weight": "7731ffe4060feaacaf71b84623eb045f",
                    "_layers.9.0.self_attn.qkv_proj.weight": "e21db4ef9d0031291bf10f0d2891130c",
                    "_layers.9.0.self_attn.q_layernorm.weight": "b093c0029ee425d8932f4767e762d3ee",
                    "_layers.9.0.self_attn.k_layernorm.weight": "dd51d26b410a1acc7fc0c09b75119b14",
                    "_layers.9.0.post_attention_layernorm.weight": "83301a39eb8f689ed4bf0a63c6765ee0",
                    "_layers.9.0.mlp.up_gate_proj.weight": "db95761551fbf7aadec1ee6b99511b76",
                    "_layers.9.0.mlp.down_proj.weight": "752dcedc986974687a6b969df201df98",
                    "_layers.9.1.input_layernorm.weight": "f6e217ef20bdd57d023bfb1342229d25",
                    "_layers.9.1.self_attn.o_proj.weight": "4640bb9563fa0f63b10d6008c8b425fb",
                    "_layers.9.1.self_attn.qkv_proj.weight": "13e3abe4d6e597e06bb67947e7ccf897",
                    "_layers.9.1.self_attn.q_layernorm.weight": "1b53ae1438558dba62da24049d730088",
                    "_layers.9.1.self_attn.k_layernorm.weight": "459774af0e0b36db81c8581dc7a45e09",
                    "_layers.9.1.post_attention_layernorm.weight": "7c21d0531529c10b4029b38d84f93ac2",
                    "_layers.9.1.mlp.up_gate_proj.weight": "5fb1cdafde7e42d49ae7d71e830e2380",
                    "_layers.9.1.mlp.down_proj.weight": "c72f73fafcea7a583a0f103da4fd75e4",
                }
                for name, p in overlap_gpt_model.named_parameters():
                    assert p.grad._md5sum() == baseline[name]
        else:
            pass


if __name__ == "__main__":
    unittest.main()
