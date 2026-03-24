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
                "_layers.shared_layers.embed.embedding.embed_tokens.weight": "d19749d6846c8764553a18821eb56ac2",
                "_layers.9.0.input_layernorm.weight": "dbe3c682a3d2c38cbdacb2995d993a68",
                "_layers.9.0.self_attn.o_proj.weight": "abf8a5178bb928ecaf321d42f5116d80",
                "_layers.9.0.self_attn.qkv_proj.weight": "4ba22a125924a5a28bd8b0a20dd94540",
                "_layers.9.0.self_attn.q_norm.weight": "89a85f33c68c6ea99c5668a467d02d6a",
                "_layers.9.0.self_attn.k_norm.weight": "db93936d424950842d263cee99fdcd5f",
                "_layers.9.0.post_attention_layernorm.weight": "8bb48bb8e92fe63ef7505cef4ebdaa39",
                "_layers.9.0.mlp.gate.weight": "ccebac6dd38547ea0fcdb82a621d80c5",
                "_layers.9.0.mlp.experts.0.up_gate_proj.weight": "431b1b42377d55064606eafdf2dee77b",
                "_layers.9.0.mlp.experts.0.down_proj.weight": "8accf32db39dc86784704b9197f6d4d5",
                "_layers.9.0.mlp.experts.1.up_gate_proj.weight": "61fc31621b7059a2ce5bda98f646668b",
                "_layers.9.0.mlp.experts.1.down_proj.weight": "b630de872aab1e7d8a976cb3359c5599",
                "_layers.9.0.mlp.experts.2.up_gate_proj.weight": "f57b01f791ddb5b1b9a3a3930486ba2d",
                "_layers.9.0.mlp.experts.2.down_proj.weight": "c557c29af22d972a2c56fd2c558a39cd",
                "_layers.9.0.mlp.experts.3.up_gate_proj.weight": "eb94739bd612f36de3a8789be5433a2f",
                "_layers.9.0.mlp.experts.3.down_proj.weight": "54e8cf46f971d8f074020bf0ba804335",
                "_layers.9.0.mlp.shared_experts.up_gate_proj.weight": "88b22b9e78aaa510da4f2d27bf93972c",
                "_layers.9.0.mlp.shared_experts.down_proj.weight": "8e84fa9052149db887f6fabbe117908c",
                "_layers.9.1.input_layernorm.weight": "eb82008d01870720457411f5f2827177",
                "_layers.9.1.self_attn.o_proj.weight": "ba8deb677f4e27886efe2a0d4996a3a3",
                "_layers.9.1.self_attn.qkv_proj.weight": "fb710c995254738ce64fa5a22458e31e",
                "_layers.9.1.self_attn.q_norm.weight": "c1db8a0ea7977276185fe6135491d4fa",
                "_layers.9.1.self_attn.k_norm.weight": "67b1a19840914588dd9f1fa882d61178",
                "_layers.9.1.post_attention_layernorm.weight": "8f18fb3aca05da38b16d8cb04b975252",
                "_layers.9.1.mlp.gate.weight": "5eabf368dcfa7805012bb3f09a42b34f",
                "_layers.9.1.mlp.experts.0.up_gate_proj.weight": "cde95a30504997e19924b381713d7136",
                "_layers.9.1.mlp.experts.0.down_proj.weight": "a78993e86c159ae6450f5021f2aeed30",
                "_layers.9.1.mlp.experts.1.up_gate_proj.weight": "60a4a0fbfb34207a1190e4364e8374eb",
                "_layers.9.1.mlp.experts.1.down_proj.weight": "cba01a287f7e5ab630157d4e5414e845",
                "_layers.9.1.mlp.experts.2.up_gate_proj.weight": "ba114487ed3bdc483ebd53de0c0ac268",
                "_layers.9.1.mlp.experts.2.down_proj.weight": "d46c61df9c715c93079daf11302dfde4",
                "_layers.9.1.mlp.experts.3.up_gate_proj.weight": "096e8ce7f02148c9961206b66286adc6",
                "_layers.9.1.mlp.experts.3.down_proj.weight": "8f62dc3095ad9d4fed042146bf49f7d5",
                "_layers.9.1.mlp.shared_experts.up_gate_proj.weight": "04e436d8fe6e548a1a71b544b61c2104",
                "_layers.9.1.mlp.shared_experts.down_proj.weight": "5c967931ff68de314af384d98620c996",
            }

            assert rst == baseline


if __name__ == "__main__":
    unittest.main()
