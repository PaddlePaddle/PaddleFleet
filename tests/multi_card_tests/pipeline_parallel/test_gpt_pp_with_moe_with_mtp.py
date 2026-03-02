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
                "_layers.shared_layers.embed.embedding.embed_tokens.weight": "9bdbd5fde8e4226c00ee0036fe55cb6b",
                "_layers.9.0.input_layernorm.weight": "8d601141530419c30c31e3ace0b2bbe6",
                "_layers.9.0.self_attn.o_proj.weight": "407aa3ad730a10157ccec274700f10d1",
                "_layers.9.0.self_attn.qkv_proj.weight": "61fe19f496533fb6ac9eef90c5aecf7a",
                "_layers.9.0.self_attn.q_norm.weight": "a9e658df86726557e817a24f2f86e256",
                "_layers.9.0.self_attn.k_norm.weight": "ca7948c37ce99a749c90772e162b8ddf",
                "_layers.9.0.post_attention_layernorm.weight": "f9b8e54a46bc29762f8863538330f48c",
                "_layers.9.0.mlp.gate.weight": "4f57a366c9d3d92387c575d219880d95",
                "_layers.9.0.mlp.experts.0.up_gate_proj.weight": "2a67cb60e6ec35a59935b56d07993905",
                "_layers.9.0.mlp.experts.0.down_proj.weight": "846f076f0f211de89ef11de9450bbd2c",
                "_layers.9.0.mlp.experts.1.up_gate_proj.weight": "09f83462ca7e5f54434f91ae7a25feba",
                "_layers.9.0.mlp.experts.1.down_proj.weight": "b13dace78fac4e973ac952250d3d7cef",
                "_layers.9.0.mlp.experts.2.up_gate_proj.weight": "11ee5be0bcafea6b4faeaa1eefc5aba5",
                "_layers.9.0.mlp.experts.2.down_proj.weight": "672045c90e2a82cca5172665926c6e0a",
                "_layers.9.0.mlp.experts.3.up_gate_proj.weight": "de1f505e18ec2e982efd23b769561ea7",
                "_layers.9.0.mlp.experts.3.down_proj.weight": "df439c081feb714c1e35e50ba8d49f5b",
                "_layers.9.0.mlp.shared_experts.up_gate_proj.weight": "21810b1acff852dd52d10a612a30649d",
                "_layers.9.0.mlp.shared_experts.down_proj.weight": "f5282c6bf8fc997da09114be9a11e855",
                "_layers.9.1.input_layernorm.weight": "9201b563e939409a205075a135dfdda6",
                "_layers.9.1.self_attn.o_proj.weight": "fe78b250e1e9fe18c3602c117108ad92",
                "_layers.9.1.self_attn.qkv_proj.weight": "05433900451ef71daae2911cc1b81e8f",
                "_layers.9.1.self_attn.q_norm.weight": "683dce3b38613b5fd15042c9840ff5cc",
                "_layers.9.1.self_attn.k_norm.weight": "f29fa36183b31ad0f2ca2b12823b1e13",
                "_layers.9.1.post_attention_layernorm.weight": "678435ae3f68aa805f003d306a3a9427",
                "_layers.9.1.mlp.gate.weight": "193111b9a01d36663300729a710495c6",
                "_layers.9.1.mlp.experts.0.up_gate_proj.weight": "272d470939e2e1ea10a85d4bee2f5342",
                "_layers.9.1.mlp.experts.0.down_proj.weight": "75c2cfeac5d1a4699397420842de0da5",
                "_layers.9.1.mlp.experts.1.up_gate_proj.weight": "ba0186672a377a18c8aa2f9dacee0a84",
                "_layers.9.1.mlp.experts.1.down_proj.weight": "540f55bc144266486f9b0dd9ea46bf1a",
                "_layers.9.1.mlp.experts.2.up_gate_proj.weight": "e8daf0e68e48383ad4a91a9a8cfec12f",
                "_layers.9.1.mlp.experts.2.down_proj.weight": "65f9c7c02f194902abc168940ca2baf7",
                "_layers.9.1.mlp.experts.3.up_gate_proj.weight": "5d5ccfa61ec6906bad2d410ae0686e02",
                "_layers.9.1.mlp.experts.3.down_proj.weight": "d7e9d857c8ab5bfa8845c19d1a305c3b",
                "_layers.9.1.mlp.shared_experts.up_gate_proj.weight": "bde135527bfaa56f9c77dfa4a7e0c70a",
                "_layers.9.1.mlp.shared_experts.down_proj.weight": "d1fad16e2f811e8c08b0c1a994e14669",
            }

            assert rst == baseline


if __name__ == "__main__":
    unittest.main()
