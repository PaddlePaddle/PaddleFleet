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
                "_layers.shared_layers.embed.embedding.embed_tokens.weight": "506d6168647863fb55d820ef674991b6",
                "_layers.9.0.input_layernorm.weight": "5b4b51eaa8bf3f7950ef07e33470d2f9",
                "_layers.9.0.self_attn.o_proj.weight": "677b18e49e973601508d997be6bfcd1c",
                "_layers.9.0.self_attn.qkv_proj.weight": "c31d2dee4fa3510cb8a141e3a5a69596",
                "_layers.9.0.self_attn.q_norm.weight": "12aea7a894f3397c8298f5be4758cabf",
                "_layers.9.0.self_attn.k_norm.weight": "ff9fc96b31ed090cbb5fadf296b45496",
                "_layers.9.0.post_attention_layernorm.weight": "18b8171c8a85ed5cb67ec49b11cbab51",
                "_layers.9.0.mlp.gate.weight": "c6641e64e02214bce3f5d500da6e902e",
                "_layers.9.0.mlp.experts.0.up_gate_proj.weight": "0c047341a2d5b94b82a2af60f59fa774",
                "_layers.9.0.mlp.experts.0.down_proj.weight": "91bd549773243e6f72172a2b5670031e",
                "_layers.9.0.mlp.experts.1.up_gate_proj.weight": "fca5eba85ce43ddfdfa8640c157bfaaa",
                "_layers.9.0.mlp.experts.1.down_proj.weight": "9b91603d32a66f25970b276215287f15",
                "_layers.9.0.mlp.experts.2.up_gate_proj.weight": "5a6be01803e5a5aefcfd9e037c1ee0cc",
                "_layers.9.0.mlp.experts.2.down_proj.weight": "8da097c213c71c44d96dec5dd88c6871",
                "_layers.9.0.mlp.experts.3.up_gate_proj.weight": "7c43545231605f66eeb7e03f21d22258",
                "_layers.9.0.mlp.experts.3.down_proj.weight": "cf39405dafd56973778708b4e3ef00b7",
                "_layers.9.0.mlp.shared_experts.up_gate_proj.weight": "8ba1eac82857e7c1016ccc685ef65742",
                "_layers.9.0.mlp.shared_experts.down_proj.weight": "987c33b8550c3469eea6d7c0efae0d58",
                "_layers.9.1.input_layernorm.weight": "0691a14caa8e568fd6a7b0aa305484a9",
                "_layers.9.1.self_attn.o_proj.weight": "02863a2818fbf051ec95b25685e7e348",
                "_layers.9.1.self_attn.qkv_proj.weight": "a0b185b3c5104e6a4cebe2c00de870da",
                "_layers.9.1.self_attn.q_norm.weight": "443624a98ffbfdd55c620ceb3bc203de",
                "_layers.9.1.self_attn.k_norm.weight": "123720da4676c46a8f9185116a5ead15",
                "_layers.9.1.post_attention_layernorm.weight": "66f761220ebd2217858288c51d7c74f8",
                "_layers.9.1.mlp.gate.weight": "a3c2740d0c352fb3778c5e1ed3cad265",
                "_layers.9.1.mlp.experts.0.up_gate_proj.weight": "592d1b7c2566c5d48c86a59ba17cf0e8",
                "_layers.9.1.mlp.experts.0.down_proj.weight": "6d25fb483d56d860c918e7c9fd40f5da",
                "_layers.9.1.mlp.experts.1.up_gate_proj.weight": "e24e4daf4496a3fb650caf41f41054b4",
                "_layers.9.1.mlp.experts.1.down_proj.weight": "6eeb2552d7a0dee8fbe91661f2b621ea",
                "_layers.9.1.mlp.experts.2.up_gate_proj.weight": "fac14125ad55f482b08eb7c033837c61",
                "_layers.9.1.mlp.experts.2.down_proj.weight": "10a2c523aa92f68047c67a4bd3959d87",
                "_layers.9.1.mlp.experts.3.up_gate_proj.weight": "7326b60c5e056fd0882007e7d54dee61",
                "_layers.9.1.mlp.experts.3.down_proj.weight": "6b11d5e3f3d361f7486c71ff05fc8eef",
                "_layers.9.1.mlp.shared_experts.up_gate_proj.weight": "e110d415a4f0fa750d4a4d1f86f0dac5",
                "_layers.9.1.mlp.shared_experts.down_proj.weight": "216888db071b00a85f4ce9d74107be18",
            }

            for name, param in overlap_gpt_model.named_parameters():
                assert param.grad._md5sum() == baseline[name]


if __name__ == "__main__":
    unittest.main()
