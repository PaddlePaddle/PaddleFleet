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
import pprint
import random
import subprocess
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
REPO_FLAG = os.getenv("repo_flag")
SKIP_TESTS = REPO_FLAG != "paddlefleet"


def get_gpu_models_via_nvidia_smi():
    try:
        output = subprocess.check_output(
            "nvidia-smi --query-gpu=name --format=csv,noheader", shell=True
        )
        models = output.decode().strip().replace("NVIDIA", "")
        return models
    except Exception as e:
        return ["Unknown"]


def judge_machine_type():
    if not paddle.is_compiled_with_cuda():
        return "No CUDA GPU"
    models = get_gpu_models_via_nvidia_smi()
    for model in models:
        name = model.upper()
        if "V" in name:
            return "V"
        elif "H" in name:
            return "H"
        elif "B" in name:
            return "B"


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


# NOTE(Pan Zhaowu): Temporary disable this test case due to PaddlePaddle PR78746
# RE-enable this test case when PR78746 and related cherry-picks is merged
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

        print("Overlap PP loss MD5:", overlap_loss._md5sum())
        rst = {}
        for name, param in overlap_gpt_model.named_parameters():
            if param.grad is not None:
                rst[name] = param.grad._md5sum()

        pp = pprint.PrettyPrinter(depth=None, width=200, compact=False)
        pp.pprint(rst)

        if judge_machine_type() == "H":
            assert overlap_loss._md5sum() == "bce3fed95247f1b7a165e32b33d6fca7"
            if paddle.distributed.get_rank() == 0:
                baseline = {
                    "_layers.9.0.input_layernorm.weight": "fe9464b2b154bf82ea7f451dd014c796",
                    "_layers.9.0.mlp.down_proj.weight": "0f997d356ae211f4d22bb6cecc5018f4",
                    "_layers.9.0.mlp.up_gate_proj.weight": "4d2b2dd20eb584ae212cc891cad7d45f",
                    "_layers.9.0.post_attention_layernorm.weight": "306febfb642695cedfcf6a2afc4acf26",
                    "_layers.9.0.self_attn.k_norm.weight": "cc963360fd3e3ed9f8ffef97fcbdf0a8",
                    "_layers.9.0.self_attn.o_proj.weight": "f21b2820d49ba7284270b5308debd360",
                    "_layers.9.0.self_attn.q_norm.weight": "79934f9fcddfe6469b95436e58fe3b46",
                    "_layers.9.0.self_attn.qkv_proj.weight": "fe596c4a672f5e2dbbba4793566ff34d",
                    "_layers.9.1.input_layernorm.weight": "66f3af21b98abcf8bdbe086664fbb0e9",
                    "_layers.9.1.mlp.down_proj.weight": "435630c8e2157ab7f24b4435bc6c7828",
                    "_layers.9.1.mlp.up_gate_proj.weight": "bef5eae242ea75b234890f0f0780fd4a",
                    "_layers.9.1.post_attention_layernorm.weight": "4b1a9ce93c5f2b23b8bd73753b43d195",
                    "_layers.9.1.self_attn.k_norm.weight": "c7e0173bf56c715d9c7eb2a3bc7c10d6",
                    "_layers.9.1.self_attn.o_proj.weight": "3552ca7607b20ecdd462375cedbffaff",
                    "_layers.9.1.self_attn.q_norm.weight": "54e827b3e6408ed147eb078cdb3bf999",
                    "_layers.9.1.self_attn.qkv_proj.weight": "ccac89b168ac82ae2d8c60297c608f15",
                    "_layers.shared_layers.embed.embedding.embed_tokens.weight": "7d1bcb081618d9c75e44c8c0e1d2488f",
                }
                for name, p in overlap_gpt_model.named_parameters():
                    assert p.grad._md5sum() == baseline[name], (
                        f"{name}'s grad has diff"
                    )
        elif judge_machine_type() == "B":
            assert overlap_loss._md5sum() == "53ef9369cbb96bb140b11987554f4954"
            if paddle.distributed.get_rank() == 0:
                baseline = {
                    "_layers.9.0.input_layernorm.weight": "b09749cbbdf3b8da21b7dc6c23e4d88a",
                    "_layers.9.0.mlp.down_proj.weight": "923676c6fbad6fc604763d42f85d7a85",
                    "_layers.9.0.mlp.up_gate_proj.weight": "3c2c005ceaf4d5280870f67b79c0698a",
                    "_layers.9.0.post_attention_layernorm.weight": "b5a58f295ae06a2b274f3526ae75014a",
                    "_layers.9.0.self_attn.k_norm.weight": "0f4bdc5650529d2f4a662ea41d453151",
                    "_layers.9.0.self_attn.o_proj.weight": "8117ecb1e205076ed8905a075803de4f",
                    "_layers.9.0.self_attn.q_norm.weight": "ab957c763bcd2c70f2c2a0c7932344ff",
                    "_layers.9.0.self_attn.qkv_proj.weight": "64a59fc8b3fae48c01bf1a88ee58fa68",
                    "_layers.9.1.input_layernorm.weight": "9ec2735a522837c23b5a79d7a9835253",
                    "_layers.9.1.mlp.down_proj.weight": "249b7248d00cf4afc0cf1126f0fcec66",
                    "_layers.9.1.mlp.up_gate_proj.weight": "b8d542554dbd8dee1a7a640bfd658658",
                    "_layers.9.1.post_attention_layernorm.weight": "fed9d29a15a2f906b1511e2f9c36c5da",
                    "_layers.9.1.self_attn.k_norm.weight": "5ce1d0b2a5f12780140542d2c6fa1229",
                    "_layers.9.1.self_attn.o_proj.weight": "cdc43b05a7e408a69137cebbeed0eeae",
                    "_layers.9.1.self_attn.q_norm.weight": "64b2cdca7ab9e0e51089af57d6cb8713",
                    "_layers.9.1.self_attn.qkv_proj.weight": "52f0b8248da51aa6235893d75c7bc2fe",
                    "_layers.shared_layers.embed.embedding.embed_tokens.weight": "278c4be9af06472930c63fdd62229c26",
                }
                for name, p in overlap_gpt_model.named_parameters():
                    assert p.grad._md5sum() == baseline[name], (
                        f"{name}'s grad has diff"
                    )


if __name__ == "__main__":
    unittest.main()
