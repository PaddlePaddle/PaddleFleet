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
SKIP_TESTS = REPO_FLAG != "paddlefleet"


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


@unittest.skipIf(
    SKIP_TESTS,
    f"Skipping tests: repo_flag={REPO_FLAG} (not 'paddlefleet')",
)
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

        assert overlap_loss._md5sum() == "bce3fed95247f1b7a165e32b33d6fca7"

        if paddle.distributed.get_rank() == 0:
            baseline = {
                "_layers.9.0.input_layernorm.weight": "34707dfb7a96f6ac9a11e7fc3713f1ea",
                "_layers.9.0.mlp.down_proj.weight": "728d03046bfb2ab0e4f191d3b045c93c",
                "_layers.9.0.mlp.up_gate_proj.weight": "9fdb6a2101890e7adef7b1ebbf353714",
                "_layers.9.0.post_attention_layernorm.weight": "fb5064a5a66b476bec0419b3687e14bf",
                "_layers.9.0.self_attn.k_norm.weight": "fde5089a9ba2b68900b994179ec0598d",
                "_layers.9.0.self_attn.o_proj.weight": "76866f06592ebadcd9b9a2c55f02cb44",
                "_layers.9.0.self_attn.q_norm.weight": "897588977d1854c1a1b7db8cd654107c",
                "_layers.9.0.self_attn.qkv_proj.weight": "9d04ca986be4a71fee8a18254014f30e",
                "_layers.9.1.input_layernorm.weight": "405839bc23ff3aef9604824c60f8f3ba",
                "_layers.9.1.mlp.down_proj.weight": "b89aa17a9e18ccaa09859bea25229e05",
                "_layers.9.1.mlp.up_gate_proj.weight": "e0834ac457fa687903cdb2dbb8178400",
                "_layers.9.1.post_attention_layernorm.weight": "8cb08725344776fc41659930b700c337",
                "_layers.9.1.self_attn.k_norm.weight": "81af5c50e3fff9ca35b6cd10050330be",
                "_layers.9.1.self_attn.o_proj.weight": "6d0f4851c8f63ea3435296248992c68f",
                "_layers.9.1.self_attn.q_norm.weight": "cb16a452bd7fb5fd4466d7cedca12e20",
                "_layers.9.1.self_attn.qkv_proj.weight": "fe969890118275f355692eedeb446083",
                "_layers.shared_layers.embed.embedding.embed_tokens.weight": "3f5909c4bd328a15662835e5c566762e",
            }

            for name, p in overlap_gpt_model.named_parameters():
                assert p.grad._md5sum() == baseline[name]


if __name__ == "__main__":
    unittest.main()
