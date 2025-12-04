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
from paddle.distributed.fleet.utils.sequence_parallel_utils import (
    register_sequence_parallel_allreduce_hooks,
)

import paddlefleet

# from tests.unit_tests.test_utilities import Utils
import paddlefleet.parallel_state as ps

# from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from paddlefleet.models.gpt.gpt_model import GPTModel
from paddlefleet.tensor_parallel.mappings import (
    _gather_along_first_dim,
    _gather_along_last_dim,
)
from paddlefleet.training.initialize import initialize_fleet
from paddlefleet.transformer.transformer_config import TransformerConfig


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


def cal_sim(a, b):
    return paddle.nn.functional.cosine_similarity(a.flatten(), b.flatten(), 0)


def check_grads(dist_model, serial_model, tp_group):
    serial_grads = {}
    for name, p in serial_model.named_parameters():
        serial_grads[name] = p.grad

    dist_grads = {}
    for name, p in dist_model.named_parameters():
        if "qkv_proj.weight" in name or "up_gate_proj.weight" in name:
            grad = _gather_along_last_dim(p.grad, tp_group)
        elif (
            "o_proj.weight" in name
            or "down_proj.weight" in name
            or "embed_tokens.weight" in name
        ):
            grad = _gather_along_first_dim(p.grad, tp_group)
        else:
            grad = p.grad
        assert (
            paddle.allclose(grad, serial_grads[name], atol=5e-8)
            and cal_sim(grad, serial_grads[name]) > 0.999
        )


def single_device_baseline(seed, batch_size, seq_len, vocab_size, config):
    seed = 46
    random.seed(seed)
    np.random.seed(seed)
    paddle.manual_seed(seed)

    transformer_layer_spec = get_gpt_layer_local_spec(
        num_experts=None,
        moe_grouped_gemm=False,
        use_qk_norm=True,
        multi_latent_attention=False,
        normalization="RMSNorm",
    )
    pre_process = True
    post_process = True
    mtp_block_spec = None
    vp_stage = None

    gpt_model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=vocab_size,
        max_sequence_length=seq_len,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=False,
        parallel_output=True,
        share_embeddings_and_output_weights=True,
        position_embedding_type="rope",
        rotary_percent=1.0,
        rotary_base=10000,
        rope_scaling=1.0,
        mtp_block_spec=mtp_block_spec,
        vp_stage=vp_stage,
    )

    data = paddle.randint(
        low=0, high=vocab_size, shape=(batch_size, seq_len + 1)
    )
    input_ids = data[:, :-1]
    labels = data[:, 1:]
    position_ids = paddle.to_tensor(data, dtype=paddle.int64).repeat(
        (batch_size, 1)
    )

    outputs = gpt_model(
        input_ids=input_ids,
        position_ids=position_ids,
        labels=labels,
    )
    loss = outputs[0]
    loss.backward()

    return loss, outputs, gpt_model


def run_tp_sp(
    seed,
    batch_size,
    seq_len,
    vocab_size,
    config,
    loss_baseline,
    outputs_baseline,
    gpt_model_baseline,
):
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 4,
        "pp_degree": 1,
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
    }
    initialize_fleet(strategy)

    _set_random_seed(seed)

    transformer_layer_spec = get_gpt_layer_local_spec(
        num_experts=None,
        moe_grouped_gemm=False,
        use_qk_norm=True,
        multi_latent_attention=False,
        normalization="RMSNorm",
    )
    pre_process = True
    post_process = True
    mtp_block_spec = None
    vp_stage = None

    gpt_model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=vocab_size,
        max_sequence_length=seq_len,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=False,
        parallel_output=True,
        share_embeddings_and_output_weights=True,
        position_embedding_type="rope",
        rotary_percent=1.0,
        rotary_base=10000,
        rope_scaling=1.0,
        mtp_block_spec=mtp_block_spec,
        vp_stage=vp_stage,
    )
    register_sequence_parallel_allreduce_hooks(gpt_model, 1, False)

    data = paddle.randint(
        low=0, high=vocab_size, shape=(batch_size, seq_len + 1)
    )
    input_ids = data[:, :-1]
    labels = data[:, 1:]
    position_ids = paddle.to_tensor(data, dtype=paddle.int64).repeat(
        (batch_size, 1)
    )

    tp_group = ps.get_tensor_model_parallel_group()

    outputs = gpt_model(
        input_ids=input_ids,
        position_ids=position_ids,
        labels=labels,
    )
    loss = outputs[0]
    loss.backward()
    assert loss == loss_baseline
    check_grads(gpt_model, gpt_model_baseline, tp_group)


class TestTPSP(unittest.TestCase):
    def setUp(self):
        self.seed = 46
        self.batch_size = 2
        self.seq_len = 128
        self.vocab_size = 1024

    def test_tp_sp(self):
        config = TransformerConfig(
            num_hidden_layers=2,
            hidden_size=512,
            num_attention_heads=4,
            intermediate_size=1024,
            normalization="RMSNorm",
            hidden_dropout_prob=0.0,
            attention_dropout=0.0,
            use_cpu_initialization=True,
            init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            output_layer_init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
        )
        loss, outputs, gpt_model = single_device_baseline(
            self.seed, self.batch_size, self.seq_len, self.vocab_size, config
        )

        dist_config = TransformerConfig(
            num_hidden_layers=2,
            hidden_size=512,
            num_attention_heads=4,
            intermediate_size=1024,
            normalization="RMSNorm",
            hidden_dropout_prob=0.0,
            attention_dropout=0.0,
            use_cpu_initialization=True,
            tensor_model_parallel_size=4,
            sequence_parallel=True,
            init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            output_layer_init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
        )
        run_tp_sp(
            self.seed,
            self.batch_size,
            self.seq_len,
            self.vocab_size,
            dist_config,
            loss,
            outputs,
            gpt_model,
        )


if __name__ == "__main__":
    unittest.main()
