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
import subprocess
import unittest

import numpy as np
import paddle
from paddle.distributed import fleet

# from tests.unit_tests.test_utilities import Utils
import paddlefleet.parallel_state as ps

# from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from paddlefleet.models.gpt.gpt_model import GPTModel
from paddlefleet.transformer.transformer_config import TransformerConfig


def get_gpu_models_via_nvidia_smi():
    try:
        output = subprocess.check_output(
            "nvidia-smi --query-gpu=name --format=csv,noheader", shell=True
        )
        models = output.decode().strip().split("\n")
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


result = judge_machine_type()
print("你的机器类型是：", result)


class TestGPTModel(unittest.TestCase):
    def setUp(self):
        seed = 46
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 1,
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
        fleet.init(is_collective=True, strategy=strategy)
        hcg = fleet.get_hybrid_communicate_group()
        ps.initialize_model_parallel(hcg)

        config = TransformerConfig(
            num_hidden_layers=2,
            hidden_size=512,
            num_attention_heads=4,
            intermediate_size=1024,
            normalization="RMSNorm",
            hidden_dropout_prob=0.0,
            attention_dropout=0.0,
            moe_num_experts=8,
            use_bias=False,
            moe_intermediate_size=1024,
            moe_token_dispatcher_type="alltoall",
            moe_shared_expert_intermediate_size=1024,
            init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            output_layer_init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
        )
        transformer_layer_spec = get_gpt_layer_local_spec(
            num_experts=8,
            moe_grouped_gemm=False,
            use_qk_norm=True,
            multi_latent_attention=False,
            normalization="RMSNorm",
        )
        pre_process = True
        post_process = True
        mtp_block_spec = None
        vp_stage = None
        self.gpt_model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=100,
            max_sequence_length=64,
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

    def test_forward(self) -> None:
        _ = self.gpt_model.config
        sequence_length = self.gpt_model.max_sequence_length
        micro_batch_size = 2

        for name, param in self.gpt_model.named_parameters():
            # 计算 L2 范数
            param_norm = param.detach().norm().item()
            param_abssum = param.detach().abs().sum().item()
            print(f"{name}: {param_norm:.6f}, {param_abssum:.6f}")

        data = list(range(sequence_length))
        input_ids = paddle.to_tensor(data, dtype=paddle.int64).repeat(
            (micro_batch_size, 1)
        )
        position_ids = paddle.to_tensor(data, dtype=paddle.int64).repeat(
            (micro_batch_size, 1)
        )
        attention_mask = paddle.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        )
        labels = paddle.to_tensor(
            list(range(1, sequence_length + 1)), dtype=paddle.int64
        ).repeat((micro_batch_size, 1))

        outputs = self.gpt_model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs["loss"]
        print("loss", loss.item())
        if judge_machine_type() == "H":
            assert loss.item() == 5.344995498657227, (
                f"loss not equal ({loss.item()} != 5.344995498657227), please check your modify"
            )
        elif judge_machine_type() == "V":
            assert loss.item() == 5.566722869873047, (
                f"loss not equal ({loss.item()} != 5.566722869873047), please check your modify"
            )

        loss.backward()

        for name, param in self.gpt_model.named_parameters():
            # 计算 L2 范数
            if param.grad is None:
                print(f"{name}: 0.000000, 0.000000")
                continue
            grad_norm = param.grad.detach().norm().item()
            grad_abssum = param.grad.detach().abs().sum().item()
            # print(f"{name}: {param.shape}, {param_norm:.6f}")
            print(f"{name}: {grad_norm:.6f}, {grad_abssum:.6f}")
            if name == "embedding.embed_tokens.weight":
                word_embeddings_grad_norm = grad_norm

        print("word_embeddings_grad_norm", word_embeddings_grad_norm)
        if judge_machine_type() == "H":
            assert word_embeddings_grad_norm == 6.112863540649414, (
                f"grad norm of word_embeddingsnot not equal ({word_embeddings_grad_norm} != 6.112863540649414), please check your modify"
            )
        elif judge_machine_type() == "V":
            assert word_embeddings_grad_norm == 9.869551658630371, (
                f"grad norm of word_embeddingsnot not equal ({word_embeddings_grad_norm} != 9.869551658630371, please check your modify"
            )


if __name__ == "__main__":
    unittest.main()
