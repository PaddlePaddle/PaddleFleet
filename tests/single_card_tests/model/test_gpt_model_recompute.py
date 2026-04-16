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
from paddle.distributed.fleet.meta_parallel import NoPipelineParallel

# from tests.unit_tests.test_utilities import Utils
import paddlefleet.parallel_state as ps

# from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddlefleet.gpt_builders import gpt_builder
from paddlefleet.models.gpt import GPTConfig


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
    if paddle.device.get_device_capability()[0] == 9:
        return "H"
    else:
        return "V"


result = judge_machine_type()
print("Your machine type is:", result)


class TestGPTModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize distributed environment, only need to execute once"""
        seed = 46
        random.seed(seed)
        np.random.seed(seed)
        paddle.manual_seed(seed)
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

    def setUp(self):
        """Reset random seed before each test case"""
        seed = 46
        random.seed(seed)
        np.random.seed(seed)
        paddle.manual_seed(seed)
        fleet_env = fleet.fleet
        self.strategy = fleet_env._user_defined_strategy

    def _create_transformer_configs(self):
        """Create multiple different transformer configurations for testing"""
        configs = []

        config1 = GPTConfig(
            num_hidden_layers=2,
            hidden_size=512,
            vocab_size=100,
            max_sequence_length=64,
            num_attention_heads=4,
            intermediate_size=1024,
            normalization="RMSNorm",
            hidden_dropout_prob=0.0,
            first_k_dense_replace=1,
            attention_dropout=0.0,
            n_routed_experts=8,
            use_bias=False,
            rotary_percent=1.0,
            rotary_base=10000,
            rope_scaling=1.0,
            moe_intermediate_size=1024,
            moe_token_dispatcher_type="alltoall",
            n_shared_experts=1,
            init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            output_layer_init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            tie_word_embeddings=True,
            use_qk_norm=True,
            recompute_granularity="selective",
            recompute_modules=[
                "core_attn",
                "norm",
                "mlp",
                "lm_head",
                "embedding",
                "loss_fn",
            ],
        )
        config1.name = "config_1"
        configs.append(config1)

        return configs

    def _create_gpt_model(self, config):
        """Create GPT model based on given configuration"""
        self.gpt_model = gpt_builder(config, num_stages=1)
        return self.gpt_model

    def _get_expected_values(self, config_name, machine_type):
        """Return expected values based on configuration name and machine type"""
        # Define expected values for different configurations on different machines
        expectations = {
            "config_1": {
                "H": {
                    "loss": 5.3251447677612305,
                    "grad_norm": 5.3691630363464355,
                },
                "V": {
                    "loss": 5.3251447677612305,
                    "grad_norm": 5.3691630363464355,
                },
            },
        }

        return expectations.get(config_name, {}).get(machine_type, {})

    def test_multiple_configurations(self):
        """Test multiple transformer configurations"""
        configs = self._create_transformer_configs()
        machine_type = judge_machine_type()

        for config in configs:
            with self.subTest(config_name=config.name):
                # Explicitly call setUp to ensure initialization for each config
                self.setUp()

                print(f"\nTesting configuration: {config.name}")
                print(
                    f"hidden_size: {config.hidden_size}, num_layers: {config.num_hidden_layers}"
                )

                # Create and test model
                gpt_model = self._create_gpt_model(config)

                # Run forward and backward propagation
                sequence_length = config.max_sequence_length
                micro_batch_size = 1

                data = list(range(sequence_length))
                input_ids = paddle.to_tensor(data, dtype=paddle.int64).repeat(
                    (micro_batch_size, 1)
                )
                position_ids = paddle.to_tensor(
                    data, dtype=paddle.int64
                ).repeat((micro_batch_size, 1))
                attention_mask = paddle.ones(
                    (micro_batch_size, 1, sequence_length, sequence_length),
                    dtype=bool,
                )
                labels = paddle.to_tensor(
                    list(range(1, sequence_length + 1)), dtype=paddle.int64
                ).repeat((micro_batch_size, 1))

                data = (
                    {
                        "input_ids": [input_ids],
                        "position_ids": [position_ids],
                        "attention_mask": [attention_mask],
                    },
                    [labels],
                )

                gpt_pipe_model = NoPipelineParallel(
                    self.gpt_model, self.strategy
                )
                loss = gpt_pipe_model.forward_backward_pipeline(data)

                for name, param in self.gpt_model.named_parameters():
                    # 计算 L2 范数
                    if param.grad is None:
                        print(f"{name}: 0.000000, 0.000000")
                        continue
                    grad_norm = param.grad.detach().norm().item()
                    grad_abssum = param.grad.detach().abs().sum().item()
                    print(f"{name}: {grad_norm:.6f}, {grad_abssum:.6f}")
                    if name == "0.embedding.embed_tokens.weight":
                        embed_tokens_grad_norm = grad_norm

                print(f"{config.name} loss: {loss.item()}")

                # Get expected values
                print("machine_type: ", machine_type)
                expected_values = self._get_expected_values(
                    config.name, machine_type
                )

                # If expected values exist, verify them
                if expected_values.get("loss") is not None:
                    self.assertAlmostEqual(
                        loss.item(),
                        expected_values["loss"],
                        places=5,
                        msg=f"{config.name} loss not equal ({loss.item()} != {expected_values['loss']})",
                    )
                else:
                    print(
                        f"Note: Expected loss value for {config.name} is not set, current loss: {loss.item()}"
                    )

                if embed_tokens_grad_norm is not None:
                    print(
                        f"{config.name} embed_tokens_grad_norm: {embed_tokens_grad_norm}"
                    )

                    if expected_values.get("grad_norm") is not None:
                        self.assertAlmostEqual(
                            embed_tokens_grad_norm,
                            expected_values["grad_norm"],
                            places=5,
                            msg=f"{config.name} grad norm not equal ({embed_tokens_grad_norm} != {expected_values['grad_norm']})",
                        )
                    else:
                        print(
                            f"Note: Expected grad_norm value for {config.name} is not set, current grad_norm: {embed_tokens_grad_norm}"
                        )
                else:
                    print(
                        f"Warning: Failed to get word_embeddings gradient for {config.name}"
                    )


if __name__ == "__main__":
    unittest.main()
