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

    def _create_transformer_configs(self):
        """Create multiple different transformer configurations for testing"""
        configs = []

        config1 = TransformerConfig(
            num_hidden_layers=2,
            hidden_size=512,
            num_attention_heads=4,
            intermediate_size=1024,
            normalization="RMSNorm",
            hidden_dropout_prob=0.0,
            attention_dropout=0.0,
            init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            output_layer_init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            fuse_rms_norm=True,
        )
        config1.name = "config_1"
        configs.append(config1)

        config2 = TransformerConfig(
            num_hidden_layers=2,
            hidden_size=512,
            num_attention_heads=4,
            intermediate_size=1024,
            normalization="RMSNorm",
            hidden_dropout_prob=0.0,
            attention_dropout=0.0,
            init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            output_layer_init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            fuse_rms_norm=True,
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=1,
        )
        config2.name = "config_2"
        configs.append(config2)

        config3 = TransformerConfig(
            num_hidden_layers=2,
            hidden_size=512,
            num_attention_heads=4,
            intermediate_size=1024,
            normalization="RMSNorm",
            hidden_dropout_prob=0.0,
            attention_dropout=0.0,
            init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            output_layer_init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            fuse_rms_norm=True,
            recompute_granularity="full",
            recompute_method="block",
            recompute_num_layers=1,
        )
        config3.name = "config_3"
        configs.append(config3)

        config4 = TransformerConfig(
            num_hidden_layers=2,
            hidden_size=512,
            num_attention_heads=4,
            intermediate_size=1024,
            normalization="RMSNorm",
            hidden_dropout_prob=0.0,
            attention_dropout=0.0,
            init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            output_layer_init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            fuse_rms_norm=True,
            recompute_granularity="selective",
            recompute_modules=["core_attn", "mlp", "layernorm"],
        )
        config4.name = "config_4"
        configs.append(config4)
        config5 = TransformerConfig(
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
            moe_shared_expert_intermediate_size=1024,
            fuse_rms_norm=True,
            init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            output_layer_init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
        )

        config5.name = "config_5"
        configs.append(config5)

        config6 = TransformerConfig(
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
            moe_shared_expert_intermediate_size=1024,
            init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            output_layer_init_method=functools.partial(
                paddle.nn.init.xavier_uniform_, gain=1.0
            ),
            fuse_rms_norm=True,
            recompute_granularity="selective",
            recompute_modules=["core_attn", "moe", "shared_experts"],
        )
        config6.name = "config_6"
        configs.append(config6)
        return configs

    def _create_gpt_model(self, config):
        """Create GPT model based on given configuration"""
        transformer_layer_spec = get_gpt_layer_local_spec(
            num_experts=config.moe_num_experts,
            moe_grouped_gemm=False,
            use_qk_norm=True,
            multi_latent_attention=False,
            normalization="RMSNorm",
        )

        pre_process = True
        post_process = True
        mtp_block_spec = None
        vp_stage = None

        return GPTModel(
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

    def _get_expected_values(self, config_name, machine_type):
        """Return expected values based on configuration name and machine type"""
        # Define expected values for different configurations on different machines
        expectations = {
            "config_1": {
                "H": {
                    "loss": 5.3645853996276855,
                    "grad_norm": 4.1039042472839355,
                },
                "V": {
                    "loss": 5.249175071716309,
                    "grad_norm": 4.636361598968506,
                },
            },
            "config_2": {
                "H": {
                    "loss": 5.3645853996276855,
                    "grad_norm": 4.1039042472839355,
                },
                "V": {
                    "loss": 5.249175071716309,
                    "grad_norm": 4.636361598968506,
                },
            },
            "config_3": {
                "H": {
                    "loss": 5.3645853996276855,
                    "grad_norm": 4.1039042472839355,
                },
                "V": {
                    "loss": 5.249175071716309,
                    "grad_norm": 4.636361598968506,
                },
            },
            "config_4": {
                "H": {
                    "loss": 5.3645853996276855,
                    "grad_norm": 4.1039042472839355,
                },
                "V": {
                    "loss": 5.249175071716309,
                    "grad_norm": 4.636361598968506,
                },
            },
            "config_5": {
                "H": {
                    "loss": 5.344995498657227,
                    "grad_norm":  6.112863540649414,
                },
                "V": {
                    "loss": 5.566722869873047,
                    "grad_norm": 9.869551658630371,
                },
            },
            "config_6": {
                "H": {
                    "loss": 5.344995498657227,
                    "grad_norm":  6.112863540649414,
                },
                "V": {
                    "loss": 5.566722869873047,
                    "grad_norm": 9.869551658630371,
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
                sequence_length = gpt_model.max_sequence_length
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

                outputs = gpt_model.forward(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs["loss"]
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

                loss.backward()

                # Check gradients
                word_embeddings_grad_norm = None
                for name, param in gpt_model.named_parameters():
                    if (
                        param.grad is not None
                        and name == "embedding.word_embeddings.weight"
                    ):
                        word_embeddings_grad_norm = (
                            param.grad.detach().norm().item()
                        )
                        break

                if word_embeddings_grad_norm is not None:
                    print(
                        f"{config.name} word_embeddings_grad_norm: {word_embeddings_grad_norm}"
                    )

                    if expected_values.get("grad_norm") is not None:
                        self.assertAlmostEqual(
                            word_embeddings_grad_norm,
                            expected_values["grad_norm"],
                            places=5,
                            msg=f"{config.name} grad norm not equal ({word_embeddings_grad_norm} != {expected_values['grad_norm']})",
                        )
                    else:
                        print(
                            f"Note: Expected grad_norm value for {config.name} is not set, current grad_norm: {word_embeddings_grad_norm}"
                        )
                else:
                    print(
                        f"Warning: Failed to get word_embeddings gradient for {config.name}"
                    )


if __name__ == "__main__":
    unittest.main()
