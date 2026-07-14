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
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)


import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _make_expert_config(**overrides):
    """Helper to create a config for expert testing."""
    from paddlefleet.transformer.transformer_config import TransformerConfig

    defaults = {
        "hidden_size": 64,
        "num_attention_heads": 2,
        "intermediate_size": 256,
        "moe_intermediate_size": 128,
        "moe_deep_gemm": False,
        "gated_linear_unit": True,
        "sequence_parallel": False,
        "tensor_model_parallel_size": 1,
        "recompute_granularity": None,
        "recompute_modules": [],
        "fp8": None,
        "using_sonic_moe": False,
        "use_bias": False,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


class TestMoeExpert(unittest.TestCase):
    """Unit tests for moe_expert module."""

    @patch("paddlefleet.transformer.moe.moe_expert.utils")
    def test_grouped_mlp_expert_init(self, mock_utils):
        """Test GroupedMLPExpert initialization."""
        from paddlefleet.transformer.moe.moe_expert import GroupedMLPExpert

        mock_utils.get_pg_size.return_value = 1
        config = _make_expert_config()
        pg_collection = MagicMock()
        pg_collection.ep = None

        expert = GroupedMLPExpert(
            num_local_experts=2,
            config=config,
            moe_deep_gemm=False,
            pg_collection=pg_collection,
        )
        self.assertEqual(expert.num_local_experts, 2)
        self.assertIsNotNone(expert.weight1)
        self.assertIsNotNone(expert.weight2)
        self.assertEqual(expert.weight1.shape[0], 2)

    def test_sonic_moe_expert_inherits_activation_clamp(self):
        """Test SonicMoE uses the shared activation clamp configuration."""
        from paddlefleet.transformer.moe import moe_expert

        for clamp_value, expected in ((7.5, 7.5), (None, 0.0)):
            with self.subTest(clamp_value=clamp_value):
                config = _make_expert_config(
                    activation_func_clamp_value=clamp_value
                )
                runtime_config = SimpleNamespace()
                with (
                    patch.object(
                        moe_expert.GroupedMLPExpert,
                        "__init__",
                        return_value=None,
                    ),
                    patch.object(
                        moe_expert.SonicMoEExpert,
                        "config",
                        config,
                        create=True,
                    ),
                    patch.object(
                        moe_expert,
                        "_refresh_fp8_config",
                        return_value=runtime_config,
                        create=True,
                    ),
                ):
                    moe_expert.SonicMoEExpert(
                        num_local_experts=2,
                        topk=2,
                        config=config,
                    )

                self.assertEqual(runtime_config.swiglu_clamp_value, expected)
