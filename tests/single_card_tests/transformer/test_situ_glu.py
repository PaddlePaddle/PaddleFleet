# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
from types import SimpleNamespace

import paddle
import paddle.nn.functional as F

from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddlefleet.transformer.activations import SituAndMul, situ, situ_glu
from paddlefleet.transformer.mlp import MLP
from paddlefleet.transformer.moe.moe_expert import (
    GroupedMLPExpert,
    SonicMoEExpert,
)
from paddlefleet.transformer.moe.moe_layer import MoELayer
from paddlefleet.transformer.transformer_config import TransformerConfig


class TestSituGLU(unittest.TestCase):
    def test_config_resolves_situ_name(self):
        config = TransformerConfig.from_config(
            SimpleNamespace(
                hidden_size=8,
                num_attention_heads=2,
                hidden_act="situ",
            )
        )

        self.assertIs(config.hidden_act, situ)

    def test_matches_official_formula(self):
        x = paddle.linspace(-8.0, 8.0, 32).reshape([2, 16])
        gate, up = paddle.chunk(x, chunks=2, axis=-1)
        expected_gate = 4.0 * paddle.tanh(gate / 4.0) * F.sigmoid(gate)
        expected = expected_gate * (25.0 * paddle.tanh(up / 25.0))

        gate_actual = situ(gate, beta=4.0)
        actual = situ_glu(x, beta=4.0, linear_beta=25.0)
        layer_actual = SituAndMul(beta=4.0, linear_beta=25.0)(x)

        self.assertTrue(paddle.allclose(gate_actual, expected_gate))
        self.assertTrue(paddle.allclose(actual, expected))
        self.assertTrue(paddle.equal_all(actual, layer_actual))

    def test_rejects_odd_projection_width(self):
        with self.assertRaisesRegex(ValueError, "even last dimension"):
            situ_glu(paddle.ones([2, 7]))

    def test_rejects_non_positive_scales(self):
        x = paddle.ones([2, 8])
        invalid_scales = (-1.0, 0.0)

        for beta in invalid_scales:
            with (
                self.subTest(beta=beta, function="situ"),
                self.assertRaises(AssertionError),
            ):
                situ(x, beta=beta)
            with (
                self.subTest(beta=beta, function="situ_glu"),
                self.assertRaises(AssertionError),
            ):
                situ_glu(x, beta=beta)

        for linear_beta in invalid_scales:
            with (
                self.subTest(linear_beta=linear_beta),
                self.assertRaises(AssertionError),
            ):
                situ_glu(x, linear_beta=linear_beta)

    def test_grouped_bf16_expert_forward_backward(self):
        model_parallel_cuda_manual_seed(2026)
        config = TransformerConfig(
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=8,
            moe_intermediate_size=4,
            gated_linear_unit=True,
            hidden_act=situ,
            activation_situ_beta=4.0,
            activation_situ_linear_beta=25.0,
            params_dtype="bfloat16",
        )
        expert = GroupedMLPExpert(
            num_local_experts=2,
            config=config,
            moe_deep_gemm=False,
        )
        self.assertIs(config.hidden_act, situ)
        hidden_states = paddle.randn([5, 8], dtype="bfloat16")
        hidden_states.stop_gradient = False

        output, output_bias = expert(
            hidden_states,
            paddle.to_tensor([2, 3], dtype="int64"),
        )
        output.sum().backward()

        self.assertIsNone(output_bias)
        self.assertEqual(list(output.shape), [5, config.hidden_size])
        self.assertTrue(
            bool(paddle.isfinite(output.astype("float32")).all().item())
        )
        self.assertTrue(
            bool(
                paddle.isfinite(hidden_states.grad.astype("float32"))
                .all()
                .item()
            )
        )

    def test_moe_layer_rejects_situ_fusion_options(self):
        model_parallel_cuda_manual_seed(2026)
        config_kwargs = {
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "intermediate_size": 8,
            "n_routed_experts": 2,
            "n_shared_experts": 0,
            "num_experts_per_tok": 1,
            "moe_intermediate_size": 4,
            "moe_deep_gemm": False,
            "gated_linear_unit": True,
            "hidden_act": situ,
        }

        for option in ("moe_use_fusion_node", "moe_expert_fusion"):
            with self.subTest(option=option):
                config = TransformerConfig(
                    **config_kwargs,
                    moe_use_fusion_node=option == "moe_use_fusion_node",
                    moe_expert_fusion=option == "moe_expert_fusion",
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "support will be added in a future release",
                ):
                    MoELayer(config)

        config = TransformerConfig(
            **config_kwargs,
            moe_use_fusion_node=False,
            moe_expert_fusion=False,
        )
        layer_spec = get_gpt_layer_local_spec(
            config,
            num_experts=config.n_routed_experts,
        )
        layer = MoELayer(
            config,
            layer_spec.sublayers_spec.mlp.extra_kwargs["sublayers"],
            SimpleNamespace(ep=None, expt_dp=None),
        )
        self.assertFalse(layer.moe_use_fusion_node)
        self.assertFalse(layer.moe_expert_fusion)

    def test_grouped_expert_preserves_standard_activation_config(self):
        model_parallel_cuda_manual_seed(2026)
        for hidden_act in (F.silu, F.gelu):
            with self.subTest(hidden_act=hidden_act):
                config = TransformerConfig(
                    hidden_size=8,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    intermediate_size=8,
                    moe_intermediate_size=4,
                    gated_linear_unit=True,
                    hidden_act=hidden_act,
                    params_dtype="bfloat16",
                )
                expert = GroupedMLPExpert(
                    num_local_experts=2,
                    config=config,
                    moe_deep_gemm=False,
                )
                x = paddle.linspace(-4.0, 4.0, 16).reshape([2, 8])
                gate, up = paddle.chunk(x, chunks=2, axis=-1)

                self.assertIs(config.hidden_act, hidden_act)
                self.assertTrue(
                    paddle.allclose(
                        expert.activation_func(x),
                        hidden_act(gate) * up,
                    )
                )

    def test_sonic_moe_rejects_non_swiglu_configurations(self):
        for hidden_act, gated_linear_unit in (
            (F.gelu, True),
            (situ, True),
            (F.silu, False),
        ):
            with self.subTest(
                hidden_act=hidden_act,
                gated_linear_unit=gated_linear_unit,
            ):
                config = TransformerConfig(
                    hidden_size=8,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    intermediate_size=8,
                    moe_intermediate_size=4,
                    gated_linear_unit=gated_linear_unit,
                    hidden_act=hidden_act,
                    params_dtype="bfloat16",
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "only supports SwiGLU",
                ):
                    SonicMoEExpert(
                        num_local_experts=2,
                        topk=2,
                        config=config,
                    )
                self.assertIs(config.hidden_act, hidden_act)

    def test_mlp_forward_backward_bypasses_swiglu_fusion(self):
        model_parallel_cuda_manual_seed(2026)
        config = TransformerConfig(
            num_hidden_layers=2,
            hidden_size=12,
            intermediate_size=24,
            num_attention_heads=4,
            use_bias=True,
            bias_activation_fusion=True,
            gated_linear_unit=True,
            hidden_act=situ,
            activation_situ_beta=4.0,
            activation_situ_linear_beta=25.0,
        )
        mlp = MLP(
            config,
            get_gpt_layer_local_spec(config).sublayers_spec.mlp.sublayers_spec,
        )
        hidden_states = paddle.randn([4, 2, config.hidden_size])
        hidden_states.stop_gradient = False

        output, output_bias = mlp(hidden_states)
        (output.sum() + output_bias.sum()).backward()

        self.assertEqual(list(output.shape), [4, 2, config.hidden_size])
        self.assertTrue(bool(paddle.isfinite(output).all().item()))
        self.assertTrue(bool(paddle.isfinite(hidden_states.grad).all().item()))
