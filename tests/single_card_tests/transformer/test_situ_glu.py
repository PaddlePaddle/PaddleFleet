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

import os
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import paddle
import paddle.nn.functional as F

from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddlefleet.transformer.activations import (
    SituAndMul,
    situ,
    situ_glu,
    situ_glu_scale_backward,
    situ_glu_scale_forward,
)
from paddlefleet.transformer.mlp import MLP
from paddlefleet.transformer.moe.fp8_utils import (
    ExpertsGroupGemmContiguousNode,
)
from paddlefleet.transformer.moe.fusion_layer_utils import FusionMoePyLayer
from paddlefleet.transformer.moe.moe_expert import (
    GroupedMLPExpert,
    SonicMoEExpert,
)
from paddlefleet.transformer.moe.moe_layer import MoELayer
from paddlefleet.transformer.transformer_config import TransformerConfig


class TestSituGLU(unittest.TestCase):
    def test_default_fused_falls_back_without_triton(self):
        x = paddle.randn([3, 8], dtype="float32")
        probs = paddle.rand([3], dtype="float32")
        out_grad = paddle.randn([3, 4], dtype="float32")
        expected_forward = situ_glu_scale_forward(
            x, probs, situ_glu_fusion=False
        )
        expected_backward = situ_glu_scale_backward(
            x, probs, out_grad, situ_glu_fusion=False
        )

        with mock.patch(
            "paddlefleet.triton_ops.utils.is_triton_available",
            return_value=False,
        ):
            actual_forward = situ_glu_scale_forward(x, probs)
            actual_backward = situ_glu_scale_backward(x, probs, out_grad)

        self.assertTrue(paddle.equal_all(actual_forward, expected_forward))
        for actual, expected in zip(
            actual_backward, expected_backward, strict=True
        ):
            self.assertTrue(paddle.equal_all(actual, expected))

    def test_config_resolves_situ_name(self):
        config = TransformerConfig.from_config(
            SimpleNamespace(
                hidden_size=8,
                num_attention_heads=2,
                hidden_act="situ",
            )
        )

        self.assertIs(config.hidden_act, situ)
        self.assertTrue(config.situ_glu_fusion)

        disabled_config = TransformerConfig.from_config(
            SimpleNamespace(
                hidden_size=8,
                num_attention_heads=2,
                hidden_act="situ",
                situ_glu_fusion=False,
            )
        )
        self.assertFalse(disabled_config.situ_glu_fusion)

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

    def test_rejects_invalid_scales(self):
        x = paddle.ones([2, 8])
        probs = paddle.ones([2])
        out_grad = paddle.ones([2, 4])
        invalid_scales = (
            -1.0,
            0.0,
            float("-inf"),
            float("inf"),
            float("nan"),
        )

        for beta in invalid_scales:
            with (
                self.subTest(beta=beta, function="situ"),
                self.assertRaisesRegex(ValueError, "positive finite"),
            ):
                situ(x, beta=beta)
            with (
                self.subTest(beta=beta, function="situ_glu"),
                self.assertRaisesRegex(ValueError, "positive finite"),
            ):
                situ_glu(x, beta=beta)
            with (
                self.subTest(beta=beta, function="situ_glu_scale_forward"),
                self.assertRaisesRegex(ValueError, "positive finite"),
            ):
                situ_glu_scale_forward(x, probs, beta=beta)
            with (
                self.subTest(beta=beta, function="situ_glu_scale_backward"),
                self.assertRaisesRegex(ValueError, "positive finite"),
            ):
                situ_glu_scale_backward(x, probs, out_grad, beta=beta)

        for linear_beta in invalid_scales:
            with (
                self.subTest(
                    linear_beta=linear_beta,
                    function="situ_glu",
                ),
                self.assertRaisesRegex(ValueError, "positive finite"),
            ):
                situ_glu(x, linear_beta=linear_beta)
            with (
                self.subTest(
                    linear_beta=linear_beta,
                    function="situ_glu_scale_forward",
                ),
                self.assertRaisesRegex(ValueError, "positive finite"),
            ):
                situ_glu_scale_forward(x, probs, linear_beta=linear_beta)
            with (
                self.subTest(
                    linear_beta=linear_beta,
                    function="situ_glu_scale_backward",
                ),
                self.assertRaisesRegex(ValueError, "positive finite"),
            ):
                situ_glu_scale_backward(
                    x,
                    probs,
                    out_grad,
                    linear_beta=linear_beta,
                )

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

    def test_grouped_deep_gemm_situ_forward_backward(self):
        model_parallel_cuda_manual_seed(2026)
        config = TransformerConfig(
            hidden_size=512,
            num_hidden_layers=1,
            num_attention_heads=8,
            intermediate_size=256,
            moe_intermediate_size=256,
            gated_linear_unit=True,
            hidden_act=situ,
            activation_situ_beta=4.0,
            activation_situ_linear_beta=25.0,
            params_dtype="bfloat16",
        )
        expert = GroupedMLPExpert(
            num_local_experts=2,
            config=config,
            moe_deep_gemm=True,
        )
        with paddle.no_grad():
            expert.weight1.set_value(
                paddle.randn(expert.weight1.shape, dtype="bfloat16") * 0.01
            )
            expert.weight2.set_value(
                paddle.randn(expert.weight2.shape, dtype="bfloat16") * 0.01
            )
        hidden_states = paddle.randn([256, 512], dtype="bfloat16")
        hidden_states.stop_gradient = False

        output, output_bias = expert(
            hidden_states,
            paddle.to_tensor([128, 128], dtype="int64"),
        )
        output.sum().backward()

        self.assertIsNone(output_bias)
        for name, tensor in (
            ("output", output),
            ("input_grad", hidden_states.grad),
            ("weight1_grad", expert.weight1.grad),
            ("weight2_grad", expert.weight2.grad),
        ):
            with self.subTest(name=name):
                self.assertIsNotNone(tensor)
                self.assertTrue(
                    bool(paddle.isfinite(tensor.astype("float32")).all())
                )

    def test_situ_glu_scale_matches_autograd(self):
        paddle.seed(2026)
        x = paddle.randn([7, 16], dtype="bfloat16")
        out_grad = paddle.randn([7, 8], dtype="bfloat16")

        for linear_beta, probs_shape in ((25.0, [7]), (None, [7, 1])):
            with self.subTest(
                linear_beta=linear_beta,
                probs_shape=probs_shape,
            ):
                probs = paddle.rand(probs_shape, dtype="float32")
                x_ref = x.detach()
                probs_ref = probs.detach()
                x_ref.stop_gradient = False
                probs_ref.stop_gradient = False
                gate, up = paddle.chunk(x_ref, chunks=2, axis=-1)
                gate = gate.astype("float32")
                up = up.astype("float32")
                gate = 4.0 * paddle.tanh(gate / 4.0) * F.sigmoid(gate)
                if linear_beta is not None:
                    up = linear_beta * paddle.tanh(up / linear_beta)
                probs_view = (
                    probs_ref.unsqueeze(-1)
                    if probs_ref.ndim == 1
                    else probs_ref
                )
                expected = (gate * up * probs_view).astype(x.dtype)
                paddle.autograd.backward([expected], [out_grad])

                actual = situ_glu_scale_forward(
                    x,
                    probs,
                    4.0,
                    linear_beta,
                )
                x_grad, recomputed, probs_grad = situ_glu_scale_backward(
                    x,
                    probs,
                    out_grad,
                    4.0,
                    linear_beta,
                )

                self.assertTrue(
                    paddle.equal_all(
                        actual.astype("float32"),
                        expected.detach().astype("float32"),
                    )
                )
                self.assertTrue(
                    paddle.equal_all(
                        recomputed.astype("float32"),
                        actual.astype("float32"),
                    )
                )
                self.assertTrue(
                    paddle.equal_all(
                        x_grad.astype("float32"),
                        x_ref.grad.astype("float32"),
                    )
                )
                self.assertTrue(
                    paddle.allclose(
                        probs_grad,
                        probs_ref.grad,
                        atol=1e-6,
                        rtol=1e-6,
                    )
                )

    def test_fused_node_rejects_unsupported_fp8_activations(self):
        for activation_type in ("geglu", "situ"):
            with self.subTest(activation_type=activation_type):
                node = ExpertsGroupGemmContiguousNode.__new__(
                    ExpertsGroupGemmContiguousNode
                )
                node.use_fp8_mlp = True
                node.activation_type = activation_type

                with self.assertRaisesRegex(
                    ValueError, "only supports.*swiglu"
                ):
                    node.fwd_down(None, None, None, 0)

    def test_runtime_guards_survive_optimized_mode(self):
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../..")
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [
                os.path.join(repo_root, "src"),
                os.path.dirname(__file__),
                env.get("PYTHONPATH", ""),
            ]
        )
        proc = subprocess.run(
            [
                sys.executable,
                "-O",
                "-m",
                "unittest",
                "-v",
                "test_situ_glu.TestSituGLU.test_rejects_invalid_scales",
                "test_situ_glu.TestSituGLU."
                "test_fused_node_rejects_unsupported_fp8_activations",
            ],
            capture_output=True,
            text=True,
            env=env,
        )

        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_moe_layer_accepts_situ_fusion_options(self):
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
            "gated_linear_unit": True,
            "hidden_act": situ,
            "activation_situ_beta": 4.0,
            "activation_situ_linear_beta": 25.0,
        }

        for expert_fusion, deep_gemm in (
            (False, False),
            (True, False),
            (True, True),
        ):
            with self.subTest(
                expert_fusion=expert_fusion,
                deep_gemm=deep_gemm,
            ):
                config = TransformerConfig(
                    **config_kwargs,
                    moe_use_fusion_node=True,
                    moe_expert_fusion=expert_fusion,
                    moe_deep_gemm=deep_gemm,
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
                self.assertEqual(layer._activation_type, "situ")
                self.assertTrue(layer.moe_use_fusion_node)
                self.assertEqual(layer.moe_expert_fusion, expert_fusion)
                self.assertEqual(layer.moe_deep_gemm, deep_gemm)

        fp8_config = TransformerConfig(
            **config_kwargs,
            moe_use_fusion_node=True,
            moe_expert_fusion=True,
            moe_deep_gemm=True,
            fp8="e4m3",
        )
        with self.assertRaisesRegex(ValueError, "supports BF16"):
            MoELayer(fp8_config)

    def test_situ_fused_grouped_deep_gemm_forward_backward(self):
        model_parallel_cuda_manual_seed(2026)
        paddle.seed(2026)
        hidden_size = 512
        intermediate_size = 256
        tokens_per_expert = [128, 128]
        config = TransformerConfig(
            hidden_size=hidden_size,
            num_hidden_layers=1,
            num_attention_heads=8,
            intermediate_size=intermediate_size,
            moe_intermediate_size=intermediate_size,
            gated_linear_unit=True,
            hidden_act=situ,
            activation_situ_beta=4.0,
            activation_situ_linear_beta=25.0,
            params_dtype="bfloat16",
        )
        weight1 = (
            paddle.randn(
                [2, hidden_size, 2 * intermediate_size], dtype="bfloat16"
            )
            * 0.01
        )
        weight2 = (
            paddle.randn([2, intermediate_size, hidden_size], dtype="bfloat16")
            * 0.01
        )
        hidden_states = paddle.randn([256, hidden_size], dtype="bfloat16")
        probs = paddle.rand([256], dtype="float32")
        out_grad = paddle.randn([256, hidden_size], dtype="bfloat16")

        def run(
            deep_gemm,
            use_accuracy_compatible=False,
            situ_glu_fusion=True,
        ):
            config.situ_glu_fusion = situ_glu_fusion
            expert = GroupedMLPExpert(
                num_local_experts=2,
                config=config,
                moe_deep_gemm=deep_gemm,
            )
            expert.weight1.set_value(weight1)
            expert.weight2.set_value(weight2)
            expert.weight1.main_grad = None
            expert.weight2.main_grad = None
            custom_map = SimpleNamespace(
                config=config,
                grouped_gemm_experts=expert,
                token_dispatcher=SimpleNamespace(
                    _comm_manager=SimpleNamespace(
                        tokens_per_expert=tokens_per_expert
                    )
                ),
            )
            node = ExpertsGroupGemmContiguousNode(
                custom_map,
                use_fp8_mlp=False,
                moe_expert_fusion=True,
                moe_deep_gemm=deep_gemm,
                activation_type="situ",
                use_accuracy_compatible=use_accuracy_compatible,
            )
            output = node.forward(hidden_states, probs, tokens_per_expert)
            input_grad, probs_grad = node.backward(out_grad, probs)
            return (
                output,
                input_grad,
                probs_grad,
                expert.weight1.main_grad,
                expert.weight2.main_grad,
            )

        grouped = run(False)
        deep_gemm = run(True)
        accuracy_compatible = run(False, use_accuracy_compatible=True)
        unfused_situ_glu = run(False, situ_glu_fusion=False)
        tolerances = (
            (1e-3, 1e-3),
            (1e-3, 1e-3),
            (1e-4, 1e-4),
            (5e-3, 5e-3),
            (5e-3, 5e-3),
        )
        for path, actuals in (
            ("deep_gemm", deep_gemm),
            ("accuracy_compatible", accuracy_compatible),
            ("unfused_situ_glu", unfused_situ_glu),
        ):
            for name, expected, actual, (atol, rtol) in zip(
                (
                    "output",
                    "input_grad",
                    "probs_grad",
                    "weight1_grad",
                    "weight2_grad",
                ),
                grouped,
                actuals,
                tolerances,
            ):
                with self.subTest(path=path, name=name):
                    expected_fp32 = expected.astype("float32")
                    actual_fp32 = actual.astype("float32")
                    self.assertTrue(
                        paddle.allclose(
                            expected_fp32,
                            actual_fp32,
                            atol=atol,
                            rtol=rtol,
                        ),
                        msg=(
                            f"{path} {name} max_abs="
                            f"{float((expected_fp32 - actual_fp32).abs().max())}"
                        ),
                    )

    def test_situ_fusion_moe_deep_gemm_smoke(self):
        model_parallel_cuda_manual_seed(2026)
        paddle.seed(2026)
        hidden_size = 512
        intermediate_size = 256
        tokens_per_expert = [128, 128]
        config = TransformerConfig(
            hidden_size=hidden_size,
            num_hidden_layers=1,
            num_attention_heads=8,
            intermediate_size=intermediate_size,
            moe_intermediate_size=intermediate_size,
            gated_linear_unit=True,
            hidden_act=situ,
            activation_situ_beta=4.0,
            activation_situ_linear_beta=25.0,
            params_dtype="bfloat16",
        )
        expert = GroupedMLPExpert(
            num_local_experts=2,
            config=config,
            moe_deep_gemm=True,
        )
        with paddle.no_grad():
            expert.weight1.set_value(
                paddle.randn(expert.weight1.shape, dtype="bfloat16") * 0.01
            )
            expert.weight2.set_value(
                paddle.randn(expert.weight2.shape, dtype="bfloat16") * 0.01
            )
        expert.weight1.main_grad = None
        expert.weight2.main_grad = None
        moe_layer = SimpleNamespace(
            config=config,
            _activation_type="situ",
            moe_use_fusion_node=True,
            grouped_gemm_experts=expert,
            token_dispatcher=SimpleNamespace(
                _comm_manager=SimpleNamespace(
                    tokens_per_expert=tokens_per_expert
                )
            ),
        )
        hidden_states = paddle.randn([256, hidden_size], dtype="bfloat16")
        hidden_states.stop_gradient = False
        probs = paddle.rand([256, 1], dtype="float32")
        probs.stop_gradient = False
        indices = paddle.concat(
            [
                paddle.zeros([128, 1], dtype="int64"),
                paddle.ones([128, 1], dtype="int64"),
            ],
            axis=0,
        )

        output = FusionMoePyLayer.apply(
            hidden_states,
            probs,
            indices,
            moe_layer,
            1,
            use_fp8_mlp=False,
            moe_deep_gemm=True,
            moe_expert_fusion=True,
        )
        output.sum().backward()

        for name, tensor in (
            ("output", output),
            ("input_grad", hidden_states.grad),
            ("probs_grad", probs.grad),
            ("weight1_grad", expert.weight1.main_grad),
            ("weight2_grad", expert.weight2.main_grad),
        ):
            with self.subTest(name=name):
                self.assertIsNotNone(tensor)
                self.assertTrue(
                    bool(paddle.isfinite(tensor.astype("float32")).all())
                )

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
