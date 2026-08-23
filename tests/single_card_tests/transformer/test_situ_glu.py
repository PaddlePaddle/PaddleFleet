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
    def test_opt_in_fusion_falls_back_without_triton(self):
        x = paddle.randn([3, 8], dtype="float32")
        probs = paddle.rand([3], dtype="float32")
        out_grad = paddle.randn([3, 4], dtype="float32")
        expected_forward = situ_glu_scale_forward(x, probs)
        expected_backward = situ_glu_scale_backward(x, probs, out_grad)

        with mock.patch(
            "paddlefleet.triton_ops.utils.is_triton_available",
            return_value=False,
        ):
            actual_forward = situ_glu_scale_forward(
                x, probs, situ_glu_fusion=True
            )
            actual_backward = situ_glu_scale_backward(
                x, probs, out_grad, situ_glu_fusion=True
            )

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
        self.assertFalse(config.situ_glu_fusion)

        fused_config = TransformerConfig.from_config(
            SimpleNamespace(
                hidden_size=8,
                num_attention_heads=2,
                hidden_act="situ",
                situ_glu_fusion=True,
            )
        )
        self.assertTrue(fused_config.situ_glu_fusion)

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
        # SiTU is supported on the fp8 path (see
        # test_situ_fp8_forward_backward_matches_bf16); GeGLU still is not.
        for activation_type in ("geglu",):
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

        # fp8 on the DeepGEMM expert path is supported now, as long as the
        # expert weight gradients stay in bf16.
        fp8_config = TransformerConfig(
            **config_kwargs,
            moe_use_fusion_node=True,
            moe_expert_fusion=True,
            moe_deep_gemm=True,
            fp8="e4m3",
            fp8_wgrad=False,
        )
        fp8_layer_spec = get_gpt_layer_local_spec(
            fp8_config,
            num_experts=fp8_config.n_routed_experts,
        )
        fp8_layer = MoELayer(
            fp8_config,
            fp8_layer_spec.sublayers_spec.mlp.extra_kwargs["sublayers"],
            SimpleNamespace(ep=None, expt_dp=None),
        )
        self.assertEqual(fp8_layer._activation_type, "situ")
        self.assertTrue(fp8_layer.fp8)

        # fp8 wgrad is the default, so it has to be rejected by name rather
        # than crash inside bwd_gate_up_weight one step into training.
        fp8_wgrad_config = TransformerConfig(
            **config_kwargs,
            moe_use_fusion_node=True,
            moe_expert_fusion=True,
            moe_deep_gemm=True,
            fp8="e4m3",
        )
        self.assertTrue(fp8_wgrad_config.fp8_wgrad)
        with self.assertRaisesRegex(ValueError, "fp8_wgrad=False"):
            MoELayer(fp8_wgrad_config)

        # SonicMoE has no SiTU fp8 kernel, so it must still be rejected.
        sonic_config = TransformerConfig(
            **config_kwargs,
            moe_use_fusion_node=True,
            moe_expert_fusion=True,
            fp8="e4m3",
            fp8_wgrad=False,
            using_sonic_moe=True,
        )
        with self.assertRaisesRegex(ValueError, "not on SonicMoE"):
            MoELayer(sonic_config)

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

    def test_situ_fp8_forward_backward_matches_bf16(self):
        """The fp8 expert path must be no less accurate for SiTU than SwiGLU.

        SiTU has no fused activation+scale+quant kernel, so ``fwd_down_fp8``
        evaluates SiTU-GLU in bf16 and hands the result to the generic
        blockwise quantizer, and ``bwd_down_input_fp8`` returns early into
        ``situ_glu_scale_backward``. How much error fp8 introduces depends on
        the GPU and the scaling recipe, so instead of pinning fixed bounds,
        measure SwiGLU under the identical configuration and require SiTU to
        stay in the same ballpark.

        ``2 * intermediate_size`` has to stay a multiple of 1024 because the
        fused SwiGLU reference kernel requires it when packing ue8m0 scales.

        The weights are scaled so that ``o1`` reaches into SiTU's saturated
        region. With the usual 0.01 scale ``beta * tanh(gate / beta)`` stays
        within a couple percent of ``gate`` itself, SiTU-GLU degenerates into
        SwiGLU numerically, and swapping the two backwards goes unnoticed.
        Saturated, the measured error ratios sit at 0.94-1.62 (max) and
        0.97-1.37 (mean), while feeding SiTU's ``o1`` through the SwiGLU
        backward pushes them past 9.

        Covered: the default separate-op SiTU-GLU and the opt-in Triton
        kernel, which the fp8 path reaches through the same
        ``situ_glu_fusion`` dispatch as bf16, plus ue8m0 packed scales on
        Blackwell only -- deep_gemm rejects the int32 scales elsewhere.

        fp8 wgrad is off because that is the only configuration SiTU + fp8
        accepts -- ``MoELayer`` rejects ``fp8_wgrad=True`` outright, since
        ``bwd_gate_up_weight``'s fp8 GEMM does not run everywhere, for SwiGLU
        just the same.
        """
        hidden_size = 1024
        intermediate_size = 512
        tokens_per_expert = [128, 128]
        num_tokens = sum(tokens_per_expert)
        names = (
            "output",
            "input_grad",
            "probs_grad",
            "weight1_grad",
            "weight2_grad",
        )

        def run(
            activation_type, use_fp8, use_ue8m0=False, situ_glu_fusion=False
        ):
            model_parallel_cuda_manual_seed(2026)
            paddle.seed(2026)
            # Draw every tensor before the expert is built so that the inputs
            # are identical no matter how much RNG the expert's own init eats.
            hidden_states = paddle.randn(
                [num_tokens, hidden_size], dtype="bfloat16"
            )
            probs = paddle.rand([num_tokens], dtype="float32")
            out_grad = paddle.randn([num_tokens, hidden_size], dtype="bfloat16")
            weight1 = (
                paddle.randn(
                    [2, hidden_size, 2 * intermediate_size], dtype="bfloat16"
                )
                * 0.1
            )
            weight2 = (
                paddle.randn(
                    [2, intermediate_size, hidden_size], dtype="bfloat16"
                )
                * 0.1
            )
            config_kwargs = {
                "hidden_size": hidden_size,
                "num_hidden_layers": 1,
                "num_attention_heads": 8,
                "intermediate_size": intermediate_size,
                "moe_intermediate_size": intermediate_size,
                "gated_linear_unit": True,
                "params_dtype": "bfloat16",
            }
            if activation_type == "situ":
                config_kwargs.update(
                    hidden_act=situ,
                    activation_situ_beta=4.0,
                    activation_situ_linear_beta=25.0,
                    situ_glu_fusion=situ_glu_fusion,
                )
            config = TransformerConfig(**config_kwargs)
            expert = GroupedMLPExpert(
                num_local_experts=2, config=config, moe_deep_gemm=True
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
                use_fp8_mlp=use_fp8,
                moe_expert_fusion=True,
                moe_deep_gemm=True,
                use_ue8m0=use_ue8m0,
                use_bf16_gemm_weight_grad=True,
                activation_type=activation_type,
            )
            output = node.forward(hidden_states, probs, tokens_per_expert)
            input_grad, probs_grad = node.backward(out_grad, probs)
            return dict(
                zip(
                    names,
                    (
                        output,
                        input_grad,
                        probs_grad,
                        expert.weight1.main_grad,
                        expert.weight2.main_grad,
                    ),
                )
            )

        def relative_error(reference, actual):
            """(max, mean) absolute error, both normalized by the bf16 amax."""
            ref = reference.astype("float32").reshape([-1])
            act = actual.astype("float32").reshape([-1])
            self.assertEqual(ref.shape, act.shape)
            amax = float(ref.abs().max())
            err = (ref - act).abs()
            return float(err.max()) / amax, float(err.mean()) / amax

        cache = {}

        def cached(activation_type, use_fp8, use_ue8m0, situ_glu_fusion):
            # The SwiGLU reference does not read situ_glu_fusion and bf16 does
            # not read use_ue8m0, so pin those to keep one entry per
            # configuration that actually differs.
            if activation_type != "situ":
                situ_glu_fusion = False
            if not use_fp8:
                use_ue8m0 = False
            key = (activation_type, use_fp8, use_ue8m0, situ_glu_fusion)
            if key not in cache:
                cache[key] = run(
                    activation_type,
                    use_fp8=use_fp8,
                    use_ue8m0=use_ue8m0,
                    situ_glu_fusion=situ_glu_fusion,
                )
            return cache[key]

        # situ_glu_fusion=False is the default; True opts into the Triton
        # kernel, which the fp8 path drives through the same dispatch.
        configs = [(False, False), (False, True)]
        if paddle.device.cuda.get_device_capability()[0] == 10:
            # ue8m0 packs the scales as int32, which deep_gemm only accepts on
            # Blackwell -- elsewhere m_grouped_fp8_gemm_nt_contiguous asserts
            # sfa_dtype == float. The same gate as MoELayer's use_ue8m0 assert.
            configs.insert(1, (True, False))
        for use_ue8m0, fusion in configs:
            swiglu_bf16 = cached("swiglu", False, use_ue8m0, fusion)
            swiglu_fp8 = cached("swiglu", True, use_ue8m0, fusion)
            situ_bf16 = cached("situ", False, use_ue8m0, fusion)
            situ_fp8 = cached("situ", True, use_ue8m0, fusion)
            for name in names:
                with self.subTest(
                    use_ue8m0=use_ue8m0, situ_glu_fusion=fusion, name=name
                ):
                    self.assertEqual(
                        situ_bf16[name].shape, situ_fp8[name].shape
                    )
                    ref_max, ref_mean = relative_error(
                        swiglu_bf16[name], swiglu_fp8[name]
                    )
                    situ_max, situ_mean = relative_error(
                        situ_bf16[name], situ_fp8[name]
                    )
                    # A silent fallback to bf16 would make the comparison
                    # vacuous, so require quantization to have happened.
                    self.assertGreater(situ_max, 0.0)
                    self.assertLessEqual(
                        situ_max,
                        max(2.5 * ref_max, 0.02),
                        msg=(
                            f"{name} max rel err {situ_max:.5f} vs swiglu "
                            f"{ref_max:.5f}"
                        ),
                    )
                    self.assertLessEqual(
                        situ_mean,
                        max(2.0 * ref_mean, 1e-3),
                        msg=(
                            f"{name} mean rel err {situ_mean:.6f} vs swiglu "
                            f"{ref_mean:.6f}"
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
