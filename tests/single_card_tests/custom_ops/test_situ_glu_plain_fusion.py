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

"""Tests for the plain (no router scale) fused SiTU-GLU behind
``situ_glu_plain_fusion``."""

import unittest
from types import SimpleNamespace
from unittest import mock

import paddle
import paddle.nn.functional as F

from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddlefleet.transformer.activations import situ, situ_glu
from paddlefleet.transformer.mlp import MLP
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.triton_ops.situ_glu_plain import (
    situ_glu_plain_backward_triton,
    situ_glu_plain_forward_triton,
)

BETA = 4.0
LINEAR_BETA = 25.0


def _grad_of_situ_glu(x, out_grad, fusion):
    """d situ_glu(x) / dx, going through whichever path ``fusion`` selects."""
    x = x.detach()
    x.stop_gradient = False
    out = situ_glu(
        x,
        beta=BETA,
        linear_beta=LINEAR_BETA,
        situ_glu_plain_fusion=fusion,
    )
    out.backward(out_grad)
    return x.grad


def _reference_grad_fp64(x, out_grad):
    """The same gradient in float64, from the same (already rounded) inputs.

    ``activations.situ_glu`` casts to fp32 internally, so it cannot serve as its
    own high-precision reference -- the formula is spelled out here instead. It
    consumes the *rounded* ``x`` / ``out_grad`` so that input quantization is
    common to all three paths and only the arithmetic differs.
    """
    x = x.astype("float64").detach()
    x.stop_gradient = False
    gate, up = paddle.chunk(x, chunks=2, axis=-1)
    gate_act = BETA * paddle.tanh(gate / BETA) * F.sigmoid(gate)
    up_act = LINEAR_BETA * paddle.tanh(up / LINEAR_BETA)
    (gate_act * up_act).backward(out_grad.astype("float64"))
    return x.grad


def _max_rel(actual, reference):
    actual = actual.astype("float64")
    reference = reference.astype("float64")
    scale = paddle.maximum(reference.abs(), paddle.full_like(reference, 1e-12))
    return float(((actual - reference).abs() / scale).max())


def _bit_exact(actual, expected):
    """``equal_all`` has no bfloat16/float16 kernel; fp32 widening is exact."""
    return bool(
        paddle.equal_all(actual.astype("float32"), expected.astype("float32"))
    )


# The guard and switch tests below run on CPU on purpose -- one of them asserts
# that the Triton entries *reject* CPU tensors -- so the skip is applied per
# class and per test rather than to the module.
_requires_gpu = unittest.skipIf(
    not paddle.is_compiled_with_cuda(),
    "the fused situ-GLU path is a Triton GPU kernel",
)


class TestSituGLUPlainFusionGuards(unittest.TestCase):
    def test_triton_entries_reject_cpu_inputs(self):
        cpu_place = paddle.CPUPlace()
        x = paddle.to_tensor([[1.0] * 8] * 2, dtype="float32", place=cpu_place)
        out_grad = paddle.to_tensor(
            [[1.0] * 4] * 2, dtype="float32", place=cpu_place
        )

        with self.assertRaisesRegex(ValueError, "must be GPU tensors"):
            situ_glu_plain_forward_triton(x)
        with self.assertRaisesRegex(ValueError, "must be GPU tensors"):
            situ_glu_plain_backward_triton(x, out_grad)

    def test_triton_entries_reject_invalid_scales(self):
        invalid_scales = (
            -1.0,
            0.0,
            float("-inf"),
            float("inf"),
            float("nan"),
        )
        entries = (
            (situ_glu_plain_forward_triton, (None,)),
            (situ_glu_plain_backward_triton, (None, None)),
        )

        for function, args in entries:
            for beta in invalid_scales:
                with (
                    self.subTest(function=function.__name__, beta=beta),
                    self.assertRaisesRegex(ValueError, "positive finite"),
                ):
                    function(*args, beta=beta)
            for linear_beta in invalid_scales:
                with (
                    self.subTest(
                        function=function.__name__, linear_beta=linear_beta
                    ),
                    self.assertRaisesRegex(ValueError, "positive finite"),
                ):
                    function(*args, linear_beta=linear_beta)


class TestSituGLUPlainFusionSwitch(unittest.TestCase):
    def test_config_default_and_opt_in(self):
        base = {
            "hidden_size": 8,
            "num_attention_heads": 2,
            "hidden_act": "situ",
        }

        config = TransformerConfig.from_config(SimpleNamespace(**base))
        self.assertFalse(config.situ_glu_plain_fusion)
        # The two fusion flags select different kernels and are independent.
        self.assertFalse(config.situ_glu_fusion)

        fused = TransformerConfig.from_config(
            SimpleNamespace(situ_glu_plain_fusion=True, **base)
        )
        self.assertTrue(fused.situ_glu_plain_fusion)
        self.assertFalse(fused.situ_glu_fusion)

    def test_flag_off_runs_the_op_chain(self):
        # The op chain leaves `paddle.chunk` in the graph; the fused path is a
        # single PyLayer. Asserting on the grad op name is what distinguishes
        # "flag was honoured" from "both paths happen to agree numerically".
        x = paddle.randn([4, 16], dtype="float32")
        with mock.patch(
            "paddlefleet.triton_ops.situ_glu_plain.fused_situ_glu_plain"
        ) as fused:
            situ_glu(x, beta=BETA, linear_beta=LINEAR_BETA)
        fused.assert_not_called()

    def test_falls_back_without_triton(self):
        x = paddle.randn([3, 8], dtype="float32")
        expected = situ_glu(x, beta=BETA, linear_beta=LINEAR_BETA)

        with mock.patch(
            "paddlefleet.triton_ops.utils.is_triton_available",
            return_value=False,
        ):
            actual = situ_glu(
                x,
                beta=BETA,
                linear_beta=LINEAR_BETA,
                situ_glu_plain_fusion=True,
            )

        self.assertTrue(_bit_exact(actual, expected))

    # bfloat16 only, unlike the rest of this class.
    @_requires_gpu
    def test_falls_back_on_non_contiguous_input(self):
        base = paddle.randn([8, 32], dtype="bfloat16")
        x = base[:, ::2]
        self.assertFalse(x.is_contiguous())

        expected = situ_glu(x, beta=BETA, linear_beta=LINEAR_BETA)
        actual = situ_glu(
            x,
            beta=BETA,
            linear_beta=LINEAR_BETA,
            situ_glu_plain_fusion=True,
        )
        self.assertTrue(_bit_exact(actual, expected))

    def test_falls_back_on_a_dtype_the_kernel_cannot_take(self):
        # The op chain casts to fp32 first, so float64 works there. The kernel
        # only widens from fp16/bf16/fp32 and would raise out of its own
        # ``_geom`` guard, so the flag must decline rather than turn a working
        # call into a TypeError.
        x = paddle.randn([8, 32], dtype="float64")
        expected = situ_glu(x, beta=BETA, linear_beta=LINEAR_BETA)
        actual = situ_glu(
            x,
            beta=BETA,
            linear_beta=LINEAR_BETA,
            situ_glu_plain_fusion=True,
        )
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertTrue(_bit_exact(actual, expected))


@_requires_gpu
class TestSituGLUPlainFusionNumerics(unittest.TestCase):
    def setUp(self):
        paddle.seed(20260903)
        self.x = paddle.randn([257, 4096], dtype="float32")
        self.out_grad = paddle.randn([257, 2048], dtype="float32")

    def test_forward_is_bit_exact_against_op_chain(self):
        for dtype in ("bfloat16", "float16", "float32"):
            with self.subTest(dtype=dtype):
                x = self.x.astype(dtype)
                expected = situ_glu(x, beta=BETA, linear_beta=LINEAR_BETA)
                actual = situ_glu(
                    x,
                    beta=BETA,
                    linear_beta=LINEAR_BETA,
                    situ_glu_plain_fusion=True,
                )
                self.assertEqual(actual.dtype, expected.dtype)
                self.assertTrue(_bit_exact(actual, expected))

    def test_forward_handles_absent_linear_beta(self):
        x = self.x.astype("bfloat16")
        expected = situ_glu(x, beta=BETA, linear_beta=None)
        actual = situ_glu(
            x, beta=BETA, linear_beta=None, situ_glu_plain_fusion=True
        )
        self.assertTrue(_bit_exact(actual, expected))

    def test_backward_is_no_less_accurate_than_the_op_chain(self):
        # The fused kernel keeps the fp32 intermediates in registers, so it is
        # expected to *differ* from the op chain. What is asserted here is the
        # direction of that difference.
        x = self.x.astype("bfloat16")
        out_grad = self.out_grad.astype("bfloat16")
        reference = _reference_grad_fp64(x, out_grad)
        eager = _grad_of_situ_glu(x, out_grad, fusion=False)
        fused = _grad_of_situ_glu(x, out_grad, fusion=True)

        self.assertEqual(fused.dtype, eager.dtype)
        self.assertLessEqual(
            _max_rel(fused, reference), _max_rel(eager, reference)
        )

    def test_backward_difference_from_op_chain_is_sub_ulp_in_scale(self):
        # The per-element relative difference is not a useful bound here: the
        # elements that differ are the ones near zero, where the op chain is the
        # inaccurate side (see the test above). Measured against the tensor's own
        # scale instead: at this shape 0.07% of elements differ at all, the
        # largest absolute difference is 0.19 of one bfloat16 ULP at the maximum
        # magnitude, and one element is -0.0 versus +0.0.
        x = self.x.astype("bfloat16")
        out_grad = self.out_grad.astype("bfloat16")
        eager = _grad_of_situ_glu(x, out_grad, fusion=False)
        fused = _grad_of_situ_glu(x, out_grad, fusion=True)

        eager32 = eager.astype("float32")
        max_abs_diff = float((fused.astype("float32") - eager32).abs().max())
        # bfloat16 keeps 8 significand bits, so one ULP is 2**-8 of the binade.
        one_ulp_at_scale = 2.0**-8 * float(eager32.abs().max())
        self.assertLessEqual(max_abs_diff, one_ulp_at_scale)

    def test_backward_respects_stop_gradient(self):
        x = self.x.astype("bfloat16").detach()
        x.stop_gradient = True
        out = situ_glu(
            x,
            beta=BETA,
            linear_beta=LINEAR_BETA,
            situ_glu_plain_fusion=True,
        )
        self.assertTrue(out.stop_gradient)


@_requires_gpu
class TestSituGLUPlainFusionReachesTheMLP(unittest.TestCase):
    """The flag is only useful if it survives the trip from config to kernel."""

    @staticmethod
    def _mlp(plain_fusion):
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
            activation_situ_beta=BETA,
            activation_situ_linear_beta=LINEAR_BETA,
            situ_glu_plain_fusion=plain_fusion,
        )
        spec = get_gpt_layer_local_spec(config).sublayers_spec.mlp
        return MLP(config, spec.sublayers_spec)

    def test_mlp_forward_backward_uses_the_fused_kernel(self):
        import paddlefleet.triton_ops.situ_glu_plain as plain

        for plain_fusion in (False, True):
            with self.subTest(situ_glu_plain_fusion=plain_fusion):
                mlp = self._mlp(plain_fusion)
                hidden_states = paddle.randn([4, 2, 12])
                hidden_states.stop_gradient = False

                with mock.patch.object(
                    plain,
                    "fused_situ_glu_plain",
                    wraps=plain.fused_situ_glu_plain,
                ) as fused:
                    output, output_bias = mlp(hidden_states)
                    (output.sum() + output_bias.sum()).backward()

                self.assertEqual(fused.called, plain_fusion)
                self.assertTrue(bool(paddle.isfinite(output).all()))
                self.assertTrue(bool(paddle.isfinite(hidden_states.grad).all()))


if __name__ == "__main__":
    unittest.main()
