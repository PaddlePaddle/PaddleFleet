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

"""Bit-exact parity tests for the fused W4A8 1x32 CUDA custom ops."""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import paddle

from paddlefleet.transformer.activations import (
    situ_glu_scale_backward,
    situ_glu_scale_forward,
)
from paddlefleet.transformer.moe import fp8_utils
from paddlefleet.transformer.moe.fp8_utils import (
    ExpertsGroupGemmContiguousNode,
    _w4a8_dequant,
    _w4a8_quant,
    _w4a8_stack_quant,
    _w4a8_weighted_swiglu_quant,
    fuse_stack_fp8_quant_python,
    fuse_stack_transpose_fp8_quant_python,
    fuse_weighted_swiglu_fp8_quant_clamp_python,
    fuse_weighted_swiglu_fp8_quant_python,
    fused_act_dequant_python,
    quant_blockwize,
)
from paddlefleet.transformer.moe.fusion_layer_utils import MlpNode
from paddlefleet.transformer.transformer_config import TransformerConfig


def _prepare_blackwell():
    if (
        not paddle.is_compiled_with_cuda()
        or paddle.device.cuda.device_count() == 0
    ):
        return False
    try:
        paddle.set_device("gpu:0")
        return paddle.device.cuda.get_device_capability()[0] >= 10
    except (RuntimeError, ValueError):
        return False


IS_BLACKWELL = _prepare_blackwell()

if IS_BLACKWELL:
    from paddlefleet_ops import (
        w4a8_dequantize_1x32,
        w4a8_quantize_1x32,
        w4a8_stack_quantize_1x32,
        w4a8_weighted_swiglu_quantize_1x32,
    )


def _fp8_bits(value):
    return value.view("int8")


class TestW4A8RuntimeDispatch(unittest.TestCase):
    def test_fused_quant_model_config(self):
        self.assertFalse(TransformerConfig().use_w4a8_fused_quant)
        self.assertTrue(
            TransformerConfig(use_w4a8_fused_quant=True).use_w4a8_fused_quant
        )

    def test_mlp_node_forwards_fused_quant_model_config(self):
        custom_map = SimpleNamespace(
            token_dispatcher=SimpleNamespace(
                _comm_manager=SimpleNamespace(tokens_per_expert=[1])
            ),
            num_experts_per_device=1,
        )
        with mock.patch(
            "paddlefleet.transformer.moe.fusion_layer_utils."
            "ExpertsGroupGemmContiguousNode"
        ) as gemm_node:
            MlpNode(
                custom_map,
                num_experts_per_tok=1,
                moe_expert_fusion=True,
                moe_deep_gemm=True,
                use_ue8m0=True,
                use_w4a8=True,
                use_w4a8_fused_quant=True,
            )

        self.assertTrue(gemm_node.call_args.kwargs["use_w4a8"])
        self.assertTrue(gemm_node.call_args.kwargs["use_ue8m0"])
        self.assertTrue(gemm_node.call_args.kwargs["use_w4a8_fused_quant"])

    def test_fused_quant_flag_requires_custom_ops(self):
        with mock.patch.object(fp8_utils, "HAS_W4A8_FUSED_QUANT", False):
            self.assertFalse(fp8_utils._use_w4a8_fused_quant(False))

        with mock.patch.object(fp8_utils, "HAS_W4A8_FUSED_QUANT", True):
            self.assertTrue(fp8_utils._use_w4a8_fused_quant(True))

        with (
            mock.patch.object(fp8_utils, "HAS_W4A8_FUSED_QUANT", False),
            self.assertRaisesRegex(RuntimeError, "requires paddlefleet_ops"),
        ):
            fp8_utils._use_w4a8_fused_quant(True)

    def test_quant_dispatches_to_fused_and_python_ops(self):
        value = mock.sentinel.value
        fused_result = (mock.sentinel.fused_q, mock.sentinel.fused_scale)
        python_result = (mock.sentinel.python_q, mock.sentinel.python_scale)

        with (
            mock.patch.object(
                fp8_utils, "_use_w4a8_fused_quant", return_value=True
            ),
            mock.patch.object(
                fp8_utils,
                "w4a8_quantize_1x32",
                create=True,
                return_value=fused_result,
            ) as fused_quant,
        ):
            self.assertEqual(_w4a8_quant(value, "fp8", True), fused_result)
            self.assertEqual(_w4a8_quant(value, "fp4", True), fused_result)
            fused_quant.assert_has_calls(
                [mock.call(value, 0), mock.call(value, 1)]
            )

        with (
            mock.patch.object(
                fp8_utils, "_use_w4a8_fused_quant", return_value=False
            ),
            mock.patch.object(
                fp8_utils, "quant_blockwize", return_value=python_result
            ) as python_quant,
        ):
            self.assertEqual(_w4a8_quant(value, "fp8"), python_result)
            python_quant.assert_called_once_with(value, quant_dtype="fp8")

    def test_stack_quant_dispatches_to_fused_and_python_ops(self):
        weights = mock.sentinel.weights
        stacked = mock.sentinel.stacked
        fused_result = (mock.sentinel.fused_q, mock.sentinel.fused_scale)

        with (
            mock.patch.object(
                fp8_utils, "_use_w4a8_fused_quant", return_value=True
            ),
            mock.patch.object(
                fp8_utils, "_stack_expert_weights", return_value=stacked
            ) as stack_weights,
            mock.patch.object(
                fp8_utils,
                "w4a8_stack_quantize_1x32",
                create=True,
                return_value=fused_result,
            ) as fused_quant,
        ):
            self.assertEqual(
                _w4a8_stack_quant(weights, True, use_w4a8_fused_quant=True),
                fused_result,
            )
            stack_weights.assert_called_once_with(weights)
            fused_quant.assert_called_once_with(stacked, True)

        for transpose, function_name in (
            (False, "fuse_stack_fp8_quant_python"),
            (True, "fuse_stack_transpose_fp8_quant_python"),
        ):
            python_result = (mock.sentinel.python_q, mock.sentinel.python_scale)
            with (
                self.subTest(transpose=transpose),
                mock.patch.object(
                    fp8_utils, "_use_w4a8_fused_quant", return_value=False
                ),
                mock.patch.object(
                    fp8_utils, function_name, return_value=python_result
                ) as python_quant,
            ):
                self.assertEqual(
                    _w4a8_stack_quant(weights, transpose), python_result
                )
                python_quant.assert_called_once_with(
                    weights,
                    quant_dtype="fp4",
                    using_ue8m0_scale=True,
                )

    def test_weighted_swiglu_dispatches_all_paths(self):
        value = mock.sentinel.value
        probs = mock.sentinel.probs
        result = (mock.sentinel.quantized, mock.sentinel.scale)

        with (
            mock.patch.object(
                fp8_utils, "_use_w4a8_fused_quant", return_value=True
            ),
            mock.patch.object(
                fp8_utils,
                "w4a8_weighted_swiglu_quantize_1x32",
                create=True,
                return_value=result,
            ) as fused_quant,
        ):
            self.assertEqual(
                _w4a8_weighted_swiglu_quant(
                    value, probs, use_w4a8_fused_quant=True
                ),
                result,
            )
            self.assertEqual(
                _w4a8_weighted_swiglu_quant(
                    value, probs, 10, use_w4a8_fused_quant=True
                ),
                result,
            )
            fused_quant.assert_has_calls(
                [mock.call(value, probs, 0.0), mock.call(value, probs, 10.0)]
            )

        for clamp_value, function_name in (
            (None, "fuse_weighted_swiglu_fp8_quant_python"),
            (0.0, "fuse_weighted_swiglu_fp8_quant_python"),
            (10.0, "fuse_weighted_swiglu_fp8_quant_clamp_python"),
        ):
            with (
                self.subTest(clamp_value=clamp_value),
                mock.patch.object(
                    fp8_utils, "_use_w4a8_fused_quant", return_value=False
                ),
                mock.patch.object(
                    fp8_utils, function_name, return_value=result
                ) as python_quant,
            ):
                self.assertEqual(
                    _w4a8_weighted_swiglu_quant(
                        value, probs, clamp_value=clamp_value
                    ),
                    result,
                )
                expected_args = (
                    (value, probs, clamp_value)
                    if clamp_value
                    else (value, probs)
                )
                python_quant.assert_called_once_with(*expected_args)

    def test_dequant_dispatches_to_fused_and_python_ops(self):
        value = mock.sentinel.value
        scale = mock.sentinel.scale

        for enabled, function_name in (
            (True, "w4a8_dequantize_1x32"),
            (False, "fused_act_dequant_python"),
        ):
            result = mock.sentinel.dequantized
            with (
                self.subTest(enabled=enabled),
                mock.patch.object(
                    fp8_utils,
                    "_use_w4a8_fused_quant",
                    return_value=enabled,
                ),
                mock.patch.object(
                    fp8_utils,
                    function_name,
                    create=True,
                    return_value=result,
                ) as dequant,
            ):
                self.assertIs(
                    _w4a8_dequant(
                        value,
                        scale,
                        use_w4a8_fused_quant=enabled,
                    ),
                    result,
                )
                dequant.assert_called_once_with(value, scale)

    @staticmethod
    def _make_node():
        node = object.__new__(ExpertsGroupGemmContiguousNode)
        node.activation_type = "swiglu"
        node.activation_situ_beta = 4.0
        node.activation_situ_linear_beta = 25.0
        node.situ_glu_fusion = False
        node.use_w4a8_fused_quant = True
        node.use_ue8m0 = True
        node._w4a8_grouped_gemm = mock.Mock(
            side_effect=lambda _x, _xs, _w, _ws, output: output
        )
        return node

    def test_w4a8_forward_methods_use_dispatch_helpers(self):
        node = self._make_node()
        node.dequant_input = False
        node.clamp_value = 10.0
        x = mock.sentinel.x
        weights = mock.sentinel.weights
        probs = mock.sentinel.probs
        x_q = mock.Mock(shape=[3, 32])
        x_scale = mock.sentinel.x_scale
        weight_q = mock.Mock(shape=[2, 64, 16])
        weight_scale = mock.sentinel.weight_scale
        gate_output = mock.Mock(shape=[3, 64])

        with (
            mock.patch.object(
                fp8_utils,
                "_w4a8_stack_quant",
                return_value=(weight_q, weight_scale),
            ) as stack_quant,
            mock.patch.object(
                fp8_utils, "_w4a8_quant", return_value=(x_q, x_scale)
            ) as quant,
            mock.patch.object(
                fp8_utils.paddle, "empty", return_value=gate_output
            ),
        ):
            self.assertIs(node._fwd_gate_up_w4a8(x, weights), gate_output)
            stack_quant.assert_called_once_with(
                weights,
                transpose=True,
                use_w4a8_fused_quant=True,
                use_ue8m0=True,
            )
            quant.assert_called_once_with(
                x,
                quant_dtype="fp8",
                use_w4a8_fused_quant=True,
            )

        swiglu_q = mock.Mock(shape=[3, 64])
        swiglu_scale = mock.sentinel.swiglu_scale
        down_output = mock.Mock(shape=[3, 64])
        with (
            mock.patch.object(
                fp8_utils,
                "_w4a8_stack_quant",
                return_value=(weight_q, weight_scale),
            ) as stack_quant,
            mock.patch.object(
                fp8_utils,
                "_w4a8_weighted_swiglu_quant",
                return_value=(swiglu_q, swiglu_scale),
            ) as swiglu_quant,
            mock.patch.object(
                fp8_utils.paddle, "empty", return_value=down_output
            ),
        ):
            self.assertIs(
                node._fwd_down_w4a8(
                    mock.sentinel.o1, probs, weights, clear_o1=False
                ),
                down_output,
            )
            stack_quant.assert_called_once_with(
                weights,
                transpose=True,
                use_w4a8_fused_quant=True,
                use_ue8m0=True,
            )
            swiglu_quant.assert_called_once_with(
                mock.sentinel.o1,
                probs,
                node.clamp_value,
                use_w4a8_fused_quant=True,
            )

    def test_w4a8_backward_methods_use_dispatch_helpers(self):
        node = self._make_node()
        node.clamp_value = None
        weights = mock.sentinel.weights
        grad = mock.Mock(dtype=paddle.bfloat16)
        weight_q = mock.Mock(shape=[2, 64, 16])
        weight_scale = mock.sentinel.weight_scale
        grad_q = mock.Mock(shape=[3, 32])
        grad_scale = mock.sentinel.grad_scale
        gemm_output = mock.Mock(shape=[3, 64])
        backward_result = (
            mock.sentinel.do1,
            mock.sentinel.probs_grad,
            mock.sentinel.o2,
        )

        with (
            mock.patch.object(
                fp8_utils,
                "_w4a8_stack_quant",
                return_value=(weight_q, weight_scale),
            ) as stack_quant,
            mock.patch.object(
                fp8_utils, "_w4a8_quant", return_value=(grad_q, grad_scale)
            ) as quant,
            mock.patch.object(
                fp8_utils.paddle, "empty", return_value=gemm_output
            ),
            mock.patch.object(fp8_utils.paddle.amp, "auto_cast") as auto_cast,
            mock.patch.object(fp8_utils, "USE_INPLACE_SWIGLU_BWD", True),
            mock.patch.object(
                fp8_utils,
                "_fused_swiglu_probs_bwd",
                return_value=backward_result,
            ),
        ):
            auto_cast.return_value.__enter__.return_value = None
            result = node._bwd_down_input_w4a8(
                weights, grad, mock.sentinel.o1, mock.sentinel.probs
            )
            self.assertEqual(
                result,
                (
                    mock.sentinel.do1,
                    mock.sentinel.o2,
                    mock.sentinel.probs_grad,
                ),
            )
            stack_quant.assert_called_once_with(
                weights,
                transpose=False,
                use_w4a8_fused_quant=True,
                use_ue8m0=True,
            )
            quant.assert_called_once_with(
                grad,
                quant_dtype="fp8",
                use_w4a8_fused_quant=True,
            )

        dx = mock.Mock(shape=[3, 64])
        with (
            mock.patch.object(
                fp8_utils,
                "_w4a8_stack_quant",
                return_value=(weight_q, weight_scale),
            ) as stack_quant,
            mock.patch.object(
                fp8_utils, "_w4a8_quant", return_value=(grad_q, grad_scale)
            ) as quant,
            mock.patch.object(fp8_utils.paddle, "empty", return_value=dx),
        ):
            self.assertIs(node._bwd_gate_up_input_w4a8(grad, weights), dx)
            stack_quant.assert_called_once_with(
                weights,
                transpose=False,
                use_w4a8_fused_quant=True,
                use_ue8m0=True,
            )
            quant.assert_called_once_with(
                grad,
                quant_dtype="fp8",
                use_w4a8_fused_quant=True,
            )

    def test_w4a8_weight_grad_uses_dispatch_dequant(self):
        class DequantCalled(Exception):
            pass

        node = self._make_node()
        node.dequant_input = True
        node.use_w4a8 = True
        node.input_fp8 = mock.sentinel.input_fp8
        node.input_scale = mock.sentinel.input_scale
        with mock.patch.object(
            fp8_utils, "_w4a8_dequant", side_effect=DequantCalled
        ) as dequant:
            with self.assertRaises(DequantCalled):
                node.bf16_weight_grad(
                    mock.sentinel.dy, None, mock.sentinel.weights
                )
            dequant.assert_called_once_with(
                node.input_fp8,
                node.input_scale,
                use_w4a8_fused_quant=True,
            )

    def test_situ_w4a8_forward_backward_matches_reference(self):
        """SiTU W4A8 uses the same forward/backward formula as the BF16 path.

        The GEMM and quantizers are stubbed so this regression is runnable
        without a compiled W4A8 extension.  It still exercises both W4A8
        activation dispatch sites and compares their complete tensor contract
        with the validated SiTU reference helpers.
        """
        node = object.__new__(ExpertsGroupGemmContiguousNode)
        node.activation_type = "situ"
        node.activation_situ_beta = 4.0
        node.activation_situ_linear_beta = 25.0
        node.situ_glu_fusion = False
        node.use_w4a8_fused_quant = False
        node.use_ue8m0 = True
        node.clamp_value = None

        x = paddle.randn([4, 8], dtype="bfloat16")
        probs = paddle.rand([4], dtype="float32")
        expected_forward = situ_glu_scale_forward(x, probs, 4.0, 25.0)
        expected_quant = (
            expected_forward,
            paddle.ones([4, 1], dtype="float32"),
        )

        with mock.patch.object(
            fp8_utils,
            "_w4a8_quant",
            return_value=expected_quant,
        ) as quant:
            actual_quant = node._w4a8_act_quant(x, probs)

        self.assertTrue(
            paddle.equal_all(
                actual_quant[0].astype("float32"),
                expected_forward.astype("float32"),
            )
        )
        self.assertTrue(
            paddle.equal_all(
                actual_quant[1].astype("float32"),
                expected_quant[1].astype("float32"),
            )
        )
        self.assertEqual(quant.call_count, 1)
        quant_args, quant_kwargs = quant.call_args
        self.assertTrue(
            paddle.equal_all(
                quant_args[0].astype("float32"),
                expected_forward.astype("float32"),
            )
        )
        self.assertEqual(
            quant_kwargs,
            {"quant_dtype": "fp8", "use_w4a8_fused_quant": False},
        )

        weights = mock.sentinel.weights
        out_grad = paddle.randn([4, 4], dtype="bfloat16")
        do2_s = paddle.full([4, 4], 0.25, dtype="bfloat16")

        def fill_gemm_output(_x, _x_scale, _w, _w_scale, output):
            output.set_value(do2_s)
            return output

        node._w4a8_grouped_gemm = mock.Mock(side_effect=fill_gemm_output)
        weight_q = paddle.zeros([1, 4, 4], dtype="int8")
        weight_scale = paddle.ones([1, 4, 1], dtype="float32")
        grad_q = mock.Mock(shape=[4, 4])
        grad_scale = paddle.ones([4, 1], dtype="float32")
        expected_backward = situ_glu_scale_backward(
            x,
            probs,
            do2_s,
            4.0,
            25.0,
        )
        with (
            mock.patch.object(
                fp8_utils,
                "_w4a8_stack_quant",
                return_value=(weight_q, weight_scale),
            ) as stack_quant,
            mock.patch.object(
                fp8_utils,
                "_w4a8_quant",
                return_value=(grad_q, grad_scale),
            ) as grad_quant,
        ):
            actual_backward = node._bwd_down_input_w4a8(
                weights,
                out_grad,
                x,
                probs,
            )

        for actual, expected in zip(actual_backward, expected_backward):
            self.assertTrue(
                paddle.equal_all(
                    actual.astype("float32"), expected.astype("float32")
                )
            )
        stack_quant.assert_called_once_with(
            weights,
            transpose=False,
            use_w4a8_fused_quant=False,
            use_ue8m0=True,
        )
        self.assertEqual(grad_quant.call_count, 1)
        grad_args, grad_kwargs = grad_quant.call_args
        self.assertTrue(
            paddle.equal_all(
                grad_args[0].astype("float32"),
                out_grad.astype("float32"),
            )
        )
        self.assertEqual(
            grad_kwargs,
            {"quant_dtype": "fp8", "use_w4a8_fused_quant": False},
        )


@unittest.skipUnless(IS_BLACKWELL, "requires a Blackwell CUDA device")
class TestW4A8CudaQuantParity(unittest.TestCase):
    def setUp(self):
        paddle.seed(23)
        np.random.seed(23)

    def assert_tensor_equal(self, actual, expected, message):
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(list(actual.shape), list(expected.shape))
        self.assertTrue(bool((actual == expected).all()), message)

    def assert_fp8_equal(self, actual, expected):
        self.assert_tensor_equal(
            _fp8_bits(actual), _fp8_bits(expected), "FP8 payload differs"
        )

    def test_quantize_fp8(self):
        for shape in ([1, 32], [255, 32], [256, 32], [257, 32], [37, 256]):
            with self.subTest(shape=shape):
                value = paddle.randn(shape, dtype="bfloat16") * 3.0
                expected_q, expected_scale = quant_blockwize(
                    value, quant_dtype="fp8"
                )
                actual_q, actual_scale = w4a8_quantize_1x32(value, 0)
                self.assert_fp8_equal(actual_q, expected_q)
                self.assert_tensor_equal(
                    actual_scale, expected_scale, "UE8M0 scale differs"
                )

    def test_quantize_misaligned_contiguous_bf16(self):
        storage = paddle.randn([257 * 32 + 1], dtype="bfloat16")
        value = storage[1:].reshape([257, 32])
        self.assertTrue(value.is_contiguous())
        expected_q, expected_scale = quant_blockwize(value, quant_dtype="fp8")
        actual_q, actual_scale = w4a8_quantize_1x32(value, 0)
        self.assert_fp8_equal(actual_q, expected_q)
        self.assert_tensor_equal(
            actual_scale, expected_scale, "misaligned UE8M0 scale differs"
        )

    def test_quantize_runtime_dispatch(self):
        value = paddle.randn([37, 256], dtype="bfloat16") * 3.0
        expected_q, expected_scale = quant_blockwize(value, quant_dtype="fp8")
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                actual_q, actual_scale = _w4a8_quant(
                    value,
                    quant_dtype="fp8",
                    use_w4a8_fused_quant=enabled,
                )
                self.assert_fp8_equal(actual_q, expected_q)
                self.assert_tensor_equal(
                    actual_scale, expected_scale, "UE8M0 scale differs"
                )

    def test_quantize_fp4_random_and_midpoints(self):
        random_value = paddle.randn([17, 256], dtype="bfloat16") * 3.0
        midpoints = np.array(
            [
                0.0,
                0.25,
                -0.25,
                0.75,
                -0.75,
                1.25,
                -1.25,
                1.75,
                -1.75,
                2.5,
                -2.5,
                3.5,
                -3.5,
                5.0,
                -5.0,
                6.0,
            ]
            + [0.0] * 16,
            dtype=np.float32,
        )
        midpoint_value = paddle.to_tensor(
            np.tile(midpoints[:32], (1, 8)), dtype="float32"
        )
        for value in (random_value, midpoint_value):
            with self.subTest(shape=list(value.shape)):
                expected_q, expected_scale = quant_blockwize(
                    value, quant_dtype="fp4"
                )
                actual_q, actual_scale = w4a8_quantize_1x32(value, 1)
                self.assert_tensor_equal(
                    actual_q, expected_q, "packed FP4 differs"
                )
                self.assert_tensor_equal(
                    actual_scale, expected_scale, "UE8M0 scale differs"
                )

    def test_stack_fp4(self):
        weights = paddle.randn([3, 64, 256], dtype="bfloat16")
        for transpose in (False, True):
            with self.subTest(transpose=transpose):
                reference = (
                    fuse_stack_transpose_fp8_quant_python
                    if transpose
                    else fuse_stack_fp8_quant_python
                )
                expected_q, expected_scale = reference(
                    weights, quant_dtype="fp4"
                )
                actual_q, actual_scale = w4a8_stack_quantize_1x32(
                    weights, transpose
                )
                self.assert_tensor_equal(
                    actual_q, expected_q, "packed FP4 differs"
                )
                self.assert_tensor_equal(
                    actual_scale, expected_scale, "UE8M0 scale differs"
                )

    def test_stack_fp4_large_bf16_fast_path(self):
        """Large stacked weights exercise the BF16 transpose fast path."""
        weights = paddle.randn([4, 256, 512], dtype="bfloat16")
        for transpose in (False, True):
            with self.subTest(transpose=transpose):
                reference = (
                    fuse_stack_transpose_fp8_quant_python
                    if transpose
                    else fuse_stack_fp8_quant_python
                )
                expected_q, expected_scale = reference(
                    weights, quant_dtype="fp4"
                )
                actual_q, actual_scale = w4a8_stack_quantize_1x32(
                    weights, transpose
                )
                self.assert_tensor_equal(
                    actual_q, expected_q, "packed FP4 differs"
                )
                self.assert_tensor_equal(
                    actual_scale, expected_scale, "UE8M0 scale differs"
                )

    def test_weighted_swiglu_fp8(self):
        value = paddle.randn([32, 512], dtype="bfloat16") * 2.0
        probs = paddle.rand([32], dtype="float32")
        for clamp_value in (0.0, 10.0):
            with self.subTest(clamp_value=clamp_value):
                if clamp_value:
                    expected_q, expected_scale = (
                        fuse_weighted_swiglu_fp8_quant_clamp_python(
                            value, probs, clamp_value
                        )
                    )
                else:
                    expected_q, expected_scale = (
                        fuse_weighted_swiglu_fp8_quant_python(value, probs)
                    )
                actual_q, actual_scale = w4a8_weighted_swiglu_quantize_1x32(
                    value, probs, clamp_value
                )
                self.assert_fp8_equal(actual_q, expected_q)
                self.assert_tensor_equal(
                    actual_scale, expected_scale, "UE8M0 scale differs"
                )
                dispatched_q, dispatched_scale = _w4a8_weighted_swiglu_quant(
                    value,
                    probs,
                    clamp_value,
                    use_w4a8_fused_quant=True,
                )
                self.assert_fp8_equal(dispatched_q, expected_q)
                self.assert_tensor_equal(
                    dispatched_scale,
                    expected_scale,
                    "runtime-dispatched UE8M0 scale differs",
                )

    def test_dequantize_fp8(self):
        value = paddle.randn([37, 256], dtype="bfloat16")
        quantized, scale = quant_blockwize(value, quant_dtype="fp8")
        expected = fused_act_dequant_python(quantized, scale)
        actual = w4a8_dequantize_1x32(quantized, scale)
        self.assert_tensor_equal(actual, expected, "BF16 dequant differs")


if __name__ == "__main__":
    unittest.main()
