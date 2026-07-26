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

import os
import unittest
from unittest import mock

import numpy as np
import paddle


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

    from paddlefleet.transformer.moe.fp8_utils import (
        _w4a8_quant,
        fuse_stack_fp8_quant_python,
        fuse_stack_transpose_fp8_quant_python,
        fuse_weighted_swiglu_fp8_quant_clamp_python,
        fuse_weighted_swiglu_fp8_quant_python,
        fused_act_dequant_python,
        quant_blockwize,
    )


def _fp8_bits(value):
    return value.view("int8")


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
        value = paddle.randn([37, 256], dtype="bfloat16") * 3.0
        expected_q, expected_scale = quant_blockwize(value, quant_dtype="fp8")
        actual_q, actual_scale = w4a8_quantize_1x32(value, 0)
        self.assert_fp8_equal(actual_q, expected_q)
        self.assert_tensor_equal(
            actual_scale, expected_scale, "UE8M0 scale differs"
        )

    def test_quantize_runtime_dispatch(self):
        value = paddle.randn([37, 256], dtype="bfloat16") * 3.0
        expected_q, expected_scale = quant_blockwize(value, quant_dtype="fp8")
        for enabled in ("0", "1"):
            with (
                self.subTest(enabled=enabled),
                mock.patch.dict(
                    os.environ, {"PADDLEFLEET_W4A8_FUSED_QUANT": enabled}
                ),
            ):
                actual_q, actual_scale = _w4a8_quant(value, quant_dtype="fp8")
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

    def test_weighted_swiglu_fp8(self):
        value = paddle.randn([31, 512], dtype="bfloat16") * 2.0
        probs = paddle.rand([31], dtype="float32")
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

    def test_dequantize_fp8(self):
        value = paddle.randn([37, 256], dtype="bfloat16")
        quantized, scale = quant_blockwize(value, quant_dtype="fp8")
        expected = fused_act_dequant_python(quantized, scale)
        actual = w4a8_dequantize_1x32(quantized, scale)
        self.assert_tensor_equal(actual, expected, "BF16 dequant differs")


if __name__ == "__main__":
    unittest.main()
