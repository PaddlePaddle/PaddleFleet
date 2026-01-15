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

import numpy as np
import paddle
import paddle.incubate.nn.functional as F

paddle.compat.enable_torch_proxy()

from paddlefleet.ops import deep_gemm, fuse_weighted_swiglu_fp8_quant
from paddlefleet.ops.deep_gemm.testing import (
    get_arch_major,
)


class TestSPAQ(unittest.TestCase):
    def setUp(self):
        self.dtypes = ["float32", "float16", "bfloat16"]
        if not paddle.is_compiled_with_cuda():
            self.skipTest("CUDA is not available")

        # Set random seed for reproducibility
        np.random.seed(42)
        paddle.seed(42)

    def dequantize_fp8_to_bf16(
        self, fp8_tensor: paddle.Tensor, scale: paddle.Tensor
    ) -> paddle.Tensor:
        """Helper function to dequantize fp8 tensor to bf16"""
        expanded_scale = paddle.repeat_interleave(scale, repeats=128, axis=-1)
        # Handle non-aligned cases by truncating
        expanded_scale = expanded_scale[:, : fp8_tensor.shape[-1]]
        return fp8_tensor.astype("float32") * expanded_scale

    def run_fused_op_test(self, M, N, K):
        x = paddle.clip(
            paddle.randn([M, K * 2]).astype("bfloat16"), min=-50, max=50
        )
        prob = paddle.randn([M, 1]).astype("float32")
        weight_x = paddle.clip(
            paddle.randn([N, K]).astype("bfloat16"), min=-50, max=50
        )

        x.stop_gradient = False
        prob.stop_gradient = False
        weight_x.stop_gradient = False

        golden_res = F.swiglu(x) * prob

        fp8_x_out_ref, fp32_x_scale_ref = fuse_weighted_swiglu_fp8_quant(
            x, prob, using_pow2_scaling=False, use_ue8m0=False
        )

        dequantized_res = self.dequantize_fp8_to_bf16(
            fp8_x_out_ref, fp32_x_scale_ref
        )

        golden_np = golden_res.astype("float32").numpy()
        fused_np = dequantized_res.numpy()

        np.testing.assert_allclose(
            golden_np,
            fused_np,
            rtol=0.01,
            atol=1,
        )

    def run_fused_op_ue8m0_test(self, M, N, K):
        x = paddle.clip(
            paddle.randn([M, K * 2]).astype("bfloat16"), min=-50, max=50
        )
        prob = paddle.randn([M, 1]).astype("float32")

        weight_x = paddle.clip(
            paddle.randn([N, K]).astype("bfloat16"), min=-50, max=50
        )

        fp8_weight, fp32_weight_scale = (
            paddle.incubate.nn.functional.fp8.fp8_quant_blockwise(
                weight_x,
                quant_method="128x128",
                input_transpose=False,
                output_scale_transpose=False,
                using_pow2_scale=True,
            )
        )

        x.stop_gradient = False
        prob.stop_gradient = False
        weight_x.stop_gradient = False

        # spaq test with using_pow2_scaling=True, use_ue8m0=False
        fp8_x_out_ref, fp32_x_scale_ref = fuse_weighted_swiglu_fp8_quant(
            x, prob, using_pow2_scaling=True, use_ue8m0=False
        )
        out_ref = paddle.empty([M, N], dtype="bfloat16")
        deep_gemm.fp8_gemm_nt(
            (fp8_x_out_ref, fp32_x_scale_ref),
            (fp8_weight, fp32_weight_scale),
            out_ref,
        )

        # spaq test with using_pow2_scaling=True, use_ue8m0=True
        fp8_x_out, ue8m0_x_scale = fuse_weighted_swiglu_fp8_quant(
            x, prob, using_pow2_scaling=True, use_ue8m0=True
        )
        out = paddle.empty([M, N], dtype="bfloat16")
        deep_gemm.fp8_gemm_nt(
            (fp8_x_out, ue8m0_x_scale), (fp8_weight, fp32_weight_scale), out
        )

        np.testing.assert_allclose(
            out_ref,
            out,
        )

    def test_spaq_0(self):
        self.run_fused_op_test(128 * 3, 128 * 4, 2048)

    def test_spaq_1(self):
        self.run_fused_op_test(128 * 10, 128 * 20, 4096)

    def test_spaq_ue8m0_0(self):
        if get_arch_major() == 10:
            self.run_fused_op_ue8m0_test(128 * 3, 128 * 4, 2048)

    def test_spaq_ue8m0_1(self):
        if get_arch_major() == 10:
            self.run_fused_op_ue8m0_test(128 * 10, 128 * 20, 4096)


if __name__ == "__main__":
    unittest.main()
