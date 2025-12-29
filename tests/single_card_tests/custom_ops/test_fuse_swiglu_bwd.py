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

import unittest

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import base
from paddle.base import core

from paddlefleet.ops import fused_swiglu_bwd


class TestFusedSwiGLUBack(unittest.TestCase):
    def setUp(self):
        self.dtypes = ["float32", "bfloat16"]
        if not paddle.is_compiled_with_cuda():
            self.skipTest("CUDA is not available")

        # Set random seed for reproducibility
        np.random.seed(42)
        paddle.seed(42)

    def get_reference_impl(self, g, y):
        """
        Paddle native implementation of SwiGLU Backward as ground truth.
        """
        # y shape: [B, S, 2*H] -> chunk -> y1, y2
        y_1, y_2 = paddle.chunk(y, chunks=2, axis=-1)

        # Calculate intermediate activations
        sigmoid_y1 = paddle.sigmoid(y_1)
        silu_y1 = F.silu(y_1)

        # Gradient calculation logic (Chain Rule)
        # 1. Gradient for y2: g * silu(y1)
        term2 = g * silu_y1

        # 2. Gradient for y1: g * y2 * (silu(y1))'
        # (silu)' = sigmoid * (1 + x * (1 - sigmoid))
        term1 = g * sigmoid_y1 * (1 + y_1 * (1 - sigmoid_y1)) * y_2

        # Concatenate to get dx
        dx = paddle.concat([term1, term2], axis=-1)
        return dx

    def run_fused_op_test(self, batch_size, seq_len, hidden_size, dtype):
        # 1. Construct input data
        # Input y: [B, S, 2*H]
        # Gradient g: [B, S, H]
        shape_y = (batch_size, seq_len, 2 * hidden_size)
        shape_g = (batch_size, seq_len, hidden_size)

        # Use a reasonable range to avoid numerical instability
        y_np = np.random.normal(0, 1.5, shape_y).astype("float32")
        g_np = np.random.normal(0, 1.0, shape_g).astype("float32")

        # 2. Create Tensors
        if dtype == "bfloat16":
            y = paddle.to_tensor(y_np).astype("bfloat16")
            g = paddle.to_tensor(g_np).astype("bfloat16")
        else:
            y = paddle.to_tensor(y_np, dtype=dtype)
            g = paddle.to_tensor(g_np, dtype=dtype)

        # 3. Run Reference Implementation
        # Naive implementation is run in the same dtype to simulate real precision loss
        # Or run in FP32 and cast down if we want to check against "Theoretical Truth"
        # Here we follow the repo style: run in target dtype.
        dx_ref = self.get_reference_impl(g, y)

        # 4. Run Custom Op Implementation
        # Note: The op is registered as fused_swiglu_bwd; invoke it here based on the actual implementation.
        dx_custom = fused_swiglu_bwd(g, y)

        # 5. Verification
        # Set tolerance
        if dtype == "bfloat16":
            if hidden_size > 1024:
                rtol, atol = 1e-1, 1e-1  # Loose tolerance for large scale
            else:
                rtol, atol = 2e-2, 2e-2
        else:
            rtol, atol = 1e-4, 1e-4

        # Convert to numpy/float32 for comparison
        ref_res = dx_ref.astype("float32").numpy()
        custom_res = dx_custom.astype("float32").numpy()

        # Check Max Difference
        max_diff = np.max(np.abs(ref_res - custom_res))

        # Check Cosine Similarity (Crucial for BF16)
        ref_flat = ref_res.flatten()
        custom_flat = custom_res.flatten()
        cos_sim = np.dot(ref_flat, custom_flat) / (
            np.linalg.norm(ref_flat) * np.linalg.norm(custom_flat) + 1e-8
        )

        print(
            f"\n[Test Case] Shape=({batch_size}, {seq_len}, {hidden_size}) | Dtype={dtype}"
        )
        print(f"  Max Diff: {max_diff:.6f} (Tol: {atol})")
        print(f"  Cos Sim : {cos_sim:.6f}")

        # Assertion
        np.testing.assert_allclose(
            custom_res,
            ref_res,
            rtol=rtol,
            atol=atol,
            err_msg=f"Output mismatch: dtype={dtype}, shape={shape_y}",
        )

        # Additional assertion for BF16 direction correctness
        if dtype == "bfloat16":
            self.assertTrue(
                cos_sim > 0.99, f"Cosine similarity too low for BF16: {cos_sim}"
            )

    def test_fused_swiglu_bwd_fp32(self):
        self.run_fused_op_test(4, 128, 1024, "float32")

    def test_fused_swiglu_bwd_bf16(self):
        if core.is_bfloat16_supported(base.CUDAPlace(0)):
            self.run_fused_op_test(4, 128, 1024, "bfloat16")

    def test_fused_swiglu_bwd_large_shape(self):
        # Test large shape to ensure no index overflow or memory alignment issues
        # Config matches LLaMA-70B dimensions roughly
        if core.is_bfloat16_supported(base.CUDAPlace(0)):
            self.run_fused_op_test(2, 2048, 4096, "bfloat16")


if __name__ == "__main__":
    unittest.main()
