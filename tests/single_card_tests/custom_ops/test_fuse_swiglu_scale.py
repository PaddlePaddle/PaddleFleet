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

from paddlefleet.ops import fused_swiglu_scale


class TestFusedSwiGLUScale(unittest.TestCase):
    def setUp(self):
        self.dtypes = ["float32", "float16", "bfloat16"]
        if not paddle.is_compiled_with_cuda():
            self.skipTest("CUDA is not available")

        # Set random seed for reproducibility
        np.random.seed(42)
        paddle.seed(42)

    def get_reference_impl(self, x, scale):
        """
        Paddle native implementation of SwiGLU + Scale as ground truth.
        """
        # x shape: [B, 2*H] -> chunk -> gate, val
        gate, val = paddle.chunk(x, chunks=2, axis=-1)

        # SwiGLU: silu(gate) * val
        # F.silu(x) = x * sigmoid(x)
        swiglu_out = F.silu(gate) * val

        # Scale: swiglu_out * scale (Broadcast multiplication)
        out = swiglu_out * scale
        return out

    def run_fused_op_test(self, batch_size, hidden_size, dtype_x, dtype_scale):
        # 1. Construct input data
        shape_x = (batch_size, 2 * hidden_size)
        shape_scale = (batch_size, 1)

        # Use a reasonable range to avoid numerical instability
        x_np = np.random.normal(0, 1.0, shape_x).astype("float32")
        scale_np = np.random.uniform(0.5, 1.5, shape_scale).astype("float32")

        # 2. Create Reference Tensors
        if dtype_x == "bfloat16":
            x_ref = paddle.to_tensor(x_np).astype("bfloat16")
        else:
            x_ref = paddle.to_tensor(x_np, dtype=dtype_x)

        if dtype_scale == "bfloat16":
            scale_ref = paddle.to_tensor(scale_np).astype("bfloat16")
        else:
            scale_ref = paddle.to_tensor(scale_np, dtype=dtype_scale)

        x_ref.stop_gradient = False
        scale_ref.stop_gradient = False

        # 3. Create Custom Op Tensors (Independent copies)
        if dtype_x == "bfloat16":
            x_custom = paddle.to_tensor(x_np).astype("bfloat16")
        else:
            x_custom = paddle.to_tensor(x_np, dtype=dtype_x)

        if dtype_scale == "bfloat16":
            scale_custom = paddle.to_tensor(scale_np).astype("bfloat16")
        else:
            scale_custom = paddle.to_tensor(scale_np, dtype=dtype_scale)

        x_custom.stop_gradient = False
        scale_custom.stop_gradient = False

        # 4. Forward Pass
        # Reference implementation
        out_ref = self.get_reference_impl(x_ref, scale_ref)

        # Custom Op implementation
        # Note: The C++ op might return a list [Tensor], handle it if necessary
        ret = fused_swiglu_scale(x_custom, scale_custom)
        out_custom = ret[0] if isinstance(ret, (list, tuple)) else ret

        # 5. Backward Pass
        # Create random output gradient
        grad_np = np.random.random(out_ref.shape).astype("float32")
        if out_ref.dtype == paddle.bfloat16:
            out_grad = paddle.to_tensor(grad_np).astype("bfloat16")
        else:
            out_grad = paddle.to_tensor(grad_np, dtype=out_ref.dtype)

        paddle.autograd.backward([out_ref], [out_grad])
        paddle.autograd.backward([out_custom], [out_grad])

        # 6. Verification
        # Set tolerance: BF16 has lower precision, requiring larger tolerance
        if "bfloat16" in [dtype_x, dtype_scale]:
            rtol, atol = 2e-2, 2e-2
            # Relax tolerance for BF16 gradient accumulation with large shape (Reduction dim > 1024)
            # This accounts for the accumulation error difference between FP32 (Fused) and BF16 (Ref)
            if hidden_size > 1024:
                rtol, atol = 0.1, 0.1
        else:
            rtol, atol = 1e-4, 1e-4

        # Verify Forward Output
        np.testing.assert_allclose(
            out_custom.astype("float32").numpy(),
            out_ref.astype("float32").numpy(),
            rtol=rtol,
            atol=atol,
            err_msg=f"Forward output mismatch: dtype_x={dtype_x}, dtype_scale={dtype_scale}",
        )

        # Verify Input Gradient (dX)
        np.testing.assert_allclose(
            x_custom.grad.astype("float32").numpy(),
            x_ref.grad.astype("float32").numpy(),
            rtol=rtol,
            atol=atol,
            err_msg=f"Gradient X mismatch: dtype_x={dtype_x}, dtype_scale={dtype_scale}",
        )

        # Verify Scale Gradient (dScale)
        np.testing.assert_allclose(
            scale_custom.grad.astype("float32").numpy(),
            scale_ref.grad.astype("float32").numpy(),
            rtol=rtol,
            atol=atol,
            err_msg=f"Gradient Scale mismatch: dtype_x={dtype_x}, dtype_scale={dtype_scale}",
        )

    def test_fused_swiglu_fp32(self):
        self.run_fused_op_test(32, 64, "float32", "float32")

    def test_fused_swiglu_bf16(self):
        if core.is_bfloat16_supported(base.CUDAPlace(0)):
            self.run_fused_op_test(32, 64, "bfloat16", "bfloat16")

    def test_fused_swiglu_mixed_precision(self):
        # Test mixed precision: Input=BF16, Scale=FP32
        if core.is_bfloat16_supported(base.CUDAPlace(0)):
            self.run_fused_op_test(16, 128, "bfloat16", "float32")

    def test_fused_swiglu_large_shape(self):
        # Test large shape to ensure no index overflow or memory alignment issues
        if core.is_bfloat16_supported(base.CUDAPlace(0)):
            self.run_fused_op_test(4, 4096, "bfloat16", "float32")


if __name__ == "__main__":
    unittest.main()
