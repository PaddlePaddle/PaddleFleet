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
from paddle import base
from paddle.base import core

try:
    from paddlefleet.extensions import ops
except ImportError:
    ops = None


class TestMatmulBwd(unittest.TestCase):
    def setUp(self):
        if ops is None:
            self.skipTest("paddlefleet.extensions.ops not available")
        self.dtypes = ["float32", "float16", "bfloat16"]
        if not paddle.is_compiled_with_cuda():
            self.skipTest("CUDA is not available")

    def run_matmul_bwd(self, shape_x, shape_y, transpose_x, transpose_y, dtype):
        # Create Inputs
        x_np = np.random.random(shape_x).astype("float32")
        y_np = np.random.random(shape_y).astype("float32")

        if dtype == "bfloat16":
            x = paddle.to_tensor(x_np, dtype="float32").astype(dtype)
            y = paddle.to_tensor(y_np, dtype="float32").astype(dtype)
        else:
            x = paddle.to_tensor(x_np, dtype=dtype)
            y = paddle.to_tensor(y_np, dtype=dtype)

        x.stop_gradient = False
        y.stop_gradient = False

        # Forward pass (Reference)
        out = paddle.matmul(
            x, y, transpose_x=transpose_x, transpose_y=transpose_y
        )

        # Create output gradient
        out_grad = paddle.to_tensor(
            np.random.random(out.shape).astype("float32"), dtype=dtype
        )

        # Backward pass (Reference)
        paddle.autograd.backward([out], [out_grad])
        dx_ref = x.grad
        dy_ref = y.grad

        # Custom Op Backward
        # matmul_bwd(x, y, out_grad, transpose_x, transpose_y) -> [dx, dy]
        dx_custom, dy_custom = ops.matmul_bwd(
            x, y, out_grad, transpose_x, transpose_y
        )

        # Compare
        np.testing.assert_allclose(
            dx_custom.astype("float32").numpy(),
            dx_ref.astype("float32").numpy(),
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"dx mismatch for dtype={dtype}, tx={transpose_x}, ty={transpose_y}",
        )
        np.testing.assert_allclose(
            dy_custom.astype("float32").numpy(),
            dy_ref.astype("float32").numpy(),
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"dy mismatch for dtype={dtype}, tx={transpose_x}, ty={transpose_y}",
        )

    def test_matmul_bwd_fp32(self):
        self.run_matmul_bwd((32, 64), (64, 32), False, False, "float32")

    def test_matmul_bwd_fp16(self):
        self.run_matmul_bwd((32, 64), (64, 32), False, False, "float16")

    def test_matmul_bwd_transpose_x(self):
        # x: [K, M], y: [K, N] -> out: [M, N]
        # x: [64, 32], y: [64, 32]
        self.run_matmul_bwd((64, 32), (64, 32), True, False, "float32")

    def test_matmul_bwd_transpose_y(self):
        # x: [M, K], y: [N, K] -> out: [M, N]
        # x: [32, 64], y: [32, 64]
        self.run_matmul_bwd((32, 64), (32, 64), False, True, "float32")

    def test_matmul_bwd_transpose_both(self):
        # x: [K, M], y: [N, K] -> out: [M, N]
        # x: [64, 32], y: [32, 64]
        self.run_matmul_bwd((64, 32), (32, 64), True, True, "float32")

    def test_matmul_bwd_bf16(self):
        if core.is_bfloat16_supported(base.CUDAPlace(0)):
            self.run_matmul_bwd((32, 64), (64, 32), False, False, "bfloat16")


if __name__ == "__main__":
    unittest.main()
