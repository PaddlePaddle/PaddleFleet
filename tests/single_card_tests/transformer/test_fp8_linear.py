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
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import random
import unittest

import numpy as np
import paddle

from paddlefleet.fp8 import FP8Linear
from paddlefleet.tensor_parallel import ColumnParallelLinear
from paddlefleet.transformer.transformer_config import TransformerConfig


def calc_diff(x: paddle.Tensor, y: paddle.Tensor):
    x, y = x.double().numpy(), y.double().numpy()
    denominator = (x * x + y * y).sum()
    if denominator == 0:  # Which means that all elements in x and y are 0
        return 0.0
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


class TestParallelMLP(unittest.TestCase):
    def setUp(self):
        self.config = TransformerConfig(
            num_hidden_layers=2,
            hidden_size=4096,
            intermediate_size=7168,
            use_bias=False,
            use_cpu_initialization=True,
        )

        paddle.manual_seed(123)
        np.random.seed(123)
        random.seed(123)
        self.fp8_linear = FP8Linear(
            self.config.hidden_size,
            self.config.intermediate_size,
            config=self.config,
            init_method=self.config.init_method,
        )

        paddle.amp.decorate(
            models=self.fp8_linear,
            level="O2",
            dtype="bfloat16",
        )
        self.fp8_linear.weight = paddle.nn.parameter.Parameter(
            self.fp8_linear.weight.T.contiguous().T
        )
        self.fp8_linear.weight.main_grad = None

        paddle.manual_seed(123)
        np.random.seed(123)
        random.seed(123)
        # self.fp32_linear = paddle.nn.Linear(self.config.hidden_size, self.config.intermediate_size, bias_attr=False)
        self.fp32_linear = ColumnParallelLinear(
            self.config.hidden_size,
            self.config.intermediate_size,
            init_method=self.config.init_method,
            bias=False,
            config=self.config,
            skip_bias_add=False,
            gather_output=False,
            tp_group=None,
        )

        self.acc_step = 10

    def test_forward_backward(self):
        np.random.seed(123)
        batch_size = 16384

        # warmup
        np_x = np.random.randn(batch_size, self.config.hidden_size).astype(
            "float32"
        )
        pd_x_fp32 = paddle.to_tensor(np_x)
        pd_x_bf16 = paddle.to_tensor(np_x).to(paddle.bfloat16)
        pd_x_fp32.stop_gradient = False
        pd_x_bf16.stop_gradient = False
        out_fp32, _ = self.fp32_linear(pd_x_fp32)
        out_fp32.sum().backward()
        out_fp8 = self.fp8_linear(pd_x_bf16)
        out_fp8.sum().backward()

        iter_start = paddle.Event(enable_timing=True)
        iter_end = paddle.Event(enable_timing=True)
        fp8_runtimes = np.zeros((self.acc_step, 2))
        fp32_runtimes = np.zeros((self.acc_step, 2))

        for i in range(self.acc_step):
            # paddle.cuda.nvtx.range_push(f"fp8_iter{i}")
            np_x = np.random.randn(batch_size, self.config.hidden_size).astype(
                "float32"
            )
            pd_x_fp32 = paddle.to_tensor(np_x)
            pd_x_bf16 = paddle.to_tensor(np_x).to(paddle.bfloat16)

            pd_x_fp32.stop_gradient = False
            pd_x_bf16.stop_gradient = False

            # paddle.cuda.nvtx.range_push(f"fp32_forward")

            iter_start.record()
            out_fp32, _ = self.fp32_linear(pd_x_fp32)
            iter_end.record()
            paddle.cuda.synchronize()
            fp32_runtimes[i, 0] = iter_start.elapsed_time(iter_end)

            # paddle.cuda.nvtx.range_pop()

            # paddle.cuda.nvtx.range_push(f"fp32_backward")
            iter_start.record()
            out_fp32.sum().backward()
            iter_end.record()
            paddle.cuda.synchronize()
            fp32_runtimes[i, 1] = iter_start.elapsed_time(iter_end)

            # paddle.cuda.nvtx.range_push(f"fp8_forward")
            iter_start.record()
            out_fp8 = self.fp8_linear(pd_x_bf16)
            iter_end.record()
            paddle.cuda.synchronize()
            fp8_runtimes[i, 0] = iter_start.elapsed_time(iter_end)

            # paddle.cuda.nvtx.range_push(f"fp8_backward")
            iter_start.record()
            out_fp8.sum().backward()
            iter_end.record()
            paddle.cuda.synchronize()
            fp8_runtimes[i, 1] = iter_start.elapsed_time(iter_end)

            out_diff = calc_diff(out_fp32, out_fp8)
            assert out_diff < 0.001, f"iter {i} failed, out_diff: {out_diff}"

            w_grad_diff = calc_diff(
                self.fp32_linear.weight.grad, self.fp8_linear.weight.main_grad
            )
            x_grad_diff = calc_diff(pd_x_fp32.grad, pd_x_bf16.grad)
            assert w_grad_diff < 0.001, (
                f"iter {i} failed, w_grad_diff: {w_grad_diff}"
            )
            assert x_grad_diff < 0.001, (
                f"iter {i} failed, x_grad_diff: {x_grad_diff}"
            )
            # paddle.cuda.nvtx.range_pop()
        # print("fp32_runtimes", fp32_runtimes)
        # print("fp8_runtimes", fp8_runtimes)


if __name__ == "__main__":
    unittest.main()
