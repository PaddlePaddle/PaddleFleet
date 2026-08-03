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

from paddlefleet.triton_ops import fused_q_rms_norm


def eager_q_rms_norm(q, eps):
    """non-high_precision_norm path: stay in input dtype throughout."""
    return q * paddle.rsqrt(q.square().mean(-1, keepdim=True) + eps)


@unittest.skipIf(
    not paddle.is_compiled_with_cuda()
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "fused_q_rms_norm requires GPU with SM80+ (bf16)",
)
class TestQRMSNormFusion(unittest.TestCase):
    def _run(self, shape, eps, dtype):
        paddle.seed(42)
        x = paddle.randn(shape).cast(dtype)

        x_f = x.clone()
        x_f.stop_gradient = False
        out_f = fused_q_rms_norm(x_f, eps=eps)
        out_f.sum().backward()

        x_e = x.clone()
        x_e.stop_gradient = False
        out_e = eager_q_rms_norm(x_e, eps)
        out_e.sum().backward()

        # kernel is designed for bit-exact alignment with the eager path
        np.testing.assert_allclose(
            out_f.astype("float32"), out_e.astype("float32"), atol=0, rtol=0
        )
        np.testing.assert_allclose(
            x_f.grad.astype("float32"),
            x_e.grad.astype("float32"),
            atol=0,
            rtol=0,
        )

    def test_forward_backward_matches_eager(self):
        for shape in ([1, 64, 64, 512], [1, 128, 64, 512], [2, 32, 64, 512]):
            for eps in (1e-5, 1e-6):
                self._run(shape, eps, "bfloat16")
