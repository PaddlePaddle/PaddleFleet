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

from __future__ import annotations

import paddle
import paddle.nn.functional as F


def situ(x: paddle.Tensor, beta: float = 1.0) -> paddle.Tensor:
    """Apply the SiTU gate activation with float32 intermediates."""

    assert beta > 0

    input_dtype = x.dtype
    x = x.astype("float32")
    output = beta * paddle.tanh(x / beta) * F.sigmoid(x)
    return output.astype(input_dtype)


def situ_glu(
    x: paddle.Tensor,
    beta: float = 1.0,
    linear_beta: float | None = None,
) -> paddle.Tensor:
    """Apply Kimi-K3 SiTU-GLU to concatenated ``[gate, up]`` projections."""

    if x.shape[-1] % 2 != 0:
        raise ValueError(
            "SiTU-GLU requires an even last dimension containing "
            f"concatenated [gate, up] projections, but got {x.shape[-1]}."
        )

    assert beta > 0
    assert linear_beta is None or linear_beta > 0

    input_dtype = x.dtype
    gate, up = paddle.chunk(x, chunks=2, axis=-1)
    gate = gate.astype("float32")
    up = up.astype("float32")

    gate = beta * paddle.tanh(gate / beta) * F.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * paddle.tanh(up / linear_beta)
    return (gate * up).astype(input_dtype)


class SituAndMul(paddle.nn.Layer):
    """Layer wrapper matching the official Kimi-K3 ``SituAndMul`` contract."""

    def __init__(
        self,
        beta: float = 1.0,
        linear_beta: float | None = None,
    ):
        super().__init__()
        self.beta = beta
        self.linear_beta = linear_beta

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return situ_glu(
            x,
            beta=self.beta,
            linear_beta=self.linear_beta,
        )


__all__ = ["SituAndMul", "situ", "situ_glu"]
