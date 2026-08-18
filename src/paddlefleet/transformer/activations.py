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

import math

import paddle
import paddle.nn.functional as F


def situ(x: paddle.Tensor, beta: float = 1.0) -> paddle.Tensor:
    """Apply the SiTU gate activation with float32 intermediates."""

    if not math.isfinite(beta) or beta <= 0:
        raise ValueError(
            f"SiTU beta must be a positive finite value, but got {beta!r}."
        )

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

    if not math.isfinite(beta) or beta <= 0:
        raise ValueError(
            f"SiTU beta must be a positive finite value, but got {beta!r}."
        )
    if linear_beta is not None and (
        not math.isfinite(linear_beta) or linear_beta <= 0
    ):
        raise ValueError(
            "SiTU linear_beta must be a positive finite value or None, "
            f"but got {linear_beta!r}."
        )

    input_dtype = x.dtype
    gate, up = paddle.chunk(x, chunks=2, axis=-1)
    gate = gate.astype("float32")
    up = up.astype("float32")

    gate = beta * paddle.tanh(gate / beta) * F.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * paddle.tanh(up / linear_beta)
    return (gate * up).astype(input_dtype)


def situ_glu_scale_forward(
    x: paddle.Tensor,
    probs: paddle.Tensor,
    beta: float = 1.0,
    linear_beta: float | None = None,
    situ_glu_fusion: bool = True,
) -> paddle.Tensor:
    """Apply SiTU-GLU and router scaling with float32 intermediates."""

    if situ_glu_fusion:
        from paddlefleet.triton_ops.utils import is_triton_available

        if is_triton_available():
            from paddlefleet.triton_ops.situ_glu import (
                situ_glu_scale_forward_triton,
            )

            return situ_glu_scale_forward_triton(x, probs, beta, linear_beta)

    input_dtype = x.dtype
    probs = probs.astype("float32")
    if probs.ndim == 1:
        probs = probs.unsqueeze(-1)
    return (
        situ_glu(x.astype("float32"), beta=beta, linear_beta=linear_beta)
        * probs
    ).astype(input_dtype)


def situ_glu_scale_backward(
    x: paddle.Tensor,
    probs: paddle.Tensor,
    out_grad: paddle.Tensor,
    beta: float = 1.0,
    linear_beta: float | None = None,
    situ_glu_fusion: bool = True,
) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """Backward for :func:`situ_glu_scale_forward`."""

    if not math.isfinite(beta) or beta <= 0:
        raise ValueError(
            f"SiTU beta must be a positive finite value, but got {beta!r}."
        )
    if linear_beta is not None and (
        not math.isfinite(linear_beta) or linear_beta <= 0
    ):
        raise ValueError(
            "SiTU linear_beta must be a positive finite value or None, "
            f"but got {linear_beta!r}."
        )
    if situ_glu_fusion:
        from paddlefleet.triton_ops.utils import is_triton_available

        if is_triton_available():
            from paddlefleet.triton_ops.situ_glu import (
                situ_glu_scale_backward_triton,
            )

            return situ_glu_scale_backward_triton(
                x, probs, out_grad, beta, linear_beta
            )

    gate, up = paddle.chunk(x, chunks=2, axis=-1)
    gate = gate.astype("float32")
    up = up.astype("float32")
    out_grad = out_grad.astype("float32")
    probs_view = probs.astype("float32")
    if probs_view.ndim == 1:
        probs_view = probs_view.unsqueeze(-1)

    gate_tanh = paddle.tanh(gate / beta)
    gate_sigmoid = F.sigmoid(gate)
    gate_act = beta * gate_tanh * gate_sigmoid
    gate_grad = (
        1.0 - gate_tanh.square()
    ) * gate_sigmoid + beta * gate_tanh * gate_sigmoid * (1.0 - gate_sigmoid)

    if linear_beta is None:
        up_act = up
        up_grad = paddle.ones_like(up)
    else:
        up_tanh = paddle.tanh(up / linear_beta)
        up_act = linear_beta * up_tanh
        up_grad = 1.0 - up_tanh.square()

    scaled_grad = out_grad * probs_view
    gate_input_grad = scaled_grad * up_act * gate_grad
    up_input_grad = scaled_grad * gate_act * up_grad
    input_grad = paddle.concat(
        [gate_input_grad, up_input_grad], axis=-1
    ).astype(x.dtype)

    unscaled_output = gate_act * up_act
    probs_grad = (out_grad * unscaled_output).sum(axis=-1, keepdim=True)
    probs_grad = probs_grad.reshape(probs.shape).astype(probs.dtype)
    scaled_output = (unscaled_output * probs_view).astype(x.dtype)
    return input_grad, scaled_output, probs_grad


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


__all__ = [
    "SituAndMul",
    "situ",
    "situ_glu",
    "situ_glu_scale_backward",
    "situ_glu_scale_forward",
]
