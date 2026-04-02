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

import paddle
import paddle.nn.functional as F
from paddle.nn.functional import swiglu


def fused_swiglu_scale_forward(x, scale):
    if paddle.is_compiled_with_cuda():
        from paddlefleet.ops import fused_swiglu_scale

        return fused_swiglu_scale(x, scale)
    else:
        out = swiglu(x)

        # scale broadcast
        scale_exp = scale.cast(x.dtype)
        while scale_exp.ndim < out.ndim:
            scale_exp = scale_exp.unsqueeze(-1)

        return out * scale_exp


def fused_swiglu_scale_backward(x, scale, out_grad):
    if paddle.is_compiled_with_cuda():
        from paddlefleet.ops import fused_swiglu_scale_bwd

        return fused_swiglu_scale_bwd(x, scale, out_grad)
    else:
        # ----------------------------
        # XPU / CPU fallback
        # ----------------------------
        hidden = x.shape[-1] // 2

        gate = x[..., :hidden]
        val = x[..., hidden:]

        sig = F.sigmoid(gate).cast(x.dtype)
        silu = gate * sig
        swiglu = silu * val

        # scale broadcast
        scale_exp = scale.cast(x.dtype)
        while scale_exp.ndim < out_grad.ndim:
            scale_exp = scale_exp.unsqueeze(-1)

        d_u = out_grad * scale_exp

        # ----------------------------
        # dv
        # ----------------------------
        d_val = d_u * silu

        # ----------------------------
        # dg
        # ----------------------------
        d_gate = d_u * val * sig * (1.0 + gate * (1.0 - sig))

        # ----------------------------
        # d_x concat back
        # ----------------------------
        d_x = paddle.concat([d_gate, d_val], axis=-1).cast(x.dtype)

        # ----------------------------
        # d_scale
        # sum(dout * swiglu) over hidden dim
        # ----------------------------
        d_scale = paddle.sum(
            out_grad.cast(paddle.float32) * swiglu.cast(paddle.float32), axis=-1
        ).cast(scale.dtype)

        return d_x, d_scale
