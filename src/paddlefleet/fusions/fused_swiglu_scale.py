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


def _broadcast_scale(scale, target_dtype, target_ndim):
    """Cast scale to target dtype and unsqueeze to match target rank."""
    scale_exp = scale.cast(target_dtype)
    while scale_exp.ndim < target_ndim:
        scale_exp = scale_exp.unsqueeze(-1)
    return scale_exp


def fused_swiglu_scale_forward(x, scale):
    if paddle.is_compiled_with_cuda():
        from paddlefleet_ops import fused_swiglu_scale

        return fused_swiglu_scale(x, scale)
    else:
        out = swiglu(x)
        scale_exp = _broadcast_scale(scale, x.dtype, out.ndim)
        return out * scale_exp


def fused_swiglu_scale_backward(x, scale, out_grad):
    if paddle.is_compiled_with_cuda():
        from paddlefleet_ops import fused_swiglu_scale_bwd

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
        swiglu_val = silu * val

        scale_exp = _broadcast_scale(scale, x.dtype, out_grad.ndim)
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
            out_grad.cast(paddle.float32) * swiglu_val.cast(paddle.float32),
            axis=-1,
        ).cast(scale.dtype)

        return d_x, d_scale


# ============================================================================
# Clamped variants (added; do NOT modify the original interfaces above)
# ============================================================================


def fused_swiglu_scale_clamp_forward(x, scale, clamp_value):
    """Clamped variant of ``fused_swiglu_scale_forward``.

    Gate is clamped to (-inf, clamp_value]; value is clamped to
    [-clamp_value, clamp_value] before SwiGLU * scale.
    """
    if paddle.is_compiled_with_cuda():
        from paddlefleet_ops import fused_swiglu_scale_clamp

        return fused_swiglu_scale_clamp(x, scale, float(clamp_value))

    # ----------------------------
    # XPU / CPU fallback
    # ----------------------------
    hidden = x.shape[-1] // 2
    gate = paddle.clip(x[..., :hidden], max=clamp_value)
    val = paddle.clip(x[..., hidden:], min=-clamp_value, max=clamp_value)
    out = F.silu(gate) * val
    scale_exp = _broadcast_scale(scale, x.dtype, out.ndim)
    return out * scale_exp


def fused_swiglu_scale_clamp_backward(x, scale, out_grad, clamp_value):
    """Clamped variant of ``fused_swiglu_scale_backward``.

    Gradients are zeroed where the corresponding input was saturated.
    """
    if paddle.is_compiled_with_cuda():
        from paddlefleet_ops import fused_swiglu_scale_clamp_bwd

        return fused_swiglu_scale_clamp_bwd(
            x, scale, out_grad, float(clamp_value)
        )

    # ----------------------------
    # XPU / CPU fallback
    # ----------------------------
    hidden = x.shape[-1] // 2
    gate_raw = x[..., :hidden]
    val_raw = x[..., hidden:]

    gate = paddle.clip(gate_raw, max=clamp_value)
    val = paddle.clip(val_raw, min=-clamp_value, max=clamp_value)
    g_mask = (gate_raw <= clamp_value).cast(x.dtype)
    v_mask = ((val_raw <= clamp_value) & (val_raw >= -clamp_value)).cast(
        x.dtype
    )

    sig = F.sigmoid(gate).cast(x.dtype)
    silu = gate * sig
    swiglu_val = silu * val

    scale_exp = _broadcast_scale(scale, x.dtype, out_grad.ndim)
    d_u = out_grad * scale_exp

    d_val = d_u * silu * v_mask
    d_gate = d_u * val * sig * (1.0 + gate * (1.0 - sig)) * g_mask
    d_x = paddle.concat([d_gate, d_val], axis=-1).cast(x.dtype)

    d_scale = paddle.sum(
        out_grad.cast(paddle.float32) * swiglu_val.cast(paddle.float32),
        axis=-1,
    ).cast(scale.dtype)

    return d_x, d_scale
