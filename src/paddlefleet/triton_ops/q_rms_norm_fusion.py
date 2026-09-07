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

"""
Triton weight-free RMS normalization for query tensors.

Fuses: q * rsqrt(mean(q^2, dim=-1) + eps)
No learnable weight parameter.

Bit-exact alignment with Paddle's high_precision_norm=False eager path
(FLAGS_use_accuracy_compatible_kernel=false, the default):

  forward : x*x in bf16, fp32 sum, mean cast to bf16, rsqrt in fp32,
            invvar cast to bf16, final mul in bf16 — EXACT match.
  backward: grad_v = (-0.5) * fp32(grad_inv) * fp32(invvar)^3 → bf16,
            replicating CudaRsqrtGradFunctor non-Compatible (MPType=float)
            path — EXACT match (both forward and backward).
"""

import paddle

from .utils import enable_compat_on_triton_kernel, is_torch_compat_available

if is_torch_compat_available():
    paddle.enable_compat(scope={"triton"})

import triton
import triton.language as tl


@enable_compat_on_triton_kernel
@triton.jit
def q_rms_norm_fwd_kernel(
    X_ptr,
    Y_ptr,
    Invvar_ptr,
    stride_x_row: tl.constexpr,
    stride_y_row: tl.constexpr,
    N1: tl.constexpr,
    actual_n2: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    eps: tl.constexpr,
):
    """Forward: y = x * rsqrt(mean(x^2) + eps), no weight.

    Mirrors Paddle's high_precision_norm=False eager path:
      x*x in native dtype (bf16), tl.sum auto-upcasts to fp32,
      mean cast back to native dtype, rsqrt in fp32 then cast back,
      final mul in native dtype.
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    cols = tl.arange(0, BLOCK_N2)
    mask = cols < actual_n2

    for row_idx in range(pid, N1, num_programs):
        x_offset = row_idx * stride_x_row
        y_offset = row_idx * stride_y_row

        # Load in native dtype (bf16) — do NOT upcast before squaring.
        x_native = tl.load(X_ptr + x_offset + cols, mask=mask, other=0.0)
        # x*x stays in native dtype (matches Paddle's q.square() which is bf16*bf16).
        x_sq = x_native * x_native
        # tl.sum auto-upcasts bf16 to fp32 for accumulation (matches Paddle mean
        # which uses fp32 reduction internally); divide in fp32.
        var_fp32 = tl.sum(x_sq.to(tl.float32), axis=0) / actual_n2
        # Paddle's bf16 mean tensor = (fp32_sum / N).cast(bf16), so cast now.
        # Then rsqrt: tl.rsqrt only supports fp32/fp64, so cast to fp32 for rsqrt
        # then cast result back to native dtype — matching Paddle's rsqrt(bf16_input)
        # which internally computes in fp32 then returns bf16.
        var_native_as_fp32 = var_fp32.to(x_native.dtype).to(tl.float32)
        invvar_fp32 = tl.rsqrt(var_native_as_fp32 + eps)
        invvar = invvar_fp32.to(x_native.dtype)
        y = x_native * invvar

        tl.store(Y_ptr + y_offset + cols, y, mask=mask)
        # invvar saved for backward; always store as fp32.
        tl.store(Invvar_ptr + row_idx, invvar.to(tl.float32))


@enable_compat_on_triton_kernel
@triton.jit
def q_rms_norm_bwd_kernel(
    DY_ptr,
    X_ptr,
    Invvar_ptr,
    DX_ptr,
    stride_dy_row: tl.constexpr,
    stride_x_row: tl.constexpr,
    stride_dx_row: tl.constexpr,
    N1: tl.constexpr,
    actual_n2: tl.constexpr,
    BLOCK_N2: tl.constexpr,
):
    """Backward: replicates Paddle's CudaRsqrtGradFunctor non-Compatible path.

    FLAGS_use_accuracy_compatible_kernel defaults to false, so Paddle uses
    MPType=float (fp32) for rsqrt_grad. The key rounding chain:
      sum(dy*x) in fp32 → round to bf16  (Paddle sum returns bf16)
      → upcast to fp32 for rsqrt_grad computation
      → (-0.5) * fp32(bf16_grad_inv) * fp32(bf16_invvar)^3 → cast to bf16
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    cols = tl.arange(0, BLOCK_N2)
    mask = cols < actual_n2

    for row_idx in range(pid, N1, num_programs):
        dy_offset = row_idx * stride_dy_row
        x_offset = row_idx * stride_x_row
        dx_offset = row_idx * stride_dx_row

        dy = tl.load(DY_ptr + dy_offset + cols, mask=mask, other=0.0)
        x = tl.load(X_ptr + x_offset + cols, mask=mask, other=0.0)
        # invvar stored as fp32; keep fp32 for grad_v computation.
        invvar_fp32 = tl.load(Invvar_ptr + row_idx)  # fp32

        # grad_x_direct = dy * inv  (native dtype — Paddle's first grad-accumulation kernel)
        invvar = invvar_fp32.to(dy.dtype)
        grad_x_direct = dy * invvar

        # Paddle's sum on bf16 input: fp32 internal accumulation but returns bf16.
        # CudaRsqrtGradFunctor then upcasts that bf16 dout back to fp32.
        # We must replicate this bf16 rounding step before the fp32 rsqrt_grad.
        grad_inv = tl.sum((dy * x).to(tl.float32), axis=0).to(
            dy.dtype
        )  # fp32 sum → bf16

        # Replicate CudaRsqrtGradFunctor non-Compatible (MPType=fp32) path:
        #   result = (-0.5f) * fp32(bf16_dout) * fp32(bf16_out)^3  -> cast to native dtype
        grad_v_fp32 = (
            -0.5
            * grad_inv.to(tl.float32)
            * invvar_fp32
            * invvar_fp32
            * invvar_fp32
        )
        grad_v = grad_v_fp32.to(dy.dtype)

        # grad_sq_per_elem = grad_v / N * 2 * x  (native dtype)
        n_inv = tl.full(grad_v.shape, 1.0 / actual_n2, dy.dtype)
        grad_x_via_sq = grad_v * n_inv * tl.full(x.shape, 2.0, dy.dtype) * x

        tl.store(
            DX_ptr + dx_offset + cols, grad_x_direct + grad_x_via_sq, mask=mask
        )


class QRMSNormFusionTriton(paddle.autograd.PyLayer):
    """Weight-free Triton RMS norm for query tensors.

    Fuses q * rsqrt(mean(q^2, dim=-1) + eps) into a single kernel.
    Mirrors Paddle's high_precision_norm=False eager path for bit-exact
    alignment: bf16 intermediates with fp32 reductions.
    """

    @staticmethod
    def forward(ctx, x, epsilon=1e-5):
        orig_shape = x.shape
        n2 = x.shape[-1]
        block_n2 = triton.next_power_of_2(n2)
        n1 = 1
        for s in orig_shape[:-1]:
            n1 *= s

        if x.ndim >= 2:
            stride_x_row = x.stride()[x.ndim - 2]
        else:
            stride_x_row = n2

        y = paddle.empty(orig_shape, dtype=x.dtype)
        stride_y_row = n2  # output is always contiguous
        invvar = paddle.empty([n1], dtype=paddle.float32)

        ROWS_PER_PROG = 128
        num_programs = min(
            n1, max(1, (n1 + ROWS_PER_PROG - 1) // ROWS_PER_PROG)
        )

        q_rms_norm_fwd_kernel[(num_programs,)](
            x,
            y,
            invvar,
            stride_x_row,
            stride_y_row,
            n1,
            n2,
            BLOCK_N2=block_n2,
            eps=epsilon,
            num_warps=4 if block_n2 > 256 else 1,
        )

        ctx.save_for_backward(x, invvar)
        ctx.n1 = n1
        ctx.n2 = n2
        ctx.block_n2 = block_n2
        ctx.stride_x_row = stride_x_row
        ctx.num_programs = num_programs
        return y

    @staticmethod
    def backward(ctx, dy):
        x, invvar = ctx.saved_tensor()
        n1 = ctx.n1
        n2 = ctx.n2
        block_n2 = ctx.block_n2
        stride_x_row = ctx.stride_x_row
        num_programs = ctx.num_programs

        dx = paddle.empty(dy.shape, dtype=dy.dtype)

        if dy.ndim >= 2:
            stride_dy_row = dy.stride()[dy.ndim - 2]
        else:
            stride_dy_row = n2
        stride_dx_row = n2  # output contiguous

        q_rms_norm_bwd_kernel[(num_programs,)](
            dy,
            x,
            invvar,
            dx,
            stride_dy_row,
            stride_x_row,
            stride_dx_row,
            n1,
            n2,
            BLOCK_N2=block_n2,
            num_warps=4 if block_n2 > 256 else 1,
        )

        return dx


def fused_q_rms_norm(x, eps=1e-5):
    """Functional API for weight-free fused RMS norm.

    Args:
        x: Input tensor (any shape, norm over last dim).
        eps: Epsilon.

    Returns:
        Normalized tensor (same shape/dtype as input).
    """
    return QRMSNormFusionTriton.apply(x, eps)
