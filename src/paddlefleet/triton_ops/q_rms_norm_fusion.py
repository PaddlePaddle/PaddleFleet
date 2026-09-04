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

Rounding chain (mirrors Paddle's high_precision_norm=False eager path,
FLAGS_use_accuracy_compatible_kernel=false, the default):

  forward : x*x in bf16, fp32 sum, mean cast to bf16, rsqrt in fp32,
            invvar cast to bf16, final mul in bf16.
  backward: grad_v = (-0.5) * fp32(grad_inv) * fp32(invvar)^3 -> bf16,
            replicating CudaRsqrtGradFunctor non-Compatible (MPType=float).

Measured agreement with the Paddle eager path on B30Z (bf16,
[4, 8192, 24, 512], 3 input distributions x 2 seeds): the rounding chain is
reproduced but the *reduction order* is not Paddle's, so the result is not
bitwise identical -- 0.42%-0.43% of the output elements differ by 1-2 bf16
ulps on N(0,1)-per-row data, 5e-7..1e-5 of elements on heavy-tailed / sparse
data. Do not rely on bitwise equality with eager.

Scheduling: rows are distributed over programs with a fixed compile-time
unroll (``UNROLL``, chosen by ``_pick_unroll``) and every row is still reduced
by the same lane set as a plain one-row-per-iteration kernel, so changing the
unroll factor or the grid does not change any output bit. Reducing a row across
a *different* number of lanes (i.e. changing ``num_warps``) does change the
fp32 accumulation order, so ``num_warps`` must stay tied to BLOCK_N2 as in
``_num_warps``.
"""

import paddle

from .utils import enable_compat_on_triton_kernel, is_torch_compat_available

if is_torch_compat_available():
    paddle.enable_compat(scope={"triton"})

import triton
import triton.language as tl


@triton.jit
def _q_rms_norm_fwd_row(
    Y_ptr,
    Invvar_ptr,
    row_idx,
    cols,
    mask,
    stride_y_row: tl.constexpr,
    actual_n2: tl.constexpr,
    eps: tl.constexpr,
    x_native,
):
    """Normalize one already-loaded row.

    Split out of the kernel so several rows can have their loads issued
    before any reduction runs; the reduction itself is unchanged (one
    ``tl.sum`` over one BLOCK_N2-wide row, i.e. the same lane set and the
    same fp32 accumulation order as a one-row-at-a-time kernel).
    """
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

    tl.store(Y_ptr + row_idx * stride_y_row + cols, y, mask=mask)
    # invvar saved for backward; always store as fp32.
    tl.store(Invvar_ptr + row_idx, invvar.to(tl.float32))


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
    UNROLL: tl.constexpr,
):
    """Forward: y = x * rsqrt(mean(x^2) + eps), no weight.

    Each program owns UNROLL rows, ``pid + j * num_programs``, and the grid is
    sized so that ``UNROLL * num_programs == N1`` exactly (the launcher picks
    UNROLL to divide N1), hence no row masking is needed.

    All UNROLL loads are issued before the first reduction. That is the whole
    point of the unroll: one row only puts ``BLOCK_N2 * 2 / threads`` bytes in
    flight per thread (8 B at BLOCK_N2=512, num_warps=4), which is about half
    of what is needed to keep the DRAM pipe full on B30Z. UNROLL independent
    loads multiply that. See the launcher for measured numbers.

    ``UNROLL == 0`` selects the pre-optimization schedule -- a grid-stride loop
    over rows with one load in flight -- and is kept only as the reference the
    unrolled schedule is checked against; ``_pick_unroll`` never returns it.
    Every row is still reduced by the same lane set, so the two schedules agree
    bit for bit; only the number of loads in flight differs.
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    cols = tl.arange(0, BLOCK_N2)
    mask = cols < actual_n2

    if UNROLL == 0:
        for row_idx in range(pid, N1, num_programs):
            xl = tl.load(
                X_ptr + row_idx * stride_x_row + cols, mask=mask, other=0.0
            )
            _q_rms_norm_fwd_row(
                Y_ptr,
                Invvar_ptr,
                row_idx,
                cols,
                mask,
                stride_y_row,
                actual_n2,
                eps,
                xl,
            )
        return

    r0 = pid
    # Load in native dtype (bf16) — do NOT upcast before squaring.
    x0 = tl.load(X_ptr + r0 * stride_x_row + cols, mask=mask, other=0.0)
    if UNROLL >= 2:
        r1 = r0 + num_programs
        x1 = tl.load(X_ptr + r1 * stride_x_row + cols, mask=mask, other=0.0)
    if UNROLL >= 4:
        r2 = r1 + num_programs
        r3 = r2 + num_programs
        x2 = tl.load(X_ptr + r2 * stride_x_row + cols, mask=mask, other=0.0)
        x3 = tl.load(X_ptr + r3 * stride_x_row + cols, mask=mask, other=0.0)
    if UNROLL >= 8:
        r4 = r3 + num_programs
        r5 = r4 + num_programs
        r6 = r5 + num_programs
        r7 = r6 + num_programs
        x4 = tl.load(X_ptr + r4 * stride_x_row + cols, mask=mask, other=0.0)
        x5 = tl.load(X_ptr + r5 * stride_x_row + cols, mask=mask, other=0.0)
        x6 = tl.load(X_ptr + r6 * stride_x_row + cols, mask=mask, other=0.0)
        x7 = tl.load(X_ptr + r7 * stride_x_row + cols, mask=mask, other=0.0)

    _q_rms_norm_fwd_row(
        Y_ptr, Invvar_ptr, r0, cols, mask, stride_y_row, actual_n2, eps, x0
    )
    if UNROLL >= 2:
        _q_rms_norm_fwd_row(
            Y_ptr, Invvar_ptr, r1, cols, mask, stride_y_row, actual_n2, eps, x1
        )
    if UNROLL >= 4:
        _q_rms_norm_fwd_row(
            Y_ptr, Invvar_ptr, r2, cols, mask, stride_y_row, actual_n2, eps, x2
        )
        _q_rms_norm_fwd_row(
            Y_ptr, Invvar_ptr, r3, cols, mask, stride_y_row, actual_n2, eps, x3
        )
    if UNROLL >= 8:
        _q_rms_norm_fwd_row(
            Y_ptr, Invvar_ptr, r4, cols, mask, stride_y_row, actual_n2, eps, x4
        )
        _q_rms_norm_fwd_row(
            Y_ptr, Invvar_ptr, r5, cols, mask, stride_y_row, actual_n2, eps, x5
        )
        _q_rms_norm_fwd_row(
            Y_ptr, Invvar_ptr, r6, cols, mask, stride_y_row, actual_n2, eps, x6
        )
        _q_rms_norm_fwd_row(
            Y_ptr, Invvar_ptr, r7, cols, mask, stride_y_row, actual_n2, eps, x7
        )


@triton.jit
def _q_rms_norm_bwd_row(
    DX_ptr,
    row_idx,
    cols,
    mask,
    stride_dx_row: tl.constexpr,
    actual_n2: tl.constexpr,
    dy,
    x,
    invvar_fp32,
):
    """Backward for one already-loaded row (reduction structure unchanged)."""
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
        -0.5 * grad_inv.to(tl.float32) * invvar_fp32 * invvar_fp32 * invvar_fp32
    )
    grad_v = grad_v_fp32.to(dy.dtype)

    # grad_sq_per_elem = grad_v / N * 2 * x  (native dtype)
    n_inv = tl.full(grad_v.shape, 1.0 / actual_n2, dy.dtype)
    grad_x_via_sq = grad_v * n_inv * tl.full(x.shape, 2.0, dy.dtype) * x

    tl.store(
        DX_ptr + row_idx * stride_dx_row + cols,
        grad_x_direct + grad_x_via_sq,
        mask=mask,
    )


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
    UNROLL: tl.constexpr,
):
    """Backward: replicates Paddle's CudaRsqrtGradFunctor non-Compatible path.

    FLAGS_use_accuracy_compatible_kernel defaults to false, so Paddle uses
    MPType=float (fp32) for rsqrt_grad. The key rounding chain:
      sum(dy*x) in fp32 → round to bf16  (Paddle sum returns bf16)
      → upcast to fp32 for rsqrt_grad computation
      → (-0.5) * fp32(bf16_grad_inv) * fp32(bf16_invvar)^3 → cast to bf16

    Same UNROLL scheme as the forward kernel; a backward row needs two loads
    (dy and x) so the useful unroll factor is half the forward one.
    ``UNROLL == 0`` is the pre-optimization grid-stride loop, kept as the
    reference for the bit-exactness check only.
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    cols = tl.arange(0, BLOCK_N2)
    mask = cols < actual_n2

    if UNROLL == 0:
        for row_idx in range(pid, N1, num_programs):
            dl = tl.load(
                DY_ptr + row_idx * stride_dy_row + cols, mask=mask, other=0.0
            )
            xl = tl.load(
                X_ptr + row_idx * stride_x_row + cols, mask=mask, other=0.0
            )
            vl = tl.load(Invvar_ptr + row_idx)
            _q_rms_norm_bwd_row(
                DX_ptr,
                row_idx,
                cols,
                mask,
                stride_dx_row,
                actual_n2,
                dl,
                xl,
                vl,
            )
        return

    r0 = pid
    d0 = tl.load(DY_ptr + r0 * stride_dy_row + cols, mask=mask, other=0.0)
    x0 = tl.load(X_ptr + r0 * stride_x_row + cols, mask=mask, other=0.0)
    # invvar stored as fp32; keep fp32 for grad_v computation.
    v0 = tl.load(Invvar_ptr + r0)
    if UNROLL >= 2:
        r1 = r0 + num_programs
        d1 = tl.load(DY_ptr + r1 * stride_dy_row + cols, mask=mask, other=0.0)
        x1 = tl.load(X_ptr + r1 * stride_x_row + cols, mask=mask, other=0.0)
        v1 = tl.load(Invvar_ptr + r1)
    if UNROLL >= 4:
        r2 = r1 + num_programs
        r3 = r2 + num_programs
        d2 = tl.load(DY_ptr + r2 * stride_dy_row + cols, mask=mask, other=0.0)
        x2 = tl.load(X_ptr + r2 * stride_x_row + cols, mask=mask, other=0.0)
        d3 = tl.load(DY_ptr + r3 * stride_dy_row + cols, mask=mask, other=0.0)
        x3 = tl.load(X_ptr + r3 * stride_x_row + cols, mask=mask, other=0.0)
        v2 = tl.load(Invvar_ptr + r2)
        v3 = tl.load(Invvar_ptr + r3)

    _q_rms_norm_bwd_row(
        DX_ptr, r0, cols, mask, stride_dx_row, actual_n2, d0, x0, v0
    )
    if UNROLL >= 2:
        _q_rms_norm_bwd_row(
            DX_ptr, r1, cols, mask, stride_dx_row, actual_n2, d1, x1, v1
        )
    if UNROLL >= 4:
        _q_rms_norm_bwd_row(
            DX_ptr, r2, cols, mask, stride_dx_row, actual_n2, d2, x2, v2
        )
        _q_rms_norm_bwd_row(
            DX_ptr, r3, cols, mask, stride_dx_row, actual_n2, d3, x3, v3
        )


def _pick_unroll(n1, block_n2, cap, elem_budget):
    """Largest power-of-two unroll factor that is <= cap, keeps the live tile
    under ``elem_budget`` elements, and divides ``n1`` (so the kernel needs no
    row masking).

    ``elem_budget`` bounds registers: one live row is ``block_n2`` bf16
    elements, so ``UNROLL * block_n2`` elements are live before the first
    reduction. 4096 elements (8 KB per program at 128 threads = 64 B/thread)
    measured 32 registers with no spill for the forward kernel.

    Never returns 0. The kernels still implement ``UNROLL == 0`` (the
    pre-optimization grid-stride loop) because it is the reference this
    schedule's bit-exactness is checked against -- see
    tests/single_card_tests/custom_ops/test_q_rms_norm_unroll.py -- but nothing
    in production selects it.

    Callers must pass a ``cap`` the kernel actually implements (8 forward, 4
    backward). A larger factor would shrink the grid to ``n1 // UNROLL`` while
    the kernel body still handled only ``cap`` rows, leaving the tail of the
    output unwritten in uninitialized memory -- silently, since the buffers come
    from ``paddle.empty``.
    """
    u = 1
    while (
        u * 2 <= cap and (u * 2) * block_n2 <= elem_budget and n1 % (u * 2) == 0
    ):
        u *= 2
    return u


# Rows per program on the ``UNROLL == 0`` (pre-optimization) path.
_LEGACY_ROWS_PER_PROG = 128


def _num_programs(n1, unroll):
    """Grid size that goes with ``unroll``.

    ``unroll >= 1``: exactly one program per ``unroll`` rows, no row masking.
    ``unroll == 0``: the old ``ceil(n1 / 128)`` programs, each looping.
    """
    if unroll == 0:
        return min(
            n1,
            max(1, (n1 + _LEGACY_ROWS_PER_PROG - 1) // _LEGACY_ROWS_PER_PROG),
        )
    return n1 // unroll


def _num_warps(block_n2):
    """Warps per program.

    This also fixes how many lanes reduce one row, i.e. the fp32 accumulation
    order, so it must not be changed for numerical-reproducibility reasons
    alone. Measured on B30Z, BLOCK_N2=512 forward: 128 lanes/row (4 warps) is
    what the shipped kernel has always used; 64 lanes/row is ~2% faster but
    re-associates the sum.
    """
    return 4 if block_n2 > 256 else 1


class QRMSNormFusionTriton(paddle.autograd.PyLayer):
    """Weight-free Triton RMS norm for query tensors.

    Fuses q * rsqrt(mean(q^2, dim=-1) + eps) into a single kernel.
    Reproduces the rounding chain of Paddle's high_precision_norm=False eager
    path (bf16 intermediates with fp32 reductions); see the module docstring
    for the measured agreement.
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

        # One row per program only puts block_n2*2/threads bytes in flight per
        # thread, roughly half of what B30Z needs to saturate DRAM. Issuing
        # UNROLL independent row loads first fixes that without touching the
        # per-row reduction. Measured, [4,8192,24,512] bf16, 6.6 TB/s HBM:
        #   UNROLL=1 (old, 128 rows/program loop) 362.7 us
        #   UNROLL=2                              284.7 us
        #   UNROLL=4                              262.6 us
        #   UNROLL=8                              240.6 us  <- 1.51x, 6.7 TB/s
        # A pure load+store kernel over the same tensors measures 238.3 us, so
        # UNROLL=8 is within 1% of the memory floor and there is nothing left.
        unroll = _pick_unroll(n1, block_n2, cap=8, elem_budget=4096)
        num_programs = _num_programs(n1, unroll)

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
            UNROLL=unroll,
            num_warps=_num_warps(block_n2),
        )

        ctx.save_for_backward(x, invvar)
        ctx.n1 = n1
        ctx.n2 = n2
        ctx.block_n2 = block_n2
        ctx.stride_x_row = stride_x_row
        return y

    @staticmethod
    def backward(ctx, dy):
        x, invvar = ctx.saved_tensor()
        n1 = ctx.n1
        n2 = ctx.n2
        block_n2 = ctx.block_n2
        stride_x_row = ctx.stride_x_row

        dx = paddle.empty(dy.shape, dtype=dy.dtype)

        if dy.ndim >= 2:
            stride_dy_row = dy.stride()[dy.ndim - 2]
        else:
            stride_dy_row = n2
        stride_dx_row = n2  # output contiguous

        # Backward reads two rows (dy and x) per output row, so it reaches the
        # same bytes-in-flight with half the unroll. Measured, same shape:
        #   UNROLL=1 (old, 128 rows/program loop) 436.0 us
        #   UNROLL=2                              364.8 us
        #   UNROLL=4                              344.3 us  <- 1.27x, 7.0 TB/s
        #   UNROLL=8                              355.9 us  (56 regs, spills to
        #                                         fewer resident programs)
        unroll = _pick_unroll(n1, block_n2, cap=4, elem_budget=2048)
        num_programs = _num_programs(n1, unroll)

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
            UNROLL=unroll,
            num_warps=_num_warps(block_n2),
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
