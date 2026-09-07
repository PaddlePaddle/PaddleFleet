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
Triton fused grouped matmul for DSv4 output projection.

Replaces paddle.einsum("...gd,grd->...gr", x, w) with a single Triton kernel
that runs on bf16 Tensor Core (no f32 upcast, no internal transpose).

Forward:  out[g, m, :] = x[g, m, :] @ w[g, :, :]^T   for each group g
Backward: dx[g, m, :] = dy[g, m, :] @ w[g, :, :]      for each group g
          dw[g, r, :] = dy[g, :, r]^T @ x[g, :, :]     for each group g
"""

from functools import partial

import paddle

from .utils import enable_compat_on_triton_kernel, is_torch_compat_available

if is_torch_compat_available():
    paddle.enable_compat(scope={"triton"})

import triton
import triton.language as tl

# dtypes the fused Triton kernel is allowed to handle. Any other dtype (e.g.
# fp32) falls back to paddle.einsum in fused_grouped_matmul so it is not
# silently downcast to fp16.
_SUPPORTED_DTYPES = (paddle.float16, paddle.bfloat16)


@enable_compat_on_triton_kernel
@triton.jit
def grouped_matmul_fwd_kernel(
    X_ptr,  # [G, M, K]
    W_ptr,  # [G, N, K]  (stored as [G, R, D], we compute X @ W^T)
    Out_ptr,  # [G, M, N]
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_xg,
    stride_xm,
    stride_xk,
    stride_wg,
    stride_wn,
    stride_wk,
    stride_og,
    stride_om,
    stride_on,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr = 128,
    BLOCK_N: tl.constexpr = 128,
    BLOCK_K: tl.constexpr = 64,
):
    """Grouped matmul forward: out[g,m,n] = sum_k x[g,m,k] * w[g,n,k]."""
    pid_g = tl.program_id(0)
    pid_mn = tl.program_id(1)

    # Decode tile indices
    num_n_blocks = tl.cdiv(N, BLOCK_N)
    pid_m = pid_mn // num_n_blocks
    pid_n = pid_mn % num_n_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Base pointers for this group
    x_base = X_ptr + pid_g * stride_xg
    w_base = W_ptr + pid_g * stride_wg
    o_base = Out_ptr + pid_g * stride_og

    # Accumulator in f32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main loop over K dimension
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k

        # Load x tile [BLOCK_M, BLOCK_K]
        x_ptrs = (
            x_base + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        )
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Load w tile [BLOCK_N, BLOCK_K] (w is [G, N, K])
        w_ptrs = (
            w_base + offs_n[:, None] * stride_wn + k_offs[None, :] * stride_wk
        )
        w_mask = (offs_n[:, None] < N) & (k_offs[None, :] < K)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Matmul: [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N]
        acc += tl.dot(x_tile, tl.trans(w_tile))

    # Store output [BLOCK_M, BLOCK_N]
    out_ptrs = (
        o_base + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    )
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(OUT_DTYPE), mask=out_mask)


@enable_compat_on_triton_kernel
@triton.jit
def grouped_matmul_dx_kernel(
    DY_ptr,  # [G, M, R]
    W_ptr,  # [G, R, K]
    DX_ptr,  # [G, M, K]
    M,
    N: tl.constexpr,  # K (output dim of dx)
    K: tl.constexpr,  # R (reduction dim)
    stride_dyg,
    stride_dym,
    stride_dyk,
    stride_wg,
    stride_wn,
    stride_wk,
    stride_dxg,
    stride_dxm,
    stride_dxn,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr = 128,
    BLOCK_N: tl.constexpr = 128,
    BLOCK_K: tl.constexpr = 64,
):
    """Backward dx: dx[g,m,d] = sum_r dy[g,m,r] * w[g,r,d]."""
    pid_g = tl.program_id(0)
    pid_mn = tl.program_id(1)

    num_n_blocks = tl.cdiv(N, BLOCK_N)
    pid_m = pid_mn // num_n_blocks
    pid_n = pid_mn % num_n_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    dy_base = DY_ptr + pid_g * stride_dyg
    w_base = W_ptr + pid_g * stride_wg
    dx_base = DX_ptr + pid_g * stride_dxg

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # dx = dy @ w: dy[M,R] @ w[R,D] -> [M,D]
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k

        # Load dy tile [BLOCK_M, BLOCK_K] where K=R
        dy_ptrs = (
            dy_base
            + offs_m[:, None] * stride_dym
            + k_offs[None, :] * stride_dyk
        )
        dy_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        dy_tile = tl.load(dy_ptrs, mask=dy_mask, other=0.0)

        # Load w tile [BLOCK_K, BLOCK_N] = w[R, D]
        # w is stored as [G, R, D], so w[g, k_offs, offs_n]
        w_ptrs = (
            w_base + k_offs[:, None] * stride_wn + offs_n[None, :] * stride_wk
        )
        w_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc += tl.dot(dy_tile, w_tile)

    dx_ptrs = (
        dx_base + offs_m[:, None] * stride_dxm + offs_n[None, :] * stride_dxn
    )
    dx_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(dx_ptrs, acc.to(OUT_DTYPE), mask=dx_mask)


@enable_compat_on_triton_kernel
@triton.jit
def grouped_matmul_dw_kernel(
    DY_ptr,  # [G, M, R]  -- transposed access as [G, R, M]
    X_ptr,  # [G, M, D]
    DW_ptr,  # [G, R, D]
    M,  # reduction dim (b*sq), dynamic
    N: tl.constexpr,  # D
    K: tl.constexpr,  # R (rows of output dw)
    stride_dyg,
    stride_dym,
    stride_dyr,
    stride_xg,
    stride_xm,
    stride_xd,
    stride_dwg,
    stride_dwr,
    stride_dwd,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr = 128,  # tile over R
    BLOCK_N: tl.constexpr = 128,  # tile over D
    BLOCK_K: tl.constexpr = 64,  # tile over M (reduction)
):
    """Backward dw: dw[g,r,d] = sum_m dy[g,m,r] * x[g,m,d]."""
    pid_g = tl.program_id(0)
    pid_mn = tl.program_id(1)

    num_n_blocks = tl.cdiv(N, BLOCK_N)
    pid_m = pid_mn // num_n_blocks
    pid_n = pid_mn % num_n_blocks

    offs_r = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # R dim
    offs_d = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # D dim
    offs_k = tl.arange(0, BLOCK_K)  # M reduction dim

    dy_base = DY_ptr + pid_g * stride_dyg
    x_base = X_ptr + pid_g * stride_xg
    dw_base = DW_ptr + pid_g * stride_dwg

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # dw = dy^T @ x: dy[R,M]^T-access @ x[M,D] -> [R, D]
    for k_start in range(0, M, BLOCK_K):
        k_offs = k_start + offs_k

        # Load dy^T tile [BLOCK_M(R), BLOCK_K(M)] = dy[m, r] transposed
        dy_ptrs = (
            dy_base
            + k_offs[None, :] * stride_dym
            + offs_r[:, None] * stride_dyr
        )
        dy_mask = (k_offs[None, :] < M) & (offs_r[:, None] < K)
        dy_tile = tl.load(dy_ptrs, mask=dy_mask, other=0.0)

        # Load x tile [BLOCK_K(M), BLOCK_N(D)]
        x_ptrs = (
            x_base + k_offs[:, None] * stride_xm + offs_d[None, :] * stride_xd
        )
        x_mask = (k_offs[:, None] < M) & (offs_d[None, :] < N)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        acc += tl.dot(dy_tile, x_tile)

    dw_ptrs = (
        dw_base + offs_r[:, None] * stride_dwr + offs_d[None, :] * stride_dwd
    )
    dw_mask = (offs_r[:, None] < K) & (offs_d[None, :] < N)
    tl.store(dw_ptrs, acc.to(OUT_DTYPE), mask=dw_mask)


def _grouped_3d_strides(t):
    """Describe ``t`` of shape ``[..., G, X]`` as ``[M, G, X]`` without a copy.

    Returns ``(stride_m, stride_g, stride_x)`` when the tensor is a plain
    non-overlapping strided view whose ``[M, G, X]`` fold is exact, else
    ``None`` -- in which case the caller materializes a dense copy exactly as
    the original implementation did.

    A strided view such as ``q[..., :qk_nope_head_dim]`` qualifies: its group
    stride is the *full* head dim rather than ``X``. Since the kernels take every
    stride explicitly they can read such a view in place, and the
    ``.contiguous()`` copy it would otherwise trigger is pure waste -- an
    expensive one, since phi's strided->dense kernel has no vectorized variant.

    The checks are deliberately conservative: only layouts that a dense copy
    would linearize element-for-element are accepted, so the kernels see the
    same values in the same order and the result is bit-identical to the copy
    path. Broadcast (stride 0), reversed (negative stride) and self-overlapping
    views are all rejected rather than reasoned about.
    """
    shape = t.shape
    strides = t.strides
    if len(shape) < 3:
        return None
    stride_m, stride_g, stride_x = strides[-3], strides[-2], strides[-1]
    # Innermost dim must be packed, and neither the group nor the row step may
    # be degenerate (0 = broadcast, < 0 = reversed) or self-overlapping.
    if stride_x != 1:
        return None
    if stride_g < shape[-1] or stride_m < shape[-2] * stride_g:
        return None
    # Leading dims collapse into M only if each is packed against the *span* of
    # everything inside it. ``span`` is that extent, grown right-to-left.
    #
    # A size-1 dim contributes nothing to M, so its own stride is irrelevant and
    # its span is passed through untouched. Comparing against
    # ``shape[i + 1] * strides[i + 1]`` instead would make a singleton neighbour
    # satisfy any outer stride -- shape (2, 1, 3, 4, 6) with strides
    # (100, 100, 24, 6, 1) would be accepted although the six M rows start at
    # 0, 24, 48, 100, 124, 148 rather than at multiples of ``stride_m``.
    span = shape[-3] * stride_m
    for i in range(len(shape) - 4, -1, -1):
        if shape[i] == 1:
            continue
        if strides[i] != span:
            return None
        span *= shape[i]
    return stride_m, stride_g, stride_x


def _launch_grouped_dw(
    dy_3d,
    x_3d,
    M,
    G,
    R,
    D,
    stride_dyg,
    stride_dym,
    stride_dyk,
    stride_xg,
    stride_xm,
    stride_xk,
):
    """dw: [G, R, D] = dy^T[G, R, M] @ x[G, M, D] via strides.

    Split out of ``GroupedMatmulTriton.backward`` so the same launch can either
    run inline or be handed to a caller as a thunk to run later.
    """
    dw = paddle.empty([G, R, D], dtype=dy_3d.dtype)
    grid_dw = lambda META: (
        G,
        triton.cdiv(R, META["BLOCK_M"]) * triton.cdiv(D, META["BLOCK_N"]),
    )
    grouped_matmul_dw_kernel[grid_dw](
        dy_3d,
        x_3d,
        dw,
        M,
        D,  # N = D (output cols)
        R,  # K = R (rows of dw, which is the R dim)
        stride_dyg,
        stride_dym,
        stride_dyk,  # stride_dyr
        stride_xg,
        stride_xm,
        stride_xk,  # stride_xd
        dw.stride()[0],
        dw.stride()[1],
        dw.stride()[2],
        tl.bfloat16 if dy_3d.dtype == paddle.bfloat16 else tl.float16,
    )
    return dw


class GroupedMatmulTriton(paddle.autograd.PyLayer):
    """Fused grouped matmul: einsum("...gd,grd->...gr") via Triton.

    Input x: [..., G, D]  (arbitrary leading dims, flattened to [M, G, D])
    Weight w: [G, R, D]
    Output: [..., G, R]

    Key optimization: NO transpose. The kernel reads [M, G, D] directly via
    stride tricks — stride_xg and stride_xm are swapped so the kernel sees
    the data as [G, M, D] without any physical data movement.
    """

    @staticmethod
    def forward(ctx, x, w, dw_accumulator=None, group_shape=None):
        # PyLayer.forward runs with grad tracking off, so a tensor created here
        # comes out with stop_gradient=True. Read the caller's intent from the
        # inputs *before* any reshape, or w_needs_grad below silently turns
        # False and the weight grad is dropped entirely.
        w_needs_grad = not w.stop_gradient
        if group_shape is not None:
            # w is the leaf parameter; view it as [G, R, D] here so the
            # deferred path can legally return None for its grad.
            assert dw_accumulator is not None, (
                "group_shape is only for the deferred-dW path"
            )
            w = w.reshape(group_shape)
        orig_shape = x.shape  # [..., G, D]
        G = w.shape[0]
        R = w.shape[1]
        D = w.shape[2]
        M = 1
        for s in orig_shape[:-2]:
            M *= s

        # x is [..., G, D] -> [M, G, D]. A foldable unit-stride view is read in
        # place; anything else is materialized as a dense copy as before.
        x_strides = _grouped_3d_strides(x)
        if x_strides is None:
            x_3d = x.reshape([M, G, D]).contiguous()
            x_strides = (x_3d.strides[0], x_3d.strides[1], x_3d.strides[2])
        else:
            x_3d = x
        stride_xm, stride_xg, stride_xk = x_strides

        # Output in [M, G, R] layout directly (no transpose after kernel)
        out = paddle.empty([M, G, R], dtype=x.dtype)

        grid = lambda META: (
            G,
            triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(R, META["BLOCK_N"]),
        )

        # Pass strides so kernel sees [G, M, D] without physical transpose:
        # kernel expects: stride_xg (group stride), stride_xm (row stride), stride_xk (col stride)
        # x_3d is [M, G, D] with strides: (G*D, D, 1)
        #   stride_xg = D (= x_3d.stride()[1]), stride_xm = G*D (= x_3d.stride()[0]), stride_xk = 1
        # out is [M, G, R] with strides: (G*R, R, 1)
        #   stride_og = R (= out.stride()[1]), stride_om = G*R (= out.stride()[0]), stride_on = 1
        grouped_matmul_fwd_kernel[grid](
            x_3d,
            w,
            out,
            M,
            R,
            D,
            stride_xg,  # stride_xg: step between groups
            stride_xm,  # stride_xm: step between rows (M dim)
            stride_xk,  # stride_xk: step between cols (D dim)
            w.stride()[0],
            w.stride()[1],
            w.stride()[2],
            out.stride()[1],  # stride_og: step between groups
            out.stride()[0],  # stride_om: step between rows (M dim)
            out.stride()[2],  # stride_on: step between cols (R dim)
            tl.bfloat16 if x.dtype == paddle.bfloat16 else tl.float16,
        )

        ctx.save_for_backward(x_3d, w)
        ctx.x_strides = x_strides
        ctx.M = M
        ctx.G = G
        ctx.R = R
        ctx.D = D
        ctx.orig_shape = orig_shape
        # Paddle PyLayer requires None for a stop_gradient input; record it here so
        # a frozen backbone (DSv4 phase 2, ``train_indexer_only``) also skips
        # the matching kernel instead of violating the contract.
        ctx.x_needs_grad = not x.stop_gradient
        ctx.w_needs_grad = w_needs_grad
        ctx.dw_accumulator = dw_accumulator

        # out is already [M, G, R] -> reshape to [..., G, R]
        out = out.reshape([*orig_shape[:-1], R])
        return out

    @staticmethod
    def backward(ctx, dy):
        x_3d, w = ctx.saved_tensor()
        M, G, R, D = ctx.M, ctx.G, ctx.R, ctx.D
        orig_shape = ctx.orig_shape

        # No "both frozen" shortcut needed: a PyLayer output is differentiable
        # only if some input is, so backward is never entered in that case.
        # dy: [..., G, R] -> [M, G, R]. Same in-place-view rule as the forward.
        dy_strides = _grouped_3d_strides(dy)
        if dy_strides is None:
            dy_3d = dy.reshape([M, G, R]).contiguous()
            dy_strides = (dy_3d.strides[0], dy_3d.strides[1], dy_3d.strides[2])
        else:
            dy_3d = dy
        stride_dym, stride_dyg, stride_dyk = dy_strides
        stride_xm, stride_xg, stride_xk = ctx.x_strides

        dx = None
        if ctx.x_needs_grad:
            # dx: [M, G, D] — kernel computes dy[G,M,R] @ w[G,R,D] via strides
            dx = paddle.empty([M, G, D], dtype=dy.dtype)
            grid_dx = lambda META: (
                G,
                triton.cdiv(M, META["BLOCK_M"])
                * triton.cdiv(D, META["BLOCK_N"]),
            )
            grouped_matmul_dx_kernel[grid_dx](
                dy_3d,
                w,
                dx,
                M,
                D,  # N = D (output cols)
                R,  # K = R (reduction)
                stride_dyg,  # stride_dyg
                stride_dym,  # stride_dym
                stride_dyk,  # stride_dyk
                w.stride()[0],
                w.stride()[1],
                w.stride()[2],
                dx.stride()[1],  # stride_dxg
                dx.stride()[0],  # stride_dxm
                dx.stride()[2],  # stride_dxn
                tl.bfloat16 if dy.dtype == paddle.bfloat16 else tl.float16,
            )
            # dx is already [M, G, D] -> reshape to [..., G, D]
            dx = dx.reshape(orig_shape)

        dw = None
        if ctx.w_needs_grad:
            compute_dw = partial(
                _launch_grouped_dw,
                dy_3d,
                x_3d,
                M,
                G,
                R,
                D,
                stride_dyg,
                stride_dym,
                stride_dyk,
                stride_xg,
                stride_xm,
                stride_xk,
            )
            if ctx.dw_accumulator is not None:
                # Hand the thunk to the caller and return no weight grad: the
                # accumulator owns both when it runs and where it lands. Used to
                # push dW into a pp p2p window (see dw_overlap.py).
                ctx.dw_accumulator(compute_dw)
            else:
                dw = compute_dw()

        return dx, dw


def fused_grouped_matmul(x, w, dw_accumulator=None, group_shape=None):
    """Fused grouped matmul replacing paddle.einsum("...gd,grd->...gr", x, w).

    Args:
        x: Input tensor [..., G, D] (typically [b, sq, G, D])
        w: Weight tensor [G, R, D]

        dw_accumulator: optional callable taking a zero-arg thunk that
            produces dw [G, R, D]. When given, the weight grad is not
            returned to autograd -- the accumulator owns when it runs and
            where it lands (see dw_overlap.deferred_grouped_dw_accumulator).
            Ignored on the einsum fallback path.

    Returns:
        Output tensor [..., G, R]
    """
    if x.dtype != w.dtype:
        raise ValueError(
            f"x and w must have the same dtype, got x={x.dtype}, w={w.dtype}"
        )
    # ``G`` and ``D`` reach the kernel from ``w`` while ``x`` is read through
    # its own strides, so a mismatching ``x`` would be read past its group (or
    # out of bounds) instead of failing. The dense path used to get this check
    # for free from ``x.reshape([M, G, D])``; the zero-copy view path never
    # reshapes, so state it here for both.
    if group_shape is not None:
        # deferred-dW path: w is the 2-D leaf parameter, viewed as [G, R, D]
        # inside the PyLayer. Validate against the view, not the parameter.
        if len(group_shape) != 3:
            raise ValueError(
                f"group_shape must be [G, R, D], got {group_shape}"
            )
        w_view_shape = list(group_shape)
    elif w.ndim != 3:
        raise ValueError(f"w must be 3-D [G, R, D], got shape {w.shape}")
    else:
        w_view_shape = list(w.shape)
    if x.ndim < 2 or tuple(x.shape[-2:]) != (w_view_shape[0], w_view_shape[2]):
        raise ValueError(
            "x's trailing dims must match w's [G, D]: expected "
            f"{(w_view_shape[0], w_view_shape[2])}, got x shape {x.shape} for "
            f"w shape {w.shape}"
        )
    if x.dtype not in _SUPPORTED_DTYPES:
        # fp32 (and any other unsupported dtype) stays on paddle.einsum to
        # preserve numerical equivalence instead of being downcast to fp16.
        return paddle.einsum(
            "...gd,grd->...gr",
            x,
            w.reshape(w_view_shape) if group_shape is not None else w,
        )
    return GroupedMatmulTriton.apply(x, w, dw_accumulator, group_shape)
