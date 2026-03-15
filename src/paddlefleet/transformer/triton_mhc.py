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
Triton-accelerated Kernels for MHC (Manifold Constrained HyperConnections).

This module provides high-performance Triton kernel implementations for:
1. Fused RMSNorm + GEMM + Split + Scale/Bias for Width Connection
2. Depth connection with optimized memory access
3. Post-Sinkhorn computation (residuals + branch_input)

Key optimizations:
- Aggressive kernel fusion to minimize global memory traffic
- Improved memory access patterns for better coalescing
- Vectorized loads/stores where possible
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

paddle.enable_compat(scope={"triton"})

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    triton = None
    tl = None

# sinkhorn_knopp is defined locally in this file


# ============================================================================
# Compact Sinkhorn-Knopp (no diag_embed optimization)
# ============================================================================

@triton.jit
def sinkhorn_4x4_compact_kernel(
    A_ptr,
    u_ptr,
    v_ptr,
    batch_size,
    stride_batch_A,
    stride_n_A,
    stride_batch_uv,
    stride_n_uv,
    iters: tl.constexpr,
    eps: tl.constexpr,
):
    """
    Same kernel as sinkhorn_kernel_static_4x4, but returns vectors directly.
    """
    pid_batch = tl.program_id(0)

    # Load the 4x4 matrix for this batch
    A_offset = pid_batch * stride_batch_A
    A00 = tl.load(A_ptr + A_offset + 0 * stride_n_A + 0)
    A01 = tl.load(A_ptr + A_offset + 0 * stride_n_A + 1)
    A02 = tl.load(A_ptr + A_offset + 0 * stride_n_A + 2)
    A03 = tl.load(A_ptr + A_offset + 0 * stride_n_A + 3)

    A10 = tl.load(A_ptr + A_offset + 1 * stride_n_A + 0)
    A11 = tl.load(A_ptr + A_offset + 1 * stride_n_A + 1)
    A12 = tl.load(A_ptr + A_offset + 1 * stride_n_A + 2)
    A13 = tl.load(A_ptr + A_offset + 1 * stride_n_A + 3)

    A20 = tl.load(A_ptr + A_offset + 2 * stride_n_A + 0)
    A21 = tl.load(A_ptr + A_offset + 2 * stride_n_A + 1)
    A22 = tl.load(A_ptr + A_offset + 2 * stride_n_A + 2)
    A23 = tl.load(A_ptr + A_offset + 2 * stride_n_A + 3)

    A30 = tl.load(A_ptr + A_offset + 3 * stride_n_A + 0)
    A31 = tl.load(A_ptr + A_offset + 3 * stride_n_A + 1)
    A32 = tl.load(A_ptr + A_offset + 3 * stride_n_A + 2)
    A33 = tl.load(A_ptr + A_offset + 3 * stride_n_A + 3)

    # Initialize u and v
    u0, u1, u2, u3 = 1.0, 1.0, 1.0, 1.0
    v0, v1, v2, v3 = 1.0, 1.0, 1.0, 1.0

    # Unrolled iterations for maximum performance
    for _ in range(iters):
        # Update u: u = 1.0 / (A @ v + eps)
        Av0 = A00*v0 + A01*v1 + A02*v2 + A03*v3 + eps
        Av1 = A10*v0 + A11*v1 + A12*v2 + A13*v3 + eps
        Av2 = A20*v0 + A21*v1 + A22*v2 + A23*v3 + eps
        Av3 = A30*v0 + A31*v1 + A32*v2 + A33*v3 + eps

        u0 = 1.0 / Av0
        u1 = 1.0 / Av1
        u2 = 1.0 / Av2
        u3 = 1.0 / Av3

        # Update v: v = 1.0 / (A^T @ u + eps)
        At_u0 = A00*u0 + A10*u1 + A20*u2 + A30*u3 + eps
        At_u1 = A01*u0 + A11*u1 + A21*u2 + A31*u3 + eps
        At_u2 = A02*u0 + A12*u1 + A22*u2 + A32*u3 + eps
        At_u3 = A03*u0 + A13*u1 + A23*u2 + A33*u3 + eps

        v0 = 1.0 / At_u0
        v1 = 1.0 / At_u1
        v2 = 1.0 / At_u2
        v3 = 1.0 / At_u3

    # Store results
    u_offset = pid_batch * stride_batch_uv
    tl.store(u_ptr + u_offset + 0 * stride_n_uv, u0)
    tl.store(u_ptr + u_offset + 1 * stride_n_uv, u1)
    tl.store(u_ptr + u_offset + 2 * stride_n_uv, u2)
    tl.store(u_ptr + u_offset + 3 * stride_n_uv, u3)

    v_offset = pid_batch * stride_batch_uv
    tl.store(v_ptr + v_offset + 0 * stride_n_uv, v0)
    tl.store(v_ptr + v_offset + 1 * stride_n_uv, v1)
    tl.store(v_ptr + v_offset + 2 * stride_n_uv, v2)
    tl.store(v_ptr + v_offset + 3 * stride_n_uv, v3)


def triton_sinkhorn_knopp_compact(A, it=20, eps=1e-8):
    """
    Compact Sinkhorn-Knopp that returns vectors (no diag_embed).

    This eliminates the extra kernel launch from diag_embed and reduces memory usage.

    Args:
        A: Input matrix of shape [B, L, N, N] or [batch_size, n, n], should be float32
        it: Number of iterations
        eps: Small epsilon for numerical stability

    Returns:
        _: Normalized matrix (not used, kept for compatibility)
        u: Left scaling vector [B*L, n] or [batch_size, n]
        v: Right scaling vector [B*L, n] or [batch_size, n]

    Note:
        For diagonal matrix operations, use element-wise multiplication:
        - U @ H @ V becomes: u[:, None] * H * v[None, :]
        No reshape operation is performed internally to reduce overhead.
    """
    B, L, n, _ = A.shape
    batch_size = B * L

    # Use optimized kernel for N=4, fallback to original for other sizes
    if n == 4:
        # Allocate output tensors (compact vector format)
        u = paddle.empty([batch_size, n], dtype=paddle.float32)
        v = paddle.empty([batch_size, n], dtype=paddle.float32)

        # Get strides
        stride_batch_A = n*n
        stride_n_A = n
        stride_batch_uv = u.stride(0)
        stride_n_uv = u.stride(1)

        # Launch optimized kernel
        grid = (batch_size,)
        sinkhorn_4x4_compact_kernel[grid](
            A,
            u,
            v,
            batch_size,
            stride_batch_A,
            stride_n_A,
            stride_batch_uv,
            stride_n_uv,
            iters=it,
            eps=eps,
            num_warps=1,
        )
        u.stop_gradient = True
        v.stop_gradient = True
        return u, v
    else:
        # Fallback: use original sinkhorn and extract diagonal elements
        _, U, V = sinkhorn_knopp(A.reshape([B*L, n, n]), it=it, eps=eps)
        # Extract diagonal from U and V: U[i,i] and V[i,i]
        # For diagonal matrix, only diagonal elements are non-zero
        u = paddle.diagonal(U, axis1=1, axis2=2)  # [batch_size, n]
        v = paddle.diagonal(V, axis1=1, axis2=2)
        u.stop_gradient = True
        v.stop_gradient = True  # [batch_size, n]
        return u, v


def sinkhorn_knopp(A, it=20, eps=1e-8):
    """
    Sinkhorn-Knopp algorithm for doubly stochastic matrix normalization.

    Args:
        A: Input non-negative matrix (already exp applied), shape [batch_size, n, n]
        it: Number of iterations
        eps: Small epsilon for numerical stability

    Returns:
        P: Normalized matrix (not used, kept for compatibility)
        U: Left scaling diagonal matrix [batch_size, n, n]
        V: Right scaling diagonal matrix [batch_size, n, n]
    """
    batch_size, n, _ = A.shape

    # Use high-performance Triton implementation when n=4
    if n == 4:
        # Allocate output tensors
        u = paddle.empty([batch_size, n], dtype=paddle.float32)
        v = paddle.empty([batch_size, n], dtype=paddle.float32)

        # Get strides
        stride_batch_A = A.stride(0)
        stride_n_A = A.stride(1)
        stride_batch_uv = u.stride(0)
        stride_n_uv = u.stride(1)

        # Launch optimized kernel
        grid = (batch_size,)
        sinkhorn_4x4_compact_kernel[grid](
            A,
            u,
            v,
            batch_size,
            stride_batch_A,
            stride_n_A,
            stride_batch_uv,
            stride_n_uv,
            iters=it,
            eps=eps,
            num_warps=1,
        )

        # Build diagonal matrices
        U = paddle.diag_embed(u)
        V = paddle.diag_embed(v)
        return _, U, V

    # Fallback to original Python implementation
    # Initialize u and v with same dtype as A
    u = paddle.ones([batch_size, n], dtype=A.dtype)
    v = paddle.ones([batch_size, n], dtype=A.dtype)

    for _ in range(it):
        # 1. Update u
        v_temp = v.unsqueeze(2)  # (B, n, 1)
        Av = paddle.matmul(A, v_temp).squeeze(2)  # (B, n)
        u = 1.0 / (Av + eps)

        # 2. Update v
        u_temp = u.unsqueeze(2)  # (B, n, 1)
        At = paddle.transpose(A, perm=[0, 2, 1])  # (B, n, n)
        At_u = paddle.matmul(At, u_temp).squeeze(2)  # (B, n)
        v = 1.0 / (At_u + eps)

    # Build diagonal matrices
    # paddle.diag_embed converts vectors to diagonal matrices (B, n) -> (B, n, n)
    if u.dtype == paddle.bfloat16:
        u = u.cast('float32')
        v = v.cast('float32')
        U = paddle.diag_embed(u)
        V = paddle.diag_embed(v)
        U = U.cast('bfloat16')
        V = V.cast('bfloat16')
    else:
        U = paddle.diag_embed(u)
        V = paddle.diag_embed(v)

    return _, U, V


@triton.jit
def exp_matmul_residuals_fused_kernel(
    # Inputs
    H_res_exp_ptr,      # [B*L, N, N]
    u_ptr,              # [B*L, N] - compact vector (diagonal of U)
    v_ptr,              # [B*L, N] - compact vector (diagonal of V)
    H_pre_ptr,          # [B*L, N]
    x_ptr,              # [B*L, N, D]
    # Outputs
    H_res_out_ptr,      # [B*L, N, N] - float32
    residuals_ptr,      # [B*L, N, D]
    branch_input_ptr,   # [B*L, D]
    # Parameters
    num_tokens: tl.constexpr,  # B*L
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
):
    """
    Fused kernel: exp_double_matmul + residuals + branch_input (N=4 specialized).

    1D grid: (num_tokens,)
      - each program handles one token, loops over D internally

    Optimizations:
      1. u/v/H_res_exp scalars loaded once per token, reused across all D-blocks
      2. H_res 4x4 computed once, stays in registers for entire D-loop
      3. H_res_out written once unconditionally (no WAW concern with 1D grid)
      4. x[token, 0..3, d_block] loaded once per D-block, shared across
         4 residual rows AND branch_input
    """
    token_idx = tl.program_id(0)

    if token_idx >= num_tokens:
        return

    stride_nn = N * N
    stride_nd = N * D

    # ── Step 1: Compute H_res[4,4] + H_pre[4] as scalars — ONCE per token ──
    uv_base = token_idx * N
    u_0 = tl.load(u_ptr + uv_base + 0).to(tl.float32)
    u_1 = tl.load(u_ptr + uv_base + 1).to(tl.float32)
    u_2 = tl.load(u_ptr + uv_base + 2).to(tl.float32)
    u_3 = tl.load(u_ptr + uv_base + 3).to(tl.float32)
    v_0 = tl.load(v_ptr + uv_base + 0).to(tl.float32)
    v_1 = tl.load(v_ptr + uv_base + 1).to(tl.float32)
    v_2 = tl.load(v_ptr + uv_base + 2).to(tl.float32)
    v_3 = tl.load(v_ptr + uv_base + 3).to(tl.float32)

    he_base = token_idx * stride_nn
    hr00 = u_0 * tl.load(H_res_exp_ptr + he_base + 0).to(tl.float32) * v_0
    hr01 = u_0 * tl.load(H_res_exp_ptr + he_base + 1).to(tl.float32) * v_1
    hr02 = u_0 * tl.load(H_res_exp_ptr + he_base + 2).to(tl.float32) * v_2
    hr03 = u_0 * tl.load(H_res_exp_ptr + he_base + 3).to(tl.float32) * v_3
    hr10 = u_1 * tl.load(H_res_exp_ptr + he_base + 4).to(tl.float32) * v_0
    hr11 = u_1 * tl.load(H_res_exp_ptr + he_base + 5).to(tl.float32) * v_1
    hr12 = u_1 * tl.load(H_res_exp_ptr + he_base + 6).to(tl.float32) * v_2
    hr13 = u_1 * tl.load(H_res_exp_ptr + he_base + 7).to(tl.float32) * v_3
    hr20 = u_2 * tl.load(H_res_exp_ptr + he_base + 8).to(tl.float32) * v_0
    hr21 = u_2 * tl.load(H_res_exp_ptr + he_base + 9).to(tl.float32) * v_1
    hr22 = u_2 * tl.load(H_res_exp_ptr + he_base + 10).to(tl.float32) * v_2
    hr23 = u_2 * tl.load(H_res_exp_ptr + he_base + 11).to(tl.float32) * v_3
    hr30 = u_3 * tl.load(H_res_exp_ptr + he_base + 12).to(tl.float32) * v_0
    hr31 = u_3 * tl.load(H_res_exp_ptr + he_base + 13).to(tl.float32) * v_1
    hr32 = u_3 * tl.load(H_res_exp_ptr + he_base + 14).to(tl.float32) * v_2
    hr33 = u_3 * tl.load(H_res_exp_ptr + he_base + 15).to(tl.float32) * v_3

    # ── Step 2: Write H_res_out — once, no branch ────────────────────────
    tl.store(H_res_out_ptr + he_base + 0, hr00)
    tl.store(H_res_out_ptr + he_base + 1, hr01)
    tl.store(H_res_out_ptr + he_base + 2, hr02)
    tl.store(H_res_out_ptr + he_base + 3, hr03)
    tl.store(H_res_out_ptr + he_base + 4, hr10)
    tl.store(H_res_out_ptr + he_base + 5, hr11)
    tl.store(H_res_out_ptr + he_base + 6, hr12)
    tl.store(H_res_out_ptr + he_base + 7, hr13)
    tl.store(H_res_out_ptr + he_base + 8, hr20)
    tl.store(H_res_out_ptr + he_base + 9, hr21)
    tl.store(H_res_out_ptr + he_base + 10, hr22)
    tl.store(H_res_out_ptr + he_base + 11, hr23)
    tl.store(H_res_out_ptr + he_base + 12, hr30)
    tl.store(H_res_out_ptr + he_base + 13, hr31)
    tl.store(H_res_out_ptr + he_base + 14, hr32)
    tl.store(H_res_out_ptr + he_base + 15, hr33)

    # Load H_pre scalars once (used in every D-block iteration)
    hp_0 = tl.load(H_pre_ptr + uv_base + 0).to(tl.float32)
    hp_1 = tl.load(H_pre_ptr + uv_base + 1).to(tl.float32)
    hp_2 = tl.load(H_pre_ptr + uv_base + 2).to(tl.float32)
    hp_3 = tl.load(H_pre_ptr + uv_base + 3).to(tl.float32)

    # ── Step 3: D-loop — residuals + branch_input ─────────────────────────
    base_x = token_idx * stride_nd
    residuals_base = token_idx * stride_nd
    branch_base = token_idx * D

    for d_start in range(0, D, BLOCK_SIZE_D):
        d_offsets = d_start + tl.arange(0, BLOCK_SIZE_D)
        d_mask = d_offsets < D

        # Load x[token, 0..3, d_block] once, reuse for residuals + branch
        x_0 = tl.load(x_ptr + base_x + 0 * D + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
        x_1 = tl.load(x_ptr + base_x + 1 * D + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
        x_2 = tl.load(x_ptr + base_x + 2 * D + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
        x_3 = tl.load(x_ptr + base_x + 3 * D + d_offsets, mask=d_mask, other=0.0).to(tl.float32)

        # Residuals: residuals[token, n, d] = sum_k H_res[n, k] * x[k, d]
        res_0 = hr00 * x_0 + hr01 * x_1 + hr02 * x_2 + hr03 * x_3
        res_1 = hr10 * x_0 + hr11 * x_1 + hr12 * x_2 + hr13 * x_3
        res_2 = hr20 * x_0 + hr21 * x_1 + hr22 * x_2 + hr23 * x_3
        res_3 = hr30 * x_0 + hr31 * x_1 + hr32 * x_2 + hr33 * x_3

        tl.store(residuals_ptr + residuals_base + 0 * D + d_offsets, res_0.to(OUTPUT_DTYPE), mask=d_mask)
        tl.store(residuals_ptr + residuals_base + 1 * D + d_offsets, res_1.to(OUTPUT_DTYPE), mask=d_mask)
        tl.store(residuals_ptr + residuals_base + 2 * D + d_offsets, res_2.to(OUTPUT_DTYPE), mask=d_mask)
        tl.store(residuals_ptr + residuals_base + 3 * D + d_offsets, res_3.to(OUTPUT_DTYPE), mask=d_mask)

        # Branch input: reuse x_0..x_3 already in registers
        branch_val = hp_0 * x_0 + hp_1 * x_1 + hp_2 * x_2 + hp_3 * x_3
        tl.store(branch_input_ptr + branch_base + d_offsets, branch_val.to(OUTPUT_DTYPE), mask=d_mask)


# ============================================================================
# Compact Backward Kernels
# ============================================================================

@triton.jit
def hres_bwd_exp_matmul_compact_kernel(
    # Inputs
    d_H_res_mat_ptr,       # [B*L, N, N]
    H_res_exp_ptr,         # [B*L, N, N]
    u_ptr,                 # [B*L, N] - compact vector
    v_ptr,                 # [B*L, N] - compact vector
    H_all_ptr,             # [B*L, N+N+N*N] - float32
    bias_terms_ptr,        # [N+N+N*N] - float32
    scaling_factors_ptr,   # [3] - float32

    # Outputs
    d_scaling_factors_ptr, # [3] - output for scale gradients (index 2: res_scale)
    d_H_all_ptr,           # [B*L, N+N+N*N] - write d_H_all[2N:] directly (bfloat16)
    d_bias_terms_ptr,      # [N+N+N*N] - atomic add d_bias_res at offset 2*N

    # Parameters
    num_tokens: tl.constexpr,  # B*L
    N: tl.constexpr,
    NN: tl.constexpr,
    N3: tl.constexpr,
    stride_d_H_res_mat,
    stride_H_res_exp,
    stride_u,               # stride for compact vectors
    stride_v,
    stride_H_all,
):
    """
    Kernel A: Fused H_res backward computation (compact version).

    Computes:
    1. d_H_res_exp = U^T @ d_H_res_mat @ V^T (simplified to element-wise with vectors)
    2. d_H_res = d_H_res_exp * H_res_exp
    3. d_H_res_scaled = d_H_res * res_scale -> write to d_H_all[2N:]
    4. d_res_scale = sum(d_H_res * H_all_raw_res)
    5. d_bias_res = d_H_res -> atomic_add to d_bias_terms[2N:]

    With compact vectors: d_H_res_exp[row, col] = u[row] * d_H_res_mat[row, col] * v[col]

    Grid: (num_tokens,) - each block handles one [N, N] matrix
    """
    pid = tl.program_id(0)

    if pid >= num_tokens:
        return

    # Load res_scale
    res_scale = tl.load(scaling_factors_ptr + 2)

    # Base offsets for current token
    d_H_res_mat_base = pid * stride_d_H_res_mat
    H_res_exp_base = pid * stride_H_res_exp
    H_all_base = pid * stride_H_all

    # Accumulator for d_res_scale
    d_res_scale_acc = tl.zeros([], dtype=tl.float32)

    # Process all NN elements element-wise
    for i in range(NN):
        # Convert flat index to (row, col)
        row = i // N
        col = i % N

        # d_H_res_exp[row, col] = u[row] * d_H_res_mat[row, col] * v[col]
        d_H_mat_rc = tl.load(d_H_res_mat_ptr + d_H_res_mat_base + row * N + col).to(tl.float32)
        u_val = tl.load(u_ptr + pid * stride_u + row).to(tl.float32)
        v_val = tl.load(v_ptr + pid * stride_v + col).to(tl.float32)

        d_H_res_exp_rc = u_val * d_H_mat_rc * v_val

        # Load H_res_exp[r, c]
        h_res_exp_rc = tl.load(H_res_exp_ptr + H_res_exp_base + row * N + col).to(tl.float32)

        # d_H_res = d_H_res_exp * H_res_exp
        d_H_res_rc = d_H_res_exp_rc * h_res_exp_rc

        # Load H_all_raw_res[row, col] (flattened)
        H_all_raw_res_rc = (tl.load(H_all_ptr + H_all_base + 2 * N + row * N + col).to(tl.float32) -
                             tl.load(bias_terms_ptr + 2 * N + row * N + col).to(tl.float32)) / res_scale

        # Accumulate d_res_scale
        d_res_scale_acc += d_H_res_rc * H_all_raw_res_rc

        # d_H_res_scaled -> write to d_H_all[2N + i] (values already in register)
        d_H_res_scaled_rc = d_H_res_rc * res_scale
        tl.store(d_H_all_ptr + H_all_base + 2 * N + i, d_H_res_scaled_rc.to(tl.bfloat16))

        # d_bias_res: d_H_res_rc already in register
        tl.atomic_add(d_bias_terms_ptr + 2 * N + i, d_H_res_rc)

    # Store d_res_scale using atomic add (index 2 in d_scaling_factors)
    tl.atomic_add(d_scaling_factors_ptr + 2, d_res_scale_acc)


# ============================================================================
# Optimized Fused Kernels for Width Connection
# ============================================================================

@triton.autotune(
    configs=[triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 128}, num_warps=4, num_stages=4)],
    key=['num_tokens', 'ND', 'N3'],
)
@triton.jit
def width_rmsnorm_gemm_kernel(
    x_ptr,
    combined_weights_ptr,
    scaling_factors_ptr,
    bias_terms_ptr,
    H_all_ptr,
    num_tokens,
    ND,
    N,
    N3,
    NN,
    stride_x_token,
    stride_cw_out,
    stride_h_all_token,   # stride for H_all output (N3 for 4D [B,L,N3])
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Optimized width connection forward kernel with RMSNorm + GEMM, outputs H_all.

    Note: This kernel uses norm_weight=1.0 (no learnable weight for RMSNorm).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    x_mask_m = rm < num_tokens
    x_offset_base = rm[:, None] * stride_x_token
    _var_acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)

    # RMSNorm: variance calculation
    for k_idx in range(0, ND, BLOCK_K):
        k_offsets = k_idx + rk
        mask_k = k_offsets < ND
        x_chunk = tl.load(x_ptr + x_offset_base + k_offsets[None, :],
                          mask=x_mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        _var_acc += x_chunk * x_chunk

    var_sum = tl.sum(_var_acc, axis=1)
    rstd = 1.0 / tl.sqrt((var_sum / ND) + eps)

    # GEMM loop with RMSNorm + weight loading + accumulation
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_idx in range(0, ND, BLOCK_K):
        k_offsets = k_idx + rk
        mask_k = k_offsets < ND
        x_chunk = tl.load(x_ptr + x_offset_base + k_offsets[None, :],
                          mask=x_mask_m[:, None] & mask_k[None, :], other=0.0)
        # norm_weight = 1.0 (no learnable weight for RMSNorm)
        x_normed = x_chunk * rstd[:, None]

        # Load weights: combined_weights[k_offsets, rn] for correct GEMM
        # combined_weights shape is [ND, N3], so element [k, j] = ptr + k * N3 + j
        w_ptr_base = combined_weights_ptr + (k_offsets[:, None] * stride_cw_out + rn[None, :])
        w_chunk = tl.load(w_ptr_base, mask=mask_k[:, None] & (rn[None, :] < N3), other=0.0).to(tl.bfloat16)
        # w_chunk shape: [BLOCK_K, BLOCK_N], no transpose needed for correct GEMM
        acc += tl.dot(x_normed.to(tl.bfloat16), w_chunk)

    # Load scaling factors
    scale_pre = tl.load(scaling_factors_ptr + 0)
    scale_post = tl.load(scaling_factors_ptr + 1)
    scale_res = tl.load(scaling_factors_ptr + 2)

    # Load bias terms for each output
    bias_pre_mask = rn < N
    bias_pre_vals = tl.load(bias_terms_ptr + rn, mask=bias_pre_mask, other=0.0)

    bias_post_mask = (rn >= N) & (rn < 2 * N)
    bias_post_vals = tl.load(bias_terms_ptr + rn, mask=bias_post_mask, other=0.0)

    bias_res_mask = (rn >= 2 * N) & (rn < N3)
    bias_res_vals = tl.load(bias_terms_ptr + rn, mask=bias_res_mask, other=0.0)

    # Store H_all with scaling and bias applied
    # H_all layout: [H_pre (N elements), H_post (N elements), H_res_flat (N*N elements)]
    # Use rn (which is global column index) for correct storage position
    # Store as bfloat16 to reduce memory bandwidth
    H_all_offset_base = rm[:, None] * stride_h_all_token + rn[None, :]

    # Store H_pre part (elements 0 to N-1)
    final_pre = acc * scale_pre + bias_pre_vals
    tl.store(H_all_ptr + H_all_offset_base, final_pre.to(tl.bfloat16),
             mask=x_mask_m[:, None] & bias_pre_mask[None, :])

    # Store H_post part (elements N to 2N-1)
    final_post = acc * scale_post + bias_post_vals
    tl.store(H_all_ptr + H_all_offset_base, final_post.to(tl.bfloat16),
             mask=x_mask_m[:, None] & bias_post_mask[None, :])

    # Store H_res part (elements 2N to N3-1)
    final_res = acc * scale_res + bias_res_vals
    tl.store(H_all_ptr + H_all_offset_base, final_res.to(tl.bfloat16),
             mask=x_mask_m[:, None] & bias_res_mask[None, :])


def width_rmsnorm_gemm_forward(x, combined_weights, scaling_factors, bias_terms, norm_eps, N, D, B, L):
    """
    Wrapper for fused RMSNorm + GEMM kernel.

    Args:
        x: [B, L, N, D]
        combined_weights: [N*D, N3]
        scaling_factors: [3]
        bias_terms: [N3]
        norm_eps: float
        N: int
        D: int
        B: int
        L: int

    Returns:
        H_all: [B, L, N3] - combined output (acc) from GEMM, before splitting

    Note:
        This function uses norm_weight=1.0 (no learnable weight for RMSNorm).
    """
    num_tokens = B * L
    NN = N * N
    N3 = N + N + NN
    ND = N * D

    H_all = paddle.empty([B, L, N3], dtype=paddle.bfloat16)

    # Use 2D grid with smallest BLOCK values from autotune configs to ensure coverage
    # Autotune configs use BLOCK_M in [16, 32, 64, 128] and BLOCK_N in [32, 64]
    # Using smallest values ensures grid is large enough for any autotune config
    grid_m = triton.cdiv(num_tokens, 16)  # Use smallest BLOCK_M=16 to ensure sufficient parallelism
    grid_n = triton.cdiv(N3, 32)          # Use smallest BLOCK_N=32 to increase N dimension coverage
    grid = (grid_m, grid_n)

    width_rmsnorm_gemm_kernel[grid](
        x,
        combined_weights,
        scaling_factors,
        bias_terms,
        H_all,
        num_tokens=num_tokens,
        ND=ND,
        N=N,
        N3=N3,
        NN=NN,
        stride_x_token=ND,
        stride_cw_out=N3,   # fixed: combined_weights shape is [ND, N3], row-major stride is N3
        stride_h_all_token=N3,   # for 4D [B, L, N3] C-contiguous
        eps=norm_eps,
    )

    return H_all


@triton.jit
def mhc_sigmoid_exp_kernel(
    # Pointers
    H_ptr,
    Out_Pre_ptr, Out_Post_ptr, Out_Res_Exp_ptr,
    # Stride info (optimized for token processing)
    stride_h_token, stride_h_dim,
    stride_out_pre_token, stride_out_pre_dim,
    stride_out_post_token, stride_out_post_dim,
    stride_out_res_token, stride_out_res_dim,
    n_tokens,
    real_N,
    # Compile-time constants
    BLOCK_N: tl.constexpr,      # Power of 2 covering N
    BLOCK_RES: tl.constexpr,    # Power of 2 covering N*N
    BLOCK_SIZE: tl.constexpr    # Number of tokens per program
):
    # 1. Determine current token batch
    pid = tl.program_id(0)
    offs_token = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_token = offs_token < n_tokens

    # 2. Prepare dimension offsets
    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < real_N

    offs_res = tl.arange(0, BLOCK_RES)
    real_N2 = real_N * real_N
    mask_res = offs_res < real_N2

    # -----------------------------------------------------------
    # 3. Compute and write Pre & Post (scale and bias already applied in H_all)
    # -----------------------------------------------------------
    base_h_ptr = H_ptr + (offs_token[:, None] * stride_h_token)

    # H_pre processing: direct sigmoid (scale and bias already applied in GEMM kernel)
    h_pre = tl.load(base_h_ptr + offs_n[None, :] * stride_h_dim,
                    mask=mask_token[:, None] & mask_n[None, :], other=0.0).to(tl.float32)
    out_pre = tl.sigmoid(h_pre)

    # H_post processing: direct sigmoid then multiply 2.0 (scale and bias already applied in GEMM kernel)
    h_post = tl.load(base_h_ptr + (real_N + offs_n[None, :]) * stride_h_dim,
                     mask=mask_token[:, None] & mask_n[None, :], other=0.0).to(tl.float32)
    out_post = 2.0 * tl.sigmoid(h_post)

    # Write results
    tl.store(Out_Pre_ptr + (offs_token[:, None] * stride_out_pre_token) + (offs_n[None, :] * stride_out_pre_dim),
             out_pre, mask=mask_token[:, None] & mask_n[None, :])
    tl.store(Out_Post_ptr + (offs_token[:, None] * stride_out_post_token) + (offs_n[None, :] * stride_out_post_dim),
             out_post, mask=mask_token[:, None] & mask_n[None, :])

    # -----------------------------------------------------------
    # 4. H_res_exp: Compute max and exp in float32 precision (matching original implementation)
    # -----------------------------------------------------------
    res_start_idx = 2 * real_N
    h_res = tl.load(base_h_ptr + (res_start_idx + offs_res[None, :]) * stride_h_dim,
                    mask=mask_token[:, None] & mask_res[None, :], other=-float('inf')).to(tl.float32)

    # Find max per row in float32 (matching original implementation precision)
    max_val = tl.max(h_res, axis=1)  # Shape: (BLOCK_SIZE, )

    # Compute exp(x - max) in float32 (matching original: paddle.exp((...).cast(paddle.float32)))
    res_exp = tl.exp(h_res - max_val[:, None])

    # Write final result (cast back to float16/bfloat16)
    tl.store(Out_Res_Exp_ptr + (offs_token[:, None] * stride_out_res_token) + (offs_res[None, :] * stride_out_res_dim),
             res_exp, mask=mask_token[:, None] & mask_res[None, :])

def mhc_fuse_sigmoid_exp(H_all, H_pre, H_post, H_res_exp, N):
    """
    Args:
        H_all: [B, L, N+N+N*N] - Input (scale and bias already applied)
        H_pre: [B, L, N] - Pre-allocated output tensor
        H_post: [B, L, N] - Pre-allocated output tensor
        H_res_exp: [B, L, N*N] - Pre-allocated output tensor
        N: int - Layer width

    Returns:
        H_pre: [B, L, N]
        H_post: [B, L, N]
        H_res_exp: [B, L, N*N]
    """
    B, L, total_dim = H_all.shape
    n_tokens = B * L

    # 2. Constant config (N <= 8 scenario)
    BLOCK_N = 8 if N <= 8 else 16  # Optimized for N=4 or 8
    BLOCK_RES = 64  # N*N max is 64
    BLOCK_SIZE = 256  # Increase block size for large L to increase parallelism

    grid = (triton.cdiv(n_tokens, BLOCK_SIZE), )

    mhc_sigmoid_exp_kernel[grid](
        H_all, H_pre, H_post, H_res_exp,  # Removed bias_terms and scaling_factors parameters
        H_all.stride(1), H_all.stride(2),
        H_pre.stride(1), H_pre.stride(2),
        H_post.stride(1), H_post.stride(2),
        H_res_exp.stride(1), 1,
        n_tokens, N,
        BLOCK_N=BLOCK_N, BLOCK_RES=BLOCK_RES, BLOCK_SIZE=BLOCK_SIZE
    )

    return H_pre, H_post, H_res_exp

# ============================================================================
# Optimized Depth Connection Forward Kernel
# ============================================================================

@triton.autotune(
    configs=[triton.Config({'BLOCK_SIZE_D': 1024, 'num_stages': 2}, num_warps=8)],
    key=['D'],
)
@triton.jit
def depth_connection_forward_kernel_optimized(
    H_post_ptr,
    branch_output_ptr,
    residuals_ptr,
    output_ptr,
    B: tl.constexpr,
    L: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    num_stages: tl.constexpr,
):
    """
    Depth connection forward kernel: output = H_post * branch_output + residuals.
    """
    pid = tl.program_id(0)

    num_tokens = B * L
    token_idx = pid // N
    stream_idx = pid % N

    if token_idx >= num_tokens:
        return

    batch_idx = token_idx // L
    seq_idx = token_idx % L

    stride_H = L * N
    stride_branch = L * 1 * D
    stride_res = L * N * D

    h_post_offset = batch_idx * stride_H + seq_idx * N + stream_idx
    h_post = tl.load(H_post_ptr + h_post_offset)

    for d_block in range(0, D, BLOCK_SIZE_D):
        d_offsets = d_block + tl.arange(0, BLOCK_SIZE_D)
        mask = d_offsets < D

        branch_offset = batch_idx * stride_branch + seq_idx * D
        branch_vals = tl.load(
            branch_output_ptr + branch_offset + d_offsets,
            mask=mask,
            other=0.0,
            eviction_policy='evict_last'
        )

        residual_offset = batch_idx * stride_res + seq_idx * N * D + stream_idx * D
        residual_vals = tl.load(
            residuals_ptr + residual_offset + d_offsets,
            mask=mask,
            other=0.0,
            eviction_policy='evict_last'
        )

        output_vals = h_post * branch_vals + residual_vals

        output_offset = batch_idx * stride_res + seq_idx * N * D + stream_idx * D
        tl.store(output_ptr + output_offset + d_offsets, output_vals, mask=mask)


def depth_connection_forward_triton_optimized(H_post, branch_output, residuals):
    """
    Depth connection forward using Triton kernel.

    Args:
        H_post: [B, L, N]
        branch_output: [B, L, 1, D]
        residuals: [B, L, N, D]

    Returns:
        output: [B, L, N, D]
    """
    B, L, N = H_post.shape
    D = branch_output.shape[-1]

    output = paddle.empty([B, L, N, D], dtype=residuals.dtype)

    num_tokens = B * L
    grid = (num_tokens * N,)

    depth_connection_forward_kernel_optimized[grid](
        H_post,
        branch_output,
        residuals,
        output,
        B=B,
        L=L,
        N=N,
        D=D,
    )

    return output


# ============================================================================
# Fused Kernels for Post-Sinkhorn Computation
# ============================================================================

@triton.jit
def fused_exp_double_matmul_kernel(
    H_res_exp_ptr,
    U_ptr,
    V_ptr,
    H_res_out_ptr,
    num_batches: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Fused kernel for: H_res = U @ H_res_exp @ V.

    Casts inputs to float32 internally for numerical stability.
    """
    pid = tl.program_id(0)
    batch_idx = pid // (N * N)
    row = (pid // N) % N
    col = pid % N

    if batch_idx >= num_batches:
        return

    stride_nn = N * N
    base_offset = batch_idx * stride_nn

    result = tl.zeros([], dtype=tl.float32)
    for m in range(N):
        u_val = tl.load(U_ptr + base_offset + row * N + m).to(tl.float32)
        inner_sum = tl.zeros([], dtype=tl.float32)
        for j in range(N):
            h_val = tl.load(H_res_exp_ptr + base_offset + m * N + j).to(tl.float32)
            v_val = tl.load(V_ptr + base_offset + j * N + col).to(tl.float32)
            inner_sum += h_val * v_val
        result += u_val * inner_sum

    tl.store(H_res_out_ptr + base_offset + row * N + col, result)


@triton.jit
def fused_residuals_branch_input_kernel(
    H_res_ptr,
    H_pre_ptr,
    x_ptr,
    residuals_ptr,
    branch_input_ptr,
    num_tokens: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
):
    """Fused kernel for computing residuals and branch_input.

    Outputs are cast to OUTPUT_DTYPE (triton type constant).

    Pass OUTPUT_DTYPE as tl.float32, tl.bfloat16, or tl.float16.
    """
    pid = tl.program_id(0)
    token_idx = pid // N
    n = pid % N

    if token_idx >= num_tokens:
        return

    stride_nn = N * N
    stride_nd = N * D

    base_h_res = token_idx * stride_nn + n * N
    base_x = token_idx * stride_nd
    base_residuals = token_idx * stride_nd + n * D

    for d_block in range(0, D, BLOCK_SIZE_D):
        d_offsets = d_block + tl.arange(0, BLOCK_SIZE_D)
        d_mask = d_offsets < D

        res_acc = tl.zeros([BLOCK_SIZE_D], dtype=tl.float32)
        for k in range(N):
            h_val = tl.load(H_res_ptr + base_h_res + k)
            x_vals = tl.load(x_ptr + base_x + k * D + d_offsets,
                           mask=d_mask, other=0.0)
            res_acc += h_val * x_vals

        tl.store(residuals_ptr + base_residuals + d_offsets,
                res_acc.to(OUTPUT_DTYPE), mask=d_mask)

    if n == 0:
        for d_block in range(0, D, BLOCK_SIZE_D):
            d_offsets = d_block + tl.arange(0, BLOCK_SIZE_D)
            d_mask = d_offsets < D

            branch_acc = tl.zeros([BLOCK_SIZE_D], dtype=tl.float32)
            for k in range(N):
                h_pre_val = tl.load(H_pre_ptr + token_idx * N + k)
                x_vals = tl.load(x_ptr + base_x + k * D + d_offsets,
                               mask=d_mask, other=0.0)
                branch_acc += h_pre_val * x_vals

            tl.store(branch_input_ptr + token_idx * D + d_offsets,
                    branch_acc.to(OUTPUT_DTYPE), mask=d_mask)


def post_sinkhorn_fused_forward(H_res_exp, u, v, H_pre, x):
    """
    Fused post-Sinkhorn forward computation (compact version).

    Optimized to use compact vectors u, v instead of diagonal matrices U, V.

    Args:
        H_res_exp: [B, L, N, N]
        u: [B*L, N] - compact left scaling vector (diagonal of U)
        v: [B*L, N] - compact right scaling vector (diagonal of V)
        H_pre: [B, L, N]
        x: [B, L, N, D]

    Returns:
        residuals: [B, L, N, D]
        branch_input: [B, L, 1, D]
        H_res: [B, L, N, N]
    """
    B, L, N, _ = H_res_exp.shape
    D = x.shape[-1]
    x_dtype = x.dtype

    # Map Paddle dtype to Triton dtype for kernel output
    dtype_map = {
        'float32': tl.float32,
        'bfloat16': tl.bfloat16,
        'float16': tl.float16,
    }
    output_dtype = dtype_map.get(str(x_dtype), tl.float32)

    num_batches = B * L

    # Allocate outputs as 4D, no reshape needed
    H_res_out = paddle.empty([B, L, N, N], dtype='float32')
    residuals = paddle.empty([B, L, N, D], dtype=x_dtype)
    branch_input = paddle.empty([B, L, 1, D], dtype=x_dtype)

    BLOCK_SIZE_D = min(512, triton.next_power_of_2(D))

    # For N=4, use single fused kernel (exp_matmul + residuals + branch_input)
    # For N!=4, fall back to two-kernel path
    if N == 4:
        grid = (num_batches,)
        exp_matmul_residuals_fused_kernel[grid](
            H_res_exp, u, v, H_pre, x,
            H_res_out, residuals, branch_input,
            num_tokens=num_batches, N=N, D=D,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
            OUTPUT_DTYPE=output_dtype,
            num_warps=8,
        )
    else:
        # Original version: H_res = U @ H_res_exp @ V (with U, V reconstructed from u, v)
        # Reconstruct diagonal matrices from vectors
        U = paddle.diag_embed(u)
        V = paddle.diag_embed(v)

        grid1 = (num_batches * N * N,)
        fused_exp_double_matmul_kernel[grid1](
            H_res_exp, U, V, H_res_out,
            num_batches=num_batches, N=N,
            BLOCK_SIZE_N=triton.next_power_of_2(N),
        )

        grid2 = (num_batches * N,)
        fused_residuals_branch_input_kernel[grid2](
            H_res_out, H_pre, x, residuals, branch_input,
            num_tokens=num_batches, N=N, D=D,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
            OUTPUT_DTYPE=output_dtype,
        )

    return residuals, branch_input, H_res_out


# ============================================================================
# Backward Kernels for Width Connection - Part 1: Branch & Residuals
# ============================================================================

@triton.jit
def width_branch_residuals_backward_kernel(
    d_branch_input_ptr,
    d_residuals_ptr,
    x_ptr,
    H_pre_ptr,
    H_res_ptr,
    d_H_pre_from_branch_ptr,
    d_H_res_mat_ptr,
    d_x_branch_add_residuals_ptr,
    B: tl.constexpr,
    L: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    d_start: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    """
    Kernel 1: Branch & Residuals Backward Fused

    Computes for a single D-block [d_start, d_start+BLOCK_SIZE_D):
    1. Branch Input backward:
       d_H_pre_from_branch += d_branch_input @ x^T  [B, L, N]
       d_x_from_branch = H_pre^T @ d_branch_input  [B, L, N, D]

    2. Residuals backward:
       d_H_res_mat += d_residuals @ x^T  [B, L, N, N]
       d_x_from_residuals = H_res^T @ d_residuals  [B, L, N, D]

    Grid: 2D (num_tokens, N) where num_tokens = B * L
    Each block handles one (token, n) pair and computes one D block.
    Results for d_H_pre_from_branch and d_H_res_mat are added to existing values.
    d_x_branch and d_x_residuals are written to the corresponding D block.

    All internal computation uses float32 for precision.
    """
    pid = tl.program_id(0)
    token_idx = pid // N
    n = pid % N

    num_tokens = B * L
    if token_idx >= num_tokens:
        return

    batch_idx = token_idx // L
    seq_idx = token_idx % L

    # Stride calculations
    stride_x_batch = L * N * D
    stride_x_seq = N * D
    stride_x_n = D
    stride_d_branch_batch = L * D
    stride_d_branch_seq = D
    stride_H_res = L * N * N
    stride_H_pre = L * N

    # D block offsets
    d_offsets = d_start + tl.arange(0, BLOCK_SIZE_D)
    d_mask = d_offsets < D

    # Base offsets for current token
    x_base = batch_idx * stride_x_batch + seq_idx * stride_x_seq

    # ============ d_H_res_mat calculation ============
    # d_H_res_mat[token, n, k] += sum over d_block of (d_residuals[token, n, d] * x[token, k, d])

    # Load d_residuals[token, n, d_block]
    d_residuals_base = batch_idx * stride_x_batch + seq_idx * stride_x_seq + n * stride_x_n
    d_residuals_vals = tl.load(d_residuals_ptr + d_residuals_base + d_offsets, mask=d_mask, other=0.0).to(tl.float32)

    # For each k, compute dot product and add to d_H_res_mat
    for k in range(N):
        # Load x[token, k, d_block]
        x_vals = tl.load(x_ptr + x_base + k * stride_x_n + d_offsets, mask=d_mask, other=0.0).to(tl.float32)

        # Compute partial sum
        d_H_res_val = tl.sum(d_residuals_vals * x_vals)

        # Load existing value and add
        H_res_base = batch_idx * stride_H_res + seq_idx * N * N + n * N + k
        existing_val = tl.load(d_H_res_mat_ptr + H_res_base)
        tl.store(d_H_res_mat_ptr + H_res_base, existing_val + d_H_res_val)

    # ============ d_x_residuals calculation ============
    # d_x_residuals[token, n, d_block] = sum over k of (H_res[token, k, n] * d_residuals[token, k, d])
    d_x_residuals_acc = tl.zeros([BLOCK_SIZE_D], dtype=tl.float32)
    for k in range(N):
        # Load H_res[token, k, n] (note the order: k, n)
        H_res_base = batch_idx * stride_H_res + seq_idx * N * N + k * N + n
        h_res_val = tl.load(H_res_ptr + H_res_base).to(tl.float32)

        # Load d_residuals[token, k, d_block]
        d_residuals_base = batch_idx * stride_x_batch + seq_idx * stride_x_seq + k * stride_x_n
        d_residuals_vals = tl.load(d_residuals_ptr + d_residuals_base + d_offsets, mask=d_mask, other=0.0).to(tl.float32)

        # Accumulate
        d_x_residuals_acc += h_res_val * d_residuals_vals

    # ============ Branch backward ============
    # d_H_pre_from_branch[token, n] += sum over d_block of (d_branch_input[token, d] * x[token, n, d])
    d_branch_base = batch_idx * stride_d_branch_batch + seq_idx * stride_d_branch_seq
    d_branch_vals = tl.load(d_branch_input_ptr + d_branch_base + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
    x_vals_n = tl.load(x_ptr + x_base + n * stride_x_n + d_offsets, mask=d_mask, other=0.0).to(tl.float32)

    d_H_pre_val = tl.sum(d_branch_vals * x_vals_n)

    # Add to existing value
    H_pre_base = batch_idx * stride_H_pre + seq_idx * N + n
    existing_val = tl.load(d_H_pre_from_branch_ptr + H_pre_base)
    tl.store(d_H_pre_from_branch_ptr + H_pre_base, existing_val + d_H_pre_val.to(tl.float32))

    # d_x_branch[token, n, d_block] = H_pre[token, n] * d_branch_input[token, d]
    H_pre_base = batch_idx * stride_H_pre + seq_idx * N
    h_pre_val = tl.load(H_pre_ptr + H_pre_base + n).to(tl.float32)
    d_x_from_branch_val = h_pre_val * d_branch_vals

    # Store d_x_branch_add_residuals[token, n, d_block] = d_x_residuals + d_x_branch
    d_x_branch_add_residuals_base = batch_idx * stride_x_batch + seq_idx * stride_x_seq + n * stride_x_n
    d_x_branch_add_residuals_val = d_x_residuals_acc + d_x_from_branch_val
    tl.store(d_x_branch_add_residuals_ptr + d_x_branch_add_residuals_base + d_offsets, d_x_branch_add_residuals_val, mask=d_mask)


def width_branch_residuals_backward_triton(d_branch_input, d_residuals, x, H_pre, H_res):
    """
    Wrapper for Branch & Residuals Backward Kernel (Kernel 1).

    All casts to float32 are handled inside the kernel.

    Args:
        d_branch_input: [B, L, 1, D]
        d_residuals: [B, L, N, D]
        x: [B, L, N, D]
        H_pre: [B, L, N]
        H_res: [B, L, N, N]

    Returns:
        d_H_pre_from_branch: [B, L, N] (float32)
        d_H_res_mat: [B, L, N, N] (float32)
        d_x_branch_add_residuals: [B, L, N, D] (float32) = d_x_branch + d_x_residuals
    """
    B, L, N, D = x.shape

    # Allocate float32 outputs
    d_H_pre_from_branch = paddle.zeros([B, L, N], dtype='float32')
    d_H_res_mat = paddle.zeros([B, L, N, N], dtype='float32')
    d_x_branch_add_residuals = paddle.empty([B, L, N, D], dtype='float32')

    num_tokens = B * L

    # Select kernel based on N and D for optimal performance
    if N == 4 and D >= 4096:
        # Use optimized kernel for N=4, D>=4096 with data preloading
        grid = (num_tokens * N,)
        width_branch_residuals_backward_kernel_n4_optimized[grid](
            d_branch_input,
            d_residuals,
            x,
            H_pre,
            H_res,
            d_H_pre_from_branch,
            d_H_res_mat,
            d_x_branch_add_residuals,
            B=B,
            L=L,
            N=N,
            D=D,
        )
    else:
        # Use generic kernel with D-blocking handled in wrapper
        # This avoids Triton JIT limitations on dynamic array indexing
        grid = (num_tokens * N,)
        BLOCK_SIZE_D = 1024  # Fixed block size for generic kernel
        for d_start in range(0, D, BLOCK_SIZE_D):
            width_branch_residuals_backward_kernel[grid](
                d_branch_input,
                d_residuals,
                x,
                H_pre,
                H_res,
                d_H_pre_from_branch,
                d_H_res_mat,
                d_x_branch_add_residuals,
                B=B,
                L=L,
                N=N,
                D=D,
                d_start=d_start,
                BLOCK_SIZE_D=BLOCK_SIZE_D,
            )

    return d_H_pre_from_branch, d_H_res_mat, d_x_branch_add_residuals


# ============================================================================
# Optimized Backward Kernel for N=4, D>=4096
# ============================================================================

@triton.autotune(
    configs=[triton.Config({'BLOCK_SIZE_D': 2048}, num_warps=8)],
    key=['D'],
)
@triton.jit
def width_branch_residuals_backward_kernel_n4_optimized(
    d_branch_input_ptr,
    d_residuals_ptr,
    x_ptr,
    H_pre_ptr,
    H_res_ptr,
    d_H_pre_from_branch_ptr,
    d_H_res_mat_ptr,
    d_x_branch_add_residuals_ptr,
    B: tl.constexpr,
    L: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    """
    Optimized Kernel 1 for N=4, D>=4096 with Data Preloading

    Key Optimizations:
    1. Preload x[N=4, BLOCK_SIZE_D] and d_residuals[N=4, BLOCK_SIZE_D] once per D-block iteration
    2. Reduces global memory access from ~12 loads to ~2 loads per D-block iteration
    3. Better data locality and cache utilization

    Computes:
    1. Branch Input backward:
       d_H_pre_from_branch = d_branch_input @ x^T  [B, L, N]
       d_x_from_branch = H_pre^T @ d_branch_input  [B, L, N, D]

    2. Residuals backward:
       d_H_res_mat = d_residuals @ x^T  [B, L, N, N]
       d_x_from_residuals = H_res^T @ d_residuals  [B, L, N, D]

    Grid: 2D (num_tokens, N) where num_tokens = B * L
    Each block handles one (token, n) pair and computes all D elements.
    All internal computation uses float32 for precision.
    """
    pid = tl.program_id(0)
    token_idx = pid // N
    n = pid % N

    num_tokens = B * L
    if token_idx >= num_tokens:
        return

    batch_idx = token_idx // L
    seq_idx = token_idx % L

    # Stride calculations
    stride_x_batch = L * N * D      # for x, d_residuals, etc.
    stride_x_seq = N * D            # stride for L dimension
    stride_x_n = D                  # stride for N dimension
    stride_d_branch_batch = L * D   # for d_branch_input [B, L, 1, D]
    stride_d_branch_seq = D        # stride for L dimension in d_branch_input
    stride_H_res = L * N * N
    stride_H_pre = L * N

    # Base offsets for current token
    x_base = batch_idx * stride_x_batch + seq_idx * stride_x_seq

    # ===== OPTIMIZATION: Load H_res and H_pre ONCE per block (outside D-loop) =====
    # H_res[token, k, n] for k=0,1,2,3 are constants for this (token, n) block
    H_res_base_k0 = batch_idx * stride_H_res + seq_idx * N * N + 0 * N + n
    H_res_base_k1 = batch_idx * stride_H_res + seq_idx * N * N + 1 * N + n
    H_res_base_k2 = batch_idx * stride_H_res + seq_idx * N * N + 2 * N + n
    H_res_base_k3 = batch_idx * stride_H_res + seq_idx * N * N + 3 * N + n

    h_res_0 = tl.load(H_res_ptr + H_res_base_k0).to(tl.float32)
    h_res_1 = tl.load(H_res_ptr + H_res_base_k1).to(tl.float32)
    h_res_2 = tl.load(H_res_ptr + H_res_base_k2).to(tl.float32)
    h_res_3 = tl.load(H_res_ptr + H_res_base_k3).to(tl.float32)

    # Load H_pre ONCE per block (outside D-loop)
    H_pre_base = batch_idx * stride_H_pre + seq_idx * N
    h_pre_val = tl.load(H_pre_ptr + H_pre_base + n).to(tl.float32)

    # Initialize accumulators for d_H_res_mat and d_H_pre_from_branch
    # These need to be accumulated across all D blocks
    d_H_res_mat_acc_0 = tl.zeros([], dtype=tl.float32)
    d_H_res_mat_acc_1 = tl.zeros([], dtype=tl.float32)
    d_H_res_mat_acc_2 = tl.zeros([], dtype=tl.float32)
    d_H_res_mat_acc_3 = tl.zeros([], dtype=tl.float32)
    d_H_pre_from_branch_acc = tl.zeros([], dtype=tl.float32)

    # Iterate over D in blocks
    for d_start in range(0, D, BLOCK_SIZE_D):
        d_offsets = d_start + tl.arange(0, BLOCK_SIZE_D)
        d_mask = d_offsets < D

        # ===== OPTIMIZATION: Preload x[N=4, BLOCK_SIZE_D] and d_residuals[N=4, BLOCK_SIZE_D] =====
        # This reduces global memory access significantly for N=4 scenario

        # Preload x[token, :, d_block] -> shape [4, BLOCK_SIZE_D]
        # Using explicit loop for clarity and guaranteed loading order
        x_block_0 = tl.load(x_ptr + x_base + 0 * stride_x_n + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
        x_block_1 = tl.load(x_ptr + x_base + 1 * stride_x_n + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
        x_block_2 = tl.load(x_ptr + x_base + 2 * stride_x_n + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
        x_block_3 = tl.load(x_ptr + x_base + 3 * stride_x_n + d_offsets, mask=d_mask, other=0.0).to(tl.float32)

        # Preload d_residuals[token, :, d_block] -> shape [4, BLOCK_SIZE_D]
        d_res_block_0 = tl.load(d_residuals_ptr + x_base + 0 * stride_x_n + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
        d_res_block_1 = tl.load(d_residuals_ptr + x_base + 1 * stride_x_n + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
        d_res_block_2 = tl.load(d_residuals_ptr + x_base + 2 * stride_x_n + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
        d_res_block_3 = tl.load(d_residuals_ptr + x_base + 3 * stride_x_n + d_offsets, mask=d_mask, other=0.0).to(tl.float32)

        # ===== Compute d_H_res_mat[token, n, k] = sum_d(d_residuals[n] * x[k]) =====
        # Use tl.where to select d_residuals[n] based on current n (avoids warp divergence)
        d_res_n = tl.where(n == 0, d_res_block_0,
                  tl.where(n == 1, d_res_block_1,
                  tl.where(n == 2, d_res_block_2, d_res_block_3)))

        # Compute dot products with preloaded x blocks and accumulate
        # d_H_res_mat[token, n, 0]
        d_H_res_mat_acc_0 += tl.sum(d_res_n * x_block_0)

        # d_H_res_mat[token, n, 1]
        d_H_res_mat_acc_1 += tl.sum(d_res_n * x_block_1)

        # d_H_res_mat[token, n, 2]
        d_H_res_mat_acc_2 += tl.sum(d_res_n * x_block_2)

        # d_H_res_mat[token, n, 3]
        d_H_res_mat_acc_3 += tl.sum(d_res_n * x_block_3)

        # ===== Compute d_x_residuals[token, n, d] = sum_k(H_res[k,n] * d_residuals[k]) =====
        # Use pre-loaded H_res values (loaded once outside the loop)
        d_x_residuals_acc = h_res_0 * d_res_block_0
        d_x_residuals_acc += h_res_1 * d_res_block_1
        d_x_residuals_acc += h_res_2 * d_res_block_2
        d_x_residuals_acc += h_res_3 * d_res_block_3

        # ===== Branch backward =====
        # d_H_pre_from_branch[token, n] = sum_d(d_branch_input[token, d] * x[token, n, d])
        d_branch_base = batch_idx * stride_d_branch_batch + seq_idx * stride_d_branch_seq
        d_branch_vals = tl.load(d_branch_input_ptr + d_branch_base + d_offsets, mask=d_mask, other=0.0).to(tl.float32)

        # Get preloaded x[token, n, d_block] using tl.where (avoids warp divergence)
        x_vals_n = tl.where(n == 0, x_block_0,
                   tl.where(n == 1, x_block_1,
                   tl.where(n == 2, x_block_2, x_block_3)))

        # Accumulate d_H_pre_from_branch across all D blocks
        d_H_pre_from_branch_acc += tl.sum(d_branch_vals * x_vals_n)

        # d_x_branch[token, n, d] = H_pre[token, n] * d_branch_input[token, d]
        # Use pre-loaded H_pre value (loaded once outside the loop)
        d_x_from_branch_val = h_pre_val * d_branch_vals

        # Store d_x_branch_add_residuals[token, n, d_block] = d_x_residuals + d_x_branch
        d_x_branch_add_residuals_base = batch_idx * stride_x_batch + seq_idx * stride_x_seq + n * stride_x_n
        d_x_branch_add_residuals_val = d_x_residuals_acc + d_x_from_branch_val
        tl.store(d_x_branch_add_residuals_ptr + d_x_branch_add_residuals_base + d_offsets, d_x_branch_add_residuals_val, mask=d_mask)

    # Store accumulated d_H_res_mat and d_H_pre_from_branch after all D blocks
    # d_H_res_mat[token, n, 0]
    H_res_base = batch_idx * stride_H_res + seq_idx * N * N + n * N + 0
    tl.store(d_H_res_mat_ptr + H_res_base, d_H_res_mat_acc_0.to(tl.float32))

    # d_H_res_mat[token, n, 1]
    H_res_base = batch_idx * stride_H_res + seq_idx * N * N + n * N + 1
    tl.store(d_H_res_mat_ptr + H_res_base, d_H_res_mat_acc_1.to(tl.float32))

    # d_H_res_mat[token, n, 2]
    H_res_base = batch_idx * stride_H_res + seq_idx * N * N + n * N + 2
    tl.store(d_H_res_mat_ptr + H_res_base, d_H_res_mat_acc_2.to(tl.float32))

    # d_H_res_mat[token, n, 3]
    H_res_base = batch_idx * stride_H_res + seq_idx * N * N + n * N + 3
    tl.store(d_H_res_mat_ptr + H_res_base, d_H_res_mat_acc_3.to(tl.float32))

    # Store d_H_pre_from_branch[token, n]
    H_pre_base = batch_idx * stride_H_pre + seq_idx * N + n
    tl.store(d_H_pre_from_branch_ptr + H_pre_base, d_H_pre_from_branch_acc.to(tl.float32))


# ============================================================================
# width_rmsnorm_gemm_backward_triton - Triton Kernel Version
# ============================================================================
# This function calls Triton kernels and will be used to fix kernel bugs
# ============================================================================

def width_rmsnorm_gemm_backward_triton(d_H_all, x, combined_weights, norm_eps, N, D, d_x_branch_add_residuals):
    """
    Triton version of width_rmsnorm_gemm_backward.
    Fused with d_x_branch_add_residuals to reduce memory access overhead.

    Args:
        d_H_all: [B, L, N+N+N*N]
        x: [B, L, N, D]
        combined_weights: [N*D, N+N+N*N]
        norm_eps: float
        N: int
        D: int
        d_x_branch_add_residuals: [B, L, N, D]

    Returns:
        d_x: [B, L, N, D] = d_x_norm + d_x_branch_add_residuals
        d_combined_weights: [N*D, N+N+N*N]

    Note:
        This function uses norm_weight=1.0 (no learnable weight for RMSNorm).
    """
    B, L = x.shape[:2]
    NN = N * N
    N3 = N + N + NN
    ND = N * D
    num_tokens = B * L

    # Get dtype directly from input tensors
    x_dtype = x.dtype
    combined_weights_dtype = combined_weights.dtype

    # Paddle dtype -> Triton dtype mapping
    def get_triton_dtype(paddle_dtype):
        if paddle_dtype == paddle.bfloat16:
            return tl.bfloat16
        elif paddle_dtype == paddle.float16:
            return tl.float16
        return tl.float32

    dx_triton_dtype = get_triton_dtype(x_dtype)
    dcw_triton_dtype = get_triton_dtype(combined_weights_dtype)

    # Get actual strides for 4D tensors
    # For [B, L, N, D], stride[1] = N*D = ND (token stride)
    # For [B, L, N3], stride[1] = N3 (token stride)
    stride_d_H_all = d_H_all.stride(1)  # [B, L, N3] -> stride[1] = N3

    # Allocate outputs with target dtype
    normed = paddle.empty([num_tokens, ND], dtype=paddle.bfloat16)
    d_normed = paddle.empty([num_tokens, ND], dtype=paddle.float32)
    d_x = paddle.empty([B, L, N, D], dtype=x_dtype)  # Direct 4D output
    d_combined_weights = paddle.zeros([ND, N3], dtype=combined_weights_dtype)

    # GEMM configuration
    BLOCK_M_FIXED = 128
    BLOCK_N_FIXED = 64
    BLOCK_K_FIXED = 32

    # ==========================================
    # Step 1: Compute d_normed (GEMM 1)
    # Compute d_normed first so the subsequent RMSNorm fused kernel can use it
    # ==========================================
    grid_m1 = triton.cdiv(num_tokens, BLOCK_M_FIXED)
    grid_n1 = triton.cdiv(ND, BLOCK_N_FIXED)
    grid1 = (grid_m1, grid_n1)

    gemm_d_normed_kernel_fixed[grid1](
        d_H_all,
        combined_weights,
        d_normed,
        M=num_tokens,
        N=ND,
        K=N3,
        stride_dh_m=stride_d_H_all,  # actual stride from [B, L, N3]
        stride_dh_k=1,      # stride for K dimension in d_H_all
        stride_cw_n=N3,    # stride for M dimension in combined_weights (K=N3)
        stride_cw_k=1,      # stride for K dimension in combined_weights
        stride_dn_m=ND,    # stride for M dimension in d_normed (N=ND)
        stride_dn_k=1,      # stride for K dimension in d_normed
        BLOCK_M=BLOCK_M_FIXED,
        BLOCK_N=BLOCK_N_FIXED,
        BLOCK_K=BLOCK_K_FIXED,
    )

    # ==========================================
    # Step 2: RMSNorm Forward + Backward Fusion
    # Single kernel completes: forward (normed, inv_std) + backward (d_x)
    # ==========================================
    # Select optimal BLOCK_ND and NUM_WARPS based on ND
    if ND <= 16384:
        BLOCK_ND_RMSNORM = 4096
        NUM_WARPS_RMSNORM = 16
    elif ND <= 32768:
        BLOCK_ND_RMSNORM = 8192
        NUM_WARPS_RMSNORM = 32
    else:
        BLOCK_ND_RMSNORM = 16384
        NUM_WARPS_RMSNORM = 32
    grid_2 = (num_tokens,)  # One block per token

    rmsnorm_fused_forward_backward[grid_2](
        x,
        d_normed,
        d_x_branch_add_residuals,
        normed,
        d_x,
        num_tokens=num_tokens,
        ND=ND,
        eps=norm_eps,
        BLOCK_ND=BLOCK_ND_RMSNORM,
        DX_DTYPE=dx_triton_dtype,
        num_warps=NUM_WARPS_RMSNORM,
    )

    # ==========================================
    # Step 3: Compute d_combined_weights (GEMM 2)
    # d_combined_weights = normed^T @ d_H_all, shape [ND, N3]
    # autotune automatically selects optimal BLOCK_M/BLOCK_K/num_warps/num_stages
    # ==========================================
    def grid_dcw(META):
        return (triton.cdiv(ND, META['BLOCK_M']), triton.cdiv(N3, META['BLOCK_N']))

    gemm_d_combined_weights_kernel_fixed[grid_dcw](
        normed,
        d_H_all,
        d_combined_weights,
        M=ND,
        N=N3,
        K=num_tokens,
        stride_normed_m=1,
        stride_normed_k=ND,
        stride_dh_k=stride_d_H_all,
        stride_dh_n=1,
        stride_dcw_m=N3,
        stride_dcw_n=1,
        DCW_DTYPE=dcw_triton_dtype,
    )

    return d_x, d_combined_weights


@triton.jit
def rmsnorm_fused_forward_backward(
    x_ptr,
    d_normed_ptr,
    d_x_branch_add_residuals_ptr,
    normed_ptr,
    d_x_ptr,
    num_tokens,
    ND,
    eps,
    BLOCK_ND: tl.constexpr,
    DX_DTYPE: tl.constexpr = tl.float32,  # d_x output type: tl.float32/tl.bfloat16/tl.float16
):
    """
    Scheme B + Scheme 3 Optimization: Fully fused Forward + Backward + reduced atomic operation contention

    Performance characteristics:
    - x load count: 2 (Phase 1 + Phase 2, Phase 2 leverages L1 cache)
    - d_normed load count: 2
    - Atomic operations: none (norm_weight=1.0, no d_norm_weight)
    - Extra memory: none

    Optimizations:
    1. Phase 1 uses eviction_policy="evict_last" to keep data in cache
    2. Phase 2 uses cache_modifier=".ca" (cache all) hint for caching

    Args:
        DX_DTYPE: d_x output type, supports tl.float32/tl.bfloat16/tl.float16

    Note:
        This kernel uses norm_weight=1.0 (no learnable weight for RMSNorm).
    """
    token_id = tl.program_id(0)

    if token_id >= num_tokens:
        return

    two_over_nd = 2.0 / ND

    # ============================================================
    # Phase 1: Compute variance + sum_dwx
    # Use eviction_policy="evict_last" to keep data in L1 cache for Phase 2
    # Optimization: Use vector accumulator, only element-wise accumulation in loop, sum at the end
    # ============================================================
    var_acc = tl.zeros([BLOCK_ND], dtype=tl.float32)
    sum_dwx_acc = tl.zeros([BLOCK_ND], dtype=tl.float32)

    for d_start in range(0, ND, BLOCK_ND):
        d_offsets = d_start + tl.arange(0, BLOCK_ND)
        d_mask = d_offsets < ND

        x_block = tl.load(
            x_ptr + token_id * ND + d_offsets,
            mask=d_mask,
            other=0.0,
            eviction_policy="evict_last"
        ).to(tl.float32)

        d_normed_block = tl.load(
            d_normed_ptr + token_id * ND + d_offsets,
            mask=d_mask,
            other=0.0,
            eviction_policy="evict_last"
        ).to(tl.float32)

        # Forward: Compute variance (element-wise accumulation, no sum)
        var_acc += x_block * x_block

        # Backward: Accumulate sum_dwx (element-wise accumulation, no sum)
        # norm_weight = 1.0 (no learnable weight)
        sum_dwx_acc += d_normed_block * x_block

    # After the loop, perform sum (reduces number of reductions)
    var_acc = tl.sum(var_acc)
    sum_dwx_acc = tl.sum(sum_dwx_acc)

    # Compute statistics
    variance = var_acc / ND
    inv_std = 1.0 / tl.sqrt(variance + eps)
    inv_std_cubed = inv_std * inv_std * inv_std

    # Compute backward intermediate values
    d_var = -0.5 * inv_std_cubed * sum_dwx_acc

    # ============================================================
    # Phase 2: Compute normed + d_x
    # Data should still be in L1 cache, lower load latency
    # ============================================================
    for d_start in range(0, ND, BLOCK_ND):
        d_offsets = d_start + tl.arange(0, BLOCK_ND)
        d_mask = d_offsets < ND

        # Reload data (should hit L1 cache)
        x_block = tl.load(
            x_ptr + token_id * ND + d_offsets,
            mask=d_mask,
            other=0.0,
            eviction_policy="evict_first"  # Not needed after Phase 2, can be evicted
        ).to(tl.float32)

        d_normed_block = tl.load(
            d_normed_ptr + token_id * ND + d_offsets,
            mask=d_mask,
            other=0.0,
            eviction_policy="evict_first"
        ).to(tl.float32)

        d_x_branch = tl.load(
            d_x_branch_add_residuals_ptr + token_id * ND + d_offsets,
            mask=d_mask,
            other=0.0,
            eviction_policy="evict_first"
        ).to(tl.float32)

        # Forward: normed (store as bfloat16 to reduce write bandwidth)
        # norm_weight = 1.0 (no learnable weight)
        normed_val = x_block * inv_std
        tl.store(
            normed_ptr + token_id * ND + d_offsets,
            normed_val.to(tl.bfloat16),
            mask=d_mask
        )

        # Backward: d_x (with type conversion)
        # norm_weight = 1.0 (no learnable weight)
        d_x_scale = d_normed_block * inv_std
        d_x_from_var = d_var * (x_block * two_over_nd)
        d_x_val = d_x_scale + d_x_from_var + d_x_branch
        tl.store(d_x_ptr + token_id * ND + d_offsets, d_x_val.to(DX_DTYPE), mask=d_mask)


@triton.jit
def gemm_d_normed_kernel_fixed(
    d_H_all_ptr,
    combined_weights_ptr,
    d_normed_ptr,
    M,
    N,
    K,
    stride_dh_m,
    stride_dh_k,
    stride_cw_n,
    stride_cw_k,
    stride_dn_m,
    stride_dn_k,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fixed version of gemm_d_normed_kernel without autotune."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    rm_mask = rm < M
    rn_mask = rn < N

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_offsets = k + rk
        k_mask = k_offsets < K

        a_block = tl.load(
            d_H_all_ptr + rm[:, None] * stride_dh_m + k_offsets[None, :] * stride_dh_k,
            mask=rm_mask[:, None] & k_mask[None, :],
            other=0.0
        )

        # Load combined_weights^T[k, rn]
        # combined_weights is [ND, N3] in row-major
        # combined_weights^T[k, rn] = combined_weights[rn, k]
        b_block = tl.load(
            combined_weights_ptr + k_offsets[:, None] * stride_cw_k + rn[None, :] * stride_cw_n,
            mask=k_mask[:, None] & rn_mask[None, :],
            other=0.0
        ).to(tl.bfloat16)

        # acc += d_H_all[rm, k] @ combined_weights^T[k, rn]
        acc += tl.dot(a_block, b_block)

    c_mask = rm_mask[:, None] & rn_mask[None, :]
    tl.store(
        d_normed_ptr + rm[:, None] * stride_dn_m + rn[None, :] * stride_dn_k,
        acc,
        mask=c_mask
    )


@triton.autotune(
    configs=[triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 128}, num_warps=4, num_stages=4)],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_d_combined_weights_kernel_fixed(
    normed_ptr,
    d_H_all_ptr,
    d_combined_weights_ptr,
    M,  # ND
    N,  # N3
    K,  # num_tokens
    stride_normed_m,
    stride_normed_k,
    stride_dh_k,
    stride_dh_n,
    stride_dcw_m,
    stride_dcw_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DCW_DTYPE: tl.constexpr = tl.bfloat16,
):
    """Autotuned GEMM for d_combined_weights = normed^T @ d_H_all."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    rm_mask = rm < M
    rn_mask = rn < N

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_offsets = k + rk
        k_mask = k_offsets < K

        a_block = tl.load(
            normed_ptr + k_offsets[:, None] * stride_normed_k + rm[None, :] * stride_normed_m,
            mask=k_mask[:, None] & rm_mask[None, :],
            other=0.0
        )

        b_block = tl.load(
            d_H_all_ptr + k_offsets[:, None] * stride_dh_k + rn[None, :] * stride_dh_n,
            mask=k_mask[:, None] & rn_mask[None, :],
            other=0.0
        )

        acc += tl.dot(tl.trans(a_block), b_block)

    c_mask = rm_mask[:, None] & rn_mask[None, :]
    tl.store(
        d_combined_weights_ptr + rm[:, None] * stride_dcw_m + rn[None, :] * stride_dcw_n,
        acc.to(DCW_DTYPE),
        mask=c_mask
    )



# ============================================================================
# H_res + Activation Backward (3-Kernel Triton Implementation)
# ============================================================================

# ----------------------------------------------------------------------------
# Kernel A: H_res_backward_fused_kernel
# Function: Fused double matrix multiplication + exponential gradient + scaling + d_res_scale computation
# ----------------------------------------------------------------------------

@triton.jit
def hres_bwd_exp_matmul_kernel(
    # Inputs
    d_H_res_mat_ptr,       # [B*L, N, N]
    H_res_exp_ptr,         # [B*L, N, N]
    U_ptr,                 # [B*L, N, N]
    V_ptr,                 # [B*L, N, N]
    H_all_ptr,             # [B*L, N+N+N*N] - float32
    bias_terms_ptr,        # [N+N+N*N] - float32
    scaling_factors_ptr,   # [3] - float32
    # Outputs
    d_scaling_factors_ptr, # [3] - output for scale gradients (index 2: res_scale)
    d_H_all_ptr,           # [B*L, N+N+N*N] - write d_H_all[2N:] directly (bfloat16)
    d_bias_terms_ptr,      # [N+N+N*N] - atomic add d_bias_res at offset 2*N
    # Parameters
    num_tokens: tl.constexpr,  # B*L
    N: tl.constexpr,
    NN: tl.constexpr,
    N3: tl.constexpr,
    stride_d_H_res_mat,
    stride_H_res_exp,
    stride_U,
    stride_V,
    stride_H_all,
):
    """
    Kernel A: Fused H_res backward computation.

    Computes:
    1. d_H_res_exp = U^T @ d_H_res_mat @ V^T
    2. d_H_res = d_H_res_exp * H_res_exp
    3. d_H_res_scaled = d_H_res * res_scale -> write to d_H_all[2N:]
    4. d_res_scale = sum(d_H_res * H_all_raw_res)
    5. d_bias_res = d_H_res -> atomic_add to d_bias_terms[2N:]

    Grid: (num_tokens,) - each block handles one [N, N] matrix
    """
    pid = tl.program_id(0)

    if pid >= num_tokens:
        return

    # Load res_scale
    res_scale = tl.load(scaling_factors_ptr + 2)

    # Base offsets for current token
    d_H_res_mat_base = pid * stride_d_H_res_mat
    H_res_exp_base = pid * stride_H_res_exp
    U_base = pid * stride_U
    V_base = pid * stride_V
    H_all_base = pid * stride_H_all

    # Accumulator for d_res_scale
    d_res_scale_acc = tl.zeros([], dtype=tl.float32)

    # Process NN elements in blocks without continue/break
    # Process all elements and use mask for bounds
    nn_offsets = tl.arange(0, NN)
    for nn_start in range(0, NN, NN):
        idx = nn_start + nn_offsets
        nn_mask = idx < NN

        # For each element in the block, compute d_H_res
        for i in range(NN):
            idx_val = nn_start + i
            if idx_val >= NN:
                pass  # Skip by not doing anything, just let the loop finish
            else:
                # Convert flat index to (row, col)
                row = idx_val // N
                col = idx_val % N

                # Compute d_H_res_exp[row, col] = U^T @ d_H_res_mat @ V^T
                d_H_res_exp_rc = tl.zeros([], dtype=tl.float32)

                for k in range(N):
                    u_kr = tl.load(U_ptr + U_base + k * N + row).to(tl.float32)

                    # sum_j d_H_res_mat[k, j] * V[col, j]
                    inner_sum = tl.zeros([], dtype=tl.float32)
                    for j in range(N):
                        d_H_mat_kj = tl.load(d_H_res_mat_ptr + d_H_res_mat_base + k * N + j).to(tl.float32)
                        v_cj = tl.load(V_ptr + V_base + col * N + j).to(tl.float32)
                        inner_sum += d_H_mat_kj * v_cj

                    d_H_res_exp_rc += u_kr * inner_sum

                # Load H_res_exp[r, c]
                h_res_exp_rc = tl.load(H_res_exp_ptr + H_res_exp_base + row * N + col).to(tl.float32)

                # d_H_res = d_H_res_exp * H_res_exp
                d_H_res_rc = d_H_res_exp_rc * h_res_exp_rc

                # Load H_all_raw_res[row, col] (flattened)
                H_all_raw_res_rc = (tl.load(H_all_ptr + H_all_base + 2 * N + row * N + col).to(tl.float32) -
                                     tl.load(bias_terms_ptr + 2 * N + row * N + col).to(tl.float32)) / res_scale

                # Accumulate d_res_scale
                d_res_scale_acc += d_H_res_rc * H_all_raw_res_rc

                # d_H_res_scaled -> write to d_H_all[2N + idx_val] (values already in register)
                d_H_res_scaled_rc = d_H_res_rc * res_scale
                tl.store(d_H_all_ptr + H_all_base + 2 * N + idx_val, d_H_res_scaled_rc.to(tl.bfloat16))

                # d_bias_res: d_H_res_rc already in register
                tl.atomic_add(d_bias_terms_ptr + 2 * N + idx_val, d_H_res_rc)

    # Store d_res_scale using atomic add (index 2 in d_scaling_factors)
    tl.atomic_add(d_scaling_factors_ptr + 2, d_res_scale_acc)


# ----------------------------------------------------------------------------
# Kernel B: Activation_backward_fused_kernel
# Function: H_pre/H_post sigmoid gradient + d_pre_scale/d_post_scale computation
# ----------------------------------------------------------------------------

@triton.jit
def pre_post_act_bwd_kernel(
    # Inputs
    d_H_post_ptr,               # [B*L, N]
    d_H_pre_from_branch_ptr,    # [B*L, N]
    H_all_ptr,                  # [B*L, N+N+N*N] - float32
    bias_terms_ptr,             # [N+N+N*N] - float32
    scaling_factors_ptr,        # [3] - float32
    # Outputs
    d_scaling_factors_ptr,      # [3] - output for scale gradients (index 0: pre_scale, index 1: post_scale)
    d_H_all_ptr,                # [B*L, N+N+N*N] - merge d_H_pre/d_H_post directly here
    d_bias_terms_ptr,           # [N+N+N*N] - bias gradient (atomic add for pre/post)
    # Parameters
    num_tokens: tl.constexpr,
    N: tl.constexpr,
    N3: tl.constexpr,
    stride_H_all,
):
    """
    Kernel B: Fused activation backward for H_pre and H_post.

    Computes sigmoid gradients + scale gradients, and directly:
    - Writes d_H_pre/d_H_post into d_H_all[:2N] (merge, eliminates Kernel C dependency)
    - Accumulates d_bias_terms[:2N] via atomic_add (bias gradient, no extra load)

    Grid: (num_tokens,) - each block handles one token
    """
    pid = tl.program_id(0)

    if pid >= num_tokens:
        return

    # Load scaling factors
    pre_scale = tl.load(scaling_factors_ptr + 0)
    post_scale = tl.load(scaling_factors_ptr + 1)

    # Base offset for H_all (compute once, use for all indexing)
    H_all_base_offset = pid * stride_H_all

    # Accumulators for scale gradients
    d_pre_scale_acc = tl.zeros([], dtype=tl.float32)
    d_post_scale_acc = tl.zeros([], dtype=tl.float32)

    # Process N elements element-by-element (N is small, <= 8)
    for i in range(N):
        # ===== H_post backward =====
        # Load H_all[N + i]
        H_all_post_val = tl.load(H_all_ptr + H_all_base_offset + N + i).to(tl.float32)

        # Load bias_terms[N + i]
        bias_post_val = tl.load(bias_terms_ptr + N + i).to(tl.float32)

        # H_all_raw_post = (H_all[N + i] - bias_terms[N + i]) / scale_post
        H_all_raw_post_val = (H_all_post_val - bias_post_val) / post_scale

        # H_post_raw = H_all_raw_post * scale_post + bias_terms[N + i]
        H_post_raw_val = H_all_raw_post_val * post_scale + bias_post_val

        # Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
        sigmoid_post = 1.0 / (1.0 + tl.exp(-H_post_raw_val))
        sigmoid_derived_post = sigmoid_post * (1.0 - sigmoid_post)

        # Load d_H_post[i]
        d_H_post_val = tl.load(d_H_post_ptr + pid * N + i).to(tl.float32)

        # d_H_post_raw = d_H_post * 2.0 * sigmoid_derived
        d_H_post_raw_val = d_H_post_val * 2.0 * sigmoid_derived_post

        # d_H_post_scaled = d_H_post_raw * scale_post
        d_H_post_scaled_val = d_H_post_raw_val * post_scale

        # Accumulate d_post_scale
        d_post_scale_acc += d_H_post_raw_val * H_all_raw_post_val

        # Merge d_H_post into d_H_all[N+i] and accumulate bias gradient (values already in register)
        tl.store(d_H_all_ptr + H_all_base_offset + N + i, d_H_post_scaled_val.to(tl.bfloat16))
        tl.atomic_add(d_bias_terms_ptr + N + i, d_H_post_raw_val)

        # ===== H_pre backward =====
        # Load H_all[i]
        H_all_pre_val = tl.load(H_all_ptr + H_all_base_offset + i).to(tl.float32)

        # Load bias_terms[i]
        bias_pre_val = tl.load(bias_terms_ptr + i).to(tl.float32)

        # H_all_raw_pre = (H_all[i] - bias_terms[i]) / scale_pre
        H_all_raw_pre_val = (H_all_pre_val - bias_pre_val) / pre_scale

        # H_pre_raw = H_all_raw_pre * scale_pre + bias_terms[i]
        H_pre_raw_val = H_all_raw_pre_val * pre_scale + bias_pre_val

        # Sigmoid derivative
        sigmoid_pre = 1.0 / (1.0 + tl.exp(-H_pre_raw_val))
        sigmoid_derived_pre = sigmoid_pre * (1.0 - sigmoid_pre)

        # Load d_H_pre_from_branch[i]
        d_H_pre_val = tl.load(d_H_pre_from_branch_ptr + pid * N + i).to(tl.float32)

        # d_H_pre_raw = d_H_pre_from_branch * sigmoid_derived
        d_H_pre_raw_val = d_H_pre_val * sigmoid_derived_pre

        # d_H_pre_scaled = d_H_pre_raw * scale_pre
        d_H_pre_scaled_val = d_H_pre_raw_val * pre_scale

        # Accumulate d_pre_scale
        d_pre_scale_acc += d_H_pre_raw_val * H_all_raw_pre_val

        # Merge d_H_pre into d_H_all[i] and accumulate bias gradient (values already in register)
        tl.store(d_H_all_ptr + H_all_base_offset + i, d_H_pre_scaled_val.to(tl.bfloat16))
        tl.atomic_add(d_bias_terms_ptr + i, d_H_pre_raw_val)

    # Store scale gradients using atomic add (index 0: pre_scale, index 1: post_scale)
    tl.atomic_add(d_scaling_factors_ptr + 0, d_pre_scale_acc)
    tl.atomic_add(d_scaling_factors_ptr + 1, d_post_scale_acc)


def width_hres_activation_backward_triton(
    d_H_res_mat,         # [B, L, N, N] - from Kernel 1
    d_H_post,             # [B, L, N] - upper layer gradient
    d_H_pre_from_branch,  # [B, L, N] - from Kernel 1
    H_all,               # [B, L, N+N+N*N]
    H_res_exp,           # [B, L, N, N] - exponential of original H_res (before Sinkhorn)
    u,                   # [B*L, N] - compact left scaling vector (diagonal of U)
    v,                   # [B*L, N] - compact right scaling vector (diagonal of V)
    scaling_factors,      # [3]
    bias_terms,           # [N+N+N*N]
    N,
    skip_sk_gradient=False,  # When True, detach SK gradients (align with width_connection_v2)
):
    """
    2-Kernel Triton implementation of H_res + Activation Backward.

    This function uses 2 Triton kernels that write directly to d_H_all and d_bias_terms
    1. Kernel A: hres_bwd_exp_matmul_kernel - H_res backward -> d_H_all[2N:] + d_bias_terms[2N:]
    2. Kernel B: pre_post_act_bwd_kernel - sigmoid gradients -> d_H_all[:2N] + d_bias_terms[:2N]

    Uses compact vectors u, v instead of full diagonal matrices U, V for efficiency.

    Args:
        d_H_res_mat: [B, L, N, N] - from Kernel 1
        d_H_post: [B, L, N] - upper layer gradient
        d_H_pre_from_branch: [B, L, N] - from Kernel 1
        H_all: [B, L, N+N+N*N]
        H_res_exp: [B, L, N, N] - exponential of original H_res (before Sinkhorn)
        u: [B*L, N] - compact left scaling vector (diagonal of U)
        v: [B*L, N] - compact right scaling vector (diagonal of V)
        scaling_factors: [3]
        bias_terms: [N+N+N*N]
        N: int
        skip_sk_gradient: bool - When True, skip gradient computation through SK
            (equivalent to detaching H_res_exp, U, V in width_connection_v2)

    Returns:
        d_H_all: [B, L, N+N+N*N]
        d_scaling_factors: [3]
        d_bias_terms: [N+N+N*N]
    """
    B, L = d_H_res_mat.shape[:2]
    NN = N * N
    N3 = N + N + NN
    num_tokens = B * L

    # Get actual strides for 4D tensors [B, L, ...]
    # For 4D tensor [B, L, N, N], stride[1] = N*N is the token stride
    # For 4D tensor [B, L, N], stride[1] = N is the token stride
    # For 2D tensor [B*L, N], stride[0] = N is the token stride
    stride_d_H_res_mat = d_H_res_mat.stride(1)  # [B, L, N, N] -> stride[1] = N*N
    stride_H_res_exp = H_res_exp.stride(1)      # [B, L, N, N] -> stride[1] = N*N
    stride_H_all = H_all.stride(1)              # [B, L, N3] -> stride[1] = N3
    stride_u = u.stride(0)                      # [B*L, N] -> stride[0] = N
    stride_v = v.stride(0)                      # [B*L, N] -> stride[0] = N

    # Allocate outputs
    # When skip_sk_gradient=True, initialize to zeros since H_res portion won't be computed
    if skip_sk_gradient:
        d_H_all = paddle.zeros([B, L, N3], dtype='bfloat16')
    else:
        d_H_all = paddle.empty([B, L, N3], dtype='bfloat16')

    # Allocate scalar outputs for scale gradients (pre-allocated [3] tensor to avoid concat)
    # Index 0: d_res_scale, Index 1: d_pre_scale, Index 2: d_post_scale
    d_scaling_factors = paddle.zeros([3], dtype='float32')

    # Allocate bias output (initialize to zero)
    d_bias_terms = paddle.zeros([N3], dtype='float32')

    # 2-Kernel pipeline: Kernel A handles res, Kernel B handles pre/post
    # Both write directly to d_H_all and d_bias_terms (no intermediate buffers)

    # ===== Kernel A: H_res backward (compact) -> d_H_all[2N:] + d_bias_terms[2N:] =====
    # When skip_sk_gradient=True, skip SK gradient computation (align with width_connection_v2)
    if not skip_sk_gradient:
        if N == 4:
            grid_a = (num_tokens,)
            hres_bwd_exp_matmul_compact_kernel[grid_a](
                d_H_res_mat, H_res_exp, u, v,
                H_all, bias_terms, scaling_factors,
                d_scaling_factors, d_H_all, d_bias_terms,
                num_tokens=num_tokens, N=N, NN=NN, N3=N3,
                stride_d_H_res_mat=stride_d_H_res_mat,
                stride_H_res_exp=stride_H_res_exp,
                stride_u=stride_u, stride_v=stride_v, stride_H_all=stride_H_all,
            )
        else:
            # For N!=4, use non-compact Kernel A with reconstructed U and V
            U = paddle.diag_embed(u)  # [num_tokens, N, N]
            V = paddle.diag_embed(v)  # [num_tokens, N, N]

            grid_a = (num_tokens,)
            hres_bwd_exp_matmul_kernel[grid_a](
                d_H_res_mat, H_res_exp, U, V,
                H_all, bias_terms, scaling_factors,
                d_scaling_factors, d_H_all, d_bias_terms,
                num_tokens=num_tokens, N=N, NN=NN, N3=N3,
                stride_d_H_res_mat=stride_d_H_res_mat, stride_H_res_exp=stride_H_res_exp,
                stride_U=NN, stride_V=NN, stride_H_all=stride_H_all,
            )
        # d_scaling_factors[0] (res_scale) and d_bias_terms[2N:] are updated by kernel
    # else: d_H_all[:, :, 2N:] remains zero, d_scaling_factors[0] and d_bias_terms[2N:] remain zero

    # ===== Kernel B: Activation backward -> d_H_all[:2N] + d_bias_terms[:2N] =====
    # This kernel handles H_pre and H_post gradients (not affected by skip_sk_gradient)
    grid_b = (num_tokens,)
    pre_post_act_bwd_kernel[grid_b](
        d_H_post, d_H_pre_from_branch,
        H_all, bias_terms, scaling_factors,
        d_scaling_factors, d_H_all, d_bias_terms,
        num_tokens=num_tokens, N=N, N3=N3, stride_H_all=stride_H_all,
    )

    # d_scaling_factors is already [3] with index 0: res_scale, 1: pre_scale, 2: post_scale

    # d_H_all is already in [B, L, N3] format, no reshape needed
    return d_H_all, d_scaling_factors, d_bias_terms


# ============================================================================
# Backward Kernel with Thread-Level Reduction
# ============================================================================

@triton.autotune(
    configs=[triton.Config({'BLOCK_SIZE_D': 1024}, num_warps=8)],
    key=['D'],
)
@triton.jit
def depth_connection_backward_fused_kernel(
    d_output_ptr,
    H_post_ptr,
    branch_output_ptr,
    d_H_post_ptr,
    d_branch_output_ptr,
    d_residuals_ptr,
    B: tl.constexpr,
    L: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    D_H_POST_DTYPE: tl.constexpr = tl.float32,
    D_BRANCH_DTYPE: tl.constexpr = tl.float32,
):
    """
    Backward kernel WITHOUT atomic_add.

    Computes:
    - d_residuals = d_output
    - d_branch_output = sum over N of (H_post * d_output)
    - d_H_post = sum over D of (branch * d_output)
    """
    pid = tl.program_id(0)

    num_tokens = B * L
    total_d_blocks = tl.cdiv(D, BLOCK_SIZE_D)

    token_idx = pid // total_d_blocks
    d_block_idx = pid % total_d_blocks

    if token_idx >= num_tokens:
        return

    batch_idx = token_idx // L
    seq_idx = token_idx % L

    stride_H = L * N
    stride_branch = L * 1 * D
    stride_grad = L * N * D

    d_start = d_block_idx * BLOCK_SIZE_D
    d_offsets = d_start + tl.arange(0, BLOCK_SIZE_D)
    d_mask = d_offsets < D

    d_branch_acc = tl.zeros([BLOCK_SIZE_D], dtype=tl.float32)

    for stream_idx in range(N):
        h_post_offset = batch_idx * stride_H + seq_idx * N + stream_idx
        h_post_val = tl.load(H_post_ptr + h_post_offset)

        d_output_offset = batch_idx * stride_grad + seq_idx * N * D + stream_idx * D
        d_output_vals = tl.load(
            d_output_ptr + d_output_offset + d_offsets,
            mask=d_mask,
            other=0.0
        )

        tl.store(
            d_residuals_ptr + d_output_offset + d_offsets,
            d_output_vals,
            mask=d_mask
        )

        d_branch_acc += h_post_val * d_output_vals

        branch_offset = batch_idx * stride_branch + seq_idx * D
        branch_vals = tl.load(
            branch_output_ptr + branch_offset + d_offsets,
            mask=d_mask,
            other=0.0
        )

        d_h_post_val = tl.sum(branch_vals * d_output_vals)

        tl.store(d_H_post_ptr + h_post_offset, d_h_post_val.to(D_H_POST_DTYPE))

    branch_offset = batch_idx * stride_branch + seq_idx * D
    tl.store(
            d_branch_output_ptr + branch_offset + d_offsets,
            d_branch_acc.to(D_BRANCH_DTYPE),
            mask=d_mask
        )


def depth_connection_backward_triton_fused(d_output, H_post, branch_output):
    """
    Backward pass WITHOUT atomic_add.
    """
    B, L, N, D = d_output.shape

    # Get dtype directly from input tensors
    H_post_dtype = H_post.dtype
    branch_output_dtype = branch_output.dtype

    # Paddle dtype -> Triton dtype mapping
    def get_triton_dtype(paddle_dtype):
        if paddle_dtype == paddle.bfloat16:
            return tl.bfloat16
        elif paddle_dtype == paddle.float16:
            return tl.float16
        return tl.float32

    d_H_post_triton_dtype = get_triton_dtype(H_post_dtype)
    d_branch_triton_dtype = get_triton_dtype(branch_output_dtype)

    d_H_post = paddle.zeros([B, L, N], dtype=H_post_dtype)
    d_branch_output = paddle.zeros([B, L, 1, D], dtype=branch_output_dtype)
    d_residuals = paddle.zeros([B, L, N, D], dtype=d_output.dtype)

    num_tokens = B * L
    BLOCK_SIZE_D_default = 1024
    num_d_blocks = (D + BLOCK_SIZE_D_default - 1) // BLOCK_SIZE_D_default
    grid = (num_tokens * num_d_blocks,)

    depth_connection_backward_fused_kernel[grid](
        d_output,
        H_post,
        branch_output,
        d_H_post,
        d_branch_output,
        d_residuals,
        B=B,
        L=L,
        N=N,
        D=D,
        D_H_POST_DTYPE=d_H_post_triton_dtype,
        D_BRANCH_DTYPE=d_branch_triton_dtype,
    )

    return d_H_post, d_branch_output, d_residuals


# ============================================================================
# Depth Connection Backward - Split Kernel Implementation (Scheme A)
# Separated into two kernels to avoid concurrent write conflicts, can use empty memory
# ============================================================================

@triton.jit
def depth_conn_bwd_residuals_kernel(
    d_output_ptr,
    H_post_ptr,
    d_residuals_ptr,
    d_branch_output_ptr,
    B: tl.constexpr,
    L: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    D_BRANCH_DTYPE: tl.constexpr = tl.float32,
):
    """
    Kernel 1: Compute d_residuals and d_branch_output

    Grid: (B * L * num_d_blocks,)
    - Each program handles one (batch_idx, seq_idx, d_block_idx)
    - Iterate over N streams, compute contributions for current D block
    - Each output position is written by only one program, can use empty memory

    Computes:
    - d_residuals = d_output
    - d_branch_output = sum over N of (H_post * d_output)

    Note: This kernel does not use autotune because grid size must be determined
          on the host side, BLOCK_SIZE_D needs to be consistent with host side.
    """
    pid = tl.program_id(0)

    num_tokens = B * L
    num_d_blocks = tl.cdiv(D, BLOCK_SIZE_D)

    token_idx = pid // num_d_blocks
    d_block_idx = pid % num_d_blocks

    if token_idx >= num_tokens:
        return

    batch_idx = token_idx // L
    seq_idx = token_idx % L

    d_start = d_block_idx * BLOCK_SIZE_D
    d_offsets = d_start + tl.arange(0, BLOCK_SIZE_D)
    d_mask = d_offsets < D

    # Accumulator for d_branch in current D block
    d_branch_acc = tl.zeros([BLOCK_SIZE_D], dtype=tl.float32)

    # Iterate over all N streams
    for stream_idx in range(N):
        # Load H_post[b, l, stream_idx] scalar
        h_post_offset = batch_idx * L * N + seq_idx * N + stream_idx
        h_post_val = tl.load(H_post_ptr + h_post_offset)

        # Load d_output[b, l, stream_idx, d_start:d_end]
        d_output_offset = batch_idx * L * N * D + seq_idx * N * D + stream_idx * D
        d_output_vals = tl.load(
            d_output_ptr + d_output_offset + d_offsets,
            mask=d_mask,
            other=0.0
        )

        # Write d_residuals = d_output (each position written only once)
        tl.store(
            d_residuals_ptr + d_output_offset + d_offsets,
            d_output_vals,
            mask=d_mask
        )

        # Accumulate d_branch_output = sum_n(H_post * d_output)
        d_branch_acc += h_post_val * d_output_vals

    # Write d_branch_output for current D block (each position written only once)
    branch_offset = batch_idx * L * D + seq_idx * D
    tl.store(
        d_branch_output_ptr + branch_offset + d_offsets,
        d_branch_acc.to(D_BRANCH_DTYPE),
        mask=d_mask
    )


@triton.autotune(
    configs=[triton.Config({'BLOCK_SIZE_D': 1024}, num_warps=8)],
    key=['D'],
)
@triton.jit
def depth_conn_bwd_hpost_kernel(
    d_output_ptr,
    branch_output_ptr,
    d_H_post_ptr,
    B: tl.constexpr,
    L: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    D_H_POST_DTYPE: tl.constexpr = tl.float32,
):
    """
    Kernel 2: Compute d_H_post

    Grid: (B * L * N,)
    - Each program handles one (batch_idx, seq_idx, stream_idx) for the full D dimension
    - Loop over D blocks, accumulate reduction results
    - Each d_H_post position is written by only one program, can use empty memory

    Computes:
    - d_H_post = sum over D of (branch * d_output)
    """
    pid = tl.program_id(0)

    total_elements = B * L * N
    if pid >= total_elements:
        return

    # Decode 3D indices
    token_stream_idx = pid // N
    stream_idx = pid % N
    batch_idx = token_stream_idx // L
    seq_idx = token_stream_idx % L

    # d_H_post scalar accumulator
    d_h_post_acc = 0.0

    # Iterate over all D blocks
    num_d_blocks = tl.cdiv(D, BLOCK_SIZE_D)
    for d_block_idx in range(num_d_blocks):
        d_start = d_block_idx * BLOCK_SIZE_D
        d_offsets = d_start + tl.arange(0, BLOCK_SIZE_D)
        d_mask = d_offsets < D

        # Load branch_output[b, l, 0, d_start:d_end]
        branch_offset = batch_idx * L * D + seq_idx * D
        branch_vals = tl.load(
            branch_output_ptr + branch_offset + d_offsets,
            mask=d_mask,
            other=0.0
        )

        # Load d_output[b, l, stream_idx, d_start:d_end]
        d_output_offset = batch_idx * L * N * D + seq_idx * N * D + stream_idx * D
        d_output_vals = tl.load(
            d_output_ptr + d_output_offset + d_offsets,
            mask=d_mask,
            other=0.0
        )

        # Accumulate contribution from current D block
        d_h_post_acc += tl.sum(branch_vals * d_output_vals)

    # Write d_H_post[b, l, stream_idx] (each position written only once)
    h_post_offset = batch_idx * L * N + seq_idx * N + stream_idx
    tl.store(d_H_post_ptr + h_post_offset, d_h_post_acc.to(D_H_POST_DTYPE))


# ============================================================================
# Depth Connection Backward - Fused Single Kernel (Optimized)
# Single kernel completes all computations, grid = (B * L,)
# ============================================================================

@triton.jit
def depth_conn_bwd_fused_v2_kernel(
    d_output_ptr,
    H_post_ptr,
    branch_output_ptr,
    d_H_post_ptr,
    d_branch_output_ptr,
    d_residuals_ptr,
    B: tl.constexpr,
    L: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    D_H_POST_DTYPE: tl.constexpr = tl.float32,
    D_BRANCH_DTYPE: tl.constexpr = tl.float32,
):
    """
    Optimized single kernel implementation - all computations in one launch.

    Grid: (B * L,)
    - Each program handles one (batch_idx, seq_idx) for the full N×D computation
    - Block-wise iteration over D, accumulate d_branch and d_H_post in registers
    - One-time write for all outputs, no concurrent write conflicts

    Computes:
    - d_residuals = d_output
    - d_branch_output = sum over N of (H_post * d_output)
    - d_H_post = sum over D of (branch * d_output)

    Note: This kernel assumes N=4, avoids dynamic indexing issues by manual loop unrolling.
    """
    pid = tl.program_id(0)

    num_tokens = B * L
    if pid >= num_tokens:
        return

    batch_idx = pid // L
    seq_idx = pid % L

    # Preload H_post[b, l, :] scalar values
    h_post_offset = batch_idx * L * N + seq_idx * N
    h_post_0 = tl.load(H_post_ptr + h_post_offset + 0)
    h_post_1 = tl.load(H_post_ptr + h_post_offset + 1)
    h_post_2 = tl.load(H_post_ptr + h_post_offset + 2)
    h_post_3 = tl.load(H_post_ptr + h_post_offset + 3)

    # d_H_post accumulators - N independent scalars
    d_h_post_0 = 0.0
    d_h_post_1 = 0.0
    d_h_post_2 = 0.0
    d_h_post_3 = 0.0

    # Base address offsets
    base_offset = batch_idx * L * N * D + seq_idx * N * D
    branch_offset = batch_idx * L * D + seq_idx * D

    # Iterate over all D blocks
    num_d_blocks = tl.cdiv(D, BLOCK_SIZE_D)
    for d_block_idx in range(num_d_blocks):
        d_start = d_block_idx * BLOCK_SIZE_D
        d_offsets = d_start + tl.arange(0, BLOCK_SIZE_D)
        d_mask = d_offsets < D

        # Load branch_output[b, l, 0, d_start:d_end]
        branch_vals = tl.load(
            branch_output_ptr + branch_offset + d_offsets,
            mask=d_mask,
            other=0.0
        )

        # Accumulator for d_branch in current D block
        d_branch_acc = tl.zeros([BLOCK_SIZE_D], dtype=tl.float32)

        # Stream 0
        d_output_vals = tl.load(d_output_ptr + base_offset + 0 * D + d_offsets, mask=d_mask, other=0.0)
        tl.store(d_residuals_ptr + base_offset + 0 * D + d_offsets, d_output_vals, mask=d_mask)
        d_branch_acc += h_post_0 * d_output_vals
        d_h_post_0 += tl.sum(branch_vals * d_output_vals)

        # Stream 1
        d_output_vals = tl.load(d_output_ptr + base_offset + 1 * D + d_offsets, mask=d_mask, other=0.0)
        tl.store(d_residuals_ptr + base_offset + 1 * D + d_offsets, d_output_vals, mask=d_mask)
        d_branch_acc += h_post_1 * d_output_vals
        d_h_post_1 += tl.sum(branch_vals * d_output_vals)

        # Stream 2
        d_output_vals = tl.load(d_output_ptr + base_offset + 2 * D + d_offsets, mask=d_mask, other=0.0)
        tl.store(d_residuals_ptr + base_offset + 2 * D + d_offsets, d_output_vals, mask=d_mask)
        d_branch_acc += h_post_2 * d_output_vals
        d_h_post_2 += tl.sum(branch_vals * d_output_vals)

        # Stream 3
        d_output_vals = tl.load(d_output_ptr + base_offset + 3 * D + d_offsets, mask=d_mask, other=0.0)
        tl.store(d_residuals_ptr + base_offset + 3 * D + d_offsets, d_output_vals, mask=d_mask)
        d_branch_acc += h_post_3 * d_output_vals
        d_h_post_3 += tl.sum(branch_vals * d_output_vals)

        # Write d_branch_output for current D block
        tl.store(
            d_branch_output_ptr + branch_offset + d_offsets,
            d_branch_acc.to(D_BRANCH_DTYPE),
            mask=d_mask
        )

    # Write d_H_post[b, l, :] vector
    tl.store(d_H_post_ptr + h_post_offset + 0, d_h_post_0.to(D_H_POST_DTYPE))
    tl.store(d_H_post_ptr + h_post_offset + 1, d_h_post_1.to(D_H_POST_DTYPE))
    tl.store(d_H_post_ptr + h_post_offset + 2, d_h_post_2.to(D_H_POST_DTYPE))
    tl.store(d_H_post_ptr + h_post_offset + 3, d_h_post_3.to(D_H_POST_DTYPE))


def depth_connection_backward_triton_fused_optimized(d_output, H_post, branch_output):
    """
    Optimized single kernel implementation - avoids double launch overhead.

    Advantages:
    - Single kernel launch
    - Memory read optimization: d_output and branch_output are read only once
    - Can use paddle.empty for memory allocation

    Note: This implementation assumes N=4. To support other N values, the kernel needs modification.
    """
    B, L, N, D = d_output.shape

    assert N == 4, f"depth_connection_backward_triton_fused_optimized only supports N=4, got N={N}"

    # Paddle dtype -> Triton dtype mapping
    def get_triton_dtype(paddle_dtype):
        if paddle_dtype == paddle.bfloat16:
            return tl.bfloat16
        elif paddle_dtype == paddle.float16:
            return tl.float16
        return tl.float32

    H_post_dtype = H_post.dtype
    branch_output_dtype = branch_output.dtype
    d_H_post_triton_dtype = get_triton_dtype(H_post_dtype)
    d_branch_triton_dtype = get_triton_dtype(branch_output_dtype)

    # Use empty for memory allocation (each position written only once)
    d_H_post = paddle.empty([B, L, N], dtype=H_post_dtype)
    d_branch_output = paddle.empty([B, L, 1, D], dtype=branch_output_dtype)
    # INPLACE OPTIMIZATION: d_residuals reuses d_output's memory
    # Since d_residuals = d_output (direct copy), and the kernel loads before stores,
    # each stream handles different memory locations, so inplace is safe
    d_residuals = d_output

    # Single kernel call
    num_tokens = B * L
    grid = (num_tokens,)

    # Use larger BLOCK_SIZE_D to reduce loop iterations
    BLOCK_SIZE_D = min(2048, D)

    depth_conn_bwd_fused_v2_kernel[grid](
        d_output,
        H_post,
        branch_output,
        d_H_post,
        d_branch_output,
        d_residuals,
        B=B,
        L=L,
        N=N,
        D=D,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        D_H_POST_DTYPE=d_H_post_triton_dtype,
        D_BRANCH_DTYPE=d_branch_triton_dtype,
    )

    return d_H_post, d_branch_output, d_residuals


def depth_connection_backward_triton_split(d_output, H_post, branch_output):
    """
    Backward pass using two separate kernels to avoid concurrent write conflicts, can use empty memory.

    Scheme A implementation:
    - Kernel 1: Compute d_residuals and d_branch_output, Grid = (B * L * num_d_blocks,)
    - Kernel 2: Compute d_H_post, Grid = (B * L * N,)

    Advantages:
    - Completely avoids concurrent writes and atomic_add
    - Can use paddle.empty for memory allocation
    - Both kernels have high parallelism
    """
    B, L, N, D = d_output.shape

    # Paddle dtype -> Triton dtype mapping
    def get_triton_dtype(paddle_dtype):
        if paddle_dtype == paddle.bfloat16:
            return tl.bfloat16
        elif paddle_dtype == paddle.float16:
            return tl.float16
        return tl.float32

    H_post_dtype = H_post.dtype
    branch_output_dtype = branch_output.dtype
    d_H_post_triton_dtype = get_triton_dtype(H_post_dtype)
    d_branch_triton_dtype = get_triton_dtype(branch_output_dtype)

    # Use empty for memory allocation (each position written only once)
    d_H_post = paddle.empty([B, L, N], dtype=H_post_dtype)
    d_branch_output = paddle.empty([B, L, 1, D], dtype=branch_output_dtype)
    # INPLACE OPTIMIZATION: d_residuals reuses d_output's memory
    # Since d_residuals = d_output (direct copy), residuals_kernel loads before stores,
    # hpost_kernel reads d_output which has been overwritten with the same value, no impact on correctness
    d_residuals = d_output

    # Kernel 1: Compute d_residuals and d_branch_output
    num_tokens = B * L
    BLOCK_SIZE_D_default = 1024
    num_d_blocks = (D + BLOCK_SIZE_D_default - 1) // BLOCK_SIZE_D_default
    grid1 = (num_tokens * num_d_blocks,)

    depth_conn_bwd_residuals_kernel[grid1](
        d_output,
        H_post,
        d_residuals,
        d_branch_output,
        B=B,
        L=L,
        N=N,
        D=D,
        BLOCK_SIZE_D=BLOCK_SIZE_D_default,
        D_BRANCH_DTYPE=d_branch_triton_dtype,
    )

    # Kernel 2: Compute d_H_post
    grid2 = (B * L * N,)

    depth_conn_bwd_hpost_kernel[grid2](
        d_output,
        branch_output,
        d_H_post,
        B=B,
        L=L,
        N=N,
        D=D,
        D_H_POST_DTYPE=d_H_post_triton_dtype,
    )

    return d_H_post, d_branch_output, d_residuals


# ============================================================================
# PyLayer Wrapper Classes
# ============================================================================

class DepthConnectionLayerTriton(paddle.autograd.PyLayer):
    """Triton-accelerated PyLayer for depth connection."""

    @staticmethod
    def forward(ctx, H_post, branch_output, residuals):
        """Forward pass using optimized Triton kernel."""
        paddle.base.core.nvprof_nvtx_push("mhc depth_connection")
        output = depth_connection_forward_triton_optimized(H_post, branch_output, residuals)

        if output.dtype != branch_output.dtype:
            output = output.cast(branch_output.dtype)

        ctx.save_for_backward(H_post, branch_output)
        ctx.B, ctx.L, ctx.N, ctx.D = output.shape
        paddle.base.core.nvprof_nvtx_pop()
        return output

    @staticmethod
    def backward(ctx, d_output):
        """Backward pass using optimized Triton kernel."""
        paddle.base.core.nvprof_nvtx_push("mhc depth_connection backward")
        saved = ctx.saved_tensor()
        H_post, branch_output = saved
        B, L, N, D = ctx.B, ctx.L, ctx.N, ctx.D

        # Use optimized single kernel version (only supports N=4)
        if N == 4:
            d_H_post, d_branch_output, d_residuals = depth_connection_backward_triton_fused_optimized(
                d_output, H_post, branch_output
            )
        else:
            # Fallback to split version
            d_H_post, d_branch_output, d_residuals = depth_connection_backward_triton_split(
                d_output, H_post, branch_output
            )

        paddle.base.core.nvprof_nvtx_pop()

        return (d_H_post,
                d_branch_output,
                d_residuals)


class WidthConnectionLayerTriton(paddle.autograd.PyLayer):
    """Triton-accelerated PyLayer implementation of width_connection.

    Note: This class uses norm_weight=1.0 (no learnable weight for RMSNorm).
    """

    @staticmethod
    def forward(ctx, x, combined_weights, scaling_factors, bias_terms,
               norm_eps, max_sinkhorn_iters, N, D,
               skip_sk_gradient=True):
        """
        Forward pass using optimized fused operations.

        Args:
            skip_sk_gradient: When True, detach Sinkhorn-Knopp gradients in backward
                (align with width_connection_v2 behavior for numerical stability)
        """
        paddle.base.core.nvprof_nvtx_push("mhc width_connection")
        B, L = x.shape[:2]
        x_dtype = x.dtype
        NN = N * N

        h_dtype = paddle.float32
        H_pre = paddle.empty((B, L, N), dtype=h_dtype)
        H_post = paddle.empty((B, L, N), dtype=h_dtype)
        H_res_exp = paddle.empty((B, L, N, N), dtype=h_dtype)

        # Get H_all from kernel (direct 4D I/O, no reshape needed)
        # norm_weight=1.0 (no learnable weight for RMSNorm)
        H_all = width_rmsnorm_gemm_forward(
            x, combined_weights, scaling_factors,
            bias_terms, norm_eps, N, D, B, L
        )

        H_pre, H_post, H_res_exp = mhc_fuse_sigmoid_exp(H_all, H_pre, H_post, H_res_exp, N)

        u, v = triton_sinkhorn_knopp_compact(H_res_exp, max_sinkhorn_iters)

        # Use compact forward function (no diag_embed overhead)
        residuals, branch_input, H_res = post_sinkhorn_fused_forward(
            H_res_exp, u, v, H_pre, x
        )

        ctx.save_for_backward(
            x, H_all, combined_weights, scaling_factors, bias_terms,
            H_pre, H_res_exp, H_res, u.detach(), v.detach()
        )

        ctx.norm_eps = norm_eps
        ctx.N = N
        ctx.D = D
        ctx.B = B
        ctx.L = L
        ctx.x_dtype = x_dtype
        ctx.skip_sk_gradient = skip_sk_gradient

        paddle.base.core.nvprof_nvtx_pop()

        return branch_input, residuals, H_post

    @staticmethod
    def backward(ctx, d_branch_input, d_residuals, d_H_post):
        """
        Backward pass using integrated Triton kernels.

        """
        try:
            saved = ctx.saved_tensor()
        except Exception as e:
            print(f"ERROR getting saved_tensor: {e}")
            raise
        paddle.base.core.nvprof_nvtx_push("mhc width_connection backward (Triton)")
        (x, H_all, combined_weights, scaling_factors, bias_terms,
         H_pre, H_res_exp, H_res, u, v) = saved

        B, L, N, D = ctx.B, ctx.L, ctx.N, ctx.D
        NN = N * N
        N3 = N + N + NN
        ND = N * D

        # ===== Kernel 1: Branch & Residuals Backward =====
        # Compute: d_H_pre_from_branch, d_H_res_mat, d_x_branch_add_residuals (fused)
        d_H_pre_from_branch, d_H_res_mat, d_x_branch_add_residuals = width_branch_residuals_backward_triton(
            d_branch_input, d_residuals, x, H_pre, H_res
        )

        # ===== Kernel 2: H_res + Activation Backward =====
        # Compute: d_H_all, d_scaling_factors, d_bias_terms using compact vectors
        d_H_all, d_scaling_factors, d_bias_terms = width_hres_activation_backward_triton(
            d_H_res_mat, d_H_post, d_H_pre_from_branch,
            H_all, H_res_exp, u, v, scaling_factors, bias_terms, N,
            skip_sk_gradient=ctx.skip_sk_gradient
        )

        # ===== Kernel 3: Combined_weights + RMSNorm Backward =====
        # Compute: d_x (fused with d_x_branch_add_residuals), d_combined_weights
        # norm_weight=1.0 (no learnable weight for RMSNorm)
        d_x, d_combined_weights = width_rmsnorm_gemm_backward_triton(
            d_H_all, x, combined_weights, ctx.norm_eps, N, D, d_x_branch_add_residuals
        )

        paddle.base.core.nvprof_nvtx_pop()

        # Return gradients for tensor inputs only (4 values)
        # d_x and d_combined_weights already have correct dtype from kernel
        return (d_x,
                d_combined_weights,
                d_scaling_factors,
                d_bias_terms)


# Export symbols
__all__ = [
    'WidthConnectionLayerTriton',
    'DepthConnectionLayerTriton',
    'TRITON_AVAILABLE',
    'sinkhorn_knopp',
]