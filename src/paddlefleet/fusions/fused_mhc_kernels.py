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

"""Fused kernels for mHC (Manifold-Constrained Hyper-Connections).

Uses Triton and cuda.tile (cuTile) kernels when available, with PaddlePaddle
reference implementations as fallback.  Reference (non-fused) implementations
live in ``paddlefleet.transformer.hyper_connection`` and are used when fused
kernels are unavailable or when the ``use_fused_mhc`` config flag is False.

Five fused operations:
  - sinkhorn:          Sinkhorn-Knopp projection to doubly stochastic matrix
  - h_aggregate:       weighted n-stream -> 1-stream aggregation
  - h_post_bda:        fused H_res @ residual + H_post * (x + bias)
  - proj_rms:          fused projection + RMS normalization
  - proj_rms_compute_h: fused projection + RMS norm + compute_h activations
"""

import logging
import math

import paddle
from paddle import Tensor

from ..triton_ops.utils import is_torch_compat_available

if is_torch_compat_available():
    paddle.enable_compat(scope={"triton"})

logger = logging.getLogger(__name__)
LOG2E = math.log2(math.e)
_INT32_MAX = 2**31 - 1


# ---------------------------------------------------------------------------
# Check Triton availability
# ---------------------------------------------------------------------------
_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Check cuTile availability
# ---------------------------------------------------------------------------
_CUTILE_AVAILABLE = False
try:
    import cuda.tile as ct

    _CUTILE_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Backend selection via environment variables
# ---------------------------------------------------------------------------


def is_triton_available() -> bool:
    """Return True if Triton is enabled for supported mHC kernels."""
    return _TRITON_AVAILABLE


def is_cutile_available() -> bool:
    """Return True if cuTile fused kernels are available."""
    return _CUTILE_AVAILABLE


def _get_cuda_stream():
    """Get current CUDA stream for cuTile launch."""
    return paddle.device.current_stream().stream_base.cuda_stream


# ============================================================================
# Triton implementations (only defined when triton is available)
# ============================================================================

if _TRITON_AVAILABLE:
    TLOG2E = tl.constexpr(LOG2E)

    # -- Sinkhorn-Knopp --------------------------------------------------------

    @triton.jit
    def _triton_sinkhorn_fwd_kernel(
        inp_ptr,
        out_ptr,
        M_init_ptr,
        N_batch,
        eps,
        HC: tl.constexpr,
        NUM_ITERS: tl.constexpr,
    ):
        """Grid: (N_batch,). Each program handles one [HC, HC] matrix."""
        pid = tl.program_id(0).to(tl.int64)
        if pid >= N_batch:
            return
        base = pid * HC * HC
        offs_r = tl.arange(0, HC)
        offs_c = tl.arange(0, HC)
        mat_ptrs = base + offs_r[:, None] * HC + offs_c[None, :]
        logits = tl.load(inp_ptr + mat_ptrs).to(tl.float32)
        row_max = tl.max(logits, axis=1)
        M = tl.exp2((logits - row_max[:, None]) * TLOG2E)
        tl.store(M_init_ptr + mat_ptrs, M.to(M_init_ptr.dtype.element_ty))
        row_sum = tl.sum(M, axis=1)
        M = M / row_sum[:, None] + eps
        col_sum = tl.sum(M, axis=0)
        M = M / (col_sum[None, :] + eps)
        for _ in range(NUM_ITERS - 1):
            row_sum = tl.sum(M, axis=1)
            M = M / (row_sum[:, None] + eps)
            col_sum = tl.sum(M, axis=0)
            M = M / (col_sum[None, :] + eps)
        tl.store(out_ptr + mat_ptrs, M.to(out_ptr.dtype.element_ty))

    @triton.jit
    def _triton_sinkhorn_bwd_kernel(
        grad_out_ptr,
        M_init_ptr,
        grad_inp_ptr,
        ws_M_ptr,
        ws_rs_ptr,
        ws_cs_ptr,
        N_batch,
        eps,
        HC: tl.constexpr,
        NUM_ITERS: tl.constexpr,
    ):
        """Grid: (N_batch,). Each program handles one [HC, HC] backward."""
        pid = tl.program_id(0).to(tl.int64)
        if pid >= N_batch:
            return
        base = pid * HC * HC
        M_ws_base = pid * 2 * NUM_ITERS * HC * HC
        v_ws_base = pid * NUM_ITERS
        offs_r = tl.arange(0, HC)
        offs_c = tl.arange(0, HC)
        mat_ptrs = base + offs_r[:, None] * HC + offs_c[None, :]
        M = tl.load(M_init_ptr + mat_ptrs).to(tl.float32)
        for t in range(NUM_ITERS):
            ws_off = M_ws_base + (2 * t) * HC * HC
            tl.store(
                ws_M_ptr + ws_off + offs_r[:, None] * HC + offs_c[None, :], M
            )
            row_sum = tl.sum(M, axis=1)
            tl.store(ws_rs_ptr + (v_ws_base + t) * HC + offs_r, row_sum)
            if t == 0:
                M = M / row_sum[:, None] + eps
            else:
                M = M / (row_sum[:, None] + eps)
            ws_off = M_ws_base + (2 * t + 1) * HC * HC
            tl.store(
                ws_M_ptr + ws_off + offs_r[:, None] * HC + offs_c[None, :], M
            )
            col_sum = tl.sum(M, axis=0)
            tl.store(ws_cs_ptr + (v_ws_base + t) * HC + offs_c, col_sum)
            M = M / (col_sum[None, :] + eps)
        grad = tl.load(grad_out_ptr + mat_ptrs).to(tl.float32)
        for t_rev in range(NUM_ITERS):
            t = NUM_ITERS - 1 - t_rev
            col_s = tl.load(ws_cs_ptr + (v_ws_base + t) * HC + offs_c).to(
                tl.float32
            )
            grad = grad / (col_s[None, :] + eps)
            col_corr = tl.sum(grad * M, axis=0)
            grad = grad - col_corr[None, :]
            M = tl.load(
                ws_M_ptr
                + M_ws_base
                + (2 * t + 1) * HC * HC
                + offs_r[:, None] * HC
                + offs_c[None, :]
            ).to(tl.float32)
            row_s = tl.load(ws_rs_ptr + (v_ws_base + t) * HC + offs_r).to(
                tl.float32
            )
            if t == 0:
                grad = grad / row_s[:, None]
                row_corr = tl.sum(grad * (M - eps), axis=1)
            else:
                grad = grad / (row_s[:, None] + eps)
                row_corr = tl.sum(grad * M, axis=1)
            grad = grad - row_corr[:, None]
            M = tl.load(
                ws_M_ptr
                + M_ws_base
                + (2 * t) * HC * HC
                + offs_r[:, None] * HC
                + offs_c[None, :]
            ).to(tl.float32)
        M_init = tl.load(M_init_ptr + mat_ptrs).to(tl.float32)
        grad = grad * M_init
        tl.store(
            grad_inp_ptr + mat_ptrs, grad.to(grad_inp_ptr.dtype.element_ty)
        )

    def _triton_sinkhorn_fwd(
        input_logits: Tensor, num_iterations: int, eps: float = 1e-6
    ) -> tuple[Tensor, Tensor]:
        original_shape = input_logits.shape
        hc = original_shape[-1]
        N_batch = input_logits.size // (hc * hc)
        out = paddle.empty(shape=[N_batch, hc, hc], dtype=input_logits.dtype)
        M_init = paddle.empty(shape=[N_batch, hc, hc], dtype=input_logits.dtype)
        inp = input_logits.reshape([N_batch, hc, hc])
        _triton_sinkhorn_fwd_kernel[(N_batch,)](
            inp, out, M_init, N_batch, eps, hc, num_iterations
        )
        return out.reshape(original_shape), M_init.reshape(original_shape)

    def _triton_sinkhorn_bwd(
        grad_output: Tensor,
        M_init: Tensor,
        num_iterations: int,
        eps: float = 1e-6,
    ) -> Tensor:
        original_shape = grad_output.shape
        hc = original_shape[-1]
        N_batch = grad_output.size // (hc * hc)
        grad_input = paddle.empty(
            shape=[N_batch, hc, hc], dtype=grad_output.dtype
        )
        go = grad_output.reshape([N_batch, hc, hc])
        mi = M_init.reshape([N_batch, hc, hc])
        ws_M = paddle.empty(
            shape=[N_batch * 2 * num_iterations * hc * hc], dtype="float32"
        )
        ws_rs = paddle.empty(
            shape=[N_batch * num_iterations * hc], dtype="float32"
        )
        ws_cs = paddle.empty(
            shape=[N_batch * num_iterations * hc], dtype="float32"
        )
        _triton_sinkhorn_bwd_kernel[(N_batch,)](
            go,
            mi,
            grad_input,
            ws_M,
            ws_rs,
            ws_cs,
            N_batch,
            eps,
            hc,
            num_iterations,
        )
        return grad_input.reshape(original_shape)

    class TritonFusedSinkhorn(paddle.autograd.PyLayer):
        """Autograd wrapper for Triton fused Sinkhorn."""

        @staticmethod
        def forward(
            ctx, input_logits: Tensor, num_iterations: int, eps: float = 1e-6
        ):
            """Run Triton Sinkhorn forward and save initial matrix for backward."""
            out, M_init = _triton_sinkhorn_fwd(
                input_logits, num_iterations, eps
            )
            ctx.save_for_backward(M_init)
            ctx.num_iterations = num_iterations
            ctx.eps = eps
            return out

        @staticmethod
        def backward(ctx, grad_output: Tensor):
            """Run Triton Sinkhorn backward."""
            (M_init,) = ctx.saved_tensor()
            grad_input = _triton_sinkhorn_bwd(
                grad_output, M_init, ctx.num_iterations, ctx.eps
            )
            return grad_input

    def triton_fused_sinkhorn(
        input_logits: Tensor, num_iterations: int, eps: float = 1e-6
    ) -> Tensor:
        """Apply Triton fused Sinkhorn with autograd support."""
        return TritonFusedSinkhorn.apply(input_logits, num_iterations, eps)

    # -- H_aggregate forward ---------------------------------------------------

    @triton.jit
    def _triton_h_agg_fwd_kernel(
        x_ptr,
        h_ptr,
        out_ptr,
        sb,
        C: tl.constexpr,
        N: tl.constexpr,
        stride_x_s,
        stride_x_n,
        stride_x_c,
        BLOCK_C: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        """out[s, c] = sum_i x[s, i, c] * h[s, i]."""
        pid_s = tl.program_id(0).to(tl.int64)
        pid_c = tl.program_id(1).to(tl.int64)
        offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_s = offs_s < sb
        mask_c = offs_c < C
        mask_2d = mask_s[:, None] & mask_c[None, :]
        acc = tl.zeros((BLOCK_S, BLOCK_C), dtype=tl.float32)
        for i in tl.static_range(N):
            x_i = tl.load(
                x_ptr
                + offs_s[:, None] * stride_x_s
                + i * stride_x_n
                + offs_c[None, :],
                mask=mask_2d,
                other=0.0,
            ).to(tl.float32)
            h_i = tl.load(h_ptr + offs_s * N + i, mask=mask_s, other=0.0).to(
                tl.float32
            )
            acc += h_i[:, None] * x_i
        tl.store(
            out_ptr + offs_s[:, None] * C + offs_c[None, :],
            acc.to(out_ptr.dtype.element_ty),
            mask=mask_2d,
        )

    def _triton_h_aggregate_fwd(x: Tensor, h_pre: Tensor) -> Tensor:
        s, b, n, C = x.shape
        sb = s * b
        out = paddle.empty(shape=[sb, C], dtype=x.dtype)
        x_flat = x.reshape([sb, n, C])
        h_flat = h_pre.reshape([sb, n])
        BLOCK_C = 256
        BLOCK_S = 4
        grid = (triton.cdiv(sb, BLOCK_S), triton.cdiv(C, BLOCK_C))
        _triton_h_agg_fwd_kernel[grid](
            x_flat,
            h_flat,
            out,
            sb,
            C,
            n,
            x_flat.stride(0),
            x_flat.stride(1),
            x_flat.stride(2),
            BLOCK_C,
            BLOCK_S,
        )
        return out.reshape([s, b, C])

    # -- H_post BDA ------------------------------------------------------------

    @triton.jit
    def _triton_hpb_fwd_kernel(
        hr_ptr,
        orig_ptr,
        hp_ptr,
        x_ptr,
        bias_ptr,
        out_ptr,
        sb,
        C: tl.constexpr,
        N: tl.constexpr,
        stride_hr_s,
        stride_hr_i,
        stride_hr_j,
        stride_orig_s,
        stride_orig_n,
        stride_orig_c,
        stride_out_s,
        stride_out_n,
        stride_out_c,
        HAS_BIAS: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        """out = hr.T @ orig + hp * (x + bias)."""
        pid_s = tl.program_id(0).to(tl.int64)
        pid_c = tl.program_id(1).to(tl.int64)
        offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_s = offs_s < sb
        mask_c = offs_c < C
        mask_2d = mask_s[:, None] & mask_c[None, :]
        x_tile = tl.load(
            x_ptr + offs_s[:, None] * C + offs_c[None, :],
            mask=mask_2d,
            other=0.0,
        ).to(tl.float32)
        if HAS_BIAS:
            bias_tile = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0).to(
                tl.float32
            )
            x_tile += bias_tile[None, :]
        for i in tl.static_range(N):
            hp_i = tl.load(hp_ptr + offs_s * N + i, mask=mask_s, other=0.0).to(
                tl.float32
            )
            out_i = hp_i[:, None] * x_tile
            for j in tl.static_range(N):
                hr_ji = tl.load(
                    hr_ptr
                    + offs_s * stride_hr_s
                    + j * stride_hr_i
                    + i * stride_hr_j,
                    mask=mask_s,
                    other=0.0,
                ).to(tl.float32)
                orig_j = tl.load(
                    orig_ptr
                    + offs_s[:, None] * stride_orig_s
                    + j * stride_orig_n
                    + offs_c[None, :],
                    mask=mask_2d,
                    other=0.0,
                ).to(tl.float32)
                out_i += hr_ji[:, None] * orig_j
            tl.store(
                out_ptr
                + offs_s[:, None] * stride_out_s
                + i * stride_out_n
                + offs_c[None, :],
                out_i.to(out_ptr.dtype.element_ty),
                mask=mask_2d,
            )

    def _triton_h_post_bda_fwd(
        h_res: Tensor,
        original_residual: Tensor,
        h_post: Tensor,
        x: Tensor,
        bias: Tensor | None,
    ) -> Tensor:
        s, b, n, C = original_residual.shape
        sb = s * b
        out = paddle.empty(shape=[sb, n, C], dtype=h_res.dtype)
        hr_flat = h_res.reshape([sb, n, n])
        orig_flat = original_residual.reshape([sb, n, C])
        hp_flat = h_post.reshape([sb, n])
        x_flat = x.reshape([sb, C])
        BLOCK_C = 256
        BLOCK_S = 4
        grid = (triton.cdiv(sb, BLOCK_S), triton.cdiv(C, BLOCK_C))
        _triton_hpb_fwd_kernel[grid](
            hr_flat,
            orig_flat,
            hp_flat,
            x_flat,
            bias if bias is not None else x_flat,
            out,
            sb,
            C,
            n,
            hr_flat.stride(0),
            hr_flat.stride(1),
            hr_flat.stride(2),  # stride_hr
            orig_flat.stride(0),
            orig_flat.stride(1),
            orig_flat.stride(2),  # stride_orig
            out.stride(0),
            out.stride(1),
            out.stride(2),  # stride_out
            HAS_BIAS=(bias is not None),
            BLOCK_C=BLOCK_C,
            BLOCK_S=BLOCK_S,
        )
        return out.reshape([s, b, n, C])

    @triton.jit
    def _triton_hpb_bwd_g_x_orig_kernel(
        go_ptr,
        hr_ptr,
        hp_ptr,
        g_orig_ptr,
        g_x_ptr,
        sb,
        C: tl.constexpr,
        N: tl.constexpr,
        stride_go_s,
        stride_go_n,
        stride_go_c,
        stride_hr_s,
        stride_hr_i,
        stride_hr_j,
        stride_orig_s,
        stride_orig_n,
        stride_orig_c,
        BLOCK_C: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        """g_x = hp @ go, g_orig = hr @ go."""
        pid_s = tl.program_id(0).to(tl.int64)
        pid_c = tl.program_id(1).to(tl.int64)
        offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_s = offs_s < sb
        mask_c = offs_c < C
        mask_2d = mask_s[:, None] & mask_c[None, :]
        g_x_acc = tl.zeros((BLOCK_S, BLOCK_C), dtype=tl.float32)
        for j in tl.static_range(N):
            go_j = tl.load(
                go_ptr
                + offs_s[:, None] * stride_go_s
                + j * stride_go_n
                + offs_c[None, :],
                mask=mask_2d,
                other=0.0,
            ).to(tl.float32)
            hp_j = tl.load(hp_ptr + offs_s * N + j, mask=mask_s, other=0.0).to(
                tl.float32
            )
            g_x_acc += hp_j[:, None] * go_j
        tl.store(
            g_x_ptr + offs_s[:, None] * C + offs_c[None, :],
            g_x_acc.to(g_x_ptr.dtype.element_ty),
            mask=mask_2d,
        )
        for i in tl.static_range(N):
            g_orig_i = tl.zeros((BLOCK_S, BLOCK_C), dtype=tl.float32)
            for j in tl.static_range(N):
                go_j = tl.load(
                    go_ptr
                    + offs_s[:, None] * stride_go_s
                    + j * stride_go_n
                    + offs_c[None, :],
                    mask=mask_2d,
                    other=0.0,
                ).to(tl.float32)
                hr_ij = tl.load(
                    hr_ptr
                    + offs_s * stride_hr_s
                    + i * stride_hr_i
                    + j * stride_hr_j,
                    mask=mask_s,
                    other=0.0,
                ).to(tl.float32)
                g_orig_i += hr_ij[:, None] * go_j
            tl.store(
                g_orig_ptr
                + offs_s[:, None] * stride_orig_s
                + i * stride_orig_n
                + offs_c[None, :],
                g_orig_i.to(g_orig_ptr.dtype.element_ty),
                mask=mask_2d,
            )

    @triton.jit
    def _triton_hpb_bwd_g_hp_hr_kernel(
        go_ptr,
        orig_ptr,
        x_ptr,
        bias_ptr,
        g_hr_ptr,
        g_hp_ptr,
        sb,
        C: tl.constexpr,
        N: tl.constexpr,
        stride_go_s,
        stride_go_n,
        stride_go_c,
        stride_orig_s,
        stride_orig_n,
        stride_orig_c,
        stride_hr_s,
        stride_hr_i,
        stride_hr_j,
        HAS_BIAS: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        """g_hp = sum_c go*(x+bias), g_hr = orig @ go.T."""
        pid_s = tl.program_id(0).to(tl.int64)
        offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
        mask_s = offs_s < sb
        g_hp_acc = tl.zeros((BLOCK_S, N), dtype=tl.float32)
        g_hr_acc = tl.zeros((BLOCK_S, N * N), dtype=tl.float32)
        for c_start in range(0, C, BLOCK_C):
            offs_c = c_start + tl.arange(0, BLOCK_C)
            mask_c = offs_c < C
            mask_2d = mask_s[:, None] & mask_c[None, :]
            x_tile = tl.load(
                x_ptr + offs_s[:, None] * C + offs_c[None, :],
                mask=mask_2d,
                other=0.0,
            ).to(tl.float32)
            if HAS_BIAS:
                bias_tile = tl.load(
                    bias_ptr + offs_c, mask=mask_c, other=0.0
                ).to(tl.float32)
                x_tile += bias_tile[None, :]
            for i in tl.static_range(N):
                go_i = tl.load(
                    go_ptr
                    + offs_s[:, None] * stride_go_s
                    + i * stride_go_n
                    + offs_c[None, :],
                    mask=mask_2d,
                    other=0.0,
                ).to(tl.float32)
                dot_hp = tl.sum(go_i * x_tile, axis=1)
                g_hp_acc += tl.where(
                    tl.arange(0, N)[None, :] == i,
                    dot_hp[:, None],
                    tl.zeros((BLOCK_S, N), dtype=tl.float32),
                )
                for j in tl.static_range(N):
                    orig_j = tl.load(
                        orig_ptr
                        + offs_s[:, None] * stride_orig_s
                        + j * stride_orig_n
                        + offs_c[None, :],
                        mask=mask_2d,
                        other=0.0,
                    ).to(tl.float32)
                    dot_hr = tl.sum(go_i * orig_j, axis=1)
                    g_hr_acc += tl.where(
                        tl.arange(0, N * N)[None, :] == j * N + i,
                        dot_hr[:, None],
                        tl.zeros((BLOCK_S, N * N), dtype=tl.float32),
                    )
        offs_n = tl.arange(0, N)
        tl.store(
            g_hp_ptr + offs_s[:, None] * N + offs_n[None, :],
            g_hp_acc.to(g_hp_ptr.dtype.element_ty),
            mask=mask_s[:, None],
        )
        nn_offs = tl.arange(0, N * N)
        for i in tl.static_range(N):
            for j in tl.static_range(N):
                col_mask = (nn_offs == (i * N + j)).to(tl.float32)
                val = tl.sum(g_hr_acc * col_mask[None, :], axis=1)
                tl.store(
                    g_hr_ptr
                    + offs_s * stride_hr_s
                    + i * stride_hr_i
                    + j * stride_hr_j,
                    val.to(g_hr_ptr.dtype.element_ty),
                    mask=mask_s,
                )

    def _triton_h_post_bda_bwd(
        grad_output: Tensor,
        h_res: Tensor,
        original_residual: Tensor,
        h_post: Tensor,
        x: Tensor,
        bias: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None]:
        s, b, n, C = original_residual.shape
        sb = s * b
        g_hr = paddle.empty(shape=[sb, n, n], dtype=h_res.dtype)
        g_res = paddle.empty(shape=[sb, n, C], dtype=original_residual.dtype)
        g_hp = paddle.empty(shape=[sb, n], dtype=h_post.dtype)
        g_x = paddle.empty(shape=[sb, C], dtype=x.dtype)
        go_flat = grad_output.reshape([sb, n, C])
        hr_flat = h_res.reshape([sb, n, n])
        orig_flat = original_residual.reshape([sb, n, C])
        hp_flat = h_post.reshape([sb, n])
        x_flat = x.reshape([sb, C])
        BLOCK_C = 256
        BLOCK_S = 4
        grid_a = (triton.cdiv(sb, BLOCK_S), triton.cdiv(C, BLOCK_C))
        _triton_hpb_bwd_g_x_orig_kernel[grid_a](
            go_flat,
            hr_flat,
            hp_flat,
            g_res,
            g_x,
            sb,
            C,
            n,
            go_flat.stride(0),
            go_flat.stride(1),
            go_flat.stride(2),  # stride_go
            hr_flat.stride(0),
            hr_flat.stride(1),
            hr_flat.stride(2),  # stride_hr
            g_res.stride(0),
            g_res.stride(1),
            g_res.stride(2),  # stride_orig
            BLOCK_C=BLOCK_C,
            BLOCK_S=BLOCK_S,
        )
        grid_b = (triton.cdiv(sb, BLOCK_S),)
        _triton_hpb_bwd_g_hp_hr_kernel[grid_b](
            go_flat,
            orig_flat,
            x_flat,
            bias if bias is not None else x_flat,
            g_hr,
            g_hp,
            sb,
            C,
            n,
            go_flat.stride(0),
            go_flat.stride(1),
            go_flat.stride(2),  # stride_go
            orig_flat.stride(0),
            orig_flat.stride(1),
            orig_flat.stride(2),  # stride_orig
            g_hr.stride(0),
            g_hr.stride(1),
            g_hr.stride(2),  # stride_hr
            HAS_BIAS=(bias is not None),
            BLOCK_C=BLOCK_C,
            BLOCK_S=BLOCK_S,
        )
        g_bias = g_x.sum(axis=0).cast(bias.dtype) if bias is not None else None
        return (
            g_hr.reshape([s, b, n, n]),
            g_res.reshape([s, b, n, C]),
            g_hp.reshape([s, b, n]),
            g_x.reshape([s, b, C]),
            g_bias,
        )

    _TRITON_IMPLS = {
        "sinkhorn": triton_fused_sinkhorn,
        "h_aggregate_fwd": _triton_h_aggregate_fwd,
        "h_post_bda_fwd": _triton_h_post_bda_fwd,
        "h_post_bda_bwd": _triton_h_post_bda_bwd,
    }
else:
    _TRITON_IMPLS = {
        "sinkhorn": None,
        "h_aggregate_fwd": None,
        "h_post_bda_fwd": None,
        "h_post_bda_bwd": None,
    }


# ============================================================================
# CuTile implementations (only defined when cuda.tile is available)
# ============================================================================

if _CUTILE_AVAILABLE:
    ConstInt = ct.Constant[int]
    PAD_ZERO = ct.PaddingMode.ZERO

    # -- Sinkhorn kernels ----------------------------------------------------

    @ct.kernel
    def _ct_sinkhorn_fwd_kernel(
        inp,
        out,
        M_init_out,
        eps,
        HC: ConstInt,
        NUM_ITERS: ConstInt,
        TILE_SIZE: ConstInt,
    ):
        pid = ct.bid(0)
        logits = ct.load(
            inp, index=(pid, 0, 0), shape=(TILE_SIZE, HC, HC)
        ).astype(ct.float32)
        row_max = ct.max(logits, axis=2, keepdims=True)
        M = ct.exp2((logits - row_max) * LOG2E)
        ct.store(
            M_init_out,
            index=(pid, 0, 0),
            tile=ct.reshape(M.astype(M_init_out.dtype), (TILE_SIZE, HC, HC)),
        )
        row_sum = ct.sum(M, axis=2, keepdims=True)
        M = M / row_sum + eps
        col_sum = ct.sum(M, axis=1, keepdims=True)
        M = M / (col_sum + eps)
        for _ in range(NUM_ITERS - 1):
            row_sum = ct.sum(M, axis=2, keepdims=True)
            M = M / (row_sum + eps)
            col_sum = ct.sum(M, axis=1, keepdims=True)
            M = M / (col_sum + eps)
        ct.store(
            out,
            index=(pid, 0, 0),
            tile=ct.reshape(M.astype(out.dtype), (TILE_SIZE, HC, HC)),
        )

    @ct.kernel
    def _ct_sinkhorn_bwd_kernel(
        grad_out,
        M_init,
        grad_inp,
        ws_M,
        ws_rs,
        ws_cs,
        eps,
        HC: ConstInt,
        NUM_ITERS: ConstInt,
        TILE_SIZE: ConstInt,
    ):
        pid = ct.bid(0)
        M_base = pid * (2 * NUM_ITERS)
        v_base = pid * NUM_ITERS

        M = ct.load(
            M_init, index=(pid, 0, 0), shape=(TILE_SIZE, HC, HC)
        ).astype(ct.float32)
        for t in range(NUM_ITERS):
            ct.store(ws_M, index=(M_base + 2 * t, 0, 0), tile=M)
            row_sum = ct.sum(M, axis=2, keepdims=True)
            ct.store(ws_rs, index=(v_base + t, 0, 0), tile=row_sum)
            if t == 0:
                M = M / row_sum + eps
            else:
                M = M / (row_sum + eps)
            ct.store(ws_M, index=(M_base + 2 * t + 1, 0, 0), tile=M)
            col_sum = ct.sum(M, axis=1, keepdims=True)
            ct.store(ws_cs, index=(v_base + t, 0, 0), tile=col_sum)
            M = M / (col_sum + eps)

        grad = ct.load(
            grad_out, index=(pid, 0, 0), shape=(TILE_SIZE, HC, HC)
        ).astype(ct.float32)
        for t_rev in range(NUM_ITERS):
            t = NUM_ITERS - 1 - t_rev
            col_s = ct.load(
                ws_cs, index=(v_base + t, 0, 0), shape=(TILE_SIZE, 1, HC)
            )
            grad = grad / (col_s + eps)
            col_corr = ct.sum(grad * M, axis=1, keepdims=True)
            grad = grad - col_corr
            M = ct.load(
                ws_M,
                index=(M_base + 2 * t + 1, 0, 0),
                shape=(TILE_SIZE, HC, HC),
            )
            row_s = ct.load(
                ws_rs, index=(v_base + t, 0, 0), shape=(TILE_SIZE, HC, 1)
            )
            if t == 0:
                grad = grad / row_s
                row_corr = ct.sum(grad * (M - eps), axis=2, keepdims=True)
            else:
                grad = grad / (row_s + eps)
                row_corr = ct.sum(grad * M, axis=2, keepdims=True)
            grad = grad - row_corr
            M = ct.load(
                ws_M, index=(M_base + 2 * t, 0, 0), shape=(TILE_SIZE, HC, HC)
            )
        grad = grad * M
        ct.store(grad_inp, index=(pid, 0, 0), tile=grad.astype(grad_inp.dtype))

    def _cutile_sinkhorn_fwd(
        input_logits: Tensor, num_iterations: int, eps: float = 1e-8
    ) -> tuple[Tensor, Tensor]:
        original_shape = input_logits.shape
        hc = original_shape[-1]
        N_batch = input_logits.size // (hc * hc)
        TILE_SIZE = math.gcd(N_batch, 128)
        out = paddle.empty(shape=[N_batch, hc, hc], dtype=input_logits.dtype)
        M_init = paddle.empty(shape=[N_batch, hc, hc], dtype=input_logits.dtype)
        ct.launch(
            _get_cuda_stream(),
            (math.ceil(N_batch / TILE_SIZE), 1, 1),
            _ct_sinkhorn_fwd_kernel,
            (
                input_logits.reshape([N_batch, hc, hc]),
                out,
                M_init,
                eps,
                hc,
                num_iterations,
                TILE_SIZE,
            ),
        )
        return out.reshape(original_shape), M_init.reshape(original_shape)

    def _cutile_sinkhorn_bwd(
        grad_output: Tensor,
        M_init: Tensor,
        num_iterations: int,
        eps: float = 1e-8,
    ) -> Tensor:
        original_shape = grad_output.shape
        hc = original_shape[-1]
        N_batch = grad_output.size // (hc * hc)
        TILE_SIZE = math.gcd(N_batch, 128)
        ws_M = paddle.empty(
            shape=[N_batch * 2 * num_iterations, hc, hc], dtype="float32"
        )
        ws_rs = paddle.empty(
            shape=[N_batch * num_iterations, hc, 1], dtype="float32"
        )
        ws_cs = paddle.empty(
            shape=[N_batch * num_iterations, 1, hc], dtype="float32"
        )
        grad_input = paddle.empty(
            shape=[N_batch, hc, hc], dtype=grad_output.dtype
        )
        ct.launch(
            _get_cuda_stream(),
            (math.ceil(N_batch / TILE_SIZE), 1, 1),
            _ct_sinkhorn_bwd_kernel,
            (
                grad_output.reshape([N_batch, hc, hc]),
                M_init.reshape([N_batch, hc, hc]),
                grad_input,
                ws_M,
                ws_rs,
                ws_cs,
                eps,
                hc,
                num_iterations,
                TILE_SIZE,
            ),
        )
        return grad_input.reshape(original_shape)

    # -- H_aggregate kernels -------------------------------------------------

    @ct.kernel
    def _ct_h_agg_fwd_kernel(
        x, h_pre, out, N: ConstInt, TILE_M: ConstInt, TILE_C: ConstInt
    ):
        pid = ct.bid(0)
        num_tiles = ct.num_tiles(x, axis=2, shape=(TILE_M, N, TILE_C))
        h_tile = ct.load(
            h_pre, index=(pid, 0), shape=(TILE_M, N), padding_mode=PAD_ZERO
        )
        h_tile = ct.expand_dims(h_tile, axis=2)
        for j in range(num_tiles):
            x_tile = ct.load(
                x,
                index=(pid, 0, j),
                shape=(TILE_M, N, TILE_C),
                padding_mode=PAD_ZERO,
            )
            acc = ct.sum(x_tile * h_tile, axis=1).astype(ct.float32)
            ct.store(out, index=(pid, j), tile=acc.astype(out.dtype))

    @ct.kernel
    def _ct_h_agg_bwd_kernel(
        go, x, h_pre, gx, gh, N: ConstInt, TILE_M: ConstInt, TILE_C: ConstInt
    ):
        pid = ct.bid(0)
        num_c_tiles = ct.num_tiles(go, axis=1, shape=(TILE_M, TILE_C))
        h_tile = ct.load(
            h_pre, index=(pid, 0), shape=(TILE_M, N), padding_mode=PAD_ZERO
        )
        h_expanded = ct.expand_dims(h_tile, axis=2)
        gh_acc = ct.full((TILE_M, N), 0, dtype=ct.float32)
        for ct_idx in range(num_c_tiles):
            go_tile = ct.load(
                go,
                index=(pid, ct_idx),
                shape=(TILE_M, TILE_C),
                padding_mode=PAD_ZERO,
            )
            go_expanded = ct.expand_dims(go_tile, axis=1)
            x_tile = ct.load(
                x,
                index=(pid, 0, ct_idx),
                shape=(TILE_M, N, TILE_C),
                padding_mode=PAD_ZERO,
            )
            gx_tile = go_expanded * h_expanded
            ct.store(gx, index=(pid, 0, ct_idx), tile=gx_tile.astype(gx.dtype))
            gh_acc += ct.sum(go_expanded * x_tile, axis=2)
        ct.store(gh, index=(pid, 0), tile=gh_acc.astype(gh.dtype))

    def _cutile_h_aggregate_fwd(x: Tensor, h_pre: Tensor) -> Tensor:
        s, b, n, C = x.shape
        sb = s * b
        TILE_SIZE = math.gcd(sb, 4)
        TILE_C = math.gcd(C, 1024)
        out = paddle.empty(shape=[sb, C], dtype=x.dtype)
        ct.launch(
            _get_cuda_stream(),
            (math.ceil(sb / TILE_SIZE),),
            _ct_h_agg_fwd_kernel,
            (
                x.reshape([sb, n, C]),
                h_pre.reshape([sb, n]),
                out,
                n,
                TILE_SIZE,
                TILE_C,
            ),
        )
        return out.reshape([s, b, C])

    def _cutile_h_aggregate_bwd(
        grad_output: Tensor, x: Tensor, h_pre: Tensor
    ) -> tuple[Tensor, Tensor]:
        s, b, n, C = x.shape
        sb = s * b
        TILE_C = math.gcd(C, 1024)
        TILE_M = math.gcd(sb, 4)
        gx = paddle.empty(shape=[sb, n, C], dtype=x.dtype)
        gh = paddle.empty(shape=[sb, n], dtype=x.dtype)
        ct.launch(
            _get_cuda_stream(),
            (math.ceil(sb / TILE_M),),
            _ct_h_agg_bwd_kernel,
            (
                grad_output.reshape([sb, C]),
                x.reshape([sb, n, C]),
                h_pre.reshape([sb, n]),
                gx,
                gh,
                n,
                TILE_M,
                TILE_C,
            ),
        )
        return gx.reshape([s, b, n, C]), gh.reshape([s, b, n])

    # -- H_post BDA kernels --------------------------------------------------

    @ct.kernel
    def _ct_hpb_fwd_kernel(
        hr, orig, hp, x, out, N: ConstInt, TILE_C: ConstInt, TILE_SIZE: ConstInt
    ):
        pid = ct.bid(0)
        num_c_tiles = ct.num_tiles(x, axis=1, shape=(TILE_SIZE, TILE_C))
        hp_tile = ct.load(
            hp, index=(pid, 0), shape=(TILE_SIZE, N), padding_mode=PAD_ZERO
        )
        hp_exp = ct.expand_dims(hp_tile, axis=2)  # (TILE_SIZE, N, 1)
        hr_tile = ct.load(
            hr,
            index=(pid, 0, 0),
            shape=(TILE_SIZE, N, N),
            padding_mode=PAD_ZERO,
        )
        for ct_idx in range(num_c_tiles):
            orig_tile = ct.load(
                orig,
                index=(pid, 0, ct_idx),
                shape=(TILE_SIZE, N, TILE_C),
                padding_mode=PAD_ZERO,
            )
            x_tile = ct.load(
                x,
                index=(pid, ct_idx),
                shape=(TILE_SIZE, TILE_C),
                padding_mode=PAD_ZERO,
            )
            x_exp = ct.expand_dims(x_tile, axis=1)  # (TILE_SIZE, 1, TILE_C)
            out_tile = hp_exp * x_exp  # (TILE_SIZE, N, TILE_C)
            for j in range(N):
                hr_row = ct.extract(hr_tile, (0, j, 0), shape=(TILE_SIZE, 1, N))
                hr_col = ct.reshape(hr_row, (TILE_SIZE, N, 1))
                orig_row = ct.extract(
                    orig_tile, (0, j, 0), shape=(TILE_SIZE, 1, TILE_C)
                )
                out_tile = out_tile + hr_col * orig_row
            ct.store(
                out, index=(pid, 0, ct_idx), tile=out_tile.astype(out.dtype)
            )

    @ct.kernel
    def _ct_hpb_fwd_bias_kernel(
        hr,
        orig,
        hp,
        x,
        bias,
        out,
        N: ConstInt,
        TILE_C: ConstInt,
        TILE_SIZE: ConstInt,
    ):
        pid = ct.bid(0)
        num_c_tiles = ct.num_tiles(x, axis=1, shape=(TILE_SIZE, TILE_C))
        hp_tile = ct.load(
            hp, index=(pid, 0), shape=(TILE_SIZE, N), padding_mode=PAD_ZERO
        )
        hp_exp = ct.expand_dims(hp_tile, axis=2)  # (TILE_SIZE, N, 1)
        hr_tile = ct.load(
            hr,
            index=(pid, 0, 0),
            shape=(TILE_SIZE, N, N),
            padding_mode=PAD_ZERO,
        )
        for ct_idx in range(num_c_tiles):
            orig_tile = ct.load(
                orig,
                index=(pid, 0, ct_idx),
                shape=(TILE_SIZE, N, TILE_C),
                padding_mode=PAD_ZERO,
            )
            x_tile = ct.load(
                x,
                index=(pid, ct_idx),
                shape=(TILE_SIZE, TILE_C),
                padding_mode=PAD_ZERO,
            )
            bias_tile = ct.load(
                bias, index=(ct_idx,), shape=(TILE_C,), padding_mode=PAD_ZERO
            )
            xb_exp = ct.expand_dims(
                x_tile + bias_tile, axis=1
            )  # (TILE_SIZE, 1, TILE_C)
            out_tile = hp_exp * xb_exp  # (TILE_SIZE, N, TILE_C)
            for j in range(N):
                hr_row = ct.extract(hr_tile, (0, j, 0), shape=(TILE_SIZE, 1, N))
                hr_col = ct.reshape(hr_row, (TILE_SIZE, N, 1))
                orig_row = ct.extract(
                    orig_tile, (0, j, 0), shape=(TILE_SIZE, 1, TILE_C)
                )
                out_tile = out_tile + hr_col * orig_row
            ct.store(
                out, index=(pid, 0, ct_idx), tile=out_tile.astype(out.dtype)
            )

    @ct.kernel
    def _ct_hpb_bwd_kernel(
        go,
        hr,
        orig,
        hp,
        x,
        g_hr,
        g_orig,
        g_hp,
        g_x,
        N: ConstInt,
        TILE_C: ConstInt,
        TILE_SIZE: ConstInt,
    ):
        pid = ct.bid(0)
        num_c_tiles = ct.cdiv(go.shape[2], TILE_C)
        hp_tile = ct.load(hp, index=(pid, 0), shape=(TILE_SIZE, N))
        hp_2d = ct.reshape(hp_tile, (1, N))
        hr_tile = ct.load(
            hr,
            index=(pid, 0, 0),
            shape=(TILE_SIZE, N, N),
            padding_mode=PAD_ZERO,
        )
        hr_2d = ct.reshape(hr_tile, (N, N))
        acc_g_hp_2d = ct.full((N, 1), 0, dtype=ct.float32)
        acc_g_hr_2d = ct.full((N, N), 0, dtype=ct.float32)
        for ct_idx in range(num_c_tiles):
            x_tile = ct.load(
                x,
                index=(pid, ct_idx),
                shape=(TILE_SIZE, TILE_C),
                padding_mode=PAD_ZERO,
            )
            x_2d = ct.reshape(x_tile, (1, TILE_C))
            go_tile = ct.load(
                go,
                index=(pid, 0, ct_idx),
                shape=(TILE_SIZE, N, TILE_C),
                padding_mode=PAD_ZERO,
            )
            go_2d = ct.reshape(go_tile, (N, TILE_C))
            orig_tile = ct.load(
                orig,
                index=(pid, 0, ct_idx),
                shape=(TILE_SIZE, N, TILE_C),
                padding_mode=PAD_ZERO,
            )
            orig_2d = ct.reshape(orig_tile, (N, TILE_C))
            g_x_2d = ct.full((1, TILE_C), 0, dtype=hp.dtype)
            g_orig_2d = ct.full((N, TILE_C), 0, dtype=hp.dtype)
            for j in range(N):
                g_x_2d += ct.extract(
                    hp_2d, (0, j), shape=(1, 1)
                ).item() * ct.extract(go_2d, (j, 0), shape=(1, TILE_C))
                g_orig_2d += ct.extract(
                    hr_2d, (0, j), shape=(N, 1)
                ) * ct.extract(go_2d, (j, 0), shape=(1, TILE_C))
            acc_g_hp_2d += ct.sum(go_2d * x_2d, axis=1, keepdims=True)
            acc_g_hr_2d += ct.sum(
                ct.expand_dims(go_2d, axis=0) * ct.expand_dims(orig_2d, axis=1),
                axis=2,
            )
            ct.store(
                g_x,
                index=(pid, ct_idx),
                tile=ct.reshape(g_x_2d, (TILE_SIZE, TILE_C)).astype(g_x.dtype),
            )
            ct.store(
                g_orig,
                index=(pid, 0, ct_idx),
                tile=ct.reshape(g_orig_2d, (TILE_SIZE, N, TILE_C)).astype(
                    g_orig.dtype
                ),
            )
        ct.store(
            g_hp,
            index=(pid, 0),
            tile=ct.reshape(acc_g_hp_2d, (TILE_SIZE, N)).astype(g_hp.dtype),
        )
        ct.store(
            g_hr,
            index=(pid, 0, 0),
            tile=ct.reshape(acc_g_hr_2d, (TILE_SIZE, N, N)).astype(g_hr.dtype),
        )

    @ct.kernel
    def _ct_hpb_bwd_bias_kernel(
        go,
        hr,
        orig,
        hp,
        x,
        bias,
        g_hr,
        g_orig,
        g_hp,
        g_x,
        N: ConstInt,
        TILE_C: ConstInt,
        TILE_SIZE: ConstInt,
    ):
        pid = ct.bid(0)
        num_c_tiles = ct.cdiv(go.shape[2], TILE_C)
        hp_tile = ct.load(hp, index=(pid, 0), shape=(TILE_SIZE, N))
        hp_2d = ct.reshape(hp_tile, (1, N))
        hr_tile = ct.load(
            hr,
            index=(pid, 0, 0),
            shape=(TILE_SIZE, N, N),
            padding_mode=PAD_ZERO,
        )
        hr_2d = ct.reshape(hr_tile, (N, N))
        acc_g_hp_2d = ct.full((N, 1), 0, dtype=ct.float32)
        acc_g_hr_2d = ct.full((N, N), 0, dtype=ct.float32)
        for ct_idx in range(num_c_tiles):
            x_tile = ct.load(
                x,
                index=(pid, ct_idx),
                shape=(TILE_SIZE, TILE_C),
                padding_mode=PAD_ZERO,
            )
            bias_tile = ct.load(
                bias, index=(ct_idx,), shape=(TILE_C,), padding_mode=PAD_ZERO
            )
            xb_2d = ct.reshape(x_tile, (1, TILE_C)) + ct.reshape(
                bias_tile, (1, TILE_C)
            )
            go_tile = ct.load(
                go,
                index=(pid, 0, ct_idx),
                shape=(TILE_SIZE, N, TILE_C),
                padding_mode=PAD_ZERO,
            )
            go_2d = ct.reshape(go_tile, (N, TILE_C))
            orig_tile = ct.load(
                orig,
                index=(pid, 0, ct_idx),
                shape=(TILE_SIZE, N, TILE_C),
                padding_mode=PAD_ZERO,
            )
            orig_2d = ct.reshape(orig_tile, (N, TILE_C))
            g_x_2d = ct.full((1, TILE_C), 0, dtype=hp.dtype)
            g_orig_2d = ct.full((N, TILE_C), 0, dtype=hp.dtype)
            for j in range(N):
                g_x_2d += ct.extract(
                    hp_2d, (0, j), shape=(1, 1)
                ).item() * ct.extract(go_2d, (j, 0), shape=(1, TILE_C))
                g_orig_2d += ct.extract(
                    hr_2d, (0, j), shape=(N, 1)
                ) * ct.extract(go_2d, (j, 0), shape=(1, TILE_C))
            acc_g_hp_2d += ct.sum(go_2d * xb_2d, axis=1, keepdims=True)
            acc_g_hr_2d += ct.sum(
                ct.expand_dims(go_2d, axis=0) * ct.expand_dims(orig_2d, axis=1),
                axis=2,
            )
            ct.store(
                g_x,
                index=(pid, ct_idx),
                tile=ct.reshape(g_x_2d, (TILE_SIZE, TILE_C)).astype(g_x.dtype),
            )
            ct.store(
                g_orig,
                index=(pid, 0, ct_idx),
                tile=ct.reshape(g_orig_2d, (TILE_SIZE, N, TILE_C)).astype(
                    g_orig.dtype
                ),
            )
        ct.store(
            g_hp,
            index=(pid, 0),
            tile=ct.reshape(acc_g_hp_2d, (TILE_SIZE, N)).astype(g_hp.dtype),
        )
        ct.store(
            g_hr,
            index=(pid, 0, 0),
            tile=ct.reshape(acc_g_hr_2d, (TILE_SIZE, N, N)).astype(g_hr.dtype),
        )

    def _cutile_h_post_bda_fwd(
        h_res: Tensor,
        original_residual: Tensor,
        h_post: Tensor,
        x: Tensor,
        bias: Tensor | None,
    ) -> Tensor:
        s, b, n, C = original_residual.shape
        sb = s * b
        TILE_C = math.gcd(C, 1024)
        TILE_SIZE = math.gcd(sb, 1)
        out = paddle.empty(shape=[sb, n, C], dtype=h_res.dtype)
        grid = (math.ceil(sb / TILE_SIZE),)
        if bias is not None:
            ct.launch(
                _get_cuda_stream(),
                grid,
                _ct_hpb_fwd_bias_kernel,
                (
                    h_res.reshape([sb, n, n]),
                    original_residual.reshape([sb, n, C]),
                    h_post.reshape([sb, n]),
                    x.reshape([sb, C]),
                    bias.detach(),
                    out,
                    n,
                    TILE_C,
                    TILE_SIZE,
                ),
            )
        else:
            ct.launch(
                _get_cuda_stream(),
                grid,
                _ct_hpb_fwd_kernel,
                (
                    h_res.reshape([sb, n, n]),
                    original_residual.reshape([sb, n, C]),
                    h_post.reshape([sb, n]),
                    x.reshape([sb, C]),
                    out,
                    n,
                    TILE_C,
                    TILE_SIZE,
                ),
            )
        return out.reshape([s, b, n, C])

    def _cutile_h_post_bda_bwd(
        grad_output: Tensor,
        h_res: Tensor,
        original_residual: Tensor,
        h_post: Tensor,
        x: Tensor,
        bias: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None]:
        s, b, n, C = original_residual.shape
        sb = s * b
        TILE_C = math.gcd(C, 1024)
        TILE_SIZE = math.gcd(sb, 1)
        g_hr = paddle.empty(shape=[sb, n, n], dtype=h_res.dtype)
        g_res = paddle.empty(shape=[sb, n, C], dtype=h_res.dtype)
        g_hp = paddle.empty(shape=[sb, n], dtype=h_res.dtype)
        g_x = paddle.empty(shape=[sb, C], dtype=h_res.dtype)
        grid = (sb,)
        if bias is not None:
            ct.launch(
                _get_cuda_stream(),
                grid,
                _ct_hpb_bwd_bias_kernel,
                (
                    grad_output.reshape([sb, n, C]),
                    h_res.reshape([sb, n, n]),
                    original_residual.reshape([sb, n, C]),
                    h_post.reshape([sb, n]),
                    x.reshape([sb, C]),
                    bias.detach(),
                    g_hr,
                    g_res,
                    g_hp,
                    g_x,
                    n,
                    TILE_C,
                    TILE_SIZE,
                ),
            )
        else:
            ct.launch(
                _get_cuda_stream(),
                grid,
                _ct_hpb_bwd_kernel,
                (
                    grad_output.reshape([sb, n, C]),
                    h_res.reshape([sb, n, n]),
                    original_residual.reshape([sb, n, C]),
                    h_post.reshape([sb, n]),
                    x.reshape([sb, C]),
                    g_hr,
                    g_res,
                    g_hp,
                    g_x,
                    n,
                    TILE_C,
                    TILE_SIZE,
                ),
            )
        g_bias = g_x.sum(axis=0) if bias is not None else None
        return (
            g_hr.reshape([s, b, n, n]),
            g_res.reshape([s, b, n, C]),
            g_hp.reshape([s, b, n]),
            g_x.reshape([s, b, C]),
            g_bias,
        )

    # -- Proj RMS kernels ----------------------------------------------------

    @ct.function
    def _ct_rms_dnorm(a_tile, norm_tile, dr_tile, K, eps):
        inv_norm = ct.where(norm_tile > 0, 1.0 / norm_tile, 0.0)
        inv_sqrt_k = 1.0 / ct.sqrt(K)
        u = norm_tile * inv_sqrt_k + eps
        coeff = -(1.0 / (u * u)) * inv_sqrt_k
        return dr_tile * coeff * a_tile * inv_norm

    @ct.kernel
    def _ct_proj_rms_fwd_kernel(
        A,
        B,
        PROJ,
        NORM,
        R,
        M: int,
        N: int,
        K: int,
        eps: float,
        TILE_M: ConstInt,
        TILE_N: ConstInt,
        TILE_K: ConstInt,
        SPLIT_K: ConstInt,
    ):
        """
        Grid: (num_tiles_m, num_tiles_k).
        Fused matmul + norm + r: proj, norm, r in one pass over K.
        R is a retained signature placeholder; r is computed after split-K
        reduction from NORM.
        """
        tile_m_id = ct.bid(0)
        split_k_id = ct.bid(1)
        num_m_tiles = ct.cdiv(M, TILE_M)
        num_k_tiles = ct.cdiv(K, TILE_K)
        num_k_tiles_per_split = ct.cdiv(num_k_tiles, SPLIT_K)
        tile_k_id_start = split_k_id * num_k_tiles_per_split
        tile_k_id_end = ct.minimum(
            tile_k_id_start + num_k_tiles_per_split, num_k_tiles
        )
        acc = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
        sum_sq = ct.full((TILE_M, 1), 0.0, dtype=ct.float32)
        for tile_k_id in range(tile_k_id_start, tile_k_id_end):
            a_tile = ct.load(
                A,
                index=(tile_m_id, tile_k_id),
                shape=(TILE_M, TILE_K),
                padding_mode=PAD_ZERO,
            )
            b_tile = ct.load(
                B,
                index=(0, tile_k_id),
                shape=(TILE_N, TILE_K),
                padding_mode=PAD_ZERO,
            )
            acc = ct.mma(
                a_tile.astype(ct.tfloat32),
                b_tile.transpose().astype(ct.tfloat32),
                acc=acc,
            )
            sum_sq += ct.sum(a_tile * a_tile, axis=1, keepdims=True)
        bid_m_k = tile_m_id + split_k_id * num_m_tiles
        ct.store(PROJ, index=(bid_m_k, 0), tile=acc.astype(PROJ.dtype))
        ct.store(NORM, index=(bid_m_k, 0), tile=sum_sq.astype(NORM.dtype))

    @ct.kernel
    def _ct_proj_rms_bwd_kernel(
        A,
        B,
        NORM,
        DD,
        DR,
        DA,
        DB,
        M: int,
        N: int,
        K: int,
        eps: float,
        TILE_SIZE_M: ConstInt,
        TILE_SIZE_N: ConstInt,
        TILE_SIZE_K: ConstInt,
    ):
        zero_pad = ct.PaddingMode.ZERO
        tile_k_id = ct.bid(0)
        NUM_M_TILES = ct.cdiv(M, TILE_SIZE_M)
        accumulator_db = ct.full(
            (TILE_SIZE_K, TILE_SIZE_N), 0.0, dtype=ct.float32
        )
        for tile_m_id in range(NUM_M_TILES):
            accumulator_da = ct.full(
                (TILE_SIZE_M, TILE_SIZE_K), 0.0, dtype=ct.float32
            )
            a_tile = ct.load(
                A,
                index=(tile_m_id, tile_k_id),
                shape=(TILE_SIZE_M, TILE_SIZE_K),
                padding_mode=zero_pad,
            )
            norm_tile = ct.load(
                NORM,
                index=(tile_m_id, 0),
                shape=(TILE_SIZE_M, 1),
                padding_mode=zero_pad,
            )
            dr_tile = ct.load(
                DR,
                index=(tile_m_id, 0),
                shape=(TILE_SIZE_M, 1),
                padding_mode=zero_pad,
            )
            accumulator_da = accumulator_da + _ct_rms_dnorm(
                a_tile, norm_tile, dr_tile, K, eps
            )
            b_tile = ct.load(
                B,
                index=(0, tile_k_id),
                shape=(TILE_SIZE_N, TILE_SIZE_K),
                padding_mode=zero_pad,
            )
            dd_tile = ct.load(
                DD,
                index=(tile_m_id, 0),
                shape=(TILE_SIZE_M, TILE_SIZE_N),
                padding_mode=zero_pad,
            )
            dd_tile = ct.astype(dd_tile, ct.tfloat32)
            accumulator_da = ct.mma(
                dd_tile, b_tile.astype(ct.tfloat32), acc=accumulator_da
            )
            ct.store(
                DA,
                index=(tile_m_id, tile_k_id),
                tile=accumulator_da.astype(DA.dtype),
            )
            accumulator_db = ct.mma(
                a_tile.transpose().astype(ct.tfloat32),
                dd_tile,
                acc=accumulator_db,
            )
        ct.store(
            DB,
            index=(0, tile_k_id),
            tile=accumulator_db.transpose().astype(DB.dtype),
        )

    @ct.kernel
    def _ct_proj_rms_bwd_small_k_kernel(
        A,
        B,
        NORM,
        DD,
        DR,
        DA,
        DB,
        M: int,
        N: int,
        K: int,
        eps: float,
        TILE_N_SIZE: ConstInt,
    ):
        zero_pad = ct.PaddingMode.ZERO
        TILE_DB_SIZE_M = 128
        TILE_DB_SIZE_K = 64
        NUM_M_TILES = ct.cdiv(M, TILE_DB_SIZE_M)
        NUM_K_TILES = ct.cdiv(K, TILE_DB_SIZE_K)
        if ct.bid(1) == 0:
            for tile_id in range(ct.bid(0), NUM_K_TILES, ct.num_blocks(0)):
                accumulator_db = ct.full(
                    (TILE_DB_SIZE_K, TILE_N_SIZE), 0.0, dtype=ct.float32
                )
                for m_tile in range(NUM_M_TILES):
                    a_tile = ct.load(
                        A,
                        index=(m_tile, tile_id),
                        shape=(TILE_DB_SIZE_M, TILE_DB_SIZE_K),
                        padding_mode=zero_pad,
                    )
                    dd_tile = ct.load(
                        DD,
                        index=(m_tile, 0),
                        shape=(TILE_DB_SIZE_M, TILE_N_SIZE),
                        padding_mode=zero_pad,
                    )
                    accumulator_db = ct.mma(
                        a_tile.transpose().astype(ct.tfloat32),
                        dd_tile.astype(ct.tfloat32),
                        acc=accumulator_db,
                    )
                ct.store(
                    DB,
                    index=(0, tile_id),
                    tile=accumulator_db.transpose().astype(DB.dtype),
                    allow_tma=False,
                )
        TILE_DA_SIZE_M = 128
        TILE_DA_SIZE_K = 256
        NUM_DA_TILES = ct.cdiv(M, TILE_DA_SIZE_M) * ct.cdiv(K, TILE_DA_SIZE_K)
        NUM_DA_K_TILES = ct.cdiv(K, TILE_DA_SIZE_K)
        if ct.bid(1) == 1:
            for tile_id in range(ct.bid(0), NUM_DA_TILES, ct.num_blocks(0)):
                b_tile_idx = tile_id % NUM_DA_K_TILES
                dd_tile_idx = tile_id // NUM_DA_K_TILES
                accumulator_da = ct.full(
                    (TILE_DA_SIZE_M, TILE_DA_SIZE_K), 0.0, dtype=ct.float32
                )
                a_tile = ct.load(
                    A,
                    index=(dd_tile_idx, b_tile_idx),
                    shape=(TILE_DA_SIZE_M, TILE_DA_SIZE_K),
                    padding_mode=zero_pad,
                )
                norm_tile = ct.load(
                    NORM,
                    index=(dd_tile_idx, 0),
                    shape=(TILE_DA_SIZE_M, 1),
                    padding_mode=zero_pad,
                )
                dr_tile = ct.load(
                    DR,
                    index=(dd_tile_idx, 0),
                    shape=(TILE_DA_SIZE_M, 1),
                    padding_mode=zero_pad,
                )
                accumulator_da = accumulator_da + _ct_rms_dnorm(
                    a_tile.astype(ct.float32), norm_tile, dr_tile, K, eps
                )
                b_tile = ct.load(
                    B,
                    index=(0, b_tile_idx),
                    shape=(TILE_N_SIZE, TILE_DA_SIZE_K),
                    padding_mode=zero_pad,
                )
                dd_tile = ct.load(
                    DD,
                    index=(dd_tile_idx, 0),
                    shape=(TILE_DA_SIZE_M, TILE_N_SIZE),
                    padding_mode=zero_pad,
                )
                accumulator_da = ct.mma(
                    dd_tile.astype(ct.tfloat32),
                    b_tile.astype(ct.tfloat32),
                    acc=accumulator_da,
                )
                ct.store(
                    DA,
                    index=(dd_tile_idx, b_tile_idx),
                    tile=accumulator_da.astype(DA.dtype),
                )

    def _ct_sigmoid(x):
        """Sigmoid via exp2: σ(x) = 1 / (1 + 2^(-x * log2(e)))."""
        return 1.0 / (1.0 + ct.exp2(-x * LOG2E))

    def _next_power_of_2(n: int) -> int:
        n -= 1
        n |= n >> 1
        n |= n >> 2
        n |= n >> 4
        n |= n >> 8
        n |= n >> 16
        n |= n >> 32
        n += 1
        return n

    def _default_tile_m(M: int) -> int:
        for tile_m in (128, 64, 32, 16, 8, 4, 2, 1):
            if tile_m <= M and M % tile_m == 0:
                return tile_m
        return 1

    def _default_proj_rms_fwd_config(M: int, K: int, TILE_N: int):
        split_k = 16 if K >= 16384 else 8 if K >= 8192 else 1
        return _default_tile_m(M), TILE_N, min(128, K), split_k

    def _cutile_proj_rms_fwd(
        x: Tensor, weight: Tensor, eps: float = 1e-8
    ) -> tuple[Tensor, Tensor, Tensor]:
        M, K = x.shape
        N = weight.shape[0]
        TILE_N = _next_power_of_2(N)
        TILE_M, _, TILE_K, split_k = _default_proj_rms_fwd_config(M, K, TILE_N)
        num_tiles_m = math.ceil(M / TILE_M)
        proj = paddle.empty(shape=[split_k * M, N], dtype=x.dtype)
        norm = paddle.empty(shape=[split_k * M, 1], dtype=x.dtype)
        r = paddle.empty(shape=[split_k * M, 1], dtype=x.dtype)
        ct.launch(
            _get_cuda_stream(),
            (num_tiles_m, split_k),
            _ct_proj_rms_fwd_kernel,
            (
                x.detach(),
                weight.detach(),
                proj,
                norm,
                r,
                M,
                N,
                K,
                eps,
                TILE_M,
                TILE_N,
                TILE_K,
                split_k,
            ),
        )
        # Reduce split_K partial results
        proj = (
            proj.reshape([split_k, M, N])
            .astype("float32")
            .sum(axis=0)
            .astype(x.dtype)
        )
        norm = (
            norm.reshape([split_k, M, 1])
            .astype("float32")
            .sum(axis=0)
            .astype(x.dtype)
        )
        # Compute norm and r after reduction
        norm = paddle.sqrt(norm)
        r = 1.0 / (norm / math.sqrt(K) + eps)
        return proj, norm, r

    def _cutile_proj_rms_bwd(
        grad_proj: Tensor,
        grad_r: Tensor,
        x: Tensor,
        weight: Tensor,
        norm: Tensor,
        eps: float = 1e-8,
    ) -> tuple[Tensor, Tensor]:
        M, K = x.shape
        N = weight.shape[0]
        da = paddle.empty(shape=x.shape, dtype=x.dtype)
        db = paddle.empty(shape=weight.shape, dtype=weight.dtype)
        TILE_SIZE_N = _next_power_of_2(N)
        assert TILE_SIZE_N <= 256, f"TILE_SIZE_N too large: {TILE_SIZE_N}"
        num_sms = (
            paddle.device.cuda.get_device_properties().multi_processor_count
        )
        if K >= 8192:
            TILE_SIZE_M, TILE_SIZE_K = 128, 128
            grid = (math.ceil(K / TILE_SIZE_K), 1)
            ct.launch(
                _get_cuda_stream(),
                grid,
                _ct_proj_rms_bwd_kernel,
                (
                    x.detach(),
                    weight.detach(),
                    norm.detach(),
                    grad_proj.detach(),
                    grad_r.detach(),
                    da,
                    db,
                    M,
                    N,
                    K,
                    eps,
                    TILE_SIZE_M,
                    TILE_SIZE_N,
                    TILE_SIZE_K,
                ),
            )
        else:
            grid = (num_sms, 2, 1)
            ct.launch(
                _get_cuda_stream(),
                grid,
                _ct_proj_rms_bwd_small_k_kernel,
                (
                    x.detach(),
                    weight.detach(),
                    norm.detach(),
                    grad_proj.detach(),
                    grad_r.detach(),
                    da,
                    db,
                    M,
                    N,
                    K,
                    eps,
                    TILE_SIZE_N,
                ),
            )
        return da, db

    # -- reduce_compute_h kernel and launchers ---------------------------------

    def _reduce_compute_h_autotune_configs(M):
        min_tile_m = 16 if M >= 16 else 1
        for tile_m in (128, 64, 32, 16, 8, 4, 2, 1):
            if tile_m < min_tile_m:
                continue
            if tile_m <= M and M % tile_m == 0:
                yield tile_m

    def _default_reduce_compute_h_tile_m(M: int) -> int:
        try:
            num_sms = (
                paddle.device.cuda.get_device_properties().multi_processor_count
            )
        except Exception:
            num_sms = 128
        valid = list(_reduce_compute_h_autotune_configs(M))
        for tm in valid:
            if math.ceil(M / tm) >= num_sms:
                return tm
        return valid[-1] if valid else 1

    @ct.kernel
    def _ct_reduce_compute_h_kernel(
        Y_acc,
        R_acc,
        Bias,
        Alpha_pre,
        Alpha_post,
        Alpha_res,
        H_PRE,
        H_POST,
        H_RES,
        R,
        PROJ_OUT,
        M: int,
        N: int,
        K: int,
        n: ConstInt,
        eps: float,
        compute_h_eps: float,
        TILE_SIZE_M: ConstInt,
        TILE_SIZE_N: ConstInt,
        SPLIT_K: ConstInt,
    ):
        """Reduce split-K partial proj/norm, compute r, and apply compute_h activations.

        Grid: (ceil(M / TILE_SIZE_M),).
        TILE_SIZE_N = next_power_of_2(N) so one tile covers the full N dimension.
        Alpha_{pre,post,res} are [1] tensors (scalar parameters).
        """
        bid_m = ct.bid(0)
        num_bid_m = ct.cdiv(M, TILE_SIZE_M)

        alpha_pre = ct.load(Alpha_pre, index=(0,), shape=(1,)).item()
        alpha_post = ct.load(Alpha_post, index=(0,), shape=(1,)).item()
        alpha_res = ct.load(Alpha_res, index=(0,), shape=(1,)).item()

        pre_accum = ct.full((TILE_SIZE_M, n), 0.0, dtype=ct.float32)
        post_accum = ct.full((TILE_SIZE_M, n), 0.0, dtype=ct.float32)
        r_accum = ct.full((TILE_SIZE_M, 1), 0.0, dtype=ct.float32)

        for split_idx in ct.static_iter(range(SPLIT_K)):
            bid_m_k = bid_m + split_idx * num_bid_m
            pre_tile = ct.load(
                Y_acc,
                index=(bid_m_k, 0),
                shape=(TILE_SIZE_M, n),
                padding_mode=PAD_ZERO,
            )
            post_tile = ct.load(
                Y_acc,
                index=(bid_m_k, 1),
                shape=(TILE_SIZE_M, n),
                padding_mode=PAD_ZERO,
            )
            pre_accum = pre_accum + ct.astype(pre_tile, ct.float32)
            post_accum = post_accum + ct.astype(post_tile, ct.float32)
            r_tile = ct.load(
                R_acc,
                index=(bid_m_k, 0),
                shape=(TILE_SIZE_M, 1),
                padding_mode=PAD_ZERO,
            )
            r_accum = r_accum + ct.astype(r_tile, ct.float32)

        ct.store(
            PROJ_OUT, index=(bid_m, 0), tile=pre_accum.astype(PROJ_OUT.dtype)
        )
        ct.store(
            PROJ_OUT, index=(bid_m, 1), tile=post_accum.astype(PROJ_OUT.dtype)
        )

        denom = ct.full((TILE_SIZE_M, 1), K * 1.0, dtype=ct.float32)
        r_val = ct.sqrt(ct.truediv(r_accum, denom))
        ct.store(R, index=(bid_m, 0), tile=r_val.astype(R.dtype))

        inv_r_eps = 1.0 / (r_val + eps)
        bias_pre = ct.astype(
            ct.load(Bias, index=(0, 0), shape=(1, n), padding_mode=PAD_ZERO),
            ct.float32,
        )
        bias_post = ct.astype(
            ct.load(Bias, index=(0, 1), shape=(1, n), padding_mode=PAD_ZERO),
            ct.float32,
        )

        h_pre_linear = pre_accum * alpha_pre * inv_r_eps + bias_pre
        h_post_linear = post_accum * alpha_post * inv_r_eps + bias_post
        h_pre = _ct_sigmoid(h_pre_linear) + compute_h_eps
        h_post = _ct_sigmoid(h_post_linear) * 2.0

        ct.store(H_PRE, index=(bid_m, 0), tile=h_pre.astype(H_PRE.dtype))
        ct.store(H_POST, index=(bid_m, 0), tile=h_post.astype(H_POST.dtype))

        for res_chunk in ct.static_iter(range(n)):
            res_accum = ct.full((TILE_SIZE_M, n), 0.0, dtype=ct.float32)
            for split_idx in ct.static_iter(range(SPLIT_K)):
                bid_m_k = bid_m + split_idx * num_bid_m
                res_tile = ct.load(
                    Y_acc,
                    index=(bid_m_k, 2 + res_chunk),
                    shape=(TILE_SIZE_M, n),
                    padding_mode=PAD_ZERO,
                )
                res_accum = res_accum + ct.astype(res_tile, ct.float32)
            bias_res = ct.astype(
                ct.load(
                    Bias,
                    index=(0, 2 + res_chunk),
                    shape=(1, n),
                    padding_mode=PAD_ZERO,
                ),
                ct.float32,
            )
            h_res = res_accum * alpha_res * inv_r_eps + bias_res
            ct.store(
                PROJ_OUT,
                index=(bid_m, 2 + res_chunk),
                tile=res_accum.astype(PROJ_OUT.dtype),
            )
            ct.store(
                H_RES, index=(bid_m, res_chunk), tile=h_res.astype(H_RES.dtype)
            )

    def _cutile_reduce_compute_h(
        proj_acc,
        norm_acc,
        bias,
        alpha_pre,
        alpha_post,
        alpha_res,
        n,
        M,
        N,
        K,
        eps,
        compute_h_eps,
        _proj_tile_m,
        tile_n,
        split_k,
    ):
        """Launch reduce split-K + compute_h kernel.

        Returns:
            h_pre: [M, n] sigmoid-activated pre weights
            h_post: [M, n] 2*sigmoid-activated post weights
            h_res: [M, n*n] residual logits
            r: [M, 1]  r = norm / sqrt(K)
            proj_reduced: [M, N] reduced projection (for backward)
        """
        stream = _get_cuda_stream()
        bias_2d = bias.detach().unsqueeze(0)
        h_pre_out = paddle.empty(shape=[M, n], dtype=proj_acc.dtype)
        h_post_out = paddle.empty(shape=[M, n], dtype=proj_acc.dtype)
        h_res_out = paddle.empty(shape=[M, N - 2 * n], dtype=proj_acc.dtype)
        r_out = paddle.empty(shape=[M, 1], dtype=proj_acc.dtype)
        proj_out = paddle.empty(shape=[M, N], dtype=proj_acc.dtype)

        tm = _default_reduce_compute_h_tile_m(M)

        ct.launch(
            stream,
            (math.ceil(M / tm),),
            _ct_reduce_compute_h_kernel,
            (
                proj_acc,
                norm_acc,
                bias_2d,
                alpha_pre.detach(),
                alpha_post.detach(),
                alpha_res.detach(),
                h_pre_out,
                h_post_out,
                h_res_out,
                r_out,
                proj_out,
                M,
                N,
                K,
                n,
                eps,
                compute_h_eps,
                tm,
                tile_n,
                split_k,
            ),
        )
        return h_pre_out, h_post_out, h_res_out, r_out, proj_out

    def _cutile_proj_rms_compute_h_fwd(
        x,
        weight,
        bias,
        alpha_pre,
        alpha_post,
        alpha_res,
        n,
        eps,
        compute_h_eps,
    ):
        """Fused proj_rms + compute_h forward.

        Launches the existing _ct_proj_rms_fwd_kernel (split-K matmul + partial norm),
        then _ct_reduce_compute_h_kernel (reduce + r + activations).

        Returns:
            h_pre: [M, n] activated pre weights
            h_post: [M, n] activated post weights
            h_res: [M, n*n] residual logits
            r: [M, 1] r = norm / sqrt(K)
            proj_reduced: [M, N] reduced projection (for backward)
        """
        M, K = x.shape
        N = weight.shape[0]
        TILE_N = _next_power_of_2(N)
        stream = _get_cuda_stream()
        tm, tn, tk, split_k = _default_proj_rms_fwd_config(M, K, TILE_N)

        proj_acc = paddle.empty(shape=[split_k * M, N], dtype=x.dtype)
        norm_acc = paddle.empty(shape=[split_k * M, 1], dtype=x.dtype)
        r_placeholder = paddle.empty(shape=[split_k * M, 1], dtype=x.dtype)

        ct.launch(
            stream,
            (math.ceil(M / tm), split_k),
            _ct_proj_rms_fwd_kernel,
            (
                x.detach(),
                weight.detach(),
                proj_acc,
                norm_acc,
                r_placeholder,
                M,
                N,
                K,
                eps,
                tm,
                tn,
                tk,
                split_k,
            ),
        )

        h_pre, h_post, h_res, r, proj_reduced = _cutile_reduce_compute_h(
            proj_acc,
            norm_acc,
            bias,
            alpha_pre,
            alpha_post,
            alpha_res,
            n,
            M,
            N,
            K,
            eps,
            compute_h_eps,
            tm,
            TILE_N,
            split_k,
        )
        return h_pre, h_post, h_res, r, proj_reduced

    # -- Backward kernels for fused compute_h + proj_rms -----------------------

    @ct.kernel
    def _ct_fused_grad_h_proj_kernel(
        GRAD_H_PRE,
        GRAD_H_POST,
        GRAD_H_RES,
        H_PRE,
        H_POST,
        PROJ,
        R,
        GRAD_R_EXT,
        Alpha_pre,
        Alpha_post,
        Alpha_res,
        GRAD_H,
        GRAD_PROJ,
        GRAD_R_TOTAL,
        M: int,
        N: int,
        n: ConstInt,
        eps: float,
        compute_h_eps: float,
        TILE_SIZE_M: ConstInt,
        TILE_SIZE_N: ConstInt,
        HAS_GRAD_H_PRE: ConstInt,
        HAS_GRAD_H_POST: ConstInt,
        HAS_GRAD_H_RES: ConstInt,
        HAS_GRAD_R_EXT: ConstInt,
    ):
        """Precompute grad_h, grad_proj, and grad_r_total for downstream backward kernels.

        Grid: (ceil(M / TILE_SIZE_M),).
        """
        tile_m_id = ct.bid(0)
        alpha_pre = ct.load(Alpha_pre, index=(0,), shape=(1,)).item()
        alpha_post = ct.load(Alpha_post, index=(0,), shape=(1,)).item()
        alpha_res = ct.load(Alpha_res, index=(0,), shape=(1,)).item()

        r_tile = ct.astype(
            ct.load(
                R,
                index=(tile_m_id, 0),
                shape=(TILE_SIZE_M, 1),
                padding_mode=PAD_ZERO,
            ),
            ct.float32,
        )
        r_eps = r_tile + eps
        inv_r_eps = 1.0 / r_eps
        grad_r_from_h = ct.full((TILE_SIZE_M, 1), 0.0, dtype=ct.float32)

        zero_full = ct.full((TILE_SIZE_M, TILE_SIZE_N), 0.0, dtype=ct.float32)
        ct.store(
            GRAD_H, index=(tile_m_id, 0), tile=zero_full.astype(GRAD_H.dtype)
        )
        ct.store(
            GRAD_PROJ,
            index=(tile_m_id, 0),
            tile=zero_full.astype(GRAD_PROJ.dtype),
        )

        # h_pre grad
        if HAS_GRAD_H_PRE:
            gy_pre = ct.astype(
                ct.load(
                    GRAD_H_PRE,
                    index=(tile_m_id, 0),
                    shape=(TILE_SIZE_M, n),
                    padding_mode=PAD_ZERO,
                ),
                ct.float32,
            )
        else:
            gy_pre = ct.full((TILE_SIZE_M, n), 0.0, dtype=ct.float32)
        h_pre = ct.astype(
            ct.load(
                H_PRE,
                index=(tile_m_id, 0),
                shape=(TILE_SIZE_M, n),
                padding_mode=PAD_ZERO,
            ),
            ct.float32,
        )
        proj_pre = ct.astype(
            ct.load(
                PROJ,
                index=(tile_m_id, 0),
                shape=(TILE_SIZE_M, n),
                padding_mode=PAD_ZERO,
            ),
            ct.float32,
        )
        sigmoid_pre = h_pre - compute_h_eps
        grad_h_pre = gy_pre * sigmoid_pre * (1.0 - sigmoid_pre)
        grad_proj_pre = grad_h_pre * alpha_pre * inv_r_eps
        grad_r_from_h += ct.sum(
            grad_h_pre * proj_pre * alpha_pre * (-inv_r_eps * inv_r_eps),
            axis=1,
            keepdims=True,
        )
        ct.store(
            GRAD_H, index=(tile_m_id, 0), tile=grad_h_pre.astype(GRAD_H.dtype)
        )
        ct.store(
            GRAD_PROJ,
            index=(tile_m_id, 0),
            tile=grad_proj_pre.astype(GRAD_PROJ.dtype),
        )

        # h_post grad
        if HAS_GRAD_H_POST:
            gy_post = ct.astype(
                ct.load(
                    GRAD_H_POST,
                    index=(tile_m_id, 0),
                    shape=(TILE_SIZE_M, n),
                    padding_mode=PAD_ZERO,
                ),
                ct.float32,
            )
        else:
            gy_post = ct.full((TILE_SIZE_M, n), 0.0, dtype=ct.float32)
        h_post = ct.astype(
            ct.load(
                H_POST,
                index=(tile_m_id, 0),
                shape=(TILE_SIZE_M, n),
                padding_mode=PAD_ZERO,
            ),
            ct.float32,
        )
        proj_post = ct.astype(
            ct.load(
                PROJ,
                index=(tile_m_id, 1),
                shape=(TILE_SIZE_M, n),
                padding_mode=PAD_ZERO,
            ),
            ct.float32,
        )
        sigmoid_post = h_post * 0.5
        grad_h_post = gy_post * sigmoid_post * (1.0 - sigmoid_post) * 2.0
        grad_proj_post = grad_h_post * alpha_post * inv_r_eps
        grad_r_from_h += ct.sum(
            grad_h_post * proj_post * alpha_post * (-inv_r_eps * inv_r_eps),
            axis=1,
            keepdims=True,
        )
        ct.store(
            GRAD_H, index=(tile_m_id, 1), tile=grad_h_post.astype(GRAD_H.dtype)
        )
        ct.store(
            GRAD_PROJ,
            index=(tile_m_id, 1),
            tile=grad_proj_post.astype(GRAD_PROJ.dtype),
        )

        # h_res grad
        for res_chunk in ct.static_iter(range(n)):
            if HAS_GRAD_H_RES:
                grad_h_res = ct.astype(
                    ct.load(
                        GRAD_H_RES,
                        index=(tile_m_id, res_chunk),
                        shape=(TILE_SIZE_M, n),
                        padding_mode=PAD_ZERO,
                    ),
                    ct.float32,
                )
            else:
                grad_h_res = ct.full((TILE_SIZE_M, n), 0.0, dtype=ct.float32)
            proj_res = ct.astype(
                ct.load(
                    PROJ,
                    index=(tile_m_id, 2 + res_chunk),
                    shape=(TILE_SIZE_M, n),
                    padding_mode=PAD_ZERO,
                ),
                ct.float32,
            )
            grad_proj_res = grad_h_res * alpha_res * inv_r_eps
            grad_r_from_h += ct.sum(
                grad_h_res * proj_res * alpha_res * (-inv_r_eps * inv_r_eps),
                axis=1,
                keepdims=True,
            )
            ct.store(
                GRAD_H,
                index=(tile_m_id, 2 + res_chunk),
                tile=grad_h_res.astype(GRAD_H.dtype),
            )
            ct.store(
                GRAD_PROJ,
                index=(tile_m_id, 2 + res_chunk),
                tile=grad_proj_res.astype(GRAD_PROJ.dtype),
            )

        if HAS_GRAD_R_EXT:
            grad_r_ext_tile = ct.astype(
                ct.load(
                    GRAD_R_EXT,
                    index=(tile_m_id, 0),
                    shape=(TILE_SIZE_M, 1),
                    padding_mode=PAD_ZERO,
                ),
                ct.float32,
            )
        else:
            grad_r_ext_tile = ct.full((TILE_SIZE_M, 1), 0.0, dtype=ct.float32)
        grad_r_total = grad_r_from_h + grad_r_ext_tile
        ct.store(
            GRAD_R_TOTAL,
            index=(tile_m_id, 0),
            tile=grad_r_total.astype(GRAD_R_TOTAL.dtype),
        )

    @ct.kernel
    def _ct_fused_grad_x_weight_kernel(
        X,
        WEIGHT,
        GRAD_PROJ,
        GRAD_R_TOTAL,
        R,
        GRAD_X,
        GRAD_WEIGHT,
        M: int,
        N: int,
        K: int,
        TILE_SIZE_M: ConstInt,
        TILE_SIZE_N: ConstInt,
        TILE_SIZE_K: ConstInt,
    ):
        """Compute grad_x and grad_weight simultaneously.

        Grid: (ceil(K / TILE_SIZE_K),).
        Each block handles one K-tile and loops over all M-tiles.
        Per M-tile: computes and stores grad_x, accumulates grad_weight.
        """
        tile_k_id = ct.bid(0)
        NUM_M_TILES = ct.cdiv(M, TILE_SIZE_M)
        weight_tile = ct.load(
            WEIGHT,
            index=(0, tile_k_id),
            shape=(TILE_SIZE_N, TILE_SIZE_K),
            padding_mode=PAD_ZERO,
        )
        acc_grad_weight = ct.full(
            (TILE_SIZE_K, TILE_SIZE_N), 0.0, dtype=ct.float32
        )

        for tile_m_id in range(NUM_M_TILES):
            grad_proj_tile = ct.load(
                GRAD_PROJ,
                index=(tile_m_id, 0),
                shape=(TILE_SIZE_M, TILE_SIZE_N),
                padding_mode=PAD_ZERO,
            )
            x_tile = ct.load(
                X,
                index=(tile_m_id, tile_k_id),
                shape=(TILE_SIZE_M, TILE_SIZE_K),
                padding_mode=PAD_ZERO,
            )
            grad_r_total = ct.load(
                GRAD_R_TOTAL,
                index=(tile_m_id, 0),
                shape=(TILE_SIZE_M, 1),
                padding_mode=PAD_ZERO,
            )
            r_tile = ct.astype(
                ct.load(
                    R,
                    index=(tile_m_id, 0),
                    shape=(TILE_SIZE_M, 1),
                    padding_mode=PAD_ZERO,
                ),
                ct.float32,
            )

            inv_rK = 1.0 / (r_tile * K)
            acc_grad_x = (grad_r_total * inv_rK) * ct.astype(x_tile, ct.float32)
            acc_grad_x = ct.mma(
                grad_proj_tile.astype(ct.tfloat32),
                weight_tile.astype(ct.tfloat32),
                acc=acc_grad_x,
            )
            ct.store(
                GRAD_X,
                index=(tile_m_id, tile_k_id),
                tile=acc_grad_x.astype(GRAD_X.dtype),
            )
            acc_grad_weight = ct.mma(
                x_tile.transpose().astype(ct.tfloat32),
                grad_proj_tile.astype(ct.tfloat32),
                acc=acc_grad_weight,
            )

        ct.store(
            GRAD_WEIGHT,
            index=(0, tile_k_id),
            tile=acc_grad_weight.transpose().astype(GRAD_WEIGHT.dtype),
        )

    @ct.kernel
    def _ct_fused_compute_h_proj_rms_bwd_small_k_kernel(
        X,
        WEIGHT,
        GRAD_PROJ,
        GRAD_R_TOTAL,
        R,
        GRAD_X,
        GRAD_WEIGHT,
        M: int,
        N: int,
        K: int,
        TILE_N_SIZE: ConstInt,
    ):
        """Fused backward (small K path) with work-stealing.

        Grid: (num_sms, 2).
        bid(1)==0: grad_weight via work-stealing over K-tiles, loops M.
        bid(1)==1: grad_x via work-stealing over (M×K) tiles.
        Scalar gradients are computed by the separate partial/reduce kernels.
        """
        zero_pad = ct.PaddingMode.ZERO
        TILE_DB_SIZE_M = 128
        TILE_DB_SIZE_K = 64
        NUM_M_TILES = ct.cdiv(M, TILE_DB_SIZE_M)
        NUM_K_TILES = ct.cdiv(K, TILE_DB_SIZE_K)

        if ct.bid(1) == 0:
            for tile_id in range(ct.bid(0), NUM_K_TILES, ct.num_blocks(0)):
                accumulator_db = ct.full(
                    (TILE_DB_SIZE_K, TILE_N_SIZE), 0.0, dtype=ct.float32
                )
                for m_tile in range(NUM_M_TILES):
                    x_tile = ct.load(
                        X,
                        index=(m_tile, tile_id),
                        shape=(TILE_DB_SIZE_M, TILE_DB_SIZE_K),
                        padding_mode=zero_pad,
                    )
                    grad_proj_tile = ct.load(
                        GRAD_PROJ,
                        index=(m_tile, 0),
                        shape=(TILE_DB_SIZE_M, TILE_N_SIZE),
                        padding_mode=zero_pad,
                    )
                    accumulator_db = ct.mma(
                        x_tile.transpose().astype(ct.tfloat32),
                        grad_proj_tile.astype(ct.tfloat32),
                        acc=accumulator_db,
                    )
                ct.store(
                    GRAD_WEIGHT,
                    index=(0, tile_id),
                    tile=accumulator_db.transpose().astype(GRAD_WEIGHT.dtype),
                    allow_tma=False,
                )

        TILE_DA_SIZE_M = 128
        TILE_DA_SIZE_K = 256
        NUM_DA_TILES = ct.cdiv(M, TILE_DA_SIZE_M) * ct.cdiv(K, TILE_DA_SIZE_K)
        NUM_DA_K_TILES = ct.cdiv(K, TILE_DA_SIZE_K)

        if ct.bid(1) == 1:
            for tile_id in range(ct.bid(0), NUM_DA_TILES, ct.num_blocks(0)):
                b_tile_idx = tile_id % NUM_DA_K_TILES
                dd_tile_idx = tile_id // NUM_DA_K_TILES
                grad_proj_tile = ct.load(
                    GRAD_PROJ,
                    index=(dd_tile_idx, 0),
                    shape=(TILE_DA_SIZE_M, TILE_N_SIZE),
                    padding_mode=zero_pad,
                )
                grad_r_total = ct.load(
                    GRAD_R_TOTAL,
                    index=(dd_tile_idx, 0),
                    shape=(TILE_DA_SIZE_M, 1),
                    padding_mode=zero_pad,
                )
                r_tile = ct.astype(
                    ct.load(
                        R,
                        index=(dd_tile_idx, 0),
                        shape=(TILE_DA_SIZE_M, 1),
                        padding_mode=zero_pad,
                    ),
                    ct.float32,
                )
                x_tile = ct.load(
                    X,
                    index=(dd_tile_idx, b_tile_idx),
                    shape=(TILE_DA_SIZE_M, TILE_DA_SIZE_K),
                    padding_mode=zero_pad,
                )
                inv_rK = 1.0 / (r_tile * K)
                accumulator_da = (grad_r_total * inv_rK) * ct.astype(
                    x_tile, ct.float32
                )
                weight_tile = ct.load(
                    WEIGHT,
                    index=(0, b_tile_idx),
                    shape=(TILE_N_SIZE, TILE_DA_SIZE_K),
                    padding_mode=zero_pad,
                )
                accumulator_da = ct.mma(
                    grad_proj_tile.astype(ct.tfloat32),
                    weight_tile.astype(ct.tfloat32),
                    acc=accumulator_da,
                )
                ct.store(
                    GRAD_X,
                    index=(dd_tile_idx, b_tile_idx),
                    tile=accumulator_da.astype(GRAD_X.dtype),
                )

    @ct.kernel
    def _ct_scalar_grads_partials_kernel(
        GRAD_H,
        PROJ,
        R,
        GRAD_ALPHA_PRE_PARTIALS,
        GRAD_ALPHA_POST_PARTIALS,
        GRAD_ALPHA_RES_PARTIALS,
        GRAD_BIAS_PARTIALS,
        M: int,
        N: int,
        n: int,
        eps: float,
        TILE_SIZE_M: ConstInt,
        TILE_SIZE_N: ConstInt,
    ):
        """Compute per-M-tile scalar-gradient partials.

        Grid: (ceil(M / TILE_SIZE_M),).  Each block processes one M-tile.
        """
        bid_m = ct.bid(0)
        offsets = ct.arange(TILE_SIZE_N, dtype=ct.int32)
        one = ct.full((TILE_SIZE_N,), 1.0, dtype=ct.float32)
        zero = ct.full((TILE_SIZE_N,), 0.0, dtype=ct.float32)
        mask_pre = ct.where(ct.less(offsets, n), one, zero)
        mask_post = ct.where(ct.less(offsets, 2 * n), one, zero) - mask_pre
        mask_res = one - mask_pre - mask_post
        mask_pre_2d = ct.reshape(mask_pre, (1, TILE_SIZE_N))
        mask_post_2d = ct.reshape(mask_post, (1, TILE_SIZE_N))
        mask_res_2d = ct.reshape(mask_res, (1, TILE_SIZE_N))

        grad_h = ct.load(
            GRAD_H,
            index=(bid_m, 0),
            shape=(TILE_SIZE_M, TILE_SIZE_N),
            padding_mode=PAD_ZERO,
        )
        proj_tile = ct.astype(
            ct.load(
                PROJ,
                index=(bid_m, 0),
                shape=(TILE_SIZE_M, TILE_SIZE_N),
                padding_mode=PAD_ZERO,
            ),
            ct.float32,
        )
        r_tile = ct.astype(
            ct.load(
                R,
                index=(bid_m, 0),
                shape=(TILE_SIZE_M, 1),
                padding_mode=PAD_ZERO,
            ),
            ct.float32,
        )
        inv_r_eps = 1.0 / (r_tile + eps)

        ga_all = grad_h * proj_tile * inv_r_eps
        ga_pre = ct.reshape(ct.sum(ga_all * mask_pre_2d), (1, 1))
        ga_post = ct.reshape(ct.sum(ga_all * mask_post_2d), (1, 1))
        ga_res = ct.reshape(ct.sum(ga_all * mask_res_2d), (1, 1))
        partial_gb = ct.sum(grad_h, axis=0, keepdims=False)
        ct.store(
            GRAD_ALPHA_PRE_PARTIALS,
            index=(bid_m, 0),
            tile=ga_pre.astype(GRAD_ALPHA_PRE_PARTIALS.dtype),
        )
        ct.store(
            GRAD_ALPHA_POST_PARTIALS,
            index=(bid_m, 0),
            tile=ga_post.astype(GRAD_ALPHA_POST_PARTIALS.dtype),
        )
        ct.store(
            GRAD_ALPHA_RES_PARTIALS,
            index=(bid_m, 0),
            tile=ga_res.astype(GRAD_ALPHA_RES_PARTIALS.dtype),
        )
        ct.store(
            GRAD_BIAS_PARTIALS,
            index=(bid_m, 0),
            tile=ct.reshape(partial_gb, (1, TILE_SIZE_N)).astype(
                GRAD_BIAS_PARTIALS.dtype
            ),
        )

    @ct.kernel
    def _ct_scalar_grads_reduce_kernel(
        GRAD_ALPHA_PRE_PARTIALS,
        GRAD_ALPHA_POST_PARTIALS,
        GRAD_ALPHA_RES_PARTIALS,
        GRAD_BIAS_PARTIALS,
        GRAD_ALPHA_PRE,
        GRAD_ALPHA_POST,
        GRAD_ALPHA_RES,
        GRAD_BIAS,
        NUM_M_BLOCKS: int,
        TILE_SIZE_N: ConstInt,
    ):
        """Reduce scalar-gradient partials and write final dtype outputs."""
        acc_pre = ct.full((1, 1), 0.0, dtype=ct.float32)
        acc_post = ct.full((1, 1), 0.0, dtype=ct.float32)
        acc_res = ct.full((1, 1), 0.0, dtype=ct.float32)
        acc_bias = ct.full((1, TILE_SIZE_N), 0.0, dtype=ct.float32)
        for bid_m in range(NUM_M_BLOCKS):
            acc_pre += ct.load(
                GRAD_ALPHA_PRE_PARTIALS,
                index=(bid_m, 0),
                shape=(1, 1),
                padding_mode=PAD_ZERO,
            ).astype(ct.float32)
            acc_post += ct.load(
                GRAD_ALPHA_POST_PARTIALS,
                index=(bid_m, 0),
                shape=(1, 1),
                padding_mode=PAD_ZERO,
            ).astype(ct.float32)
            acc_res += ct.load(
                GRAD_ALPHA_RES_PARTIALS,
                index=(bid_m, 0),
                shape=(1, 1),
                padding_mode=PAD_ZERO,
            ).astype(ct.float32)
            acc_bias += ct.load(
                GRAD_BIAS_PARTIALS,
                index=(bid_m, 0),
                shape=(1, TILE_SIZE_N),
                padding_mode=PAD_ZERO,
            ).astype(ct.float32)
        ct.store(
            GRAD_ALPHA_PRE,
            index=(0, 0),
            tile=acc_pre.astype(GRAD_ALPHA_PRE.dtype),
        )
        ct.store(
            GRAD_ALPHA_POST,
            index=(0, 0),
            tile=acc_post.astype(GRAD_ALPHA_POST.dtype),
        )
        ct.store(
            GRAD_ALPHA_RES,
            index=(0, 0),
            tile=acc_res.astype(GRAD_ALPHA_RES.dtype),
        )
        ct.store(GRAD_BIAS, index=(0, 0), tile=acc_bias.astype(GRAD_BIAS.dtype))

    def _cutile_fused_compute_h_proj_rms_bwd(
        x,
        weight,
        grad_h_pre,
        grad_h_post,
        grad_h_res,
        h_pre,
        h_post,
        h_res,
        proj,
        r,
        grad_r_ext,
        alpha_pre,
        alpha_post,
        alpha_res,
        bias,
        n,
        eps,
        compute_h_eps,
    ):
        """Fused compute_h + proj_rms backward.

        Returns:
            grad_x: [M, K]
            grad_weight: [N, K]
            grad_alpha_pre: [1]
            grad_alpha_post: [1]
            grad_alpha_res: [1]
            grad_bias: [N]
        """
        M, K = x.shape
        N = weight.shape[0]
        TILE_N = _next_power_of_2(N)
        assert TILE_N <= 256
        stream = _get_cuda_stream()

        grad_x = paddle.empty(shape=x.shape, dtype=x.dtype)
        grad_weight = paddle.empty(shape=weight.shape, dtype=weight.dtype)
        has_grad_r_ext = grad_r_ext is not None
        grad_r_ext_arg = grad_r_ext if has_grad_r_ext else r
        has_grad_h_pre = grad_h_pre is not None
        has_grad_h_post = grad_h_post is not None
        has_grad_h_res = grad_h_res is not None
        grad_h_pre_arg = grad_h_pre if has_grad_h_pre else h_pre
        grad_h_post_arg = grad_h_post if has_grad_h_post else h_post
        grad_h_res_arg = grad_h_res if has_grad_h_res else h_res

        grad_h_buf = paddle.empty(shape=[M, TILE_N], dtype="float32")
        grad_proj_buf = paddle.empty(shape=[M, TILE_N], dtype="float32")
        grad_r_total_buf = paddle.empty(shape=[M, 1], dtype="float32")

        tile_m_precomp = _default_tile_m(M)
        ct.launch(
            stream,
            (math.ceil(M / tile_m_precomp),),
            _ct_fused_grad_h_proj_kernel,
            (
                grad_h_pre_arg,
                grad_h_post_arg,
                grad_h_res_arg,
                h_pre,
                h_post,
                proj,
                r,
                grad_r_ext_arg,
                alpha_pre,
                alpha_post,
                alpha_res,
                grad_h_buf,
                grad_proj_buf,
                grad_r_total_buf,
                M,
                N,
                n,
                eps,
                compute_h_eps,
                tile_m_precomp,
                TILE_N,
                int(has_grad_h_pre),
                int(has_grad_h_post),
                int(has_grad_h_res),
                int(has_grad_r_ext),
            ),
        )

        if K >= 8192:
            tm, tn, tk = 128, TILE_N, 128
            ct.launch(
                stream,
                (math.ceil(K / tk),),
                _ct_fused_grad_x_weight_kernel,
                (
                    x,
                    weight,
                    grad_proj_buf,
                    grad_r_total_buf,
                    r,
                    grad_x,
                    grad_weight,
                    M,
                    N,
                    K,
                    tm,
                    tn,
                    tk,
                ),
            )
        else:
            num_sms = (
                paddle.device.cuda.get_device_properties().multi_processor_count
            )
            ct.launch(
                stream,
                (num_sms, 2, 1),
                _ct_fused_compute_h_proj_rms_bwd_small_k_kernel,
                (
                    x,
                    weight,
                    grad_proj_buf,
                    grad_r_total_buf,
                    r,
                    grad_x,
                    grad_weight,
                    M,
                    N,
                    K,
                    TILE_N,
                ),
            )

        tile_m_scalar = min(128, M)
        num_m_blocks = math.ceil(M / tile_m_scalar)
        grad_alpha_pre_partials = paddle.empty(
            shape=[num_m_blocks, 1], dtype="float32"
        )
        grad_alpha_post_partials = paddle.empty(
            shape=[num_m_blocks, 1], dtype="float32"
        )
        grad_alpha_res_partials = paddle.empty(
            shape=[num_m_blocks, 1], dtype="float32"
        )
        grad_bias_partials = paddle.empty(
            shape=[num_m_blocks, TILE_N], dtype="float32"
        )
        grad_alpha_pre_out = paddle.empty(shape=[1, 1], dtype=alpha_pre.dtype)
        grad_alpha_post_out = paddle.empty(shape=[1, 1], dtype=alpha_post.dtype)
        grad_alpha_res_out = paddle.empty(shape=[1, 1], dtype=alpha_res.dtype)
        grad_bias_out = paddle.empty(shape=[1, TILE_N], dtype=bias.dtype)

        ct.launch(
            stream,
            (num_m_blocks,),
            _ct_scalar_grads_partials_kernel,
            (
                grad_h_buf,
                proj,
                r,
                grad_alpha_pre_partials,
                grad_alpha_post_partials,
                grad_alpha_res_partials,
                grad_bias_partials,
                M,
                N,
                n,
                eps,
                tile_m_scalar,
                TILE_N,
            ),
        )
        ct.launch(
            stream,
            (1,),
            _ct_scalar_grads_reduce_kernel,
            (
                grad_alpha_pre_partials,
                grad_alpha_post_partials,
                grad_alpha_res_partials,
                grad_bias_partials,
                grad_alpha_pre_out,
                grad_alpha_post_out,
                grad_alpha_res_out,
                grad_bias_out,
                num_m_blocks,
                TILE_N,
            ),
        )

        return (
            grad_x,
            grad_weight,
            grad_alpha_pre_out.reshape(alpha_pre.shape),
            grad_alpha_post_out.reshape(alpha_post.shape),
            grad_alpha_res_out.reshape(alpha_res.shape),
            grad_bias_out.reshape([-1])[:N],
        )


# ============================================================================
# Unified public API with Triton > cuTile > native dispatch
# ============================================================================


def _get_triton_impl(key):
    if not _TRITON_AVAILABLE:
        return None
    return _TRITON_IMPLS.get(key)


if _CUTILE_AVAILABLE:

    class _CutileSinkhornPyLayer(paddle.autograd.PyLayer):
        @staticmethod
        def forward(ctx, input_logits, num_iterations, eps):
            output, M_init = _cutile_sinkhorn_fwd(
                input_logits, num_iterations, eps
            )
            ctx.save_for_backward(M_init)
            ctx.num_iterations = num_iterations
            ctx.eps = eps
            return output

        @staticmethod
        def backward(ctx, grad_output):
            (M_init,) = ctx.saved_tensor()
            return _cutile_sinkhorn_bwd(
                grad_output, M_init, ctx.num_iterations, ctx.eps
            )

    def _cutile_sinkhorn_apply(input_logits, num_iterations, eps):
        return _CutileSinkhornPyLayer.apply(input_logits, num_iterations, eps)

    class CutileProjRmsComputeH(paddle.autograd.PyLayer):
        """cuTile projection + RMS norm + compute_h activations."""

        @staticmethod
        def forward(
            ctx,
            x,
            weight,
            alpha_pre,
            alpha_post,
            alpha_res,
            bias,
            n,
            eps=1e-6,
            compute_h_eps=1e-6,
        ):
            h_pre, h_post, h_res, r, proj_reduced = (
                _cutile_proj_rms_compute_h_fwd(
                    x,
                    weight,
                    bias,
                    alpha_pre,
                    alpha_post,
                    alpha_res,
                    n,
                    eps,
                    compute_h_eps,
                )
            )
            ctx.save_for_backward(
                x,
                weight,
                h_pre,
                h_post,
                h_res,
                proj_reduced,
                r,
                alpha_pre,
                alpha_post,
                alpha_res,
                bias,
            )
            ctx.n = n
            ctx.eps = eps
            ctx.compute_h_eps = compute_h_eps
            return h_pre, h_post, h_res, r

        @staticmethod
        def backward(ctx, grad_h_pre, grad_h_post, grad_h_res, grad_r_ext):
            (
                x,
                weight,
                h_pre,
                h_post,
                h_res,
                proj,
                r,
                alpha_pre,
                alpha_post,
                alpha_res,
                bias_param,
            ) = ctx.saved_tensor()
            # detach all tensors to allow __dlpack__ access in cuTile kernels
            x, weight = x.detach(), weight.detach()
            h_pre, h_post, h_res = (
                h_pre.detach(),
                h_post.detach(),
                h_res.detach(),
            )
            proj, r = proj.detach(), r.detach()
            alpha_pre, alpha_post, alpha_res = (
                alpha_pre.detach(),
                alpha_post.detach(),
                alpha_res.detach(),
            )
            bias_param = bias_param.detach()
            grad_h_pre = grad_h_pre.detach() if grad_h_pre is not None else None
            grad_h_post = (
                grad_h_post.detach() if grad_h_post is not None else None
            )
            grad_h_res = grad_h_res.detach() if grad_h_res is not None else None
            grad_r_ext = grad_r_ext.detach() if grad_r_ext is not None else None
            grad_x, grad_weight, grad_ap, grad_apo, grad_ar, grad_bias = (
                _cutile_fused_compute_h_proj_rms_bwd(
                    x,
                    weight,
                    grad_h_pre,
                    grad_h_post,
                    grad_h_res,
                    h_pre,
                    h_post,
                    h_res,
                    proj,
                    r,
                    grad_r_ext,
                    alpha_pre,
                    alpha_post,
                    alpha_res,
                    bias_param,
                    ctx.n,
                    ctx.eps,
                    ctx.compute_h_eps,
                )
            )
            return grad_x, grad_weight, grad_ap, grad_apo, grad_ar, grad_bias

else:

    def _cutile_sinkhorn_apply(input_logits, num_iterations, eps):
        raise RuntimeError("cuTile not available")


class FusedHAggregate(paddle.autograd.PyLayer):
    """H_aggregate with Triton/cuTile forward and backward."""

    @staticmethod
    def forward(ctx, x: Tensor, h_pre: Tensor):
        triton_fwd = _get_triton_impl("h_aggregate_fwd")
        if triton_fwd is not None:
            output = triton_fwd(x, h_pre)
        elif _CUTILE_AVAILABLE:
            output = _cutile_h_aggregate_fwd(x, h_pre)
        else:
            raise RuntimeError(
                "cuTile(required) must be available when config.use_fused_mhc is True."
            )
        ctx.save_for_backward(x, h_pre)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, h_pre = ctx.saved_tensor()
        if _CUTILE_AVAILABLE:
            return _cutile_h_aggregate_bwd(grad_output, x, h_pre)
        raise RuntimeError(
            "cuTile(required) must be available when config.use_fused_mhc is True."
        )


class FusedHPostBDA(paddle.autograd.PyLayer):
    """H_post_bda with Triton/cuTile forward and backward."""

    @staticmethod
    def forward(ctx, h_res, original_residual, h_post, x, bias):
        triton_fwd = _get_triton_impl("h_post_bda_fwd")
        if triton_fwd is not None:
            output = triton_fwd(h_res, original_residual, h_post, x, bias)
        elif _CUTILE_AVAILABLE:
            output = _cutile_h_post_bda_fwd(
                h_res, original_residual, h_post, x, bias
            )
        else:
            raise RuntimeError(
                "cuTile(required) must be available when config.use_fused_mhc is True."
            )
        if bias is not None:
            ctx.save_for_backward(h_res, original_residual, h_post, x, bias)
            ctx.has_bias = True
        else:
            ctx.save_for_backward(h_res, original_residual, h_post, x)
            ctx.has_bias = False
        ctx.x_stop_gradient = x.stop_gradient
        ctx.bias_stop_gradient = (
            bias.stop_gradient if bias is not None else True
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.has_bias:
            h_res, orig_res, h_post, x, bias = ctx.saved_tensor()
        else:
            h_res, orig_res, h_post, x = ctx.saved_tensor()
            bias = None
        triton_bwd = _get_triton_impl("h_post_bda_bwd")
        if triton_bwd is not None:
            g_hr, g_res, g_hp, g_x, g_bias = triton_bwd(
                grad_output, h_res, orig_res, h_post, x, bias
            )
        elif _CUTILE_AVAILABLE:
            g_hr, g_res, g_hp, g_x, g_bias = _cutile_h_post_bda_bwd(
                grad_output, h_res, orig_res, h_post, x, bias
            )
        else:
            raise RuntimeError(
                "cuTile(required) must be available when config.use_fused_mhc is True."
            )

        if not ctx.has_bias:
            return g_hr, g_res, g_hp, g_x
        else:
            if ctx.x_stop_gradient:
                g_x = None
            if ctx.bias_stop_gradient:
                g_bias = None
            return g_hr, g_res, g_hp, g_x, g_bias


class FusedProjRms(paddle.autograd.PyLayer):
    """Fused projection + RMS normalization."""

    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, eps: float = 1e-6):
        original_shape = x.shape
        K = original_shape[-1]
        x_2d = x.reshape([-1, K])
        proj, norm, r = _cutile_proj_rms_fwd(x_2d, weight, eps)
        ctx.save_for_backward(x_2d, weight, norm)
        ctx.eps = eps
        ctx.original_shape = original_shape
        N = weight.shape[0]
        batch_shape = list(original_shape[:-1])
        return proj.reshape([*batch_shape, N]), r.reshape([*batch_shape, 1])

    @staticmethod
    def backward(ctx, grad_proj, grad_r):
        x_2d, weight, norm = ctx.saved_tensor()
        original_shape = ctx.original_shape
        grad_proj_2d = grad_proj.reshape([-1, grad_proj.shape[-1]])
        grad_r_2d = grad_r.reshape([-1, 1])
        grad_x, grad_weight = _cutile_proj_rms_bwd(
            grad_proj_2d, grad_r_2d, x_2d, weight, norm, ctx.eps
        )
        return grad_x.reshape(original_shape), grad_weight


# -- Public API functions ------------------------------------------------------


def fused_sinkhorn(
    input_logits: Tensor, num_iterations: int, eps: float = 1e-6
) -> Tensor:
    """Project logits to doubly stochastic matrix (Triton > cuTile).
    Args:
        input_logits: [..., n, n] raw logits
        num_iterations: Sinkhorn iterations
        eps: numerical stability

    Returns:
        [..., n, n] doubly stochastic matrix
    """
    assert input_logits.ndim >= 2, (
        f"fused_sinkhorn: input must be at least 2D, got shape {list(input_logits.shape)}"
    )
    assert input_logits.shape[-1] == input_logits.shape[-2], (
        f"fused_sinkhorn: last two dims must be equal (square matrix), "
        f"got shape {list(input_logits.shape)}"
    )
    hc = input_logits.shape[-1]
    N_batch = input_logits.size // (hc * hc)
    assert N_batch <= _INT32_MAX, (
        f"fused_sinkhorn: N_batch={N_batch} exceeds int32 max ({_INT32_MAX})"
    )

    input_logits = input_logits.contiguous()
    triton_impl = _get_triton_impl("sinkhorn")
    if triton_impl is not None:
        return triton_impl(input_logits, num_iterations, eps)
    if _CUTILE_AVAILABLE:
        return _cutile_sinkhorn_apply(input_logits, num_iterations, eps)
    raise RuntimeError(
        "cuTile(required) must be available when config.use_fused_mhc is True."
    )


def fused_h_aggregate(x: Tensor, h_pre: Tensor) -> Tensor:
    """Weighted n-stream to 1-stream aggregation.

    Args:
        x: [s, b, n, C] n-stream hidden states
        h_pre: [s, b, n] aggregation weights

    Returns:
        [s, b, C] aggregated hidden states
    """
    assert x.ndim == 4, (
        f"fused_h_aggregate: x must be 4D [s,b,n,C], got shape {list(x.shape)}"
    )
    assert h_pre.ndim == 3, (
        f"fused_h_aggregate: h_pre must be 3D [s,b,n], got shape {list(h_pre.shape)}"
    )
    assert x.shape[:3] == h_pre.shape[:3], (
        f"fused_h_aggregate: x shape {list(x.shape)} and h_pre shape {list(h_pre.shape)} "
        f"must match on first 3 dims [s,b,n]"
    )
    s, b, n, C = x.shape
    assert s * b <= _INT32_MAX, (
        f"fused_h_aggregate: s*b={s * b} exceeds int32 max ({_INT32_MAX})"
    )
    assert C <= _INT32_MAX, (
        f"fused_h_aggregate: C={C} exceeds int32 max ({_INT32_MAX})"
    )

    if _TRITON_AVAILABLE or _CUTILE_AVAILABLE:
        return FusedHAggregate.apply(x, h_pre)
    raise RuntimeError(
        "cuTile(required) must be available when config.use_fused_mhc is True."
    )


def fused_h_post_bda(
    h_res: Tensor,
    original_residual: Tensor,
    h_post: Tensor,
    x: Tensor,
    bias: Tensor | None,
) -> Tensor:
    """Fused H_res @ residual + H_post * (x + bias).

    Args:
        h_res: [s, b, n, n] residual mixing matrix
        original_residual: [s, b, n, C] n-stream residual
        h_post: [s, b, n] expansion weights
        x: [s, b, C] layer output
        bias: [C] or None

    Returns:
        [s, b, n, C] fused output
    """
    assert h_res.ndim == 4 and h_res.shape[-1] == h_res.shape[-2], (
        f"fused_h_post_bda: h_res must be 4D [s,b,n,n], got shape {list(h_res.shape)}"
    )
    assert original_residual.ndim == 4, (
        f"fused_h_post_bda: original_residual must be 4D [s,b,n,C], got shape {list(original_residual.shape)}"
    )
    n = h_res.shape[-1]
    assert original_residual.shape[2] == n, (
        f"fused_h_post_bda: original_residual dim2={original_residual.shape[2]} != n={n}"
    )
    assert h_post.ndim == 3 and h_post.shape[-1] == n, (
        f"fused_h_post_bda: h_post must be 3D [s,b,n], got shape {list(h_post.shape)}"
    )
    assert x.ndim == 3 and x.shape[-1] == original_residual.shape[-1], (
        f"fused_h_post_bda: x must be 3D [s,b,C] with C={original_residual.shape[-1]}, got shape {list(x.shape)}"
    )
    s, b = original_residual.shape[:2]
    C = original_residual.shape[-1]
    assert s * b <= _INT32_MAX, (
        f"fused_h_post_bda: s*b={s * b} exceeds int32 max ({_INT32_MAX})"
    )
    assert C <= _INT32_MAX, (
        f"fused_h_post_bda: C={C} exceeds int32 max ({_INT32_MAX})"
    )

    if _TRITON_AVAILABLE or _CUTILE_AVAILABLE:
        return FusedHPostBDA.apply(h_res, original_residual, h_post, x, bias)
    raise RuntimeError(
        "cuTile(required) must be available when config.use_fused_mhc is True."
    )


def fused_proj_rms(
    x: Tensor, weight: Tensor, eps: float = 1e-6
) -> tuple[Tensor, Tensor]:
    """Fused projection + RMS normalization.

    Args:
        x: [..., K] input (last dim is K)
        weight: [K, N] projection weight
        eps: stability epsilon

    Returns:
        proj: [..., N] = x @ weight^T
        r: [..., 1] = 1 / (||x|| / sqrt(K) + eps)
    """
    # [K, N] --> [N, K]
    weight = weight.t()
    assert weight.ndim == 2, (
        f"fused_proj_rms: weight must be 2D [N, K], got shape {list(weight.shape)}"
    )
    K = x.shape[-1]
    N, K_w = weight.shape
    assert K == K_w, (
        f"fused_proj_rms: x last dim (K={K}) must match weight dim1 (K={K_w}). "
        f"x.shape={list(x.shape)}, weight.shape={list(weight.shape)}. "
        f"If weight is [K, N], you need to transpose it: fused_proj_rms(x, weight.t())"
    )
    assert N <= 256, (
        f"fused_proj_rms: N={N} exceeds max supported tile size 256. "
        f"weight.shape={list(weight.shape)}. Check if weight needs transposing."
    )
    M = x.size // K
    assert M <= _INT32_MAX, (
        f"fused_proj_rms: M={M} (x reshaped to [M, K]) exceeds int32 max ({_INT32_MAX})"
    )
    assert K <= _INT32_MAX, (
        f"fused_proj_rms: K={K} exceeds int32 max ({_INT32_MAX})"
    )

    if _CUTILE_AVAILABLE:
        return FusedProjRms.apply(x, weight, eps)
    raise RuntimeError(
        "cuTile(required) must be available when config.use_fused_mhc is True."
    )


def fused_proj_rms_compute_h(
    x: Tensor,
    weight: Tensor,
    alpha_pre: Tensor,
    alpha_post: Tensor,
    alpha_res: Tensor,
    bias: Tensor,
    n: int,
    eps: float = 1e-6,
    compute_h_eps: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Fused projection + RMS norm + compute_h (cuTile).

    Combines three steps in one kernel launch:
        1. proj = x @ weight^T, r = 1 / (||x|| / sqrt(K) + eps)
        2. h = r * proj * alpha + bias
        3. Split h into h_pre (sigmoid+eps), h_post (2*sigmoid), h_res

    Args:
        x: [M, K] input hidden states (K = n * C)
        weight: [K, N] projection weight (N = n^2 + 2n)
        alpha_pre: [1] scaling factor for h_pre
        alpha_post: [1] scaling factor for h_post
        alpha_res: [1] scaling factor for h_res
        bias: [N] bias vector (N = n^2 + 2n)
        n: number of residual streams
        eps: RMS norm stability epsilon
        compute_h_eps: epsilon added to sigmoid(h_pre) for stability

    Returns:
        h_pre: [M, n] aggregation weights (sigmoid + compute_h_eps)
        h_post: [M, n] expansion weights (2 * sigmoid)
        h_res: [M, n*n] residual mixing logits
        r: [M, 1] inverse RMS scaling factor
    """
    assert x.ndim == 2, (
        f"fused_proj_rms_compute_h: x must be 2D [M, K], got shape {list(x.shape)}"
    )
    assert weight.ndim == 2, (
        f"fused_proj_rms_compute_h: weight must be 2D [K, N], got shape {list(weight.shape)}"
    )
    M, K = x.shape
    K_w, N = weight.shape
    assert K == K_w, (
        f"fused_proj_rms_compute_h: x last dim (K={K}) must match weight dim0 (K={K_w}). "
        f"x.shape={list(x.shape)}, weight.shape={list(weight.shape)}"
    )
    expected_N = n * n + 2 * n
    assert N == expected_N, (
        f"fused_proj_rms_compute_h: weight dim1 (N={N}) must equal n^2+2n={expected_N} for n={n}"
    )
    assert alpha_pre.size == 1, (
        f"fused_proj_rms_compute_h: alpha_pre must be scalar [1], got shape {list(alpha_pre.shape)}"
    )
    assert alpha_post.size == 1, (
        f"fused_proj_rms_compute_h: alpha_post must be scalar [1], got shape {list(alpha_post.shape)}"
    )
    assert alpha_res.size == 1, (
        f"fused_proj_rms_compute_h: alpha_res must be scalar [1], got shape {list(alpha_res.shape)}"
    )
    assert bias.shape[-1] == N, (
        f"fused_proj_rms_compute_h: bias last dim must be N={N}, got shape {list(bias.shape)}"
    )
    assert M * K <= _INT32_MAX, (
        f"fused_proj_rms_compute_h: x address offset M*K={M * K} exceeds int32 max ({_INT32_MAX})"
    )
    assert M * N <= _INT32_MAX, (
        f"fused_proj_rms_compute_h: output address offset M*N={M * N} exceeds int32 max ({_INT32_MAX})"
    )

    if _CUTILE_AVAILABLE:
        return CutileProjRmsComputeH.apply(
            x,
            weight.t(),
            alpha_pre,
            alpha_post,
            alpha_res,
            bias,
            n,
            eps,
            compute_h_eps,
        )
    raise RuntimeError(
        "cuTile(required) must be available when config.use_fused_mhc is True."
    )
