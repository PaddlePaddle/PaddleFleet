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

"""Fused Triton kernels for Kimi-K3 SiTU-GLU with router scaling."""

from __future__ import annotations

import math

import paddle

from .utils import enable_compat_on_triton_kernel, is_torch_compat_available

if is_torch_compat_available():
    paddle.enable_compat(scope={"triton"})

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _sigmoid_precise(x):
    return libdevice.div_rn(1.0, 1.0 + libdevice.exp(-x))


@enable_compat_on_triton_kernel
@triton.jit
def _situ_glu_scale_fwd_kernel(
    x_ptr,
    probs_ptr,
    out_ptr,
    n_cols: tl.constexpr,
    beta: tl.constexpr,
    linear_beta: tl.constexpr,
    has_linear_tanh: tl.constexpr,
    block_n: tl.constexpr,
):
    # Long-context MoE batches can make row * stride exceed int32. Keep all
    # flattened pointer offsets in int64.
    row = tl.program_id(0).to(tl.int64)
    col_block = tl.program_id(1)
    cols = col_block * block_n + tl.arange(0, block_n)
    mask = cols < n_cols

    gate = tl.load(x_ptr + row * (2 * n_cols) + cols, mask=mask).to(tl.float32)
    up = tl.load(x_ptr + row * (2 * n_cols) + n_cols + cols, mask=mask).to(
        tl.float32
    )
    prob = tl.load(probs_ptr + row).to(tl.float32)

    gate_tanh = libdevice.tanh(gate / beta)
    gate_act = beta * gate_tanh * _sigmoid_precise(gate)
    if has_linear_tanh:
        up = linear_beta * libdevice.tanh(up / linear_beta)
    out = gate_act * up * prob
    tl.store(out_ptr + row * n_cols + cols, out, mask=mask)


@enable_compat_on_triton_kernel
@triton.jit
def _situ_glu_scale_bwd_kernel(
    x_ptr,
    probs_ptr,
    out_grad_ptr,
    x_grad_ptr,
    recomputed_ptr,
    probs_grad_ptr,
    n_cols: tl.constexpr,
    beta: tl.constexpr,
    linear_beta: tl.constexpr,
    has_linear_tanh: tl.constexpr,
    block_n: tl.constexpr,
):
    # Long-context MoE batches can make row * stride exceed int32. Keep all
    # flattened pointer offsets in int64.
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, block_n)
    mask = cols < n_cols

    gate = tl.load(x_ptr + row * (2 * n_cols) + cols, mask=mask).to(tl.float32)
    up = tl.load(x_ptr + row * (2 * n_cols) + n_cols + cols, mask=mask).to(
        tl.float32
    )
    out_grad = tl.load(
        out_grad_ptr + row * n_cols + cols, mask=mask, other=0.0
    ).to(tl.float32)
    prob = tl.load(probs_ptr + row).to(tl.float32)

    gate_tanh = libdevice.tanh(gate / beta)
    gate_sigmoid = _sigmoid_precise(gate)
    gate_act = beta * gate_tanh * gate_sigmoid
    gate_grad = (
        1.0 - gate_tanh * gate_tanh
    ) * gate_sigmoid + beta * gate_tanh * gate_sigmoid * (1.0 - gate_sigmoid)
    if has_linear_tanh:
        up_tanh = libdevice.tanh(up / linear_beta)
        up_act = linear_beta * up_tanh
        up_grad = 1.0 - up_tanh * up_tanh
    else:
        up_act = up
        up_grad = 1.0

    unscaled = gate_act * up_act
    scaled_grad = out_grad * prob
    gate_input_grad = scaled_grad * up_act * gate_grad
    up_input_grad = scaled_grad * gate_act * up_grad

    tl.store(
        x_grad_ptr + row * (2 * n_cols) + cols,
        gate_input_grad,
        mask=mask,
    )
    tl.store(
        x_grad_ptr + row * (2 * n_cols) + n_cols + cols,
        up_input_grad,
        mask=mask,
    )
    tl.store(
        recomputed_ptr + row * n_cols + cols,
        unscaled * prob,
        mask=mask,
    )
    probs_grad = tl.sum(tl.where(mask, out_grad * unscaled, 0.0), axis=0)
    tl.store(probs_grad_ptr + row, probs_grad)


@enable_compat_on_triton_kernel
@triton.jit
def _situ_glu_scale_bwd_segmented_kernel(
    x_ptr,
    probs_ptr,
    out_grad_ptr,
    x_grad_ptr,
    recomputed_ptr,
    probs_grad_ptr,
    n_cols: tl.constexpr,
    beta: tl.constexpr,
    linear_beta: tl.constexpr,
    has_linear_tanh: tl.constexpr,
    chunk_n: tl.constexpr,
):
    # N=3072 padded to 4096 makes the full-row kernel register-heavy. Process
    # three exact 1024-element chunks in one CTA so each chunk's intermediates
    # die before the next one, while preserving one final router-score grad.
    row = tl.program_id(0).to(tl.int64)
    lane_cols = tl.arange(0, chunk_n)
    probs_grad = 0.0

    for chunk_start in tl.range(
        0,
        n_cols,
        chunk_n,
        loop_unroll_factor=1,
        disable_licm=True,
    ):
        cols = chunk_start + lane_cols
        gate = tl.load(x_ptr + row * (2 * n_cols) + cols).to(tl.float32)
        up = tl.load(x_ptr + row * (2 * n_cols) + n_cols + cols).to(tl.float32)
        out_grad = tl.load(out_grad_ptr + row * n_cols + cols).to(tl.float32)
        prob = tl.load(probs_ptr + row).to(tl.float32)

        gate_tanh = libdevice.tanh(gate / beta)
        gate_sigmoid = _sigmoid_precise(gate)
        gate_act = beta * gate_tanh * gate_sigmoid
        gate_grad = (
            1.0 - gate_tanh * gate_tanh
        ) * gate_sigmoid + beta * gate_tanh * gate_sigmoid * (
            1.0 - gate_sigmoid
        )
        if has_linear_tanh:
            up_tanh = libdevice.tanh(up / linear_beta)
            up_act = linear_beta * up_tanh
            up_grad = 1.0 - up_tanh * up_tanh
        else:
            up_act = up
            up_grad = 1.0

        unscaled = gate_act * up_act
        scaled_grad = out_grad * prob
        tl.store(
            x_grad_ptr + row * (2 * n_cols) + cols,
            scaled_grad * up_act * gate_grad,
        )
        tl.store(
            x_grad_ptr + row * (2 * n_cols) + n_cols + cols,
            scaled_grad * gate_act * up_grad,
        )
        tl.store(recomputed_ptr + row * n_cols + cols, unscaled * prob)
        probs_grad += tl.sum(out_grad * unscaled, axis=0)

    tl.store(probs_grad_ptr + row, probs_grad)


def _validate_inputs(
    x: paddle.Tensor,
    probs: paddle.Tensor,
    out_grad: paddle.Tensor | None = None,
) -> tuple[int, int]:
    tensors = (x, probs) if out_grad is None else (x, probs, out_grad)
    if any(not tensor.place.is_gpu_place() for tensor in tensors):
        raise ValueError("SiTU-GLU Triton inputs must be GPU tensors")
    if any(tensor.place != x.place for tensor in tensors[1:]):
        raise ValueError("SiTU-GLU Triton inputs must be on the same GPU")
    if x.ndim != 2 or x.shape[1] % 2:
        raise ValueError(f"x must have shape [M, 2N], got {list(x.shape)}")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if x.dtype not in (paddle.float16, paddle.bfloat16, paddle.float32):
        raise TypeError(f"unsupported x dtype: {x.dtype}")
    rows, n_cols = int(x.shape[0]), int(x.shape[1]) // 2
    if list(probs.shape) not in ([rows], [rows, 1]):
        raise ValueError(
            f"probs must have shape [M] or [M, 1], got {list(probs.shape)}"
        )
    if not probs.is_contiguous():
        raise ValueError("probs must be contiguous")
    if probs.dtype not in (paddle.float16, paddle.bfloat16, paddle.float32):
        raise TypeError(f"unsupported probs dtype: {probs.dtype}")
    if out_grad is not None:
        if list(out_grad.shape) != [rows, n_cols]:
            raise ValueError(
                f"out_grad must have shape [M, N], got {list(out_grad.shape)}"
            )
        if out_grad.dtype != x.dtype:
            raise TypeError(
                f"out_grad dtype must match x: {out_grad.dtype} vs {x.dtype}"
            )
        if not out_grad.is_contiguous():
            raise ValueError("out_grad must be contiguous")
    return rows, n_cols


def situ_glu_scale_forward_triton(
    x: paddle.Tensor,
    probs: paddle.Tensor,
    beta: float = 1.0,
    linear_beta: float | None = None,
) -> paddle.Tensor:
    """Compute ``SiTU-GLU(x) * probs`` in one Triton kernel."""

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
    rows, n_cols = _validate_inputs(x, probs)
    out = paddle.empty([rows, n_cols], dtype=x.dtype)
    if rows == 0 or n_cols == 0:
        return out
    block_n = min(1024, triton.next_power_of_2(n_cols))
    grid = (rows, triton.cdiv(n_cols, block_n))
    # 8 elements per thread, i.e. 32 B in flight per thread. The old rule
    # (`8 if block_n >= 512 else 4`) left only 4 elements per thread at
    # block_n=1024, which is the shape this kernel actually runs at. Measured at
    # M=82432 / N=2048 / bf16, the fastest point at every block_n is the
    # "8 elements/thread" diagonal: 512:w2 328.0, 1024:w4 327.4, 2048:w8 331.6
    # us, versus 338.4 us for the old rule's 1024:w8.
    # This is bit-exact, not a numeric tradeoff: the forward kernel has no
    # cross-element reduction, so each output element's arithmetic is
    # independent of which block or thread computes it (verified over all
    # 168 820 736 elements at M=82432, zero differing bits).
    _situ_glu_scale_fwd_kernel[grid](
        x,
        probs,
        out,
        n_cols=n_cols,
        beta=float(beta),
        linear_beta=1.0 if linear_beta is None else float(linear_beta),
        has_linear_tanh=linear_beta is not None,
        block_n=block_n,
        num_warps=max(1, min(32, block_n // 256)),
    )
    return out


def situ_glu_scale_backward_triton(
    x: paddle.Tensor,
    probs: paddle.Tensor,
    out_grad: paddle.Tensor,
    beta: float = 1.0,
    linear_beta: float | None = None,
) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """Fused SiTU-GLU backward, including recompute and router-score grad."""

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
    rows, n_cols = _validate_inputs(x, probs, out_grad)
    x_grad = paddle.empty_like(x)
    recomputed = paddle.empty([rows, n_cols], dtype=x.dtype)
    probs_grad = paddle.empty_like(probs)
    if rows == 0 or n_cols == 0:
        return x_grad, recomputed, probs_grad
    if n_cols == 3072:
        _situ_glu_scale_bwd_segmented_kernel[(rows,)](
            x,
            probs,
            out_grad,
            x_grad,
            recomputed,
            probs_grad,
            n_cols=n_cols,
            beta=float(beta),
            linear_beta=1.0 if linear_beta is None else float(linear_beta),
            has_linear_tanh=linear_beta is not None,
            chunk_n=1024,
            num_warps=4,
        )
        return x_grad, recomputed, probs_grad

    block_n = triton.next_power_of_2(n_cols)
    if block_n > 65536:
        raise ValueError(f"SiTU-GLU width is too large for one row: {n_cols}")
    _situ_glu_scale_bwd_kernel[(rows,)](
        x,
        probs,
        out_grad,
        x_grad,
        recomputed,
        probs_grad,
        n_cols=n_cols,
        beta=float(beta),
        linear_beta=1.0 if linear_beta is None else float(linear_beta),
        has_linear_tanh=linear_beta is not None,
        block_n=block_n,
        num_warps=8 if block_n >= 2048 else 4,
    )
    return x_grad, recomputed, probs_grad


__all__ = [
    "situ_glu_scale_backward_triton",
    "situ_glu_scale_forward_triton",
]
