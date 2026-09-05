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

"""Fused Triton SiTU-GLU **without** router scaling (the plain ``situ_glu``).

Why a second file next to ``situ_glu.py``: that one fuses SiTU-GLU *and* the
per-token router scale, and its whole signature is built around ``probs`` plus
the ``recomputed`` / ``probs_grad`` outputs the fp8 grouped-GEMM path needs.
The dense call sites -- ``mlp.py`` (which is what ``StandardMLPSharedExpert``
and the ``first_k_dense_replace`` layer run) and the non-fp8
``moe_expert.py`` -- have no ``probs`` at all and otherwise go through the eager
op composition in ``activations.situ_glu``. Feeding them a ones-vector would add
a read and drag them onto the other kernel's contract, so they get their own
kernel instead.

The eager composition it replaces issues, per call, on ``[M, 2N]`` input:
split, 2 casts, 4 scales, 2 tanh, 1 sigmoid, 2 multiplies, 1 cast -- 13 fp32
launches, each of which round-trips a full ``[M, N]`` fp32 tensor through DRAM.
Only the first read and the last write are necessary.

Numerics: the kernel performs the *same* fp32 operations in the *same* order as
the eager chain, so the forward is bit-exact against it. ``sigmoid`` is spelled
as an IEEE divide of ``1 + expf(-x)`` rather than ``tl.sigmoid`` because
Paddle's ``CudaSigmoidFunctor<float>`` is ``one / (one + exp(-x))`` in fp32 and
``tl.sigmoid`` lowers to the ``ex2.approx`` path. The backward is *not* bit-exact
against the eager chain: at ``[257, 4096]`` bf16, 0.07% of gradient elements
differ, by at most 8 bfloat16 ULP and only where the gradient is near zero (the
largest absolute difference is 0.19 of one ULP at the tensor's maximum
magnitude). The difference is in the direction of being more accurate -- against
an fp64 reference from the same rounded inputs, max relative error 3.1e-2 for the
eager chain versus 5.3e-3 for this kernel, with equal RMS -- because the
intermediates stay in registers instead of round-tripping through bf16.

Backward saves nothing but ``x``: the activations are cheap to recompute and
saving the five fp32 intermediates the eager chain keeps alive would cost more
DRAM traffic in the forward than the recompute costs in the backward
(4 x [M,N] fp32 written + read, versus one [M,2N] narrow read). The recompute
is counted in this file's byte budget, not hidden.
"""

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
    """``1 / (1 + exp(-x))`` in fp32, matching CudaSigmoidFunctor<float>."""
    return libdevice.div_rn(1.0, 1.0 + libdevice.exp(-x))


@enable_compat_on_triton_kernel
@triton.jit
def _situ_glu_plain_fwd_kernel(
    x_ptr,
    out_ptr,
    n_cols: tl.constexpr,
    beta: tl.constexpr,
    inv_beta: tl.constexpr,
    linear_beta: tl.constexpr,
    inv_linear_beta: tl.constexpr,
    has_linear_tanh: tl.constexpr,
    block_n: tl.constexpr,
):
    # Long-context batches make row * stride exceed int32; keep offsets int64.
    row = tl.program_id(0).to(tl.int64)
    col_block = tl.program_id(1)
    cols = col_block * block_n + tl.arange(0, block_n)
    mask = cols < n_cols

    base = row * (2 * n_cols)
    gate = tl.load(x_ptr + base + cols, mask=mask).to(tl.float32)
    up = tl.load(x_ptr + base + n_cols + cols, mask=mask).to(tl.float32)

    # Same association as `beta * paddle.tanh(gate / beta) * F.sigmoid(gate)`.
    # `gate * inv_beta` (not `gate / beta`) because Paddle's `/ scalar` lowers
    # to the `scale` op, i.e. a multiply by the host-rounded reciprocal.
    gate_act = (beta * libdevice.tanh(gate * inv_beta)) * _sigmoid_precise(gate)
    if has_linear_tanh:
        up = linear_beta * libdevice.tanh(up * inv_linear_beta)
    tl.store(out_ptr + row * n_cols + cols, gate_act * up, mask=mask)


@enable_compat_on_triton_kernel
@triton.jit
def _situ_glu_plain_bwd_kernel(
    x_ptr,
    out_grad_ptr,
    x_grad_ptr,
    n_cols: tl.constexpr,
    beta: tl.constexpr,
    inv_beta: tl.constexpr,
    linear_beta: tl.constexpr,
    inv_linear_beta: tl.constexpr,
    has_linear_tanh: tl.constexpr,
    block_n: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col_block = tl.program_id(1)
    cols = col_block * block_n + tl.arange(0, block_n)
    mask = cols < n_cols

    base = row * (2 * n_cols)
    gate = tl.load(x_ptr + base + cols, mask=mask).to(tl.float32)
    up = tl.load(x_ptr + base + n_cols + cols, mask=mask).to(tl.float32)
    out_grad = tl.load(
        out_grad_ptr + row * n_cols + cols, mask=mask, other=0.0
    ).to(tl.float32)

    gate_tanh = libdevice.tanh(gate * inv_beta)
    gate_sigmoid = _sigmoid_precise(gate)
    gate_act = (beta * gate_tanh) * gate_sigmoid
    gate_grad = (
        1.0 - gate_tanh * gate_tanh
    ) * gate_sigmoid + beta * gate_tanh * gate_sigmoid * (1.0 - gate_sigmoid)
    if has_linear_tanh:
        up_tanh = libdevice.tanh(up * inv_linear_beta)
        up_act = linear_beta * up_tanh
        up_grad = 1.0 - up_tanh * up_tanh
    else:
        up_act = up
        up_grad = 1.0

    tl.store(x_grad_ptr + base + cols, out_grad * up_act * gate_grad, mask=mask)
    tl.store(
        x_grad_ptr + base + n_cols + cols,
        out_grad * gate_act * up_grad,
        mask=mask,
    )


def _check(beta: float, linear_beta: float | None) -> None:
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


def _geom(x: paddle.Tensor) -> tuple[int, int]:
    if not x.place.is_gpu_place():
        raise ValueError("SiTU-GLU Triton inputs must be GPU tensors")
    if x.ndim < 1 or x.shape[-1] % 2:
        raise ValueError(
            f"x last dim must be even ([.., 2N]), got {list(x.shape)}"
        )
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if x.dtype not in (paddle.float16, paddle.bfloat16, paddle.float32):
        raise TypeError(f"unsupported x dtype: {x.dtype}")
    n_cols = int(x.shape[-1]) // 2
    rows = 1
    for d in x.shape[:-1]:
        rows *= int(d)
    return rows, n_cols


def _launch(n_cols: int) -> tuple[int, int]:
    """(block_n, num_warps).

    Same rule ``triton_ops/situ_glu.py`` already launches its scale kernel
    with, so this file introduces no launch geometry of its own. Both kernels
    here are elementwise, so the choice cannot change any output bit -- see the
    module docstring.
    """
    block_n = min(1024, triton.next_power_of_2(n_cols))
    return block_n, 8 if block_n >= 512 else 4


def situ_glu_plain_forward_triton(
    x: paddle.Tensor,
    beta: float = 1.0,
    linear_beta: float | None = None,
) -> paddle.Tensor:
    """``SiTU-GLU(x)`` in one Triton kernel. ``x`` is ``[..., 2N]``."""
    _check(beta, linear_beta)
    rows, n_cols = _geom(x)
    out_shape = [*[int(d) for d in x.shape[:-1]], n_cols]
    out = paddle.empty(out_shape, dtype=x.dtype)
    if rows == 0 or n_cols == 0:
        return out
    block_n, num_warps = _launch(n_cols)
    grid = (rows, triton.cdiv(n_cols, block_n))
    _situ_glu_plain_fwd_kernel[grid](
        x,
        out,
        n_cols=n_cols,
        beta=float(beta),
        inv_beta=1.0 / float(beta),
        linear_beta=1.0 if linear_beta is None else float(linear_beta),
        inv_linear_beta=(
            1.0 if linear_beta is None else 1.0 / float(linear_beta)
        ),
        has_linear_tanh=linear_beta is not None,
        block_n=block_n,
        num_warps=num_warps,
    )
    return out


def situ_glu_plain_backward_triton(
    x: paddle.Tensor,
    out_grad: paddle.Tensor,
    beta: float = 1.0,
    linear_beta: float | None = None,
) -> paddle.Tensor:
    """Gradient of :func:`situ_glu_plain_forward_triton` wrt ``x``."""
    _check(beta, linear_beta)
    rows, n_cols = _geom(x)
    x_grad = paddle.empty_like(x)
    if rows == 0 or n_cols == 0:
        return x_grad
    if not out_grad.is_contiguous():
        out_grad = out_grad.contiguous()
    block_n, num_warps = _launch(n_cols)
    grid = (rows, triton.cdiv(n_cols, block_n))
    _situ_glu_plain_bwd_kernel[grid](
        x,
        out_grad,
        x_grad,
        n_cols=n_cols,
        beta=float(beta),
        inv_beta=1.0 / float(beta),
        linear_beta=1.0 if linear_beta is None else float(linear_beta),
        inv_linear_beta=(
            1.0 if linear_beta is None else 1.0 / float(linear_beta)
        ),
        has_linear_tanh=linear_beta is not None,
        block_n=block_n,
        num_warps=num_warps,
    )
    return x_grad


class FusedSituGluPlain(paddle.autograd.PyLayer):
    """autograd node for the plain fused SiTU-GLU.

    Only ``x`` is saved. ``beta`` / ``linear_beta`` are Python floats baked
    into the kernel as ``constexpr``, so they carry no gradient and are stashed
    on ``ctx`` rather than passed through ``save_for_backward``.
    """

    @staticmethod
    def forward(ctx, x, beta, linear_beta):
        out = situ_glu_plain_forward_triton(x, beta, linear_beta)
        ctx.save_for_backward(x)
        ctx.beta = beta
        ctx.linear_beta = linear_beta
        # `stop_gradient` is only trustworthy on a PyLayer's forward inputs
        # (same guard as SinkhornKnopp / FusedSinkhornKnopp in
        # transformer/hyper_connection.py): a frozen-backbone segment can still
        # be handed a backward, and Paddle then demands `None` at that slot.
        ctx.x_stop_gradient = x.stop_gradient
        return out

    @staticmethod
    def backward(ctx, out_grad):
        if ctx.x_stop_gradient:
            return None
        (x,) = ctx.saved_tensor()
        return situ_glu_plain_backward_triton(
            x, out_grad, ctx.beta, ctx.linear_beta
        )


def fused_situ_glu_plain(
    x: paddle.Tensor,
    beta: float = 1.0,
    linear_beta: float | None = None,
) -> paddle.Tensor:
    """Differentiable fused ``situ_glu``."""
    return FusedSituGluPlain.apply(x, beta, linear_beta)


__all__ = [
    "FusedSituGluPlain",
    "fused_situ_glu_plain",
    "situ_glu_plain_backward_triton",
    "situ_glu_plain_forward_triton",
]
