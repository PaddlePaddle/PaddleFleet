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

"""Fused HCA inverse RoPE + ungrouped VHA postmix.

Computes ``out = M @ inv_rope(O)`` -- the ``pos_dim > 0`` inverse-RoPE block of
``DSv4HybridAttention`` followed by the ``grouped=False`` branch of
``_apply_vha_postmix`` -- without ever materialising ``inv_rope(O)``.

Why a channel split is exact. RoPE only mixes the trailing ``pe_dim`` channels of
each head while the postmix GEMM contracts the *head* axis, so the channel axis
is a pure N dimension of the GEMM:

    out[t,h,c] = sum_h' M[h,h'] * roped(O)[t,h',c]

Splitting that N axis leaves the accumulation order untouched, so
``matmul(M, X[..., a:b]) == matmul(M, X)[..., a:b]`` bit for bit. That is a
cuBLAS property rather than a documented guarantee, so it is asserted in
``tests/single_card_tests/test_inv_rope_vha_postmix_fusion.py`` alongside every
other equality this module leans on.

All arithmetic is shared with ``mla_rope_inplace_fusion`` rather than duplicated:
the rotation reuses that module's ``_mul_round_bf16`` (whose inline-PTX
``cvt.rn.bf16.f32`` is what blocks FFMA folding and keeps the fused result equal
to eager Paddle), its ``_fused_cos_sin`` prelude, and its in-place forward /
backward kernels. Duplicating any of it would let the two drift apart silently.
"""

import paddle
import triton
import triton.language as tl

from .mla_rope_inplace_fusion import (
    _fused_cos_sin,
    _get_block_h,
    _mul_round_bf16,
    _rope_mla_inplace_bwd_kernel,
    _rope_mla_inplace_fwd_kernel,
)


def _check_shape(t, nope_dim, pe_dim):
    if t.dim() != 3:
        raise ValueError(f"t must be [B*S, H, D]; got {t.shape}")
    if t.stride(-1) != 1:
        raise ValueError("t must have a contiguous last dim")
    if t.shape[-1] != nope_dim + pe_dim:
        raise ValueError(
            f"t last dim {t.shape[-1]} != nope_dim + pe_dim "
            f"({nope_dim} + {pe_dim})"
        )
    if t.shape[1] % _get_block_h(t.shape[1]) != 0:
        raise ValueError(f"head_num {t.shape[1]} not divisible by its BLOCK_H")


def _check_rope_args(t, cos, sin, nope_dim, pe_dim):
    _check_shape(t, nope_dim, pe_dim)
    if pe_dim % 4 != 0:
        raise ValueError(f"pe_dim must be a multiple of 4; got {pe_dim}")
    if not cos.is_contiguous() or not sin.is_contiguous():
        raise ValueError("cos/sin must be contiguous")
    if cos.shape[-1] != pe_dim or sin.shape[-1] != pe_dim:
        raise ValueError(
            f"cos/sin last dim must be pe_dim={pe_dim}; got "
            f"{cos.shape[-1]}/{sin.shape[-1]}"
        )


def build_mla_rope_cos_sin(
    freqs: paddle.Tensor,
    b: int,
    s: int,
    pe_dim: int,
    mscale: float,
    inverse: bool,
    dtype: paddle.dtype,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """cos/sin prelude of ``fused_apply_mla_rope_inplace``, exposed separately.

    Driving the raw kernels only stays bitwise equal to the standalone op if the
    cos/sin tensors are built identically, so keep the checks here in sync with
    that wrapper's.
    """
    if freqs.dim() != 4 or freqs.shape[2] != 1:
        raise ValueError(f"freqs must be [B,S,1,D]; got {freqs.shape}")
    b_f, s_f, _, d_f = freqs.shape
    if s_f != s or d_f != pe_dim:
        raise ValueError(
            f"freqs {freqs.shape} mismatches [B,S]=[{b},{s}], pe_dim={pe_dim}"
        )
    if not (b_f == 1 or b_f == b):
        raise ValueError(f"freqs B {b_f} must be 1 or {b}")
    if b_f < b:
        freqs = freqs.broadcast_to([b, s, 1, pe_dim])
    return _fused_cos_sin(freqs, mscale, inverse, dtype)


@triton.jit
def _rope_pe_gather_kernel(
    T,
    T_OUT,
    COS,
    SIN,
    nope_dim,
    pe_dim: tl.constexpr,
    head_num: tl.constexpr,
    stride_in_seq,
    stride_in_nheads,
    stride_out_seq,
    stride_out_nheads,
    BLOCK_H: tl.constexpr,
):
    """Rotate ``T[..., nope_dim:]`` into a *compact* ``[B*S, H, pe_dim]`` buffer.

    Same loads and same ``_mul_round_bf16`` sequence as
    ``_rope_mla_inplace_fwd_kernel``; only the output addressing differs (the
    input keeps the wide row stride and the ``nope_dim`` channel offset, the
    output is densely packed). The forward needs the rotated pe channels as a
    standalone GEMM operand and never wants the nope channels copied.
    """
    pid_m = tl.program_id(axis=0).to(tl.int64)
    pid_head = tl.program_id(axis=1).to(tl.int64)

    half: tl.constexpr = pe_dim // 2

    cos_left = tl.load(COS + pid_m * pe_dim + tl.arange(0, half))
    sin_left = tl.load(SIN + pid_m * pe_dim + tl.arange(0, half))
    cos_right = tl.load(COS + pid_m * pe_dim + half + tl.arange(0, half))
    sin_right = tl.load(SIN + pid_m * pe_dim + half + tl.arange(0, half))
    cos_left = cos_left.expand_dims(0).broadcast_to(BLOCK_H, half)
    sin_left = sin_left.expand_dims(0).broadcast_to(BLOCK_H, half)
    cos_right = cos_right.expand_dims(0).broadcast_to(BLOCK_H, half)
    sin_right = sin_right.expand_dims(0).broadcast_to(BLOCK_H, half)

    T = T + pid_m * stride_in_seq + pid_head * BLOCK_H * stride_in_nheads
    T_OUT = (
        T_OUT + pid_m * stride_out_seq + pid_head * BLOCK_H * stride_out_nheads
    )
    head_mask = (pid_head * BLOCK_H + tl.arange(0, BLOCK_H))[:, None] < head_num
    in_off = (
        tl.arange(0, BLOCK_H)[:, None] * stride_in_nheads.to(tl.int64)
        + nope_dim
        + tl.arange(0, pe_dim)[None, :]
    )
    out_off = (
        tl.arange(0, BLOCK_H)[:, None] * stride_out_nheads.to(tl.int64)
        + tl.arange(0, pe_dim)[None, :]
    )

    x = tl.load(T + in_off, mask=head_mask)
    x = tl.reshape(x, (BLOCK_H, half, 2))
    x_1, x_2 = tl.split(x)

    y_left = _mul_round_bf16(x_1, cos_left).to(tl.float32) - _mul_round_bf16(
        x_2, sin_left
    ).to(tl.float32)
    y_right = _mul_round_bf16(x_2, cos_right).to(tl.float32) + _mul_round_bf16(
        x_1, sin_right
    ).to(tl.float32)

    y = tl.join(y_left, y_right)
    y = tl.reshape(y, (BLOCK_H, pe_dim))
    tl.store(T_OUT + out_off, y, mask=head_mask)


@triton.jit
def _pe_scatter_kernel(
    SRC,
    DST,
    pe_dim: tl.constexpr,
    head_num: tl.constexpr,
    src_seq_stride,
    src_head_stride,
    dst_seq_stride,
    dst_head_stride,
    dst_chan_off,
    BLOCK_H: tl.constexpr,
):
    """``DST[..., dst_chan_off:] = SRC`` for a compact ``[M, H, pe_dim]`` source.

    Paddle can express this as ``t[..., nope:] = compact``, but its strided
    elementwise copy does not vectorise the pattern: at nope=448/pe=64 it spends
    56 us on a 128 MiB slice against 49 us here, and the matching gather is 7x
    off (332 us). One contiguous ``pe_dim``-wide load/store per head fixes it.
    """
    pid_m = tl.program_id(axis=0).to(tl.int64)
    pid_head = tl.program_id(axis=1).to(tl.int64)

    head_mask = (pid_head * BLOCK_H + tl.arange(0, BLOCK_H))[:, None] < head_num
    rows = tl.arange(0, BLOCK_H)[:, None]
    chan = tl.arange(0, pe_dim)[None, :]

    src = SRC + pid_m * src_seq_stride + pid_head * BLOCK_H * src_head_stride
    dst = DST + pid_m * dst_seq_stride + pid_head * BLOCK_H * dst_head_stride
    src_off = rows * src_head_stride.to(tl.int64) + chan
    dst_off = rows * dst_head_stride.to(tl.int64) + dst_chan_off + chan

    tl.store(
        dst + dst_off, tl.load(src + src_off, mask=head_mask), mask=head_mask
    )


def rope_pe_to_compact(
    t: paddle.Tensor,
    cos: paddle.Tensor,
    sin: paddle.Tensor,
    nope_dim: int,
    pe_dim: int,
) -> paddle.Tensor:
    """Rotate the pe channels of ``t`` [B*S, H, D] into a fresh compact buffer.

    Returns [B*S, H, pe_dim], bitwise equal to
    ``fused_apply_mla_rope_inplace(t, ...)[..., nope_dim:]``.
    """
    _check_rope_args(t, cos, sin, nope_dim, pe_dim)
    m, h, _ = t.shape
    out = paddle.empty([m, h, pe_dim], dtype=t.dtype)
    block_h = _get_block_h(h)
    _rope_pe_gather_kernel[(m, triton.cdiv(h, block_h))](
        t,
        out,
        cos,
        sin,
        nope_dim,
        pe_dim,
        h,
        t.stride(0),
        t.stride(1),
        out.stride(0),
        out.stride(1),
        block_h,
    )
    return out


def rope_full_out_of_place(
    t: paddle.Tensor,
    cos: paddle.Tensor,
    sin: paddle.Tensor,
    nope_dim: int,
    pe_dim: int,
) -> paddle.Tensor:
    """RoPE the pe channels of ``t`` [B*S, H, D] into a fresh [B*S, H, D] buffer.

    Same kernel and arguments as ``RoPEMLAInplaceFusion.forward`` with
    ``clone_input=True``, hence bitwise equal to it.
    """
    _check_rope_args(t, cos, sin, nope_dim, pe_dim)
    m, h, d = t.shape
    out = paddle.empty([m, h, d], dtype=t.dtype)
    block_h = _get_block_h(h)
    _rope_mla_inplace_fwd_kernel[(m, triton.cdiv(h, block_h))](
        t,
        out,
        cos,
        sin,
        nope_dim,
        pe_dim,
        h,
        t.stride(0),
        t.stride(1),
        block_h,
        triton.next_power_of_2(max(nope_dim, 1)),
        True,
    )
    return out


def rope_pe_transpose_inplace_(
    t: paddle.Tensor,
    cos: paddle.Tensor,
    sin: paddle.Tensor,
    nope_dim: int,
    pe_dim: int,
) -> paddle.Tensor:
    """Apply the transpose rotation to ``t[..., nope_dim:]`` in place.

    Same kernel and arguments as ``RoPEMLAInplaceFusion.backward``, so a gradient
    pushed through here matches the standalone op bit for bit.
    """
    _check_rope_args(t, cos, sin, nope_dim, pe_dim)
    m, h, _ = t.shape
    block_h = _get_block_h(h)
    _rope_mla_inplace_bwd_kernel[(m, triton.cdiv(h, block_h))](
        t,
        cos,
        sin,
        nope_dim,
        pe_dim,
        h,
        t.stride(0),
        t.stride(1),
        block_h,
    )
    return t


def scatter_pe_slice_(
    t: paddle.Tensor, compact: paddle.Tensor, nope_dim: int, pe_dim: int
) -> paddle.Tensor:
    """``t[..., nope_dim:] = compact`` for a [B*S, H, D] tensor, vectorised."""
    _check_shape(t, nope_dim, pe_dim)
    if list(compact.shape) != [t.shape[0], t.shape[1], pe_dim]:
        raise ValueError(
            f"compact must be {[t.shape[0], t.shape[1], pe_dim]}; "
            f"got {compact.shape}"
        )
    m, h, _ = t.shape
    block_h = _get_block_h(h)
    _pe_scatter_kernel[(m, triton.cdiv(h, block_h))](
        compact,
        t,
        pe_dim,
        h,
        compact.stride(0),
        compact.stride(1),
        t.stride(0),
        t.stride(1),
        nope_dim,
        block_h,
    )
    return t


class InvRopeVhaPostmixFusion(paddle.autograd.PyLayer):
    """``out = M @ inv_rope(O)`` without ever materialising ``inv_rope(O)``.

    The unfused pair costs 4N of traffic and keeps two full-width copies of the
    attention output alive; this costs 2.75N and keeps one.

    Forward:
        pe_roped = rope(O[..., nope:])   -> compact [B*S, nh, pe]
        out      = matmul(M, O)          -> full width; the pe block it computes
                                            from unrotated data is discarded
        out[..., nope:] = matmul(M, pe_roped)

    Backward. The composite really is ``out = M @ O_roped``, so:

    - The activation gradient needs no rotated operand at all: ``dO_roped =
      M^T @ dOut`` full width, then the transpose rotation in place on the pe
      block. That is exactly the sequence the unfused path runs, hence bitwise
      identical.
    - The weight gradient ``dM[h,h'] = sum_{t,c} dOut[t,h,c] * O_roped[t,h',c]``
      contracts over the *full* channel axis, so it cannot be assembled from two
      partial sums without changing the reduction tree. The rotated tensor is
      therefore rebuilt here and both gradients are taken from
      ``paddle._C_ops.matmul_grad`` -- the same op the unfused postmix GEMM's
      backward calls, so they match by construction on every architecture.

      Hand-rolling that GEMM over head-major ``[H, B*S, D]`` operands (which is
      what ``matmul_grad`` internally builds with two ``TilingSwapDim1And2``
      passes at 2.5 TB/s) is ~870us/layer faster at the production shape, and
      was bitwise equal across every shape tested on sm10.3 -- but not on sm90,
      where CI caught ``head-major wgrad (128,4,64)`` differing in 8/16 elements.
      For a small ``[nh,nh] x K`` GEMM cuBLAS selects its algorithm per
      architecture, so no shape-based gate can make that route safe and it was
      dropped. Only the *forward* split still rests on a cuBLAS property
      (``matmul(M, X[..., a:b]) == matmul(M, X)[..., a:b]``), which is asserted
      over a wide shape sweep in the unit test.

    Nothing is ever mutated in place on the saved attention output, which the CSA
    backward also reads for its ``delta = rowsum(dO * O)``.
    """

    @staticmethod
    def forward(ctx, o_flat, m, cos, sin, nope_dim, pe_dim):
        pe_roped = rope_pe_to_compact(o_flat, cos, sin, nope_dim, pe_dim)
        out = paddle.matmul(m, o_flat)
        # The pe block `out` just computed came from unrotated channels; replace
        # it with the narrow GEMM on the rotated operand.
        scatter_pe_slice_(out, paddle.matmul(m, pe_roped), nope_dim, pe_dim)
        ctx.save_for_backward(o_flat, m, cos, sin)
        ctx.nope_dim = nope_dim
        ctx.pe_dim = pe_dim
        ctx.m_needs_grad = not m.stop_gradient
        return out

    @staticmethod
    def backward(ctx, d_out):
        o_flat, m, cos, sin = ctx.saved_tensors
        nope_dim, pe_dim = ctx.nope_dim, ctx.pe_dim
        if not d_out.is_contiguous():
            d_out = d_out.contiguous()

        if not ctx.m_needs_grad:
            # Frozen postmix (an indexer-only warmup stage, say): only the
            # activation gradient is wanted, and that never needs the rotated
            # operand, so skip rebuilding it.
            d_o = paddle.matmul(m, d_out, transpose_x=True)
            rope_pe_transpose_inplace_(d_o, cos, sin, nope_dim, pe_dim)
            return d_o, None, None, None

        # Rebuild the rotated tensor and hand both gradients to the very op the
        # unfused path used, so they come out of the same kernels by
        # construction rather than by a cuBLAS coincidence. Hand-rolling the
        # weight-gradient GEMM over head-major operands is ~870us/layer faster
        # at the production shape and was bitwise equal on sm10.3, but *not* on
        # sm90 (CI caught `head-major wgrad (128,4,64)` differing by 8/16
        # elements): for a small [nh,nh] x K GEMM cuBLAS picks its algorithm per
        # architecture, so no shape-based gate can make that route safe.
        o_roped = rope_full_out_of_place(o_flat, cos, sin, nope_dim, pe_dim)
        d_m, d_o = paddle._C_ops.matmul_grad(m, o_roped, d_out, False, False)
        del o_roped

        # d_o is the gradient wrt the rotated tensor; push it back through the
        # rotation exactly as the standalone RoPE op's backward does.
        rope_pe_transpose_inplace_(d_o, cos, sin, nope_dim, pe_dim)
        return d_o, d_m, None, None


def fused_inv_rope_vha_postmix(
    attn_out: paddle.Tensor,
    freqs: paddle.Tensor,
    postmix_u: paddle.Tensor,
    postmix_v: paddle.Tensor,
    nope_dim: int,
    pe_dim: int,
    mscale: float = 1.0,
) -> paddle.Tensor:
    """Inverse RoPE + ungrouped VHA postmix in one pass.

    Args:
        attn_out: [b, sq, nh, v_head_dim], contiguous bf16 attention output.
        freqs: [b_or_1, sq, 1, pe_dim] fp32 angle tensor.
        postmix_u, postmix_v: the [nh, rank] postmix factors.

    Returns:
        [b, sq, nh * v_head_dim], bitwise equal to what the unfused RoPE followed
        by ``_apply_vha_postmix``'s ungrouped branch returns.
    """
    b, sq, nh, d = attn_out.shape
    if not attn_out.is_contiguous():
        attn_out = attn_out.contiguous()
    cos, sin = build_mla_rope_cos_sin(
        freqs, b, sq, pe_dim, mscale, True, attn_out.dtype
    )
    # Same construction as _apply_vha_postmix's ungrouped branch; keep the two
    # in sync or the fused path stops being bitwise equal.
    m = paddle.matmul(postmix_v, postmix_u, transpose_y=True)
    m = m + paddle.eye(nh, dtype=m.dtype)
    out = InvRopeVhaPostmixFusion.apply(
        attn_out.reshape([b * sq, nh, d]), m, cos, sin, nope_dim, pe_dim
    )
    return out.reshape([b, sq, nh * d])
