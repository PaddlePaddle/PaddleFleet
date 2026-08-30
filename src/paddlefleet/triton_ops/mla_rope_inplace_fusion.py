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
High performance in-place RoPE for DSv4 hybrid attention.

This is a hand-specialised fusion of the call

    t_pe = t[..., nope_dim:]
    t_pe = _apply_rotary_pos_emb_bshd(
        t_pe, freqs,
        mscale=mscale,
        rotary_interleaved=False,
        multi_latent_attention=True,
        inverse=inverse,
        mla_output_remove_interleaving=True,
    )
    out = paddle.concat([t[..., :nope_dim], t_pe], axis=-1)

Features:
- Binary equal to Paddle eager mode by using triton asm.
- Coalesced memory access and tuned block size for best performance.
- Generically supports q/kv and o (inverse rope) in one kernel.
"""

import paddle

from .utils import is_torch_compat_available

if is_torch_compat_available():
    paddle.enable_compat(scope={"triton"})

import triton
import triton.language as tl


def _get_block_h(nheads: int) -> int:
    """Largest power-of-2 dividing nheads, capped at 32."""
    block_h = min(32, triton.next_power_of_2(nheads))
    while block_h > 1 and nheads % block_h != 0:
        block_h //= 2
    return block_h


@triton.jit
def _mul_round_bf16(a, b):
    """Compute (a * b) in fp32, then round to bf16 via inline PTX.

    Forces an explicit `cvt.rn.bf16.f32` after the multiply so the Triton
    compiler cannot fuse it into an FFMA with the surrounding add/sub. This
    is what gives us bit-exact parity with eager Paddle, which stores each
    intermediate to bf16 memory between elementwise ops.
    """
    return tl.inline_asm_elementwise(
        "cvt.rn.bf16.f32 $0, $1;",
        "=h,r",
        [(a.to(tl.float32) * b.to(tl.float32))],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _cos_sin_kernel(
    FREQS,
    COS_OUT,
    SIN_OUT,
    n_elements,
    mscale,
    INVERSE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fused port of the eager cos/sin prelude."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements

    f = tl.load(FREQS + offs, mask=mask)  # fp32

    c = (tl.cos(f) * mscale).to(tl.bfloat16)
    s = (tl.sin(f) * mscale).to(tl.bfloat16)
    if INVERSE:
        s = -s

    tl.store(COS_OUT + offs, c, mask=mask)
    tl.store(SIN_OUT + offs, s, mask=mask)


@triton.jit
def _rope_mla_inplace_fwd_kernel(
    T,
    T_OUT,
    COS,
    SIN,
    nope_dim,
    pe_dim: tl.constexpr,
    head_num: tl.constexpr,
    stride_x_seq,
    stride_x_nheads,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    OUT_OF_PLACE: tl.constexpr,
):
    """Forward: rotate t[..., nope_dim:] (interleaved in/out).

    Reads from ``T`` and writes to ``T_OUT``. With ``OUT_OF_PLACE=False`` the
    caller passes the same pointer for both and the kernel behaves exactly as
    the original in-place version: the nope channels are never touched, so
    there is no extra traffic and no extra allocation. With
    ``OUT_OF_PLACE=True`` the nope channels are additionally copied across,
    which lets the caller keep the input buffer intact without paying for a
    separate ``clone()`` pass over the whole tensor.
    """
    pid_m = tl.program_id(axis=0).to(tl.int64)
    pid_head = tl.program_id(axis=1).to(tl.int64)

    # COS/SIN are pre-broadcast to [B*S, pe_dim] (contiguous), so each token
    # index maps directly to a row regardless of whether freqs originally had
    # B=1 or B>1, and regardless of slicing along the seq axis.
    half: tl.constexpr = pe_dim // 2

    cos_left = tl.load(COS + pid_m * pe_dim + tl.arange(0, half))
    sin_left = tl.load(SIN + pid_m * pe_dim + tl.arange(0, half))
    cos_right = tl.load(COS + pid_m * pe_dim + half + tl.arange(0, half))
    sin_right = tl.load(SIN + pid_m * pe_dim + half + tl.arange(0, half))
    cos_left = cos_left.expand_dims(0).broadcast_to(BLOCK_H, half)
    sin_left = sin_left.expand_dims(0).broadcast_to(BLOCK_H, half)
    cos_right = cos_right.expand_dims(0).broadcast_to(BLOCK_H, half)
    sin_right = sin_right.expand_dims(0).broadcast_to(BLOCK_H, half)

    # Pointer to the start of this token's (head_block_first) row, then advance
    # past the nope channels to land on the rope slice.
    row_off = pid_m * stride_x_seq + pid_head * BLOCK_H * stride_x_nheads
    T = T + row_off
    T_OUT = T_OUT + row_off
    head_off = tl.arange(0, BLOCK_H)[:, None] * stride_x_nheads.to(tl.int64)
    head_mask = (pid_head * BLOCK_H + tl.arange(0, BLOCK_H))[:, None] < head_num

    # Out-of-place only: carry the untouched nope channels over to the output
    # buffer. BLOCK_D is next_power_of_2(nope_dim) on the host, so this is a
    # single masked pass; the in-place path compiles this branch away entirely.
    if OUT_OF_PLACE:
        offs_d = tl.arange(0, BLOCK_D)
        nope_off = head_off + offs_d[None, :]
        nope_mask = head_mask & (offs_d[None, :] < nope_dim)
        tl.store(
            T_OUT + nope_off,
            tl.load(T + nope_off, mask=nope_mask),
            mask=nope_mask,
        )

    # Offsets into the rope slice: [BLOCK_H, pe_dim] with last dim STRIDE=1.
    # We deliberately load the whole pe_dim contiguously instead of poking
    # at 2k / 2k+1 with stride-2 offsets — Triton's lowering for stride-2
    # int64 offsets has historically been flaky (extra sector requests, no
    # vectorization), and explicit contiguous loads compile down to
    # `ld.global.v4.b32` which is the theoretical optimum for bf16.
    flat_off = head_off + nope_dim + tl.arange(0, pe_dim)[None, :]

    # One contiguous load per program, then de-interleave in registers.
    x = tl.load(T + flat_off, mask=head_mask)  # [BLOCK_H, pe_dim] bf16
    x = tl.reshape(x, (BLOCK_H, half, 2))
    x_1, x_2 = tl.split(x)  # both [BLOCK_H, half]

    # ---- bit-exact match with eager Paddle path -------------------------
    # Eager runs this as three separate elementwise ops: each `a*b` and
    # each `+`/`-` lands in bf16 memory before the next op reads it. We
    # mirror that by keeping every intermediate as bf16 and forcing a
    # store/reload through a hand-written __nv_bf16 round, which the
    # Triton/PTX compiler cannot fuse into FFMA.
    y_left = _mul_round_bf16(x_1, cos_left).to(tl.float32) - _mul_round_bf16(
        x_2, sin_left
    ).to(tl.float32)
    y_right = _mul_round_bf16(x_2, cos_right).to(tl.float32) + _mul_round_bf16(
        x_1, sin_right
    ).to(tl.float32)

    # Re-interleave (mla_output_remove_interleaving=True writes back to the
    # same 2k / 2k+1 positions) and store in one contiguous write.
    y = tl.join(y_left, y_right)  # [BLOCK_H, half, 2]
    y = tl.reshape(y, (BLOCK_H, pe_dim))
    tl.store(T_OUT + flat_off, y, mask=head_mask)


@triton.jit
def _rope_mla_inplace_bwd_kernel(
    DO,
    COS,
    SIN,
    nope_dim,
    pe_dim: tl.constexpr,
    head_num: tl.constexpr,
    stride_x_seq,
    stride_x_nheads,
    BLOCK_H: tl.constexpr,
):
    """Backward: transform grad in place by the transpose of the forward 2x2."""
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

    DO = DO + pid_m * stride_x_seq + pid_head * BLOCK_H * stride_x_nheads
    flat_off = (
        tl.arange(0, BLOCK_H)[:, None] * stride_x_nheads.to(tl.int64)
        + nope_dim
        + tl.arange(0, pe_dim)[None, :]
    )
    head_mask = (pid_head * BLOCK_H + tl.arange(0, BLOCK_H))[:, None] < head_num

    g = tl.load(DO + flat_off, mask=head_mask)  # [BLOCK_H, pe_dim] bf16
    g = tl.reshape(g, (BLOCK_H, half, 2))
    g1, g2 = tl.split(g)  # both [BLOCK_H, half]

    # Same FFMA-blocking trick as forward (see `_mul_round_bf16`): each
    # product is independently rounded to bf16 before the add, matching
    # eager Paddle's elementwise op-by-op execution.
    dx_1 = _mul_round_bf16(g1, cos_left).to(tl.float32) + _mul_round_bf16(
        g2, sin_right
    ).to(tl.float32)
    dx_2 = _mul_round_bf16(g2, cos_right).to(tl.float32) - _mul_round_bf16(
        g1, sin_left
    ).to(tl.float32)

    dx = tl.join(dx_1, dx_2)  # [BLOCK_H, half, 2]
    dx = tl.reshape(dx, (BLOCK_H, pe_dim))
    tl.store(DO + flat_off, dx, mask=head_mask)


class RoPEMLAInplaceFusion(paddle.autograd.PyLayer):
    """PyLayer wrapping the in-place fwd/bwd kernels."""

    @staticmethod
    def forward(ctx, t, cos, sin, nope_dim, pe_dim, clone_input):
        assert t.stride(-1) == 1
        assert cos.is_contiguous()
        assert sin.is_contiguous()
        B, S, H, D = t.shape
        assert D == nope_dim + pe_dim
        assert pe_dim % 4 == 0
        assert cos.shape[-1] == pe_dim
        assert sin.shape[-1] == pe_dim

        # Flatten BS for the kernel (view, no copy on contiguous tensors).
        t_flat = t.reshape([B * S, H, D])
        BLOCK_H = _get_block_h(H)
        assert H % BLOCK_H == 0, (
            f"head_num must be divisible by BLOCK_H ({BLOCK_H}), got {H}"
        )

        # When the upstream still needs `t`, write to a fresh buffer instead of
        # cloning first: `clone()` would read+write the whole tensor and the
        # kernel would then read+write the rope slice again. Letting the kernel
        # read `t` and write `out` (carrying the nope channels across on the
        # way) is a single pass, and leaves `t` untouched just the same.
        # clone_input=False keeps the true in-place behaviour: same pointer in
        # and out, nope branch compiled away, no allocation.
        out = paddle.empty(t.shape, dtype=t.dtype) if clone_input else t
        out_flat = out.reshape([B * S, H, D]) if clone_input else t_flat
        BLOCK_D = triton.next_power_of_2(max(nope_dim, 1))

        grid = (B * S, triton.cdiv(H, BLOCK_H))
        _rope_mla_inplace_fwd_kernel[grid](
            t_flat,
            out_flat,
            cos,
            sin,
            nope_dim,
            pe_dim,
            H,
            t_flat.stride(0),
            t_flat.stride(1),
            BLOCK_H,
            BLOCK_D,
            clone_input,
        )

        ctx.save_for_backward(cos, sin)
        ctx.nope_dim = nope_dim
        ctx.pe_dim = pe_dim
        ctx.block_h = BLOCK_H
        ctx.shape = (B, S, H, D)
        # clone_input=False returns the reshape-back view of the input, whose
        # storage is identical to `t`; clone_input=True returns the new buffer.
        return out

    @staticmethod
    def backward(ctx, grad):
        cos, sin = ctx.saved_tensors
        B, S, H, D = ctx.shape
        # Run in place on grad; nope channels are passed through unchanged.
        grad_flat = grad.contiguous().reshape([B * S, H, D])
        BLOCK_H = ctx.block_h
        grid = (B * S, triton.cdiv(H, BLOCK_H))
        _rope_mla_inplace_bwd_kernel[grid](
            grad_flat,
            cos,
            sin,
            ctx.nope_dim,
            ctx.pe_dim,
            H,
            grad_flat.stride(0),
            grad_flat.stride(1),
            BLOCK_H,
        )
        return grad, None, None  # (t, cos, sin)


def _fused_cos_sin(
    freqs: paddle.Tensor,
    mscale: float,
    inverse: bool,
    dtype: paddle.dtype,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Triton replacement for the three-line eager cos/sin prelude:

    cos = (paddle.cos(freqs) * mscale).to(dtype)
    sin = (paddle.sin(freqs) * mscale).to(dtype)
    if inverse:
        sin = -sin
    """
    freqs = freqs.contiguous()
    n = freqs.size
    cos = paddle.empty(freqs.shape, dtype=dtype)
    sin = paddle.empty(freqs.shape, dtype=dtype)

    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    _cos_sin_kernel[grid](
        freqs,
        cos,
        sin,
        n,
        mscale,
        inverse,
        BLOCK,
    )
    return cos, sin


def fused_apply_mla_rope_inplace(
    t: paddle.Tensor,
    freqs: paddle.Tensor,
    nope_dim: int,
    mscale: float = 1.0,
    inverse: bool = False,
    clone_input: bool = False,
) -> paddle.Tensor:
    """In-place RoPE on t's trailing rope channels.

    Specialised for DSv4 hybrid attention's MLA pe path:
      bshd, multi_latent_attention=True, mla_output_remove_interleaving=True,
      rotary_interleaved=False, mscale=1.0, no SP/CP, no THD,
      high_precision_rope=False (cos/sin computed in fp32, cast to t.dtype
      to match the unfused path's bf16 arithmetic).

    `t` is generic — both q and k/v can be passed through, since the only
    assumption is that the last `pe_dim` channels carry the interleaved
    rope pairs.

    Args:
        t: [B, S, H, nope_dim + pe_dim], contiguous, bf16 (or fp16/fp32).
            Mutated in place unless clone_input=True.
        freqs: [B, S, 1, pe_dim], fp32 angle tensor. May be non-contiguous.
        nope_dim: number of leading nope channels left untouched.
        mscale: scaling factor for rotary embedding.
        inverse: if True, apply the inverse rotation (used by the
            inv_rope post-attention canonicalisation step).
        clone_input: if True, leave `t` untouched and return a new tensor
            instead (needed when the upstream still reads `t`, e.g. an
            attention output that its own backward has saved).

    Returns:
        With clone_input=False, `t` itself (same storage as the input).
        With clone_input=True, a freshly allocated tensor; `t` is not
        modified. Either way channels [..., :nope_dim] carry the input's
        nope values unchanged and [..., nope_dim:] are rotated.
    """
    # Check t
    assert t.is_contiguous(), (
        "fused_apply_mla_rope_inplace requires t to be contiguous, "
        f"got shape={t.shape} strides={t.strides}"
    )
    B, S, H, D = t.shape
    pe_dim = freqs.shape[-1]
    assert D >= pe_dim, f"t last dim {D} must be at least rope pe_dim {pe_dim}"
    nope_dim_check = D - pe_dim
    assert nope_dim == nope_dim_check, (
        f"nope_dim {nope_dim} mismatches D-pe_dim {nope_dim_check}"
    )
    assert t.dtype == paddle.bfloat16, (
        f"fused_apply_mla_rope_inplace is designed for bf16, got {t.dtype}"
    )

    # Check freqs
    assert freqs.dim() == 4 and freqs.shape[2] == 1, (
        f"freqs must be [B,S,1,D]; got {freqs.shape}"
    )
    B_f, S_f, _, D_f = freqs.shape
    assert S_f == S and D_f == pe_dim, (
        f"freqs [B,S,1,D]={freqs.shape} mismatches t [B,S]=[{B},{S}], "
        f"pe_dim={pe_dim}"
    )
    assert B_f == 1 or B_f == B, f"freqs B {B_f} must be 1 or {B}"
    if B_f < B:
        freqs = freqs.broadcast_to([B, S, 1, pe_dim])

    # Compute cos/sin on freqs
    cos, sin = _fused_cos_sin(freqs, mscale, inverse, t.dtype)

    return RoPEMLAInplaceFusion.apply(
        t, cos, sin, nope_dim, pe_dim, clone_input
    )


# ---------------------------------------------------------------------------
# rope + concat, absorbed-MQA key  (multi_latent_attention.py)
# ---------------------------------------------------------------------------
#
# The absorbed-MQA key is built as
#
#     k_pos_emb = <rotate_half rope>(k_pos_emb)          # [b, s, 1, pe]
#     key = paddle.cat([kv_compressed.unsqueeze(-2), k_pos_emb], axis=-1)
#
# i.e. an ``[b, s, 1, latent + pe]`` buffer whose leading ``latent`` channels
# are the (absorbed) value and whose trailing ``pe`` channels are the rotated
# positional part.  Fusing the two writes the buffer once instead of rotating
# into a temporary and then copying both pieces.
#
# Like ``fused_apply_rope_half`` below this does NOT work in place, which is
# the point: ``k_pos_emb`` is an *argument* of
# ``qkv_up_proj_and_rope_apply``, created outside it, so a closure replayed by
# ``recompute_qkv_up_porj_and_rope`` would rotate an already-rotated tensor a
# second time.  Writing to a fresh buffer is replay-safe.


@triton.jit
def _rope_cat_key_fwd_kernel(
    KV,
    KPE,
    OUT,
    PE_OUT,
    COS,
    SIN,
    latent_dim: tl.constexpr,
    pe_dim: tl.constexpr,
    stride_kv,
    stride_kpe,
    stride_out,
    stride_pe_out,
    ADJACENT_IN: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    """Forward: OUT[:latent] = KV, OUT[latent:] = PE_OUT = rotate_half(KPE)."""
    pid = tl.program_id(axis=0).to(tl.int64)
    half: tl.constexpr = pe_dim // 2

    # ---- latent block: straight copy ----
    offs = tl.arange(0, BLOCK_L)
    lat_mask = offs < latent_dim
    v = tl.load(KV + pid * stride_kv + offs, mask=lat_mask)
    tl.store(OUT + pid * stride_out + offs, v, mask=lat_mask)

    # ---- pe block: same 2x2 rotation as _rope_half_out_fwd_kernel ----
    cos_left = tl.load(COS + pid * pe_dim + tl.arange(0, half))
    sin_left = tl.load(SIN + pid * pe_dim + tl.arange(0, half))
    cos_right = tl.load(COS + pid * pe_dim + half + tl.arange(0, half))
    sin_right = tl.load(SIN + pid * pe_dim + half + tl.arange(0, half))

    h = tl.arange(0, half)
    if ADJACENT_IN:
        # Adjacent source pairing; see ``_rope_half_out_fwd_kernel``. The stores
        # below are untouched, so the output stays half-split.
        x_1 = tl.load(KPE + pid * stride_kpe + 2 * h)
        x_2 = tl.load(KPE + pid * stride_kpe + 2 * h + 1)
    else:
        x_1 = tl.load(KPE + pid * stride_kpe + h)
        x_2 = tl.load(KPE + pid * stride_kpe + half + h)

    y_left = _mul_round_bf16(x_1, cos_left).to(tl.float32) - _mul_round_bf16(
        x_2, sin_left
    ).to(tl.float32)
    y_right = _mul_round_bf16(x_2, cos_right).to(tl.float32) + _mul_round_bf16(
        x_1, sin_right
    ).to(tl.float32)

    tl.store(OUT + pid * stride_out + latent_dim + h, y_left)
    tl.store(OUT + pid * stride_out + latent_dim + half + h, y_right)
    # The rotated pe is already in registers, so handing it back as its own
    # contiguous tensor costs one extra store and no extra read. The caller
    # needs it (as ``k_pe``) and slicing it back out of OUT afterwards would be
    # a non-contiguous read plus a copy, and would put a second consumer on
    # OUT's gradient.
    tl.store(PE_OUT + pid * stride_pe_out + h, y_left)
    tl.store(PE_OUT + pid * stride_pe_out + half + h, y_right)


@triton.jit
def _rope_cat_key_bwd_kernel(
    DOUT,
    DPE_OUT,
    DKV,
    DKPE,
    COS,
    SIN,
    latent_dim: tl.constexpr,
    pe_dim: tl.constexpr,
    stride_kv,
    stride_kpe,
    stride_out,
    stride_pe_out,
    HAS_DPE_OUT: tl.constexpr,
    ADJACENT_IN: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    """Backward: split DOUT, passing the latent part through unchanged."""
    pid = tl.program_id(axis=0).to(tl.int64)
    half: tl.constexpr = pe_dim // 2

    offs = tl.arange(0, BLOCK_L)
    lat_mask = offs < latent_dim
    g = tl.load(DOUT + pid * stride_out + offs, mask=lat_mask)
    tl.store(DKV + pid * stride_kv + offs, g, mask=lat_mask)

    cos_left = tl.load(COS + pid * pe_dim + tl.arange(0, half))
    sin_left = tl.load(SIN + pid * pe_dim + tl.arange(0, half))
    cos_right = tl.load(COS + pid * pe_dim + half + tl.arange(0, half))
    sin_right = tl.load(SIN + pid * pe_dim + half + tl.arange(0, half))

    h = tl.arange(0, half)
    g1 = tl.load(DOUT + pid * stride_out + latent_dim + h)
    g2 = tl.load(DOUT + pid * stride_out + latent_dim + half + h)
    if HAS_DPE_OUT:
        # The pe block feeds two outputs, so its incoming gradients add. Round
        # the sum back to bf16 first: that is what paddle's own accumulation
        # would do for a tensor with two consumers, and the rotation below is
        # written to match eager op-by-op rounding.
        g1 = (
            g1.to(tl.float32)
            + tl.load(DPE_OUT + pid * stride_pe_out + h).to(tl.float32)
        ).to(g1.dtype)
        g2 = (
            g2.to(tl.float32)
            + tl.load(DPE_OUT + pid * stride_pe_out + half + h).to(tl.float32)
        ).to(g2.dtype)

    dx_1 = _mul_round_bf16(g1, cos_left).to(tl.float32) + _mul_round_bf16(
        g2, sin_right
    ).to(tl.float32)
    dx_2 = _mul_round_bf16(g2, cos_right).to(tl.float32) - _mul_round_bf16(
        g1, sin_left
    ).to(tl.float32)

    if ADJACENT_IN:
        # Mirrors the forward gather; the DOUT/DPE_OUT reads above follow the
        # output layout and are unaffected.
        tl.store(DKPE + pid * stride_kpe + 2 * h, dx_1)
        tl.store(DKPE + pid * stride_kpe + 2 * h + 1, dx_2)
    else:
        tl.store(DKPE + pid * stride_kpe + h, dx_1)
        tl.store(DKPE + pid * stride_kpe + half + h, dx_2)


class RoPECatKeyFusion(paddle.autograd.PyLayer):
    """PyLayer wrapping the rope+concat absorbed-key kernels."""

    @staticmethod
    def forward(ctx, kv, kpe, cos, sin, latent_dim, pe_dim, adjacent_in):
        # Input validation via explicit exceptions (not ``assert``): this is the
        # PyLayer behind the public ``fused_rope_cat_key`` entry, so under
        # ``python -O`` a stripped assert would let a wrong shape / non-contiguous
        # buffer into the Triton kernel.
        if not (kv.stride(-1) == 1 and kpe.stride(-1) == 1):
            raise ValueError(
                "RoPECatKeyFusion needs kv/kpe last dim contiguous, got "
                f"kv.strides={kv.strides}, kpe.strides={kpe.strides}"
            )
        if not (cos.is_contiguous() and sin.is_contiguous()):
            raise ValueError("RoPECatKeyFusion needs cos/sin contiguous")
        n = kv.shape[0] * kv.shape[1]
        if kpe.shape[0] * kpe.shape[1] != n:
            raise ValueError(
                f"kpe leading dims {kpe.shape[:2]} do not flatten to kv's n={n}"
            )
        if kv.shape[-1] != latent_dim:
            raise ValueError(
                f"kv last dim {kv.shape[-1]} != latent_dim {latent_dim}"
            )
        if kpe.shape[-1] != pe_dim:
            raise ValueError(f"kpe last dim {kpe.shape[-1]} != pe_dim {pe_dim}")
        if pe_dim % 4 != 0:
            raise ValueError(f"pe_dim {pe_dim} must be a multiple of 4")
        if not (cos.shape[-1] == pe_dim and sin.shape[-1] == pe_dim):
            raise ValueError(
                f"cos/sin last dim {cos.shape[-1]}/{sin.shape[-1]} != "
                f"pe_dim {pe_dim}"
            )

        kv_flat = kv.reshape([n, latent_dim])
        kpe_flat = kpe.reshape([n, pe_dim])
        out = paddle.empty([n, latent_dim + pe_dim], dtype=kv.dtype)
        pe_out = paddle.empty([n, pe_dim], dtype=kv.dtype)

        BLOCK_L = triton.next_power_of_2(latent_dim)
        _rope_cat_key_fwd_kernel[(n,)](
            kv_flat,
            kpe_flat,
            out,
            pe_out,
            cos,
            sin,
            latent_dim,
            pe_dim,
            kv_flat.stride(0),
            kpe_flat.stride(0),
            out.stride(0),
            pe_out.stride(0),
            adjacent_in,
            BLOCK_L,
        )

        ctx.save_for_backward(cos, sin)
        ctx.dims = (latent_dim, pe_dim, BLOCK_L)
        ctx.adjacent_in = adjacent_in
        ctx.kv_shape = kv.shape
        ctx.kpe_shape = kpe.shape
        b, s = kv.shape[0], kv.shape[1]
        return (
            out.reshape([b, s, 1, latent_dim + pe_dim]),
            pe_out.reshape([b, s, 1, pe_dim]),
        )

    @staticmethod
    def backward(ctx, dout, dpe_out=None):
        cos, sin = ctx.saved_tensors
        latent_dim, pe_dim, BLOCK_L = ctx.dims
        n = ctx.kv_shape[0] * ctx.kv_shape[1]
        dout_flat = dout.contiguous().reshape([n, latent_dim + pe_dim])
        # ``k_pe`` may have no consumer at all (``MQALatentAttention`` takes it
        # and never reads it), in which case paddle hands back nothing for it and
        # the extra load is compiled away rather than reading a zero buffer.
        has_dpe = dpe_out is not None
        if has_dpe:
            dpe_flat = dpe_out.contiguous().reshape([n, pe_dim])
        else:
            dpe_flat = dout_flat  # unused; kernel needs a valid pointer
        dkv = paddle.empty([n, latent_dim], dtype=dout.dtype)
        dkpe = paddle.empty([n, pe_dim], dtype=dout.dtype)

        _rope_cat_key_bwd_kernel[(n,)](
            dout_flat,
            dpe_flat,
            dkv,
            dkpe,
            cos,
            sin,
            latent_dim,
            pe_dim,
            dkv.stride(0),
            dkpe.stride(0),
            dout_flat.stride(0),
            dpe_flat.stride(0),
            has_dpe,
            ctx.adjacent_in,
            BLOCK_L,
        )
        return (
            dkv.reshape(ctx.kv_shape),
            dkpe.reshape(ctx.kpe_shape),
            None,
            None,
        )


def fused_rope_cat_key(
    kv_compressed: paddle.Tensor,
    k_pos_emb: paddle.Tensor,
    freqs: paddle.Tensor,
    latent_dim: int,
    pe_dim: int,
    mscale: float = 1.0,
    adjacent_in: bool = False,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Build the absorbed-MQA key: rotate ``k_pos_emb`` and concat, in one pass.

    Replaces, for the ``mqa_latent`` branch of
    ``MLASelfAttention.get_query_key_value_tensors``::

        k_pos_emb = <rotate_half rope>(k_pos_emb)
        key = paddle.cat([kv_compressed.unsqueeze(-2), k_pos_emb], axis=-1)

    Both of that snippet's results are returned, because the caller needs both:
    ``key`` for the attention and the rotated pe on its own as ``k_pe``. The
    second one is a separate contiguous tensor rather than a view into ``key``'s
    tail -- the rotated values are already in registers, so it costs one store
    and no extra read, while slicing it back out afterwards would cost a
    non-contiguous read plus a copy, put a second consumer on ``key``'s
    gradient, and produce a non-contiguous tensor that
    ``RecomputeWithoutOutput(share_grad_holder=True)`` refuses.

    Same ``rotate_half`` convention (and the same bf16 rounding) as
    ``fused_apply_rope_half``, so it is bit-exact with that pair. Writes
    fresh buffers rather than rotating in place, which keeps it safe when
    ``recompute_qkv_up_porj_and_rope`` replays the closure that owns
    ``k_pos_emb``.

    Args:
        kv_compressed: [b, s, latent_dim], last-dim-contiguous, bf16. Already
            normalised; copied through unchanged.
        k_pos_emb: [b, s, pe_dim] or [b, s, 1, pe_dim], last-dim-contiguous,
            bf16. NOT yet rotated.
        freqs: [b or 1, s, 1, pe_dim] angle tensor, fp32 or bf16.
        latent_dim: width of the value/nope block that leads the key.
        pe_dim: width of the rope block that trails it.
        mscale: scaling factor; must be 1.0 when ``freqs`` is not fp32.
        adjacent_in: pair ``k_pos_emb``'s source channels ``(2k, 2k+1)`` instead
            of ``(k, k+half)``. Must match the q side's ``adjacent_in``, since
            the two meet in ``q @ k^T``. Default False leaves existing callers
            unchanged.

    Returns:
        (key, k_pe) where key is [b, s, 1, latent_dim + pe_dim] with the
        absorbed value in its leading ``latent_dim`` channels, and k_pe is
        [b, s, 1, pe_dim] holding the same rotated pe as key's tail.
    """
    # Production input validation: these guard user/config-facing dtype, shape
    # and mscale before the values reach the Triton kernel, so they must be
    # explicit exceptions rather than ``assert`` -- ``python -O`` strips
    # asserts, which would let fp16 into the bf16-rounding path or a wrong
    # shape into Triton address arithmetic and silently corrupt results.
    if kv_compressed.dtype != paddle.bfloat16:
        raise ValueError(
            f"fused_rope_cat_key is designed for bf16, got {kv_compressed.dtype}"
        )
    if k_pos_emb.dtype != kv_compressed.dtype:
        raise ValueError(
            "fused_rope_cat_key needs k_pos_emb and kv_compressed to share a "
            f"dtype; got {k_pos_emb.dtype} vs {kv_compressed.dtype}"
        )
    b, s = kv_compressed.shape[0], kv_compressed.shape[1]

    if not (freqs.dim() == 4 and freqs.shape[2] == 1):
        raise ValueError(f"freqs must be [B,S,1,D]; got {freqs.shape}")
    B_f, S_f, _, D_f = freqs.shape
    if not (S_f == s and D_f == pe_dim):
        raise ValueError(
            f"freqs {freqs.shape} mismatches [b,s]=[{b},{s}], pe_dim={pe_dim}"
        )
    if not (B_f == 1 or B_f == b):
        raise ValueError(f"freqs B {B_f} must be 1 or {b}")
    if B_f < b:
        freqs = freqs.broadcast_to([b, s, 1, pe_dim])

    # Same upcast rule as ``fused_apply_rope_half``; see the note there.
    if freqs.dtype != paddle.float32:
        if mscale != 1.0:
            raise ValueError(
                "fused_rope_cat_key needs fp32 freqs when mscale != 1 "
                f"(got freqs.dtype={freqs.dtype}, mscale={mscale})"
            )
        freqs = freqs.astype(paddle.float32)

    cos, sin = _fused_cos_sin(freqs, mscale, False, kv_compressed.dtype)
    return RoPECatKeyFusion.apply(
        kv_compressed, k_pos_emb, cos, sin, latent_dim, pe_dim, adjacent_in
    )


# ---------------------------------------------------------------------------
# rotate_half layout  (DSA indexer, MLA q / k_pos_emb)
# ---------------------------------------------------------------------------
#
# A second convention, used by the DSA indexer and by MLA's eager RoPE branch
# (both run with ``multi_latent_attention=False``):
#
#   x_pe   = x[..., :rope_head_dim]          # DSA: pe FIRST, nope trails
#   x_nope = x[..., rope_head_dim:]          # MLA q: the other way round
#   x_pe   = _apply_rotary_pos_emb_bshd(x_pe, freqs,
#                rotary_interleaved=False, multi_latent_attention=False)
#   return paddle.concat([x_pe, x_nope], axis=-1)
#
# ``multi_latent_attention=False`` means no 0::2/1::2 de-interleaving: the
# rotated pair is (first half, second half) of the pe slice, and the output
# stays in those same halves.  So the arithmetic is identical to the MLA
# kernels above -- only the load/store index mapping differs, and it is
# strictly simpler (two contiguous half-width accesses, no register shuffle).
#
# The whole three-step sequence collapses into one kernel: rotate the pe block
# and copy the other channels across, writing the destination once.  Two
# properties worth stating explicitly, because both are load-bearing:
#
# 1. It does not work in place.  ``qkv_up_proj_and_rope_apply``
#    (multi_latent_attention.py:1692) can be replayed by
#    ``recompute_qkv_up_porj_and_rope``, and ``k_pos_emb`` is one of its
#    *arguments* -- created outside, so rotating it in place would be applied
#    twice on replay.  ``q`` happens to be created inside (:1710) and would
#    survive that, but that is a property of where a line sits rather than of
#    the code doing the rotating, and it breaks silently if either moves.
#
# 2. Copying the other channels is what removes the concat.  The DSA indexer's
#    ``_apply_rope`` ends in ``paddle.concat([x_pe, x_nope], axis=-1)``, which
#    rebuilds the whole [b,s,h,128] tensor to update 64 channels; the kernel's
#    output already *is* that tensor, so one read + write replaces a pe-sized
#    rope followed by a full-width concat.
#
# When ``pe_dim == head_dim`` (MLA k_pos_emb) there are no other channels and
# ``COPY_OTHER`` compiles away, leaving a plain out-of-place rope.


def _get_block_h_tiled(nheads: int, block_d: int) -> int:
    """``_get_block_h`` capped so one tile stays around 2K elements.

    The copy path materialises a [BLOCK_H, BLOCK_D] tile, which for the MLA q
    shape (D = 256) would be 8K elements at BLOCK_H = 32.
    """
    block_h = _get_block_h(nheads)
    while block_h > 1 and block_h * block_d > 2048:
        block_h //= 2
    return block_h


@triton.jit
def _rope_half_out_fwd_kernel(
    X,
    OUT,
    COS,
    SIN,
    pe_offset,
    pe_dim: tl.constexpr,
    head_dim: tl.constexpr,
    head_num: tl.constexpr,
    stride_x_seq,
    stride_x_nheads,
    stride_o_seq,
    stride_o_nheads,
    COPY_OTHER: tl.constexpr,
    ADJACENT_IN: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """out = x with [pe_offset, pe_offset + pe_dim) rotated, rest copied."""
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

    X = X + pid_m * stride_x_seq + pid_head * BLOCK_H * stride_x_nheads
    OUT = OUT + pid_m * stride_o_seq + pid_head * BLOCK_H * stride_o_nheads

    hrow = tl.arange(0, BLOCK_H)[:, None]
    head_mask = (pid_head * BLOCK_H + tl.arange(0, BLOCK_H))[:, None] < head_num
    xs = hrow * stride_x_nheads.to(tl.int64)
    os = hrow * stride_o_nheads.to(tl.int64)

    if COPY_OTHER:
        # Everything outside the rope block moves across untouched. Masking one
        # full-width tile is cheaper than two separate lead/trail windows and
        # handles pe_offset == 0 (trail only) and pe_offset + pe_dim == head_dim
        # (lead only) without a special case.
        d = tl.arange(0, BLOCK_D)[None, :]
        other = (
            (d < head_dim)
            & ((d < pe_offset) | (d >= pe_offset + pe_dim))
            & head_mask
        )
        tl.store(OUT + os + d, tl.load(X + xs + d, mask=other), mask=other)

    if ADJACENT_IN:
        # The pair sharing a frequency is adjacent in the *source*, as in
        # ``rotary_fwd_q_kernel`` in ``fused_mla_yarn_rope_apply.py``. Only
        # the gather positions move: cos/sin indexing, the arithmetic below, the
        # rounding and the half-split store positions are all unchanged, which
        # is what keeps this bit-exact with the eager
        # ``multi_latent_attention=True, mla_output_remove_interleaving=False``
        # pair. 32 strided loads, no register shuffle.
        off_l = xs + pe_offset + tl.arange(0, half)[None, :] * 2
        off_r = off_l + 1
    else:
        off_l = xs + pe_offset + tl.arange(0, half)[None, :]
        off_r = off_l + half
    out_l = os + pe_offset + tl.arange(0, half)[None, :]
    out_r = out_l + half

    x_1 = tl.load(X + off_l, mask=head_mask)
    x_2 = tl.load(X + off_r, mask=head_mask)

    # Same rounding discipline as the MLA kernels above; see
    # ``_mul_round_bf16`` for why each product is rounded independently.
    y_left = _mul_round_bf16(x_1, cos_left).to(tl.float32) - _mul_round_bf16(
        x_2, sin_left
    ).to(tl.float32)
    y_right = _mul_round_bf16(x_2, cos_right).to(tl.float32) + _mul_round_bf16(
        x_1, sin_right
    ).to(tl.float32)

    tl.store(OUT + out_l, y_left, mask=head_mask)
    tl.store(OUT + out_r, y_right, mask=head_mask)


@triton.jit
def _rope_half_out_bwd_kernel(
    DO,
    DX,
    COS,
    SIN,
    pe_offset,
    pe_dim: tl.constexpr,
    head_dim: tl.constexpr,
    head_num: tl.constexpr,
    stride_o_seq,
    stride_o_nheads,
    stride_x_seq,
    stride_x_nheads,
    COPY_OTHER: tl.constexpr,
    ADJACENT_IN: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """dx = do with the rope block transformed by the forward 2x2 transpose."""
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

    DO = DO + pid_m * stride_o_seq + pid_head * BLOCK_H * stride_o_nheads
    DX = DX + pid_m * stride_x_seq + pid_head * BLOCK_H * stride_x_nheads

    hrow = tl.arange(0, BLOCK_H)[:, None]
    head_mask = (pid_head * BLOCK_H + tl.arange(0, BLOCK_H))[:, None] < head_num
    os = hrow * stride_o_nheads.to(tl.int64)
    xs = hrow * stride_x_nheads.to(tl.int64)

    if COPY_OTHER:
        # The forward copied these channels, so the VJP copies the gradient.
        d = tl.arange(0, BLOCK_D)[None, :]
        other = (
            (d < head_dim)
            & ((d < pe_offset) | (d >= pe_offset + pe_dim))
            & head_mask
        )
        tl.store(DX + xs + d, tl.load(DO + os + d, mask=other), mask=other)

    off_l = os + pe_offset + tl.arange(0, half)[None, :]
    off_r = off_l + half
    if ADJACENT_IN:
        # ``off_l``/``off_r`` follow the *output* layout, which ADJACENT_IN does
        # not change; only the input-gradient scatter does, mirroring the
        # forward gather. Together they still cover every channel of the block
        # exactly once, so ``dx`` stays fully written.
        dx_l = xs + pe_offset + tl.arange(0, half)[None, :] * 2
        dx_r = dx_l + 1
    else:
        dx_l = xs + pe_offset + tl.arange(0, half)[None, :]
        dx_r = dx_l + half

    g1 = tl.load(DO + off_l, mask=head_mask)
    g2 = tl.load(DO + off_r, mask=head_mask)

    dx_1 = _mul_round_bf16(g1, cos_left).to(tl.float32) + _mul_round_bf16(
        g2, sin_right
    ).to(tl.float32)
    dx_2 = _mul_round_bf16(g2, cos_right).to(tl.float32) - _mul_round_bf16(
        g1, sin_left
    ).to(tl.float32)

    tl.store(DX + dx_l, dx_1, mask=head_mask)
    tl.store(DX + dx_r, dx_2, mask=head_mask)


class RoPEHalfOutFusion(paddle.autograd.PyLayer):
    """PyLayer wrapping the out-of-place rotate_half fwd/bwd kernels."""

    @staticmethod
    def forward(ctx, t, cos, sin, pe_dim, pe_offset, adjacent_in):
        # Input validation via explicit exceptions (not ``assert``): this is the
        # PyLayer behind the public ``fused_apply_rope_half`` entry, so under
        # ``python -O`` a stripped assert would let a wrong shape / non-contiguous
        # buffer into the Triton kernel.
        if t.stride(-1) != 1:
            raise ValueError(
                f"RoPEHalfOutFusion needs t last dim contiguous, got "
                f"strides={t.strides}"
            )
        if not cos.is_contiguous():
            raise ValueError("RoPEHalfOutFusion needs cos contiguous")
        if not sin.is_contiguous():
            raise ValueError("RoPEHalfOutFusion needs sin contiguous")
        B, S, H, D = t.shape
        if D < pe_offset + pe_dim:
            raise ValueError(
                f"rope block [{pe_offset}, {pe_offset + pe_dim}) does not fit "
                f"in t's last dim {D}"
            )
        if pe_dim % 4 != 0:
            raise ValueError(f"pe_dim {pe_dim} must be a multiple of 4")
        if cos.shape[-1] != pe_dim:
            raise ValueError(f"cos last dim {cos.shape[-1]} != pe_dim {pe_dim}")
        if sin.shape[-1] != pe_dim:
            raise ValueError(f"sin last dim {sin.shape[-1]} != pe_dim {pe_dim}")

        out = paddle.empty([B, S, H, D], dtype=t.dtype)
        t_flat = t.reshape([B * S, H, D])
        o_flat = out.reshape([B * S, H, D])

        copy_other = pe_dim < D
        block_d = triton.next_power_of_2(D)
        BLOCK_H = (
            _get_block_h_tiled(H, block_d) if copy_other else _get_block_h(H)
        )
        grid = (B * S, triton.cdiv(H, BLOCK_H))
        _rope_half_out_fwd_kernel[grid](
            t_flat,
            o_flat,
            cos,
            sin,
            pe_offset,
            pe_dim,
            D,
            H,
            t_flat.stride(0),
            t_flat.stride(1),
            o_flat.stride(0),
            o_flat.stride(1),
            copy_other,
            adjacent_in,
            block_d,
            BLOCK_H,
        )

        ctx.save_for_backward(cos, sin)
        ctx.pe_dim = pe_dim
        ctx.pe_offset = pe_offset
        ctx.block_h = BLOCK_H
        ctx.block_d = block_d
        ctx.copy_other = copy_other
        ctx.adjacent_in = adjacent_in
        ctx.shape = (B, S, H, D)
        return out

    @staticmethod
    def backward(ctx, grad):
        cos, sin = ctx.saved_tensors
        B, S, H, D = ctx.shape
        grad_flat = grad.contiguous().reshape([B * S, H, D])
        dx = paddle.empty([B * S, H, D], dtype=grad.dtype)
        BLOCK_H = ctx.block_h
        grid = (B * S, triton.cdiv(H, BLOCK_H))
        _rope_half_out_bwd_kernel[grid](
            grad_flat,
            dx,
            cos,
            sin,
            ctx.pe_offset,
            ctx.pe_dim,
            D,
            H,
            grad_flat.stride(0),
            grad_flat.stride(1),
            dx.stride(0),
            dx.stride(1),
            ctx.copy_other,
            ctx.adjacent_in,
            ctx.block_d,
            BLOCK_H,
        )
        return dx.reshape([B, S, H, D]), None, None  # (t, cos, sin)


def fused_apply_rope_half(
    t: paddle.Tensor,
    freqs: paddle.Tensor,
    pe_dim: int,
    mscale: float = 1.0,
    pe_offset: int = 0,
    adjacent_in: bool = False,
) -> paddle.Tensor:
    """Out-of-place RoPE on a contiguous rope block, rotate_half convention.

    Bit-identical to the eager slice + rotate_half + concat it replaces.
    ``t`` is left untouched: the rotated pe block and a verbatim copy of
    every other channel land in a fresh buffer, so the result is the
    assembled tensor and no concat is needed afterwards.  See the section
    comment above for why nothing here works in place.

    Args:
        t: [B, S, H, D] with D >= pe_offset + pe_dim, last-dim-contiguous,
            bf16. Not modified.
        freqs: [B or 1, S, 1, pe_dim] angle tensor, fp32 or bf16. May be
            non-contiguous.
        pe_dim: width of the rope block.
        mscale: rotary scaling factor. Must be 1.0 when ``freqs`` is not fp32:
            ``tl.cos``/``tl.sin`` need fp32, and while upcasting bf16 freqs is
            bit-exact for the transcendental itself (paddle also evaluates it
            in fp32 and rounds once), the eager prelude then multiplies the
            *already rounded* bf16 cosine by mscale while this kernel scales
            in fp32.  Those agree only at mscale == 1.0.
        pe_offset: channel offset of the rope block inside each head.
        adjacent_in: pair source channels ``(2k, 2k+1)`` instead of
            ``(k, k+half)``. The output still lands in halves, so this is
            bit-exact with the eager ``multi_latent_attention=True,
            mla_output_remove_interleaving=False`` pair (and with
            ``fused_apply_mla_rope_for_q``) rather than with the default's
            ``multi_latent_attention=False``. Default False leaves every
            existing caller, including the DSA indexer, unchanged.

    Returns:
        A new [B, S, H, D] tensor: [..., pe_offset : pe_offset + pe_dim] is
        rotated, every other channel is a verbatim copy of ``t``.
    """
    # Production input validation -- explicit exceptions, not ``assert``: under
    # ``python -O`` a stripped assert would let fp16 into the bf16-rounding path
    # or a wrong shape into Triton address arithmetic and silently corrupt
    # results. dsa_attention.py:465 explicitly relies on these checks.
    if t.stride(-1) != 1:
        raise ValueError(
            "fused_apply_rope_half requires t's last dim to be contiguous, got "
            f"shape={t.shape} strides={t.strides}"
        )
    B, S, H, D = t.shape
    if not (pe_offset >= 0 and D >= pe_offset + pe_dim):
        raise ValueError(
            f"rope block [{pe_offset}, {pe_offset + pe_dim}) does not fit in t's "
            f"last dim {D}"
        )
    if t.dtype != paddle.bfloat16:
        raise ValueError(
            f"fused_apply_rope_half is designed for bf16, got {t.dtype}"
        )

    if not (freqs.dim() == 4 and freqs.shape[2] == 1):
        raise ValueError(f"freqs must be [B,S,1,D]; got {freqs.shape}")
    B_f, S_f, _, D_f = freqs.shape
    if not (S_f == S and D_f == pe_dim):
        raise ValueError(
            f"freqs [B,S,1,D]={freqs.shape} mismatches t [B,S]=[{B},{S}], "
            f"pe_dim={pe_dim}"
        )
    if not (B_f == 1 or B_f == B):
        raise ValueError(f"freqs B {B_f} must be 1 or {B}")
    if B_f < B:
        freqs = freqs.broadcast_to([B, S, 1, pe_dim])

    if freqs.dtype != paddle.float32:
        if mscale != 1.0:
            raise ValueError(
                "fused_apply_rope_half needs fp32 freqs when mscale != 1 (got "
                f"freqs.dtype={freqs.dtype}, mscale={mscale}): the eager path "
                "rounds cos/sin to bf16 before applying mscale."
            )
        freqs = freqs.astype(paddle.float32)

    cos, sin = _fused_cos_sin(freqs, mscale, False, t.dtype)

    return RoPEHalfOutFusion.apply(t, cos, sin, pe_dim, pe_offset, adjacent_in)
