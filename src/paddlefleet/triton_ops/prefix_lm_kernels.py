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

"""The 4 Triton kernels for packed prefix-LM attention.

These are hand-tuned Triton kernels implementing packed prefix-LM flash
attention: one forward kernel plus three backward kernels (delta, dK/dV, and
dQ). The kernel bodies use only `triton` / `tl` and do not touch
any deep-learning framework.

Note: Triton's `@triton.jit` retrieves source code via `inspect`, so these
kernels must live in a real `.py` file; defining them by `exec` on a string
would raise
`ValueError: @jit functions should be defined in a Python file`.
"""

import triton
import triton.language as tl

__all__ = [
    "_prefix_lm_fwd_kernel",
    "_prefix_lm_delta_kernel",
    "_prefix_lm_dkdv_kernel",
    "_prefix_lm_dq_kernel",
]

# ==== Triton kernels ====


@triton.jit
def _prefix_lm_fwd_kernel(
    Q,
    K,
    V,
    Out,
    Lse,
    seg_start_ptr,
    seg_ctx_ptr,
    kv_lo_ptr,
    kv_hi_ptr,
    full_ptr,
    stride_qb,
    stride_qh,
    stride_qt,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kt,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vt,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_ot,
    stride_od,
    stride_lb,
    stride_lh,
    stride_lt,
    stride_fm,
    stride_fn,
    seq_len,
    pad_start,
    scale,
    H: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Flash-attention forward with the packed prefix-LM three-region mask.

    grid = (num_q_blocks, B * H). Online softmax in fp32; output cast to the
    input dtype on store. ``Lse`` (m + log l, fp32) is saved for backward.

    ``full_ptr`` is a per-``(q-block, kv-block)`` int8 flag (see
    :func:`build_block_full_flags`): on a FULL block every pair is allowed,
    so the whole three-region predicate, the two ``seg_start`` / ``seg_ctx``
    loads, and both ``tl.where`` calls are skipped -- the compute/issue-side
    overhead that dominates long single-segment forwards. Partial blocks run
    the exact masked path. Both branches are numerically identical on the
    blocks they handle (validated bit-for-bit vs the dense reference).
    """

    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    q_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offs = tl.arange(0, HEAD_DIM)

    q_base = Q + b * stride_qb + h * stride_qh
    q_ptrs = q_base + q_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd
    q_mask = q_offs[:, None] < seq_len
    # Keep Q/K/V in their native dtype (bf16 in training) so ``tl.dot`` runs
    # on the bf16 tensor cores with an fp32 accumulator -- same numerics as
    # flex_attention. Upcasting to fp32 here would force the non-tensor-core
    # ieee path (~10-30x slower). For fp32 inputs (parity tests) the load
    # stays fp32 and ``input_precision="ieee"`` keeps the dot exact.
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # per-q-row segment metadata
    q_seg_start = tl.load(
        seg_start_ptr + q_offs, mask=q_offs < seq_len, other=-1
    )
    q_seg_ctx = tl.load(seg_ctx_ptr + q_offs, mask=q_offs < seq_len, other=0)
    q_local = q_offs - q_seg_start
    q_in_ctx = q_local < q_seg_ctx
    q_is_real = q_offs < pad_start

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    kv_lo = tl.load(kv_lo_ptr + pid_m)
    kv_hi = tl.load(kv_hi_ptr + pid_m)

    k_base = K + b * stride_kb + h * stride_kh
    v_base = V + b * stride_vb + h * stride_vh
    full_row = full_ptr + pid_m * stride_fm

    for blk in range(kv_lo, kv_hi):
        k_offs = blk * BLOCK_N + tl.arange(0, BLOCK_N)
        k_valid = k_offs < seq_len
        k_ptrs = (
            k_base + k_offs[:, None] * stride_kt + d_offs[None, :] * stride_kd
        )
        k = tl.load(k_ptrs, mask=k_valid[:, None], other=0.0)

        s = (
            tl.dot(q, tl.trans(k), input_precision="ieee") * scale
        )  # [BLOCK_M, BLOCK_N]

        is_full = tl.load(full_row + blk * stride_fn) != 0
        if is_full:
            # FULL block: every pair allowed -> no predicate, no tl.where.
            # ``build_block_full_flags`` only flags a block full when it is a
            # complete in-range single-segment tile with no pad, so k_valid
            # is all-true here too and needs no re-masking.
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            p = tl.exp(s - m_safe[:, None])
            alpha = tl.exp(m_i - m_safe)
            alpha = tl.where(m_i == float("-inf"), 0.0, alpha)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            v_ptrs = (
                v_base
                + k_offs[:, None] * stride_vt
                + d_offs[None, :] * stride_vd
            )
            v = tl.load(v_ptrs, mask=k_valid[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(
                p.to(v.dtype), v, input_precision="ieee"
            )
            m_i = m_new
        else:
            k_seg_start = tl.load(
                seg_start_ptr + k_offs, mask=k_valid, other=-2
            )
            k_seg_ctx = tl.load(seg_ctx_ptr + k_offs, mask=k_valid, other=0)
            k_local = k_offs - k_seg_start
            k_in_ctx = k_local < k_seg_ctx
            k_is_real = k_offs < pad_start

            same_seg = (
                (q_seg_start[:, None] == k_seg_start[None, :])
                & q_is_real[:, None]
                & k_is_real[None, :]
            )
            qc = q_in_ctx[:, None]
            kc = k_in_ctx[None, :]
            ge = q_local[:, None] >= k_local[None, :]
            region = (qc & kc) | ((~qc) & kc) | ((~qc) & (~kc) & ge)
            allow = same_seg & region
            # pad diagonal self-loop: q in pad and q_idx == k_idx
            pad_diag = (~q_is_real[:, None]) & (
                q_offs[:, None] == k_offs[None, :]
            )
            allow = allow | pad_diag
            allow = allow & k_valid[None, :]

            s = tl.where(allow, s, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            p = tl.exp(s - m_safe[:, None])
            p = tl.where(allow, p, 0.0)
            alpha = tl.exp(m_i - m_safe)
            alpha = tl.where(m_i == float("-inf"), 0.0, alpha)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            v_ptrs = (
                v_base
                + k_offs[:, None] * stride_vt
                + d_offs[None, :] * stride_vd
            )
            v = tl.load(v_ptrs, mask=k_valid[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(
                p.to(v.dtype), v, input_precision="ieee"
            )
            m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc / l_safe[:, None]
    lse = tl.where(l_i == 0.0, float("-inf"), m_i + tl.log(l_safe))

    o_base = Out + b * stride_ob + h * stride_oh
    o_ptrs = o_base + q_offs[:, None] * stride_ot + d_offs[None, :] * stride_od
    tl.store(
        o_ptrs, out.to(Out.dtype.element_ty), mask=q_offs[:, None] < seq_len
    )
    l_base = Lse + b * stride_lb + h * stride_lh
    tl.store(l_base + q_offs * stride_lt, lse, mask=q_offs < seq_len)


@triton.jit
def _prefix_lm_delta_kernel(
    Out,
    DOut,
    Delta,
    stride_ob,
    stride_oh,
    stride_ot,
    stride_od,
    stride_dob,
    stride_doh,
    stride_dot,
    stride_dod,
    stride_deb,
    stride_deh,
    stride_det,
    seq_len,
    H: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """delta[i] = rowsum(dO[i] * O[i]); one fp32 scalar per query row."""

    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offs = tl.arange(0, HEAD_DIM)
    valid = offs < seq_len

    o_ptrs = (
        Out
        + b * stride_ob
        + h * stride_oh
        + offs[:, None] * stride_ot
        + d_offs[None, :] * stride_od
    )
    do_ptrs = (
        DOut
        + b * stride_dob
        + h * stride_doh
        + offs[:, None] * stride_dot
        + d_offs[None, :] * stride_dod
    )
    o = tl.load(o_ptrs, mask=valid[:, None], other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=valid[:, None], other=0.0).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(
        Delta + b * stride_deb + h * stride_deh + offs * stride_det,
        delta,
        mask=valid,
    )


@triton.jit
def _prefix_lm_dkdv_kernel(
    Q,
    K,
    V,
    DOut,
    Lse,
    Delta,
    DK,
    DV,
    seg_start_ptr,
    seg_ctx_ptr,
    q_lo_ptr,
    q_hi_ptr,
    full_ptr,
    stride_qb,
    stride_qh,
    stride_qt,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kt,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vt,
    stride_vd,
    stride_dob,
    stride_doh,
    stride_dot,
    stride_dod,
    stride_lb,
    stride_lh,
    stride_lt,
    stride_deb,
    stride_deh,
    stride_det,
    stride_dkb,
    stride_dkh,
    stride_dkt,
    stride_dkd,
    stride_dvb,
    stride_dvh,
    stride_dvt,
    stride_dvd,
    stride_fm,
    stride_fn,
    seq_len,
    pad_start,
    scale,
    H: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Accumulate dK, dV for one KV block over its relevant Q blocks.

    grid = (num_kv_blocks, B*H). Recomputes ``P = exp(scale*QKᵀ − lse)`` with
    the same three-region mask as forward. ``dV = Pᵀ@dO``,
    ``dS = P∘(dP−delta)`` with ``dP = dO@Vᵀ``, ``dK = scale·(dSᵀ@Q)``.

    ``full_ptr`` is the int8 ``[num_q_blocks, num_kv_blocks]`` flag indexed
    ``[qb, kb]`` (see :func:`build_block_full_flags_bwd`): on a FULL block
    every pair is allowed, so the q-side ``seg_start`` / ``seg_ctx`` loads,
    the three-region predicate, and both ``tl.where`` calls are skipped.
    Partial blocks run the exact masked path; numerically identical per block.
    """

    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    k_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    d_offs = tl.arange(0, HEAD_DIM)
    k_valid = k_offs < seq_len

    k_ptrs = (
        K
        + b * stride_kb
        + h * stride_kh
        + k_offs[:, None] * stride_kt
        + d_offs[None, :] * stride_kd
    )
    v_ptrs = (
        V
        + b * stride_vb
        + h * stride_vh
        + k_offs[:, None] * stride_vt
        + d_offs[None, :] * stride_vd
    )
    k = tl.load(k_ptrs, mask=k_valid[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=k_valid[:, None], other=0.0)

    k_seg_start = tl.load(seg_start_ptr + k_offs, mask=k_valid, other=-2)
    k_seg_ctx = tl.load(seg_ctx_ptr + k_offs, mask=k_valid, other=0)
    k_local = k_offs - k_seg_start
    k_in_ctx = k_local < k_seg_ctx
    k_is_real = k_offs < pad_start

    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    q_lo = tl.load(q_lo_ptr + pid_n)
    q_hi = tl.load(q_hi_ptr + pid_n)
    full_col = full_ptr + pid_n * stride_fn

    for blk in range(q_lo, q_hi):
        q_offs = blk * BLOCK_M + tl.arange(0, BLOCK_M)
        q_valid = q_offs < seq_len
        q_ptrs = (
            Q
            + b * stride_qb
            + h * stride_qh
            + q_offs[:, None] * stride_qt
            + d_offs[None, :] * stride_qd
        )
        q = tl.load(q_ptrs, mask=q_valid[:, None], other=0.0)
        do_ptrs = (
            DOut
            + b * stride_dob
            + h * stride_doh
            + q_offs[:, None] * stride_dot
            + d_offs[None, :] * stride_dod
        )
        do = tl.load(do_ptrs, mask=q_valid[:, None], other=0.0)
        lse = tl.load(
            Lse + b * stride_lb + h * stride_lh + q_offs * stride_lt,
            mask=q_valid,
            other=0.0,
        )
        delta = tl.load(
            Delta + b * stride_deb + h * stride_deh + q_offs * stride_det,
            mask=q_valid,
            other=0.0,
        )

        s = (
            tl.dot(q, tl.trans(k), input_precision="ieee") * scale
        )  # [BLOCK_M, BLOCK_N]

        is_full = tl.load(full_col + blk * stride_fm) != 0
        if is_full:
            # FULL block: every pair allowed -> no q-side seg loads, no
            # predicate, no tl.where. ``build_block_full_flags_bwd`` only
            # flags a block full when it is a complete in-range single-segment
            # tile with no pad, so k_valid/q_valid are all-true here too.
            p = tl.exp(s - lse[:, None])  # [BLOCK_M, BLOCK_N]
            dv += tl.dot(tl.trans(p).to(do.dtype), do, input_precision="ieee")
            dp = tl.dot(do, tl.trans(v), input_precision="ieee")
            ds = p * (dp - delta[:, None])
            dk += (
                tl.dot(tl.trans(ds).to(q.dtype), q, input_precision="ieee")
                * scale
            )
        else:
            q_seg_start = tl.load(
                seg_start_ptr + q_offs, mask=q_valid, other=-1
            )
            q_seg_ctx = tl.load(seg_ctx_ptr + q_offs, mask=q_valid, other=0)
            q_local = q_offs - q_seg_start
            q_in_ctx = q_local < q_seg_ctx
            q_is_real = q_offs < pad_start

            same_seg = (
                (q_seg_start[:, None] == k_seg_start[None, :])
                & q_is_real[:, None]
                & k_is_real[None, :]
            )
            qc = q_in_ctx[:, None]
            kc = k_in_ctx[None, :]
            ge = q_local[:, None] >= k_local[None, :]
            region = (qc & kc) | ((~qc) & kc) | ((~qc) & (~kc) & ge)
            allow = same_seg & region
            pad_diag = (~q_is_real[:, None]) & (
                q_offs[:, None] == k_offs[None, :]
            )
            allow = (allow | pad_diag) & k_valid[None, :] & q_valid[:, None]

            p = tl.where(
                allow, tl.exp(s - lse[:, None]), 0.0
            )  # [BLOCK_M, BLOCK_N]

            # dV += Pᵀ @ dO
            dv += tl.dot(tl.trans(p).to(do.dtype), do, input_precision="ieee")
            # dP = dO @ Vᵀ ; dS = P∘(dP−delta) ; dK += scale·(dSᵀ@Q)
            dp = tl.dot(do, tl.trans(v), input_precision="ieee")
            ds = p * (dp - delta[:, None])
            ds = tl.where(allow, ds, 0.0)
            dk += (
                tl.dot(tl.trans(ds).to(q.dtype), q, input_precision="ieee")
                * scale
            )

    dk_ptrs = (
        DK
        + b * stride_dkb
        + h * stride_dkh
        + k_offs[:, None] * stride_dkt
        + d_offs[None, :] * stride_dkd
    )
    dv_ptrs = (
        DV
        + b * stride_dvb
        + h * stride_dvh
        + k_offs[:, None] * stride_dvt
        + d_offs[None, :] * stride_dvd
    )
    tl.store(dk_ptrs, dk.to(DK.dtype.element_ty), mask=k_valid[:, None])
    tl.store(dv_ptrs, dv.to(DV.dtype.element_ty), mask=k_valid[:, None])


@triton.jit
def _prefix_lm_dq_kernel(
    Q,
    K,
    V,
    DOut,
    Lse,
    Delta,
    DQ,
    seg_start_ptr,
    seg_ctx_ptr,
    kv_lo_ptr,
    kv_hi_ptr,
    full_ptr,
    stride_qb,
    stride_qh,
    stride_qt,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kt,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vt,
    stride_vd,
    stride_dob,
    stride_doh,
    stride_dot,
    stride_dod,
    stride_lb,
    stride_lh,
    stride_lt,
    stride_deb,
    stride_deh,
    stride_det,
    stride_dqb,
    stride_dqh,
    stride_dqt,
    stride_dqd,
    stride_fm,
    stride_fn,
    seq_len,
    pad_start,
    scale,
    H: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Accumulate dQ for one Q block over its relevant KV blocks (no atomics).

    grid = (num_q_blocks, B*H). ``dS = P∘(dP−delta)``, ``dQ = scale·(dS@K)``.

    Shares the FORWARD scan orientation (per-q-block KV range), so it reuses
    the forward ``full_flags`` indexed ``[pid_m, blk]``: on a FULL block the
    k-side ``seg_start`` / ``seg_ctx`` loads, the predicate, and both
    ``tl.where`` calls are skipped. Partial blocks run the exact masked path.
    """

    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    q_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offs = tl.arange(0, HEAD_DIM)
    q_valid = q_offs < seq_len

    q_ptrs = (
        Q
        + b * stride_qb
        + h * stride_qh
        + q_offs[:, None] * stride_qt
        + d_offs[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=q_valid[:, None], other=0.0)
    do_ptrs = (
        DOut
        + b * stride_dob
        + h * stride_doh
        + q_offs[:, None] * stride_dot
        + d_offs[None, :] * stride_dod
    )
    do = tl.load(do_ptrs, mask=q_valid[:, None], other=0.0)
    lse = tl.load(
        Lse + b * stride_lb + h * stride_lh + q_offs * stride_lt,
        mask=q_valid,
        other=0.0,
    )
    delta = tl.load(
        Delta + b * stride_deb + h * stride_deh + q_offs * stride_det,
        mask=q_valid,
        other=0.0,
    )

    q_seg_start = tl.load(seg_start_ptr + q_offs, mask=q_valid, other=-1)
    q_seg_ctx = tl.load(seg_ctx_ptr + q_offs, mask=q_valid, other=0)
    q_local = q_offs - q_seg_start
    q_in_ctx = q_local < q_seg_ctx
    q_is_real = q_offs < pad_start

    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    kv_lo = tl.load(kv_lo_ptr + pid_m)
    kv_hi = tl.load(kv_hi_ptr + pid_m)
    full_row = full_ptr + pid_m * stride_fm

    for blk in range(kv_lo, kv_hi):
        k_offs = blk * BLOCK_N + tl.arange(0, BLOCK_N)
        k_valid = k_offs < seq_len
        k_ptrs = (
            K
            + b * stride_kb
            + h * stride_kh
            + k_offs[:, None] * stride_kt
            + d_offs[None, :] * stride_kd
        )
        v_ptrs = (
            V
            + b * stride_vb
            + h * stride_vh
            + k_offs[:, None] * stride_vt
            + d_offs[None, :] * stride_vd
        )
        k = tl.load(k_ptrs, mask=k_valid[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=k_valid[:, None], other=0.0)

        s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale

        is_full = tl.load(full_row + blk * stride_fn) != 0
        if is_full:
            # FULL block: every pair allowed -> no k-side seg loads, no
            # predicate, no tl.where.
            p = tl.exp(s - lse[:, None])
            dp = tl.dot(do, tl.trans(v), input_precision="ieee")
            ds = p * (dp - delta[:, None])
            dq += tl.dot(ds.to(k.dtype), k, input_precision="ieee") * scale
        else:
            k_seg_start = tl.load(
                seg_start_ptr + k_offs, mask=k_valid, other=-2
            )
            k_seg_ctx = tl.load(seg_ctx_ptr + k_offs, mask=k_valid, other=0)
            k_local = k_offs - k_seg_start
            k_in_ctx = k_local < k_seg_ctx
            k_is_real = k_offs < pad_start

            same_seg = (
                (q_seg_start[:, None] == k_seg_start[None, :])
                & q_is_real[:, None]
                & k_is_real[None, :]
            )
            qc = q_in_ctx[:, None]
            kc = k_in_ctx[None, :]
            ge = q_local[:, None] >= k_local[None, :]
            region = (qc & kc) | ((~qc) & kc) | ((~qc) & (~kc) & ge)
            allow = same_seg & region
            pad_diag = (~q_is_real[:, None]) & (
                q_offs[:, None] == k_offs[None, :]
            )
            allow = (allow | pad_diag) & k_valid[None, :] & q_valid[:, None]

            p = tl.where(allow, tl.exp(s - lse[:, None]), 0.0)
            dp = tl.dot(do, tl.trans(v), input_precision="ieee")
            ds = p * (dp - delta[:, None])
            ds = tl.where(allow, ds, 0.0)
            dq += tl.dot(ds.to(k.dtype), k, input_precision="ieee") * scale

    dq_ptrs = (
        DQ
        + b * stride_dqb
        + h * stride_dqh
        + q_offs[:, None] * stride_dqt
        + d_offs[None, :] * stride_dqd
    )
    tl.store(dq_ptrs, dq.to(DQ.dtype.element_ty), mask=q_valid[:, None])
