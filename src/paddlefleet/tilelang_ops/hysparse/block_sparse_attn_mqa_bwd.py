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

"""Backward for the MQA gather block-sparse attention
(:mod:`block_sparse_attn_mqa`).

Mirrors the forward's layout: one program per query token, with the ``H``
query heads placed on the GEMM ``M`` dimension and the single shared K/V head
gathered ``block_B`` at a time over the ``nsel`` selected blocks. dQ for the
token is accumulated locally; dK/dV are scattered into the shared K/V head via
``atomic_add`` (many query tokens may select the same block).
"""

import paddle
import tilelang
from tilelang import language as T


@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def block_sparse_mqa_bwd(
    H,
    D,
    nsel,
    sm_scale,
    block_B=64,
    block_H=None,
    num_stages=1,
    threads=128,
):
    assert D % 16 == 0, (
        f"D must be a multiple of 16 (tensor-core k-tile), got {D}"
    )
    assert H <= 128, "this kernel supports up to 128 query heads"

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    q_shape = [batch, seq_len, H, D]
    kv_shape = [batch, seq_len_kv, D]
    idx_shape = [batch, seq_len, nsel]
    vr_shape = [batch, seq_len, 2]
    lse_shape = [batch, seq_len, H]

    dtype = T.bfloat16
    accum_dtype = T.float32
    idx_dtype = T.int32
    # Heads are tiled on the GEMM M dimension in groups of ``BH`` so that the
    # ``[·, D]`` shared buffers fit on-chip for large head dims (e.g. MLA
    # D=576); ``BH == H`` keeps the single-program-per-token fast path.
    BH = block_H if block_H is not None else H
    PH = max(tilelang.math.next_power_of_2(BH), 16)
    num_hg = (H + BH - 1) // BH
    BB = block_B

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        dO: T.Tensor(q_shape, dtype),
        Lse: T.Tensor(lse_shape, accum_dtype),
        Delta: T.Tensor(lse_shape, accum_dtype),
        Indices: T.Tensor(idx_shape, idx_dtype),
        ValidRange: T.Tensor(vr_shape, idx_dtype),
        dK: T.Tensor(kv_shape, accum_dtype),
        dV: T.Tensor(kv_shape, accum_dtype),
        dQ: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(num_hg, seq_len, batch, threads=threads) as (hg, bs, bb):
            h0 = hg * BH
            Q_shared = T.alloc_shared([PH, D], dtype)
            dO_shared = T.alloc_shared([PH, D], dtype)
            K_shared = T.alloc_shared([BB, D], dtype)
            V_shared = T.alloc_shared([BB, D], dtype)
            P_shared = T.alloc_shared([PH, BB], dtype)
            dS_shared = T.alloc_shared([PH, BB], dtype)
            dQ_shared = T.alloc_shared([PH, D], dtype)

            acc_s = T.alloc_fragment([PH, BB], accum_dtype)
            acc_p = T.alloc_fragment([PH, BB], accum_dtype)
            acc_dp = T.alloc_fragment([PH, BB], accum_dtype)
            acc_dq = T.alloc_fragment([PH, D], accum_dtype)
            acc_dk = T.alloc_fragment([BB, D], accum_dtype)
            acc_dv = T.alloc_fragment([BB, D], accum_dtype)
            lse_f = T.alloc_fragment([PH], accum_dtype)
            delta_f = T.alloc_fragment([PH], accum_dtype)

            bos = ValidRange[bb, bs, 0]
            eos = ValidRange[bb, bs, 1]

            for h, d in T.Parallel(PH, D):
                gh = h0 + h
                use = (h < BH) and (gh < H)
                sh = T.if_then_else(use, gh, 0)
                Q_shared[h, d] = T.if_then_else(
                    use, Q[bb, bs, sh, d], T.cast(0, dtype)
                )
                dO_shared[h, d] = T.if_then_else(
                    use, dO[bb, bs, sh, d], T.cast(0, dtype)
                )
            for h in T.Parallel(PH):
                gh = h0 + h
                use = (h < BH) and (gh < H)
                sh = T.if_then_else(use, gh, 0)
                lse_f[h] = T.if_then_else(use, Lse[bb, bs, sh], 0.0)
                delta_f[h] = T.if_then_else(use, Delta[bb, bs, sh], 0.0)

            T.clear(acc_dq)

            for i in T.Pipelined(nsel, num_stages=num_stages):
                blk = Indices[bb, bs, i]
                valid_blk = blk >= 0
                safe_blk = T.if_then_else(valid_blk, blk, 0)

                # document-relative gather: relative block ``blk`` spans
                # absolute columns [bos + blk*BB, bos + (blk+1)*BB). Guard the
                # read against the padded K/V length.
                for c, d in T.Parallel(BB, D):
                    col = bos + safe_blk * BB + c
                    in_bounds = col < seq_len_kv
                    safe_col = T.if_then_else(in_bounds, col, 0)
                    K_shared[c, d] = T.if_then_else(
                        in_bounds, K[bb, safe_col, d], T.cast(0, dtype)
                    )
                    V_shared[c, d] = T.if_then_else(
                        in_bounds, V[bb, safe_col, d], T.cast(0, dtype)
                    )

                # P = softmax prob = exp(raw*sm_scale - lse); masked -> 0.
                T.clear(acc_s)
                T.gemm(
                    Q_shared,
                    K_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                for h, c in T.Parallel(PH, BB):
                    col = bos + safe_blk * BB + c
                    keep = valid_blk and (col < eos)
                    acc_p[h, c] = T.if_then_else(
                        keep,
                        T.exp2(
                            (acc_s[h, c] * sm_scale - lse_f[h]) * 1.44269504
                        ),
                        0.0,
                    )
                T.copy(acc_p, P_shared)

                # dP = dO @ V^T
                T.clear(acc_dp)
                T.gemm(
                    dO_shared,
                    V_shared,
                    acc_dp,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                # dS = P * (dP - Delta) * sm_scale
                for h, c in T.Parallel(PH, BB):
                    acc_dp[h, c] = (
                        acc_p[h, c] * (acc_dp[h, c] - delta_f[h]) * sm_scale
                    )
                T.copy(acc_dp, dS_shared)

                # dQ += dS @ K
                T.gemm(
                    dS_shared,
                    K_shared,
                    acc_dq,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                # dV = P^T @ dO ; dK = dS^T @ Q  (scattered to shared KV)
                T.clear(acc_dv)
                T.gemm(
                    P_shared,
                    dO_shared,
                    acc_dv,
                    transpose_A=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.clear(acc_dk)
                T.gemm(
                    dS_shared,
                    Q_shared,
                    acc_dk,
                    transpose_A=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                for c, d in T.Parallel(BB, D):
                    col = bos + safe_blk * BB + c
                    if valid_blk and (col < seq_len_kv):
                        T.atomic_add(dV[bb, col, d], acc_dv[c, d])
                        T.atomic_add(dK[bb, col, d], acc_dk[c, d])

            T.copy(acc_dq, dQ_shared)
            for h, d in T.Parallel(PH, D):
                gh = h0 + h
                if (h < BH) and (gh < H):
                    dQ[bb, bs, gh, d] = dQ_shared[h, d]

    return main


@tilelang.jit(out_idx=[-1])
def _cast_bf16_kv(D, block_N=64, threads=128):
    batch = T.dynamic("batch")
    seq_len_kv = T.dynamic("seq_len_kv")
    shape = [batch, seq_len_kv, D]

    @T.prim_func
    def main(
        X: T.Tensor(shape, T.float32),
        Out: T.Tensor(shape, T.bfloat16),
    ):
        with T.Kernel(
            T.ceildiv(seq_len_kv, block_N), batch, threads=threads
        ) as (bn, bb):
            for i, d in T.Parallel(block_N, D):
                if bn * block_N + i < seq_len_kv:
                    Out[bb, bn * block_N + i, d] = X[bb, bn * block_N + i, d]

    return main


def _fit_block_h(H, D, block_B, cap_bytes=200000):
    """Largest head-group (a divisor-ish of H) whose per-token backward shared
    buffers fit. Q/dO/dQ ``[PH, D]`` + K/V ``[BB, D]`` + P/dS ``[PH, BB]`` in
    bf16; for large D (MLA 576) all H=64 heads on M overflow, so tile them.
    """
    for bh in (H, 64, 32, 16):
        if bh > H:
            continue
        ph = max(tilelang.math.next_power_of_2(bh), 16)
        shared = 6 * ph * D + 4 * block_B * D + 4 * ph * block_B
        if shared <= cap_bytes:
            return bh
    return min(H, 16)


def block_sparse_mqa_bwd_interface(
    q, k, v, o, do, lse, indices, valid_range, sm_scale=None, block_B=64
):
    """Backward interface for the MQA gather block-sparse attention.

    Args:
        q:           [B, S, H, D] bf16 forward query.
        k, v:        [B, S_kv, D] bf16 shared key/value (forward inputs).
        o:           [B, S, H, D] bf16 forward output.
        do:          [B, S, H, D] bf16 grad of output.
        lse:         [B, S, H] fp32 natural-log LSE from forward.
        indices:     [B, S, nsel] int block ids selected per query token.
        valid_range: [B, S, 2] int32.
        sm_scale:    softmax scale; defaults to D**-0.5.
        block_B:     key block size.

    Returns:
        dq [B,S,H,D] bf16, dk/dv [B,S_kv,D] bf16.
    """
    assert q.is_contiguous() and do.is_contiguous() and lse.is_contiguous()
    b, s, h, d = q.shape
    s_kv = k.shape[1]
    if sm_scale is None:
        sm_scale = d**-0.5
    if valid_range.dtype != paddle.int32:
        valid_range = valid_range.cast("int32")
    if indices.dtype != paddle.int32:
        indices = indices.cast("int32")

    # gather reads whole blocks without clamping -> pad K/V (and dK/dV) so the
    # last block is addressable.
    pad = (block_B - s_kv % block_B) % block_B
    s_kv_pad = s_kv + pad
    if pad > 0:
        k = paddle.nn.functional.pad(k, [0, 0, 0, pad])
        v = paddle.nn.functional.pad(v, [0, 0, 0, pad])
    k = k.contiguous()
    v = v.contiguous()
    valid_range = valid_range.contiguous()
    indices = indices.contiguous()

    delta = (o.astype("float32") * do.astype("float32")).sum(-1).contiguous()
    dk = paddle.zeros([b, s_kv_pad, d], dtype="float32")
    dv = paddle.zeros([b, s_kv_pad, d], dtype="float32")

    bwd = block_sparse_mqa_bwd(
        h,
        d,
        indices.shape[-1],
        float(sm_scale),
        block_B=block_B,
        block_H=_fit_block_h(h, d, block_B),
    )
    dq = bwd(q, k, v, do, lse, delta, indices, valid_range, dk, dv)

    cast = _cast_bf16_kv(d)
    dk_bf = cast(dk)
    dv_bf = cast(dv)
    if pad > 0:
        dk_bf = dk_bf[:, :s_kv, :].contiguous()
        dv_bf = dv_bf[:, :s_kv, :].contiguous()
    return dq, dk_bf, dv_bf
