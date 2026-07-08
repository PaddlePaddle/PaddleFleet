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

"""Backward for the MQA block-score attention: flash-attention backward
(dQ, dK, dV) for full attention with a single shared K/V head and causal +
document masking.

K/V are one shared head ``[B, S_kv, D]``; dK/dV from every query head are
scattered into that single head via ``atomic_add``. The block-score output of
the forward is consumed by a non-differentiable TopK and therefore contributes
no gradient here.
"""

import paddle
import tilelang
from tilelang import language as T

from .block_sparse_attn_mqa_bwd import _cast_bf16_kv


@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def block_score_mqa_bwd(
    H,
    D,
    sm_scale,
    block_M=64,
    block_N=64,
    block_B=64,
    num_stages=1,
    threads=128,
):
    assert D % 16 == 0, (
        f"D must be a multiple of 16 (tensor-core k-tile), got {D}"
    )
    assert block_B % block_N == 0, "block_B must be a multiple of block_N"

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")
    num_bm = T.dynamic("num_bm")

    q_shape = [batch, seq_len, H, D]
    kv_shape = [batch, seq_len_kv, D]
    lse_shape = [batch, seq_len, H]
    vr_shape = [batch, seq_len, 2]
    br_shape = [batch, num_bm, 2]

    dtype = T.bfloat16
    accum_dtype = T.float32
    idx_dtype = T.int32
    BM = block_M
    BN = block_N
    ratio = (
        block_B // block_N
    )  # key sub-tiles per block_B-sized selection block

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        dO: T.Tensor(q_shape, dtype),
        Lse: T.Tensor(lse_shape, accum_dtype),
        Delta: T.Tensor(lse_shape, accum_dtype),
        ValidRange: T.Tensor(vr_shape, idx_dtype),
        BlockRange: T.Tensor(br_shape, idx_dtype),
        dK: T.Tensor(kv_shape, accum_dtype),
        dV: T.Tensor(kv_shape, accum_dtype),
        dQ: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, BM), H, batch, threads=threads) as (
            bm,
            bh,
            bb,
        ):
            Q_shared = T.alloc_shared([BM, D], dtype)
            dO_shared = T.alloc_shared([BM, D], dtype)
            K_shared = T.alloc_shared([BN, D], dtype)
            V_shared = T.alloc_shared([BN, D], dtype)
            P_shared = T.alloc_shared([BM, BN], dtype)
            dS_shared = T.alloc_shared([BM, BN], dtype)

            acc_s = T.alloc_fragment([BM, BN], accum_dtype)
            acc_p = T.alloc_fragment([BM, BN], accum_dtype)
            acc_dp = T.alloc_fragment([BM, BN], accum_dtype)
            acc_dq = T.alloc_fragment([BM, D], accum_dtype)
            acc_dk = T.alloc_fragment([BN, D], accum_dtype)
            acc_dv = T.alloc_fragment([BN, D], accum_dtype)
            lse_f = T.alloc_fragment([BM], accum_dtype)
            delta_f = T.alloc_fragment([BM], accum_dtype)
            bos = T.alloc_fragment([BM], idx_dtype)
            eos = T.alloc_fragment([BM], idx_dtype)

            for i in T.Parallel(BM):
                row = bm * BM + i
                in_range = row < seq_len
                bos[i] = T.if_then_else(in_range, ValidRange[bb, row, 0], 0)
                eos[i] = T.if_then_else(in_range, ValidRange[bb, row, 1], 0)
                lse_f[i] = T.if_then_else(in_range, Lse[bb, row, bh], 0)
                delta_f[i] = T.if_then_else(in_range, Delta[bb, row, bh], 0)

            T.copy(Q[bb, bm * BM : (bm + 1) * BM, bh, :], Q_shared)
            T.copy(dO[bb, bm * BM : (bm + 1) * BM, bh, :], dO_shared)
            T.clear(acc_dq)

            # document-tight early-exit: the host precomputes, per query tile,
            # the block-B key window [jl, jh) actually reachable by this tile's
            # rows -- jl = min_i(bos_i) // block_B skips leading blocks before
            # the tile's document start; jh = ceil(max_i(eos_i) / block_B) skips
            # blocks past the (causal) end. Every skipped block is fully masked
            # (col < bos_i or col >= eos_i) for all rows, so dQ/dK/dV are
            # unchanged. The window is in block_B units; we iterate it in
            # ``block_N`` sub-tiles (``ratio`` per block) so the K/V/P/dS shared
            # buffers stay small enough to fit a full block_M=64 query tile at
            # large head dims (D=576) -- bigger M matches the forward's
            # tensor-core utilisation and K/V amortisation.
            jl = BlockRange[bb, bm, 0]
            jh = BlockRange[bb, bm, 1]
            for nn in T.Pipelined((jh - jl) * ratio, num_stages=num_stages):
                col0 = (jl * ratio + nn) * BN
                # single shared K/V head -> no head index
                T.copy(K[bb, col0 : col0 + BN, :], K_shared)
                T.copy(V[bb, col0 : col0 + BN, :], V_shared)

                # P = softmax prob = exp(raw*sm_scale - lse); masked -> 0
                T.clear(acc_s)
                T.gemm(
                    Q_shared,
                    K_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                for i, c in T.Parallel(BM, BN):
                    col = col0 + c
                    keep = (col >= bos[i]) and (col < eos[i])
                    acc_p[i, c] = T.if_then_else(
                        keep,
                        T.exp2(
                            (acc_s[i, c] * sm_scale - lse_f[i]) * 1.44269504
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
                for i, c in T.Parallel(BM, BN):
                    acc_dp[i, c] = (
                        acc_p[i, c] * (acc_dp[i, c] - delta_f[i]) * sm_scale
                    )
                T.copy(acc_dp, dS_shared)

                # dQ += dS @ K
                T.gemm(
                    dS_shared,
                    K_shared,
                    acc_dq,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                # dV += P^T @ dO ; dK += dS^T @ Q  (scattered to shared KV)
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
                for c, d in T.Parallel(BN, D):
                    T.atomic_add(dV[bb, col0 + c, d], acc_dv[c, d])
                    T.atomic_add(dK[bb, col0 + c, d], acc_dk[c, d])

            # write dQ straight from the accumulator fragment (no shared-memory
            # staging) -- one fewer [BM, D] bf16 buffer lets the shared budget
            # fit a larger query tile (bigger M -> better tensor-core use).
            for i, d in T.Parallel(BM, D):
                if bm * BM + i < seq_len:
                    dQ[bb, bm * BM + i, bh, d] = acc_dq[i, d]

    return main


def _fit_block_mn(D, block_B, cap_bytes=230000):
    """Pick (block_M, block_N) maximising the query tile then the key sub-tile.

    The backward holds Q/dO ``[block_M, D]`` + K/V ``[block_N, D]`` + P/dS
    ``[block_M, block_N]`` in bf16 shared memory (dQ writes straight from its
    accumulator). Prefer the largest ``block_M`` (<=64) -- big M matches the
    forward's tensor-core utilisation and K/V amortisation -- and, for that M,
    the largest ``block_N`` (dividing ``block_B``) that fits. At MLA D=576 a
    full block_M=64 needs block_N=32 to fit Blackwell's ~227 KB; small dims
    (D=64) keep block_N=block_B (single-tile fast path).
    """
    cands_n = [n for n in (block_B, 32, 16) if block_B % n == 0]
    cands_n = sorted(set(cands_n), reverse=True)
    for bm in (64, 48, 32, 16):
        for bn in cands_n:
            shared = 4 * (bm * D + bn * D + bm * bn)
            if shared <= cap_bytes:
                return bm, bn
    return 16, min(16, block_B)


def block_score_mqa_bwd_interface(
    q,
    k,
    v,
    o,
    do,
    lse,
    valid_range,
    sm_scale=None,
    block_B=64,
    block_M=None,
    block_N=None,
):
    """Backward interface for the MQA block-score attention.

    Args:
        q:           [B, S, H, D] bf16 forward query.
        k, v:        [B, S_kv, D] bf16 single shared key/value head.
        o:           [B, S, H, D] bf16 forward output.
        do:          [B, S, H, D] bf16 grad of output.
        lse:         [B, S, H] fp32 natural-log LSE from forward.
        valid_range: [B, S, 2] int32.
        sm_scale:    softmax scale; defaults to D**-0.5.
        block_B:     key block size.
        block_M:     query tile size on the GEMM M dim; ``None`` auto-fits the
                     largest tile that fits shared memory (:func:`_fit_block_mn`).
        block_N:     key sub-tile size (must divide ``block_B``); ``None``
                     auto-fits. Overrides are for tuning/testing the tiling; the
                     result is mathematically identical for any valid choice.

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

    # kernel reads whole block_B blocks; pad K/V (and dK/dV) so the last block
    # is addressable.
    pad = (block_B - s_kv % block_B) % block_B
    s_kv_pad = s_kv + pad
    if pad > 0:
        k = paddle.nn.functional.pad(k, [0, 0, 0, pad])
        v = paddle.nn.functional.pad(v, [0, 0, 0, pad])
    k = k.contiguous()
    v = v.contiguous()
    valid_range = valid_range.contiguous()

    delta = (o.astype("float32") * do.astype("float32")).sum(-1).contiguous()
    dk = paddle.zeros([b, s_kv_pad, d], dtype="float32")
    dv = paddle.zeros([b, s_kv_pad, d], dtype="float32")

    fit_m, fit_n = _fit_block_mn(d, block_B)
    if block_M is None:
        block_M = fit_m
    if block_N is None:
        block_N = fit_n
    num_kv_blocks = s_kv_pad // block_B

    # Per query-tile key-block window [jl, jh): jl skips leading blocks before
    # the tile's document start, jh caps at the tile's (causal) reach. Rows are
    # grouped into tiles of ``block_M``; padded rows get bos=+big / eos=0 so
    # they never widen a tile's window.
    num_bm = (s + block_M - 1) // block_M
    pad_rows = num_bm * block_M - s
    bos = valid_range[:, :, 0]
    eos = valid_range[:, :, 1]
    if pad_rows > 0:
        bos = paddle.nn.functional.pad(bos, [0, pad_rows], value=s_kv_pad)
        eos = paddle.nn.functional.pad(eos, [0, pad_rows], value=0)
    bos = bos.reshape([b, num_bm, block_M])
    eos = eos.reshape([b, num_bm, block_M])
    jl = (bos.min(-1) // block_B).clip(0, num_kv_blocks)
    jh = ((eos.max(-1) + block_B - 1) // block_B).clip(0, num_kv_blocks)
    jh = paddle.maximum(jh, jl)
    block_range = paddle.stack([jl, jh], axis=-1).astype("int32").contiguous()

    bwd = block_score_mqa_bwd(
        h,
        d,
        float(sm_scale),
        block_M=block_M,
        block_N=block_N,
        block_B=block_B,
    )
    dq = bwd(q, k, v, do, lse, delta, valid_range, block_range, dk, dv)

    cast = _cast_bf16_kv(d)
    dk_bf = cast(dk)
    dv_bf = cast(dv)
    if pad > 0:
        dk_bf = dk_bf[:, :s_kv, :].contiguous()
        dv_bf = dv_bf[:, :s_kv, :].contiguous()
    return dq, dk_bf, dv_bf
