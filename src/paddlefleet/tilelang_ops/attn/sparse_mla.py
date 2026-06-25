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

"""DSA TileLang sparse MLA kernels and interfaces (fwd/bwd/topk_reducesum)."""

import os

import paddle
from paddle import Tensor

try:
    paddle.enable_compat(scope={"tilelang"})
    import tilelang
    import tilelang.language as T

    HAS_TILELANG = True
except ImportError:
    HAS_TILELANG = False

from paddlefleet.tilelang_ops.indexer.dsa_indexer import (
    _cache_get_or_compile,
)

# ===========================================================================
# 1. Sparse MLA Forward
# ===========================================================================

if HAS_TILELANG:

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def _tl_sparse_mla_fwd(
        heads,
        dim,
        tail_dim,
        topk,
        kv_group=1,
        sm_scale=None,
        is_causal=True,
        block_I=32,
        num_stages=2,
        threads=128,
        dtype="bfloat16",
    ):
        assert dim == tilelang.math.next_power_of_2(dim)
        assert tail_dim == tilelang.math.next_power_of_2(tail_dim)
        assert is_causal
        assert topk % block_I == 0
        if sm_scale is None:
            sm_scale = (1.0 / (dim + tail_dim)) ** 0.5

        batch_plus_one = T.symbolic("batch_plus_one")
        seq_len = T.symbolic("seq_len")

        head_kv = heads // kv_group
        D = dim
        D_tail = tail_dim
        BI = block_I
        NI = tilelang.cdiv(topk, block_I)

        q_shape = [seq_len, heads, dim + tail_dim]
        kv_shape = [seq_len, kv_group, dim + tail_dim]
        o_shape = [seq_len, heads, dim]
        indices_shape = [seq_len, kv_group, topk]
        lse_shape = [seq_len, heads]
        offsets_shape = [batch_plus_one]
        token_indices_shape = [seq_len, 2]

        padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
        if padded_H != head_kv:
            assert kv_group == 1

        if head_kv > 64:
            assert head_kv % 64 == 0
            REPLICATE_H = head_kv // 64
        else:
            REPLICATE_H = 1
        H_per_block = padded_H if REPLICATE_H == 1 else 64

        @T.prim_func
        def main(
            Q: T.Tensor(q_shape, dtype),
            KV: T.Tensor(kv_shape, dtype),
            Indices: T.Tensor(indices_shape, "int32"),
            Offsets: T.Tensor(offsets_shape, "int32"),
            TokenIndices: T.Tensor(token_indices_shape, "int32"),
            Output: T.Tensor(o_shape, dtype),
            Lse: T.Tensor(lse_shape, "float"),
        ):
            with T.Kernel(seq_len * REPLICATE_H, kv_group, threads=threads) as (
                bx,
                by,
            ):
                Q_shared = T.alloc_shared([H_per_block, D], dtype)
                Q_tail_shared = T.alloc_shared([H_per_block, D_tail], dtype)
                KV_shared = T.alloc_shared([BI, D], dtype)
                K_tail_shared = T.alloc_shared([BI, D_tail], dtype)
                mask = T.alloc_fragment([BI], "bool")

                acc_o = T.alloc_fragment([H_per_block, D], "float")
                acc_s = T.alloc_fragment([H_per_block, BI], "float")
                S_shared = T.alloc_shared([H_per_block, BI], dtype)
                sumexp = T.alloc_fragment([H_per_block], "float")
                sumexp_i = T.alloc_fragment([H_per_block], "float")
                alpha = T.alloc_fragment([H_per_block], "float")
                m_i = T.alloc_fragment([H_per_block], "float")
                m_i_prev = T.alloc_fragment([H_per_block], "float")

                T.fill(acc_o, 0)
                T.fill(sumexp, 0)
                T.fill(m_i, -(2**30))

                b_s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)
                b_i, s_i = TokenIndices[b_s_i, 0], TokenIndices[b_s_i, 1]
                bos, eos = Offsets[b_i], Offsets[b_i + 1]
                g_i = by
                max_kv_i = s_i

                H0 = g_i * padded_H + (
                    0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64
                )
                H1 = H0 + H_per_block

                T.copy(Q[bos + s_i, H0:H1, :D], Q_shared)
                T.copy(Q[bos + s_i, H0:H1, D:], Q_tail_shared)

                for i_i in T.Pipelined(NI, num_stages=num_stages):
                    for bi_i in T.Parallel(BI):
                        mask[bi_i] = (
                            Indices[bos + s_i, g_i, i_i * BI + bi_i] <= max_kv_i
                        ) & (Indices[bos + s_i, g_i, i_i * BI + bi_i] != -1)

                    for bi_i, d_i in T.Parallel(BI, D):
                        idx = Indices[bos + s_i, g_i, i_i * BI + bi_i]
                        safe_idx = T.if_then_else(idx != -1, idx, 0)
                        KV_shared[bi_i, d_i] = KV[bos + safe_idx, g_i, d_i]
                    for bi_i, d_i in T.Parallel(BI, D_tail):
                        idx = Indices[bos + s_i, g_i, i_i * BI + bi_i]
                        safe_idx = T.if_then_else(idx != -1, idx, 0)
                        K_tail_shared[bi_i, d_i] = KV[
                            bos + safe_idx, g_i, D + d_i
                        ]

                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.if_then_else(
                            mask[bi_i], 0, -T.infinity(acc_s.dtype)
                        )
                    T.gemm(
                        Q_shared,
                        KV_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    T.gemm(
                        Q_tail_shared,
                        K_tail_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                    T.copy(m_i, m_i_prev)
                    T.reduce_max(acc_s, m_i, dim=1, clear=False)
                    for h_i in T.Parallel(H_per_block):
                        alpha[h_i] = T.exp(
                            (m_i_prev[h_i] - m_i[h_i]) * sm_scale
                        )
                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.exp(
                            acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale
                        )
                    T.reduce_sum(acc_s, sumexp_i, dim=1)
                    for h_i in T.Parallel(H_per_block):
                        sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                    for h_i, d_i in T.Parallel(H_per_block, D):
                        acc_o[h_i, d_i] = acc_o[h_i, d_i] * alpha[h_i]

                    T.copy(acc_s, S_shared)
                    T.gemm(
                        S_shared,
                        KV_shared,
                        acc_o,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                for h_i, d_i in T.Parallel(H_per_block, D):
                    acc_o[h_i, d_i] /= sumexp[h_i]
                for h_i in T.Parallel(H_per_block):
                    sumexp[h_i] = T.log(sumexp[h_i]) + m_i[h_i] * sm_scale

                T.copy(acc_o, Output[bos + s_i, H0:H1, :])
                T.copy(sumexp, Lse[bos + s_i, H0:H1])

        return main


def dsa_sparse_mla_fwd_interface(
    q: Tensor,
    kv: Tensor,
    indices: Tensor,
    offsets: Tensor,
    token_indices: Tensor,
    sm_scale: float,
    d_v: int,
) -> tuple[Tensor, Tensor]:
    """Sparse MLA forward (THD format).

    Returns:
        output: [S, heads, d_v] bfloat16
        lse: [S, heads] float32
    """
    if not HAS_TILELANG:
        raise RuntimeError(
            "tilelang is required for dsa_sparse_mla_fwd_interface"
        )

    seq_len, heads, dim_plus_tail_dim = q.shape
    dim = d_v
    tail_dim = dim_plus_tail_dim - dim
    if kv.ndim == 2:
        kv = kv.unsqueeze(1)
    _, kv_group, _ = kv.shape
    _, _, topk = indices.shape

    kernel_dtype = (
        "float" if os.getenv("DSA_TILELANG_FP32", "0") == "1" else "bfloat16"
    )

    def _compile():
        return _tl_sparse_mla_fwd(
            heads=heads,
            dim=dim,
            tail_dim=tail_dim,
            topk=topk,
            kv_group=kv_group,
            sm_scale=sm_scale,
            block_I=32,
            num_stages=2,
            threads=128,
            dtype=kernel_dtype,
        )

    key = (
        "sparse_mla_fwd",
        heads,
        dim,
        tail_dim,
        topk,
        kv_group,
        sm_scale,
        kernel_dtype,
    )
    kernel = _cache_get_or_compile("sparse_mla_fwd", key, _compile)

    q_kernel = q.cast("float32") if kernel_dtype == "float" else q
    kv_kernel = kv.cast("float32") if kernel_dtype == "float" else kv

    out, lse = kernel(q_kernel, kv_kernel, indices, offsets, token_indices)
    return out, lse


# ===========================================================================
# 2. Sparse MLA Backward
# ===========================================================================

if HAS_TILELANG:

    @tilelang.jit(out_idx=[-1])
    def _tl_preprocess(
        H, D, block_ND=32, num_stages=5, dtype="bfloat16", accum_dtype="float"
    ):
        S = T.symbolic("S")
        shape = [S, H, D]

        @T.prim_func
        def preprocess_kernel(
            O: T.Tensor(shape, dtype),
            dO: T.Tensor(shape, dtype),
            Delta: T.Tensor([S, H], accum_dtype),
        ):
            with T.Kernel(H, T.ceildiv(S, block_ND)) as (bx, by):
                o = T.alloc_fragment([block_ND, block_ND], accum_dtype)
                do = T.alloc_fragment([block_ND, block_ND], accum_dtype)
                delta = T.alloc_fragment([block_ND], accum_dtype)
                acc = T.alloc_fragment([block_ND, block_ND], accum_dtype)
                T.clear(acc)
                for k in T.Pipelined(
                    T.ceildiv(D, block_ND), num_stages=num_stages
                ):
                    T.copy(
                        O[
                            by * block_ND : (by + 1) * block_ND,
                            bx,
                            k * block_ND : (k + 1) * block_ND,
                        ],
                        o,
                    )
                    T.copy(
                        dO[
                            by * block_ND : (by + 1) * block_ND,
                            bx,
                            k * block_ND : (k + 1) * block_ND,
                        ],
                        do,
                    )
                    for i, j in T.Parallel(block_ND, block_ND):
                        acc[i, j] += o[i, j] * do[i, j]
                T.reduce_sum(acc, delta, 1)
                T.copy(delta, Delta[by * block_ND : (by + 1) * block_ND, bx])

        return preprocess_kernel

    @tilelang.jit(out_idx=[-1])
    def _tl_postprocess(
        D,
        D_tail,
        kv_group=1,
        block_N=64,
        threads=128,
        dtype="bfloat16",
        accum_dtype="float",
    ):
        S_kv = T.symbolic("S_kv")
        dkv_shape = [S_kv, kv_group, D + D_tail]

        @T.prim_func
        def postprocess_kernel(
            dKV: T.Tensor(dkv_shape, accum_dtype),
            dKV_out: T.Tensor(dkv_shape, dtype),
        ):
            with T.Kernel(
                T.ceildiv(S_kv, block_N), kv_group, threads=threads
            ) as (bx, by):
                T.copy(
                    dKV[bx * block_N : (bx + 1) * block_N, by, :],
                    dKV_out[bx * block_N : (bx + 1) * block_N, by, :],
                )

        return postprocess_kernel

    @tilelang.jit(
        out_idx=[-2],
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: False,
        },
    )
    def _tl_bwd(
        H,
        D,
        D_tail,
        topk,
        kv_group=1,
        sm_scale=None,
        is_causal=True,
        block_size=32,
        num_stages=0,
        threads=256,
        dtype="bfloat16",
    ):
        assert kv_group == 1
        assert topk % block_size == 0
        assert is_causal

        if sm_scale is None:
            sm_scale = (D + D_tail) ** (-0.5)

        B_plus_one = T.symbolic("B_plus_one")
        S = T.symbolic("S")

        if H > 64:
            assert H % 64 == 0
            kv_group_view = H // 64
        else:
            kv_group_view = 1
        H_kv = H // kv_group_view

        q_shape = [S, H, D + D_tail]
        k_shape = [S, kv_group, D + D_tail]
        o_shape = [S, H, D]
        indices_shape = [S, kv_group, topk]
        delta_shape = [S, H]
        lse_shape = [S, H]
        offsets_shape = [B_plus_one]
        token_indices_shape = [S, 2]

        padded_H = max(tilelang.math.next_power_of_2(H_kv), 16)
        BS = block_size
        NS = tilelang.cdiv(topk, block_size)
        split_store = 2

        @T.prim_func
        def sparse_mla_bwd_kernel(
            Q: T.Tensor(q_shape, dtype),
            KV: T.Tensor(k_shape, dtype),
            dO: T.Tensor(o_shape, dtype),
            Indices: T.Tensor(indices_shape, "int32"),
            Lse: T.Tensor(lse_shape, "float"),
            Delta: T.Tensor(delta_shape, "float"),
            Offsets: T.Tensor(offsets_shape, "int32"),
            TokenIndices: T.Tensor(token_indices_shape, "int32"),
            dQ: T.Tensor(q_shape, dtype),
            dKV: T.Tensor(k_shape, "float"),
        ):
            with T.Kernel(S, kv_group_view, threads=threads) as (b_s_i, bz):
                Q_shared = T.alloc_shared([padded_H, D], dtype)
                Q_tail_shared = T.alloc_shared([padded_H, D_tail], dtype)
                KV_shared = T.alloc_shared([BS, D], dtype)
                KV_tail_shared = T.alloc_shared([BS, D_tail], dtype)
                dO_shared = T.alloc_shared([padded_H, D], dtype)
                mask = T.alloc_fragment([BS], "bool")

                P_shared_cast = T.alloc_shared([padded_H, BS], dtype)
                dP_shared_cast = T.alloc_shared([padded_H, BS], dtype)
                dQ_shared = T.alloc_shared([padded_H, D], dtype)
                dQ_tail_shared = T.alloc_shared([padded_H, D_tail], dtype)

                acc_p = T.alloc_fragment([padded_H, BS], "float")
                acc_dp = T.alloc_fragment([padded_H, BS], "float")
                acc_dq = T.alloc_fragment([padded_H, D], "float")
                acc_dq_tail = T.alloc_fragment([padded_H, D_tail], "float")
                acc_dkv = T.alloc_fragment([BS, D], "float")
                acc_dkv_tail = T.alloc_fragment([BS, D_tail], "float")
                acc_dkv_shared = T.alloc_shared([BS // split_store, D], "float")
                acc_dkv_tail_shared = T.alloc_shared(
                    [BS // split_store, D_tail], "float"
                )

                b_i, s_i = TokenIndices[b_s_i, 0], TokenIndices[b_s_i, 1]
                bos = Offsets[b_i]
                max_kv_i = s_i

                T.copy(
                    Q[bos + s_i, bz * padded_H : (bz + 1) * padded_H, :D],
                    Q_shared,
                )
                T.copy(
                    Q[bos + s_i, bz * padded_H : (bz + 1) * padded_H, D:],
                    Q_tail_shared,
                )
                T.copy(
                    dO[bos + s_i, bz * padded_H : (bz + 1) * padded_H, :D],
                    dO_shared,
                )

                T.clear(acc_dq)
                T.clear(acc_dq_tail)

                for i_i in T.Pipelined(NS, num_stages=num_stages):
                    for bi_i in T.Parallel(BS):
                        idx = Indices[bos + s_i, 0, i_i * BS + bi_i]
                        mask[bi_i] = (idx <= max_kv_i) & (idx != -1)

                    for h_i, bi_i in T.Parallel(padded_H, BS):
                        acc_p[h_i, bi_i] = T.if_then_else(
                            mask[bi_i], 0, -T.infinity(acc_p.dtype)
                        )

                    for bi_i, d_i in T.Parallel(BS, D):
                        idx = Indices[bos + s_i, 0, i_i * BS + bi_i]
                        safe_idx = T.if_then_else(idx != -1, idx, 0)
                        KV_shared[bi_i, d_i] = KV[bos + safe_idx, 0, d_i]
                    T.gemm(
                        Q_shared,
                        KV_shared,
                        acc_p,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )

                    for bi_i, d_i in T.Parallel(BS, D_tail):
                        idx = Indices[bos + s_i, 0, i_i * BS + bi_i]
                        safe_idx = T.if_then_else(idx != -1, idx, 0)
                        KV_tail_shared[bi_i, d_i] = KV[
                            bos + safe_idx, 0, D + d_i
                        ]
                    T.gemm(
                        Q_tail_shared,
                        KV_tail_shared,
                        acc_p,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )

                    for h_i, bi_i in T.Parallel(padded_H, BS):
                        acc_p[h_i, bi_i] = T.exp(
                            acc_p[h_i, bi_i] * sm_scale
                            - Lse[bos + s_i, bz * padded_H + h_i]
                        )

                    T.copy(acc_p, P_shared_cast)
                    T.gemm(
                        dO_shared,
                        KV_shared,
                        acc_dp,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullCol,
                        clear_accum=True,
                    )

                    for h_i, bi_i in T.Parallel(padded_H, BS):
                        acc_dp[h_i, bi_i] = (
                            acc_p[h_i, bi_i]
                            * (
                                acc_dp[h_i, bi_i]
                                - Delta[bos + s_i, bz * padded_H + h_i]
                            )
                            * sm_scale
                        )

                    T.copy(acc_dp, dP_shared_cast)
                    T.gemm(
                        dP_shared_cast,
                        KV_shared,
                        acc_dq,
                        policy=T.GemmWarpPolicy.FullCol,
                    )
                    T.gemm(
                        dP_shared_cast,
                        KV_tail_shared,
                        acc_dq_tail,
                        policy=T.GemmWarpPolicy.FullCol,
                    )

                    T.gemm(
                        dP_shared_cast,
                        Q_shared,
                        acc_dkv,
                        transpose_A=True,
                        policy=T.GemmWarpPolicy.FullCol,
                        clear_accum=True,
                    )
                    T.gemm(
                        P_shared_cast,
                        dO_shared,
                        acc_dkv,
                        transpose_A=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )

                    T.clear(acc_dkv_tail)
                    T.gemm(
                        dP_shared_cast,
                        Q_tail_shared,
                        acc_dkv_tail,
                        transpose_A=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )

                    for s in range(split_store):
                        for bi_i, d_i in T.Parallel(BS, D):
                            if bi_i < BS // split_store:
                                acc_dkv_shared[bi_i, d_i] = acc_dkv[
                                    bi_i + s * (BS // split_store), d_i
                                ]
                        for bi_i, d_i in T.Parallel(BS, D_tail):
                            if bi_i < BS // split_store:
                                acc_dkv_tail_shared[bi_i, d_i] = acc_dkv_tail[
                                    bi_i + s * (BS // split_store), d_i
                                ]
                        for bi_i, d_i in T.Parallel(BS // split_store, D):
                            idx = Indices[
                                bos + s_i,
                                0,
                                i_i * BS + bi_i + s * (BS // split_store),
                            ]
                            safe_idx = T.if_then_else(idx != -1, idx, 0)
                            if idx != -1:
                                T.atomic_add(
                                    dKV[bos + safe_idx, 0, d_i],
                                    acc_dkv_shared[bi_i, d_i],
                                )
                        for bi_i, d_i in T.Parallel(BS // split_store, D_tail):
                            idx = Indices[
                                bos + s_i,
                                0,
                                i_i * BS + bi_i + s * (BS // split_store),
                            ]
                            safe_idx = T.if_then_else(idx != -1, idx, 0)
                            if idx != -1:
                                T.atomic_add(
                                    dKV[bos + safe_idx, 0, D + d_i],
                                    acc_dkv_tail_shared[bi_i, d_i],
                                )

                T.copy(acc_dq, dQ_shared)
                T.copy(acc_dq_tail, dQ_tail_shared)
                T.copy(
                    dQ_shared,
                    dQ[bos + s_i, bz * padded_H : (bz + 1) * padded_H, :D],
                )
                T.copy(
                    dQ_tail_shared,
                    dQ[bos + s_i, bz * padded_H : (bz + 1) * padded_H, D:],
                )

        return sparse_mla_bwd_kernel


def dsa_sparse_mla_bwd_interface(
    q: Tensor,
    kv: Tensor,
    o: Tensor,
    do: Tensor,
    indices: Tensor,
    lse: Tensor,
    offsets: Tensor,
    token_indices: Tensor,
    sm_scale: float,
    d_v: int,
) -> tuple[Tensor, Tensor]:
    """Sparse MLA backward. THD format.

    Returns:
        dq: [S, heads, dim+tail_dim] bfloat16
        dkv: [S, kv_group, dim+tail_dim] bfloat16
    """
    if not HAS_TILELANG:
        raise RuntimeError(
            "tilelang is required for dsa_sparse_mla_bwd_interface"
        )

    S, H, dim_plus_tail_dim = q.shape
    D = d_v
    D_tail = dim_plus_tail_dim - D
    if kv.ndim == 2:
        kv = kv.unsqueeze(1)
    _, kv_group, _ = kv.shape
    topk = indices.shape[-1]

    kernel_dtype = (
        "float" if os.getenv("DSA_TILELANG_FP32", "0") == "1" else "bfloat16"
    )

    def _compile_pre():
        return _tl_preprocess(H, D, dtype=kernel_dtype)

    def _compile_bwd():
        threads = 256 if H > 16 else 128
        return _tl_bwd(
            H,
            D,
            D_tail,
            topk,
            kv_group,
            sm_scale,
            threads=threads,
            dtype=kernel_dtype,
        )

    def _compile_post():
        return _tl_postprocess(D, D_tail, kv_group, dtype=kernel_dtype)

    preprocess_kernel = _cache_get_or_compile(
        "sparse_mla_pre", ("pre", H, D, kernel_dtype), _compile_pre
    )
    bwd_kernel = _cache_get_or_compile(
        "sparse_mla_bwd",
        ("bwd", H, D, D_tail, topk, kv_group, sm_scale, kernel_dtype),
        _compile_bwd,
    )
    postprocess_kernel = _cache_get_or_compile(
        "sparse_mla_post",
        ("post", D, D_tail, kv_group, kernel_dtype),
        _compile_post,
    )

    q_kernel = q.cast("float32") if kernel_dtype == "float" else q
    kv_kernel = kv.cast("float32") if kernel_dtype == "float" else kv
    o_kernel = o.cast("float32") if kernel_dtype == "float" else o
    do_kernel = do.cast("float32") if kernel_dtype == "float" else do
    delta = preprocess_kernel(o_kernel, do_kernel)
    dkv = paddle.zeros_like(kv, dtype="float32")
    dq = bwd_kernel(
        q_kernel,
        kv_kernel,
        do_kernel,
        indices,
        lse,
        delta,
        offsets,
        token_indices,
        dkv,
    )
    dkv = postprocess_kernel(dkv)
    return dq, dkv


# ===========================================================================
# 3. Sparse MLA TopK ReduceSum (for indexer backward loss)
# ===========================================================================

if HAS_TILELANG:

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        }
    )
    def _tl_sparse_mla_topk_reducesum(
        heads,
        dim,
        tail_dim,
        topk,
        kv_group=1,
        sm_scale=None,
        block_I=32,
        num_stages=2,
        threads=128,
    ):
        assert dim == tilelang.math.next_power_of_2(dim)
        assert tail_dim == tilelang.math.next_power_of_2(tail_dim)
        assert topk % block_I == 0
        if sm_scale is None:
            sm_scale = (1.0 / (dim + tail_dim)) ** 0.5

        batch_plus_one = T.symbolic("batch_plus_one")
        seq_len = T.symbolic("seq_len")
        seq_len_kv = T.symbolic("seq_len_kv")

        head_kv = heads // kv_group
        D = dim
        D_tail = tail_dim
        BI = block_I
        NI = tilelang.cdiv(topk, block_I)

        padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
        if padded_H != head_kv:
            assert kv_group == 1

        if head_kv > 64:
            assert head_kv % 64 == 0
            REPLICATE_H = head_kv // 64
        else:
            REPLICATE_H = 1
        H_per_block = padded_H if REPLICATE_H == 1 else 64

        q_shape = [seq_len, heads, dim + tail_dim]
        kv_shape = [seq_len_kv, kv_group, dim + tail_dim]
        indices_shape = [seq_len, kv_group, topk]
        lse_shape = [seq_len, heads]
        reducesum_shape = [seq_len, kv_group, REPLICATE_H, topk]
        offsets_shape = [batch_plus_one]
        token_indices_shape = [seq_len, 2]

        @T.prim_func
        def tl_sparse_mla_topk_reducesum_kernel(
            Q: T.Tensor(q_shape, "bfloat16"),
            KV: T.Tensor(kv_shape, "bfloat16"),
            Indices: T.Tensor(indices_shape, "int32"),
            Lse: T.Tensor(lse_shape, "float"),
            Offsets: T.Tensor(offsets_shape, "int32"),
            TokenIndices: T.Tensor(token_indices_shape, "int32"),
            ReduceSum: T.Tensor(reducesum_shape, "float"),
        ):
            with T.Kernel(seq_len * REPLICATE_H, kv_group, threads=threads) as (
                bx,
                by,
            ):
                Q_shared = T.alloc_shared([H_per_block, D], "bfloat16")
                Q_tail_shared = T.alloc_shared(
                    [H_per_block, D_tail], "bfloat16"
                )
                KV_shared = T.alloc_shared([BI, D], "bfloat16")
                K_tail_shared = T.alloc_shared([BI, D_tail], "bfloat16")
                mask = T.alloc_fragment([BI], "bool")

                acc_s = T.alloc_fragment([H_per_block, BI], "float")
                reducesum = T.alloc_fragment([BI], "float")
                lse = T.alloc_fragment([H_per_block], "float")
                T.fill(lse, 0)

                b_s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)
                b_i, s_i = TokenIndices[b_s_i, 0], TokenIndices[b_s_i, 1]
                bos, eos = Offsets[b_i], Offsets[b_i + 1]
                r_i = bx % REPLICATE_H
                g_i = by
                max_kv_i = s_i

                H0 = g_i * padded_H + (
                    0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64
                )
                H1 = H0 + H_per_block

                T.copy(Q[bos + s_i, H0:H1, :D], Q_shared)
                T.copy(Q[bos + s_i, H0:H1, D:], Q_tail_shared)
                T.copy(Lse[bos + s_i, H0:H1], lse)

                for i_i in T.Pipelined(NI, num_stages=num_stages):
                    for bi_i in T.Parallel(BI):
                        mask[bi_i] = (
                            Indices[bos + s_i, g_i, i_i * BI + bi_i] <= max_kv_i
                        ) & (Indices[bos + s_i, g_i, i_i * BI + bi_i] != -1)

                    for bi_i, d_i in T.Parallel(BI, D):
                        idx = Indices[bos + s_i, g_i, i_i * BI + bi_i]
                        safe_idx = T.if_then_else(idx != -1, idx, 0)
                        KV_shared[bi_i, d_i] = KV[bos + safe_idx, g_i, d_i]
                    for bi_i, d_i in T.Parallel(BI, D_tail):
                        idx = Indices[bos + s_i, g_i, i_i * BI + bi_i]
                        safe_idx = T.if_then_else(idx != -1, idx, 0)
                        K_tail_shared[bi_i, d_i] = KV[
                            bos + safe_idx, g_i, D + d_i
                        ]

                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.if_then_else(
                            mask[bi_i], 0, -T.infinity(acc_s.dtype)
                        )
                    T.gemm(
                        Q_shared,
                        KV_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    T.gemm(
                        Q_tail_shared,
                        K_tail_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.exp(
                            acc_s[h_i, bi_i] * sm_scale - lse[h_i]
                        )
                    T.reduce_sum(acc_s, reducesum, dim=0)
                    T.copy(
                        reducesum,
                        ReduceSum[
                            bos + s_i, g_i, r_i, i_i * BI : i_i * BI + BI
                        ],
                    )

        return tl_sparse_mla_topk_reducesum_kernel


def dsa_sparse_mla_topk_reducesum_interface(
    q: Tensor,
    kv: Tensor,
    topk_indices: Tensor,
    lse: Tensor,
    offsets: Tensor,
    token_indices: Tensor,
    dim_v: int,
    sm_scale: float,
) -> Tensor:
    """Sparse MLA topk reducesum. THD format.

    Returns:
        attn_score: [S, kv_group, topk] float32
    """
    if not HAS_TILELANG:
        raise RuntimeError(
            "tilelang is required for dsa_sparse_mla_topk_reducesum_interface"
        )

    seq_len, heads, dim_plus_tail_dim = q.shape
    tail_dim = dim_plus_tail_dim - dim_v
    topk = topk_indices.shape[-1]
    kv_group = kv.shape[1] if kv.ndim == 3 else 1
    REPLICATE_H = max(heads // 64, 1)

    def _compile():
        return _tl_sparse_mla_topk_reducesum(
            heads=heads,
            dim=dim_v,
            tail_dim=tail_dim,
            topk=topk,
            kv_group=kv_group,
            sm_scale=sm_scale,
        )

    key = ("topk_reducesum", heads, dim_v, tail_dim, topk, kv_group, sm_scale)
    kernel = _cache_get_or_compile("sparse_mla_topk_rs", key, _compile)

    reducesum = paddle.zeros(
        [seq_len, kv_group, REPLICATE_H, topk], dtype="float32"
    )
    kernel(q, kv, topk_indices, lse, offsets, token_indices, reducesum)
    reducesum = reducesum.sum(axis=2)  # [S, kv_group, topk]
    attn_score = reducesum / reducesum.sum(axis=-1, keepdim=True).clip(
        min=1e-10
    )
    return attn_score
