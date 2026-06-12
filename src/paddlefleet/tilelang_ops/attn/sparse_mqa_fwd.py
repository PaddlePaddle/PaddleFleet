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

# Refer to https://github.com/radixark/miles/pull/1045/

import paddle
import tilelang
from tilelang import language as T


@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def sparse_mqa_fwd(
    heads,
    dim,
    topk,
    sm_scale=None,
    block_I=64,
    num_stages=2,
    threads=256,
    D_chunk=128,
):
    assert dim == tilelang.math.next_power_of_2(dim), (
        f"dim must be power of 2, got {dim}"
    )
    assert topk % block_I == 0, (
        f"topk ({topk}) must be divisible by block_I ({block_I})"
    )
    if sm_scale is None:
        sm_scale = (1.0 / dim) ** 0.5 * 1.44269504
    else:
        sm_scale = sm_scale * 1.44269504

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len_kv, dim]
    o_shape = [batch, seq_len, heads, dim]
    indices_shape = [batch, seq_len, topk]
    lse_shape = [batch, seq_len, heads]
    attn_sink_shape = [heads]
    indices_dtype = T.int32
    dtype = T.bfloat16
    score_dtype = T.tfloat32
    accum_dtype = T.float32

    H = heads
    padded_H = max(tilelang.math.next_power_of_2(heads), 16)
    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    D = dim
    DC = D_chunk
    assert D % DC == 0, f"dim ({D}) must be divisible by D_chunk ({DC})"
    N_DC = D // DC

    if heads > 64:
        assert heads % 64 == 0, "heads should be a multiple of 64"
        REPLICATE_H = heads // 64
    else:
        REPLICATE_H = 1

    H_per_block = padded_H if REPLICATE_H == 1 else 64

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        KV: T.Tensor(kv_shape, dtype),
        AttnSink: T.Tensor(attn_sink_shape, accum_dtype),
        Indices: T.Tensor(indices_shape, indices_dtype),
        Output: T.Tensor(o_shape, dtype),
        Lse: T.Tensor(lse_shape, accum_dtype),
    ):
        with T.Kernel(seq_len * REPLICATE_H, batch, threads=threads) as (
            bx,
            by,
        ):
            Q_shared = T.alloc_shared([H_per_block, DC], score_dtype)
            KV_shared = T.alloc_shared([BI, DC], score_dtype)
            KV_shared_bf16 = T.alloc_shared([BI, DC], dtype)
            S_shared = T.alloc_shared([H_per_block, BI], dtype)
            O_shared = T.alloc_shared([H_per_block, DC], dtype)
            mask = T.alloc_fragment([BI], "bool")

            # Per-chunk output accumulators (GEMM-compatible layout)
            acc_o_0 = T.alloc_fragment([H_per_block, DC], accum_dtype)
            acc_o_1 = T.alloc_fragment([H_per_block, DC], accum_dtype)
            acc_o_2 = T.alloc_fragment([H_per_block, DC], accum_dtype)
            acc_o_3 = T.alloc_fragment([H_per_block, DC], accum_dtype)
            acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
            sumexp = T.alloc_fragment([H_per_block], accum_dtype)
            sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
            alpha = T.alloc_fragment([H_per_block], accum_dtype)
            m_i = T.alloc_fragment([H_per_block], accum_dtype)
            m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)
            m_i_log2 = T.alloc_fragment([H_per_block], accum_dtype)

            T.fill(acc_o_0, 0)
            T.fill(acc_o_1, 0)
            T.fill(acc_o_2, 0)
            T.fill(acc_o_3, 0)
            T.fill(sumexp, 0)
            T.fill(m_i, -(2**30))

            b_i = by
            s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)

            H0 = 0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64
            H1 = H0 + H_per_block

            for i_i in T.Serial(NI):
                for bi_i in T.Parallel(BI):
                    mask[bi_i] = Indices[b_i, s_i, i_i * BI + bi_i] != -1

                # Score GEMM: K-reduction across D-chunks (tf32)
                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.if_then_else(
                        mask[bi_i], 0, -T.infinity(acc_s.dtype)
                    )
                for d_c in range(N_DC):
                    T.copy(
                        Q[b_i, s_i, H0:H1, d_c * DC : (d_c + 1) * DC], Q_shared
                    )
                    for bi_i, d_i in T.Parallel(BI, DC):
                        KV_shared[bi_i, d_i] = KV[
                            b_i,
                            Indices[b_i, s_i, i_i * BI + bi_i],
                            d_c * DC + d_i,
                        ]
                    T.gemm(
                        Q_shared,
                        KV_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                # Online softmax
                T.copy(m_i, m_i_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for h_i in T.Parallel(H_per_block):
                    m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
                for h_i in T.Parallel(H_per_block):
                    alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.exp2(
                        acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale
                    )
                T.reduce_sum(acc_s, sumexp_i, dim=1)
                for h_i in T.Parallel(H_per_block):
                    sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                # Rescale per-chunk accumulators
                for h_i, d_i in T.Parallel(H_per_block, DC):
                    acc_o_0[h_i, d_i] = acc_o_0[h_i, d_i] * alpha[h_i]
                for h_i, d_i in T.Parallel(H_per_block, DC):
                    acc_o_1[h_i, d_i] = acc_o_1[h_i, d_i] * alpha[h_i]
                for h_i, d_i in T.Parallel(H_per_block, DC):
                    acc_o_2[h_i, d_i] = acc_o_2[h_i, d_i] * alpha[h_i]
                for h_i, d_i in T.Parallel(H_per_block, DC):
                    acc_o_3[h_i, d_i] = acc_o_3[h_i, d_i] * alpha[h_i]

                # Value GEMMs (unrolled per D-chunk)
                T.copy(acc_s, S_shared)
                for bi_i, d_i in T.Parallel(BI, DC):
                    KV_shared_bf16[bi_i, d_i] = KV[
                        b_i, Indices[b_i, s_i, i_i * BI + bi_i], 0 * DC + d_i
                    ]
                T.gemm(
                    S_shared,
                    KV_shared_bf16,
                    acc_o_0,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                for bi_i, d_i in T.Parallel(BI, DC):
                    KV_shared_bf16[bi_i, d_i] = KV[
                        b_i, Indices[b_i, s_i, i_i * BI + bi_i], 1 * DC + d_i
                    ]
                T.gemm(
                    S_shared,
                    KV_shared_bf16,
                    acc_o_1,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                for bi_i, d_i in T.Parallel(BI, DC):
                    KV_shared_bf16[bi_i, d_i] = KV[
                        b_i, Indices[b_i, s_i, i_i * BI + bi_i], 2 * DC + d_i
                    ]
                T.gemm(
                    S_shared,
                    KV_shared_bf16,
                    acc_o_2,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                for bi_i, d_i in T.Parallel(BI, DC):
                    KV_shared_bf16[bi_i, d_i] = KV[
                        b_i, Indices[b_i, s_i, i_i * BI + bi_i], 3 * DC + d_i
                    ]
                T.gemm(
                    S_shared,
                    KV_shared_bf16,
                    acc_o_3,
                    policy=T.GemmWarpPolicy.FullRow,
                )

            # AttnSink epilogue
            for h_i in T.Parallel(H_per_block):
                m_i_log2[h_i] = T.max(
                    m_i[h_i] * sm_scale,
                    AttnSink[H0 + h_i] * 1.44269504,
                )
            for h_i in T.Parallel(H_per_block):
                alpha[h_i] = T.exp2(m_i[h_i] * sm_scale - m_i_log2[h_i])
            for h_i in T.Parallel(H_per_block):
                sumexp[h_i] = sumexp[h_i] * alpha[h_i] + T.exp2(
                    AttnSink[H0 + h_i] * 1.44269504 - m_i_log2[h_i]
                )
            for h_i, d_i in T.Parallel(H_per_block, DC):
                acc_o_0[h_i, d_i] = acc_o_0[h_i, d_i] * alpha[h_i] / sumexp[h_i]
            for h_i, d_i in T.Parallel(H_per_block, DC):
                acc_o_1[h_i, d_i] = acc_o_1[h_i, d_i] * alpha[h_i] / sumexp[h_i]
            for h_i, d_i in T.Parallel(H_per_block, DC):
                acc_o_2[h_i, d_i] = acc_o_2[h_i, d_i] * alpha[h_i] / sumexp[h_i]
            for h_i, d_i in T.Parallel(H_per_block, DC):
                acc_o_3[h_i, d_i] = acc_o_3[h_i, d_i] * alpha[h_i] / sumexp[h_i]
            for h_i in T.Parallel(H_per_block):
                sumexp[h_i] = T.log2(sumexp[h_i]) + m_i_log2[h_i]

            # Write output per chunk
            T.copy(acc_o_0, O_shared)
            T.copy(O_shared, Output[b_i, s_i, H0:H1, 0 * DC : 1 * DC])
            T.copy(acc_o_1, O_shared)
            T.copy(O_shared, Output[b_i, s_i, H0:H1, 1 * DC : 2 * DC])
            T.copy(acc_o_2, O_shared)
            T.copy(O_shared, Output[b_i, s_i, H0:H1, 2 * DC : 3 * DC])
            T.copy(acc_o_3, O_shared)
            T.copy(O_shared, Output[b_i, s_i, H0:H1, 3 * DC : 4 * DC])
            T.copy(sumexp, Lse[b_i, s_i, H0:H1])

    return main


def sparse_mqa_fwd_interface(
    q,
    kv,
    attn_sink,
    topk_idxs,
    sm_scale=None,
    block_I=64,
    num_stages=2,
    threads=256,
    D_chunk=128,
):
    """Forward interface for DSv4 sparse MQA attention."""
    assert (
        q.is_contiguous() and kv.is_contiguous() and topk_idxs.is_contiguous()
    )
    batch, seq_len, heads, dim = q.shape
    _, _, topk = topk_idxs.shape
    _, _, kv_dim = kv.shape
    assert kv_dim == dim

    padded_topk = (topk + block_I - 1) // block_I * block_I
    if padded_topk != topk:
        pad = paddle.full(
            [batch, seq_len, padded_topk - topk], -1, dtype=topk_idxs.dtype
        )
        topk_idxs = paddle.concat([topk_idxs, pad], axis=-1).contiguous()
        topk = padded_topk

    kernel = sparse_mqa_fwd(
        heads,
        dim,
        topk,
        sm_scale,
        block_I=block_I,
        num_stages=num_stages,
        threads=threads,
        D_chunk=D_chunk,
    )
    out, lse = kernel(q, kv, attn_sink, topk_idxs)
    return out, lse
