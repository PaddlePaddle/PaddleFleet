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

"""Block-score attention (paper "Algorithm 1") with **per-head** K/V (MHA):
the decompressed MLA layout where each of the ``H`` query heads has its own
Key/Value head. Emitted per-block max logits feed the same block-level TopK
selection as the MQA variant.

This is the sibling of :mod:`block_score_attn` (MQA/absorbed MLA, single shared
K/V head). The two exist to measure the wall-clock cost of the ``H``x KV-access
factor: MHA saves attention FLOPs (smaller head dim, no RoPE slice) but reads
``H`` independent K/V heads instead of one, so in the memory-bound regime it can
be *slower* despite the FLOP saving.

Q is ``[B, S, H, D]``; K is ``[B, S_kv, H, D]``; V is ``[B, S_kv, H, D_v]`` --
one K/V head per query head. Masking (causal + document) is expressed through
``valid_range`` ``[B, S, 2]`` giving each query's half-open valid key column
range ``[bos, eos)``.

Efficient structure (vs the MQA per-token / heads-on-M kernel): one program per
``(query-tile, head, batch)`` with ``block_M`` query tokens on the GEMM ``M``
dim sharing that head's K -- the standard dense-attention tiling that fills the
tensor cores. The two layouts are each mode's natural best structure.

Block coordinates are **absolute** (``col // block_B``), not document-relative:
tiling ``block_M`` query rows together makes per-row document-relative binning
(different ``bos`` per row within a tile) awkward to scatter. For the causal
single-document case used in the accuracy tests (``bos == 0``)
absolute == relative, so the emitted block scores are directly comparable to the
MQA kernel's. Document-relative binning is out of scope here (it does not affect
the FLOP/memory comparison, since binning is only a cheap ``reduce_max``).

The forward returns:
* ``Output``    ``[B, S, H, D_v]`` attention output.
* ``Lse``       ``[B, S, H]`` natural log-sum-exp of the *scaled* logits.
* ``BlockLogit`` ``[B, H, S, num_blocks]`` per-(query, key-block) max of the
  *raw* (unscaled) ``q·k`` logit over valid columns; fully-masked blocks store
  ``-inf``. Recover the eq.(3) probability score on the host with
  :func:`~block_score_attn.block_scores_from_logit`.
"""

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
def block_score_mha_fwd(
    H,
    D,
    sm_scale,
    D_v=None,
    block_B=64,
    block_M=64,
    num_stages=2,
    threads=128,
):
    """Per-head (MHA) block-score kernel.

    One program per ``(query-tile, head, batch)``: ``block_M`` query tokens on
    the GEMM ``M`` dimension, sharing this head's K/V. ``D`` is the query/key
    head dim (``q·k`` logit); ``D_v`` is the value/output head dim (defaults to
    ``D``). For decompressed MLA ``D == D_v`` (e.g. 256).
    """
    if D_v is None:
        D_v = D
    assert D % 16 == 0, (
        f"D must be a multiple of 16 (tensor-core k-tile), got {D}"
    )
    assert D_v % 16 == 0, (
        f"D_v must be a multiple of 16 (tensor-core k-tile), got {D_v}"
    )
    scale_log2 = sm_scale * 1.44269504  # log2(e), online softmax in base 2

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")
    num_blocks = T.dynamic("num_blocks")
    num_bm = T.dynamic("num_bm")

    q_shape = [batch, seq_len, H, D]
    k_shape = [batch, seq_len_kv, H, D]
    v_shape = [batch, seq_len_kv, H, D_v]
    o_shape = [batch, seq_len, H, D_v]
    lse_shape = [batch, seq_len, H]
    vr_shape = [batch, seq_len, 2]
    br_shape = [batch, num_bm, 2]
    blk_shape = [batch, H, seq_len, num_blocks]

    dtype = T.bfloat16
    accum_dtype = T.float32
    idx_dtype = T.int32
    BM = block_M
    BB = block_B

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        V: T.Tensor(v_shape, dtype),
        ValidRange: T.Tensor(vr_shape, idx_dtype),
        BlockRange: T.Tensor(br_shape, idx_dtype),
        BlockLogit: T.Tensor(blk_shape, accum_dtype),
        Output: T.Tensor(o_shape, dtype),
        Lse: T.Tensor(lse_shape, accum_dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, BM), H, batch, threads=threads) as (
            bm,
            bh,
            bb,
        ):
            Q_shared = T.alloc_shared([BM, D], dtype)
            K_shared = T.alloc_shared([BB, D], dtype)
            V_shared = T.alloc_shared([BB, D_v], dtype)
            P_shared = T.alloc_shared([BM, BB], dtype)

            acc_o = T.alloc_fragment([BM, D_v], accum_dtype)
            acc_s = T.alloc_fragment([BM, BB], accum_dtype)
            blk_max = T.alloc_fragment([BM], accum_dtype)
            m_i = T.alloc_fragment([BM], accum_dtype)
            m_prev = T.alloc_fragment([BM], accum_dtype)
            l_i = T.alloc_fragment([BM], accum_dtype)
            l_new = T.alloc_fragment([BM], accum_dtype)
            alpha = T.alloc_fragment([BM], accum_dtype)
            bos = T.alloc_fragment([BM], idx_dtype)
            eos = T.alloc_fragment([BM], idx_dtype)

            T.fill(acc_o, 0)
            T.fill(m_i, -(2**30))
            T.fill(l_i, 0)

            for i in T.Parallel(BM):
                row = bm * BM + i
                in_range = row < seq_len
                bos[i] = T.if_then_else(in_range, ValidRange[bb, row, 0], 0)
                eos[i] = T.if_then_else(in_range, ValidRange[bb, row, 1], 0)

            T.copy(Q[bb, bm * BM : (bm + 1) * BM, bh, :], Q_shared)

            # causal/document early-exit: the host precomputes, per query tile,
            # the block-B key window [jl, jh) reachable by this tile's rows --
            # jl = min_i(bos_i)//block_B (skip leading blocks before the tile's
            # document start); jh = ceil(max_i(eos_i)/block_B) (skip blocks past
            # the causal end). Blocks outside are fully masked for every row and
            # contribute nothing to out/lse; their per-block max logit stays the
            # host-prefilled -inf.
            jl = BlockRange[bb, bm, 0]
            jh = BlockRange[bb, bm, 1]
            for j in T.Pipelined(jh - jl, num_stages=num_stages):
                jb = jl + j  # absolute block index
                col0 = jb * BB
                for c, d in T.Parallel(BB, D):
                    col = col0 + c
                    in_bounds = col < seq_len_kv
                    safe_col = T.if_then_else(in_bounds, col, 0)
                    K_shared[c, d] = T.if_then_else(
                        in_bounds, K[bb, safe_col, bh, d], T.cast(0, dtype)
                    )
                for c, d in T.Parallel(BB, D_v):
                    col = col0 + c
                    in_bounds = col < seq_len_kv
                    safe_col = T.if_then_else(in_bounds, col, 0)
                    V_shared[c, d] = T.if_then_else(
                        in_bounds, V[bb, safe_col, bh, d], T.cast(0, dtype)
                    )

                # raw q·k^T
                T.clear(acc_s)
                T.gemm(
                    Q_shared,
                    K_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )

                # per-row causal + document mask: col in [bos_i, eos_i)
                for i, c in T.Parallel(BM, BB):
                    col = col0 + c
                    keep = (col >= bos[i]) and (col < eos[i])
                    acc_s[i, c] = T.if_then_else(
                        keep, acc_s[i, c], -T.infinity(accum_dtype)
                    )

                # per-block max of raw logit over valid cols -> block score src
                # (absolute block coordinate jb)
                T.reduce_max(acc_s, blk_max, dim=1, clear=True)
                for i in T.Parallel(BM):
                    row = bm * BM + i
                    if row < seq_len:
                        BlockLogit[bb, bh, row, jb] = blk_max[i]

                # online softmax (base 2) over scaled logits
                T.copy(m_i, m_prev)
                for i in T.Parallel(BM):
                    m_i[i] = T.max(m_i[i], blk_max[i] * sm_scale)
                for i in T.Parallel(BM):
                    alpha[i] = T.exp2((m_prev[i] - m_i[i]) * 1.44269504)
                for i, c in T.Parallel(BM, BB):
                    acc_s[i, c] = T.exp2(
                        acc_s[i, c] * scale_log2 - m_i[i] * 1.44269504
                    )
                T.reduce_sum(acc_s, l_new, dim=1)
                for i in T.Parallel(BM):
                    l_i[i] = l_i[i] * alpha[i] + l_new[i]
                for i, d in T.Parallel(BM, D_v):
                    acc_o[i, d] = acc_o[i, d] * alpha[i]
                T.copy(acc_s, P_shared)
                T.gemm(
                    P_shared, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow
                )

            # normalize; empty rows (no valid key) -> 0 out / -inf lse
            for i, d in T.Parallel(BM, D_v):
                acc_o[i, d] = T.if_then_else(
                    l_i[i] > 0, acc_o[i, d] / l_i[i], 0.0
                )
            for i, d in T.Parallel(BM, D_v):
                row = bm * BM + i
                if row < seq_len:
                    Output[bb, row, bh, d] = acc_o[i, d]
            for i in T.Parallel(BM):
                row = bm * BM + i
                if row < seq_len:
                    Lse[bb, row, bh] = T.if_then_else(
                        l_i[i] > 0,
                        m_i[i] + T.log(l_i[i]),
                        -T.infinity(accum_dtype),
                    )

    return main


def block_score_mha_attn_fwd(q, k, v, valid_range, sm_scale=None, block_B=64):
    """Forward interface for the MHA (per-head K/V) block-score attention.

    Args:
        q:           [B, S, H, D] bf16 query (H heads).
        k:           [B, S_kv, H, D] bf16 key (H heads).
        v:           [B, S_kv, H, D_v] bf16 value (H heads).
        valid_range: [B, S, 2] int32 per-query [bos, eos) valid key columns.
        sm_scale:    softmax scale; defaults to D**-0.5.
        block_B:     key block size.

    Returns:
        out [B,S,H,D_v], lse [B,S,H], block_logit [B,H,S,num_blocks] where
        num_blocks = ceil(S_kv / block_B). Block coordinates are absolute.
    """
    assert q.is_contiguous()
    b, s, h, d = q.shape
    s_kv = k.shape[1]
    d_v = v.shape[-1]
    assert list(k.shape) == [b, s_kv, h, d], (
        f"k must be [B, S_kv, H, D] matching q; got k {k.shape}, q {q.shape}"
    )
    assert list(v.shape[:3]) == [b, s_kv, h], (
        f"v must be [B, S_kv, H, D_v] matching k; got {v.shape}, k {k.shape}"
    )
    assert list(valid_range.shape) == [b, s, 2], (
        f"valid_range must be [B, S, 2]; got {valid_range.shape}"
    )
    if sm_scale is None:
        sm_scale = d**-0.5
    if valid_range.dtype != paddle.int32:
        valid_range = valid_range.cast("int32")

    # kernel reads whole block_B blocks; pad K/V so the last block is in bounds
    pad = (block_B - s_kv % block_B) % block_B
    s_kv_pad = s_kv + pad
    if pad > 0:
        # pad along the sequence axis (axis 1) of [B, S_kv, H, D(_v)]
        k = paddle.nn.functional.pad(k, [0, 0, 0, 0, 0, pad])
        v = paddle.nn.functional.pad(v, [0, 0, 0, 0, 0, pad])
    k = k.contiguous()
    v = v.contiguous()
    valid_range = valid_range.contiguous()
    num_blocks = s_kv_pad // block_B

    # Per query-tile key-block window [jl, jh): jl skips leading blocks before
    # the tile's document start, jh caps at the tile's (causal) reach. Padded
    # rows get bos=+big / eos=0 so they never widen a tile's window.
    block_M = 64
    num_bm_v = (s + block_M - 1) // block_M
    pad_rows = num_bm_v * block_M - s
    bos = valid_range[:, :, 0]
    eos = valid_range[:, :, 1]
    if pad_rows > 0:
        bos = paddle.nn.functional.pad(bos, [0, pad_rows], value=s_kv_pad)
        eos = paddle.nn.functional.pad(eos, [0, pad_rows], value=0)
    bos = bos.reshape([b, num_bm_v, block_M])
    eos = eos.reshape([b, num_bm_v, block_M])
    jl = (bos.min(-1) // block_B).clip(0, num_blocks)
    jh = ((eos.max(-1) + block_B - 1) // block_B).clip(0, num_blocks)
    jh = paddle.maximum(jh, jl)
    block_range = paddle.stack([jl, jh], axis=-1).astype("int32").contiguous()

    # Skipped blocks (outside a tile's window) keep -inf so their host-side
    # block-score is 0; pre-fill instead of leaving the buffer uninitialised.
    block_logit = paddle.full(
        [b, h, s, num_blocks], float("-inf"), dtype="float32"
    )
    kernel = block_score_mha_fwd(
        h, d, float(sm_scale), D_v=d_v, block_B=block_B, block_M=block_M
    )
    out, lse = kernel(q, k, v, valid_range, block_range, block_logit)
    return out, lse, block_logit
