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

"""Block-score attention (paper "Algorithm 1") with a single shared K/V head
(MQA/MLA): full attention that additionally emits per-block max logits used to
derive block-level selection scores.

Q is ``[B, S, H, D]`` (H query heads); K, V are one shared head
``[B, S_kv, D]``. Masking (causal + document) is expressed through
``valid_range`` ``[B, S, 2]`` giving each query's half-open valid key column
range ``[bos, eos)``.

The forward returns:
* ``Output``    ``[B, S, H, D]`` attention output.
* ``Lse``       ``[B, S, H]`` natural log-sum-exp of the *scaled* logits.
* ``BlockLogit`` ``[B, H, S, num_blocks]`` per-(query, key-block) max of the
  *raw* (unscaled) ``q·k`` logit over valid columns; fully-masked blocks store
  ``-inf``. The block-max softmax *probability* score of eq. (3) is recovered
  on the host as ``exp(BlockLogit * sm_scale - Lse)`` (see
  :func:`block_scores_from_logit`).

The backward (:mod:`block_score_attn_bwd`) is a standard flash-attention
backward (dQ, dK, dV); the block scores feed a non-differentiable TopK and
therefore carry no gradient.
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
def block_score_mqa_fwd(
    H,
    D,
    sm_scale,
    block_B=64,
    num_stages=2,
    threads=128,
):
    """Block-score kernel with a single shared K/V head (MQA/MLA scoring).

    One program per query **token**, with the ``H`` query heads placed on the
    GEMM ``M`` dimension (like the gather kernel). Blocks are
    **document-relative**: block ``j`` of a query spans absolute key columns
    ``[bos + j*block_B, bos + (j+1)*block_B)`` where ``bos`` is the query's
    document start. The full-attention ``Output``/``Lse`` are unchanged (they
    still sum over every valid key in ``[bos, eos)``); only the emitted
    per-block max logit is binned relative to ``bos`` so downstream TopK
    selects blocks exactly as if each document were processed standalone.
    """
    assert D % 16 == 0, (
        f"D must be a multiple of 16 (tensor-core k-tile), got {D}"
    )
    assert H <= 128, "this kernel supports up to 128 query heads"
    scale_log2 = sm_scale * 1.44269504  # log2(e), online softmax in base 2

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")
    num_blocks = T.dynamic("num_blocks")

    q_shape = [batch, seq_len, H, D]
    kv_shape = [batch, seq_len_kv, D]
    o_shape = [batch, seq_len, H, D]
    lse_shape = [batch, seq_len, H]
    vr_shape = [batch, seq_len, 2]
    blk_shape = [batch, H, seq_len, num_blocks]

    dtype = T.bfloat16
    accum_dtype = T.float32
    idx_dtype = T.int32
    PH = max(tilelang.math.next_power_of_2(H), 16)  # padded heads on M
    BB = block_B

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        ValidRange: T.Tensor(vr_shape, idx_dtype),
        BlockLogit: T.Tensor(blk_shape, accum_dtype),
        Output: T.Tensor(o_shape, dtype),
        Lse: T.Tensor(lse_shape, accum_dtype),
    ):
        with T.Kernel(seq_len, batch, threads=threads) as (bs, bb):
            Q_shared = T.alloc_shared([PH, D], dtype)
            K_shared = T.alloc_shared([BB, D], dtype)
            V_shared = T.alloc_shared([BB, D], dtype)
            P_shared = T.alloc_shared([PH, BB], dtype)

            acc_o = T.alloc_fragment([PH, D], accum_dtype)
            acc_s = T.alloc_fragment([PH, BB], accum_dtype)
            blk_max = T.alloc_fragment([PH], accum_dtype)
            m_i = T.alloc_fragment([PH], accum_dtype)
            m_prev = T.alloc_fragment([PH], accum_dtype)
            l_i = T.alloc_fragment([PH], accum_dtype)
            l_new = T.alloc_fragment([PH], accum_dtype)
            alpha = T.alloc_fragment([PH], accum_dtype)

            T.fill(acc_o, 0)
            T.fill(m_i, -(2**30))
            T.fill(l_i, 0)

            bos = ValidRange[bb, bs, 0]
            eos = ValidRange[bb, bs, 1]

            # load this token's H query heads onto M (pad rows >= H with 0)
            for h, d in T.Parallel(PH, D):
                Q_shared[h, d] = T.if_then_else(
                    h < H, Q[bb, bs, h, d], T.cast(0, dtype)
                )

            # causal/document early-exit: only this token's own valid blocks
            # (relative block j in [0, ceil((eos-bos)/block_B))) can hold a
            # valid key; every later block is fully masked (all cols >= eos)
            # and contributes nothing to out/lse. Its per-block max logit would
            # be -inf, which the host ``BlockLogit`` buffer is pre-filled with,
            # so skipping those blocks is exact. For pure causal this halves the
            # work; for packed documents a token only scans its own document.
            num_valid_blocks = T.ceildiv(eos - bos, BB)
            for j in T.Pipelined(num_valid_blocks, num_stages=num_stages):
                # gather relative block j: cols [bos + j*BB, bos + (j+1)*BB).
                # Guard the read against the padded K/V length (cols >= eos are
                # masked below, so a clamped dummy read is harmless).
                for c, d in T.Parallel(BB, D):
                    col = bos + j * BB + c
                    in_bounds = col < seq_len_kv
                    safe_col = T.if_then_else(in_bounds, col, 0)
                    K_shared[c, d] = T.if_then_else(
                        in_bounds, K[bb, safe_col, d], T.cast(0, dtype)
                    )
                    V_shared[c, d] = T.if_then_else(
                        in_bounds, V[bb, safe_col, d], T.cast(0, dtype)
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

                # causal + document mask (col >= bos automatic for block j)
                for h, c in T.Parallel(PH, BB):
                    col = bos + j * BB + c
                    acc_s[h, c] = T.if_then_else(
                        col < eos, acc_s[h, c], -T.infinity(accum_dtype)
                    )

                # per-block max of raw logit over valid cols -> block score src
                T.reduce_max(acc_s, blk_max, dim=1, clear=True)
                for h in T.Parallel(PH):
                    if h < H:
                        BlockLogit[bb, h, bs, j] = blk_max[h]

                # online softmax (base 2) over scaled logits
                T.copy(m_i, m_prev)
                for h in T.Parallel(PH):
                    m_i[h] = T.max(m_i[h], blk_max[h] * sm_scale)
                for h in T.Parallel(PH):
                    alpha[h] = T.exp2((m_prev[h] - m_i[h]) * 1.44269504)
                for h, c in T.Parallel(PH, BB):
                    acc_s[h, c] = T.exp2(
                        acc_s[h, c] * scale_log2 - m_i[h] * 1.44269504
                    )
                T.reduce_sum(acc_s, l_new, dim=1)
                for h in T.Parallel(PH):
                    l_i[h] = l_i[h] * alpha[h] + l_new[h]
                for h, d in T.Parallel(PH, D):
                    acc_o[h, d] = acc_o[h, d] * alpha[h]
                T.copy(acc_s, P_shared)
                T.gemm(
                    P_shared, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow
                )

            # normalize; empty rows (no valid key) -> 0 out / -inf lse
            for h, d in T.Parallel(PH, D):
                acc_o[h, d] = T.if_then_else(
                    l_i[h] > 0, acc_o[h, d] / l_i[h], 0.0
                )
            for h, d in T.Parallel(PH, D):
                if h < H:
                    Output[bb, bs, h, d] = acc_o[h, d]
            for h in T.Parallel(PH):
                if h < H:
                    Lse[bb, bs, h] = T.if_then_else(
                        l_i[h] > 0,
                        m_i[h] + T.log(l_i[h]),
                        -T.infinity(accum_dtype),
                    )

    return main


def block_scores_from_logit(block_logit, lse, sm_scale):
    """Recover eq.(3) block-max probability scores on the host.

    Args:
        block_logit: [B, H, S, num_blocks] raw per-block max logit (-inf if
            fully masked), as returned by the forward kernel.
        lse:         [B, S, H] natural-log LSE from the forward kernel.
        sm_scale:    softmax scale.

    Returns:
        [B, H, S, num_blocks] block-max softmax probabilities in [0, 1];
        fully-masked blocks are 0.
    """
    lse_bhs = lse.transpose([0, 2, 1]).unsqueeze(-1)  # [B,H,S,1]
    scaled = block_logit.astype("float32") * sm_scale - lse_bhs
    scores = paddle.exp(scaled)
    # exp(-inf) already 0, but guard any nan from (-inf)-(-inf) style edge.
    scores = paddle.where(
        paddle.isfinite(scores), scores, paddle.zeros_like(scores)
    )
    return scores


def block_score_mqa_attn_fwd(q, k, v, valid_range, sm_scale=None, block_B=64):
    """Forward interface for the MQA (shared K/V head) block-score attention.

    Args:
        q:           [B, S, H, D] bf16 query (H heads).
        k, v:        [B, S_kv, D] bf16 single shared key/value head.
        valid_range: [B, S, 2] int32 per-query [bos, eos) valid key columns.
        sm_scale:    softmax scale; defaults to D**-0.5.
        block_B:     key block size.

    Returns:
        out [B,S,H,D], lse [B,S,H], block_logit [B,H,S,num_blocks] where
        num_blocks = ceil(S_kv / block_B).
    """
    assert q.is_contiguous()
    b, s, h, d = q.shape
    s_kv = k.shape[1]
    # lightweight host-side shape checks (no device sync) so mismatched inputs
    # fail early and clearly instead of causing undefined kernel behaviour.
    assert len(k.shape) == 3 and list(k.shape) == list(v.shape), (
        f"k/v must be [B, S_kv, D] and match; got {k.shape} vs {v.shape}"
    )
    assert k.shape[0] == b and k.shape[2] == d, (
        f"k/v batch/head_dim must match q; got k {k.shape}, q {q.shape}"
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
    if pad > 0:
        k = paddle.nn.functional.pad(k, [0, 0, 0, pad])
        v = paddle.nn.functional.pad(v, [0, 0, 0, pad])
    k = k.contiguous()
    v = v.contiguous()
    valid_range = valid_range.contiguous()
    num_blocks = (s_kv + pad) // block_B

    # Blocks past a token's own valid range are skipped by the kernel's
    # causal/document early-exit, so their per-block max logit must already read
    # as -inf (score 0). Pre-fill instead of leaving the buffer uninitialised.
    block_logit = paddle.full(
        [b, h, s, num_blocks], float("-inf"), dtype="float32"
    )
    kernel = block_score_mqa_fwd(h, d, float(sm_scale), block_B=block_B)
    out, lse = kernel(q, k, v, valid_range, block_logit)
    return out, lse, block_logit
