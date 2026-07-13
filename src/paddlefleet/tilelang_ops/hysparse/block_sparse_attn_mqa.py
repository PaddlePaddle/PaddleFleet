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

"""Block-sparse attention (MQA gather): per-query-token block-sparse attention
with a single shared Key/Value head.

Unlike the dense-mask variant (:mod:`block_sparse_attn`), this kernel actually
**gathers only the selected key blocks** and skips the rest, so its cost scales
with ``nsel`` (selected blocks) instead of the full sequence length.

Efficiency comes from the MQA/MLA layout: K/V is a single head shared by all
query heads, and the TopK block indices are shared across heads. The kernel
therefore places the ``H`` query heads on the GEMM ``M`` dimension (one query
token per program), so a single gathered key block feeds every head and the
GEMM is wide enough to saturate tensor cores. This mirrors the repo's
``sparse_mqa`` kernel but gathers whole ``block_B``-sized key blocks and applies
the causal + document ``valid_range`` column mask inside the kernel.

Layout:
* ``Q``           ``[B, S, H, D]`` bf16.
* ``K``, ``V``    ``[B, S_kv, D]`` bf16 (single shared head).
* ``Indices``     ``[B, S, nsel]`` int32 block ids (``-1`` = padding), shared
  across heads.
* ``ValidRange``  ``[B, S, 2]`` int32 per-query ``[bos, eos)``.
* ``Output``      ``[B, S, H, D]`` bf16, ``Lse`` ``[B, S, H]`` fp32 natural-log.
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
def block_sparse_mqa_fwd(
    H,
    D,
    nsel,
    sm_scale,
    block_B=64,
    num_stages=2,
    threads=128,
):
    assert D % 16 == 0, (
        f"D must be a multiple of 16 (tensor-core k-tile), got {D}"
    )
    assert H <= 128, "this kernel supports up to 128 query heads"
    scale_log2 = sm_scale * 1.44269504

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    q_shape = [batch, seq_len, H, D]
    kv_shape = [batch, seq_len_kv, D]
    o_shape = [batch, seq_len, H, D]
    idx_shape = [batch, seq_len, nsel]
    vr_shape = [batch, seq_len, 2]
    lse_shape = [batch, seq_len, H]

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
        Indices: T.Tensor(idx_shape, idx_dtype),
        ValidRange: T.Tensor(vr_shape, idx_dtype),
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
            row_max = T.alloc_fragment([PH], accum_dtype)
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

            for i in T.Pipelined(nsel, num_stages=num_stages):
                blk = Indices[bb, bs, i]
                valid_blk = blk >= 0
                safe_blk = T.if_then_else(valid_blk, blk, 0)

                # gather one selected block. Block ids are document-relative:
                # relative block ``blk`` spans absolute columns
                # [bos + blk*BB, bos + (blk+1)*BB). Guard the read against the
                # padded K/V length (columns >= eos are masked out below, so a
                # clamped dummy read is harmless).
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

                T.clear(acc_s)
                T.gemm(
                    Q_shared,
                    K_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )

                # mask: keep column iff block valid and in [bos, eos). The
                # relative column bos + blk*BB + c is always >= bos, so only the
                # upper bound needs checking.
                for h, c in T.Parallel(PH, BB):
                    col = bos + safe_blk * BB + c
                    keep = valid_blk and (col < eos)
                    acc_s[h, c] = T.if_then_else(
                        keep, acc_s[h, c], -T.infinity(accum_dtype)
                    )

                # online softmax (base 2)
                T.reduce_max(acc_s, row_max, dim=1, clear=True)
                T.copy(m_i, m_prev)
                for h in T.Parallel(PH):
                    m_i[h] = T.max(m_i[h], row_max[h] * sm_scale)
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

            # normalize; empty rows (no selected valid key) -> 0 out / -inf lse
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


def block_sparse_mqa_attn_fwd(
    q, k, v, indices, valid_range, sm_scale=None, block_B=64
):
    """Forward interface for the MQA gather block-sparse attention.

    Args:
        q:           [B, S, H, D] bf16.
        k, v:        [B, S_kv, D] bf16 shared key/value.
        indices:     [B, S, nsel] int32 block ids (-1 padding), shared heads.
        valid_range: [B, S, 2] int32.
        sm_scale:    softmax scale; defaults to D**-0.5.
        block_B:     key block size.

    Returns:
        out [B,S,H,D] bf16, lse [B,S,H] fp32 (natural log; empty rows -inf).
    """
    assert q.is_contiguous()
    b, s, h, d = q.shape
    s_kv = k.shape[1]
    nsel = indices.shape[-1]
    # lightweight host-side shape checks (no device sync) so mismatched inputs
    # fail early and clearly instead of causing undefined kernel behaviour.
    assert len(k.shape) == 3 and list(k.shape) == list(v.shape), (
        f"k/v must be [B, S_kv, D] and match; got {k.shape} vs {v.shape}"
    )
    assert k.shape[0] == b and k.shape[2] == d, (
        f"k/v batch/head_dim must match q; got k {k.shape}, q {q.shape}"
    )
    assert list(indices.shape[:2]) == [b, s], (
        f"indices must be [B, S, nsel] matching q; got {indices.shape}"
    )
    assert list(valid_range.shape) == [b, s, 2], (
        f"valid_range must be [B, S, 2]; got {valid_range.shape}"
    )
    if sm_scale is None:
        sm_scale = d**-0.5
    if valid_range.dtype != paddle.int32:
        valid_range = valid_range.cast("int32")
    if indices.dtype != paddle.int32:
        indices = indices.cast("int32")

    # kernel gathers whole block_B blocks without bounds clamping, so pad
    # K/V so the last block is fully addressable.
    pad = (block_B - s_kv % block_B) % block_B
    if pad > 0:
        k = paddle.nn.functional.pad(k, [0, 0, 0, pad])
        v = paddle.nn.functional.pad(v, [0, 0, 0, pad])
    k = k.contiguous()
    v = v.contiguous()
    valid_range = valid_range.contiguous()
    indices = indices.contiguous()

    kernel = block_sparse_mqa_fwd(h, d, nsel, float(sm_scale), block_B=block_B)
    out, lse = kernel(q, k, v, indices, valid_range)
    return out, lse
