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

"""HySparse forward pipeline: chain block-score attention, block-TopK, and
block-sparse attention.

All stages share a single K/V head (MQA/MLA sparse branch):

1. **Block-score attention** (:func:`block_score_mqa_attn_fwd`): full attention
   with the shared K/V head that also emits per-(query, key-block) max raw
   logits ``BlockLogit``.
2. **Block TopK selection** (:func:`select_topk_blocks`): recover eq.(3) block
   scores, aggregate them across heads by a group-wise **maximum** (shared
   block selection), mask blocks that hold no causal/document-valid key, and
   TopK to per-query block indices.
3. **Block-sparse attention** (:func:`block_sparse_mqa_attn_fwd`): MQA
   block-sparse attention that gathers only the selected blocks, so its cost
   scales with ``topk`` rather than the sequence length.

The block scores feed a non-differentiable TopK, so only the two attention
operators carry gradient.
"""

import paddle

from .block_score_attn import (
    block_score_mqa_attn_fwd,
    block_scores_from_logit,
)
from .block_sparse_attn_mqa import block_sparse_mqa_attn_fwd


def _valid_block_mask(valid_range, num_blocks, block_B):
    """Boolean [B, S, num_blocks]: relative block j holds >=1 valid key.

    Block ids are **document-relative**: block j of a query spans key columns
    ``[bos + j*block_B, bos + (j+1)*block_B)``. It holds at least one valid key
    iff its start still lies inside the query's valid range, i.e.
    ``bos + j*block_B < eos``.
    """
    bos = valid_range[..., 0:1].astype("int64")  # [B, S, 1]
    eos = valid_range[..., 1:2].astype("int64")  # [B, S, 1]
    j = paddle.arange(num_blocks, dtype="int64").reshape([1, 1, num_blocks])
    start = bos + j * block_B  # absolute column where relative block j starts
    return start < eos  # [B, S, num_blocks]


def select_topk_blocks(
    block_logit, lse, valid_range, sm_scale, topk, block_B, head_agg="max"
):
    """Select per-query TopK key blocks from block-score attention outputs.

    Args:
        block_logit: [B, H, S, num_blocks] raw per-block max logit.
        lse:         [B, S, H] natural-log LSE from block-score attention.
        valid_range: [B, S, 2] int, per-query [bos, eos) valid key columns.
        sm_scale:    softmax scale.
        topk:        number of blocks to select per query token.
        block_B:     key block size.
        head_agg:    how to aggregate block scores across heads so the whole
                     query group shares one selection. ``"max"`` (paper eq. 3
                     group-wise maximum) or ``"sum"``.

    Returns:
        indices: [B, S, topk] int32 selected block ids, shared across heads;
                 slots beyond the number of valid blocks are -1. The width is
                 always ``topk`` even when ``topk > num_blocks`` (the extra
                 slots are -1 padding), keeping the shape contract stable.
    """
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    b, h, s, num_blocks = block_logit.shape
    scores = block_scores_from_logit(block_logit, lse, sm_scale)  # [B,H,S,nb]
    # aggregate across heads (block selection shared across the query group)
    if head_agg == "max":
        agg = scores.max(axis=1)  # [B, S, num_blocks]
    elif head_agg == "sum":
        agg = scores.sum(axis=1)  # [B, S, num_blocks]
    else:
        raise ValueError(f"unknown head_agg={head_agg!r}")

    valid = _valid_block_mask(valid_range, num_blocks, block_B)  # [B,S,nb]
    neg = paddle.full_like(agg, -1.0)
    agg = paddle.where(valid, agg, neg)  # invalid blocks pushed to the bottom

    k = min(topk, num_blocks)
    top_val, top_idx = paddle.topk(agg, k=k, axis=-1)  # [B, S, k]
    # slots that landed on an invalid block (negative score) -> -1
    top_idx = paddle.where(top_val >= 0, top_idx, paddle.full_like(top_idx, -1))
    top_idx = top_idx.astype("int32")
    if k < topk:
        # honour the promised [B, S, topk] width when topk exceeds the number
        # of blocks; the surplus slots are -1 padding (already ignored by the
        # gather kernel and the reference).
        pad = paddle.full([b, s, topk - k], -1, dtype="int32")
        top_idx = paddle.concat([top_idx, pad], axis=-1)
    return top_idx.contiguous()


def hysparse_forward_mqa(q, k, v, valid_range, topk, sm_scale=None, block_B=64):
    """HySparse forward with a single shared K/V head (MQA/MLA sparse branch).

    Three stages (block-score -> TopK -> block-sparse) with K/V as one shared
    head
    ``[B, S_kv, D]``: the block selection is aggregated across the whole query
    group by a group-wise **maximum** (paper eq. 3) so all heads share indices,
    and the sparse branch uses the MQA gather kernel
    (:func:`block_sparse_mqa_attn_fwd`) whose cost scales with ``topk`` rather
    than the sequence length.

    Both stages keep K/V as a single shared head: block-score attention uses
    the MQA scoring kernel (:func:`block_score_mqa_attn_fwd`, no head broadcast)
    and the sparse branch uses the MQA gather kernel.

    Args:
        q:           [B, S, H, D] bf16 query (H heads).
        k, v:        [B, S_kv, D] bf16 single shared key/value head.
        valid_range: [B, S, 2] int32 causal + document valid key range.
        topk:        number of blocks selected per query token.
        sm_scale:    softmax scale; defaults to D**-0.5.
        block_B:     key block size.

    Returns:
        sparse_out:  [B, S, H, D] MQA block-sparse attention output.
        sparse_lse:  [B, S, H] natural-log LSE of the sparse branch.
        indices:     [B, S, topk] selected block ids (int32, -1 padding).
        full_out:    [B, S, H, D] full attention output (block-score).
        full_lse:    [B, S, H] natural-log LSE of block-score attention.
    """
    b, s, h, d = q.shape
    if sm_scale is None:
        sm_scale = d**-0.5

    # block-score attention with the single shared K/V head -- no broadcast.
    full_out, full_lse, block_logit = block_score_mqa_attn_fwd(
        q, k, v, valid_range, sm_scale=sm_scale, block_B=block_B
    )
    indices = select_topk_blocks(
        block_logit,
        full_lse,
        valid_range,
        sm_scale,
        topk,
        block_B,
        head_agg="max",
    )
    # sparse branch uses the shared single-head K/V gather kernel
    sparse_out, sparse_lse = block_sparse_mqa_attn_fwd(
        q, k, v, indices, valid_range, sm_scale=sm_scale, block_B=block_B
    )
    return sparse_out, sparse_lse, indices, full_out, full_lse
