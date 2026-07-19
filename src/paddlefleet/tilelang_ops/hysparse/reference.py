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

"""Naive Paddle reference (散算子) for HySparse block attention operators.

Two operators, mirroring the paper (arXiv 2602.03560), with a single shared
Key/Value head (MQA/MLA):

* ``ref_block_score_attn_mqa``  — block-score attention, "Algorithm 1": standard
  full attention (shared K/V head) that additionally emits block-level max
  attention *probability* scores ``S`` (eq. 3), used for TopK block selection.
* ``ref_block_sparse_attn_mqa`` — block-sparse attention: each query token
  attends only to a per-token selected set of key blocks gathered from the
  shared K/V head.

Masking (causal + document) is expressed through ``valid_range`` of shape
``[B, S, 2]`` giving, per query token, the half-open valid key column range
``[bos, eos)``. Causal masking sets ``eos = t + 1``; document masking sets
``bos`` to the document start.

These functions are written for readability/correctness, not speed, and are
the ground truth the TileLang kernels are validated against.
"""

import paddle
import paddle.nn.functional as F

NEG_INF = float("-inf")


def _range_mask(valid_range, seq_len_kv):
    """Build a boolean key mask [B, 1, S, S_kv] from valid_range [B, S, 2]."""
    bos = valid_range[..., 0].unsqueeze(1).unsqueeze(-1)  # [B, 1, S, 1]
    eos = valid_range[..., 1].unsqueeze(1).unsqueeze(-1)  # [B, 1, S, 1]
    col = paddle.arange(seq_len_kv, dtype=valid_range.dtype)
    col = col.reshape([1, 1, 1, seq_len_kv])  # [1, 1, 1, S_kv]
    return (col >= bos) & (col < eos)  # [B, 1, S, S_kv]


def _to_bhsd(x):
    """[B, S, H, D] -> [B, H, S, D]."""
    return x.transpose([0, 2, 1, 3])


def ref_block_score_attn_mqa(q, k, v, valid_range, sm_scale=None, block_B=64):
    """Block-score attention reference (MQA): full attention with a single
    shared K/V head plus block-max probability scores (eq. 3).

    Args:
        q:           [B, S, H, D] query (H heads).
        k, v:        [B, S_kv, D] shared key/value (single head).
        valid_range: [B, S, 2] int, per-query [bos, eos) valid key columns.
        sm_scale:    softmax scale; defaults to D**-0.5.
        block_B:     key block size for the emitted block scores.

    Returns:
        out:     [B, S, H, D] attention output.
        lse:     [B, S, H] natural-log-sum-exp of the masked logits.
        s_block: [B, H, S, num_blocks] block-max softmax probability (eq. 3),
                 num_blocks = ceil(S_kv / block_B). Fully-masked blocks -> 0.
    """
    b, s, h, d = q.shape
    s_kv = k.shape[1]
    if sm_scale is None:
        sm_scale = d**-0.5

    qb = _to_bhsd(q).astype("float32")  # [B, H, S, D]
    # broadcast the single shared K/V head across all query heads
    kb = k.astype("float32").unsqueeze(1)  # [B, 1, S_kv, D]
    vb = v.astype("float32").unsqueeze(1)  # [B, 1, S_kv, D]

    logits = paddle.matmul(qb, kb, transpose_y=True) * sm_scale  # [B,H,S,S_kv]
    mask = _range_mask(valid_range, s_kv)  # [B,1,S,S_kv]
    neg = paddle.full_like(logits, NEG_INF)
    logits = paddle.where(mask, logits, neg)

    # Rows with no valid key (bos >= eos) would make softmax/logsumexp nan;
    # guard them to 0 output and leave lse as -inf (matches the kernel).
    row_has_key = mask.any(axis=-1, keepdim=True)  # [B,1,S,1]
    lse = paddle.logsumexp(logits, axis=-1)  # [B,H,S]
    probs = F.softmax(logits, axis=-1)  # [B,H,S,S_kv]
    probs = paddle.where(
        row_has_key.expand_as(probs), probs, paddle.zeros_like(probs)
    )
    out = paddle.matmul(probs, vb.expand([b, h, s_kv, d]))  # [B,H,S,D]

    # Block-max probability with **document-relative** block coordinates: block j
    # of a query spans key columns [bos + j*block_B, bos + (j+1)*block_B), i.e.
    # the grid is anchored at that query's document start ``bos`` (not absolute
    # sequence columns). This matches processing each document standalone.
    num_blocks = (s_kv + block_B - 1) // block_B
    col = paddle.arange(s_kv, dtype="int64").reshape([1, 1, s_kv])  # [1,1,S_kv]
    bos = valid_range[..., 0:1].astype("int64")  # [B, S, 1]
    rel = col - bos  # [B, S, S_kv] column position relative to doc start
    rel_id = paddle.where(  # relative block id; -1 for cols before doc start
        rel >= 0, rel // block_B, paddle.full_like(rel, -1)
    )  # [B, S, S_kv]
    rel_id = rel_id.unsqueeze(1)  # [B, 1, S, S_kv]
    s_block_list = []
    for j in range(num_blocks):
        hit = rel_id == j  # [B, 1, S, S_kv]
        masked = paddle.where(hit, probs, paddle.zeros_like(probs))
        s_block_list.append(masked.max(axis=-1))  # [B, H, S]
    s_block = paddle.stack(s_block_list, axis=-1)  # [B,H,S,num_blocks]

    out = out.transpose([0, 2, 1, 3])  # back to [B,S,H,D]
    lse = lse.transpose([0, 2, 1])  # [B,S,H]
    return out.astype(q.dtype), lse, s_block


def _selected_key_mask(indices, valid_range, seq_len_kv, block_B):
    """Boolean [B, S, S_kv]: key column selected by this query's block ids.

    Block ids are **document-relative**: block ``j`` of a query spans key
    columns ``[bos + j*block_B, bos + (j+1)*block_B)`` where ``bos`` is the
    query's document start (``valid_range[..., 0]``). A column is selected iff
    it lies at or after ``bos`` and its relative block id is among the query's
    (valid) selected ids.

    indices: [B, S, nsel] int block ids; -1 marks an invalid/padding slot.
    """
    col = paddle.arange(seq_len_kv, dtype="int64").reshape([1, 1, seq_len_kv])
    bos = valid_range[..., 0:1].astype("int64")  # [B, S, 1]
    rel = col - bos  # [B, S, S_kv]
    col_block = paddle.where(  # relative block id of each key column
        rel >= 0, rel // block_B, paddle.full_like(rel, -1)
    )  # [B, S, S_kv]
    idx = indices.astype("int64").unsqueeze(-2)  # [B,S,1,nsel]
    col_block_e = col_block.unsqueeze(-1)  # [B,S,S_kv,1]
    # a column is selected iff its relative block id equals any valid selected id
    hit = (col_block_e == idx) & (idx >= 0)  # [B,S,S_kv,nsel]
    return hit.any(axis=-1)  # [B,S,S_kv]


def ref_block_sparse_attn_mqa(
    q, k, v, indices, valid_range, sm_scale=None, block_B=64
):
    """Block-sparse attention reference (MQA): per-query-token block-sparse
    attention with a single Key/Value head shared across all query heads.

    This mirrors the efficient HySparse sparse branch: the block indices are
    shared across heads (GQA group-wise max upstream), and K/V are a single
    shared head so one gathered block feeds every query head.

    Args:
        q:           [B, S, H, D] query (H heads).
        k, v:        [B, S_kv, D] shared key/value (single head).
        indices:     [B, S, nsel] int block ids selected per query token
                     (-1 = padding). Shared across heads.
        valid_range: [B, S, 2] int, per-query [bos, eos) valid key columns.
        sm_scale:    softmax scale; defaults to D**-0.5.
        block_B:     key block size.

    Returns:
        out: [B, S, H, D]; lse: [B, S, H].
    """
    b, s, h, d = q.shape
    s_kv = k.shape[1]
    if sm_scale is None:
        sm_scale = d**-0.5

    qb = _to_bhsd(q).astype("float32")  # [B, H, S, D]
    # broadcast the single shared KV head across all query heads
    kb = k.astype("float32").unsqueeze(1)  # [B, 1, S_kv, D]
    vb = v.astype("float32").unsqueeze(1)  # [B, 1, S_kv, D]

    logits = paddle.matmul(qb, kb, transpose_y=True) * sm_scale  # [B,H,S,S_kv]

    range_m = _range_mask(valid_range, s_kv)  # [B,1,S,S_kv]
    sel_m = _selected_key_mask(indices, valid_range, s_kv, block_B).unsqueeze(
        1
    )  # [B,1,S,S_kv]
    mask = range_m & sel_m
    neg = paddle.full_like(logits, NEG_INF)
    logits = paddle.where(mask, logits, neg)

    # LSE is taken from the fully -inf-masked logits so empty rows (no valid
    # key) stay -inf, matching the kernel and the docstring. softmax then uses
    # the same logits and its NaN empty rows are zeroed via ``row_has_key``.
    row_has_key = mask.any(axis=-1, keepdim=True)  # [B,1,S,1]
    lse = paddle.logsumexp(logits, axis=-1)  # [B,H,S]
    probs = F.softmax(logits, axis=-1)
    probs = paddle.where(
        row_has_key.expand_as(probs), probs, paddle.zeros_like(probs)
    )
    out = paddle.matmul(probs, vb.expand([b, h, s_kv, d]))  # [B,H,S,D]

    out = out.transpose([0, 2, 1, 3])
    lse = lse.transpose([0, 2, 1])
    return out.astype(q.dtype), lse


def make_causal_valid_range(seq_len, batch=1, doc_lengths=None):
    """Helper: build valid_range [B, S, 2] for causal (+ optional document) mask.

    Args:
        seq_len:     total sequence length S (== S_kv).
        batch:       batch size B.
        doc_lengths: optional list of document lengths (packed along S). If
                     given, sum must equal seq_len; each token's bos is set to
                     its document start. If None, a single document is assumed.

    Returns:
        valid_range: [B, S, 2] int32.
    """
    pos = paddle.arange(seq_len, dtype="int32")
    eos = pos + 1  # causal: attend up to and including self
    if doc_lengths is None:
        bos = paddle.zeros([seq_len], dtype="int32")
    else:
        assert sum(doc_lengths) == seq_len
        starts = []
        cur = 0
        for dl in doc_lengths:
            starts += [cur] * dl
            cur += dl
        bos = paddle.to_tensor(starts, dtype="int32")
    vr = paddle.stack([bos, eos], axis=-1)  # [S, 2]
    return vr.unsqueeze(0).expand([batch, seq_len, 2]).contiguous()
