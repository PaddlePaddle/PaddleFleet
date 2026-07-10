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

"""Autograd (``paddle.autograd.PyLayer``) wrappers for the HySparse MQA block
attention operators, combining the separate TileLang forward and backward
kernels into differentiable ops usable inside a training network.

Two ops, both with a single shared K/V head:

* :class:`BlockScoreMQAAttn` -- full block-score attention. Returns the
  attention output (differentiable) plus ``lse`` and per-block ``block_logit``
  (both non-differentiable: they feed a non-differentiable TopK selection).
* :class:`BlockSparseMQAAttn` -- block-sparse gather attention over a
  per-query-token selection of key blocks. Returns output (differentiable) and
  ``lse`` (non-differentiable).

The convenience :func:`hysparse_mqa_attention` chains score -> TopK -> sparse:
the TopK sits between the two PyLayers and carries no gradient, exactly as in
the non-differentiable :func:`.pipeline.hysparse_forward_mqa`.
"""

import paddle

from .block_score_attn import block_score_mqa_attn_fwd
from .block_score_attn_bwd import block_score_mqa_bwd_interface
from .block_sparse_attn_mqa import block_sparse_mqa_attn_fwd
from .block_sparse_attn_mqa_bwd import block_sparse_mqa_bwd_interface
from .pipeline import select_topk_blocks


class BlockScoreMQAAttn(paddle.autograd.PyLayer):
    """Differentiable MQA block-score attention.

    forward inputs: q [B,S,H,Dk], k [B,S_kv,Dk], v [B,S_kv,Dv],
    valid_range [B,S,2] int32, plus scalar ``sm_scale`` and ``block_B``.
    outputs: out [B,S,H,Dv] (differentiable), lse [B,S,H] and
    block_logit [B,H,S,num_blocks] (both non-differentiable).
    """

    @staticmethod
    def forward(ctx, q, k, v, valid_range, sm_scale, block_B):
        out, lse, block_logit = block_score_mqa_attn_fwd(
            q, k, v, valid_range, sm_scale=sm_scale, block_B=block_B
        )
        ctx.save_for_backward(q, k, v, out, lse, valid_range)
        ctx.sm_scale = sm_scale
        ctx.block_B = block_B
        # lse / block_logit feed a non-differentiable TopK -> no gradient.
        ctx.mark_non_differentiable(lse, block_logit)
        return out, lse, block_logit

    @staticmethod
    def backward(ctx, grad_out, *_):
        q, k, v, out, lse, valid_range = ctx.saved_tensor()
        dq, dk, dv = block_score_mqa_bwd_interface(
            q,
            k,
            v,
            out,
            grad_out.contiguous(),
            lse,
            valid_range,
            sm_scale=ctx.sm_scale,
            block_B=ctx.block_B,
        )
        return dq, dk, dv


class BlockSparseMQAAttn(paddle.autograd.PyLayer):
    """Differentiable MQA block-sparse gather attention.

    forward inputs: q [B,S,H,Dk], k [B,S_kv,Dk], v [B,S_kv,Dv],
    indices [B,S,nsel] int32 (document-relative block ids, -1 padding),
    valid_range [B,S,2] int32, plus scalar ``sm_scale`` and ``block_B``.
    outputs: out [B,S,H,Dv] (differentiable), lse [B,S,H] (non-differentiable).
    """

    @staticmethod
    def forward(ctx, q, k, v, indices, valid_range, sm_scale, block_B):
        out, lse = block_sparse_mqa_attn_fwd(
            q, k, v, indices, valid_range, sm_scale=sm_scale, block_B=block_B
        )
        ctx.save_for_backward(q, k, v, out, lse, indices, valid_range)
        ctx.sm_scale = sm_scale
        ctx.block_B = block_B
        ctx.mark_non_differentiable(lse)
        return out, lse

    @staticmethod
    def backward(ctx, grad_out, *_):
        q, k, v, out, lse, indices, valid_range = ctx.saved_tensor()
        dq, dk, dv = block_sparse_mqa_bwd_interface(
            q,
            k,
            v,
            out,
            grad_out.contiguous(),
            lse,
            indices,
            valid_range,
            sm_scale=ctx.sm_scale,
            block_B=ctx.block_B,
        )
        return dq, dk, dv


def block_score_mqa_attention(q, k, v, valid_range, sm_scale=None, block_B=64):
    """Differentiable block-score attention (see :class:`BlockScoreMQAAttn`).

    Returns ``(out, lse, block_logit)``; only ``out`` carries gradient.
    """
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    return BlockScoreMQAAttn.apply(
        q, k, v, valid_range, float(sm_scale), block_B
    )


def block_sparse_mqa_attention(
    q, k, v, indices, valid_range, sm_scale=None, block_B=64
):
    """Differentiable block-sparse gather attention.

    Returns ``(out, lse)``; only ``out`` carries gradient.
    """
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    return BlockSparseMQAAttn.apply(
        q, k, v, indices, valid_range, float(sm_scale), block_B
    )


def hysparse_mqa_attention(
    q, k, v, valid_range, topk, sm_scale=None, block_B=64, head_agg="max"
):
    """Differentiable HySparse forward: block-score -> TopK -> block-sparse.

    The TopK selection between the two attention ops is non-differentiable
    (block scores feed a hard selection), so gradient flows only through the
    two attention PyLayers.

    Args:
        q:           [B, S, H, Dk] bf16 query.
        k:           [B, S_kv, Dk] bf16 shared key head.
        v:           [B, S_kv, Dv] bf16 shared value head.
        valid_range: [B, S, 2] int32 causal + document valid key range.
        topk:        number of key blocks selected per query token.
        sm_scale:    softmax scale; defaults to Dk**-0.5.
        block_B:     key block size.
        head_agg:    cross-head block-score aggregation ("max" or "sum").

    Returns:
        sparse_out [B,S,H,Dv], sparse_lse [B,S,H], indices [B,S,topk],
        full_out [B,S,H,Dv], full_lse [B,S,H]. ``sparse_out`` and ``full_out``
        carry gradient; the rest are detached side outputs.
    """
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    sm_scale = float(sm_scale)

    full_out, full_lse, block_logit = block_score_mqa_attention(
        q, k, v, valid_range, sm_scale=sm_scale, block_B=block_B
    )
    # non-differentiable block selection (detached inputs)
    with paddle.no_grad():
        indices = select_topk_blocks(
            block_logit.detach(),
            full_lse.detach(),
            valid_range,
            sm_scale,
            topk,
            block_B,
            head_agg=head_agg,
        )
    sparse_out, sparse_lse = block_sparse_mqa_attention(
        q, k, v, indices, valid_range, sm_scale=sm_scale, block_B=block_B
    )
    return sparse_out, sparse_lse, indices, full_out, full_lse
