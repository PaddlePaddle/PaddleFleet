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

"""Full-candidate (dense) DSA indexer KL, on the cuDNN-frontend dense triple.

The DSA warmup phase supervises the indexer over **every** causal candidate, so
there is no top-k to shrink the width. The TileLang full-candidate indexer
cannot serve that past 8192 columns: its two bitonic buffers are sized
``2 * topk``, so one block asks for ``16 * topk + 25344`` B of shared memory
against an SM100 opt-in limit of 232448 B, and the launch fails with
``Failed to set the allowed dynamic shared memory size``.

The three dense cuDNN-frontend ops used here have no top-k stage at all, hence
no width-proportional shared memory and no ``[s, width]`` tensor that has to
survive until the backward:

* :func:`dense_indexer_kl_scores` -- raw indexer score + its LSE, so
  ``probs = exp(score - lse)`` is the indexer's prediction ``Q``;
* :func:`dense_attn_kl_scores` -- ``sum_h exp(QK*scale - LSE_h)`` + its L1
  norm, so ``score / l1norm`` is the KL target ``P``;
* :func:`dense_indexer_kl_bwd` -- consumes both raw score matrices and emits
  ``d_index_q / d_weights / d_index_k`` for ``coeff * (Q - P)``.

Everything runs in THD (varlen) layout, which is what per-document packing and
context parallel both need: ``cu_seqlens_q`` addresses the local query rows and
``cu_seqlens_k`` the globally-gathered keys, so a query row's candidate set is
its own document's causal prefix and nothing else. THD also compacts the score
width to the longest *visible* document instead of the whole packed sequence.

Weight convention: unlike the TileLang indexer and the sparse cuDNN pair, the
dense indexer score is called with ``sm_scale=1.0`` and ``weights`` exactly as
``DSAIndexer`` produced them, i.e. still carrying the pre-baked
``head_dim**-0.5``. That skips the un-bake/re-bake bf16 round trip (measured:
max_rel 8e-7 against an fp32 reference, versus 4.2e-4 when un-baked), and it
means ``d_weights`` comes back in the same pre-baked space as ``weights``.
"""

from __future__ import annotations

import paddle
from paddlefleet_ops import CUDNN_FRONTEND_HINT, is_cudnn_frontend_available


def _require_cudnn_frontend():
    if not is_cudnn_frontend_available():
        raise ImportError(CUDNN_FRONTEND_HINT)


class _HashableTensor(paddle.Tensor):
    """``paddle.Tensor`` with hashable ``shape`` / ``stride()``.

    The dense wrappers key their compiled-kernel cache on ``q.shape`` and
    ``q.stride()`` directly (``score_recompute/api.py`` ``key = (...)``), and
    Paddle returns both as lists, which are unhashable.
    """

    @property
    def shape(self):
        return tuple(super().shape)

    def stride(self, dim=None):
        if dim is None:
            return tuple(super().stride())
        return super().stride(dim)


def dense_kl_cu_seqlens(
    segment_lens: list[int], seq_offset: int, seq_len: int
) -> tuple:
    """``(cu_seqlens_q, cu_seqlens_k, max_q, max_k, q_causal_offsets)``.

    Thin ``ratio=1`` alias of the CP-aware helper the sparse indexer forward
    already uses, so the two paths cannot drift on where a document border sits.

    ``segment_lens`` must tile the **whole** global sequence, padding included
    (see ``_doc_segment_lens`` on the caller side): a segment shorter than its
    slot would let the next document's tokens into the causal prefix.
    """
    from .csa_indexer_fwd_cudnn import _make_cu_seqlens

    return _make_cu_seqlens(
        [int(n) for n in segment_lens], int(seq_offset), int(seq_len), 1
    )


def dense_indexer_kl_scores(
    index_q: paddle.Tensor,
    index_k: paddle.Tensor,
    weights: paddle.Tensor,
    cu_seqlens_q: paddle.Tensor,
    cu_seqlens_k: paddle.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    q_causal_offsets: paddle.Tensor | None = None,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Raw indexer score over every in-document causal candidate.

    Args:
        index_q: ``[s_local, H_i, D_i]`` bf16 indexer queries (local rows).
        index_k: ``[s_global, D_i]`` bf16 indexer keys (globally gathered).
        weights: ``[s_local, H_i]`` bf16 per-head weights, **pre-baked** with
            ``head_dim**-0.5`` exactly as ``DSAIndexer`` emits them.
        cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        q_causal_offsets: from :func:`dense_kl_cu_seqlens`.

    Returns:
        ``(score [s_local, max_seqlen_k] fp32, lse [s_local] fp32)``.
        ``score`` is ``-inf`` outside the causal candidate set, so
        ``exp(score - lse[:, None])`` is the softmax prediction with an exact
        zero everywhere else.
    """
    _require_cudnn_frontend()
    from paddlefleet_ops.cudnn.deepseek_sparse_attention.score_recompute.api import (
        dense_indexer_score_recompute_wrapper,
    )

    res = dense_indexer_score_recompute_wrapper(
        _HashableTensor(index_q.contiguous()),
        _HashableTensor(index_k.unsqueeze(1).contiguous()),
        _HashableTensor(weights.contiguous()),
        qhead_per_kv_head=int(index_q.shape[1]),
        # Pre-baked weights already carry ``head_dim**-0.5``; see the module
        # docstring for why this is not the TileLang convention.
        sm_scale=1.0,
        ratio=1,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=int(max_seqlen_q),
        max_seqlen_k=int(max_seqlen_k),
        q_causal_offsets=q_causal_offsets,
    )
    return res["out"], res["denom"]


def dense_indexer_kl_bwd(
    index_q: paddle.Tensor,
    weights: paddle.Tensor,
    index_k: paddle.Tensor,
    attn_score: paddle.Tensor,
    attn_l1norm: paddle.Tensor,
    index_score: paddle.Tensor,
    index_lse: paddle.Tensor,
    loss_coeff: float,
    cu_seqlens_q: paddle.Tensor,
    cu_seqlens_k: paddle.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    q_causal_offsets: paddle.Tensor | None = None,
    grad_loss: paddle.Tensor | float | None = None,
    block_I: int = 128,
) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """``d_index_q / d_weights / d_index_k`` of the dense full-candidate KL.

    The kernel derives both distributions from the raw score matrices --
    ``predict = exp(index_score - index_lse)``,
    ``target = attn_score / attn_l1norm`` -- and emits
    ``grad_scale * (predict - target)`` as the score gradient, which is the same
    quantity the TileLang branch spells out as
    ``(topk_probs - target) * loss_coeff / num_rows``.

    ``grad_scale = loss_coeff * grad_loss / total_q`` is fixed inside the
    kernel, so the ``1 / total_q`` is *not* optional: a caller whose forward
    divided by something else (a valid-token count, say) has to pre-multiply
    ``loss_coeff`` by ``total_q / its_own_denominator``.

    ``attn_score`` and ``index_score`` are consumed **in place** by the
    score-gradient stage and are *not* copied here: at 64k/cp=8 one of them is
    2 GiB, so cloning both would double the backward's width-proportional
    footprint (measured 4.82 vs 3.79 matrix-equivalents of peak at s=16384).
    Callers must therefore hand over matrices they are done with -- the warmup
    backward recomputes both immediately before this call and drops them right
    after. ``attn_l1norm`` and ``index_lse`` are read-only.

    Setting ``index_lse[q] = +inf`` zeroes that row's gradient exactly: it
    drives both ``predict`` and the kernel's ``log_clip_mask`` to 0. Zeroing
    ``attn_l1norm`` instead would not -- the target clamp
    ``max(target, exp(-100))`` leaves a residue.

    ``d_index_k`` is allocated in fp32 and cast back here. The kernel reduces
    dK through fp32 atomics regardless, and its own bf16 landing path ends in
    ``d_index_k.copy_(d_index_k_f32)`` (``indexer_backward/api.py:414``), which
    Paddle's ``copy_`` rejects across dtypes ("Tensor Copy cannot be
    performed"). Handing it an fp32 buffer takes the branch that reduces in
    place instead.
    """
    _require_cudnn_frontend()
    from paddlefleet_ops.cudnn.deepseek_sparse_attention.indexer_backward.api import (
        dense_indexer_backward_wrapper,
    )

    if grad_loss is None:
        grad_loss = paddle.ones([], dtype=paddle.float32)
    elif not isinstance(grad_loss, paddle.Tensor):
        grad_loss = paddle.to_tensor(float(grad_loss), dtype=paddle.float32)
    elif grad_loss.dtype != paddle.float32:
        grad_loss = grad_loss.cast(paddle.float32)

    out = dense_indexer_backward_wrapper(
        _HashableTensor(index_q.contiguous()),
        _HashableTensor(weights.contiguous()),
        _HashableTensor(index_k.contiguous()),
        _HashableTensor(attn_score),
        _HashableTensor(attn_l1norm.contiguous()),
        _HashableTensor(index_score),
        _HashableTensor(index_lse.contiguous()),
        # Matches the ``sm_scale=1.0`` / pre-baked-weights convention of
        # ``dense_indexer_kl_scores``; the two must agree or the gradient is
        # built on a different score than the forward measured.
        sm_scale=1.0,
        loss_coeff=float(loss_coeff),
        grad_loss=grad_loss,
        block_I=int(block_I),
        ratio=1,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=int(max_seqlen_q),
        max_seqlen_k=int(max_seqlen_k),
        q_causal_offsets=q_causal_offsets,
        d_index_k=_HashableTensor(
            paddle.zeros(index_k.shape, dtype=paddle.float32)
        ),
    )
    return (
        out["d_index_q"],
        out["d_weights"],
        out["d_index_k"].cast(index_k.dtype),
    )


def dense_attn_kl_scores(
    query: paddle.Tensor,
    kv: paddle.Tensor,
    lse: paddle.Tensor,
    softmax_scale: float,
    cu_seqlens_q: paddle.Tensor,
    cu_seqlens_k: paddle.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    q_causal_offsets: paddle.Tensor | None = None,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Head-summed attention probabilities, i.e. the raw KL target.

    Computes ``score[q, t] = sum_h exp(Q_h.K_t * softmax_scale - LSE_h)`` and
    ``l1norm[q] = sum_t score[q, t]``, so ``score / l1norm[:, None]`` is the
    uniform head mixture ``(1/H) sum_h softmax_h`` the DSA KL uses as its
    target -- **provided** ``lse`` is the true per-head log-sum-exp over the
    same candidate set. Any other per-head constant silently produces a
    *mass-weighted* mixture instead, so ``lse`` is not a free normaliser.

    Args:
        query: ``[s_local, H, D]`` bf16 absorbed queries (local rows).
        kv: ``[s_global, D]`` bf16 latent keys (globally gathered).
        lse: ``[s_local, H]`` fp32 per-head LSE over the candidate set. Rows set
            to ``+inf`` come back as an all-zero score row, which is how an
            inactive row is switched out of the loss.
        softmax_scale: the attention scale, unchanged from the forward.

    Returns:
        ``(score [s_local, max_seqlen_k] fp32, l1norm [s_local] fp32)``.
        Masked positions are ``-inf`` in ``score`` while ``l1norm`` only sums
        the valid ones, so callers must ``clip(min=0)`` before normalising.
    """
    _require_cudnn_frontend()
    from paddlefleet_ops.cudnn.deepseek_sparse_attention.score_recompute.api import (
        dense_attn_score_recompute_wrapper,
    )

    res = dense_attn_score_recompute_wrapper(
        _HashableTensor(query.contiguous()),
        _HashableTensor(kv.unsqueeze(1).contiguous()),
        _HashableTensor(lse.contiguous()),
        float(softmax_scale),
        qhead_per_kv_head=int(query.shape[1]),
        ratio=1,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=int(max_seqlen_q),
        max_seqlen_k=int(max_seqlen_k),
        q_causal_offsets=q_causal_offsets,
    )
    return res["out"], res["denom"]
