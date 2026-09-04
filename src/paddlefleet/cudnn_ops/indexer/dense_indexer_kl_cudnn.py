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


# Query-head counts the two dense score epilogues can tile. This is a property
# of the device, not of the op, because the two arch paths are separate kernels
# (``api.py`` dispatches on ``get_device_capability``).
#
# On SM100+ both epilogues address TMEM
# through ``tcgen05.copy.Repetition(m_block_size // 4)``
# (``dense_score_recompute_sm100.py:163,1043,1332``), and that enum admits only
# powers of two -- x1 .. x128 (``cutlass/cute/nvgpu/tcgen05/copy.py:46``). The
# tile dispatch derives ``m_block_size = qhead_per_kv_head * 2``, falling back
# to ``* 1`` when shared memory is tight (``_interface_sm100.py:699-713``), so a
# head count that is not a power of two has no valid tile at all: ``h == 24``
# with ``head_dim == 576`` picks ``m_block_size = 48`` and dies with
# ``ValueError: 12 is not a valid Repetition`` raised from inside the CuTe
# trace, i.e. from wherever the recompute segment is replayed rather than from
# the call site.
#
# 16 rather than 4 as the SM100+ floor: the attention epilogue unrolls its head
# sum by ``2 * LSE_ILP == 8`` (``dense_score_recompute_sm100.py:1378-1382``) and
# the indexer one by ``2 * W_ILP == 16`` (``:1095-1097``), both with a
# truncating ``range_constexpr``, so narrower tiles start dropping heads from
# the sum. 16 is also where the sparse sibling kernel stops returning an
# all-zero target (``mqa_latent_attention._TARGET_QHEAD_MIN``).
#
# SM90 has neither constraint. It caps ``tile_m`` at 64 and loops over
# ``qhead_per_kvhead // tile_m`` head tiles
# (``_interface_sm90._compute_tile_m``), with no TMEM copy and no fixed-width
# head-sum unroll, so it serves any width up to 64 in a single tile and any
# multiple of 64 above that -- ``index_n_heads == 192`` included. Holding it to
# the SM100+ rule would reject widths its kernel handles natively, so the
# supported set is computed per device instead.
#
# Its one floor is a single head: the same ``_compute_tile_m`` opens with
# ``assert qhead_per_kvhead > 1, "SM90 kernel requires MQA/GQA"``, and
# ``_validate_and_prepare_*`` re-derives the ratio as ``num_head // num_head_kv``
# behind ``assert num_head > num_head_kv`` (``_interface_sm90.py:71,129,338``).
# Since these callers hand the kernel a single latent KV head, ``h == 1`` trips
# both. Two is enough to clear them and is the narrowest pad that does.
_DENSE_SCORE_QHEAD_MIN = 16
_SM90_HEAD_TILE = 64
_SM90_QHEAD_MIN = 2


def _dense_score_qheads(num_heads: int) -> int:
    """Narrowest width this device's dense score can tile ``num_heads`` at."""
    num_heads = int(num_heads)
    major, _ = paddle.device.cuda.get_device_capability()
    if major < 10:
        if num_heads <= _SM90_HEAD_TILE:
            return max(_SM90_QHEAD_MIN, num_heads)
        return -(-num_heads // _SM90_HEAD_TILE) * _SM90_HEAD_TILE
    return max(_DENSE_SCORE_QHEAD_MIN, 1 << (num_heads - 1).bit_length())


def _require_dense_score_qheads(num_heads: int, name: str) -> None:
    """Reject a head count the dense score cannot tile, from the call site.

    The indexer pair is checked rather than padded: unlike the attention score,
    whose outputs are head-reduced, ``dense_indexer_kl_bwd`` returns per-head
    ``d_index_q`` / ``d_weights``, so padding would have to be threaded through
    the backward's tiling as well for no gain -- every indexer this path is built
    for runs at ``index_n_heads == 64``, which is a supported width on both arch
    paths (and ``mqa_latent_attention._check_cudnn_dense_indexer_support``
    rejects narrower ones at the layer). This exists so a future config lands on
    a named error instead of ``N is not a valid Repetition`` from inside a
    recompute replay.
    """
    num_heads = int(num_heads)
    supported = _dense_score_qheads(num_heads)
    if num_heads != supported:
        raise ValueError(
            f"the dense cuDNN score kernel tiles its MMA ``M`` on {name}, and "
            f"the narrowest tile this device can cover {num_heads} heads with "
            f"is {supported}, so {name} must be {supported}, got {num_heads}. "
            "On SM100+ an untileable width instead fails as 'N is not a valid "
            "Repetition' from inside the CuTe trace, i.e. from wherever the "
            "segment is replayed."
        )


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

    _require_dense_score_qheads(index_q.shape[1], "index_n_heads")
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


def _pad_attn_kl_heads(
    query: paddle.Tensor, lse: paddle.Tensor
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Widen ``query``/``lse`` to a head count the dense attention score tiles.

    Zeros for the query pad and ``+inf`` for its LSE, which is the same pair the
    sparse sibling path uses (``mqa_latent_attention._attn_target_cudnn``) and it
    is exact rather than approximate here:

    * A pad head scores ``0`` against every key (its query row is zeros), so the
      epilogue evaluates ``0 * scale - inf = -inf``, clamps it to
      ``EXP2_ARG_MIN = -126`` (``dense_score_recompute_sm100.py:1403``) and adds
      ``exp2(-126) = 1.18e-38`` to that column's head sum. At the production
      geometry a real column sum is ~3.7e-4, whose fp32 ULP is ~4.4e-11, so
      eight pad heads move the result by 27 orders of magnitude less than one
      unit in the last place: the fp32 accumulation is bit-identical.
    * The ``l1norm`` denominator accumulates the same per-column sums, so it
      inherits the same non-perturbation.
    * Columns outside the causal candidate set are force-zeroed by the kernel
      before they reach either accumulator (``:1419-1421``), pad heads included.

    Nothing is sliced back off: both kernel outputs are already head-reduced
    (``[s_local, max_seqlen_k]`` and ``[s_local]``), so the pad heads leave no
    trace in the returned shapes. And this function is on the *score* path only,
    which carries no sink and no forced window (see
    ``mqa_latent_attention._dense_kl_attn_lse``) -- the sink-bearing DSA paths do
    their own head padding with ``_NEG_SINK`` in ``fusions/mqa_sparse_attn.py``
    and ``fusions/csa_sparse_attn.py``.

    Called from both the forward and the backward of the warmup KL, which is why
    it lives here rather than at either call site: the two must hand the kernel
    the same head count or the backward would build its gradient on a target the
    forward never measured.
    """
    h = int(query.shape[1])
    h_padded = _dense_score_qheads(h)
    if h_padded == h:
        return query, lse
    pad = h_padded - h
    s_local, head_dim = int(query.shape[0]), int(query.shape[2])
    query = paddle.concat(
        [query, paddle.zeros([s_local, pad, head_dim], dtype=query.dtype)],
        axis=1,
    )
    lse = paddle.concat(
        [
            lse.cast("float32"),
            paddle.full(
                [int(lse.shape[0]), pad], float("inf"), dtype="float32"
            ),
        ],
        axis=1,
    )
    return query, lse


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
        query: ``[s_local, H, D]`` bf16 absorbed queries (local rows). ``H`` is
            padded up to a supported tile width here when it is not one already;
            see :func:`_dense_score_qheads`.
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

    query, lse = _pad_attn_kl_heads(query, lse)
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
