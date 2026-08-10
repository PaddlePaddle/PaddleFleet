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

"""Paddle wrapper around the cuDNN-frontend DSA indexer forward.

Calls ``paddlefleet_ops.cudnn.deepseek_sparse_attention.indexer_forward.api
.indexer_forward_wrapper`` and ``paddlefleet_ops.cudnn.deepseek_sparse_attention
.indexer_top_k.api.indexer_top_k_wrapper`` directly on Paddle tensors.

Returns selected compressed KV indices and per-row valid counts.
"""

from __future__ import annotations

import paddle
from paddlefleet_ops import CUDNN_FRONTEND_HINT, is_cudnn_frontend_available

# Packed-global fallback materializes a dense ``[B, S_q, S_k]`` fp32 score
# matrix plus left-align gather buffers. At 128K with CP that peaks at tens of
# GiB (e.g. S_q=S_k=32768 => 4GiB scores + ~16GiB int64 gather indices). Tiling
# the query dimension bounds the peak to ``O(tile * S_k)`` without changing the
# result: the indexer forward and radix top-k are both per-query-row
# independent, so a tile's output equals the matching slice of the full run.
# Default target keeps one fp32 score tile near 256MiB (tile*S_k <= 64Mi elems).
_DEFAULT_QUERY_TILE_ELEMS = 1 << 26


def _resolve_indexer_query_tile(sq: int, sk: int) -> int:
    """Query-dim tile size bounding the dense score matrix per kernel call."""
    sq = int(sq)
    if sk <= 0:
        return sq
    tile = max(1, _DEFAULT_QUERY_TILE_ELEMS // int(sk))
    return min(tile, sq)


def _require_cudnn_frontend():
    if not is_cudnn_frontend_available():
        raise ImportError(CUDNN_FRONTEND_HINT)


def _check_cudnn_indexer_shape_support(
    index_q, index_k_comp, ratio, seq_offset=0
):
    """Guard host-side shape contracts the cuDNN indexer forward cannot honor.

    The cuDNN CSA indexer forward kernel has historically crashed for short
    compressed-KV shapes: ``S_k == 1`` triggers ``cudaErrorIllegalInstruction``
    (715) rather than failing cleanly. Keep this cheap host-side assert so an
    unsupported short-sequence case fails clearly instead of poisoning the CUDA
    context. TileLang / pure-Paddle remain the recommended backends for
    standalone short sequences.

    cuDNN-frontend v1.26 no longer needs a host-side ``S_q + seq_offset <=
    S_k * ratio`` guard: the SM100 kernel clamps the ratio-causal block count to
    ``seqlen_k`` and skipped/masked positions remain ``-inf``. Tail query rows in
    non-ratio-aligned sequences are therefore valid and should be allowed.
    """
    sk = int(index_k_comp.shape[1])
    if sk < 2:
        raise ValueError(
            "cuDNN CSA indexer currently requires compressed KV length >= 2; "
            f"got S_k={sk}. Use the TileLang/Paddle indexer for short sequences."
        )


def _validate_indexer_inputs(index_q, index_k_comp, weights):
    if not isinstance(index_q, paddle.Tensor):
        raise TypeError(
            f"index_q must be a paddle.Tensor, got {type(index_q)!r}"
        )
    if not isinstance(index_k_comp, paddle.Tensor):
        raise TypeError(
            f"index_k_comp must be a paddle.Tensor, got {type(index_k_comp)!r}"
        )
    if not isinstance(weights, paddle.Tensor):
        raise TypeError(
            f"weights must be a paddle.Tensor, got {type(weights)!r}"
        )
    if len(index_q.shape) != 4:
        raise ValueError(
            f"index_q must have shape [B, S, H_i, D_i], got {index_q.shape}"
        )
    if len(index_k_comp.shape) != 3:
        raise ValueError(
            f"index_k_comp must have shape [B, S_comp, D_i], got {index_k_comp.shape}"
        )
    if len(weights.shape) != 3:
        raise ValueError(
            f"weights must have shape [B, S, H_i], got {weights.shape}"
        )

    batch, seq_len, heads, dim = index_q.shape
    batch_k, _, dim_k = index_k_comp.shape
    batch_w, seq_len_w, heads_w = weights.shape
    if batch != batch_k or batch != batch_w:
        raise ValueError(
            f"batch mismatch: index_q={index_q.shape}, "
            f"index_k_comp={index_k_comp.shape}, weights={weights.shape}"
        )
    if seq_len != seq_len_w or heads != heads_w or dim != dim_k:
        raise ValueError(
            f"shape mismatch: index_q={index_q.shape}, "
            f"index_k_comp={index_k_comp.shape}, weights={weights.shape}"
        )
    if heads not in (32, 64):
        raise ValueError(
            f"cuDNN IndexerForward requires H_i (qhead_per_kv_head) in {{32, 64}}, got {heads}"
        )
    if dim != 128:
        raise ValueError(f"cuDNN IndexerForward requires D_i=128, got {dim}")


def _indexer_top_k_unfused(
    input_values: paddle.Tensor,
    seq_lens: paddle.Tensor,
    top_k: int,
    return_val: bool = True,
):
    """
    Deterministic topk in replacement of cudnn indexer_top_k_wrapper.

    Expects input_values be masked by seq_lens (as indexer_forward_wrapper does).
    """
    # Note: paddle.topk doesn't allow k greater than the axis size.
    k = min(top_k, input_values.shape[-1])
    topk_values, topk_indices = paddle.topk(input_values, k, axis=-1)
    topk_indices = topk_indices.astype("int32")

    if k < top_k:
        topk_values = paddle.nn.functional.pad(
            topk_values, (0, 0, 0, top_k - k), value=float("-inf")
        )
        topk_indices = paddle.nn.functional.pad(
            topk_indices, (0, 0, 0, top_k - k), value=-1
        )

    # Entries outside each row's valid prefix [0, seq_lens) are already -inf.
    # Since topk returns results in descending order, these invalid entries occupy
    # output slots [seq_lens, top_k), whose indices can therefore be set to -1.
    topk_indices = paddle.where(
        paddle.arange(top_k, dtype="int32") < seq_lens[:, None],
        topk_indices,
        paddle.full_like(topk_indices, -1),
    )

    return {
        "indices": topk_indices,
        "values": topk_values if return_val else None,
    }


def cudnn_indexer_forward(
    index_q, index_k_comp, weights, ratio=4, sm_scale=None, seq_offset=0
):
    """Compute indexer scores using cuDNN CuTe-DSL kernel (SM100).

    Args:
        index_q:       [B, S_q, H_i, D_i] bf16, indexer queries.
        index_k_comp:  [B, S_k, D_i] bf16, compressed indexer keys.
        weights:       [B, S_q, H_i] bf16, per-head weights.
        ratio:         compression ratio for the causal mask.
        sm_scale:      scale factor applied to QK scores (default: dim**-0.5).

    Returns:
        scores: [B, S_q, S_k] fp32 Paddle tensor. Masked positions are -inf.
    """
    seq_offset = int(seq_offset)
    if seq_offset < 0:
        raise ValueError(f"seq_offset must be >= 0, got {seq_offset}")
    _check_cudnn_indexer_shape_support(
        index_q, index_k_comp, ratio, seq_offset=seq_offset
    )
    if sm_scale is None:
        sm_scale = float(index_q.shape[-1]) ** -0.5
    _require_cudnn_frontend()
    from paddlefleet_ops.cudnn.deepseek_sparse_attention.indexer_forward.api import (
        indexer_forward_wrapper,
    )

    q_causal_offsets = None
    if seq_offset != 0:
        q_causal_offsets = paddle.full(
            [int(index_q.shape[0])],
            seq_offset,
            dtype="int32",
        )

    result = indexer_forward_wrapper(
        index_q.contiguous(),
        index_k_comp.unsqueeze(2).contiguous(),
        weights.contiguous(),
        ratio=int(ratio),
        sm_scale=float(sm_scale),
        q_causal_offsets=q_causal_offsets,
    )
    return result["scores"]


def cudnn_indexer_topk(scores, sq, ratio, topk, valid_range=None, seq_offset=0):
    """Select top-K indices using cuDNN TRT-LLM radix kernel (SM100).

    Args:
        scores:  [B, S_q, S_k] fp32 Paddle tensor.
        sq:      query sequence length.
        ratio:   compression ratio.
        topk:    number of entries to select per query position.
        valid_range: optional [B, S_q, 2] int32 per-query left-closed
            compressed-KV range ``[valid_start, valid_end)`` for document-mask
            (packed multi-document) training. ``None`` => causal-only mode
            (legacy single-document behavior, byte-for-byte unchanged).

    Returns:
        topk_indices: [B, S_q, topk] int32 **global** compressed-buffer ids,
            invalid slots are -1.
        topk_length:  [B, S_q] int32, per-row valid count.
    """
    batch = int(scores.shape[0])
    sk = int(scores.shape[2])
    sq = int(sq)
    topk = int(topk)
    seq_offset = int(seq_offset)
    topk_k = min(topk, sk)

    if valid_range is None:
        # Causal-only (single-document): the radix kernel's per-row prefix
        # length is exactly the ratio-causal limit. No id remap needed —
        # local == global because there is a single compressed buffer.
        q_idx = paddle.arange(seq_offset, seq_offset + sq, dtype="int32")
        seq_lens = paddle.clip((q_idx + 1) // int(ratio), max=sk).tile([batch])
        scores_for_topk = scores
        valid_range_for_remap = None
    else:
        # Document-mask: the valid window [valid_start, valid_end) is an
        # arbitrary sub-interval, but the radix kernel only honors prefixes
        # [0, seq_lens). Left-align each query's window to [0, count), run
        # top-k in that local space, then map the selected local ids back to
        # global compressed-buffer ids by adding valid_start.
        from .docmask_utils import (
            shift_scores_to_local_window,
            topk_local_to_global,
        )

        if valid_range.shape[0] != batch or valid_range.shape[1] != sq:
            raise ValueError(
                f"valid_range must have shape [{batch}, {sq}, 2], got "
                f"{list(valid_range.shape)}"
            )
        scores_for_topk, counts = shift_scores_to_local_window(
            scores, valid_range
        )
        seq_lens = counts.reshape([batch * sq]).cast("int32")
        valid_range_for_remap = valid_range

    result = _indexer_top_k_unfused(
        scores_for_topk.reshape([batch * sq, sk]).contiguous(),
        seq_lens,
        top_k=topk_k,
        return_val=False,
    )
    topk_indices = result["indices"].reshape([batch, sq, topk_k]).cast("int32")

    if valid_range_for_remap is not None:
        # local (per-document, [0, count)) -> global; -1 slots preserved.
        topk_indices = topk_local_to_global(topk_indices, valid_range_for_remap)

    if topk_k < topk:
        padding = paddle.full([batch, sq, topk - topk_k], -1, dtype="int32")
        topk_indices = paddle.concat([topk_indices, padding], axis=-1)

    topk_length = (topk_indices >= 0).sum(axis=-1).cast("int32")
    return topk_indices, topk_length


def cudnn_indexer_topk_fwd(
    index_q,
    index_k_comp,
    weights,
    ratio=4,
    topk_effective=64,
    indexer_softmax_scale=1.0,
    valid_range=None,
    startend_row_indices=None,
    doc_lens=None,
    seq_offset=0,
    return_topk_scores=False,
):
    """Run cuDNN-frontend DSA indexer forward on Paddle tensors.

    Args:
        index_q:                [B, S, H_i, D_i] bf16, indexer queries.
        index_k_comp:           [B, S_comp, D_i] bf16, compressed indexer keys.
        weights:                [B, S, H_i] bf16, per-head weights.
        ratio:                  compression ratio (e.g. 4).
        topk_effective:         number of entries to select per query position.
        indexer_softmax_scale:  additional scale on weights.
        valid_range:            optional [B, S, 2] int32 per-query left-closed
            compressed-KV range for document-mask (packed multi-document)
            training. ``None`` => causal-only single-document mode (unchanged).

        startend_row_indices: optional [1, S, 1] doc end metadata. When present
            with ``valid_range`` and ``seq_offset == 0``, the docmask path uses
            cuDNN THD/varlen forward so score computation is document-local
            instead of packed-global. CP docmask uses the packed-global fallback
            because local query slices do not match the global docmask length.
        doc_lens: optional precomputed document lengths from reusable CSA
            docmask metadata. When supplied, the THD docmask path reuses it
            instead of reparsing ``startend_row_indices``.
        seq_offset: global query position offset for CP causal-only mode.
        return_topk_scores: return selected raw scores as a third output. This
            avoids gathering from a packed-global score tensor on the THD path.

    Returns:
        topk_indices: [B, S, topk_effective] int32 global compressed-buffer ids,
            invalid slots are -1.
        topk_length:  [B, S] int32, per-row valid count.
        topk_scores:  optional [B, S, topk_effective] fp32 selected scores.
    """
    return _cudnn_indexer_topk_fwd_impl(
        index_q,
        index_k_comp,
        weights,
        ratio=ratio,
        topk_effective=topk_effective,
        indexer_softmax_scale=indexer_softmax_scale,
        valid_range=valid_range,
        startend_row_indices=startend_row_indices,
        doc_lens=doc_lens,
        seq_offset=seq_offset,
        return_topk_scores=return_topk_scores,
    )


def _cudnn_indexer_topk_fwd_impl(
    index_q,
    index_k_comp,
    weights,
    ratio=4,
    topk_effective=64,
    indexer_softmax_scale=1.0,
    valid_range=None,
    startend_row_indices=None,
    doc_lens=None,
    seq_offset=0,
    return_topk_scores=False,
):
    _validate_indexer_inputs(index_q, index_k_comp, weights)
    if int(topk_effective) <= 0:
        raise ValueError(
            f"topk_effective must be positive, got {topk_effective}"
        )
    seq_offset = int(seq_offset)
    if seq_offset < 0:
        raise ValueError(f"seq_offset must be >= 0, got {seq_offset}")

    # sm_scale combines base dim**-0.5 with any additional indexer_softmax_scale
    _sm = float(index_q.shape[-1]) ** -0.5
    if float(indexer_softmax_scale) != 1.0:
        _sm = _sm * float(indexer_softmax_scale)

    if valid_range is not None and doc_lens is not None:
        thd_result = _cudnn_indexer_topk_fwd_docmask_thd(
            index_q,
            index_k_comp,
            weights,
            ratio,
            topk_effective,
            _sm,
            valid_range,
            seq_offset,
            doc_lens=doc_lens,
            return_topk_scores=return_topk_scores,
        )
        if thd_result is not None:
            return thd_result

    sq_total = int(index_q.shape[1])
    sk = int(index_k_comp.shape[1])
    query_tile = _resolve_indexer_query_tile(sq_total, sk)

    if query_tile >= sq_total:
        return _dense_indexer_topk_single(
            index_q,
            index_k_comp,
            weights,
            ratio,
            topk_effective,
            _sm,
            valid_range,
            seq_offset,
            return_topk_scores,
        )

    # Tile the query dimension: each tile's forward + top-k is independent of
    # the others, so concatenating tile outputs reproduces the single-shot
    # result byte-for-byte while capping peak memory at ``O(query_tile * S_k)``.
    idx_parts = []
    len_parts = []
    score_parts = [] if return_topk_scores else None
    for start in range(0, sq_total, query_tile):
        end = min(start + query_tile, sq_total)
        vr_chunk = None if valid_range is None else valid_range[:, start:end]
        chunk = _dense_indexer_topk_single(
            index_q[:, start:end],
            index_k_comp,
            weights[:, start:end],
            ratio,
            topk_effective,
            _sm,
            vr_chunk,
            seq_offset + start,
            return_topk_scores,
        )
        if return_topk_scores:
            idx_chunk, len_chunk, score_chunk = chunk
            score_parts.append(score_chunk)
        else:
            idx_chunk, len_chunk = chunk
        idx_parts.append(idx_chunk)
        len_parts.append(len_chunk)

    topk_indices = paddle.concat(idx_parts, axis=1)
    topk_length = paddle.concat(len_parts, axis=1)
    if return_topk_scores:
        return topk_indices, topk_length, paddle.concat(score_parts, axis=1)
    return topk_indices, topk_length


def _make_cu_seqlens(
    doc_lens: list[int], seq_offset: int, seq_len: int, ratio: int
):
    """
    Generate CP-local cu_seqlens_q/k for the indexer_top_k kernel.

    The cu_seqlens_q only contains the document covered by this CP rank, with
    both left and right borders clipped. The cu_seqlens_k is similar, but only
    clips the right border because the left part is causal, thus visible.

    Example:
    Say there are 4 docs in this batch and the current CP rank covers part of
    doc1, the total doc2, and part of doc3. The cu_seqlens_q exactly contains
    the `seq_len` part of q, while the cu_seqlens_k extends its left bound to
    cover the full doc1.

    q: |<----- doc0 ----->|<-- doc1 -->|<--- doc2 --->|<---- doc3 ---->|
       {-----  seq_offset  -----}{------  seq_len  ------}

    k: |<----- doc0 ----->|<-- doc1 -->|<--- doc2 --->|<---- doc3 ---->|
                          {--------   visible_k  --------}
    """
    end_q = end_k = 0
    max_q = max_k = 0
    cu_seqlens_q = [0]  # q is local, always starts from 0
    cu_seqlens_k = []  # k is global, may be offsetted
    q_causal_offsets = []

    for n in doc_lens:
        start_q, start_k = end_q, end_k
        end_q += n
        end_k += n // ratio

        # skip documents that are fully before this CP rank
        if end_q <= seq_offset:
            continue

        cu_seqlens_q.append(min(end_q - seq_offset, seq_len))
        q_causal_offsets.append(max(seq_offset - start_q, 0))

        # the last document in this CP rank can only see the causal part of k
        n_visible = min(seq_offset + seq_len - start_q, n)
        n_visible_comp = n_visible // ratio
        max_q = max(n_visible, max_q)
        max_k = max(n_visible_comp, max_k)

        if not cu_seqlens_k:
            cu_seqlens_k.append(start_k)
        cu_seqlens_k.append(start_k + n_visible_comp)

        # break when the end of the current document exceeds this CP rank
        if end_q >= seq_offset + seq_len:
            break

    cu_seqlens_q = paddle.to_tensor(cu_seqlens_q, "int32")
    cu_seqlens_k = paddle.to_tensor(cu_seqlens_k, "int32")
    q_causal_offsets = (
        paddle.to_tensor(q_causal_offsets, "int32") if seq_offset > 0 else None
    )  # not necessary for non-CP
    return cu_seqlens_q, cu_seqlens_k, max_q, max_k, q_causal_offsets


def _cudnn_indexer_topk_fwd_docmask_thd(
    index_q: paddle.Tensor,
    index_k_comp: paddle.Tensor,
    weights: paddle.Tensor,
    ratio: int,
    topk_effective: int,
    sm_scale: float,
    valid_range: paddle.Tensor,
    seq_offset: int,
    doc_lens: list[int],
    return_topk_scores: bool = False,
):
    batch, seq_len, _, _ = index_q.shape
    if batch != 1:
        return None

    _require_cudnn_frontend()
    from paddlefleet_ops.cudnn.deepseek_sparse_attention.indexer_forward.api import (
        indexer_forward_wrapper,
    )

    from .docmask_utils import topk_local_to_global, valid_range_to_counts

    cu_q, cu_k, max_q, max_k, q_causal_offsets = _make_cu_seqlens(
        doc_lens, seq_offset, seq_len, ratio
    )
    # Fallback to dense path if there is no visible doc in this CP rank.
    if len(cu_k) == 0:
        return None

    # The indexer kernel requires the last dim (max_k) padded to 16-byte.
    # If we don't pad it here, the kernel will allocate a padded scratch buffer
    # and copying results back, which causes extra contiguous kernels.
    TMA_ALIGN_ELEMS = 4
    max_k = (max_k + TMA_ALIGN_ELEMS - 1) // TMA_ALIGN_ELEMS * TMA_ALIGN_ELEMS

    q_thd = index_q.squeeze(0)  # [local_seqlen, heads, dim]
    # [global_seqlen // ratio, 1, dim]
    k_thd = index_k_comp.squeeze(0).unsqueeze(1)
    w_thd = weights.squeeze(0)  # [local_seqlen, heads]

    # Compute scores in THD compacted shape [local_seqlen, max_k]
    scores = indexer_forward_wrapper(
        q_thd,
        k_thd,
        w_thd,
        ratio=ratio,
        sm_scale=sm_scale,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
        q_causal_offsets=q_causal_offsets,
    )["scores"]

    counts = valid_range_to_counts(valid_range)

    # Compute topk indices [local_seqlen, topk]
    results = _indexer_top_k_unfused(
        scores,
        counts.squeeze(0),
        top_k=topk_effective,
        return_val=return_topk_scores,
    )

    topk_global = topk_local_to_global(
        results["indices"].unsqueeze(0), valid_range
    )
    topk_length = counts.clip(max=topk_effective)

    if return_topk_scores:
        topk_scores = results["values"].unsqueeze(0)
        return topk_global, topk_length, topk_scores
    return topk_global, topk_length


def _dense_indexer_topk_single(
    index_q,
    index_k_comp,
    weights,
    ratio,
    topk_effective,
    sm_scale,
    valid_range,
    seq_offset,
    return_topk_scores,
):
    """Single-shot packed-global forward + top-k over the full query slice."""
    scores = cudnn_indexer_forward(
        index_q,
        index_k_comp,
        weights,
        ratio=ratio,
        sm_scale=sm_scale,
        seq_offset=seq_offset,
    )
    topk_indices, topk_length = cudnn_indexer_topk(
        scores,
        int(index_q.shape[1]),
        ratio,
        topk_effective,
        valid_range=valid_range,
        seq_offset=seq_offset,
    )
    if not return_topk_scores:
        return topk_indices, topk_length

    invalid_mask = topk_indices < 0
    safe_indices = paddle.where(
        invalid_mask, paddle.zeros_like(topk_indices), topk_indices
    )
    topk_scores = paddle.take_along_axis(
        scores, safe_indices.cast("int64"), axis=2
    )
    topk_scores = paddle.where(
        invalid_mask,
        paddle.full_like(topk_scores, float("-inf")),
        topk_scores,
    )
    return topk_indices, topk_length, topk_scores
