# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""
Compressed Sparse Attention (CSA) for DeepSeekV4 Hybrid Attention.

Ported from Megatron-LM experimental_attention_variant/csa.py (commit bf4e1db).

Components:
  - Compressor: Gated pooling compressor with overlap (ratio=4) or non-overlap (ratio=128)
  - CSAIndexer: Learned top-k retrieval over compressed positions
  - CompressedSparseAttention: Core attention combining sliding window + compressed KV
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING

import paddle
import paddle.nn.functional as F
from paddle import Tensor, nn
from paddle.distributed.fleet.meta_parallel import LayerSpec, build_spec_layer

from paddlefleet.models.common.embeddings.rope_utils import (
    _apply_rotary_pos_emb_bshd,
)
from paddlefleet.transformer import FleetLayer
from paddlefleet.transformer.dsa_attention import (
    DSAIndexerLossAutoScaler,
    DSAIndexerLossLoggingHelper,
    FusedDSAIndexerLoss,
    fused_qk_topk_naive,
    rotate_activation,
)

if TYPE_CHECKING:
    from paddlefleet.process_groups_config import ProcessGroupCollection
    from paddlefleet.transformer.enums import AttnMaskType
    from paddlefleet.transformer.transformer_config import TransformerConfig

# ---------------------------------------------------------------------------
# Helper functions for index computation
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=8)
def _get_window_topk_idxs_cached(
    window_size: int, seqlen: int, device_str: str
) -> Tensor:
    """Compute sliding window indices for a single sequence (cached)."""
    base = paddle.arange(seqlen).unsqueeze(1)  # [seqlen, 1]
    offsets = paddle.arange(window_size)  # [window_size]
    matrix = paddle.clip(base - window_size + 1, min=0) + offsets
    matrix = paddle.where(matrix > base, paddle.full_like(matrix, -1), matrix)
    return matrix


@functools.lru_cache(maxsize=8)
def _get_compress_topk_idxs_cached(
    ratio: int, seqlen: int, offset: int, device_str: str
) -> Tensor:
    """Compute compressed indices for a single sequence (cached)."""
    n_compressed = seqlen // ratio
    k_indices = paddle.arange(n_compressed)
    matrix = k_indices.unsqueeze(0).expand([seqlen, -1])
    causal_bound = paddle.arange(1, seqlen + 1).unsqueeze(1) // ratio
    causal_invalid = matrix >= causal_bound
    matrix = paddle.where(
        causal_invalid, paddle.full_like(matrix, -1), matrix + offset
    )
    return matrix


def get_window_topk_idxs(
    window_size: int,
    batch_size: int,
    seqlen: int,
    device=None,
) -> Tensor:
    """Get sliding window indices: [b, seqlen, window_size]."""
    indices = _get_window_topk_idxs_cached(window_size, seqlen, "gpu")
    return indices.unsqueeze(0).expand([batch_size, -1, -1])


def _build_compressed_causal_mask(
    ratio: int,
    batch_size: int,
    seqlen: int,
    n_compressed: int,
) -> Tensor:
    compressed_ids = paddle.arange(n_compressed).unsqueeze(0)
    positions = paddle.arange(1, seqlen + 1).unsqueeze(1)
    invalid = compressed_ids >= (positions // ratio)
    invalid = invalid.unsqueeze(0).expand([batch_size, seqlen, n_compressed])
    return paddle.where(
        invalid,
        paddle.full([1], float("-inf"), dtype="float32"),
        paddle.zeros([1], dtype="float32"),
    )


def get_compress_topk_idxs(
    ratio: int,
    batch_size: int,
    seqlen: int,
    offset: int,
    device=None,
) -> Tensor:
    """Get compressed indices: [b, seqlen, seqlen // ratio]."""
    matrix = _get_compress_topk_idxs_cached(ratio, seqlen, offset, "gpu")
    return matrix.unsqueeze(0).expand([batch_size, -1, -1])


# ---------------------------------------------------------------------------
# RoPE helper for CSA
# ---------------------------------------------------------------------------


def _apply_rope(
    x: Tensor,
    nope_dim: int,
    pos_dim: int,
    rotary_pos_emb_module,
    config: TransformerConfig,
    rotary_seq_len: int,
    ratio: int = 1,
) -> Tensor:
    """Apply RoPE to the last pos_dim dims, leaving first nope_dim unchanged.

    For compressed positions (ratio > 1), subsamples the RoPE frequencies
    by taking every ratio-th position.

    Args:
        x: [b, seq, ...dim...] where last dim = nope_dim + pos_dim
        nope_dim: dimensions that don't get RoPE
        pos_dim: dimensions that get RoPE
        rotary_pos_emb_module: RotaryEmbedding instance
        config: transformer config
        rotary_seq_len: sequence length for this tensor
        ratio: compression ratio for position subsampling
    """
    total_seq_len = rotary_seq_len * ratio if ratio > 1 else rotary_seq_len
    result = rotary_pos_emb_module(total_seq_len, packed_seq=False)
    if isinstance(result, tuple):
        freqs, mscale = result
    else:
        freqs, mscale = result, 1.0
    # freqs: [1, total_seq_len, pos_dim]
    if ratio > 1:
        freqs = freqs[:, :total_seq_len:ratio, :][:, :rotary_seq_len, :]

    squeeze_head = x.ndim == 3
    if squeeze_head:
        x = x.unsqueeze(2)  # [b, s, 1, dim]

    x_nope = x[..., :nope_dim]
    x_pe = x[..., nope_dim:]

    x_pe = _apply_rotary_pos_emb_bshd(
        x_pe,
        freqs,
        mscale=mscale,
        rotary_interleaved=False,
        multi_latent_attention=True,
        mla_output_remove_interleaving=True,
    )

    out = paddle.concat([x_nope, x_pe], axis=-1)
    if squeeze_head:
        out = out.squeeze(2)
    return out


# ---------------------------------------------------------------------------
# Unfused compressed sparse attention
# ---------------------------------------------------------------------------


def unfused_compressed_sparse_attn(
    query: Tensor,
    kv_full: Tensor,
    attn_sink: Tensor,
    topk_indices: Tensor,
    softmax_scale: float,
) -> Tensor:
    """Sparse attention with MQA and learnable attention sink.

    Args:
        query: [b, sq, np, hn] multi-head query
        kv_full: [b, n_kv, hn] single-head KV (original + compressed concatenated)
        attn_sink: [np] per-head learnable bias (attention sink)
        topk_indices: [b, sq, topk] indices into kv_full dim=1 (-1 = invalid)
        softmax_scale: attention scale factor

    Returns:
        output: [b, sq, np * hn]
    """
    b, sq, np_heads, hn = query.shape
    topk = topk_indices.shape[-1]

    # Clamp negative indices to 0 for gathering, mask them later
    safe_indices = paddle.clip(topk_indices, min=0).cast(
        paddle.int64
    )  # [b, sq, topk]
    safe_indices_exp = safe_indices.unsqueeze(-1).expand(
        [-1, -1, -1, hn]
    )  # [b, sq, topk, hn]

    # Gather KV at selected positions: [b, n_kv, hn] -> [b, sq, topk, hn]
    kv_gathered = paddle.gather(
        kv_full.unsqueeze(1).expand([-1, sq, -1, -1]),
        dim=2,
        index=safe_indices_exp,
    )

    # Compute attention scores: [b, np, sq, topk]
    q = query.transpose([0, 2, 1, 3])  # [b, np, sq, hn]
    # scores = einsum("bnsh,bskh->bnsk", q, kv_gathered)
    scores = (
        paddle.einsum(
            "bnsh,bskh->bnsk", q.cast("float32"), kv_gathered.cast("float32")
        )
        * softmax_scale
    )  # [b, np, sq, topk]

    # Mask invalid positions (topk_indices < 0) with -inf
    invalid_mask = (topk_indices < 0).unsqueeze(1)  # [b, 1, sq, topk]
    scores = scores.masked_fill(invalid_mask, float("-inf"))

    # Softmax with attention sink
    # sink: [np] -> [1, np, 1, 1]
    sink = attn_sink.reshape([1, np_heads, 1, 1])
    # Compute stable softmax: max over scores and sink
    scores_max = scores.max(axis=-1, keepdim=True)  # [b, np, sq, 1]
    scores_max = paddle.maximum(scores_max, sink)

    exp_scores = paddle.exp(scores - scores_max)  # [b, np, sq, topk]
    exp_sink = paddle.exp(sink - scores_max)  # [b, np, sq, 1]

    sum_exp = exp_scores.sum(axis=-1, keepdim=True) + exp_sink  # [b, np, sq, 1]
    attn_weights = exp_scores / sum_exp  # [b, np, sq, topk]

    # Weighted sum: [b, np, sq, topk] x [b, sq, topk, hn] -> [b, np, sq, hn]
    output = paddle.einsum(
        "bnsk,bskh->bnsh", attn_weights, kv_gathered.cast("float32")
    )
    output = output.cast(query.dtype)

    # Reshape: [b, np, sq, hn] -> [b, sq, np * hn]
    output = output.transpose([0, 2, 1, 3]).reshape([b, sq, np_heads * hn])
    return output


def _resolve_csa_indexer_loss_topk_effective(
    config, index_topk: int, n_compressed: int
) -> int:
    """Return the TileLang CSA indexer top-k width used by indexer loss.

    Phase semantics (driven by ``dsa_indexer_use_sparse_loss``, **not** by the
    new TileLang switches):

    * Phase 3 (``dsa_indexer_use_sparse_loss=True``): ``min(index_topk,
      n_compressed)`` — selected-topk semantics, same as the existing
      ``FusedDSAIndexerLoss`` / ``CSAIndexer.forward`` choice.
    * Phase 2 (``dsa_indexer_use_sparse_loss=False``): ``n_compressed`` — the
      selected set covers the full compressed candidate range and is later
      consumed as full-range KL by the indexer loss path.
    """
    use_sparse_loss = bool(getattr(config, "dsa_indexer_use_sparse_loss", True))
    if use_sparse_loss:
        return min(int(index_topk), int(n_compressed))
    return int(n_compressed)


def _resolve_csa_indexer_attn_topk_effective(
    index_topk: int, n_compressed: int
) -> int:
    """Return the compressed top-k width consumed by main CSA attention."""
    return min(int(index_topk), int(n_compressed))


def _resolve_csa_tilelang_switch(config, field_name: str) -> bool:
    backend_enabled = (
        getattr(config, "csa_tilelang_backend", None)
        == "attention_paddle_compat"
    )
    override = getattr(config, field_name, None)
    if override is None:
        return backend_enabled
    return bool(override)


def _map_compressed_topk_to_kv_full(
    topk_indices_compressed: Tensor,
    sq: int,
    ratio: int,
    offset: int,
) -> Tensor:
    """Map compressed block ids to ``kv_full`` indices.

    For each query position ``t``, only ``(t + 1) // ratio`` compressed blocks
    are causally valid. Slots whose compressed id is out of that range are
    written back as ``-1``; valid slots are shifted by ``offset`` (which is
    the original sequence length so that compressed entries follow the raw
    KV positions inside ``kv_full``).
    """
    n_valid_per_pos = (
        paddle.arange(1, sq + 1, dtype=topk_indices_compressed.dtype).unsqueeze(
            1
        )
        // ratio
    ).unsqueeze(0)  # [1, sq, 1]
    valid = (topk_indices_compressed >= 0) & (
        topk_indices_compressed < n_valid_per_pos
    )
    return paddle.where(
        valid,
        topk_indices_compressed + offset,
        paddle.full_like(topk_indices_compressed, -1),
    )


def _compute_attn_target_on_selected_set(
    query_mla: Tensor,  # [b, sq, np, hn]  DETACHED
    key_comp_mla: Tensor,  # [b, sk, hn] shared compressed KV, or legacy [b, sk, np, hn]
    topk_indices: Tensor,  # [b, sq, topk_eff] int32, -1 for invalid slots
    softmax_scale: float,
    tp_group=None,
) -> Tensor:
    """Construct attention target ``p[t, S_t]`` on the selected compressed set.

    Mathematically equivalent to ``_compute_dsa_indexer_loss`` with
    ``sparse_loss=True``, but evaluated only on the selected slots ``S_t``
    given by ``topk_indices`` instead of materializing the full ``[B,Sq,Sk]``
    distribution. Invalid (``-1``) slots are masked out before softmax and
    receive zero target probability after L1 normalization.

    The result has shape ``[b, sq, topk_eff]`` in fp32 and is the multi-head
    aggregated, L1 normalized target distribution used as the second argument
    of ``KL(target || index_prob)``.
    """
    b, sq, np, hn = query_mla.shape
    topk_eff = topk_indices.shape[-1]

    # Per-head full attention scores [b, np, sq, sk]. DSv4 compressed KV is
    # shared across query heads as [b, sk, hn]; the legacy per-head-expanded
    # [b, sk, np, hn] shape is still accepted for non-TileLang references.
    q = query_mla.transpose([0, 2, 1, 3]).cast("float32")  # [b, np, sq, hn]
    if len(key_comp_mla.shape) == 3:
        k = key_comp_mla.transpose([0, 2, 1]).cast("float32").unsqueeze(1)
    else:
        k = key_comp_mla.transpose([0, 2, 3, 1]).cast("float32")
    attn_scores = paddle.matmul(q, k) * float(softmax_scale)  # [b, np, sq, sk]

    # Replace -1 with 0 for safe gather; then mask back to -inf afterwards.
    valid = topk_indices >= 0  # [b, sq, topk_eff]
    safe_indices = paddle.where(
        valid, topk_indices, paddle.zeros_like(topk_indices)
    ).cast("int64")
    safe_indices_exp = safe_indices.unsqueeze(1).expand([b, np, sq, topk_eff])
    selected_logits = paddle.take_along_axis(
        attn_scores, safe_indices_exp, axis=-1
    )  # [b, np, sq, topk_eff]

    # Mask invalid slots so softmax assigns them zero probability.
    valid_bn = valid.unsqueeze(1)  # [b, 1, sq, topk_eff]
    neg_inf = paddle.full([1], float("-inf"), dtype="float32")
    selected_logits = paddle.where(valid_bn, selected_logits, neg_inf)

    # Avoid all-(-inf) rows producing NaN in softmax: zero such rows out.
    row_valid = valid.any(axis=-1, keepdim=True)  # [b, sq, 1]
    row_valid_bn = row_valid.unsqueeze(1)  # [b, 1, sq, 1]
    selected_logits = paddle.where(
        row_valid_bn, selected_logits, paddle.zeros_like(selected_logits)
    )

    probs = F.softmax(selected_logits, axis=-1, dtype="float32")
    # Re-zero fully invalid rows post-softmax (softmax of zeros is uniform).
    probs = probs * row_valid_bn.cast("float32")

    # Aggregate over heads, optional TP all-reduce, then L1 normalize.
    target = probs.sum(axis=1)  # [b, sq, topk_eff]
    if tp_group is not None and getattr(tp_group, "nranks", 1) > 1:
        paddle.distributed.all_reduce(target.contiguous(), group=tp_group)
    target = target / target.sum(axis=-1, keepdim=True).clip(min=1e-10)

    # Zero out invalid slots (so they contribute nothing to KL).
    target = paddle.where(valid, target, paddle.zeros_like(target))
    return target


def _compute_tilelang_csa_indexer_loss_forward(
    index_q: Tensor,
    weights: Tensor,
    index_k_comp: Tensor,
    query_mla: Tensor,
    key_comp_mla: Tensor,
    ratio: int,
    topk_effective: int,
    softmax_scale: float,
    loss_coeff: float,
    tp_group=None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    from paddlefleet.tilelang_ops import (
        csa_attn_target_reducesum,
        csa_indexer_topk_fwd,
    )

    topk_indices, topk_probs = csa_indexer_topk_fwd(
        index_q,
        index_k_comp,
        weights,
        ratio=int(ratio),
        topk_effective=int(topk_effective),
    )

    if tp_group is not None and getattr(tp_group, "nranks", 1) > 1:
        target = _compute_attn_target_on_selected_set(
            query_mla, key_comp_mla, topk_indices, softmax_scale, tp_group
        )
    else:
        target = csa_attn_target_reducesum(
            query_mla,
            key_comp_mla,
            topk_indices,
            softmax_scale,
        )

    eps = 1e-10
    kl_per_elem = target * (
        paddle.log(target + eps) - paddle.log(topk_probs + eps)
    )
    loss = kl_per_elem.sum(axis=-1).mean() * float(loss_coeff)
    return loss, topk_indices, topk_probs, target


class TileLangCSAIndexerLoss(paddle.autograd.PyLayer):
    """Selected-topk KL loss using TileLang CSA Indexer fwd/bwd kernels.

    Forward:
        1. Calls TileLang fused indexer to obtain
           ``topk_indices [B, S, topk_effective]`` and post-softmax
           ``topk_probs [B, S, topk_effective]`` over the selected set.
        2. Constructs the multi-head aggregated attention target
           ``p[t, S_t]`` on the same selected set via
           ``_compute_attn_target_on_selected_set``.
        3. Returns the scalar KL loss
           ``KL(p[t,S_t] || softmax(I[t,S_t])) * loss_coeff``.

        ``topk_indices`` is returned alongside the scalar loss. The outer
        ``CompressedSparseAttention.forward`` may trim it back to the main
        attention top-k width before feeding sparse attention.

    Backward:
        Following the Megatron / Miles selected-topk convention, the gradient
        of KL w.r.t. the post-softmax indexer probabilities at the selected
        slots is ``q - p`` (the fused softmax+KL identity). The kernel then
        backprops only through the ReLU + per-head weighting + scaled QK GEMM.

        ``grad_index_scores = (topk_probs - target) * loss_coeff / num_rows``
        is multiplied by the upstream ``grad_loss`` and forwarded to
        ``csa_indexer_bwd``.

    Phase semantics:
        * Phase 2 (``dsa_indexer_use_sparse_loss=False``): caller passes
          ``topk_effective=n_compressed`` so the selected set covers the full
          causal compressed range — equivalent to the Paddle full-range KL.
        * Phase 3 (``dsa_indexer_use_sparse_loss=True``): caller passes
          ``topk_effective=min(index_topk, n_compressed)`` for the standard
          selected-topk KL semantics.

    Phase 1 (``csa_dense_mode=True``) never reaches this PyLayer because
    ``self.indexer`` is ``None`` in that configuration.
    """

    @staticmethod
    def forward(
        ctx,
        index_q: Tensor,  # [b, sq, h_i, d_i]
        weights: Tensor,  # [b, sq, h_i]   (RAW weights, no softmax_scale baked in)
        index_k_comp: Tensor,  # [b, sk, d_i]
        query_mla: Tensor,  # [b, sq, np, hn]      DETACHED MLA query
        key_comp_mla: Tensor,  # [b, sk, hn]           DETACHED shared compressed KV
        ratio: int,
        topk_effective: int,
        softmax_scale: float,
        loss_coeff: float,
        tp_group=None,
    ) -> Tensor:
        # The TileLang kernel applies its own ``dim**-0.5`` scale on index_q
        # and consumes raw weights, so we pass them through unmodified. The
        # PyLayer treats this as a single fused op: forward materializes only
        # the selected ``[B,S,topk_effective]`` tensors and backward never
        # touches the full ``[B,S,S_comp]`` logits.
        loss, topk_indices, topk_probs, target = (
            _compute_tilelang_csa_indexer_loss_forward(
                index_q,
                weights,
                index_k_comp,
                query_mla,
                key_comp_mla,
                ratio,
                topk_effective,
                softmax_scale,
                loss_coeff,
                tp_group,
            )
        )

        ctx.save_for_backward(
            index_q.detach(),
            weights.detach(),
            index_k_comp.detach(),
            topk_indices.detach(),
            topk_probs.detach(),
            target.detach(),
        )
        ctx.loss_coeff = float(loss_coeff)
        ctx.num_rows = float(target.shape[0] * target.shape[1])
        return loss, topk_indices.detach()

    @staticmethod
    def backward(ctx, grad_loss: Tensor, grad_topk_indices: Tensor = None):
        from paddlefleet.tilelang_ops import (
            csa_indexer_bwd,
        )

        (
            index_q,
            weights,
            index_k_comp,
            topk_indices,
            topk_probs,
            target,
        ) = ctx.saved_tensor()

        # Treat the forward eps as numerical protection only and use the
        # fused softmax + KL gradient w.r.t. the selected logits.
        scale = ctx.loss_coeff / max(ctx.num_rows, 1.0)
        grad_index_scores = (topk_probs - target) * scale
        if grad_loss is not None:
            grad_index_scores = grad_index_scores * grad_loss

        grad_q, grad_weights, grad_k = csa_indexer_bwd(
            index_q,
            weights,
            index_k_comp,
            topk_indices,
            grad_index_scores,
        )

        if grad_q.dtype != index_q.dtype:
            grad_q = grad_q.cast(index_q.dtype)
        if grad_weights.dtype != weights.dtype:
            grad_weights = grad_weights.cast(weights.dtype)
        if grad_k.dtype != index_k_comp.dtype:
            grad_k = grad_k.cast(index_k_comp.dtype)

        return (
            grad_q,
            grad_weights,
            grad_k,
            None,
            None,
        )


class TileLangCSAIndexerLossAutoScaler(paddle.autograd.PyLayer):
    """Attach TileLang CSA indexer loss gradients to the main output.

    This is the TileLang analogue of ``DSAIndexerLossAutoScaler``. It avoids
    chaining a scalar-loss PyLayer behind another PyLayer in the full training
    graph while preserving the same gradient scale semantics.
    """

    @staticmethod
    def forward(
        ctx,
        output: Tensor,
        index_q: Tensor,
        weights: Tensor,
        index_k_comp: Tensor,
        topk_indices: Tensor,
        topk_probs: Tensor,
        target: Tensor,
        loss_coeff: float,
    ) -> Tensor:
        ctx.save_for_backward(
            index_q.detach(),
            weights.detach(),
            index_k_comp.detach(),
            topk_indices.detach(),
            topk_probs.detach(),
            target.detach(),
        )
        ctx.loss_coeff = float(loss_coeff)
        ctx.num_rows = float(target.shape[0] * target.shape[1])
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        from paddlefleet.tilelang_ops import csa_indexer_bwd

        (
            index_q,
            weights,
            index_k_comp,
            topk_indices,
            topk_probs,
            target,
        ) = ctx.saved_tensor()

        grad_index_scores = (topk_probs - target) * (
            ctx.loss_coeff / max(ctx.num_rows, 1.0)
        )
        scale = DSAIndexerLossAutoScaler._main_loss_backward_scale
        if scale is not None:
            grad_index_scores = grad_index_scores * scale

        grad_q, grad_weights, grad_k = csa_indexer_bwd(
            index_q,
            weights,
            index_k_comp,
            topk_indices,
            grad_index_scores,
        )

        if grad_q.dtype != index_q.dtype:
            grad_q = grad_q.cast(index_q.dtype)
        if grad_weights.dtype != weights.dtype:
            grad_weights = grad_weights.cast(weights.dtype)
        if grad_k.dtype != index_k_comp.dtype:
            grad_k = grad_k.cast(index_k_comp.dtype)

        return (
            grad_output,
            grad_q,
            grad_weights,
            grad_k,
            None,
            None,
            None,
        )


# ---------------------------------------------------------------------------
# Compressor
# ---------------------------------------------------------------------------


@dataclass
class CompressorSublayersSpec:
    """Sublayer specifications for CSA Compressor."""

    linear_wkv: type | LayerSpec = None
    linear_wgate: type | LayerSpec = None
    norm: type | LayerSpec = None


class Compressor(nn.Layer):
    """Gated pooling compressor for CSA.

    Compresses a sequence by pooling groups of compress_ratio tokens using
    learned gated weights.

    For ratio=4: overlapping compression (coff=2)
    For ratio=128: non-overlapping compression (coff=1)
    """

    def __init__(
        self,
        config: TransformerConfig,
        sublayers_spec: CompressorSublayersSpec,
        compress_ratio: int,
        head_dim: int,
        rotate: bool = False,
        rotary_pos_emb=None,
    ):
        super().__init__()
        self.config = config
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.overlap = compress_ratio == 4
        self.coff = 1 + int(self.overlap)
        self.rotate = rotate
        self.qk_pos_emb_head_dim = config.qk_pos_emb_head_dim or 0
        self.rotary_pos_emb = rotary_pos_emb

        proj_out_dim = self.coff * head_dim

        self.linear_wkv = build_spec_layer(
            sublayers_spec.linear_wkv,
            config.hidden_size,
            proj_out_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            skip_weight_param_allocation=False,
            tp_group=None,
        )
        self.linear_wgate = build_spec_layer(
            sublayers_spec.linear_wgate,
            config.hidden_size,
            proj_out_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            skip_weight_param_allocation=False,
            tp_group=None,
        )

        self.ape = self.create_parameter(
            shape=[compress_ratio, proj_out_dim],
            dtype="float32",
            default_initializer=nn.initializer.Normal(
                std=config.init_method_std
                if hasattr(config, "init_method_std")
                else 0.02
            ),
        )

        self.norm = build_spec_layer(
            sublayers_spec.norm,
            config=config,
            hidden_size=head_dim,
            eps=getattr(config, "layernorm_epsilon", 1e-5),
        )

    def _overlap_transform(
        self, tensor: Tensor, fill_value: float = 0
    ) -> Tensor:
        """Apply overlapping window transform for 4x compression.

        Input shape:  [b, n_groups, ratio, coff * head_dim]
        Output shape: [b, n_groups, 2 * ratio, head_dim]
        """
        b, n_groups, ratio, _ = tensor.shape
        d = self.head_dim
        new_tensor = paddle.full(
            [b, n_groups, 2 * ratio, d], fill_value, dtype=tensor.dtype
        )
        # Second half of each group's projection goes to positions [ratio:]
        new_tensor[:, :, ratio:, :] = tensor[:, :, :, d:]
        # First half of previous group goes to positions [:ratio] (skip group 0)
        new_tensor[:, 1:, :ratio, :] = tensor[:, :-1, :, :d]
        return new_tensor

    def forward(
        self,
        x: Tensor,
    ) -> Tensor | None:
        """Compress hidden states into shorter KV sequence.

        Args:
            x: [b, sq, hidden_size]

        Returns:
            compressed_kv: [b, sq // ratio, head_dim] or None if too short.
        """
        b, sq, _ = x.shape
        ratio = self.compress_ratio

        if sq < ratio:
            return None

        kv, _ = self.linear_wkv(x)  # [b, sq, coff * head_dim]
        score, _ = self.linear_wgate(x)  # [b, sq, coff * head_dim]

        cutoff = (sq // ratio) * ratio
        if cutoff < sq:
            kv = kv[:, :cutoff, :]
            score = score[:, :cutoff, :]

        n_compressed = cutoff // ratio

        # Reshape: [b, n_compressed, ratio, coff * head_dim]
        kv = kv.reshape([b, n_compressed, ratio, -1])
        score = score.reshape([b, n_compressed, ratio, -1])

        # APE: [ratio, coff * head_dim] -> [1, 1, ratio, coff * head_dim]
        score = score + self.ape.reshape([1, 1, ratio, -1])

        if self.overlap:
            kv = self._overlap_transform(kv, fill_value=0)
            score = self._overlap_transform(score, fill_value=float("-inf"))

        # Gated pooling: softmax over the pool_dim, weighted sum
        kv = (kv * F.softmax(score, axis=2)).sum(
            axis=2
        )  # [b, n_compressed, head_dim]

        kv = self.norm(kv.cast(x.dtype))

        # Apply RoPE with subsampled positions
        if self.rotary_pos_emb is not None and self.qk_pos_emb_head_dim > 0:
            kv = _apply_rope(
                kv,
                self.head_dim - self.qk_pos_emb_head_dim,
                self.qk_pos_emb_head_dim,
                self.rotary_pos_emb,
                self.config,
                n_compressed,
                ratio=ratio,
            )

        if self.rotate:
            kv = rotate_activation(kv)

        return kv  # [b, n_compressed, head_dim]


# ---------------------------------------------------------------------------
# CSAIndexer
# ---------------------------------------------------------------------------


@dataclass
class CSAIndexerSublayersSpec:
    """Sublayer specifications for CSAIndexer."""

    linear_wq_b: type | LayerSpec = None
    linear_weights_proj: type | LayerSpec = None
    compressor: type | LayerSpec = None


class CSAIndexer(nn.Layer):
    """Learned top-k retrieval over compressed positions for CSA.

    Computes index scores to select the most relevant compressed KV positions
    for each query token. Uses its own nested Compressor with Hadamard rotation
    for key generation.
    """

    def __init__(
        self,
        config: TransformerConfig,
        sublayers_spec: CSAIndexerSublayersSpec,
        compress_ratio: int,
        rotary_pos_emb=None,
    ):
        super().__init__()
        self.config = config
        self.compress_ratio = compress_ratio
        self.hidden_size = config.hidden_size
        self.qk_pos_emb_head_dim = config.qk_pos_emb_head_dim or 0
        self.q_lora_rank = config.q_lora_rank

        self.index_n_heads = config.dsa_index_n_heads
        self.index_head_dim = config.dsa_index_head_dim
        self.index_topk = config.dsa_index_topk

        self.softmax_scale: float = self.index_head_dim**-0.5

        self.rotary_pos_emb = rotary_pos_emb

        # Q projection: q_lora_rank -> n_heads * head_dim
        self.linear_wq_b = build_spec_layer(
            sublayers_spec.linear_wq_b,
            self.q_lora_rank,
            self.index_n_heads * self.index_head_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            skip_weight_param_allocation=False,
            tp_group=None,
        )

        # Weights projection: hidden_size -> n_heads
        self.linear_weights_proj = build_spec_layer(
            sublayers_spec.linear_weights_proj,
            self.hidden_size,
            self.index_n_heads,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            skip_weight_param_allocation=False,
            tp_group=None,
        )

        # Own compressor (smaller head_dim, with Hadamard rotation)
        self.compressor = build_spec_layer(
            sublayers_spec.compressor,
            config=config,
            compress_ratio=compress_ratio,
            head_dim=self.index_head_dim,
            rotate=True,
            rotary_pos_emb=rotary_pos_emb,
        )

    def forward_before_topk(
        self,
        x: Tensor,  # [b, sq, hidden_size]
        qr: Tensor,  # [b, sq, q_lora_rank]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute Q, compressed K, and weights before top-k selection."""
        b, sq, _ = x.shape

        # Q path
        q, _ = self.linear_wq_b(qr)  # [b, sq, n_heads * head_dim]
        q = q.reshape([b, sq, self.index_n_heads, self.index_head_dim])
        if self.rotary_pos_emb is not None and self.qk_pos_emb_head_dim > 0:
            q = _apply_rope(
                q,
                self.index_head_dim - self.qk_pos_emb_head_dim,
                self.qk_pos_emb_head_dim,
                self.rotary_pos_emb,
                self.config,
                sq,
                ratio=1,
            )
        q = rotate_activation(q)

        # K path: own compressor (already applies RoPE and rotation internally)
        k = self.compressor(x)  # [b, n_compressed, index_head_dim]

        # Weights
        weights, _ = self.linear_weights_proj(x)  # [b, sq, n_heads]
        weights = weights * (self.index_n_heads**-0.5)

        return q, k, weights

    def forward(
        self,
        x: Tensor,
        qr: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return (index_scores, topk_indices).

        Args:
            x: [b, sq, hidden_size]
            qr: [b, sq, q_lora_rank]
            mask: [b, sq, n_compressed] optional causal mask

        Returns:
            index_scores: [b, sq, n_compressed]
            topk_indices: [b, sq, topk]
        """
        q, k, weights = self.forward_before_topk(x, qr)
        effective_topk = min(self.index_topk, k.shape[1])
        index_scores, topk_indices = fused_qk_topk_naive(
            q, k, weights, effective_topk, mask
        )
        return index_scores, topk_indices


# ---------------------------------------------------------------------------
# CompressedSparseAttention (core attention)
# ---------------------------------------------------------------------------


@dataclass
class CompressedSparseAttentionSublayersSpec:
    """Sublayer specifications for CompressedSparseAttention."""

    compressor: type | LayerSpec = None
    indexer: type | LayerSpec = None


class CompressedSparseAttention(FleetLayer):
    """Core attention combining sliding window + compressed KV attention.

    Conditionally builds Compressor and CSAIndexer based on compress_ratio:
      - ratio=0: window-only attention
      - ratio=4: window + 4x compressed + learned CSAIndexer
      - ratio=128: window + 128x compressed, attend to all compressed positions
    """

    def __init__(
        self,
        config: TransformerConfig,
        sublayers_spec: CompressedSparseAttentionSublayersSpec,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float | None = None,
        softmax_scale: float | None = None,
        k_channels: int | None = None,
        v_channels: int | None = None,
        is_mtp_layer: bool = False,
        is_swa: bool = False,
        num_attention_heads: int | None = None,
        num_key_value_heads: int | None = None,
        cp_comm_type: str = "p2p",
        pg_collection: ProcessGroupCollection = None,
        rotary_pos_emb: nn.Layer = None,
        compress_ratio: int = 0,
    ):
        super().__init__(config)
        self.config = config
        self.layer_number = layer_number
        self.pg_collection = pg_collection
        self.tp_group = (
            pg_collection.tp
            if pg_collection is not None
            and pg_collection.tp is not None
            and getattr(pg_collection.tp, "nranks", 1) > 1
            else None
        )
        self.compress_ratio = compress_ratio
        self.window_size = config.csa_window_size
        self.v_head_dim = config.v_head_dim
        self.n_local_heads = config.num_attention_heads
        self.softmax_scale = config.v_head_dim**-0.5

        # Learnable attention sink per head
        self.attn_sink = self.create_parameter(
            shape=[self.n_local_heads],
            dtype="float32",
            default_initializer=nn.initializer.Constant(0.0),
        )

        # Conditionally build Compressor (ratio > 1)
        if self.compress_ratio > 1:
            self.compressor = build_spec_layer(
                sublayers_spec.compressor,
                config=config,
                compress_ratio=self.compress_ratio,
                head_dim=config.v_head_dim,
                rotate=False,
                rotary_pos_emb=rotary_pos_emb,
            )
        else:
            self.compressor = None

        # Conditionally build Indexer (ratio == 4 and not dense_mode)
        if self.compress_ratio == 4 and not config.csa_dense_mode:
            self.indexer = build_spec_layer(
                sublayers_spec.indexer,
                config=config,
                compress_ratio=self.compress_ratio,
                rotary_pos_emb=rotary_pos_emb,
            )
        else:
            self.indexer = None

    def _compute_indexer_compressed_topk_idxs(
        self,
        query: Tensor,
        x: Tensor,
        qr: Tensor,
        compressed_kv: Tensor,
        n_compressed: int,
        offset: int,
    ) -> tuple[Tensor, Tensor | None, tuple | None]:
        """Build indexer-selected compressed KV indices and loss state."""
        b, sq, np_heads, _ = query.shape
        indexer_loss = None
        tilelang_indexer_loss_state = None

        x_det = x.detach()
        qr_det = qr.detach()
        if self.training:
            x_det.stop_gradient = False
            qr_det.stop_gradient = False

        # Loss and main attention intentionally use different top-k widths
        # during phase 2. ``dsa_indexer_use_sparse_loss=False`` expands only
        # the indexer loss to the full compressed range; the main CSA attention
        # remains sparse and consumes ``min(index_topk, n_compressed)``.
        use_tilelang_indexer = _resolve_csa_tilelang_switch(
            self.config,
            "csa_tilelang_enable_indexer",
        )
        # The fused TileLang indexer-loss path is only active during the
        # grad-enabled forward. Full recompute runs the first forward under
        # no_grad; that pass should only materialize main-attention indices.
        use_tilelang_loss_path = (
            use_tilelang_indexer and self.training and paddle.is_grad_enabled()
        )
        loss_topk_effective = _resolve_csa_indexer_loss_topk_effective(
            self.config,
            self.indexer.index_topk,
            n_compressed,
        )
        attn_topk_effective = _resolve_csa_indexer_attn_topk_effective(
            self.indexer.index_topk,
            n_compressed,
        )

        causal_mask = _build_compressed_causal_mask(
            self.compress_ratio,
            b,
            sq,
            n_compressed,
        )

        if use_tilelang_loss_path:
            # Fused TileLang fwd + selected-set KL + TileLang bwd. The returned
            # indices may be wider than main attention top-k during phase 2 and
            # are trimmed below before sparse attention consumes them.
            indexer_loss_coeff = getattr(
                self.config, "dsa_indexer_loss_coeff", 0.0
            )
            q_indexer_bf, k_indexer_bf, weights_indexer_bf = (
                self.indexer.forward_before_topk(x_det, qr_det)
            )
            # compressed_kv is shared across query heads; pass it directly to
            # the target/reducesum path instead of materializing [B,S,H,D].
            key_comp_mla = compressed_kv.detach()
            (
                indexer_loss,
                topk_indices_compressed,
                topk_probs,
                target,
            ) = _compute_tilelang_csa_indexer_loss_forward(
                q_indexer_bf,
                weights_indexer_bf,
                k_indexer_bf,
                query.detach(),
                key_comp_mla,
                int(self.compress_ratio),
                int(loss_topk_effective),
                float(self.softmax_scale),
                float(indexer_loss_coeff),
                self.tp_group,
            )
            tilelang_indexer_loss_state = (
                q_indexer_bf,
                weights_indexer_bf,
                k_indexer_bf,
                topk_indices_compressed,
                topk_probs,
                target,
                float(indexer_loss_coeff),
            )
            if indexer_loss_coeff > 0:
                DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                    loss=indexer_loss,
                    layer_number=self.layer_number,
                    num_layers=self.config.num_hidden_layers,
                )
        elif self.training and not use_tilelang_indexer:
            q_indexer, k_indexer, weights_indexer = (
                self.indexer.forward_before_topk(x_det, qr_det)
            )
            indexer_loss_coeff = getattr(
                self.config, "dsa_indexer_loss_coeff", 0.0
            )
            key_for_loss = (
                compressed_kv.transpose([1, 0, 2])
                .unsqueeze(2)
                .expand([-1, -1, np_heads, -1])
            )

            q_sf = q_indexer.transpose([1, 0, 2, 3])
            k_sf = (
                k_indexer.transpose([1, 0, 2])
                if k_indexer.ndim == 3
                else k_indexer.transpose([1, 0, 2, 3])
            )
            weights_sf = (
                weights_indexer * self.indexer.softmax_scale
            ).transpose([1, 0, 2])
            query_sf = query.transpose([1, 0, 2, 3]).detach()
            mask_for_loss = causal_mask.unsqueeze(1)

            indexer_loss = FusedDSAIndexerLoss.apply(
                q_sf,
                weights_sf,
                k_sf,
                query_sf,
                key_for_loss.detach(),
                self.softmax_scale,
                min(self.indexer.index_topk, n_compressed),
                indexer_loss_coeff,
                mask_for_loss,
                getattr(self.config, "dsa_indexer_use_sparse_loss", True),
                self.tp_group,
            )
            topk_indices_compressed = FusedDSAIndexerLoss._last_topk_indices

            if indexer_loss_coeff > 0:
                DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                    loss=indexer_loss,
                    layer_number=self.layer_number,
                    num_layers=self.config.num_hidden_layers,
                )
        elif not use_tilelang_indexer:
            _, topk_indices_compressed = self.indexer(
                x_det,
                qr_det,
                mask=causal_mask,
            )

        # Optionally replace topk producer with TileLang fused compressed
        # indexer forward. This only swaps the indices fed to sparse attention.
        if use_tilelang_indexer and not use_tilelang_loss_path:
            from paddlefleet.tilelang_ops import (
                csa_indexer_topk_fwd,
            )

            with paddle.no_grad():
                q_indexer_tl, k_indexer_tl, weights_indexer_tl = (
                    self.indexer.forward_before_topk(x_det, qr_det)
                )
                tl_topk_indices, _tl_topk_scores = csa_indexer_topk_fwd(
                    q_indexer_tl,
                    k_indexer_tl,
                    weights_indexer_tl,
                    ratio=self.compress_ratio,
                    topk_effective=attn_topk_effective,
                )

            topk_indices_compressed = tl_topk_indices

        if topk_indices_compressed.shape[-1] > attn_topk_effective:
            topk_indices_compressed = topk_indices_compressed[
                ..., :attn_topk_effective
            ].contiguous()

        compress_topk_idxs = _map_compressed_topk_to_kv_full(
            topk_indices_compressed,
            sq,
            self.compress_ratio,
            offset,
        )

        return compress_topk_idxs, indexer_loss, tilelang_indexer_loss_state

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None,
        x: Tensor = None,
        qr: Tensor = None,
    ) -> Tensor:
        """Forward pass for CompressedSparseAttention.

        Args:
            query: [b, sq, np, v_head_dim]
            key:   [b, sq, 1, v_head_dim] (single-head MQA)
            value: unused (key == value in DSv4 Hybrid MQA)
            attention_mask: unused (causal is implicit)
            x:     [b, sq, hidden_size] original hidden states
            qr:    [b, sq, q_lora_rank] compressed query representation

        Returns:
            output: [b, sq, np * v_head_dim]
        """
        b, sq, np_heads, hn = query.shape

        # Step 1: Prepare single-head KV
        kv = key.squeeze(2)  # [b, sq, v_head_dim]

        # Step 2: Compression
        if self.compressor is not None and self.compress_ratio > 1:
            compressed_kv = self.compressor(x)  # [b, n_compressed, v_head_dim]
            if compressed_kv is not None:
                kv_full = paddle.concat([kv, compressed_kv], axis=1)
                n_compressed = compressed_kv.shape[1]
            else:
                kv_full = kv
                n_compressed = 0
        else:
            kv_full = kv
            n_compressed = 0

        offset = sq  # compressed indices start after original positions

        # Step 3: Window indices
        window_idxs = get_window_topk_idxs(self.window_size, b, sq)

        # Step 4: Compressed indices
        indexer_loss = None
        tilelang_indexer_loss_state = None

        if self.compress_ratio > 1 and n_compressed > 0:
            if self.indexer is not None:
                (
                    compress_topk_idxs,
                    indexer_loss,
                    tilelang_indexer_loss_state,
                ) = self._compute_indexer_compressed_topk_idxs(
                    query,
                    x,
                    qr,
                    compressed_kv,
                    n_compressed,
                    offset,
                )
            else:
                # ratio=128: attend to all compressed positions
                compress_topk_idxs = get_compress_topk_idxs(
                    self.compress_ratio,
                    b,
                    sq,
                    offset,
                )

            if compress_topk_idxs.dtype != window_idxs.dtype:
                compress_topk_idxs = compress_topk_idxs.cast(window_idxs.dtype)
            topk_idxs = paddle.concat(
                [window_idxs, compress_topk_idxs], axis=-1
            )
        else:
            topk_idxs = window_idxs

        topk_idxs = topk_idxs.cast("int32")

        # Step 5: Sparse attention
        output = self.compressed_sparse_attn(
            query,
            kv_full,
            self.attn_sink,
            topk_idxs,
            self.softmax_scale,
        )

        # Step 6: Attach indexer loss
        if tilelang_indexer_loss_state is not None and self.training:
            output = TileLangCSAIndexerLossAutoScaler.apply(
                output,
                *tilelang_indexer_loss_state,
            )
        elif indexer_loss is not None and self.training:
            output = DSAIndexerLossAutoScaler.apply(output, indexer_loss)

        return output

    def compressed_sparse_attn(
        self,
        query: Tensor,
        kv_full: Tensor,
        attn_sink: Tensor,
        topk_idxs: Tensor,
        softmax_scale: float,
    ):
        if _resolve_csa_tilelang_switch(
            self.config,
            "csa_tilelang_enable_sparse_attn",
        ):
            from paddlefleet.tilelang_ops import csa_sparse_attn

            output = csa_sparse_attn(
                query,
                kv_full,
                attn_sink.cast("float32"),
                topk_idxs,
                softmax_scale,
            )
        else:
            output = unfused_compressed_sparse_attn(
                query,
                kv_full,
                attn_sink.cast("float32"),
                topk_idxs,
                softmax_scale,
            )
        return output
