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


def _get_doc_start(
    attn_mask_startend_row_indices: Tensor, seqlen: int
) -> Tensor:
    """Derive per-position document start from attn_mask_startend_row_indices.

    Pure tensor operations using cummax, no Python for-loop.

    Logic:
      1. Detect boundary: where the end-boundary value changes → new doc starts
      2. Mark boundary positions with their index, non-boundary with 0
      3. cummax along seqlen propagates each boundary's index to all subsequent
         positions until the next boundary (since boundaries are monotonically increasing)

    Args:
        attn_mask_startend_row_indices: [b, seqlen] or [seqlen] tensor where each value
            is the end boundary (exclusive) of the document that position belongs to.

    Returns:
        doc_start: [b, seqlen] int64 tensor with start position of each token's document.
    """
    mask = attn_mask_startend_row_indices
    squeeze_batch: bool = mask.ndim == 1
    if squeeze_batch:
        mask = mask.unsqueeze(0)  # [1, seqlen]

    # Step 1: 检测文本边界 (相邻值不同 → 新文本开始)
    changed = paddle.zeros_like(mask, dtype="int64")
    changed[:, 0] = 1
    changed[:, 1:] = (mask[:, 1:] != mask[:, :-1]).cast("int64")

    # Step 2: 在边界位置标记其索引，非边界标记0
    positions = (
        paddle.arange(seqlen, dtype="int64").unsqueeze(0).expand_as(mask)
    )
    start_marker = changed * positions
    # 例: 两个文本(24+8) → start_marker = [0, 0, ..., 0, 24, 0, ..., 0]
    #                                       ^pos0(边界)     ^pos24(边界)
    # 注: pos0 的 changed=1, positions=0, 所以 start_marker[0]=0 是正确的

    # Step 3: cummax 沿 seqlen 方向取累积最大值
    # 由于文本段起始位置单调递增，cummax 的结果就是每个位置的 doc_start
    # 例: [0,0,...,0,24,0,...,0] → cummax → [0,0,...,0,24,24,...,24]
    doc_start = paddle.cummax(start_marker, axis=1).values

    if squeeze_batch:
        doc_start = doc_start.squeeze(0)

    return doc_start


@functools.lru_cache(maxsize=8)
def _get_window_topk_idxs_no_doc(
    window_size: int, seqlen: int, device_str: str
) -> Tensor:
    """Compute sliding window indices without document mask (cached)."""
    base = paddle.arange(seqlen).unsqueeze(1)  # [seqlen, 1]
    offsets = paddle.arange(window_size)  # [window_size]
    matrix = paddle.clip(base - window_size + 1, min=0) + offsets
    matrix = paddle.where(matrix > base, paddle.full_like(matrix, -1), matrix)
    return matrix


@functools.lru_cache(maxsize=8)
def _get_compress_topk_idxs_no_doc(
    ratio: int, seqlen: int, offset: int, device_str: str
) -> Tensor:
    """Compute compress indices without document mask (cached)."""
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
    attn_mask_startend_row_indices: Tensor,
    device=None,
) -> Tensor:
    """Get sliding window indices: [b, seqlen, window_size].

    Args:
        attn_mask_startend_row_indices: None for single-document (no mask), or
            a tensor of shape [b, 1, seqlen, 1] or [b, seqlen] with per-position
            document end boundaries.
    """
    if attn_mask_startend_row_indices is None:
        indices = _get_window_topk_idxs_no_doc(window_size, seqlen, "gpu")
        return indices.unsqueeze(0).expand([batch_size, -1, -1])

    # Reshape to [b, seqlen]
    assert (
        attn_mask_startend_row_indices.shape[1] == 1
        and attn_mask_startend_row_indices.shape[3] == 1
    ), (
        f"attn_mask_startend_row_indices shape must be [b, 1, seqlen, 1] now, but got {attn_mask_startend_row_indices.shape}"
    )
    mask = attn_mask_startend_row_indices.reshape([batch_size, seqlen])
    doc_start = _get_doc_start(mask, seqlen)  # [b, seqlen]

    # Vectorized computation for all batches at once
    base = paddle.arange(seqlen).unsqueeze(1)  # [seqlen, 1]
    offsets = paddle.arange(window_size)  # [window_size]

    # doc_start: [b, seqlen] -> [b, seqlen, 1]
    doc_start_3d = doc_start.unsqueeze(2)  # [b, seqlen, 1]
    base_3d = base.unsqueeze(0)  # [1, seqlen, 1]

    win_start = paddle.maximum(
        base_3d - window_size + 1, doc_start_3d
    )  # [b, seqlen, 1]
    matrix = win_start + offsets.unsqueeze(0).unsqueeze(
        0
    )  # [b, seqlen, window_size]
    matrix = paddle.where(
        matrix > base_3d, paddle.full_like(matrix, -1), matrix
    )
    return matrix


def get_compress_topk_idxs(
    ratio: int,
    batch_size: int,
    seqlen: int,
    offset: int,
    attn_mask_startend_row_indices: Tensor,
    device=None,
) -> Tensor:
    """Get compressed indices: [b, seqlen, seqlen // ratio].

    Args:
        attn_mask_startend_row_indices: None for single-document (no mask), or
            a tensor of shape [b, 1, seqlen, 1] or [b, seqlen] with per-position
            document end boundaries.
    """
    if attn_mask_startend_row_indices is None:
        matrix = _get_compress_topk_idxs_no_doc(ratio, seqlen, offset, "gpu")
        return matrix.unsqueeze(0).expand([batch_size, -1, -1])

    # Reshape to [b, seqlen]
    mask = attn_mask_startend_row_indices.reshape([batch_size, seqlen])
    doc_start = _get_doc_start(mask, seqlen)  # [b, seqlen]

    n_compressed = seqlen // ratio
    k_indices = paddle.arange(n_compressed)  # [n_compressed]

    # Expand: [b, seqlen, n_compressed]
    matrix = (
        k_indices.unsqueeze(0).unsqueeze(0).expand([batch_size, seqlen, -1])
    )  # [b, seqlen, n_compressed]

    # Causal mask: k >= (i+1) // ratio
    causal_bound = (
        paddle.arange(1, seqlen + 1).unsqueeze(1) // ratio
    ).unsqueeze(0)  # [1, seqlen, 1]
    causal_invalid = matrix >= causal_bound

    # Document boundary mask: k < ceil(doc_start[i] / ratio)
    doc_start_compressed = ((doc_start + ratio - 1) // ratio).unsqueeze(
        2
    )  # [b, seqlen, 1]
    doc_invalid = matrix < doc_start_compressed

    invalid = causal_invalid | doc_invalid
    matrix = paddle.where(
        invalid, paddle.full_like(matrix, -1), matrix + offset
    )
    return matrix


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


from paddlefleet.tilelang_ops import tilelang_compressed_sparse_attn

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

    def forward(self, x: Tensor) -> Tensor | None:
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
        cp_comm_type: str = "p2p",
        pg_collection: ProcessGroupCollection = None,
        rotary_pos_emb: nn.Layer = None,
        compress_ratio: int = 0,
    ):
        super().__init__(config)
        self.config = config
        self.layer_number = layer_number
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

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None,
        x: Tensor = None,
        qr: Tensor = None,
        attn_mask_startend_row_indices: Tensor = None,
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
        window_idxs = get_window_topk_idxs(
            self.window_size, b, sq, attn_mask_startend_row_indices
        )

        # Step 4: Compressed indices
        indexer_loss = None

        if self.compress_ratio > 1 and n_compressed > 0:
            if self.indexer is not None:
                x_det = x.detach()
                qr_det = qr.detach()
                if self.training:
                    x_det.stop_gradient = False
                    qr_det.stop_gradient = False

                # Build causal mask for compressed positions: [b, sq, n_compressed]
                compressed_ids = paddle.arange(n_compressed).unsqueeze(
                    0
                )  # [1, n_compressed]
                positions = paddle.arange(1, sq + 1).unsqueeze(1)  # [sq, 1]
                causal_mask = paddle.where(
                    compressed_ids >= (positions // self.compress_ratio),
                    paddle.full([1], float("-inf"), dtype="float32"),
                    paddle.zeros([1], dtype="float32"),
                )  # [sq, n_compressed]
                causal_mask = causal_mask.unsqueeze(0).expand(
                    [b, sq, n_compressed]
                )  # [b, sq, n_compressed]

                if self.training:
                    q_indexer, k_indexer, weights_indexer = (
                        self.indexer.forward_before_topk(x_det, qr_det)
                    )
                    indexer_loss_coeff = getattr(
                        self.config, "dsa_indexer_loss_coeff", 0.0
                    )
                    # compressed_kv: [b, n, hn] -> expand for loss: [n, b, np, hn]
                    key_for_loss = (
                        compressed_kv.transpose([1, 0, 2])
                        .unsqueeze(2)
                        .expand([-1, -1, np_heads, -1])
                    )

                    # Convert batch-first to seq-first for FusedDSAIndexerLoss
                    q_sf = q_indexer.transpose([1, 0, 2, 3])  # [sq, b, h, d]
                    k_sf = (
                        k_indexer.transpose([1, 0, 2])
                        if k_indexer.ndim == 3
                        else k_indexer.transpose([1, 0, 2, 3])
                    )  # [n_compressed, b, d]
                    weights_sf = (
                        weights_indexer * self.indexer.softmax_scale
                    ).transpose([1, 0, 2])  # [sq, b, h]
                    query_sf = query.transpose(
                        [1, 0, 2, 3]
                    ).detach()  # [sq, b, np, hn]

                    # causal_mask for FusedDSAIndexerLoss: [b, 1, sq, n_compressed]
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
                        getattr(
                            self.config, "dsa_indexer_use_sparse_loss", True
                        ),
                        None,  # tp_group
                    )
                    topk_indices_compressed = (
                        FusedDSAIndexerLoss._last_topk_indices
                    )

                    if indexer_loss_coeff > 0:
                        DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                            loss=indexer_loss,
                            layer_number=self.layer_number,
                            num_layers=self.config.num_hidden_layers,
                        )
                else:
                    _, topk_indices_compressed = self.indexer(
                        x_det, qr_det, mask=causal_mask
                    )

                # Filter invalid indices and shift to kv_full space
                n_valid_per_pos = (
                    paddle.arange(1, sq + 1).unsqueeze(1) // self.compress_ratio
                )  # [sq, 1]
                n_valid_per_pos = n_valid_per_pos.unsqueeze(0)  # [1, sq, 1]
                valid = topk_indices_compressed < n_valid_per_pos
                compress_topk_idxs = paddle.where(
                    valid,
                    topk_indices_compressed + offset,
                    paddle.full_like(topk_indices_compressed, -1),
                )
            else:
                # ratio=128: attend to all compressed positions
                compress_topk_idxs = get_compress_topk_idxs(
                    self.compress_ratio,
                    b,
                    sq,
                    offset,
                    attn_mask_startend_row_indices,
                )

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
        if indexer_loss is not None and self.training:
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
        if self.config.csa_sparse_attn_fusion:
            output = tilelang_compressed_sparse_attn(
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
