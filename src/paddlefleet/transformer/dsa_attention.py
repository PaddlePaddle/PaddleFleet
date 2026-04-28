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
DeepSeek Sparse Attention (DSA) extension for Multi-Latent Attention.

This module extends the upstream MLASelfAttention with DSA Indexer support
(DeepSeek V3.2 architecture):
  - Indexer: Token scoring module that selects top-k relevant positions
  - FusedDSAIndexerLoss: Fused KL-divergence loss with full manual backward
  - DSAIndexerLossAutoScaler: Loss scaling helper
  - MLASelfAttentionWithDSA: Subclass of MLASelfAttention with DSA integration

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import paddle
import paddle.nn.functional as F
from paddle import Tensor
from paddle.distributed.fleet.utils import recompute

from paddlefleet import parallel_state
from paddlefleet.models.common.embeddings.rope_utils import (
    _apply_rotary_pos_emb_bshd,
)
from paddlefleet.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
)
from paddlefleet.transformer.enums import AttnMaskType
from paddlefleet.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSublayersSpec,
)

if TYPE_CHECKING:
    from paddlefleet.process_groups_config import ProcessGroupCollection
    from paddlefleet.transformer.transformer_config import TransformerConfig


def hadamard_transform(x: Tensor, scale: float = 1.0) -> Tensor:
    """Fast Walsh-Hadamard Transform using the butterfly algorithm.

    Pure Paddle implementation, equivalent to:
        F.linear(x, hadamard_matrix(dim)) * scale

    Uses O(N log N) butterfly operations instead of O(N^2) matrix multiply.
    The Hadamard matrix is symmetric and orthogonal, so backward is the same
    transform applied to grad_output (handled automatically by Paddle autograd).

    Reference:
        - fast-hadamard-transform (Tri Dao): csrc/fast_hadamard_transform_cuda.cu
        - PaddleFormers/paddleformers/quantization/hadamard_utils.py (matmul_hadU)

    Args:
        x: Input tensor of shape (..., dim). dim must be a power of 2.
        scale: Scaling factor applied to the output.

    Returns:
        Hadamard-transformed tensor of the same shape.
    """
    original_shape = x.shape
    dim = original_shape[-1]
    assert dim > 0 and (dim & (dim - 1)) == 0, (
        f"hadamard_transform requires dim to be a power of 2, got {dim}"
    )

    # Flatten batch dims: (..., dim) -> (batch, dim)
    x = x.reshape([-1, dim])

    h = 1
    while h < dim:
        x = x.reshape([-1, dim // (2 * h), 2, h])
        a = x[:, :, 0, :]
        b = x[:, :, 1, :]
        x = paddle.stack([a + b, a - b], axis=2)
        x = x.reshape([-1, dim])
        h *= 2

    return x.reshape(original_shape) * scale


def rotate_activation(x: Tensor) -> Tensor:
    """Apply Hadamard rotation activation.

    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L424-L428

    Args:
        x: Input tensor (must be bfloat16).

    Returns:
        Rotated tensor.
    """
    assert x.dtype == paddle.bfloat16, (
        f"rotate_activation only support bf16 input, but got {x.dtype}"
    )
    hidden_size = x.shape[-1]
    return hadamard_transform(x, scale=hidden_size**-0.5)


# ---------------------------------------------------------------------------
# Unfused DSA attention (explicit bmm, supports asymmetric Q/K vs V dims)
# ---------------------------------------------------------------------------
def _unfused_dsa_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    combined_mask: Tensor | None,
    softmax_scale: float,
) -> Tensor:
    """Unfused DSA sparse attention

    Uses explicit bmm instead of flash attention to support:
    - Different Q/K head_dim vs V head_dim (MLA architecture)
    - Arbitrary per-token sparse masks from DSA Indexer

    Args:
        query: [b, s, nhpp, qk_head_dim]
        key:   [b, s, nhpp, qk_head_dim]
        value: [b, s, nhpp, v_head_dim]   (v_head_dim may differ from qk_head_dim)
        combined_mask: [b, 1, s, s]  (causal + sparse index mask, -inf for masked)
        softmax_scale: 1/sqrt(qk_head_dim)

    Returns:
        output: [b, s, nhpp * v_head_dim]
    """
    b, s, nhpp, qk_hd = query.shape
    v_hd = value.shape[-1]

    # Reshape for bmm: [b*nhpp, s, hd]
    q = query.transpose([0, 2, 1, 3]).reshape([b * nhpp, s, qk_hd])
    k = key.transpose([0, 2, 1, 3]).reshape([b * nhpp, s, qk_hd])
    v = value.transpose([0, 2, 1, 3]).reshape([b * nhpp, s, v_hd])

    # Q * K^T with scale: [b*nhpp, s, s]
    attn_scores = (
        paddle.bmm(q.cast("float32"), k.cast("float32").transpose([0, 2, 1]))
        * softmax_scale
    )

    # Apply combined mask (causal + sparse index mask)
    if combined_mask is not None:
        mask = (
            combined_mask.expand([b, nhpp, s, s])
            .contiguous()
            .reshape([b * nhpp, s, s])
        )
        attn_scores = attn_scores + mask.cast("float32")

    attn_weights = F.softmax(attn_scores, axis=-1)

    # Attention_weights * V: [b*nhpp, s, v_hd]
    output = paddle.bmm(attn_weights.cast(v.dtype), v)

    # [b*nhpp, s, v_hd] -> [b, s, nhpp*v_hd]
    output = (
        output.reshape([b, nhpp, s, v_hd])
        .transpose([0, 2, 1, 3])
        .reshape([b, s, nhpp * v_hd])
    )

    return output


# ---------------------------------------------------------------------------
# DSA Indexer
# ---------------------------------------------------------------------------
class Indexer(paddle.nn.Layer):
    """DSA Indexer: DeepSeek Sparse Attention token selection module.

    For each query token, scores all cached key positions using a lightweight
    n_heads-head attention mechanism, then selects the top-k most relevant
    positions for the full MLA attention computation.

    Key design notes:
    - Uses non-interleaved RoPE (unlike MLA which uses interleaved)
    - Uses LayerNorm (not RMSNorm) on K
    - nope/pe split order: [nope | pe]
    - Uses ReLU-aggregated scoring across heads
    - Per-head learned importance weights via weights_proj
    - weights absorbs softmax_scale

    """

    def __init__(self, config: TransformerConfig, layer_number: int):
        super().__init__()
        self.config = config

        self.n_heads = config.dsa_index_n_heads
        self.head_dim = config.dsa_index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.index_topk = config.dsa_index_topk
        self.softmax_scale = self.head_dim**-0.5
        self.layer_number = layer_number

        # wq_b: q_lora_rank -> n_heads * head_dim (duplicated)
        self.wq_b = paddle.nn.Linear(
            config.q_lora_rank,
            self.n_heads * self.head_dim,
            bias_attr=False,
        )

        # wk: hidden_size -> head_dim (single shared K, duplicated)
        self.wk = paddle.nn.Linear(
            config.hidden_size,
            self.head_dim,
            bias_attr=False,
        )

        # k_norm: LayerNorm (NOT RMSNorm) per reference
        self.k_norm = paddle.nn.LayerNorm(self.head_dim, epsilon=1e-6)

        # weights_proj: learned per-head importance [hidden -> n_heads]
        self.weights_proj = paddle.nn.Linear(
            config.hidden_size,
            self.n_heads,
            bias_attr=False,
        )

    def _apply_rope(
        self, x: Tensor, freqs: Tensor, mscale: float = 1.0
    ) -> Tensor:
        """Apply RoPE to the pe portion of x.

        Split order: [pe | nope], matching DeepSeek-V3.2 Indexer (model.py:462).

        RoPE format is controlled by config.dsa_indexer_rotary_interleaved:
        - False (default): non-interleaved RoPE with half-head frequencies [θ₁,θ₂,...,θ₁,θ₂,...]
        - True: interleaved RoPE with paired frequencies [θ₁,θ₁,θ₂,θ₂,...]

        Args:
            x: [..., head_dim] (rope_dim + nope_dim)
            freqs: RoPE frequencies
            mscale: YaRN concentration factor (1.0 for plain RoPE, ~1.37 for YaRN)
        """
        x_pe = x[..., : self.rope_head_dim]
        x_nope = x[..., self.rope_head_dim :]
        x_pe = _apply_rotary_pos_emb_bshd(
            x_pe,
            freqs,
            rotary_interleaved=self.config.dsa_indexer_rotary_interleaved,
            multi_latent_attention=False,
            mscale=mscale,
        )
        return paddle.concat([x_pe, x_nope], axis=-1)

    def forward_before_topk(
        self,
        hidden_states: Tensor,  # [b, s, hidden_size]
        q_latent: Tensor,  # [b, s, q_lora_rank]
        freqs: Tensor,
        mscale: float = 1.0,
    ):
        """Compute q, k, weights before top-k selection."""
        bsz, seqlen, _ = hidden_states.shape
        q = self.wq_b(q_latent)  # [b, s, n_heads * head_dim]
        q = q.reshape([bsz, seqlen, self.n_heads, self.head_dim])
        if freqs is not None:
            q = self._apply_rope(q, freqs, mscale)

        k = self.wk(hidden_states)  # [b, s, head_dim]
        k = self.k_norm(k)
        if freqs is not None:
            k = self._apply_rope(k.unsqueeze(2), freqs, mscale).squeeze(2)

        # Rotate activation (Hadamard transform)
        q = rotate_activation(q)
        k = rotate_activation(k)

        weights = (
            self.weights_proj(hidden_states.cast("float32"))
            * (self.n_heads**-0.5)
            * self.softmax_scale
        )

        return q, k, weights

    def compute_index_scores(
        self,
        q: Tensor,  # [b, s, n_heads, head_dim]
        k: Tensor,  # [b, t, head_dim]
        weights: Tensor,  # [b, s, n_heads]
        mask: Tensor | None = None,
    ):
        """Compute index scores and select top-k."""
        q_fp32 = q.cast("float32")
        k_fp32 = k.cast("float32")

        scores = paddle.einsum("bshd,btd->bsht", q_fp32, k_fp32)
        index_scores = (weights.unsqueeze(-1) * F.relu(scores)).sum(axis=2)

        if mask is not None:
            index_scores = index_scores + mask.squeeze(1)

        topk_k = min(self.index_topk, index_scores.shape[-1])
        topk_indices = paddle.topk(index_scores, k=topk_k, axis=-1)[1]
        # Clamp indices to valid range: paddle.topk may return garbage indices
        # for -inf input values
        topk_indices = paddle.clip(
            topk_indices, min=0, max=index_scores.shape[-1] - 1
        )

        return index_scores, topk_indices

    def forward(
        self,
        hidden_states: Tensor,
        q_latent: Tensor,
        freqs: Tensor,
        attention_mask: Tensor,
        mscale: float = 1.0,
    ) -> tuple[Tensor, Tensor]:
        """Compute DSA token importance scores and return scores + top-k indices."""
        q, k, weights = self.forward_before_topk(
            hidden_states, q_latent, freqs, mscale
        )
        index_scores, topk_indices = self.compute_index_scores(
            q, k, weights, attention_mask
        )
        return index_scores, topk_indices


def _compute_index_scores_fused(
    q: Tensor, weights: Tensor, k: Tensor
) -> Tensor:
    """Compute index scores from Indexer outputs.

    Args:
        q:       [sq, b, h, d]  (Indexer query, after RoPE + Hadamard)
        weights: [sq, b, h]     (per-head importance weights)
        k:       [sk, b, d]     (Indexer key, after RoPE + Hadamard)

    Returns:
        index_scores: [b, sq, sk]
    """
    # q @ k^T -> [sq, b, h, sk]
    index_scores = paddle.einsum(
        "sbhd,tbd->sbht", q.cast("float32"), k.cast("float32")
    )
    # ReLU activation
    index_scores = F.relu(index_scores)
    # Weight each head: [sq, b, h, sk] * [sq, b, h, 1] -> [sq, b, h, sk]
    index_scores = index_scores * weights.unsqueeze(-1)
    # Sum across heads: [sq, b, h, sk] -> [sq, b, sk]
    index_scores = index_scores.sum(axis=2)
    # Transpose to [b, sq, sk]
    index_scores = index_scores.transpose([1, 0, 2])
    return index_scores


def _compute_dsa_indexer_loss(
    index_scores: Tensor,
    topk_indices: Tensor,
    query: Tensor,
    key: Tensor,
    softmax_scale: float,
    loss_coeff: float,
    sparse_loss: bool,
    tp_group,
) -> Tensor:
    """Compute KL divergence loss between index_scores and true attention_scores.

    Args:
        index_scores: [b, sq, sk]
        topk_indices: [b, sq, topk]
        query: [sq, b, np, hn]  (MLA query, DETACHED)
        key:   [sk, b, np, hn]  (MLA key, DETACHED)
        softmax_scale: Scale coefficient after q @ k^T
        loss_coeff: Coefficient for the indexer KL divergence loss
        sparse_loss: Whether to apply sparse index mask
        tp_group: TP process group (or None)

    Returns:
        indexer_loss: scalar
    """
    sq, b, np, hn = query.shape
    sk = key.shape[0]

    # [sq, b, np, hn] -> [b, np, sq, hn] -> [b * np, sq, hn]
    query_reshaped = query.transpose([1, 2, 0, 3]).reshape([b * np, sq, hn])
    # [sk, b, np, hn] -> [b, np, hn, sk] -> [b * np, hn, sk]
    key_reshaped = key.transpose([1, 2, 3, 0]).reshape([b * np, hn, sk])
    # Compute attention scores [b * np, sq, sk]
    attention_scores = (
        paddle.bmm(query_reshaped.cast("float32"), key_reshaped.cast("float32"))
        * softmax_scale
    )
    # Reshape to [b, np, sq, sk]
    attention_scores = attention_scores.reshape([b, np, sq, sk])

    # causal_mask [sq, sk]
    causal_mask = paddle.triu(
        paddle.full([sq, sk], float("-inf"), dtype="float32"),
        diagonal=1,
    )
    # index_mask [b, sq, sk]
    index_mask = paddle.full([b, sq, sk], float("-inf"), dtype="float32")
    index_mask = paddle.put_along_axis(
        index_mask,
        topk_indices,
        paddle.zeros_like(topk_indices, dtype="float32"),
        axis=-1,
    )

    # [b, np, sq, sk] + [1, 1, sq, sk] -> [b, np, sq, sk]
    attention_scores = attention_scores + causal_mask.reshape([1, 1, sq, sk])
    if sparse_loss:
        # [b, np, sq, sk] + [b, 1, sq, sk] -> [b, np, sq, sk]
        attention_scores = attention_scores + index_mask.reshape([b, 1, sq, sk])
        # [b, sq, sk] + [b, sq, sk] -> [b, sq, sk]
        index_scores = index_scores + index_mask

    # [b, np, sq, sk] -> [b, np, sq, sk]
    attention_scores = F.softmax(attention_scores, axis=-1, dtype="float32")
    # [b, sq, sk] -> [b, sq, sk]
    index_scores = F.softmax(index_scores, axis=-1, dtype="float32")

    # Sum attention scores across heads: [b, np, sq, sk] -> [b, sq, sk]
    attention_scores = attention_scores.sum(axis=1)
    if tp_group is not None and tp_group.nranks > 1:
        paddle.distributed.all_reduce(
            attention_scores.contiguous(), group=tp_group
        )
    # L1 normalize target on the last dimension
    attention_scores = attention_scores / attention_scores.sum(
        axis=-1, keepdim=True
    )

    # KL divergence: KL(target || index) = target * log(target / index)
    kl_per_element = attention_scores * (
        paddle.log(attention_scores + 1e-10) - paddle.log(index_scores + 1e-10)
    )

    # [b, sq, sk] -> [b, sq] -> [1]
    kl_div = kl_per_element.sum(axis=-1).mean()
    indexer_loss = kl_div * loss_coeff

    return indexer_loss


def _bwd_fused_indexer_loss(
    q: Tensor,
    weights: Tensor,
    k: Tensor,
    query: Tensor,
    key: Tensor,
    topk_indices: Tensor,
    softmax_scale: float,
    loss_coeff: float,
    sparse_loss: bool,
    grad_loss: Tensor,
    tp_group,
) -> tuple[Tensor, Tensor, Tensor]:
    """Manual backward for fused indexer loss.


    All tensor layouts (sequence-first):
        q:       [sq, b, h, d]
        weights: [sq, b, h]
        k:       [sk, b, d]
        query:   [sq, b, np, hn]  (MLA query)
        key:     [sk, b, np, hn]  (MLA key)

    Returns:
        grad_q:       [sq, b, h, d]
        grad_weights: [sq, b, h]
        grad_k:       [sk, b, d]
    """
    # Recompute index_scores from (q, weights, k)
    index_scores = _compute_index_scores_fused(q, weights, k)  # [b, sq, sk]

    sq, b, np, hn = query.shape
    sk = key.shape[0]

    # [sq, b, np, hn] -> [b, np, sq, hn] -> [b * np, sq, hn]
    query_reshaped = query.transpose([1, 2, 0, 3]).reshape([b * np, sq, hn])
    # [sk, b, np, hn] -> [b, np, hn, sk] -> [b * np, hn, sk]
    key_reshaped = key.transpose([1, 2, 3, 0]).reshape([b * np, hn, sk])
    # Compute attention scores [b * np, sq, sk]
    attention_scores = (
        paddle.bmm(query_reshaped.cast("float32"), key_reshaped.cast("float32"))
        * softmax_scale
    )
    del query_reshaped, key_reshaped

    # Reshape to [b, np, sq, sk]
    attention_scores = attention_scores.reshape([b, np, sq, sk])

    # causal_mask [sq, sk]
    causal_mask = paddle.triu(
        paddle.full([sq, sk], float("-inf"), dtype="float32"),
        diagonal=1,
    )
    # index_mask [b, sq, sk]
    index_mask = paddle.full([b, sq, sk], float("-inf"), dtype="float32")
    index_mask = paddle.put_along_axis(
        index_mask,
        topk_indices,
        paddle.zeros_like(topk_indices, dtype="float32"),
        axis=-1,
    )

    # Apply causal mask to both attention and index scores
    attention_scores = attention_scores + causal_mask.reshape([1, 1, sq, sk])
    index_scores = index_scores + causal_mask.unsqueeze(0)
    del causal_mask

    if sparse_loss:
        attention_scores = attention_scores + index_mask.reshape([b, 1, sq, sk])
        index_scores = index_scores + index_mask

    # Compute softmax for both
    attention_scores_softmax = F.softmax(
        attention_scores, axis=-1, dtype="float32"
    )
    del attention_scores

    index_scores_softmax = F.softmax(index_scores, axis=-1, dtype="float32")
    del index_scores

    # Sum attention scores across heads: [b, np, sq, sk] -> [b, sq, sk]
    attention_scores_sum = attention_scores_softmax.sum(axis=1)
    del attention_scores_softmax

    if tp_group is not None and tp_group.nranks > 1:
        paddle.distributed.all_reduce(
            attention_scores_sum.contiguous(), group=tp_group
        )

    # L1 normalize
    attention_scores_normalized = (
        attention_scores_sum / attention_scores_sum.sum(axis=-1, keepdim=True)
    )
    del attention_scores_sum

    # Backward through loss = kl_div * loss_coeff
    # where kl_div = kl_per_element.sum(dim=-1).mean()
    grad_kl_div = grad_loss.cast("float32") * loss_coeff  # scalar

    # Backward through mean: distribute gradient equally
    grad_kl_per_row = grad_kl_div / (b * sq)  # scalar

    # Backward through sum(dim=-1): broadcast back to [b, sq, sk]
    grad_kl_per_element = grad_kl_per_row.reshape([1, 1, 1]).expand([b, sq, sk])

    # Backward through kl: ∂kl/∂index_softmax = -target / index_softmax
    grad_index_scores_softmax = (
        -attention_scores_normalized
        / (index_scores_softmax + 1e-10)
        * grad_kl_per_element
    )
    del attention_scores_normalized

    # Backward through softmax:
    # ∂L/∂x = softmax * (∂L/∂softmax - sum(∂L/∂softmax * softmax))
    sum_grad = (grad_index_scores_softmax * index_scores_softmax).sum(
        axis=-1, keepdim=True
    )
    grad_index_scores_logits = index_scores_softmax * (
        grad_index_scores_softmax - sum_grad
    )
    del index_scores_softmax, grad_index_scores_softmax, sum_grad

    # Zero out gradients for masked positions
    causal_valid_mask = paddle.tril(
        paddle.ones([sq, sk], dtype="bool")
    )  # [sq, sk]
    if sparse_loss:
        index_valid_mask = index_mask == 0  # [b, sq, sk]
        del index_mask
        valid_mask = (
            causal_valid_mask.unsqueeze(0) & index_valid_mask
        )  # [b, sq, sk]
        del index_valid_mask
    else:
        del index_mask
        valid_mask = causal_valid_mask.unsqueeze(0).expand(
            [b, sq, sk]
        )  # [b, sq, sk]
    del causal_valid_mask

    grad_index_scores_logits = grad_index_scores_logits * valid_mask.cast(
        "float32"
    )
    del valid_mask

    # Transpose from [b, sq, sk] to [sq, b, sk]
    grad_index_scores = grad_index_scores_logits.transpose(
        [1, 0, 2]
    )  # [sq, b, sk]
    del grad_index_scores_logits

    # Backward through sum over heads: expand gradient
    grad_weighted_scores = grad_index_scores.unsqueeze(2)  # [sq, b, 1, sk]
    del grad_index_scores

    # Compute forward values needed for backward (recomputation)
    scores = paddle.einsum(
        "sbhd,tbd->sbht", q.cast("float32"), k.cast("float32")
    )  # [sq, b, h, sk]
    relu_mask = scores > 0
    scores_after_relu = F.relu(scores)
    del scores

    # Backward through multiplication by weights:
    # ∂L/∂weights = grad * relu_scores (sum over sk)
    grad_weights = (grad_weighted_scores * scores_after_relu).sum(
        axis=-1
    )  # [sq, b, h]

    # ∂L/∂relu_scores = grad * weights
    grad_scores_after_relu = grad_weighted_scores * weights.unsqueeze(
        -1
    )  # [sq, b, h, sk]
    del grad_weighted_scores, scores_after_relu

    # Backward through ReLU
    grad_scores = grad_scores_after_relu * relu_mask.cast(
        "float32"
    )  # [sq, b, h, sk]
    del grad_scores_after_relu, relu_mask

    # Backward through einsum 'sbhd,tbd->sbht'
    # ∂L/∂q = einsum('sbht,tbd->sbhd', grad_scores, k)
    grad_q = paddle.einsum(
        "sbht,tbd->sbhd", grad_scores, k.cast("float32")
    )  # [sq, b, h, d]
    # ∂L/∂k = einsum('sbht,sbhd->tbd', grad_scores, q)
    grad_k = paddle.einsum(
        "sbht,sbhd->tbd", grad_scores, q.cast("float32")
    )  # [sk, b, d]
    del grad_scores

    return (
        grad_q.cast(q.dtype),
        grad_weights.cast(weights.dtype),
        grad_k.cast(k.dtype),
    )


class FusedDSAIndexerLoss(paddle.autograd.PyLayer):
    """Fused DSA Indexer Loss: index_scores + topk + KL loss + full manual backward."""

    _last_topk_indices: Tensor | None = None

    @staticmethod
    def forward(
        ctx,
        q: Tensor,  # [sq, b, h, d]  — Indexer query output
        weights: Tensor,  # [sq, b, h]     — Indexer per-head weights
        k: Tensor,  # [sk, b, d]     — Indexer key output
        query: Tensor,  # [sq, b, np, hn] — MLA query (DETACHED)
        key: Tensor,  # [sk, b, np, hn] — MLA key (DETACHED)
        # Non-tensor params follow (stored on ctx, not in backward returns)
        softmax_scale: float = 1.0,
        topk: int = 64,
        loss_coeff: float = 1.0,
        mask: Tensor | None = None,
        sparse_loss: bool = True,
        tp_group=None,
    ) -> Tensor:
        """Fused forward: compute index_scores, topk, and KL loss.

        Args:
            q:       Indexer query after RoPE+Hadamard [sq, b, h, d]
            weights: Per-head importance weights [sq, b, h]
            k:       Indexer key after RoPE+Hadamard [sk, b, d]
            query:   MLA query (detached) [sq, b, np, hn]
            key:     MLA key (detached) [sk, b, np, hn]
            softmax_scale: MLA attention softmax scale
            topk:    Number of top-k indices to select
            loss_coeff: Coefficient for KL loss
            mask:    Optional mask for index_scores [b, 1, sq, sk] or [1, 1, sq, sk]
            sparse_loss: Whether to use sparse index mask in loss
            tp_group: TP process group (or None)

        Returns:
            indexer_loss: scalar KL divergence loss
        """
        # Step 1: Compute index_scores from (q, weights, k)
        index_scores = _compute_index_scores_fused(q, weights, k)  # [b, sq, sk]

        # Step 2: Apply mask and select topk
        if mask is not None:
            masked_scores = index_scores + mask.squeeze(1)
        else:
            masked_scores = index_scores
        topk_k = min(topk, masked_scores.shape[-1])
        topk_indices = paddle.topk(masked_scores, k=topk_k, axis=-1)[1]
        # Clamp indices to valid range: paddle.topk may return garbage indices
        # for -inf input values
        topk_indices = paddle.clip(
            topk_indices, min=0, max=masked_scores.shape[-1] - 1
        )

        FusedDSAIndexerLoss._last_topk_indices = topk_indices.detach()

        # Step 3: Compute KL loss (use masked_scores)
        indexer_loss = _compute_dsa_indexer_loss(
            masked_scores,
            topk_indices,
            query,
            key,
            softmax_scale,
            loss_coeff,
            sparse_loss,
            tp_group,
        )

        ctx.save_for_backward(q, weights, k, query, key, topk_indices)
        ctx.softmax_scale = softmax_scale
        ctx.loss_coeff = loss_coeff
        ctx.sparse_loss = sparse_loss
        ctx.tp_group = tp_group

        return indexer_loss

    @staticmethod
    def backward(ctx, grad_loss: Tensor):
        """Backward: recompute and manually backprop to (q, weights, k).

        Returns 6 gradients for the 6 Tensor inputs to forward:
            q, weights, k, query, key, mask
        (Paddle PyLayer only counts Tensor params, not float/int/bool/None.)
        """
        q, weights, k, query, key, topk_indices = ctx.saved_tensor()

        grad_q, grad_weights, grad_k = _bwd_fused_indexer_loss(
            q,
            weights,
            k,
            query,
            key,
            topk_indices,
            ctx.softmax_scale,
            ctx.loss_coeff,
            ctx.sparse_loss,
            grad_loss,
            ctx.tp_group,
        )

        return grad_q, grad_weights, grad_k, None, None, None


class DSAIndexerLossAutoScaler(paddle.autograd.PyLayer):
    """Attaches indexer_loss to the backward graph without changing output value."""

    _main_loss_backward_scale: Tensor | None = None

    @staticmethod
    def forward(ctx, output: Tensor, indexer_loss: Tensor) -> Tensor:
        ctx.save_for_backward(indexer_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (indexer_loss,) = ctx.saved_tensor()
        scale = DSAIndexerLossAutoScaler._main_loss_backward_scale
        if scale is None:
            scale = paddle.ones([1], dtype=indexer_loss.dtype)
        scaled_grad = paddle.ones_like(indexer_loss) * scale
        return grad_output, scaled_grad

    @staticmethod
    def set_loss_scale(scale: Tensor):
        DSAIndexerLossAutoScaler._main_loss_backward_scale = scale


logger = logging.getLogger(__name__)


class DSAIndexerLossLoggingHelper:
    """Helper class for logging sparse attention indexer losses across layers and ranks."""

    tracker: dict = {}

    @staticmethod
    def save_loss_to_tracker(
        loss: Tensor,
        layer_number: int,
        num_layers: int,
        reduce_group=None,
        avg_group=None,
    ):
        """Save the indexer loss for logging.

        Args:
            loss: The loss tensor (scalar).
            layer_number: Layer index of the loss, 1-indexed.
            num_layers: The number of total layers.
            reduce_group: The group for reducing the loss.
            avg_group: The group for averaging the loss.
        """
        if layer_number is None:
            return

        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" not in tracker:
            tracker["values"] = paddle.zeros([num_layers])
        tracker["values"][layer_number - 1] += loss.detach()
        tracker["reduce_group"] = reduce_group
        tracker["avg_group"] = avg_group

    @staticmethod
    def clean_loss_in_tracker():
        """Clear the indexer losses."""
        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" in tracker:
            tracker["values"].zero_()
        tracker["reduce_group"] = None
        tracker["avg_group"] = None

    @staticmethod
    def reduce_loss_in_tracker():
        """Collect and reduce the indexer losses across ranks."""
        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" not in tracker:
            return
        values = tracker["values"]

        # PP all-reduce
        pp_group = parallel_state.get_pipeline_model_parallel_group(
            check_initialized=False
        )
        if pp_group is not None and pp_group.nranks > 1:
            paddle.distributed.all_reduce(values, group=pp_group)

        # TP reduce
        if tracker.get("reduce_group") is not None:
            paddle.distributed.all_reduce(values, group=tracker["reduce_group"])

        # CP avg
        if tracker.get("avg_group") is not None:
            paddle.distributed.all_reduce(values, group=tracker["avg_group"])
            values /= tracker["avg_group"].nranks

        # DP avg
        dp_group = parallel_state.get_data_parallel_group(
            check_initialized=False
        )
        if dp_group is not None and dp_group.nranks > 1:
            paddle.distributed.all_reduce(values, group=dp_group)
            values /= dp_group.nranks

    @staticmethod
    def track_indexer_metrics(
        loss_scale: float,
        iteration: int,
        writer=None,
        total_loss_dict: dict | None = None,
    ):
        """Track the sparse attention indexer metrics for logging.

        Args:
            loss_scale: Scale factor for the loss (e.g. 1/num_microbatches).
            iteration: Current training iteration.
            writer: TensorBoard writer (optional).
            total_loss_dict: Dictionary to accumulate total losses (optional).
        """
        DSAIndexerLossLoggingHelper.reduce_loss_in_tracker()
        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" not in tracker:
            return

        indexer_loss_values = tracker["values"] * loss_scale
        num_layers = indexer_loss_values.shape[0]
        avg_indexer_loss = indexer_loss_values.sum() / num_layers

        if total_loss_dict is not None:
            if "indexer loss" in total_loss_dict:
                total_loss_dict["indexer loss"] += avg_indexer_loss
            else:
                total_loss_dict["indexer loss"] = avg_indexer_loss

        if writer is not None:
            writer.add_scalar(
                "indexer loss", avg_indexer_loss.item(), iteration
            )

        logger.info(
            "Iteration %d | indexer loss: %.6f",
            iteration,
            avg_indexer_loss.item(),
        )

        DSAIndexerLossLoggingHelper.clean_loss_in_tracker()


# ---------------------------------------------------------------------------
# MLASelfAttentionWithDSA — extends upstream MLASelfAttention
# ---------------------------------------------------------------------------
class MLASelfAttentionWithDSA(MLASelfAttention):
    """MLA Self-attention with DeepSeek Sparse Attention (DSA) Indexer.

    Extends the upstream MLASelfAttention by:
    1. Reusing parent's get_query_key_value_tensors() for Q/K/V + q_compressed.
    2. Overriding forward() to run the DSA Indexer, build a sparse mask,
       and use unfused bmm attention.

    The Indexer needs q_compressed (normed, from MLA down-projection) and
    hidden_states in [b, s, h] format. The parent's get_query_key_value_tensors
    now returns (query, key, value, q_compressed, kv_compressed)
    """

    def __init__(
        self,
        config: TransformerConfig,
        sublayers_spec: MLASelfAttentionSublayersSpec,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection | None = None,
    ):
        super().__init__(
            config=config,
            sublayers_spec=sublayers_spec,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
        )

        # DSA Indexer
        self.indexer = Indexer(config, layer_number)

        # DSA loss config
        self.dsa_indexer_loss_coeff = getattr(
            config, "dsa_indexer_loss_coeff", None
        )
        self.dsa_indexer_use_sparse_loss = getattr(
            config, "dsa_indexer_use_sparse_loss", False
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        attn_mask_startend_row_indices: Tensor | None = None,
        key_value_states=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        in_recompute: bool = False,
        position_ids=None,
    ):
        """Forward: MLA projections + DSA Indexer + sparse attention.

        Overrides the parent forward to:
        1. Reuse parent's get_query_key_value_tensors for Q/K/V + q_compressed
        2. Prepare Indexer inputs from returned q_compressed (no re-computation)
        3. Run DSA Indexer to get top-k indices
        4. Build sparse mask and use unfused bmm attention
        5. Compute DSA Indexer KL loss if enabled
        """
        assert rotary_pos_emb is None, (
            "Rotary position embeddings should not be passed into MLA."
        )
        assert attention_bias is None
        assert rotary_pos_cos is None and rotary_pos_sin is None
        assert packed_seq_params is None, (
            "packed_seq_params is not supported yet."
        )

        # =====================
        # Query, Key, Value + compressed intermediates
        # =====================
        query, key, value, q_compressed, kv_compressed = (
            self.get_query_key_value_tensors(
                hidden_states,
                key_value_states,
                position_ids,
                packed_seq_params,
            )
        )

        #   Non-SP: batch-first [b, s, ...]
        #   SP:     seq-first   [s/tp, b, ...]
        if self.config.sequence_parallel:
            # SP: q_compressed [s/tp, b, q_lora_rank] -> gather -> [s, b, q_lora_rank]
            #     -> transpose -> [b, s, q_lora_rank]
            indexer_q_latent = gather_from_sequence_parallel_region(
                q_compressed,
                tensor_parallel_output_grad=True,
                group=self.pg_collection.tp,
            ).detach()
            indexer_hidden = gather_from_sequence_parallel_region(
                hidden_states,
                tensor_parallel_output_grad=True,
                group=self.pg_collection.tp,
            ).detach()
            indexer_q_latent = indexer_q_latent.transpose([1, 0, 2])
            indexer_hidden = indexer_hidden.transpose([1, 0, 2])
        else:
            # Non-SP: already batch-first [b, s, ...], no transpose needed
            indexer_q_latent = q_compressed.detach()
            indexer_hidden = hidden_states.detach()

        # Convert query/key/value to sequence-first [s, b, n, h] for DSA
        if not self.config.sequence_parallel:
            # Non-SP: [b, s, n, h] -> [s, b, n, h]
            query = query.transpose([1, 0, 2, 3])
            key = key.transpose([1, 0, 2, 3])
            value = value.transpose([1, 0, 2, 3])
        # SP: already seq-first [s/tp, b, n, h], no transpose needed

        # indexer_hidden: [b, s, h]
        # indexer_q_latent: [b, s, q_lora_rank]
        # query: [s, b, n, h]
        # key: [s, b, n, h]
        # value: [s, b, n, h]

        # Get RoPE freqs for Indexer (non-interleaved, computed from rotary_pos_emb)
        # Re-compute from self.rotary_pos_emb since parent doesn't expose it
        with paddle.no_grad():
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                hidden_states, self.config, packed_seq_params
            )
            packed_seq = (
                packed_seq_params is not None
                and packed_seq_params.qkv_format == "thd"
            )
            assert not self.config.apply_rope_fusion, (
                "DSA Indexer requires unfused RoPE (apply_rope_fusion=False). "
                "Fused RoPE returns cos/sin instead of freqs, which the Indexer cannot use."
            )
            indexer_mscale = 1.0
            if self.config.rope_type == "rope":
                indexer_freqs = self.rotary_pos_emb(
                    rotary_seq_len, packed_seq=packed_seq
                )
            else:
                indexer_freqs, indexer_mscale = self.rotary_pos_emb(
                    rotary_seq_len, packed_seq=packed_seq
                )
            # rotary_pos_emb returns [1, seq_len, 1, dim].
            # Indexer tensors are batch-first [b, s, heads, head_dim], so keep
            # freqs as 4D [1, seq_len, 1, dim] for correct broadcasting.
            # Do NOT squeeze(0) — that would make it [seq_len, 1, dim] (seq-first)
            # which causes _apply_rotary_pos_emb_bshd to mis-broadcast.
            if indexer_freqs is not None and (
                packed_seq_params is None
                or self.config.context_parallel_size == 1
            ):
                # Use the actual full sequence length from gathered indexer input,
                # not the potentially-sharded hidden_states.shape[0] (s/tp in SP mode).
                actual_seq_len = indexer_hidden.shape[
                    1
                ]  # [b, s, h] — always full seq
                indexer_freqs = indexer_freqs[
                    :, 0:actual_seq_len
                ]  # slice seq dim (dim 1)

            # MLA's YarnRotaryEmbedding generates interleaved freqs [θ₁,θ₁,θ₂,θ₂,...]
            # when config.rotary_interleaved=True, but Indexer uses non-interleaved
            # RoPE which expects half-half freqs [θ₁,θ₂,...,θ₁,θ₂,...].
            # Convert format here to match Indexer's rotary_interleaved=False.
            # if self.config.rotary_interleaved and indexer_freqs is not None:
            #     indexer_freqs = paddle.concat(
            #         [indexer_freqs[..., 0::2], indexer_freqs[..., 1::2]],
            #         axis=-1,
            #     )

        indexer_seq_len = indexer_hidden.shape[1]  # [b, s, h] — always full seq
        indexer_causal_mask = paddle.triu(
            paddle.full(
                [indexer_seq_len, indexer_seq_len],
                float("-inf"),
                dtype="float32",
            ),
            diagonal=1,
        )  # [s, s]
        if attention_mask is not None:
            # attention_mask: [b, 1, sq, sk] — may contain padding info beyond causal
            indexer_float_mask = attention_mask.cast(
                "float32"
            ) + indexer_causal_mask.unsqueeze(0).unsqueeze(0)  # [b, 1, s, s]
        else:
            indexer_float_mask = indexer_causal_mask.unsqueeze(0).unsqueeze(
                0
            )  # [1, 1, s, s]

        # Indexer forward_before_topk runs WITH gradient tracking so that
        # FusedDSAIndexerLoss can backprop through (q, weights, k) to
        # Indexer parameters (wq_b, wk, weights_proj).
        q_idx, k_idx, weights_idx = self.indexer.forward_before_topk(
            indexer_hidden,
            indexer_q_latent,
            indexer_freqs,
            mscale=indexer_mscale,
        )

        # Convert Indexer outputs from batch-first [b, s, h, d] to
        # sequence-first [s, b, h, d]
        q_idx_sf = q_idx.transpose([1, 0, 2, 3])  # [sq, b, h, d]
        k_idx_sf = k_idx.transpose([1, 0, 2])  # [sk, b, d]
        weights_idx_sf = weights_idx.transpose([1, 0, 2])  # [sq, b, h]

        # FusedDSAIndexerLoss: compute index_scores + topk + KL loss inside PyLayer,
        # with full manual backward to (q, weights, k).
        if self.training and self.dsa_indexer_loss_coeff is not None:
            indexer_loss = FusedDSAIndexerLoss.apply(
                q_idx_sf,
                weights_idx_sf,
                k_idx_sf,
                query.detach(),
                key.detach(),
                self.softmax_scale,
                self.indexer.index_topk,
                float(self.dsa_indexer_loss_coeff),
                indexer_float_mask,
                bool(self.dsa_indexer_use_sparse_loss),
                self.pg_collection.tp
                if self.pg_collection.tp is not None
                and self.pg_collection.tp.nranks > 1
                else None,
            )
            topk_indices = FusedDSAIndexerLoss._last_topk_indices
        else:
            # Inference or no loss: compute index_scores + topk directly
            index_scores, topk_indices = self.indexer.compute_index_scores(
                q_idx, k_idx, weights_idx, indexer_float_mask
            )
            topk_indices = topk_indices.detach()

        # =====================
        # Build sparse mask
        # =====================
        seqlen = query.shape[0]  # [s, b, nhpp, hd]
        bsz = query.shape[1]

        index_mask = paddle.full(
            [bsz, seqlen, seqlen],
            fill_value=float("-inf"),
            dtype="float32",
        )
        zeros = paddle.zeros(
            [
                topk_indices.shape[0],
                topk_indices.shape[1],
                topk_indices.shape[2],
            ],
            dtype="float32",
        )
        index_mask = paddle.put_along_axis(
            index_mask, topk_indices, zeros, axis=-1
        )
        # Merge causal + index into [b, s, s], then unsqueeze to [b, 1, s, s]
        # causal_mask is [s, s], reuse the one built for indexer (same seqlen)
        causal_mask = paddle.triu(
            paddle.full([seqlen, seqlen], float("-inf"), dtype="float32"),
            diagonal=1,
        )
        index_mask = index_mask + causal_mask.unsqueeze(0)
        combined_mask = index_mask.unsqueeze(1)  # [b, 1, s, s]

        if attention_mask is not None:
            combined_mask = attention_mask.cast("float32") + combined_mask

        # =====================
        # Core attention (unfused bmm for DSA)
        # =====================
        query = query.transpose([1, 0, 2, 3]).contiguous()
        key = key.transpose([1, 0, 2, 3]).contiguous()
        value = value.transpose([1, 0, 2, 3]).contiguous()

        if self.recompute_core_attention and self.training:
            core_attn_out = recompute(
                _unfused_dsa_attention,
                query,
                key,
                value,
                combined_mask.clone() if combined_mask is not None else None,
                self.softmax_scale,
            )
        else:
            core_attn_out = _unfused_dsa_attention(
                query,
                key,
                value,
                combined_mask,
                self.softmax_scale,
            )

        # =====================
        # Output projection
        # =====================
        if self.config.sequence_parallel:
            core_attn_out = core_attn_out.transpose([1, 0, 2]).contiguous()
        output, bias = self.o_proj(core_attn_out)

        # =====================
        # DSA Indexer KL loss (already computed by FusedDSAIndexerLoss above)
        # =====================
        if self.training and self.dsa_indexer_loss_coeff is not None:
            if self.dsa_indexer_loss_coeff > 0:
                DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                    loss=indexer_loss,
                    layer_number=self.layer_number,
                    num_layers=self.config.num_hidden_layers,
                )
            output = DSAIndexerLossAutoScaler.apply(output, indexer_loss)

        return output, bias
