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
  - DSAIndexerLoss: KL-divergence loss for Indexer training
  - DSAIndexerLossAutoScaler: Loss scaling helper
  - MLASelfAttentionWithDSA: Subclass of MLASelfAttention with DSA integration

Reference: Megatron-LM/megatron/core/transformer/experimental_attention_variant/dsa.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import paddle
import paddle.nn.functional as F
from paddle import Tensor
from paddle.distributed.fleet.utils import recompute

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

    # Butterfly: iteratively halve and compute sum/diff pairs.
    # Uses paddle.stack (not in-place index assignment) to keep autograd intact.
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
    """Unfused DSA sparse attention (matches Megatron-Core unfused_dsa_fn).

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

    Reference: Megatron-LM dsa.py DSAIndexer
    """

    def __init__(self, config: TransformerConfig, layer_number: int):
        super().__init__()
        self.config = config

        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.index_topk = config.index_topk
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
        """Apply non-interleaved RoPE to the pe portion of x.

        Split order: [nope | pe], matching Megatron-Core dsa.py _apply_rope.

        Args:
            x: [..., head_dim] (nope_dim + rope_dim)
            freqs: RoPE frequencies
            mscale: YaRN concentration factor (1.0 for plain RoPE, ~1.37 for YaRN)
        """
        x_nope = x[..., : self.nope_head_dim]
        x_pe = x[..., self.nope_head_dim :]
        x_pe = _apply_rotary_pos_emb_bshd(
            x_pe,
            freqs,
            rotary_interleaved=False,
            multi_latent_attention=self.config.multi_latent_attention,
            mscale=mscale,
        )
        return paddle.concat([x_nope, x_pe], axis=-1)

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


# ---------------------------------------------------------------------------
# DSA Indexer Loss (PyLayer)
# ---------------------------------------------------------------------------


class DSAIndexerLoss(paddle.autograd.PyLayer):
    """Fused DSA Indexer KL-divergence loss.

    Trains the Indexer to predict which tokens receive high attention weights.
    Reference: Megatron-Core dsa.py FusedDSAIndexerLoss
    """

    @staticmethod
    def forward(
        ctx,
        index_scores: Tensor,  # [b, sq, sk]
        topk_indices: Tensor,  # [b, sq, topk]
        query: Tensor,  # [b, sq, nhpp, qk_head_dim] (DETACHED)
        key: Tensor,  # [b, sk, nhpp, qk_head_dim] (DETACHED)
        mla_softmax_scale: float,
        loss_coeff: float,
        sparse_loss: bool,
        tp_group,
    ) -> Tensor:
        b, sq, sk = index_scores.shape
        nhpp = query.shape[2]

        q_f = query.cast("float32")
        k_f = key.cast("float32")
        attention_scores = (
            paddle.einsum("bshd,bthd->bhst", q_f, k_f) * mla_softmax_scale
        )

        causal_mask = paddle.triu(
            paddle.full([sq, sk], float("-inf"), dtype="float32"),
            diagonal=1,
        )
        attention_scores = attention_scores + causal_mask.unsqueeze([0, 1])

        index_mask = paddle.full([b, sq, sk], float("-inf"), dtype="float32")
        index_mask = paddle.put_along_axis(
            index_mask,
            topk_indices,
            paddle.zeros_like(topk_indices, dtype="float32"),
            axis=-1,
        )

        masked_index_scores = index_scores.cast(
            "float32"
        ) + causal_mask.unsqueeze(0)
        if sparse_loss:
            attention_scores = attention_scores + index_mask.unsqueeze(1)
            masked_index_scores = masked_index_scores + index_mask

        attn_probs = F.softmax(attention_scores, axis=-1)
        idx_probs = F.softmax(masked_index_scores, axis=-1)

        attn_probs_sum = attn_probs.sum(axis=1)
        if tp_group is not None and tp_group.nranks > 1:
            paddle.distributed.all_reduce(attn_probs_sum, group=tp_group)

        target = attn_probs_sum / (
            attn_probs_sum.sum(axis=-1, keepdim=True) + 1e-10
        )

        kl = target * (
            paddle.log(target + 1e-10) - paddle.log(idx_probs + 1e-10)
        )
        kl_div = kl.sum(axis=-1).mean()
        indexer_loss = kl_div * loss_coeff

        ctx.save_for_backward(
            target, idx_probs, index_mask if sparse_loss else None
        )
        ctx.b = b
        ctx.sq = sq
        ctx.sparse_loss = sparse_loss
        ctx.loss_coeff = loss_coeff

        return indexer_loss

    @staticmethod
    def backward(ctx, grad_loss: Tensor):
        target, idx_probs, index_mask = ctx.saved_tensor()
        b, sq = ctx.b, ctx.sq
        sparse_loss = ctx.sparse_loss
        loss_coeff = ctx.loss_coeff
        sk = target.shape[-1]

        grad_idx_probs = (
            -target
            / (idx_probs + 1e-10)
            * (grad_loss.cast("float32") * loss_coeff / (b * sq))
        )
        sum_grad = (grad_idx_probs * idx_probs).sum(axis=-1, keepdim=True)
        grad_index_scores = idx_probs * (grad_idx_probs - sum_grad)

        causal_valid = paddle.tril(paddle.ones([sq, sk], dtype="bool"))
        if sparse_loss and index_mask is not None:
            valid_mask = causal_valid.unsqueeze(0) & (index_mask == 0)
        else:
            valid_mask = causal_valid.unsqueeze(0).expand([b, sq, sk])

        grad_index_scores = grad_index_scores * valid_mask.cast("float32")

        # Gradients for Tensor inputs only (Paddle PyLayer convention):
        # index_scores, topk_indices(None), query(None), key(None)
        return grad_index_scores.cast(idx_probs.dtype), None, None, None


class DSAIndexerLossAutoScaler(paddle.autograd.PyLayer):
    """Attaches indexer_loss to the backward graph without changing output value."""

    _main_loss_backward_scale: Tensor | None = None

    @staticmethod
    def forward(ctx, output: Tensor, indexer_loss: Tensor) -> Tensor:
        print(f"===========> indexer_loss: {indexer_loss}")
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
    now returns (query, key, value, q_compressed, kv_compressed) — aligned with
    Megatron-LM — so we directly reuse q_compressed for the Indexer path.
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
            config, "indexer_loss_coeff", None
        )
        self.dsa_indexer_use_sparse_loss = getattr(
            config, "indexer_use_sparse_loss", False
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

        # =====================
        # Query, Key, Value + compressed intermediates (aligned with Megatron)
        # =====================
        query, key, value, q_compressed, kv_compressed = (
            self.get_query_key_value_tensors(
                hidden_states,
                key_value_states,
                position_ids,
                packed_seq_params,
            )
        )

        # =====================
        # DSA Indexer inputs (reuse q_compressed from parent, no re-computation)
        # =====================
        # q_compressed: [s/tp, b, q_lora_rank] (SP) or [s, b, q_lora_rank] (non-SP)
        # Indexer needs full-sequence [b, s, ...] tensors, all detached.
        with paddle.no_grad():
            if self.config.sequence_parallel:
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
            else:
                indexer_q_latent = q_compressed.detach()
                indexer_hidden = hidden_states.detach()

            # Convert [s, b, ...] -> [b, s, ...]
            indexer_q_latent = indexer_q_latent.transpose([1, 0, 2])
            indexer_hidden = indexer_hidden.transpose([1, 0, 2])

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

        # Build causal float_mask for Indexer scoring, matching MG DSAttention.forward:
        # MG always passes a causal-aware mask to the Indexer so that topk selection
        # never picks future positions.  Without this, Indexer could select future
        # tokens which are then killed by the causal mask in attention → all-inf rows
        # → softmax NaN.
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

        index_scores, topk_indices = self.indexer(
            indexer_hidden,
            indexer_q_latent,
            indexer_freqs,
            indexer_float_mask,
            mscale=indexer_mscale,
        )
        # Detach topk_indices: int64 index tensor, no meaningful gradients.
        topk_indices = topk_indices.detach()

        # =====================
        # Build sparse mask
        # =====================
        if self.config.sequence_parallel:
            seqlen = query.shape[0]  # [s, b, nhpp, hd]
            bsz = query.shape[1]
        else:
            bsz = query.shape[0]  # [b, s, nhpp, hd]
            seqlen = query.shape[1]

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
        if self.config.sequence_parallel:
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
        # DSA Indexer KL loss
        # =====================
        if self.training and self.dsa_indexer_loss_coeff is not None:
            indexer_loss = DSAIndexerLoss.apply(
                index_scores,
                topk_indices,
                query.detach(),
                key.detach(),
                self.softmax_scale,
                float(self.dsa_indexer_loss_coeff),
                bool(self.dsa_indexer_use_sparse_loss),
                self.pg_collection.tp
                if self.pg_collection.tp.nranks > 1
                else None,
            )
            output = DSAIndexerLossAutoScaler.apply(output, indexer_loss)

        return output, bias
