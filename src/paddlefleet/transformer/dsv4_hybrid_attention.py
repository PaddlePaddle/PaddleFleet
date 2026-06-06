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
DeepSeekV4 Hybrid Attention with Compressed Sparse Attention.

Ported from Megatron-LM experimental_attention_variant/deepseek_v4_hybrid_attention.py
(commit bf4e1db).

Components:
  - DSv4HybridAttention: Base class with inverse RoPE, grouped output projection
  - DSv4HybridSelfAttention: Self-attention with Q low-rank, single-head KV
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import paddle
from paddle import Tensor, nn
from paddle.distributed.fleet.meta_parallel import LayerSpec, build_spec_layer

from paddlefleet.models.common.embeddings.rope_utils import (
    _apply_rotary_pos_emb_bshd,
)
from paddlefleet.models.common.embeddings.rotary_pos_embedding import (
    RotaryEmbedding,
)
from paddlefleet.models.common.embeddings.yarn_rotary_pos_embedding import (
    YarnRotaryEmbedding,
)
from paddlefleet.transformer.attention import Attention

if TYPE_CHECKING:
    from paddlefleet.process_groups_config import ProcessGroupCollection
    from paddlefleet.transformer.enums import AttnMaskType
    from paddlefleet.transformer.transformer_config import TransformerConfig


class _DSV4AttentionInputBranches(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x: Tensor):
        return x.clone(), x.clone(), x.clone()

    @staticmethod
    def backward(ctx, q_grad: Tensor, kv_grad: Tensor, core_x_grad: Tensor):
        return (q_grad + kv_grad) + core_x_grad


def _dsv4_attention_split_input_branches(tensor: Tensor):
    if os.environ.get("DSV4_FLEET_ATTN_INPUT_ORDER", "0") != "1":
        return tensor, tensor, tensor
    if tensor is None or not isinstance(tensor, paddle.Tensor) or tensor.stop_gradient:
        return tensor, tensor, tensor
    return _DSV4AttentionInputBranches.apply(tensor)


class _DSV4QRMSNorm(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, q: Tensor, eps: float) -> Tensor:
        r = paddle.rsqrt(q.square().mean(axis=-1, keepdim=True) + eps)
        ctx.save_for_backward(q, r)
        return q * r

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        q, r = ctx.saved_tensor()
        hidden_size = q.shape[-1]
        grad_r = (grad_output * q).sum(axis=-1, keepdim=True)
        grad_q = grad_output * r
        grad_add = grad_r * (-0.5) * (r * r * r)
        grad_q = grad_q + (paddle.full_like(q, 2.0) * q) * (grad_add / hidden_size)
        return grad_q


def _q_rms_norm(q: Tensor, eps: float) -> Tensor:
    """RMS normalization for query (no learnable weight)."""
    return _DSV4QRMSNorm.apply(q, eps)


# ---------------------------------------------------------------------------
# Sublayers spec dataclass
# ---------------------------------------------------------------------------


@dataclass
class DSv4HybridSelfAttentionSublayersSpec:
    """Sublayer specifications for DSv4 Hybrid Self-Attention."""

    linear_q_down_proj: type | LayerSpec = None
    linear_q_up_proj: type | LayerSpec = None
    linear_kv_proj: type | LayerSpec = None
    core_attention: type | LayerSpec | None = None
    o_proj: type | LayerSpec = None
    q_layernorm: type | LayerSpec = None
    kv_layernorm: type | LayerSpec = None


# ---------------------------------------------------------------------------
# DSv4HybridAttention
# ---------------------------------------------------------------------------


class DSv4HybridAttention(Attention):
    """DSv4 Hybrid Attention with CSA core attention, inverse RoPE, and grouped output.

    This class:
    1. Builds per-layer RotaryEmbedding (with configurable base for compressed layers)
    2. Builds CompressedSparseAttention as core attention
    3. Applies inverse RoPE on attention output
    4. Performs grouped low-rank output projection
    """

    def __init__(
        self,
        config: TransformerConfig,
        sublayers_spec: DSv4HybridSelfAttentionSublayersSpec,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection = None,
        is_mtp_layer: bool = False,
    ):
        super().__init__(
            config=config,
            sublayers_spec=sublayers_spec,
            layer_number=layer_number,
            attention_type=attention_type,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
        )

        self.num_attention_heads = config.num_attention_heads
        self.v_head_dim = config.v_head_dim
        self.qk_pos_emb_head_dim = config.qk_pos_emb_head_dim or 0
        self.query_projection_size = self.num_attention_heads * self.v_head_dim
        self.q_head_dim = self.v_head_dim
        self.key_hidden_size = self.q_head_dim
        self.val_hidden_size = self.v_head_dim

        # Per-layer compress ratio
        if is_mtp_layer:
            layer_idx = self.config.num_hidden_layers + layer_number
            compress_ratio = self.config.csa_compress_ratios[layer_idx]
        else:
            compress_ratio = self.config.csa_compress_ratios[layer_number]
        # Per-layer RoPE (potentially different base for compressed layers)
        rope_base = getattr(config, "rotary_base", 10000)
        if compress_ratio > 1:
            rope_base = config.csa_compress_rotary_base

        use_compressed_yarn = compress_ratio > 1
        if not use_compressed_yarn:
            self.rotary_pos_emb = RotaryEmbedding(
                self.qk_pos_emb_head_dim,
                rotary_percent=getattr(config, "rotary_percent", 1.0),
                rotary_base=rope_base,
            )
        else:
            self.rotary_pos_emb = YarnRotaryEmbedding(
                self.qk_pos_emb_head_dim,
                rotary_base=rope_base,
                scaling_factor=getattr(config, "rotary_scaling_factor", 40),
                original_max_position_embeddings=getattr(
                    config, "original_max_position_embeddings", 4096
                ),
                beta_fast=getattr(config, "beta_fast", 32),
                beta_slow=getattr(config, "beta_slow", 1),
                mscale=getattr(config, "mscale", 1.0),
                mscale_all_dim=getattr(config, "mscale_all_dim", 0.0),
            )

        self.core_attention = build_spec_layer(
            sublayers_spec.core_attention,
            config=config,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type=attention_type,
            attention_dropout=None,
            softmax_scale=getattr(config, "softmax_scale", None),
            k_channels=self.q_head_dim,
            v_channels=self.v_head_dim,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=1,
            cp_comm_type=cp_comm_type,
            pg_collection=self.pg_collection,
            compress_ratio=compress_ratio,
            rotary_pos_emb=self.rotary_pos_emb,
        )

        # Grouped output projection
        self.o_local_groups = config.o_groups
        assert self.query_projection_size % config.o_groups == 0, (
            "num_attention_heads * v_head_dim must be divisible by o_groups"
        )
        group_proj_in_size = self.query_projection_size // config.o_groups
        group_proj_out_size = config.o_groups * config.o_lora_rank

        self.linear_o_group_proj = self.create_parameter(
            shape=[group_proj_out_size, group_proj_in_size],
            dtype=config.dtype if hasattr(config, "dtype") else "bfloat16",
            default_initializer=nn.initializer.Normal(
                std=getattr(config, "init_method_std", 0.02)
            ),
        )

        linear_proj_in_size = config.o_groups * config.o_lora_rank
        self.o_proj = build_spec_layer(
            sublayers_spec.o_proj,
            linear_proj_in_size,
            config.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_group=self.pg_collection.tp,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        **kwargs,
    ) -> tuple[Tensor, None]:
        """Forward pass.

        Args:
            hidden_states: [b, sq, hidden_size]
            attention_mask: optional mask

        Returns:
            (output [b, sq, hidden_size], bias=None)
        """
        q_input, kv_input, core_x = _dsv4_attention_split_input_branches(hidden_states)
        # Get Q, K, V tensors
        query, key, value, q_compressed, kv_compressed = (
            self.get_query_key_value_tensors(q_input, kv_input=kv_input)
        )
        # Core attention (CompressedSparseAttention)
        core_attn_out = self.core_attention(
            query,
            key,
            value,
            attention_mask,
            x=core_x,
            qr=q_compressed,
        )
        # core_attn_out: [b, sq, np * v_head_dim]

        # Inverse RoPE on last qk_pos_emb_head_dim of each head
        b, sq, _ = core_attn_out.shape
        pos_dim = self.qk_pos_emb_head_dim
        nope_dim = self.v_head_dim - pos_dim

        if pos_dim > 0:
            core_attn_out = core_attn_out.reshape(
                [b, sq, self.num_attention_heads, self.v_head_dim]
            )
            # Get RoPE frequencies for inverse
            _rope_result = self.rotary_pos_emb(sq, packed_seq=False)
            if isinstance(_rope_result, tuple):
                freqs, mscale = _rope_result
            else:
                freqs, mscale = _rope_result, 1.0
            # DSv4 reference uses pure norm-preserving RoPE; YaRN's mscale is not applied.
            mscale = 1.0

            content_part = core_attn_out[..., :nope_dim]
            rot_part = core_attn_out[..., nope_dim:]

            rot_part = _apply_rotary_pos_emb_bshd(
                rot_part,
                freqs,
                mscale=mscale,
                rotary_interleaved=False,
                multi_latent_attention=True,
                inverse=True,
                mla_output_remove_interleaving=True,
            )
            core_attn_out = paddle.concat([content_part, rot_part], axis=-1)
            core_attn_out = core_attn_out.reshape([b, sq, -1])
        # Grouped output projection
        core_attn_out = core_attn_out.reshape([b, sq, self.o_local_groups, -1])
        wo_a_weight = self.linear_o_group_proj.reshape(
            [self.o_local_groups, self.config.o_lora_rank, -1]
        )
        core_attn_out = paddle.einsum(
            "...gd,grd->...gr", core_attn_out, wo_a_weight
        )
        core_attn_out = core_attn_out.reshape([b, sq, -1])
        # Output projection
        output, bias = self.o_proj(core_attn_out)

        return output, bias

    def get_query_key_value_tensors(self, hidden_states: Tensor):
        """Override in subclass."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# DSv4HybridSelfAttention
# ---------------------------------------------------------------------------


class DSv4HybridSelfAttention(DSv4HybridAttention):
    """DSv4 Hybrid Self-Attention with Q low-rank decomposition and single-head KV.

    Q path: hidden -> q_down_proj -> q_layernorm -> q_up_proj -> rms_norm -> RoPE
    KV path: hidden -> kv_proj -> kv_layernorm -> RoPE (single head, key == value)
    """

    def __init__(
        self,
        config: TransformerConfig,
        sublayers_spec: DSv4HybridSelfAttentionSublayersSpec,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection = None,
        is_mtp_layer: bool = False,
    ):
        super().__init__(
            config=config,
            sublayers_spec=sublayers_spec,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
        )

        self.q_lora_rank = config.q_lora_rank
        q_head_dim = self.v_head_dim  # In DSv4 Hybrid, q_head_dim == v_head_dim

        # Q down projection: hidden_size -> q_lora_rank (duplicated)
        self.linear_q_down_proj = build_spec_layer(
            sublayers_spec.linear_q_down_proj,
            config.hidden_size,
            config.q_lora_rank,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            skip_weight_param_allocation=False,
            tp_group=None,
        )

        # Q layernorm
        self.q_layernorm = build_spec_layer(
            sublayers_spec.q_layernorm,
            config=config,
            hidden_size=config.q_lora_rank,
            eps=getattr(config, "rms_norm_eps", 1e-5),
        )

        # Q up projection: q_lora_rank -> num_heads * q_head_dim (column parallel)
        self.linear_q_up_proj = build_spec_layer(
            sublayers_spec.linear_q_up_proj,
            config.q_lora_rank,
            self.num_attention_heads * q_head_dim,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_group=self.pg_collection.tp,
        )

        # KV projection: hidden_size -> v_head_dim (single head)
        self.linear_kv_proj = build_spec_layer(
            sublayers_spec.linear_kv_proj,
            config.hidden_size,
            config.v_head_dim,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_group=self.pg_collection.tp,
        )

        # KV layernorm
        self.kv_layernorm = build_spec_layer(
            sublayers_spec.kv_layernorm,
            config=config,
            hidden_size=config.v_head_dim,
            eps=getattr(config, "rms_norm_eps", 1e-5),
        )

    def get_query_key_value_tensors(
        self, hidden_states: Tensor, kv_input: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Derive query, key, value from hidden_states.

        Args:
            hidden_states: [b, sq, hidden_size]

        Returns:
            query: [b, sq, num_heads, v_head_dim]
            key:   [b, sq, 1, v_head_dim]
            value: [b, sq, 1, v_head_dim]
            q_compressed: [b, sq, q_lora_rank]
            kv_compressed: [b, sq, hidden_size] (== hidden_states)
        """
        b, sq, _ = hidden_states.shape
        kv_hidden_states = hidden_states if kv_input is None else kv_input

        # Q path
        q_compressed, _ = self.linear_q_down_proj(
            hidden_states
        )  # [b, sq, q_lora_rank]
        q_compressed = self.q_layernorm(q_compressed)

        q, _ = self.linear_q_up_proj(q_compressed)  # [b, sq, n * v_head_dim]
        q = q.reshape([b, sq, self.num_attention_heads, self.v_head_dim])
        q = _q_rms_norm(q, getattr(self.config, "rms_norm_eps", 1e-5))

        # KV path
        kv, _ = self.linear_kv_proj(kv_hidden_states)  # [b, sq, v_head_dim]
        kv = self.kv_layernorm(kv)

        # Apply RoPE to both Q and KV
        pos_dim = self.qk_pos_emb_head_dim
        nope_dim = self.v_head_dim - pos_dim

        if pos_dim > 0:
            # Get RoPE frequencies
            _rope_result = self.rotary_pos_emb(sq, packed_seq=False)
            if isinstance(_rope_result, tuple):
                freqs, mscale = _rope_result
            else:
                freqs, mscale = _rope_result, 1.0
            # DSv4 reference uses pure norm-preserving RoPE; YaRN's mscale is not applied.
            mscale = 1.0

            # Q RoPE: split nope/pe, apply RoPE to pe part
            q_nope = q[..., :nope_dim]
            q_pe = q[..., nope_dim:]
            q_pe = _apply_rotary_pos_emb_bshd(
                q_pe,
                freqs,
                mscale=mscale,
                rotary_interleaved=False,
                multi_latent_attention=True,
                mla_output_remove_interleaving=True,
            )
            query = paddle.concat([q_nope, q_pe], axis=-1)

            # KV RoPE: split nope/pe, apply RoPE to pe part
            kv_nope = kv[..., :nope_dim]
            kv_pe = kv[..., nope_dim:]
            # Add head dim for RoPE: [b, sq, pos_dim] -> [b, sq, 1, pos_dim]
            kv_pe = kv_pe.unsqueeze(2)
            kv_pe = _apply_rotary_pos_emb_bshd(
                kv_pe,
                freqs,
                mscale=mscale,
                rotary_interleaved=False,
                multi_latent_attention=True,
                mla_output_remove_interleaving=True,
            )
            kv_pe = kv_pe.squeeze(2)
            kv = paddle.concat([kv_nope, kv_pe], axis=-1)
        else:
            query = q

        # Single head: key = value = [b, sq, 1, v_head_dim]
        key = kv.unsqueeze(2)
        value = key

        return query, key, value, q_compressed, hidden_states
