# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""Paddle Qwen3_VL model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed.fleet.utils import recompute

from ...nn.activation import ACT2FN
from ...nn.attention.interface import ALL_ATTENTION_FUNCTIONS
from ...nn.linear import Linear as GeneralLinear
from ..cache_utils import Cache
from ..model_outputs import ModelOutput
from ..model_utils import PretrainedModel
from .configuration import Qwen3VLConfig, Qwen3VLVisionConfig

if TYPE_CHECKING:
    from .modeling_fleet import (
        Qwen3VLForCausalLMPipe,
        Qwen3VLForConditionalGeneration,
        Qwen3VLModel,
        Qwen3VLModelPipe,
    )


def __getattr__(name):
    if name == "Qwen3VLModel":
        from .modeling_fleet import Qwen3VLModel

        return Qwen3VLModel
    elif name == "Qwen3VLForConditionalGeneration":
        from .modeling_fleet import Qwen3VLForConditionalGeneration

        return Qwen3VLForConditionalGeneration
    elif name == "Qwen3VLForCausalLMPipe":
        from .modeling_fleet import Qwen3VLForCausalLMPipe

        return Qwen3VLForCausalLMPipe
    elif name == "Qwen3VLModelPipe":
        from .modeling_fleet import Qwen3VLModelPipe

        return Qwen3VLModelPipe

    raise AttributeError(f"module {__name__} has no attribute {name}")


class Qwen3VLVisionMLP(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_type = config.get("hidden_act", "silu")
        self.linear_fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias_attr=True)
        self.linear_fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias_attr=True)
        self.act_fn = ACT2FN[self.act_type]

    def forward(self, hidden_state):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


class Qwen3VLVisionPatchEmbed(nn.Layer):
    def __init__(
        self,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True)

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.reshape(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).reshape(-1, self.embed_dim)
        return hidden_states


class Qwen3VLVisionRotaryEmbedding(nn.Layer):
    inv_freq: paddle.Tensor

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (paddle.arange(0, dim, 2, dtype=paddle.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistable=False)

    def forward(self, seqlen: int) -> paddle.Tensor:
        seq = paddle.arange(seqlen, dtype=self.inv_freq.dtype)
        freqs = paddle.outer(seq, self.inv_freq)
        return freqs


class Qwen3VLVisionPatchMerger(nn.Layer):
    def __init__(
        self,
        config: Qwen3VLConfig,
        dim: int = None,
        context_dim: int = None,
        spatial_merge_size: int = None,
        use_postshuffle_norm: bool = False,
    ) -> None:
        super().__init__()
        context_dim = context_dim if context_dim is not None else config.hidden_size
        dim = dim if dim is not None else config.out_hidden_size
        spatial_merge_size = spatial_merge_size if spatial_merge_size is not None else config.spatial_merge_size

        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        norm_dim = self.hidden_size if use_postshuffle_norm else context_dim
        self.norm = nn.LayerNorm(norm_dim, epsilon=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, dim)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        if self.use_postshuffle_norm:
            x = self.norm(x.reshape([-1, self.hidden_size]))
            x = x.reshape([-1, self.hidden_size])
        else:
            x = self.norm(x)
            x = x.reshape([-1, self.hidden_size])

        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.cat([-x2, x1], axis=-1)


def apply_rotary_pos_emb_vision(q, k, cos, sin):
    """Applies Rotary Position Embedding to the query and key tensors."""
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    with paddle.amp.auto_cast(False):
        q, k = q.astype(dtype="float32"), k.astype(dtype="float32")
        cos, sin = cos.unsqueeze(-2).astype(dtype="float32"), sin.unsqueeze(-2).astype(dtype="float32")
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed.astype(orig_q_dtype), k_embed.astype(orig_k_dtype)


class Qwen3VLVisionAttention(nn.Layer):
    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1  # needed for eager attention
        self.qkv = GeneralLinear.create(
            self.dim,
            self.dim * 3,
            has_bias=True,
            linear_type="default",
        )
        self.proj = GeneralLinear.create(
            self.dim,
            self.dim,
            linear_type="default",
        )
        self.scaling = self.head_dim**-0.5
        self.config = config
        self.attention_dropout = 0.0
        self.is_causal = False

    def forward(
        self,
        hidden_states: paddle.Tensor,
        cu_seqlens: paddle.Tensor,
        rotary_pos_emb: Optional[paddle.Tensor] = None,
        position_embeddings: Optional[tuple[paddle.Tensor, paddle.Tensor]] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        **kwargs,
    ) -> paddle.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=None,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape([seq_length, -1]).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen3VLVisionBlock(nn.Layer):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3VLVisionAttention(config=config)
        self.mlp = Qwen3VLVisionMLP(config)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        cu_seqlens: paddle.Tensor,
        rotary_pos_emb: Optional[paddle.Tensor] = None,
        position_embeddings: Optional[tuple[paddle.Tensor, paddle.Tensor]] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        **kwargs,
    ) -> paddle.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            **kwargs,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen3VLPretrainedModel(PretrainedModel):
    config_class = Qwen3VLConfig
    base_model_prefix = "model"
    input_modalities = ["image", "video", "text"]
    _no_split_modules = ["Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"]
    _keys_to_ignore_on_load_unexpected = [r"self_attn.rotary_emb.inv_freq"]
    transpose_weight_keys = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "qkv",
        "gate_proj",
        "up_proj",
        "down_proj",
        "proj",
        r"linear_fc\d+",
    ]

    @classmethod
    def _gen_aoa_config(cls, config: Qwen3VLConfig):
        mapping = cls._checkpoint_conversion_mapping
        llm_target = next((v for v in mapping.values() if "language_model" in v), "language_model")
        visual_target = next((v for v in mapping.values() if "visual" in v), "visual")
        llm_prefix = f"{llm_target}." if not llm_target.endswith(".") else llm_target
        visual_prefix = f"{visual_target}." if not visual_target.endswith(".") else visual_target

        # language model
        aoa_config = {
            "aoa_statements": [
                f"model.language_model.embed_tokens.weight -> {llm_prefix}embed_tokens.weight",
                f"model.language_model.norm.weight -> {llm_prefix}norm.weight",
                f"model.language_model.layers.$LAYER_ID.input_layernorm.weight -> {llm_prefix}layers.$LAYER_ID.input_layernorm.weight",
                f"model.language_model.layers.$LAYER_ID.post_attention_layernorm.weight -> {llm_prefix}layers.$LAYER_ID.post_attention_layernorm.weight",
                f"model.language_model.layers.$LAYER_ID.self_attn.o_proj.weight^T -> {llm_prefix}layers.$LAYER_ID.self_attn.o_proj.weight",
                f"model.language_model.layers.$LAYER_ID.mlp.down_proj.weight^T -> {llm_prefix}layers.$LAYER_ID.mlp.down_proj.weight",
                f"model.language_model.layers.$LAYER_ID.self_attn.q_norm.weight -> {llm_prefix}layers.$LAYER_ID.self_attn.q_norm.weight",
                f"model.language_model.layers.$LAYER_ID.self_attn.k_norm.weight -> {llm_prefix}layers.$LAYER_ID.self_attn.k_norm.weight",
            ]
        }

        # visual model
        aoa_config["aoa_statements"] += (
            [
                f"model.visual.blocks.$LAYER_ID.attn.{x}.weight^T -> {visual_prefix}blocks.$LAYER_ID.attn.{x}.weight"
                for x in ("qkv", "proj")
            ]
            + [
                f"model.visual.blocks.$LAYER_ID.attn.{x}.bias -> {visual_prefix}blocks.$LAYER_ID.attn.{x}.bias"
                for x in ("qkv", "proj")
            ]
            + [
                f"model.visual.blocks.$LAYER_ID.mlp.linear_fc{x}.weight^T -> {visual_prefix}blocks.$LAYER_ID.mlp.linear_fc{x}.weight"
                for x in ("1", "2")
            ]
            + [
                f"model.visual.blocks.$LAYER_ID.mlp.linear_fc{x}.bias -> {visual_prefix}blocks.$LAYER_ID.mlp.linear_fc{x}.bias"
                for x in ("1", "2")
            ]
        )
        aoa_config["aoa_statements"] += [
            f"model.visual.patch_embed.proj.weight -> {visual_prefix}patch_embed.proj.weight",
            f"model.visual.patch_embed.proj.bias -> {visual_prefix}patch_embed.proj.bias",
            f"model.visual.pos_embed.weight -> {visual_prefix}pos_embed.weight",
            f"model.visual.merger.norm.weight -> {visual_prefix}merger.norm.weight",
            f"model.visual.merger.norm.bias -> {visual_prefix}merger.norm.bias",
            f"model.visual.blocks.$LAYER_ID.norm1.weight -> {visual_prefix}blocks.$LAYER_ID.norm1.weight",
            f"model.visual.blocks.$LAYER_ID.norm1.bias -> {visual_prefix}blocks.$LAYER_ID.norm1.bias",
            f"model.visual.blocks.$LAYER_ID.norm2.weight -> {visual_prefix}blocks.$LAYER_ID.norm2.weight",
            f"model.visual.blocks.$LAYER_ID.norm2.bias -> {visual_prefix}blocks.$LAYER_ID.norm2.bias",
        ]
        aoa_config["aoa_statements"] += [
            f"model.visual.merger.linear_fc1.weight^T -> {visual_prefix}merger.linear_fc1.weight",
            f"model.visual.merger.linear_fc1.bias -> {visual_prefix}merger.linear_fc1.bias",
            f"model.visual.merger.linear_fc2.weight^T -> {visual_prefix}merger.linear_fc2.weight",
            f"model.visual.merger.linear_fc2.bias -> {visual_prefix}merger.linear_fc2.bias",
        ]

        aoa_config["aoa_statements"] += [
            f"model.visual.deepstack_merger_list.$LAYER_ID.linear_fc1.weight^T -> {visual_prefix}deepstack_merger_list.$LAYER_ID.linear_fc1.weight",
            f"model.visual.deepstack_merger_list.$LAYER_ID.linear_fc1.bias -> {visual_prefix}deepstack_merger_list.$LAYER_ID.linear_fc1.bias",
            f"model.visual.deepstack_merger_list.$LAYER_ID.linear_fc2.weight^T -> {visual_prefix}deepstack_merger_list.$LAYER_ID.linear_fc2.weight",
            f"model.visual.deepstack_merger_list.$LAYER_ID.linear_fc2.bias -> {visual_prefix}deepstack_merger_list.$LAYER_ID.linear_fc2.bias",
            f"model.visual.deepstack_merger_list.$LAYER_ID.norm.weight -> {visual_prefix}deepstack_merger_list.$LAYER_ID.norm.weight",
            f"model.visual.deepstack_merger_list.$LAYER_ID.norm.bias -> {visual_prefix}deepstack_merger_list.$LAYER_ID.norm.bias",
        ]

        # attention qkv
        aoa_config["aoa_statements"] += [
            f"model.language_model.layers.$LAYER_ID.self_attn.q_proj.weight^T, model.language_model.layers.$LAYER_ID.self_attn.k_proj.weight^T, model.language_model.layers.$LAYER_ID.self_attn.v_proj.weight^T -> {llm_prefix}layers.$LAYER_ID.self_attn.qkv_proj.weight, fused_qkv, num_heads={config.text_config.num_attention_heads}, num_key_value_groups={config.text_config.num_key_value_heads}"
        ]
        if config.attention_bias:
            aoa_config["aoa_statements"] += [
                f"model.language_model.layers.$LAYER_ID.self_attn.q_proj.bias, model.language_model.layers.$LAYER_ID.self_attn.k_proj.bias, model.language_model.layers.$LAYER_ID.self_attn.v_proj.bias -> {llm_prefix}layers.$LAYER_ID.self_attn.qkv_proj.bias, fused_qkv, num_heads={config.text_config.num_attention_heads}, num_key_value_groups={config.text_config.num_key_value_heads}"
            ]

        # FFN
        aoa_config["aoa_statements"] += [
            f"model.language_model.layers.$LAYER_ID.mlp.gate_proj.weight^T, model.language_model.layers.$LAYER_ID.mlp.up_proj.weight^T -> {llm_prefix}layers.$LAYER_ID.mlp.up_gate_proj.weight, fused_ffn",
        ]

        # Qwen3_VLModel without lm_head
        if cls._tied_weights_keys:
            aoa_config["aoa_statements"] += [
                f"{'model.language_model.embed_tokens.weight' if config.tie_word_embeddings else 'lm_head.weight'} -> lm_head.weight",
            ]

        return aoa_config

    @classmethod
    def _gen_inv_aoa_config(cls, config: Qwen3VLConfig):
        mapping = cls._checkpoint_conversion_mapping
        llm_target = next((v for v in mapping.values() if "language_model" in v), "language_model")
        visual_target = next((v for v in mapping.values() if "visual" in v), "visual")
        llm_prefix = f"{llm_target}." if not llm_target.endswith(".") else llm_target
        visual_prefix = f"{visual_target}." if not visual_target.endswith(".") else visual_target

        # language model
        aoa_config = {
            "aoa_statements": [
                f"{llm_prefix}embed_tokens.weight -> model.language_model.embed_tokens.weight",
                f"{llm_prefix}norm.weight -> model.language_model.norm.weight",
                f"{llm_prefix}layers.$LAYER_ID.input_layernorm.weight -> model.language_model.layers.$LAYER_ID.input_layernorm.weight",
                f"{llm_prefix}layers.$LAYER_ID.post_attention_layernorm.weight -> model.language_model.layers.$LAYER_ID.post_attention_layernorm.weight",
                f"{llm_prefix}layers.$LAYER_ID.self_attn.o_proj.weight^T -> model.language_model.layers.$LAYER_ID.self_attn.o_proj.weight",
                f"{llm_prefix}layers.$LAYER_ID.mlp.down_proj.weight^T -> model.language_model.layers.$LAYER_ID.mlp.down_proj.weight",
                f"{llm_prefix}layers.$LAYER_ID.self_attn.q_norm.weight -> model.language_model.layers.$LAYER_ID.self_attn.q_norm.weight",
                f"{llm_prefix}layers.$LAYER_ID.self_attn.k_norm.weight -> model.language_model.layers.$LAYER_ID.self_attn.k_norm.weight",
            ]
        }

        # visual model
        aoa_config["aoa_statements"] += (
            [
                f"{visual_prefix}blocks.$LAYER_ID.attn.{x}.weight^T -> model.visual.blocks.$LAYER_ID.attn.{x}.weight"
                for x in ("qkv", "proj")
            ]
            + [
                f"{visual_prefix}blocks.$LAYER_ID.attn.{x}.bias -> model.visual.blocks.$LAYER_ID.attn.{x}.bias"
                for x in ("qkv", "proj")
            ]
            + [
                f"{visual_prefix}blocks.$LAYER_ID.mlp.linear_fc{x}.weight^T -> model.visual.blocks.$LAYER_ID.mlp.linear_fc{x}.weight"
                for x in ("1", "2")
            ]
            + [
                f"{visual_prefix}blocks.$LAYER_ID.mlp.linear_fc{x}.bias -> model.visual.blocks.$LAYER_ID.mlp.linear_fc{x}.bias"
                for x in ("1", "2")
            ]
        )
        aoa_config["aoa_statements"] += [
            f"{visual_prefix}patch_embed.proj.weight -> model.visual.patch_embed.proj.weight",
            f"{visual_prefix}patch_embed.proj.bias -> model.visual.patch_embed.proj.bias",
            f"{visual_prefix}pos_embed.weight -> model.visual.pos_embed.weight",
            f"{visual_prefix}merger.norm.weight -> model.visual.merger.norm.weight",
            f"{visual_prefix}merger.norm.bias -> model.visual.merger.norm.bias",
            f"{visual_prefix}blocks.$LAYER_ID.norm1.weight -> model.visual.blocks.$LAYER_ID.norm1.weight",
            f"{visual_prefix}blocks.$LAYER_ID.norm1.bias -> model.visual.blocks.$LAYER_ID.norm1.bias",
            f"{visual_prefix}blocks.$LAYER_ID.norm2.weight -> model.visual.blocks.$LAYER_ID.norm2.weight",
            f"{visual_prefix}blocks.$LAYER_ID.norm2.bias -> model.visual.blocks.$LAYER_ID.norm2.bias",
        ]
        aoa_config["aoa_statements"] += [
            f"{visual_prefix}merger.linear_fc1.weight^T -> model.visual.merger.linear_fc1.weight",
            f"{visual_prefix}merger.linear_fc1.bias -> model.visual.merger.linear_fc1.bias",
            f"{visual_prefix}merger.linear_fc2.weight^T -> model.visual.merger.linear_fc2.weight",
            f"{visual_prefix}merger.linear_fc2.bias -> model.visual.merger.linear_fc2.bias",
        ]
        aoa_config["aoa_statements"] += [
            f"{visual_prefix}deepstack_merger_list.$LAYER_ID.linear_fc1.weight^T -> model.visual.deepstack_merger_list.$LAYER_ID.linear_fc1.weight",
            f"{visual_prefix}deepstack_merger_list.$LAYER_ID.linear_fc1.bias -> model.visual.deepstack_merger_list.$LAYER_ID.linear_fc1.bias",
            f"{visual_prefix}deepstack_merger_list.$LAYER_ID.linear_fc2.weight^T -> model.visual.deepstack_merger_list.$LAYER_ID.linear_fc2.weight",
            f"{visual_prefix}deepstack_merger_list.$LAYER_ID.linear_fc2.bias -> model.visual.deepstack_merger_list.$LAYER_ID.linear_fc2.bias",
            f"{visual_prefix}deepstack_merger_list.$LAYER_ID.norm.weight -> model.visual.deepstack_merger_list.$LAYER_ID.norm.weight",
            f"{visual_prefix}deepstack_merger_list.$LAYER_ID.norm.bias -> model.visual.deepstack_merger_list.$LAYER_ID.norm.bias",
        ]

        # attention qkv
        aoa_config["aoa_statements"] += [
            f"{llm_prefix}layers.$LAYER_ID.self_attn.qkv_proj.weight  -> {llm_prefix}layers.$LAYER_ID.self_attn.q_proj.weight, {llm_prefix}layers.$LAYER_ID.self_attn.k_proj.weight, {llm_prefix}layers.$LAYER_ID.self_attn.v_proj.weight, fused_qkv, num_heads={config.text_config.num_attention_heads}, num_key_value_groups = {config.text_config.num_key_value_heads}",
        ]
        aoa_config["aoa_statements"] += [
            f"{llm_prefix}layers.{layer_id}.self_attn.{x}_proj.weight^T -> model.language_model.layers.{layer_id}.self_attn.{x}_proj.weight"
            for layer_id in range(config.text_config.num_hidden_layers)
            for x in ("q", "k", "v")
        ]
        if config.attention_bias:
            aoa_config["aoa_statements"] += [
                f"{llm_prefix}layers.$LAYER_ID.self_attn.qkv_proj.bias  -> model.language_model.layers.$LAYER_ID.self_attn.q_proj.bias, model.language_model.layers.$LAYER_ID.self_attn.k_proj.bias, model.language_model.layers.$LAYER_ID.self_attn.v_proj.bias, fused_qkv, num_heads={config.text_config.num_attention_heads}, num_key_value_groups = {config.text_config.num_key_value_heads}",
            ]

        # FFN
        aoa_config["aoa_statements"] += [
            f"{llm_prefix}layers.$LAYER_ID.mlp.up_gate_proj.weight -> {llm_prefix}layers.$LAYER_ID.mlp.gate_proj.weight, {llm_prefix}layers.$LAYER_ID.mlp.up_proj.weight, fused_ffn"
        ]
        aoa_config["aoa_statements"] += [
            f"{llm_prefix}layers.{layer_id}.mlp.{x}_proj.weight^T -> model.language_model.layers.{layer_id}.mlp.{x}_proj.weight"
            for layer_id in range(config.text_config.num_hidden_layers)
            for x in ("gate", "up")
        ]

        # Qwen3VLModel without lm_head
        if cls._tied_weights_keys:
            aoa_config["aoa_statements"] += [
                f"lm_head.weight -> {'_' if config.tie_word_embeddings else 'lm_head.weight'}",
            ]

        return aoa_config


class Qwen3VLVisionModel(Qwen3VLPretrainedModel):
    config_class = Qwen3VLVisionConfig
    _no_split_modules = ["Qwen3VLVisionBlock"]

    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.deepstack_visual_indexes = config.deepstack_visual_indexes
        self.deepstack_merger_list = nn.LayerList(
            [
                Qwen3VLVisionPatchMerger(
                    config=config,
                    use_postshuffle_norm=True,
                )
                for _ in range(len(config.deepstack_visual_indexes))
            ]
        )

        self.patch_embed = Qwen3VLVisionPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.LayerList([Qwen3VLVisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen3VLVisionPatchMerger(
            config=config,
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
        )
        self.gradient_checkpointing = False

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = paddle.arange(h).unsqueeze(1).expand([-1, w])
            hpos_ids = hpos_ids.reshape(
                [
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                ]
            )
            hpos_ids = hpos_ids.transpose(perm=[0, 2, 1, 3])
            hpos_ids = hpos_ids.flatten()

            wpos_ids = paddle.arange(w).unsqueeze(0).expand([h, -1])
            wpos_ids = wpos_ids.reshape(
                [
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                ]
            )
            wpos_ids = wpos_ids.transpose([0, 2, 1, 3])
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(paddle.stack(x=[hpos_ids, wpos_ids], axis=-1).tile(repeat_times=[t, 1]))
        pos_ids = paddle.cat(x=pos_ids, axis=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(start_axis=1)
        return rotary_pos_emb

    @paddle.jit.not_to_static
    def recompute_training_full(
        self,
        layer_module: nn.Layer,
        hidden_states: paddle.Tensor,
        cu_seqlens: paddle.Tensor,
        rotary_pos_emb: Optional[paddle.Tensor] = None,
        position_embeddings: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
    ):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        hidden_states = recompute(
            create_custom_forward(layer_module),
            hidden_states,
            cu_seqlens,
            rotary_pos_emb,
            position_embeddings,
        )
        return hidden_states

    def fast_pos_embed_interpolate(self, grid_thw):
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
        device = paddle.get_device()

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = paddle.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = paddle.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor.astype("float32")
            dw = w_idxs - w_idxs_floor.astype("float32")

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = paddle.tensor(idx_list, dtype=paddle.long, device=device)
        weight_tensor = paddle.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
        pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.tile([t, 1])
            pos_embed = (
                pos_embed.reshape([t, h // merge_size, merge_size, w // merge_size, merge_size, -1])
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = paddle.cat(patch_pos_embeds_permute)
        return patch_pos_embeds

    def forward(self, hidden_states: paddle.Tensor, grid_thw: paddle.Tensor) -> paddle.Tensor:
        """
        Args:
            hidden_states (`paddle.Tensor` of shape `(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`paddle.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `paddle.Tensor`: hidden_states.
        """
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        seq_len, _ = tuple(hidden_states.shape)

        hidden_states = hidden_states.reshape([seq_len, -1])
        rotary_pos_emb = rotary_pos_emb.reshape([seq_len, -1])
        emb = paddle.cat((rotary_pos_emb, rotary_pos_emb), axis=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = paddle.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            axis=0, dtype="int32"
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        indices_per_segment = paddle.stack(
            [
                cu_seqlens[1:],
                paddle.full_like(cu_seqlens[1:], cu_seqlens[-1]),
                paddle.zeros_like(cu_seqlens[:-1]),
                cu_seqlens[:-1],
            ],
            axis=1,
        )
        attn_mask_startend_row_indices = paddle.repeat_interleave(indices_per_segment, lengths, axis=0)[
            None, None, ...
        ]

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            cu_seqlens_now = cu_seqlens

            has_gradient = not hidden_states.stop_gradient
            if (
                self.config.recompute_granularity == "full"
                and self.config.recompute_method == "uniform"
                and self.config.recompute_num_layers == 1
                and has_gradient
            ):
                hidden_states = self.recompute_training_full(
                    blk,
                    hidden_states,
                    cu_seqlens=cu_seqlens_now,
                    position_embeddings=position_embeddings,
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                )
            else:
                hidden_states = blk(
                    hidden_states,
                    cu_seqlens=cu_seqlens_now,
                    position_embeddings=position_embeddings,
                    attn_mask_startend_row_indices=attn_mask_startend_row_indices,
                )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                    hidden_states
                )
                deepstack_feature_lists.append(deepstack_feature)

        hidden_states = self.merger(hidden_states)

        return hidden_states, deepstack_feature_lists


@dataclass
class Qwen3VLCausalLMOutputWithPast(ModelOutput):
    r"""
    loss (`paddle.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`paddle.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache)`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    rope_deltas (`paddle.Tensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    """

    loss: Optional[paddle.Tensor] = None
    logits: Optional[paddle.Tensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[paddle.Tensor]] = None
    attentions: Optional[tuple[paddle.Tensor]] = None
    rope_deltas: Optional[paddle.Tensor] = None


__all__ = [
    "Qwen3VLPretrainedModel",
    "Qwen3VLVisionMLP",
    "Qwen3VLVisionPatchEmbed",
    "Qwen3VLVisionRotaryEmbedding",
    "Qwen3VLVisionPatchMerger",
    "rotate_half",
    "apply_rotary_pos_emb_vision",
    "Qwen3VLVisionAttention",
    "Qwen3VLVisionBlock",
    "Qwen3VLVisionModel",
    "Qwen3VLCausalLMOutputWithPast",
]
