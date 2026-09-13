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
"""Paddle Qwen3_VL_Moe model."""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Optional, Union

import paddle

from ...nn.criterion.interface import CriterionLayer
from ..cache_utils import Cache
from ..model_outputs import ModelOutput
from ..model_utils import PretrainedModel
from ..qwen3_vl.modeling_fleet import Qwen3VLModelDist, Qwen3VLProvider
from .configuration import (
    Qwen3VLMoeConfig,
)


class Qwen3VLMoePretrainedModelFleet(PretrainedModel):
    config_class = Qwen3VLMoeConfig
    base_model_prefix = "model"
    input_modalities = ["image", "video", "text"]
    _no_split_modules = ["Qwen3VLMoeTextDecoderLayer", "Qwen3VLMoeVisionBlock"]
    _keys_to_ignore_on_load_unexpected = [r"self_attn.rotary_emb.inv_freq"]
    transpose_weight_keys = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "qkv",
        "proj",
        r"linear_fc\d+",
        "gate",
    ]

    @classmethod
    def _gen_aoa_config(cls, config: Qwen3VLMoeConfig):
        mapping = cls._checkpoint_conversion_mapping
        llm_target = next(
            (v for v in mapping.values() if "language_model" in v),
            "language_model",
        )
        visual_target = "model.vision_model"
        llm_prefix = (
            f"{llm_target}." if not llm_target.endswith(".") else llm_target
        )
        visual_prefix = (
            f"{visual_target}."
            if not visual_target.endswith(".")
            else visual_target
        )

        # language model
        aoa_config = {
            "aoa_statements": [
                f"model.language_model.embed_tokens.weight -> {llm_prefix}embedding.embed_tokens.weight",
                f"model.language_model.norm.weight ->  {llm_prefix}norm.weight",
                f"model.language_model.layers.$LAYER_ID.self_attn.o_proj.weight^T -> {llm_prefix}layers.$LAYER_ID.self_attn.o_proj.weight",
                f"model.language_model.layers.$LAYER_ID.input_layernorm.weight -> {llm_prefix}layers.$LAYER_ID.input_layernorm.weight",
                f"model.language_model.layers.$LAYER_ID.post_attention_layernorm.weight -> {llm_prefix}layers.$LAYER_ID.post_attention_layernorm.weight",
                f"model.language_model.layers.$LAYER_ID.self_attn.q_norm.weight -> {llm_prefix}layers.$LAYER_ID.self_attn.q_norm.weight",
                f"model.language_model.layers.$LAYER_ID.self_attn.k_norm.weight -> {llm_prefix}layers.$LAYER_ID.self_attn.k_norm.weight",
                f"model.language_model.layers.$LAYER_ID.mlp.gate.weight -> {llm_prefix}layers.$LAYER_ID.mlp.gate.weight, dtype='float32'",
            ]
        }
        # language attention qkv
        aoa_config["aoa_statements"] += [
            f"model.language_model.layers.{layer_id}.self_attn.q_proj.weight^T, model.language_model.layers.{layer_id}.self_attn.k_proj.weight^T, model.language_model.layers.{layer_id}.self_attn.v_proj.weight^T -> {llm_prefix}layers.{layer_id}.self_attn.qkv_proj.weight, fused_qkv, num_heads={config.text_config.num_attention_heads}, num_key_value_groups={config.text_config.num_key_value_heads}"
            for layer_id in range(config.text_config.num_hidden_layers)
        ]
        if config.attention_bias:
            aoa_config["aoa_statements"] += [
                f"model.language_model.layers.{layer_id}.self_attn.q_proj.bias, model.language_model.layers.{layer_id}.self_attn.k_proj.bias, model.language_model.layers.{layer_id}.self_attn.v_proj.bias -> {llm_prefix}{layer_id + 1}.self_attn.qkv_proj.bias, fused_qkv, num_heads={config.text_config.num_attention_heads}, num_key_value_groups={config.text_config.num_key_value_heads}"
                for layer_id in range(config.text_config.num_hidden_layers)
            ]

        # language moe experts
        for layer_id in range(config.num_hidden_layers):
            if config.moe_expert_fusion:
                aoa_config["aoa_statements"] += [
                    f"model.language_model.layers.{layer_id}.mlp.experts.gate_up_proj -> {llm_prefix}layers.{layer_id}.mlp.grouped_gemm_experts.weight1",
                    f"model.language_model.layers.{layer_id}.mlp.experts.down_proj -> {llm_prefix}layers.{layer_id}.mlp.grouped_gemm_experts.weight2",
                ]
            else:
                split_experts_up_gate = ""
                split_experts_down = ""
                for expert_id in range(config.text_config.num_experts):
                    split_experts_up_gate += f"{llm_prefix}layers.{layer_id}.mlp.experts.{expert_id}.up_gate_proj.weight,"
                    split_experts_down += f"{llm_prefix}layers.{layer_id}.mlp.experts.{expert_id}.down_proj.weight,"
                split_experts_down += "axis=0"
                split_experts_up_gate += "axis=0"
                aoa_config["aoa_statements"] += [
                    f"model.language_model.layers.{layer_id}.mlp.experts.gate_up_proj -> {split_experts_up_gate}",
                    f"model.language_model.layers.{layer_id}.mlp.experts.down_proj -> {split_experts_down}",
                ]
        # visual model
        # visual attention qkv : qqqq,kkkk,vvvv->qkv,qkv,qkv
        aoa_config["aoa_statements"] += [
            stmt
            for layer_id in range(config.vision_config.depth)
            for stmt in (
                f"model.visual.blocks.{layer_id}.attn.qkv.weight -> model.visual.blocks.{layer_id}.attn.q.weight, model.visual.blocks.{layer_id}.attn.k.weight,model.visual.blocks.{layer_id}.attn.v.weight,axis=0",
                f"model.visual.blocks.{layer_id}.attn.q.weight^T, model.visual.blocks.{layer_id}.attn.k.weight^T, model.visual.blocks.{layer_id}.attn.v.weight^T -> {visual_prefix}decoder.layers.{layer_id}.self_attn.qkv_proj.weight,fused_qkv, num_heads={config.vision_config.num_heads}, num_key_value_groups={config.vision_config.num_heads}",
                f"model.visual.blocks.{layer_id}.attn.qkv.bias -> model.visual.blocks.{layer_id}.attn.q.bias, model.visual.blocks.{layer_id}.attn.k.bias, model.visual.blocks.{layer_id}.attn.v.bias,axis=0",
                f"model.visual.blocks.{layer_id}.attn.q.bias, model.visual.blocks.{layer_id}.attn.k.bias, model.visual.blocks.{layer_id}.attn.v.bias -> {visual_prefix}decoder.layers.{layer_id}.self_attn.qkv_proj.bias, fused_qkv, num_heads={config.vision_config.num_heads}, num_key_value_groups={config.vision_config.num_heads},axis=0",
            )
        ]
        aoa_config["aoa_statements"] += (
            [
                f"model.visual.blocks.$LAYER_ID.attn.proj.weight^T -> {visual_prefix}decoder.layers.$LAYER_ID.self_attn.o_proj.weight"
            ]
            + [
                f"model.visual.blocks.$LAYER_ID.attn.proj.bias -> {visual_prefix}decoder.layers.$LAYER_ID.self_attn.o_proj.bias"
            ]
            + [
                f"model.visual.blocks.$LAYER_ID.mlp.{x}.weight^T -> {visual_prefix}decoder.layers.$LAYER_ID.mlp.{y}.weight"
                for x, y in (
                    ("linear_fc1", "up_gate_proj"),
                    ("linear_fc2", "down_proj"),
                )
            ]
            + [
                f"model.visual.blocks.$LAYER_ID.mlp.{x}.bias -> {visual_prefix}decoder.layers.$LAYER_ID.mlp.{y}.bias"
                for x, y in (
                    ("linear_fc1", "up_gate_proj"),
                    ("linear_fc2", "down_proj"),
                )
            ]
        )
        aoa_config["aoa_statements"] += [
            f"model.visual.patch_embed.proj.weight -> {visual_prefix}patch_embed.proj.weight",
            f"model.visual.patch_embed.proj.bias -> {visual_prefix}patch_embed.proj.bias",
            f"model.visual.pos_embed.weight -> {visual_prefix}pos_embed.weight",
            f"model.visual.merger.norm.weight -> {visual_prefix}decoder.merger.norm.weight",
            f"model.visual.merger.norm.bias -> {visual_prefix}decoder.merger.norm.bias",
            f"model.visual.blocks.$LAYER_ID.norm1.weight -> {visual_prefix}decoder.layers.$LAYER_ID.input_layernorm.weight",
            f"model.visual.blocks.$LAYER_ID.norm1.bias -> {visual_prefix}decoder.layers.$LAYER_ID.input_layernorm.bias",
            f"model.visual.blocks.$LAYER_ID.norm2.weight -> {visual_prefix}decoder.layers.$LAYER_ID.post_attention_layernorm.weight",
            f"model.visual.blocks.$LAYER_ID.norm2.bias -> {visual_prefix}decoder.layers.$LAYER_ID.post_attention_layernorm.bias",
        ]
        aoa_config["aoa_statements"] += [
            f"model.visual.merger.linear_fc1.weight^T -> {visual_prefix}decoder.merger.linear_fc1.weight",
            f"model.visual.merger.linear_fc1.bias -> {visual_prefix}decoder.merger.linear_fc1.bias",
            f"model.visual.merger.linear_fc2.weight^T -> {visual_prefix}decoder.merger.linear_fc2.weight",
            f"model.visual.merger.linear_fc2.bias -> {visual_prefix}decoder.merger.linear_fc2.bias",
        ]
        aoa_config["aoa_statements"] += [
            f"model.visual.deepstack_merger_list.$LAYER_ID.linear_fc1.weight^T -> {visual_prefix}decoder.deepstack_merger_list.$LAYER_ID.linear_fc1.weight",
            f"model.visual.deepstack_merger_list.$LAYER_ID.linear_fc1.bias -> {visual_prefix}decoder.deepstack_merger_list.$LAYER_ID.linear_fc1.bias",
            f"model.visual.deepstack_merger_list.$LAYER_ID.linear_fc2.weight^T -> {visual_prefix}decoder.deepstack_merger_list.$LAYER_ID.linear_fc2.weight",
            f"model.visual.deepstack_merger_list.$LAYER_ID.linear_fc2.bias -> {visual_prefix}decoder.deepstack_merger_list.$LAYER_ID.linear_fc2.bias",
            f"model.visual.deepstack_merger_list.$LAYER_ID.norm.weight -> {visual_prefix}decoder.deepstack_merger_list.$LAYER_ID.norm.weight",
            f"model.visual.deepstack_merger_list.$LAYER_ID.norm.bias -> {visual_prefix}decoder.deepstack_merger_list.$LAYER_ID.norm.bias",
        ]

        # Qwen3_VLModel without lm_head
        if cls._tied_weights_keys:
            aoa_config["aoa_statements"] += [
                f"{'model.language_model.embed_tokens.weight' if config.tie_word_embeddings else 'lm_head.weight'} -> {llm_prefix}lm_head.weight",
            ]

        return aoa_config

    @classmethod
    def _gen_inv_aoa_config(cls, config: Qwen3VLMoeConfig):
        mapping = cls._checkpoint_conversion_mapping
        llm_target = next(
            (v for v in mapping.values() if "language_model" in v),
            "language_model",
        )
        # visual_target = next((v for v in mapping.values() if "visual" in v), "visual")
        visual_target = "model.vision_model"
        llm_prefix = (
            f"{llm_target}." if not llm_target.endswith(".") else llm_target
        )
        visual_prefix = (
            f"{visual_target}."
            if not visual_target.endswith(".")
            else visual_target
        )
        # language model
        aoa_config = {
            "aoa_statements": [
                f"{llm_prefix}embedding.embed_tokens.weight -> model.language_model.embed_tokens.weight",
                f"{llm_prefix}norm.weight -> model.language_model.norm.weight",
                f"{llm_prefix}layers.$LAYER_ID.input_layernorm.weight -> model.language_model.layers.$LAYER_ID.input_layernorm.weight",
                f"{llm_prefix}layers.$LAYER_ID.post_attention_layernorm.weight -> model.language_model.layers.$LAYER_ID.post_attention_layernorm.weight",
                f"{llm_prefix}layers.$LAYER_ID.self_attn.o_proj.weight^T -> model.language_model.layers.$LAYER_ID.self_attn.o_proj.weight",
                f"{llm_prefix}layers.$LAYER_ID.mlp.gate.weight -> model.language_model.layers.$LAYER_ID.mlp.gate.weight",
                f"{llm_prefix}layers.$LAYER_ID.self_attn.q_norm.weight -> model.language_model.layers.$LAYER_ID.self_attn.q_norm.weight",
                f"{llm_prefix}layers.$LAYER_ID.self_attn.k_norm.weight -> model.language_model.layers.$LAYER_ID.self_attn.k_norm.weight",
                f"{llm_prefix}layers.$LAYER_ID.mlp.grouped_gemm_experts.weight1 -> model.language_model.layers.$LAYER_ID.mlp.experts.gate_up_proj",
                f"{llm_prefix}layers.$LAYER_ID.mlp.grouped_gemm_experts.weight2 -> model.language_model.layers.$LAYER_ID.mlp.experts.down_proj",
            ]
        }

        # visual model
        aoa_config["aoa_statements"] += [
            stmt
            for layer_id in range(config.vision_config.depth)
            for stmt in (
                f"{visual_prefix}decoder.layers.{layer_id}.self_attn.qkv_proj.weight -> model.visual.blocks.{layer_id}.attn.q.weight, model.visual.blocks.{layer_id}.attn.k.weight, model.visual.blocks.{layer_id}.attn.v.weight, fused_qkv, num_heads={config.vision_config.num_heads}, num_key_value_groups={config.vision_config.num_heads}",
                f"model.visual.blocks.{layer_id}.attn.q.weight^T, model.visual.blocks.{layer_id}.attn.k.weight^T, model.visual.blocks.{layer_id}.attn.v.weight^T -> model.visual.blocks.{layer_id}.attn.qkv.weight, axis=0",
                f"{visual_prefix}decoder.layers.{layer_id}.self_attn.qkv_proj.bias -> model.visual.blocks.{layer_id}.attn.q.bias, model.visual.blocks.{layer_id}.attn.k.bias, model.visual.blocks.{layer_id}.attn.v.bias, fused_qkv, num_heads={config.vision_config.num_heads}, num_key_value_groups={config.vision_config.num_heads},axis=0",
                f"model.visual.blocks.{layer_id}.attn.q.bias, model.visual.blocks.{layer_id}.attn.k.bias, model.visual.blocks.{layer_id}.attn.v.bias -> model.visual.blocks.{layer_id}.attn.qkv.bias, axis=0",
            )
        ]
        aoa_config["aoa_statements"] += (
            [
                f"{visual_prefix}decoder.layers.$LAYER_ID.self_attn.o_proj.weight^T -> model.visual.blocks.$LAYER_ID.attn.proj.weight"
            ]
            + [
                f"{visual_prefix}decoder.layers.$LAYER_ID.self_attn.o_proj.bias -> model.visual.blocks.$LAYER_ID.attn.proj.bias"
            ]
            + [
                f"{visual_prefix}decoder.layers.$LAYER_ID.mlp.{y}.weight^T -> model.visual.blocks.$LAYER_ID.mlp.{x}.weight"
                for x, y in (
                    ("linear_fc1", "up_gate_proj"),
                    ("linear_fc2", "down_proj"),
                )
            ]
            + [
                f"{visual_prefix}decoder.layers.$LAYER_ID.mlp.{y}.bias -> model.visual.blocks.$LAYER_ID.mlp.{x}.bias"
                for x, y in (
                    ("linear_fc1", "up_gate_proj"),
                    ("linear_fc2", "down_proj"),
                )
            ]
        )
        aoa_config["aoa_statements"] += [
            f"{visual_prefix}patch_embed.proj.weight -> model.visual.patch_embed.proj.weight",
            f"{visual_prefix}patch_embed.proj.bias -> model.visual.patch_embed.proj.bias",
            f"{visual_prefix}pos_embed.weight -> model.visual.pos_embed.weight",
            f"{visual_prefix}decoder.merger.norm.weight -> model.visual.merger.norm.weight",
            f"{visual_prefix}decoder.merger.norm.bias -> model.visual.merger.norm.bias",
            f"{visual_prefix}decoder.layers.$LAYER_ID.input_layernorm.weight -> model.visual.blocks.$LAYER_ID.norm1.weight",
            f"{visual_prefix}decoder.layers.$LAYER_ID.input_layernorm.bias -> model.visual.blocks.$LAYER_ID.norm1.bias",
            f"{visual_prefix}decoder.layers.$LAYER_ID.post_attention_layernorm.weight -> model.visual.blocks.$LAYER_ID.norm2.weight",
            f"{visual_prefix}decoder.layers.$LAYER_ID.post_attention_layernorm.bias -> model.visual.blocks.$LAYER_ID.norm2.bias",
        ]
        aoa_config["aoa_statements"] += [
            f"{visual_prefix}decoder.merger.linear_fc1.weight^T -> model.visual.merger.linear_fc1.weight",
            f"{visual_prefix}decoder.merger.linear_fc1.bias -> model.visual.merger.linear_fc1.bias",
            f"{visual_prefix}decoder.merger.linear_fc2.weight^T -> model.visual.merger.linear_fc2.weight",
            f"{visual_prefix}decoder.merger.linear_fc2.bias -> model.visual.merger.linear_fc2.bias",
        ]
        aoa_config["aoa_statements"] += [
            f"{visual_prefix}decoder.deepstack_merger_list.$LAYER_ID.linear_fc1.weight^T -> model.visual.deepstack_merger_list.$LAYER_ID.linear_fc1.weight",
            f"{visual_prefix}decoder.deepstack_merger_list.$LAYER_ID.linear_fc1.bias -> model.visual.deepstack_merger_list.$LAYER_ID.linear_fc1.bias",
            f"{visual_prefix}decoder.deepstack_merger_list.$LAYER_ID.linear_fc2.weight^T -> model.visual.deepstack_merger_list.$LAYER_ID.linear_fc2.weight",
            f"{visual_prefix}decoder.deepstack_merger_list.$LAYER_ID.linear_fc2.bias -> model.visual.deepstack_merger_list.$LAYER_ID.linear_fc2.bias",
            f"{visual_prefix}decoder.deepstack_merger_list.$LAYER_ID.norm.weight -> model.visual.deepstack_merger_list.$LAYER_ID.norm.weight",
            f"{visual_prefix}decoder.deepstack_merger_list.$LAYER_ID.norm.bias -> model.visual.deepstack_merger_list.$LAYER_ID.norm.bias",
        ]

        # attention qkv
        aoa_config["aoa_statements"] += [
            f"{llm_prefix}layers.{layer_id}.self_attn.qkv_proj.weight  -> model.language_model.layers.{layer_id}.self_attn.q_proj.weight, model.language_model.layers.{layer_id}.self_attn.k_proj.weight, model.language_model.layers.{layer_id}.self_attn.v_proj.weight, fused_qkv, num_heads={config.text_config.num_attention_heads}, num_key_value_groups = {config.text_config.num_key_value_heads}"
            for layer_id in range(config.text_config.num_hidden_layers)
        ]
        if config.attention_bias:
            aoa_config["aoa_statements"] += [
                f"{llm_prefix}{layer_id + 1}.self_attn.qkv_proj.bias  -> model.language_model.layers.{layer_id}.self_attn.q_proj.bias, model.language_model.layers.{layer_id}.self_attn.k_proj.bias, model.language_model.layers.{layer_id}.self_attn.v_proj.bias, fused_qkv, num_heads={config.text_config.num_attention_heads}, num_key_value_groups = {config.text_config.num_key_value_heads}"
                for layer_id in range(config.text_config.num_hidden_layers)
            ]

        aoa_config["aoa_statements"] += [
            f"{llm_prefix}layers.{layer_id}.self_attn.{x}_proj.weight^T -> model.language_model.layers.{layer_id}.self_attn.{x}_proj.weight"
            for layer_id in range(config.text_config.num_hidden_layers)
            for x in ("q", "k", "v")
        ]
        # Qwen3VLMoeModel without lm_head
        if cls._tied_weights_keys:
            aoa_config["aoa_statements"] += [
                f"{llm_prefix}lm_head.weight -> {'_' if config.tie_word_embeddings else 'lm_head.weight'}",
            ]

        return aoa_config


@dataclass
class Qwen3VLMoeCausalLMOutputWithPast(ModelOutput):
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
    aux_loss: Optional[paddle.Tensor] = None


class Qwen3VLMoeModel(Qwen3VLMoePretrainedModelFleet):
    is_fleet = True

    def __new__(cls, config, have_criterion=True):
        config.tensor_model_parallel_size = max(
            config.tensor_model_parallel_size, 1
        )
        config.context_parallel_size = max(config.context_parallel_size, 1)
        config.pipeline_model_parallel_size = max(
            config.pipeline_model_parallel_size, 1
        )
        config.virtual_pipeline_model_parallel_size = max(
            config.virtual_pipeline_model_parallel_size, 1
        )
        config.expert_model_parallel_size = max(
            config.expert_model_parallel_size, 1
        )
        config.moe_expert_fusion = True
        criterion = None
        if have_criterion:
            criterion = CriterionLayer(config.text_config)
        model_provider_class = Qwen3VLProvider
        model_provider = model_provider_class.from_config(config)
        qwen3vl_model = Qwen3VLModelDist(
            model_provider, model_version=config.model_type, criterion=criterion
        )
        qwen3vl_model._gen_aoa_config = cls._gen_aoa_config
        qwen3vl_model._gen_inv_aoa_config = cls._gen_inv_aoa_config
        qwen3vl_model._get_tensor_parallel_mappings = (
            cls._get_tensor_parallel_mappings
        )
        qwen3vl_model.config_to_save = config
        qwen3vl_model.get_hardware_flops = types.MethodType(
            cls.get_hardware_flops, qwen3vl_model
        )

        return qwen3vl_model


class Qwen3VLMoeForConditionalGeneration(Qwen3VLMoePretrainedModelFleet):
    _checkpoint_conversion_mapping = {
        "^visual": "model.visual",
        r"^model(?!\.(language_model|visual))": "model.language_model",
    }
    _tied_weights_keys = {
        "lm_head.weight": "model.language_model.embed_tokens.weight"
    }
    config_class = Qwen3VLMoeConfig

    def __init__(self, config):
        super().__init__(config)
        # model_provider = Qwen3VLProvider.from_config(config)
        self.model = Qwen3VLMoeModel(config, have_criterion=False)
        self.criterion = CriterionLayer(config.text_config)
        # self.tie_weights()

    def state_dict(self, *args, **kwargs):
        # Override state_dict method to handle language_model's custom state_dict
        state_dict = super().state_dict(*args, **kwargs)
        # Remove existing language_model keys to avoid duplicates
        delete_key = []
        for key in state_dict.keys():
            if key.startswith("model.language_model."):
                delete_key.append(key)
        for key in delete_key:
            state_dict.pop(key)
        if self.model.language_model is not None:
            # Get language_model's state_dict
            language_state_dict = self.model.language_model.state_dict(
                *args, **kwargs
            )

            # Merge language_model parameters into main state_dict
            for key, value in language_state_dict.items():
                state_dict[key] = value
        return state_dict

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        pixel_values: Optional[paddle.Tensor] = None,
        pixel_values_videos: Optional[paddle.Tensor] = None,
        image_grid_thw: Optional[paddle.Tensor] = None,
        video_grid_thw: Optional[paddle.Tensor] = None,
        rope_deltas: Optional[paddle.Tensor] = None,
        cache_position: Optional[paddle.Tensor] = None,
        logits_to_keep: Union[int, paddle.Tensor] = 0,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> Union[tuple, Qwen3VLMoeCausalLMOutputWithPast]:
        r"""
        labels (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`paddle.Tensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`paddle.Tensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        rope_deltas (`paddle.Tensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.

        Example:

        ```python
        >>> from paddlefleet.transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        >>> model = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example1.jpg",
                    },
                    {"type": "text", "text": "Describe the image."},
                ],
            }
        ]

        >>> inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pd"
        )

        >>> # Generate
        >>> generated_ids = model.generate(**inputs, max_new_tokens=1024)
        >>> output_text = processor.batch_decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        >>> print(output_text)
        ```
        """

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            **kwargs,
        )

        logits = outputs

        loss = None
        if labels is not None:
            loss, _ = self.criterion(logits, labels)

        return Qwen3VLMoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=None,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            rope_deltas=None,
        )


__all__ = [
    "Qwen3VLMoeForConditionalGeneration",
    "Qwen3VLMoeModel",
    "Qwen3VLMoePretrainedModelFleet",
    "Qwen3VLMoeCausalLMOutputWithPast",
]
