# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
"""Paddle Qwen2 model."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Union

from paddlefleet.transformers.gpt_provider import GPTModelProvider

from ...nn.pp_model import CriterionLayerPipe, GeneralModelForCausalLMPipe
from ...utils.log import logger
from ..model_utils import PretrainedModel
from .configuration import Qwen2Config


@dataclass
class Qwen2ModelProvider(GPTModelProvider):
    """Base provider for Qwen2 Models."""

    model_type = "qwen2"

    attention_bias: bool = True

    bias_activation_fusion: bool = True
    bias_dropout_fusion: bool = True

    transform_rules = {
        "dtype": "params_dtype",
    }

    persist_layer_norm: bool = True
    share_embeddings_and_output_weights: bool = False

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        """
        Save a configuration object to the directory `save_directory`, so that it can be re-loaded using the
        [`~PretrainedConfig.from_pretrained`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the configuration JSON file will be saved (will be created if it does not exist).
            kwargs:
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        output_config_file = os.path.join(save_directory, self.CONFIG_NAME)
        config_dict = asdict(self)

        # Filter out non-serializable values
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items() if make_serializable(v) is not None}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(item) for item in obj if make_serializable(item) is not None]
            elif isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            else:
                # Skip non-serializable types like partial, function, etc.
                return None

        serializable_config = make_serializable(config_dict)

        with open(output_config_file, "w", encoding="utf-8") as writer:
            writer.write(json.dumps(serializable_config, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
        logger.info(f"Configuration saved in {output_config_file}")


class Qwen2PretrainedModel(PretrainedModel):
    config_class = Qwen2Config
    base_model_prefix = "model"
    _keys_to_ignore_on_load_unexpected = [r"self_attn.rotary_emb.inv_freq"]
    transpose_weight_keys = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    @classmethod
    def _gen_aoa_config(cls, config: Qwen2Config):
        model_prefix = "" if cls == getattr(cls, "base_model_class", None) else "model."
        is_fleet = getattr(cls, "is_fleet", False)

        aoa_config = {
            "aoa_statements": [
                f"model.layers.$LAYER_ID.self_attn.o_proj.weight^T -> {model_prefix}layers.$LAYER_ID.self_attn.o_proj.weight",
                f"model.layers.$LAYER_ID.mlp.down_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.down_proj.weight",
                f"model.layers.$LAYER_ID.input_layernorm.weight -> {model_prefix}layers.$LAYER_ID.input_layernorm.weight",
                f"model.layers.$LAYER_ID.post_attention_layernorm.weight -> {model_prefix}layers.$LAYER_ID.post_attention_layernorm.weight",
                f"model.norm.weight -> {model_prefix}norm.weight",
            ]
        }

        if is_fleet:
            aoa_config["aoa_statements"] += [
                f"model.embed_tokens.weight -> {model_prefix}embedding.embed_tokens.weight",
            ]
        else:
            aoa_config["aoa_statements"] += [
                f"model.embed_tokens.weight -> {model_prefix}embed_tokens.weight",
            ]

        # attention qkv
        aoa_config["aoa_statements"] += [
            f"model.layers.$LAYER_ID.self_attn.q_proj.weight^T, model.layers.$LAYER_ID.self_attn.k_proj.weight^T, model.layers.$LAYER_ID.self_attn.v_proj.weight^T -> {model_prefix}layers.$LAYER_ID.self_attn.qkv_proj.weight, fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups={config.num_key_value_heads}",
        ]
        aoa_config["aoa_statements"] += [
            f"model.layers.$LAYER_ID.self_attn.q_proj.bias, model.layers.$LAYER_ID.self_attn.k_proj.bias, model.layers.$LAYER_ID.self_attn.v_proj.bias -> {model_prefix}layers.$LAYER_ID.self_attn.qkv_proj.bias, fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups={config.num_key_value_heads}, axis=0",
        ]

        # FFN
        aoa_config["aoa_statements"] += [
            f"model.layers.$LAYER_ID.mlp.gate_proj.weight^T, model.layers.$LAYER_ID.mlp.up_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.up_gate_proj.weight, fused_ffn",
        ]

        # lm_head
        if config.tie_word_embeddings:
            if is_fleet:
                aoa_config["aoa_statements"] += [f"model.embed_tokens.weight -> {model_prefix}lm_head.weight"]
            else:
                aoa_config["aoa_statements"] += ["model.embed_tokens.weight -> lm_head.weight"]
        else:
            if is_fleet:
                aoa_config["aoa_statements"] += [f"lm_head.weight -> {model_prefix}lm_head.weight"]

        return aoa_config

    @classmethod
    def _gen_inv_aoa_config(cls, config: Qwen2Config):
        model_prefix = "" if cls == getattr(cls, "base_model_class", None) else "model."
        is_fleet = getattr(cls, "is_fleet", False)

        aoa_statements = [
            f"{model_prefix}layers.$LAYER_ID.self_attn.o_proj.weight^T -> model.layers.$LAYER_ID.self_attn.o_proj.weight",
            f"{model_prefix}layers.$LAYER_ID.mlp.down_proj.weight^T -> model.layers.$LAYER_ID.mlp.down_proj.weight",
            f"{model_prefix}layers.$LAYER_ID.input_layernorm.weight -> model.layers.$LAYER_ID.input_layernorm.weight",
            f"{model_prefix}layers.$LAYER_ID.post_attention_layernorm.weight -> model.layers.$LAYER_ID.post_attention_layernorm.weight",
            f"{model_prefix}norm.weight -> model.norm.weight",
        ]

        if is_fleet:
            aoa_statements += [
                f"{model_prefix}embedding.embed_tokens.weight -> model.embed_tokens.weight",
            ]
        else:
            aoa_statements += [
                f"{model_prefix}embed_tokens.weight -> model.embed_tokens.weight",
            ]

        aoa_statements += [
            f"{model_prefix}layers.$LAYER_ID.self_attn.qkv_proj.weight -> model.layers.$LAYER_ID.self_attn.q_proj.weight, model.layers.$LAYER_ID.self_attn.k_proj.weight, model.layers.$LAYER_ID.self_attn.v_proj.weight , fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups = {config.num_key_value_heads}",
        ]
        for layer_id in range(config.num_hidden_layers):
            for x in ("q", "k", "v"):
                aoa_statements += [
                    f"model.layers.{layer_id}.self_attn.{x}_proj.weight^T -> model.layers.{layer_id}.self_attn.{x}_proj.weight"
                ]
        aoa_statements += [
            f"{model_prefix}layers.$LAYER_ID.self_attn.qkv_proj.bias -> model.layers.$LAYER_ID.self_attn.q_proj.bias, model.layers.$LAYER_ID.self_attn.k_proj.bias, model.layers.$LAYER_ID.self_attn.v_proj.bias, fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups={config.num_key_value_heads}, axis=0",
        ]

        aoa_statements += [
            f"{model_prefix}layers.$LAYER_ID.mlp.up_gate_proj.weight -> model.layers.$LAYER_ID.mlp.gate_proj.weight, model.layers.$LAYER_ID.mlp.up_proj.weight, fused_ffn",
        ]
        for layer_id in range(config.num_hidden_layers):
            aoa_statements += [
                f"model.layers.{layer_id}.mlp.gate_proj.weight^T -> model.layers.{layer_id}.mlp.gate_proj.weight",
                f"model.layers.{layer_id}.mlp.up_proj.weight^T -> model.layers.{layer_id}.mlp.up_proj.weight",
            ]

        if config.tie_word_embeddings:
            if is_fleet:
                aoa_statements += [f"{model_prefix}lm_head.weight -> _"]
            else:
                aoa_statements += ["lm_head.weight -> _"]
        else:
            if is_fleet:
                aoa_statements += [f"{model_prefix}lm_head.weight -> lm_head.weight"]

        aoa_config = {"aoa_statements": aoa_statements}
        return aoa_config


class Qwen2ForCausalLM(Qwen2PretrainedModel):
    is_fleet = True

    def __new__(cls, config):
        # Hybrid parallel config convert.
        config.tensor_model_parallel_size = max(config.tensor_model_parallel_size, 1)
        config.context_parallel_size = max(config.context_parallel_size, 1)
        config.pipeline_model_parallel_size = max(config.pipeline_model_parallel_size, 1)
        config.virtual_pipeline_model_parallel_size = max(config.virtual_pipeline_model_parallel_size, 1)
        config.expert_model_parallel_size = max(config.expert_model_parallel_size, 1)

        model_provider_class = Qwen2ModelProvider
        model_provider = model_provider_class.from_config(config)
        loss_fn = None
        if getattr(config, "dpo_config", None):
            loss_fn = CriterionLayerPipe(config, use_infohub=True)
        gpt_model = model_provider.provide(loss_fn=loss_fn)
        gpt_model._gen_aoa_config = cls._gen_aoa_config
        gpt_model._gen_inv_aoa_config = cls._gen_inv_aoa_config
        gpt_model.config_to_save = config
        gpt_model.is_fleet = cls.is_fleet
        return gpt_model


class Qwen2ForCausalLMPipe(Qwen2PretrainedModel, GeneralModelForCausalLMPipe):
    is_fleet = True

    def __new__(cls, config):
        # Hybrid parallel config convert.
        config.tensor_model_parallel_size = max(config.tensor_model_parallel_size, 1)
        config.context_parallel_size = max(config.context_parallel_size, 1)
        config.pipeline_model_parallel_size = max(config.pipeline_model_parallel_size, 1)
        config.virtual_pipeline_model_parallel_size = max(config.virtual_pipeline_model_parallel_size, 1)
        config.expert_model_parallel_size = max(config.expert_model_parallel_size, 1)

        model_provider_class = Qwen2ModelProvider
        model_provider = model_provider_class.from_config(config)
        loss_fn = None
        if getattr(config, "dpo_config", None):
            loss_fn = CriterionLayerPipe(config, use_infohub=True)
        gpt_model = model_provider.provide(loss_fn=loss_fn)
        gpt_model._gen_aoa_config = cls._gen_aoa_config
        gpt_model._gen_inv_aoa_config = cls._gen_inv_aoa_config
        if not hasattr(config, "architectures"):
            config.architectures = [cls.__name__.replace("Pipe", "")]
        gpt_model.config_to_save = config
        gpt_model.is_fleet = cls.is_fleet
        return gpt_model


__all__ = [
    "Qwen2PretrainedModel",
    "Qwen2ForCausalLM",
    "Qwen2ForCausalLMPipe",
    "Qwen2ModelProvider",
]
