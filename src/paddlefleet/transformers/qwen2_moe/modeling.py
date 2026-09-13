# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
"""Paddle Qwen2Moe model."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Union

from paddlefleet.transformers.gpt_provider import GPTModelProvider

from ...nn.pp_model import CriterionLayerPipe, GeneralModelForCausalLMPipe
from ...utils.log import logger
from ..model_utils import PretrainedModel
from .configuration import Qwen2MoeConfig


@dataclass
class Qwen2MoeModelProvider(GPTModelProvider):
    """Base provider for Qwen2Moe Models."""

    model_type = "qwen2_moe"

    moe_shared_expert_gate: bool = True

    attention_bias: bool = True

    bias_activation_fusion: bool = True
    bias_dropout_fusion: bool = True

    transform_rules = {
        "dtype": "params_dtype",
        "num_experts": "n_routed_experts",
    }

    persist_layer_norm: bool = True
    share_embeddings_and_output_weights: bool = False

    def save_pretrained(
        self, save_directory: Union[str, os.PathLike], **kwargs
    ):
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
            raise AssertionError(
                f"Provided path ({save_directory}) should be a directory, not a file"
            )

        os.makedirs(save_directory, exist_ok=True)

        output_config_file = os.path.join(save_directory, self.CONFIG_NAME)
        config_dict = asdict(self)

        # Filter out non-serializable values
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {
                    k: make_serializable(v)
                    for k, v in obj.items()
                    if make_serializable(v) is not None
                }
            elif isinstance(obj, (list, tuple)):
                return [
                    make_serializable(item)
                    for item in obj
                    if make_serializable(item) is not None
                ]
            elif isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            else:
                # Skip non-serializable types like partial, function, etc.
                return None

        serializable_config = make_serializable(config_dict)

        with open(output_config_file, "w", encoding="utf-8") as writer:
            writer.write(
                json.dumps(
                    serializable_config,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n"
            )
        logger.info(f"Configuration saved in {output_config_file}")


class Qwen2MoePretrainedModel(PretrainedModel):
    config_class = Qwen2MoeConfig
    base_model_prefix = "model"
    _keys_to_ignore_on_load_unexpected = [r"self_attn.rotary_emb.inv_freq"]
    transpose_weight_keys = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "gate",
        "shared_expert_gate",
    ]

    @classmethod
    def _gen_aoa_config(cls, config: Qwen2MoeConfig):
        if hasattr(config, "n_routed_experts"):
            num_experts = config.n_routed_experts
        else:
            num_experts = config.num_experts
        model_prefix = (
            "" if cls == getattr(cls, "base_model_class", None) else "model."
        )
        is_fleet = getattr(cls, "is_fleet", False)
        aoa_config = {
            "aoa_statements": [
                f"model.layers.$LAYER_ID.self_attn.o_proj.weight^T -> {model_prefix}layers.$LAYER_ID.self_attn.o_proj.weight",
                f"model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.down_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.experts.$EXPERT_ID.down_proj.weight",
                f"model.layers.$LAYER_ID.input_layernorm.weight -> {model_prefix}layers.$LAYER_ID.input_layernorm.weight",
                f"model.layers.$LAYER_ID.post_attention_layernorm.weight -> {model_prefix}layers.$LAYER_ID.post_attention_layernorm.weight",
                f"model.norm.weight -> {model_prefix}norm.weight",
            ]
        }

        if is_fleet:
            aoa_config["aoa_statements"] += [
                f"model.layers.$LAYER_ID.mlp.gate.weight -> {model_prefix}layers.$LAYER_ID.mlp.gate.weight, dtype='float32'",
                f"model.embed_tokens.weight -> {model_prefix}embedding.embed_tokens.weight",
                f"model.layers.$LAYER_ID.mlp.shared_expert.down_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.shared_experts.down_proj.weight",
                f"model.layers.$LAYER_ID.mlp.shared_expert_gate.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.shared_experts.gate_weight, dtype='float32'",
            ]
        else:
            aoa_config["aoa_statements"] += [
                f"model.layers.$LAYER_ID.mlp.gate.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.gate.weight, dtype='float32'",
                f"model.embed_tokens.weight -> {model_prefix}embed_tokens.weight",
                f"model.layers.$LAYER_ID.mlp.shared_expert.down_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.shared_expert.down_proj.weight",
                f"model.layers.$LAYER_ID.mlp.shared_expert_gate.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.shared_expert_gate.weight, dtype='float32'",
            ]

        # attention qkv
        aoa_config["aoa_statements"] += [
            f"model.layers.$LAYER_ID.self_attn.q_proj.weight^T, model.layers.$LAYER_ID.self_attn.k_proj.weight^T, model.layers.$LAYER_ID.self_attn.v_proj.weight^T -> {model_prefix}layers.$LAYER_ID.self_attn.qkv_proj.weight, fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups={config.num_key_value_heads}",
        ]
        if config.qkv_bias:
            aoa_config["aoa_statements"] += [
                f"model.layers.$LAYER_ID.self_attn.q_proj.bias, model.layers.$LAYER_ID.self_attn.k_proj.bias, model.layers.$LAYER_ID.self_attn.v_proj.bias -> {model_prefix}layers.$LAYER_ID.self_attn.qkv_proj.bias, fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups={config.num_key_value_heads}, axis=0",
            ]

        # FFN
        if is_fleet:
            aoa_config["aoa_statements"] += [
                f"model.layers.$LAYER_ID.mlp.shared_expert.gate_proj.weight^T, model.layers.$LAYER_ID.mlp.shared_expert.up_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.shared_experts.up_gate_proj.weight, fused_ffn",
                f"model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.gate_proj.weight^T, model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.up_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.experts.$EXPERT_ID.up_gate_proj.weight, axis=1",
            ]
        else:
            aoa_config["aoa_statements"] += [
                f"model.layers.$LAYER_ID.mlp.shared_expert.gate_proj.weight^T, model.layers.$LAYER_ID.mlp.shared_expert.up_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.shared_expert.up_gate_proj.weight, fused_ffn",
                f"model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.gate_proj.weight^T, model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.up_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.experts.$EXPERT_ID.up_gate_proj.weight, fused_ffn",
            ]

        if config.get("fd_fallback", False):
            for layer_idx in range(0, config.num_hidden_layers):
                src_prefix = f"model.layers.{layer_idx}"
                tgt_prefix = f"{model_prefix}layers.{layer_idx}"
                ep_weight1 = []
                ep_weight2 = []
                for expert_id in range(num_experts):
                    ep_weight1.append(
                        f"{src_prefix}.mlp.experts.{expert_id}.up_gate_proj.weight"
                    )
                    ep_weight2.append(
                        f"{src_prefix}.mlp.experts.{expert_id}.down_proj.weight"
                    )
                group1 = ",".join(ep_weight1)
                group2 = ",".join(ep_weight2)
                aoa_config["aoa_statements"] += [
                    f"{group1} -> {tgt_prefix}.mlp.experts.gate_up_proj, axis=0"
                    f"{group2} -> {tgt_prefix}.mlp.experts.down_proj, axis=0"
                ]

        # lm_head
        if config.tie_word_embeddings:
            if is_fleet:
                aoa_config["aoa_statements"] += [
                    f"model.embed_tokens.weight -> {model_prefix}lm_head.weight"
                ]
            else:
                aoa_config["aoa_statements"] += [
                    "model.embed_tokens.weight -> lm_head.weight"
                ]
        else:
            if is_fleet:
                aoa_config["aoa_statements"] += [
                    f"lm_head.weight -> {model_prefix}lm_head.weight"
                ]

        return aoa_config

    @classmethod
    def _gen_inv_aoa_config(cls, config: Qwen2MoeConfig):
        if hasattr(config, "n_routed_experts"):
            num_experts = config.n_routed_experts
        else:
            num_experts = config.num_experts
        model_prefix = (
            "" if cls == getattr(cls, "base_model_class", None) else "model."
        )
        is_fleet = getattr(cls, "is_fleet", False)
        aoa_statements = [
            f"{model_prefix}layers.$LAYER_ID.self_attn.o_proj.weight^T -> model.layers.$LAYER_ID.self_attn.o_proj.weight",
            f"{model_prefix}layers.$LAYER_ID.mlp.experts.$EXPERT_ID.down_proj.weight^T -> model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.down_proj.weight",
            f"{model_prefix}layers.$LAYER_ID.input_layernorm.weight -> model.layers.$LAYER_ID.input_layernorm.weight",
            f"{model_prefix}layers.$LAYER_ID.post_attention_layernorm.weight -> model.layers.$LAYER_ID.post_attention_layernorm.weight",
            f"{model_prefix}norm.weight -> model.norm.weight",
        ]

        if is_fleet:
            aoa_statements += [
                f"{model_prefix}layers.$LAYER_ID.mlp.gate.weight -> model.layers.$LAYER_ID.mlp.gate.weight, dtype='bfloat16'",
                f"{model_prefix}embedding.embed_tokens.weight -> model.embed_tokens.weight",
                f"{model_prefix}layers.$LAYER_ID.mlp.shared_experts.down_proj.weight^T -> model.layers.$LAYER_ID.mlp.shared_expert.down_proj.weight",
                f"{model_prefix}layers.$LAYER_ID.mlp.shared_experts.gate_weight^T -> model.layers.$LAYER_ID.mlp.shared_expert_gate.weight, dtype='bfloat16'",
            ]
        else:
            aoa_statements += [
                f"{model_prefix}layers.$LAYER_ID.mlp.gate.weight^T -> model.layers.$LAYER_ID.mlp.gate.weight, dtype='bfloat16'",
                f"{model_prefix}embed_tokens.weight -> model.embed_tokens.weight",
                f"{model_prefix}layers.$LAYER_ID.mlp.shared_expert.down_proj.weight^T -> model.layers.$LAYER_ID.mlp.shared_expert.down_proj.weight",
                f"{model_prefix}layers.$LAYER_ID.mlp.shared_expert_gate.weight^T -> model.layers.$LAYER_ID.mlp.shared_expert_gate.weight, dtype='bfloat16'",
            ]

        aoa_statements += [
            f"{model_prefix}layers.$LAYER_ID.self_attn.qkv_proj.weight -> model.layers.$LAYER_ID.self_attn.q_proj.weight, model.layers.$LAYER_ID.self_attn.k_proj.weight, model.layers.$LAYER_ID.self_attn.v_proj.weight , fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups = {config.num_key_value_heads}",
        ]
        for layer_id in range(config.num_hidden_layers):
            for x in ("q", "k", "v"):
                aoa_statements += [
                    f"model.layers.{layer_id}.self_attn.{x}_proj.weight^T -> model.layers.{layer_id}.self_attn.{x}_proj.weight"
                ]
        if config.qkv_bias:
            aoa_statements += [
                f"{model_prefix}layers.$LAYER_ID.self_attn.qkv_proj.bias -> model.layers.$LAYER_ID.self_attn.q_proj.bias, model.layers.$LAYER_ID.self_attn.k_proj.bias, model.layers.$LAYER_ID.self_attn.v_proj.bias, fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups={config.num_key_value_heads}, axis=0",
            ]

        if config.get("fd_fallback", False):
            for layer_id in range(config.num_hidden_layers):
                ep_weight1 = []
                ep_weight2 = []
                for expert_id in range(num_experts):
                    ep_weight1.append(
                        f"{model_prefix}layers.{layer_id}.mlp.experts.{expert_id}.up_gate_proj.weight"
                    )
                    ep_weight2.append(
                        f"{model_prefix}layers.{layer_id}.mlp.experts.{expert_id}.down_proj.weight"
                    )
                group1 = ",".join(ep_weight1)
                group2 = ",".join(ep_weight2)
                aoa_statements += [
                    f"{model_prefix}layers.{layer_id}.mlp.experts.gate_up_proj -> {group1}, axis=0"
                    f"{model_prefix}layers.{layer_id}.mlp.experts.down_proj -> {group2}, axis=0"
                ]
            if is_fleet:
                aoa_statements += [
                    f"{model_prefix}layers.$LAYER_ID.mlp.shared_experts.up_gate_proj.weight -> model.layers.$LAYER_ID.mlp.shared_expert.gate_proj.weight, model.layers.$LAYER_ID.mlp.shared_expert.up_proj.weight, fused_ffn",
                ]
            else:
                aoa_statements += [
                    f"{model_prefix}layers.$LAYER_ID.mlp.shared_expert.up_gate_proj.weight -> model.layers.$LAYER_ID.mlp.shared_expert.gate_proj.weight, model.layers.$LAYER_ID.mlp.shared_expert.up_proj.weight, fused_ffn",
                ]
            for layer_id in range(config.num_hidden_layers):
                for expert_id in range(num_experts):
                    aoa_statements += [
                        f"{model_prefix}layers.{layer_id}.mlp.experts.{expert_id}.up_gate_proj.weight -> model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.weight, model.layers.{layer_id}.mlp.experts.{expert_id}.up_proj.weight, axis=1",
                    ]
                    aoa_statements += [
                        f"model.layers.{layer_id}.mlp.experts.{expert_id}.down_proj.weight^T -> model.layers.{layer_id}.mlp.experts.{expert_id}.down_proj.weight",
                    ]
        else:
            if is_fleet:
                aoa_statements += [
                    f"{model_prefix}layers.$LAYER_ID.mlp.shared_experts.up_gate_proj.weight -> model.layers.$LAYER_ID.mlp.shared_expert.gate_proj.weight, model.layers.$LAYER_ID.mlp.shared_expert.up_proj.weight, fused_ffn",
                    f"{model_prefix}layers.$LAYER_ID.mlp.experts.$EXPERT_ID.up_gate_proj.weight -> model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.gate_proj.weight, model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.up_proj.weight, axis=1",
                ]
            else:
                aoa_statements += [
                    f"{model_prefix}layers.$LAYER_ID.mlp.shared_expert.up_gate_proj.weight -> model.layers.$LAYER_ID.mlp.shared_expert.gate_proj.weight, model.layers.$LAYER_ID.mlp.shared_expert.up_proj.weight, fused_ffn",
                    f"{model_prefix}layers.$LAYER_ID.mlp.experts.$EXPERT_ID.up_gate_proj.weight -> model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.gate_proj.weight, model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.up_proj.weight, fused_ffn",
                ]
        for layer_id in range(config.num_hidden_layers):
            aoa_statements += [
                f"model.layers.{layer_id}.mlp.shared_expert.gate_proj.weight^T -> model.layers.{layer_id}.mlp.shared_expert.gate_proj.weight",
                f"model.layers.{layer_id}.mlp.shared_expert.up_proj.weight^T -> model.layers.{layer_id}.mlp.shared_expert.up_proj.weight",
            ]
            for expert_id in range(num_experts):
                aoa_statements += [
                    f"model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.weight^T -> model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.weight",
                    f"model.layers.{layer_id}.mlp.experts.{expert_id}.up_proj.weight^T -> model.layers.{layer_id}.mlp.experts.{expert_id}.up_proj.weight",
                ]

        if config.tie_word_embeddings:
            if is_fleet:
                aoa_statements += [f"{model_prefix}lm_head.weight -> _"]
            else:
                aoa_statements += ["lm_head.weight -> _"]
        else:
            if is_fleet:
                aoa_statements += [
                    f"{model_prefix}lm_head.weight -> lm_head.weight"
                ]

        aoa_config = {"aoa_statements": aoa_statements}
        return aoa_config


class Qwen2MoeForCausalLM(Qwen2MoePretrainedModel):
    is_fleet = True

    def __new__(cls, config):
        # Hybrid parallel config convert.
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
        config.n_shared_experts = (
            config.shared_expert_intermediate_size
            // config.moe_intermediate_size
        )

        model_provider_class = Qwen2MoeModelProvider
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


class Qwen2MoeForCausalLMPipe(
    Qwen2MoePretrainedModel, GeneralModelForCausalLMPipe
):
    is_fleet = True

    def __new__(cls, config):
        # Hybrid parallel config convert.
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
        config.n_shared_experts = (
            config.shared_expert_intermediate_size
            // config.moe_intermediate_size
        )

        model_provider_class = Qwen2MoeModelProvider
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
    "Qwen2MoePretrainedModel",
    "Qwen2MoeForCausalLM",
    "Qwen2MoeForCausalLMPipe",
    "Qwen2MoeModelProvider",
]
