# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
"""Paddle Qwen3Moe model."""

from __future__ import annotations

from dataclasses import dataclass

from ...nn.pp_model import CriterionLayerPipe, GeneralModelForCausalLMPipe
from ..gpt_provider import GPTModelProvider
from ..model_utils import PretrainedModel
from .configuration import Qwen3MoeConfig


@dataclass
class Qwen3MoEModelProvider(GPTModelProvider):
    """Base provider for Qwen3 MoE Models."""

    moe_router_load_balancing_type: str = "aux_loss"

    gated_linear_unit: bool = True

    bias_activation_fusion: bool = True

    transform_rules = {
        "tensor_parallel_degree": "tensor_model_parallel_size",
        "pipeline_parallel_degree": "pipeline_model_parallel_size",
        "context_parallel_degree": "context_parallel_size",
        "expert_parallel_degree": "expert_model_parallel_size",
        "dtype": "params_dtype",
        "num_experts": "n_routed_experts",
        "num_local_experts": "n_routed_experts",
    }

    rotary_base: float = 1000000.0
    moe_router_pre_softmax: bool = False
    moe_permute_fusion: bool = True
    moe_router_dtype: str = "fp32"
    moe_router_enable_expert_bias: bool = False
    moe_router_bias_update_rate: float = 0
    persist_layer_norm: bool = True
    moe_router_force_load_balancing: bool = False
    share_embeddings_and_output_weights: bool = False

    apply_rope_fusion: bool = True
    recompute_granularity: str = None
    virtual_pipeline_model_parallel_size: int = None

    rope_scaling: float = 1.0
    bias_dropout_fusion: bool = True
    moe_expert_fusion: bool = True

    n_shared_experts: int = 0

    use_qk_norm: bool = True


class Qwen3MoePretrainedModel(PretrainedModel):
    config_class = Qwen3MoeConfig
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
    ]

    @classmethod
    def _gen_aoa_config(cls, config: Qwen3MoeConfig):
        if hasattr(config, "n_routed_experts"):
            num_experts = config.n_routed_experts
        elif hasattr(config, "num_local_experts"):
            num_experts = config.num_local_experts
        else:
            num_experts = config.num_experts

        model_prefix = (
            "" if cls == getattr(cls, "base_model_class", None) else "model."
        )
        using_sonic_moe = config.using_sonic_moe
        aoa_config = {
            "aoa_statements": [
                f"model.layers.$LAYER_ID.self_attn.o_proj.weight^T -> {model_prefix}layers.$LAYER_ID.self_attn.o_proj.weight",
                f"model.layers.$LAYER_ID.input_layernorm.weight -> {model_prefix}layers.$LAYER_ID.input_layernorm.weight",
                f"model.layers.$LAYER_ID.post_attention_layernorm.weight -> {model_prefix}layers.$LAYER_ID.post_attention_layernorm.weight",
                f"model.norm.weight -> {model_prefix}norm.weight",
            ]
        }

        if using_sonic_moe:
            aoa_config["aoa_statements"] += [
                f"model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.down_proj.weight -> {model_prefix}layers.$LAYER_ID.mlp.experts.$EXPERT_ID.down_proj.weight",
            ]
        else:
            aoa_config["aoa_statements"] += [
                f"model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.down_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.experts.$EXPERT_ID.down_proj.weight",
            ]

        if getattr(cls, "is_fleet", False):
            aoa_config["aoa_statements"] += [
                f"model.embed_tokens.weight -> {model_prefix}embedding.embed_tokens.weight",
                f"model.layers.$LAYER_ID.mlp.gate.weight -> {model_prefix}layers.$LAYER_ID.mlp.gate.weight, dtype='float32'",
                f"model.layers.$LAYER_ID.self_attn.q_norm.weight -> {model_prefix}layers.$LAYER_ID.self_attn.q_layernorm.weight",
                f"model.layers.$LAYER_ID.self_attn.k_norm.weight -> {model_prefix}layers.$LAYER_ID.self_attn.k_layernorm.weight",
                f"lm_head.weight -> {model_prefix}lm_head.weight",
            ]
        else:
            aoa_config["aoa_statements"] += [
                f"model.embed_tokens.weight -> {model_prefix}embed_tokens.weight",
                f"model.layers.$LAYER_ID.mlp.gate.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.gate.weight, dtype='float32'",
                f"model.layers.$LAYER_ID.self_attn.q_norm.weight -> {model_prefix}layers.$LAYER_ID.self_attn.q_norm.weight",
                f"model.layers.$LAYER_ID.self_attn.k_norm.weight -> {model_prefix}layers.$LAYER_ID.self_attn.k_norm.weight",
            ]

        # attention qkv
        aoa_config["aoa_statements"] += [
            f"model.layers.$LAYER_ID.self_attn.q_proj.weight^T, model.layers.$LAYER_ID.self_attn.k_proj.weight^T, model.layers.$LAYER_ID.self_attn.v_proj.weight^T -> {model_prefix}layers.$LAYER_ID.self_attn.qkv_proj.weight, fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups={config.num_key_value_heads}",
        ]
        if config.attention_bias:
            aoa_config["aoa_statements"] += [
                f"model.layers.$LAYER_ID.self_attn.q_proj.bias, model.layers.$LAYER_ID.self_attn.k_proj.bias, model.layers.$LAYER_ID.self_attn.v_proj.bias -> {model_prefix}layers.$LAYER_ID.self_attn.qkv_proj.bias, fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups={config.num_key_value_heads}, axis=0",
            ]

        # FFN
        if getattr(cls, "is_fleet", False):
            if using_sonic_moe:
                aoa_config["aoa_statements"] += [
                    f"model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.gate_proj.weight, model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.up_proj.weight -> {model_prefix}layers.$LAYER_ID.mlp.experts.$EXPERT_ID.up_gate_proj.weight, axis=0",
                ]
            else:
                aoa_config["aoa_statements"] += [
                    f"model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.gate_proj.weight^T, model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.up_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.experts.$EXPERT_ID.up_gate_proj.weight, axis=1",
                ]

        else:
            aoa_config["aoa_statements"] += [
                f"model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.gate_proj.weight^T, model.layers.$LAYER_ID.mlp.experts.$EXPERT_ID.up_proj.weight^T -> {model_prefix}layers.$LAYER_ID.mlp.experts.$EXPERT_ID.up_gate_proj.weight, fused_ffn",
            ]

        if getattr(cls, "is_fleet", False) and (
            config.moe_expert_fusion or using_sonic_moe
        ):
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
                group_gemm1 = ",".join(ep_weight1)
                group_gemm2 = ",".join(ep_weight2)
                aoa_config["aoa_statements"] += [
                    f"{group_gemm1} -> {tgt_prefix}.mlp.grouped_gemm_experts.weight1, axis=0",
                    f"{group_gemm2} -> {tgt_prefix}.mlp.grouped_gemm_experts.weight2, axis=0",
                ]
        else:
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
                        f"{group1} -> {tgt_prefix}.mlp.experts.gate_up_proj, axis=0",
                        f"{group2} -> {tgt_prefix}.mlp.experts.down_proj, axis=0",
                    ]

        # lm_head
        if config.tie_word_embeddings:
            aoa_config["aoa_statements"] += [
                "model.embed_tokens.weight -> lm_head.weight"
            ]

        return aoa_config

    @classmethod
    def _gen_inv_aoa_config(cls, config: Qwen3MoeConfig):
        if hasattr(config, "n_routed_experts"):
            num_experts = config.n_routed_experts
        elif hasattr(config, "num_local_experts"):
            num_experts = config.num_local_experts
        else:
            num_experts = config.num_experts

        model_prefix = (
            "" if cls == getattr(cls, "base_model_class", None) else "model."
        )
        using_sonic_moe = config.using_sonic_moe
        aoa_statements = [
            f"{model_prefix}layers.$LAYER_ID.self_attn.o_proj.weight^T -> model.layers.$LAYER_ID.self_attn.o_proj.weight",
            f"{model_prefix}layers.$LAYER_ID.input_layernorm.weight -> model.layers.$LAYER_ID.input_layernorm.weight",
            f"{model_prefix}layers.$LAYER_ID.post_attention_layernorm.weight -> model.layers.$LAYER_ID.post_attention_layernorm.weight",
            f"{model_prefix}norm.weight -> model.norm.weight",
        ]

        if getattr(cls, "is_fleet", False):
            aoa_statements += [
                f"{model_prefix}embedding.embed_tokens.weight -> model.embed_tokens.weight",
                f"{model_prefix}layers.$LAYER_ID.mlp.gate.weight -> model.layers.$LAYER_ID.mlp.gate.weight, dtype='bfloat16'",
                f"{model_prefix}layers.$LAYER_ID.self_attn.q_layernorm.weight -> model.layers.$LAYER_ID.self_attn.q_norm.weight",
                f"{model_prefix}layers.$LAYER_ID.self_attn.k_layernorm.weight -> model.layers.$LAYER_ID.self_attn.k_norm.weight",
                f"{model_prefix}lm_head.weight -> lm_head.weight",
            ]
        else:
            aoa_statements += [
                f"{model_prefix}embed_tokens.weight -> model.embed_tokens.weight",
                f"{model_prefix}layers.$LAYER_ID.mlp.gate.weight^T -> model.layers.$LAYER_ID.mlp.gate.weight, dtype='bfloat16'",
                f"{model_prefix}layers.$LAYER_ID.self_attn.q_norm.weight -> model.layers.$LAYER_ID.self_attn.q_norm.weight",
                f"{model_prefix}layers.$LAYER_ID.self_attn.k_norm.weight -> model.layers.$LAYER_ID.self_attn.k_norm.weight",
            ]

        aoa_statements += [
            f"{model_prefix}layers.$LAYER_ID.self_attn.qkv_proj.weight -> model.layers.$LAYER_ID.self_attn.q_proj.weight, model.layers.$LAYER_ID.self_attn.k_proj.weight, model.layers.$LAYER_ID.self_attn.v_proj.weight , fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups = {config.num_key_value_heads}",
        ]
        for layer_id in range(config.num_hidden_layers):
            for x in ("q", "k", "v"):
                aoa_statements += [
                    f"model.layers.{layer_id}.self_attn.{x}_proj.weight^T -> model.layers.{layer_id}.self_attn.{x}_proj.weight"
                ]
        if config.attention_bias:
            aoa_statements += [
                f"{model_prefix}layers.$LAYER_ID.self_attn.qkv_proj.bias -> model.layers.$LAYER_ID.self_attn.q_proj.bias, model.layers.$LAYER_ID.self_attn.k_proj.bias, model.layers.$LAYER_ID.self_attn.v_proj.bias, fused_qkv, num_heads={config.num_attention_heads}, num_key_value_groups={config.num_key_value_heads}, axis=0",
            ]

        if getattr(cls, "is_fleet", False) and (
            config.moe_expert_fusion or using_sonic_moe
        ):
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
                group_gemm1 = ",".join(ep_weight1)
                group_gemm2 = ",".join(ep_weight2)
                aoa_statements += [
                    f"{model_prefix}layers.{layer_id}.mlp.grouped_gemm_experts.weight1 -> {group_gemm1}, axis=0",
                    f"{model_prefix}layers.{layer_id}.mlp.grouped_gemm_experts.weight2 -> {group_gemm2}, axis=0",
                ]
        else:
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
                        f"{model_prefix}layers.{layer_id}.mlp.experts.gate_up_proj -> {group1}, axis=0",
                        f"{model_prefix}layers.{layer_id}.mlp.experts.down_proj -> {group2}, axis=0",
                    ]

        for layer_id in range(config.num_hidden_layers):
            for expert_id in range(num_experts):
                if getattr(cls, "is_fleet", False):
                    if using_sonic_moe:
                        aoa_statements += [
                            f"{model_prefix}layers.{layer_id}.mlp.experts.{expert_id}.up_gate_proj.weight -> model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.weight, model.layers.{layer_id}.mlp.experts.{expert_id}.up_proj.weight, axis=0",
                        ]
                    else:
                        aoa_statements += [
                            f"{model_prefix}layers.{layer_id}.mlp.experts.{expert_id}.up_gate_proj.weight -> model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.weight, model.layers.{layer_id}.mlp.experts.{expert_id}.up_proj.weight, axis=1",
                        ]
                else:
                    aoa_statements += [
                        f"{model_prefix}layers.{layer_id}.mlp.experts.{expert_id}.up_gate_proj.weight -> model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.weight, model.layers.{layer_id}.mlp.experts.{expert_id}.up_proj.weight, fused_ffn",
                    ]

                if not using_sonic_moe:
                    aoa_statements += [
                        f"model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.weight^T -> model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.weight",
                        f"model.layers.{layer_id}.mlp.experts.{expert_id}.up_proj.weight^T -> model.layers.{layer_id}.mlp.experts.{expert_id}.up_proj.weight",
                        f"model.layers.{layer_id}.mlp.experts.{expert_id}.down_proj.weight^T -> model.layers.{layer_id}.mlp.experts.{expert_id}.down_proj.weight",
                    ]

        if config.tie_word_embeddings:
            aoa_statements += ["lm_head.weight -> _"]

        aoa_config = {"aoa_statements": aoa_statements}
        return aoa_config


class Qwen3MoeForCausalLM(Qwen3MoePretrainedModel):
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
        config.fuse_rms_norm = True

        model_provider_class = Qwen3MoEModelProvider
        model_provider = model_provider_class.from_config(config)
        loss_fn = None
        if getattr(config, "dpo_config", None):
            loss_fn = CriterionLayerPipe(config, use_infohub=True)
        gpt_model = model_provider.provide(loss_fn=loss_fn)
        gpt_model._gen_aoa_config = cls._gen_aoa_config
        gpt_model._gen_inv_aoa_config = cls._gen_inv_aoa_config
        gpt_model._get_tensor_parallel_mappings = (
            cls._get_tensor_parallel_mappings
        )
        gpt_model.config_to_save = config
        gpt_model.is_fleet = cls.is_fleet

        return gpt_model


class Qwen3MoeForCausalLMPipe(
    Qwen3MoePretrainedModel, GeneralModelForCausalLMPipe
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
        config.fuse_rms_norm = True

        model_provider_class = Qwen3MoEModelProvider
        model_provider = model_provider_class.from_config(config)
        loss_fn = None
        if getattr(config, "dpo_config", None):
            loss_fn = CriterionLayerPipe(config, use_infohub=True)
        gpt_model = model_provider.provide(loss_fn=loss_fn)
        gpt_model._gen_aoa_config = cls._gen_aoa_config
        gpt_model._gen_inv_aoa_config = cls._gen_inv_aoa_config
        gpt_model._get_tensor_parallel_mappings = (
            cls._get_tensor_parallel_mappings
        )
        gpt_model.config_to_save = config
        gpt_model.is_fleet = cls.is_fleet
        return gpt_model


__all__ = [
    "Qwen3MoePretrainedModel",
    "Qwen3MoeForCausalLM",
    "Qwen3MoeForCausalLMPipe",
    "Qwen3MoEModelProvider",
]
