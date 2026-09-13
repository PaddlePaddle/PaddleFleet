# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
DeepSeek V3.2 PaddleFleet model bridge.

This module bridges the HuggingFace-style PretrainedConfig/PretrainedModel
interface used by PaddleFleet with the PaddleFleet provider system.

Pattern follows glm_moe_dsa/modeling.py (GLM5 PR #3940) exactly:
  - DeepseekV32ForCausalLM.__new__() calls DeepSeekV3_2BaseProvider.from_config(config)
  - provider.provide() calls paddlefleet.gpt_builders.gpt_builder()
  - Returns a PaddleFleet GPT model (Megatron-style)
"""

from dataclasses import dataclass
from typing import Callable, Optional

import paddle
import paddle.nn.functional as F

from ..aoa_config_base import MoEAOAConfigGenerator
from ..gpt_provider import GPTModelProvider
from ..model_utils import PretrainedModel
from .configuration import DeepseekV32Config


@dataclass
class DeepSeekV3_2BaseProvider(GPTModelProvider):
    """
    Base provider for DeepSeek V3.2 architecture.

    Key components:
    - MLA: Multi-Latent Attention with low-rank KV compression
    - DSA: DeepSeek Sparse Attention (Indexer selects top-2048 tokens per query)
    - MoE: Mixture of Experts with group-limited routing
    - MTP: Multi-Token Prediction auxiliary loss

    Reference: DeepSeek-V3.2-Exp/inference/model.py
    Config:    DeepSeek-V3.2-Exp/inference/config_671B_v3.2.json
    """

    # ---- Normalization and activation ----
    normalization: str = "RMSNorm"
    hidden_act: Callable = F.silu
    gated_linear_unit: bool = True
    use_bias: bool = False
    attention_bias: bool = False
    rms_norm_eps: float = 1e-6

    # ---- Precision ----
    autocast_dtype: paddle.dtype = paddle.bfloat16
    params_dtype: paddle.dtype = paddle.bfloat16
    bf16: bool = True

    # ---- Embedding ----
    tie_word_embeddings: bool = False

    # ---- Sequence ----
    seq_length: int = 4096
    max_sequence_length: int = 4096
    hidden_dropout_prob: float = 0.0
    attention_dropout: float = 0.0
    init_method_std: float = 0.006  # ~1/sqrt(7168)

    # ---- MLA: Multi-Latent Attention ----
    # MLA de-interleave in rope_utils is NOT needed when rotary_interleaved=True,
    # because _rotate_half(interleaved=True) already pairs adjacent dims correctly
    # (matching DeepSeek-V3.2 reference apply_rotary_emb(interleaved=True)).
    multi_latent_attention: bool = False
    num_attention_heads: int = 128
    # head_dim matches v_head_dim=128 so o_proj sizing in Attention base is correct
    head_dim: int = 128
    # num_key_value_heads must be set for Attention base class;
    # in MLA, KV is latent-compressed but we set this equal to num_attention_heads
    # so TP sharding logic in Attention.__init__ works correctly
    num_key_value_heads: int = 128

    # MLA low-rank projection dimensions (matches DeepSeek V3.2 671B config)
    q_lora_rank: int = 1536  # wq_a: hidden -> q_lora_rank
    kv_lora_rank: int = 512  # wkv_a: hidden -> kv_lora_rank + qk_rope_head_dim
    qk_nope_head_dim: int = 128  # per-head non-RoPE Q/K dim
    qk_rope_head_dim: int = 64  # per-head RoPE Q/K dim
    v_head_dim: int = 128  # per-head V dim (= head_dim, so o_proj ok)

    # ---- DSA: DeepSeek Sparse Attention Indexer ----
    # Non-None activates the DeepSeek V3.2 path in gpt_builders.py
    # Field names mirror HuggingFace config.json keys for zero-copy from_config().
    index_n_heads: int = 64  # Indexer scoring heads
    index_head_dim: int = 128  # Indexer Q/K head dim
    index_topk: int = 2048  # Tokens selected per query
    # KL loss trains wq_b/wk/weights_proj via KL(true_attn_dist || indexer_dist)
    # Coefficient ~0.01 matches Megatron-Core default; set to None to disable
    indexer_loss_coeff: float = 0.01
    indexer_use_sparse_loss: bool = (
        False  # use full-sequence KL (denser gradients)
    )

    # ---- RoPE ----
    position_embedding_type: str = "rope"
    # DeepSeek V3.2 uses YaRN-style RoPE with base 10000
    rotary_base: float = 10000.0
    # MLA uses interleaved RoPE; Indexer uses non-interleaved (handled internally)
    # Setting rotary_interleaved=True here enables the interleaved path for MLA Q/K
    rotary_interleaved: bool = True
    # Disable fused RoPE kernel: MLA applies RoPE only to qk_rope_head_dim subspace,
    # which is incompatible with the fused kernel that expects full head_dim
    apply_rope_fusion: bool = False
    # Use fp32 RoPE for numerical stability (matches reference implementation)
    high_precision_rope: bool = True

    # ---- MoE routing ----
    scoring_func: str = "sigmoid"  # Score experts with sigmoid
    num_experts_per_tok: int = 8  # n_activated_experts
    n_group: int = 8  # n_expert_groups: 256 experts / 8 groups = 32 per group
    topk_group: int = 4  # n_limited_groups: select top-4 groups
    routed_scaling_factor: float = (
        2.5  # route_scale: scale selected expert weights
    )
    topk_method: str = "group_limited_greedy"  # group-limited top-k routing
    norm_topk_prob: bool = True  # normalize expert weights to sum to 1
    moe_token_dispatcher_type: str = "deepep"
    moe_router_load_balancing_type: str = "seq_aux_loss"
    moe_router_pre_softmax: bool = False
    moe_expert_fusion: bool = False
    moe_shared_expert_overlap: bool = True
    moe_router_dtype: str = "fp32"
    moe_router_enable_expert_bias: bool = True
    moe_router_bias_update_rate: float = 0.0

    # ---- MTP: Multi-Token Prediction ----
    # 1 MTP layer for auxiliary next-token prediction loss
    num_nextn_predict_layers: Optional[int] = 1
    mtp_loss_scaling_factor: float = 0.1  # MTP loss weight

    # ---- Optimization ----
    persist_layer_norm: bool = True
    bias_activation_fusion: bool = True
    bias_dropout_fusion: bool = True


class DeepseekV32PreTrainedModel(PretrainedModel):
    config_class = DeepseekV32Config
    base_model_prefix = "model"

    # Layernorm weight names that need dtype cast (fleet model skips generic dtype mapping)
    _NORM_WEIGHT_KEYS = (
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "q_a_layernorm.weight",
        "kv_a_layernorm.weight",
        "k_norm.weight",
        "k_norm.bias",
        "norm.weight",
    )

    @classmethod
    def _gen_aoa_config(cls, config: DeepseekV32Config):
        aoa_config = MoEAOAConfigGenerator.gen_aoa_config(config)
        cls._inject_norm_dtype(aoa_config["aoa_statements"], "bfloat16")
        return aoa_config

    @classmethod
    def _gen_inv_aoa_config(cls, config: DeepseekV32Config):
        inv_aoa_config = MoEAOAConfigGenerator.gen_inv_aoa_config(config)
        cls._inject_norm_dtype(inv_aoa_config["aoa_statements"], "float32")
        return inv_aoa_config

    @classmethod
    def _inject_norm_dtype(cls, aoa_statements, target_dtype):
        """Inject dtype into existing layernorm statements generated by base class."""
        for i, stmt in enumerate(aoa_statements):
            if (
                any(k in stmt for k in cls._NORM_WEIGHT_KEYS)
                and "dtype=" not in stmt
            ):
                aoa_statements[i] = f"{stmt}, dtype='{target_dtype}'"


def _build_model(config):
    """
    Common __new__ logic shared by ForCausalLM and ForCausalLMPipe.

    Steps:
    1. Normalise parallel config attributes (same as GLM5).
    2. Call DeepSeekV3_2BaseProvider.from_config(config) to populate provider fields,
       then provider.provide() which runs gpt_builder() and returns the PaddleFleet model.
       (moe_layer_freq + first_k_dense_replace conversion is handled by
        TransformerConfig.__post_init__ automatically.)
    """
    # 1. Normalise parallel config (guard against missing attrs from old configs)
    config.tensor_model_parallel_size = max(
        getattr(config, "tensor_model_parallel_size", 1), 1
    )
    config.pipeline_model_parallel_size = max(
        getattr(config, "pipeline_model_parallel_size", 1), 1
    )
    config.context_parallel_size = max(
        getattr(config, "context_parallel_size", 1), 1
    )
    config.virtual_pipeline_model_parallel_size = max(
        getattr(config, "virtual_pipeline_model_parallel_size", 1), 1
    )
    config.expert_model_parallel_size = max(
        getattr(config, "expert_model_parallel_size", 1), 1
    )

    # 2. Build model via provider
    model_provider = DeepSeekV3_2BaseProvider.from_config(config)
    gpt_model = model_provider.provide()
    gpt_model.config_to_save = config
    return gpt_model


class DeepseekV32ForCausalLM(DeepseekV32PreTrainedModel):
    """DeepSeek V3.2 model for pipeline_model_parallel_size == 1."""

    is_fleet = True

    def __new__(cls, config):
        gpt_model = _build_model(config)
        gpt_model.is_fleet = cls.is_fleet
        gpt_model._gen_aoa_config = cls._gen_aoa_config
        gpt_model._gen_inv_aoa_config = cls._gen_inv_aoa_config
        return gpt_model


class DeepseekV32ForCausalLMPipe(DeepseekV32PreTrainedModel):
    """DeepSeek V3.2 model for pipeline_model_parallel_size > 1."""

    is_fleet = True

    def __new__(cls, config):
        if not hasattr(config, "architectures"):
            config.architectures = ["DeepseekV32ForCausalLM"]
        gpt_model = _build_model(config)
        gpt_model.is_fleet = cls.is_fleet
        gpt_model._gen_aoa_config = cls._gen_aoa_config
        gpt_model._gen_inv_aoa_config = cls._gen_inv_aoa_config
        return gpt_model


__all__ = [
    "DeepSeekV3_2BaseProvider",
    "DeepseekV32PreTrainedModel",
    "DeepseekV32ForCausalLM",
    "DeepseekV32ForCausalLMPipe",
]
