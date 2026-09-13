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
Concrete DeepSeek V3.2 model providers for PaddleFleet-based pretraining.

Architecture: MLA (Multi-Latent Attention) + DSA Indexer (DeepSeek Sparse Attention)
             + MoE (Mixture of Experts) + MTP (Multi-Token Prediction)

The shared base provider lives in the library at
``paddlefleet.transformers.deepseek_v32.modeling.DeepSeekV3_2BaseProvider``;
this file only carries the concrete size/debug variants.

Usage:
    provider = DeepSeekV3_2_671BProvider()
    model = provider.provide(loss_fn=loss_fn)
"""

from dataclasses import dataclass, field

from paddlefleet.transformers.deepseek_v32.modeling import (
    DeepSeekV3_2BaseProvider,
)


@dataclass
class DeepSeekV3_2_671BProvider(DeepSeekV3_2BaseProvider):
    """
    Provider for DeepSeek V3.2 671B model (full production config).

    Architecture:
    - 61 transformer layers: first 3 dense MLP + 58 MoE
    - All layers use MLA + DSA Indexer attention
    - 256 routed experts + 1 shared expert per MoE layer

    Config reference: DeepSeek-V3.2-Exp/inference/config_671B_v3.2.json
    """

    # ---- Model dimensions ----
    hidden_size: int = 7168  # dim
    num_hidden_layers: int = 61  # n_layers
    vocab_size: int = 129280

    # ---- FFN dimensions ----
    intermediate_size: int = 18432  # inter_dim: dense MLP hidden size
    moe_intermediate_size: int = (
        2048  # moe_inter_dim: per-expert MLP hidden size
    )

    # ---- MoE architecture ----
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    # Layer pattern: first 3 layers dense (0), then 58 MoE (1)
    moe_layer_freq: int | list[int] = field(
        default_factory=lambda: [0] * 3 + [1] * 58
    )


@dataclass
class DeepSeekV3_2_671BDebugProvider(DeepSeekV3_2_671BProvider):
    """
    Small debug variant of DeepSeek V3.2 for single-card validation.

    Reduces all dimensions to fit on a single GPU for smoke testing.
    Pattern: 1 dense layer + 3 MoE layers.
    """

    # ---- Reduced model dimensions ----
    num_hidden_layers: int = 4
    hidden_size: int = 1024
    vocab_size: int = 129280

    # ---- Reduced attention dimensions ----
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 64
    q_lora_rank: int = 256
    kv_lora_rank: int = 128
    qk_nope_head_dim: int = 64
    qk_rope_head_dim: int = 32
    v_head_dim: int = 64

    # ---- Reduced Indexer dimensions ----
    index_n_heads: int = 8
    index_head_dim: int = 64
    index_topk: int = 128
    indexer_loss_coeff: float = 0.01
    indexer_use_sparse_loss: bool = False

    # ---- Reduced FFN dimensions ----
    intermediate_size: int = 2048
    moe_intermediate_size: int = 512

    # ---- Reduced MoE ----
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    moe_layer_freq: int | list[int] = field(
        default_factory=lambda: [0] * 1 + [1] * 3
    )

    # ---- Disable MTP for simplicity ----
    num_nextn_predict_layers: int | None = 0

    # ---- Short sequence for debug ----
    seq_length: int = 512
    max_sequence_length: int = 512

    # ---- Single card: no model parallel ----
    sequence_parallel: bool = False
    expert_model_parallel_size: int = 1
    tensor_model_parallel_size: int = 1
    moe_router_force_load_balancing: bool = True


@dataclass
class DeepSeekV3_2_8GPUDebugProvider(DeepSeekV3_2BaseProvider):
    """
    Debug provider for DeepSeek V3.2 on a single node with 8 GPUs.

    Scales up from the single-card DebugProvider to exercise multi-card
    communication paths (all-reduce, all-gather, DeepEP routing) without
    the memory footprint of the full 671B model.

    Key dimension constraints for parallelism:
        num_attention_heads (32) and index_n_heads (16) must be
        divisible by whatever tensor_model_parallel_size is used.
        n_routed_experts (16) must be divisible by expert_model_parallel_size.

    Pattern: 2 dense layers + 6 MoE layers (8 total).
    """

    # ---- Reduced model dimensions ----
    num_hidden_layers: int = 8
    hidden_size: int = 2048
    vocab_size: int = 129280

    # ---- Reduced attention dimensions ----
    num_attention_heads: int = 32  # divisible by TP=1/2/4/8
    num_key_value_heads: int = 32
    head_dim: int = 64
    q_lora_rank: int = 512
    kv_lora_rank: int = 128
    qk_nope_head_dim: int = 64
    qk_rope_head_dim: int = 32
    v_head_dim: int = 64

    # ---- Reduced Indexer dimensions ----
    index_n_heads: int = 16  # divisible by TP=1/2/4/8
    index_head_dim: int = 64
    index_topk: int = 256
    indexer_loss_coeff: float = 0.01
    indexer_use_sparse_loss: bool = False

    # ---- Reduced FFN dimensions ----
    intermediate_size: int = 4096
    moe_intermediate_size: int = 1024

    # ---- Reduced MoE ----
    n_routed_experts: int = 16  # divisible by EP=1/2/4/8
    n_shared_experts: int = 1
    moe_layer_freq: int | list[int] = field(
        default_factory=lambda: [0] * 2 + [1] * 6
    )

    # ---- Disable MTP for simplicity ----
    num_nextn_predict_layers: int | None = 0

    # ---- Moderate sequence length ----
    seq_length: int = 1024
    max_sequence_length: int = 1024
