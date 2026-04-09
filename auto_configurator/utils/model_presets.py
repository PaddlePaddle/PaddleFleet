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
Model Preset Configurations for AutoConfigurator

Contains dataclass definitions for various model presets (GPT, LLaMA, Qwen,
Mixtral, Gemma, etc.) that can be used with AutoConfigurator.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

# GPTConfig 类型声明 (用于类型注解)
if TYPE_CHECKING:
    from paddlefleet.models.gpt.gpt_config import GPTConfig
else:
    # 运行时导入，避免循环依赖
    try:
        from paddlefleet.models.gpt.gpt_config import GPTConfig
    except ImportError:
        # 如果 paddlefleet 不可用，使用一个占位符类型
        GPTConfig: type = object  # type: ignore


# ============================================================================
# 支持的模型类型
# ============================================================================

GPT_BASED_MODELS = [
    "gpt",
    "llama",
    "llama2",
    "llama3",
    "qwen",
    "qwen2",
    "qwen3",
    "mixtral",
    "mistral",
    "gemma",
    "glm",
]

# ============================================================================
# 预设模型配置 (类似 NeMo 的 recipe 工厂)
# ============================================================================


@dataclass
class GPT3_175BConfig(GPTConfig):
    """GPT-3 175B 配置预设."""

    num_hidden_layers: int = 96
    hidden_size: int = 12288
    num_attention_heads: int = 96
    intermediate_size: int = 4 * 12288  # 4x
    max_sequence_length: int = 2048
    vocab_size: int = 50000
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    bf16: bool = True


@dataclass
class Llama2_7BConfig(GPTConfig):
    """LLaMA-2 7B 配置预设."""

    num_hidden_layers: int = 32
    hidden_size: int = 4096
    num_attention_heads: int = 32
    intermediate_size: int = 11008
    max_sequence_length: int = 4096
    vocab_size: int = 32000
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    bf16: bool = True


@dataclass
class Llama2_70BConfig(GPTConfig):
    """LLaMA-2 70B 配置预设."""

    num_hidden_layers: int = 80
    hidden_size: int = 8192
    num_attention_heads: int = 64
    intermediate_size: int = 28672
    max_sequence_length: int = 4096
    vocab_size: int = 32000
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    bf16: bool = True


@dataclass
class Llama3_8BConfig(GPTConfig):
    """LLaMA-3 8B 配置预设."""

    num_hidden_layers: int = 32
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    intermediate_size: int = 14336
    max_sequence_length: int = 8192
    vocab_size: int = 128256
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    bf16: bool = True


@dataclass
class Llama3_70BConfig(GPTConfig):
    """LLaMA-3 70B 配置预设."""

    num_hidden_layers: int = 80
    hidden_size: int = 8192
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    intermediate_size: int = 28672
    max_sequence_length: int = 8192
    vocab_size: int = 128256
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    bf16: bool = True


@dataclass
class Qwen2_7BConfig(GPTConfig):
    """Qwen2 7B 配置预设."""

    num_hidden_layers: int = 28
    hidden_size: int = 3584
    num_attention_heads: int = 28
    intermediate_size: int = 18944
    max_sequence_length: int = 32768
    vocab_size: int = 152064
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    bf16: bool = True


@dataclass
class Qwen2_72BConfig(GPTConfig):
    """Qwen2 72B 配置预设."""

    num_hidden_layers: int = 80
    hidden_size: int = 8192
    num_attention_heads: int = 64
    intermediate_size: int = 29568
    max_sequence_length: int = 32768
    vocab_size: int = 152064
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    bf16: bool = True


@dataclass
class Qwen3MoE30BConfig(GPTConfig):
    """Qwen3-30B-A3B MoE 配置预设."""

    num_hidden_layers: int = 48
    hidden_size: int = 2048
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    intermediate_size: int = 6144
    max_sequence_length: int = 8192
    vocab_size: int = 151936
    # MoE 参数
    n_routed_experts: int = 128
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 768
    moe_grouped_gemm: bool = True
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    bf16: bool = True


@dataclass
class Mixtral8x7BConfig(GPTConfig):
    """Mixtral 8x7B MoE 配置预设."""

    num_hidden_layers: int = 32
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    intermediate_size: int = 14336
    max_sequence_length: int = 32768
    vocab_size: int = 32000
    # MoE 参数
    n_routed_experts: int = 8
    num_experts_per_tok: int = 2
    moe_intermediate_size: int = 5120
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    bf16: bool = True


@dataclass
class Mixtral8x22BConfig(GPTConfig):
    """Mixtral 8x22B MoE 配置预设."""

    num_hidden_layers: int = 56
    hidden_size: int = 6144
    num_attention_heads: int = 48
    num_key_value_heads: int = 8
    intermediate_size: int = 16384
    max_sequence_length: int = 32768
    vocab_size: int = 32000
    # MoE 参数
    n_routed_experts: int = 8
    num_experts_per_tok: int = 2
    moe_intermediate_size: int = 7680
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    bf16: bool = True


@dataclass
class Gemma2_9BConfig(GPTConfig):
    """Gemma-2 9B 配置预设."""

    num_hidden_layers: int = 42
    hidden_size: int = 3584
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    intermediate_size: int = 14336
    max_sequence_length: int = 8192
    vocab_size: int = 256000
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    bf16: bool = True


@dataclass
class Gemma2_27BConfig(GPTConfig):
    """Gemma-2 27B 配置预设."""

    num_hidden_layers: int = 46
    hidden_size: int = 4608
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    intermediate_size: int = 30720
    max_sequence_length: int = 8192
    vocab_size: int = 256000
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    bf16: bool = True


# ============================================================================
# 预设映射表
# ============================================================================

MODEL_PRESETS: dict[str, type[GPTConfig]] = {
    # GPT variants
    "gpt3_175b": GPT3_175BConfig,
    # LLaMA variants
    "llama2_7b": Llama2_7BConfig,
    "llama2_70b": Llama2_70BConfig,
    "llama3_8b": Llama3_8BConfig,
    "llama3_70b": Llama3_70BConfig,
    "llama_7b": Llama2_7BConfig,  # alias
    "llama_70b": Llama2_70BConfig,  # alias
    # Qwen variants
    "qwen2_7b": Qwen2_7BConfig,
    "qwen2_72b": Qwen2_72BConfig,
    "qwen3_moe_30b": Qwen3MoE30BConfig,
    "qwen3_30b": Qwen3MoE30BConfig,  # alias
    "qwen_7b": Qwen2_7BConfig,  # alias
    "qwen_72b": Qwen2_72BConfig,  # alias
    # Mixtral variants
    "mixtral_8x7b": Mixtral8x7BConfig,
    "mixtral_8x22b": Mixtral8x22BConfig,
    "mixtral_7b": Mixtral8x7BConfig,  # alias
    "mixtral_22b": Mixtral8x22BConfig,  # alias
    # Gemma variants
    "gemma2_9b": Gemma2_9BConfig,
    "gemma2_27b": Gemma2_27BConfig,
    "gemma_9b": Gemma2_9BConfig,  # alias
    "gemma_27b": Gemma2_27BConfig,  # alias
}


# ============================================================================
# 辅助函数
# ============================================================================


def list_presets() -> None:
    """列出所有可用的预设模型配置."""
    print("\n可用的预设模型配置:")
    print("=" * 60)
    print(f"{'名称':<20} {'模型':<15} {'参数':<30}")
    print("-" * 60)

    for name, config_cls in MODEL_PRESETS.items():
        if config_cls.__name__.startswith("GPT"):
            model = "GPT"
            params = f"{config_cls().num_hidden_layers}L, {config_cls().hidden_size}H, {config_cls().vocab_size}V"
        elif config_cls.__name__.startswith("Llama"):
            model = "LLaMA"
            params = f"{config_cls().num_hidden_layers}L, {config_cls().hidden_size}H, {config_cls().vocab_size}V"
        elif config_cls.__name__.startswith("Qwen"):
            model = "Qwen"
            params = f"{config_cls().num_hidden_layers}L, {config_cls().hidden_size}H, {config_cls().vocab_size}V"
        elif config_cls.__name__.startswith("Mixtral"):
            model = "Mixtral"
            moe = ", MoE" if hasattr(config_cls(), "n_routed_experts") else ""
            params = f"{config_cls().num_hidden_layers}L, {config_cls().hidden_size}H{moe}"
        elif config_cls.__name__.startswith("Gemma"):
            model = "Gemma"
            params = f"{config_cls().num_hidden_layers}L, {config_cls().hidden_size}H, {config_cls().vocab_size}V"
        else:
            model = config_cls.__name__
            params = "-"
        print(f"  {name:<20} {model:<15} {params:<30}")
    print("=" * 60)
