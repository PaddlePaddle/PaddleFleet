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
Command Line Arguments Parser for AutoConfigurator Example

Provides argument parsing aligned with NeMo's auto_config.py design.
"""

import argparse
import re

import yaml

from .model_presets import GPT_BASED_MODELS, MODEL_PRESETS


def _parse_size_list(sizes: str) -> list[int]:
    """Parse comma-separated size list.

    Args:
        sizes: Comma-separated string (e.g., "1,2,4" or "auto")

    Returns:
        List of integers, or None if "auto" is specified
    """
    if "auto" in sizes.lower():
        return None
    try:
        return [int(x.strip()) for x in sizes.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid size list: {sizes}")


def get_args():
    """解析命令行参数，对齐 NeMo auto_config.py 的参数设计.

    Returns:
        argparse.Namespace with parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="AutoConfigurator - 通用大模型并行策略自动配置",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用预设模型
  python main.py --model_type llama --model_size 7b --batch_mode

  # MoE 模型
  python main.py --model_type mixtral --model_size 7b --moe --batch_mode

  # 自定义参数
  python main.py \\
      --num_layers 32 \\
      --hidden_size 4096 \\
      --num_heads 32 \\
      --tensor_parallel_sizes 1,2,4 \\
      --batch_mode

  # 仅生成配置
  python main.py --model_type llama --model_size 7b

  # 收集结果
  python main.py --get_results
        """,
    )

    # 运行控制
    run_group = parser.add_argument_group("运行控制")
    run_group.add_argument(
        "--run_number",
        type=int,
        help="运行第 N 个配置 (1-based index). 不指定则仅生成配置.",
    )
    run_group.add_argument(
        "--batch_mode",
        action="store_true",
        help="批量运行所有候选配置",
    )
    run_group.add_argument(
        "--get_results",
        action="store_true",
        help="从现有日志收集性能结果",
    )
    run_group.add_argument(
        "--max_configs",
        type=int,
        default=None,
        help="评分后取 Top-N 配置运行 (需配合 scoring_fn 使用)",
    )

    # 模型配置
    model_group = parser.add_argument_group("模型配置")
    model_group.add_argument(
        "--model_type",
        type=str,
        default="llama",
        choices=GPT_BASED_MODELS,
        help="模型类型 (default: llama)",
    )
    model_group.add_argument(
        "--model_size",
        type=str,
        help="预设模型大小 (如: 7b, 70b, 175b, 8x7b, 30b)",
    )
    model_group.add_argument(
        "--moe",
        action="store_true",
        help="启用 MoE 模型 (自动添加专家并行搜索)",
    )

    # 自定义模型参数
    custom_group = parser.add_argument_group("自定义模型参数 (覆盖预设)")
    custom_group.add_argument(
        "--num_layers",
        type=int,
        help="层数 (num_hidden_layers)",
    )
    custom_group.add_argument(
        "--hidden_size",
        type=int,
        help="隐藏层维度",
    )
    custom_group.add_argument(
        "--num_heads",
        type=int,
        help="注意力头数 (num_attention_heads)",
    )
    custom_group.add_argument(
        "--num_kv_heads",
        type=int,
        help="KV头数 (num_key_value_heads, for GQA)",
    )
    custom_group.add_argument(
        "--ffn_size",
        type=int,
        help="FFN维度 (intermediate_size)",
    )
    custom_group.add_argument(
        "--seq_length",
        type=int,
        help="序列长度 (max_sequence_length)",
    )
    custom_group.add_argument(
        "--vocab_size",
        type=int,
        help="词表大小",
    )

    # MoE 参数
    moe_group = parser.add_argument_group("MoE 参数")
    moe_group.add_argument(
        "--num_experts",
        type=int,
        help="专家数量 (n_routed_experts)",
    )
    moe_group.add_argument(
        "--experts_per_tok",
        type=int,
        help="每token的专家数 (num_experts_per_tok)",
    )
    moe_group.add_argument(
        "--moe_ffn_size",
        type=int,
        help="MoE FFN维度 (moe_intermediate_size)",
    )

    # 并行搜索空间
    parallel_group = parser.add_argument_group("并行搜索空间")
    parallel_group.add_argument(
        "--tensor_parallel_sizes",
        type=str,
        default="auto",
        help="张量并行尺寸，逗号分隔或 'auto' (default: auto)",
    )
    parallel_group.add_argument(
        "--pipeline_parallel_sizes",
        type=str,
        default="auto",
        help="流水线并行尺寸，逗号分隔或 'auto' (default: auto)",
    )
    parallel_group.add_argument(
        "--context_parallel_sizes",
        type=str,
        default="1",
        help="上下文并行尺寸，逗号分隔 (default: 1)",
    )
    parallel_group.add_argument(
        "--expert_parallel_sizes",
        type=str,
        default="1",
        help="专家并行尺寸，逗号分隔 (default: 1)",
    )
    parallel_group.add_argument(
        "--micro_batch_sizes",
        type=str,
        default="auto",
        help="微批次大小，逗号分隔或 'auto' (default: auto)",
    )

    # 训练配置
    training_group = parser.add_argument_group("训练配置")
    training_group.add_argument(
        "--global_batch_size",
        type=int,
        default=512,
        help="全局批次大小 (default: 512)",
    )
    training_group.add_argument(
        "--max_steps",
        type=int,
        default=50,
        help="每个配置的最大训练步数 (default: 50)",
    )
    training_group.add_argument(
        "--base_yaml",
        type=str,
        help="基础训练配置 YAML 路径 (用于实际训练运行)",
    )

    # 集群配置
    cluster_group = parser.add_argument_group("集群配置")
    cluster_group.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="节点数 (default: 1)",
    )
    cluster_group.add_argument(
        "--gpus_per_node",
        type=int,
        default=8,
        help="每节点 GPU 数 (default: 8)",
    )
    cluster_group.add_argument(
        "--gpu_memory",
        type=int,
        default=80,
        choices=[40, 80],
        help="每GPU显存大小 (default: 80)",
    )

    # 其他选项
    other_group = parser.add_argument_group("其他选项")
    other_group.add_argument(
        "--log_dir",
        type=str,
        default="./auto_config_logs",
        help="日志输出目录 (default: ./auto_config_logs)",
    )
    other_group.add_argument(
        "--list_presets",
        action="store_true",
        help="列出所有可用的预设模型配置",
    )
    other_group.add_argument(
        "--dry_run",
        action="store_true",
        help="仅打印命令，不实际执行",
    )
    other_group.add_argument(
        "--extra_metrics",
        action="store_true",
        help="启用额外指标监控",
    )

    return parser.parse_args()


def _infer_model_from_path(model_name_or_path: str) -> str | None:
    """从 model_name_or_path 推断 MODEL_PRESETS key.

    匹配规则: 将路径末段转为小写并去除连字符，与 MODEL_PRESETS keys 做子串匹配。
    例如:
      ".../Qwen3-30B-A3B" → "qwen3_30b" (匹配 "qwen3_30b" in MODEL_PRESETS)
      ".../Llama-3-8B"    → "llama3_8b"
      ".../Mixtral-8x7B"  → "mixtral_8x7b"
    """
    basename = model_name_or_path.rstrip("/").rsplit("/", 1)[-1]
    # 标准化: "Qwen3-30B-A3B" → "qwen3_30b_a3b"
    normalized = basename.lower().replace("-", "_")

    best_key = None
    best_len = 0
    for key in MODEL_PRESETS:
        # key 如 "qwen3_30b", "llama3_8b", "mixtral_8x7b"
        if key in normalized and len(key) > best_len:
            best_key = key
            best_len = len(key)

    # 若未匹配，尝试去除所有下划线后再匹配
    # 处理 "Llama-3-8B" → "llama_3_8b" vs key "llama3_8b" 的情况
    if best_key is None:
        collapsed = normalized.replace("_", "")
        for key in MODEL_PRESETS:
            key_collapsed = key.replace("_", "")
            if key_collapsed in collapsed and len(key) > best_len:
                best_key = key
                best_len = len(key)

    return best_key


def _parse_preset_key(preset_key: str):
    """从 preset key 解析 model_type, model_size, moe.

    例如:
      "qwen3_30b"     → ("qwen3", "30b", False)
      "qwen3_moe_30b" → ("qwen3", "30b", True)
      "llama3_8b"     → ("llama3", "8b", False)
      "mixtral_8x7b"  → ("mixtral", "8x7b", False)
    """
    # 尝试 {type}_moe_{size} 格式
    m = re.match(r"^(\w+?)_moe_(\w+)$", preset_key)
    if m:
        return m.group(1), m.group(2), True

    # 尝试 {type}_{size} 格式 (size 以数字开头)
    m = re.match(r"^(\w+?)_(\d\w*)$", preset_key)
    if m:
        model_type = m.group(1)
        model_size = m.group(2)
        # 检查该 preset 的 config 是否带 MoE 参数
        config_cls = MODEL_PRESETS.get(preset_key)
        moe = False
        if config_cls is not None:
            instance = config_cls()
            moe = getattr(instance, "n_routed_experts", 0) > 0
        return model_type, model_size, moe

    return None, None, False


def load_args_from_yaml(yaml_path: str, **overrides) -> argparse.Namespace:
    """从标准 PaddleFormers 训练 YAML 推断 AutoConfigurator 参数.

    自动从 YAML 中的 model_name_or_path 推断 model_type/model_size/moe，
    从 use_expert_parallel 推断 MoE 模式。搜索空间等参数通过 overrides 传入。

    Args:
        yaml_path: 标准 PaddleFormers 训练 YAML 路径
        **overrides: 覆盖推断值的键值对，如 tensor_parallel_sizes="1", max_steps=10

    Returns:
        argparse.Namespace，与 get_args() 返回格式一致
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    # --- 从 model_name_or_path 推断 model_type / model_size / moe ---
    model_path = data.get("model_name_or_path", "")
    preset_key = _infer_model_from_path(model_path)

    model_type, model_size, moe = None, None, False
    if preset_key:
        model_type, model_size, moe = _parse_preset_key(preset_key)

    # use_expert_parallel 字段也可判断 MoE
    if data.get("use_expert_parallel", False):
        moe = True

    # 构建 Namespace，overrides 优先
    defaults = {
        # model
        "model_type": model_type or "llama",
        "model_size": model_size,
        "moe": moe,
        # search space
        "tensor_parallel_sizes": "auto",
        "pipeline_parallel_sizes": "auto",
        "context_parallel_sizes": "1",
        "expert_parallel_sizes": "auto",
        "micro_batch_sizes": "auto",
        # training
        "global_batch_size": 128,
        "max_steps": 50,
        "max_configs": None,
        # cluster
        "num_nodes": 1,
        "gpus_per_node": 8,
        "gpu_memory": 80,
        # paths
        "log_dir": "./auto_config_logs",
        "base_yaml": yaml_path,
        # run mode
        "batch_mode": True,
        "run_number": None,
        "get_results": False,
        "list_presets": False,
        "dry_run": False,
        "extra_metrics": False,
        # custom model params (not used when preset is available)
        "num_layers": None,
        "hidden_size": None,
        "num_heads": None,
        "num_kv_heads": None,
        "ffn_size": None,
        "seq_length": None,
        "vocab_size": None,
        "num_experts": None,
        "experts_per_tok": None,
        "moe_ffn_size": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)
