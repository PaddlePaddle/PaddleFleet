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
AutoConfigurator Main Entry - 通用大模型并行策略自动配置

对齐 NeMo examples/llm/auto_configurator/auto_config.py 的设计模式。

支持模型类型:
  - GPT-based: gpt, llama, qwen, mixtral, mistral, gemma, glm
  - MoE: 指定 --moe 参数启用专家并行

运行模式:
  1. 仅生成配置: python main.py
  2. 单次运行: python main.py --run_number 1
  3. 批量运行: python main.py --batch_mode
  4. 结果收集: python main.py --get_results

示例命令:
  # 7B 模型
  python main.py --model_type llama --model_size 7b --batch_mode

  # 30B MoE 模型
  python main.py --model_type qwen3 --model_size 30b --moe --batch_mode

  # 自定义参数
  python main.py \\
      --model_type gpt \\
      --num_layers 32 \\
      --hidden_size 4096 \\
      --num_heads 32 \\
      --tensor_parallel_sizes 1,2,4 \\
      --batch_mode
"""

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

try:
    from paddlefleet.models.gpt.gpt_config import GPTConfig
except ImportError as e:
    print("ERROR: Failed to import paddlefleet.")
    print("Please set PYTHONPATH before running this script:")
    print()
    print(
        "  export PYTHONPATH=<path/to/PaddleFleet/src>:<path/to/PaddleFormers>:$PYTHONPATH"
    )
    print()
    raise e

from auto_configurator import (
    AutoConfigurator,
    PaddleFleetRecipe,
    generate_configs,
)
from auto_configurator.utils.cli_args import _parse_size_list, get_args
from auto_configurator.utils.model_presets import (
    MODEL_PRESETS,
    list_presets,
)
from auto_configurator.utils.results_formatter import (
    parse_training_logs,
    print_results,
    save_results_to_csv,
)
from auto_configurator.utils.training_runner import run_single_config

# ============================================================================
# Recipe 工厂函数
# ============================================================================


@dataclass
class CustomModelConfig(GPTConfig):
    """Custom model config from CLI args."""

    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    bf16: bool = True


def create_recipe(args) -> PaddleFleetRecipe:
    """Create PaddleFleetRecipe from CLI args.

    Args:
        args: Parsed command line arguments

    Returns:
        PaddleFleetRecipe instance
    """
    # 预设模型配置
    model_config: GPTConfig | None = None

    if args.model_size:
        preset_key = f"{args.model_type}_{args.model_size}"
        if preset_key in MODEL_PRESETS:
            model_config = MODEL_PRESETS[preset_key]()
        else:
            # Try alias
            alias_key = f"{args.model_type}_{args.model_size.replace('b', '')}b"
            if alias_key in MODEL_PRESETS:
                model_config = MODEL_PRESETS[alias_key]()
            elif args.moe:
                # MoE 模型可能使用 {type}_moe_{size} 命名
                moe_key = f"{args.model_type}_moe_{args.model_size}"
                if moe_key in MODEL_PRESETS:
                    model_config = MODEL_PRESETS[moe_key]()
                else:
                    raise ValueError(f"Unknown model preset: {preset_key}")
            else:
                raise ValueError(f"Unknown model preset: {preset_key}")
    else:
        # 使用自定义参数
        model_config = CustomModelConfig()
        if args.num_layers is not None:
            model_config.num_hidden_layers = args.num_layers
        if args.hidden_size is not None:
            model_config.hidden_size = args.hidden_size
        if args.num_heads is not None:
            model_config.num_attention_heads = args.num_heads
        if args.num_kv_heads is not None:
            model_config.num_key_value_heads = args.num_kv_heads
        if args.ffn_size is not None:
            model_config.intermediate_size = args.ffn_size
        if args.seq_length is not None:
            model_config.max_sequence_length = args.seq_length
        if args.vocab_size is not None:
            model_config.vocab_size = args.vocab_size

    # MoE 参数
    if args.moe:
        if args.num_experts is not None:
            model_config.n_routed_experts = args.num_experts
        if args.experts_per_tok is not None:
            model_config.num_experts_per_tok = args.experts_per_tok
        if args.moe_ffn_size is not None:
            model_config.moe_intermediate_size = args.moe_ffn_size
            model_config.moe_expert_fusion = True
            model_config.expert_model_parallel_size = 1

    return PaddleFleetRecipe(
        model_config=model_config,
        micro_batch_size=1,
        global_batch_size=args.global_batch_size,
        num_nodes=args.num_nodes,
        num_gpus_per_node=args.gpus_per_node,
        max_steps=args.max_steps,
        log_dir=args.log_dir,
    )


# ============================================================================
# 主流程
# ============================================================================


def train_config(
    args,
    scoring_fn: Callable | None = None,
    max_configs: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    """AutoConfigurator 主流程.

    Args:
        args: 命令行参数
        scoring_fn: 可选的配置评分函数 (GeneratedConfig -> float)，用于筛选 Top-N 配置
        max_configs: 评分后取 Top-N 配置。需配合 scoring_fn 使用

    Returns:
        (base_config, configs) 元组
    """
    # 列出预设模型
    if args.list_presets:
        list_presets()
        return None, {}

    # 解析搜索空间
    tensor_parallel_sizes = _parse_size_list(args.tensor_parallel_sizes)
    pipeline_parallel_sizes = _parse_size_list(args.pipeline_parallel_sizes)
    context_parallel_sizes = _parse_size_list(args.context_parallel_sizes)
    expert_parallel_sizes = _parse_size_list(args.expert_parallel_sizes)
    micro_batch_sizes = _parse_size_list(args.micro_batch_sizes)

    # MoE 模式默认启用 EP 搜索
    if args.moe and args.expert_parallel_sizes == "1":
        expert_parallel_sizes = None  # 使用 auto

    # 创建 recipe
    recipe = create_recipe(args)

    # 创建 AutoConfigurator
    runner = AutoConfigurator(
        recipe=recipe,
        path_to_logs=args.log_dir,
        mode="pretrain",
        gpu_memory_gb=args.gpu_memory,
        tensor_parallel_sizes=tensor_parallel_sizes,
        pipeline_parallel_sizes=pipeline_parallel_sizes,
        micro_batch_sizes=micro_batch_sizes,
        context_parallel_sizes=context_parallel_sizes,
        expert_parallel_sizes=expert_parallel_sizes,
        max_steps_per_run=args.max_steps,
        calculate_model_size=False,
    )

    # 打印配置信息
    print()
    print("=" * 70)
    print("  AutoConfigurator - 并行策略自动配置")
    print("=" * 70)
    print(f"  模型类型:   {args.model_type}")
    if args.model_size:
        print(f"  模型大小:   {args.model_size}")
    if args.moe:
        print("  MoE 模式:   是")
    print(
        f"  GPU:        {runner.gpu_count} GPUs "
        f"({args.num_nodes} nodes x {args.gpus_per_node}, {args.gpu_memory}GB)"
    )
    print(f"  GBS:        {args.global_batch_size}")
    print(
        f"  搜索空间:   TP={args.tensor_parallel_sizes}, PP={args.pipeline_parallel_sizes}, "
        f"EP={args.expert_parallel_sizes}, MBS={args.micro_batch_sizes}"
    )

    # 生成配置（支持评分筛选）
    effective_max = max_configs or getattr(args, "max_configs", None)
    base_config, configs = generate_configs(
        runner, max_configs=effective_max, scoring_fn=scoring_fn
    )

    print(f"  候选配置:   {len(configs)} 个")
    print(f"  日志目录:   {args.log_dir}")
    if args.base_yaml:
        print(f"  基础 YAML:  {args.base_yaml}")
    print("=" * 70)

    # 模式 1: 仅生成配置
    if not args.run_number and not args.batch_mode and not args.get_results:
        print("\n  ✓ 配置生成完成")
        print("  使用 --run_number <N> 运行单个配置")
        print("  使用 --batch_mode 运行所有配置")
        return base_config, configs

    # 模式 4: 收集结果
    if args.get_results:
        print("\n  收集训练结果...")
        results = parse_training_logs(
            args.log_dir, configs, args.num_nodes, args.gpus_per_node
        )
        if results:
            print_results(results)
            csv_path = os.path.join(args.log_dir, "results_summary.csv")
            save_results_to_csv(results, csv_path)
            print(f"\n  ✓ 结果已保存到: {args.log_dir}")
        else:
            print(f"\n  ✗ 未能从日志中提取到性能数据，请检查: {args.log_dir}")
        return base_config, configs

    # 模式 2: 单次运行
    if args.run_number and not args.batch_mode:
        if args.run_number < 1 or args.run_number > len(configs):
            print(
                f"\n  ✗ 无效的 run_number: {args.run_number} (范围: 1-{len(configs)})"
            )
            return base_config, configs

        config_names = list(configs.keys())
        name = config_names[args.run_number - 1]
        config = configs[name]

        print(f"\n  运行配置 #{args.run_number}/{len(configs)}: {name}")
        if args.extra_metrics:
            print("  已启用额外指标监控")

        run_single_config(
            name,
            config,
            args.log_dir,
            args.base_yaml,
            args.num_nodes,
            args.gpus_per_node,
            args.global_batch_size,
            args.max_steps,
            dry_run=args.dry_run,
        )
        return base_config, configs

    # 模式 3: 批量运行
    if args.batch_mode:
        print("\n  批量运行所有配置...")
        total = len(configs)
        succeeded = 0
        failed_names = []

        for i, (name, config) in enumerate(configs.items(), 1):
            print(f"\n[{i}/{total}]", end="")
            ok = run_single_config(
                name,
                config,
                args.log_dir,
                args.base_yaml,
                args.num_nodes,
                args.gpus_per_node,
                args.global_batch_size,
                args.max_steps,
                dry_run=args.dry_run,
            )
            if ok:
                succeeded += 1
            else:
                failed_names.append(name)

        print(f"\n\n{'=' * 70}")
        print(
            f"  Benchmark 完成: {succeeded}/{total} 成功, {len(failed_names)} 失败"
        )
        if failed_names:
            print(f"  失败配置: {', '.join(failed_names)}")
        print(f"{'=' * 70}")

        # 自动收集结果
        print("\n  自动收集训练结果...")
        results = parse_training_logs(
            args.log_dir, configs, args.num_nodes, args.gpus_per_node
        )
        if results:
            print_results(results)
            csv_path = os.path.join(args.log_dir, "results_summary.csv")
            save_results_to_csv(results, csv_path)
            print(f"\n  ✓ 结果已保存到: {args.log_dir}")
        else:
            print(f"\n  ✗ 未能从日志中提取到性能数据，请检查: {args.log_dir}")

        return base_config, configs

    return base_config, configs


def main():
    """主入口函数."""
    args = get_args()
    train_config(args)


if __name__ == "__main__":
    main()
