#!/usr/bin/env python3

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

"""AutoConfigurator Search Entry - 自动搜索最优并行策略并 benchmark

从标准 PaddleFormers 训练 YAML 自动推断模型类型，
通过评分函数筛选 Top-N 配置后批量运行 benchmark。

用法:
  # 指定训练 YAML
  python auto_search.py utils/qwen3_moe_30b.yaml

  # 覆盖搜索空间
  python auto_search.py utils/qwen3_moe_30b.yaml --tensor_parallel_sizes 1,2 --expert_parallel_sizes 4,8

  # 覆盖训练/集群参数
  python auto_search.py utils/qwen3_moe_30b.yaml --max_steps 20 --max_configs 5
"""

import argparse
import os

from auto_configurator.core import GeneratedConfig
from auto_configurator.utils.cli_args import load_args_from_yaml

# ============================================================================
# Example scoring function for MoE on single-node 8 GPUs
# ============================================================================


def moe_scoring_fn(cfg: GeneratedConfig) -> float:
    """Heuristic scoring for MoE models on single-node setup.

    Prioritizes: high EP > low TP > PP=1 > reasonable MBS.
    Also penalizes configs that will OOM (too many experts per GPU).
    """
    total_gpus = 8
    num_experts = 128
    gpu_memory_gb = 80

    tp = cfg.tensor_parallel_size
    pp = cfg.pipeline_parallel_size
    ep = cfg.expert_parallel_size
    mbs = cfg.micro_batch_size

    mp = tp * pp * ep
    dp = total_gpus // mp if mp <= total_gpus else 0

    # Infeasible: model parallelism exceeds available GPUs
    if mp > total_gpus:
        return -1.0

    # OOM guard: estimate total memory per GPU including dense + MoE + optimizer
    dense_params_per_layer = (
        2048 * 2048 * 4  # Q/K/V/O projections (simplified)
        + 2048 * 6144 * 2  # up/down FFN
    )
    total_dense_params = 48 * dense_params_per_layer / tp
    experts_per_gpu = (
        num_experts // ep if num_experts % ep == 0 else num_experts
    )
    expert_params_per_gpu = experts_per_gpu * 3 * 2048 * 768  # gate/up/down
    total_params = (total_dense_params + expert_params_per_gpu) / pp
    param_memory_gb = total_params * 18 / (1024**3)
    activation_memory_gb = 10 * mbs / tp
    estimated_total_gb = param_memory_gb + activation_memory_gb
    if estimated_total_gb > gpu_memory_gb * 0.95:
        return -1.0

    s_dp = dp / total_gpus if dp > 0 else 0

    if num_experts % ep == 0:
        s_ep = ep / min(total_gpus, num_experts)
    else:
        s_ep = 0.1

    if pp == 1:
        s_pp = 1.0
    else:
        gbs = cfg.global_batch_size
        num_microbatches = gbs // (mbs * max(dp, 1)) if dp > 0 else 1
        bubble_ratio = (pp - 1) / (num_microbatches + pp - 1)
        s_pp = 1.0 - bubble_ratio

    s_tp = 1.0 / tp

    mbs_scores = {1: 0.5, 2: 1.0, 4: 0.85, 8: 0.6}
    s_mbs = mbs_scores.get(mbs, 0.3)

    total = 0.30 * s_dp + 0.25 * s_ep + 0.20 * s_pp + 0.15 * s_tp + 0.10 * s_mbs
    return round(total, 4)


# ============================================================================
# Main
# ============================================================================

DEFAULT_YAML = os.path.join(
    os.path.dirname(__file__), "utils", "qwen3_moe_30b.yaml"
)


def main():
    parser = argparse.ArgumentParser(
        description="AutoConfigurator Search - 自动搜索最优并行策略并 benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python auto_search.py utils/qwen3_moe_30b.yaml
  python auto_search.py path/to/model.yaml --tensor_parallel_sizes 1,2
  python auto_search.py path/to/model.yaml --max_configs 5 --dry_run
        """,
    )
    parser.add_argument(
        "base_yaml",
        type=str,
        nargs="?",
        default=DEFAULT_YAML,
        help="PaddleFormers 训练 YAML 路径 (default: utils/qwen3_moe_30b.yaml)",
    )
    # search space (optional overrides)
    parser.add_argument(
        "--tensor_parallel_sizes",
        type=str,
        help="TP 搜索空间，逗号分隔或 'auto'",
    )
    parser.add_argument(
        "--pipeline_parallel_sizes",
        type=str,
        help="PP 搜索空间，逗号分隔或 'auto'",
    )
    parser.add_argument(
        "--expert_parallel_sizes",
        type=str,
        help="EP 搜索空间，逗号分隔或 'auto'",
    )
    parser.add_argument(
        "--micro_batch_sizes", type=str, help="MBS 搜索空间，逗号分隔或 'auto'"
    )
    # training
    parser.add_argument("--global_batch_size", type=int, help="全局批大小")
    parser.add_argument("--max_steps", type=int, help="每配置最大步数")
    parser.add_argument(
        "--max_configs", type=int, default=3, help="Top-N 配置数 (default: 3)"
    )
    # cluster
    parser.add_argument("--num_nodes", type=int, help="节点数")
    parser.add_argument("--gpus_per_node", type=int, help="每节点 GPU 数")
    parser.add_argument("--gpu_memory", type=int, help="GPU 显存 GB")
    # output
    parser.add_argument("--log_dir", type=str, help="日志目录")
    parser.add_argument(
        "--dry_run", action="store_true", help="仅打印命令，不实际执行"
    )

    cli = parser.parse_args()

    # 只将用户显式传入的参数作为 overrides（跳过 None 和 base_yaml）
    overrides = {
        k: v for k, v in vars(cli).items() if k != "base_yaml" and v is not None
    }
    args = load_args_from_yaml(cli.base_yaml, **overrides)

    from auto_configurator.main import train_config

    train_config(args, scoring_fn=moe_scoring_fn, max_configs=cli.max_configs)


if __name__ == "__main__":
    main()
