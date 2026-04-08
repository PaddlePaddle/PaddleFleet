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
Results Formatter for AutoConfigurator

Provides output formatting and CSV export functionality for training results.
This module formats parsed metrics for display and saves to CSV.
"""

import csv
import os
from typing import Any

from auto_configurator.core.log_parser import (
    parse_training_logs as parse_single_log,
)


def print_results(results: list[dict[str, Any]]) -> None:
    """打印格式化的结果表格.

    Args:
        results: 结果列表，每个结果包含 tp/pp/ep/mbs/dp/time_per_step 等字段
    """
    print(f"\n{'=' * 85}")
    print("  性能排名 (按 time_per_step 升序, 越小越好)")
    print(f"{'=' * 85}")
    print(
        f"  {'排名':<4} {'TP':>2} {'PP':>2} {'EP':>3} {'MBS':>4} {'DP':>3}  "
        f"{'time/step(s)':>12}  {'tokens/s/GPU':>12}  {'samples/s':>10}"
    )
    print(
        f"  {'─' * 4} {'─' * 2} {'─' * 2} {'─' * 3} {'─' * 4} {'─' * 3}  "
        f"{'─' * 12}  {'─' * 12}  {'─' * 10}"
    )

    for rank, r in enumerate(results, 1):
        tps = (
            f"{r['time_per_step']:>12.2f}"
            if "time_per_step" in r
            else f"{'N/A':>12}"
        )
        tok = (
            f"{r['tokens_per_second_per_device']:>12.2f}"
            if "tokens_per_second_per_device" in r
            else f"{'N/A':>12}"
        )
        thr = (
            f"{r['throughput']:>10.4f}" if "throughput" in r else f"{'N/A':>10}"
        )
        print(
            f"  {rank:<4} {r['tp']:>2} {r['pp']:>2} {r['ep']:>3} {r['mbs']:>4} "
            f"{r['dp']:>3}  {tps}  {tok}  {thr}"
        )

    # 最优配置
    if results:
        best = results[0]
        print(f"\n{'=' * 85}")
        tok_info = (
            f", tokens/s/GPU={best['tokens_per_second_per_device']:.2f}"
            if "tokens_per_second_per_device" in best
            else ""
        )
        print(
            f"  ★ 最优配置: TP={best['tp']} PP={best['pp']} EP={best['ep']} "
            f"MBS={best['mbs']} (time/step={best.get('time_per_step', 'N/A')}s{tok_info})"
        )
        print(f"{'=' * 85}")


def save_results_to_csv(results: list[dict[str, Any]], csv_path: str) -> bool:
    """保存结果到 CSV.

    Args:
        results: 结果列表
        csv_path: CSV 文件路径

    Returns:
        是否成功
    """
    try:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        return True
    except Exception:
        return False


def parse_training_logs(
    base_log_dir: str,
    configs: dict[str, Any],
    num_nodes: int,
    num_gpus_per_node: int,
) -> list[dict[str, Any]]:
    """解析所有候选配置的训练日志（组合多个配置的日志解析）.

    Args:
        base_log_dir: 基础日志目录
        configs: 配置字典 {name: config}
        num_nodes: 节点数
        num_gpus_per_node: 每节点 GPU 数

    Returns:
        排序后的结果列表
    """
    results = []

    for name, config in configs.items():
        log_dir = os.path.join(base_log_dir, name)
        metrics = parse_single_log(log_dir)
        if metrics:
            total_gpus = num_nodes * num_gpus_per_node
            model_parallel = (
                config.tensor_parallel_size
                * config.pipeline_parallel_size
                * config.expert_parallel_size
            )
            dp_size = total_gpus // model_parallel

            results.append(
                {
                    "name": name,
                    "tp": config.tensor_parallel_size,
                    "pp": config.pipeline_parallel_size,
                    "ep": config.expert_parallel_size,
                    "mbs": config.micro_batch_size,
                    "dp": dp_size,
                    **metrics,
                }
            )

    results.sort(key=lambda x: x.get("time_per_step", float("inf")))
    return results
