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
Training Runner for AutoConfigurator

Contains functions for building and running training commands with different
parallel strategies.
"""

import os
import subprocess
import time
from typing import Any


def build_launch_cmd(
    config: Any,
    log_dir: str,
    base_yaml: str | None,
    num_nodes: int,
    num_gpus_per_node: int,
    global_batch_size: int,
    max_steps: int,
) -> str:
    """构建训练命令.

    Args:
        config: 生成的并行配置对象
        log_dir: 日志输出目录
        base_yaml: 基础训练 YAML 路径
        num_nodes: 节点数
        num_gpus_per_node: 每节点 GPU 数
        global_batch_size: 全局批大小
        max_steps: 最大步数

    Returns:
        完整的训练命令字符串
    """
    # 集群信息（从环境变量获取）
    nnodes = os.environ.get("NNODES", str(num_nodes))
    rank = os.environ.get("RANK", "0")
    master = os.environ.get("MASTER_ADDR", "127.0.0.1")
    port = os.environ.get("MASTER_PORT", "36677")

    use_ep = "true" if config.expert_parallel_size > 1 else "false"

    # 数据集路径（从环境变量获取）
    train_data = os.environ.get("AUTOCONFIG_TRAIN_DATA", "")
    eval_data = os.environ.get("AUTOCONFIG_EVAL_DATA", "")

    # 构建 paddleformers-cli train 命令
    # 如果没有提供 base_yaml，这里只打印配置信息
    if base_yaml is None:
        return (
            f"# No base_yaml provided. "
            f"Config: TP={config.tensor_parallel_size} "
            f"PP={config.pipeline_parallel_size} "
            f"EP={config.expert_parallel_size} "
            f"MBS={config.micro_batch_size}"
        )

    overrides = [
        f"tensor_model_parallel_size={config.tensor_parallel_size}",
        f"pipeline_model_parallel_size={config.pipeline_parallel_size}",
        f"expert_model_parallel_size={config.expert_parallel_size}",
        f"use_expert_parallel={use_ep}",
        f"per_device_train_batch_size={config.micro_batch_size}",
        f"max_steps={max_steps}",
        f"logging_dir={log_dir}",
        f"output_dir={log_dir}/ckpt",
        "save_steps=999999",
        "do_eval=false",
        "benchmark=true",
        "using_sonic_moe=false",
    ]

    # EP + TP > 1 时必须启用 sequence_parallel
    if config.tensor_parallel_size > 1 and config.expert_parallel_size > 1:
        overrides.append("sequence_parallel=true")

    # 添加数据集路径（如果已设置）
    if train_data:
        overrides.append(f"train_dataset_path={train_data}")
    if eval_data:
        overrides.append(f"eval_dataset_path={eval_data}")

    # 计算梯度累积步数
    total_gpus = num_nodes * num_gpus_per_node
    model_parallel = (
        config.tensor_parallel_size
        * config.pipeline_parallel_size
        * config.expert_parallel_size
    )
    dp_size = max(total_gpus // model_parallel, 1)
    grad_accum = max(
        global_batch_size // (config.micro_batch_size * dp_size), 1
    )
    overrides.append(f"gradient_accumulation_steps={grad_accum}")

    # 对于 benchmark 模式，覆盖 num_samples_each_epoch 以避免小数据集无限循环
    # 只需要 max_steps * grad_accum * mbs * dp_size 个样本
    num_samples = max_steps * grad_accum * config.micro_batch_size * dp_size * 2
    overrides.append(f"num_samples_each_epoch={num_samples}")

    cmd = (
        f"NNODES={nnodes} RANK={rank} "
        f"MASTER_ADDR={master} MASTER_PORT={port} "
        f"CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "
        f"paddleformers-cli train {base_yaml} " + " ".join(overrides)
    )
    return cmd


def run_single_config(
    name: str,
    config: Any,
    base_log_dir: str,
    base_yaml: str | None,
    num_nodes: int,
    num_gpus_per_node: int,
    global_batch_size: int,
    max_steps: int,
    timeout: int = 1800,
    dry_run: bool = False,
) -> bool:
    """运行单个候选配置.

    Args:
        name: 配置名称
        config: 并行配置对象
        base_log_dir: 基础日志目录
        base_yaml: 基础训练 YAML 路径
        num_nodes: 节点数
        num_gpus_per_node: 每节点 GPU 数
        global_batch_size: 全局批大小
        max_steps: 最大步数
        timeout: 超时时间（秒）
        dry_run: 仅打印命令，不执行

    Returns:
        是否成功
    """
    log_dir = os.path.join(base_log_dir, name)
    os.makedirs(log_dir, exist_ok=True)

    cmd = build_launch_cmd(
        config,
        log_dir,
        base_yaml,
        num_nodes,
        num_gpus_per_node,
        global_batch_size,
        max_steps,
    )

    print(f"\n{'─' * 70}")
    print(f"  配置: {name}")
    print(
        f"  TP={config.tensor_parallel_size}  PP={config.pipeline_parallel_size}  "
        f"EP={config.expert_parallel_size}  MBS={config.micro_batch_size}"
    )
    if base_yaml:
        print(f"  命令: {cmd[:120]}...")
    else:
        print(f"  {cmd}")
    print(f"{'─' * 70}")

    if dry_run:
        return True

    if not base_yaml:
        print("  ⚠ 跳过: 未指定 --base_yaml")
        return False

    t0 = time.time()
    try:
        proc = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Kill entire process group to avoid orphaned GPU workers
            import signal

            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()
            print(f"  ✗ 超时 (>{timeout}s)")
            return False

        elapsed = time.time() - t0

        # 保存日志
        with open(os.path.join(log_dir, "stdout.log"), "w") as f:
            f.write(stdout)
        if stderr:
            with open(os.path.join(log_dir, "stderr.log"), "w") as f:
                f.write(stderr)

        if proc.returncode == 0:
            print(f"  ✓ 完成 ({elapsed:.0f}s)")
            return True
        else:
            print(f"  ✗ 失败 (exit={proc.returncode}, {elapsed:.0f}s)")
            return False

    except Exception as e:
        print(f"  ✗ 异常: {e}")
        return False
