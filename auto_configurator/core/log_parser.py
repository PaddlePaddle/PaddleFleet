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
Training Log Parser for AutoConfigurator

Provides core parsing logic for extracting metrics from training logs.
This module contains pure parsing functions that return structured data
without any output formatting or side effects.
"""

import json
import os
import re


def parse_training_logs(
    log_dir: str,
    log_file_patterns: list[str] | None = None,
) -> dict[str, float]:
    """从训练日志目录中提取性能指标（纯解析，无输出）.

    Args:
        log_dir: 日志目录路径
        log_file_patterns: 日志文件名模式列表，默认为常见模式

    Returns:
        指标字典，未找到返回空字典
    """
    if not os.path.isdir(log_dir):
        return {}

    # 默认日志文件模式
    if log_file_patterns is None:
        log_file_patterns = ["trainer_log.jsonl", "train.log", "stdout.log"]

    # 收集日志文件
    log_files = []
    for fname in log_file_patterns:
        fpath = os.path.join(log_dir, fname)
        if os.path.exists(fpath):
            log_files.append(fpath)

    # 递归搜索
    for root, _, files in os.walk(log_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            if fpath not in log_files and fname.endswith((".log", ".jsonl")):
                log_files.append(fpath)

    # 解析每个日志文件
    for fpath in log_files:
        try:
            with open(fpath, "r") as f:
                content = f.read()
        except Exception:
            continue

        # Strip ANSI escape codes for reliable matching
        content_clean = re.sub(r"\x1b\[[0-9;]*m", "", content)

        # JSONL 格式
        if fpath.endswith(".jsonl"):
            time_values = []
            for line in content_clean.strip().split("\n"):
                try:
                    entry = json.loads(line)
                    if "train_step_time" in entry:
                        time_values.append(entry["train_step_time"])
                except (json.JSONDecodeError, KeyError):
                    continue
            if time_values:
                # Skip warmup: discard first 1/3 of samples
                skip = max(len(time_values) // 3, 1)
                stable = time_values[skip:]
                if stable:
                    return {
                        "time_per_step": round(sum(stable) / len(stable), 4)
                    }

        # PaddleFormers log format: extract per-step metrics
        # Format: interval_runtime: 162.57, interval_tokens_per_second_per_device: 6449.99, ...
        runtime_values = []
        tokens_per_s_values = []
        samples_per_s_values = []
        for line in content_clean.split("\n"):
            rt_m = re.search(r"interval_runtime:\s*([0-9.]+)", line)
            tk_m = re.search(
                r"interval_tokens_per_second_per_device:\s*([0-9.]+)", line
            )
            sp_m = re.search(r"interval_samples_per_second:\s*([0-9.]+)", line)
            if rt_m:
                runtime_values.append(float(rt_m.group(1)))
            if tk_m:
                tokens_per_s_values.append(float(tk_m.group(1)))
            if sp_m:
                samples_per_s_values.append(float(sp_m.group(1)))

        if runtime_values:
            # Skip warmup: discard first 1/3 of samples
            skip = max(len(runtime_values) // 3, 1)
            stable_rt = runtime_values[skip:]
            stable_tk = (
                tokens_per_s_values[skip:] if tokens_per_s_values else []
            )
            stable_sp = (
                samples_per_s_values[skip:] if samples_per_s_values else []
            )

            metrics = {}
            if stable_rt:
                metrics["time_per_step"] = round(
                    sum(stable_rt) / len(stable_rt), 4
                )
            if stable_tk:
                metrics["tokens_per_second_per_device"] = round(
                    sum(stable_tk) / len(stable_tk), 4
                )
            if stable_sp:
                metrics["throughput"] = round(
                    sum(stable_sp) / len(stable_sp), 4
                )
            if metrics:
                return metrics

        # Fallback: legacy text log patterns
        for pattern, key in [
            (r"train_step_time[:\s=]+([0-9.]+)", "time_per_step"),
            (r"time_per_step[:\s=]+([0-9.]+)", "time_per_step"),
            (r"ips[:\s=]+([0-9.]+)", "throughput"),
            (r"throughput[:\s=]+([0-9.]+)", "throughput"),
        ]:
            m = re.search(pattern, content_clean, re.IGNORECASE)
            if m and key not in locals():
                return {key: round(float(m.group(1)), 4)}

    return {}
