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

"""Run Top-3 Qwen3-30B-A3B MoE configurations end-to-end.

Uses auto_search's MoE scoring function to select the top 3 parallel
strategy configurations for Qwen3-30B-A3B on 8×H800 80GB, then
benchmarks each one via paddleformers-cli train.

Usage:
  # Full benchmark (default: 20 steps per config)
  python run_top3_qwen30b.py

  # Dry run (only print commands, no execution)
  python run_top3_qwen30b.py --dry_run

  # Custom steps
  python run_top3_qwen30b.py --max_steps 10
"""

import os
import sys

# Ensure project paths are on PYTHONPATH
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_this_dir)
_paddleformers_root = os.path.join(
    os.path.dirname(_project_root), "PaddleFormers"
)
for p in [
    _project_root,
    os.path.join(_project_root, "src"),
    _paddleformers_root,
]:
    if p not in sys.path:
        sys.path.insert(0, p)

from auto_configurator.auto_search import moe_scoring_fn
from auto_configurator.utils.cli_args import load_args_from_yaml

# Use benchmark dataset
os.environ.setdefault("AUTOCONFIG_TRAIN_DATA", "/tmp/benchmark_train.jsonl")
os.environ.setdefault("AUTOCONFIG_EVAL_DATA", "/tmp/benchmark_train.jsonl")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Top-3 Qwen3-30B-A3B benchmark"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=20,
        help="Max steps per config (default: 20)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print commands, do not execute",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./auto_config_logs_qwen30b",
        help="Log output directory",
    )
    cli = parser.parse_args()

    base_yaml = os.path.join(
        os.path.dirname(__file__), "utils", "qwen3_moe_30b.yaml"
    )

    # Explicitly set search space for Qwen3-30B-A3B MoE on 8×H800.
    #
    # Why concrete values instead of "auto":
    #   load_args_from_yaml defaults to "auto" for TP/PP/EP/MBS, but
    #   _parse_size_list("auto") returns None which breaks grid_search
    #   (get_grid_search_params treats None != "auto" as explicit override).
    #
    # Why EP=8 only:
    #   Qwen3-30B-A3B has 128 experts. EP<=4 puts >=32 experts/GPU → OOM.
    #   Only EP=8 (16 experts/GPU) fits in 80GB memory.
    #
    # Why MBS=1 only:
    #   EP=8 + MBS=2 uses ~74-77GB; cross-entropy allocates extra ~9GB → OOM.
    #   MBS=1 (~70GB) is the only safe choice.
    # GBS=8 keeps gradient_accumulation_steps=8 (vs 128 at GBS=128),
    # which dramatically speeds up binpacking data preprocessing.
    args = load_args_from_yaml(
        base_yaml,
        tensor_parallel_sizes="1",
        pipeline_parallel_sizes="1",
        expert_parallel_sizes="8",
        micro_batch_sizes="1",
        global_batch_size=8,
        max_steps=cli.max_steps,
        max_configs=3,
        log_dir=cli.log_dir,
        dry_run=cli.dry_run,
        batch_mode=True,
    )

    from main import train_config

    train_config(args, scoring_fn=moe_scoring_fn, max_configs=3)


if __name__ == "__main__":
    main()
