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
Results aggregation for AutoConfigurator.

Provides functions to collect performance metrics from training logs,
calculate TFLOPS, and generate results summaries.
"""

from __future__ import annotations

import csv
import logging
import os

from .log_parser import parse_training_logs as _parse_log_dir
from .performance import calculate_tflops

logger = logging.getLogger(__name__)


def get_results(
    base_config: object,
    runner_config,
    path_to_save: str,
    output_top_n: int = 10,
):
    """Generate performance results from training logs.

    Parses training logs (if available) and generates a summary
    of performance metrics for each configuration.

    Args:
        base_config: Base configuration object
        runner_config: AutoConfigurator instance
        path_to_save: Path to save results
        output_top_n: Number of top configs to display

    Returns:
        None (results are saved to file)
    """
    logger.info(f"Generating results summary to {path_to_save}...")

    # Check if log directory exists
    if not os.path.exists(path_to_save):
        logger.warning(f"Log directory not found: {path_to_save}")
        return

    # List all subdirectories (each represents a config run)
    config_dirs = [
        d
        for d in os.listdir(path_to_save)
        if os.path.isdir(os.path.join(path_to_save, d))
    ]

    if not config_dirs:
        logger.warning("No configuration runs found in log directory")
        return

    # Generate results summary
    results = []
    for config_dir in config_dirs:
        # Parse config name for parameters
        config_name = config_dir
        params = _parse_config_name(config_name)

        # Try to find training metrics using core log parser
        metrics = _parse_log_dir(os.path.join(path_to_save, config_dir))
        if metrics:
            results.append({**params, **metrics})

    # Sort by training time (lower is better)
    results.sort(key=lambda x: x.get("time_per_step", float("inf")))

    # Display top results
    logger.info("=" * 60)
    logger.info("Top Performance Results")
    logger.info("=" * 60)
    for i, result in enumerate(results[:output_top_n]):
        logger.info(
            f"#{i + 1}: {result.get('config_name')} - "
            f"{result.get('time_per_step', 'N/A')}s/step, "
            f"{result.get('tflops_per_gpu', 'N/A')} TFLOPS/GPU"
        )

    # Save results to CSV
    _save_results_to_csv(results, path_to_save)

    logger.info(f"Results saved to {path_to_save}/results_summary.csv")


def _parse_config_name(config_name: str) -> dict:
    """Parse configuration name to extract parameters.

    Expected format: {model}_{size}b_{nodes}nodes_tp_{tp}_pp_{pp}_cp_{cp}_ep_{ep}_mbs_{mbs}

    Args:
        config_name: Configuration directory name

    Returns:
        Dictionary with extracted parameters
    """
    parts = config_name.split("_")

    return {
        "config_name": config_name,
        "tp": int(parts[parts.index("tp") + 1]) if "tp" in parts else 1,
        "pp": int(parts[parts.index("pp") + 1]) if "pp" in parts else 1,
        "cp": int(parts[parts.index("cp") + 1]) if "cp" in parts else 1,
        "ep": int(parts[parts.index("ep") + 1]) if "ep" in parts else 1,
        "mbs": int(parts[parts.index("mbs") + 1]) if "mbs" in parts else 1,
    }


def _calculate_tflops_from_metrics(metrics: dict, log_data: dict) -> dict:
    """Calculate TFLOPS from metrics and configuration.

    Args:
        metrics: Existing metrics dictionary
        log_data: Full log data for config access

    Returns:
        Updated metrics dictionary with TFLOPS
    """
    if "time_per_step" not in metrics:
        return metrics

    time_per_step = metrics["time_per_step"]

    # Try to use the formal TFLOPS formula if config params are available
    try:
        gbs = log_data.get("global_batch_size", 0)
        seq_len = log_data.get("seq_length", 0)
        hidden_size = log_data.get("hidden_size", 0)
        ffn_size = log_data.get("ffn_size", 0)
        num_layers = log_data.get("num_layers", 0)
        vocab = log_data.get("vocab_size", 0)
        num_nodes = log_data.get("num_nodes", 1)
        gpus_per_node = log_data.get("gpus_per_node", 8)
        model_name = log_data.get("model_name", "gpt")

        if all([gbs, seq_len, hidden_size, num_layers, vocab]):
            _, per_gpu_tflops = calculate_tflops(
                model_name=model_name,
                gbs=gbs,
                enc_seq_len=seq_len,
                dec_seq_len=seq_len,
                hidden_size=hidden_size,
                ffn_size=ffn_size or 4 * hidden_size,
                num_layers=num_layers,
                vocab=vocab,
                num_nodes=num_nodes,
                gpus_per_node=gpus_per_node,
                time_per_step=time_per_step,
            )
            metrics["tflops_per_gpu"] = per_gpu_tflops
            return metrics
    except Exception:
        pass

    # Fallback: approximate TFLOPS when config params are not available
    metrics["tflops_per_gpu"] = round(100.0 / time_per_step, 2)

    return metrics


def _save_results_to_csv(results: list, path: str):
    """Save results to CSV file.

    Args:
        results: List of result dictionaries
        path: Directory to save CSV
    """
    csv_path = os.path.join(path, "results_summary.csv")

    with open(csv_path, "w", newline="") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    logger.info(f"Results summary saved to {csv_path}")
