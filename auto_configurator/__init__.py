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
AutoConfigurator for PaddleFleet.

Migrated and adapted from NVIDIA NeMo's AutoConfigurator to support
PaddlePaddle's PaddleFleet distributed training framework.

This module provides automatic configuration generation for large model training,
including model architecture inference and grid search for optimal parallel strategies.
"""

from __future__ import annotations

import csv
import glob
import logging
import os
import re
from dataclasses import dataclass

from .core import (
    ModelSizeParams,
    calculate_model_size,
    generate_grid_search_configs,
)
from .paddlefleet_adapters import (
    CombinedConfigAdapter,
    DataConfigAdapter,
    ModelConfigAdapter,
    PaddleFleetRecipe,
    ParallelStrategyAdapter,
    TrainingConfigAdapter,
    create_paddlefleet_adapter,
)

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# Supported Model Types
# Note: T5/mT5 and BERT models are currently not supported in PaddleFleet

SUPPORTED_MODELS = {
    "gpt": "GPT",
    "llama": "Llama",
    "mixtral": "Mixtral",
    "mistral": "Mistral",
    "gemma": "Gemma",
    "qwen": "Qwen",
    "glm": "GLM",
}

# ============================================================================
# AutoConfigurator Main Class
# ============================================================================


@dataclass
class AutoConfigurator:
    """Auto Configurator runner config class for PaddleFleet.

    This class manages the automatic configuration generation for large model
    training in PaddleFleet, including:
    - Model architecture parameter inference
    - Grid search for parallel strategies
    - Configuration validation

    Args:
        recipe (PaddleFleetRecipe): Recipe containing model and training configs
        path_to_logs (str): Path to directory for saving logs
        mode (Optional[str]): 'pretrain' or 'finetune' recipe mode
        gpu_memory_gb (Optional[int]): Memory per GPU in GB (40 or 80)
        tensor_parallel_sizes (Optional[List[int]]): TP sizes to search, or "auto"
        pipeline_parallel_sizes (Optional[List[int]]): PP sizes to search, or "auto"
        micro_batch_sizes (Optional[List[int]]): MBS sizes to search, or "auto"
        context_parallel_sizes (Optional[List[int]]): CP sizes to search
        expert_parallel_sizes (Optional[List[int]]): EP sizes to search
        min_model_parallel_size (Optional[int]): Min desired parallelism
        max_model_parallel_size (Optional[int]): Max desired parallelism
        num_tokens_in_b (Optional[int]): Number of tokens in billions in dataset
        tflops_per_gpu (Optional[int]): Estimated TFLOPS per GPU
        max_minutes_per_run (Optional[int]): Max minutes per grid search run
        max_training_days (Optional[int]): Expected training days
        max_steps_per_run (Optional[int]): Max steps per grid search run
        vocab_size (Optional[int]): Tokenizer vocabulary size
        calculate_model_size (Optional[bool]): Auto-calculate model architecture
    """

    recipe: PaddleFleetRecipe
    path_to_logs: str

    # Mode
    mode: str | None = "pretrain"

    # Hardware constraints
    gpu_memory_gb: int | None = 80
    tensor_parallel_sizes: list[int] | None = None
    pipeline_parallel_sizes: list[int] | None = None
    micro_batch_sizes: list[int] | None = None
    context_parallel_sizes: list[int] | None = None
    expert_parallel_sizes: list[int] | None = None
    min_model_parallel_size: int | None = None
    max_model_parallel_size: int | None = None

    # Training constraints
    num_tokens_in_b: int | None = 1400
    tflops_per_gpu: int | None = 140
    max_minutes_per_run: int | None = 30
    max_training_days: int | None = 2
    max_steps_per_run: int | None = 50
    vocab_size: int | None = 32000

    # Model calculation
    calculate_model_size: bool | None = False

    # Internal state (set during initialization)
    model_type: str = None
    model_size_in_b: float = None
    gpu_count: int = None
    seq_length: int = None
    global_batch_size: int = None

    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Create adapter for unified config access
        self.adapter = create_paddlefleet_adapter(self.recipe)

        # Extract core parameters
        model_config = self.adapter.get_model_config()
        parallel_config = self.adapter.get_parallel_strategy()
        data_config = self.adapter.get_data_config()
        training_config = self.adapter.get_training_config()

        # Get model type
        self.model_type = model_config.get_model_type()

        # Validate model type
        if self.model_type not in SUPPORTED_MODELS:
            raise ValueError(
                f"Model type '{self.model_type}' not supported. "
                f"Supported: {list(SUPPORTED_MODELS.keys())}"
            )

        # Validate sequence length
        self.seq_length = model_config.get_seq_length()
        if self.seq_length not in [2048, 4096, 8192, 16384, 32768]:
            raise ValueError(
                f"seq_length {self.seq_length} not supported. "
                f"Supported: [2048, 4096, 8192, 16384, 32768]"
            )

        # Validate mode
        if self.mode not in ["pretrain", "finetune"]:
            raise ValueError(
                f"Mode must be 'pretrain' or 'finetune'. Got {self.mode}"
            )

        # Validate parallel config for finetune mode
        if self.mode == "finetune":
            if (
                self.tensor_parallel_sizes is None
                or "auto" in str(self.tensor_parallel_sizes)
                if self.tensor_parallel_sizes
                else []
            ):
                raise ValueError(
                    "tensor_parallel_sizes must be specified for finetune mode"
                )
            if (
                self.pipeline_parallel_sizes is None
                or "auto" in str(self.pipeline_parallel_sizes)
                if self.pipeline_parallel_sizes
                else []
            ):
                raise ValueError(
                    "pipeline_parallel_sizes must be specified for finetune mode"
                )
            if self.calculate_model_size:
                raise ValueError(
                    "calculate_model_size not supported for finetune mode"
                )

        # Validate hardware constraints
        if self.gpu_memory_gb not in [40, 80]:
            raise ValueError(
                f"gpu_memory_gb must be 40 or 80. Got {self.gpu_memory_gb}"
            )

        # Set GPU count
        self.gpu_count = (
            training_config.get_num_nodes()
            * training_config.get_num_gpus_per_node()
        )
        if self.gpu_count <= 0:
            raise ValueError("num_nodes * gpus_per_node must be > 0")

        # Validate training constraints
        if self.num_tokens_in_b is not None and self.num_tokens_in_b <= 0:
            raise ValueError("num_tokens_in_b must be > 0")
        if self.tflops_per_gpu is not None and self.tflops_per_gpu <= 0:
            raise ValueError("tflops_per_gpu must be > 0")
        if (
            self.max_minutes_per_run is not None
            and self.max_minutes_per_run < 10
        ):
            raise ValueError("max_minutes_per_run must be >= 10")
        if self.max_steps_per_run is not None and self.max_steps_per_run < 10:
            raise ValueError("max_steps_per_run must be >= 10")

        # Validate context and expert parallel sizes
        if (
            self.context_parallel_sizes is not None
            and len(self.context_parallel_sizes) > 0
        ):
            cp_str = str(self.context_parallel_sizes[0])
            if "auto" in cp_str.lower():
                raise ValueError(
                    "'auto' not supported for context_parallel_sizes"
                )
        if (
            self.expert_parallel_sizes is not None
            and len(self.expert_parallel_sizes) > 0
        ):
            ep_str = str(self.expert_parallel_sizes[0])
            if "auto" in ep_str.lower():
                raise ValueError(
                    "'auto' not supported for expert_parallel_sizes"
                )

        # Get batch size
        self.global_batch_size = data_config.get_global_batch_size()

        # Log configuration
        logger.info("=" * 60)
        logger.info("AutoConfigurator for PaddleFleet Configuration")
        logger.info("=" * 60)
        logger.info(f"Model type: {self.model_type}")
        logger.info(f"Mode: {self.mode}")
        logger.info(f"GPU memory: {self.gpu_memory_gb}GB")
        logger.info(
            f"GPU count: {self.gpu_count} ({self.gpu_count // training_config.get_num_nodes()} nodes * {training_config.get_num_gpus_per_node()} GPUs)"
        )
        logger.info(f"Seq length: {self.seq_length}")
        logger.info(f"Global batch size: {self.global_batch_size}")
        logger.info(f"Max steps per run: {self.max_steps_per_run}")
        logger.info(f"Max training days: {self.max_training_days}")
        logger.info("=" * 60)

    def get_adapter(self) -> CombinedConfigAdapter:
        """Get combined configuration adapter."""
        return self.adapter

    def get_model_config(self) -> ModelConfigAdapter:
        """Get model configuration adapter."""
        return self.adapter.get_model_config()

    def get_parallel_config(self) -> ParallelStrategyAdapter:
        """Get parallel strategy adapter."""
        return self.adapter.get_parallel_strategy()

    def get_data_config(self) -> DataConfigAdapter:
        """Get data configuration adapter."""
        return self.adapter.get_data_config()

    def get_training_config(self) -> TrainingConfigAdapter:
        """Get training configuration adapter."""
        return self.adapter.get_training_config()


# ============================================================================
# Public API Functions
# ============================================================================


def generate_configs(
    runner: AutoConfigurator,
) -> tuple[object, dict[str, object]]:
    """Generate configurations for grid search.

    This is the main entry point for AutoConfigurator.
    It generates all candidate configurations based on the
    runner's parameters and validates them.

    Args:
        runner: AutoConfigurator instance with all configuration

    Returns:
        Tuple of (base_config, configs_dict) where:
        - base_config: The base configuration object
        - configs_dict: Dictionary mapping config names to generated configs
    """
    logger.info("Generating grid search configurations...")

    # Apply model size calculation if enabled
    if runner.calculate_model_size:
        _apply_model_size_calculation(runner)
    else:
        runner.model_size_in_b = _extract_model_size_from_config(runner)

    configs = generate_grid_search_configs(
        runner_config=runner,
        adapter=runner.adapter,
    )

    logger.info(f"Generated {len(configs)} candidate configurations")

    # Return base config and generated configs
    return runner.recipe.model_config, configs


def _apply_model_size_calculation(runner: AutoConfigurator):
    """Calculate optimal model architecture based on training constraints.

    Uses estimate_model_size to calculate model size that fits
    training constraints, then infers architecture parameters.

    Args:
        runner: AutoConfigurator instance
    """
    # Calculate model size based on constraints
    runner.model_size_in_b = estimate_model_size(
        gpu_count=runner.gpu_count,
        max_training_days=runner.max_training_days,
        model_size_in_b=None,
        tflops_per_gpu=runner.tflops_per_gpu,
        num_tokens_in_b=runner.num_tokens_in_b,
        model_name=runner.model_type,
    )

    # Infer architecture parameters
    model_config = runner.adapter.get_model_config()
    params = ModelSizeParams(
        model_size_in_b=runner.model_size_in_b,
        vocab_size=runner.vocab_size,
        seq_length=runner.seq_length,
        model_name=runner.model_type,
    )
    params.init_params()

    # Apply inferred parameters to config
    model_config.set_num_layers(params.layers)
    model_config.set_hidden_size(params.hidden_size)
    model_config.set_num_attention_heads(params.num_attention_heads)
    model_config.set_ffn_hidden_size(params.ffn_size)

    logger.info(
        f"Applied model architecture: {params.layers} layers, "
        f"hidden_size={params.hidden_size}, "
        f"num_attention_heads={params.num_attention_heads}, "
        f"ffn_size={params.ffn_size}"
    )


def _extract_model_size_from_config(runner: AutoConfigurator) -> float:
    """Extract model size from existing config if not calculating.

    Args:
        runner: AutoConfigurator instance

    Returns:
        Model size in billions
    """
    model_config = runner.adapter.get_model_config()

    # Try to calculate from config parameters
    try:
        size = calculate_model_size(
            vocab_size=model_config.get_vocab_size(),
            seq_length=model_config.get_seq_length(),
            hidden_size=model_config.get_hidden_size(),
            num_layers=model_config.get_num_layers(),
            ffn_size=model_config.get_ffn_hidden_size(),
            model_name=runner.model_type,
        )
        return size
    except Exception:
        # Return default if calculation fails
        return 1.0


def estimate_model_size(
    gpu_count: int,
    max_training_days: float,
    model_size_in_b: float | None = None,
    tflops_per_gpu: int = 140,
    num_tokens_in_b: int = 300,
    model_name: str = "gpt",
) -> float:
    """Estimates model size to train given constraints.

    If model_size is provided, estimates the time to train it.
    If not provided, estimates the model size that can be trained.

    Args:
        gpu_count: Number of GPUs to use
        max_training_days: Number of days to train
        model_size_in_b: Model size in billions (if known)
        tflops_per_gpu: Estimated TFLOPS per GPU
        num_tokens_in_b: Number of tokens in dataset (billions)
        model_name: Model type for penalty adjustment (unused - kept for API compatibility)

    Returns:
        Estimated model size in billions of parameters
    """
    # Calculate model size if not provided
    if model_size_in_b is None:
        model_size_in_b = (
            (max_training_days * 3600 * 24 * gpu_count * tflops_per_gpu * 1e12)
            / (8 * num_tokens_in_b * 1e9)
            / 1e9
        )
        model_size_in_b = round(model_size_in_b, 2)
    else:
        # Calculate training time if model size is provided
        max_training_days = (
            model_size_in_b * 1e9 * 8 * num_tokens_in_b * 1e9
        ) / (3600 * 24 * gpu_count * tflops_per_gpu * 1e12)
        max_training_days = round(max_training_days, 2)

    logger.info(
        f"You can train a {model_size_in_b}B parameter model in "
        f"{max_training_days} days using {gpu_count} GPUs. "
        f"Assuming {num_tokens_in_b}B tokens, {tflops_per_gpu} TFLOPS/GPU."
    )

    return model_size_in_b


def get_results(
    base_config: object,
    runner_config: AutoConfigurator,
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

        # Try to find training metrics
        metrics = _extract_metrics(os.path.join(path_to_save, config_dir))
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


def _extract_metrics(config_dir: str) -> dict:
    """Extract training metrics from configuration directory.

    Args:
        config_dir: Path to configuration directory

    Returns:
        Dictionary with extracted metrics
    """
    metrics = {}

    # Look for common log files
    for log_file in ["train.log", "metrics.json", "events.out.tfevents.*"]:
        log_path = os.path.join(config_dir, log_file)
        # Check for wildcard pattern
        if "*" in log_file:
            matches = glob.glob(log_path)
            if matches:
                log_path = matches[0]
        if os.path.exists(log_path):
            # Parse log file based on type
            if log_file.endswith(".json"):
                metrics.update(_parse_json_log(log_path))
            elif log_file.endswith(".log"):
                metrics.update(_parse_text_log(log_path))
            break

    return metrics


def _parse_json_log(log_path: str) -> dict:
    """Parse JSON log file for metrics.

    Args:
        log_path: Path to JSON log file

    Returns:
        Dictionary with extracted metrics
    """
    import json

    with open(log_path, "r") as f:
        data = json.load(f)

    # Extract common metrics
    metrics = {}
    if "time_per_step" in data:
        metrics["time_per_step"] = data["time_per_step"]
    if "throughput" in data:
        metrics["throughput"] = data["throughput"]

    # Calculate TFLOPS if enough info available
    if metrics:
        metrics = _calculate_tflops_from_metrics(metrics, data)

    return metrics


def _parse_text_log(log_path: str) -> dict:
    """Parse text log file for metrics.

    Args:
        log_path: Path to text log file

    Returns:
        Dictionary with extracted metrics
    """
    metrics = {}

    # Simple parsing - look for common patterns
    with open(log_path, "r") as f:
        content = f.read()

    # Extract timing information
    time_match = re.search(r"time_per_step[:\s]+([0-9.]+)", content)
    if time_match:
        metrics["time_per_step"] = float(time_match.group(1))

    # Extract throughput
    throughput_match = re.search(r"throughput[:\s]+([0-9.]+)", content)
    if throughput_match:
        metrics["throughput"] = float(throughput_match.group(1))

    return metrics


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

    # Get config parameters
    # This would need access to full config, simplified here
    time_per_step = metrics["time_per_step"]

    # Calculate approximate TFLOPS (simplified)
    # In real implementation, use calculate_tflops with full parameters
    metrics["tflops_per_gpu"] = 100.0 / time_per_step  # Simplified

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


__all__ = [
    "AutoConfigurator",
    "PaddleFleetRecipe",
    "generate_configs",
    "estimate_model_size",
    "get_results",
]
