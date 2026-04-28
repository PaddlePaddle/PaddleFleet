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
AutoConfigurator class and generate_configs API for PaddleFleet.

This module provides the main AutoConfigurator class for automatic
configuration generation for large model distributed training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

from .core import (
    GeneratedConfig,
    ModelSizeParams,
    calculate_model_size,
    estimate_model_size,
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
    "llama2": "Llama",
    "llama3": "Llama",
    "mixtral": "Mixtral",
    "mistral": "Mistral",
    "gemma": "Gemma",
    "qwen": "Qwen",
    "qwen2": "Qwen",
    "qwen3": "Qwen",
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
        tflops_per_gpu (Optional[int]): Estimated BF16 TFLOPS per GPU (default 989 for H100)
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
    tflops_per_gpu: int | None = 989  # H100 BF16 peak TFLOPS
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

        # Validate sequence length (must be a positive multiple of 1024, up to 1M)
        self.seq_length = model_config.get_seq_length()
        if (
            self.seq_length is None
            or self.seq_length < 1024
            or self.seq_length % 1024 != 0
            or self.seq_length > 1048576
        ):
            raise ValueError(
                f"seq_length {self.seq_length} not supported. "
                f"Must be a positive multiple of 1024 (up to 1048576)."
            )

        # Validate mode
        if self.mode not in ["pretrain", "finetune"]:
            raise ValueError(
                f"Mode must be 'pretrain' or 'finetune'. Got {self.mode}"
            )

        # Validate parallel config for finetune mode
        if self.mode == "finetune":
            if self.tensor_parallel_sizes is None or (
                self.tensor_parallel_sizes
                and "auto" in str(self.tensor_parallel_sizes)
            ):
                raise ValueError(
                    "tensor_parallel_sizes must be specified for finetune mode"
                )
            if self.pipeline_parallel_sizes is None or (
                self.pipeline_parallel_sizes
                and "auto" in str(self.pipeline_parallel_sizes)
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
    max_configs: int | None = None,
    scoring_fn: Callable[[GeneratedConfig], float] | None = None,
) -> tuple[object, dict[str, object]]:
    """Generate configurations for grid search.

    This is the main entry point for AutoConfigurator.
    It generates all candidate configurations based on the
    runner's parameters and validates them.

    When scoring_fn is provided, configs are scored, deduplicated by
    parallel strategy (TP, PP, CP, EP) keeping the best MBS per group,
    and optionally truncated to max_configs. When scoring_fn is None,
    all configs are returned unmodified (NeMo-aligned behavior).

    Args:
        runner: AutoConfigurator instance with all configuration
        max_configs: Maximum number of configs to return. Only takes
            effect when scoring_fn is provided. None means no limit.
        scoring_fn: Optional function that takes a GeneratedConfig and
            returns a float score (higher is better). When provided,
            enables scoring + diversity dedup + top-N selection.

    Returns:
        Tuple of (base_config, configs_dict) where:
        - base_config: The base configuration object
        - configs_dict: Dictionary mapping config names to generated configs

    Raises:
        ValueError: If max_configs is negative
    """
    if max_configs is not None and max_configs < 0:
        raise ValueError("max_configs must be a non-negative integer or None")

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

    # Apply scoring + diversity dedup + top-N when scoring_fn is provided
    if scoring_fn is not None:
        configs = _apply_scoring_and_dedup(configs, scoring_fn, max_configs)
        logger.info(
            f"After scoring and dedup: {len(configs)} configurations retained"
        )

    # Return base config and generated configs
    return runner.recipe.model_config, configs


def _apply_scoring_and_dedup(
    configs: dict[str, GeneratedConfig],
    scoring_fn: Callable[[GeneratedConfig], float],
    max_configs: int | None,
) -> dict[str, GeneratedConfig]:
    """Score configs, apply diversity dedup, and return top-N.

    Diversity dedup groups configs by (TP, PP, CP, EP) and keeps only
    the highest-scoring MBS variant per group, ensuring the returned
    configs cover diverse parallel strategies.

    Args:
        configs: Dictionary of config name -> GeneratedConfig
        scoring_fn: Function that takes GeneratedConfig and returns a
            float score (higher is better)
        max_configs: Max number of configs to return; None means no limit

    Returns:
        Filtered dictionary of config name -> GeneratedConfig, ordered
        by score descending
    """
    if not configs:
        return {}

    # Step 1: Score every config
    scored: list[tuple[str, GeneratedConfig, float]] = []
    for name, config in configs.items():
        score = scoring_fn(config)
        scored.append((name, config, score))

    # Step 2: Diversity dedup — group by (TP, PP, CP, EP), keep best MBS
    best_per_group: dict[
        tuple[int, int, int, int], tuple[str, GeneratedConfig, float]
    ] = {}
    for name, config, score in scored:
        key = (
            config.tensor_parallel_size,
            config.pipeline_parallel_size,
            config.context_parallel_size,
            config.expert_parallel_size,
        )
        if key not in best_per_group or score > best_per_group[key][2]:
            best_per_group[key] = (name, config, score)

    # Step 3: Sort by score descending
    deduped = sorted(best_per_group.values(), key=lambda x: x[2], reverse=True)

    # Step 4: Apply max_configs limit
    if max_configs is not None and max_configs > 0:
        deduped = deduped[:max_configs]

    # Step 5: Rebuild ordered dict
    return {name: config for name, config, _score in deduped}


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
