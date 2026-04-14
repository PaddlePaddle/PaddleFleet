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

# Core API
from .autoconfigurator import (
    SUPPORTED_MODELS,
    AutoConfigurator,
    _apply_scoring_and_dedup,
    generate_configs,
)

# Core data types and utilities
from .core import (
    GeneratedConfig,
    ModelSizeParams,
    calculate_model_size,
    calculate_tflops,
    estimate_model_size,
    generate_grid_search_configs,
)

# Results
from .core.results import get_results

# Adapters
from .paddlefleet_adapters import (
    CombinedConfigAdapter,
    DataConfigAdapter,
    ModelConfigAdapter,
    PaddleFleetRecipe,
    ParallelStrategyAdapter,
    TrainingConfigAdapter,
    create_paddlefleet_adapter,
)

# Utils (backward-compatible re-exports)
from .utils import (
    GPT_BASED_MODELS,
    MODEL_PRESETS,
    _parse_size_list,
    build_launch_cmd,
    get_args,
    print_results,
    run_single_config,
    save_results_to_csv,
)

__all__ = [
    # Core API
    "SUPPORTED_MODELS",
    "AutoConfigurator",
    "GeneratedConfig",
    "PaddleFleetRecipe",
    "generate_configs",
    "estimate_model_size",
    "get_results",
    # Core data types and utilities
    "ModelSizeParams",
    "calculate_model_size",
    "calculate_tflops",
    "generate_grid_search_configs",
    # Adapters
    "CombinedConfigAdapter",
    "DataConfigAdapter",
    "ModelConfigAdapter",
    "ParallelStrategyAdapter",
    "TrainingConfigAdapter",
    "create_paddlefleet_adapter",
    # Utils
    "GPT_BASED_MODELS",
    "MODEL_PRESETS",
    "get_args",
    "build_launch_cmd",
    "run_single_config",
    "print_results",
    "save_results_to_csv",
    # Internal helpers (exported for testing / advanced use)
    "_apply_scoring_and_dedup",
    "_parse_size_list",
]
