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
Core modules for AutoConfigurator in PaddleFleet.

This module contains:
- Model size calculation and estimation
- Architecture parameter inference
- Grid search generation for parallel strategies (GPT only)
- TFLOPS calculation
- Training log parsing
- Results aggregation
"""

from .grid_search import (
    GeneratedConfig,
    GPTGridSearch,
    GridSearchConfig,
    generate_grid_search_configs,
    get_grid_search_params,
)
from .log_parser import parse_training_logs
from .model_size import (
    GPT_BASED_MODELS,
    ModelSizeParams,
    calculate_model_size,
    estimate_model_size,
)
from .performance import calculate_tflops
from .results import get_results

__all__ = [
    # From model_size
    "GPT_BASED_MODELS",
    "ModelSizeParams",
    "calculate_model_size",
    "estimate_model_size",
    # From grid_search
    "GeneratedConfig",
    "GPTGridSearch",
    "GridSearchConfig",
    "generate_grid_search_configs",
    "get_grid_search_params",
    # From performance
    "calculate_tflops",
    # From log_parser
    "parse_training_logs",
    # From results
    "get_results",
]
