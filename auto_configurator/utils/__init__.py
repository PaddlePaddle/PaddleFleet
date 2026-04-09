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
Utility modules for AutoConfigurator.

Contains CLI support, model presets, training runner, and results formatting.
"""

from .cli_args import _parse_size_list, get_args, load_args_from_yaml
from .model_presets import GPT_BASED_MODELS, MODEL_PRESETS, list_presets
from .results_formatter import (
    parse_training_logs,
    print_results,
    save_results_to_csv,
)
from .training_runner import build_launch_cmd, run_single_config

__all__ = [
    "GPT_BASED_MODELS",
    "MODEL_PRESETS",
    "list_presets",
    "get_args",
    "_parse_size_list",
    "load_args_from_yaml",
    "build_launch_cmd",
    "run_single_config",
    "print_results",
    "save_results_to_csv",
    "parse_training_logs",
]
