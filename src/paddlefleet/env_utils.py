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

# paddlefleet/env_utils.py

import os


def get_env_int(name: str, default: int = 0) -> int:
    """
    Get environment variable as int.

    Args:
        name (str): Environment variable name.
        default (int): Default value if not found or invalid.

    Returns:
        int: Parsed integer value.
    """
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def is_ci_env() -> bool:
    """
    Detect whether running in CI environment.

    Returns:
        bool: True if in CI.
    """
    ci_keys = ["CI", "GITHUB_ACTIONS", "GITLAB_CI"]
    return any(os.getenv(k) for k in ci_keys)


def get_env_str(name: str, default: str | None = None) -> str | None:
    """
    Get environment variable as string.

    Args:
        name (str): Environment variable name.
        default (Optional[str]): Default value.

    Returns:
        Optional[str]: Environment value.
    """
    return os.getenv(name, default)
