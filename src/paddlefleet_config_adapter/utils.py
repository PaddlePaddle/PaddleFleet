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

"""Small helpers shared across the config adapter."""

from __future__ import annotations

import math

# YAML key of every parallel dimension the adapter reasons about.
PARALLEL_FIELDS = {
    "tp": "tensor_model_parallel_size",
    "pp": "pipeline_model_parallel_size",
    "ep": "expert_model_parallel_size",
    "cp": "context_parallel_size",
    "sep": "sep_parallel_size",
}


def lcm(a, b):
    """Least common multiple of two integers."""
    return abs(a * b) // math.gcd(a, b)


def multi_lcm(*args):
    """Least common multiple of an arbitrary number of integers."""
    result = args[0]
    for x in args[1:]:
        result = lcm(result, x)
    return result


def parse_value(value_str):
    """Infer a Python value from a CLI string.

    Supports bool / None / int / float, falling back to the raw string.
    """
    lowered = value_str.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in ("null", "none"):
        return None
    try:
        return int(value_str)
    except ValueError:
        pass
    try:
        return float(value_str)
    except ValueError:
        pass
    return value_str


def extract_parallel_params(config):
    """Return ``(tp, pp, ep, cp, sep)`` read from a training YAML.

    Missing / commented-out / null values default to 1.
    """
    dims = []
    for key in ("tp", "pp", "ep", "cp", "sep"):
        raw = config.get(PARALLEL_FIELDS[key], 1)
        dims.append(max(int(raw), 1) if raw is not None else 1)
    return tuple(dims)
