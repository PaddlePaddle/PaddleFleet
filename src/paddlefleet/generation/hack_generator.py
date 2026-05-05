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

"""Deprecated — use :mod:`paddlefleet.generation.greedy_generator` instead.

This module provided a Step 1 hack that used monkey-patching to inject KV cache
into FleetGPTModel.  Step 2 showed that PaddleFleet already has native KV cache
support wired through the entire stack, so the monkey-patching is unnecessary.

The public API is re-exported here for backward compatibility.
"""

import warnings

warnings.warn(
    "paddlefleet.generation.hack_generator is deprecated. "
    "Use paddlefleet.generation.GreedyGenerator instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .greedy_generator import DynamicKVCache, GreedyGenerator  # noqa: E402
from .inference_utils import init_inference_fleet  # noqa: E402

# Backward-compatible aliases
HackGreedyGenerator = GreedyGenerator

__all__ = [
    "DynamicKVCache",
    "HackGreedyGenerator",
    "GreedyGenerator",
    "init_inference_fleet",
]
