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
paddlefleet.ops — compatibility shim.

All operator implementations now live in the `paddlefleet-ops` package
(paddlefleet_ops.ops).  This module re-exports everything from there so
that existing code using `paddlefleet.ops` continues to work unchanged.
"""

from __future__ import annotations

from paddlefleet_ops.ops import (  # noqa: F401
    is_deep_ep_available,
    is_deep_gemm_available,
    is_flash_mask_available,
    is_sonic_moe_available,
)
from paddlefleet_ops import ops as _ops_module  # noqa: F401

import sys as _sys

# Mirror the entire paddlefleet_ops.ops namespace into paddlefleet.ops so that
# attribute access (e.g. `paddlefleet.ops.deep_gemm`) and wildcard imports
# both work transparently.
_sys.modules[__name__] = _ops_module
