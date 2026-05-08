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

import sys as _sys

from paddlefleet_ops import ops as _ops_module
from paddlefleet_ops.ops import (  # noqa: F401
    HardwareIncompatibleBlocker,
    blocked_import_messages,
    is_deep_ep_available,
    is_deep_gemm_available,
    is_flash_mask_available,
    is_sonic_moe_available,
)

# Mirror the entire paddlefleet_ops.ops namespace into paddlefleet.ops so that
# attribute access (e.g. `paddlefleet.ops.deep_gemm`) and wildcard imports
# both work transparently.
_sys.modules[__name__] = _ops_module

# The HardwareIncompatibleBlocker installed by paddlefleet_ops.ops only
# catches ``import paddlefleet_ops.ops.*``.  Install a second blocker that
# intercepts ``import paddlefleet.ops.*`` so that old-namespace import
# attempts also raise the informative RuntimeError.
_compat_blocker_messages = {
    k.replace("paddlefleet_ops.ops.", "paddlefleet.ops."): v
    for k, v in blocked_import_messages.items()
    if k.startswith("paddlefleet_ops.ops.")
}
if _compat_blocker_messages:
    _sys.meta_path.insert(
        0, HardwareIncompatibleBlocker(_compat_blocker_messages)
    )
