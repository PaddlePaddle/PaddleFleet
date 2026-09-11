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

"""Fine-grained activation offloading.

In the forward pass, activations saved inside a marked region are copied
asynchronously to pinned host memory and released from device memory. In the
backward pass they are prefetched back, driven by the pipeline schedule. This
trades D2H/H2D bandwidth for device memory.

A region name maps to the execution span of one module, not to a single tensor:
every activation that region saves for backward is offloaded. Model code marks
the region around the module itself::

    from paddlefleet.activation_offload import offload_groups, offload_region

    # in __init__
    self.offload_qkv_linear = "qkv_linear" in offload_groups(config)

    # in forward
    with offload_region(self.offload_qkv_linear, "qkv_linear"):
        mixed_qkv, _ = self.qkv_proj(hidden_states)

The region must wrap the module rather than tag its output. What backward needs
is what the module consumes -- its input for a linear, the q/k/v and softmax
intermediates for an attention -- whereas a module's output is the input of the
next module and belongs to the next region.

The set of recognized region names comes from the config
(``offload_modules``); leaving it unset offloads every supported region.

Under pipeline parallelism, register the schedule hooks once after the model is
wrapped and close every accumulation step so per-iteration bookkeeping is
reset::

    model = fleet.distributed_model(model)
    enable_fleet_prefetch(model)
    ...
    loss = model.train_batch([x, y], opt)
    get_offload_manager().end_iteration()

``OffloadManager.format_stats()`` reports whether prefetch is keeping up:
``prefetched`` should track ``packed``, and ``late`` and ``exposed`` should stay
at zero.
"""

from .fleet_hooks import enable_fleet_prefetch
from .manager import (
    OffloadManager,
    current_offload_manager,
    get_offload_manager,
    manager_from_config,
    offload_enabled,
    offload_groups,
    offload_kwargs_from_config,
    offload_region,
    reset_offload_manager,
)
from .pinned_pool import PinnedPool
from .pylayer_shim import install as install_pylayer_shim

__all__ = [
    "OffloadManager",
    "PinnedPool",
    "current_offload_manager",
    "enable_fleet_prefetch",
    "get_offload_manager",
    "install_pylayer_shim",
    "manager_from_config",
    "offload_enabled",
    "offload_groups",
    "offload_kwargs_from_config",
    "offload_region",
    "reset_offload_manager",
]
