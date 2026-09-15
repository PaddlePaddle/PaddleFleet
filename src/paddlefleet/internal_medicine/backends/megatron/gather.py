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

"""Megatron distributed gather function for training_logs."""

import logging

logger = logging.getLogger(__name__)


def _torch_gather(local_metrics: dict) -> list:
    """Gather metrics from all ranks using torch.distributed."""
    import torch.distributed as dist

    if not dist.is_initialized():
        return None
    world_size = dist.get_world_size()
    info_list = [None] * world_size
    dist.all_gather_object(info_list, local_metrics)
    return info_list


def install_gather_fn():
    """Install the torch-based gather function into the global training_logs."""
    from ...core.training_logs import training_logs

    try:
        import torch.distributed as dist

        if dist.is_initialized():
            training_logs.set_gather_fn(_torch_gather)
    except ImportError:
        pass
