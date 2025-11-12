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

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import paddle.distributed.fleet.base.topology as tp
import warnings

# Intra-layer model parallel group that the current rank belongs to.
_TENSOR_MODEL_PARALLEL_GROUP = None
_TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = None

# Inter-layer model parallel group that the current rank belongs to.
_PIPELINE_MODEL_PARALLEL_GROUP = None
_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None

_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None

# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None

# Expert model parallel group that current rank belongs to.
_EXPERT_MODEL_PARALLEL_GROUP = None

# Expert data parallel group
_EXPERT_DATA_PARALLEL_GROUP = None

# Context parallel group that the current rank belongs to
_CONTEXT_PARALLEL_GROUP = None

# Data parallel group information with context parallel combined.
_DATA_PARALLEL_GROUP_WITH_CP = None


def initialize_model_parallel(
    hcg: tp.EPHybridCommunicateGroup | tp.HybridCommunicateGroup,
    virtual_pipeline_model_parallel_size: int | None = None,
):
    global _TENSOR_MODEL_PARALLEL_GROUP
    global _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS
    _TENSOR_MODEL_PARALLEL_GROUP = hcg._mp_comm_group
    _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = hcg._mp_group

    global _PIPELINE_MODEL_PARALLEL_GROUP
    _PIPELINE_MODEL_PARALLEL_GROUP = hcg._pp_comm_group

    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = hcg._sharding_comm_group

    global _EXPERT_MODEL_PARALLEL_GROUP
    global _EXPERT_DATA_PARALLEL_GROUP
    _EXPERT_MODEL_PARALLEL_GROUP = hcg._ep_comm_group
    _EXPERT_DATA_PARALLEL_GROUP = hcg._moe_sharding_comm_group

    global _CONTEXT_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP_WITH_CP
    _CONTEXT_PARALLEL_GROUP = hcg._cp_comm_group
    _DATA_PARALLEL_GROUP_WITH_CP = hcg._cp_sharding_comm_group

    if virtual_pipeline_model_parallel_size is not None:
        if not hcg._pp_comm_group.nranks > 1:
            raise RuntimeError(
                "pipeline-model-parallel size should be greater than 1 with interleaved schedule"
            )
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = (
            virtual_pipeline_model_parallel_size
        )


def get_tensor_model_parallel_group(check_initialized=True):
    """Get the tensor-model-parallel group the caller rank belongs to."""
    if check_initialized:
        assert _TENSOR_MODEL_PARALLEL_GROUP is not None, (
            "tensor model parallel group is not initialized"
        )
    return _TENSOR_MODEL_PARALLEL_GROUP


def get_pipeline_model_parallel_group(check_initialized=True):
    """Get the pipeline-model-parallel group the caller rank belongs to."""
    if check_initialized:
        assert _PIPELINE_MODEL_PARALLEL_GROUP is not None, (
            "pipeline_model parallel group is not initialized"
        )
    return _PIPELINE_MODEL_PARALLEL_GROUP


def get_data_parallel_group(with_context_parallel=False):
    """Get the data-parallel group the caller rank belongs to."""
    if with_context_parallel:
        assert _DATA_PARALLEL_GROUP_WITH_CP is not None, (
            "data parallel group with context parallel combined is not initialized"
        )
        return _DATA_PARALLEL_GROUP_WITH_CP
    else:
        assert _DATA_PARALLEL_GROUP is not None, (
            "data parallel group is not initialized"
        )
        return _DATA_PARALLEL_GROUP


def get_expert_model_parallel_group(check_initialized=True):
    """Get the expert-model-parallel group the caller rank belongs to."""
    if check_initialized:
        assert _EXPERT_MODEL_PARALLEL_GROUP is not None, (
            "expert model parallel group is not initialized"
        )
    return _EXPERT_MODEL_PARALLEL_GROUP


def get_expert_data_parallel_group(check_initialized=True):
    """Get expert data parallel group."""
    if check_initialized:
        assert _EXPERT_DATA_PARALLEL_GROUP is not None, (
            "Expert data parallel group is not initialized"
        )
    return _EXPERT_DATA_PARALLEL_GROUP


def get_context_parallel_group(check_initialized=True):
    """Get the context-parallel group the caller rank belongs to."""
    if check_initialized:
        assert _CONTEXT_PARALLEL_GROUP is not None, (
            "context parallel group is not initialized"
        )
    return _CONTEXT_PARALLEL_GROUP


def get_pipeline_model_parallel_world_size():
    """Return world size for the pipeline-model-parallel group."""
    global _PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    if _PIPELINE_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return get_pipeline_model_parallel_group().nranks


def set_pipeline_model_parallel_world_size(world_size):
    """Set the pipeline-model-parallel size"""
    global _PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _PIPELINE_MODEL_PARALLEL_WORLD_SIZE = world_size


def get_virtual_pipeline_model_parallel_rank():
    """Return the virtual pipeline-parallel rank."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK


def set_virtual_pipeline_model_parallel_rank(rank):
    """Set the virtual pipeline-parallel rank."""
    warnings.warn(
        "set_virtual_pipeline_model_parallel_rank in global scope is deprecated. "
        "Pass vp_stage explicitly to is_pipeline_first_stage, is_pipeline_last_stage, etc.",
        DeprecationWarning,
    )
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = rank


def get_virtual_pipeline_model_parallel_world_size():
    """Return the virtual pipeline-parallel world size."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
