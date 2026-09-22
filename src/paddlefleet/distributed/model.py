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

"""Explicit distributed wrappers; never mutate Paddle's global factories."""

from paddle.distributed import fleet

from paddlefleet.pipeline_parallel.indexcache_adapter import (
    IndexCachePipelineLayer,
    IndexCachePipelineParallel,
)


def distributed_model(model):
    """Select the IndexCache 1F1B wrapper, or delegate unchanged to Paddle."""
    config = getattr(model, "config", None)
    if not getattr(config, "indexcache_topk_pattern", None):
        return fleet.distributed_model(model)
    if not isinstance(model, IndexCachePipelineLayer):
        raise TypeError(
            "IndexCache requires a model with IndexCachePipelineLayer boundaries"
        )
    if model.get_num_virtual_stages() != 1:
        raise NotImplementedError(
            "IndexCache supports only ordinary 1F1B (VPP=1)"
        )
    strategy = fleet.fleet._user_defined_strategy
    pp_config = strategy.hybrid_configs["pp_configs"]
    if pp_config.use_dualpipev or pp_config.forward_backward_overlap_scheduler:
        raise NotImplementedError(
            "IndexCache does not support DualPipeV or compute-overlap scheduling"
        )
    hcg = fleet.get_hybrid_communicate_group()
    if hcg.get_pipe_parallel_world_size() <= 1:
        return fleet.distributed_model(model)
    if strategy.amp:
        raise NotImplementedError(
            "IndexCache PP requires Trainer-managed AMP/scaler rather than strategy.amp"
        )
    wrapper = IndexCachePipelineParallel(model, hcg, strategy=strategy)
    model._indexcache_instance_wrapper = True
    return wrapper
