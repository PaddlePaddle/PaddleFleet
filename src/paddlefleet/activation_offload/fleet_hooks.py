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

"""Pipeline schedule wiring: micro-step callbacks drive grouping and prefetch.

Built on the global micro-step hooks of ``paddle.distributed.fleet``
(``register_global_pipeline_parallel_hook``), using all four anchors. Custom
schedules that do not go through those hooks, such as dualpipe, are not
supported.

Ordering under interleaved 1F1B::

    FORWARD_BEGIN(chunk)  -> begin_forward_group(chunk)      open a group
    [forward kernels; activations inside a region are packed to pinned memory]
    FORWARD_END           -> clear_current_group()           close it
    ...
    BACKWARD_BEGIN(chunk) -> prefetch_next_group(chunk)      queue that group
    [backward kernels; a prefetched unpack only waits on an event, and each
     consumed tensor refills the budget]
    BACKWARD_END(chunk)   -> prefetch_next_group_head(chunk) also queue the group
                                                            that is next to run
                                                            backward, so its
                                                            leading tensors have
                                                            work to hide behind

Forward and backward step ids are numbered independently of each other, so the
callbacks never use ``step_id``: groups are keyed by chunk id plus the manager's
own micro-batch counters.
"""

from __future__ import annotations

from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
    PipelineParallelMicroStepLocations as _Loc,
    register_global_pipeline_parallel_hook as _register_hook,
)

from .manager import get_offload_manager

_wired = False
_pp_model = None


def _chunk_id():
    # Interleaved schedules set _virtual_pp_rank before every micro-step. Without
    # interleaving (VPP=1) the attribute is absent and everything collapses into
    # a single group, which is still correct.
    return getattr(_pp_model, "_virtual_pp_rank", 0) or 0


def enable_fleet_prefetch(pp_model=None):
    """Register the global micro-step callbacks that drive group prefetch.

    Call once after ``fleet.distributed_model()`` and before the first
    ``train_batch()``. After every accumulation step, the caller should call
    ``get_offload_manager().end_iteration()`` to clear leftover bookkeeping and
    counters.

    Args:
        pp_model: the ``PipelineParallel`` instance returned by
            ``fleet.distributed_model()``, read for ``_virtual_pp_rank`` (the
            VPP chunk id). Passing None collapses grouping into a single chain
            keyed by micro-batch order, which is correct when VPP=1.
    """
    global _wired, _pp_model
    _pp_model = pp_model
    if _wired:
        return
    _wired = True

    mgr = get_offload_manager()

    def on_fwd_begin(step_id=None, **kw):
        mgr.begin_forward_group(_chunk_id())

    def on_fwd_end(step_id=None, **kw):
        mgr.clear_current_group()

    def on_bwd_begin(step_id=None, **kw):
        # Prefetch anchor: the next micro-batch of this chunk is about to run
        # backward and no backward kernel has been issued yet, so the H2D copies
        # can hide behind the whole backward pass.
        mgr.prefetch_next_group(_chunk_id())

    def on_bwd_end(step_id=None, **kw):
        # Cross-group anchor: this group's backward has finished and its queue is
        # drained, so queue the group that runs backward next. Gated by
        # mgr.cross_group_prefetch; a no-op when that is off.
        mgr.prefetch_next_group_head(_chunk_id())

    _register_hook(_Loc.FORWARD_BEGIN, on_fwd_begin)
    _register_hook(_Loc.FORWARD_END, on_fwd_end)
    _register_hook(_Loc.BACKWARD_BEGIN, on_bwd_begin)
    _register_hook(_Loc.BACKWARD_END, on_bwd_end)
