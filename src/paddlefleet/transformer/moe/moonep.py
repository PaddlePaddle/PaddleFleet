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

from __future__ import annotations

import os
import weakref
from dataclasses import dataclass
from functools import partial

import paddle

from .moe_expert import RuntimeExpertWeights

try:
    from paddlefleet_ops import is_moonep_available as _is_moonep_available
except ImportError:
    _MOONEP_AVAILABLE = False
else:
    _MOONEP_AVAILABLE = _is_moonep_available()
if _MOONEP_AVAILABLE:
    from paddlefleet_ops.moonep import Buffer as MoonEPBuffer
    from paddlefleet_ops.moonep._C import (
        nvl_dist_alloc,
        nvl_dist_map,
        nvl_release_mem_handle,
    )
    from paddlefleet_ops.moonep.buffer import (
        _all_gather_shareables,
        _exchange_ipc_fds,
        _use_fabric_for_group,
        get_vmm_granularity,
    )
    from paddlefleet_ops.moonep.grad_reduce import launch_grad_reduce
    from paddlefleet_ops.moonep.inter_rank_sync import launch_inter_rank_sync
    from paddlefleet_ops.moonep.prefetch import launch_prefetch
else:
    MoonEPBuffer = None


_buffer_cache = {}
_bridges = weakref.WeakSet()


def is_moonep_available() -> bool:
    """Return whether paddlefleet-ops loaded the optional MoonEP runtime."""
    return _MOONEP_AVAILABLE


def _require_moonep() -> None:
    if not _MOONEP_AVAILABLE:
        raise ImportError(
            "moe_token_dispatcher_type=moonep but MoonEP is unavailable. "
            "Install paddlefleet-ops with MoonEP support."
        )


def get_moonep_buffer(
    *,
    S: int,
    H: int,
    K: int,
    E: int,
    B: int,
    num_ep_ranks: int,
    group,
    num_sms: int | None = None,
):
    """Get a shared Buffer for one EP group, device, and fixed signature."""
    _require_moonep()
    S, H, K, E, B, num_ep_ranks = (
        int(value) for value in (S, H, K, E, B, num_ep_ranks)
    )
    if num_sms is not None:
        num_sms = int(num_sms)
    signature = (
        id(group),
        paddle.get_device(),
        S,
        H,
        K,
        E,
        B,
        num_ep_ranks,
        num_sms,
    )
    buffer = _buffer_cache.get(signature)
    if buffer is None:
        buffer = MoonEPBuffer(
            S=S,
            H=H,
            K=K,
            E=E,
            B=B,
            num_ep_ranks=num_ep_ranks,
            group=group,
            num_sms=num_sms,
            explicitly_destroy=True,
        )
        _buffer_cache[signature] = buffer
    return buffer


def finalize_moonep() -> None:
    """Destroy live communication buffers before process-group teardown."""
    for buffer in list(_buffer_cache.values()):
        buffer.destroy()
    _buffer_cache.clear()
    for bridge in list(_bridges):
        bridge.destroy()
    _bridges.clear()


def _close_fds(fds) -> None:
    for fd in set(fds):
        os.close(fd)


def _allocate_mapping(
    chunk_shape: list[int],
    dtype,
    group,
    *,
    with_reduce_view: bool,
):
    """Map all owner chunks followed by this rank's redundant slot chunk."""
    _require_moonep()
    rank = paddle.distributed.get_rank(group)
    world_size = paddle.distributed.get_world_size(group)

    chunk_nbytes = paddle.empty([], dtype=dtype).element_size()
    for dim in chunk_shape:
        chunk_nbytes *= int(dim)
    granularity = int(get_vmm_granularity())
    if chunk_nbytes % granularity:
        raise ValueError(
            "MoonEP expert storage must be VMM aligned: "
            f"shape={tuple(chunk_shape)}, dtype={dtype}, "
            f"bytes={chunk_nbytes}, granularity={granularity}."
        )

    use_fabric = _use_fabric_for_group(group)
    expert_keepalive = slot_keepalive = None
    expert_owned = slot_owned = None
    open_fds = set()
    try:
        expert_keepalive, expert_handle, expert_owned = nvl_dist_alloc(
            shape=chunk_shape, dtype=dtype, use_fabric=use_fabric
        )
        if not use_fabric:
            expert_fd = int(expert_handle.item())
            open_fds.add(expert_fd)
        slot_keepalive, slot_handle, slot_owned = nvl_dist_alloc(
            shape=chunk_shape, dtype=dtype, use_fabric=use_fabric
        )
        if not use_fabric:
            slot_fd = int(slot_handle.item())
            open_fds.add(slot_fd)
        if use_fabric:
            expert_handles = _all_gather_shareables(expert_handle, group)
            full_handles = paddle.concat(
                [expert_handles, slot_handle.reshape([1, -1])], axis=0
            )
            full = nvl_dist_map(
                chunk_shape=chunk_shape,
                dtype=dtype,
                shareables=full_handles,
                local_rank=rank,
                world_size=world_size + 1,
                use_fabric=True,
            )
            reduce_view = None
            if with_reduce_view:
                slot_handles = _all_gather_shareables(slot_handle, group)
                reduce_view = nvl_dist_map(
                    chunk_shape=chunk_shape,
                    dtype=dtype,
                    shareables=slot_handles,
                    local_rank=rank,
                    world_size=world_size,
                    use_fabric=True,
                )
        else:
            expert_fds = _exchange_ipc_fds(
                expert_fd,
                list(range(world_size)),
                rank,
                world_size,
                group,
            )
            open_fds.update(expert_fds.values())
            os.close(expert_fd)
            open_fds.remove(expert_fd)

            if with_reduce_view:
                slot_fds = _exchange_ipc_fds(
                    slot_fd,
                    list(range(world_size)),
                    rank,
                    world_size,
                    group,
                )
                open_fds.update(slot_fds.values())
                os.close(slot_fd)
                open_fds.remove(slot_fd)
                full_fds = [
                    *(expert_fds[index] for index in range(world_size)),
                    slot_fds[rank],
                ]
                full = nvl_dist_map(
                    chunk_shape=chunk_shape,
                    dtype=dtype,
                    shareables=paddle.to_tensor(
                        full_fds, dtype="int64", place=paddle.CPUPlace()
                    ),
                    local_rank=rank,
                    world_size=world_size + 1,
                    use_fabric=False,
                )
                reduce_view = nvl_dist_map(
                    chunk_shape=chunk_shape,
                    dtype=dtype,
                    shareables=paddle.to_tensor(
                        [slot_fds[index] for index in range(world_size)],
                        dtype="int64",
                        place=paddle.CPUPlace(),
                    ),
                    local_rank=rank,
                    world_size=world_size,
                    use_fabric=False,
                )
            else:
                full_fds = [
                    *(expert_fds[index] for index in range(world_size)),
                    slot_fd,
                ]
                full = nvl_dist_map(
                    chunk_shape=chunk_shape,
                    dtype=dtype,
                    shareables=paddle.to_tensor(
                        full_fds, dtype="int64", place=paddle.CPUPlace()
                    ),
                    local_rank=rank,
                    world_size=world_size + 1,
                    use_fabric=False,
                )
                reduce_view = None
    finally:
        if not use_fabric:
            _close_fds(open_fds)
        if expert_owned is not None:
            nvl_release_mem_handle(expert_owned)
        if slot_owned is not None:
            nvl_release_mem_handle(slot_owned)

    return full, reduce_view, (expert_keepalive, slot_keepalive)


@dataclass
class _Projection:
    full_weight: paddle.Tensor
    full_grad: paddle.Tensor
    reduce_buffer: paddle.Tensor
    keepalives: tuple


class MoonEPWeightBridge:
    """Bridge registered local expert parameters to MoonEP's E+B layout."""

    def __init__(
        self,
        *,
        group,
        num_experts: int,
        num_local_experts: int,
        weight1_shape,
        weight2_shape,
    ):
        self.group = group
        self.rank = paddle.distributed.get_rank(group)
        self.world_size = paddle.distributed.get_world_size(group)
        self.num_experts = int(num_experts)
        self.num_local_experts = int(num_local_experts)
        self.num_runtime_experts = self.num_experts + self.num_local_experts
        self.buffer = None
        self._destroyed = False

        if self.num_experts != self.world_size * self.num_local_experts:
            raise ValueError(
                "MoonEP requires an even expert distribution: "
                f"num_experts={self.num_experts}, "
                f"world_size={self.world_size}, "
                f"num_local_experts={self.num_local_experts}."
            )
        if len(weight1_shape) != 3 or len(weight2_shape) != 3:
            raise ValueError("MoonEP requires grouped 3-D expert weights.")

        projection_shapes = (
            [int(value) for value in weight1_shape],
            [int(value) for value in weight2_shape],
        )
        self.projections = []
        for shape in projection_shapes:
            full_weight, _, weight_keepalives = _allocate_mapping(
                shape, paddle.bfloat16, group, with_reduce_view=False
            )
            full_grad, reduce_buffer, grad_keepalives = _allocate_mapping(
                shape, paddle.float32, group, with_reduce_view=True
            )
            runtime_shape = [
                self.num_runtime_experts,
                shape[1],
                shape[2],
            ]
            self.projections.append(
                _Projection(
                    full_weight=full_weight.reshape(runtime_shape),
                    full_grad=full_grad.reshape(runtime_shape),
                    reduce_buffer=reduce_buffer.reshape(
                        [self.world_size, *shape]
                    ),
                    keepalives=(*weight_keepalives, *grad_keepalives),
                )
            )
        _bridges.add(self)

    @property
    def full_weights(self):
        return tuple(projection.full_weight for projection in self.projections)

    def attach_buffer(self, buffer) -> None:
        self.buffer = buffer

    def _local_slice(self):
        start = self.rank * self.num_local_experts
        return slice(start, start + self.num_local_experts)

    @paddle.no_grad()
    def prepare(self, weight1, weight2, plan) -> None:
        if self.buffer is None:
            raise RuntimeError(
                "MoonEP weight bridge has no communication buffer."
            )
        local_slice = self._local_slice()
        for source, projection in zip((weight1, weight2), self.projections):
            projection.full_weight[local_slice].copy_(source.contiguous())

        launch_inter_rank_sync(self.buffer._require_ctx())
        self.prefetch(plan)

    @paddle.no_grad()
    def prefetch(self, plan, projection_index=None) -> None:
        ctx = self.buffer._require_ctx()
        experts_to_copy = plan.experts_to_copy[self.rank].contiguous()
        projections = (
            self.projections
            if projection_index is None
            else (self.projections[projection_index],)
        )
        for projection in projections:
            launch_prefetch(
                projection.full_weight[: self.num_experts],
                projection.full_weight[self.num_experts :],
                experts_to_copy,
                num_sms=int(ctx["num_sms"]),
            )

    @paddle.no_grad()
    def reduce_grads(self, plan, grad_weight1, grad_weight2):
        local_slice = self._local_slice()
        slot_slice = slice(self.num_experts, self.num_runtime_experts)
        for grad, projection in zip(
            (grad_weight1, grad_weight2), self.projections
        ):
            grad = grad.astype(paddle.float32).contiguous()
            projection.full_grad[local_slice].copy_(grad[local_slice])
            projection.full_grad[slot_slice].copy_(grad[slot_slice])

        ctx = self.buffer._require_ctx()
        launch_inter_rank_sync(ctx)
        for projection in self.projections:
            launch_grad_reduce(
                projection.full_grad[: self.num_experts],
                projection.reduce_buffer,
                plan.experts_to_copy,
                rank=self.rank,
                num_sms=int(ctx["num_sms"]),
                meta_buf=ctx["meta_buf"],
                meta_stride=int(ctx["meta_chunk_padded"]),
                barrier_off=int(ctx["BARRIER_OFF"]),
                grid_sync_bar=ctx["grid_sync_bar"],
            )
        return tuple(
            projection.full_grad[local_slice].clone()
            for projection in self.projections
        )

    def destroy(self) -> None:
        if self._destroyed:
            return
        self.projections.clear()
        self.buffer = None
        self._destroyed = True


class MoonEPDispatch(paddle.autograd.PyLayer):
    """Autograd-aware MoonEP dispatch with a saved communication plan."""

    @staticmethod
    def forward(
        ctx,
        hidden_states,
        topk_probs,
        topk_indices,
        tokens_per_expert,
        buffer,
        state,
    ):
        dispatched, dispatched_probs, cu_seqlens, plan = buffer.dispatch(
            hidden_states.contiguous(),
            topk_probs.astype(paddle.float32).contiguous(),
            topk_indices.astype(paddle.int32).contiguous(),
            tokens_per_expert.astype(paddle.int32).contiguous(),
        )
        starts = paddle.concat(
            [paddle.zeros([1], dtype=cu_seqlens.dtype), cu_seqlens[:-1]]
        )
        runtime_tokens_per_expert = (cu_seqlens - starts).astype("int64")
        runtime_tokens_per_expert.stop_gradient = True
        ctx.buffer = buffer
        ctx.plan = plan
        state["plan"] = plan
        ctx.set_grad_in_dtype_consistent(False)
        return dispatched, dispatched_probs, runtime_tokens_per_expert

    @staticmethod
    def backward(ctx, grad_hidden, grad_probs, _grad_counts):
        grad_hidden_states, grad_topk_probs, _ = ctx.buffer.combine(
            plan=ctx.plan,
            hidden_nvsh=grad_hidden.contiguous(),
            route_weights_nvs=(
                grad_probs.astype(paddle.float32).contiguous()
                if grad_probs is not None
                else None
            ),
        )
        return grad_hidden_states, grad_topk_probs, None, None


class MoonEPCombine(paddle.autograd.PyLayer):
    """Combine expert output and re-dispatch its gradient with the saved plan."""

    @staticmethod
    def forward(ctx, expert_output, buffer, plan, _bridge):
        combined, _, _ = buffer.combine(
            plan=plan, hidden_nvsh=expert_output.contiguous()
        )
        ctx.buffer = buffer
        ctx.plan = plan
        return combined

    @staticmethod
    def backward(ctx, grad_output):
        grad_expert_output, _, _, _ = ctx.buffer.dispatch(
            grad_output.contiguous(), plan=ctx.plan
        )
        return grad_expert_output


class MoonEPRuntimeWeights(paddle.autograd.PyLayer):
    """Expose prefetched E+B weights while reducing their grads to owners."""

    @staticmethod
    def forward(ctx, weight1, weight2, bridge, plan):
        bridge.prepare(weight1, weight2, plan)
        ctx.bridge = bridge
        ctx.plan = plan
        ctx.weight1_dtype = weight1.dtype
        ctx.weight2_dtype = weight2.dtype
        ctx.set_grad_in_dtype_consistent(False)
        return bridge.full_weights

    @staticmethod
    def backward(ctx, grad_weight1, grad_weight2):
        grad_weight1, grad_weight2 = ctx.bridge.reduce_grads(
            ctx.plan,
            grad_weight1,
            grad_weight2,
        )
        return (
            grad_weight1.astype(ctx.weight1_dtype),
            grad_weight2.astype(ctx.weight2_dtype),
        )


def moonep_dispatch(
    hidden_states,
    topk_probs,
    topk_indices,
    tokens_per_expert,
    buffer,
    state,
):
    return MoonEPDispatch.apply(
        hidden_states,
        topk_probs,
        topk_indices,
        tokens_per_expert,
        buffer,
        state,
    )


def moonep_combine(expert_output, buffer, plan, bridge):
    return MoonEPCombine.apply(expert_output, buffer, plan, bridge)


def moonep_runtime_weights(grouped_experts, bridge, plan):
    tensors = MoonEPRuntimeWeights.apply(
        grouped_experts.weight1, grouped_experts.weight2, bridge, plan
    )
    return RuntimeExpertWeights(
        tensors=tensors,
        restore_before_backward=tuple(
            partial(bridge.prefetch, plan, projection_index=index)
            for index in range(len(tensors))
        ),
    )
