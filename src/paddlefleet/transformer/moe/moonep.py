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

import importlib
import os
import weakref
from dataclasses import dataclass
from types import SimpleNamespace

import paddle

_runtime = None
_buffers = weakref.WeakSet()
_bridges = weakref.WeakSet()


def is_moonep_available() -> bool:
    """Return whether paddlefleet-ops loaded the optional MoonEP runtime."""
    try:
        from paddlefleet_ops import is_moonep_available as available

        return available()
    except ImportError:
        return False


def _load_runtime():
    """Load MoonEP only after paddlefleet-ops enabled its compat scope."""
    global _runtime
    if _runtime is not None:
        return _runtime
    if not is_moonep_available():
        raise ImportError(
            "moe_token_dispatcher_type=moonep but MoonEP is unavailable. "
            "Install paddlefleet-ops with MoonEP support."
        )

    from paddlefleet_ops.moonep import Buffer

    package = "paddlefleet_ops.moonep"
    buffer_module = importlib.import_module(f"{package}.buffer")
    c_module = importlib.import_module(f"{package}._C")
    grad_reduce_module = importlib.import_module(f"{package}.grad_reduce")
    prefetch_module = importlib.import_module(f"{package}.prefetch")
    sync_module = importlib.import_module(f"{package}.inter_rank_sync")
    _runtime = SimpleNamespace(
        Buffer=Buffer,
        alloc_chunk=buffer_module._alloc_nvl_chunk,
        exchange_fabric_handles=buffer_module._exchange_fabric_handles,
        exchange_ipc_fds=buffer_module._exchange_ipc_fds,
        get_vmm_granularity=buffer_module.get_vmm_granularity,
        mnnvl_enabled=buffer_module._mnnvl_enabled,
        map_fabric=c_module.nvl_dist_map_fabric,
        map_posix=c_module.nvl_dist_map,
        release_handle=c_module.nvl_release_mem_handle,
        launch_grad_reduce=grad_reduce_module.launch_grad_reduce,
        launch_prefetch=prefetch_module.launch_prefetch,
        inter_rank_sync=sync_module.launch_inter_rank_sync,
    )
    return _runtime


def new_moonep_buffer(**kwargs):
    """Create a Buffer tracked for collective teardown."""
    runtime = _load_runtime()
    buffer = runtime.Buffer(explicitly_destroy=True, **kwargs)
    _buffers.add(buffer)
    return buffer


def finalize_moonep() -> None:
    """Destroy live communication buffers before process-group teardown."""
    for buffer in list(_buffers):
        buffer.destroy()
    _buffers.clear()
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
    runtime = _load_runtime()
    rank = paddle.distributed.get_rank(group)
    world_size = paddle.distributed.get_world_size(group)

    chunk_nbytes = paddle.empty([], dtype=dtype).element_size()
    for dim in chunk_shape:
        chunk_nbytes *= int(dim)
    granularity = int(runtime.get_vmm_granularity())
    if chunk_nbytes % granularity:
        raise ValueError(
            "MoonEP expert storage must be VMM aligned: "
            f"shape={tuple(chunk_shape)}, dtype={dtype}, "
            f"bytes={chunk_nbytes}, granularity={granularity}."
        )

    use_mnnvl = runtime.mnnvl_enabled()
    expert_keepalive = slot_keepalive = None
    expert_owned = slot_owned = None
    open_fds = set()
    try:
        expert_keepalive, expert_handle, expert_owned = runtime.alloc_chunk(
            chunk_shape, dtype
        )
        if not use_mnnvl:
            open_fds.add(expert_handle)
        slot_keepalive, slot_handle, slot_owned = runtime.alloc_chunk(
            chunk_shape, dtype
        )
        if not use_mnnvl:
            open_fds.add(slot_handle)
        if use_mnnvl:
            expert_handles = runtime.exchange_fabric_handles(
                expert_handle, world_size, group
            )
            full = runtime.map_fabric(
                chunk_shape=chunk_shape,
                dtype=dtype,
                fabric_handles=[*expert_handles, slot_handle],
                local_rank=rank,
                world_size=world_size + 1,
            )
            reduce_view = None
            if with_reduce_view:
                slot_handles = runtime.exchange_fabric_handles(
                    slot_handle, world_size, group
                )
                reduce_view = runtime.map_fabric(
                    chunk_shape=chunk_shape,
                    dtype=dtype,
                    fabric_handles=slot_handles,
                    local_rank=rank,
                    world_size=world_size,
                )
        else:
            expert_fds = runtime.exchange_ipc_fds(
                expert_handle,
                list(range(world_size)),
                rank,
                world_size,
                group,
            )
            open_fds.update(expert_fds.values())
            os.close(expert_handle)
            open_fds.remove(expert_handle)

            if with_reduce_view:
                slot_fds = runtime.exchange_ipc_fds(
                    slot_handle,
                    list(range(world_size)),
                    rank,
                    world_size,
                    group,
                )
                open_fds.update(slot_fds.values())
                os.close(slot_handle)
                open_fds.remove(slot_handle)
                full_fds = [
                    *(expert_fds[index] for index in range(world_size)),
                    slot_fds[rank],
                ]
                full = runtime.map_posix(
                    chunk_shape=chunk_shape,
                    dtype=dtype,
                    fds=full_fds,
                    local_rank=rank,
                    world_size=world_size + 1,
                )
                reduce_view = runtime.map_posix(
                    chunk_shape=chunk_shape,
                    dtype=dtype,
                    fds=[slot_fds[index] for index in range(world_size)],
                    local_rank=rank,
                    world_size=world_size,
                )
            else:
                full_fds = [
                    *(expert_fds[index] for index in range(world_size)),
                    slot_handle,
                ]
                full = runtime.map_posix(
                    chunk_shape=chunk_shape,
                    dtype=dtype,
                    fds=full_fds,
                    local_rank=rank,
                    world_size=world_size + 1,
                )
                reduce_view = None
    finally:
        if not use_mnnvl:
            _close_fds(open_fds)
        if expert_owned is not None:
            runtime.release_handle(expert_owned)
        if slot_owned is not None:
            runtime.release_handle(slot_owned)

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
        intermediate_size = int(weight2_shape[1])
        if int(weight1_shape[2]) != 2 * intermediate_size:
            raise ValueError(
                "MoonEP requires packed SwiGLU weight1 with shape "
                "[local_experts, hidden, 2 * intermediate]."
            )

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

        runtime = _load_runtime()
        runtime.inter_rank_sync(self.buffer._require_ctx())
        self.prefetch(plan)

    @paddle.no_grad()
    def prefetch(self, plan) -> None:
        runtime = _load_runtime()
        ctx = self.buffer._require_ctx()
        experts_to_copy = plan.experts_to_copy[self.rank].contiguous()
        for projection in self.projections:
            runtime.launch_prefetch(
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

        runtime = _load_runtime()
        ctx = self.buffer._require_ctx()
        runtime.inter_rank_sync(ctx)
        for projection in self.projections:
            runtime.launch_grad_reduce(
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
    def forward(ctx, expert_output, buffer, plan, bridge):
        combined, _, _ = buffer.combine(
            plan=plan, hidden_nvsh=expert_output.contiguous()
        )
        ctx.buffer = buffer
        ctx.plan = plan
        ctx.bridge = bridge
        return combined

    @staticmethod
    def backward(ctx, grad_output):
        # A later microbatch may have reused the redundant slots.
        ctx.bridge.prefetch(ctx.plan)
        grad_expert_output, _, _, _ = ctx.buffer.dispatch(
            grad_output.contiguous(), plan=ctx.plan
        )
        return grad_expert_output


class MoonEPExperts(paddle.autograd.PyLayer):
    """SwiGLU grouped MLP over MoonEP's padded E+B expert groups."""

    @staticmethod
    def forward(
        ctx,
        hidden_states,
        weight1,
        weight2,
        tokens_per_expert,
        bridge,
        plan,
    ):
        bridge.prepare(weight1, weight2, plan)
        weight1_runtime, weight2_runtime = bridge.full_weights
        batch_sizes = [int(value) for value in tokens_per_expert.cpu().tolist()]
        num_valid_tokens = sum(batch_sizes)
        hidden = hidden_states[:num_valid_tokens]
        fc1_output = paddle.incubate.nn.functional.batched_gemm(
            hidden, weight1_runtime, batch_sizes
        )
        gate, up = paddle.chunk(fc1_output, 2, axis=-1)
        activated = paddle.nn.functional.silu(gate) * up
        output = paddle.incubate.nn.functional.batched_gemm(
            activated, weight2_runtime, batch_sizes
        )
        if num_valid_tokens != hidden_states.shape[0]:
            output = paddle.concat(
                [
                    output,
                    paddle.zeros(
                        [
                            hidden_states.shape[0] - num_valid_tokens,
                            hidden_states.shape[1],
                        ],
                        dtype=output.dtype,
                    ),
                ],
                axis=0,
            )

        ctx.save_for_backward(hidden, gate, up, activated)
        ctx.batch_sizes = batch_sizes
        ctx.num_input_tokens = hidden_states.shape[0]
        ctx.bridge = bridge
        ctx.plan = plan
        ctx.weight1_dtype = weight1.dtype
        ctx.weight2_dtype = weight2.dtype
        ctx.set_grad_in_dtype_consistent(False)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        hidden, gate, up, activated = ctx.saved_tensor()
        weight1_runtime, weight2_runtime = ctx.bridge.full_weights
        grad_output = grad_output[: hidden.shape[0]].contiguous()

        grad_weight2 = paddle.incubate.nn.functional.batched_gemm(
            activated,
            grad_output,
            ctx.batch_sizes,
            trans_lhs=True,
        )
        grad_activated = paddle.incubate.nn.functional.batched_gemm(
            grad_output,
            weight2_runtime,
            ctx.batch_sizes,
            trans_rhs=True,
        )
        sigmoid = paddle.nn.functional.sigmoid(gate)
        silu_gate = gate * sigmoid
        grad_gate = grad_activated * up * sigmoid * (1 + gate * (1 - sigmoid))
        grad_up = grad_activated * silu_gate
        grad_fc1 = paddle.concat([grad_gate, grad_up], axis=-1)
        grad_hidden = paddle.incubate.nn.functional.batched_gemm(
            grad_fc1,
            weight1_runtime,
            ctx.batch_sizes,
            trans_rhs=True,
        )
        grad_weight1 = paddle.incubate.nn.functional.batched_gemm(
            hidden,
            grad_fc1,
            ctx.batch_sizes,
            trans_lhs=True,
        )
        grad_weight1, grad_weight2 = ctx.bridge.reduce_grads(
            ctx.plan,
            grad_weight1,
            grad_weight2,
        )
        if hidden.shape[0] != ctx.num_input_tokens:
            grad_hidden = paddle.concat(
                [
                    grad_hidden,
                    paddle.zeros(
                        [
                            ctx.num_input_tokens - hidden.shape[0],
                            hidden.shape[1],
                        ],
                        dtype=grad_hidden.dtype,
                    ),
                ],
                axis=0,
            )
        return (
            grad_hidden,
            grad_weight1.astype(ctx.weight1_dtype),
            grad_weight2.astype(ctx.weight2_dtype),
            None,
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


def moonep_experts(
    hidden_states,
    grouped_experts,
    tokens_per_expert,
    bridge,
    plan,
):
    return MoonEPExperts.apply(
        hidden_states,
        grouped_experts.weight1,
        grouped_experts.weight2,
        tokens_per_expert,
        bridge,
        plan,
    )
