# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025 DeepSeek
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

try:
    from paddle.distributed.communication import deep_ep

    HAVE_DEEP_EP = True
except ImportError:
    HAVE_DEEP_EP = False

import importlib

import paddle
from paddle import framework
from paddle.autograd import PyLayer
from paddle.distributed.communication.group import Group

from .moe_utils import manual_backward

_buffer = None


def set_pfcc_deep_ep_backend():
    global deep_ep, HAVE_DEEP_EP, _buffer
    pfcc_deep_ep = importlib.import_module("paddlefleet.ops.deep_ep")
    deep_ep = pfcc_deep_ep
    _buffer = None
    HAVE_DEEP_EP = deep_ep is not None


def barrier_ep(ep_group):
    """barrier_ep"""
    paddle.distributed.barrier(ep_group)


def wait_for_deepep(group_id):
    """wait_for_deepep"""
    comm_event = deep_ep.get_event_from_comm_stream(group_id)
    comm_event.calc_stream_wait(group_id)


def get_hidden_bytes(x: paddle.Tensor) -> int:
    """Calculate the number of hidden bytes for a tensor.

    Args:
        x (paddle.Tensor): Input tensor

    Returns:
        int: Number of hidden bytes
    """
    return x.shape[1] * max(x.element_size(), 2)


def get_buffer(group: Group, hidden_bytes: int):
    """Get or create a buffer for all-to-all communication.

    Args:
        group (paddle.distributed.ProcessGroup): Process group for communication
        hidden_bytes (int): Number of hidden bytes needed

    Returns:
        Buffer: Communication buffer
    """
    global _buffer
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (
        deep_ep.Buffer.get_dispatch_config(group.world_size),
        deep_ep.Buffer.get_combine_config(group.world_size),
    ):
        # Split long line for PEP8 compliance
        num_nvl_bytes = max(
            config.get_nvl_buffer_size_hint(hidden_bytes, group.world_size),
            num_nvl_bytes,
        )
        num_rdma_bytes = max(
            config.get_rdma_buffer_size_hint(hidden_bytes, group.world_size),
            num_rdma_bytes,
        )

    # Allocate buffer if not existed or not enough buffer
    # NOTES: the adaptive routing configuration of the network **must be off**
    if (
        _buffer is None
        or _buffer.group != group
        or _buffer.num_nvl_bytes < num_nvl_bytes
        or _buffer.num_rdma_bytes < num_rdma_bytes
    ):
        _buffer = deep_ep.Buffer(group, num_nvl_bytes, num_rdma_bytes)
    return _buffer


def fused_dispatch_forward_func(
    x,
    token_indices,
    token_probs,
    num_experts,
    group,
    previous_event=None,
    async_finish=False,
    allocate_on_comm_stream=False,
    moe_ep_barrier: bool = True,
):
    """Forward pass of fused dispatch."""
    if moe_ep_barrier:
        barrier_ep(group)
    # Calculate layout before actual dispatch
    if isinstance(x, tuple):
        buffer = get_buffer(group, get_hidden_bytes(x[0]))
    else:
        buffer = get_buffer(group, get_hidden_bytes(x))
    (
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        previous_event_,
    ) = buffer.get_dispatch_layout(
        token_indices,
        num_experts,
        previous_event=previous_event,
        async_finish=async_finish,
        allocate_on_comm_stream=allocate_on_comm_stream,
    )

    assert token_probs.dtype == paddle.float32
    # Do MoE dispatch
    # NOTES: the CPU will wait for GPU's signal to arrive,
    # so this is not compatible with CUDA graph
    (
        recv_x,
        recv_token_indices,
        recv_token_probs,
        num_recv_tokens_per_expert_list,
        handle,
        event,
    ) = buffer.dispatch(
        x,
        topk_idx=token_indices,
        topk_weights=token_probs,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        previous_event=previous_event,
        async_finish=async_finish,
        allocate_on_comm_stream=allocate_on_comm_stream,
    )

    states = {}
    states["dispatched_indices"] = recv_token_indices
    states["tokens_per_expert"] = num_recv_tokens_per_expert_list
    states["handle"] = handle

    return recv_x, recv_token_probs, states, event


def fused_dispatch_backward_func(
    grad_output,
    grad_token_probs,
    group,
    handle,
    previous_event=None,
    async_finish=False,
    allocate_on_comm_stream=False,
    moe_ep_barrier: bool = True,
):
    """Backward pass of fused dispatch."""
    if moe_ep_barrier:
        barrier_ep(group)

    buffer = get_buffer(group, get_hidden_bytes(grad_output))

    grad_x, grad_token_probs, event = buffer.combine(
        grad_output.contiguous(),
        handle,
        topk_weights=grad_token_probs.cast(paddle.float32),
        previous_event=previous_event,
        async_finish=async_finish,
        allocate_on_comm_stream=allocate_on_comm_stream,
    )
    return grad_x, None, grad_token_probs


def fused_combine_forward_func(
    x,
    group,
    states,
    previous_event=None,
    async_finish=False,
    allocate_on_comm_stream=False,
    moe_ep_barrier: bool = True,
):
    """Forward pass of fused combine."""
    if moe_ep_barrier:
        barrier_ep(group)

    handle = states["handle"]
    buffer = get_buffer(group, get_hidden_bytes(x))
    combined_x, _, event = buffer.combine(
        x,
        handle=handle,
        async_finish=async_finish,
        previous_event=previous_event,
        allocate_on_comm_stream=allocate_on_comm_stream,
    )
    return combined_x


def fused_combine_backward_func(
    grad_output,
    group,
    handle,
    previous_event=None,
    async_finish=False,
    allocate_on_comm_stream=False,
    moe_ep_barrier: bool = True,
):
    """Backward pass of fused combine."""
    if moe_ep_barrier:
        barrier_ep(group)

    if isinstance(grad_output, tuple):
        buffer = get_buffer(group, get_hidden_bytes(grad_output[0]))
        grad_x, _, _, _, _, event = buffer.dispatch(
            (grad_output[0].contiguous(), grad_output[1].contiguous()),
            handle=handle,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
    else:
        buffer = get_buffer(group, get_hidden_bytes(grad_output))
        grad_x, _, _, _, _, event = buffer.dispatch(
            grad_output.contiguous(),
            handle=handle,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
    return grad_x


class FusedDispatch(PyLayer):
    """Fused dispatch operation for MoE routing combining computation and communication."""

    @staticmethod
    def forward(
        ctx,
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        previous_event=None,
        fp8_dispatch: bool = False,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
        moe_ep_barrier: bool = True,
    ):
        """Forward pass of fused dispatch."""
        if fp8_dispatch:
            x_fp8, scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
                x,
                quant_method="1x128",
                input_transpose=False,
                output_scale_transpose=True,
                return_transpose_only=False,
            )
            scale = scale.T.contiguous()
            x = (x_fp8, scale)
        recv_x, recv_token_probs, states, event = fused_dispatch_forward_func(
            x,
            token_indices,
            token_probs,
            num_experts,
            group,
            previous_event,
            async_finish,
            allocate_on_comm_stream,
            moe_ep_barrier=moe_ep_barrier,
        )

        ctx.group = group
        ctx.handle = states["handle"]
        ctx.event = event
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        ctx.set_grad_in_dtype_consistent(False)
        ctx.moe_ep_barrier = moe_ep_barrier
        if fp8_dispatch:
            recv_x, scale = recv_x
            return recv_x, recv_token_probs, states, {"scale": scale}
        return recv_x, recv_token_probs, states, None

    @staticmethod
    def backward(ctx, grad_output, grad_token_probs):
        """Backward pass of fused dispatch."""
        return fused_dispatch_backward_func(
            grad_output,
            grad_token_probs,
            ctx.group,
            ctx.handle,
            None,  # previous_event
            ctx.async_finish,
            ctx.allocate_on_comm_stream,
            moe_ep_barrier=ctx.moe_ep_barrier,
        )


class FusedCombine(PyLayer):
    """Fused combine operation for MoE output combining computation and communication."""

    @staticmethod
    def forward(
        ctx,
        x,
        group,
        states,
        previous_event=None,
        async_finish=False,
        allocate_on_comm_stream=False,
        moe_ep_barrier: bool = True,
    ):
        """Forward pass of fused combine."""
        combined_x = fused_combine_forward_func(
            x, group, states, previous_event, moe_ep_barrier=moe_ep_barrier
        )

        ctx.handle = states["handle"]
        ctx.group = group
        ctx.previous_event = previous_event
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        ctx.moe_ep_barrier = moe_ep_barrier

        return combined_x

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass of fused combine."""
        return fused_combine_backward_func(
            grad_output,
            ctx.group,
            ctx.handle,
            ctx.previous_event,
            ctx.async_finish,
            ctx.allocate_on_comm_stream,
            moe_ep_barrier=ctx.moe_ep_barrier,
        )


class FusedCombineAsync(PyLayer):
    """FusedCombineAsync."""

    @staticmethod
    def forward(
        ctx,
        x,
        group,
        states,
        *fn_args,
        fn,
        is_first_fwd=False,
    ):
        """Forward pass of fused combine."""
        combined_x = fused_combine_forward_func(
            x,
            group,
            states,
            async_finish=True,
        )

        assert fn is not None, "use FusedCombineAsync async, but fn is None."
        ctx.bwf, fn_out = manual_backward(fn, is_first_fwd, *fn_args)

        ctx.handle = states["handle"]
        ctx.group = group

        wait_for_deepep(group.id)

        return (combined_x,) + fn_out  # noqa: RUF005

    @staticmethod
    def backward(ctx, grad_output, *fn_out_grads):
        """Backward pass of fused combine."""
        grad_x = fused_combine_backward_func(
            grad_output,
            ctx.group,
            ctx.handle,
            async_finish=True,
        )

        fn_args_grads = ctx.bwf(*fn_out_grads)

        wait_for_deepep(ctx.group.id)
        return (grad_x,) + fn_args_grads  # noqa: RUF005


if HAVE_DEEP_EP:

    def fused_dispatch(
        x,
        token_indices,
        token_probs,
        num_experts,
        group: Group,
        previous_event=None,
        fp8_dispatch: bool = False,
        async_finish=False,
        allocate_on_comm_stream=False,
        moe_ep_barrier: bool = True,
    ):
        """Perform fused dispatch operation if deep_ep is available.

        Args:
            x: Input tensor [num_tokens, hidden_size]
            token_indices: Token routing indices [num_tokens, topk]
            token_probs: Token routing probabilities [num_tokens, topk]
            num_experts: Number of experts
            group: Process group
            previous_event: Previous CUDA event
            moe_ep_barrier: Whether to use barrier for expert parallelism

        Returns:
            Result of FusedDispatch
        """
        return FusedDispatch.apply(
            x.contiguous(),
            token_indices,
            token_probs,
            num_experts,
            group,
            previous_event,
            fp8_dispatch,
            async_finish,
            allocate_on_comm_stream,
            moe_ep_barrier=moe_ep_barrier,
        )

    def fused_combine(
        x,
        group,
        handle,
        previous_event=None,
        combine_overlap_handle=None,
        async_finish=False,
        moe_ep_barrier: bool = True,
    ):
        """Perform fused combine operation if deep_ep is available.

        Args:
            x: Input tensor
            group: Process group
            handle: Communication handle
            previous_event: Previous CUDA event
            combine_overlap_handle: Handle for overlapping with shared experts
            moe_ep_barrier: Whether to use barrier for expert parallelism

        Returns:
            Result of FusedCombine
        """
        states = {}
        states["handle"] = handle
        if combine_overlap_handle is None:
            return FusedCombine.apply(
                x,
                group,
                states,
                previous_event,
                async_finish,
                moe_ep_barrier=moe_ep_barrier,
            )
        else:
            assert previous_event is None
            assert isinstance(combine_overlap_handle, dict)
            assert "fn" in combine_overlap_handle
            assert "fn_args" in combine_overlap_handle
            assert isinstance(combine_overlap_handle["fn_args"], tuple)
            combined_x, *fn_out = FusedCombineAsync.apply(
                x,
                group,
                states,
                *(combine_overlap_handle["fn_args"]),
                fn=combine_overlap_handle["fn"],
                is_first_fwd=not framework._dygraph_tracer()._has_grad,
            )
            combine_overlap_handle["fn_out"] = fn_out
            return combined_x

else:
    fused_dispatch = None
    fused_combine = None


class DispatchNode:
    def __init__(self, name="dispatch"):
        self.name = name

    def reset_statue(self):
        self.handle = None

    def forward(
        self,
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        previous_event=None,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        """Forward pass of fused dispatch."""
        recv_x, recv_token_probs, states, event = fused_dispatch_forward_func(
            x,
            token_indices,
            token_probs,
            num_experts,
            group,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

        self.group = group
        self.handle = states["handle"]
        self.event = event

        return recv_x, recv_token_probs, states

    def backward(
        self,
        grad_output,
        grad_token_probs,
        previous_event=None,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        """Backward pass of fused dispatch."""
        out = fused_dispatch_backward_func(
            grad_output,
            grad_token_probs,
            self.group,
            self.handle,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        self.reset_statue()
        return out


class CombineNode:
    def __init__(self, name="combine"):
        self.name = name

    def reset_statue(self):
        self.handle = None

    def forward(
        self,
        x,
        group,
        handle,
        previous_event=None,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        """Forward pass of fused combine."""
        states = {}
        states["handle"] = handle
        combined_x = fused_combine_forward_func(
            x,
            group,
            states,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

        self.handle = handle
        self.group = group
        self.previous_event = previous_event

        return combined_x

    def backward(
        self,
        grad_output,
        previous_event=None,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        """Backward pass of fused combine."""
        out = fused_combine_backward_func(
            grad_output,
            self.group,
            self.handle,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        self.reset_statue()
        return out
