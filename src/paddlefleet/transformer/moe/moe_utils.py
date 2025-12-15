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
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import paddle
from paddle import Tensor, framework

try:
    from paddle import scatter_add_
except ImportError:
    scatter_add_ = None
import paddle.distributed as dist

if TYPE_CHECKING:
    from collections.abc import Callable

    from paddle.distributed.communication.group import Group


def permute(
    tokens,
    routing_map,
    num_out_tokens: int | None = None,
    drop_and_pad: bool = False,
):
    """Permute the tokens and probs based on the mask.
    Tokens with the same designated expert will be grouped together.
    The shape of mask is [tokens, num_experts], it indicates which experts were selected
    by each token.

    Args:
        tokens (paddle.Tensor): The input token tensor, [num_tokens, hidden].
        routing_map (paddle.Tensor): The sparse token to expert mapping, [num_tokens, num_experts].
        num_out_tokens (int, optional): The number of output tokens. If None, it's set to
                                        the number of input tokens.
        drop_and_pad (bool, optional): Whether or not the token dispatcher uses token-drop
                                       and pads the number of tokens to the expert capacity.
    """
    assert not drop_and_pad, "token-drop and pads is not supported"
    num_tokens, hidden = tokens.shape
    num_experts = routing_map.shape[1]

    # mask [num_tokens, num_experts] -> [num_experts, num_tokens]
    routing_map = routing_map.cast(paddle.bool).T.contiguous()

    # Create a dense expert-to-token mapping from the sparse token-to-expert mapping
    token_indices = (
        paddle.arange(num_tokens).unsqueeze(0).expand([num_experts, -1])
    )
    sorted_indices = token_indices.masked_select(routing_map)

    # use the mapping to permute the tokens
    permuted_input = tokens.index_select(axis=0, index=sorted_indices)

    return permuted_input, sorted_indices


def unpermute(
    permuted_tokens: paddle.Tensor,
    sorted_indices: paddle.Tensor,
    restore_shape: paddle.shape,
    probs: paddle.Tensor = None,
    routing_map: paddle.Tensor = None,
    drop_and_pad: bool = False,
):
    """
    Restore the original order of tokens after permutation. If probs are provided, it
    will also apply them to the tokens before restoring the order.

    Args:
        permuted_tokens (paddle.Tensor): The permuted token tensor.
        sorted_indices (paddle.Tensor): The indices used to sort the tokens.
        restore_shape (paddle.shape): The shape of the unpermuted tensor.
        probs (paddle.Tensor, optional): The unpermuted probs tensor,
        routing_map (paddle.Tensor, optional): Token to expert mapping, shape
            [num_tokens, num_experts].
        drop_and_pad (bool, optional): Whether or not the token dispatcher uses token-drop
                                       and pads the number of tokens to the expert capacity.

    Returns:
        paddle.Tensor: The tokens restored to their original order.
    """
    assert not drop_and_pad, "token-drop and pads is not supported"
    _, hidden = restore_shape

    if probs is not None:
        assert routing_map is not None, (
            "Mask must be provided to permute the probs."
        )
        permuted_probs = probs.T.contiguous().masked_select(
            routing_map.T.contiguous().cast(paddle.bool)
        )
        permuted_tokens = permuted_tokens * permuted_probs.unsqueeze(-1)

    # Create an output tensor filled with zeros
    output_tokens = paddle.zeros(restore_shape, dtype=permuted_tokens.dtype)
    # Scatter add the permuted_input back to the original positions
    # if scatter_add_ is not None:
    #     # NOTE: this expand will cause a big memory usage, so disable this method
    #     sorted_indices = sorted_indices.unsqueeze(1).expand(-1, hidden)
    #     output_tokens.scatter_add_(0, sorted_indices, permuted_tokens)
    # else:
    # NOTE: Calling multiple times of scatter_ will not accumulate,
    # Instead, it reset to zero and then accumulated again.
    # so can't do subbatch here.
    output_tokens.scatter_(
        index=sorted_indices, updates=permuted_tokens, overwrite=False
    )
    return output_tokens


class AddAuxiliaryLoss(paddle.autograd.PyLayer):
    """
    The trick function of adding auxiliary (aux) loss,
    which includes the gradient of the aux loss during backpropagation.
    """

    @staticmethod
    def forward(ctx, x, loss):
        assert paddle.numel(loss) == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = not loss.stop_gradient
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = paddle.ones(1, dtype=ctx.dtype)
        return grad_output, grad_loss


class _AllToAll(paddle.autograd.PyLayer):
    @staticmethod
    def forward(
        ctx: Any,
        output_shape: list,
        input: Tensor,
        out_split_sizes: list | None = None,
        in_split_sizes: list | None = None,
        group: Group = None,
    ) -> Tensor:  # type: ignore
        """
        All-to-all communication in the group.
        Args:
            ctx (Any): Context object.
            output_shape (list): Output shape.
            input (Tensor): Input tensor.
            out_split_sizes (list): Output split sizes.
            in_split_sizes (list): Input split sizes.
            group (Group): The group object.
        Returns:
            Tensor: Output tensor.
        """

        ctx.group = group
        ctx.input_shape = input.shape
        ctx.out_split_sizes = out_split_sizes
        ctx.in_split_sizes = in_split_sizes

        # return input
        if dist.get_world_size(group) <= 1:
            return input

        output = paddle.empty(
            output_shape, dtype=input.dtype, requires_grad=True
        )
        task = dist.alltoall_single(
            output,
            input,
            out_split_sizes=out_split_sizes,
            in_split_sizes=in_split_sizes,
            sync_op=False,
            group=group,
        )
        task.wait()

        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> tuple[Tensor]:
        """
        Aggregates gradient information from all input tensors into a single tensor.
        Args:
            ctx (Any): The context object used to store information that needs to be passed.
            *grad_output (Tensor): A list of input tensors whose gradients are to be aggregated.
        Returns:
            tuple[Tensor]: A tuple containing a tensor that holds the gradients of all input tensors.
        """
        # return grad_output
        return _AllToAll.apply(
            ctx.input_shape,
            *grad_output,
            ctx.in_split_sizes,
            ctx.out_split_sizes,
            ctx.group,
        )


class RandomSTE(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x):
        ctx.x_shape = x.shape
        ctx.x_dtype = x.dtype
        return paddle.randn(x.shape).cast(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return paddle.zeros(ctx.x_shape, dtype=ctx.x_dtype)


def apply_random_logits(logits):
    """
    Apply the RandomSTE function to the logits.
    """
    return RandomSTE.apply(logits)


def is_tensor(data):
    """Check if data is a tensor"""
    return isinstance(data, (paddle.Tensor, paddle.base.core.eager.Tensor))


def detach_and_requires_grad_(*args):
    """Detach tensors and preserve their requires_grad settings"""
    ret = [a.detach() if is_tensor(a) else a for a in args]
    for r, a in zip(ret, args):
        if is_tensor(a):
            r.stop_gradient = a.stop_gradient
    return ret


class FakeClone(paddle.autograd.PyLayer):
    """
    In manual_backward, in order to preserve the local computation graph for temporary backward computation,
    we need to clone the output of manual_backward. This clone operation essentially doesn't need the value
    of the output, but rather needs to obtain the computation graph attached to the output.

    However, calling paddle.clone would perform an unnecessary data copy.
    FakeClone avoids this data copy and achieves the goal of extracting the computation graph.
    """

    @staticmethod
    def forward(ctx, input):
        """Forward pass"""
        if input.is_contiguous():
            fake_output = paddle.Tensor()
            fake_output.get_tensor()._share_data_nocheck_with(
                input.get_tensor()
            )
        else:
            fake_output = input.clone()
        return fake_output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass"""
        return grad_output


def manual_backward(f: Callable, is_first_fwd: bool, *args: list[Any]):
    """
    Args:
        f(callable)
        args(*Any)
    Returns
        bw_f(callable): manual backward fn
        out(List[Tensor]): output of f(*args)
    """
    tracer = framework._dygraph_tracer()
    orig = tracer._has_grad
    if not is_first_fwd:
        tracer._has_grad = True  # turn on grad trace so we can manual backward

    detached_args = detach_and_requires_grad_(*args)
    detached_args_clone = [
        FakeClone.apply(a) if is_tensor(a) else a for a in detached_args
    ]
    out = f(*detached_args_clone)
    if isinstance(out, list):
        out = tuple(out)
    elif not isinstance(out, tuple):
        out = (out,)

    if is_first_fwd:
        tracer._has_grad = orig
        return None, out

    out_cached = [
        FakeClone.apply(o) for o in out if o is not None
    ]  # do not cache stop_gradient output

    for o in out_cached:
        o._clear_dataptr()  # free mem
    tracer._has_grad = orig

    def bwd_f(*grad):
        nonlocal out_cached, detached_args, f
        grad = list(grad)
        grad = [g for g in grad if g is not None]
        assert grad and out_cached, (len(grad), len(out_cached))
        grad, out_cached = zip(
            *[(g, o) for g, o in zip(grad, out_cached) if not o.stop_gradient]
        )

        assert len(grad) == len(out_cached), (len(grad), len(out_cached), f)

        paddle.autograd.backward(out_cached, grad)
        return tuple([t.grad for t in detached_args if is_tensor(t)])

    return bwd_f, out
