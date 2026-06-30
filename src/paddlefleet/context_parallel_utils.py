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

import inspect

import paddle
from paddle import distributed as dist
from paddle.autograd.py_layer import PyLayer
from paddle.distributed import fleet
from paddle.nn.functional.flash_attention import flashmask_attention

_flash_mask_available = False
try:
    if (
        paddle.cuda.is_available()
        and paddle.cuda.get_device_capability()[0] == 10
    ):
        from paddlefleet_ops.flash_mask.cute.flashmask_utils import (
            FlashMaskInfoPaddle,
        )
        from paddlefleet_ops.flash_mask.cute.interface import (
            _flash_attn_bwd,
            _flash_attn_fwd,
        )

        _flash_mask_available = True
except (ImportError, AttributeError):
    _flash_mask_available = False


def mark_context_parallel_parameter_disable_scale_grad(param_or_layer):
    """
    Mark parameters or layers to disable context parallel gradient scaling.

    This function sets the attribute `context_parallel_disable_scale_grad` to `True` for the given parameter,
    tensor, or layer. When set, this flag indicates that the specified parameter or layer should not have
    its gradient scaled during context parallel training.
    - If a `paddle.nn.Layer` is provided, both its `weight` and (if present) `bias` will be marked.
    - If a `paddle.base.framework.Parameter` or `paddle.Tensor` is provided, it will be marked directly.
    - Raises a `TypeError` if the input is not a supported type.
    Args:
        param_or_layer (paddle.nn.Layer or paddle.base.framework.Parameter or paddle.Tensor):
            The parameter, tensor, or layer to mark as disabling context parallel gradient scaling.
    Raises:
        TypeError: If `param_or_layer` is not a `Parameter`, `Tensor`, or `Layer`.
    Example:
        >>> mark_context_parallel_parameter_disable_scale_grad(layer)
        >>> mark_context_parallel_parameter_disable_scale_grad(param)
    """

    if isinstance(param_or_layer, paddle.nn.Layer):
        param_or_layer.weight.context_parallel_disable_scale_grad = True
        if hasattr(param_or_layer, "bias") and param_or_layer.bias is not None:
            param_or_layer.bias.context_parallel_disable_scale_grad = True
    elif isinstance(
        param_or_layer, (paddle.base.framework.Parameter, paddle.Tensor)
    ):
        param_or_layer.context_parallel_disable_scale_grad = True
    else:
        raise TypeError(
            f"param should be 'Parameter' or 'Tensor' or 'Layer', but received {type(param_or_layer)}"
        )


def context_parallel_parameter_disable_scale_grad(param):
    """
    Check whether context parallel gradient scaling is disabled for the parameter or tensor.
    Returns the value of the `context_parallel_disable_scale_grad` attribute for the given parameter or tensor.
    If the attribute is not set, returns `False` by default.
    Args:
        param (paddle.base.framework.Parameter or paddle.Tensor):
            The parameter or tensor to check.
    Returns:
        bool: True if context parallel gradient scaling is disabled, False otherwise.
    Example:
        >>> if context_parallel_parameter_disable_scale_grad(param):
        ...     # Handle parameter that should not have its gradient scaled
        ...     pass
    """
    return getattr(param, "context_parallel_disable_scale_grad", False)


def scatter_balance(input_tensor, group=None, axis=0):
    """
    Evenly split input tensor along the specified axis across model parallel ranks.
    This function implements balanced scattering by taking chunks from both ends
    of the tensor to ensure load balancing across ranks.
    Args:
        input_tensor (paddle.Tensor): Input tensor to be scattered
        group (paddle.distributed.Group, optional): Communication group.
            If None, uses model parallel group from fleet
        axis (int, optional): Axis along which to scatter. Defaults to 0
    Returns:
        paddle.Tensor: Scattered tensor chunk for current rank
    Note:
        This API is different from distributed.scatter - it performs balanced
        splitting by taking chunks from both ends of the sequence.
    """
    if group is None:
        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_model_parallel_group()

    parallelism = group.nranks
    if parallelism == 1:
        return input_tensor.clone()

    rank = group.rank
    seq_len = input_tensor.shape[axis]

    # Ensure sequence length is divisible by parallelism * 2 for balanced splitting
    assert seq_len % (parallelism * 2) == 0, (
        f"Input sequence length {seq_len} can't be divided exactly by sequence parallelism * 2 {parallelism * 2}"
    )

    interval = seq_len // parallelism // 2
    total_len = input_tensor.shape[axis]

    # Take chunk from the beginning
    chunk_start = paddle.slice(
        input_tensor,
        axes=[axis],
        starts=[interval * rank],
        ends=[interval * (rank + 1)],
    )

    # Take chunk from the end (in reverse order)
    chunk_end = paddle.slice(
        input_tensor,
        axes=[axis],
        starts=[total_len - interval * (rank + 1)],
        ends=[total_len - interval * rank],
    )

    # Concatenate chunks
    result = paddle.concat([chunk_start, chunk_end], axis=axis)

    # Use assign to free the memory of the whole input tensor to avoid OOM
    # since slice uses stride and maintains reference to original tensor
    result = paddle.assign(result)
    return result


def all_gather_balance(input_tensor, group=None, axis=0):
    """
    Balanced all-gather operation using Triton reorder kernel.

    Gathers tensors from all ranks via all_gather, then reorders the gathered data
    using a Triton kernel (balanced_gather_reorder_kernel) to reconstruct the original
    sequence order from the DualChunkSwap balanced layout. Each rank's local tensor
    contains two chunks (one from the start, one from the end of the sequence), and
    this function reassembles them into the full contiguous sequence.

    This is the inverse of reduce_scatter_any_axis_balance and scatter_balance.

    Args:
        input_tensor (paddle.Tensor): Local tensor chunk to gather. Each rank's
            chunk size along `axis` must be even (split into two halves by the
            balanced strategy).
        group (paddle.distributed.Group, optional): Communication group. If None,
            uses the model parallel group from fleet.
        axis (int, optional): Axis along which to gather and reorder. Defaults to 0.

    Returns:
        paddle.Tensor: Full gathered tensor with shape[axis] = input_shape[axis] * parallelism,
            reordered to restore the original sequence order.
    """
    import triton

    from paddlefleet.triton_ops.balanced_reorder import (
        balanced_gather_reorder_kernel,
    )

    if group is None:
        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_model_parallel_group()

    parallelism = group.nranks
    if parallelism == 1:
        return input_tensor.clone()

    # Single all_gather (gathers along axis=0)
    shape = list(input_tensor.shape)
    gathered_shape = list(shape)
    gathered_shape[0] = shape[0] * parallelism
    gathered = paddle.empty(gathered_shape, dtype=input_tensor.dtype)
    dist.stream.all_gather(
        gathered, input_tensor.contiguous(), group=group, use_calc_stream=True
    )

    # Compute strides for reorder kernel
    axis_size = shape[axis]
    chunk_size = axis_size // 2
    N = parallelism

    # outer_size: product of all dims left of axis in the *original* (per-rank) shape
    outer_size = 1
    for i in range(axis):
        outer_size *= shape[i]

    # inner_size: product of all dims right of axis
    inner_size = 1
    for i in range(axis + 1, len(shape)):
        inner_size *= shape[i]

    # src is gathered along axis=0: shape = [N*S0, S1, ..., S_axis, ..., S_last]
    # src_rank_stride = elements per rank = product of original shape
    src_rank_stride = 1
    for s in shape:
        src_rank_stride *= s

    # src_outer_stride = elements to skip per outer index = S_axis * inner_size
    src_outer_stride = axis_size * inner_size

    out_shape = list(shape)
    out_shape[axis] = 2 * N * chunk_size
    output = paddle.empty(out_shape, dtype=input_tensor.dtype)

    BLOCK_SIZE = 1024
    num_blocks_per_chunk = triton.cdiv(chunk_size * inner_size, BLOCK_SIZE)
    grid = (num_blocks_per_chunk * 2 * N, outer_size, 1)

    balanced_gather_reorder_kernel[grid](
        gathered,
        output,
        N,
        chunk_size,
        inner_size,
        src_rank_stride,
        src_outer_stride,
        num_blocks_per_chunk,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


def reduce_scatter_any_axis(input_tensor, axis, group=None):
    """
    Reduce-scatter operation along any axis.
    Performs element-wise reduction (sum) across ranks and scatters the result
    so each rank gets a portion of the reduced tensor.
    Args:
        input_tensor (paddle.Tensor): Input tensor to reduce and scatter
        axis (int): Axis along which to perform reduce-scatter
        group (paddle.distributed.Group, optional): Communication group
    Returns:
        paddle.Tensor: Reduced and scattered tensor chunk
    """
    if group is None:
        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_context_parallel_group()

    parallelism = group.nranks
    if parallelism == 1:
        return input_tensor.clone()

    assert input_tensor.shape[axis] % parallelism == 0, (
        f"Input sequence length {input_tensor.shape[axis]} can't be ",
        f"divided exactly by context parallelism {parallelism}",
    )

    if axis == 0:
        # Optimized path for axis=0
        output_shape = list(input_tensor.shape)
        output_shape[0] = output_shape[0] // parallelism

        output = paddle.empty(shape=output_shape, dtype=input_tensor.dtype)
        dist.stream.reduce_scatter(
            output,
            input_tensor,
            op=dist.ReduceOp.SUM,
            group=group,
            use_calc_stream=True,
        )
        return output
    else:
        # General case for other axes using alltoall
        input_chunks = paddle.split(input_tensor, parallelism, axis=axis)

        output_buffers = [
            paddle.empty(input_chunks[0].shape, dtype=input_tensor.dtype)
            for _ in range(parallelism)
        ]

        dist.stream.alltoall(
            output_buffers, input_chunks, group=group, use_calc_stream=True
        )

        # Sum the received chunks
        result = paddle.stack(output_buffers, axis=0).sum(axis=0)
        return result


def reduce_scatter_any_axis_balance(input_tensor, axis, group=None):
    """
    Balanced reduce-scatter operation along any axis using Triton reorder kernel.

    Performs reduce-scatter with the DualChunkSwap balanced strategy: first reorders
    the input tensor via a Triton kernel (balanced_scatter_reorder_kernel) to prepare
    balanced chunks for each rank, then uses alltoall_single to exchange data, and
    finally sums the received chunks to produce the reduced result.

    This is the inverse of all_gather_balance and is used in backward passes of
    context parallel attention (e.g., to reduce-scatter key/value gradients).

    Args:
        input_tensor (paddle.Tensor): Input tensor to reduce and scatter. The size
            along `axis` must be divisible by (parallelism * 2).
        axis (int): Axis along which to perform the balanced reduce-scatter.
        group (paddle.distributed.Group, optional): Communication group. If None,
            uses the context parallel group from fleet.

    Returns:
        paddle.Tensor: Reduced tensor with shape[axis] = input_shape[axis] / parallelism.
    """
    import triton

    from paddlefleet.triton_ops.balanced_reorder import (
        balanced_scatter_reorder_kernel,
    )

    if group is None:
        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_context_parallel_group()

    parallelism = group.nranks
    if parallelism == 1:
        return input_tensor.clone()

    N = parallelism
    shape = list(input_tensor.shape)

    assert shape[axis] % (N * 2) == 0, (
        f"Input sequence length {shape[axis]} can't be "
        f"divided exactly by context parallelism * 2 {N * 2}"
    )

    chunk_size = shape[axis] // (2 * N)

    outer_size = 1
    for i in range(axis):
        outer_size *= shape[i]

    inner_size = 1
    for i in range(axis + 1, len(shape)):
        inner_size *= shape[i]

    src_outer_stride = shape[axis] * inner_size
    dst_outer_stride = 2 * chunk_size * inner_size
    dst_rank_stride = outer_size * dst_outer_stride

    per_rank_shape = list(shape)
    per_rank_shape[axis] = 2 * chunk_size
    # send_buf: [N, *per_rank_shape], contiguous, kernel writes into it
    send_buf = paddle.empty([N, *per_rank_shape], dtype=input_tensor.dtype)

    BLOCK_SIZE = 1024
    num_blocks_per_chunk = triton.cdiv(chunk_size * inner_size, BLOCK_SIZE)
    grid = (num_blocks_per_chunk * 2 * N, outer_size, 1)

    balanced_scatter_reorder_kernel[grid](
        input_tensor,
        send_buf,
        N,
        chunk_size,
        inner_size,
        src_outer_stride,
        dst_rank_stride,
        dst_outer_stride,
        num_blocks_per_chunk,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # alltoall_single: send_buf[r] -> rank r's recv_buf[my_rank]
    recv_buf = paddle.empty_like(send_buf)
    dist.stream.alltoall_single(
        recv_buf.reshape([-1]),
        send_buf.reshape([-1]),
        group=group,
        use_calc_stream=True,
    )

    # sum across N received chunks: same order as original stack+sum
    result = recv_buf.reshape([N, *per_rank_shape]).sum(axis=0)
    return result


def scatter_contiguous(input_tensor, group=None, axis=0):
    """Contiguous scatter: rank r gets slice [r*chunk, (r+1)*chunk] along axis."""
    if group is None:
        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_context_parallel_group()
    nranks = group.nranks
    if nranks == 1:
        return input_tensor.clone()
    rank = group.rank
    chunk_size = input_tensor.shape[axis] // nranks
    result = paddle.slice(
        input_tensor,
        axes=[axis],
        starts=[rank * chunk_size],
        ends=[(rank + 1) * chunk_size],
    )
    return paddle.assign(result)


def all_gather_contiguous(input_tensor, group=None, axis=0):
    """Contiguous all-gather: concatenate all ranks' local tensors in rank order."""
    if group is None:
        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_context_parallel_group()
    nranks = group.nranks
    if nranks == 1:
        return input_tensor.clone()
    if axis == 0:
        shape = list(input_tensor.shape)
        shape[0] *= nranks
        gathered = paddle.empty(shape=shape, dtype=input_tensor.dtype)
        dist.stream.all_gather(
            gathered,
            input_tensor.contiguous(),
            group=group,
            use_calc_stream=True,
        )
        return gathered
    else:
        tensor_list = [
            paddle.empty(input_tensor.shape, dtype=input_tensor.dtype)
            for _ in range(nranks)
        ]
        dist.stream.all_gather(
            tensor_list,
            input_tensor.contiguous(),
            group=group,
            use_calc_stream=True,
        )
        return paddle.concat(tensor_list, axis=axis)


def reduce_scatter_contiguous(input_tensor, axis, group=None):
    """Contiguous reduce-scatter: reduce_scatter for axis=0, alltoall+sum otherwise."""
    if group is None:
        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_context_parallel_group()
    nranks = group.nranks
    if nranks == 1:
        return input_tensor.clone()
    if axis == 0:
        output_shape = list(input_tensor.shape)
        output_shape[0] //= nranks
        output = paddle.empty(shape=output_shape, dtype=input_tensor.dtype)
        dist.stream.reduce_scatter(
            output,
            input_tensor.contiguous(),
            op=dist.ReduceOp.SUM,
            group=group,
            use_calc_stream=True,
        )
        return output
    else:
        chunks = paddle.split(input_tensor, nranks, axis=axis)
        bufs = [
            paddle.empty(chunks[0].shape, dtype=input_tensor.dtype)
            for _ in range(nranks)
        ]
        dist.stream.alltoall(
            bufs,
            [c.contiguous() for c in chunks],
            group=group,
            use_calc_stream=True,
        )
        return (
            paddle.stack(bufs).cast("float32").sum(0).cast(input_tensor.dtype)
        )


class ContextParallelScatterOp(PyLayer):
    """
    Context parallel scatter operation using PyLayer for automatic differentiation.
    Forward: Scatter input tensor (balanced or contiguous based on mode)
    Backward: All-gather gradients (inverse of forward scatter)
    """

    @staticmethod
    def forward(ctx, input_tensor, axis=0, mode="dualchunk_allgather"):
        ctx.axis = axis
        ctx.mode = mode
        hcg = fleet.get_hybrid_communicate_group()

        assert hcg.get_context_parallel_world_size() > 1, (
            "ScatterOpCP must be used with context parallel, ",
            f"context_parallel_world_size={hcg.get_context_parallel_world_size()}",
        )

        group = hcg.get_context_parallel_group()
        ctx.group = group

        if mode == "contiguous_allgather":
            return scatter_contiguous(input_tensor, group=group, axis=axis)
        return scatter_balance(input_tensor, axis=axis, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.mode == "contiguous_allgather":
            return all_gather_contiguous(
                grad_output, group=ctx.group, axis=ctx.axis
            )
        return all_gather_balance(grad_output, axis=ctx.axis, group=ctx.group)


class ContextParallelGatherOp(PyLayer):
    """
    Context parallel gather operation using PyLayer for automatic differentiation.
    Forward: All-gather input tensor (balanced or contiguous based on mode)
    Backward: Scatter gradients (inverse of forward gather)
    """

    @staticmethod
    def forward(ctx, input_tensor, axis=0, mode="dualchunk_allgather"):
        ctx.axis = axis
        ctx.mode = mode
        hcg = fleet.get_hybrid_communicate_group()

        assert hcg.get_context_parallel_world_size() > 1, (
            "GatherOpCP must be used with context parallel, ",
            f"context_parallel_world_size={hcg.get_context_parallel_world_size()}",
        )

        group = hcg.get_context_parallel_group()
        ctx.group = group

        if mode == "contiguous_allgather":
            return all_gather_contiguous(input_tensor, group=group, axis=axis)
        return all_gather_balance(input_tensor, axis=axis, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.mode == "contiguous_allgather":
            return scatter_contiguous(
                grad_output, group=ctx.group, axis=ctx.axis
            )
        return scatter_balance(grad_output, axis=ctx.axis, group=ctx.group)


class ContextParallelAllGatherOp(PyLayer):
    """
    Context parallel all-gather operation with gradient reduction.
    Forward: All-gather input tensor (balanced or contiguous based on mode)
    Backward: Reduce-scatter gradients (sum + scatter)
    """

    @staticmethod
    def forward(ctx, input_tensor, axis, mode="dualchunk_allgather"):
        ctx.axis = axis
        ctx.mode = mode
        hcg = fleet.get_hybrid_communicate_group()

        assert hcg.get_context_parallel_world_size() > 1, (
            "AllGatherOpCP must be used with context parallel, ",
            f"context_parallel_world_size={hcg.get_context_parallel_world_size()}",
        )

        group = hcg.get_context_parallel_group()
        ctx.group = group

        if mode == "contiguous_allgather":
            return all_gather_contiguous(input_tensor, group=group, axis=axis)
        return all_gather_balance(input_tensor, axis=axis, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.mode == "contiguous_allgather":
            return reduce_scatter_contiguous(
                grad_output, axis=ctx.axis, group=ctx.group
            )
        return reduce_scatter_any_axis_balance(
            grad_output, axis=ctx.axis, group=ctx.group
        )


def preprocess_index(
    startend_row_indices, chunk_id, seq_blocksize, max_seqlen_q
):
    """
    Preprocess startend row indices for a single chunk.
    Adjusts the startend_row_indices relative to the chunk's starting position and
    clips them to valid range.
    Args:
        startend_row_indices (paddle.Tensor): Original startend row indices
        chunk_id (int): ID of the current chunk
        seq_blocksize (int): Size of each sequence block
        max_seqlen_q (int): Maximum sequence length for queries
    Returns:
        paddle.Tensor: Preprocessed row indices
    """
    rows_min = chunk_id * seq_blocksize
    adjusted_indices = startend_row_indices - rows_min
    clipped_indices = paddle.clip(adjusted_indices, min=0, max=max_seqlen_q)
    return clipped_indices


def preprocess_index_dual_chunks(
    startend_row_indices,
    chunk_id_first,
    chunk_id_second,
    seq_blocksize,
    max_seqlen_q,
):
    """
    Preprocess row indices for dual chunks (DualChunkSwap strategy).
    This function handles the index preprocessing for the balanced dual-chunk
    strategy where each rank processes chunks from both ends of the sequence.
    Args:
        startend_row_indices (paddle.Tensor): Original row indices
        chunk_id_first (int): ID of the first chunk
        chunk_id_second (int): ID of the second chunk
        seq_blocksize (int): Size of each sequence block
        max_seqlen_q (int): Maximum sequence length for queries
    Returns:
        paddle.Tensor: Preprocessed row indices for dual chunks
    """
    # Calculate starting positions for both chunks
    rows_min_first = chunk_id_first * seq_blocksize
    rows_min_second = chunk_id_second * seq_blocksize

    # Process first chunk indices
    indices_first = startend_row_indices - rows_min_first
    indices_first = paddle.clip(indices_first, min=0, max=max_seqlen_q)

    # Process second chunk indices
    indices_second = startend_row_indices - rows_min_second
    indices_second = paddle.clip(indices_second, min=0, max=max_seqlen_q)

    # Offset second chunk indices to avoid overlap
    indices_second = paddle.where(
        indices_second != 0, indices_second + max_seqlen_q, indices_second
    )

    # Combine indices from both chunks
    combined_indices = paddle.maximum(indices_first, indices_second)
    return combined_indices


def cp_flashmask_allgatherkv_balance_forward(
    query,
    key,
    value,
    startend_row_indices,
    learnable_sink,
    group,
    causal,
    is_training,
):
    """
    Forward pass of context parallel flashmask attention with balanced all-gather strategy.
    This function implements the forward pass of flash attention with context parallelism
    using the DualChunkSwap strategy for load balancing.
    Args:
        query (paddle.Tensor): Query tensor with shape [batch, seq_len/n, num_heads, head_dim]
        key (paddle.Tensor): Key tensor with shape [batch, seq_len/n, num_heads, head_dim]
        value (paddle.Tensor): Value tensor with shape [batch, seq_len/n, num_heads, head_dim]
        startend_row_indices (paddle.Tensor): Row indices for attention mask
        group (paddle.distributed.Group): Communication group
        causal (bool): Whether to use causal attention
        is_training (bool): Whether in training mode
    Returns:
        tuple: (output, log_sum_exp, processed_indices, fa_version)
            ``fa_version`` is the effective FlashAttention version actually
            used by the forward kernel and must be passed to the backward
            counterpart to keep fwd/bwd consistent.
    """
    paddle.base.core.nvprof_nvtx_push(
        "cp_flashmask_allgatherkv_balance_forward"
    )

    rank = group.rank
    cp_size = group.world_size

    # All-gather key tensors across context parallel ranks
    key_gathered = all_gather_balance(key, axis=1, group=group)

    # All-gather value tensors across context parallel ranks
    value_gathered = all_gather_balance(value, axis=1, group=group)

    # Calculate sequence block size for dual-chunk strategy
    seq_blocksize = query.shape[1] // 2

    # Preprocess indices for dual-chunk strategy
    startend_row_indices = preprocess_index_dual_chunks(
        startend_row_indices,
        chunk_id_first=rank,
        chunk_id_second=2 * cp_size - rank - 1,
        seq_blocksize=seq_blocksize,
        max_seqlen_q=seq_blocksize,
    )

    # Perform flashmask attention with startend_row_indices
    fa_version = paddle.base.framework.get_flags(["FLAGS_flash_attn_version"])[
        "FLAGS_flash_attn_version"
    ]
    # Apply deterministic override here so forward and backward use the same
    # effective fa_version (mirrors backward's previous logic and the
    # framework flashmask_attention's internal deterministic fallback).
    deterministic = paddle.get_flags(["FLAGS_cudnn_deterministic"])[
        "FLAGS_cudnn_deterministic"
    ]
    if "block_mask" in inspect.signature(flashmask_attention).parameters:
        if deterministic and query.shape[-1] > 128:
            fa_version = 2
    elif deterministic:
        fa_version = 2

    if fa_version == 4 and _flash_mask_available:
        output, log_sum_exp = _flash_attn_fwd(
            query,
            key_gathered,
            value_gathered,
            causal=causal,
            return_lse=True,
            startend_row_indices=startend_row_indices,
            learnable_sink=learnable_sink,
            pack_gqa=False,
        )
    else:
        if learnable_sink is not None:
            raise NotImplementedError(
                "learnable_sink only supported on fa_version==4 cute backend"
            )
        output, log_sum_exp = flashmask_attention(
            query,
            key_gathered,
            value_gathered,
            startend_row_indices=startend_row_indices,
            causal=causal,
            return_softmax_lse=True,
            training=is_training,
        )

    paddle.base.core.nvprof_nvtx_pop()
    return output, log_sum_exp, startend_row_indices, fa_version


def cp_flashmask_allgatherkv_balance_backward(
    query,
    key,
    value,
    startend_row_indices,
    output,
    log_sum_exp,
    output_grad,
    learnable_sink,
    group,
    causal,
    fa_version: int,
):
    """
    Backward pass of context parallel flashmask attention with balanced all-gather strategy.
    This function implements the backward pass of flashmask attention with context parallelism,
    computing gradients for query, key, and value tensors.
    Args:
        query (paddle.Tensor): Query tensor
        key (paddle.Tensor): Key tensor
        value (paddle.Tensor): Value tensor
        startend_row_indices (paddle.Tensor): Processed startend_row_indices
        output (paddle.Tensor): Forward pass output
        log_sum_exp (paddle.Tensor): Log-sum-exp from forward pass
        output_grad (paddle.Tensor): Gradient of output
        group (paddle.distributed.Group): Communication group
        causal (bool): Whether causal attention was used
        fa_version (int): FlashAttention version that was actually used by the
            forward kernel. Must be propagated from the forward call to keep
            fwd/bwd consistent.
    Returns:
        tuple: (query_grad, key_grad, value_grad, grad_sink)
    """
    paddle.base.core.nvprof_nvtx_push(
        "cp_flashmask_allgatherkv_balance_backward"
    )

    # All-gather key and value tensors (same as forward pass)
    key_gathered = all_gather_balance(key, axis=1, group=group)
    value_gathered = all_gather_balance(value, axis=1, group=group)

    grad_sink = None
    if fa_version == 2:
        if learnable_sink is not None:
            raise NotImplementedError(
                "learnable_sink only supported on fa_version==4 cute backend"
            )
        # Create seed offset tensor (required for gradient computation)
        seed_offset = paddle.zeros(
            shape=[query.shape[1], query.shape[2]], dtype=paddle.int64
        )

        # Compute gradients using flashmask attention backward pass
        query_grad, key_grad_gathered, value_grad_gathered = (
            paddle._C_ops.flashmask_attention_grad(
                query,
                key_gathered,
                value_gathered,
                startend_row_indices,
                output,
                log_sum_exp,
                seed_offset,
                output_grad,
                0.0,  # dropout probability
                causal,
            )
        )
    elif fa_version == 3:
        if learnable_sink is not None:
            raise NotImplementedError(
                "learnable_sink only supported on fa_version==4 cute backend"
            )
        sig_params = inspect.signature(flashmask_attention).parameters
        if "group" in sig_params:
            query_grad, key_grad_gathered, value_grad_gathered = (
                paddle._C_ops.flashmask_attention_v2_grad(
                    query,
                    key_gathered,
                    value_gathered,
                    output,
                    log_sum_exp,
                    startend_row_indices,
                    None,  # block_mask
                    output_grad,
                    query.shape[-1] ** (-0.5),
                    False,
                    0,  # rank
                    1,  # nranks
                )
            )
        elif "block_mask" in sig_params:
            query_grad, key_grad_gathered, value_grad_gathered = (
                paddle._C_ops.flashmask_attention_v2_grad(
                    query,
                    key_gathered,
                    value_gathered,
                    output,
                    log_sum_exp,
                    startend_row_indices,
                    None,  # block_mask
                    output_grad,
                    query.shape[-1] ** (-0.5),
                    False,
                )
            )
        else:
            query_grad, key_grad_gathered, value_grad_gathered = (
                paddle._C_ops.flashmask_attention_v2_grad(
                    query,
                    key_gathered,
                    value_gathered,
                    output,
                    log_sum_exp,
                    startend_row_indices,
                    output_grad,
                    query.shape[-1] ** (-0.5),
                    False,
                )
            )
    elif fa_version == 4 and _flash_mask_available:
        if startend_row_indices is not None:
            flashmask_info = FlashMaskInfoPaddle(
                startend_row_indices=startend_row_indices,
                is_causal=causal,
            )
        else:
            flashmask_info = None
        query_grad, key_grad_gathered, value_grad_gathered, grad_sink = (
            _flash_attn_bwd(
                query,
                key_gathered,
                value_gathered,
                output,
                output_grad,
                log_sum_exp,
                flashmask_info,
                learnable_sink=learnable_sink,
                causal=causal,
                deterministic=paddle.get_flags(["FLAGS_cudnn_deterministic"])[
                    "FLAGS_cudnn_deterministic"
                ],
            )
        )
    else:
        raise ValueError(
            f"FlashAttention version {fa_version} is not supported."
        )

    # Reduce-scatter key and value gradients
    key_grad = reduce_scatter_any_axis_balance(
        key_grad_gathered, axis=1, group=group
    )
    value_grad = reduce_scatter_any_axis_balance(
        value_grad_gathered, axis=1, group=group
    )

    paddle.base.core.nvprof_nvtx_pop()
    return query_grad, key_grad, value_grad, grad_sink


def scatter_with_padding(input_tensor, num_pad, axis, group):
    """scatter_with_padding"""
    cp_degree = group.nranks
    cp_rank = group.rank

    total_num = input_tensor.shape[axis]
    avg_num = (total_num + num_pad) // cp_degree

    split_sections = []
    cnt = 0
    rank_idx = 0
    rank_pad = 0
    for _ in range(0, cp_degree):
        if cnt + avg_num < total_num:
            split_sections.append(avg_num)
        elif cnt < total_num:
            split_sections.append(total_num - cnt)
            rank_pad = avg_num - total_num + cnt
        else:
            break
        cnt += avg_num
        rank_idx += 1

    if cp_rank < rank_idx:
        list_of_res = paddle.split(input_tensor, num_or_sections=split_sections)
        cur_res = list_of_res[cp_rank]
        if rank_pad > 0 and cp_rank == rank_idx - 1:
            pad_list = [0 for _ in range(0, input_tensor.ndim * 2)]
            pad_list[axis * input_tensor.ndim * 2 + 1] = rank_pad
            cur_res = paddle.nn.functional.pad(
                cur_res, pad_list, mode="constant", value=0
            )
    else:
        shape = input_tensor.shape
        shape[axis] = avg_num
        cur_res = paddle.zeros(shape, input_tensor.dtype)
        cur_res.stop_gradient = False
    return cur_res


def all_gather_without_padding(input_tensor, num_pad, axis, group):
    """all_gather_without_padding"""
    output_shape = list(input_tensor.shape)
    output_shape[axis] = output_shape[axis] * group.nranks
    output_tensor = paddle.empty(shape=output_shape, dtype=input_tensor.dtype)
    dist.stream.all_gather(output_tensor, input_tensor, group)
    if num_pad > 0:
        pad_start = output_tensor.shape[axis] - num_pad
        output_tensor = paddle.slice(
            output_tensor, axes=[axis], starts=[0], ends=[pad_start]
        )
    return output_tensor


class ContextParallelNormalScatter(PyLayer):
    """ContextParallelNormalScatter"""

    @staticmethod
    def forward(ctx, input_tensor, num_pad, axis=0):
        """forward"""
        ctx.axis = axis
        hcg = fleet.get_hybrid_communicate_group()
        cp_degree = hcg.get_context_parallel_world_size()

        if cp_degree == 1:
            return input_tensor.clone()

        group = hcg.get_context_parallel_group()
        ctx.group = group
        ctx.num_pad = num_pad
        ctx.axis = axis

        return scatter_with_padding(input_tensor, num_pad, axis, ctx.group)

    @staticmethod
    def backward(ctx, grad_output):
        """backward"""
        if ctx.group.nranks == 1:
            return grad_output.clone()

        return all_gather_without_padding(
            grad_output, ctx.num_pad, ctx.axis, ctx.group
        )


class ContextParallelNormalGather(PyLayer):
    """ContextParallelNormalGather"""

    @staticmethod
    def forward(ctx, input_tensor, num_pad, axis=0):
        """forward"""
        ctx.axis = axis
        hcg = fleet.get_hybrid_communicate_group()
        cp_degree = hcg.get_context_parallel_world_size()
        group = hcg.get_context_parallel_group()
        ctx.group = group
        ctx.num_pad = num_pad

        if cp_degree == 1:
            return input_tensor.clone()

        return all_gather_without_padding(input_tensor, num_pad, axis, group)

    @staticmethod
    def backward(ctx, grad_output):
        """backward"""
        if ctx.group.nranks == 1:
            return grad_output.clone()

        return scatter_with_padding(
            grad_output, ctx.num_pad, ctx.axis, ctx.group
        )


class FlashMaskContextParallel(PyLayer):
    """
    FlashMask attention with context parallelism implementation.
    This class implements flashmask attention with context parallelism (CP) using PyLayer
    for automatic differentiation. CP partitions tensors along the sequence dimension
    to enable long-context LLMs in a distributed fashion.
    The implementation uses the DualChunkSwap strategy to ensure load balancing
    across CP ranks by processing chunks from both ends of the sequence.
    """

    @staticmethod
    def forward(
        ctx,
        query,
        key,
        value,
        startend_row_indices,
        fixed_seed_offset=None,
        dropout=0.0,
        causal=False,
        training=True,
        learnable_sink=None,
        mode="allgather_kv",
    ):
        """
        Forward pass of FlashMask attention with context parallelism.
        Args:
            ctx: Context object for saving information for backward pass
            query (paddle.Tensor): Query tensor, pre-divided by CP size
            key (paddle.Tensor): Key tensor, pre-divided by CP size
            value (paddle.Tensor): Value tensor, pre-divided by CP size
            startend_row_indices (paddle.Tensor): Row indices for attention mask
            fixed_seed_offset (paddle.Tensor, optional): Fixed seed offset for dropout
            dropout (float): Dropout probability
            causal (bool): Whether to use causal attention
            training (bool): Whether in training mode
            mode (str): Attention mode, currently supports "allgather_kv"
        Returns:
            paddle.Tensor: Attention output
        Raises:
            NotImplementedError: If dropout > 0.0 or causal=True
            AssertionError: If query sequence length is not divisible by 2
        """
        # Validate input parameters
        if dropout > 0.0:
            raise NotImplementedError(
                "Dropout is not supported in FlashMask context parallel yet."
            )

        if causal:
            raise NotImplementedError(
                "FlashMaskContextParallel does not support causal=True yet."
            )

        if fixed_seed_offset is not None:
            raise NotImplementedError("Fixed seed offset is not supported yet.")

        # Get communication group
        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_context_parallel_group()

        # Validate query sequence length for DualChunkSwap strategy
        assert query.shape[1] % 2 == 0, (
            f"Query sequence length must be divisible by 2. "
            f"FlashMaskContextParallel uses DualChunkSwap strategy for load balancing. "
            f"Current query sequence length: {query.shape[1]}"
        )

        # Perform forward pass
        output, log_sum_exp, startend_row_indices, fa_version = (
            cp_flashmask_allgatherkv_balance_forward(
                query,
                key,
                value,
                startend_row_indices,
                learnable_sink,
                group,
                causal,
                training,
            )
        )

        # Save tensors for backward pass
        ctx.save_for_backward(
            query, key, value, output, log_sum_exp, startend_row_indices
        )
        ctx.group = group
        ctx.causal = causal
        ctx.fa_version = fa_version
        ctx.learnable_sink = learnable_sink
        # Only a trainable sink (a Parameter) needs a gradient returned from
        # backward. A fixed off-by-one sink is created as a stop_gradient=True
        # Tensor, and Paddle's PyLayer requires None in that return slot.
        ctx.sink_requires_grad = (
            learnable_sink is not None and not learnable_sink.stop_gradient
        )

        return output

    @staticmethod
    def backward(ctx, output_grad):
        """
        Backward pass of FlashMask attention with context parallelism.
        Args:
            ctx: Context object with saved information
            output_grad (paddle.Tensor): Gradient of output
        Returns:
            tuple: Gradients for all input arguments
        """
        # Retrieve saved tensors
        query, key, value, output, log_sum_exp, startend_row_indices = (
            ctx.saved_tensor()
        )
        group = ctx.group
        causal = ctx.causal
        fa_version = ctx.fa_version
        learnable_sink = ctx.learnable_sink

        # Compute gradients
        query_grad, key_grad, value_grad, grad_sink = (
            cp_flashmask_allgatherkv_balance_backward(
                query,
                key,
                value,
                startend_row_indices,
                output,
                log_sum_exp,
                output_grad,
                learnable_sink,
                group,
                causal,
                fa_version,
            )
        )

        # PyLayer maps backward returns positionally onto the forward TENSOR
        # inputs: query(0)/key(1)/value(2)/startend_row_indices(3)/
        # learnable_sink(4). startend_row_indices is stop_gradient=True, so its
        # slot (position 3) must be None -- grad_sink belongs in position 4.
        # A fixed off-by-one sink is also stop_gradient=True, so for it the
        # 3-tuple (sink slot omitted) is correct.
        if ctx.sink_requires_grad:
            return query_grad, key_grad, value_grad, None, grad_sink
        return query_grad, key_grad, value_grad


def flashmask_attention_cp(
    query,
    key,
    value,
    startend_row_indices,
    fixed_seed_offset=None,
    dropout=0.0,
    causal=False,
    training=True,
    learnable_sink=None,
    mode="allgather_kv",
):
    """
    FlashMask attention with context parallelism - public API.
    This is the main entry point for using FlashMask attention with context parallelism.
    It provides a convenient interface that wraps the FlashMaskContextParallel PyLayer.
    Args:
        query (paddle.Tensor): Query tensor with shape [batch, seq_len/n, num_heads, head_dim]
        key (paddle.Tensor): Key tensor with shape [batch, seq_len/n, num_heads, head_dim]
        value (paddle.Tensor): Value tensor with shape [batch, seq_len/n, num_heads, head_dim]
        startend_row_indices (paddle.Tensor): Row indices for attention mask
        fixed_seed_offset (paddle.Tensor, optional): Fixed seed offset for dropout
        dropout (float, optional): Dropout probability. Defaults to 0.0
        causal (bool, optional): Whether to use causal attention. Defaults to False
        training (bool, optional): Whether in training mode. Defaults to True
        mode (str, optional): Attention mode. Defaults to "allgather_kv"
    Returns:
        paddle.Tensor: Attention output with shape [batch, seq_len/n, num_heads, head_dim]
    Example:
        ```python
        # Initialize tensors (assuming context parallelism is set up)
        query = paddle.randn([2, 512, 8, 64])  # [batch, seq_len/n, heads, head_dim]
        key = paddle.randn([2, 512, 8, 64])    # [batch, seq_len/n, heads, head_dim]
        value = paddle.randn([2, 512, 8, 64])  # [batch, seq_len/n, heads, head_dim]
        mask_indices = paddle.randint(0, 1024, [100, 2])
        # Apply FlashMask attention with context parallelism
        output = flashmask_attention_cp(
            query=query,
            key=key,
            value=value,
            startend_row_indices=mask_indices,
            training=True
        )
        ```
    """
    if learnable_sink is not None:
        raise NotImplementedError(
            "learnable_sink is not supported on flashmask_attention_cp"
        )

    output = FlashMaskContextParallel.apply(
        query,
        key,
        value,
        startend_row_indices,
        fixed_seed_offset,
        dropout,
        causal,
        training,
        learnable_sink,
        mode,
    )
    return output
