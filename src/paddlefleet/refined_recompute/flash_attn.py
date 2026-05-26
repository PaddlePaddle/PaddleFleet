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

import logging
import queue

import paddle
from paddle import framework
from paddle.autograd import PyLayer
from paddle.distributed import fleet
from paddlefleet_ops.flash_mask_facade import (
    flash_attn_dispatch_bwd,
    flash_attn_dispatch_fwd,
)

from paddlefleet.context_parallel_utils import (
    cp_flashmask_allgatherkv_balance_backward,
    cp_flashmask_allgatherkv_balance_forward,
)
from paddlefleet.refined_recompute.queue_check import global_rr_queue_log

logger = logging.getLogger(__name__)


def flashattn_auto_cast(q, k, v, dtype=paddle.bfloat16):
    """
    A utility function to ensure that the Query, Key, and Value tensors
    are cast to a specific data type (typically bfloat16) before being
    passed to the FlashAttention kernel, which often requires a specific precision.

    Args:
        q (paddle.Tensor): The query tensor.
        k (paddle.Tensor): The key tensor.
        v (paddle.Tensor): The value tensor.
        dtype (paddle.dtype, optional): The target data type. Defaults to paddle.bfloat16.

    Returns:
        tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]: The casted Q, K, and V tensors.
    """
    if q.dtype != dtype:
        q = q.astype(dtype)
    if k.dtype != dtype:
        k = k.astype(dtype)
    if v.dtype != dtype:
        v = v.astype(dtype)
    return q, k, v


class FlashAttnFunctor(PyLayer):
    """
    A custom PyLayer designed for the refined recompute strategy.

    This class does not perform any actual computation in its forward pass. Instead,
    it serves as a "surrogate" or "fake" layer during the second forward pass of
    recomputation. Its primary role is to reconstruct the computation graph,
    allowing PaddlePaddle's autograd engine to correctly execute the custom
    backward pass defined here.
    """

    @staticmethod
    def forward(ctx, q, k, v, hold_tensors):
        """
        The forward pass of the surrogate layer. It simply retrieves the pre-computed
        attention output from the `hold_tensors` dictionary and saves all necessary
        tensors for the backward pass using `ctx.save_for_backward`.
        """
        fa_version = hold_tensors["fa_version"]
        ctx.fa_version = fa_version
        ctx.need_pad = hold_tensors.get("need_pad", False)
        ctx.head_dim_v = hold_tensors.get("head_dim_v", None)
        ctx.padded_value = hold_tensors.get("padded_value", None)

        result_attention = hold_tensors["result_attention"]
        padded_output = hold_tensors.get("padded_output", result_attention)
        softmax_lse = hold_tensors["softmax_lse"]
        causal = hold_tensors["causal"]

        if fa_version == 2:
            seed_offset = hold_tensors["seed_offset"]
            dropout = hold_tensors["dropout"]
            ctx.save_for_backward(
                q,
                k,
                v,
                padded_output,
                softmax_lse,
                seed_offset,
                dropout,
                causal,
            )
        else:
            # FA3 and FA4 share the same saved tensors
            ctx.save_for_backward(q, k, v, padded_output, softmax_lse, causal)
        # TODO: 这里为什么不用 raise ValueError(f"Invalid flash attention version: {fa_version}")

        return result_attention

    @staticmethod
    def backward(ctx, grad):
        """
        Defines the custom backward pass for FlashAttention.
        Uses flash_attn_dispatch_bwd for unified gradient computation.
        """
        fa_version = ctx.fa_version

        if fa_version == 2:
            (
                q,
                k,
                v,
                padded_output,
                softmax_lse,
                seed_offset,
                dropout,
                causal,
            ) = ctx.saved_tensor()
            q_grad, k_grad, v_grad = flash_attn_dispatch_bwd(
                q.detach(),
                k.detach(),
                v.detach(),
                padded_output,
                grad,
                softmax_lse,
                fa_version=fa_version,
                startend_row_indices=None,
                seed_offset=seed_offset,
                dropout=dropout,
                causal=causal,
                need_pad=ctx.need_pad,
                head_dim_v=ctx.head_dim_v,
                padded_value=ctx.padded_value,
            )
            seed_offset._clear_dataptr()
        else:
            q, k, v, padded_output, softmax_lse, causal = ctx.saved_tensor()
            q_grad, k_grad, v_grad = flash_attn_dispatch_bwd(
                q.detach(),
                k.detach(),
                v.detach(),
                padded_output,
                grad,
                softmax_lse,
                fa_version=fa_version,
                startend_row_indices=None,
                causal=causal,
                need_pad=ctx.need_pad,
                head_dim_v=ctx.head_dim_v,
                padded_value=ctx.padded_value,
            )

        # Manually release memory of intermediate tensors to save GPU memory.
        padded_output._clear_dataptr()
        softmax_lse._clear_dataptr()

        return q_grad, k_grad, v_grad


class RefinedRcomputeFlashAttention:
    """
    Implements the refined recompute strategy for standard (non-masked) FlashAttention.
    This class is designed to be used within a `recompute` block.
    """

    def __init__(self):
        """Initializes the class, creating a queue to hold intermediate tensors."""
        self._hold_tensors_queue = queue.Queue()
        global_rr_queue_log.update(self._hold_tensors_queue, "flash_attention")

    def forward(
        self,
        query_states,
        key_states,
        value_states,
        dropout=0.0,
        causal=True,
        return_softmax=False,
        training=True,
    ):
        """
        The main entry point for the forward pass.
        It checks if autograd is active. If not, it executes the first forward pass.
        If autograd is active (which happens during recomputation's backward pass),
        it executes the second forward pass.
        """
        if not framework._dygraph_tracer()._has_grad:
            # This is the initial, normal forward pass.
            attn_output, attn_weights = self._first_fwd(
                query_states,
                key_states,
                value_states,
                dropout=dropout,
                causal=causal,
                return_softmax=return_softmax,
                training=training,
            )
        else:
            # This is the second forward pass, executed during the backward pass of recompute.
            assert not self._hold_tensors_queue.empty(), (
                "queue should not be empty"
            )
            attn_output, attn_weights = self._second_fwd(
                query_states, key_states, value_states
            )

        return attn_output, attn_weights

    @paddle.no_grad()
    def _first_fwd(
        self,
        query_states,
        key_states,
        value_states,
        dropout=0.0,
        causal=True,
        return_softmax=False,
        training=True,
    ):
        """
        The first forward pass. It runs the actual FlashAttention computation
        without tracking gradients (`@paddle.no_grad()`). It saves the necessary
        intermediate tensors for the backward pass into a queue and returns the final output.
        """
        query_states, key_states, value_states = flashattn_auto_cast(
            query_states, key_states, value_states
        )

        fwd_result = flash_attn_dispatch_fwd(
            query_states,
            key_states,
            value_states,
            startend_row_indices=None,
            causal=causal,
            dropout=dropout,
            training=training,
            return_softmax=return_softmax,
        )

        hold_tensors = {
            "result_attention": fwd_result["output"],
            "padded_output": fwd_result.get("padded_output"),
            "softmax_lse": fwd_result["softmax_lse"],
            "seed_offset": fwd_result["seed_offset"],
            "result_softmax": fwd_result["result_softmax"],
            "dropout": dropout,
            "causal": causal,
            "fa_version": fwd_result["fa_version"],
            "need_pad": fwd_result["need_pad"],
            "head_dim_v": fwd_result["head_dim_v"],
            "padded_value": fwd_result.get("padded_value"),
        }

        # Put the dictionary of saved tensors into the queue.
        self._hold_tensors_queue.put(hold_tensors)
        return fwd_result["output"], fwd_result[
            "result_softmax"
        ] if return_softmax else None

    def _second_fwd(self, query_states, key_states, value_states):
        """
        The second forward pass. It retrieves the saved tensors from the queue
        and passes them to the `FlashAttnFunctor` surrogate layer. This action
        reconstructs the computation graph, enabling the custom backward pass to run.
        """
        hold_tensors = self._hold_tensors_queue.get()
        query_states, key_states, value_states = flashattn_auto_cast(
            query_states, key_states, value_states
        )
        # Call the surrogate PyLayer to link the backward pass.
        output = FlashAttnFunctor.apply(
            query_states, key_states, value_states, hold_tensors
        )
        return output, hold_tensors.get(
            "result_softmax"
        )  # Use .get for safety with FA v3/v4

    def __call__(self, *args, **kwds):
        """Makes the class instance callable, similar to a standard nn.Layer."""
        return self.forward(*args, **kwds)


class FlashMaskAttnFunctor(PyLayer):
    """
    A custom PyLayer for the **masked** version of FlashAttention.

    This class serves the same purpose as `FlashAttnFunctor` but is tailored for
    the `flashmask_attention` operator, which takes an additional `startend_row_indices`
    tensor to handle variable-length sequences or sparse attention patterns.
    """

    @staticmethod
    def forward(ctx, q, k, v, startend_row_indices, hold_tensors):
        """
        The forward pass for the masked attention surrogate layer.
        It saves all necessary tensors, including `startend_row_indices`, for the backward pass.
        """
        fa_version = hold_tensors["fa_version"]
        ctx.fa_version = fa_version
        ctx.need_pad = hold_tensors.get("need_pad", False)
        ctx.head_dim_v = hold_tensors.get("head_dim_v", None)
        ctx.padded_value = hold_tensors.get("padded_value", None)

        result_attention = hold_tensors["result_attention"]
        padded_output = hold_tensors.get("padded_output", result_attention)
        softmax_lse = hold_tensors["softmax_lse"]
        causal = hold_tensors["causal"]

        if fa_version == 2:
            seed_offset = hold_tensors["seed_offset"]
            dropout = hold_tensors["dropout"]
            ctx.save_for_backward(
                q,
                k,
                v,
                startend_row_indices,
                padded_output,
                softmax_lse,
                seed_offset,
                dropout,
                causal,
            )
        else:
            # FA3 and FA4 share the same saved tensors
            ctx.save_for_backward(
                q,
                k,
                v,
                startend_row_indices,
                padded_output,
                softmax_lse,
                causal,
            )

        return result_attention

    @staticmethod
    def backward(ctx, grad):
        """
        Defines the custom backward pass for masked FlashAttention.
        Uses flash_attn_dispatch_bwd for unified gradient computation.
        """
        fa_version = ctx.fa_version

        if fa_version == 2:
            (
                q,
                k,
                v,
                startend_row_indices,
                padded_output,
                softmax_lse,
                seed_offset,
                dropout,
                causal,
            ) = ctx.saved_tensor()
            q_grad, k_grad, v_grad = flash_attn_dispatch_bwd(
                q.detach(),
                k.detach(),
                v.detach(),
                padded_output,
                grad,
                softmax_lse,
                fa_version=fa_version,
                startend_row_indices=startend_row_indices,
                seed_offset=seed_offset,
                dropout=dropout,
                causal=causal,
                need_pad=ctx.need_pad,
                head_dim_v=ctx.head_dim_v,
                padded_value=ctx.padded_value,
            )
            seed_offset._clear_dataptr()
        else:
            (
                q,
                k,
                v,
                startend_row_indices,
                padded_output,
                softmax_lse,
                causal,
            ) = ctx.saved_tensor()
            q_grad, k_grad, v_grad = flash_attn_dispatch_bwd(
                q.detach(),
                k.detach(),
                v.detach(),
                padded_output,
                grad,
                softmax_lse,
                fa_version=fa_version,
                startend_row_indices=startend_row_indices,
                causal=causal,
                need_pad=ctx.need_pad,
                head_dim_v=ctx.head_dim_v,
                padded_value=ctx.padded_value,
            )

        # Manually release memory.
        padded_output._clear_dataptr()
        softmax_lse._clear_dataptr()

        return q_grad, k_grad, v_grad


class RefinedRcomputeFlashMaskAttention:
    """
    Implements the refined recompute strategy for masked FlashAttention.
    This class is designed to be used within a `recompute` block.
    """

    def __init__(self):
        """Initializes the class, creating a queue to hold intermediate tensors."""
        self._hold_tensors_queue = queue.Queue()
        global_rr_queue_log.update(
            self._hold_tensors_queue, "flashmask_attention"
        )

    def forward(
        self,
        query_states,
        key_states,
        value_states,
        startend_row_indices,
        dropout=0.0,
        causal=True,
        return_softmax=False,
        training=True,
    ):
        """
        The main entry point for the forward pass.
        Dispatches to either the first or second forward pass based on autograd state.
        """
        if not framework._dygraph_tracer()._has_grad:
            # This is the initial, normal forward pass.
            attn_output = self._first_fwd(
                query_states,
                key_states,
                value_states,
                startend_row_indices,
                dropout=dropout,
                causal=causal,
                return_softmax=return_softmax,
                training=training,
            )
        else:
            # This is the second forward pass, executed during the backward pass of recompute.
            assert not self._hold_tensors_queue.empty(), (
                "queue should not be empty"
            )
            attn_output = self._second_fwd(
                query_states, key_states, value_states, startend_row_indices
            )

        return attn_output

    @paddle.no_grad()
    def _first_fwd(
        self,
        query_states,
        key_states,
        value_states,
        startend_row_indices,
        dropout=0.0,
        causal=True,
        return_softmax=False,
        training=True,
    ):
        """
        The first forward pass for masked attention. It runs the actual computation,
        saves intermediate tensors to the queue, and returns the output.
        """
        query_states, key_states, value_states = flashattn_auto_cast(
            query_states, key_states, value_states
        )

        fwd_result = flash_attn_dispatch_fwd(
            query_states,
            key_states,
            value_states,
            startend_row_indices=startend_row_indices,
            causal=causal,
            dropout=dropout,
            training=training,
            return_softmax=return_softmax,
        )

        hold_tensors = {
            "result_attention": fwd_result["output"],
            "padded_output": fwd_result.get("padded_output"),
            "softmax_lse": fwd_result["softmax_lse"],
            "seed_offset": fwd_result["seed_offset"],
            "result_softmax": fwd_result["result_softmax"],
            "dropout": dropout,
            "causal": causal,
            "fa_version": fwd_result["fa_version"],
            "need_pad": fwd_result["need_pad"],
            "head_dim_v": fwd_result["head_dim_v"],
            "padded_value": fwd_result.get("padded_value"),
        }

        self._hold_tensors_queue.put(hold_tensors)
        return fwd_result["output"]

    def _second_fwd(
        self, query_states, key_states, value_states, startend_row_indices
    ):
        """
        The second forward pass for masked attention. It reconstructs the graph
        by calling the `FlashMaskAttnFunctor` surrogate layer.
        """
        hold_tensors = self._hold_tensors_queue.get()
        query_states, key_states, value_states = flashattn_auto_cast(
            query_states, key_states, value_states
        )
        output = FlashMaskAttnFunctor.apply(
            query_states,
            key_states,
            value_states,
            startend_row_indices,
            hold_tensors,
        )
        return output

    def __call__(self, *args, **kwds):
        """Makes the class instance callable, similar to a standard nn.Layer."""
        return self.forward(*args, **kwds)


class FlashMaskAttnCpFunctor(PyLayer):
    """
    A custom PyLayer for FlashAttention with Context Parallelism.

    This class serves as the surrogate layer for the refined recompute strategy
    in the context parallel path.
    """

    @staticmethod
    def forward(ctx, q, k, v, hold_tensors):
        """
        The forward pass for the CP masked attention surrogate layer.
        """
        result_attention = hold_tensors["result_attention"]
        softmax_lse = hold_tensors["softmax_lse"]
        startend_row_indices = hold_tensors["startend_row_indices"]
        fa_version = hold_tensors["fa_version"]
        group = hold_tensors["group"]
        causal = hold_tensors["causal"]

        ctx.fa_version = fa_version
        ctx.need_pad = hold_tensors["need_pad"]
        ctx.head_dim_v = hold_tensors["head_dim_v"]
        ctx.save_for_backward(
            q,
            k,
            v,
            startend_row_indices,
            result_attention,
            softmax_lse,
            group,
            causal,
        )

        return result_attention

    @staticmethod
    def backward(ctx, grad):
        """
        Defines the custom backward pass for CP masked FlashAttention.
        """
        (
            q,
            k,
            v,
            startend_row_indices,
            result_attention,
            softmax_lse,
            group,
            causal,
        ) = ctx.saved_tensor()
        fa_version = ctx.fa_version

        # Compute gradients via context parallel backward
        query_grad, key_grad, value_grad = (
            cp_flashmask_allgatherkv_balance_backward(
                q,
                k,
                v,
                startend_row_indices,
                result_attention,
                softmax_lse,
                grad,
                group,
                causal,
                fa_version,
                need_pad=ctx.need_pad,
                v_head_dim=ctx.head_dim_v,
            )
        )

        # Manually release memory.
        result_attention._clear_dataptr()
        softmax_lse._clear_dataptr()

        return query_grad, key_grad, value_grad


class RefinedRcomputeFlashMaskCpAttention:
    """
    Implements the refined recompute strategy for masked FlashAttention.
    This class is designed to be used within a `recompute` block in Context Parallel.
    """

    def __init__(self):
        """Initializes the class, creating a queue to hold intermediate tensors."""
        self._hold_tensors_queue = queue.Queue()
        global_rr_queue_log.update(
            self._hold_tensors_queue, "flashmask_attention_rr"
        )

    def forward(
        self,
        query_states,
        key_states,
        value_states,
        startend_row_indices,
        fixed_seed_offset=None,
        dropout=0.0,
        causal=False,
        training=True,
        mode="allgather_kv",
    ):
        """
        The main entry point for the forward pass.
        Dispatches to either the first or second forward pass based on autograd state.
        """
        if not framework._dygraph_tracer()._has_grad:
            # This is the initial, normal forward pass.
            attn_output = self._first_fwd(
                query_states,
                key_states,
                value_states,
                startend_row_indices,
                fixed_seed_offset=fixed_seed_offset,
                dropout=dropout,
                causal=causal,
                training=training,
                mode=mode,
            )
        else:
            # This is the second forward pass, executed during the backward pass of recompute.
            assert not self._hold_tensors_queue.empty(), (
                "queue should not be empty"
            )
            attn_output = self._second_fwd(
                query_states, key_states, value_states
            )

        return attn_output

    @paddle.no_grad()
    def _first_fwd(
        self,
        query_states,
        key_states,
        value_states,
        startend_row_indices,
        fixed_seed_offset=None,
        dropout=0.0,
        causal=False,
        training=True,
        mode="allgather_kv",
    ):
        """
        The first forward pass for masked attention. It runs the actual computation,
        saves intermediate tensors to the queue, and returns the output.
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
        assert query_states.shape[1] % 2 == 0, (
            f"Query sequence length must be divisible by 2. "
            f"FlashMaskContextParallel uses DualChunkSwap strategy for load balancing. "
            f"Current query sequence length: {query_states.shape[1]}"
        )

        (
            result_attention,
            softmax_lse,
            startend_row_indices,
            fa_version,
            need_pad,
            head_dim_v,
        ) = cp_flashmask_allgatherkv_balance_forward(
            query_states,
            key_states,
            value_states,
            startend_row_indices,
            group,
            causal,
            training,
        )

        hold_tensors = {
            "result_attention": result_attention,
            "softmax_lse": softmax_lse,
            "startend_row_indices": startend_row_indices,
            "fa_version": fa_version,
            "group": group,
            "causal": causal,
            "need_pad": need_pad,
            "head_dim_v": head_dim_v,
        }

        self._hold_tensors_queue.put(hold_tensors)
        return result_attention

    def _second_fwd(self, query_states, key_states, value_states):
        """
        The second forward pass for masked attention. It reconstructs the graph
        by calling the `FlashMaskAttnCpFunctor` surrogate layer.
        """
        hold_tensors = self._hold_tensors_queue.get()
        output = FlashMaskAttnCpFunctor.apply(
            query_states, key_states, value_states, hold_tensors
        )
        return output

    def __call__(self, *args, **kwds):
        """Makes the class instance callable, similar to a standard nn.Layer."""
        return self.forward(*args, **kwds)
