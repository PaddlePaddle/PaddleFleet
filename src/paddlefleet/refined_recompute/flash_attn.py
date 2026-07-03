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
import logging
import queue

import paddle
from paddle import _C_ops, framework
from paddle.autograd import PyLayer
from paddle.distributed import fleet
from paddle.nn.functional.flash_attention import flashmask_attention

from paddlefleet.context_parallel_utils import (
    cp_flashmask_allgatherkv_balance_backward,
    cp_flashmask_allgatherkv_balance_forward,
)
from paddlefleet.refined_recompute.queue_check import global_rr_queue_log

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

logger = logging.getLogger(__name__)


def _get_fa_version(hdim):
    """
    Determines which version of the FlashAttention C++ operator to use.
    It checks environment flags to decide between version 2 and version 3,
    and defaults to version 2 for XPU devices.

    Returns:
        int: The version number of FlashAttention to be used (2 or 3).
    """
    if "xpu" in paddle.get_device():
        return 2
    # Xiangrui: For deterministic, NOT support for hdim > 128 currently.
    if "block_mask" in inspect.signature(flashmask_attention).parameters:
        if (
            paddle.get_flags(["FLAGS_cudnn_deterministic"])[
                "FLAGS_cudnn_deterministic"
            ]
            and hdim > 128
        ):
            return 2
    elif paddle.get_flags(["FLAGS_cudnn_deterministic"])[
        "FLAGS_cudnn_deterministic"
    ]:
        return 2
    fa_version = paddle.base.framework.get_flags(["FLAGS_flash_attn_version"])[
        "FLAGS_flash_attn_version"
    ]
    # Fall back to version 3 if flash_mask is not available
    if fa_version == 4 and not _flash_mask_available:
        logger.warning(
            "FlashMask (fa_version=4) is not available, falling back to fa_version=3"
        )
        return 3
    return fa_version


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

        Args:
            ctx (Context): The context object to save tensors for backward.
            q, k, v (paddle.Tensor): The input tensors from the second forward pass.
            hold_tensors (dict): A dictionary containing intermediate results from the
                                 first forward pass (e.g., the actual attention output, softmax_lse).

        Returns:
            paddle.Tensor: The pre-computed attention output.
        """
        fa_version = _get_fa_version(q.shape[-1])
        ctx.fa_version = fa_version
        ctx.softmax_scale = hold_tensors.get("softmax_scale")

        # Save the necessary tensors that will be needed to compute the gradient.
        if fa_version == 2:
            result_attention = hold_tensors["result_attention"]
            result_softmax = hold_tensors["result_softmax"]
            softmax_lse = hold_tensors["softmax_lse"]
            seed_offset = hold_tensors["seed_offset"]
            dropout = hold_tensors["dropout"]
            causal = hold_tensors["causal"]
            ctx.save_for_backward(
                q,
                k,
                v,
                result_attention,
                softmax_lse,
                seed_offset,
                dropout,
                causal,
            )
        elif fa_version == 3:
            result_attention = hold_tensors["result_attention"]
            softmax_lse = hold_tensors["softmax_lse"]
            causal = hold_tensors["causal"]
            ctx.save_for_backward(
                q, k, v, result_attention, softmax_lse, causal
            )
        elif fa_version == 4:
            result_attention = hold_tensors["result_attention"]
            softmax_lse = hold_tensors["softmax_lse"]
            causal = hold_tensors["causal"]
            ctx.save_for_backward(
                q, k, v, result_attention, softmax_lse, causal
            )
        else:
            raise ValueError(f"Invalid flash attention version: {fa_version}")

        # Return the actual output computed during the first forward pass.
        return result_attention

    @staticmethod
    def backward(ctx, grad):
        """
        Defines the custom backward pass for FlashAttention.
        It retrieves the saved tensors from the context and calls the low-level
        C++ gradient operator (`_C_ops.flash_attn_grad`) to compute the gradients
        for Q, K, and V.

        Args:
            ctx (Context): The context object to retrieve saved tensors.
            grad (paddle.Tensor): The gradient of the output tensor.

        Returns:
            tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]: The gradients for Q, K, and V.
        """
        fa_version = ctx.fa_version

        if fa_version == 2:
            (
                q,
                k,
                v,
                result_attention,
                softmax_lse,
                seed_offset,
                dropout,
                causal,
            ) = ctx.saved_tensor()
            # Call the underlying C++ gradient kernel.
            q_grad, k_grad, v_grad = _C_ops.flash_attn_grad(
                q.detach(),
                k.detach(),
                v.detach(),
                result_attention,
                softmax_lse,
                seed_offset,
                None,  # attn_mask (dense mask)
                grad,
                dropout,
                causal,
            )
            seed_offset._clear_dataptr()
        elif fa_version == 3:
            q, k, v, result_attention, softmax_lse, causal = ctx.saved_tensor()
            q_grad, k_grad, v_grad = _C_ops.flash_attn_v3_grad(
                q.detach(),
                k.detach(),
                v.detach(),
                result_attention,
                softmax_lse,
                grad,
                q.shape[-1] ** (-0.5)
                if ctx.softmax_scale is None
                else ctx.softmax_scale,  # softmax_scale
                causal,
                -1,  # window_size_left
                -1,  # window_size_right
                0.0,  # softcap
                0,  # sm_margin
            )
        elif fa_version == 4:
            flashmask_info = None
            q, k, v, result_attention, softmax_lse, causal = ctx.saved_tensor()
            q_grad, k_grad, v_grad, _ = _flash_attn_bwd(
                q.detach(),
                k.detach(),
                v.detach(),
                result_attention,
                grad,
                softmax_lse,
                flashmask_info,
                softmax_scale=ctx.softmax_scale,
                causal=causal,
                deterministic=bool(
                    paddle.get_flags(["FLAGS_cudnn_deterministic"])[
                        "FLAGS_cudnn_deterministic"
                    ]
                ),
            )
        else:
            raise ValueError(f"Invalid flash attention version: {fa_version}")

        # Manually release memory of intermediate tensors to save GPU memory.
        result_attention._clear_dataptr()
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
        softmax_scale=None,
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
                softmax_scale=softmax_scale,
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
        softmax_scale=None,
    ):
        """
        The first forward pass. It runs the actual FlashAttention computation
        without tracking gradients (`@paddle.no_grad()`). It saves the necessary
        intermediate tensors for the backward pass into a queue and returns the final output.
        """
        query_states, key_states, value_states = flashattn_auto_cast(
            query_states, key_states, value_states
        )

        fa_version = _get_fa_version(query_states.shape[-1])
        if fa_version == 2:
            if softmax_scale is not None:
                raise NotImplementedError(
                    "fa_version==2 does not support setting softmax_scale"
                )
            (result_attention, result_softmax, softmax_lse, seed_offset) = (
                _C_ops.flash_attn(
                    query_states,
                    key_states,
                    value_states,
                    None,
                    None,
                    dropout,
                    causal,
                    return_softmax,
                    not training,
                    "",
                )
            )
            # Store all tensors needed for the backward pass in a dictionary.
            hold_tensors = {
                "result_attention": result_attention,
                "result_softmax": result_softmax,
                "softmax_lse": softmax_lse,
                "seed_offset": seed_offset,
                "dropout": dropout,
                "causal": causal,
            }
        elif fa_version == 3:
            (result_attention, softmax_lse) = _C_ops.flash_attn_v3(
                query_states,
                key_states,
                value_states,
                None,
                None,
                None,
                None,
                query_states.shape[-1] ** (-0.5)
                if softmax_scale is None
                else softmax_scale,
                causal,
                -1,
                -1,
                0.0,
                1,
                False,
                False,
                0,
            )
            hold_tensors = {
                "result_attention": result_attention,
                "softmax_lse": softmax_lse,
                "causal": causal,
                "softmax_scale": softmax_scale,
            }
            result_softmax = None  # FA v3 does not return softmax.
        elif fa_version == 4:
            (result_attention, softmax_lse) = _flash_attn_fwd(
                query_states,
                key_states,
                value_states,
                causal=causal,
                return_lse=True,
                startend_row_indices=None,
                softmax_scale=softmax_scale,
                pack_gqa=False,
            )
            hold_tensors = {
                "result_attention": result_attention,
                "softmax_lse": softmax_lse,
                "causal": causal,
                "softmax_scale": softmax_scale,
            }
            result_softmax = None
        else:
            raise ValueError(f"Invalid flash attention version: {fa_version}")

        # Put the dictionary of saved tensors into the queue.
        self._hold_tensors_queue.put(hold_tensors)
        return result_attention, result_softmax if return_softmax else None

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
        )  # Use .get for safety with FA v3

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
    def forward(
        ctx, q, k, v, startend_row_indices, learnable_sink, hold_tensors
    ):
        """
        The forward pass for the masked attention surrogate layer.
        It saves all necessary tensors, including `startend_row_indices`, for the backward pass.
        """
        fa_version = _get_fa_version(q.shape[-1])
        ctx.fa_version = fa_version
        ctx.softmax_scale = hold_tensors.get("softmax_scale")
        ctx.sink_requires_grad = (
            learnable_sink is not None and not learnable_sink.stop_gradient
        )

        if fa_version == 2:
            result_attention = hold_tensors["result_attention"]
            softmax_lse = hold_tensors["softmax_lse"]
            seed_offset = hold_tensors["seed_offset"]
            dropout = hold_tensors["dropout"]
            causal = hold_tensors["causal"]
            ctx.save_for_backward(
                q,
                k,
                v,
                startend_row_indices,
                result_attention,
                softmax_lse,
                seed_offset,
                dropout,
                causal,
            )
        elif fa_version == 3:
            result_attention = hold_tensors["result_attention"]
            softmax_lse = hold_tensors["softmax_lse"]
            causal = hold_tensors["causal"]
            ctx.save_for_backward(
                q,
                k,
                v,
                startend_row_indices,
                result_attention,
                softmax_lse,
                causal,
            )
        elif fa_version == 4:
            result_attention = hold_tensors["result_attention"]
            softmax_lse = hold_tensors["softmax_lse"]
            causal = hold_tensors["causal"]
            ctx.save_for_backward(
                q,
                k,
                v,
                startend_row_indices,
                result_attention,
                softmax_lse,
                causal,
                learnable_sink,
            )
        else:
            raise ValueError(f"Invalid flash attention version: {fa_version}")

        return result_attention

    @staticmethod
    def backward(ctx, grad):
        """
        Defines the custom backward pass for masked FlashAttention.
        It calls the corresponding low-level C++ gradient operator (`_C_ops.flashmask_attention_grad`).
        """
        fa_version = ctx.fa_version

        if fa_version == 2:
            (
                q,
                k,
                v,
                startend_row_indices,
                result_attention,
                softmax_lse,
                seed_offset,
                dropout,
                causal,
            ) = ctx.saved_tensor()
            # Call the underlying C++ gradient kernel for masked attention.
            q_grad, k_grad, v_grad = _C_ops.flashmask_attention_grad(
                q.detach(),
                k.detach(),
                v.detach(),
                startend_row_indices,
                result_attention,
                softmax_lse,
                seed_offset,
                grad,
                dropout,
                causal,
            )
            seed_offset._clear_dataptr()
        elif fa_version == 3:
            (
                q,
                k,
                v,
                startend_row_indices,
                result_attention,
                softmax_lse,
                causal,
            ) = ctx.saved_tensor()

            sig_params = inspect.signature(flashmask_attention).parameters

            softmax_scale = (
                q.shape[-1] ** (-0.5)
                if ctx.softmax_scale is None
                else ctx.softmax_scale,
            )

            if "group" in sig_params:
                q_grad, k_grad, v_grad = _C_ops.flashmask_attention_v2_grad(
                    q.detach(),
                    k.detach(),
                    v.detach(),
                    result_attention,
                    softmax_lse,
                    startend_row_indices,
                    None,  # block_mask
                    grad,
                    softmax_scale,
                    causal,
                    0,  # rank
                    1,  # nranks
                )
            elif "block_mask" in sig_params:
                q_grad, k_grad, v_grad = _C_ops.flashmask_attention_v2_grad(
                    q.detach(),
                    k.detach(),
                    v.detach(),
                    result_attention,
                    softmax_lse,
                    startend_row_indices,
                    None,  # block_mask
                    grad,
                    softmax_scale,
                    causal,
                )
            else:
                q_grad, k_grad, v_grad = _C_ops.flashmask_attention_v2_grad(
                    q.detach(),
                    k.detach(),
                    v.detach(),
                    result_attention,
                    softmax_lse,
                    startend_row_indices,
                    grad,
                    softmax_scale,
                    causal,
                )
        elif fa_version == 4:
            (
                q,
                k,
                v,
                startend_row_indices,
                result_attention,
                softmax_lse,
                causal,
                learnable_sink,
            ) = ctx.saved_tensor()
            if startend_row_indices is not None:
                flashmask_info = FlashMaskInfoPaddle(
                    startend_row_indices=startend_row_indices,
                    is_causal=causal,
                )
            else:
                flashmask_info = None
            q_grad, k_grad, v_grad, sink_grad = _flash_attn_bwd(
                q.detach(),
                k.detach(),
                v.detach(),
                result_attention,
                grad,
                softmax_lse,
                flashmask_info,
                learnable_sink=learnable_sink,
                softmax_scale=ctx.softmax_scale,
                causal=causal,
                deterministic=bool(
                    paddle.get_flags(["FLAGS_cudnn_deterministic"])[
                        "FLAGS_cudnn_deterministic"
                    ]
                ),
            )
        else:
            raise ValueError(f"Invalid flash attention version: {fa_version}")

        # Manually release memory.
        result_attention._clear_dataptr()
        softmax_lse._clear_dataptr()

        # PyLayer maps backward returns positionally onto the forward TENSOR
        # inputs: q(0)/k(1)/v(2)/startend_row_indices(3)/learnable_sink(4).
        # startend_row_indices is stop_gradient=True, so its slot (position 3)
        # must be None -- sink_grad belongs in position 4. A fixed off-by-one
        # sink is also stop_gradient=True, so for it the 3-tuple is correct.
        if ctx.sink_requires_grad:
            return q_grad, k_grad, v_grad, None, sink_grad
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
        learnable_sink=None,
        softmax_scale=None,
    ):
        """
        The main entry point for the forward pass.
        Dispatches to either the first or second forward pass based on autograd state.
        """
        if learnable_sink is not None:
            fa_version = _get_fa_version(query_states.shape[-1])
            if fa_version != 4:
                raise NotImplementedError(
                    "learnable_sink only supported on fa_version==4 cute backend"
                )
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
                learnable_sink=learnable_sink,
                softmax_scale=softmax_scale,
            )
        else:
            # This is the second forward pass, executed during the backward pass of recompute.
            assert not self._hold_tensors_queue.empty(), (
                "queue should not be empty"
            )
            attn_output = self._second_fwd(
                query_states,
                key_states,
                value_states,
                startend_row_indices,
                learnable_sink=learnable_sink,
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
        learnable_sink=None,
        softmax_scale=None,
    ):
        """
        The first forward pass for masked attention. It runs the actual computation,
        saves intermediate tensors to the queue, and returns the output.
        """
        query_states, key_states, value_states = flashattn_auto_cast(
            query_states, key_states, value_states
        )
        fa_version = _get_fa_version(query_states.shape[-1])
        if fa_version == 2:
            if softmax_scale is not None:
                raise NotImplementedError(
                    "fa_version==2 does not support setting softmax_scale"
                )
            (result_attention, result_softmax, softmax_lse, seed_offset) = (
                _C_ops.flashmask_attention(
                    query_states,
                    key_states,
                    value_states,
                    startend_row_indices,
                    None,
                    dropout,
                    causal,
                    return_softmax,
                    not training,
                    "",
                )
            )
            hold_tensors = {
                "result_attention": result_attention,
                "softmax_lse": softmax_lse,
                "seed_offset": seed_offset,
                "result_softmax": result_softmax,
                "dropout": dropout,
                "causal": causal,
            }
        elif fa_version == 3:
            sig_params = inspect.signature(flashmask_attention).parameters
            scale = (
                query_states.shape[-1] ** (-0.5)
                if softmax_scale is None
                else softmax_scale
            )
            if "group" in sig_params:
                (result_attention, softmax_lse) = _C_ops.flashmask_attention_v2(
                    query_states,
                    key_states,
                    value_states,
                    startend_row_indices,
                    None,  # block_mask
                    None,  # nvshmem unique id
                    scale,
                    causal,
                    0,  # rank
                    1,  # nranks
                )
            elif "block_mask" in sig_params:
                (result_attention, softmax_lse) = _C_ops.flashmask_attention_v2(
                    query_states,
                    key_states,
                    value_states,
                    startend_row_indices,
                    None,  # block_mask
                    scale,
                    causal,
                )
            else:
                (result_attention, softmax_lse) = _C_ops.flashmask_attention_v2(
                    query_states,
                    key_states,
                    value_states,
                    startend_row_indices,
                    scale,
                    causal,
                )
            hold_tensors = {
                "result_attention": result_attention,
                "softmax_lse": softmax_lse,
                "causal": causal,
                "softmax_scale": softmax_scale,
            }
        elif fa_version == 4:
            (result_attention, softmax_lse) = _flash_attn_fwd(
                query_states,
                key_states,
                value_states,
                causal=causal,
                return_lse=True,
                startend_row_indices=startend_row_indices,
                learnable_sink=learnable_sink,
                softmax_scale=softmax_scale,
                pack_gqa=False,
            )
            hold_tensors = {
                "result_attention": result_attention,
                "softmax_lse": softmax_lse,
                "causal": causal,
                "learnable_sink": learnable_sink,
                "softmax_scale": softmax_scale,
            }
        else:
            raise ValueError(f"Invalid flash attention version: {fa_version}")

        self._hold_tensors_queue.put(hold_tensors)
        return result_attention

    def _second_fwd(
        self,
        query_states,
        key_states,
        value_states,
        startend_row_indices,
        learnable_sink=None,
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
            learnable_sink,
            hold_tensors,
        )
        return output

    def __call__(self, *args, **kwds):
        """Makes the class instance callable, similar to a standard nn.Layer."""
        return self.forward(*args, **kwds)


class FlashMaskAttnCpFunctor(PyLayer):
    """
    A custom PyLayer for the **masked** version of FlashAttention.

    This class serves the same purpose as `FlashAttnFunctor` but is tailored for
    the `flashmask_attention` operator, which takes an additional `startend_row_indices`
    tensor to handle variable-length sequences or sparse attention patterns.
    """

    @staticmethod
    def forward(ctx, q, k, v, learnable_sink, hold_tensors):
        """
        The forward pass for the masked attention surrogate layer.
        It saves all necessary tensors, including `startend_row_indices`, for the backward pass.
        """

        result_attention = hold_tensors["result_attention"]
        softmax_lse = hold_tensors["softmax_lse"]
        startend_row_indices = hold_tensors["startend_row_indices"]
        fa_version = hold_tensors["fa_version"]
        group = hold_tensors["group"]
        causal = hold_tensors["causal"]

        ctx.fa_version = fa_version
        ctx.softmax_scale = hold_tensors.get("softmax_scale")
        ctx.sink_requires_grad = (
            learnable_sink is not None and not learnable_sink.stop_gradient
        )
        ctx.save_for_backward(
            q,
            k,
            v,
            startend_row_indices,
            result_attention,
            softmax_lse,
            group,
            causal,
            learnable_sink,
        )

        return result_attention

    @staticmethod
    def backward(ctx, grad):
        """
        Defines the custom backward pass for masked FlashAttention.
        It calls the corresponding low-level C++ gradient operator (`_C_ops.flashmask_attention_grad`).
        """
        # Retrieve saved tensors
        (
            q,
            k,
            v,
            startend_row_indices,
            result_attention,
            softmax_lse,
            group,
            causal,
            learnable_sink,
        ) = ctx.saved_tensor()
        fa_version = ctx.fa_version

        # Compute gradients
        query_grad, key_grad, value_grad, sink_grad = (
            cp_flashmask_allgatherkv_balance_backward(
                q,
                k,
                v,
                startend_row_indices,
                result_attention,
                softmax_lse,
                grad,
                learnable_sink,
                group,
                causal,
                fa_version,
                ctx.softmax_scale,  # softmax_scale
            )
        )

        # Manually release memory.
        result_attention._clear_dataptr()
        softmax_lse._clear_dataptr()

        # PyLayer maps backward returns positionally onto the forward TENSOR
        # inputs: q(0)/k(1)/v(2)/learnable_sink(3). A trainable sink needs its
        # grad returned in slot 3; a fixed off-by-one sink (stop_gradient=True)
        # or sink=None uses the 3-tuple.
        if ctx.sink_requires_grad:
            return query_grad, key_grad, value_grad, sink_grad
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
        learnable_sink=None,
        softmax_scale=None,
    ):
        """
        The main entry point for the forward pass.
        Dispatches to either the first or second forward pass based on autograd state.
        """
        if learnable_sink is not None:
            fa_version = paddle.base.framework.get_flags(
                ["FLAGS_flash_attn_version"]
            )["FLAGS_flash_attn_version"]
            if not (fa_version == 4 and _flash_mask_available):
                raise NotImplementedError(
                    "learnable_sink only supported on fa_version==4 cute backend"
                )
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
                learnable_sink=learnable_sink,
                softmax_scale=softmax_scale,
            )
        else:
            # This is the second forward pass, executed during the backward pass of recompute.
            assert not self._hold_tensors_queue.empty(), (
                "queue should not be empty"
            )
            attn_output = self._second_fwd(
                query_states,
                key_states,
                value_states,
                learnable_sink=learnable_sink,
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
        learnable_sink=None,
        softmax_scale=None,
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

        result_attention, softmax_lse, startend_row_indices, fa_version = (
            cp_flashmask_allgatherkv_balance_forward(
                query_states,
                key_states,
                value_states,
                startend_row_indices,
                learnable_sink,
                group,
                causal,
                training,
                softmax_scale,
            )
        )

        hold_tensors = {
            "result_attention": result_attention,
            "softmax_lse": softmax_lse,
            "startend_row_indices": startend_row_indices,
            "fa_version": fa_version,
            "group": group,
            "causal": causal,
            "learnable_sink": learnable_sink,
            "softmax_scale": softmax_scale,
        }

        self._hold_tensors_queue.put(hold_tensors)
        return result_attention

    def _second_fwd(
        self, query_states, key_states, value_states, learnable_sink=None
    ):
        """
        The second forward pass for masked attention. It reconstructs the graph
        by calling the `FlashMaskAttnFunctor` surrogate layer.
        """
        hold_tensors = self._hold_tensors_queue.get()
        output = FlashMaskAttnCpFunctor.apply(
            query_states,
            key_states,
            value_states,
            learnable_sink,
            hold_tensors,
        )
        return output

    def __call__(self, *args, **kwds):
        """Makes the class instance callable, similar to a standard nn.Layer."""
        return self.forward(*args, **kwds)
