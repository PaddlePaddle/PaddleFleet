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

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import paddle
from paddle import nn

if TYPE_CHECKING:
    from paddle.distributed.communication.group import Group

logger = logging.getLogger(__name__)

from .fp8_utils import FP8_ALIGN
from .fused_a2a import (
    DeepEPCombineAsyncRefinedRecompute,
    fused_combine,
    fused_dispatch,
    get_hybrid_ep_buffer,
    hybrid_ep_combine,
    hybrid_ep_dispatch,
    quantize_activation_blockscaled_fast,
)
from .moe_utils import (
    AllGatherGroupOp,
    ReduceScatterGroupOp,
    _AllToAll,
    all_gather_group,
    manual_backward,
    permute,
    reduce_scatter_group,
    sort_chunks_by_idxs,
    unpermute,
    use_accuracy_compatible_kernel,
)

HAVE_HYBRID_EP = False
HYBRID_EP_LOAD_CACHED_KERNELS = True


def _sort_chunks_like_tokens(
    input: paddle.Tensor,
    split_sizes: list[int],
    sorted_idxs: list[int],
) -> paddle.Tensor:
    chunks = paddle.split(input, split_sizes, axis=0)
    return paddle.concat([chunks[i] for i in sorted_idxs], axis=0)


try:
    from paddlefleet_ops import is_hybrid_ep_available

    HAVE_HYBRID_EP = is_hybrid_ep_available()
except ImportError:
    HAVE_HYBRID_EP = False


def is_hybrid_ep_backend_selected(
    dispatcher_type: str | None = None,
) -> bool:
    selected_dispatcher = dispatcher_type or "deepep"
    if selected_dispatcher not in (
        "allgather",
        "alltoall",
        "deepep",
        "hybridep",
    ):
        raise ValueError(
            "moe_token_dispatcher_type must be one of: allgather, alltoall, deepep, hybridep"
        )
    if selected_dispatcher != "hybridep":
        return False
    if not HAVE_HYBRID_EP:
        raise ImportError(
            "moe_token_dispatcher_type=hybridep but HybridEP runtime is unavailable."
        )
    return True


def _try_setup_router_topk_metadata(
    manager,
    num_tokens: int,
    topk_weights: paddle.Tensor | None,
    topk_indices: paddle.Tensor | None,
) -> bool:
    if topk_weights is None or topk_indices is None:
        return False
    manager.token_probs = topk_weights.reshape(
        [num_tokens, manager.router_topk]
    )
    manager.token_indices = topk_indices.reshape(
        [num_tokens, manager.router_topk]
    )
    manager.token_indices.stop_gradient = True
    return True


class _DispatchManager(ABC):
    """
    A manager class to handle dispatch and combine processes for MoE models.

    DispatcherManager handles token dispatching according to the routing_map of format
    [num_local_tokens, world_size, num_instances]. The routing_map is a 3D tensor where each
    element indicates whether a token should be sent to a specific rank.

    num_instances is the maximum number of tokens instances dispatched into a target rank, it
    can be the number of local experts, or the size of sub_group.
    """

    @abstractmethod
    def setup_metadata(
        self,
        routing_map: paddle.Tensor,
        probs: paddle.Tensor,
        topk_weights: paddle.Tensor | None = None,
        topk_indices: paddle.Tensor | None = None,
    ):
        """Set up metadata of routing_map and probs.

        If ``topk_weights`` and ``topk_indices`` are provided (e.g. produced by
        the router), they will be used directly and the internal ``paddle.topk``
        call will be skipped.
        """
        pass

    @abstractmethod
    def dispatch(
        self,
        hidden_states: paddle.Tensor,
        fp8_dispatch: bool,
        async_finish: bool,
    ) -> paddle.Tensor:
        """Dispatch the hidden_states according to the routing_map."""
        pass

    @abstractmethod
    def combine(
        self, hidden_states: paddle.Tensor, combine_overlap_handle: dict | None
    ) -> paddle.Tensor:
        """Combine the hidden_states after expert processing."""
        pass

    @abstractmethod
    def get_dispatched_metadata(self) -> paddle.Tensor:
        """Get the metadata of the dispatched hidden_states."""
        pass

    @abstractmethod
    def get_permuted_hidden_states_by_experts(
        self, hidden_states: paddle.Tensor
    ) -> paddle.Tensor:
        """Get the permuted hidden states by instances."""
        pass

    @abstractmethod
    def get_restored_hidden_states_by_experts(
        self, hidden_states: paddle.Tensor
    ) -> paddle.Tensor:
        """Get the restored hidden states by instances."""
        pass


class _HybridEPManager(_DispatchManager):
    """
    HybridEP path using dispatch_with_permute/combine_with_unpermute only.

    The manager owns per-layer handles and count metadata. The communication
    buffer is shared at fused_a2a module scope, matching DeepEP and Megatron.
    """

    def __init__(
        self,
        group: Group,
        router_topk: int,
        num_experts: int | None = None,
        num_local_experts: int | None = None,
        moe_ep_barrier: bool = True,
        hybridep_buffer_configs: dict | None = None,
    ):
        if not HAVE_HYBRID_EP:
            raise ImportError("HybridEP runtime is not available.")

        self.group = group
        self.router_topk = router_topk
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.routing_map = None
        self.routing_probs = None
        self.token_indices = None
        self.token_probs = None
        self.dispatched_indices = None
        self.dispatched_probs = None
        self.tokens_per_expert = None
        self.padded_tokens_per_expert = None
        self.handle = None
        self._active_buffer = None
        self.hybridep_buffer_configs = hybridep_buffer_configs or {}

    def _get_buffer(
        self,
        hidden_states: paddle.Tensor,
        max_num_of_tokens_per_rank: int | None = None,
    ):
        hidden_dim = hidden_states.shape[-1]
        if max_num_of_tokens_per_rank is None:
            max_num_of_tokens_per_rank = hidden_states.shape[0]
        self._active_buffer = get_hybrid_ep_buffer(
            group=self.group,
            hidden_dim=hidden_dim,
            max_num_of_tokens_per_rank=max_num_of_tokens_per_rank,
            num_local_experts=self.num_local_experts,
            load_cached_kernels=HYBRID_EP_LOAD_CACHED_KERNELS,
            **self.hybridep_buffer_configs,
        )
        return self._active_buffer

    def _get_num_permuted_tokens_upper_bound(
        self, num_local_tokens: int
    ) -> int:
        total_routed_tokens = (
            num_local_tokens * self.group.nranks * self.router_topk
        )
        if FP8_ALIGN > 1:
            total_routed_tokens += self.num_local_experts * (FP8_ALIGN - 1)
        return total_routed_tokens

    def _indices_to_dense_metadata(
        self,
        token_indices: paddle.Tensor,
        token_weights: paddle.Tensor | None,
    ) -> tuple[paddle.Tensor, paddle.Tensor | None]:
        safe_indices = paddle.where(
            token_indices >= 0,
            token_indices,
            paddle.zeros_like(token_indices),
        ).astype("int64")
        one_hot = paddle.nn.functional.one_hot(
            safe_indices, num_classes=self.num_experts
        )
        valid_mask = (token_indices >= 0).astype(one_hot.dtype).unsqueeze(-1)
        one_hot = one_hot * valid_mask
        routing_map = paddle.sum(one_hot, axis=1).astype("bool")

        probs = None
        if token_weights is not None:
            probs = paddle.sum(
                one_hot.astype(token_weights.dtype)
                * token_weights.unsqueeze(-1),
                axis=1,
            )
            if probs.dtype != paddle.float32:
                probs = probs.astype("float32")
        return routing_map, probs

    def _get_dispatch_metadata(
        self,
        token_indices: paddle.Tensor | None,
        token_weights: paddle.Tensor | None,
    ) -> tuple[paddle.Tensor, paddle.Tensor | None]:
        if self.routing_map is not None:
            return self.routing_map, self.routing_probs
        assert token_indices is not None, (
            "HybridEP dispatch requires routing metadata."
        )
        return self._indices_to_dense_metadata(token_indices, token_weights)

    def setup_metadata(
        self,
        routing_map: paddle.Tensor,
        probs: paddle.Tensor,
        topk_weights: paddle.Tensor | None = None,
        topk_indices: paddle.Tensor | None = None,
    ):
        num_tokens = routing_map.shape[0]
        self.routing_map = routing_map.reshape(
            [num_tokens, self.num_experts]
        ).astype("bool")
        self.routing_probs = probs.reshape([num_tokens, self.num_experts])
        if self.routing_probs.dtype != paddle.float32:
            self.routing_probs = self.routing_probs.astype("float32")
        if _try_setup_router_topk_metadata(
            self, num_tokens, topk_weights, topk_indices
        ):
            return
        self.token_probs, self.token_indices = paddle.topk(
            self.routing_probs, self.router_topk, axis=-1
        )

    def _extract_tokens_per_expert(
        self,
        num_dispatched_tokens: int,
        local_expert_routing_map: paddle.Tensor,
    ):
        return (
            local_expert_routing_map[:num_dispatched_tokens]
            .astype("int64")
            .sum(axis=0)
        )

    def dispatch_overlap(
        self,
        hidden_states: paddle.Tensor,
        token_indices: paddle.Tensor,
        token_weights: paddle.Tensor,
        fp8_dispatch: bool = False,
        async_finish: bool = False,
        use_ue8m0: bool = False,
    ) -> paddle.Tensor:
        del async_finish
        self.token_indices = token_indices
        self.token_probs = token_weights
        hidden_states, self.dispatched_probs, scale = hybrid_ep_dispatch(
            hidden_states,
            token_indices,
            token_weights,
            self,
            fp8_dispatch,
        )
        self.dispatched_indices = None
        return hidden_states, None if scale is None else {"scale": scale}

    def _dispatch_with_permute_impl(
        self,
        hidden_states: paddle.Tensor,
        token_indices: paddle.Tensor,
        token_weights: paddle.Tensor,
        use_fp8: bool = False,
    ):
        buffer = self._get_buffer(hidden_states)
        routing_map, probs = self._get_dispatch_metadata(
            token_indices, token_weights
        )
        num_permuted_tokens = self._get_num_permuted_tokens_upper_bound(
            hidden_states.shape[0]
        )
        scaling_factor = None
        if use_fp8:
            hidden_states, scaling_factor = (
                paddle.incubate.nn.functional.fp8_quant_blockwise(
                    hidden_states,
                    quant_method="1x128",
                    input_transpose=False,
                    output_scale_transpose=True,
                    return_transpose_only=False,
                )
            )
            scaling_factor = scaling_factor.T.contiguous()
        (
            hidden_states,
            dispatched_probs,
            scale,
            tokens_per_expert,
            self.handle,
        ) = buffer.dispatch_with_permute(
            hidden=hidden_states,
            routing_map=routing_map,
            probs=probs,
            num_of_experts_per_rank=self.num_local_experts,
            use_fp8=use_fp8,
            scaling_factor=scaling_factor,
            pad_multiple=FP8_ALIGN if use_fp8 else None,
            num_permuted_tokens=num_permuted_tokens,
            non_blocking=True,
        )
        self.padded_tokens_per_expert = tokens_per_expert
        (
            _sparse_to_dense_map,
            _rdma_to_attn_map,
            _attn_to_rdma_map,
            num_dispatched_tokens_tensor,
            local_expert_routing_map,
            *_,
        ) = self.handle
        num_dispatched_tokens = int(num_dispatched_tokens_tensor.item())
        self.tokens_per_expert = self._extract_tokens_per_expert(
            num_dispatched_tokens,
            local_expert_routing_map,
        )
        return hidden_states, dispatched_probs, scale

    def dispatch(
        self,
        hidden_states: paddle.Tensor,
        fp8_dispatch: bool = False,
        async_finish: bool = False,
        use_ue8m0: bool = False,
        using_sonic_moe: bool = False,
    ) -> paddle.Tensor:
        return self.dispatch_overlap(
            hidden_states,
            self.token_indices,
            self.token_probs,
            fp8_dispatch=fp8_dispatch,
            async_finish=async_finish,
        )

    def combine(
        self,
        hidden_states: paddle.Tensor,
        combine_overlap_handle: dict | None = None,
        async_finish: bool = False,
        use_rr_deepep_combine: bool = False,
        fp8_dispatch: bool = False,
        combine_grad_handle: dict | None = None,
    ) -> paddle.Tensor:
        del async_finish, use_rr_deepep_combine
        if combine_overlap_handle is not None:
            raise NotImplementedError(
                "HybridEP backend does not support combine overlap in PaddleFleet."
            )
        hidden_states = hybrid_ep_combine(hidden_states, self)
        self.dispatched_probs = None
        self.handle = None
        return hidden_states

    def get_dispatched_metadata(self) -> paddle.Tensor:
        if self.dispatched_indices is None or self.dispatched_probs is None:
            raise NotImplementedError(
                "HybridEP backend does not expose fused-node dispatch metadata for the current mode."
            )
        return self.dispatched_indices, self.dispatched_probs

    def get_number_of_tokens_per_expert(self) -> paddle.Tensor:
        return self.tokens_per_expert

    def get_permuted_hidden_states_by_experts(
        self, hidden_states: paddle.Tensor
    ) -> paddle.Tensor:
        return hidden_states

    def get_restored_hidden_states_by_experts(
        self, hidden_states: paddle.Tensor
    ) -> paddle.Tensor:
        if self.dispatched_probs is None:
            return hidden_states
        return hidden_states * self.dispatched_probs.astype(
            hidden_states.dtype
        ).unsqueeze(-1)


class _DeepEPManager(_DispatchManager):
    """
    A manager class to handle fused all-to-all communication processes for MoE models using
    DeepEP backend. See https://github.com/deepseek-ai/deepep for more details.

    The workflow of the DeepEP dispatcher is:
    (1) setup_metadata(): Process routing map and probabilities to prepare dispatch metadata
    (2) dispatch():
        - Use fused kernel to permute tokens and perform all-to-all communication in single step
    (3) get_permuted_hidden_states_by_instances():
        - Convert routing map and probabilities to multihot format
        - Permute tokens using fused kernel
    (4) get_restored_hidden_states_by_instances():
        - Reverse permutation using fused kernel
    (5) combine():
        - Reverse process using fused kernel to unpermute and perform all-to-all in single step

    This implementation uses fused communication kernels (fused_dispatch/fused_combine) that
    combine permutation and communication operations for improved efficiency compared to
    separate permute+alltoall steps.
    """

    def __init__(
        self,
        group: Group,
        router_topk: int,
        num_experts: int | None = None,
        num_local_experts: int | None = None,
        moe_ep_barrier: bool = True,
    ):
        self.group = group
        self.router_topk = router_topk
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.moe_ep_barrier = moe_ep_barrier

        # Metadata
        self.token_indices = None
        self.token_probs = None
        # Handle used for combine operation
        self.handle = None

        if fused_dispatch is None:
            raise ImportError(
                "DeepEP is not supported in your paddlepaddle whl package."
            )
        self._rr_fusedcombined = None

    def setup_metadata(
        self,
        routing_map: paddle.Tensor,
        probs: paddle.Tensor,
        topk_weights: paddle.Tensor | None = None,
        topk_indices: paddle.Tensor | None = None,
    ):
        num_tokens = routing_map.shape[0]

        if _try_setup_router_topk_metadata(
            self, num_tokens, topk_weights, topk_indices
        ):
            return

        routing_map = routing_map.reshape([num_tokens, self.num_experts])
        probs = probs.reshape([num_tokens, self.num_experts])
        # Convert the format of routing map from multihot to indices.
        self.token_probs, self.token_indices = paddle.topk(
            probs, self.router_topk, axis=-1
        )

    def dispatch_overlap(
        self,
        hidden_states: paddle.Tensor,
        token_indices: paddle.Tensor,
        token_weights: paddle.Tensor,
        fp8_dispatch: bool = False,
        async_finish: bool = False,
        use_ue8m0: bool = False,
    ) -> paddle.Tensor:
        hidden_states, dispatched_probs, states, scale = fused_dispatch(
            hidden_states,
            token_indices,
            token_weights,
            self.num_experts,
            self.group,
            fp8_dispatch=fp8_dispatch,
            async_finish=async_finish,
            use_ue8m0=use_ue8m0,
        )
        self.handle = states["handle"]
        self.tokens_per_expert = states["tokens_per_expert"]
        self.dispatched_indices = states["dispatched_indices"]
        self.dispatched_probs = dispatched_probs

        return hidden_states, scale

    def dispatch(
        self,
        hidden_states: paddle.Tensor,
        fp8_dispatch: bool = False,
        async_finish: bool = False,
        use_ue8m0: bool = False,
        using_sonic_moe: bool = False,
    ) -> paddle.Tensor:
        hidden_states, dispatched_probs, states, scale = fused_dispatch(
            hidden_states,
            self.token_indices,
            self.token_probs,
            self.num_experts,
            self.group,
            fp8_dispatch=fp8_dispatch,
            async_finish=async_finish,
            moe_ep_barrier=self.moe_ep_barrier,
            use_ue8m0=use_ue8m0,
            using_sonic_moe=using_sonic_moe,
        )
        self.handle = states["handle"]
        self.tokens_per_expert = states["tokens_per_expert"]
        self.dispatched_indices = states["dispatched_indices"]
        self.dispatched_probs = dispatched_probs

        return hidden_states, scale

    def _indices_to_multihot(self, indices, probs):
        """
        Converts a tensor of indices to a multihot vector.

        Args:
            indices (paddle.Tensor): [num_tokens, topk] token indices, where -1 means masked out.
            probs (paddle.Tensor): [num_tokens, topk] token probabilities.

        Returns:
            tuple[paddle.Tensor, paddle.Tensor]:
                - routing_map: Multihot vector.
                - probs: Multihot probabilities.
        """
        batch_size = indices.shape[0]
        multihot_routing_map = paddle.zeros(
            (batch_size, self.num_local_experts), dtype=paddle.int64
        )

        multihot_probs = paddle.zeros(
            (batch_size, self.num_local_experts), dtype=paddle.float32
        )

        mask = indices != -1
        valid_indices = indices[mask]
        row_indices = paddle.arange(batch_size).repeat_interleave(
            mask.sum(axis=1)
        )
        multihot_routing_map[row_indices, valid_indices] = 1
        multihot_probs[row_indices, valid_indices] = probs[mask]
        return multihot_routing_map.cast(paddle.bool), multihot_probs

    def get_dispatched_metadata(self) -> paddle.Tensor:
        return self.dispatched_indices, self.dispatched_probs

    def get_number_of_tokens_per_expert(self) -> paddle.Tensor:
        """
        Get the number of tokens per expert.
        """
        return self.tokens_per_expert

    def combine(
        self,
        hidden_states: paddle.Tensor,
        combine_overlap_handle: dict | None = None,
        async_finish: bool = False,
        use_rr_deepep_combine: bool = False,
        fp8_dispatch: bool = False,
        combine_grad_handle: dict | None = None,
    ) -> paddle.Tensor:
        if combine_overlap_handle is not None and use_rr_deepep_combine:
            if self._rr_fusedcombined is None:
                self._rr_fusedcombined = DeepEPCombineAsyncRefinedRecompute()
            elif not isinstance(
                self._rr_fusedcombined, DeepEPCombineAsyncRefinedRecompute
            ):
                raise RuntimeError(
                    f"_rr_fusedcombined type mismatch: expected DeepEPCombineAsyncRefinedRecompute, "
                    f"got {type(self._rr_fusedcombined).__name__}."
                )
        if fp8_dispatch is True:
            assert combine_grad_handle is not None, (
                "fp8_dispatch=True, but combine_grad_handle is None."
            )
        hidden_states = fused_combine(
            hidden_states,
            self.group,
            self.handle,
            _rr_fusedcombined=self._rr_fusedcombined,
            combine_overlap_handle=combine_overlap_handle,
            async_finish=async_finish,
            moe_ep_barrier=self.moe_ep_barrier,
            use_rr_deepep_combine=use_rr_deepep_combine,
            fp8_dispatch=fp8_dispatch,
            combine_grad_handle=combine_grad_handle,
        )
        # Release the handle and token_indices after combine operation
        self.handle = None
        self.token_indices = None
        self.token_probs = None
        self.dispatched_probs = None
        self.dispatched_indices = None
        return hidden_states

    def get_permuted_hidden_states_by_experts(
        self, hidden_states: paddle.Tensor
    ) -> paddle.Tensor:
        self.dispatched_routing_map, self.dispatched_probs = (
            self._indices_to_multihot(
                self.dispatched_indices, self.dispatched_probs
            )
        )
        self.hidden_shape_before_permute = hidden_states.shape
        hidden_states, self.reversed_mapping_for_combine = permute(
            hidden_states,
            self.dispatched_routing_map,
            num_out_tokens=sum(self.tokens_per_expert),
        )
        return hidden_states

    def get_restored_hidden_states_by_experts(
        self, hidden_states: paddle.Tensor
    ) -> paddle.Tensor:
        input_dtype = hidden_states.dtype
        assert self.dispatched_probs.dtype == paddle.float32, (
            "DeepEP only supports float32 probs"
        )
        hidden_states = unpermute(
            hidden_states,
            self.reversed_mapping_for_combine,
            restore_shape=self.hidden_shape_before_permute,
            routing_map=self.dispatched_routing_map,
            probs=self.dispatched_probs,
        )
        return hidden_states.to(input_dtype)


class MoETokenDispatcher:
    """
    MoE Token Dispatcher
    """

    def __init__(self, ep_group) -> None:
        """
        Initialize the MoE Token Dispatcher.
        """
        self._ep_group = ep_group

    @property
    def ep_group(self):
        """Get expert model parallel group."""
        return self._ep_group

    @property
    def ep_size(self):
        """Get expert model parallel world_size."""
        return self.ep_group.world_size

    @abstractmethod
    def token_permutation(
        self,
        tokens: paddle.Tensor,
        probs: paddle.Tensor,
        routing_map: paddle.Tensor,
    ):
        """Dispatch tokens to experts.

        Args:
            tokens (paddle.Tensor): Input tokens.
            probs (paddle.Tensor): The routing probability tensor [num_tokens, num_experts].
            routing_map (paddle.Tensor): Token to expert mapping tensor.

        Returns:
            paddle.Tensor: Tokens tensor.
        """
        raise NotImplementedError("Dispatch function not implemented.")

    @abstractmethod
    def token_unpermutation(
        self, expert_output: paddle.Tensor, bias: paddle.Tensor = None
    ):
        """Restores the expert output to its original ordering.

        Args:
            expert_output (paddle.Tensor): The output tensor from the expert models.
            bias (paddle.Tensor): The bias tensor.

        Returns:
            (paddle.Tensor, paddle.Tensor): Unpermuted activation and optional bias.
        """
        raise NotImplementedError("Restore function not implemented.")


class MoEFlexTokenDispatcher(MoETokenDispatcher):
    """
    Flexible token dispatcher for MoE models with Efficient-A2A communication kernels.
    """

    def __init__(
        self,
        num_local_experts: int,
        num_experts_per_tok: int,
        n_routed_experts: int,
        ep_group: Group,
        moe_ep_barrier: bool = True,
        dispatcher_type: str | None = None,
        hybridep_buffer_configs: dict | None = None,
    ):
        super().__init__(ep_group)

        self.num_local_experts = num_local_experts
        assert self.ep_size > 1, "Flex token dispatcher requires EP > 1"
        manager_cls = (
            _HybridEPManager
            if is_hybrid_ep_backend_selected(dispatcher_type)
            else _DeepEPManager
        )
        manager_kwargs = {
            "group": self.ep_group,
            "router_topk": num_experts_per_tok,
            "num_experts": n_routed_experts,
            "num_local_experts": self.num_local_experts,
            "moe_ep_barrier": moe_ep_barrier,
        }
        if manager_cls is _HybridEPManager:
            manager_kwargs["hybridep_buffer_configs"] = hybridep_buffer_configs
        self._comm_manager = manager_cls(**manager_kwargs)

    def dispatch_preprocess(
        self,
        hidden_states: paddle.Tensor,
        probs: paddle.Tensor,
        routing_map: paddle.Tensor,
        topk_weights: paddle.Tensor | None = None,
        topk_indices: paddle.Tensor | None = None,
    ):
        self.hidden_shape = hidden_states.shape
        hidden_states = hidden_states.view([-1, self.hidden_shape[-1]])
        self._comm_manager.setup_metadata(
            routing_map, probs, topk_weights, topk_indices
        )
        return hidden_states

    def dispatch_preprocess_overlap(
        self,
        hidden_states: paddle.Tensor,
        token_probs: paddle.Tensor,
        token_indices: paddle.Tensor,
    ):
        self.hidden_shape = hidden_states.shape
        hidden_states = hidden_states.view([-1, self.hidden_shape[-1]])
        self._comm_manager.routing_map = None
        self._comm_manager.routing_probs = None
        self._comm_manager.token_probs = token_probs
        self._comm_manager.token_indices = token_indices
        return hidden_states

    def token_dispatch_overlap(
        self,
        hidden_states: paddle.Tensor,
        token_indices: paddle.Tensor,
        token_weights: paddle.Tensor,
        fp8_dispatch: bool,
        async_finish: bool = False,
        use_ue8m0: bool = False,
    ):
        return self._comm_manager.dispatch_overlap(
            hidden_states,
            token_indices,
            token_weights,
            fp8_dispatch,
            async_finish,
            use_ue8m0=use_ue8m0,
        )

    def token_dispatch(
        self,
        hidden_states: paddle.Tensor,
        fp8_dispatch: bool,
        async_finish: bool = False,
        use_ue8m0: bool = False,
        using_sonic_moe: bool = False,
    ):
        return self._comm_manager.dispatch(
            hidden_states,
            fp8_dispatch,
            async_finish,
            use_ue8m0=use_ue8m0,
            using_sonic_moe=using_sonic_moe,
        )

    def dispatch_postprocess(
        self,
        hidden_states: paddle.Tensor,
    ):
        global_input_tokens = (
            self._comm_manager.get_permuted_hidden_states_by_experts(
                hidden_states
            )
        )
        tokens_per_expert = self._comm_manager.get_number_of_tokens_per_expert()

        return global_input_tokens, tokens_per_expert

    def combine_preprocess(self, hidden_states: paddle.Tensor):
        return self._comm_manager.get_restored_hidden_states_by_experts(
            hidden_states
        )

    def token_combine(
        self,
        hidden_states: paddle.Tensor,
        combine_overlap_handle: dict | None = None,
        async_finish=False,
        fp8_combine_grad_handle: dict | None = None,
    ):
        del fp8_combine_grad_handle
        if combine_overlap_handle is not None:
            raise ValueError(
                "MoEFlexTokenDispatcher (alltoall) does not support "
                "combine_overlap_handle"
            )
        return self._comm_manager.combine(
            hidden_states, async_finish=async_finish
        )

    def combine_postprocess(self, hidden_states: paddle.Tensor):
        return hidden_states.reshape(self.hidden_shape)

    def get_dispatched_routing(self):
        """Return (dispatched_indices, dispatched_probs, tokens_per_expert)."""
        return (
            self._comm_manager.dispatched_indices,
            self._comm_manager.dispatched_probs,
            self._comm_manager.tokens_per_expert,
        )

    def token_permutation(
        self,
        hidden_states: paddle.Tensor,
        probs: paddle.Tensor,
        routing_map: paddle.Tensor,
        topk_weights: paddle.Tensor | None = None,
        topk_indices: paddle.Tensor | None = None,
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        self.hidden_shape = hidden_states.shape
        hidden_states = hidden_states.view([-1, self.hidden_shape[-1]])

        self._comm_manager.setup_metadata(
            routing_map, probs, topk_weights, topk_indices
        )
        hidden_states, scale = self._comm_manager.dispatch(hidden_states)
        global_input_tokens = (
            self._comm_manager.get_permuted_hidden_states_by_experts(
                hidden_states
            )
        )
        tokens_per_expert = self._comm_manager.get_number_of_tokens_per_expert()

        return global_input_tokens, tokens_per_expert

    def token_unpermutation(
        self, hidden_states: paddle.Tensor, bias: paddle.Tensor | None = None
    ) -> tuple[paddle.Tensor, paddle.Tensor | None]:
        assert bias is None, "Bias is not supported in MoEFlexTokenDispatcher"
        hidden_states = (
            self._comm_manager.get_restored_hidden_states_by_experts(
                hidden_states
            )
        )
        hidden_states = self._comm_manager.combine(hidden_states)

        hidden_states = hidden_states.reshape(self.hidden_shape)
        return hidden_states, None


class AllToAllTokenDispatcher(nn.Layer):
    """
    All-to-All EP
    """

    def __init__(
        self,
        moe_group: Group,
        expert_model_parallel_size: int,
        num_experts_per_device: int,
        local_expert_indices: list,
    ):
        nn.Layer.__init__(self)
        self.moe_group = moe_group
        self.expert_model_parallel_size = expert_model_parallel_size
        self.num_experts_per_device = num_experts_per_device
        self.local_expert_indices = local_expert_indices
        self.num_local_experts = len(local_expert_indices)

    def dispatch_preprocess(
        self,
        hidden_states: paddle.Tensor,
        probs: paddle.Tensor,
        mask: paddle.Tensor,  # routing_map
        topk_weights: paddle.Tensor | None = None,
        topk_indices: paddle.Tensor | None = None,
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        self.routing_map = mask
        self.probs = probs
        self.num_experts = (
            self.num_experts_per_device * self.expert_model_parallel_size
        )
        mask = mask.to(paddle.int32)

        if len(hidden_states.shape) == 3:
            batch_size, seq_len, d_model = hidden_states.shape
        else:
            seq_len, d_model = hidden_states.shape
        reshaped_input = hidden_states.reshape([-1, d_model])
        self.d_model = d_model
        self.reshaped_input_shape = reshaped_input.shape
        tokens_per_expert = mask.sum(axis=0)  # Shape: [num_experts]
        tokens_per_expert = tokens_per_expert.detach()
        tokens_per_ep_rank = tokens_per_expert.reshape(
            [self.expert_model_parallel_size, -1]
        ).sum(axis=1)
        # First All-to-All: Exchange expert token counts across ranks
        # Returns `tokens_per_expert_group` is for current rank
        num_global_tokens_per_expert = AllGatherGroupOp.apply(
            tokens_per_expert, group=self.moe_group
        ).reshape(self.expert_model_parallel_size, self.num_experts)
        num_global_tokens_per_local_expert = num_global_tokens_per_expert[
            :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
        ].clone()

        # Can also use the two AllToAll functions below instead of the above AllGather
        # It will save memory , but also has more accuracy diff with DeepEP version
        # global_tokens_per_expert = _AllToAll.apply(
        #     [tokens_per_expert.shape[0]],
        #     tokens_per_expert,
        #     group=self.moe_group,
        # )
        # num_global_tokens_per_local_expert = global_tokens_per_expert.reshape(self.expert_model_parallel_size, self.num_local_experts)

        if num_global_tokens_per_local_expert.sum().item() == 0:
            self.is_empty_tokens = True
        else:
            self.is_empty_tokens = False

        self.tokens_per_expert = num_global_tokens_per_local_expert.sum(axis=0)

        num_global_tokens_per_rank = num_global_tokens_per_local_expert.sum(
            axis=1
        )

        self.num_global_tokens_per_local_expert = (
            num_global_tokens_per_local_expert.reshape(
                -1, self.num_local_experts
            )
        )

        self.output_splits = num_global_tokens_per_rank.cpu().tolist()
        num_local_tokens_per_expert = self.routing_map.sum(dim=0)
        self.input_split_sizes = num_local_tokens_per_expert.reshape(
            self.expert_model_parallel_size, self.num_local_experts
        ).sum(axis=1)
        self.output_shape_tokens = [
            num_global_tokens_per_rank.sum().cpu().item(),
            d_model,
        ]

        (
            permutated_local_input_tokens,
            self.reversed_local_input_permutation_mapping,
        ) = permute(reshaped_input, self.routing_map)
        if use_accuracy_compatible_kernel():
            num_routed_tokens = int(tokens_per_expert.sum().item())
            routing_map = self.routing_map.cast(paddle.bool).T.contiguous()
            flat_sorted = paddle.argsort(
                routing_map.reshape([-1]).cast("int32"),
                descending=True,
                stable=True,
            )[:num_routed_tokens]
            self.permuted_local_probs = paddle.index_select(
                self.probs.T.contiguous().reshape([-1]),
                flat_sorted,
                axis=0,
            )
        self.permutated_local_input_tokens_shape = (
            permutated_local_input_tokens.shape
        )

        return permutated_local_input_tokens

    def token_dispatch(
        self,
        permutated_local_input_tokens: paddle.Tensor,
        fp8_dispatch: bool = False,
        async_finish: bool = False,
        use_ue8m0: bool = False,
        using_sonic_moe: bool = False,
    ):
        # Second All-to-All: Exchange expert tokens across ranks. `gathered_tokens` are the tokens that will be processed by current rank
        global_input_tokens = _AllToAll.apply(
            self.output_shape_tokens,
            permutated_local_input_tokens,  # sorted_tokens,
            out_split_sizes=self.output_splits,
            in_split_sizes=self.input_split_sizes,
            group=self.moe_group,
        )
        if use_accuracy_compatible_kernel():
            # Match Megatron's all-to-all backward numerics by routing probs through a
            # 2D [tokens, 1] tensor, like hidden-state dispatch.
            global_input_probs_2d = _AllToAll.apply(
                [self.output_shape_tokens[0], 1],
                self.permuted_local_probs.unsqueeze(-1),
                out_split_sizes=self.output_splits,
                in_split_sizes=self.input_split_sizes,
                group=self.moe_group,
            )
            self.global_input_probs = global_input_probs_2d.squeeze(-1)

        return global_input_tokens, None

    def dispatch_postprocess(
        self,
        global_input_tokens: paddle.Tensor,
    ):
        input_chunk_idxs = paddle.arange(self.num_experts)
        # [num_local_experts, ep_size]. Sort the input chunks by local experts.
        self.sort_input_by_local_experts = input_chunk_idxs.reshape(
            -1, self.num_local_experts
        ).T.ravel()
        # [ep_size, num_local_experts]. Restore the output chunks by local experts.
        self.restore_output_by_local_experts = input_chunk_idxs.reshape(
            self.num_local_experts, -1
        ).T.ravel()

        if self.num_local_experts > 1 and not self.is_empty_tokens:
            split_sizes_list = (
                self.num_global_tokens_per_local_expert.ravel().tolist()
            )
            sorted_idxs_list = self.sort_input_by_local_experts.tolist()
            global_input_tokens, _ = sort_chunks_by_idxs(
                global_input_tokens,
                self.num_global_tokens_per_local_expert.ravel(),
                self.sort_input_by_local_experts,
            )
            if use_accuracy_compatible_kernel():
                self.global_input_probs = _sort_chunks_like_tokens(
                    self.global_input_probs,
                    split_sizes_list,
                    sorted_idxs_list,
                )
        sorted_tokens = global_input_tokens
        self.tokens_per_expert_post_gather = self.tokens_per_expert
        return sorted_tokens, self.tokens_per_expert_post_gather

    def combine_preprocess(self, hidden_states: paddle.Tensor):
        if self.num_local_experts > 1 and not self.is_empty_tokens:
            hidden_states, _ = sort_chunks_by_idxs(
                hidden_states,
                self.num_global_tokens_per_local_expert.T.ravel(),
                self.restore_output_by_local_experts,
            )
        return hidden_states

    def token_combine(
        self,
        hidden_states: paddle.Tensor,
        combine_overlap_handle: dict | None = None,
        async_finish: bool = False,
        fp8_combine_grad_handle: dict | None = None,
    ):
        del combine_overlap_handle, async_finish, fp8_combine_grad_handle
        permutated_local_input_tokens = _AllToAll.apply(
            self.permutated_local_input_tokens_shape,
            hidden_states,
            out_split_sizes=self.input_split_sizes,
            in_split_sizes=self.output_splits,
            group=self.moe_group,
        )
        return permutated_local_input_tokens

    def combine_postprocess(self, permutated_local_input_tokens: paddle.Tensor):
        output = unpermute(
            permutated_local_input_tokens,
            self.reversed_local_input_permutation_mapping,
            restore_shape=self.reshaped_input_shape,
            probs=(None if use_accuracy_compatible_kernel() else self.probs),
            routing_map=self.routing_map,
        )
        return output


class _RouterAllGather(paddle.autograd.PyLayer):
    """AllGather for router-local tensors (allgather dispatcher only).

    Forward: same as ``AllGatherGroupOp`` — concatenates the local
    ``[seq_local, ...]`` tensor across the EP group along axis 0 to
    ``[seq_global, ...]``.

    Backward: **scatter (not reduce-scatter)**. Each token has exactly one
    origin rank — its router lives there and only there. Only the slice owned
    by the current rank is a valid gradient for *this* rank's router; the rest
    belongs to other ranks' routers and must not be summed in.
    ``AllGatherGroupOp`` was designed for hidden_states (unique-per-rank
    pre-AllGather, hence reduce-scatter on backward); reusing it for
    router-local metadata is a semantics mismatch.

    Gradient shape: this layer's output (``_global_topk_weights``, shape
    ``[T_global, K]``) is consumed downstream *only* through
    ``_differentiable_router_scores``, which gathers it via
    ``dispatched_probs.reshape(-1)[gather_idx]``. SonicMoE's ``_DownProjection``
    emits a 1-D ``ds`` over the gathered/padded layout, but that flows back
    through ``_GatherRouterScores`` (scatter into a flat ``[T_global*K]``
    buffer) and the ``reshape(-1)`` op, whose autograd restores the original
    2-D ``[T_global, K]`` shape. So the gradient arriving here matches the
    forward output shape exactly; no flattening reaches this edge.
    """

    @staticmethod
    def forward(ctx, input, group):
        ctx.group = group
        ctx.input_shape = list(input.shape)
        if group is None or group.nranks == 1:
            return input.clone()
        output_shape = list(input.shape)
        output_shape[0] = output_shape[0] * group.nranks
        output = paddle.empty(shape=output_shape, dtype=input.dtype)
        paddle.distributed.stream.all_gather(
            output, input, group=group, use_calc_stream=True
        )
        return output

    @staticmethod
    def backward(ctx, grad):
        group = ctx.group
        local_shape = ctx.input_shape
        if group is None or group.nranks == 1:
            if list(grad.shape) != local_shape:
                grad = grad.reshape(local_shape)
            return grad
        # The incoming grad matches the forward output shape
        # ([T_global, *trailing]); see class docstring for why no flattening
        # reaches this edge. Split along axis 0 and take this rank's segment
        # (scatter — no cross-rank reduction).
        global_shape = [local_shape[0] * group.nranks, *local_shape[1:]]
        if list(grad.shape) != global_shape:
            # Defensive: a downstream kernel emitted an unexpected layout.
            # Reshape only if the element count still matches, else fail loud.
            expected_numel = 1
            for _d in global_shape:
                expected_numel *= _d
            if int(grad.numel()) != expected_numel:
                raise ValueError(
                    "_RouterAllGather.backward: incoming grad has "
                    f"{int(grad.numel())} elements but the AllGather'd router "
                    f"tensor requires {expected_numel} (global_shape="
                    f"{global_shape})."
                )
            grad = grad.reshape(global_shape)
        chunks = paddle.split(grad, num_or_sections=group.nranks, axis=0)
        out = chunks[group.rank].contiguous()
        if list(out.shape) != local_shape:
            out = out.reshape(local_shape)
        return out


class _PreAllGatherResult(paddle.autograd.PyLayer):
    """Wraps a pre-issued async AllGather result with proper autograd.

    Used by :class:`AllGatherTokenDispatcher` to overlap the hidden_states
    AllGather (issued on the comm stream before gate) with gate computation
    (on the calc stream).  This layer ``task.wait()``-s for the async
    AllGather to complete and returns the pre-filled output buffer.
    Backward is ReduceScatter (same as ``AllGatherGroupOp`` backward).
    """

    @staticmethod
    def forward(ctx, hidden_states, handle):
        # handle: dict {"output": Tensor, "task": Task, "group": Group}
        handle["task"].wait()
        ctx.group = handle["group"]
        return handle["output"]

    @staticmethod
    def backward(ctx, grad):
        # Backward of AllGather is ReduceScatter.
        # ``handle`` is a plain dict (not a Tensor): Paddle's PyLayer
        # counts only tensor positional args when matching backward
        # returns to forward inputs, so backward must return exactly 1
        # value (the grad for ``hidden_states``). Adding a ``None`` for
        # ``handle`` would raise a tuple-arity mismatch on current
        # Paddle. Verified by ``test_consumes_fake_handle``.
        grad_input = ReduceScatterGroupOp.apply(grad, ctx.group)
        return grad_input


class _PreAllGatherFP8Result(paddle.autograd.PyLayer):
    """FP8 counterpart of :class:`_PreAllGatherResult`.

    Consumes a handle pre-issued by ``AllGatherTokenDispatcher.pre_allgather``
    when ``fp8_dispatch=True``: local quantize ran on calc stream before this,
    and two async AllGathers (fp8 uint8 view + fp32 scale) were launched on
    the comm stream so they overlap with gate compute. Forward waits both
    tasks and returns ``(fp8_global, scale_global)``.

    The fp8 output is fed straight into SonicMoE's fused ``_UpProjection`` as
    ``prequant_activation_payload=(x_fp8_global, scale_global)``. On backward,
    ``_UpProjection`` produces a **bf16** ``dx`` for that activation input,
    which the autograd engine delivers here as ``grad_output`` on the fp8
    output edge (Paddle would normally try to materialize an fp8-typed grad to
    match the fp8 forward output dtype; ``set_grad_in_dtype_consistent(False)``
    + ``set_materialize_grads(False)`` disable that so we receive the native
    bf16 grad untouched). Backward then ReduceScatters that bf16 ``dx`` back to
    the local token shard — the dual of the forward AllGather — without any
    fp8 reduction or dtype cast.
    """

    @staticmethod
    def forward(ctx, hidden_states, handle):
        handle["task_x"].wait()
        handle["task_s"].wait()
        ctx.group = handle["group"]
        x_fp8_global = handle["x_fp8_global_uint8"].view("float8_e4m3fn")
        scale_global = handle["scale_global"]
        # The fp8 output carries a bf16 gradient (SonicMoE's _UpProjection emits
        # bf16 dx for its activation input). Tell Paddle not to coerce the
        # incoming grad to the fp8 output dtype, and not to materialize a zero
        # fp8 grad buffer if the edge is unused.
        ctx.set_grad_in_dtype_consistent(False)
        ctx.set_materialize_grads(False)
        return x_fp8_global, scale_global

    @staticmethod
    def backward(ctx, grad_output, grad_scale=None):
        # grad_output: bf16 dx [T_global, H] from _UpProjection.backward.
        # grad_scale: gradient for the AllGather'd scale tensor — the scale is
        # consumed only as a non-differentiable prequant payload, so it is None.
        del grad_scale
        group = ctx.group
        if grad_output is None:
            # No gradient reached the fp8 activation edge (e.g. recompute warm-up
            # / inference). Nothing to scatter back.
            return None
        if group is None or group.nranks == 1:
            return grad_output
        grad_input = ReduceScatterGroupOp.apply(grad_output, group)
        return grad_input


def _reduce_scatter_async(input, group):
    """Async ReduceScatter (sum, axis 0) on the comm stream. Returns ``(output, task)``.

    Issues a non-blocking ReduceScatter on the dedicated NCCL comm stream so
    the caller can run independent compute on the calc stream and only
    ``task.wait()`` right before consuming ``output``. Mirrors the async
    AllGather pattern used by ``pre_allgather``.
    """
    input = input.contiguous()
    out_shape = list(input.shape)
    assert out_shape[0] % group.nranks == 0, (
        f"ReduceScatter input rows {out_shape[0]} not divisible by "
        f"nranks {group.nranks}"
    )
    out_shape[0] //= group.nranks
    output = paddle.empty(shape=out_shape, dtype=input.dtype)
    task = paddle.distributed.stream.reduce_scatter(
        output,
        input,
        op=paddle.distributed.ReduceOp.SUM,
        group=group,
        sync_op=False,
        use_calc_stream=False,
    )
    return output, task


def _all_gather_async(input, group):
    """Async AllGather (axis 0) on the comm stream. Returns ``(output, task)``.

    Dual of :func:`_reduce_scatter_async`; used by the combine backward to
    overlap the gradient AllGather with the shared-expert backward compute.
    """
    input = input.contiguous()
    out_shape = list(input.shape)
    out_shape[0] *= group.nranks
    output = paddle.empty(shape=out_shape, dtype=input.dtype)
    task = paddle.distributed.stream.all_gather(
        output,
        input,
        group=group,
        sync_op=False,
        use_calc_stream=False,
    )
    return output, task


class _AllGatherCombineAsync(paddle.autograd.PyLayer):
    """ReduceScatter combine fused with a user-supplied subgraph (e.g. shared experts).

    AllGather-path counterpart of :class:`DeepEPCombineAsync`. Forward issues
    the combine ReduceScatter on the **comm stream** (async), then runs
    ``fn(*fn_args)`` (the shared-expert subgraph) on the **calc stream** via
    :func:`manual_backward` so the two genuinely overlap, and only waits on the
    collective right before returning. Backward mirrors this: the gradient
    AllGather (dual of ReduceScatter) is issued async on the comm stream while
    ``bwf`` runs the shared-expert backward on the calc stream, then waits.

    Forward signature: ``(x, group, *fn_args, fn, is_first_fwd)``.
    Backward returns ``(grad_x, *fn_args_grads)`` — Paddle's PyLayer
    skips non-tensor positional args (``group``) when matching arity.

    Overlap is valid because ``fn``'s inputs (local hidden states) are
    independent of ``x`` (the gathered expert outputs), so the shared-expert
    compute has no data dependency on the in-flight combine collective — the
    same precondition that makes ``DeepEPCombineAsync`` correct. Async only
    changes the execution stream/timing; the ReduceScatter/AllGather semantics
    (and thus numerics) are identical to the synchronous path, and combine
    stays bf16 in both directions to match deepep — ``_DownProjection.backward``
    self-quantizes ``dout`` to fp8, which (since AllGather is a lossless row
    concat) is numerically identical to quantizing before the collective.
    """

    @staticmethod
    def forward(
        ctx,
        x,
        group,
        *fn_args,
        fn,
        is_first_fwd=False,
    ):
        if fn is None:
            raise ValueError(
                "_AllGatherCombineAsync requires a non-None fn for overlap."
            )
        ctx.group = group

        if group is None or group.nranks == 1:
            combined_x = x.clone()
            ctx.bwf, fn_out = manual_backward(fn, is_first_fwd, *fn_args)
            return (combined_x,) + fn_out  # noqa: RUF005

        # 1) Issue the combine ReduceScatter on the comm stream (async). NCCL
        #    inserts the cross-stream dependency on ``x`` (produced on calc
        #    stream) automatically.
        combined_x, task = _reduce_scatter_async(x, group)
        # 2) Run the shared-expert subgraph on the calc stream so it overlaps
        #    with the in-flight ReduceScatter (its inputs are independent of x).
        ctx.bwf, fn_out = manual_backward(fn, is_first_fwd, *fn_args)
        # 3) Synchronize before the combined output is consumed downstream.
        task.wait()

        return (combined_x,) + fn_out  # noqa: RUF005

    @staticmethod
    def backward(ctx, grad_output, *fn_out_grads):
        group = ctx.group
        if group is None or group.nranks == 1:
            grad_x = grad_output.clone()
            fn_args_grads = ctx.bwf(*fn_out_grads)
            return (grad_x,) + fn_args_grads  # noqa: RUF005

        # Dual of forward: AllGather the gradient on the comm stream (async)
        # while the shared-expert backward runs on the calc stream, then wait.
        grad_x, task = _all_gather_async(grad_output, group)
        fn_args_grads = ctx.bwf(*fn_out_grads)
        task.wait()
        return (grad_x,) + fn_args_grads  # noqa: RUF005



class AllGatherTokenDispatcher(nn.Layer):
    """
    AllGather + ReduceScatter EP dispatcher (fused-kernel only).

    Every expert is sharded along its ``intermediate`` dim into ``ep_size``
    partitions; every rank therefore holds one shard of *every* expert. The
    forward path is:

        [seq/ep, h] --AllGather(ep)--> [seq, h]
                  --SonicMoE fused _UpProjection / _DownProjection
                    (gather + grouped GEMM + activation + scatter, no
                    explicit permute / unpermute)-->
                  [seq, h] --ReduceScatter(ep, sum)--> [seq/ep, h]

    Routing is computed *before* the AllGather, so each rank only knows its
    local routing map. The dispatcher AllGathers ``hidden_states`` plus the
    compact ``[N, topk]`` routing tensors so that every rank performs the
    same expert compute on the global token list (saving bandwidth versus
    AllGathering the full ``[N, num_experts]`` sparse tensors).

    This dispatcher only reuses SonicMoE's fused expert-compute kernels; it
    does not engage any other SonicMoE component (router, dispatcher, fp8
    protocol).
    """

    def __init__(
        self,
        moe_group: Group,
        expert_model_parallel_size: int,
        num_experts: int,
        fp8_dispatch: bool = False,
        use_ue8m0: bool = False,
    ):
        nn.Layer.__init__(self)
        self.moe_group = moe_group
        self.ep_size = expert_model_parallel_size
        self.num_experts = num_experts
        # In allgather mode every rank holds a shard of every expert.
        self.num_local_experts = num_experts
        # fp8 dispatch quantizes hidden_states to fp8 *before* the AllGather so
        # the inter-rank communication carries fp8 bytes (plus a compact
        # blockwise scale) instead of bf16, mirroring DeepEP. The fp8 activation
        # is consumed directly by SonicMoE's fused ``_UpProjection`` via
        # ``prequant_activation_payload=(x_fp8, scale)`` (no re-quantization).
        # Backward: ``_UpProjection`` emits a bf16 ``dx`` for its activation
        # input — which is exactly ``_PreAllGatherFP8Result``'s fp8 output edge.
        # ``_PreAllGatherFP8Result`` declares grad-dtype inconsistency so it can
        # receive that bf16 grad on its fp8 output and ReduceScatter it back to
        # the local shard.
        self.fp8_dispatch = fp8_dispatch
        self.use_ue8m0 = use_ue8m0
        # Handle for a pre-issued async AllGather of hidden_states (set by
        # ``pre_allgather``, consumed by ``dispatch_preprocess``).
        self._pre_ag_handle: dict | None = None
        # Populated by ``dispatch_preprocess`` — global routing metadata
        # after AllGather across the EP group.
        self._global_topk_indices = None
        self._global_topk_weights = None
        # Populated by ``dispatch_preprocess`` when fp8_dispatch is active —
        # the AllGather'd blockwise scale tensor, reused by downstream fused
        # kernels (``run_sonic_moe``) as ``prequant_activation_payload`` to
        # skip redundant re-quantization, mirroring DeepEP's contract.
        self._fp8_dispatch_scale = None
        # Cache for the overlap-combine path (set by ``token_combine``,
        # consumed by ``combine_postprocess``).
        self._overlap_combined = None

    def pre_allgather(self, hidden_states: paddle.Tensor):
        """Issue an async AllGather for *hidden_states* on the comm stream.

        Called **before** gate computation so that the AllGather runs on the
        dedicated NCCL comm stream while the gate MLP runs on the calc stream.
        The result is stored in ``self._pre_ag_handle`` and consumed by
        :meth:`dispatch_preprocess`.

        Two flavors:
        - bf16 path: a single async AllGather on the comm stream.
        - fp8 path (``fp8_dispatch=True``): quantize locally (calc stream),
          then issue two async AllGathers (fp8 uint8 view + fp32 scale) on
          the comm stream so both overlap with gate compute. Consumed via
          :class:`_PreAllGatherFP8Result`.
        """
        if self.moe_group is None or self.moe_group.nranks == 1:
            self._pre_ag_handle = None
            return

        # Drain any leftover handle from a previous (possibly aborted)
        # forward, so its NCCL task and output buffer are released before
        # we issue a new one. This protects against OOM retries / aborted
        # iterations where ``dispatch_preprocess`` never consumed the
        # previous handle.
        if self._pre_ag_handle is not None:
            try:
                if "task" in self._pre_ag_handle:
                    self._pre_ag_handle["task"].wait()
                else:
                    self._pre_ag_handle["task_x"].wait()
                    self._pre_ag_handle["task_s"].wait()
            except (RuntimeError, OSError) as _e:
                logger.warning(
                    "pre_allgather: leftover async task wait failed (%s), "
                    "discarding handle.",
                    _e,
                )
            self._pre_ag_handle = None

        if len(hidden_states.shape) == 3:
            _, _, d_model = hidden_states.shape
        else:
            _, d_model = hidden_states.shape
        reshaped_input = hidden_states.reshape([-1, d_model]).contiguous()

        if self.fp8_dispatch:
            # FP8 async path: quantize on calc stream, then issue both
            # AllGathers (fp8 data + scale) on the comm stream so they
            # overlap with gate MLP. quantize_activation_blockscaled_fast
            # itself runs on the calc stream and finishes before the
            # comm-stream collectives consume its outputs (NCCL handles
            # the cross-stream sync).
            assert quantize_activation_blockscaled_fast is not None, (
                "Cannot find quantize_activation_blockscaled_fast, "
                "please update sonicmoe."
            )
            x_fp8, scale = quantize_activation_blockscaled_fast(
                reshaped_input, scale_dtype=paddle.int32
            )
            T_local, H = x_fp8.shape
            T_global = T_local * self.moe_group.nranks
            H128 = scale.shape[1]
            x_fp8_global_uint8 = paddle.empty([T_global, H], dtype="uint8")
            task_x = paddle.distributed.stream.all_gather(
                x_fp8_global_uint8,
                x_fp8.view("uint8"),
                group=self.moe_group,
                sync_op=False,
                use_calc_stream=False,
            )
            scale_global = paddle.empty([T_global, H128], dtype=scale.dtype)
            task_s = paddle.distributed.stream.all_gather(
                scale_global,
                scale,
                group=self.moe_group,
                sync_op=False,
                use_calc_stream=False,
            )
            self._pre_ag_handle = {
                "x_fp8_global_uint8": x_fp8_global_uint8,
                "scale_global": scale_global,
                "task_x": task_x,
                "task_s": task_s,
                "group": self.moe_group,
                "fp8": True,
            }
            return

        output_shape = list(reshaped_input.shape)
        output_shape[0] = output_shape[0] * self.moe_group.nranks
        global_hidden_states = paddle.empty(
            shape=output_shape, dtype=reshaped_input.dtype
        )
        task = paddle.distributed.stream.all_gather(
            global_hidden_states,
            reshaped_input,
            group=self.moe_group,
            sync_op=False,
            use_calc_stream=False,
        )

        self._pre_ag_handle = {
            "output": global_hidden_states,
            "task": task,
            "group": self.moe_group,
        }

    def dispatch_preprocess(
        self,
        hidden_states: paddle.Tensor,
        probs: paddle.Tensor,
        mask: paddle.Tensor,  # routing_map
        topk_weights: paddle.Tensor | None = None,
        topk_indices: paddle.Tensor | None = None,
    ) -> paddle.Tensor:
        """AllGather hidden_states and topk metadata across the EP group.

        Reuses ``_pre_ag_handle`` if :meth:`pre_allgather` was called;
        otherwise performs a sync AllGather (FP8 path when ``fp8_dispatch``
        is on, plain bf16 otherwise). The unpermuted global hidden states
        are returned — fused SonicMoE kernels handle gather/scatter inside
        the expert compute, so no explicit permute happens here.

        Side effects: caches ``_global_topk_indices`` and
        ``_global_topk_weights`` for the fused expert compute, and sets
        ``tokens_per_expert = None`` (the consumer recomputes it).

        Raises:
            ValueError: if ``topk_indices`` or ``topk_weights`` is None.
        """
        if len(hidden_states.shape) == 3:
            _, _, d_model = hidden_states.shape
        else:
            _, d_model = hidden_states.shape
        reshaped_input = hidden_states.reshape([-1, d_model]).contiguous()

        # Reset fp8 dispatch state — will be set below only when fp8 path is taken.
        self._fp8_dispatch_scale = None

        # AllGather hidden_states along token dim across the EP group.
        # If pre_allgather() was called before gate, reuse the async result;
        # otherwise fall back to the synchronous path.
        if self._pre_ag_handle is not None:
            if self._pre_ag_handle.get("fp8", False):
                global_hidden_states, self._fp8_dispatch_scale = (
                    _PreAllGatherFP8Result.apply(
                        reshaped_input, self._pre_ag_handle
                    )
                )
            else:
                global_hidden_states = _PreAllGatherResult.apply(
                    reshaped_input, self._pre_ag_handle
                )
            self._pre_ag_handle = None
        elif self.fp8_dispatch:
            # Overlap disabled (moe_allgather_gate_overlap=False): no handle was
            # pre-issued before gate. Issue the fp8 AllGather synchronously here
            # and consume it immediately, so fp8 dispatch still works without the
            # gate-overlap optimization.
            self.pre_allgather(reshaped_input)
            global_hidden_states, self._fp8_dispatch_scale = (
                _PreAllGatherFP8Result.apply(
                    reshaped_input, self._pre_ag_handle
                )
            )
            self._pre_ag_handle = None
        else:
            global_hidden_states = AllGatherGroupOp.apply(
                reshaped_input, self.moe_group
            )

        # Memory-saving fused path: skip permute entirely; the fused SonicMoE
        # kernels (_UpProjection/_DownProjection) handle gather/scatter
        # internally. We only AllGather the topk metadata and return the
        # unpermuted global hidden states.
        if topk_indices is None or topk_weights is None:
            raise ValueError(
                "AllGatherTokenDispatcher requires topk_indices and "
                "topk_weights to be provided."
            )
        # Indices are routing IDs in [0, num_experts) (or -1 for padding), so
        # int32 covers the range with room to spare (num_experts << 2^31) while
        # halving the AllGather payload vs int64. The downstream SonicMoE
        # metadata kernel (deepep_topk_to_sonic_metadata) casts to int32 anyway.
        self._global_topk_indices = AllGatherGroupOp.apply(
            topk_indices.detach().cast("int32"), self.moe_group
        )
        # Padding tokens have topk_indices == -1 (set by TopKRouter).
        # Keep the sentinel in indices so SonicMoE metadata can treat those
        # slots as masked; only zero the corresponding routing weights below.
        self._padding_mask = self._global_topk_indices < 0
        # Router-local AllGather: every token has exactly one origin rank,
        # so the topk_weights gradient flowing back from sonic-moe's
        # _DownProjection (shape [seq_global, K]) must be *scattered* (slice
        # this rank's segment), not reduce-scattered. _RouterAllGather
        # implements scatter on backward, which is the correct semantics
        # for router-owned tensors and lets the router receive main-loss
        # gradients on this rank — matching the alltoall/deepep contract
        # where router weights are also trained by the main loss via the
        # unpermute(probs=...) multiplication.
        self._global_topk_weights = _RouterAllGather.apply(
            topk_weights.cast(probs.dtype), self.moe_group
        )
        # Zero out weights for padding tokens (indices were clipped above).
        # Apply the mask unconditionally: a ``.any()`` guard would force a
        # GPU->CPU sync every step just to decide whether to run the where.
        # ``_padding_mask`` is already a bool tensor (``< 0``), so feed it to
        # ``paddle.where`` directly — the previous bool->float->bool cast was a
        # no-op roundtrip. Numerically identical (all-False mask leaves the
        # weights untouched).
        self._global_topk_weights = paddle.where(
            self._padding_mask,
            paddle.zeros_like(self._global_topk_weights),
            self._global_topk_weights,
        )
        # NOTE: tokens_per_expert is intentionally NOT computed here.
        # The allgather branch of fusion_moe_forward (moe_layer.py) passes
        # `_global_topk_indices` directly to `SonicMoEExpert.forward()` /
        # `run_sonic_moe`, which internally computes the token counts and
        # cumsum offsets needed by the fused kernels. A bincount here would
        # be pure waste. We return None to keep the dispatcher contract
        # `(global_input_tokens, tokens_per_expert)` but the consumer
        # ignores it on this path.
        self.tokens_per_expert = None
        # Return unpermuted global hidden states — fused kernels do the rest
        return global_hidden_states

    def token_dispatch(
        self,
        permuted_global_input_tokens: paddle.Tensor,
        fp8_dispatch: bool = False,
        async_finish: bool = False,
        use_ue8m0: bool = False,
        using_sonic_moe: bool = True,
    ):
        """No-op pass-through for the AllGather path.

        After :meth:`dispatch_preprocess`, every rank already holds the full
        global token list, so no further inter-rank communication is needed.
        ``fp8_dispatch`` / ``async_finish`` / ``use_ue8m0`` are accepted for
        signature compatibility with other dispatchers; the actual FP8
        quantization is handled inside ``dispatch_preprocess``.

        ``using_sonic_moe`` must be True on this path: the AllGather
        dispatcher feeds ``_global_topk_indices`` directly into
        ``SonicMoEExpert.forward`` / ``run_sonic_moe`` (see
        ``dispatch_preprocess``). Any other expert path is unsupported here.

        Returns:
            tuple: ``(global_tokens, fp8_handle)``. When ``_fp8_dispatch_scale``
            was captured during :meth:`dispatch_preprocess` (fp8 path), the handle
            is ``{"scale": scale}`` matching DeepEP's contract so downstream
            fused kernels can reuse the dispatch scale as
            ``prequant_activation_payload``.
        """
        if not using_sonic_moe:
            raise ValueError(
                "AllGatherTokenDispatcher requires using_sonic_moe=True; "
                "the AllGather path is only wired for the fused SonicMoE "
                "expert kernels. Switch dispatcher type or enable SonicMoE."
            )
        fp8_handle = (
            {"scale": self._fp8_dispatch_scale}
            if self._fp8_dispatch_scale is not None
            else None
        )
        return permuted_global_input_tokens, fp8_handle

    def get_dispatched_routing(self):
        """Return (dispatched_indices, dispatched_probs, tokens_per_expert).

        For AllGather, tokens_per_expert is computed on demand via bincount
        since every rank holds all experts and the global token counts are
        already complete. ``paddle.bincount`` defaults to int64; cast to
        int32 to match the dtype contract expected by the downstream
        SonicMoE metadata kernel (``deepep_topk_to_sonic_metadata``).
        """
        indices = self._global_topk_indices
        flat_indices = indices.reshape([-1])
        valid_indices = paddle.masked_select(
            flat_indices,
            flat_indices >= 0,
        )
        tokens_per_expert = paddle.bincount(
            valid_indices,
            minlength=self.num_experts,
        ).cast("int32")
        return (
            indices,
            self._global_topk_weights,
            tokens_per_expert,
        )

    def dispatch_postprocess(
        self,
        global_input_tokens: paddle.Tensor,
    ):
        """Return the cached ``(global_tokens, tokens_per_expert)`` tuple.

        Mirrors the dispatcher protocol; ``tokens_per_expert`` is ``None``
        on this path because the fused SonicMoE kernels recompute it from
        ``_global_topk_indices``.
        """
        return global_input_tokens, self.tokens_per_expert

    def combine_preprocess(self, hidden_states: paddle.Tensor):
        """No-op pass-through (no unpermute on the fused path)."""
        return hidden_states

    def token_combine(
        self,
        hidden_states: paddle.Tensor,
        combine_overlap_handle: dict | None = None,
        async_finish: bool = False,
        fp8_combine_grad_handle: dict | None = None,
    ):
        """ReduceScatter the expert outputs back to the local token shard.

        When ``combine_overlap_handle`` is None this is a no-op; the actual
        ReduceScatter is deferred to :meth:`combine_postprocess`. When it
        is provided, the ReduceScatter is fused with a user-supplied
        subgraph (typically the shared-expert MLP) via
        :class:`_AllGatherCombineAsync` and the combined output is cached
        for ``combine_postprocess`` to return as-is. The handle dict is
        mutated in place: ``fn_out`` receives the subgraph outputs.

        Args:
            combine_overlap_handle: dict with keys ``fn`` (callable) and
                ``fn_args`` (tuple). On return, ``fn_out`` is populated.
            fp8_combine_grad_handle: accepted for caller-signature parity but
                unused on the allgather path. The combine collectives stay
                bf16 in both directions; ``_DownProjection.backward`` quantizes
                ``dout`` itself. Since the backward AllGather is a lossless row
                concat (no reduction), self-quant-after-AllGather is numerically
                identical to quant-before — so this path already matches
                deepep+sonic precision without any extra fp8 collective.
        """
        del fp8_combine_grad_handle

        if combine_overlap_handle is None:
            self._overlap_combined = None
            return hidden_states
        if not isinstance(combine_overlap_handle, dict):
            raise TypeError(
                "combine_overlap_handle must be a dict, got "
                f"{type(combine_overlap_handle).__name__}"
            )
        if (
            "fn" not in combine_overlap_handle
            or "fn_args" not in combine_overlap_handle
        ):
            raise ValueError(
                "combine_overlap_handle must contain 'fn' and 'fn_args' keys"
            )
        if not isinstance(combine_overlap_handle["fn_args"], tuple):
            raise TypeError(
                "combine_overlap_handle['fn_args'] must be a tuple, got "
                f"{type(combine_overlap_handle['fn_args']).__name__}"
            )
        from paddle import framework as _framework

        combined_x, *fn_out = _AllGatherCombineAsync.apply(
            hidden_states,
            self.moe_group,
            *(combine_overlap_handle["fn_args"]),
            fn=combine_overlap_handle["fn"],
            is_first_fwd=not _framework._dygraph_tracer()._has_grad,
        )
        combine_overlap_handle["fn_out"] = tuple(fn_out)
        self._overlap_combined = combined_x
        return combined_x

    def combine_postprocess(self, hidden_states: paddle.Tensor):
        """ReduceScatter the partial-sum expert outputs to the local shard.

        The fused ``_DownProjection`` already scattered back to
        ``[global_T, h]`` with topk weights applied, but the result is a
        partial sum across the EP-sharded intermediate dim. This step sums
        across EP ranks and slices to the per-rank token segment.

        If :meth:`token_combine` already performed the ReduceScatter (the
        overlap path), the cached output is returned and the cache cleared.
        """
        if getattr(self, "_overlap_combined", None) is not None:
            # ReduceScatter was already performed (fused with shared experts) in
            # ``token_combine``; pass the cached output through.
            out = self._overlap_combined
            self._overlap_combined = None
            return out
        return ReduceScatterGroupOp.apply(hidden_states, self.moe_group)
