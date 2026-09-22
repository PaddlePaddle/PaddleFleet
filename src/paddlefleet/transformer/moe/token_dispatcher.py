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

from paddlefleet.transformer.utils import profile

from .fp8_utils import FP8_ALIGN
from .fused_a2a import (
    HYBRIDEP_TOKEN_ALIGNMENT,
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
from .moonep import (
    MoonEPWeightBridge,
    get_moonep_buffer,
    is_moonep_available,
    moonep_combine,
    moonep_dispatch,
    moonep_runtime_weights,
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
        "moonep",
        "ringmoe",
    ):
        raise ValueError(
            "moe_token_dispatcher_type must be one of: "
            "allgather, alltoall, deepep, hybridep, moonep, ringmoe"
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
    buffer is shared at fused_a2a module scope.
    """

    def __init__(
        self,
        group: Group,
        router_topk: int,
        num_experts: int | None = None,
        num_local_experts: int | None = None,
        moe_ep_barrier: bool = True,
        hybridep_buffer_configs: dict | None = None,
        moe_deep_gemm: bool = False,
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
        self.num_permuted_tokens = None
        self.handle = None
        self._active_buffer = None
        self.hybridep_buffer_configs = hybridep_buffer_configs or {}
        self._moe_deep_gemm = moe_deep_gemm
        self._reset_dispatch_state()
        self._num_unpadded_tokens = None

    def _reset_dispatch_state(self):
        self._dispatch_uses_fp8 = None
        self._dispatch_pad_multiple = None

    def _set_dispatch_state(self, use_fp8: bool):
        self._dispatch_uses_fp8 = use_fp8
        self._dispatch_pad_multiple = (
            FP8_ALIGN if use_fp8 or self._moe_deep_gemm else None
        )

    def _get_max_num_tokens_per_rank(self, num_local_tokens: int, place) -> int:
        max_num_tokens = num_local_tokens
        if self.group.nranks > 1:
            max_num_tokens_tensor = paddle.to_tensor(
                [num_local_tokens], dtype="int64", place=place
            )
            paddle.distributed.all_reduce(
                max_num_tokens_tensor,
                op=paddle.distributed.ReduceOp.MAX,
                group=self.group,
            )
            max_num_tokens = int(max_num_tokens_tensor.item())
        return (
            (max_num_tokens + HYBRIDEP_TOKEN_ALIGNMENT - 1)
            // HYBRIDEP_TOKEN_ALIGNMENT
            * HYBRIDEP_TOKEN_ALIGNMENT
        )

    def _pad_tokens_to_rank_max(
        self, tensor: paddle.Tensor | None, max_num_tokens: int
    ) -> paddle.Tensor | None:
        if tensor is None or tensor.shape[0] == max_num_tokens:
            return tensor
        assert tensor.shape[0] < max_num_tokens, (
            f"HybridEP token padding expects local tokens <= EP max, got "
            f"{tensor.shape[0]} > {max_num_tokens}."
        )
        pad_shape = [max_num_tokens - tensor.shape[0], *tensor.shape[1:]]
        padding = paddle.zeros(pad_shape, dtype=tensor.dtype)
        return paddle.concat([tensor, padding], axis=0)

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

    def _set_num_permuted_tokens(self, tokens_per_expert: paddle.Tensor) -> int:
        self.num_permuted_tokens = int(
            paddle.sum(tokens_per_expert.astype("int64")).item()
        )
        return self.num_permuted_tokens

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
        num_unpadded_tokens = hidden_states.shape[0]
        max_num_tokens = self._get_max_num_tokens_per_rank(
            num_unpadded_tokens, hidden_states.place
        )
        self._num_unpadded_tokens = num_unpadded_tokens
        routing_map, probs = self._get_dispatch_metadata(
            token_indices, token_weights
        )
        hidden_states = self._pad_tokens_to_rank_max(
            hidden_states, max_num_tokens
        )
        routing_map = self._pad_tokens_to_rank_max(routing_map, max_num_tokens)
        probs = self._pad_tokens_to_rank_max(probs, max_num_tokens)
        buffer = self._get_buffer(hidden_states, max_num_tokens)
        num_permuted_tokens = self._get_num_permuted_tokens_upper_bound(
            max_num_tokens
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
        self._set_dispatch_state(use_fp8)
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
            pad_multiple=self._dispatch_pad_multiple,
            num_permuted_tokens=num_permuted_tokens,
            non_blocking=True,
        )
        self.padded_tokens_per_expert = tokens_per_expert
        num_permuted_tokens = self._set_num_permuted_tokens(tokens_per_expert)
        hidden_states = hidden_states[:num_permuted_tokens]
        if dispatched_probs is not None:
            dispatched_probs = dispatched_probs[:num_permuted_tokens]
        if scale is not None:
            scale = scale[:num_permuted_tokens]
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
        hidden_states = hybrid_ep_combine(
            hidden_states, self, self.num_permuted_tokens
        )
        self.dispatched_probs = None
        self.handle = None
        self.num_permuted_tokens = None
        self._reset_dispatch_state()
        if (
            self._num_unpadded_tokens is not None
            and hidden_states.shape[0] != self._num_unpadded_tokens
        ):
            hidden_states = hidden_states[: self._num_unpadded_tokens]
        self._num_unpadded_tokens = None
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


class _MoonEPManager(_DispatchManager):
    """MoonEP manager using fixed-capacity E+B expert groups."""

    def __init__(
        self,
        group: Group,
        router_topk: int,
        num_experts: int,
        num_local_experts: int,
        moe_ep_barrier: bool = True,
    ):
        del moe_ep_barrier
        if not is_moonep_available():
            raise ImportError(
                "moe_token_dispatcher_type=moonep but MoonEP is unavailable."
            )
        self.group = group
        self.router_topk = router_topk
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.token_indices = None
        self.token_probs = None
        self.tokens_per_expert = None
        self.dispatched_indices = None
        self.dispatched_probs = None
        self.handle = None
        self._buffer = None
        self._buffer_signature = None
        self._bridge = None
        self._num_dispatched_tokens = None

    def bind_experts(self, grouped_experts) -> None:
        self._bridge = MoonEPWeightBridge(
            group=self.group,
            num_experts=self.num_experts,
            num_local_experts=self.num_local_experts,
            weight1_shape=grouped_experts.weight1.shape,
            weight2_shape=grouped_experts.weight2.shape,
        )

    def setup_metadata(
        self,
        routing_map: paddle.Tensor,
        probs: paddle.Tensor,
        topk_weights: paddle.Tensor | None = None,
        topk_indices: paddle.Tensor | None = None,
    ):
        num_tokens = routing_map.shape[0]
        routing_map = routing_map.reshape(
            [num_tokens, self.num_experts]
        ).astype("bool")
        probs = probs.reshape([num_tokens, self.num_experts])
        if not _try_setup_router_topk_metadata(
            self, num_tokens, topk_weights, topk_indices
        ):
            self.token_probs, self.token_indices = paddle.topk(
                probs, self.router_topk, axis=-1
            )
        self.token_probs = self.token_probs.astype("float32")
        self.token_indices = self.token_indices.astype("int32")
        padding_mask = self.token_indices < 0
        self.token_probs = paddle.where(
            padding_mask,
            paddle.zeros_like(self.token_probs),
            self.token_probs,
        )
        self.token_indices = paddle.where(
            padding_mask,
            paddle.zeros_like(self.token_indices),
            self.token_indices,
        )
        self.token_indices.stop_gradient = True
        self.tokens_per_expert = _tokens_per_expert_histogram(
            self.token_indices, self.num_experts
        )

    def _ensure_buffer(self, hidden_states: paddle.Tensor) -> None:
        if self._bridge is None:
            raise RuntimeError(
                "MoonEP grouped experts must be bound before dispatch."
            )
        signature = (
            int(hidden_states.shape[0]),
            int(hidden_states.shape[1]),
            str(hidden_states.dtype),
            int(self.router_topk),
            int(self.num_experts),
            int(self.num_local_experts),
        )
        if self._buffer is not None:
            if hidden_states.dtype != paddle.bfloat16:
                raise ValueError(
                    "MoonEP dispatch requires BF16 hidden states, "
                    f"got {hidden_states.dtype}."
                )
            if signature != self._buffer_signature:
                raise ValueError(
                    "MoonEP requires a fixed dispatch signature while a layer "
                    f"buffer is live: expected {self._buffer_signature}, "
                    f"got {signature}."
                )
            return
        world_size = paddle.distributed.get_world_size(self.group)
        rank_metadata = [None] * world_size
        paddle.distributed.all_gather_object(
            rank_metadata,
            signature,
            group=self.group,
        )
        if any(metadata != rank_metadata[0] for metadata in rank_metadata):
            raise ValueError(
                "MoonEP requires an identical dispatch signature across its "
                f"ranks, got {rank_metadata}."
            )
        if hidden_states.dtype != paddle.bfloat16:
            raise ValueError(
                "MoonEP dispatch requires BF16 hidden states, "
                f"got {hidden_states.dtype}."
            )
        self._buffer_signature = signature
        self._buffer = get_moonep_buffer(
            S=signature[0],
            H=hidden_states.shape[1],
            K=self.router_topk,
            E=self.num_experts,
            B=self.num_local_experts,
            num_ep_ranks=world_size,
            group=self.group,
        )
        self._bridge.attach_buffer(self._buffer)

    def dispatch_overlap(
        self,
        hidden_states: paddle.Tensor,
        token_indices: paddle.Tensor,
        token_weights: paddle.Tensor,
        fp8_dispatch: bool = False,
        async_finish: bool = False,
        use_ue8m0: bool = False,
    ):
        del (
            hidden_states,
            token_indices,
            token_weights,
            fp8_dispatch,
            async_finish,
            use_ue8m0,
        )
        raise NotImplementedError("MoonEP does not support dispatch overlap.")

    def dispatch(
        self,
        hidden_states: paddle.Tensor,
        fp8_dispatch: bool = False,
        async_finish: bool = False,
        use_ue8m0: bool = False,
        using_sonic_moe: bool = False,
    ):
        if fp8_dispatch or use_ue8m0 or using_sonic_moe:
            raise NotImplementedError(
                "MoonEP currently supports BF16 dispatch."
            )
        del async_finish
        self._ensure_buffer(hidden_states)
        state = {}
        (
            dispatched_hidden,
            self.dispatched_probs,
            self.tokens_per_expert,
        ) = moonep_dispatch(
            hidden_states,
            self.token_probs,
            self.token_indices,
            self.tokens_per_expert,
            self._buffer,
            state,
        )
        self.handle = state["plan"]
        return dispatched_hidden, None

    def combine(
        self,
        hidden_states: paddle.Tensor,
        combine_overlap_handle: dict | None = None,
        async_finish: bool = False,
        use_rr_deepep_combine: bool = False,
        fp8_dispatch: bool = False,
        combine_grad_handle: dict | None = None,
    ):
        if combine_overlap_handle is not None:
            raise NotImplementedError(
                "MoonEP does not support shared-expert combine overlap."
            )
        del (
            async_finish,
            use_rr_deepep_combine,
            fp8_dispatch,
            combine_grad_handle,
        )
        hidden_states = moonep_combine(
            hidden_states, self._buffer, self.handle, self._bridge
        )
        self.handle = None
        self.token_indices = None
        self.token_probs = None
        self.tokens_per_expert = None
        self.dispatched_probs = None
        self._num_dispatched_tokens = None
        return hidden_states

    def runtime_expert_weights(self, grouped_experts):
        if self.handle is None:
            raise RuntimeError("MoonEP runtime weights require an active plan.")
        return moonep_runtime_weights(
            grouped_experts, self._bridge, self.handle
        )

    def get_dispatched_metadata(self):
        raise NotImplementedError(
            "MoonEP exposes E+B group counts instead of DeepEP routing metadata."
        )

    def get_number_of_tokens_per_expert(self):
        return self.tokens_per_expert

    def get_permuted_hidden_states_by_experts(self, hidden_states):
        self._num_dispatched_tokens = int(hidden_states.shape[0])
        num_valid_tokens = int(self.tokens_per_expert.sum().item())
        return hidden_states[:num_valid_tokens]

    def get_restored_hidden_states_by_experts(self, hidden_states):
        if self.dispatched_probs is not None:
            hidden_states = hidden_states * self.dispatched_probs[
                : hidden_states.shape[0]
            ].astype(hidden_states.dtype).unsqueeze(-1)
        if hidden_states.shape[0] != self._num_dispatched_tokens:
            hidden_states = paddle.concat(
                [
                    hidden_states,
                    paddle.zeros(
                        [
                            self._num_dispatched_tokens
                            - hidden_states.shape[0],
                            hidden_states.shape[1],
                        ],
                        dtype=hidden_states.dtype,
                    ),
                ],
                axis=0,
            )
        return hidden_states


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
        use_accuracy_compatible: bool = False,
    ):
        self.group = group
        self.router_topk = router_topk
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.moe_ep_barrier = moe_ep_barrier
        self.use_accuracy_compatible = use_accuracy_compatible

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
            use_accuracy_compatible=self.use_accuracy_compatible,
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
            use_accuracy_compatible=self.use_accuracy_compatible,
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
        moe_deep_gemm: bool = False,
        use_accuracy_compatible: bool = False,
    ):
        super().__init__(ep_group)

        self.use_accuracy_compatible = use_accuracy_compatible
        self.num_local_experts = num_local_experts
        assert self.ep_size > 1, "Flex token dispatcher requires EP > 1"
        if dispatcher_type == "moonep":
            manager_cls = _MoonEPManager
        elif is_hybrid_ep_backend_selected(dispatcher_type):
            manager_cls = _HybridEPManager
        else:
            manager_cls = _DeepEPManager
        manager_kwargs = {
            "group": self.ep_group,
            "router_topk": num_experts_per_tok,
            "num_experts": n_routed_experts,
            "num_local_experts": self.num_local_experts,
            "moe_ep_barrier": moe_ep_barrier,
        }
        if manager_cls is _HybridEPManager:
            manager_kwargs["hybridep_buffer_configs"] = hybridep_buffer_configs
            manager_kwargs["moe_deep_gemm"] = moe_deep_gemm
        elif manager_cls is _DeepEPManager:
            manager_kwargs["use_accuracy_compatible"] = use_accuracy_compatible
        self._comm_manager = manager_cls(**manager_kwargs)

    def bind_experts(self, grouped_experts) -> None:
        if isinstance(self._comm_manager, _MoonEPManager):
            self._comm_manager.bind_experts(grouped_experts)

    def runtime_expert_weights(self, grouped_experts):
        if not isinstance(self._comm_manager, _MoonEPManager):
            raise RuntimeError(
                "runtime_expert_weights is only provided by the MoonEP manager."
            )
        return self._comm_manager.runtime_expert_weights(grouped_experts)

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

    def token_combine(self, hidden_states: paddle.Tensor, async_finish=False):
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
        use_accuracy_compatible: bool = False,
    ):
        nn.Layer.__init__(self)
        self.moe_group = moe_group
        self.expert_model_parallel_size = expert_model_parallel_size
        self.num_experts_per_device = num_experts_per_device
        self.local_expert_indices = local_expert_indices
        self.num_local_experts = len(local_expert_indices)
        self.use_accuracy_compatible = use_accuracy_compatible

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
        self.hidden_states_dtype = hidden_states.dtype
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
        ) = permute(
            reshaped_input,
            self.routing_map,
            use_accuracy_compatible=self.use_accuracy_compatible,
        )
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

    def get_dispatched_routing(self):
        """Return (dispatched_indices, dispatched_probs, tokens_per_expert).

        AllToAll uses tokens_per_expert-based expert processing
        (expert_forward), so dispatched_indices and dispatched_probs are None.
        The corresponding branch in ``fusion_moe_forward`` selects
        ``expert_forward`` instead of index-based fusion kernels.
        """
        return (None, None, self.tokens_per_expert)

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
            use_accuracy_compatible=self.use_accuracy_compatible,
        )
        output_dtype = getattr(self, "hidden_states_dtype", output.dtype)
        return output.cast(output_dtype)


class _RouterAllGather(paddle.autograd.PyLayer):
    """AllGather for router topk weights, shared by the allgather and ringmoe
    dispatchers (the ring calls it per round via ``_ag_router``).

    Forward:  [T_local, K] --AllGather(EP)--> [T_global, K]  (identical on all ranks)
    Backward: [T_global, K] --ReduceScatter(EP, SUM)--> [T_local, K]

    Every EP rank holds an intermediate-dim shard of every expert and computes a
    partial expert output for ALL global tokens using the same all-gathered
    router weights.  Each rank therefore produces its own partial gradient for
    the shared weight tensor; the backward must sum these partials (reduce) and
    return each rank its own token segment (scatter).  A plain scatter would keep
    only the origin rank's partial and discard the rest, under-training the
    router.
    """

    @staticmethod
    def forward(ctx, input, group):
        ctx.group = group
        ctx.input_shape = list(input.shape)
        # The router-weight gather stays on the CALC stream, not the comm
        # stream: the token/index gathers already occupy the comm stream, so
        # queuing this small gather behind them would serialize it and add
        # dispatch latency. On calc it overlaps the in-flight comm-stream
        # gathers instead. Backward (reduce-scatter) is unaffected.
        return _calc_stream_all_gather(input, group)

    @staticmethod
    def backward(ctx, grad):
        group = ctx.group
        local_shape = ctx.input_shape
        if group is None or group.nranks == 1:
            if list(grad.shape) != local_shape:
                grad = grad.reshape(local_shape)
            return grad
        global_shape = [local_shape[0] * group.nranks, *local_shape[1:]]
        if list(grad.shape) != global_shape:
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
        out = reduce_scatter_group(grad.contiguous(), group=group)
        if list(out.shape) != local_shape:
            out = out.reshape(local_shape)
        return out


def _drain_async_handle(handle: dict | None, where: str) -> None:
    """Wait out a leftover pre-issued collective and discard its handle.

    A prefetch whose consumer was never reached -- an exception between issuing
    and consuming -- leaves an in-flight NCCL task owning an output buffer.
    Waiting before the slot is overwritten keeps the next prefetch from racing
    it. A failed wait is logged rather than raised: there is nothing left to
    recover, and raising here would mask whatever aborted the previous forward.

    Always returns None, so the caller can assign the result back over the slot
    it just drained.
    """
    if handle is None:
        return None
    try:
        handle["task"].wait()
    except (RuntimeError, OSError) as exc:
        logger.warning(
            "%s: leftover async task wait failed (%s), discarding handle.",
            where,
            exc,
        )
    return None


class _PreAllGatherResult(paddle.autograd.PyLayer):
    """Consume a pre-issued async AllGather of hidden_states.

    Forward waits for the async NCCL task and returns the filled buffer.
    Backward is ReduceScatter (dual of AllGather).
    The ``handle`` is a plain dict, not a tensor, so Paddle's PyLayer does not
    expect a gradient for it — backward returns only one value.
    """

    @staticmethod
    def forward(ctx, hidden_states, handle):
        handle["task"].wait()
        ctx.group = handle["group"]
        return handle["output"]

    @staticmethod
    def backward(ctx, grad):
        grad_input = ReduceScatterGroupOp.apply(grad, ctx.group)
        return grad_input


class _PreAllGatherFP8Result(paddle.autograd.PyLayer):
    """FP8 variant of _PreAllGatherResult.

    Forward waits for the single fused async AllGather task (fp8 data ++ block
    scale packed per token) and returns ``(x_fp8_global, scale_global)`` after
    splitting.  The fp8 tensor is consumed directly by SonicMoE's _UpProjection
    via prequant_activation_payload.

    _UpProjection.backward produces a bf16 dx for the activation input, but
    the forward output here is fp8.  set_grad_in_dtype_consistent(False) and
    set_materialize_grads(False) tell Paddle to pass the bf16 grad through
    without dtype coercion.  Backward then ReduceScatters that bf16 grad back
    to the local token shard.
    """

    @staticmethod
    def forward(ctx, hidden_states, handle):
        handle["task"].wait()
        ctx.group = handle["group"]
        x_fp8_global, scale_global = _split_fused_fp8_gather(
            handle["fused_global"],
            handle["H"],
            handle["H128"],
            handle["scale_dtype"],
        )
        ctx.set_grad_in_dtype_consistent(False)
        ctx.set_materialize_grads(False)
        return x_fp8_global, scale_global

    @staticmethod
    def backward(ctx, grad_output, grad_scale=None):
        del grad_scale
        group = ctx.group
        if grad_output is None:
            return None
        if group is None or group.nranks == 1:
            return grad_output
        grad_input = ReduceScatterGroupOp.apply(grad_output, group)
        return grad_input


def _reduce_scatter_async(input, group):
    """Async ReduceScatter (SUM, axis 0) on the comm stream.
    Returns (output, task)."""
    input = input.contiguous()
    out_shape = list(input.shape)
    if out_shape[0] % group.nranks != 0:
        raise ValueError(
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
    """Async AllGather (axis 0) on the comm stream.
    Returns (output, task)."""
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


def _quantize_and_pack_fp8(x):
    """Quantize ``x`` to fp8 e4m3 + int32 1x128 block scale and pack into a
    single uint8 row ``[T_local, H + 4*num_scale_blocks]``
    (fp8 data ++ scale bytes).

    Returns ``(fused_local, H, num_scale_blocks, scale_dtype)``. The caller
    is responsible for AllGather (sync or async) and for unpacking via
    :func:`_split_fused_fp8_gather`. Bit-identical to two separate gather of
    data and scale: AllGather is a lossless row concat and the 1x128 scale
    lives within a single token's hidden vector, so packing along axis 1
    before the gather yields the same per-rank bytes as gathering separately.
    """
    if quantize_activation_blockscaled_fast is None:
        raise RuntimeError(
            "Cannot find quantize_activation_blockscaled_fast, "
            "please update sonicmoe."
        )
    x = x.contiguous()
    x_fp8, scale = quantize_activation_blockscaled_fast(
        x, scale_dtype=paddle.int32
    )
    _, H = x_fp8.shape
    num_scale_blocks = scale.shape[1]
    scale_dtype = scale.dtype
    fused_local = paddle.concat(
        [x_fp8.view("uint8"), scale.view("uint8")], axis=1
    ).contiguous()
    return fused_local, H, num_scale_blocks, scale_dtype


def _fused_fp8_all_gather_async(x, group):
    """Quantize local tensor to fp8 e4m3 + int32 1x128 block scale, then a SINGLE
    async AllGather of the fused (fp8-data-as-uint8 ++ scale-as-uint8) per-token
    byte row on the comm stream.  Returns
    ``(fused_global, H, num_scale_blocks, scale_dtype, task)``.

    Fusing data and scale into one AllGather (vs two back-to-back collectives)
    is bit-identical: AllGather concatenates rows along axis 0 and every rank
    carries an identical ``T_local``, so packing the two payloads along axis 1
    before the gather yields the same per-rank bytes as gathering them
    separately.  It removes one NCCL kernel launch (and its peer-wait fixed
    cost) per collective.

    Because AllGather is a lossless row concat and the 1x128 scale lives within
    a single token's hidden vector, this is bit-identical to
    bf16-AllGather-then-quantize (halving on-wire bytes vs bf16).  The caller
    waits ``task`` and then calls :func:`_split_fused_fp8_gather` to recover
    ``(data_e4m3_global, scale_global)``.

    Used both by the combine backward (``_AllGatherCombineAsync.backward``) and
    by ``AllGatherTokenDispatcher.pre_allgather`` for the forward activation.
    """
    fused_local, H, num_scale_blocks, scale_dtype = _quantize_and_pack_fp8(x)
    T_global = fused_local.shape[0] * group.nranks
    fused_global = paddle.empty([T_global, fused_local.shape[1]], dtype="uint8")
    task = paddle.distributed.stream.all_gather(
        fused_global,
        fused_local,
        group=group,
        sync_op=False,
        use_calc_stream=False,
    )
    return fused_global, H, num_scale_blocks, scale_dtype, task


def _split_fused_fp8_gather(fused_global, H, num_scale_blocks, scale_dtype):
    """Recover ``(data_e4m3_global, scale_global)`` from a fused gather buffer.

    Inverse of the packing in :func:`_fused_fp8_all_gather_async` /
    :meth:`AllGatherTokenDispatcher.pre_allgather`. Must be called only after the
    gather task has completed.
    """
    T_global = fused_global.shape[0]
    data_global_u8 = fused_global[:, :H].contiguous()
    scale_global = fused_global[:, H:].contiguous().view(scale_dtype)
    return (
        data_global_u8.view("float8_e4m3fn"),
        scale_global.reshape([T_global, num_scale_blocks]),
    )


def _fp8_split_all_gather(x, group):
    """fp8 token gather as TWO separate plain AllGathers (data + scale).

    Each payload lands in its OWN contiguous output, so there is no strided
    unpack: the fused-pack path's :func:`_split_fused_fp8_gather` did a full
    ``[T_global, H]`` ``.contiguous()`` per gather (measured the single biggest
    copy on the ring's critical path). Here the gathered data buffer is already
    contiguous ``[T_global, H]`` -- ``.view("float8_e4m3fn")`` feeds the fp8 GEMM
    directly with zero copy.

    Uses the ordinary ``all_gather_group``: the gathered fp8 data is saved for
    the expert GEMM backward, i.e. retained until this microbatch's backward
    (pipeline depth), and a regular allocator tensor retains fine.

    Bit-identical to the fused path: AllGather is a lossless row concat and each
    1x128 scale block lives within a single token's row, so nothing spans ranks.
    Backward stays with the caller (straight-through ReduceScatter of bf16 grad).
    """
    if quantize_activation_blockscaled_fast is None:
        raise RuntimeError(
            "Cannot find quantize_activation_blockscaled_fast, "
            "please update sonicmoe."
        )
    x = x.contiguous()
    x_fp8, scale = quantize_activation_blockscaled_fast(
        x, scale_dtype=paddle.int32
    )
    num_scale_blocks = scale.shape[1]
    scale_dtype = scale.dtype
    if group is None or group.nranks == 1:
        return x_fp8, scale
    data_global = all_gather_group(x_fp8.view("uint8"), group=group)
    scale_global = all_gather_group(scale.view("uint8"), group=group)
    T_global = data_global.shape[0]
    return (
        data_global.view("float8_e4m3fn"),
        scale_global.view(scale_dtype).reshape([T_global, num_scale_blocks]),
    )


class _AllGatherCombineAsync(paddle.autograd.PyLayer):
    """Fuse the combine ReduceScatter with a shared-expert subgraph for overlap.

    Forward:
      1. Issue async ReduceScatter of expert output on comm stream.
      2. Run shared-expert subgraph on calc stream (concurrent with step 1).
      3. Wait for ReduceScatter, return (combined_x,) + fn_out.

    Backward (dual):
      - bf16 path: async AllGather of grad on comm stream while shared-expert
        backward runs on calc stream.
      - fp8 path (fp8_combine_grad_handle != None): quantize local grad to fp8,
        async AllGather both data and scale on comm stream, write results into
        the handle for _DownProjection.backward to consume directly.

    Overlap is safe because fn's inputs are independent of x (expert output).
    """

    @staticmethod
    def forward(
        ctx,
        x,
        group,
        *fn_args,
        fn,
        is_first_fwd=False,
        fp8_combine_grad_handle=None,
    ):
        if fn is None:
            raise ValueError(
                "_AllGatherCombineAsync requires a non-None fn for overlap."
            )
        ctx.group = group
        ctx.fp8_combine_grad_handle = fp8_combine_grad_handle
        if fp8_combine_grad_handle is not None:
            ctx.set_grad_in_dtype_consistent(False)
            ctx.set_materialize_grads(False)

        if group is None or group.nranks == 1:
            combined_x = x.clone()
            ctx.bwf, fn_out = manual_backward(fn, is_first_fwd, *fn_args)
            return (combined_x,) + fn_out  # noqa: RUF005

        combined_x, task = _reduce_scatter_async(x, group)
        ctx.bwf, fn_out = manual_backward(fn, is_first_fwd, *fn_args)
        task.wait()

        return (combined_x,) + fn_out  # noqa: RUF005

    @staticmethod
    def backward(ctx, grad_output, *fn_out_grads):
        group = ctx.group
        handle = ctx.fp8_combine_grad_handle
        if group is None or group.nranks == 1:
            grad_x = grad_output.clone()
            fn_args_grads = ctx.bwf(*fn_out_grads)
            return (grad_x,) + fn_args_grads  # noqa: RUF005

        if handle is not None:
            fused_global, _H, _num_scale_blocks, _sdt, task = (
                _fused_fp8_all_gather_async(grad_output, group)
            )
            fn_args_grads = ctx.bwf(*fn_out_grads)
            task.wait()
            data_e4m3, scale_global = _split_fused_fp8_gather(
                fused_global, _H, _num_scale_blocks, _sdt
            )
            handle["data"] = data_e4m3
            handle["scale"] = scale_global
            return (data_e4m3,) + fn_args_grads  # noqa: RUF005

        grad_x, task = _all_gather_async(grad_output, group)
        fn_args_grads = ctx.bwf(*fn_out_grads)
        task.wait()
        return (grad_x,) + fn_args_grads  # noqa: RUF005


class _AllGatherCombineNoOverlap(paddle.autograd.PyLayer):
    """ReduceScatter combine without overlap subgraph (sync on calc stream).

    Mirrors ``_AllGatherCombineAsync``'s fp8/bf16 grad collection but skips the
    shared-expert overlap. All collectives use the same sync calc-stream
    wrappers (``reduce_scatter_group`` / ``all_gather_group``) as
    :class:`ReduceScatterGroupOp` and the ``_PreAllGather*Result`` backward
    paths, so they FIFO-order naturally after the expert MLP forward and
    ``_DownProjection.backward`` without cross-stream event synchronization.
    Needed because the no-overlap path must still populate
    ``fp8_combine_grad_handle`` in backward for ``_DownProjection.backward``
    to consume; previously the no-overlap branch returned ``hidden_states``
    directly and dropped the handle.
    """

    @staticmethod
    def forward(ctx, x, group, fp8_combine_grad_handle=None):
        ctx.group = group
        ctx.fp8_combine_grad_handle = fp8_combine_grad_handle
        if fp8_combine_grad_handle is not None:
            ctx.set_grad_in_dtype_consistent(False)
            ctx.set_materialize_grads(False)
        if group is None:
            return x.clone()
        return reduce_scatter_group(x, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group
        handle = ctx.fp8_combine_grad_handle
        if group is None:
            return grad_output.clone()
        if handle is not None:
            fused_local, H, num_scale_blocks, scale_dtype = (
                _quantize_and_pack_fp8(grad_output)
            )
            fused_global = all_gather_group(fused_local, group=group)
            data_e4m3, scale_global = _split_fused_fp8_gather(
                fused_global, H, num_scale_blocks, scale_dtype
            )
            handle["data"] = data_e4m3
            handle["scale"] = scale_global
            return data_e4m3
        return all_gather_group(grad_output, group=group)


@paddle.no_grad()
def _tokens_per_expert_histogram(indices, num_experts):
    """Count tokens per expert WITHOUT any GPU->CPU synchronization.

    ``indices`` is the [..., K] routing tensor that may contain ``-1`` for
    padding tokens.  The result is an int32 ``[num_experts]`` histogram.

    This replaces ``masked_select`` + ``bincount``: ``masked_select`` produces a
    *variable-length* output (its size is data-dependent), which forces Paddle
    to copy the element count back to the CPU, stalling the host.  ``bincount``
    likewise must resolve ``max(input)+1`` on the CPU to size its output.  Both
    serialize the pipeline.

    Instead we build a fixed-size one-hot histogram: padding (-1) is sent to a
    sink column ``== num_experts`` so it is dropped, and every other value is in
    ``[0, num_experts)``.  All shapes are known a priori, so no host sync occurs.
    Numerically identical to ``bincount(masked_select(indices, indices>=0),
    minlength=num_experts)``.
    """
    flat = indices.reshape([-1]).cast("int64")
    sink = paddle.full_like(flat, num_experts)
    clamped = paddle.where(flat >= 0, flat, sink)
    # scatter(overwrite=False) accumulates updates[i] into counts[clamped[i]]
    # via atomic add — fixed-shape [num_experts + 1] output, no host sync,
    # and O(T*K + E) temp memory instead of O(T*K*E) from a full one-hot.
    counts = paddle.zeros([num_experts + 1], dtype="int32")
    ones = paddle.ones([flat.shape[0]], dtype="int32")
    counts = paddle.scatter(counts, clamped, ones, overwrite=False)
    return counts[:num_experts]


class AllGatherTokenDispatcher(nn.Layer):
    """AllGather + ReduceScatter EP dispatcher (SonicMoE fused-kernel only).

    Every expert is sharded along its intermediate dim into EP partitions;
    every rank holds all experts but only its I/EP shard of each.  The forward
    data flow is:

        [T_local, H] --AllGather--> [T_global, H] --SonicMoE fused
        _UpProjection / _DownProjection (all tokens, partial I)-->
        [T_global, H] --ReduceScatter(SUM)--> [T_local, H]

    Routing metadata (topk_indices, topk_weights) is also AllGathered so every
    rank sees the same global token assignments.  No explicit permute/unpermute
    is needed — the SonicMoE fused kernels handle token gather/scatter
    internally based on the indices.
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
        self.num_local_experts = num_experts  # every rank holds all experts
        self.fp8_dispatch = fp8_dispatch
        self.use_ue8m0 = use_ue8m0
        self._pre_ag_handle: dict | None = None
        self._global_topk_indices = None
        self._global_topk_weights = None
        self._fp8_dispatch_scale = None
        self._overlap_combined = None

    def pre_allgather(self, hidden_states: paddle.Tensor):
        """Issue an async AllGather of hidden_states on the comm stream.

        Called before gate computation so the AllGather overlaps with the gate
        MLP on the calc stream.  Result is stored in self._pre_ag_handle and
        consumed by dispatch_preprocess.

        bf16 path: single async AllGather.
        fp8 path: quantize + a single fused async AllGather of (data ++ scale)
            via ``_fused_fp8_all_gather_async``.
        """
        if self.moe_group is None or self.moe_group.nranks == 1:
            self._pre_ag_handle = None
            return

        # Drain leftover handle from a possibly-aborted previous forward.
        self._pre_ag_handle = _drain_async_handle(
            self._pre_ag_handle, "pre_allgather"
        )

        if len(hidden_states.shape) == 3:
            _, _, d_model = hidden_states.shape
        else:
            _, d_model = hidden_states.shape
        reshaped_input = hidden_states.reshape([-1, d_model]).contiguous()

        if self.fp8_dispatch:
            (
                fused_global,
                H,
                num_scale_blocks,
                scale_dtype,
                task,
            ) = _fused_fp8_all_gather_async(reshaped_input, self.moe_group)
            self._pre_ag_handle = {
                "fused_global": fused_global,
                "H": H,
                "H128": num_scale_blocks,
                "scale_dtype": scale_dtype,
                "task": task,
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
        """AllGather hidden_states and routing metadata across the EP group.

        Steps:
        1. Reshape to [T_local, H].
        2. AllGather hidden_states (reuse pre-issued async handle if available).
        3. AllGather topk_indices async on comm stream (int32, no gradient).
        4. AllGather topk_weights via _RouterAllGather on calc stream (has
           gradient, backward = reduce-scatter).
        5. Wait for indices, build padding mask (indices < 0), zero padding weights.
        6. Return global hidden_states (unpermuted — SonicMoE handles gather).

        Caches _global_topk_indices, _global_topk_weights for downstream use.

        Note:
            If ``_pre_ag_handle`` is None on entry (gate-overlap did not fire,
            e.g. ``moe_allgather_gate_overlap=False`` or a direct call that
            bypasses ``_maybe_pre_allgather_overlap``) and ``fp8_dispatch`` is
            True, this method issues the fp8 AllGather inline via
            ``pre_allgather`` and immediately waits on it.  This is correct but
            forfeits the gate-compute overlap that the pre-issued path provides.
        """
        if len(hidden_states.shape) == 3:
            _, _, d_model = hidden_states.shape
        else:
            _, d_model = hidden_states.shape
        reshaped_input = hidden_states.reshape([-1, d_model]).contiguous()

        self._fp8_dispatch_scale = None

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

        if topk_indices is None or topk_weights is None:
            raise ValueError(
                "AllGatherTokenDispatcher requires topk_indices and "
                "topk_weights to be provided."
            )
        # AllGather indices as int32 on comm stream (async, no gradient).
        # Issued before weights AllGather so both collectives are in flight.
        topk_indices_i32 = topk_indices.detach().cast("int32").contiguous()
        if self.moe_group is None or self.moe_group.nranks == 1:
            self._global_topk_indices = topk_indices_i32.clone()
            _idx_task = None
        else:
            _idx_out_shape = list(topk_indices_i32.shape)
            _idx_out_shape[0] *= self.moe_group.nranks
            self._global_topk_indices = paddle.empty(
                shape=_idx_out_shape, dtype=topk_indices_i32.dtype
            )
            _idx_task = paddle.distributed.stream.all_gather(
                self._global_topk_indices,
                topk_indices_i32,
                group=self.moe_group,
                sync_op=False,
                use_calc_stream=False,
            )
        # AllGather router weights on calc stream (has gradient).
        self._global_topk_weights = _RouterAllGather.apply(
            topk_weights.cast(probs.dtype), self.moe_group
        )
        # Wait for indices AllGather right before its first consumer.
        if _idx_task is not None:
            _idx_task.wait()
        # Build padding mask and zero corresponding weights.
        padding_mask = self._global_topk_indices < 0
        self._global_topk_weights = paddle.where(
            padding_mask,
            paddle.zeros_like(self._global_topk_weights),
            self._global_topk_weights,
        )
        self.tokens_per_expert = None
        return global_hidden_states

    def token_dispatch(
        self,
        permuted_global_input_tokens: paddle.Tensor,
        fp8_dispatch: bool = False,
        async_finish: bool = False,
        use_ue8m0: bool = False,
        using_sonic_moe: bool = True,
    ):
        """No-op pass-through.  AllGather already happened in dispatch_preprocess,
        so every rank already holds the full global token list.  Returns
        (tokens, fp8_handle) where fp8_handle carries the dispatch scale if
        fp8_dispatch is active."""
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
        """Return (global_indices, global_weights, tokens_per_expert).

        tokens_per_expert uses a sync-free scatter histogram
        (:func:`_tokens_per_expert_histogram`) — no GPU->CPU sync, no
        full one-hot materialization.
        """
        tokens_per_expert = _tokens_per_expert_histogram(
            self._global_topk_indices, self.num_experts
        )
        return (
            self._global_topk_indices,
            self._global_topk_weights,
            tokens_per_expert,
        )

    def dispatch_postprocess(
        self,
        global_input_tokens: paddle.Tensor,
    ):
        """Return (global_tokens, tokens_per_expert). tokens_per_expert is None
        on this path — SonicMoE kernels recompute it from indices."""
        return global_input_tokens, self.tokens_per_expert

    def combine_preprocess(self, hidden_states: paddle.Tensor):
        """No-op pass-through."""
        return hidden_states

    def token_combine(
        self,
        hidden_states: paddle.Tensor,
        combine_overlap_handle: dict | None = None,
        async_finish: bool = False,
        fp8_combine_grad_handle: dict | None = None,
    ):
        """Combine expert outputs via ReduceScatter.

        If combine_overlap_handle is provided, fuse the ReduceScatter with the
        shared-expert subgraph via _AllGatherCombineAsync for overlap.  The
        combined output is cached for combine_postprocess to return.

        fp8_combine_grad_handle, when non-None, enables fp8 quantization of the
        combine backward gradient (halves bandwidth vs bf16).  The gathered fp8
        data+scale are written into the handle for _DownProjection.backward.
        """
        if combine_overlap_handle is None:
            # Must wrap in a PyLayer so backward populates
            # fp8_combine_grad_handle for _DownProjection.backward.
            combined_x = _AllGatherCombineNoOverlap.apply(
                hidden_states, self.moe_group, fp8_combine_grad_handle
            )
            self._overlap_combined = combined_x
            return combined_x
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
            fp8_combine_grad_handle=fp8_combine_grad_handle,
        )
        combine_overlap_handle["fn_out"] = tuple(fn_out)
        self._overlap_combined = combined_x
        return combined_x

    def combine_postprocess(self, hidden_states: paddle.Tensor):
        """Return cached ReduceScatter result from token_combine.

        token_combine sets _overlap_combined on both paths (overlap and
        no-overlap), so the fallback ReduceScatterGroupOp is defensive only.
        """
        if getattr(self, "_overlap_combined", None) is not None:
            out = self._overlap_combined
            self._overlap_combined = None
            return out
        return ReduceScatterGroupOp.apply(hidden_states, self.moe_group)


_RING_SUBGROUP_CACHE: dict = {}

# Intra-node GPU count (G) the ring splits on. Fixed rather than detected: the
# split has to match the real NVLink domain, and over-estimating it silently
# routes inter-node traffic as if it were intra-node, which shows up as an
# unexplained slowdown instead of an error. Set to the NVLink domain size of
# the target machines.
_RING_GPUS_PER_NODE = 8

# Warn only once if the gather-ahead ordering can't be installed (_order_after).
_ORDER_AFTER_WARNED = False


def _order_after(dst_group, src_group):
    """Order ``dst_group``'s comm stream after everything on ``src_group``'s.

    A comm->comm dependency. Unlike ``task.wait()``, which makes the calc stream
    wait (and so also pins any compute enqueued after it), this touches only the
    two communication streams. Both are created lazily together with the group's
    NCCL comm on its first collective, so ``get_stream`` raises until then --
    treat that as "nothing to order against yet" and report it, so the caller can
    fall back to issuing the gather the ordinary way.
    """
    from paddle import framework as _framework

    place = _framework._current_expected_place()
    try:
        dst_s = dst_group.process_group.get_stream(place)
        src_s = src_group.process_group.get_stream(place)
    except Exception as exc:
        global _ORDER_AFTER_WARNED
        if not _ORDER_AFTER_WARNED:
            logger.warning(
                "RingMoE gather-ahead disabled: no comm stream yet for one of "
                "the groups (%s). The group needs one collective before "
                "get_stream works -- see the warm-up in _build_ring_subgroups.",
                exc,
            )
            _ORDER_AFTER_WARNED = True
        return False
    dst_s.wait_stream(src_s)
    return True


def _calc_stream_all_gather(input, group):
    """Degenerate-safe AllGather over ``group``, on the calc stream.

    Not ring-specific -- used by ``_RouterAllGather`` (flat allgather AND ring),
    ``_ag_indices`` and the ring PyLayers alike. Semantics identical to
    :func:`all_gather_group` (axis-0 concat, result ready on the calc stream for
    the very next op), with the degenerate single-rank group short-circuited to a
    clone. The output is a plain tensor, so the enclosing PyLayer's backward
    (ReduceScatter) is unchanged.
    """
    if group is None or group.nranks == 1:
        return input.clone()
    return all_gather_group(input, group=group)


def _build_ring_subgroups(moe_group, gpus_per_node: int = _RING_GPUS_PER_NODE):
    """Split the EP group into intra-node and inter-node sub-groups.

    Assumes EP ranks are node-contiguous: the first G EP ranks live on one
    machine, the next G on the next, etc. ``G = min(gpus_per_node, EP)``, so an
    EP group that fits inside one machine collapses to the intra level only.
    Returns ``(G, N, intra_group, inter_group, ...)``; a level's group is None
    when degenerate (size 1).

    ``new_group`` and ``all_gather_object`` are world collectives, so every rank
    must reach this in the same order. Each rank contributes its own EP group's
    partition via ``all_gather_object``; the merged set is deduplicated and
    sorted so all ranks build identical groups in identical order. Ranks owning
    no MoE layer never reach here, which is why production pre-creates the groups
    via ``init_ring_subgroups`` at build time.
    """
    ep_ranks = list(moe_group.ranks)
    ep_size = len(ep_ranks)
    G = min(gpus_per_node, ep_size)
    if ep_size % G != 0:
        raise ValueError(
            f"RingMoE requires EP size ({ep_size}) to be a multiple of the "
            f"per-node GPU count ({G}), so that EP == N*G. Pick an EP degree "
            f"that divides evenly across whole machines."
        )
    N = ep_size // G

    key = (tuple(ep_ranks), G)
    if key in _RING_SUBGROUP_CACHE:
        return _RING_SUBGROUP_CACHE[key]

    intra_lists = (
        [ep_ranks[n * G : (n + 1) * G] for n in range(N)] if G > 1 else []
    )
    inter_lists = (
        [[ep_ranks[n * G + j] for n in range(N)] for j in range(G)]
        if N > 1
        else []
    )

    gathered: list = []
    paddle.distributed.all_gather_object(
        gathered, {"intra": intra_lists, "inter": inter_lists}
    )

    def _unique_sorted(kind: str):
        seen, out = set(), []
        for proposal in gathered:
            for lst in proposal[kind]:
                k = tuple(sorted(lst))
                if len(k) > 1 and k not in seen:
                    seen.add(k)
                    out.append(list(k))
        out.sort()
        return out

    global_rank = paddle.distributed.get_rank()
    intra_group = inter_group = intra_rs_group = intra_ag_group = (
        intra_rt_group
    ) = None
    for lst in _unique_sorted("intra"):
        g = paddle.distributed.new_group(ranks=lst)
        # Three more plain groups over the SAME intra ranks, each with its own
        # NCCL comm/stream: the output ReduceScatter (rsg), the token
        # gather-ahead (agg) and the routing gather-ahead (rtg). They must never
        # share the token-AllGather's comm, or a round's collective serializes
        # behind the next round's on that stream and the overlap is lost. Always
        # built on every rank (never env-gated) to keep the world-collective
        # new_group count consistent.
        rsg = paddle.distributed.new_group(ranks=lst)
        agg = paddle.distributed.new_group(ranks=lst)
        rtg = paddle.distributed.new_group(ranks=lst)
        if global_rank in lst:
            intra_group = g
            intra_rs_group = rsg
            intra_ag_group = agg
            intra_rt_group = rtg
    for lst in _unique_sorted("inter"):
        g = paddle.distributed.new_group(ranks=lst)
        if global_rank in lst:
            inter_group = g

    # Force the prefetch groups' NCCL comms (and with them their streams) into
    # existence now. Paddle creates both lazily on a group's first collective,
    # and these groups' only users are the gather-aheads, which need
    # ``get_stream`` to work BEFORE they issue anything -- so without a warm-up
    # the cross-stream ordering silently never installs and the optimization
    # turns itself off. Symmetric across the group's ranks, so it cannot hang.
    for _g in (intra_ag_group, intra_rt_group):
        if _g is not None:
            _warm = paddle.zeros([1], dtype="float32")
            paddle.distributed.all_reduce(_warm, group=_g)

    # One-time confirmation of the topology this rank ended up with.
    logger.info("RingMoE ring topology: G=%d N=%d", G, N)

    result = (
        G,
        N,
        intra_group,
        inter_group,
        intra_rs_group,
        intra_ag_group,
        intra_rt_group,
    )
    _RING_SUBGROUP_CACHE[key] = result
    return result


def init_ring_subgroups(moe_group=None, gpus_per_node: int | None = None):
    """Pre-create the RingMoE sub-groups on every rank, once, at build time.

    Building them lazily in ``RingMoETokenDispatcher.__init__`` would deadlock:
    the creation path is a *world* collective, but a dispatcher only exists on
    ranks whose PP stage owns a MoE layer, so a MoE-free stage would skip the
    collective while MoE-bearing stages block forever. The model builder calls
    this on every rank instead, so the groups are cached before any dispatcher
    is built. Idempotent, and a no-op when EP is degenerate.
    """
    if gpus_per_node is None:
        gpus_per_node = _RING_GPUS_PER_NODE
    if moe_group is None:
        from paddlefleet import parallel_state

        moe_group = parallel_state.get_expert_model_parallel_group(
            check_initialized=False
        )
    if moe_group is None or moe_group.nranks == 1:
        return None
    return _build_ring_subgroups(moe_group, gpus_per_node)


class _InterRingShift(paddle.autograd.PyLayer):
    """Autograd-safe async cyclic ring shift over an inter-node group.

    forward launches a non-blocking all-to-all on the comm stream (send my rows
    to ``dst``, receive from ``src``) and stashes the task in ``handle`` so the
    caller can wait lazily — this is what lets the shift overlap with the intra
    expert GEMM. backward performs the reverse (sync) shift.
    """

    @staticmethod
    def forward(ctx, x, group, dst, src, handle):
        ctx.group, ctx.dst, ctx.src = group, dst, src
        n = group.nranks
        rows = x.shape[0]
        in_split = [0] * n
        in_split[dst] = rows
        out_split = [0] * n
        out_split[src] = rows
        out = paddle.empty(x.shape, dtype=x.dtype)
        handle["task"] = paddle.distributed.stream.alltoall_single(
            out,
            x.contiguous(),
            out_split_sizes=out_split,
            in_split_sizes=in_split,
            group=group,
            sync_op=False,
            use_calc_stream=False,
        )
        return out

    @staticmethod
    def backward(ctx, grad):
        n = ctx.group.nranks
        rows = grad.shape[0]
        # Reverse direction: send to src, receive from dst.
        in_split = [0] * n
        in_split[ctx.src] = rows
        out_split = [0] * n
        out_split[ctx.dst] = rows
        out = paddle.empty(grad.shape, dtype=grad.dtype)
        task = paddle.distributed.stream.alltoall_single(
            out,
            grad.contiguous(),
            out_split_sizes=out_split,
            in_split_sizes=in_split,
            group=ctx.group,
            sync_op=True,
        )
        return out


class _RingAllGather(paddle.autograd.PyLayer):
    """AllGather over a ring sub-group. Backward is ReduceScatter-SUM.

    Same semantics as :class:`~.moe_utils.AllGatherGroupOp` but without its
    ``paddle.distributed.barrier``: the ring owns its sub-groups and output
    buffers, so it has nothing to protect, and the barrier's device-wide sync
    (once per round per layer per micro-batch, both directions) would drain the
    in-flight ``_InterRingShift`` and pipeline p2p sends -- defeating both this
    ring's overlap and VPP's ``overlap_p2p_comm``.
    """

    @staticmethod
    def forward(ctx, input, group):
        ctx.group = group
        return _calc_stream_all_gather(input, group)

    @staticmethod
    def backward(ctx, grad):
        return reduce_scatter_group(grad.contiguous(), group=ctx.group)


class _RingReduceScatterAsync(paddle.autograd.PyLayer):
    """Non-blocking ReduceScatter-SUM over a ring sub-group.

    Same autograd dual as :class:`~.moe_utils.ReduceScatterGroupOp` (forward RS,
    backward AllGather), but forward issues on the *comm* stream and stashes the
    task in ``handle`` instead of blocking the calc stream, so a round's output
    reduce overlaps the next round. Nothing reads ``partials[d]`` until
    ``ring_forward`` concatenates them, so the caller MUST drain every handle
    before touching the result. Backward stays synchronous (autograd schedules
    it in reverse order with no slack to overlap into).
    """

    @staticmethod
    def forward(ctx, x, group, handle):
        ctx.group = group
        out_shape = list(x.shape)
        out_shape[0] = out_shape[0] // group.nranks
        out = paddle.empty(shape=out_shape, dtype=x.dtype)
        handle["task"] = paddle.distributed.stream.reduce_scatter(
            out,
            x.contiguous(),
            op=paddle.distributed.ReduceOp.SUM,
            group=group,
            sync_op=False,
            use_calc_stream=False,
        )
        return out

    @staticmethod
    def backward(ctx, grad):
        return all_gather_group(grad, group=ctx.group)


class _RingFP8AllGather(paddle.autograd.PyLayer):
    """fp8 intra-node AllGather of the ring's tokens.

    Forward:  quantize the bf16 tokens to e4m3 plus a per-block int32 scale, then
    AllGather the data and the scale as TWO separate plain collectives, each into
    its own contiguous buffer (no strided unpack copy -- see
    :func:`_fp8_split_all_gather`).
    Backward: ReduceScatter-SUM the incoming bf16 grad back to the local rows.

    Same contract as :class:`_PreAllGatherFP8Result` on the flat path, and for
    the same reasons:

    * Quantizing *before* the gather is bit-identical to gathering bf16 and
      quantizing after -- AllGather is a lossless row concat and a scale block
      lives inside a single token's hidden vector, so no block ever spans two
      ranks. It also cuts on-wire bytes versus gathering bf16.
    * The uint8 packing stays *inside* this PyLayer on purpose: once packed the
      tensor is an integer dtype and autograd stops there, so the pack must
      never be an autograd-visible intermediate. This is also why the ring
      quantizes per round rather than once up front -- the inter-node rotation
      sits between quantize and gather, and hoisting it would mean folding the
      whole ring into one PyLayer. Keeping the rotation in bf16 is effectively
      free.
    * Quantization is straight-through: the grad of the fp8 output flows to the
      bf16 input unchanged (``grad_scale`` is dropped), matching the flat path.
    """

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        # Two separate plain AllGathers (data + scale), each into its own
        # contiguous buffer -> no strided unpack copy. See _fp8_split_all_gather.
        out = _fp8_split_all_gather(x, group)
        # _UpProjection.backward hands back a bf16 dx while this forward output
        # is e4m3.  Without these two, Paddle coerces the grad to the output
        # dtype and the ReduceScatter below trips NCCL's
        # "float8 dtypes are not currently supported for NCCL reductions".
        # Same pair as _PreAllGatherFP8Result on the flat path.
        ctx.set_grad_in_dtype_consistent(False)
        ctx.set_materialize_grads(False)
        return out

    @staticmethod
    def backward(ctx, grad_output, grad_scale=None):
        del grad_scale  # quantization is straight-through
        if grad_output is None:  # set_materialize_grads(False)
            return None
        group = ctx.group
        if group is None or group.nranks == 1:
            return grad_output
        return reduce_scatter_group(grad_output.contiguous(), group=group)


class _PreAllGatherFP8Ring(paddle.autograd.PyLayer):
    """Consume a pre-issued fp8 intra AllGather (Tier1 prefetch).

    Ring analogue of :class:`_PreAllGatherFP8Result`: forward waits the two async
    tasks and views the gathered fp8 data + scale (no unpack copy); backward
    ReduceScatters the bf16 grad back to the local shard. The two flags keep the
    bf16 grad from being coerced to the fp8 output dtype (else the RS trips
    NCCL's "float8 not supported for reductions"). ``handle`` is a plain dict, so
    backward returns a single value.
    """

    @staticmethod
    def forward(ctx, tok, handle):
        handle["data_task"].wait()
        handle["scale_task"].wait()
        ctx.group = handle["group"]
        # Data and scale were gathered into their OWN contiguous regular tensors
        # (see _prefetch_tok_ag_fp8), so this is a pure view -- no strided unpack
        # copy. Regular allocator tensors, so retaining them for backward is
        # ordinary activation memory.
        T_global = handle["data_buf"].shape[0]
        out = (
            handle["data_buf"].view("float8_e4m3fn"),
            handle["scale_buf"]
            .view(handle["sdt"])
            .reshape([T_global, handle["nsb"]]),
        )
        # Drop the handle's refs (the tensors live on via ``out``); keeps the
        # handle from pinning anything past consumption.
        handle["data_buf"] = None
        handle["scale_buf"] = None
        handle["data_task"] = None
        handle["scale_task"] = None
        ctx.set_grad_in_dtype_consistent(False)
        ctx.set_materialize_grads(False)
        return out

    @staticmethod
    def backward(ctx, grad_output, grad_scale=None):
        del grad_scale  # quantization is straight-through
        if grad_output is None:
            return None
        group = ctx.group
        if group is None or group.nranks == 1:
            return grad_output
        return reduce_scatter_group(grad_output.contiguous(), group=group)


class _FP8TokenQuant(paddle.autograd.PyLayer):
    """Quantize the local token ONCE; hand back the fp8 data and scale as-is.

    The ring's single quantization. Data and scale come back as two separate
    uint8 views (not concatenated -- fusing them cost a full token-payload copy
    to save only one collective). These byte tensors, not a bf16 copy, are what
    the ring passes around: hops forward them verbatim and intra gathers read
    them directly, so no round re-quantizes. Autograd runs on the uint8 carrier
    directly; backward is straight-through (quantization is identity in reverse,
    the scale carries no gradient). ``meta`` receives ``nsb``/``sdt`` for the
    gathers to interpret the scale buffer.
    """

    @staticmethod
    def forward(ctx, tok, meta):
        if quantize_activation_blockscaled_fast is None:
            raise RuntimeError(
                "Cannot find quantize_activation_blockscaled_fast, "
                "please update sonicmoe."
            )
        x = tok.contiguous()
        x_fp8, scale = quantize_activation_blockscaled_fast(
            x, scale_dtype=paddle.int32
        )
        meta["nsb"] = scale.shape[1]
        meta["sdt"] = scale.dtype
        ctx.tok_shape = list(tok.shape)
        ctx.tok_dtype = tok.dtype
        ctx.set_grad_in_dtype_consistent(False)
        ctx.set_materialize_grads(False)
        return x_fp8.view("uint8"), scale.view("uint8")

    @staticmethod
    def backward(ctx, grad, grad_scale=None):
        del grad_scale  # the block scale is not differentiable
        if grad is None:
            return None
        # Straight-through: the incoming grad is already the bf16 gradient of the
        # local token (summed by autograd over the gather and the next hop).
        return grad.astype(ctx.tok_dtype).reshape(ctx.tok_shape)


class _FP8TokenShift(paddle.autograd.PyLayer):
    """One cyclic inter-node hop of the fp8 token, data and scale side by side.

    Two non-blocking all-to-all collectives rather than one over a fused buffer:
    packing them together would cost a full copy of the data every hop, which is
    more than the second launch costs. Both tasks land in ``handle`` so the caller can
    drain them after the expert call is enqueued; that deferred wait is what lets
    the hop run underneath the GEMM. The returned buffers are valid only after
    that drain, exactly like :class:`_InterRingShift`.

    Forward moves bytes -- no dtype interpretation, no re-quantization. Backward
    is the reverse hop on the bf16 gradient of the data (send to ``src``, receive
    from ``dst``), synchronous, matching ``_InterRingShift.backward``; the scale
    has no gradient. Forward outputs are uint8 while the gradient is bf16, hence
    the dtype-consistency opt-out.
    """

    @staticmethod
    def forward(ctx, data, scale, group, dst, src, handle):
        ctx.group, ctx.dst, ctx.src = group, dst, src
        n = group.nranks
        tasks = []
        outs = []
        for t in (data, scale):
            rows = t.shape[0]
            in_split = [0] * n
            in_split[dst] = rows
            out_split = [0] * n
            out_split[src] = rows
            out = paddle.empty(t.shape, dtype=t.dtype)
            tasks.append(
                paddle.distributed.stream.alltoall_single(
                    out,
                    t.contiguous(),
                    out_split_sizes=out_split,
                    in_split_sizes=in_split,
                    group=group,
                    sync_op=False,
                    use_calc_stream=False,
                )
            )
            outs.append(out)
        handle["tasks"] = tasks
        ctx.set_grad_in_dtype_consistent(False)
        ctx.set_materialize_grads(False)
        return outs[0], outs[1]

    @staticmethod
    def backward(ctx, grad, grad_scale=None):
        del grad_scale  # the block scale is not differentiable
        if grad is None:
            return None, None
        n = ctx.group.nranks
        rows = grad.shape[0]
        in_split = [0] * n
        in_split[ctx.src] = rows
        out_split = [0] * n
        out_split[ctx.dst] = rows
        out = paddle.empty(grad.shape, dtype=grad.dtype)
        paddle.distributed.stream.alltoall_single(
            out,
            grad.contiguous(),
            out_split_sizes=out_split,
            in_split_sizes=in_split,
            group=ctx.group,
            sync_op=True,
        )
        return out, None


def _fp8_pair(data, scale, meta):
    """Wrap the fp8 data/scale pair in the dict the intra gathers expect.

    No slicing and no copy -- data and scale have been separate tensors since
    :class:`_FP8TokenQuant`, so this is pure bookkeeping and is safe to call on
    buffers whose all-to-all is still in flight (the reader just has to be
    ordered after that hop on the communication side).
    """
    return {
        "data": data,
        "scale": scale,
        "sdt": meta["sdt"],
        "nsb": meta["nsb"],
        "H": data.shape[1],
    }


def _combine_a2a_issue(partials, group, handle):
    """Start the all-to-all half of the final inter-node combine.

    ``ReduceScatter == all-to-all + local reduce``, and splitting the two buys
    three things here, all of them free:

    * The concat goes away. The monolithic ReduceScatter needed every partial
      laid out in one ``[N*T, H]`` buffer, i.e. a full copy of the combine
      payload on the critical path. The all-to-all only needs the chunks that
      actually travel, so the home chunk is never copied and for ``N==2`` there
      is no concat at all.
    * Only the traveling chunks are sent. The home chunk stays where it is and is
      added locally.
    * The transfer starts before the reduce, so whatever the caller enqueues in
      between (the shared expert) overlaps it.

    Numerically this is the same as before: for ``N==2`` NCCL's bf16
    ReduceScatter is one bf16 add per element, and so is the local sum in
    :class:`_InterCombineSum`.

    Raw and value-only -- :class:`_InterCombineSum` owns the gradient.
    """
    n, r0 = group.nranks, group.rank
    T = partials[0].shape[0]
    peers = [d for d in range(n) if d != r0]
    # alltoall_single splits the input by rank order, so the send buffer has to
    # be the traveling chunks in ascending rank order. With one peer that is a
    # single existing tensor -- no copy.
    send = (
        partials[peers[0]]
        if len(peers) == 1
        else paddle.concat([partials[d] for d in peers], axis=0)
    )
    in_split = [0] * n
    out_split = [0] * n
    for d in peers:
        in_split[d] = T
        out_split[d] = T
    recv = paddle.empty(
        [T * len(peers), *partials[0].shape[1:]], dtype=partials[0].dtype
    )
    task = paddle.distributed.stream.alltoall_single(
        recv,
        send.contiguous(),
        out_split_sizes=out_split,
        in_split_sizes=in_split,
        group=group,
        sync_op=False,
        use_calc_stream=False,
    )
    handle.update(
        {"task": task, "recv": recv, "peers": peers, "T": T, "r0": r0, "n": n}
    )


class _InterCombineSum(paddle.autograd.PyLayer):
    """Finish the split combine: wait the all-to-all, then sum in bf16.

    One bf16 add per remote chunk, which for ``N==2`` is exactly the single add
    NCCL's ``ReduceScatter_Sum_bf16`` did -- same dtype, same number of
    roundings, and bf16 addition is commutative, so the result is bit-identical
    to the unsplit ReduceScatter.

    Backward is that ReduceScatter's dual, an AllGather, sliced back per input:
    the gradient of ``partials[d]`` is the output gradient as computed on rank
    ``d``. Before the split the concat did the slicing; now it is explicit.
    """

    @staticmethod
    def forward(ctx, *partials, group, handle):
        ctx.group = group
        ctx.n = handle["n"]
        ctx.shapes = [list(p.shape) for p in partials]
        handle["task"].wait()
        recv, peers, T, r0 = (
            handle["recv"],
            handle["peers"],
            handle["T"],
            handle["r0"],
        )
        out = partials[r0]
        for i in range(len(peers)):
            out = out + recv[i * T : (i + 1) * T]
        handle.clear()
        return out

    @staticmethod
    def backward(ctx, grad):
        g = all_gather_group(grad.contiguous(), group=ctx.group)
        T = ctx.shapes[0][0]
        outs = []
        for d in range(ctx.n):
            gd = g[d * T : (d + 1) * T]
            if list(gd.shape) != ctx.shapes[d]:
                gd = gd.reshape(ctx.shapes[d])
            outs.append(gd)
        return tuple(outs)


def _routing_pair(idx, w):
    """Pack routing indices and router weights into ONE float32 tensor.

    ``[T, K] int32`` and ``[T, K] probs_dtype`` become ``[T, 2K] float32``, so the
    hop and the intra gather each move one tensor instead of two. Both payloads
    are tiny, so the concat is cheap -- unlike the token, where fusing cost a
    multi-MB copy and was undone.

    float32 is chosen so nothing is lost either way: a bf16 (or fp32) weight
    round-trips through float32 exactly, and expert ids are integers far below
    2^24. And because this is built from ordinary ``concat``/``cast`` ops, the
    gradient of ``w`` flows back through them natively -- no carrier tricks, no
    byte views.
    """
    return paddle.concat([idx.astype("float32"), w.astype("float32")], axis=1)


def _routing_unpair(pair, k, w_dtype):
    """Split a ``[..., 2K] float32`` routing pair back into ``(idx, w)``."""
    idx = pair[:, :k].astype("int32")
    w = pair[:, k:]
    if w.dtype != w_dtype:
        w = w.astype(w_dtype)
    return idx, w


def _prefetch_pair_ag(pair, group):
    """Pre-issue the intra AllGather of the fused routing pair.

    On the calc stream the routing gathers are dead serial time in front of the
    expert call. Issued here on the comm stream instead, early enough to hide
    under the GEMM. Unlike the token the hop wrote straight into a plain tensor,
    so this is an ordinary collective -- the caller only has to order it after
    that hop (see :func:`_order_after`).

    Returns a handle, or None for a degenerate group (caller falls back to the
    inline calc-stream gather).
    """
    if group is None or group.nranks == 1:
        return None
    out, task = _all_gather_async(pair, group)
    return {
        "out": out,
        "task": task,
        "shape": list(pair.shape),
        "group": group,
    }


class _PreAllGatherPair(paddle.autograd.PyLayer):
    """Consume a pre-issued AllGather of routing data (the fused pair, or w alone).

    Routing analogue of :class:`_PreAllGatherFP8Ring`: forward waits the async
    task and hands back the gathered pair, backward reduce-scatters the gradient
    to this rank's shard -- the same forward-AG / backward-RS dual
    :class:`_RouterAllGather` has, so autograd sees no difference between the
    prefetched and the inline path. ``pair`` is the differentiable local tensor
    and exists to carry the gradient; the value comes from ``handle``. Nothing
    here depends on the payload being the fused pair, so round 0 reuses it for
    the bare router weights.
    """

    @staticmethod
    def forward(ctx, pair, handle):
        handle["task"].wait()
        ctx.group = handle["group"]
        ctx.local_shape = handle["shape"]
        out = handle["out"]
        handle["out"] = None
        handle["task"] = None
        ctx.set_materialize_grads(False)
        return out

    @staticmethod
    def backward(ctx, grad):
        if grad is None:
            return None
        group = ctx.group
        if group is None or group.nranks == 1:
            return grad.reshape(ctx.local_shape)
        out = reduce_scatter_group(grad.contiguous(), group=group)
        if list(out.shape) != ctx.local_shape:
            out = out.reshape(ctx.local_shape)
        return out


def _prefetch_tok_ag_fp8(tok_fp8, group):
    """Pre-issue the intra AllGather of an ALREADY fp8 token.

    Used by the quantize-once ring: a shifted token arrives pre-quantized, so
    this skips the quantization and just issues the two async intra AllGathers
    of the fp8 data and the int32 scale. Bit-identical to quantizing bf16 then
    gathering -- the forwarded fp8 equals what this rank would have quantized
    from the (round-invariant) bf16 token. Consumed by :class:`_PreAllGatherFP8Ring`.
    """
    if group is None or group.nranks == 1:
        return None
    data_local = tok_fp8["data"]
    scale_local = tok_fp8["scale"]
    rows = data_local.shape[0]
    data_buf = paddle.empty(
        [rows * group.nranks, *data_local.shape[1:]], dtype="uint8"
    )
    scale_buf = paddle.empty(
        [rows * group.nranks, *scale_local.shape[1:]], dtype="uint8"
    )
    data_task = paddle.distributed.stream.all_gather(
        data_buf,
        data_local,
        group=group,
        sync_op=False,
        use_calc_stream=False,
    )
    scale_task = paddle.distributed.stream.all_gather(
        scale_buf,
        scale_local,
        group=group,
        sync_op=False,
        use_calc_stream=False,
    )
    return {
        "data_buf": data_buf,
        "scale_buf": scale_buf,
        "data_task": data_task,
        "scale_task": scale_task,
        "nsb": tok_fp8["nsb"],
        "sdt": tok_fp8["sdt"],
        "group": group,
    }


class RingMoETokenDispatcher(AllGatherTokenDispatcher):
    """Two-level ring dispatcher for intermediate-sharded experts.

    Mathematically identical to :class:`AllGatherTokenDispatcher`: every rank
    holds all experts sharded along the intermediate dim, and a token's output
    is the sum of every I-shard's down-proj contribution. Only the shape of the
    global gather/reduce differs. Instead of one flat AllGather/ReduceScatter
    over the whole EP group, tokens rotate through the N nodes while each node
    applies its own I-shard via intra-node AllGather/ReduceScatter over its G
    GPUs, so bulk traffic stays intra-node and only the token rows cross the
    network per hop.

    ``MoELayer`` drives :meth:`ring_forward` directly instead of the flat
    dispatch/compute/combine pipeline; the inherited AllGather methods
    (checkpoint IO, ``token_combine``, ...) keep the two paths interchangeable
    everywhere else. Numerical equivalence against the ``allgather`` dispatcher
    is the intended correctness check.
    """

    def __init__(
        self,
        moe_group: Group,
        expert_model_parallel_size: int,
        num_experts: int,
        fp8_dispatch: bool = False,
        use_ue8m0: bool = False,
    ):
        super().__init__(
            moe_group,
            expert_model_parallel_size,
            num_experts,
            fp8_dispatch=fp8_dispatch,
            use_ue8m0=use_ue8m0,
        )
        # ``use_ue8m0`` is deliberately not special-cased here. The flat
        # AllGatherTokenDispatcher stores it and never reads it either: fp8
        # *dispatch* quantization always goes through _quantize_and_pack_fp8
        # with an int32 scale, and the flag applies to other fp8 machinery
        # (weights) instead. The ring reuses that exact helper, so it is in the
        # same position as the flat path -- rejecting the combination would
        # block a config allgather runs fine.
        self.fp8_dispatch = fp8_dispatch
        # ring_forward validates the inter-node token layout on its first step.
        self._equal_tokens_checked = False
        # Handle for round 0's token AllGather, pre-issued before the gate by
        # pre_gate_token_ag and consumed as round 0's handle in the fold.
        # MoELayer issues it for every ringmoe forward (not gated on
        # moe_allgather_gate_overlap), so on the two-level ring it is always set;
        # None only on the degenerate single-node ring (no intra level).
        self._gate_pf_handle = None
        # Packed token + in-flight round-0 hop from pre_gate_token_ag; consumed
        # (and cleared) by ring_forward. None only on the degenerate ring.
        self._gate_pack = None
        # Build the 2-level ring topology on a fixed per-node GPU count
        # (``_RING_GPUS_PER_NODE``). RingMoE only
        # makes sense when EP spans MORE than one node (N>1): with N==1 the ring
        # degenerates to a single intra-node AllGather == the flat 'allgather'
        # dispatcher. Reject that (and single-GPU/no-EP) at construction with a
        # clear message instead of silently running a degenerate path.
        assert moe_group is not None and moe_group.nranks > 1, (
            "RingMoETokenDispatcher needs a real EP group (nranks>1); got "
            f"nranks={None if moe_group is None else moe_group.nranks}. Use "
            "moe_token_dispatcher_type='allgather' for single-GPU / no-EP."
        )
        (
            self.G,
            self.N,
            self.intra_group,
            self.inter_group,
            self.intra_rs_group,
            self.intra_ag_group,
            self.intra_rt_group,
        ) = _build_ring_subgroups(moe_group, _RING_GPUS_PER_NODE)
        assert self.N > 1, (
            f"RingMoETokenDispatcher requires EP to span >1 node (got N={self.N}, "
            f"G={self.G}). With N==1 the two-level ring is just a single "
            "intra-node AllGather -- set moe_token_dispatcher_type='allgather' "
            "for single-node EP (expert_model_parallel_size <= "
            f"{_RING_GPUS_PER_NODE})."
        )

    def _ag(self, t, group):
        """Autograd-safe AllGather over a sub-group (backward = ReduceScatter)."""
        if group is None or group.nranks == 1:
            return t
        return _RingAllGather.apply(t, group)

    def _ag_tokens(self, tok, group):
        """Gather the round's tokens, in fp8 when ``fp8_dispatch`` is on.

        Returns ``(gathered, scale)``; ``scale`` is None on the bf16 path. A
        degenerate group still has to produce a scale in fp8 mode, so quantize
        locally and skip only the collective.
        """
        if not self.fp8_dispatch:
            return self._ag(tok, group), None
        return _RingFP8AllGather.apply(tok, group)

    def _drain_stale_pregate(self):
        """Wait out and drop any pre-gate prefetch a previous forward left behind.

        ``pre_gate_token_ag`` issues collectives whose only consumer is
        ``ring_forward``; if that never ran (an exception in the gate, say), their
        tasks are still in flight and own their output buffers. Waiting before the
        slots are overwritten keeps the next forward's collectives from racing
        them. Failures are logged rather than raised: there is nothing left to
        recover, and raising here would mask whatever aborted the previous
        forward.
        """

        def _tasks_of(h):
            # fp8 token gather -> data_task/scale_task; plain gather and
            # _InterRingShift -> task; _FP8TokenShift -> tasks list.
            if not h:
                return ()
            out = [h.get("data_task"), h.get("scale_task"), h.get("task")]
            out.extend(h.get("tasks") or ())
            return out

        pending = list(_tasks_of(self._gate_pf_handle))
        if self._gate_pack is not None:
            pending.extend(_tasks_of(self._gate_pack.get("h_shift")))
        self._gate_pf_handle = None
        self._gate_pack = None
        for task in pending:
            if task is None:
                continue
            try:
                task.wait()
            except (RuntimeError, OSError) as exc:
                logger.warning(
                    "ringmoe pre-gate: leftover async task wait failed (%s), "
                    "discarding handle.",
                    exc,
                )

    def pre_gate_token_ag(self, tokens):
        """Pack the token ONCE and start round 0's collectives, before the gate.

        Called by MoELayer for every ringmoe forward (NOT gated on
        ``moe_allgather_gate_overlap``), with the same tensor
        ``_project_to_latent`` will hand to :meth:`ring_forward` (it caches it as
        ``_latent_hidden``), so the pack done here is the one the whole ring uses
        -- autograd stays connected through this tensor.

        Two things get issued ahead of the gate:

        * round 0's intra token AllGather, read straight off the packed bytes;
        * the round 0 -> 1 inter hop, whose payload is the local token and so
          does not need anything the gate produces. ``idx``/``w`` DO come out of
          the gate, so their (tiny) hops stay in the loop.

        Both dtypes go through here. On fp8 this also removes a redundant
        quantization (the gate prefetch used to quantize for the gather and the
        ring again for the hop); on bf16 there is nothing to quantize, so the
        token itself is the carrier and the two collectives are plain ones.

        No-op only when there is no intra level to gather over (single-node ring,
        ``intra_group is None``); on the two-level ring this is the ring's sole
        token-gather entry point, so :meth:`ring_forward` asserts it ran.
        """
        # A previous forward that aborted between here and ring_forward leaves
        # in-flight NCCL tasks owning output buffers (the round-0 gather and the
        # round 0 -> 1 hop). Wait them out before dropping the references, or the
        # collectives issued below race buffers nobody is going to read. Same
        # reasoning as _drain_async_handle on the flat allgather path.
        self._drain_stale_pregate()
        if not (self.N > 1 and self.intra_group is not None):
            return
        t = tokens.reshape([-1, tokens.shape[-1]]).contiguous()
        n = self.N
        r0 = self.inter_group.rank
        dst, src = (r0 + 1) % n, (r0 - 1) % n
        meta = {}
        h_shift = {}
        if self.fp8_dispatch:
            data, scale = _FP8TokenQuant.apply(t, meta)
            self._gate_pf_handle = _prefetch_tok_ag_fp8(
                _fp8_pair(data, scale, meta), self.intra_group
            )
            nxt_data, nxt_scale = _FP8TokenShift.apply(
                data, scale, self.inter_group, dst, src, h_shift
            )
        else:
            # bf16 ring: the same schedule with one tensor instead of two and no
            # quantization. _prefetch_pair_ag / _PreAllGatherPair are just the
            # prefetched form of _RingAllGather -- identical AllGather forward and
            # ReduceScatter backward -- so round 0's gather and its hop hide under
            # the gate exactly like they do on the fp8 path.
            data, scale, nxt_scale = t, None, None
            self._gate_pf_handle = _prefetch_pair_ag(t, self.intra_group)
            nxt_data = _InterRingShift.apply(
                t, self.inter_group, dst, src, h_shift
            )
        self._gate_pack = {
            "meta": meta,
            "data": data,
            "scale": scale,
            "nxt_data": nxt_data,
            "nxt_scale": nxt_scale,
            "h_shift": h_shift,
        }

    def _rs(self, t, group):
        """Autograd-safe ReduceScatter-SUM over a sub-group (bwd = AllGather)."""
        if group is None or group.nranks == 1:
            return t
        return ReduceScatterGroupOp.apply(t, group)

    def _rs_async(self, t, group, handle):
        """Non-blocking variant of :meth:`_rs`; caller drains ``handle``.

        Degenerate groups need no collective, so ``handle`` stays taskless and
        the drain below is a no-op -- callers can treat both cases uniformly.
        """
        if group is None or group.nranks == 1:
            return t
        return _RingReduceScatterAsync.apply(t, group, handle)

    @staticmethod
    def _drain(handles):
        """Wait out every in-flight async collective, in issue order.

        A handle carries either a single ``task`` (one collective, e.g.
        :class:`_InterRingShift`) or a ``tasks`` list (several, e.g.
        :class:`_FP8TokenShift`, which moves data and scale separately). Missing
        both means a degenerate group that issued nothing.
        """
        for h in handles:
            task = h.get("task")
            if task is not None:
                task.wait()
            for task in h.get("tasks") or ():
                if task is not None:
                    task.wait()

    def _ag_indices(self, idx, group):
        """AllGather int32 routing indices (no gradient).

        Runs on the CALC stream: the result is consumed by the very next op in
        this round, so there is nothing to overlap with, and ``sync_op=True`` on
        the comm stream would additionally make the calc stream wait on a
        cross-stream event once per round per layer for no benefit.
        """
        if group is None or group.nranks == 1:
            return idx
        return _calc_stream_all_gather(idx, group)

    def _ag_router(self, weights, group):
        """AllGather router weights (has gradient, backward = reduce-scatter)."""
        if group is None or group.nranks == 1:
            return weights
        return _RouterAllGather.apply(weights, group)

    def _check_equal_tokens(self, local_tokens: int):
        """Assert every inter-group rank holds the same token count.

        ``_InterRingShift`` fills its in/out split sizes from the *local*
        ``x.shape[0]``, so sender and receiver disagree the moment token counts
        diverge (e.g. unbalanced DP, or variable_seq_lengths without packing to
        a fixed length). NCCL then mismatches sizes, which surfaces as a hang or
        corrupted data rather than an exception. ``ring_forward`` runs this once
        per dispatcher, on the first step, since it costs an all_gather.
        """
        counts = paddle.full([1], local_tokens, dtype="int64")
        gathered = paddle.empty([self.inter_group.nranks], dtype="int64")
        paddle.distributed.stream.all_gather(
            gathered, counts, group=self.inter_group, sync_op=True
        )
        counts_list = gathered.tolist()
        if len(set(counts_list)) != 1:
            raise ValueError(
                "RingMoE requires an equal local token count on every rank of "
                f"the inter-node group, got {counts_list} (this rank: "
                f"{local_tokens}). Pack sequences to a fixed length or keep DP "
                "balanced; otherwise the inter-node shift mismatches NCCL "
                "buffer sizes."
            )

    def global_tokens_per_expert(self, topk_indices):
        """EP-wide tokens-per-expert histogram, for MoE balance logging.

        The ring never materializes the flat global index list that
        :meth:`AllGatherTokenDispatcher.get_dispatched_routing` histograms, so
        sum the per-rank histograms over the EP group instead — numerically
        identical to histogramming the AllGathered indices.
        """
        counts = _tokens_per_expert_histogram(
            topk_indices.detach().cast("int32"), self.num_experts
        )
        if self.moe_group is not None and self.moe_group.nranks > 1:
            paddle.distributed.all_reduce(counts, group=self.moe_group)
        return counts

    def _inter_combine(self, partials, group, combine_overlap_handle):
        """Final inter-node combine: fp8-free, concat-free, overlapped.

        The ReduceScatter is split into its two halves -- an all-to-all of the
        chunks that actually travel, then a local bf16 sum with the home chunk
        (see :func:`_combine_a2a_issue`). Bit-identical to the unsplit version,
        and it drops the ``[N*T, H]`` concat the monolithic collective needed.

        Quantizing that transfer to fp8 was tried and rejected: an all-to-all
        does not reduce, so fp8 IS allowed there (NCCL refuses fp8 reductions),
        but the e4m3 round-trip measurably degraded the MoE output and its
        gradients. bf16 on the wire, no exceptions.

        A ``combine_overlap_handle`` means the shared-expert subgraph runs
        between the all-to-all and the wait, on the calc stream, so the transfer
        hides behind it. The subgraph reads the pre-MoE residual, never the ring
        output, so there is no dependency to violate. It is called plainly here,
        not through ``manual_backward`` -- ordinary autograd tracks it, and the
        ring no longer interleaves anything in the reverse pass.

        ``group=None`` is the single-node ring: one partial, already final.
        """
        if group is None:
            if combine_overlap_handle is not None:
                se_out = combine_overlap_handle["fn"](
                    *combine_overlap_handle["fn_args"]
                )
                if not isinstance(se_out, tuple):
                    se_out = (se_out,)
                combine_overlap_handle["fn_out"] = tuple(se_out)
            return partials[0]

        h = {}
        _combine_a2a_issue(partials, group, h)
        if combine_overlap_handle is not None:
            se_out = combine_overlap_handle["fn"](
                *combine_overlap_handle["fn_args"]
            )
            if not isinstance(se_out, tuple):
                se_out = (se_out,)
            combine_overlap_handle["fn_out"] = tuple(se_out)
        return _InterCombineSum.apply(*partials, group=group, handle=h)

    def ring_forward(
        self,
        x_l,
        topk_weights,
        topk_indices,
        expert_fn,
        probs_dtype,
        recompute_moe_gate_up=False,
        combine_overlap_handle=None,
    ):
        """Run the two-level ring and return this rank's combined output.

        Only the token/routing tensors rotate around the N nodes, one hop at a
        time via async ``_InterRingShift``. At each stop this node applies its
        own mid-slice -- intra AllGather -> expert GEMM -> intra ReduceScatter --
        producing one partial per round; a single inter-node ReduceScatter then
        sums every node's contribution back to each home rank. Keeping the shift
        independent of the compute is what lets it overlap the intra GEMM. All
        collectives are autograd-safe PyLayers, so backward needs no extra
        wiring. Requires an equal local token count across the inter group (see
        :meth:`_check_equal_tokens`).

        Peak activation is not reduced versus the flat AllGather path (all N
        rounds' gathered tokens stay resident for their backward); the wins are
        the traffic pattern and the overlap headroom.

        ``combine_overlap_handle`` (optional, no-op when absent) runs the
        shared-expert subgraph while the final inter ReduceScatter is in flight,
        consumed in :meth:`_inter_combine`.
        """
        x_l = x_l.reshape([-1, x_l.shape[-1]]).contiguous()
        if self.fp8_dispatch and x_l.shape[-1] % 128 != 0:
            # Constrains the *latent* dim, a different axis from MoELayer's
            # per-shard fp8 alignment check.
            #
            # Do NOT relax the tile width: the fp8 kernel consumes several
            # scale blocks per tile, so the hidden width must be a multiple of
            # the full block-scale tile, not just one scale block.
            raise ValueError(
                f"RingMoE + fp8 requires the MoE hidden width "
                f"({x_l.shape[-1]}, i.e. moe_latent_size when latent MoE is on, "
                f"else hidden_size) to be a multiple of 128 (fp8 block-scale "
                f"tile)."
            )
        tok = x_l
        cur_idx = topk_indices.detach().cast("int32").contiguous()
        cur_w = topk_weights.cast(probs_dtype).contiguous()
        # Zero the padded routing lanes once, on the local tensor, instead of
        # once per round on the gathered one. Masking commutes with the
        # gathers (both elementwise on matching rows) and with their backward
        # reduce-scatter (the mask is a constant 0/1 diagonal, identical on every
        # rank of either group), so the router gradient is unchanged.
        cur_w = paddle.where(cur_idx < 0, paddle.zeros_like(cur_w), cur_w)

        n = self.N
        r0 = self.inter_group.rank
        if not self._equal_tokens_checked:
            # Once per dispatcher: catches a misconfigured token layout at the
            # first step instead of hanging inside NCCL, without paying for a
            # collective on every later step.
            self._equal_tokens_checked = True
            self._check_equal_tokens(tok.shape[0])

        intra, inter = self.intra_group, self.inter_group
        intra_on = intra is not None and intra.nranks > 1
        ag_group = self.intra_ag_group or intra
        rt_group = self.intra_rt_group
        dst, src = (r0 + 1) % n, (r0 - 1) % n

        partials = [None] * n
        rs_handles = []
        cur_tok, cur_idx_r, cur_w_r = tok, cur_idx, cur_w
        k_topk = cur_idx.shape[1]
        cur_pair = None
        # fp8 ring: quantize ONCE, here, and carry the packed bytes around the
        # ring from now on. cur_data is the autograd edge for the token (a uint8
        # tensor can hold one, so no zero-leaf indirection is needed); cur_scale
        # rides along without a gradient; cur_tok stays the carrier only on the
        # bf16 path.
        fp8_ring = self.fp8_dispatch and intra_on
        # pre_gate_token_ag may already have packed the token and started the
        # round 0 -> 1 hop, ahead of the gate. Take it over; drop our reference
        # so an aborted forward cannot hand a stale in-flight task to the next.
        pre = self._gate_pack
        self._gate_pack = None
        meta = {}
        cur_data = cur_scale = None
        if fp8_ring:
            # The fp8 ring has exactly one entry point: pre_gate_token_ag packed
            # the token and started round 0's gather and hop before the gate.
            # A real raise (not assert) so the invariant survives ``python -O``
            # instead of degenerating into a None subscript.
            if pre is None:
                raise RuntimeError(
                    "RingMoE fp8 dispatch requires the pre-gate prefetch: "
                    "MoELayer must call pre_gate_token_ag() before ring_forward()."
                )
            meta = pre["meta"]
            cur_data, cur_scale = pre["data"], pre["scale"]
        elif intra_on:
            # bf16 two-level: same single always-on entry point. MoELayer issues
            # pre_gate_token_ag for every ringmoe forward (not gated on
            # moe_allgather_gate_overlap), so round 0's gather and hop are already
            # in flight. Adopt the pre-gate's reshaped copy as the carrier so the
            # tensor whose bytes are on the wire is the one carrying the gradient.
            if pre is None:
                raise RuntimeError(
                    "RingMoE dispatch requires the pre-gate prefetch: MoELayer "
                    "must call pre_gate_token_ag() before ring_forward()."
                )
            cur_tok = pre["data"]
        # else: degenerate single-node ring (no intra level) -- nothing was
        # pre-gated, cur_tok stays the local token and round 0 gathers nothing.
        # Round 0's gather may already be in flight (issued before the gate);
        # None on the degenerate ring. Dropped after use so a later forward
        # without a fresh pre_gate_token_ag cannot reuse a stale handle.
        pf_handle = self._gate_pf_handle
        self._gate_pf_handle = None
        # Routing (idx/w) gather for THIS round, pre-issued by the previous one.
        # Round 0 has none -- idx/w come out of the gate, so unlike the token
        # there is nothing to prefetch before the loop starts.
        pf_rt = None

        for step in range(n):
            # Round 0's router-weight gather goes out FIRST, before anything else
            # queues on rt_group. It is the only routing gather that gets waited
            # on ahead of the expert call, and a wait picks up everything queued
            # before it on that comm -- issue it after the cross-stream wait on
            # the hop and the next round's pair gather, and waiting for it would
            # drag the GEMM behind the hop again (the serial prologue, once more).
            h_w0 = None
            if pf_rt is None and rt_group is not None:
                h_w0 = _prefetch_pair_ag(cur_w_r, rt_group)
            # Start the hop to the next node BEFORE the expert call. An async
            # collective makes its comm stream wait on an event recorded on the
            # calc stream at ISSUE time, so issuing it afterwards would pin the
            # transfer behind the whole GEMM -- and its payload is this round's
            # inputs, which the GEMM does not produce. The waits happen after
            # the expert call is enqueued, so the transfer runs underneath it.
            shift_handles = []
            nxt_tok = nxt_data = nxt_scale = nxt_pair = None
            pf_ahead = None
            pf_rt_ahead = None
            if step < n - 1:
                h_t, h_i = {}, {}
                if step == 0 and pre is not None:
                    # Already in flight since before the gate.
                    nxt_data, nxt_scale = pre["nxt_data"], pre["nxt_scale"]
                    if not fp8_ring:
                        nxt_tok = nxt_data
                    h_t = pre["h_shift"]
                elif fp8_ring:
                    # Bytes on the wire: half the payload of bf16, and the
                    # receiving round reads them as-is.
                    nxt_data, nxt_scale = _FP8TokenShift.apply(
                        cur_data, cur_scale, inter, dst, src, h_t
                    )
                else:
                    nxt_tok = _InterRingShift.apply(
                        cur_tok, inter, dst, src, h_t
                    )
                # idx and w travel together as one float32 pair: two tiny
                # collectives collapse into one and the concat costs almost
                # nothing (contrast the token, where fusing cost a multi-MB copy).
                nxt_pair = _InterRingShift.apply(
                    _routing_pair(cur_idx_r, cur_w_r), inter, dst, src, h_i
                )
                shift_handles = [h_t, h_i]
                if fp8_ring:
                    # Also pre-issue round step+1's gather ahead of the expert
                    # call, so it hides under the GEMM like the hop does.
                    #
                    # It reads the hop's output, and the only wait the
                    # distributed API offers (``task.wait()``) is comm->calc: it
                    # makes the CALC stream wait, which would pin the GEMM behind
                    # the hop too and lose more than this gains. So the
                    # dependency goes comm->comm instead -- the gather's stream
                    # waits on the inter stream -- and the gather reads views of
                    # the still-in-flight carrier. This works only because the
                    # carrier is raw bytes: slicing it enqueues nothing on the
                    # calc stream.
                    #
                    # On its own comm (NOT intra_group): sharing the token
                    # AllGather's comm makes this round's own gather-wait pick up
                    # everything queued behind it there -- the cross-stream wait
                    # and the next round's gather -- which put a serial
                    # prologue in front of round 0's GEMM. Same lesson as
                    # intra_rs_group.
                    if _order_after(ag_group, inter):
                        pf_ahead = _prefetch_tok_ag_fp8(
                            _fp8_pair(nxt_data, nxt_scale, meta), ag_group
                        )
                elif intra_on:
                    # Same on the bf16 path: one tensor instead of two, and no
                    # quantization, but the schedule is identical -- issue the
                    # next round's gather on its own comm, ordered after the hop
                    # comm->comm so the GEMM is not pinned behind the hop. The
                    # AllGather reads the hop's output buffer while it is still in
                    # flight, which is safe for the same reason as fp8: enqueuing
                    # it touches the calc stream not at all.
                    if _order_after(ag_group, inter):
                        pf_ahead = _prefetch_pair_ag(nxt_tok, ag_group)
                # Same treatment for the routing pair: its own comm, ordered
                # after the hop the same way. One gather instead of two.
                if rt_group is not None:
                    if _order_after(rt_group, inter):
                        pf_rt_ahead = _prefetch_pair_ag(nxt_pair, rt_group)

            if fp8_ring:
                # Always a prefetched gather: round 0's was issued before the
                # gate, every later round's during the previous round's GEMM.
                g_tok, g_scale = _PreAllGatherFP8Ring.apply(cur_data, pf_handle)
            elif intra_on:
                # bf16 two-level: the ring has a SINGLE, always-on intra-gather
                # path -- prefetched. Round 0's gather came from
                # pre_gate_token_ag (MoELayer issues it for every ringmoe
                # forward, regardless of moe_allgather_gate_overlap), later
                # rounds' from the previous round's gather-ahead. No inline
                # fallback: the ring never gathers on the calc stream.
                g_tok, g_scale = (
                    _PreAllGatherPair.apply(cur_tok, pf_handle),
                    None,
                )
            else:
                # Degenerate single-node ring (no intra level to gather over):
                # tokens are already local, so this is a passthrough -- fp8 still
                # quantizes locally to produce a scale, the collective is skipped.
                g_tok, g_scale = self._ag_tokens(cur_tok, intra)
            # Round 0 has no prefetched routing gather -- idx and w come out of
            # the gate, so there is nothing to pre-issue before the loop. Split
            # the two by what the expert call actually waits for:
            #
            #   idx  -> the histogram consumes it immediately, and the GEMM needs
            #           the histogram, so this gather stays on the CALC stream
            #           where its result is used next. Moving it to a comm stream
            #           would only add a cross-stream event in front of work that
            #           cannot start any earlier.
            #   w    -> nothing reads it until expert_fn, so it goes on the
            #           routing comm and overlaps the idx gather AND the
            #           histogram.
            #
            # (The old "weights on the comm stream slow the step" result was
            # measured when they queued behind the big token AllGather on a
            # shared comm; rt_group is a comm of its own, and round 0's token
            # gather was issued before the gate anyway.)
            if pf_rt is not None:
                g_pair = _PreAllGatherPair.apply(cur_pair, pf_rt)
                g_idx, g_w = _routing_unpair(g_pair, k_topk, cur_w_r.dtype)
            else:
                g_idx = self._ag_indices(cur_idx_r, intra)
                g_w = None
            hist = _tokens_per_expert_histogram(g_idx, self.num_experts)
            if g_w is None:
                g_w = (
                    _PreAllGatherPair.apply(cur_w_r, h_w0)
                    if h_w0 is not None
                    else self._ag_router(cur_w_r, intra)
                )

            with profile("fusion_mlp"):
                res = expert_fn(
                    g_tok,
                    g_idx,
                    g_w,
                    self.fp8_dispatch,
                    tokens_per_expert=hist,
                    fp8_scale=g_scale,
                    recompute_moe_gate_up=recompute_moe_gate_up,
                    fp8_combine_grad_handle=None,
                    # ``hist`` lives on device by design; letting SonicMoE read it
                    # back to size its metadata is a blocking D2H on the launch
                    # path, which delays this rank's collective enqueue and shows
                    # up as intra-node skew. Size from shapes instead -- rows here
                    # are dense (every rank holds all experts), so the bound is
                    # within a small margin of the exact row count.
                    sync_free_sizing=True,
                )
            y = res[0] if isinstance(res, (tuple, list)) else res

            # Enqueue the output ReduceScatter as soon as the expert output
            # exists, on a comm of its own (intra_rs_group, never the token
            # AllGather's). It depends on y alone and is not waited for here, so
            # it runs concurrently with the shift still in flight and with the
            # next round's gather. All of them are drained after the loop.
            h_rs = {}
            partials[(r0 - step) % n] = self._rs_async(
                y, self.intra_rs_group, h_rs
            )
            rs_handles.append(h_rs)

            if step < n - 1:
                self._drain(shift_handles)
                cur_pair = nxt_pair
                cur_idx_r, cur_w_r = _routing_unpair(
                    nxt_pair, k_topk, cur_w_r.dtype
                )
                if fp8_ring:
                    cur_data, cur_scale = nxt_data, nxt_scale
                else:
                    cur_tok = nxt_tok
                # Pre-issue the gather round step+1 will consume. Normally it was
                # already launched above, ahead of the expert call; this is the
                # fallback for when the comm->comm ordering could not be
                # installed (see _order_after), where it has to wait until the
                # hop is drained. Still on its own comm so this round's own
                # gather-wait cannot get stuck behind it.
                pf_handle = pf_ahead
                if pf_handle is None and fp8_ring:
                    pf_handle = _prefetch_tok_ag_fp8(
                        _fp8_pair(cur_data, cur_scale, meta), ag_group
                    )
                elif pf_handle is None and intra_on:
                    pf_handle = _prefetch_pair_ag(cur_tok, ag_group)
                # Same for the routing gather: normally already in flight from
                # ahead of the expert call, otherwise issued now.
                pf_rt = pf_rt_ahead
                if pf_rt is None and rt_group is not None:
                    pf_rt = _prefetch_pair_ag(cur_pair, rt_group)

        self._drain(rs_handles)
        out = self._inter_combine(partials, inter, combine_overlap_handle)
        return out
