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

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import paddle
from paddle import nn

if TYPE_CHECKING:
    from paddle.distributed.communication.group import Group


from .fused_a2a import fused_combine, fused_dispatch
from .moe_utils import (
    AllGatherGroupOp,
    _AllToAll,
    permute,
    sort_chunks_by_idxs,
    unpermute,
)


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
    def setup_metadata(self, routing_map: paddle.Tensor, probs: paddle.Tensor):
        """Set up metadata of routing_map and probs."""
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


class _DeepepManager(_DispatchManager):
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

    def setup_metadata(self, routing_map: paddle.Tensor, probs: paddle.Tensor):
        num_tokens = routing_map.shape[0]

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
    ) -> paddle.Tensor:
        hidden_states, dispatched_probs, states, scale = fused_dispatch(
            hidden_states,
            token_indices,
            token_weights,
            self.num_experts,
            self.group,
            fp8_dispatch=fp8_dispatch,
            async_finish=async_finish,
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
    ) -> paddle.Tensor:
        hidden_states = fused_combine(
            hidden_states,
            self.group,
            self.handle,
            None,
            combine_overlap_handle,
            async_finish,
            moe_ep_barrier=self.moe_ep_barrier,
        )
        # Release the handle after combine operation
        self.handle = None
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
    ):
        super().__init__(ep_group)

        self.num_local_experts = num_local_experts
        assert self.ep_size > 1, "Flex token dispatcher requires EP > 1"
        self._comm_manager = _DeepepManager(
            group=self.ep_group,
            router_topk=num_experts_per_tok,
            num_experts=n_routed_experts,
            num_local_experts=self.num_local_experts,
            moe_ep_barrier=moe_ep_barrier,
        )

    def dispatch_preprocess(
        self,
        hidden_states: paddle.Tensor,
        probs: paddle.Tensor,
        routing_map: paddle.Tensor,
    ):
        self.hidden_shape = hidden_states.shape
        hidden_states = hidden_states.view([-1, self.hidden_shape[-1]])
        self._comm_manager.setup_metadata(routing_map, probs)
        return hidden_states

    def dispatch_preprocess_overlap(
        self,
        hidden_states: paddle.Tensor,
        token_indices: paddle.Tensor,
        token_weights: paddle.Tensor,
        probs: paddle.Tensor,
        routing_map: paddle.Tensor,
    ):
        self.hidden_shape = hidden_states.shape
        hidden_states = hidden_states.view([-1, self.hidden_shape[-1]])
        self._comm_manager.setup_metadata(routing_map, probs)
        return hidden_states, token_indices, token_weights, probs, routing_map

    def token_dispatch_overlap(
        self,
        hidden_states: paddle.Tensor,
        token_indices: paddle.Tensor,
        token_weights: paddle.Tensor,
        fp8_dispatch: bool,
        async_finish: bool = False,
    ):
        return self._comm_manager.dispatch_overlap(
            hidden_states,
            token_indices,
            token_weights,
            fp8_dispatch,
            async_finish,
        )

    def token_dispatch(
        self,
        hidden_states: paddle.Tensor,
        fp8_dispatch: bool,
        async_finish: bool = False,
    ):
        return self._comm_manager.dispatch(
            hidden_states, fp8_dispatch, async_finish
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

    def token_permutation(
        self,
        hidden_states: paddle.Tensor,
        probs: paddle.Tensor,
        routing_map: paddle.Tensor,
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        self.hidden_shape = hidden_states.shape
        hidden_states = hidden_states.view([-1, self.hidden_shape[-1]])

        self._comm_manager.setup_metadata(routing_map, probs)
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
        gates_masked: paddle.Tensor,  # probs
        mask: paddle.Tensor,  # routing_map
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        self.routing_map = mask
        self.gates_masked = gates_masked
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
        self.permutated_local_input_tokens_shape = (
            permutated_local_input_tokens.shape
        )

        return permutated_local_input_tokens

    def token_dispatch(
        self,
        permutated_local_input_tokens: paddle.Tensor,
        fp8_dispatch: bool = False,
        async_finish: bool = False,
    ):
        # Second All-to-All: Exchange expert tokens across ranks. `gathered_tokens` are the tokens that will be processed by current rank
        global_input_tokens = _AllToAll.apply(
            self.output_shape_tokens,
            permutated_local_input_tokens,  # sorted_tokens,
            out_split_sizes=self.output_splits,
            in_split_sizes=self.input_split_sizes,
            group=self.moe_group,
        )

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
            global_input_tokens, _ = sort_chunks_by_idxs(
                global_input_tokens,
                self.num_global_tokens_per_local_expert.ravel(),
                self.sort_input_by_local_experts,
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
            probs=self.gates_masked,
            routing_map=self.routing_map,
        )

        return output
