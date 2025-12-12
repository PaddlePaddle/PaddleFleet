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

import numpy as np

from .fused_a2a import fused_combine, fused_dispatch
from .moe_utils import _AllToAll, permute, unpermute


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
    def dispatch(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        """Dispatch the hidden_states according to the routing_map."""
        pass

    @abstractmethod
    def combine(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
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
    ):
        self.group = group
        self.router_topk = router_topk
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts

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

    def dispatch(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        hidden_states, dispatched_probs, states = fused_dispatch(
            hidden_states,
            self.token_indices,
            self.token_probs,
            self.num_experts,
            self.group,
        )
        self.handle = states["handle"]
        self.tokens_per_expert = states["tokens_per_expert"]
        self.dispatched_indices = states["dispatched_indices"]
        self.dispatched_probs = dispatched_probs

        return hidden_states

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

    def combine(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        hidden_states = fused_combine(hidden_states, self.group, self.handle)
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
    ):
        super().__init__(ep_group)

        self.num_local_experts = num_local_experts
        assert self.ep_size > 1, "Flex token dispatcher requires EP > 1"
        self._comm_manager = _DeepepManager(
            group=self.ep_group,
            router_topk=num_experts_per_tok,
            num_experts=n_routed_experts,
            num_local_experts=self.num_local_experts,
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

    def token_dispatch(self, hidden_states: paddle.Tensor):
        return self._comm_manager.dispatch(hidden_states)

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

    def token_combine(self, hidden_states: paddle.Tensor):
        return self._comm_manager.combine(hidden_states)

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
        hidden_states = self._comm_manager.dispatch(hidden_states)
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
    ):
        nn.Layer.__init__(self)
        self.moe_group = moe_group
        self.expert_model_parallel_size = expert_model_parallel_size
        self.num_experts_per_device = num_experts_per_device

    def dispatch_preprocess(
        self,
        hidden_states: paddle.Tensor,
        gates_masked: paddle.Tensor,  # probs
        mask: paddle.Tensor,  # routing_map
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        self.gates_masked = gates_masked
        if self.expert_model_parallel_size <= 1:
            return hidden_states
        mask = mask.to(paddle.int64)

        if len(hidden_states.shape) == 3:
            batch_size, seq_len, d_model = hidden_states.shape
        else:
            seq_len, d_model = hidden_states.shape
        reshaped_input = hidden_states.reshape([-1, d_model])
        self.d_model = d_model
        self.reshaped_input_shape = reshaped_input.shape
        tokens_per_expert = mask.sum(axis=0)  # Shape: [num_experts]
        token_indices, expert_indices = paddle.where(mask == 1)
        combined_key = expert_indices * seq_len + token_indices
        sort_indices = paddle.argsort(combined_key)
        self.sorted_token_indices = token_indices[sort_indices]
        self.sorted_expert_indices = expert_indices[sort_indices]
        # `sorted_tokens` are tokens that sorted by expert id.
        # First `tokens_per_expert[0]` tokens belong to expert 0, next `tokens_per_expert[1]` tokens belong to expert 1, etc.
        # Shape: [batch_size * seq_len * num_experts_per_token, d_model]
        sorted_tokens = reshaped_input[self.sorted_token_indices]

        tokens_per_expert = tokens_per_expert.detach()
        self.sorted_tokens_shape = sorted_tokens.shape

        tokens_per_ep_rank = tokens_per_expert.reshape(
            [self.expert_model_parallel_size, -1]
        ).sum(axis=1)
        # First All-to-All: Exchange expert token counts across ranks
        # Returns `self.global_tokens_per_expert` is for current rank
        self.global_tokens_per_expert = _AllToAll.apply(
            [tokens_per_expert.shape[0]],
            tokens_per_expert,
            group=self.moe_group,
        )

        if self.global_tokens_per_expert.sum().item() == 0:
            self.is_empty_tokens = True
        else:
            self.is_empty_tokens = False

        tokens_per_expert_group_sum = self.global_tokens_per_expert.reshape(
            [self.expert_model_parallel_size, -1]
        )
        self.output_splits = (
            tokens_per_expert_group_sum.sum(axis=1).cpu().tolist()
        )
        self.input_split_sizes = tokens_per_ep_rank.cpu().tolist()
        self.dispatch_output_shape = [
            self.global_tokens_per_expert.sum(axis=0).cpu().item(),
            sorted_tokens.shape[1],
        ]

        return sorted_tokens

    def token_dispatch(self, sorted_tokens: paddle.Tensor):
        # Second All-to-All: Exchange expert tokens across ranks. `gathered_tokens` are the tokens that will be processed by current rank
        gathered_tokens = _AllToAll.apply(
            self.dispatch_output_shape,
            sorted_tokens,
            out_split_sizes=self.output_splits,
            in_split_sizes=self.input_split_sizes,
            group=self.moe_group,
        )
        return gathered_tokens

    def dispatch_postprocess(
        self,
        gathered_tokens: paddle.Tensor,
    ):
        # Next, we should sort `gathered_tokens` by expert ids, so that the tokens for the same expert are contiguous.
        tokens_per_expert_post_gather = self.global_tokens_per_expert.reshape(
            [self.expert_model_parallel_size, self.num_experts_per_device]
        ).sum(axis=0)
        gatherd_idxs = np.zeros(
            shape=(gathered_tokens.shape[0],), dtype=np.int32
        )
        s = 0
        for i, k in enumerate(self.global_tokens_per_expert.cpu().numpy()):
            gatherd_idxs[s : s + k] = i % self.num_experts_per_device
            s += k
        self.gatherd_idxs = gatherd_idxs.argsort()
        sorted_tokens = gathered_tokens[self.gatherd_idxs]

        return sorted_tokens, tokens_per_expert_post_gather

    def combine_preprocess(self, expert_outs: paddle.Tensor):
        # Restore the original order of tokens, prepare for the third All-to-All.
        if self.is_empty_tokens:
            new_x = expert_outs
        else:
            new_x = paddle.empty_like(expert_outs)
            new_x[self.gatherd_idxs] = expert_outs
        return new_x

    def token_combine(self, new_x: paddle.Tensor):
        # Third All-to-All: Exchange expert outputs back to original rank. `gathered_tokens` are the tokens that originally belong to current rank
        gathered_tokens = _AllToAll.apply(
            self.sorted_tokens_shape,
            new_x,
            out_split_sizes=self.input_split_sizes,
            in_split_sizes=self.output_splits,
            group=self.moe_group,
        )
        return gathered_tokens

    def combine_postprocess(self, gathered_tokens: paddle.Tensor):
        # For every processed token, need to multiply the expert weight.
        expert_major_weights = self.gates_masked[
            self.sorted_token_indices, self.sorted_expert_indices
        ]  # shape [batch_size * seq_len * num_experts_per_token]
        weighted_gathered_tokens = (
            gathered_tokens
            * expert_major_weights.unsqueeze(-1).to(gathered_tokens.dtype)
        )  # shape [batch_size * seq_len * num_experts_per_token, d_model]

        final_output_empty = paddle.zeros(
            self.reshaped_input_shape, dtype=gathered_tokens.dtype
        )
        token_indices_for_scatter = self.sorted_token_indices.unsqueeze(
            -1
        ).expand(
            -1, self.d_model
        )  # shape [batch_size * seq_len * num_experts_per_token, d_model]

        token_indices_for_scatter_single = token_indices_for_scatter[
            :, 0:1
        ].squeeze()  # shape [batch_size * seq_len * num_experts_per_token, 1]

        final_output = paddle.scatter(
            final_output_empty,
            token_indices_for_scatter_single,
            weighted_gathered_tokens,
            overwrite=False,
        )

        return final_output
