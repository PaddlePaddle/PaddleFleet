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

from abc import ABC, abstractmethod

import numpy as np
import paddle
from paddle import nn
from paddle.distributed.communication.group import Group

from .moe_utils import _AllToAll


class MoECommunicationInterface(ABC):
    def __init__(
        self,
        moe_group: Group,
        expert_parallel_degree: int,
        num_experts_per_device: int,
    ):
        self.moe_group = moe_group
        self.expert_parallel_degree = expert_parallel_degree
        self.num_experts_per_device = num_experts_per_device

    @abstractmethod
    def dispatch_and_permute(
        self,
        hidden_states: paddle.Tensor,
        gates_masked: paddle.Tensor,
        mask: paddle.Tensor,
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        """
        Dispatch and permute input tensors based on the provided mask.

        Args:
            hidden_states: Input tensor to be dispatched and permuted. Shape: [batch_size * seq_len, d_model]
            gates_masked: Masked gates. For each token(row), the selected experts are remainded with their normalized gate values, others are 0. Shape: [batch_size * seq_len, num_experts]
            mask: Mask. For each token(row), the selected experts are marked with 1, others are 0. Shape: [batch_size * seq_len, num_experts]

        Returns:
            sorted_tokens: Sorted tokens after dispatching and permuting. Shape: [num_tokens_processed_by_experts_in_current_rank, d_model]
            tokens_per_expert_current_rank: Tokens per expert for current rank. Shape: [num_experts_per_rank]
        """
        pass

    @abstractmethod
    def combine_and_unpermute(
        self, expert_outs: paddle.Tensor
    ) -> paddle.Tensor:
        """
        Combine and unpermute expert outputs.

        Args:
            expert_outs: Expert outputs. Shape: [num_tokens_processed_by_experts_in_current_rank, d_model]

        Returns:
            output: Output tensor. Shape: [batch_size * seq_len, d_model]
        """
        pass


class AllToAllMoECommunication(MoECommunicationInterface, nn.Layer):
    """
    All-to-All EP
    """

    def __init__(
        self,
        moe_group: Group,
        expert_parallel_degree: int,
        num_experts_per_device: int,
    ):
        nn.Layer.__init__(self)
        MoECommunicationInterface.__init__(
            self, moe_group, expert_parallel_degree, num_experts_per_device
        )

    def dispatch_and_permute(
        self,
        hidden_states: paddle.Tensor,
        gates_masked: paddle.Tensor,
        mask: paddle.Tensor,
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        self.gates_masked = gates_masked
        if self.expert_parallel_degree <= 1:
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
            [self.expert_parallel_degree, -1]
        ).sum(axis=1)
        # First All-to-All: Exchange expert token counts across ranks
        # Returns `tokens_per_expert_group` is for current rank
        tokens_per_expert_group = _AllToAll.apply(
            [tokens_per_expert.shape[0]],
            tokens_per_expert,
            group=self.moe_group,
        )

        if tokens_per_expert_group.sum().item() == 0:
            self.is_empty_tokens = True
        else:
            self.is_empty_tokens = False

        tokens_per_expert_group_sum = tokens_per_expert_group.reshape(
            [self.expert_parallel_degree, -1]
        )
        self.output_splits = (
            tokens_per_expert_group_sum.sum(axis=1).cpu().tolist()
        )
        self.input_split_sizes = tokens_per_ep_rank.cpu().tolist()
        output_shape = [
            tokens_per_expert_group.sum(axis=0).cpu().item(),
            sorted_tokens.shape[1],
        ]

        # Second All-to-All: Exchange expert tokens across ranks. `gathered_tokens` are the tokens that will be processed by current rank
        gathered_tokens = _AllToAll.apply(
            output_shape,
            sorted_tokens,
            out_split_sizes=self.output_splits,
            in_split_sizes=self.input_split_sizes,
            group=self.moe_group,
        )

        # Next, we should sort `gathered_tokens` by expert ids, so that the tokens for the same expert are contiguous.
        tokens_per_expert_post_gather = tokens_per_expert_group.reshape(
            [self.expert_parallel_degree, self.num_experts_per_device]
        ).sum(axis=0)
        gatherd_idxs = np.zeros(
            shape=(gathered_tokens.shape[0],), dtype=np.int32
        )
        s = 0
        for i, k in enumerate(tokens_per_expert_group.cpu().numpy()):
            gatherd_idxs[s : s + k] = i % self.num_experts_per_device
            s += k
        self.gatherd_idxs = gatherd_idxs.argsort()
        sorted_tokens = gathered_tokens[self.gatherd_idxs]

        return sorted_tokens, tokens_per_expert_post_gather

    def combine_and_unpermute(
        self,
        expert_outs: paddle.Tensor,
    ) -> paddle.Tensor:
        # Restore the original order of tokens, prepare for the third All-to-All.
        if self.is_empty_tokens:
            new_x = expert_outs
        else:
            new_x = paddle.empty_like(expert_outs)
            new_x[self.gatherd_idxs] = expert_outs

        # Third All-to-All: Exchange expert outputs back to original rank. `gathered_tokens` are the tokens that originally belong to current rank
        gathered_tokens = _AllToAll.apply(
            self.sorted_tokens_shape,
            new_x,
            out_split_sizes=self.input_split_sizes,
            in_split_sizes=self.output_splits,
            group=self.moe_group,
        )

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


class DeepEPMoECommunication(MoECommunicationInterface, nn.Layer):
    """
    DeepEP EP
    """

    def __init__(
        self,
        moe_group: Group,
        expert_parallel_degree: int,
        num_experts_per_device: int,
        token_dispatcher,
    ):
        nn.Layer.__init__(self)
        MoECommunicationInterface.__init__(
            self, moe_group, expert_parallel_degree, num_experts_per_device
        )
        self.token_dispatcher = token_dispatcher

    def dispatch_and_permute(
        self,
        hidden_states: paddle.Tensor,
        gates_masked: paddle.Tensor,
        mask: paddle.Tensor,
    ):
        (dispatched_input, tokens_per_expert) = (
            self.token_dispatcher.token_permutation(
                hidden_states,
                gates_masked,
                mask,
            )
        )
        return dispatched_input, tokens_per_expert

    def combine_and_unpermute(self, expert_outs: paddle.Tensor):
        output, _ = self.token_dispatcher.token_unpermutation(expert_outs, None)
        return output
