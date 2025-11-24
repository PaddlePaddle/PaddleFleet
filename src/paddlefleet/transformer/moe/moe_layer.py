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

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import paddle
from paddle import nn
from paddle.distributed.fleet.utils.sequence_parallel_utils import (
    GatherOp,
    ScatterOp,
)

if TYPE_CHECKING:
    from paddlefleet.process_groups_config import ProcessGroupCollection
    from paddlefleet.spec_utils import LayerSpec
    from paddlefleet.transformer.transformer_config import TransformerConfig

from paddlefleet import utils

from .moe_communication import AllToAllMoECommunication, DeepEPMoECommunication
from .moe_expert import StandardMLPExpert
from .moe_router import StandardMoERouter
from .moe_shared_expert import StandardMLPSharedExpert
from .moe_utils import AddAuxiliaryLoss
from .token_dispatcher import MoEFlexTokenDispatcher


@dataclass
class MoESublayers:
    """MoE Layer Sublayers spec"""

    mlp_spec: LayerSpec | type = None  # Used by experts


class MoELayer(nn.Layer):
    def __init__(
        self,
        config: TransformerConfig,
        sublayers: MoESublayers | None = None,
        pg_collection: ProcessGroupCollection | None = None,
    ):
        super().__init__()
        self.sublayers = sublayers
        routed_expert_config = deepcopy(config)
        shared_expert_config = deepcopy(config)
        config = asdict(config)
        self.pg_collection = pg_collection
        self.hidden_size = config["hidden_size"]
        self.moe_intermediate_size = config.get(
            "moe_intermediate_size", config.get("moe_ffn_hidden_size", -1)
        )
        self.num_experts = config.get(
            "moe_num_experts",
            config.get("n_routed_experts", config.get("moe_num_experts", -1)),
        )
        self.num_shared_experts = config.get("moe_num_shared_experts", 0)
        self.moe_shared_expert_intermediate_size = config.get(
            "moe_shared_expert_intermediate_size",
            self.moe_intermediate_size * self.num_shared_experts,
        )
        assert (
            self.moe_shared_expert_intermediate_size
            % self.moe_intermediate_size
            == 0
        ), (
            "moe_shared_expert_intermediate_size must be divisible by moe_intermediate_size"
        )
        self.num_shared_experts = (
            self.moe_shared_expert_intermediate_size
            // self.moe_intermediate_size
        )
        self.num_experts_per_tok = config.get(
            "num_experts_per_tok",
            config.get("num_experts_per_tok", config.get("moe_k", -1)),
        )
        self.expert_activation = config.get(
            "hidden_act", config.get("expert_activation", "silu")
        )
        self.transpose_gate_weight = config.get("transpose_gate_weight", False)
        self.sequence_parallel = config.get("sequence_parallel", False)
        self.tensor_parallel_degree = config.get("tensor_parallel_degree", 1)
        self.fuse_up_gate = config.get(
            "fuse_attention_ffn", config.get("fuse_up_gate", False)
        )
        self.moe_token_dispatcher_type = config.get(
            "moe_token_dispatcher_type", "deepep"
        )
        self.aux_loss_alpha = config.get(
            "moe_aux_loss_coeff", config.get("aux_loss_alpha", 0.0)
        )

        self.moe_group = pg_collection.ep
        self.expert_parallel_degree = (
            utils.get_pg_size(self.moe_group)
            if self.moe_group is not None
            else 1
        )

        # MoE-Related Configs
        self.expert_dropout = config.get("expert_dropout", 0.0)

        self._init_expert_parallel()
        self.router = StandardMoERouter(
            config=config, pg_collection=pg_collection
        )

        self.expert_class = StandardMLPExpert
        self.shared_expert_class = StandardMLPSharedExpert

        if (
            self.expert_parallel_degree <= 1
            and self.sequence_parallel
            and self.tensor_parallel_degree > 1
        ):
            routed_expert_config.sequence_parallel = False
            shared_expert_config.sequence_parallel = False
        elif (
            self.expert_parallel_degree > 1 and self.tensor_parallel_degree >= 1
        ):
            routed_expert_config.tensor_parallel_degree = 1

        expert_args = {}
        expert_args["config"] = routed_expert_config
        expert_args["moe_intermediate_size"] = self.moe_intermediate_size
        expert_args["fuse_up_gate"] = self.fuse_up_gate
        expert_args["is_expert"] = True
        expert_args["mlp_spec"] = self.sublayers.mlp_spec
        self.experts = nn.LayerList([])
        for i in range(self.num_experts):
            if i // self.num_experts_per_device == self.moe_rank:
                self.experts.append(self.expert_class(**expert_args))
            else:
                self.experts.append(None)

        shared_expert_args = deepcopy(expert_args)
        shared_expert_args["moe_intermediate_size"] = (
            self.moe_shared_expert_intermediate_size
        )
        shared_expert_args["is_expert"] = False
        if self.num_shared_experts > 0:
            self.shared_experts = self.shared_expert_class(**shared_expert_args)
        else:
            self.shared_experts = None

        if self.expert_parallel_degree > 1:
            if self.moe_token_dispatcher_type == "deepep":
                self.token_dispatcher = MoEFlexTokenDispatcher(
                    self.num_experts_per_device,
                    self.num_experts_per_tok,
                    self.num_experts,
                    self.moe_group,
                )
                self.communication = DeepEPMoECommunication(
                    self.moe_group,
                    self.expert_parallel_degree,
                    self.num_experts_per_device,
                    self.token_dispatcher,
                )
            elif self.moe_token_dispatcher_type == "alltoall":
                self.communication = AllToAllMoECommunication(
                    self.moe_group,
                    self.expert_parallel_degree,
                    self.num_experts_per_device,
                )
            else:
                raise NotImplementedError(
                    f"Unsupported moe_token_dispatcher_type {self.moe_token_dispatcher_type}"
                )

        if self.expert_parallel_degree > 1:
            self.is_mp_moe = False
            self.is_ep_moe = True
            for p in self.experts.parameters():
                p.is_moe_param = True
                p.color = {"color": "moe_expert", "group": self.moe_grad_group}
                p.no_sync = not self.is_mp_moe
                p.expert = not self.is_mp_moe
                if self.is_mp_moe or self.is_ep_moe:
                    p.is_distributed = True

    def _init_expert_parallel(self):
        def _parse_moe_expert_parallel(
            num_experts: int, expert_parallel_degree: int
        ) -> int:
            """
            Args:
                num_experts: Total number of experts
                expert_parallel_degree: Expert parallel groups

            Returns:
                moe_num_experts_per_device: Number of experts per device
            """
            assert num_experts >= expert_parallel_degree, (
                f"expert num_experts={num_experts} >= moe_world_size={expert_parallel_degree}"
            )
            assert num_experts % expert_parallel_degree == 0, (
                f"expert num_experts={num_experts} % moe_world_size={expert_parallel_degree} == 0"
            )

            moe_num_experts_per_device = num_experts // expert_parallel_degree
            return moe_num_experts_per_device

        if self.expert_parallel_degree > 1:
            self.moe_grad_group = self.pg_collection.expt_dp
            self.moe_rank = utils.get_pg_rank(self.moe_group)
            self.moe_rank = max(self.moe_rank, 0)
            self.num_experts_per_device = _parse_moe_expert_parallel(
                self.num_experts, self.expert_parallel_degree
            )
        else:
            self.moe_group = None
            self.moe_rank = 0
            self.expert_parallel_degree = 1
            self.num_experts_per_device = self.num_experts

    def expert_forward(
        self,
        dispatched_input,
        tokens_per_expert,
        experts,
        moe_rank,
        num_experts_per_device,
    ):
        outputs = []
        tokens_per_expert = (
            tokens_per_expert.tolist()
            if not isinstance(tokens_per_expert, list)
            else tokens_per_expert
        )
        chunks = paddle.split(
            dispatched_input, num_or_sections=tokens_per_expert, axis=0
        )
        for i, chunk in enumerate(chunks):
            if tokens_per_expert[i] == 0:
                continue
            chunk = chunk.contiguous()
            current_expert_idx = i + moe_rank * num_experts_per_device
            expert = experts[current_expert_idx]
            outputs += [expert(chunk)[0]]

        if not outputs:
            return paddle.empty(
                [0, dispatched_input.shape[-1]],
                dtype=dispatched_input.dtype,
                requires_grad=True,
            )

        return paddle.concat(outputs, axis=0)

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        """
        Args:
            hidden_states: Shape: [batch_size, seq_len, hidden_size]

        Returns:
            output: Shape: [batch_size, seq_len, hidden_size]
        """
        if self.expert_parallel_degree <= 1 and self.sequence_parallel:
            hidden_states = GatherOp.apply(hidden_states)
        orig_shape = hidden_states.shape
        residuals = hidden_states
        (
            capacity,
            topk_weights,
            topk_indices,
            gates_masked,
            mask,
            priorities,
            aux_loss,
            z_loss,
        ) = self.router(hidden_states)
        # topk_weights, topk_indices will be used in AllToAllMoECommunication
        # gates_masked, mask will be used in DeepEPMoECommunication
        # capacity, priorities are not used currently

        if self.expert_parallel_degree > 1:
            sorted_tokens, tokens_per_expert_current_rank = (
                self.communication.dispatch_and_permute(
                    hidden_states, gates_masked, mask
                )
            )
            expert_outs = self.expert_forward(
                sorted_tokens,
                tokens_per_expert_current_rank,
                self.experts,
                self.moe_rank,
                self.num_experts_per_device,
            )
            output = self.communication.combine_and_unpermute(expert_outs)
        else:
            if len(hidden_states.shape) == 3:
                batch_size, seq_len, d_model = hidden_states.shape
                reshaped_input = hidden_states.reshape([-1, d_model])
            else:
                reshaped_input = hidden_states
            output = self._forward_traditional_moe(
                reshaped_input, topk_indices, topk_weights
            )

        if self.training and self.aux_loss_alpha > 0.0:
            aux_loss = aux_loss * self.aux_loss_alpha
            output = AddAuxiliaryLoss.apply(output, aux_loss)

        output = output.reshape(orig_shape)

        if self.shared_experts is not None:
            shared_output = self.shared_experts(residuals)[0]
            output = output + shared_output

        if self.expert_parallel_degree <= 1 and self.sequence_parallel:
            output = ScatterOp.apply(output)

        return output, None  # None is bias

    def _forward_traditional_moe(
        self,
        hidden_states: paddle.Tensor,
        selected_experts: paddle.Tensor,
        topk_weights: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Forward without expert parallelism

        Args:
            hidden_states: Input hidden states, shape: [batch_size*seq_len, hidden_size]
            selected_experts: TopK experts indices, shape: [seq_len, num_experts_per_tok]
            topk_weights: TopK weights, shape: [seq_len, num_experts_per_tok]

        Returns:
            output: Output hidden states, shape: [seq_len, hidden_size]
        """

        _, d_model = hidden_states.shape
        final_hidden_states = paddle.zeros_like(
            hidden_states, dtype=hidden_states.dtype
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = paddle.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).transpose([2, 1, 0])
        tokens_per_expert = expert_mask.reshape([expert_mask.shape[0], -1]).sum(
            axis=-1
        )
        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            top_x, idx = paddle.where(expert_mask[expert_idx])
            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            if tokens_per_expert[expert_idx] <= 0.1:
                continue
            current_state = hidden_states[idx, None].reshape([-1, d_model])
            expert_out = expert_layer(current_state)[0]
            current_weight = topk_weights[idx, top_x].unsqueeze(-1)
            current_hidden_states = expert_out * current_weight
            final_hidden_states.index_add_(
                index=idx.reshape([-1]),
                axis=0,
                value=current_hidden_states.to(hidden_states.dtype),
            )

        return final_hidden_states.cast(hidden_states.dtype)
