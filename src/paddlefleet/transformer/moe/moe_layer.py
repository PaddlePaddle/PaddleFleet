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

import logging
from copy import deepcopy
from dataclasses import dataclass
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

from .fusion_layer_utils import FusionMoePyLayer
from .moe_expert import StandardMLPExpert
from .moe_router import DeepEPTopKRouter, StandardMoERouter
from .moe_shared_expert import StandardMLPSharedExpert
from .moe_utils import AddAuxiliaryLoss
from .token_dispatcher import AllToAllTokenDispatcher, MoEFlexTokenDispatcher

logger = logging.getLogger(__name__)


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
        self.config = config
        self.sublayers = sublayers
        routed_expert_config = deepcopy(config)
        shared_expert_config = deepcopy(config)
        self.pg_collection = pg_collection
        self.hidden_size = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size
        self.num_experts = config.n_routed_experts
        self.n_shared_experts = config.n_shared_experts
        self.moe_shared_expert_intermediate_size = None
        if self.n_shared_experts:
            self.moe_shared_expert_intermediate_size = (
                self.moe_intermediate_size * self.n_shared_experts
            )
        self.num_experts_per_tok = config.num_experts_per_tok
        self.hidden_act = config.hidden_act
        self.sequence_parallel = config.sequence_parallel
        self.tensor_model_parallel_size = config.tensor_model_parallel_size
        self.moe_token_dispatcher_type = config.moe_token_dispatcher_type
        self.moe_shared_expert_overlap = config.moe_shared_expert_overlap
        self.fp8 = config.fp8
        self.fp8_dispatch = bool(config.fp8)
        self.fp8_wgrad = config.fp8_wgrad
        self.moe_use_fusion_node = False
        if self.moe_token_dispatcher_type == "deepep":
            self.moe_use_fusion_node = True

        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.moe_grouped_gemm = config.moe_grouped_gemm
        self.moe_group = pg_collection.ep
        self.expert_model_parallel_size = (
            utils.get_pg_size(self.moe_group)
            if self.moe_group is not None
            else 1
        )
        self.num_local_experts = (
            self.num_experts // self.expert_model_parallel_size
        )

        # MoE-Related Configs
        self._init_expert_parallel()
        if config.moe_router_fusion:
            self.gate = DeepEPTopKRouter(
                config=config, pg_collection=pg_collection
            )
        else:
            self.gate = StandardMoERouter(
                config=config, pg_collection=pg_collection
            )

        self.expert_class = StandardMLPExpert
        self.shared_expert_class = StandardMLPSharedExpert

        if (
            self.expert_model_parallel_size <= 1
            and self.sequence_parallel
            and self.tensor_model_parallel_size > 1
        ):
            routed_expert_config.sequence_parallel = False
            shared_expert_config.sequence_parallel = False
        elif (
            self.expert_model_parallel_size > 1
            and self.tensor_model_parallel_size >= 1
        ):
            routed_expert_config.tensor_model_parallel_size = 1

        expert_args = {}
        expert_args["config"] = routed_expert_config
        expert_args["moe_intermediate_size"] = self.moe_intermediate_size
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
        if self.n_shared_experts > 0:
            self.shared_experts = self.shared_expert_class(**shared_expert_args)
        else:
            self.shared_experts = None

        if (
            self.moe_token_dispatcher_type == "deepep"
            and not paddle.device.current_device_is_cpu
            and paddle.device.get_device_capability()[0] < 9
        ):
            # TODO: Support Ampere architecture after upgrade deepep in paddlepaddle
            logger.info(
                "deepep in paddlepaddle does not support compute capability < 9.0, "
                "fallback to alltoall token dispatcher."
            )
            self.moe_token_dispatcher_type = "alltoall"
            self.moe_use_fusion_node = False
            self.fp8_dispatch = False

        if self.fp8:
            assert self.moe_use_fusion_node, (
                "fp8 can only be used when moe_use_fusion_node = True."
            )
        if self.moe_use_fusion_node and not self.moe_grouped_gemm:
            logger.warning(
                "moe_use_fusion_node must work with moe_grouped_gemm, but currently moe_grouped_gemm is False. "
                "Will turn on moe_grouped_gemm."
            )
            self.moe_grouped_gemm = True

        if self.expert_model_parallel_size > 1:
            if self.moe_token_dispatcher_type == "deepep":
                self.token_dispatcher = MoEFlexTokenDispatcher(
                    self.num_experts_per_device,
                    self.num_experts_per_tok,
                    self.num_experts,
                    self.moe_group,
                )
            elif self.moe_token_dispatcher_type == "alltoall":
                self.token_dispatcher = AllToAllTokenDispatcher(
                    self.moe_group,
                    self.expert_model_parallel_size,
                    self.num_experts_per_device,
                )
            else:
                raise NotImplementedError(
                    f"Unsupported moe_token_dispatcher_type {self.moe_token_dispatcher_type}"
                )

        self.recompute_moe_gate_up = (
            self.config.recompute_granularity == "selective"
            and "moe_gate_up" in self.config.recompute_modules
        )
        self.recompute_moe_premute = (
            self.config.recompute_granularity == "selective"
            and "moe_premute" in self.config.recompute_modules
        )

        if self.expert_model_parallel_size > 1:
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
            num_experts: int, expert_model_parallel_size: int
        ) -> int:
            """
            Args:
                num_experts: Total number of experts
                expert_model_parallel_size: Expert parallel groups

            Returns:
                n_routed_experts_per_device: Number of experts per device
            """
            assert num_experts >= expert_model_parallel_size, (
                f"expert num_experts={num_experts} >= moe_world_size={expert_model_parallel_size}"
            )
            assert num_experts % expert_model_parallel_size == 0, (
                f"expert num_experts={num_experts} % moe_world_size={expert_model_parallel_size} == 0"
            )

            n_routed_experts_per_device = (
                num_experts // expert_model_parallel_size
            )
            return n_routed_experts_per_device

        if self.expert_model_parallel_size > 1:
            self.moe_grad_group = self.pg_collection.expt_dp
            self.moe_rank = utils.get_pg_rank(self.moe_group)
            self.moe_rank = max(self.moe_rank, 0)
            self.num_experts_per_device = _parse_moe_expert_parallel(
                self.num_experts, self.expert_model_parallel_size
            )
        else:
            self.moe_group = None
            self.moe_rank = 0
            self.expert_model_parallel_size = 1
            self.num_experts_per_device = self.num_experts

    def expert_forward(
        self,
        dispatched_input,
        tokens_per_expert,
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
            current_expert_idx = i + self.moe_rank * self.num_experts_per_device
            expert = self.experts[current_expert_idx]
            outputs += [expert(chunk)[0]]

        if not outputs:
            return dispatched_input

        return paddle.concat(outputs, axis=0)

    def dispatch(
        self,
        hidden_states: paddle.Tensor,
        probs: paddle.Tensor,
        routing_map: paddle.Tensor,
    ):
        hidden_states = self.token_dispatcher.dispatch_preprocess(
            hidden_states, probs, routing_map
        )
        hidden_states, fp8_dispatched_handle = (
            self.token_dispatcher.token_dispatch(
                hidden_states, self.fp8_dispatch
            )
        )
        return hidden_states, fp8_dispatched_handle

    def permute(self, hidden_states: paddle.Tensor):
        global_input_tokens, tokens_per_expert = (
            self.token_dispatcher.dispatch_postprocess(hidden_states)
        )
        return global_input_tokens, tokens_per_expert

    def unpermute(self, hidden_states: paddle.Tensor):
        return self.token_dispatcher.combine_preprocess(hidden_states)

    def combine(self, hidden_states: paddle.Tensor):
        hidden_states = self.token_dispatcher.token_combine(hidden_states)
        return self.token_dispatcher.combine_postprocess(hidden_states)

    def routed_experts_compute(
        self,
        hidden_states: paddle.Tensor,
    ):
        global_input_tokens, tokens_per_expert = self.permute(hidden_states)
        expert_outs = self.expert_forward(
            global_input_tokens,
            tokens_per_expert,
        )
        return self.unpermute(expert_outs)

    # MoE forward: dispatch -> permute -> compute ->unpermute -> combine
    def custom_forward(
        self,
        hidden_states: paddle.Tensor,
        probs: paddle.Tensor,
        routing_map: paddle.Tensor,
    ):
        hidden_states, _ = self.dispatch(hidden_states, probs, routing_map)
        hidden_states = self.routed_experts_compute(hidden_states)
        return self.combine(hidden_states)

    def fusion_moe_forward(
        self,
        hidden_states: paddle.Tensor,
        probs: paddle.Tensor,
        routing_map: paddle.Tensor,
        combine_overlap_handle: dict,
    ):
        # TODO(deepllz): add fp8 dispatch config && implementation
        dispatched_hidden_states, fp8_dispatched_handle = self.dispatch(
            hidden_states, probs, routing_map
        )
        dispatched_indices = (
            self.token_dispatcher._comm_manager.dispatched_indices
        )
        dispatched_probs = self.token_dispatcher._comm_manager.dispatched_probs

        hidden_states = FusionMoePyLayer.apply(
            dispatched_hidden_states,
            dispatched_probs,
            dispatched_indices,
            self,
            self.num_experts_per_tok,
            use_fp8_mlp=self.fp8,
            recompute_moe_gate_up=self.recompute_moe_gate_up,
            recompute_moe_premute=self.recompute_moe_premute,
            fp8_dispatched_handle=fp8_dispatched_handle,
            use_bf16_gemm_weight_grad=not self.fp8_wgrad,
        )
        hidden_states = self.token_dispatcher._comm_manager.combine(
            hidden_states, combine_overlap_handle
        )
        return hidden_states

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        """
        Args:
            hidden_states: Shape: [batch_size, seq_len, hidden_size]

        Returns:
            output: Shape: [batch_size, seq_len, hidden_size]
        """
        if self.expert_model_parallel_size <= 1 and self.sequence_parallel:
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
        ) = self.gate(hidden_states)
        # topk_weights, topk_indices: Shape is [seq_len, moe_router_topk]
        # gates_masked, mask: Shape is [seq_len, num_experts], sometimes their names are "probs" and "routing_map"
        # capacity, priorities are used for dropping tokens, currently they are not used

        if (
            self.shared_experts is not None
            and self.moe_shared_expert_overlap
            and self.moe_use_fusion_node
            and self.expert_model_parallel_size > 1
        ):
            combine_overlap_handle = {
                "fn": self.shared_experts,
                "fn_args": (residuals,),
            }
        else:
            combine_overlap_handle = None

        if self.expert_model_parallel_size > 1:
            if self.moe_use_fusion_node:
                output = self.fusion_moe_forward(
                    hidden_states, gates_masked, mask, combine_overlap_handle
                )
            else:
                output = self.custom_forward(hidden_states, gates_masked, mask)
        else:
            if len(hidden_states.shape) == 3:
                batch_size, seq_len, d_model = hidden_states.shape
                reshaped_input = hidden_states.reshape([-1, d_model])
            else:
                reshaped_input = hidden_states
            output = self._forward_single_card_moe(
                reshaped_input, topk_indices, topk_weights
            )

        if self.training and self.router_aux_loss_coef:
            aux_loss = aux_loss * self.router_aux_loss_coef
            output = AddAuxiliaryLoss.apply(output, aux_loss)

        output = output.reshape(orig_shape)
        if self.shared_experts is not None:
            if combine_overlap_handle is not None:
                shared_output = combine_overlap_handle["fn_out"][0]
            else:
                shared_output = self.shared_experts(residuals)[0]
            output = output + shared_output

        if self.expert_model_parallel_size <= 1 and self.sequence_parallel:
            output = ScatterOp.apply(output)
        return output, None  # None is bias

    def _forward_single_card_moe(
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

            # use scatter to replace index_add
            final_hidden_states_tmp = paddle.zeros_like(final_hidden_states)
            final_hidden_states_tmp = paddle.scatter(
                final_hidden_states_tmp,
                idx.reshape([-1]),
                current_hidden_states.to(hidden_states.dtype),
                overwrite=False,
            )
            final_hidden_states = final_hidden_states + final_hidden_states_tmp
        return final_hidden_states.cast(hidden_states.dtype)

    def fp8_quant_weight(self, batch_mode=False, quant_transpose=True):
        if not (self.moe_use_fusion_node and self.fp8):
            return

        def quantize_weights(
            weight_list, weight_obj=None, quant_transpose=None
        ):
            """Helper function to quantize a list of weights."""
            if weight_obj is None:
                weight_obj = weight_list[0]
            if hasattr(weight_obj, "fp8_weight_stacked") or hasattr(
                weight_obj, "fp8_weight_stacked_transpose"
            ):
                return

            if quant_transpose is None:
                fp8_weight, fp8_scale = (
                    paddle.incubate.nn.functional.fused_stack_transpose_quant(
                        weight_list, transpose=False
                    )
                )
                weight_obj.fp8_weight_stacked = fp8_weight
                weight_obj.fp8_scale_stacked = fp8_scale

                fp8_weight_t, fp8_scale_t = (
                    paddle.incubate.nn.functional.fused_stack_transpose_quant(
                        weight_list, transpose=True
                    )
                )
                weight_obj.fp8_weight_stacked_transpose = fp8_weight_t
                weight_obj.fp8_scale_stacked_transpose = fp8_scale_t
            elif quant_transpose is False:
                # Only quantize without transpose
                fp8_weight, fp8_scale = (
                    paddle.incubate.nn.functional.fused_stack_transpose_quant(
                        weight_list, transpose=False
                    )
                )
                weight_obj.fp8_weight_stacked = fp8_weight
                weight_obj.fp8_scale_stacked = fp8_scale
            elif quant_transpose is True:
                # Only quantize with transpose
                fp8_weight_t, fp8_scale_t = (
                    paddle.incubate.nn.functional.fused_stack_transpose_quant(
                        weight_list, transpose=True
                    )
                )
                weight_obj.fp8_weight_stacked_transpose = fp8_weight_t
                weight_obj.fp8_scale_stacked_transpose = fp8_scale_t
            else:
                raise ValueError("Invalid value for `quant_transpose`.")

        if batch_mode:
            # Batch mode: process all experts' weights together
            expert_w1_list = [
                expert.up_gate_proj.weight
                for expert in self.experts
                if expert is not None
            ]
            expert_w2_list = [
                expert.down_proj.weight
                for expert in self.experts
                if expert is not None
            ]
            if expert_w1_list:
                quantize_weights(
                    expert_w1_list, expert_w1_list[0], quant_transpose
                )
            if expert_w2_list:
                quantize_weights(
                    expert_w2_list, expert_w2_list[0], quant_transpose
                )
        else:
            # Individual mode: process each expert's weights separately
            for expert in self.experts:
                if expert is not None:
                    quantize_weights(
                        [expert.up_gate_proj.weight],
                        quant_transpose=quant_transpose,
                    )
                    quantize_weights(
                        [expert.down_proj.weight],
                        quant_transpose=quant_transpose,
                    )

    def use_fp8(self):
        if self.moe_use_fusion_node and self.fp8:
            return True
        return False
