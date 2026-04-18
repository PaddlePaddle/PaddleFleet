# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) Microsoft Corporation.
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# Copyright (C) 2024 THL A29 Limited, a Tencent company.  All rights reserved.
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

import hashlib
import logging
import os
from typing import TYPE_CHECKING

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed.fleet.utils.sequence_parallel_utils import AllGatherOp

if TYPE_CHECKING:
    from paddlefleet.process_groups_config import ProcessGroupCollection
    from paddlefleet.transformer.transformer_config import TransformerConfig
from paddle._C_ops import matmul_grad

from paddlefleet.context_parallel_utils import ContextParallelAllGatherOp
from paddlefleet.parallel_state import get_context_parallel_world_size
from paddlefleet.transformer.moe.moe_utils import apply_random_logits

# Alignment switch: match ErnieCore's MoE router behavior for bit-level alignment
_ERNIECORE_ALIGNMENT = (
    os.environ.get("gpt_model_use_experimental_version", "0") == "1"
)
_LOG_LAYER_MD5 = os.environ.get("LOG_LAYER_MD5", "0") == "1"

# Lazy-loaded EC FusedMoETopk Triton kernel for bit-exact alignment
_FusedMoETopk = None


def _get_fused_moe_topk():
    global _FusedMoETopk
    if _FusedMoETopk is None:
        from ernie_core.ops.triton_ops.fused_moe_topk import FusedMoETopk

        _FusedMoETopk = FusedMoETopk
    return _FusedMoETopk


_moe_router_logger = logging.getLogger(__name__)


def _log_moe_md5(tensor, name, layer_idx=None):
    """Log MD5 of a tensor for MoE precision alignment debugging."""
    if _LOG_LAYER_MD5 and _ERNIECORE_ALIGNMENT:
        from paddlefleet.transformer.transformer_layer import TransformerLayer

        if TransformerLayer._skip_mtp_probes:
            return  # Skip MTP passes — EC has no MTP
        data = tensor.detach().cast("float32").numpy().tobytes()
        md5 = hashlib.md5(data).hexdigest()
        rank = (
            paddle.distributed.get_rank()
            if paddle.distributed.is_initialized()
            else 0
        )
        layer_str = f" Layer={layer_idx}" if layer_idx is not None else ""
        print(
            f"[MD5 MoE] Rank={rank}{layer_str} {name} MD5={md5} shape={list(tensor.shape)}",
            flush=True,
        )


class FusedGateDetachMatmul(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x, w):
        ctx.dtype = paddle.float32
        ctx.save_for_backward(x, w)
        return F.linear(x.cast(ctx.dtype), w.cast(ctx.dtype))

    @staticmethod
    def backward(ctx, y_grad):
        x, w = ctx.saved_tensor()
        assert ctx.dtype == y_grad.dtype, "dtype not match"
        x_g, w_g = matmul_grad(
            x.cast(ctx.dtype),
            w.cast(ctx.dtype),
            y_grad,
            False,
            False,
        )

        x_grad = x_g.cast(x.dtype) if not x.stop_gradient else None
        w_grad = w_g.cast(w.dtype) if not w.stop_gradient else None
        return x_grad, w_grad


def gate_detach_matmul(
    x, weight, use_fuse, moe_router_force_load_balancing=False
):
    if use_fuse:
        score = FusedGateDetachMatmul.apply(x, weight)
    else:
        x = x.cast(paddle.float32)
        score = F.linear(x, weight)

    if moe_router_force_load_balancing:
        score = apply_random_logits(score)
    return score


class StandardMoERouter(nn.Layer):
    def __init__(
        self,
        config: TransformerConfig,
        pg_collection: ProcessGroupCollection | None = None,
    ):
        super().__init__()
        self.config = config

        self.hidden_size = config.hidden_size
        self.num_experts = config.n_routed_experts

        self.topk_method = config.topk_method
        self.num_experts_per_tok = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        # force keep in float32 when using amp
        self._cast_to_low_precision = False

        self.n_group = config.n_group

        self.topk_group = config.topk_group

        self.routed_scaling_factor = config.routed_scaling_factor

        self.tensor_model_parallel_size = config.tensor_model_parallel_size
        self.sequence_parallel = config.sequence_parallel
        self.context_parallel_size = max(get_context_parallel_world_size(), 1)

        self.scoring_func = config.scoring_func

        self.routing_type = config.moe_router_load_balancing_type

        if self.routing_type != "seq_aux_loss" and config.get("seq_aux", False):
            raise ValueError(
                f"seq_aux is True but routing_type is {self.routing_type}. Please check."
            )

        # According to the shape of gate weights in model checkpoint
        self.weight = paddle.create_parameter(
            shape=[self.num_experts, self.hidden_size],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Uniform(),
        )

        if self.topk_method == "noaux_tc":
            self.register_buffer(
                "e_score_correction_bias",
                paddle.zeros((self.num_experts,), dtype=paddle.float32),
            )
            self._cast_to_low_precision = False
            self.expert_usage = paddle.zeros(
                shape=[self.num_experts],
                dtype=paddle.int64,
            )  # Used in MoECorrectionBiasAdjustCallback
            self.expert_usage.stop_gradient = True

    def gate_score_func(
        self, logits: paddle.Tensor, logits_type_promotion: bool = True
    ) -> paddle.Tensor:
        # [..., hidden_dim] -> [..., num_experts]
        with paddle.amp.auto_cast(False):
            if logits_type_promotion:
                logits = logits.cast("float32")
            scoring_func = self.scoring_func
            if scoring_func == "softmax":
                scores = F.softmax(logits, axis=-1)
            elif scoring_func == "sigmoid":
                scores = F.sigmoid(logits)
            elif scoring_func == "tanh":
                scores = F.tanh(logits)
            elif scoring_func == "relu":
                scores = F.relu(logits)
            elif scoring_func == "gelu":
                scores = F.gelu(logits)
            elif scoring_func == "leaky_relu":
                scores = F.leaky_relu(logits)
            else:
                raise NotImplementedError(f"{scoring_func} is not implemented.")
        return scores

    @paddle.no_grad()
    def _capacity(
        self,
        gates: paddle.Tensor,
        capacity_factor: float,
        max_capacity: int,
        min_capacity: int,
    ) -> paddle.Tensor:
        """Calculate the capacity for each expert based on the gates and capacity factor.

        Args:
            gates (paddle.Tensor): A tensor of shape [num_tokens, num_experts] representing the probability distribution
                over experts for each token.
            capacity_factor (float): A scalar float value representing the capacity factor for each expert.
            min_capacity (int): A scalar integer value representing the minimum capacity for each expert.

        Returns:
            int: A tensor value representing the calculated capacity for each expert.
        """
        assert gates.ndim == 2, (
            f"gates should be 2D, but got {gates.ndim}, {gates.shape}"
        )
        # gates has shape of SE
        num_tokens = gates.shape[0]
        num_experts = gates.shape[1]
        capacity = int((num_tokens // num_experts) * capacity_factor)
        if capacity < min_capacity:
            capacity = min_capacity
        if capacity > max_capacity:
            capacity = max_capacity
        assert capacity > 0, (
            f"requires capacity > 0, capacity_factor: {capacity_factor}, input_shape: {gates.shape}"
        )

        return capacity

    def _cal_aux_loss(self, gates, mask):
        """
        Calculate auxiliary loss

        Args:
            gates (paddle.Tensor): Represents the output probability of each expert. The shape is [batch_size, num_experts]
            mask (paddle.Tensor): Represents whether each sample belongs to a certain expert. The shape is [batch_size, num_experts]

        Returns:
            paddle.Tensor: The value of auxiliary loss.

        """
        # TODO: @DrownFish19 update aux_loss for Qwen2MoE and DeepSeekV2&V3
        me = paddle.mean(gates, axis=0)
        ce = paddle.mean(mask.cast("float32"), axis=0)
        aux_loss = paddle.sum(me * ce) * float(self.num_experts)
        return aux_loss

    def _cal_seq_aux_loss(self, probs, top_k, routing_map, seq_len, batch_size):
        # all_probs and routing_map should be computed using the runtime local sequence length on each worker.
        if (
            self.tensor_model_parallel_size > 1
            or self.context_parallel_size > 1
        ):
            local_seq_len = seq_len
            # [B*S, E]
            if self.sequence_parallel and self.tensor_model_parallel_size > 1:
                all_probs = AllGatherOp.apply(probs)
                local_seq_len = local_seq_len * self.tensor_model_parallel_size
            else:
                all_probs = probs
            # [B, S, E]
            if self.context_parallel_size > 1:
                all_probs = all_probs.reshape(
                    [
                        -1,
                        local_seq_len,
                        self.num_experts,
                    ]
                )
                # [B, S, E]
                all_probs = ContextParallelAllGatherOp.apply(all_probs, axis=1)
                local_seq_len = local_seq_len * self.context_parallel_size
            else:
                # [B, S, E]
                all_probs = all_probs.reshape(
                    [-1, local_seq_len, self.num_experts]
                )
            batch_size = all_probs.shape[0]
            # [B, S, E]
            routing_map = routing_map.reshape([batch_size, seq_len, -1])
            max_seq_len = local_seq_len
        else:
            # [B, S, E]
            if len(probs.shape) == 2:
                probs = probs.reshape([batch_size, seq_len, probs.shape[-1]])
            batch_size, local_seq_len, _ = probs.shape
            all_probs = probs
            routing_map = routing_map.reshape([batch_size, local_seq_len, -1])
            max_seq_len = local_seq_len

        seq_axis = 1
        # Both cost_coeff and seq_aux_loss must be computed with the global sequence length visible to all workers.
        # [B, E]
        cost_coeff = routing_map.sum(
            axis=seq_axis, dtype="float32"
        ) / paddle.to_tensor(
            max_seq_len * top_k / self.num_experts, dtype="float32"
        )
        # [B, E] -> [B] -> []
        seq_aux_loss = (
            (cost_coeff * all_probs.sum(axis=seq_axis) / max_seq_len)
            .sum(axis=1)
            .mean()
        )
        return seq_aux_loss

    def _cal_z_loss(self, logits) -> paddle.Tensor:
        """
        Calculate the z loss.

        Args:
            logits (paddle.Tensor): Model output. The shape is [batch_size, num_experts].

        Returns:
            paddle.Tensor: The z loss value.
        """
        l_zloss = paddle.logsumexp(logits, axis=1).square().mean()
        return l_zloss

    def _priority(
        self, topk_idx: paddle.Tensor, capacity: int
    ) -> paddle.Tensor:
        """_summary_
            The priority is the cumulative sum of the expert indices.

            This method is used in hunyuan model
        Args:
            topk_idx (paddle.Tensor): [batch_size * seq_len, topk]

        Returns:
            paddle.Tensor: cumsum locations
        """
        _, k = topk_idx.shape
        # Shape: [seq_len * k]
        chosen_expert = topk_idx.reshape([-1])
        # Shape: [seq_len * k, num_experts].
        token_priority = F.one_hot(chosen_expert, self.num_experts).cast(
            paddle.int32
        )
        token_priority = paddle.logical_and(
            token_priority > 0, token_priority.cumsum(axis=0) <= capacity
        )
        # Shape: [seq_len, num_experts].
        token_priority = token_priority.reshape([-1, k, self.num_experts]).sum(
            axis=1
        )

        return (token_priority > 0.0).astype("float32")

    def _probs_drop_policy(
        self,
        scores: paddle.Tensor,
        capacity: int,
    ) -> paddle.Tensor:
        """
        Implements the Probability-based (Probs) drop policy to enforce expert capacity.

        A token is assigned (mask value 1.0) to an expert if:
        1. It chose that expert (score > 0). (Implicitly handled by input scores).
        2. Its score for that expert is among the top 'capacity' scores for that expert.

        Args:
            scores (paddle.Tensor): [num_tokens, num_total_experts].
                                This should already contain zeros for non-selected
                                experts (i.e., the result of top-K gating).
            capacity (int): The maximum number of tokens any single expert can handle.
                                    (Not strictly used here, but good practice to include).

        Returns:
            paddle.Tensor: [num_tokens, num_total_experts] boolean mask (converted to float).
                        1.0 = Assigned and within capacity. 0.0 = Dropped or unassigned.
        """
        num_tokens, num_experts = scores.shape

        # --- Step 1: Find the 'capacity' best tokens for *each* expert ---

        # Use paddle.topk along dim=0 (the token dimension) to find the indices
        # of the tokens that have the highest scores for each expert (column).
        # Since 'scores' has shape [Tokens, Experts], dim=0 returns the token indices.

        # topk_token_indices has shape [capacity, num_total_experts]
        # It tells us WHICH tokens (row indices) are prioritized by capacity.

        # We use min(num_tokens, capacity) just in case there are fewer tokens than capacity.
        k_to_use = min(num_tokens, capacity)

        # We only care about the indices of the selected tokens
        _, topk_token_indices = paddle.topk(
            scores,
            k=k_to_use,
            dim=0,
            sorted=True,  # Sorted=True is usually faster, but we only use the indices.
        )

        # --- Step 2: Create the final assignment mask using scatter ---

        # Initialize the mask to all zeros (tokens are initially dropped/unassigned).
        # We use boolean type for efficient scattering, then convert to float later.
        final_mask = paddle.zeros(num_tokens, num_experts, dtype=paddle.bool)

        # 2a. Create the column indices for the assignment.
        # We need a tensor of shape [k_to_use, num_experts] where each row is [0, 1, 2, ..., num_experts-1].
        col_indices = (
            paddle.arange(num_experts)
            .unsqueeze(0)
            .expand_as(topk_token_indices)
        )

        # 2b. Flatten the row (token) and column (expert) indices for advanced indexing.
        token_indices_flat = topk_token_indices.flatten()
        col_indices_flat = col_indices.flatten()

        # 2c. Use advanced indexing to set the mask positions to True.
        # This sets mask[token_index, expert_index] = True for all prioritized tokens.
        final_mask[token_indices_flat, col_indices_flat] = True

        # --- Step 3: Ensure only originally selected tokens are kept ---

        # Since paddle.topk can pick up tokens with score 0 if num_tokens < capacity,
        # we must ensure that we only keep tokens that had a positive score initially.
        # This step implicitly cleans up any spurious assignments made by topk on zero scores.

        token_priority_mask = final_mask.float() * (scores > 0).float()

        return token_priority_mask

    def _topk_greedy(
        self, scores: paddle.Tensor, k: int
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        """_summary_

        Args:
            scores (paddle.Tensor): [bsz*seq_len, n_experts]
            k (int): select the top k experts

        Returns:
            tuple[paddle.Tensor, paddle.Tensor]: topk_weight, topk_idx
            topk_weight: [bsz*seq_len, k]
            topk_idx: [bsz*seq_len, k]
        """
        topk_weight, topk_idx = paddle.topk(scores, k=k, axis=-1, sorted=True)

        return topk_weight, topk_idx

    def _topk_group_limited_greedy(
        self, scores: paddle.Tensor, k: int, n_group: int, topk_group: int
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        """_summary_

        Args:
            scores (paddle.Tensor): [bsz*seq_len, n_experts]
            k (int): select the top k experts in each group
            n_groups (int): the number of groups for all experts
            topk_group (int): the number of groups selected

        Returns:
            tuple[paddle.Tensor, paddle.Tensor]: topk_weight, topk_idx
            topk_weight: [bsz*seq_len, k]
            topk_idx: [bsz*seq_len, k]

        Note: the group size is normal greater than the number of k
        """
        bsz_seq_len, n_experts = scores.shape
        assert n_experts % n_group == 0, (
            "n_experts must be divisible by n_groups"
        )

        group_scores = scores.reshape([0, n_group, -1]).max(
            axis=-1
        )  # [n, n_group]
        group_idx = paddle.topk(
            group_scores, k=topk_group, axis=-1, sorted=True
        )[1]  # [n, top_k_group]
        group_mask = paddle.zeros_like(group_scores).put_along_axis(group_idx, paddle.to_tensor(1.0), axis=-1)  # fmt:skip
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand([bsz_seq_len, n_group, n_experts // n_group])
            .reshape([bsz_seq_len, -1])
        )  # [n, e]
        tmp_scores = scores * score_mask  # [n, e]
        topk_weight, topk_idx = paddle.topk(
            tmp_scores, k=k, axis=-1, sorted=True
        )

        return topk_weight, topk_idx

    def _topk_noaux_tc(
        self, scores: paddle.Tensor, k: int, n_group: int, topk_group: int
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        """_summary_

        Args:
            scores (paddle.Tensor): [bsz*seq_len, n_experts]
            k (int): select the top k experts in each group
            n_groups (int): the number of groups for all experts
            topk_group (int): the number of groups selected

        Returns:
            tuple[paddle.Tensor, paddle.Tensor]: topk_weight, topk_idx
            topk_weight: [bsz*seq_len, k]
            topk_idx: [bsz*seq_len, k]

        Note: the group size is normal greater than the number of k
        """
        bsz_seq_len, n_experts = scores.shape
        assert n_experts % n_group == 0, (
            "n_experts must be divisible by n_groups"
        )

        assert self.e_score_correction_bias is not None, (
            "e_score_correction_bias is None"
        )
        scores_for_choice = scores.reshape(
            [bsz_seq_len, -1]
        ) + self.e_score_correction_bias.detach().unsqueeze(0)
        if n_group == 1:
            topk_weight, topk_idx = paddle.topk(
                scores_for_choice, k=k, axis=-1, sorted=True
            )
        else:
            group_scores = (
                scores_for_choice.reshape([bsz_seq_len, self.n_group, -1])
                .topk(2, axis=-1)[0]
                .sum(axis=-1)
            )  # fmt:skip [n, n_group]
            group_idx = paddle.topk(
                group_scores, k=topk_group, axis=-1, sorted=True
            )[1]  # [n, top_k_group]
            group_mask = paddle.zeros_like(group_scores).put_along_axis(group_idx, paddle.to_tensor(1.0, dtype="float32"), axis=-1)  # fmt:skip
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand([bsz_seq_len, n_group, n_experts // n_group])
                .reshape([bsz_seq_len, -1])
            )  # [n, e]
            tmp_scores = scores_for_choice * score_mask  # [n, e]
            topk_weight, topk_idx = paddle.topk(
                tmp_scores, k=k, axis=-1, sorted=True
            )

        # The bias term b is used only to adjust affinity scores for Top-K expert selection (routing); it does not affect gating.
        # The gate applied during dispatch and to weight the FFN output is computed from the original affinity score s_{i,t} (without the bias).
        topk_weight = scores.take_along_axis(topk_idx, axis=1)

        return topk_weight, topk_idx

    def _call_topk_method(
        self, topk_method, gates, k, n_group=None, topk_group=None
    ):
        if topk_method == "greedy":
            top_gate, top_idx = self._topk_greedy(gates, k=k)
        elif topk_method == "group_limited_greedy":
            top_gate, top_idx = self._topk_group_limited_greedy(
                gates,
                k,
                n_group,
                topk_group,
            )
        elif topk_method == "noaux_tc":
            top_gate, top_idx = self._topk_noaux_tc(
                gates,
                k,
                n_group,
                topk_group,
            )
        else:
            raise NotImplementedError(f"Invalid topk_method: {topk_method}")
        return top_gate, top_idx

    def set_layer_number(self, layer_number):
        self.layer_number = layer_number


class TopKRouter(StandardMoERouter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._layer_number = None

    def set_layer_number(self, layer_number):
        self._layer_number = layer_number

    def forward(self, input):
        if len(input.shape) == 3:
            if not self.sequence_parallel:
                batch_size, seq_len, d_model = input.shape
            else:
                seq_len, batch_size, d_model = input.shape
            input = input.reshape([-1, d_model])
        elif len(input.shape) == 2:
            raise ValueError(
                "The input tensor should have shape [batch_size, sequence_length, hidden_size]"
            )

        with paddle.amp.auto_cast(False):
            logits = gate_detach_matmul(
                input,
                self.weight.T,
                True,
                self.config.moe_router_force_load_balancing,
            )

        _log_moe_md5(logits, "gate_logits", self._layer_number)

        gates = self.gate_score_func(logits)

        _log_moe_md5(gates, "gate_probs_sigmoid", self._layer_number)

        gates_ori = gates
        if self.scoring_func == "sigmoid":
            gates_ori = gates_ori / (
                gates_ori.sum(axis=-1, keepdim=True) + 1e-20
            )

        if _ERNIECORE_ALIGNMENT:
            # Use EC's FusedMoETopk Triton kernel for bit-exact alignment.
            # This ensures the topk selection + normalization uses the exact same
            # GPU kernel as ErnieCore, avoiding FP32 rounding differences between
            # Triton's scalar loop and Paddle's tensor ops.
            if not hasattr(self, "_ec_topk_path_logged"):
                print(
                    "[ALIGNMENT PATH HIT] moe_router: FusedMoETopk Triton kernel used",
                    flush=True,
                )
                self._ec_topk_path_logged = True
            FusedMoETopk = _get_fused_moe_topk()
            use_node_limit = self.n_group > 1
            probs_for_choice = (
                gates + self.e_score_correction_bias.detach().unsqueeze(0)
            )
            if _LOG_LAYER_MD5 and self._layer_number == 0:
                _log_moe_md5(
                    self.e_score_correction_bias,
                    "e_score_correction_bias",
                    self._layer_number,
                )
                _log_moe_md5(
                    probs_for_choice, "probs_for_choice", self._layer_number
                )
            top_gate, top_idx = FusedMoETopk.apply(
                gates,  # gate_probs (original sigmoid scores)
                probs_for_choice,  # probs_for_choice (with correction bias)
                self.num_experts_per_tok,
                use_node_limit,
                self.n_group,
                self.topk_group,
                self.norm_topk_prob,  # norm_gate_logits
            )
            # top_gate is already normalized by the Triton kernel when norm_topk_prob=True

            _log_moe_md5(
                top_idx.cast("float32"), "topk_indices", self._layer_number
            )
            # Log raw weights and sum for alignment verification (re-computed from gate_probs)
            if _LOG_LAYER_MD5:
                raw_topk_weights = paddle.take_along_axis(
                    gates, top_idx, axis=-1
                )
                _log_moe_md5(
                    raw_topk_weights, "topk_weights_raw", self._layer_number
                )
                raw_sum = raw_topk_weights.sum(axis=-1, keepdim=True)
                _log_moe_md5(raw_sum, "topk_raw_sum", self._layer_number)
        else:
            # top_gate: [B*S, K], top_idx: [B*S, K]
            top_gate, top_idx = self._call_topk_method(
                self.topk_method,
                gates,
                k=self.num_experts_per_tok,
                n_group=self.n_group,
                topk_group=self.topk_group,
            )

            _log_moe_md5(
                top_idx.cast("float32"), "topk_indices", self._layer_number
            )
            _log_moe_md5(top_gate, "topk_weights_raw", self._layer_number)

        # z-loss
        if self.config.router_z_loss_coef:
            l_zloss = self._cal_z_loss(logits) * self.config.router_z_loss_coef
        else:
            l_zloss = None

        mask = paddle.zeros_like(gates).put_along_axis(
            top_idx, paddle.to_tensor(1.0, dtype=gates.dtype), axis=1
        )

        gates_masked = gates * mask

        # norm
        if not _ERNIECORE_ALIGNMENT:
            # When _ERNIECORE_ALIGNMENT is True, top_gate is already normalized by FusedMoETopk
            if self.norm_topk_prob:
                denominator = top_gate.sum(axis=-1, keepdim=True) + 1e-20
                top_gate = top_gate / denominator
                if self.num_experts_per_tok > 1:
                    gates_s = paddle.sum(gates_masked, axis=-1, keepdim=True)
                    denom_s = paddle.clip(
                        gates_s, min=paddle.finfo(gates_masked.dtype).eps
                    )
                    gates_masked = gates_masked / denom_s
        else:
            # Reconstruct gates_masked from top_gate (Triton kernel output) to ensure
            # bit-exact alignment. Instead of normalizing gates_masked independently
            # (which uses different FP32 reduction over E=32 elements vs K=8),
            # scatter the already-normalized top_gate values back to [S, E] layout.
            gates_masked = paddle.zeros_like(gates).put_along_axis(
                top_idx, top_gate, axis=1
            )

        if abs(self.routed_scaling_factor - 1.0) > 1e-6:
            top_gate = top_gate * self.routed_scaling_factor
            gates_masked *= self.routed_scaling_factor

        _log_moe_md5(gates_masked, "gates_masked", self._layer_number)
        _log_moe_md5(top_gate, "topk_weights_normed", self._layer_number)

        if self.topk_method == "noaux_tc":
            exp_counts = paddle.sum(mask.cast(paddle.int64), axis=0)
            with paddle.no_grad():
                self.expert_usage += exp_counts

        # aux_loss
        if self.config.router_aux_loss_coef:
            if self.routing_type == "seq_aux_loss":
                l_aux = self._cal_seq_aux_loss(
                    gates_ori,
                    self.num_experts_per_tok,
                    mask,
                    seq_len,
                    batch_size,
                )
            else:
                l_aux = self._cal_aux_loss(gates, mask)
        else:
            l_aux = None

        return (
            None,  # new capacity
            top_gate,  # weights of selected experts for each token [num_tokens, num_experts_per_token]
            top_idx,  # indices of selected experts for each token [num_tokens, num_experts_per_token]
            gates_masked,  # masked gates. for each token, the selected experts are remainded with their original values, others are 0 [num_tokens, num_experts]
            mask,  # mask. for each token, the selected experts are marked with 1s [num_tokens, num_experts]
            None,  # token priority
            l_aux,
            l_zloss,
        )
