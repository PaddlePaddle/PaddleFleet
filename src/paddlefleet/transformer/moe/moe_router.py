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
from functools import partial
from typing import TYPE_CHECKING

import paddle
import paddle.nn.functional as F
from paddle import framework, nn
from paddle.distributed.fleet.utils.sequence_parallel_utils import (
    AllGatherOp,
    mark_as_sequence_parallel_parameter,
)

from paddlefleet.tensor_parallel.sequence_parallel_utils_legacy import (
    GatherOpLegacy,
)

if TYPE_CHECKING:
    from paddlefleet.process_groups_config import ProcessGroupCollection
    from paddlefleet.transformer.transformer_config import TransformerConfig
from paddle._C_ops import matmul_grad
from paddle.distributed.fleet.meta_parallel.zero_bubble_utils import (
    WeightGradStore,
)
from paddle.distributed.fleet.utils.sequence_parallel_utils import ScatterOp

from paddlefleet.context_parallel_utils import (
    ContextParallelAllGatherOp,
    ContextParallelGatherOp,
    ContextParallelScatterOp,
)
from paddlefleet.parallel_state import (
    get_context_parallel_rank,
    get_context_parallel_world_size,
    get_tensor_model_parallel_group,
)
from paddlefleet.transformer.moe.moe_utils import apply_random_logits
from paddlefleet.transformer.transformer_config import dw_overlap_enabled

# MD5 logging for MoE router precision debugging
_LOG_LAYER_MD5 = os.environ.get("LOG_LAYER_MD5", "0") == "1"

# Lazy-loaded MoETopkFusion Triton kernel for bit-exact alignment
_MoETopkFusion = None


def _get_moe_topk_fusion():
    global _MoETopkFusion
    if _MoETopkFusion is None:
        from paddlefleet.triton_ops.moe_topk_fusion import MoETopkFusion

        _MoETopkFusion = MoETopkFusion
    return _MoETopkFusion


_moe_router_logger = logging.getLogger(__name__)


def _log_moe_md5(tensor, name, layer_idx=None):
    """Log MD5 of a tensor for MoE precision alignment debugging."""
    from paddlefleet.transformer.transformer_layer import TransformerLayer

    if _LOG_LAYER_MD5 and TransformerLayer._gpt_model_use_experimental_version:
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


_ROUTER_SCALE_FAST = None


def _router_scale_fast_enabled():
    """Opt-in switch for the atomic-free routed-scaling-factor gather backward.

    Default OFF keeps the original ``F.embedding`` path bit-for-bit.
    """
    global _ROUTER_SCALE_FAST
    if _ROUTER_SCALE_FAST is None:
        _ROUTER_SCALE_FAST = (
            os.environ.get("FLEET_MOE_ROUTER_SCALE_FAST", "0") == "1"
        )
    return _ROUTER_SCALE_FAST


class GatherExpertScale(paddle.autograd.PyLayer):
    """Gather per-expert scales for the selected experts.

    Forward is identical to ``F.embedding(idx, param.unsqueeze(1)).squeeze(-1)``
    (both are a plain gather). The backward differs: instead of atomically
    accumulating num_tokens*topk gradients into only num_experts addresses, it
    scatters them into a dense [num_tokens, num_experts] fp32 buffer and does a
    column sum. The atomic version is both slow (a few hundred addresses take
    ~1e3 atomics each) and, in bf16, badly inaccurate (the accumulator saturates
    once it grows past 2**8 times the increment).
    """

    @staticmethod
    def forward(ctx, param, idx):
        """forward"""
        ctx.save_for_backward(idx)
        ctx.num_experts = param.shape[0]
        return paddle.gather(param, idx.reshape([-1])).reshape(idx.shape)

    @staticmethod
    def backward(ctx, grad):
        """backward"""
        (idx,) = ctx.saved_tensor()
        dense = paddle.zeros(
            [idx.shape[0], ctx.num_experts], dtype=paddle.float32
        )
        dense = paddle.put_along_axis(
            dense, idx, grad.astype(paddle.float32), axis=1, reduce="add"
        )
        return dense.sum(axis=0).astype(grad.dtype), None


def apply_learnable_routed_scaling(top_gate, top_idx, param):
    """Scale top_gate by the learnable per-expert routed scaling factor.

    top_idx may contain -1 for padded tokens; those are clipped to 0 exactly as
    before (their top_gate is already 0).
    """
    safe_topk_indices = paddle.clip(top_idx, min=0)
    if _router_scale_fast_enabled():
        gathered_scales = GatherExpertScale.apply(param, safe_topk_indices)
    else:
        gathered_scales = F.embedding(
            safe_topk_indices, param.unsqueeze(1)
        ).squeeze(-1)
    return top_gate * gathered_scales


class FusedGateDetachMatmul(paddle.autograd.PyLayer):
    """
    FusedGateDetachMatmul
    """

    @staticmethod
    def forward(ctx, x, w, defer_dw=False, use_accuracy_compatible=False):
        """
        forward
        """
        ctx.defer_dw = defer_dw
        ctx.use_accuracy_compatible = use_accuracy_compatible

        ctx.dtype = paddle.float32
        ctx.save_for_backward(x, w)
        w = w.T
        return F.linear(x.cast(ctx.dtype), w.cast(ctx.dtype))

    @staticmethod
    def backward(ctx, y_grad):
        """
        backward
        """
        x, w = ctx.saved_tensor()
        assert ctx.dtype == y_grad.dtype, "dtype not match"

        w_stop_grad = w.stop_gradient
        x_stop_grad = x.stop_gradient

        def _compute_weight_grad(x_cast, y_grad, weight):
            with paddle.amp.auto_cast(False):
                w_grad = paddle.matmul(
                    x_cast, y_grad, transpose_x=True
                ).T  # 始终先算梯度

            # MG returns `grad_weight.to(weight_dtype)` and only then accumulates
            # it into the fp32 main_grad, so the wgrad passes through the weight
            # storage dtype first.
            if ctx.use_accuracy_compatible:
                w_grad = w_grad.cast(weight.dtype).cast(paddle.float32)

            if hasattr(weight, "main_grad"):
                if weight.main_grad is None:
                    weight.main_grad = paddle.zeros(
                        weight.shape, dtype=paddle.float32
                    )
                assert w_grad.dtype == weight.main_grad.dtype, (
                    f"w_grad dtype {w_grad.dtype} != main_grad dtype {weight.main_grad.dtype}"
                )
                weight.main_grad.add_(w_grad)
            else:
                raise AssertionError("fp8 overlap need main_grad attribute")

            if hasattr(weight, "_apply_backward_hook"):
                weight._apply_backward_hook()

        if ctx.defer_dw:
            x_cast = x.cast(ctx.dtype)
            w_cast = w.cast(ctx.dtype)

            x_g = paddle.matmul(y_grad, w_cast.T, transpose_y=True)
            x_grad = x_g.cast(x.dtype) if not x_stop_grad else None

            if w_stop_grad:
                return x_grad, None
            else:
                WeightGradStore.enabled = True
                WeightGradStore.put(
                    partial(
                        _compute_weight_grad,
                        x_cast.detach(),
                        y_grad.detach(),
                        w,
                    )
                )
                WeightGradStore.enabled = False
                return x_grad, None
        else:
            if ctx.use_accuracy_compatible:
                # Mirror MG `RouterGatingLinearFunction.backward`:
                #   grad_input  = torch.mm(grad_output, weight.to(router_dtype))
                #   grad_weight = torch.mm(grad_output.t(), inp.to(router_dtype))
                # i.e. two separate GEMMs (not a fused matmul_grad), then each
                # gradient cast back to its own storage dtype.
                x_g = paddle.matmul(y_grad, w.cast(ctx.dtype))
                w_g = paddle.matmul(y_grad, x.cast(ctx.dtype), transpose_x=True)
                x_grad = x_g.cast(x.dtype) if not x_stop_grad else None
                w_grad = w_g.cast(w.dtype) if not w_stop_grad else None
                return x_grad, w_grad
            else:
                w = w.T
                x_g, w_g = matmul_grad(
                    x.cast(ctx.dtype),
                    w.cast(ctx.dtype),
                    y_grad,
                    False,
                    False,
                )

                x_grad = x_g.cast(x.dtype) if not x_stop_grad else None
                w_grad = w_g.cast(w.dtype) if not w_stop_grad else None
                if w_grad is not None:
                    w_grad = w_grad.T

                return x_grad, w_grad


def gate_detach_matmul(
    x,
    weight,
    use_fuse,
    moe_router_force_load_balancing=False,
    defer_dw=False,
    use_accuracy_compatible=False,
):
    if use_fuse:
        score = FusedGateDetachMatmul.apply(
            x, weight, defer_dw, use_accuracy_compatible
        )
    else:
        x = x.cast(paddle.float32)
        score = F.linear(x, weight)

    if moe_router_force_load_balancing:
        score = apply_random_logits(score)
    return score


def _apply_routing_map_fusion(
    gates, top_idx, input_ids_none_zero_mask, input_ids=None, pad_token_id=0
):
    from paddlefleet.triton_ops import routing_map_fusion_forward

    if input_ids_none_zero_mask is not None and input_ids is not None:
        fused_input_ids = input_ids.reshape([-1])
    else:
        fused_input_ids = None
    fused_mask, top_idx, exp_counts = routing_map_fusion_forward(
        gates,
        top_idx,
        input_ids=fused_input_ids,
        is_pure_text_line=None,
        pad_token_id=pad_token_id,
    )
    mask = fused_mask.cast(gates.dtype)
    return mask, top_idx, exp_counts


class StandardMoERouter(nn.Layer):
    def __init__(
        self,
        config: TransformerConfig,
        pg_collection: ProcessGroupCollection | None = None,
    ):
        super().__init__()
        self.config = config
        self.use_accuracy_compatible = getattr(
            config, "use_accuracy_compatible", False
        )

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
        self.routed_scaling_factor_learnable = (
            config.routed_scaling_factor_learnable
        )

        self.tensor_model_parallel_size = config.tensor_model_parallel_size
        self.sequence_parallel = config.sequence_parallel
        self.context_parallel_size = max(get_context_parallel_world_size(), 1)

        self.scoring_func = config.scoring_func

        self.routing_type = config.moe_router_load_balancing_type

        if self.routing_type != "seq_aux_loss" and config.get("seq_aux", False):
            raise ValueError(
                f"seq_aux is True but routing_type is {self.routing_type}. Please check."
            )

        if self.routing_type == "seq_aux_loss" and self.scoring_func not in (
            "softmax",
            "sigmoid",
            "relu",
            "sftplus",
            "sqrtsoftplus",
        ):
            raise ValueError(
                "seq_aux_loss requires a non-negative MoE scoring_func, "
                f"but got {self.scoring_func!r}. "
            )

        # Initialize gate weight with Normal distribution aligned with Megatron.
        if self.use_accuracy_compatible:
            self.weight = paddle.create_parameter(
                shape=[self.num_experts, self.hidden_size],
                dtype=config.params_dtype,
                default_initializer=paddle.nn.initializer.Constant(0.0),
            )
        else:
            self.weight = paddle.create_parameter(
                shape=[self.num_experts, self.hidden_size],
                dtype="float32",
                default_initializer=paddle.nn.initializer.Constant(0.0),
            )
        config.init_method(self.weight)

        if (
            self.sequence_parallel
            and self.config.expert_model_parallel_size > 1
        ):
            mark_as_sequence_parallel_parameter(self.weight)

        # Multi-view (split-feature) routing: instead of a single gate
        # projection, score each expert with the SUM of two independent views.
        # The routing score is score_func(logits_0) + score_func(logits_1),
        # where logits_0 reuses the existing ``self.weight`` gate and logits_1
        # comes from a new projection ``self.weight_1``. This gives the router
        # two independent "views" of each token while adding only one extra
        # projection and keeping the expert FFN compute unchanged.
        #
        # Disabled by default so that existing configs / checkpoints keep using
        # the single ``self.weight`` gate unchanged; enable via the
        # ``moe_split_feature_routing`` config flag. Hash-routing layers keep
        # using ``self.weight`` as the single gate regardless of this flag.
        self.moe_split_feature_routing = getattr(
            config, "moe_split_feature_routing", False
        )
        if self.moe_split_feature_routing:
            # Same layout / init as ``self.weight`` ([num_experts, hidden_size])
            # so the two views are symmetric and the projection can reuse the
            # fused gate matmul (force-load-balancing and defer_dw paths
            # included). ``self.weight`` is reused as the first view, so no
            # extra gate is wasted. The scoring_func == "sigmoid" contract is
            # checked later in set_layer_number(), once we know whether this is
            # a hash-routing layer (hash layers bypass split routing and may use
            # a non-sigmoid scoring_func).
            self.weight_1 = paddle.create_parameter(
                shape=[self.num_experts, self.hidden_size],
                dtype="float32",
                default_initializer=paddle.nn.initializer.Constant(0.0),
            )
            config.init_method(self.weight_1)
            if (
                self.sequence_parallel
                and self.config.expert_model_parallel_size > 1
            ):
                mark_as_sequence_parallel_parameter(self.weight_1)

        if self.routed_scaling_factor_learnable:
            self.routed_scaling_factor_param = self.create_parameter(
                shape=[self.num_experts],
                dtype="float32",
                default_initializer=nn.initializer.Constant(
                    self.routed_scaling_factor
                ),
            )
            if (
                self.sequence_parallel
                and self.config.expert_model_parallel_size > 1
            ):
                mark_as_sequence_parallel_parameter(
                    self.routed_scaling_factor_param
                )

        if self.topk_method == "noaux_tc":
            if not self.config.gpt_model_use_experimental_version:
                self.register_buffer(
                    "e_score_correction_bias",
                    paddle.zeros((self.num_experts,), dtype=paddle.float32),
                )
            else:
                self.register_buffer(
                    "e_score_correction_bias",
                    paddle.zeros((1, self.num_experts), dtype=paddle.float32),
                )
            self._cast_to_low_precision = False
            self.expert_usage = paddle.zeros(
                shape=[self.num_experts],
                dtype=paddle.int64,
            )  # Used in MoECorrectionBiasAdjustCallback
            self.expert_usage.stop_gradient = True

        if self.topk_method == "quantile_balancing":
            if self.routing_type != "none":
                raise ValueError(
                    "quantile_balancing is a self-contained load balancing method, "
                    "so the auxiliary-loss based balancing must be turned off "
                    "explicitly. Please set moe_router_load_balancing_type='none', "
                    f"but got {self.routing_type!r}."
                )
            if self.config.router_aux_loss_coef:
                raise ValueError(
                    "quantile_balancing is a self-contained load balancing method. "
                    "A non-zero router_aux_loss_coef keeps the auxiliary load "
                    "balancing loss active and optimizes a second, competing "
                    "balancing objective. Please set router_aux_loss_coef=0 when "
                    "using quantile_balancing, but got "
                    f"{self.config.router_aux_loss_coef!r}."
                )
            if self.n_group != 1:
                raise ValueError(
                    "Quantile Balancing currently only supports n_group=1. "
                    "Multi-group routing (n_group>1) is not compatible with QB because "
                    "the group pre-selection changes the effective cutoff in a way that "
                    "cannot be captured by a single per-expert histogram. "
                    f"Got n_group={self.n_group}."
                )
            # Bias vector -- same name as noaux_tc for checkpoint compatibility
            if not self.config.gpt_model_use_experimental_version:
                self.register_buffer(
                    "e_score_correction_bias",
                    paddle.zeros((self.num_experts,), dtype=paddle.float32),
                )
            else:
                self.register_buffer(
                    "e_score_correction_bias",
                    paddle.zeros((1, self.num_experts), dtype=paddle.float32),
                )
            self._cast_to_low_precision = False

            # Histogram accumulator: [n_experts, B]
            self.qb_n_bins = getattr(config, "qb_n_bins", 1000)
            self.qb_histogram = paddle.zeros(
                shape=[self.num_experts, self.qb_n_bins],
                dtype=paddle.int32,
            )
            self.qb_histogram.stop_gradient = True

            # Expert usage for logging/diagnostics
            self.expert_usage = paddle.zeros(
                shape=[self.num_experts],
                dtype=paddle.int64,
            )
            self.expert_usage.stop_gradient = True

            # QB Binning range -- persisted because it affects the next step's
            # histogram and therefore must survive checkpoint resumption.
            self.register_buffer(
                "qb_bin_min", paddle.to_tensor(-1.0, dtype=paddle.float32)
            )
            self.register_buffer(
                "qb_bin_max", paddle.to_tensor(1.0, dtype=paddle.float32)
            )

        # Hash-routing state. Activated lazily via set_layer_number() so that the
        # router knows its layer index.
        self.is_hash_layer = False
        self.tid2eid = None

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
            elif scoring_func == "sftplus":
                scores = F.softplus(logits)
            elif scoring_func == "sqrtsoftplus":
                scores = paddle.sqrt(F.softplus(logits) + 1e-20)
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

    def _cal_seq_aux_loss(
        self,
        probs,
        top_k,
        routing_map,
        seq_len,
        batch_size,
        input_ids=None,
        origin_input_ids=None,
    ):
        if self.use_accuracy_compatible:
            _probs_2d_pf = (
                probs
                if probs.dim() == 2
                else probs.reshape([-1, probs.shape[-1]])
            )
            _aux_top_idx_pf = paddle.topk(
                _probs_2d_pf, k=top_k, axis=-1
            ).indices.cast("int64")
            _aux_routing_map_pf = paddle.zeros_like(
                _probs_2d_pf
            ).put_along_axis_(
                _aux_top_idx_pf,
                paddle.to_tensor(1.0, dtype=_probs_2d_pf.dtype),
                axis=-1,
            )
            _aux_routing_map_pf = _aux_routing_map_pf.reshape(routing_map.shape)
            # Mask out padding/invalid tokens (rows where original routing_map is all-zero)
            row_mask = (
                routing_map.cast("int64").sum(axis=-1, keepdim=True) > 0
            ).cast(_aux_routing_map_pf.dtype)
            routing_map = _aux_routing_map_pf * row_mask

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
                all_probs = ContextParallelAllGatherOp.apply(
                    all_probs, axis=1, mode=self.config.cp_balance_mode
                )
                local_seq_len = local_seq_len * self.context_parallel_size
            else:
                # [B, S, E]
                all_probs = all_probs.reshape(
                    [-1, local_seq_len, self.num_experts]
                )
            batch_size = all_probs.shape[0]
            # [B, S, E]: align with EC by GatherOp + split on routing_map
            if self.sequence_parallel and self.tensor_model_parallel_size > 1:
                tp_group = get_tensor_model_parallel_group()
                routing_map_gathered = GatherOpLegacy.apply(
                    routing_map, 0, tp_group
                ).reshape(
                    [
                        -1,
                        seq_len * self.tensor_model_parallel_size,
                        routing_map.shape[-1],
                    ]
                )
                routing_map = paddle.split(
                    routing_map_gathered,
                    num_or_sections=self.tensor_model_parallel_size,
                    axis=1,
                )[tp_group.rank]
            else:
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
        # Align with EC: use per-line valid token count as denominator instead of
        # fixed max_seq_len. PF's input_ids plays the role of EC's origin_input_ids.
        # [B, 1]
        if input_ids is not None:
            if (
                self.config.sequence_parallel
                and self.config.experimental_dataflow
            ):
                # input_ids [b, s/(cp*tp)] -> gather seq dim -> [b, s/cp]
                b, s = input_ids.shape
                input_ids = AllGatherOp.apply(input_ids.reshape([-1])).reshape(
                    [b, -1]
                )
            if (
                get_context_parallel_world_size() > 1
                and self.config.experimental_dataflow
            ):
                # In EB data flow, we need to gather input_ids here to get right denom.
                input_ids = ContextParallelGatherOp.apply(
                    input_ids, axis=1, mode=self.config.cp_balance_mode
                )
            _ids = input_ids
            if _ids.ndim == 1:
                _ids = _ids.unsqueeze(axis=0)
            pad_token_id = getattr(self.config, "pad_token_id", 0)
            if pad_token_id is None:
                pad_token_id = 0
            origin_valid_mask = (_ids != pad_token_id).astype(paddle.float32)
            if getattr(
                self.config, "gpt_model_use_experimental_version", False
            ):
                if origin_input_ids is not None:
                    # origin_input_ids is the full un-scattered ids (already
                    # includes MTP-shifted tokens); no AllGather/CP gather and
                    # no additional num_nextn_predict_layers offset.
                    _origin_ids = origin_input_ids
                    if _origin_ids.ndim == 1:
                        _origin_ids = _origin_ids.unsqueeze(axis=0)
                    origin_valid_mask_for_count = (
                        _origin_ids != pad_token_id
                    ).astype(paddle.float32)
                    token_count_per_line = origin_valid_mask_for_count.sum(
                        axis=-1, keepdim=True
                    )
                else:
                    token_count_per_line = origin_valid_mask.sum()
            else:
                token_count_per_line = origin_valid_mask.sum(
                    axis=-1, keepdim=True
                )
            is_invalid_line_float = (token_count_per_line == 0).astype(
                paddle.float32
            )
            denom = token_count_per_line + 1e-6 * is_invalid_line_float
        else:
            denom = paddle.to_tensor(float(max_seq_len), dtype="float32")

        if getattr(self.config, "gpt_model_use_experimental_version", False):
            # Align with ernie: divide by S first, then multiply by E/K (two-step to match float order)
            # [B, E]
            cost_coeff = (
                routing_map.sum(axis=seq_axis, dtype="float32")
                / denom
                * paddle.to_tensor(
                    float(self.num_experts) / top_k, dtype="float32"
                )
            )
            # Align with ernie: use mean instead of sum/S
            # [B, E] -> [B] -> []
            seq_aux_loss = (
                (cost_coeff * all_probs.mean(axis=seq_axis)).sum(axis=1).mean()
            )
        else:
            # [B, E]
            if self.use_accuracy_compatible:
                tokens_per_expert = routing_map.sum(
                    axis=seq_axis, dtype="float32"
                )  # [B, E]
                _aggregated = all_probs.sum(axis=seq_axis)  # [B, E]
                _per_expert = _aggregated * tokens_per_expert  # [B, E]
                _bsz = _per_expert.shape[0]
                # MG get_tokens_per_expert_and_token_count():
                #   total_num_tokens = tokens_per_expert.sum() / (topk * bsz)
                # i.e. the number of *valid routing* tokens per line (padding
                # rows contribute no routing entry). Without padding this is
                # exactly max_seq_len, so the no-padding path stays bit-exact.
                # Kept as a Python scalar (single division) to match MG's
                # scalar arithmetic instead of a multi-kernel tensor path.
                _total_num_tokens = float(
                    tokens_per_expert.sum().item()
                ) / float(top_k * _bsz)
                if _total_num_tokens <= 0.0:
                    # Every line is padding: no routed token, no loss.
                    return _per_expert.sum() * 0.0
                _scalar = float(self.num_experts) / (
                    float(top_k) * _total_num_tokens * _total_num_tokens
                )
                seq_aux_loss = _per_expert.sum() * _scalar
                # MG divides the sequence-level loss by bsz; keep the
                # single-line case free of the extra op (bit-exact).
                if _bsz > 1:
                    seq_aux_loss = seq_aux_loss / float(_bsz)
            else:
                cost_coeff = routing_map.sum(axis=seq_axis, dtype="float32") / (
                    denom
                    * paddle.to_tensor(
                        top_k / self.num_experts, dtype="float32"
                    )
                )
                # [B, E] -> [B] -> []
                seq_aux_loss = (
                    (cost_coeff * all_probs.sum(axis=seq_axis) / denom)
                    .sum(axis=1)
                    .mean()
                )
        return seq_aux_loss

    def _cal_z_loss(
        self, logits, input_ids=None, origin_input_ids=None
    ) -> paddle.Tensor:
        """
        Calculate the z loss.

        Args:
            logits (paddle.Tensor): Model output. The shape is [batch_size, num_experts].
            input_ids (paddle.Tensor, optional): Input token ids used to compute loss mask.

        Returns:
            paddle.Tensor: The z loss value.
        """
        if input_ids is not None:
            gathered_input_ids = input_ids
            if (
                self.config.sequence_parallel
                and self.config.experimental_dataflow
            ):
                # input_ids [b, s/(cp*tp)] -> gather seq dim -> [b, s/cp]
                b, s = input_ids.shape
                gathered_input_ids = AllGatherOp.apply(
                    gathered_input_ids.reshape([-1])
                ).reshape([b, -1])
            if (
                get_context_parallel_world_size() > 1
                and self.config.experimental_dataflow
            ):
                # In EB data flow, we need to gather input_ids here to get right denom.
                gathered_input_ids = ContextParallelGatherOp.apply(
                    gathered_input_ids, axis=1, mode=self.config.cp_balance_mode
                )

            pad_token_id = getattr(self.config, "pad_token_id", 0)
            if pad_token_id is None:
                pad_token_id = 0

            if getattr(
                self.config, "gpt_model_use_experimental_version", False
            ) and (origin_input_ids is not None):
                origin_loss_mask = (origin_input_ids != pad_token_id).astype(
                    paddle.float32
                )
            else:
                origin_loss_mask = (gathered_input_ids != pad_token_id).astype(
                    paddle.float32
                )
            loss_mask = (input_ids != pad_token_id).astype(paddle.float32)
            loss_mask = loss_mask.reshape([-1])
            denom = origin_loss_mask.sum()

            l_zloss = (
                logits.logsumexp(1).square() * loss_mask
            ).sum() / paddle.clip(denom, min=1e-6)
        else:
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
        if not self.config.gpt_model_use_experimental_version:
            scores_for_choice = scores.reshape(
                [bsz_seq_len, -1]
            ) + self.e_score_correction_bias.detach().unsqueeze(0)
        else:
            scores_for_choice = (
                scores.reshape([bsz_seq_len, -1])
                + self.e_score_correction_bias.detach()
            )
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
        if self.use_accuracy_compatible:
            row_idx = paddle.arange(
                bsz_seq_len, dtype=topk_idx.dtype
            ).unsqueeze(-1)
            row_idx = row_idx.expand(topk_idx.shape)
            gather_idx = paddle.stack([row_idx, topk_idx], axis=-1)
            topk_weight = paddle.gather_nd(scores, gather_idx)
        else:
            topk_weight = scores.take_along_axis(topk_idx, axis=1)

        return topk_weight, topk_idx

    def _topk_quantile_balancing(
        self,
        scores: paddle.Tensor,
        k: int,
        n_group: int,
        topk_group: int,
        valid_mask: paddle.Tensor | None = None,
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        """Quantile Balancing top-k selection.

        Same routing logic as noaux_tc: bias affects only expert selection,
        not gate weights. Additionally accumulates per-expert required_bias
        into a histogram for the QB callback to recover quantile-based bias.

        Args:
            scores (paddle.Tensor): [bsz*seq_len, n_experts] raw gating scores
            k (int): number of experts to select per token
            n_group (int): number of expert groups
            topk_group (int): number of groups selected
            valid_mask (paddle.Tensor | None): [bsz*seq_len, 1] mask marking
                non-padding tokens. Padding rows must not enter the histogram:
                their scores have been zeroed out by the caller, so they would
                all collapse into a single bin and skew the recovered quantile.
                None means no padding information is available and every row is
                treated as a valid token.

        Returns:
            tuple[paddle.Tensor, paddle.Tensor]: topk_weight, topk_idx
            topk_weight: [bsz*seq_len, k] -- gate weights from ORIGINAL scores
            topk_idx: [bsz*seq_len, k] -- selected expert indices
        """
        bsz_seq_len, n_experts = scores.shape
        if n_group != 1:
            raise ValueError(
                "Quantile Balancing currently only supports n_group=1. "
                "Multi-group routing (n_group>1) is not compatible with QB because "
                "the group pre-selection changes the effective cutoff in a way that "
                "cannot be captured by a single per-expert histogram. "
                f"Got n_group={n_group}."
            )

        assert self.e_score_correction_bias is not None, (
            "e_score_correction_bias is None for quantile_balancing"
        )

        # Step 1: Add bias for selection (detached, no gradient)
        if not self.config.gpt_model_use_experimental_version:
            scores_for_choice = scores.reshape(
                [bsz_seq_len, -1]
            ) + self.e_score_correction_bias.detach().unsqueeze(0)
        else:
            scores_for_choice = (
                scores.reshape([bsz_seq_len, -1])
                + self.e_score_correction_bias.detach()
            )

        # Step 2: Top-k selection (n_group=1, straightforward topk)
        topk_weight, topk_idx = paddle.topk(
            scores_for_choice, k=k, axis=-1, sorted=True
        )

        # Step 3: Gate weight from ORIGINAL scores (bias not in gate)
        topk_weight = scores.take_along_axis(topk_idx, axis=1)

        # Step 4: Accumulate histogram for QB (only during training)
        if framework._dygraph_tracer()._has_grad:
            self._accumulate_qb_histogram(
                scores, scores_for_choice, k, valid_mask
            )

        return topk_weight, topk_idx

    @paddle.no_grad()
    def _accumulate_qb_histogram(
        self,
        raw_scores: paddle.Tensor,
        biased_scores: paddle.Tensor,
        k: int,
        valid_mask: paddle.Tensor | None = None,
        alpha: paddle.Tensor | None = None,
    ):
        """Accumulate required_bias into the QB histogram.

        For each token t and expert e:
            alpha_t = (k+1)-th largest biased score for token t (the cutoff)
            required_bias[t, e] = alpha_t - raw_scores[t, e]

        We bin required_bias into self.qb_histogram[e, bin_idx].

        Padding tokens must be excluded. The caller zeroes out the gating
        scores of padding rows, so for such a row alpha is the (k+1)-th largest
        bias and required_bias equals that same constant for *every* expert:
        each padding token would add one count to one identical bin of all
        experts, inflating the per-expert total (and therefore the target
        quantile) and spiking the CDF exactly where it is read out.

        Args:
            raw_scores: [N, E] original gating scores (without bias)
            biased_scores: [N, E] scores with bias added (used for cutoff)
            k: top-k value
            valid_mask: [N, 1] mask marking non-padding tokens, or None when no
                padding information is available (all rows count).
            alpha: [N] or [N, 1] pre-computed cutoff values from MoETopkFusion
                kernel (optional). When provided, skips the internal topk
                computation for cutoff.
        """
        N, E = raw_scores.shape
        B = self.qb_n_bins

        # Compute alpha: the (k+1)-th largest biased score per token
        # This is the "cutoff" -- the highest biased score NOT selected
        if alpha is not None:
            # Alpha provided externally (e.g., from MoETopkFusion kernel)
            if alpha.ndim == 1:
                alpha = alpha.unsqueeze(-1)  # [N] -> [N, 1]
        else:
            # Clamp k+1 to at most E (in case of degenerate config)
            topk_val = min(k + 1, int(E))
            alpha = paddle.topk(
                biased_scores, k=topk_val, axis=-1, sorted=True
            )[0][:, -1:]  # [N, 1] -- the smallest of top-(k+1), i.e., cutoff

        # required_bias[t, e] = alpha[t] - raw_scores[t, e]
        required_bias = alpha - raw_scores  # [N, E]

        # Bin into histogram
        b_min = self.qb_bin_min
        b_max = self.qb_bin_max
        total_range = b_max - b_min
        if total_range < 1e-8:
            total_range = 2.0  # fallback for first step when bias is all-zero

        # Quantize to bin index: [0, B-1]
        # bin_idx = floor((required_bias - b_min) / total_range * B)
        bin_idx = ((required_bias - b_min) / total_range * B).cast(paddle.int64)
        bin_idx = paddle.clip(bin_idx, min=0, max=B - 1)

        # Vectorized histogram accumulation:
        # Offset each expert's bins by e*B, then do a single flat bincount
        offsets = (
            paddle.arange(E, dtype=paddle.int64).unsqueeze(0) * B
        )  # [1, E]
        flat_bins = (bin_idx + offsets).reshape([-1])  # [N*E]
        counts = paddle.zeros([E * B], dtype=paddle.int32)
        if valid_mask is None:
            weights = paddle.ones([N * E], dtype=paddle.int32)
        else:
            # [N,1] -> [N,E] -> [N*E]: padding rows carry weight 0, so their
            # bins receive += 0. Keeping the shape static (instead of selecting
            # valid rows) avoids a data-dependent N and an empty-tensor case.
            weights = (
                valid_mask.reshape([N, 1])
                .cast(paddle.int32)
                .expand([N, E])
                .reshape([-1])
            )
        counts.put_along_axis_(flat_bins, weights, axis=0, reduce="add")
        self.qb_histogram += counts.reshape([E, B])

        # Also accumulate expert_usage for diagnostics
        # Count how many tokens selected each expert (from topk_idx)
        # This is done separately in forward via exp_counts, so skip here

    def _hash_routing(
        self,
        logits: paddle.Tensor,
        flat_ids: paddle.Tensor,
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        """Hash-based routing: expert indices come from the tid2eid lookup table.

        Scores are still computed from the gating logits for weight computation,
        but expert selection is determined by the pre-computed hash table.

        Aligned with the upstream hash-routing reference implementation
        (``TopKRouter._hash_routing``).

        Args:
            logits (paddle.Tensor): Gating logits, shape [num_tokens, num_experts].
            flat_ids (paddle.Tensor): Token IDs flattened to match the row order
                of ``logits``. Shape [num_tokens], dtype int64.

        Returns:
            top_gate (paddle.Tensor): Per-token weights for the selected experts,
                shape [num_tokens, topk]. Already normalized for non-softmax
                score functions.
            top_idx (paddle.Tensor): Selected expert indices, shape
                [num_tokens, topk], dtype int64.
        """
        if self.tid2eid is None:
            raise ValueError(
                "tid2eid buffer is not registered; hash routing is not initialized."
            )
        score_function = self.scoring_func
        orig_dtype = logits.dtype
        logits_fp32 = logits.cast("float32")
        if score_function == "softmax":
            scores = F.softmax(logits_fp32, axis=-1).cast(orig_dtype)
        elif score_function == "sigmoid":
            scores = F.sigmoid(logits_fp32).cast(orig_dtype)
        elif score_function == "sqrtsoftplus":
            scores = paddle.sqrt(F.softplus(logits_fp32) + 1e-20).cast(
                orig_dtype
            )
        else:
            raise ValueError(
                f"Unsupported scoring_func in hash routing: {score_function!r}"
            )

        top_idx = self.tid2eid[flat_ids].cast(paddle.int64)  # [N, topk]
        top_gate = paddle.take_along_axis(scores, top_idx, axis=1)  # [N, topk]
        if score_function != "softmax":
            top_gate = top_gate / (top_gate.sum(axis=-1, keepdim=True) + 1e-20)

        # Apply routed_scaling_factor to the gathered top_gate.
        # Mirrors the non-hash path (see forward(): routed_scaling_factor[_learnable]
        # is multiplied onto top_gate after normalization).
        if self.routed_scaling_factor_learnable:
            top_gate = apply_learnable_routed_scaling(
                top_gate, top_idx, self.routed_scaling_factor_param
            )
        elif abs(self.routed_scaling_factor - 1.0) > 1e-6:
            top_gate = top_gate * self.routed_scaling_factor

        return top_gate, top_idx

    def _call_topk_method(
        self,
        topk_method,
        gates,
        k,
        n_group=None,
        topk_group=None,
        valid_mask=None,
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
        elif topk_method == "quantile_balancing":
            top_gate, top_idx = self._topk_quantile_balancing(
                gates,
                k,
                n_group,
                topk_group,
                valid_mask=valid_mask,
            )
        else:
            raise NotImplementedError(f"Invalid topk_method: {topk_method}")
        return top_gate, top_idx

    def set_layer_number(self, layer_number, is_mtp_layer: bool = False):
        self.layer_number = layer_number
        self._setup_hash_layer(layer_number, is_mtp_layer)

    def _setup_hash_layer(self, layer_number, is_mtp_layer: bool = False):
        """Activate hash routing for this layer if it falls in the hash range.

        Activation condition (0-indexed layer_number):
            is_hash_layer = (
                not is_mtp_layer
                and moe_n_hash_layers > 0
                and layer_number < moe_n_hash_layers
            )
        i.e. the first ``moe_n_hash_layers`` MoE layers use hash routing.

        Side effects on hash layers:
        - Registers the ``tid2eid`` buffer (round-robin placeholder; production
          deployments are expected to load a pretrained tid2eid from checkpoint).
        - Validates ``scoring_func`` and ``actual_vocab_size``.
        - Disables expert-bias state (e_score_correction_bias / expert_usage)
          on hash layers.
        """
        n_hash = getattr(self.config, "moe_n_hash_layers", 0)
        head_empty_layers = getattr(
            self.config, "num_empty_layers_add_in_head", 0
        )
        logical_layer_number = (
            None if layer_number is None else layer_number - head_empty_layers
        )
        self.is_hash_layer = (
            not is_mtp_layer
            and n_hash > 0
            and logical_layer_number is not None
            and 0 <= logical_layer_number < n_hash
        )
        # Enforce the split-feature routing contract now that is_hash_layer is
        # known. Split routing only applies to non-hash layers (hash layers
        # bypass it and may legitimately use a non-sigmoid scoring_func), so
        # validate scoring_func only on layers that will run the two-view
        # sigmoid path.
        if self.moe_split_feature_routing and not self.is_hash_layer:
            if self.scoring_func != "sigmoid":
                raise ValueError(
                    "moe_split_feature_routing requires scoring_func "
                    f"== 'sigmoid', but got {self.scoring_func!r}."
                )
        if not self.is_hash_layer:
            return

        if self.scoring_func not in ("softmax", "sigmoid", "sqrtsoftplus"):
            raise ValueError(
                f"Hash routing requires scoring_func in "
                f"{{'softmax', 'sigmoid', 'sqrtsoftplus'}}, got "
                f"{self.scoring_func!r}."
            )
        vocab_size = getattr(self.config, "actual_vocab_size", None)
        if vocab_size is None:
            raise ValueError(
                "actual_vocab_size must be set when moe_n_hash_layers > 0; "
                "it is required to allocate the tid2eid lookup buffer."
            )

        # Production deployments load a pretrained tid2eid table from the
        # inference checkpoint; no public initialization recipe is documented.
        # Round-robin is used here only as a placeholder so the layer is
        # runnable from scratch.
        ids = paddle.arange(vocab_size, dtype=paddle.int64)
        tid2eid = paddle.stack(
            [
                (ids + k) % self.num_experts
                for k in range(self.num_experts_per_tok)
            ],
            axis=1,
        )
        # Replace the placeholder attribute with a registered buffer.
        if hasattr(self, "tid2eid"):
            del self.tid2eid
        self.register_buffer("tid2eid", tid2eid)

        # Hash layers do not participate in expert-bias correction: drop the
        # buffers allocated under ``topk_method == 'noaux_tc'`` in __init__.
        # ``del self.<name>`` goes through ``paddle.nn.Layer.__delattr__``,
        # which removes the entry from both ``_buffers`` and
        # ``_non_persistable_buffer_names_set`` for registered buffers.
        if hasattr(self, "e_score_correction_bias"):
            del self.e_score_correction_bias
        if hasattr(self, "expert_usage"):
            del self.expert_usage

        # Hash layers bypass split-feature routing (see forward's ``use_split``
        # guard), so the second-view gate created in __init__ is never used
        # here. Drop it to avoid registering an unused parameter.
        if hasattr(self, "weight_1"):
            del self.weight_1


class TopKRouter(StandardMoERouter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._layer_number = None

    def set_layer_number(self, layer_number, is_mtp_layer: bool = False):
        self._layer_number = layer_number
        self.layer_number = layer_number
        self._setup_hash_layer(layer_number, is_mtp_layer=is_mtp_layer)

    def forward(self, input, input_ids=None, origin_input_ids=None):
        if len(input.shape) == 3:
            if not self.sequence_parallel:
                batch_size, seq_len, d_model = input.shape
            else:
                seq_len, batch_size, d_model = input.shape
            input = input.reshape([-1, d_model])
            if (
                get_context_parallel_world_size() > 1
                and self.config.experimental_dataflow
            ):
                # In EB dataflow, shape of input_ids [b, s],
                # but shape of input is [b, s/cp, h] ([s/cp, b, h] in sp),
                # so we need to scatter input_ids here to avid the assertion below
                input_ids = ContextParallelScatterOp.apply(
                    input_ids, axis=1, mode=self.config.cp_balance_mode
                )
            elif (
                get_context_parallel_world_size() > 1
                and getattr(self.config, "use_erndata", False)
                and input_ids is not None
                and input_ids.shape[1] != seq_len
            ):
                # erndata MTP path: PaddleFleet dataloader broadcasts
                # input_ids full-length [B, L] to every CP rank (unlike
                # experimental_dataflow which pre-scatters). Embedding was
                # already zigzag-sliced to [B, L/cp, H] via
                # extract_local_zigzag_chunks (see gpt_embedding.py). Slice
                # input_ids to the matching zigzag chunks here — no comm
                # needed since every rank holds the same [B, L] tensor.
                from paddlefleet.transformer.multi_token_prediction import (
                    extract_local_zigzag_chunks,
                )

                _cp_size = get_context_parallel_world_size()
                _cp_rank = get_context_parallel_rank()
                input_ids = extract_local_zigzag_chunks(
                    input_ids, _cp_rank, _cp_size, axis=1
                )
            if input_ids is not None:
                pad_token_id = getattr(self.config, "pad_token_id", 0)
                if pad_token_id is None:
                    pad_token_id = 0
                if self.sequence_parallel:
                    input_ids_none_zero_mask = (
                        (input_ids != pad_token_id)
                        .transpose([1, 0])
                        .reshape([-1, 1])
                    )
                else:
                    input_ids_none_zero_mask = (
                        input_ids != pad_token_id
                    ).reshape([-1, 1])
                batch_size_, seq_len_ = input_ids.shape
                assert (batch_size_ == batch_size) and (seq_len_ == seq_len), (
                    f"input_ids shape mismatch with input: "
                    f"input_ids=[{batch_size_}, {seq_len_}], "
                    f"expected [batch_size={batch_size}, seq_len={seq_len}]"
                )
            else:
                input_ids_none_zero_mask = None
        elif len(input.shape) == 2:
            if not self.config.gpt_model_use_experimental_version:
                raise ValueError(
                    "input must be a 3D tensor when "
                    "gpt_model_use_experimental_version=True."
                )
            cp_size = (
                max(get_context_parallel_world_size(), 1)
                if getattr(self.config, "experimental_dataflow", False)
                else 1
            )
            tp_size = (
                self.tensor_model_parallel_size if self.sequence_parallel else 1
            )
            seq_len = self.config.max_sequence_length // (cp_size * tp_size)
            batch_size = input.shape[0] // seq_len
            if (
                max(get_context_parallel_world_size(), 1) > 1
                and self.config.experimental_dataflow
                and input_ids is not None
            ):
                # In EB dataflow, shape of input_ids [b, s],
                # but shape of input is [b, s/cp, h] ([s/cp, b, h] in sp),
                # so we need to scatter input_ids here to avid the assertion below
                input_ids = ContextParallelScatterOp.apply(
                    input_ids, axis=1, mode=self.config.cp_balance_mode
                )
            if (
                input_ids is not None
                and self.sequence_parallel
                and self.config.experimental_dataflow
            ):
                # SP: input_ids [b, s/cp] -> [b, s/(cp*tp)]
                b, s = input_ids.shape
                input_ids = ScatterOp.apply(input_ids.reshape([-1])).reshape(
                    [b, -1]
                )
            if input_ids is not None:
                pad_token_id = getattr(self.config, "pad_token_id", 0)
                if pad_token_id is None:
                    pad_token_id = 0
                input_ids_none_zero_mask = (input_ids != pad_token_id).reshape(
                    [-1, 1]
                )
                batch_size_, seq_len_ = input_ids.shape
                assert (batch_size_ == batch_size) and (seq_len_ == seq_len), (
                    f"input_ids shape mismatch with input: "
                    f"input_ids=[{batch_size_}, {seq_len_}], "
                    f"expected [batch_size={batch_size}, seq_len={seq_len}]"
                )
            else:
                input_ids_none_zero_mask = None

        # Hash routing requires input_ids; verify early.
        if self.is_hash_layer and input_ids is None:
            raise ValueError(
                "Hash routing (moe_n_hash_layers > 0) requires input_ids. "
                "Make sure input_ids is passed through the model forward "
                "to the MoE layer."
            )

        # Split-feature routing applies to non-hash layers only; hash-routing
        # layers keep using the original single gate projection.
        use_split = self.moe_split_feature_routing and not self.is_hash_layer

        with paddle.amp.auto_cast(False):
            if use_split:
                # The two-view contract is sigmoid + sigmoid. Re-assert it here
                # at the point of use: set_layer_number() validates scoring_func
                # early, but if the router is invoked before set_layer_number()
                # (is_hash_layer still at its __init__ default of False), this
                # guard prevents the split branch from running under a
                # non-sigmoid scoring_func.
                if self.scoring_func != "sigmoid":
                    raise ValueError(
                        "moe_split_feature_routing requires scoring_func "
                        f"== 'sigmoid', but got {self.scoring_func!r}."
                    )
                # Two independent views; the routing score is the SUM of their
                # per-expert scores. View 0 reuses the existing self.weight
                # gate, view 1 uses the new self.weight_1 projection. Both
                # reuse the fused gate matmul so they share the
                # force-load-balancing and defer_dw paths.
                logits_0 = gate_detach_matmul(
                    input,
                    self.weight,
                    True,
                    self.config.moe_router_force_load_balancing,
                    dw_overlap_enabled(self.config, "moe_router_gate"),
                    self.use_accuracy_compatible,
                )
                logits_1 = gate_detach_matmul(
                    input,
                    self.weight_1,
                    True,
                    self.config.moe_router_force_load_balancing,
                    dw_overlap_enabled(self.config, "moe_router_gate"),
                    self.use_accuracy_compatible,
                )
                # The two-view contract is sigmoid + sigmoid. scoring_func is
                # guaranteed to be "sigmoid" here (validated above and in
                # set_layer_number), so route both views through the shared
                # gate_score_func instead of hardcoding F.sigmoid.
                gates = self.gate_score_func(logits_0) + self.gate_score_func(
                    logits_1
                )
                logits = logits_0 + logits_1  # used by z-loss
            else:
                logits = gate_detach_matmul(
                    input,
                    self.weight,
                    True,
                    self.config.moe_router_force_load_balancing,
                    dw_overlap_enabled(self.config, "moe_router_gate"),
                    self.use_accuracy_compatible,
                )

        _log_moe_md5(logits, "gate_logits", self._layer_number)

        # ---- Hash routing branch ----
        if self.is_hash_layer:
            if self.sequence_parallel:
                flat_ids = (
                    input_ids.transpose([1, 0]).reshape([-1]).cast(paddle.int64)
                )
            else:
                flat_ids = input_ids.reshape([-1]).cast(paddle.int64)

            top_gate, top_idx = self._hash_routing(logits, flat_ids)

            # Build full [num_tokens, num_experts] probs and routing mask.
            probs = paddle.zeros_like(logits).put_along_axis(
                top_idx, top_gate.cast(logits.dtype), axis=1
            )
            mask = (probs > 0).cast(logits.dtype)

            # Apply padding (input_ids == 0):
            # routing_map = routing_map & ~padding_mask.
            # input_ids_none_zero_mask shape: [N, 1], broadcast over expert/topk dim.
            if input_ids_none_zero_mask is not None:
                valid_mask = input_ids_none_zero_mask.cast(mask.dtype)
                mask = mask * valid_mask
                probs = probs * valid_mask
                top_gate = top_gate * valid_mask
                top_idx = top_idx.masked_fill(~valid_mask.cast(paddle.bool), -1)

            _log_moe_md5(
                top_idx.cast(paddle.float32),
                "hash_topk_indices",
                self._layer_number,
            )
            # No aux/z loss, no expert-bias updates on hash layers.
            return (None, top_gate, top_idx, probs, mask, None, None, None)
        # ---- end hash routing ----

        # Split-feature routing already produced `gates` (sum of the two
        # score_func views) inside the auto_cast block above; only the
        # single-gate path needs the scoring function applied here. (Hash
        # layers have already returned above, so `use_split` here is equivalent
        # to `self.moe_split_feature_routing`.)
        if not use_split:
            gates = self.gate_score_func(logits)

        if input_ids_none_zero_mask is not None:
            # input_ids_none_zero_mask shape: [b*s,1]
            valid_mask = input_ids_none_zero_mask.astype(paddle.float32)
            assert valid_mask.shape[0] == logits.shape[0], (
                f"check valid_mask shape {valid_mask.shape}"
            )
            logits = logits * valid_mask
            gates = gates * valid_mask

        _log_moe_md5(gates, "gate_probs_sigmoid", self._layer_number)

        # Use clone() to ensure that the execution order of the grad nodes is consistent with EC.
        if self.use_accuracy_compatible and not use_split:
            gates_ori = self.gate_score_func(logits).cast(logits.dtype)
            if input_ids_none_zero_mask is not None:
                gates_ori = gates_ori * valid_mask
        else:
            gates_ori = gates.clone()
        if self.config.router_aux_loss_coef and self.scoring_func != "softmax":
            if not getattr(
                self.config, "gpt_model_use_experimental_version", False
            ):
                gates_ori = gates_ori / (
                    gates_ori.sum(axis=-1, keepdim=True) + 1e-20
                )
            else:
                # Use clip() to ensure the computation logic is consistent with EC; it may be useful when gradients are very small.
                gates_ori = gates_ori / paddle.clip(
                    gates_ori.sum(-1, keepdim=True), min=1e-12
                )

        if getattr(self.config, "moe_topk_fusion", False):
            # Use MoETopkFusion Triton kernel for fused topk selection.
            MoETopkFusion = _get_moe_topk_fusion()
            use_node_limit = self.n_group > 1
            if not self.config.gpt_model_use_experimental_version:
                probs_for_choice = (
                    gates + self.e_score_correction_bias.detach().unsqueeze(0)
                )
            else:
                probs_for_choice = gates + self.e_score_correction_bias.detach()
            if _LOG_LAYER_MD5 and self._layer_number == 0:
                _log_moe_md5(
                    self.e_score_correction_bias,
                    "e_score_correction_bias",
                    self._layer_number,
                )
                _log_moe_md5(
                    probs_for_choice, "probs_for_choice", self._layer_number
                )

            if self.topk_method == "quantile_balancing":
                # QB path: kernel does NOT normalize (normalization stays eager
                # for bit-exact alignment with original QB). Additionally returns
                # alpha (the k+1-th largest choice value) for histogram accumulation.
                top_gate, top_idx, alpha = MoETopkFusion.apply(
                    gates,  # gate_probs (original sigmoid scores)
                    probs_for_choice,  # probs_for_choice (with correction bias)
                    self.num_experts_per_tok,
                    use_node_limit,
                    self.n_group,
                    self.topk_group,
                    False,  # norm_gate_logits=False (normalization stays eager)
                    True,  # return_alpha=True
                )
                # Accumulate QB histogram with pre-computed alpha (skips internal topk)
                if framework._dygraph_tracer()._has_grad:
                    self._accumulate_qb_histogram(
                        gates,
                        probs_for_choice,
                        self.num_experts_per_tok,
                        valid_mask=input_ids_none_zero_mask,
                        alpha=alpha,
                    )
            else:
                # noaux_tc / other paths: kernel handles normalization
                top_gate, top_idx = MoETopkFusion.apply(
                    gates,  # gate_probs (original sigmoid scores)
                    probs_for_choice,  # probs_for_choice (with correction bias)
                    self.num_experts_per_tok,
                    use_node_limit,
                    self.n_group,
                    self.topk_group,
                    self.norm_topk_prob,  # norm_gate_logits
                )
            # top_gate is already normalized by the Triton kernel when norm_topk_prob=True (non-QB)

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
                # Only quantile_balancing consumes this: padding rows have
                # already been zeroed out above and must stay out of the QB
                # histogram.
                valid_mask=input_ids_none_zero_mask,
            )

            _log_moe_md5(
                top_idx.cast("float32"), "topk_indices", self._layer_number
            )
            _log_moe_md5(top_gate, "topk_weights_raw", self._layer_number)

        # z-loss
        if self.config.router_z_loss_coef:
            l_zloss = (
                self._cal_z_loss(logits, input_ids, origin_input_ids)
                * self.config.router_z_loss_coef
            )
        else:
            l_zloss = None

        if getattr(self.config, "routing_map_fusion", False):
            pad_token_id = getattr(self.config, "pad_token_id", 0)
            if pad_token_id is None:
                pad_token_id = 0
            mask, top_idx, exp_counts = _apply_routing_map_fusion(
                gates,
                top_idx,
                input_ids_none_zero_mask,
                input_ids,
                pad_token_id=pad_token_id,
            )
        else:
            with paddle.amp.auto_cast(enable=False):
                mask = paddle.zeros_like(gates).put_along_axis_(
                    top_idx, paddle.to_tensor(1.0, dtype=gates.dtype), axis=1
                )
            if input_ids_none_zero_mask is not None:
                valid_mask = input_ids_none_zero_mask
                mask = mask * valid_mask.cast(mask.dtype)
                # -1 means neither participates in routing nor expert calculation
                top_idx = top_idx.masked_fill(~valid_mask.cast(paddle.bool), -1)
            exp_counts = paddle.sum(mask.cast(paddle.int64), axis=0)

        # norm
        if self.norm_topk_prob:
            if (
                not getattr(self.config, "moe_topk_fusion", False)
                or self.topk_method == "quantile_balancing"
            ):
                # QB fusion path passes norm_gate_logits=False to the kernel,
                # so normalization must happen here in eager (for bit-exact alignment).
                # Non-fusion paths also normalize here.
                if self.use_accuracy_compatible:
                    _sum_f64 = top_gate.cast(paddle.float64).sum(
                        axis=-1, keepdim=True
                    )
                    denominator = _sum_f64.cast(paddle.float32) + 1e-20
                else:
                    denominator = top_gate.sum(axis=-1, keepdim=True) + 1e-20
                top_gate = top_gate / denominator
            # When moe_topk_fusion=True and not QB, top_gate is already normalized by MoETopkFusion

        if self.routed_scaling_factor_learnable:
            top_gate = apply_learnable_routed_scaling(
                top_gate, top_idx, self.routed_scaling_factor_param
            )
        elif abs(self.routed_scaling_factor - 1.0) > 1e-6:
            top_gate = top_gate * self.routed_scaling_factor

        # Reconstruct probs (combine weights in [S, E] sparse layout) from final top_gate.
        probs = paddle.zeros_like(gates, dtype=top_gate.dtype).put_along_axis_(
            top_idx, top_gate, axis=1
        )

        _log_moe_md5(probs, "probs", self._layer_number)
        _log_moe_md5(top_gate, "topk_weights_normed", self._layer_number)

        if (
            self.topk_method in ("noaux_tc", "quantile_balancing")
            and framework._dygraph_tracer()._has_grad
        ):
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
                    input_ids=input_ids,
                    origin_input_ids=origin_input_ids,
                )

            else:
                l_aux = self._cal_aux_loss(gates, mask)
        else:
            l_aux = None

        return (
            None,  # new capacity
            top_gate,  # weights of selected experts for each token [num_tokens, num_experts_per_token]
            top_idx,  # indices of selected experts for each token [num_tokens, num_experts_per_token]
            probs,  # combine weights in [S, E] sparse layout; non-selected positions are 0 [num_tokens, num_experts]
            mask,  # mask. for each token, the selected experts are marked with 1s [num_tokens, num_experts]
            None,  # token priority
            l_aux,
            l_zloss,
        )
