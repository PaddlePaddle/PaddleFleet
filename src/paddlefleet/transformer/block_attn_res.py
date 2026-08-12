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

"""Block Attention Residuals (Block AttnRes).

Implements the Block AttnRes mechanism from "Attention Residuals"
(Kimi Team, 2026). Replaces standard fixed-weight residual connections
with learned softmax attention over block-level representations.

Standard residuals accumulate with fixed unit weights:
    h_l = h_{l-1} + f_{l-1}(h_{l-1})

Block AttnRes partitions layers into N blocks, uses standard residual
accumulation within blocks, and applies softmax attention over
block-level representations across blocks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import paddle
from paddle import Tensor, nn
from paddle.distributed.fleet.meta_parallel import (
    LayerSpec,
    build_spec_layer,
)

from paddlefleet.transformer.identity_op import IdentityOp
from paddlefleet.transformer.layer import FleetLayer

from .paddle_norm import RMSNorm, get_norm_extra_args

try:
    from paddle.distributed.fleet.utils.sequence_parallel_utils import (
        mark_as_sequence_parallel_parameter,
    )
except ImportError:
    logging.warn("Fail to import mark_as_sequence_parallel_parameter!")

    def mark_as_sequence_parallel_parameter(parameter):
        return parameter


if TYPE_CHECKING:
    from paddlefleet.transformer.transformer_config import (
        TransformerConfig,
    )


@dataclass
class BlockAttnResSublayersSpec:
    norm: LayerSpec | type = IdentityOp


def _block_attn_res_rmsnorm(
    partial_block: Tensor,
    blocks: list[Tensor],
    proj_weight: Tensor,
    norm_weight: Tensor,
    norm_eps: float,
) -> Tensor:
    """Apply the official FP32 AttentionRes reduction for RMSNorm inputs."""
    with paddle.amp.auto_cast(enable=False):
        all_repr = [*blocks, partial_block]
        hidden_size = partial_block.shape[-1]
        values = paddle.stack(
            [
                representation.reshape([-1, hidden_size])
                for representation in all_repr
            ],
            axis=1,
        ).astype("float32")
        variance = values.pow(2).mean(axis=-1, keepdim=True)
        normalized = values * paddle.rsqrt(variance + norm_eps)
        score_weight = norm_weight.astype("float32") * proj_weight.astype(
            "float32"
        ).reshape([-1])
        scores = (normalized * score_weight).sum(axis=-1)
        probabilities = paddle.nn.functional.softmax(scores, axis=-1).unsqueeze(
            1
        )
        output = paddle.matmul(probabilities, values).squeeze(1)
    return output.reshape(partial_block.shape)


class BlockAttnResFunc(paddle.autograd.PyLayer):
    """Custom forward/backward for BlockAttnRes to save memory.

    Forward runs under no_grad and only saves inputs.
    Backward recomputes intermediate activations (norm outputs, logits, weights)
    one block at a time to minimize peak memory.
    """

    @staticmethod
    def forward(
        ctx, partial_block, proj_weight, norm_weight, norm_eps, *blocks
    ):
        # Save only inputs for backward recomputation
        ctx.save_for_backward(partial_block, proj_weight, norm_weight, *blocks)
        ctx.norm_eps = norm_eps
        ctx.num_blocks = len(blocks)

        # Compute forward without building autograd graph
        with paddle.no_grad():
            h = _block_attn_res_rmsnorm(
                partial_block,
                list(blocks),
                proj_weight,
                norm_weight,
                norm_eps,
            )

        return h

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensor()
        partial_block = saved[0]
        proj_weight = saved[1]
        norm_weight = saved[2]
        blocks = list(saved[3:])
        norm_eps = ctx.norm_eps

        all_repr = [*blocks, partial_block]
        num_repr = len(all_repr)

        # Determine which forward tensor inputs need gradients
        # Forward tensor arg order: partial_block, proj_weight, norm_weight, *blocks
        # (norm_eps is a non-tensor arg, handled via None below)
        all_forward_tensors = [partial_block, proj_weight, norm_weight, *blocks]
        needs_grad = [not t.stop_gradient for t in all_forward_tensors]

        # Recompute forward values needed for backward
        with paddle.enable_grad():
            repr_detached = []
            for r in all_repr:
                rd = r.detach()
                rd.stop_gradient = False
                repr_detached.append(rd)

            proj_weight_d = proj_weight.detach()
            proj_weight_d.stop_gradient = False
            norm_weight_d = norm_weight.detach()
            norm_weight_d.stop_gradient = False

            h = _block_attn_res_rmsnorm(
                repr_detached[-1],
                repr_detached[:-1],
                proj_weight_d,
                norm_weight_d,
                norm_eps,
            )

            # Compute gradients via autograd
            grad_targets = [*repr_detached, proj_weight_d, norm_weight_d]
            grads = paddle.autograd.grad(
                outputs=[h],
                inputs=grad_targets,
                grad_outputs=[grad_output],
            )

        # grads layout: [repr_0, ..., repr_N, proj_weight, norm_weight]
        # repr order is [*blocks, partial_block]
        grad_blocks_list = list(grads[: num_repr - 1])
        grad_partial_block = grads[num_repr - 1]
        grad_proj_weight = grads[num_repr]
        grad_norm_weight = grads[num_repr + 1]

        # Return order matches forward TENSOR args:
        # partial_block, proj_weight, norm_weight, *blocks
        # (norm_eps is a non-tensor arg — no gradient returned for it)
        result = []
        raw_grads = [
            grad_partial_block,
            grad_proj_weight,
            grad_norm_weight,
            *grad_blocks_list,
        ]
        for need, g in zip(needs_grad, raw_grads):
            result.append(g if need else None)

        # This PyLayer only supports first-order backward.  Release the saved
        # distributed activations as soon as their gradients are materialized;
        # otherwise the Python context can retain the AttentionRes graph until
        # interpreter shutdown, after NCCL communicators have already begun
        # teardown.
        ctx.container = ()
        return tuple(result)


class BlockAttnRes(FleetLayer):
    """Per-layer module for Block Attention Residuals."""

    def __init__(
        self,
        config: TransformerConfig,
        sublayers_spec: BlockAttnResSublayersSpec,
    ):
        super().__init__(config=config)
        self.hidden_size = config.hidden_size

        # TODO: check when this parameter should be
        # marked as sequence parallel param,
        # i.e., its gradient should be all-reduced.
        self.proj_weight = self.create_parameter(
            # Keep the Linear(hidden_size, 1) checkpoint layout.  The leading
            # singleton dimension broadcasts identically in the dot product,
            # while allowing AOA to load HF Attention Residual weights without
            # an unsupported squeeze/reshape operation.
            shape=[1, self.hidden_size],
            default_initializer=nn.initializer.Constant(0.0),
        )

        input_is_parallel = (
            True
            if self.config.tensor_model_parallel_size > 1
            and self.config.sequence_parallel
            else False
        )
        if input_is_parallel:
            mark_as_sequence_parallel_parameter(self.proj_weight)
        extra_args = get_norm_extra_args(
            sublayers_spec.norm,
            self.config,
            self.hidden_size,
            self.config.rms_norm_eps,
            input_is_parallel,
        )
        self.norm = build_spec_layer(sublayers_spec.norm, **extra_args)

        # BlockAttnResFunc (PyLayer) hardcodes RMSNorm math internally;
        # fall back to the standard autograd path for other norms.
        self._use_pylayer = isinstance(self.norm, RMSNorm)

    def forward(self, partial_block: Tensor, blocks: list[Tensor]) -> Tensor:
        """Compute Block Attention Residual.

        During training, uses BlockAttnResFunc (PyLayer) to avoid retaining
        intermediate activations (norm outputs, logits, weights) in the
        autograd graph — they are recomputed in the backward pass.

        Args:
            partial_block: Current in-progress block representation,
                shape [B, S, H].
            blocks: List of completed block representations.
                Each tensor has shape [B, S, H].
        Returns:
            Tensor of shape [B, S, H] — the attention-weighted
            combination of all block representations.
        """
        if self.training and self._use_pylayer:
            h = BlockAttnResFunc.apply(
                partial_block,
                self.proj_weight,
                self.norm.weight,
                self.norm.variance_epsilon,
                *blocks,
            )
        else:
            all_repr = [*blocks, partial_block]
            n = len(all_repr)

            logits_list = []
            for r in all_repr:
                normed = self.norm(r)
                logits_list.append((normed * self.proj_weight).sum(axis=-1))

            logits = paddle.stack(logits_list, axis=0)
            weights = paddle.nn.functional.softmax(logits, axis=0)

            h = weights[0].unsqueeze(-1) * all_repr[0]
            for i in range(1, n):
                h = h + weights[i].unsqueeze(-1) * all_repr[i]

        if partial_block is not None and h.dtype != partial_block.dtype:
            h = h.to(partial_block.dtype)

        return h


class OutputBlockAttnResPipe(paddle.nn.Layer):
    """Pipeline-compatible wrapper: run output BlockAttnRes BEFORE the final norm."""

    def __init__(
        self,
        config: TransformerConfig,
        sublayers_spec: BlockAttnResSublayersSpec,
    ):
        super().__init__()
        self.config = config
        self.block_attn_res = BlockAttnRes(config, sublayers_spec)

    def _mtp_active(self) -> bool:
        # Guard mirrors WrappedPaddleNormPipe.forward (paddle_norm.py) so that
        # this pipe splits under exactly the same conditions as the final norm.
        return (
            self.config.num_nextn_predict_layers is not None
            and self.config.num_nextn_predict_layers > 0
            and not self.config.mtp_load_weight_only
            and not (
                not self.config.gpt_model_use_experimental_version
                and self.config.enable_mtp_magic_send
            )
        )

    def forward(self, dict_args: dict):
        if not self.config.block_attention_residuals:
            return dict_args

        hidden_states = dict_args["hidden_states"]
        blocks = dict_args.get("blocks", []) or []

        if self._mtp_active():
            tensor_list = paddle.split(
                hidden_states, self.config.num_nextn_predict_layers + 1
            )
            main_hidden = self.block_attn_res(tensor_list[0], blocks)
            hidden_states = paddle.concat([main_hidden, *tensor_list[1:]])
        else:
            hidden_states = self.block_attn_res(hidden_states, blocks)

        rst = {**dict_args, "hidden_states": hidden_states}
        # Blocks have been consumed; downstream (final norm, LM head) no
        # longer needs them.
        rst.pop("blocks", None)
        return rst

    def build_schedule_node(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            ScheduleNode,
        )

        return ScheduleNode(self.forward, name="OutputBlockAttnResPipe")
