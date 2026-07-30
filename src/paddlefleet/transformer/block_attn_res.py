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

from .paddle_norm import get_norm_extra_args

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
            shape=[self.hidden_size],
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

    def forward(self, partial_block: Tensor, blocks: list[Tensor]) -> Tensor:
        """Compute Block Attention Residual."""
        all_repr = [*blocks, partial_block]
        n = len(all_repr)

        logits_list = []
        for r in all_repr:
            normed = self.norm(r)
            logits_list.append((normed * self.proj_weight).sum(axis=-1))

        # Stack logits only (small tensor: [N, B, S]), safe from alignment issue
        logits = paddle.stack(logits_list, axis=0)

        # Softmax over block dimension (axis=0)
        weights = paddle.nn.functional.softmax(logits, axis=0)

        # Weighted sum via for-loop
        h = weights[0].unsqueeze(-1) * all_repr[0]
        for i in range(1, n):
            h = h + weights[i].unsqueeze(-1) * all_repr[i]

        if partial_block is not None and h.dtype != partial_block.dtype:
            h = h.to(partial_block.dtype)

        return h
