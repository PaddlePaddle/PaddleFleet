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

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from paddlefleet.packed_seq_params import PackedSeqParams
    from paddlefleet.transformer.enums import AttnMaskType
    from paddlefleet.transformer.transformer_config import TransformerConfig

import paddle
from paddle import Tensor

from paddlefleet.fusions.fused_softmax import FusedScaleMaskSoftmax
from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.transformer.layer import FleetLayer
from paddlefleet.transformer.utils import (
    attention_mask_func,
    is_layer_window_attention,
)
from paddlefleet.utils import divide


class DotProductAttention(FleetLayer):
    """
    Region where selective activation recomputation is applied.
    This region is memory intensive but less compute intensive which
    makes activation checkpointing more efficient for LLMs (20B+).
    See Reducing Activation Recomputation in Large Transformer Models:
    https://arxiv.org/abs/2205.05198 for more details.

    We use the following notation:
     h: hidden size
     n: number of attention heads
     p: number of tensor model parallel partitions
     b: batch size
     s: sequence length
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float | None = None,
        softmax_scale: float | None = None,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection = None,
        **kwargs,
    ):
        super().__init__(config=config)

        self.config: TransformerConfig = config

        assert self.config.context_parallel_size == 1, (
            "Context parallelism is only supported by TEDotProductAttention!"
        )

        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type  # unused for now

        projection_size = (
            self.config.kv_channels * self.config.num_attention_heads
        )

        # Per attention head and per partition values.
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(
                required_pgs=["tp"]
            )
        else:
            assert hasattr(pg_collection, "tp"), (
                "DotProductAttention pg_collection must have tp process group"
            )

        world_size = (
            pg_collection.tp.world_size
            if pg_collection.tp is not None and pg_collection.tp.world_size >= 1
            else 1
        )
        self.hidden_size_per_partition = divide(projection_size, world_size)
        self.hidden_size_per_attention_head = divide(
            projection_size, config.num_attention_heads
        )
        self.num_attention_heads_per_partition = divide(
            self.config.num_attention_heads, world_size
        )
        self.num_query_groups_per_partition = divide(
            self.config.num_query_groups, world_size
        )

        coeff = None
        if softmax_scale is None:
            self.softmax_scale = 1.0 / math.sqrt(
                self.hidden_size_per_attention_head
            )
        else:
            self.softmax_scale = softmax_scale

        if self.config.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.softmax_scale /= coeff

        if is_layer_window_attention(
            self.config.window_size,
            self.config.window_attn_skip_freq,
            layer_number,
        ):
            window_size = self.config.window_size
        else:
            window_size = None

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            input_in_fp16=self.config.fp16,
            input_in_bf16=self.config.bf16,
            attn_mask_type=self.attn_mask_type,
            scaled_masked_softmax_fusion=self.config.masked_softmax_fusion,
            mask_func=attention_mask_func,
            softmax_in_fp32=self.config.attention_softmax_in_fp32,
            scale=coeff,
            window_size=window_size,
        )

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = paddle.nn.Dropout(
            self.config.attention_dropout
            if attention_dropout is None
            else attention_dropout
        )

        if self.config.softmax_type == "vanilla":
            self.softmax_offset = None
        elif self.config.softmax_type == "off-by-one":
            self.softmax_offset = paddle.zeros(
                self.num_attention_heads_per_partition
            )
        elif self.config.softmax_type == "learnable":
            self.register_parameter(
                "softmax_offset",
                paddle.nn.Parameter(
                    paddle.empty(
                        self.num_attention_heads_per_partition,
                        dtype=self.config.params_dtype,
                    )
                ),
            )
            if config.perform_initialization:
                self.softmax_offset = config.init_method(self.softmax_offset)
        else:
            raise ValueError("Softmax type not supported")

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType = None,
        attention_bias: Tensor = None,
        packed_seq_params: PackedSeqParams | None = None,
    ):
        """Forward."""
        assert packed_seq_params is None, (
            "Packed sequence is not supported by DotProductAttention."
            "Please use TEDotProductAttention instead."
        )
        assert attention_bias is None, (
            "Attention bias is not supported for DotProductAttention."
        )

        # ===================================
        # Raw attention scores. [b, n/p, s, s]
        # ===================================

        # expand the key and value [b, sk, ng, hn] -> [b, sk, np, hn]
        # This is a noop for normal attention where ng == np. When using group query attention this
        # creates a view that has the keys and values virtually repeated along their dimension to
        # match the number of queries.

        # attn_mask_type is not used.
        if (
            self.num_attention_heads_per_partition
            // self.num_query_groups_per_partition
            > 1
        ):
            key = key.repeat_interleave(
                self.num_attention_heads_per_partition
                // self.num_query_groups_per_partition,
                dim=2,
            )
            value = value.repeat_interleave(
                self.num_attention_heads_per_partition
                // self.num_query_groups_per_partition,
                dim=2,
            )

        # [b, np, sq, sk]
        output_size = (
            query.shape[0],
            query.shape[2],
            query.shape[1],
            key.shape[1],
        )

        # [b, sq, np, hn] -> [b * np, sq, hn]
        # This will be a simple view when doing normal attention, but in group query attention
        # the key and value tensors are repeated to match the queries so you can't use
        # simple strides to extract the queries.
        query = query.transpose([0, 2, 1, 3]).reshape(
            output_size[0] * output_size[1], output_size[2], -1
        )
        # [b, sk, np, hn] -> [b * np, hn, sk]
        key = key.transpose([0, 2, 3, 1]).reshape(
            output_size[0] * output_size[1], -1, output_size[3]
        )

        # preallocting input tensor: [b * np, sq, sk]
        matmul_input_buffer = paddle.empty(
            (output_size[0] * output_size[1], output_size[2], output_size[3]),
            query.dtype,
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = paddle.baddbmm(
            matmul_input_buffer,
            query,
            key,
            beta=0.0,
            alpha=self.softmax_scale,
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.reshape(*output_size)

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs: Tensor = self.scale_mask_softmax(
            attention_scores, attention_mask, self.softmax_offset
        )

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value -> context layer.
        # [b, sk, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (
            value.shape[0],
            value.shape[2],
            query.shape[1],
            value.shape[3],
        )

        # change view [b * np, sk, hn]
        value = value.transpose([0, 2, 1, 3]).reshape(
            output_size[0] * output_size[1], value.shape[1], -1
        )

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.reshape(
            output_size[0] * output_size[1], output_size[2], -1
        )

        # matmul: [b * np, sq, hn]
        context = paddle.bmm(attention_probs, value)

        # change view [b, np, sq, hn]
        context = context.reshape(*output_size)

        # [b, np, sq, hn] --> [b, sq, np, hn]
        context = context.transpose([0, 2, 1, 3]).contiguous()

        # [b, sq, np, hn] --> [b, sq, hp]
        new_context_shape = (
            *context.shape[:-2],
            self.hidden_size_per_partition,
        )
        context = context.reshape(*new_context_shape)

        return context
