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
import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from paddlefleet.packed_seq_params import PackedSeqParams
    from paddlefleet.transformer.transformer_config import TransformerConfig

logger = logging.getLogger(__name__)

import paddle
from paddle import Tensor
from paddle.nn.functional.flash_attention import flashmask_attention

from paddlefleet.context_parallel_utils import flashmask_attention_cp
from paddlefleet.fusions.fused_softmax import FusedScaleMaskSoftmax
from paddlefleet.parallel_state import get_context_parallel_world_size
from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.refined_recompute import (
    RefinedRcomputeFlashMaskAttention as rr_flashmask_attention,
    RefinedRcomputeFlashMaskCpAttention as rr_flashmask_attention_cp,
)
from paddlefleet.transformer.enums import AttnMaskType
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

        projection_size = self.config.head_dim * self.config.num_attention_heads

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
            self.config.num_key_value_heads, world_size
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
            self.config.sliding_window,
            self.config.window_attn_skip_freq,
            layer_number,
        ):
            sliding_window = self.config.sliding_window
        else:
            sliding_window = None

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            input_in_fp16=self.config.fp16,
            input_in_bf16=self.config.bf16,
            attn_mask_type=self.attn_mask_type,
            scaled_masked_softmax_fusion=self.config.masked_softmax_fusion,
            mask_func=attention_mask_func,
            softmax_in_fp32=self.config.attention_softmax_in_fp32,
            scale=coeff,
            sliding_window=sliding_window,
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
        self.rr_flashmask_attention_func = rr_flashmask_attention()

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_startend_row_indices: Tensor = None,
        attn_mask_type: AttnMaskType = None,
        attention_bias: Tensor = None,
        packed_seq_params: PackedSeqParams | None = None,
        use_rr_flash_attention: bool = False,
    ):
        """Forward."""
        assert attention_bias is None, (
            "Attention bias is not supported for DotProductAttention."
        )
        if packed_seq_params is not None:
            assert (
                query.dtype == paddle.bfloat16 or query.dtype == paddle.float16
            ), "attention only support fp16/bf16 when use packed_seq_params"
            lengths = (
                packed_seq_params.cu_seqlens_kv[1:]
                - packed_seq_params.cu_seqlens_kv[:-1]
            )
            splits = [
                paddle.split(tensor, lengths.tolist(), axis=1)
                for tensor in (query, key, value)
            ]
            attn_outputs = []
            for q, k, v in zip(*splits):
                attn_outputs.append(
                    paddle.nn.functional.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        None,
                        self.config.attention_dropout,
                        is_causal=False,
                    )
                )
            # [b,s,h_n,h_dim]
            attn_output = paddle.cat(attn_outputs, axis=1)
            return attn_output.reshape(
                [0, 0, attn_output.shape[2] * attn_output.shape[3]]
            )
        if (
            query.dtype == paddle.bfloat16 or query.dtype == paddle.float16
        ) and attn_mask_startend_row_indices is None:
            # Note:
            # attention_mask is None in default
            # is_causal is True in default
            # training is True in default
            # Default values above maybe changed in the future
            attn_output = paddle.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attention_mask,
                self.config.attention_dropout,
                is_causal=True,
                training=True,
            )

            attn_output = paddle.reshape(
                x=attn_output,
                shape=[0, 0, attn_output.shape[2] * attn_output.shape[3]],
            )

            return attn_output

        elif (
            query.dtype == paddle.bfloat16 or query.dtype == paddle.float16
        ) and attn_mask_startend_row_indices is not None:
            # Note:
            # attn_mask_startend_row_indices is not None for flashmask
            flashmask_attention_func = (
                self.rr_flashmask_attention_func
                if use_rr_flash_attention
                else flashmask_attention
            )

            # Handle MLA case where query/key head_dim != value head_dim
            # flashmask_attention requires head_dim_q == head_dim_v for backward pass
            q_head_dim = query.shape[-1]
            v_head_dim = value.shape[-1]
            need_value_padding = q_head_dim != v_head_dim

            if need_value_padding:
                # Pad value to match query head_dim
                # value: [b, s, h, v_head_dim] -> [b, s, h, q_head_dim]
                bsz, seq_len, num_heads, _ = value.shape
                value_padding = paddle.zeros(
                    [bsz, seq_len, num_heads, q_head_dim - v_head_dim],
                    dtype=value.dtype,
                )
                value_padded = paddle.concat([value, value_padding], axis=-1)
            else:
                value_padded = value

            attn_output = flashmask_attention_func(
                query.astype(value.dtype),
                key.astype(value.dtype),
                value_padded.astype(value.dtype),
                startend_row_indices=attn_mask_startend_row_indices,
                dropout=self.config.attention_dropout,
                causal=(attn_mask_type == AttnMaskType.causal),
            )
            attn_output = attn_output.reshape([0, 0, -1])

            if need_value_padding:
                # Truncate output back to original v_head_dim
                # attn_output: [b, s, h, q_head_dim] -> [b, s, h, v_head_dim]
                attn_output = attn_output[..., :v_head_dim]

            return attn_output

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


class CPDotProductAttention(FleetLayer):
    """
    Attention use flashmask
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

        # self.context_parallel_size = self.config.context_parallel_size
        self.context_parallel_size = get_context_parallel_world_size()

        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type  # unused for now
        self.rr_flashmask_attention_cp_func = rr_flashmask_attention_cp()

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_startend_row_indices: Tensor = None,
        attn_mask_type: AttnMaskType = None,
        attention_bias: Tensor = None,
        packed_seq_params: PackedSeqParams | None = None,
        use_rr_flash_attention: bool = False,
    ):
        """Forward."""
        assert packed_seq_params is None, (
            "Packed sequence is not supported by CPDotProductAttention now."
        )
        assert attention_bias is None, (
            "Attention bias is not supported for CPDotProductAttention now."
        )
        assert self.context_parallel_size > 1, (
            "CPDotProductAttention is only for context_parallel_size > 1."
        )

        b, seq_len = key.shape[0], key.shape[1]
        seq_len = seq_len * self.context_parallel_size

        if attn_mask_startend_row_indices is None:
            attn_mask_startend_row_indices = paddle.full(
                shape=[b, 1, seq_len, 1],
                fill_value=seq_len,
                dtype=paddle.int32,
            ).cuda()

        if attn_mask_startend_row_indices.shape[-1] == 1:
            b, k_heads, k_seqlen, _ = attn_mask_startend_row_indices.shape
            append_indices = paddle.to_tensor(
                np.arange(seq_len),
                dtype=attn_mask_startend_row_indices.dtype,
            ).cuda()
            append_indices = append_indices.reshape(1, 1, seq_len, 1)
            append_indices_expand = append_indices.expand(
                b, k_heads, k_seqlen, 1
            )
            attn_mask_startend_row_indices = paddle.concat(
                [attn_mask_startend_row_indices, append_indices_expand],
                axis=-1,
            )
        elif attn_mask_startend_row_indices.shape[-1] == 2:
            b, k_heads, k_seqlen, _ = attn_mask_startend_row_indices.shape
            append_indices = paddle.to_tensor(
                np.arange(seq_len),
                dtype=attn_mask_startend_row_indices.dtype,
            )
            append_indices = append_indices.reshape(1, 1, seq_len, 1)
            append_indices_expand0 = append_indices.expand(
                b, k_heads, k_seqlen, 1
            )
            append_indices_expand1 = append_indices_expand0.clone()
            attn_mask_startend_row_indices = paddle.concat(
                [
                    attn_mask_startend_row_indices,
                    append_indices_expand0,
                    append_indices_expand1,
                ],
                axis=-1,
            )
        else:
            raise ValueError(
                "Invalid attention mask shape, when using context parallel, attn_mask_startend_row_indices.shape[-1] must be either 1 or 2"
            )
        flashmask_attention_func = (
            self.rr_flashmask_attention_cp_func
            if use_rr_flash_attention
            else flashmask_attention_cp
        )
        attn_output = flashmask_attention_func(
            self.config,
            query.astype(value.dtype),
            key.astype(value.dtype),
            value.astype(value.dtype),
            startend_row_indices=attn_mask_startend_row_indices,
            dropout=self.config.attention_dropout,
            causal=False,  # mask for cp causal is False
        )
        attn_output = attn_output.reshape([0, 0, -1])
        return attn_output
