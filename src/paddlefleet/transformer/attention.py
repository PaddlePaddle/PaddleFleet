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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, NoReturn

if TYPE_CHECKING:
    from .transformer_config import TransformerConfig

import paddle
from paddle import Tensor

from paddlefleet import tensor_parallel
from paddlefleet.models.common.embeddings import (
    apply_rotary_pos_emb,
)
from paddlefleet.models.common.embeddings.yarn_rotary_pos_embedding import (
    _yarn_get_concentration_factor_from_config,
)
from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.spec_utils import LayerSpec, build_layer
from paddlefleet.transformer.layer import FleetLayer
from paddlefleet.utils import divide, get_pg_size

from .enums import AttnMaskType


@dataclass
class SelfAttentionSublayersSpec:
    """
    Configuration class for specifying the sublayers_spec of a self-attention.
    """

    qkv_proj: LayerSpec | type = None
    core_attention: LayerSpec | type = None
    o_proj: LayerSpec | type = None
    q_layernorm: LayerSpec | type = None
    k_layernorm: LayerSpec | type = None


@dataclass
class CrossAttentionSublayersSpec:
    """
    Configuration class for specifying the sublayers_spec of a cross-attention.
    """

    linear_q: LayerSpec | type = None
    linear_kv: LayerSpec | type = None
    core_attention: LayerSpec | type = None
    o_proj: LayerSpec | type = None


class Attention(FleetLayer, ABC):
    """Attention layer abstract class.

    This layer only contains common layers required for the "self attn" and
    "cross attn" specializations.
    """

    def __init__(
        self,
        config: TransformerConfig,
        sublayers_spec: SelfAttentionSublayersSpec
        | CrossAttentionSublayersSpec,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection = None,
    ):
        super().__init__(config=config)

        self.config = config
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type

        # For normal attention without groups, num_key_value_heads == num_attention_heads,
        # so these two will be the same
        self.query_projection_size = (
            self.config.head_dim * self.config.num_attention_heads
        )
        self.kv_projection_size = (
            self.config.head_dim * self.config.num_key_value_heads
        )

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(
                required_pgs=["tp", "cp"]
            )
        else:
            assert hasattr(pg_collection, "tp"), (
                "Attention pg_collection must have tp process group"
            )
            assert hasattr(pg_collection, "cp"), (
                "Attention pg_collection must have cp process group"
            )
        self.pg_collection = pg_collection

        # Per attention head and per partition values
        world_size = get_pg_size(self.pg_collection.tp)
        self.hidden_size_per_attention_head = divide(
            self.query_projection_size, self.config.num_attention_heads
        )
        self.num_attention_heads_per_partition = divide(
            self.config.num_attention_heads, world_size
        )
        self.num_query_groups_per_partition = divide(
            self.config.num_key_value_heads, world_size
        )

        # To support both CUDA Graphs and key value with different hidden size
        self.key_hidden_size = self.hidden_size_per_attention_head
        self.val_hidden_size = self.hidden_size_per_attention_head

        self.core_attention = build_layer(
            sublayers_spec.core_attention,
            config=self.config,
            layer_number=self.layer_number,
            attn_mask_type=self.attn_mask_type,
            attention_type=self.attention_type,
            cp_comm_type=cp_comm_type,
            softmax_scale=self.config.softmax_scale,
            pg_collection=self.pg_collection,
        )

        self.checkpoint_core_attention = (
            self.config.recompute_granularity == "selective"
            and "core_attn" in self.config.recompute_modules
        )

        # Output.
        self.o_proj = build_layer(
            sublayers_spec.o_proj,
            self.query_projection_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.use_bias,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_group=self.pg_collection.tp,
        )

    def _checkpointed_attention_forward(
        self,
        query,
        key,
        value,
        attention_mask,
        rotary_pos_emb=None,
        attn_mask_type=None,
        attention_bias=None,
        packed_seq_params=None,
    ):
        """Forward method with selective activation checkpointing."""

        def custom_forward(*inputs):
            query = inputs[0]
            key = inputs[1]
            value = inputs[2]
            attention_mask = inputs[3]
            attn_mask_type = inputs[5]
            attn_mask_type = AttnMaskType(attn_mask_type.item())
            output_ = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
            return output_

        if attn_mask_type is None:
            attn_mask_type = self.attn_mask_type
        attn_mask_type = paddle.to_tensor(
            [attn_mask_type.value], dtype=paddle.int
        )
        hidden_states = tensor_parallel.checkpoint(
            custom_forward,
            False,
            query,
            key,
            value,
            attention_mask,
            rotary_pos_emb,
            attn_mask_type,
        )

        return hidden_states

    @abstractmethod
    def get_query_key_value_tensors(
        self, hidden_states, key_value_states, split_qkv=True
    ):
        """
        This method needs to be implemented based on whether the derived class
        is "self-attn" or "cross-attn".
        """

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        key_value_states: Tensor | None = None,
        rotary_pos_emb: Tensor | tuple[Tensor, Tensor] | None = None,
        rotary_pos_cos: Tensor | None = None,
        rotary_pos_sin: Tensor | None = None,
        rotary_pos_cos_sin: Tensor | None = None,
        attention_bias: Tensor | None = None,
        packed_seq_params: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Perform a forward pass through the attention layer.

        Args:
            hidden_states (Tensor): Hidden states.
            attention_mask (Tensor): Attention mask.
            key_value_states (Optional[Tensor]): Key/value states (for cross attention).
            rotary_pos_emb (Optional[Union[Tensor, tuple[Tensor, Tensor]]]): Rotary
                embedding tensor(s).
            attention_bias (Optional[Tensor]): Attention bias.
            packed_seq_params (Optional[PackedSeqparams]): Parameters used for THD format.

        Return:
            (tuple[Tensor, Tensor]) Attention output and bias.

        """
        # Check if we need to skip RoPE
        # no_rope is 0-indexed array and self.layer_number is 1-indexed
        # no_rope = (
        #    self.config.no_rope_freq[self.layer_number - 1]
        #    if self.config.no_rope_freq
        #    else False
        # )
        no_rope = False

        if no_rope:
            rotary_pos_emb = None

        # hidden_states: [b, sq, h]

        # For self attention we just duplicate the rotary_pos_emb if it isn't already
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        # =====================
        # Query, Key, and Value
        # =====================
        # Check if fused_single_qkv_rope is requested but either unavailable or not
        # supported for the current use case.
        # if self.attention_type != "cross":
        #   assert not (self.config.fused_single_qkv_rope), (
        #        "fused_single_qkv_rope requested but not available/supported for the config."
        #    )

        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        qkv_output = self.get_query_key_value_tensors(
            hidden_states, key_value_states, split_qkv=True
        )
        attn_mask_type = self.attn_mask_type
        block_table = None
        query, key, value = qkv_output

        # ================================================
        # relative positional embedding (rotary embedding)
        # ================================================
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb

            if packed_seq_params is not None:
                if packed_seq_params.cu_seqlens_q_padded is not None:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
                else:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q
                if packed_seq_params.cu_seqlens_kv_padded is not None:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
                else:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
            else:
                cu_seqlens_q = cu_seqlens_kv = None

            if (
                self.config.apply_rope_fusion
                and q_pos_emb is not None
                and k_pos_emb is not None
            ):
                query, key, _ = apply_rotary_pos_emb(
                    (query, key),
                    None,
                    config=self.config,
                    cu_seqlens=cu_seqlens_q,
                    mscale=None,
                    cp_group=self.pg_collection.cp,
                )
            else:
                if q_pos_emb is not None:
                    query = apply_rotary_pos_emb(
                        query,
                        q_pos_emb,
                        config=self.config,
                        cu_seqlens=cu_seqlens_q,
                        mscale=_yarn_get_concentration_factor_from_config(
                            self.config
                        ),
                        cp_group=self.pg_collection.cp,
                    )

                if k_pos_emb is not None:
                    key = apply_rotary_pos_emb(
                        key,
                        k_pos_emb,
                        config=self.config,
                        cu_seqlens=cu_seqlens_kv,
                        mscale=_yarn_get_concentration_factor_from_config(
                            self.config
                        ),
                        cp_group=self.pg_collection.cp,
                    )

        # ==================================
        # core attention computation
        # ==================================

        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        else:
            # Static batching attention kernel.
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )

        if (
            packed_seq_params is not None
            and packed_seq_params.qkv_format == "thd"
        ):
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.o_proj(core_attn_out)

        return output, bias

    def set_for_recompute_input_layernorm(self):
        """Set the attention layer for recompute input_layernorm. Only needed for fp8."""
        raise NotImplementedError(
            "set_for_recompute_input_layernorm is not implemented."
        )


class SelfAttention(Attention):
    """Self-attention layer class

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        sublayers_spec: SelfAttentionSublayersSpec,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection = None,
    ):
        super().__init__(
            config=config,
            sublayers_spec=sublayers_spec,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
        )

        self.qkv_proj = build_layer(
            sublayers_spec.qkv_proj,
            self.config.hidden_size,
            self.query_projection_size + 2 * self.kv_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.use_bias or self.config.attention_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_group=self.pg_collection.tp,
        )

        if sublayers_spec.q_layernorm is not None:
            self.q_layernorm = build_layer(
                sublayers_spec.q_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.rms_norm_eps,
            )
        else:
            self.q_layernorm = None

        if sublayers_spec.k_layernorm is not None:
            self.k_layernorm = build_layer(
                sublayers_spec.k_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.rms_norm_eps,
            )
        else:
            self.k_layernorm = None

    def get_query_key_value_tensors(
        self, hidden_states, key_value_states=None, split_qkv=True
    ):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`. If `split_qkv=False`, then
        the unsplit mixed_qkv tensor is returned.
        """
        # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
        mixed_qkv, _ = self.qkv_proj(hidden_states)

        # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
        new_tensor_shape = (
            *mixed_qkv.shape[:-1],
            self.num_query_groups_per_partition,
            (
                (
                    self.num_attention_heads_per_partition
                    // self.num_query_groups_per_partition
                    + 2
                )
                * self.hidden_size_per_attention_head
            ),
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)

        split_arg_list = [
            (
                self.num_attention_heads_per_partition
                // self.num_query_groups_per_partition
                * self.hidden_size_per_attention_head
            ),
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
        ]

        # Return unsplit mixed_qkv and split_arg_list
        if not split_qkv:
            return mixed_qkv, split_arg_list

        # [sq, b, ng, (np/ng + 2) * hn]
        # --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
        (query, key, value) = paddle.split(mixed_qkv, split_arg_list, axis=3)

        # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        query = query.reshape(
            query.shape[0],
            query.shape[1],
            -1,
            self.hidden_size_per_attention_head,
        )

        if self.q_layernorm is not None:
            query = self.q_layernorm(query)

        if self.k_layernorm is not None:
            key = self.k_layernorm(key)

        return query, key, value

    def backward_dw(self) -> NoReturn:
        """Execute weight update operations"""
        self._backward_qkv_proj()
        self._backward_output_proj()

    def _backward_qkv_proj(self):
        """Update weights for QKV projection layer"""
        self.qkv_proj.backward_dw()

    def _backward_output_proj(self):
        """Update weights for output projection layer"""
        self.o_proj.backward_dw()


class CrossAttention(Attention):
    """Cross-attention layer class

    Cross-attention layer takes input with size [s, b, h] and context with size
    [s, b, h] and returns output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        sublayers_spec: CrossAttentionSublayersSpec,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection = None,
    ):
        super().__init__(
            config=config,
            sublayers_spec=sublayers_spec,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="cross",
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
        )

        if self.config.num_key_value_heads != self.config.num_attention_heads:
            raise ValueError(
                "Group query attention is not currently supported in cross attention."
            )
        assert self.query_projection_size == self.kv_projection_size

        self.linear_q = build_layer(
            sublayers_spec.linear_q,
            self.config.hidden_size,
            self.query_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.use_bias,
            skip_bias_add=False,
            is_expert=False,
        )

        self.linear_kv = build_layer(
            sublayers_spec.linear_kv,
            self.config.hidden_size,
            2 * self.kv_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.use_bias,
            skip_bias_add=False,
            is_expert=False,
        )

    def get_query_key_value_tensors(
        self, hidden_states, key_value_states, split_qkv=True
    ):
        """
        Derives `query` tensor from `hidden_states`, and `key`/`value` tensors
        from `key_value_states`.
        """
        assert split_qkv, "split_qkv must be True for CrossAttention"
        # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
        mixed_kv, _ = self.linear_kv(key_value_states)

        # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
        new_tensor_shape = (
            *mixed_kv.size()[:-1],
            self.num_attention_heads_per_partition,
            2 * self.hidden_size_per_attention_head,
        )
        mixed_kv = mixed_kv.view(*new_tensor_shape)

        # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
        (key, value) = tensor_parallel.split_tensor_along_last_dim(mixed_kv, 2)

        # Attention head [sq, b, h] --> [sq, b, hp]
        query, _ = self.linear_q(hidden_states)

        # [sq, b, hp] --> [sq, b, np, hn]
        new_tensor_shape = (
            *query.size()[:-1],
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        )
        query = query.view(*new_tensor_shape)

        return query, key, value
