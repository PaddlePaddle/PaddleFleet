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
from paddle.distributed.fleet.utils import recompute

from paddlefleet import tensor_parallel
from paddlefleet.models.common.embeddings import (
    apply_rotary_pos_emb,
)
from paddlefleet.models.common.embeddings.yarn_rotary_pos_embedding import (
    _yarn_get_concentration_factor_from_config,
)
from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.recompute_utils import (
    need_recompute_in_block,
    need_recompute_in_first_n,
)
from paddlefleet.spec_utils import LayerSpec, build_layer
from paddlefleet.tensor_parallel.mappings import (
    gather_from_tensor_model_parallel_region,
    scatter_to_tensor_model_parallel_region,
)
from paddlefleet.transformer.layer import FleetLayer
from paddlefleet.utils import divide, get_pg_rank, get_pg_size

from .dot_product_attention import CPDotProductAttention
from .enums import AttnMaskType


@dataclass
class SelfAttentionSublayersSpec:
    """
    Configuration class for specifying the sublayers_spec of a self-attention.
    """

    qkv_proj: LayerSpec | type = None
    core_attention: LayerSpec | type = None
    o_proj: LayerSpec | type = None
    q_norm: LayerSpec | type = None
    k_norm: LayerSpec | type = None


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
        self.use_rr_flash_attention = False
        self.recompute_core_attention = False
        if self.config.recompute_granularity == "selective":
            if isinstance(self.config.recompute_modules, list):
                if self.config.recompute_num_layers is None:
                    # selective all submodels to recompute
                    if "core_attn" in self.config.recompute_modules:
                        self.recompute_core_attention = True
                else:
                    # selective submodels in special layers to recompute
                    assert self.config.recompute_method in ["first_n", "block"]
                    if "core_attn" in self.config.recompute_modules:
                        self.recompute_core_attention = (
                            need_recompute_in_block(
                                self.layer_number,
                                self.config,
                                self.config.recompute_num_layers,
                            )
                            if self.config.recompute_method == "block"
                            else need_recompute_in_first_n(
                                self.layer_number,
                                self.config,
                                self.config.recompute_num_layers,
                            )
                        )
            elif isinstance(self.config.recompute_modules, dict):
                assert self.config.recompute_method in ["first_n", "block"]
                if "core_attn" in self.config.recompute_modules:
                    self.recompute_core_attention = (
                        need_recompute_in_block(
                            self.layer_number,
                            self.config,
                            self.config.recompute_modules["core_attn"],
                        )
                        if self.config.recompute_method == "block"
                        else need_recompute_in_first_n(
                            self.layer_number,
                            self.config,
                            self.config.recompute_modules["core_attn"],
                        )
                    )
        if (
            self.config.recompute_modules is not None
            and "flash_attn" in self.config.recompute_modules
        ):
            assert self.config.recompute_granularity is not None, (
                "rr must be used when recompute is enabled"
            )
            if isinstance(self.config.recompute_modules, list):
                self.use_rr_flash_attention = True
            elif isinstance(self.config.recompute_modules, dict):
                self.use_rr_flash_attention = not need_recompute_in_first_n(
                    self.layer_number,
                    self.config,
                    self.config.recompute_modules["flash_attn"],
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
        attn_mask_startend_row_indices: Tensor | None = None,
        key_value_states: Tensor | None = None,
        rotary_pos_emb: Tensor | tuple[Tensor, Tensor] | None = None,
        rotary_pos_cos: Tensor | None = None,
        rotary_pos_sin: Tensor | None = None,
        rope_freqs_cis: Tensor | None = None,
        position_ids: Tensor | None = None,
        attention_bias: Tensor | None = None,
        packed_seq_params: Tensor | None = None,
        in_recompute: bool = False,
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
        if len(qkv_output) == 4:
            query, key, value, gate = qkv_output
        else:
            query, key, value = qkv_output
            gate = None

        # ================================================
        # relative positional embedding (rotary embedding)
        # ================================================
        if rope_freqs_cis is not None:
            rope_freqs_cis = rope_freqs_cis.unsqueeze(-2)  # ..., 1, head_dim/2
            # ..., num_heads, head_dim/2
            query_ = paddle.view_as_complex(
                query.float().view(*query.shape[:-1], -1, 2)
            )
            key_ = paddle.view_as_complex(
                key.float().view(*key.shape[:-1], -1, 2)
            )
            query = (
                paddle.view_as_real(query_ * rope_freqs_cis)
                .flatten(-2)
                .to(hidden_states.dtype)
            )  # ..., num_heads, head_dim
            key = (
                paddle.view_as_real(key_ * rope_freqs_cis)
                .flatten(-2)
                .to(hidden_states.dtype)
            )  # ..., num_heads, head_dim

        elif rotary_pos_emb is not None:
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
                total_seqlen_q = packed_seq_params.total_seqlen_q
                total_seqlen_kv = packed_seq_params.total_seqlen_kv
            else:
                cu_seqlens_q = cu_seqlens_kv = None
                total_seqlen_q = total_seqlen_kv = None

            if (
                self.config.apply_rope_fusion
                and q_pos_emb is not None
                and k_pos_emb is not None
            ):
                query, key, _ = apply_rotary_pos_emb(
                    (query, key),
                    None,
                    rotary_pos_cos,
                    rotary_pos_sin,
                    config=self.config,
                    cu_seqlens=cu_seqlens_q,
                    position_ids=position_ids,
                    mscale=None,
                    cp_group=self.pg_collection.cp,
                )
            # elif self.config.apply_vision_rope:
            #     query, key = apply_rotary_pos_emb_vision(query,key,rotary_pos_cos,rotary_pos_sin)
            else:
                if q_pos_emb is not None:
                    query = apply_rotary_pos_emb(
                        query,
                        q_pos_emb,
                        None,
                        None,
                        config=self.config,
                        cu_seqlens=cu_seqlens_q,
                        total_seq_len=total_seqlen_q,
                        position_ids=position_ids,
                        mscale=_yarn_get_concentration_factor_from_config(
                            self.config
                        ),
                        cp_group=self.pg_collection.cp,
                    )

                if k_pos_emb is not None:
                    key = apply_rotary_pos_emb(
                        key,
                        k_pos_emb,
                        None,
                        None,
                        config=self.config,
                        cu_seqlens=cu_seqlens_kv,
                        total_seq_len=total_seqlen_kv,
                        position_ids=position_ids,
                        mscale=_yarn_get_concentration_factor_from_config(
                            self.config
                        ),
                        cp_group=self.pg_collection.cp,
                    )

        # ==================================
        # core attention computation
        # ==================================

        # NOTE: For sequence parallel, the input is [seq, b, h],
        # transpose back to [b, seq, h] for attention computation
        # TODO: supports [seq, b, h] input in attention computation
        if self.config.sequence_parallel:
            query = query.transpose([1, 0, 2, 3]).contiguous()
            key = key.transpose([1, 0, 2, 3]).contiguous()
            value = value.transpose([1, 0, 2, 3]).contiguous()
            # Slice and adjust attn_mask_startend_row_indices for the local SP sequence
            # range. The full mask has shape [B, 1, S, 1] with absolute row indices.
            # Each SP rank processes key/query positions [tp_rank*L : (tp_rank+1)*L],
            # so we need the local slice with row indices adjusted to local space.
            if attn_mask_startend_row_indices is not None and not isinstance(
                self.core_attention, CPDotProductAttention
            ):
                # Skip this adjustment when CP is active, as CPDotProductAttention
                # expects the full global mask and handles CP splitting internally.
                local_seq = key.shape[1]  # S / tp_size after transpose
                if attn_mask_startend_row_indices.shape[2] != local_seq:
                    tp_rank = get_pg_rank(self.pg_collection.tp)
                    offset = tp_rank * local_seq
                    attn_mask_startend_row_indices = paddle.clip(
                        attn_mask_startend_row_indices[
                            :, :, offset : offset + local_seq, :
                        ]
                        - offset,
                        min=0,
                    ).astype(paddle.int32)

        if self.recompute_core_attention and self.training:
            core_attn_out = recompute(
                self.core_attention,
                query,
                key,
                value,
                attention_mask.clone() if attention_mask is not None else None,
                attn_mask_startend_row_indices.clone()
                if attn_mask_startend_row_indices is not None
                else None,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                use_rr_flash_attention=self.use_rr_flash_attention,
            )
        else:
            # Static batching attention kernel.
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_startend_row_indices,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                use_rr_flash_attention=self.use_rr_flash_attention
                and in_recompute,
            )
        # =================
        # Output. [b, sq, h]
        # =================

        if self.config.sequence_parallel:
            core_attn_out = core_attn_out.transpose([1, 0, 2]).contiguous()

        # Apply gated attention: gate the attention output before output projection
        if gate is not None:
            core_attn_out = core_attn_out * paddle.nn.functional.sigmoid(gate)

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

        self.gated_attention = getattr(self.config, "gated_attention", False)
        gate_projection_size = (
            self.query_projection_size if self.gated_attention else 0
        )

        self.qkv_proj = build_layer(
            sublayers_spec.qkv_proj,
            self.config.hidden_size,
            self.query_projection_size
            + 2 * self.kv_projection_size
            + gate_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.use_bias or self.config.attention_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_group=self.pg_collection.tp,
        )

        # For per_layer qk_norm, norm operates on gathered (full) tensors,
        # so input_is_parallel should be False to avoid extra allreduce.
        if getattr(self.config, "qk_norm_type", "per_head") == "per_layer":
            norm_input_parallel = False
        else:
            norm_input_parallel = config.tensor_model_parallel_size > 1

        if sublayers_spec.q_norm is not None:
            if getattr(self.config, "qk_norm_type", "per_head") == "per_layer":
                q_norm_hidden_size = (
                    self.hidden_size_per_attention_head
                    * self.config.num_attention_heads
                )
            else:
                q_norm_hidden_size = self.hidden_size_per_attention_head
            self.q_norm = build_layer(
                sublayers_spec.q_norm,
                hidden_size=q_norm_hidden_size,
                config=self.config,
                eps=self.config.rms_norm_eps,
                input_is_parallel=norm_input_parallel,
            )
        else:
            self.q_norm = None

        if sublayers_spec.k_norm is not None:
            if getattr(self.config, "qk_norm_type", "per_head") == "per_layer":
                k_norm_hidden_size = (
                    self.hidden_size_per_attention_head
                    * self.config.num_key_value_heads
                )
            else:
                k_norm_hidden_size = self.hidden_size_per_attention_head
            self.k_norm = build_layer(
                sublayers_spec.k_norm,
                hidden_size=k_norm_hidden_size,
                config=self.config,
                eps=self.config.rms_norm_eps,
                input_is_parallel=norm_input_parallel,
            )
        else:
            self.k_norm = None

    def get_query_key_value_tensors(
        self, hidden_states, key_value_states=None, split_qkv=True
    ):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`. If `split_qkv=False`, then
        the unsplit mixed_qkv tensor is returned.
        When gated_attention is enabled, also returns a gate tensor for output gating.
        """
        # Attention heads [b, sq, h] --> [b, sq, ng * group_dim]
        mixed_qkv, _ = self.qkv_proj(hidden_states)

        heads_per_group = (
            self.num_attention_heads_per_partition
            // self.num_query_groups_per_partition
        )
        q_dim = heads_per_group * self.hidden_size_per_attention_head

        if self.gated_attention:
            # Per group: Q + Gate + K + V
            group_dim = (
                heads_per_group * 2 + 2
            ) * self.hidden_size_per_attention_head
        else:
            # Per group: Q + K + V
            group_dim = (
                heads_per_group + 2
            ) * self.hidden_size_per_attention_head

        # [b, sq, hp] --> [b, sq, ng, group_dim]
        new_tensor_shape = (
            *mixed_qkv.shape[:-1],
            self.num_query_groups_per_partition,
            group_dim,
        )
        mixed_qkv = mixed_qkv.reshape(*new_tensor_shape)

        if self.gated_attention:
            split_arg_list = [
                q_dim,
                q_dim,
                self.hidden_size_per_attention_head,
                self.hidden_size_per_attention_head,
            ]
        else:
            split_arg_list = [
                q_dim,
                self.hidden_size_per_attention_head,
                self.hidden_size_per_attention_head,
            ]

        # Return unsplit mixed_qkv and split_arg_list
        if not split_qkv:
            return mixed_qkv, split_arg_list

        parts = paddle.split(mixed_qkv, split_arg_list, axis=3)

        if self.gated_attention:
            query, gate, key, value = parts
        else:
            query, key, value = parts
            gate = None

        if getattr(self.config, "qk_norm_type", "per_head") == "per_layer" and (
            self.q_norm is not None or self.k_norm is not None
        ):
            # per_layer qk_norm: normalize across all heads jointly

            # Flatten to [b, sq, np * hn] / [b, sq, ng * hn]
            query = query.reshape(*query.shape[:2], -1)
            key = key.reshape(*key.shape[:2], -1)

            # TP gather: collect all TP shards so norm sees the full dimension
            enable_tp = get_pg_size(self.pg_collection.tp) > 1
            if enable_tp:
                query = gather_from_tensor_model_parallel_region(
                    query, group=self.pg_collection.tp
                )
                key = gather_from_tensor_model_parallel_region(
                    key, group=self.pg_collection.tp
                )

            if self.q_norm is not None:
                query = self.q_norm(query)
            if self.k_norm is not None:
                key = self.k_norm(key)

            # TP scatter: split back to per-rank shards
            if enable_tp:
                query = scatter_to_tensor_model_parallel_region(
                    query, group=self.pg_collection.tp
                )
                key = scatter_to_tensor_model_parallel_region(
                    key, group=self.pg_collection.tp
                )

            # Reshape to per-head layout [b, sq, np, hn] / [b, sq, ng, hn]
            query = query.reshape(
                query.shape[0],
                query.shape[1],
                -1,
                self.hidden_size_per_attention_head,
            )
            key = key.reshape(
                key.shape[0],
                key.shape[1],
                -1,
                self.hidden_size_per_attention_head,
            )
        else:
            # per_head qk_norm (default): reshape first, then normalize per head
            # [b, sq, ng, np/ng * hn] -> [b, sq, np, hn]
            query = query.reshape(
                query.shape[0],
                query.shape[1],
                -1,
                self.hidden_size_per_attention_head,
            )

            if self.q_norm is not None:
                query = self.q_norm(query)

            if self.k_norm is not None:
                key = self.k_norm(key)

        if gate is not None:
            # [b, sq, ng, np/ng * hn] -> [b, sq, np * hn]
            gate = gate.reshape(*gate.shape[:2], -1)
            return query, key, value, gate

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

        # Attention head [b, sq, h] --> [b, sq, hp]
        query, _ = self.linear_q(hidden_states)

        # [b, sq, hp] --> [b, sq, np, hn]
        new_tensor_shape = (
            *query.size()[:-1],
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        )
        query = query.view(*new_tensor_shape)

        return query, key, value
