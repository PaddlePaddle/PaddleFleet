# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import math
import os
from dataclasses import dataclass
from typing import NoReturn

import paddle
from paddle import Tensor
from paddle.distributed.fleet.meta_parallel import LayerSpec, build_spec_layer
from paddle.distributed.fleet.utils import recompute

from paddlefleet.models.common.embeddings import (
    apply_rotary_pos_emb,
)
from paddlefleet.models.common.embeddings.rotary_pos_embedding import (
    RotaryEmbedding,
)
from paddlefleet.models.common.embeddings.yarn_rotary_pos_embedding import (
    YarnRotaryEmbedding as YarnRotaryEmbedding,
    _yarn_get_mscale,
)
from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    gather_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)
from paddlefleet.transformer.attention import Attention
from paddlefleet.transformer.enums import AttnMaskType
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.utils import get_pg_size

try:
    from paddlefleet.transformer.dot_product_attention import (
        CPDotProductAttention,
    )
except Exception:
    CPDotProductAttention = None

try:
    from paddlefleet.fusions.fused_mla_yarn_rope_apply import (
        fused_apply_mla_rope_for_kv,
        fused_apply_mla_rope_for_q,
    )
except:
    fused_apply_mla_rope_for_kv = None
    fused_apply_mla_rope_for_q = None


def _ec_compatible_rope_apply(q_pe, k_pe, seq_len, rope_base=1000000.0):
    """Apply RoPE using EC's complex multiplication method (no YaRN, no mscale).

    This exactly matches ErnieCore's compute_freqs_cis_mrope_and_apply_rotary_3d
    when position_ids are sequential [0, 1, 2, ..., seq_len-1] (text-only case
    where all 3 mRoPE axes have the same value).

    Args:
        q_pe: [B, S, H, D] query positional embedding portion
        k_pe: [B, S, 1, D] key positional embedding portion
        seq_len: sequence length
        rope_base: base frequency (default 1e6)
    """
    head_dim = q_pe.shape[-1]
    # inv_freq same as EC: 1 / (base^(arange(0, dim, 2) / dim))
    freqs = 1.0 / (
        rope_base
        ** (paddle.arange(0, head_dim, 2, dtype="float32") / float(head_dim))
    )
    # position ids: [0, 1, ..., seq_len-1]
    positions = paddle.arange(0, seq_len, dtype="float32")
    # freqs_table: [S, D/2]
    freqs_table = paddle.outer(positions, freqs)
    # Expand for batch: [1, S, D/2]
    freqs_expanded = freqs_table.unsqueeze(0)
    # Expand to match q_pe batch size: [B, S, D/2]
    freqs_expanded = freqs_expanded.expand(
        [q_pe.shape[0], seq_len, head_dim // 2]
    )
    # freqs_cis: complex [B, S, D/2] -> [B, S, 1, D/2]
    freqs_cis = paddle.polar(paddle.ones_like(freqs_expanded), freqs_expanded)
    freqs_cis = freqs_cis.unsqueeze(2)  # [B, S, 1, D/2]

    # MD5 debug
    import hashlib as _hl

    _log_md5 = os.environ.get("LOG_LAYER_MD5", "0") == "1"
    if _log_md5:
        import paddle.distributed as _dist

        _r = _dist.get_rank() if _dist.is_initialized() else 0
        _fc_real = paddle.as_real(freqs_cis)
        _md5 = _hl.md5(_fc_real.cast("float32").numpy().tobytes()).hexdigest()
        print(
            f"[MD5 Probe PF] Rank={_r} freqs_cis MD5={_md5} shape={list(freqs_cis.shape)} q_dtype={q_pe.dtype}",
            flush=True,
        )

    # Apply to q_pe via complex multiplication (EC style: interleaved pairs)
    xq = paddle.reshape(
        q_pe.cast("float32"), [*q_pe.shape[:-1], -1, 2]
    )  # [B,S,H,D/2,2]
    xk = paddle.reshape(
        k_pe.cast("float32"), [*k_pe.shape[:-1], -1, 2]
    )  # [B,S,1,D/2,2]
    xq_ = paddle.as_complex(xq)  # [B,S,H,D/2]
    xk_ = paddle.as_complex(xk)  # [B,S,1,D/2]

    if _log_md5:
        _xq_data = paddle.as_real(xq_).cast("float32").numpy().tobytes()
        _xq_md5 = _hl.md5(_xq_data).hexdigest()
        print(
            f"[MD5 Probe PF] Rank={_r} xq_complex MD5={_xq_md5} shape={list(xq_.shape)}",
            flush=True,
        )

    xq_out = paddle.as_real(xq_ * freqs_cis)  # [B,S,H,D/2,2]
    xk_out = paddle.as_real(xk_ * freqs_cis)  # [B,S,1,D/2,2]

    xq_out = paddle.flatten(xq_out, start_axis=3)  # [B,S,H,D]
    xk_out = paddle.flatten(xk_out, start_axis=3)  # [B,S,1,D]

    return xq_out.cast(q_pe.dtype), xk_out.cast(k_pe.dtype)


@dataclass
class MLASelfAttentionSublayersSpec:
    """Sublayers for MLA self-attention layer."""

    q_a_layernorm: LayerSpec | type = None
    kv_a_layernorm: LayerSpec | type = None

    q_proj: LayerSpec | type = None
    q_a_proj: LayerSpec | type = None
    q_b_proj: LayerSpec | type = None
    kv_a_proj_with_mqa: LayerSpec | type = None
    kv_b_proj: LayerSpec | type = None
    core_attention: LayerSpec | type = None
    o_proj: LayerSpec | type = None
    gate_proj: LayerSpec | type = None


class MultiLatentAttention(Attention):
    """Multi-Latent Attention layer abstract class."""

    def __init__(
        self,
        config: TransformerConfig,
        sublayers_spec: MLASelfAttentionSublayersSpec,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection | None = None,
    ) -> None:
        super().__init__(
            config=config,
            sublayers_spec=sublayers_spec,
            layer_number=layer_number,
            attention_type=attention_type,
            attn_mask_type=attn_mask_type,
            pg_collection=pg_collection,
        )
        self.config: TransformerConfig

        self.query_projection_size = (
            self.config.v_head_dim * self.config.num_attention_heads
        )

        self.q_head_dim = (
            self.config.qk_nope_head_dim + self.config.qk_rope_head_dim
        )

        mscale = _yarn_get_mscale(
            self.config.rotary_scaling_factor, self.config.mscale_all_dim
        )
        self.softmax_scale = mscale * mscale / math.sqrt(self.q_head_dim)

        if self.config.rope_type == "rope":
            self.rotary_pos_emb = RotaryEmbedding(
                self.config.qk_rope_head_dim,
                rotary_interleaved=self.config.rotary_interleaved,
                rotary_percent=self.config.rotary_percent,
                rotary_base=self.config.rope_theta,
                cp_group=self.pg_collection.cp,
            )
        elif self.config.rope_type == "yarn":
            self.rotary_pos_emb = YarnRotaryEmbedding(
                self.config.qk_rope_head_dim,
                rotary_interleaved=self.config.rotary_interleaved,
                rotary_base=self.config.rope_theta,
                scaling_factor=self.config.rotary_scaling_factor,
                original_max_position_embeddings=self.config.original_max_position_embeddings,
                beta_fast=self.config.beta_fast,
                beta_slow=self.config.beta_slow,
                mscale=self.config.mscale,
                mscale_all_dim=self.config.mscale_all_dim,
                # cp_group=self.pg_collection.cp,
            )
        else:
            raise ValueError(
                f"Unsupported RoPE type: {self.config.rope_type}, supported types are "
                "'rope' and 'yarn'"
            )

        self.core_attention = build_spec_layer(
            sublayers_spec.core_attention,
            config=self.config,
            layer_number=self.layer_number,
            attn_mask_type=self.attn_mask_type,
            attention_type=self.attention_type,
            softmax_scale=self.softmax_scale,
            k_channels=self.q_head_dim,
            v_channels=self.config.v_head_dim,
            cp_comm_type=cp_comm_type,
            pg_collection=self.pg_collection,
        )

        # Output.
        self.o_proj = build_spec_layer(
            sublayers_spec.o_proj,
            self.query_projection_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.use_bias,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="proj",
            tp_group=self.pg_collection.tp,
        )

        # Gated attention
        self.gated_attention = getattr(self.config, "gated_attention", False)
        if self.gated_attention and sublayers_spec.gate_proj is not None:
            self.gate_proj = build_spec_layer(
                sublayers_spec.gate_proj,
                self.config.hidden_size,
                self.query_projection_size,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=self.config.use_bias,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name="mla_gate",
                tp_group=self.pg_collection.tp,
            )
        else:
            self.gated_attention = False
            self.gate_proj = None

    def forward(
        self,
        hidden_states,
        attention_mask,
        attn_mask_startend_row_indices: Tensor | None = None,
        key_value_states=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        in_recompute: bool = False,
        position_ids=None,
    ):
        """Forward pass for multi-latent attention"""
        from paddlefleet.transformer.transformer_layer import TransformerLayer

        _log = TransformerLayer._log_md5

        assert rotary_pos_emb is None, (
            "Rotary position embeddings should not be passed into MLA."
        )
        assert attention_bias is None, (
            "Attention bias should not be passed into MLA."
        )
        assert rotary_pos_cos is None and rotary_pos_sin is None, (
            "MLA does not support Flash Decoding"
        )

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention

        query, key, value, _, _ = self.get_query_key_value_tensors(
            hidden_states,
            key_value_states,
            position_ids,
            packed_seq_params,
        )

        layer_num = getattr(self, "layer_number", -1)
        _log(query, "attn_query", layer_num)
        _log(key, "attn_key", layer_num)
        if value is not None:
            _log(value, "attn_value", layer_num)

        attn_mask_type = self.attn_mask_type
        query = query.contiguous()
        key = key.contiguous()

        if value is not None:
            value = value.contiguous()

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

        _log(core_attn_out, "core_attn_out", layer_num)

        # =================
        # Output. [b, sq, h]
        # =================
        if self.config.sequence_parallel:
            core_attn_out = core_attn_out.transpose([1, 0, 2]).contiguous()

        # Apply gated attention
        if self.gated_attention:
            gate, _ = self.gate_proj(hidden_states)
            core_attn_out = core_attn_out * paddle.nn.functional.sigmoid(gate)

        output, bias = self.o_proj(core_attn_out)

        _log(output, "attn_o_proj_out", layer_num)

        return output, bias


class MLASelfAttention(MultiLatentAttention):
    """MLA Self-attention layer class

    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        sublayers_spec: MLASelfAttentionSublayersSpec,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection | None = None,
    ):
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        super().__init__(
            config=config,
            sublayers_spec=sublayers_spec,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
        )

        if self.config.q_lora_rank is None:
            # Not projecting query
            self.q_proj = build_spec_layer(
                sublayers_spec.q_proj,
                self.config.hidden_size,
                self.config.num_attention_heads * self.q_head_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name="q_proj",
            )

        else:
            self.q_a_proj = build_spec_layer(
                sublayers_spec.q_a_proj,
                self.config.hidden_size,
                self.config.q_lora_rank,
                config=self.config,
                init_method=self.config.init_method,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name="q_a_proj",
                skip_weight_param_allocation=False,
                tp_group=pg_collection.tp,
            )

            self.q_b_proj = build_spec_layer(
                sublayers_spec.q_b_proj,
                self.config.q_lora_rank,
                self.config.num_attention_heads * self.q_head_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name="q_b_proj",
                tp_group=pg_collection.tp,
            )

        self.kv_a_proj_with_mqa = build_spec_layer(
            sublayers_spec.kv_a_proj_with_mqa,
            self.config.hidden_size,
            self.config.kv_lora_rank + self.config.qk_rope_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="kv_a_proj_with_mqa",
            skip_weight_param_allocation=False,
            tp_group=pg_collection.tp,
        )

        self.kv_b_proj = build_spec_layer(
            sublayers_spec.kv_b_proj,
            self.config.kv_lora_rank,
            self.config.num_attention_heads
            * (self.config.qk_nope_head_dim + self.config.v_head_dim),
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="kv_b_proj",
            tp_group=pg_collection.tp,
        )

        if self.config.q_lora_rank is not None:
            self.q_a_layernorm = build_spec_layer(
                sublayers_spec.q_a_layernorm,
                hidden_size=self.config.q_lora_rank,
                config=self.config,
                eps=self.config.rms_norm_eps,
            )

        self.kv_a_layernorm = build_spec_layer(
            sublayers_spec.kv_a_layernorm,
            hidden_size=self.config.kv_lora_rank,
            config=self.config,
            eps=self.config.rms_norm_eps,
        )

    def get_query_key_value_tensors(
        self,
        hidden_states,
        key_value_states=None,
        position_ids=None,
        packed_seq_params=None,
    ):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # b = batch size, s = sequence length, h = hidden size, n = num attention heads
        # Attention heads [b, s, n*h]
        assert hidden_states.ndim == 3, (
            f"hidden_states should be 3D, [b, s, n*h], got {hidden_states.ndim}D"
        )

        # =========================================
        # Prepare RoPE and seqlen related params
        # =========================================
        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
            hidden_states, self.config, packed_seq_params
        )

        # rotary_pos_emb:[1, s, 1, 64]
        mscale = 1.0
        rotary_pos_cos = None
        rotary_pos_sin = None
        packed_seq = (
            packed_seq_params is not None
            and packed_seq_params.qkv_format == "thd"
        )
        if self.config.rope_type == "rope":
            rotary_pos_emb = self.rotary_pos_emb(
                rotary_seq_len, packed_seq=packed_seq
            )
        else:
            if self.config.apply_rope_fusion:
                rotary_pos_cos, rotary_pos_sin = (
                    self.rotary_pos_emb.get_cached_cos_sin(
                        rotary_seq_len,
                        dtype=hidden_states.dtype,
                        packed_seq=packed_seq,
                    )
                )
                rotary_pos_emb = None
                assert (
                    fused_apply_mla_rope_for_q is not None
                    and fused_apply_mla_rope_for_kv is not None
                ), "Fused MLA RoPE apply is not imported successfully"
            else:
                rotary_pos_emb, mscale = self.rotary_pos_emb(
                    rotary_seq_len, packed_seq=packed_seq
                )
                # mscale is already accounted for in self.softmax_scale; set to 1.0 to avoid double-applying
                # mscale = 1.0

        if (
            packed_seq_params is not None
            and packed_seq_params.qkv_format == "thd"
        ):
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

        # =========================================
        # QKV down projection and layernorm
        # =========================================
        if self.config.q_lora_rank is not None:
            # if q_a_proj is ColumnParallelLinear:
            #     q_compressed: [b, s, q_lora_rank / TP]
            q_compressed, _ = self.q_a_proj(hidden_states)

            # When output is sharded (ColumnParallelLinear):
            # Gather output to restore output dim q_lora_rank;
            # Scatter sequence back to s / TP if sequence-parallel
            if q_compressed.size(-1) != self.config.q_lora_rank:
                q_compressed = gather_from_tensor_model_parallel_region(
                    q_compressed
                )
                if self.config.sequence_parallel:
                    q_compressed = scatter_to_sequence_parallel_region(
                        q_compressed
                    )
        else:
            q_compressed = hidden_states

        # if kv_a_proj_with_mqa is ColumnParallelLinear:
        #     kv_combined: [b, s, (kv_lora_rank + qk_rope_head_dim) / TP]
        kv_combined, _ = self.kv_a_proj_with_mqa(hidden_states)
        if (
            kv_combined.size(-1)
            != self.config.kv_lora_rank + self.config.qk_rope_head_dim
        ):
            # kv_combined: [b, s, (kv_lora_rank + qk_rope_head_dim)]
            kv_combined = gather_from_tensor_model_parallel_region(kv_combined)
            # kv_compressed:[b, s, kv_lora_rank], k_pos_emb: [b, s, qk_rope_head_dim]
            kv_compressed, k_pos_emb = paddle.split(
                kv_combined,
                [self.config.kv_lora_rank, self.config.qk_rope_head_dim],
                axis=-1,
            )
            if self.config.sequence_parallel:
                # kv_compressed:[b, s / TP, kv_lora_rank]
                kv_compressed = scatter_to_sequence_parallel_region(
                    kv_compressed
                )
        else:
            # kv_compressed:[b, s / TP, kv_lora_rank], k_pos_emb: [b, s / TP, qk_rope_head_dim]
            kv_compressed, k_pos_emb = paddle.split(
                kv_combined,
                [self.config.kv_lora_rank, self.config.qk_rope_head_dim],
                axis=-1,
            )
            if (
                get_pg_size(self.pg_collection.tp) > 1
                and self.config.sequence_parallel
            ):
                # k_pos_emb: [b, s, qk_rope_head_dim]
                k_pos_emb = gather_from_sequence_parallel_region(
                    k_pos_emb, group=self.pg_collection.tp
                )

        # if packed_seq_params is not None:
        #     # PaddleFleet batch-first: [b=1, t, h] -> squeeze dim0 (batch) -> [t, h]
        #     # (SP seq-first: [t, b=1, h] -> squeeze dim1 (batch) -> [t, h])
        #     batch_dim = 1 if self.config.sequence_parallel else 0
        #     q_compressed = q_compressed.squeeze(batch_dim)
        #     kv_compressed = kv_compressed.squeeze(batch_dim)
        #     k_pos_emb = k_pos_emb.squeeze(batch_dim)

        # =========================================
        # Apply norm
        # =========================================

        if self.config.q_lora_rank is not None:
            # q_compressed: [num_tokens, q_lora_rank]
            q_compressed = self.q_a_layernorm(q_compressed)

        kv_compressed = self.kv_a_layernorm(kv_compressed)

        # === MD5 probes for MLA intermediate values ===
        from paddlefleet.transformer.transformer_layer import TransformerLayer

        _log = TransformerLayer._log_md5
        _log(q_compressed, "mla_q_compressed_normed", self.layer_number)
        _log(kv_compressed, "mla_kv_compressed_normed", self.layer_number)
        _log(k_pos_emb, "mla_k_pos_emb_raw", self.layer_number)

        # =========================================
        # QKV up projection and RoPE apply
        # =========================================

        def qkv_up_proj_and_rope_apply(
            q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb
        ):
            """
            Apply the up projection and RoPE to the query and key.
            When sequence packing enabled, the input tensors adopt a packed shape of [t, ...];
            otherwise, they maintain the unpacked shape [b, s, ...]. In subsequent code comments,
            we uniformly use [num_tokens, ...] to denote [b, s, ...] or [t, ...] for two cases.
            """
            if self.config.q_lora_rank is not None:
                # q_compressed: [num_tokens, q_lora_rank]
                # q: [num_tokens, n * (qk_nope_head_dim + qk_rope_head_dim)]
                q, _ = self.q_b_proj(q_compressed)
            else:
                # q_compressed: [num_tokens, hidden_size]
                # q: [num_tokens, n * (qk_nope_head_dim + qk_rope_head_dim)]
                q, _ = self.q_proj(q_compressed)

            # q: [num_tokens, n, q_head_dim]
            q = q.view(
                *q.size()[:-1],
                self.num_attention_heads_per_partition,
                self.q_head_dim,
            )

            # kv: [num_tokens, n * (qk_nope_head_dim + v_head_dim)]
            kv, _ = self.kv_b_proj(kv_compressed)

            # kv: [num_tokens, n, (qk_nope_head_dim + v_head_dim)]
            kv = kv.view(
                *kv.size()[:-1],
                self.num_attention_heads_per_partition,
                self.config.qk_nope_head_dim + self.config.v_head_dim,
            )

            # [num_tokens, qk_rope_head_dim] -> [num_tokens, 1, qk_rope_head_dim]
            k_pos_emb = paddle.unsqueeze(k_pos_emb, -2)

            if self.config.apply_rope_fusion:
                cp_rank = self.pg_collection.cp.rank()
                cp_size = self.pg_collection.cp.size()
                query = fused_apply_mla_rope_for_q(
                    q,
                    rotary_pos_cos,
                    rotary_pos_sin,
                    self.config.qk_nope_head_dim,
                    self.config.qk_rope_head_dim,
                    cu_seqlens_q,
                    cp_rank,
                    cp_size,
                )
                key, value = fused_apply_mla_rope_for_kv(
                    kv,
                    k_pos_emb,
                    rotary_pos_cos,
                    rotary_pos_sin,
                    self.config.qk_rope_head_dim,
                    self.config.qk_nope_head_dim,
                    self.config.v_head_dim,
                    cu_seqlens_kv,
                    cp_rank,
                    cp_size,
                )
            else:
                # Determine seq length:
                #   packed 3D [t, n, d]      -> dim 0
                #   SP     4D [s, b, n, d]   -> dim 0
                #   normal 4D [b, s, n, d]   -> dim 1
                if q.ndim == 3 or self.config.sequence_parallel:
                    q_len = q.size(0)
                else:
                    q_len = q.size(1)
                # rotary_pos_emb: squeeze [1, seq_len, 1, headdim]

                if (
                    packed_seq_params is None
                    or self.config.context_parallel_size == 1
                ):
                    # During training, the sequence length is always
                    # the full rotary_pos_emb length, except for sequence packing + CP.
                    # We need the full rotary_pos_emb to cover the full sequence,
                    # so we do not shorten it here.
                    rotary_pos_emb = rotary_pos_emb[:, 0:q_len]

                # q_no_pe: [num_tokens, n, qk_nope_head_dim]
                # q_pos_emb: [num_tokens, n, qk_rope_head_dim]
                q_no_pe, q_pos_emb = paddle.split(
                    q,
                    [
                        self.config.qk_nope_head_dim,
                        self.config.qk_rope_head_dim,
                    ],
                    axis=-1,
                )

                # k_no_pe: [num_tokens, n, qk_nope_head_dim]
                # value: [num_tokens, n, v_head_dim]
                k_no_pe, value = paddle.split(
                    kv,
                    [self.config.qk_nope_head_dim, self.config.v_head_dim],
                    axis=-1,
                )

                # When sequence_parallel is enabled and not packed,
                # q/k are seq-first [s, b, n, d] but rotary_pos_emb is
                # batch-first [1, s, 1, d]. Transpose to [s, 1, 1, d]
                # so broadcasting aligns correctly in _apply_rotary_pos_emb_bshd.
                if self.config.sequence_parallel and rotary_pos_emb.ndim == 4:
                    rotary_pos_emb = rotary_pos_emb.transpose([1, 0, 2, 3])

                if self.config.gpt_model_use_experimental_version:
                    # EC-compatible RoPE: complex rotation, no YaRN, no mscale
                    from paddlefleet.transformer.transformer_layer import (
                        TransformerLayer,
                    )

                    _log = TransformerLayer._log_md5
                    _log(q_pos_emb, "mla_q_pe_before_rope", self.layer_number)
                    _log(k_pos_emb, "mla_k_pe_before_rope", self.layer_number)
                    q_pos_emb, k_pos_emb = _ec_compatible_rope_apply(
                        q_pos_emb,
                        k_pos_emb,
                        q_len,
                        rope_base=self.config.rope_theta,  # Must match EC's config.rope_theta
                    )
                    _log(q_pos_emb, "mla_q_pe_after_rope", self.layer_number)
                    _log(k_pos_emb, "mla_k_pe_after_rope", self.layer_number)
                else:
                    # q_pos_emb: [num_tokens, n, qk_rope_head_dim]
                    q_pos_emb = apply_rotary_pos_emb(
                        q_pos_emb,
                        rotary_pos_emb,
                        rotary_pos_cos,
                        rotary_pos_sin,
                        config=self.config,
                        cu_seqlens=cu_seqlens_q,
                        mscale=mscale,
                        cp_group=self.pg_collection.cp,
                    )
                    # k_pos_emb:[num_tokens, 1, qk_rope_head_dim]
                    k_pos_emb = apply_rotary_pos_emb(
                        k_pos_emb,
                        rotary_pos_emb,
                        rotary_pos_cos,
                        rotary_pos_sin,
                        config=self.config,
                        cu_seqlens=cu_seqlens_kv,
                        mscale=mscale,
                        cp_group=self.pg_collection.cp,
                    )

                # query: [num_tokens, n, (qk_nope_head_dim + qk_rope_head_dim)]
                query = paddle.cat([q_no_pe, q_pos_emb], axis=-1)

                # key: [num_tokens, n, (qk_nope_head_dim + qk_rope_head_dim)]
                if k_pos_emb.ndim == 4:
                    k_pos_emb = k_pos_emb.expand(
                        -1, -1, self.num_attention_heads_per_partition, -1
                    )
                else:
                    assert k_pos_emb.ndim == 3
                    k_pos_emb = k_pos_emb.expand(
                        -1, self.num_attention_heads_per_partition, -1
                    )
                key = paddle.cat([k_no_pe, k_pos_emb], axis=-1)

            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

            return query, key, value

        query, key, value = qkv_up_proj_and_rope_apply(
            q_compressed, kv_compressed, k_pos_emb, rotary_pos_emb
        )

        return query, key, value, q_compressed, kv_compressed

    def backward_dw(self) -> NoReturn:
        """Execute weight gradient computation"""
        self._backward_kv_proj()
        self._backward_q_proj()
        self._backward_output_proj()
        # GATE backward?

    def _backward_kv_proj(self):
        """Computes weight gradients of KV projection layers"""
        self.kv_b_proj.backward_dw()
        self.kv_a_proj_with_mqa.backward_dw()

    def _backward_q_proj(self):
        """Computes weight gradients of Q projection layers"""
        if self.config.q_lora_rank is None:
            self.q_proj.backward_dw()
        else:
            self.q_a_proj.backward_dw()
            self.q_b_proj.backward_dw()

    def _backward_output_proj(self):
        """Computes weight gradients of output projection layer"""
        self.o_proj.backward_dw()
