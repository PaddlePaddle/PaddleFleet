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

"""Flat configuration for the unified HyperBody model.

Unlike the earlier composite design (which nested a ``text_config`` +
``encoder_config`` via ``sub_configs``), this is a **single flat config**:

* decoder (language model) fields keep their **bare** names -- so
  ``HyperBodyDecoderModelProvider.from_config(this_config)`` reads them directly
  via the framework's ``register_attributes`` mechanism (same names as
  ``HyperBodyDecoderConfig``);
* HyperEncoder fields carry an **``encoder_`` prefix** -- ``modeling.py`` strips
  the prefix and rebuilds a transient ``HyperEncoderConfig`` to drive the
  encoder spec builders;
* the two bridge token ids (``image_token_id`` / ``video_token_id``) drive the
  ``gpt_embedding`` multimodal-merge scatter that splices encoder latents into
  the decoder embedding stream.

The decoder field set / ``super().__init__`` routing mirrors
``HyperBodyDecoderConfig`` exactly (llm_meta fields MUST go through
``super().__init__(**kwargs)``, else ``set_expected_keys`` resets them to
defaults -- see ``HyperBodyDecoderConfig`` docstring step 2).
"""

from __future__ import annotations

from ..configuration_utils import PretrainedConfig

__all__ = [
    "HyperBodyConfig",
    "CONTEXT_TOKEN",
    "IM_PATCH_TOKEN",
    "AUDIO_PATCH_TOKEN",
    "VIDEO_TOKEN_SENTINEL",
]

# ---------------------------------------------------------------------------
# Special tokens (single source of truth, inlined so this package is
# self-contained). ``CONTEXT_TOKEN``: wherever <context> appears in input_ids,
# the encoder latents are spliced into the decoder embedding stream.
# ``IM_PATCH_TOKEN`` / ``AUDIO_PATCH_TOKEN``: image / audio patch placeholders
# consumed by the encoder frontend when building context embeds.
# ---------------------------------------------------------------------------
IM_PATCH_TOKEN = 128815
AUDIO_PATCH_TOKEN = 128829
CONTEXT_TOKEN = 128830

# Sentinel for the (unused) video placeholder. ``gpt_embedding`` reads
# ``video_token_id`` unconditionally; -1 never matches a real token id, so the
# video branch stays inert.
VIDEO_TOKEN_SENTINEL = -1

# ---------------------------------------------------------------------------
# HyperBody decoder geometry (production values), inlined here so the flat
# config no longer imports from ``transformers/hyperbody_decoder``.
# ---------------------------------------------------------------------------
HYPERBODY_DECODER_VOCAB_SIZE = 129280
HYPERBODY_DECODER_HIDDEN_SIZE = 1280
HYPERBODY_DECODER_NUM_LAYERS = 12
HYPERBODY_DECODER_NUM_HEADS = 10
HYPERBODY_DECODER_FFN_HIDDEN = 6848  # dense FFN, layer 0 only
HYPERBODY_DECODER_MOE_FFN_HIDDEN = 896
HYPERBODY_DECODER_NUM_MOE_EXPERTS = 64
HYPERBODY_DECODER_MOE_TOPK = 6
# Shared-expert intermediate size 1792 = n_shared_experts x moe_intermediate_size
# = 2 x 896.
HYPERBODY_DECODER_NUM_SHARED_EXPERTS = 2

# HyperEncoder geometry defaults (production values), mirrored here so the flat
# config is self-contained.
_ENC_VOCAB_SIZE = 129280
_ENC_HIDDEN_SIZE = 1280
_ENC_FFN_HIDDEN = 6848
_ENC_MOE_FFN_HIDDEN = 896
_ENC_NUM_MOE_EXPERTS = 64
_ENC_NUM_LAYERS = 12
_ENC_NUM_HEADS = 10


def build_moe_layer_freq(
    num_hidden_layers: int = HYPERBODY_DECODER_NUM_LAYERS,
) -> list[int]:
    """Per-layer dense/MoE 0-1 table: ``[0] + [1]*(L-1)`` (layer 0 dense, rest MoE).

    WARNING: must be a list. Passing an int makes Paddle take ``i % N``, which is
    not the intended per-layer semantics.
    """
    if num_hidden_layers < 1:
        raise ValueError(
            f"num_hidden_layers must be >= 1, got {num_hidden_layers}"
        )
    return [0] + [1] * (num_hidden_layers - 1)


class HyperBodyConfig(PretrainedConfig):
    r"""Flat config for the unified HyperBody model (encoder + decoder in one).

    Decoder fields keep their **bare** names (identical to
    ``HyperBodyDecoderConfig``) so ``HyperBodyDecoderModelProvider.from_config``
    reads them directly. Encoder fields carry an ``encoder_`` prefix;
    ``modeling.py`` strips the prefix to rebuild a transient
    ``HyperEncoderConfig``. The two bridge token ids drive the multimodal-merge
    scatter in ``gpt_embedding``.

    The ``super().__init__`` routing of llm_meta fields mirrors
    ``HyperBodyDecoderConfig`` exactly -- llm_meta fields MUST go through
    ``super().__init__(**kwargs)``, else ``set_expected_keys`` resets them.
    """

    model_type = "hyperbody"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        # ===================== DECODER (bare names) =====================
        # === structure ===
        vocab_size: int = HYPERBODY_DECODER_VOCAB_SIZE,
        hidden_size: int = HYPERBODY_DECODER_HIDDEN_SIZE,
        num_hidden_layers: int = HYPERBODY_DECODER_NUM_LAYERS,
        num_attention_heads: int = HYPERBODY_DECODER_NUM_HEADS,
        num_key_value_heads: int | None = None,
        head_dim: int | None = None,
        intermediate_size: int = HYPERBODY_DECODER_FFN_HIDDEN,
        hidden_act: str = "silu",
        gated_linear_unit: bool = True,
        multi_latent_attention: bool = False,
        use_qk_norm: bool = False,
        normalization: str = "RMSNorm",
        rms_norm_eps: float = 1e-6,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        # === position embedding ===
        position_embedding_type: str = "rope",
        rope_theta: float = 10000,
        rotary_percent: float = 1.0,
        max_position_embeddings: int = 8192,
        # === dropout / bias ===
        attention_dropout: float = 0.0,
        hidden_dropout_prob: float = 0.0,
        use_bias: bool = False,
        attention_bias: bool = False,
        # === MoE ===
        n_routed_experts: int = HYPERBODY_DECODER_NUM_MOE_EXPERTS,
        moe_intermediate_size: int = HYPERBODY_DECODER_MOE_FFN_HIDDEN,
        num_experts_per_tok: int = HYPERBODY_DECODER_MOE_TOPK,
        n_shared_experts: int = HYPERBODY_DECODER_NUM_SHARED_EXPERTS,
        moe_layer_freq: list[int] | None = None,
        first_k_dense_replace: None = None,
        moe_token_dispatcher_type: str = "alltoall",
        moe_expert_fusion: bool = True,
        n_group: int = 1,
        topk_group: int = 1,
        router_aux_loss_coef: float = 0.001,
        scoring_func: str = "softmax",
        topk_method: str = "greedy",
        norm_topk_prob: bool = False,
        moe_router_load_balancing_type: str = "seq_aux_loss",
        moe_shared_expert_overlap: bool = True,
        routed_scaling_factor: float = 1.0,
        routed_scaling_factor_learnable: bool = False,
        # === dtype / numerics ===
        bf16: bool = True,
        attention_softmax_in_fp32: bool = False,
        fp32_residual_connection: bool = False,
        variable_seq_lengths: bool = True,
        calculate_per_token_loss: bool = False,
        # === fusion switches ===
        bias_activation_fusion: bool = True,
        masked_softmax_fusion: bool = True,
        bias_dropout_fusion: bool = True,
        apply_rope_fusion: bool = False,
        cross_entropy_loss_fusion: bool = False,
        # === initialization ===
        use_cpu_initialization: bool = False,
        use_accuracy_compatible: bool = False,
        # === parallelism (overridden at runtime by yaml) ===
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        virtual_pipeline_model_parallel_size: int = 1,
        expert_model_parallel_size: int = 1,
        context_parallel_size: int = 1,
        sequence_parallel: bool = False,
        pp_seg_method: str = "layer:TransformerLayer|EmptyLayer",
        # === misc decoder ===
        tie_word_embeddings: bool = False,
        hyperbody_context_token_id: int = CONTEXT_TOKEN,
        # ===================== ENCODER (encoder_ prefix) =====================
        encoder_vocab_size: int = _ENC_VOCAB_SIZE,
        encoder_hidden_size: int = _ENC_HIDDEN_SIZE,
        encoder_intermediate_size: int = _ENC_FFN_HIDDEN,
        encoder_num_hidden_layers: int = _ENC_NUM_LAYERS,
        encoder_num_attention_heads: int = _ENC_NUM_HEADS,
        encoder_num_key_value_heads: int | None = None,
        encoder_rms_norm_eps: float = 1e-6,
        encoder_rope_theta: float = 10000,
        encoder_attention_dropout: float = 0.0,
        encoder_hidden_dropout_prob: float = 0.0,
        encoder_attention_bias: bool = False,
        encoder_moe_intermediate_size: int = _ENC_MOE_FFN_HIDDEN,
        encoder_n_routed_experts: int = _ENC_NUM_MOE_EXPERTS,
        encoder_num_experts_per_tok: int = 6,
        encoder_n_shared_experts: int = 2,
        encoder_first_k_dense_replace: int = 1,
        encoder_routed_scaling_factor: float = 1.0,
        encoder_n_group: int = 1,
        encoder_topk_group: int = 1,
        encoder_norm_topk_prob: bool = False,
        encoder_scoring_func: str = "softmax",
        encoder_topk_method: str = "greedy",
        # hyperencoder-specific geometry
        hyperencoder_query_lengths: tuple[int, int] = (256, 8192),
        hyperencoder_seq_align: int = 128,
        hyperencoder_attn_backend: str = "dp",
        hyperencoder_packed_decoder: bool = False,
        # Decoder packed-RoPE gate, decoupled from ``hyperencoder_packed_decoder``.
        # ``None`` (default) => fall back to ``hyperencoder_packed_decoder`` so the
        # current behavior is unchanged; set explicitly to control the two knobs
        # independently.
        decoder_packed_rope: bool | None = None,
        # ===================== BRIDGE =====================
        image_token_id: int = CONTEXT_TOKEN,
        video_token_id: int = VIDEO_TOKEN_SENTINEL,
        **kwargs,
    ):
        # ================= DECODER bare fields (mirror HyperBodyDecoderConfig) =================
        # ---- structure ----
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = (
            num_key_value_heads
            if num_key_value_heads is not None
            else num_attention_heads
        )
        self.head_dim = (
            head_dim
            if head_dim is not None
            else hidden_size // num_attention_heads
        )
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        # ---- position embedding ----
        self.rope_theta = rope_theta
        self.rotary_percent = rotary_percent
        self.max_position_embeddings = max_position_embeddings
        self.max_sequence_length = max_position_embeddings
        self.seq_length = max_position_embeddings
        # ---- dropout / bias ----
        self.attention_dropout = attention_dropout
        self.hidden_dropout_prob = hidden_dropout_prob
        self.use_bias = use_bias
        self.attention_bias = attention_bias
        # ---- MoE ----
        self.n_routed_experts = n_routed_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.moe_layer_freq = (
            build_moe_layer_freq(num_hidden_layers)
            if moe_layer_freq is None
            else list(moe_layer_freq)
        )
        self.first_k_dense_replace = first_k_dense_replace
        self.n_group = n_group
        self.topk_group = topk_group
        self.scoring_func = scoring_func
        self.topk_method = topk_method
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.routed_scaling_factor_learnable = routed_scaling_factor_learnable
        # ---- dtype / numerics ----
        self.bf16 = bf16
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.variable_seq_lengths = variable_seq_lengths
        self.calculate_per_token_loss = calculate_per_token_loss
        # ---- fusion switches ----
        self.bias_activation_fusion = bias_activation_fusion
        self.masked_softmax_fusion = masked_softmax_fusion
        self.bias_dropout_fusion = bias_dropout_fusion
        self.cross_entropy_loss_fusion = cross_entropy_loss_fusion
        # ---- initialization ----
        self.use_cpu_initialization = use_cpu_initialization
        self.use_accuracy_compatible = use_accuracy_compatible
        self.pp_seg_method = pp_seg_method
        self.hyperbody_context_token_id = hyperbody_context_token_id

        # ================= ENCODER encoder_* fields (plain attributes) =================
        self.encoder_vocab_size = encoder_vocab_size
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_intermediate_size = encoder_intermediate_size
        self.encoder_num_hidden_layers = encoder_num_hidden_layers
        self.encoder_num_attention_heads = encoder_num_attention_heads
        self.encoder_num_key_value_heads = (
            encoder_num_key_value_heads
            if encoder_num_key_value_heads is not None
            else encoder_num_attention_heads
        )
        self.encoder_rms_norm_eps = encoder_rms_norm_eps
        self.encoder_rope_theta = encoder_rope_theta
        self.encoder_attention_dropout = encoder_attention_dropout
        self.encoder_hidden_dropout_prob = encoder_hidden_dropout_prob
        self.encoder_attention_bias = encoder_attention_bias
        self.encoder_moe_intermediate_size = encoder_moe_intermediate_size
        self.encoder_n_routed_experts = encoder_n_routed_experts
        self.encoder_num_experts_per_tok = encoder_num_experts_per_tok
        self.encoder_n_shared_experts = encoder_n_shared_experts
        self.encoder_first_k_dense_replace = encoder_first_k_dense_replace
        self.encoder_routed_scaling_factor = encoder_routed_scaling_factor
        self.encoder_n_group = encoder_n_group
        self.encoder_topk_group = encoder_topk_group
        self.encoder_norm_topk_prob = encoder_norm_topk_prob
        self.encoder_scoring_func = encoder_scoring_func
        self.encoder_topk_method = encoder_topk_method
        self.hyperencoder_query_lengths = tuple(hyperencoder_query_lengths)
        self.hyperencoder_seq_align = hyperencoder_seq_align
        self.hyperencoder_attn_backend = hyperencoder_attn_backend
        self.hyperencoder_packed_decoder = hyperencoder_packed_decoder
        self.decoder_packed_rope = decoder_packed_rope

        # Reject the triton frontend without packed decoding at construction time.
        # A non-packed decoder builds a dense attention mask, but the 'triton'
        # backend routes the trunk through PrefixLMTritonCore, which rejects an
        # explicit attention_mask and would otherwise only fail at the first
        # forward. Fail loudly here instead.
        #
        # Match case-insensitively to mirror ``encoder_attn_backend`` /
        # ``use_triton_encoder_attn`` (both lower-case the field before
        # comparing); otherwise ``'TRITON'`` would bypass this guard and only
        # fail at the first forward.
        if (
            str(self.hyperencoder_attn_backend).lower() == "triton"
            and not self.hyperencoder_packed_decoder
        ):
            raise ValueError(
                "hyperencoder_attn_backend='triton' requires "
                "hyperencoder_packed_decoder=True (PrefixLMTritonCore does not "
                "accept a dense attention mask)."
            )

        # ================= BRIDGE =================
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id

        # WARNING: llm_meta fields must go through kwargs so set_expected_keys does
        # not reset them (see module + HyperBodyDecoderConfig docstrings).
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            multi_latent_attention=multi_latent_attention,
            use_qk_norm=use_qk_norm,
            normalization=normalization,
            gated_linear_unit=gated_linear_unit,
            position_embedding_type=position_embedding_type,
            fp32_residual_connection=fp32_residual_connection,
            apply_rope_fusion=apply_rope_fusion,
            moe_token_dispatcher_type=moe_token_dispatcher_type,
            moe_expert_fusion=moe_expert_fusion,
            moe_router_load_balancing_type=moe_router_load_balancing_type,
            moe_shared_expert_overlap=moe_shared_expert_overlap,
            router_aux_loss_coef=router_aux_loss_coef,
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
            expert_model_parallel_size=expert_model_parallel_size,
            context_parallel_size=context_parallel_size,
            sequence_parallel=sequence_parallel,
            **kwargs,
        )
