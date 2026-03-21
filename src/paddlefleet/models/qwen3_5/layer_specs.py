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

"""Layer specs for Qwen3.5 models (vision encoder + hybrid language decoder)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from paddlefleet.transformer.enums import AttnMaskType
from paddlefleet.transformer.paddle_norm import (
    WrappedPaddleNorm,
    WrappedPaddleNormPipe,
)

from ...spec_utils import LayerSpec
from ..backends import LocalSpecProvider
from ..common.embeddings.rotary_pos_embedding import (
    RotaryEmbedding,
)
from ..common.empty_layer import EmptyLayer
from ..gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_spec,
)
from ..qwen3_vl.embedding import VisionEmbedding, VisionEmbeddingSpec
from ..qwen3_vl.patch_merger import (
    Qwen3VLVisionPatchMergerSpec,
    Qwen3VLVisionPathMerger,
)
from .qwen3_5_model import (
    Qwen3_5RMSNorm,
    Qwen3_5RMSNormPipe,
    Qwen3_5VisionModel,
    Qwen3_5VisionSublayersSpec,
)

if TYPE_CHECKING:
    from paddlefleet.models.gpt import GPTConfig

    from ...transformer.transformer_config import TransformerConfig


# ======================================================================
# Vision model specs
# ======================================================================


def get_qwen3_5_vision_spec(config: TransformerConfig) -> LayerSpec:
    """Build the complete Qwen3.5 vision model spec."""
    backend = LocalSpecProvider()

    # --- Empty layers for pipeline parallel padding ---
    empty_layer_spec = LayerSpec(
        layer=EmptyLayer, extra_kwargs={"config": config}
    )
    head_empty_layers = [empty_layer_spec] * config.num_empty_layers_add_in_head
    tail_empty_layers = [empty_layer_spec] * config.num_empty_layers_add_in_tail

    # --- Transformer encoder layers ---
    head_offset = config.num_empty_layers_add_in_head
    transformer_layers = [
        get_gpt_layer_local_spec(
            config=config,
            layer_number=i + head_offset,
            attn_mask_type=AttnMaskType.no_mask,
        )
        for i in range(config.num_hidden_layers)
    ]

    # --- Vision embedding with rotary position embedding ---
    embedding_spec = LayerSpec(
        layer=VisionEmbedding,
        sublayers_spec=VisionEmbeddingSpec(
            rope_embedding=LayerSpec(
                layer=RotaryEmbedding,
                extra_kwargs={
                    "head_dim": config.head_dim // 2,
                    "rotary_base": config.rope_theta,
                    "rope_scaling": config.rope_scaling,
                    "rotary_percent": config.rotary_percent,
                },
            )
        ),
        extra_kwargs={"config": config},
    )

    # --- Patch merger ---
    config.merger_hidden_size = config.hidden_size * (
        config.spatial_merge_size**2
    )
    merger_spec = LayerSpec(
        layer=Qwen3VLVisionPathMerger,
        sublayers_spec=Qwen3VLVisionPatchMergerSpec(
            norm=backend.layer_norm(
                rms_norm=(config.normalization == "RMSNorm"), for_qk=False
            ),
        ),
        extra_kwargs={
            "config": config,
            "dim": config.out_hidden_size,
            "context_dim": config.hidden_size,
        },
    )

    # --- Assemble full vision model spec ---
    return LayerSpec(
        layer=Qwen3_5VisionModel,
        extra_kwargs={"config": config, "modal": "vision_model"},
        sublayers_spec=Qwen3_5VisionSublayersSpec(
            embedding=embedding_spec,
            head_empty_layers=head_empty_layers,
            transformer_layers=transformer_layers,
            tail_empty_layers=tail_empty_layers,
            merger=merger_spec,
        ),
    )


# ======================================================================
# Language model (hybrid decoder) specs
# ======================================================================


def get_qwen3_5_language_spec(config: GPTConfig) -> LayerSpec:
    """Build the complete Qwen3.5 language model spec.

    1. Creates per-layer transformer specs via ``get_gpt_layer_local_spec``,
       mapping Qwen3.5 layer types (``"full_attention"`` / ``"linear_attention"``)
       to the corresponding ``attention_layer_type``.
    2. Assembles the full ``GPTModel`` spec via ``get_gpt_spec``.
    """
    # -- Step 1: build transformer layer specs --------------------------------
    layer_types = getattr(config, "layer_types", None)
    if layer_types is None:
        layer_types = ["full_attention"] * config.num_hidden_layers

    # --- Empty layers for pipeline parallel padding ---
    empty_layer_spec = LayerSpec(
        layer=EmptyLayer, extra_kwargs={"config": config}
    )
    head_empty_layers = [empty_layer_spec] * config.num_empty_layers_add_in_head
    tail_empty_layers = [empty_layer_spec] * config.num_empty_layers_add_in_tail

    head_offset = getattr(config, "num_empty_layers_add_in_head", 0)

    LAYER_TYPE_MAP = {
        "full_attention": "self_attention",
        "linear_attention": "gated_delta_net",
    }

    transformer_layers_spec = []
    for i, lt in enumerate(layer_types):
        attn_type = LAYER_TYPE_MAP.get(lt)
        if attn_type is None:
            raise ValueError(f"Unknown layer type: {lt!r} at index {i}")
        spec = get_gpt_layer_local_spec(
            config=config,
            normalization=config.normalization,
            layer_number=i + head_offset,
            attention_layer_type=attn_type,
            num_experts=config.n_routed_experts,
            moe_grouped_gemm=config.moe_grouped_gemm,
        )

        # Post-process: replace standard RMSNorm with 1-centered
        # Qwen3_5RMSNorm for decoder layer norms and QK norms.
        # GatedDeltaNet out_norm is left unchanged (already equivalent).
        sub = spec.sublayers_spec
        if sub.input_layernorm is WrappedPaddleNorm:
            sub.input_layernorm = Qwen3_5RMSNorm
        if sub.post_attention_layernorm is WrappedPaddleNorm:
            sub.post_attention_layernorm = Qwen3_5RMSNorm

        # Replace q_norm / k_norm in self-attention (only for
        # full_attention layers where they are WrappedPaddleNorm)
        attn_spec = sub.self_attn
        if hasattr(attn_spec, "sublayers_spec"):
            attn_sub = attn_spec.sublayers_spec
            if (
                hasattr(attn_sub, "q_norm")
                and attn_sub.q_norm is WrappedPaddleNorm
            ):
                attn_sub.q_norm = Qwen3_5RMSNorm
            if (
                hasattr(attn_sub, "k_norm")
                and attn_sub.k_norm is WrappedPaddleNorm
            ):
                attn_sub.k_norm = Qwen3_5RMSNorm

        transformer_layers_spec.append(spec)

    # -- Step 2: assemble full language model spec via get_gpt_spec -----------
    full_spec = get_gpt_spec(
        config=config,
        transformer_layers_spec=transformer_layers_spec,
        mtp_layers_spec=None,
        vocab_size=config.vocab_size,
        max_sequence_length=config.max_sequence_length,
        head_empty_layers_spec=head_empty_layers,
        tail_empty_layers_spec=tail_empty_layers,
        position_embedding_type=config.position_embedding_type,
        rotary_percent=config.rotary_percent,
        rotary_base=config.rotary_base,
        rope_scaling=config.rope_scaling,
        parallel_output=config.parallel_output,
        tie_word_embeddings=config.tie_word_embeddings,
    )

    # Post-process: replace final layer norm with 1-centered variant
    final_norm_spec = full_spec.sublayers_spec.layer_norm
    if final_norm_spec.layer is WrappedPaddleNormPipe:
        final_norm_spec.layer = Qwen3_5RMSNormPipe

    return full_spec
