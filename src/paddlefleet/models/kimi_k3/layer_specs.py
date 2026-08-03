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

"""LayerSpecs of the Kimi-K3 vision tower."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from ...fusions.fused_bias_dropout import get_bias_dropout_add
from ...spec_utils import LayerSpec
from ...transformer.attention import SelfAttention, SelfAttentionSublayersSpec
from ...transformer.identity_op import IdentityOp
from ...transformer.paddle_norm import WrappedPaddleNormPipe
from ...transformer.transformer_layer import TransformerLayerSublayersSpec
from ..backends import LocalSpecProvider
from ..common.embeddings.rotary_pos_embedding import Rope2DPosEmbRepeated
from ..gpt.gpt_layer_specs import get_mlp_layer_spec_for_backend
from .embedding import KimiK3VisionEmbedding, KimiK3VisionEmbeddingSpec
from .kimi_k3_model import (
    KimiK3VisionModel,
    KimiK3VisionSublayersSpec,
    KimiK3VisionTransformerLayer,
)
from .sd2_tpool_merge import (
    KimiK3VisionPatchMerger,
    KimiK3VisionPatchMergerSpec,
    KimiK3VisionSd2TpoolMerger,
)

if TYPE_CHECKING:
    from ...transformer.transformer_config import TransformerConfig


def get_kimi_k3_vision_head_dim(config: TransformerConfig) -> int:
    """MoonViT projects to ``qkv_hidden_size`` (1536), which is *not*
    ``hidden_size`` (1024), so the head dim cannot be derived from
    ``hidden_size // num_attention_heads``.
    """
    qkv_hidden_size = getattr(config, "qkv_hidden_size", None) or (
        config.hidden_size
    )
    assert qkv_hidden_size % config.num_attention_heads == 0, (
        f"qkv_hidden_size {qkv_hidden_size} must be divisible by "
        f"num_attention_heads {config.num_attention_heads}"
    )
    return qkv_hidden_size // config.num_attention_heads


def get_kimi_k3_vision_layer_local_spec(
    config: TransformerConfig = None,
    layer_number: int = 1,
) -> LayerSpec:
    backend = LocalSpecProvider()
    rms_norm = config.normalization == "RMSNorm"
    layer_norm = backend.layer_norm(rms_norm=rms_norm, for_qk=False)
    mlp = get_mlp_layer_spec_for_backend(backend=backend)

    return LayerSpec(
        layer=KimiK3VisionTransformerLayer,
        sublayers_spec=TransformerLayerSublayersSpec(
            input_layernorm=layer_norm,
            self_attn=LayerSpec(
                layer=SelfAttention,
                sublayers_spec=SelfAttentionSublayersSpec(
                    qkv_proj=backend.column_parallel_linear(),
                    core_attention=backend.core_attention(),
                    o_proj=backend.row_parallel_linear(),
                    # Kimi-K3 MoonViT has no qk norm.
                    q_norm=IdentityOp,
                    k_norm=IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            post_attention_layernorm=layer_norm,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
            sharded_state_dict_keys_map={
                "input_layernorm.": "self_attn.qkv_proj.layer_norm_",
                "post_attention_layernorm.": "mlp.up_gate_proj.layer_norm_",
            },
        ),
        extra_kwargs={
            "config": config,
            "layer_number": layer_number,
            "hidden_dropout_prob": config.hidden_dropout_prob
            if config is not None
            else None,
            "modal": "vision_model",
        },
    )


def get_kimi_k3_vision_encoder_layers_spec(
    config: TransformerConfig,
) -> list[LayerSpec]:
    layer_spec_func = partial(
        get_kimi_k3_vision_layer_local_spec,
        config=config,
    )
    layer_specs = []
    for layer_number in range(config.num_hidden_layers):
        real_layer_number = layer_number + config.num_empty_layers_add_in_head
        layer_specs.append(layer_spec_func(layer_number=real_layer_number))

    return layer_specs


def get_kimi_k3_vision_spec(
    config: TransformerConfig,
    transformer_layers_spec: list[LayerSpec],
    head_empty_layers_spec: list[LayerSpec] | None = None,
    tail_empty_layer_spec: list[LayerSpec] | None = None,
):
    backend = LocalSpecProvider()
    rms_norm = config.normalization == "RMSNorm"

    embedding_spec = KimiK3VisionEmbeddingSpec(
        rope_embedding=LayerSpec(
            layer=Rope2DPosEmbRepeated,
            extra_kwargs={
                "head_dim": get_kimi_k3_vision_head_dim(config),
                "max_height": getattr(config, "max_height", 512),
                "max_width": getattr(config, "max_width", 512),
                "rotary_base": getattr(config, "rotary_base", 10000),
            },
        )
    )

    merger_spec = LayerSpec(
        layer=KimiK3VisionPatchMerger,
        sublayers_spec=KimiK3VisionPatchMergerSpec(
            norm=backend.layer_norm(rms_norm=rms_norm, for_qk=False)
        ),
        extra_kwargs={"config": config},
    )

    return LayerSpec(
        layer=KimiK3VisionModel,
        sublayers_spec=KimiK3VisionSublayersSpec(
            embedding=LayerSpec(
                layer=KimiK3VisionEmbedding,
                sublayers_spec=embedding_spec,
                extra_kwargs={"config": config},
            ),
            head_empty_layers=head_empty_layers_spec,
            transformer_layers=transformer_layers_spec,
            tail_empty_layers=tail_empty_layer_spec,
            final_layernorm=LayerSpec(
                layer=WrappedPaddleNormPipe,
                extra_kwargs={
                    "config": config,
                    "hidden_size": config.hidden_size,
                    "eps": config.rms_norm_eps,
                },
            ),
            sdtpool_merger=LayerSpec(
                layer=KimiK3VisionSd2TpoolMerger,
                extra_kwargs={"config": config},
            ),
            merger=merger_spec,
        ),
        extra_kwargs={"config": config, "modal": "vision_model"},
    )
