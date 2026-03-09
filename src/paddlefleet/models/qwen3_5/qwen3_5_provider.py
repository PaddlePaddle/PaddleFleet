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

"""Config providers for Qwen3.5 models.

Providers are dataclass configs that know how to instantiate PaddleFleet model
shells via their ``provide()`` method.

* ``Qwen3_5VisionProvider`` – builds the Qwen3.5 vision encoder.
* ``Qwen3_5VLProvider`` – composes vision + language into the full VL model.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import paddle
from paddle.nn import functional as F

from ...spec_utils import LayerSpec, build_layer
from ...transformer import TransformerConfig
from ...transformer.transformer_layer import TransformerLayer
from ..common.empty_layer import EmptyLayer
from .layer_specs import (
    get_qwen3_5_language_spec,
    get_qwen3_5_vision_spec,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from .qwen3_5_model import (
        Qwen3_5Model,
        Qwen3_5VisionModel,
    )


# ======================================================================
# Vision provider
# ======================================================================


@dataclass
class Qwen3_5VisionProvider(TransformerConfig):
    patch_size: int = 16
    use_bias: bool = True
    add_qkv_bias: bool = True
    num_position_embeddings: int = 2304
    embed_dim: int = (1152,)
    hidden_size: int = 1152
    out_hidden_size: int = 3584
    in_channels: int = 3
    spatial_merge_size: int = 2
    spatial_patch_size: int = 16
    temporal_patch_size: int = 2
    hidden_dropout_prob: float = 0.0
    attention_dropout: float = 0.0
    intermediate_size: int = 4304
    initializer_range: float = 0.02
    gated_linear_unit: bool = False
    activation_func: Callable = F.gelu
    layernorm_zero_centered_gamma: bool = False
    apply_query_key_layer_scaling: bool = False
    persist_layer_norm: bool = True
    bias_activation_fusion: bool = False
    bias_dropout_fusion: bool = False
    attention_softmax_in_fp32: bool = True
    normalization: str = "LayerNorm"
    apply_rope_fusion: bool = True
    rms_norm_eps: float = 1e-6
    transformer_layer_spec: LayerSpec = TransformerLayer
    model_version: str = "qwen3_5"
    img_h: int = 336
    img_w: int = 336
    add_class_token: bool = False
    class_token_len: int = 1
    high_precision_rope: bool = True
    rotary_percent: float = 1.0
    transform_rules = {
        "dtype": "params_dtype",
        "num_heads": "num_attention_heads",
        "depth": "num_hidden_layers",
        "initializer_range": "init_method_std",
    }

    def provide(self) -> Qwen3_5VisionModel:
        pp_size = self.pipeline_model_parallel_size

        device_context = contextlib.nullcontext
        if self.init_model_with_meta_device:
            device_context = partial(paddle.device, device="meta")

        with device_context():
            spec = get_qwen3_5_vision_spec(self)
            return build_layer(
                spec,
                seg_method="layer:TransformerLayer|EmptyLayer",
                num_stages=pp_size,
            )


# ======================================================================
# VL model provider (vision + language composite)
# ======================================================================


@dataclass
class Qwen3_5VLProvider:
    """Config provider that composes vision + language into the full VL model.

    The language model is built directly from ``GPTModel`` via
    ``get_qwen3_5_language_spec``.  No separate language provider is needed.

    Usage::

        vl_provider = Qwen3_5VLProvider(
            vision_config=Qwen3_5VisionProvider(...),
            language_config=TransformerConfig(...),
        )
        model = vl_provider.provide()
    """

    vision_config: Qwen3_5VisionProvider | None = None
    language_config: TransformerConfig | None = None
    spatial_merge_size: int = 2
    image_token_id: int | None = None
    video_token_id: int | None = None

    def provide(self) -> Qwen3_5Model:
        from .qwen3_5_model import Qwen3_5Model

        vision_model = None
        if self.vision_config is not None:
            vision_model = self.vision_config.provide()

        language_model = None
        if self.language_config is not None:
            config = self.language_config
            pp_size = config.pipeline_model_parallel_size

            device_context = contextlib.nullcontext
            if config.init_model_with_meta_device:
                device_context = partial(paddle.device, device="meta")

            empty_layer_spec = LayerSpec(
                layer=EmptyLayer, extra_kwargs={"config": config}
            )
            head_empty = [empty_layer_spec] * getattr(
                config, "num_empty_layers_add_in_head", 0
            )
            tail_empty = [empty_layer_spec] * getattr(
                config, "num_empty_layers_add_in_tail", 0
            )

            spec = get_qwen3_5_language_spec(
                config=config,
                vocab_size=getattr(config, "vocab_size", 151936),
                max_sequence_length=getattr(
                    config, "max_sequence_length", 32768
                ),
                head_empty_layers_spec=head_empty,
                tail_empty_layers_spec=tail_empty,
                position_embedding_type=getattr(
                    config, "position_embedding_type", "mrope"
                ),
                rotary_percent=getattr(config, "rotary_percent", 1.0),
                rotary_base=getattr(config, "rope_theta", 10000),
                rope_scaling=getattr(config, "rope_scaling", False),
                parallel_output=getattr(config, "parallel_output", False),
                tie_word_embeddings=getattr(
                    config, "tie_word_embeddings", False
                ),
            )

            with device_context():
                language_model = build_layer(
                    spec,
                    seg_method="layer:TransformerLayer|EmptyLayer",
                    num_stages=pp_size,
                )

        return Qwen3_5Model(
            config=self.language_config,
            vision_model=vision_model,
            language_model=language_model,
            spatial_merge_size=self.spatial_merge_size,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
        )
