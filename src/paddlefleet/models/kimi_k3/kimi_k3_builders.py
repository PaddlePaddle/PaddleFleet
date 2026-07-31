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
import functools

import numpy as np
from paddle.distributed.fleet.meta_parallel import (
    LayerSpec,
    build_spec_layer,
)
from paddle.nn import functional as F

from ...transformer.transformer_config import TransformerConfig
from ..common.empty_layer import EmptyLayer
from .layer_specs import (
    get_kimi_k3_vision_encoder_layers_spec,
    get_kimi_k3_vision_head_dim,
    get_kimi_k3_vision_spec,
)


def kimi_k3_vision_builder(config, **kwargs):
    """Build the Kimi-K3 vision tower from a ``TransformerConfig``."""
    # MoonViT qkv width differs from hidden_size; make the attention head dim
    # explicit before any spec is built.
    config.head_dim = get_kimi_k3_vision_head_dim(config)
    config.v_head_dim = config.head_dim

    transformer_layer_specs = get_kimi_k3_vision_encoder_layers_spec(
        config=config
    )

    head_empty_layers_spec = [
        LayerSpec(layer=EmptyLayer, extra_kwargs={"config": config})
        for _ in range(config.num_empty_layers_add_in_head)
    ]
    tail_empty_layers_spec = [
        LayerSpec(layer=EmptyLayer, extra_kwargs={"config": config})
        for _ in range(config.num_empty_layers_add_in_tail)
    ]

    res_spec = get_kimi_k3_vision_spec(
        config=config,
        head_empty_layers_spec=head_empty_layers_spec,
        transformer_layers_spec=transformer_layer_specs,
        tail_empty_layer_spec=tail_empty_layers_spec,
    )

    return build_spec_layer(res_spec, **kwargs)


def build_kimi_k3_vision_config(**overrides) -> TransformerConfig:
    """``TransformerConfig`` carrying the Kimi-K3 ``vision_config`` defaults.

    Defaults mirror ``Kimi-K3/config.json``; the K3-specific fields that are
    not part of ``TransformerConfig`` are attached afterwards.
    """
    vision_fields = {
        "patch_size": 14,
        "in_channels": 3,
        "qkv_hidden_size": 1536,
        "init_pos_emb_height": 64,
        "init_pos_emb_width": 64,
        "init_pos_emb_time": 4,
        "pos_emb_type": "divided_fixed",
        "pos_emb_interpolation_mode": "bilinear",
        "patch_embed_proj_bias": False,
        "merge_kernel_size": (2, 2),
        "mm_hidden_size": 1024,
        "text_hidden_size": 7168,
        "projector_ln_eps": 1e-5,
        "max_height": 512,
        "max_width": 512,
    }
    extra = {
        key: overrides.pop(key)
        for key in list(vision_fields)
        if key in overrides
    }
    vision_fields.update(extra)

    base = {
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_hidden_layers": 27,
        "num_attention_heads": 12,
        "normalization": "RMSNorm",
        "hidden_act": functools.partial(F.gelu, approximate=True),
        "use_bias": False,
        "gated_linear_unit": False,
        "attention_bias": False,
        "use_qk_norm": False,
        "hidden_dropout_prob": 0.0,
        "attention_dropout": 0.0,
        # HF builds the encoder norms as ``nn.RMSNorm(dim)`` without an explicit
        # eps, so torch falls back to ``finfo(dtype).eps``. Mirror the fp32 value
        # so that forward alignment against the reference is apples-to-apples.
        "rms_norm_eps": float(np.finfo(np.float32).eps),
    }
    base.update(overrides)

    config = TransformerConfig(**base)
    for key, value in vision_fields.items():
        setattr(config, key, value)
    config.head_dim = get_kimi_k3_vision_head_dim(config)
    config.v_head_dim = config.head_dim
    return config
