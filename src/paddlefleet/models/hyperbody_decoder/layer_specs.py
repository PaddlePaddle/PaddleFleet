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

"""Layer specs for the HyperBody decoder backbone -- a **component** provided on the PaddleFleet side.

## Split of responsibilities

| Side | Responsibility |
|---|---|
| **PaddleFleet (this file)** | Provides the ``LayerSpec`` list and block spec for the backbone |
| PaddleFormers `transformers/hyperbody_decoder/` | HF-style config -> ``GPTConfig`` conversion + model assembly |

Geometry constants and ``GPTConfig`` assembly do not live here; they belong to the
"HF-style config to GPTConfig conversion" responsibility on the PaddleFormers side.

## Backbone shape

The HyperBody decoder is a DeepSeekV2-Lite MoE backbone: hidden 1280 / MHA 10x128 /
dense FFN 6848 (layer 0 only) / 64 experts top-6 / moe FFN 896 / 2 shared experts /
RMSNorm 1e-6 / RoPE base 10000. Each layer is built by looping over
``get_gpt_layer_local_spec``. :func:`_is_moe_layer` decides per-layer whether the
MLP is dense or MoE by reading the per-layer ``moe_layer_freq`` list.

The attention mask type is ``AttnMaskType.causal`` -- this is a standard causal LM,
so ``DotProductAttention``'s SDPA branch forcing ``is_causal=True`` is exactly right.
The per-layer norms use whatever ``get_gpt_layer_local_spec`` gives for
``normalization="RMSNorm"``.

## Why the spec construction is collected here

The generic path in ``paddlefleet.gpt_builders.gpt_builder`` (which internally calls
``get_gpt_decoder_layers_spec``) would already work behaviorally. Collecting the spec
construction explicitly here gives one practical benefit: when ``moe_layer_freq`` is
passed as an int, ``get_gpt_decoder_layers_spec`` (``gpt_layer_specs.py:803-807``)
generates the dense/MoE pattern via ``i % N``, which is not the intended per-layer
semantics for this model. :func:`_is_moe_layer` **rejects** ints outright, turning
that into an explicit error instead of a silent mismatch.
"""

from __future__ import annotations

from paddle.distributed.fleet.meta_parallel import LayerSpec

from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from paddlefleet.transformer.enums import AttnMaskType
from paddlefleet.transformer.transformer_block import (
    TransformerBlockSublayersSpec,
)
from paddlefleet.transformer.transformer_config import TransformerConfig

__all__ = [
    "get_hyperbody_decoder_layer_specs",
    "get_hyperbody_decoder_block_spec",
]


def _is_moe_layer(config: TransformerConfig, layer_idx: int) -> bool:
    """Whether layer ``i`` is a MoE layer.

    ``moe_layer_freq`` is a **per-layer 0/1 list** (``[0] + [1]*(L-1)``), so the
    criterion is simply taking element ``i`` of the list.

    WARNING: this **rejects int**. Passing an int would make Paddle take the
    ``i % N`` path (``gpt_layer_specs.py:803-807``), which is not the intended
    per-layer semantics for this model -- an inevitable mismatch.
    """
    freq = config.moe_layer_freq
    if isinstance(freq, (list, tuple)):
        if len(freq) != config.num_hidden_layers:
            raise ValueError(
                f"moe_layer_freq length {len(freq)} != num_hidden_layers "
                f"{config.num_hidden_layers}"
            )
        return bool(freq[layer_idx])
    raise TypeError(
        f"HyperBody decoder requires moe_layer_freq to be a per-layer list, got {type(freq).__name__}. "
        "Passing an int would take the `i % N` semantics, an inevitable mismatch."
    )


def get_hyperbody_decoder_layer_specs(
    config: TransformerConfig,
) -> list[LayerSpec]:
    """Produce a :class:`LayerSpec` list of ``num_hidden_layers`` layers.

    Layer 0 is a dense FFN (``intermediate_size``); the rest are MoE
    (``n_routed_experts`` x ``moe_intermediate_size`` + shared experts).

    ``moe_expert_fusion`` only applies to MoE layers -- dense layers pass ``False``,
    otherwise ``get_mlp_layer_spec_for_backend`` would build the 3-D grouped-GEMM
    weights.
    """
    if config.multi_latent_attention:
        raise ValueError(
            "HyperBody decoder is pure MHA (num_query_groups == num_attention_heads). "
            "multi_latent_attention must be False."
        )

    specs: list[LayerSpec] = []
    for i in range(config.num_hidden_layers):
        is_moe = _is_moe_layer(config, i)
        specs.append(
            get_gpt_layer_local_spec(
                config=config,
                layer_number=i + config.num_empty_layers_add_in_head,
                num_experts=config.n_routed_experts if is_moe else None,
                moe_expert_fusion=config.moe_expert_fusion if is_moe else False,
                use_qk_norm=config.use_qk_norm,
                normalization=config.normalization,
                # Standard causal LM.
                attn_mask_type=AttnMaskType.causal,
            )
        )
    return specs


def get_hyperbody_decoder_block_spec(
    config: TransformerConfig,
) -> TransformerBlockSublayersSpec:
    """Produce the spec for the whole block (N layers + final norm).

    ``layer_norm=None`` means we keep the one that ``TransformerBlock`` builds itself
    according to ``config.normalization``.

    This function is not used on the ``paddleformers-cli`` pipeline path (that path
    wants the bare ``list[LayerSpec]``, see ``get_hyperbody_decoder_layer_specs``);
    it is for single-card scripts / unit tests that build a ``TransformerBlock``
    directly without going through ``PipelineLayer``.
    """
    return TransformerBlockSublayersSpec(
        layer_specs=get_hyperbody_decoder_layer_specs(config),
        layer_norm=None,
    )
