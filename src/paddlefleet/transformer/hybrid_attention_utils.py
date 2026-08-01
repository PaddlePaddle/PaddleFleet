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

"""Canonical layer indexing for DSV4 layer-wise hybrid attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple, Protocol

HybridAttentionKind = Literal["mla", "hca", "csa", "window"]


class HybridAttentionLayerInfo(NamedTuple):
    """Resolved identity of one decoder or MTP attention layer."""

    logical_index: int
    layer_kind: HybridAttentionKind
    compress_ratio: int


@dataclass(frozen=True)
class LayerAttentionConfig:
    """Immutable dimensions for one hybrid-attention layer.

    DSV4 and MLA intentionally use disjoint QK dimension namespaces. Fields
    which do not belong to the selected attention family are left as ``None``.
    """

    layer_kind: HybridAttentionKind
    logical_index: int
    compress_ratio: int
    q_lora_rank: int
    v_head_dim: int
    num_attention_heads: int
    num_key_value_heads: int
    qk_pos_emb_head_dim: int | None = None
    kv_lora_rank: int | None = None
    qk_nope_head_dim: int | None = None
    qk_rope_head_dim: int | None = None
    hybrid_index_n_heads: int | None = None
    hybrid_index_head_dim: int | None = None
    hybrid_index_topk: int | None = None


class _HybridAttentionConfig(Protocol):
    num_hidden_layers: int
    num_empty_layers_add_in_head: int
    num_nextn_predict_layers: int | None
    mtp_num_layers: int | None
    csa_compress_ratios: list[int] | None


def resolve_layer_attention_config(
    config: _HybridAttentionConfig,
    layer_number: int,
    is_mtp_layer: bool = False,
) -> LayerAttentionConfig:
    """Resolve immutable, family-specific dimensions for one layer."""
    info = resolve_hybrid_attention_layer(config, layer_number, is_mtp_layer)
    if info.layer_kind == "mla":
        field_names = (
            "hybrid_mla_q_lora_rank",
            "hybrid_mla_kv_lora_rank",
            "hybrid_mla_qk_nope_head_dim",
            "hybrid_mla_qk_rope_head_dim",
            "hybrid_mla_v_head_dim",
            "hybrid_mla_num_attention_heads",
            "hybrid_mla_num_key_value_heads",
        )
        dimensions = {name: getattr(config, name, None) for name in field_names}
        invalid = [
            name
            for name, value in dimensions.items()
            if not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
        ]
        if invalid:
            raise ValueError(
                "hybrid MLA dimensions must be explicit positive integers; "
                f"invalid fields: {', '.join(invalid)}"
            )
        hybrid_index_dimensions = {
            "hybrid_index_n_heads": getattr(
                config, "hybrid_index_n_heads", None
            ),
            "hybrid_index_head_dim": getattr(
                config, "hybrid_index_head_dim", None
            ),
            "hybrid_index_topk": getattr(config, "hybrid_index_topk", None),
        }
        invalid_hybrid_index = [
            name
            for name, value in hybrid_index_dimensions.items()
            if value is not None
            and (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            )
        ]
        if invalid_hybrid_index:
            raise ValueError(
                "hybrid MLA indexer dimensions must be positive integers when set; "
                f"invalid fields: {', '.join(invalid_hybrid_index)}"
            )
        tp_size = getattr(config, "tensor_model_parallel_size", 1) or 1
        if tp_size != 1:
            raise ValueError(
                "hybrid MLA currently requires tensor_model_parallel_size=1, "
                f"got {tp_size}"
            )
        return LayerAttentionConfig(
            layer_kind=info.layer_kind,
            logical_index=info.logical_index,
            compress_ratio=info.compress_ratio,
            q_lora_rank=dimensions["hybrid_mla_q_lora_rank"],
            kv_lora_rank=dimensions["hybrid_mla_kv_lora_rank"],
            qk_nope_head_dim=dimensions["hybrid_mla_qk_nope_head_dim"],
            qk_rope_head_dim=dimensions["hybrid_mla_qk_rope_head_dim"],
            v_head_dim=dimensions["hybrid_mla_v_head_dim"],
            num_attention_heads=dimensions["hybrid_mla_num_attention_heads"],
            num_key_value_heads=dimensions["hybrid_mla_num_key_value_heads"],
            hybrid_index_n_heads=hybrid_index_dimensions[
                "hybrid_index_n_heads"
            ],
            hybrid_index_head_dim=hybrid_index_dimensions[
                "hybrid_index_head_dim"
            ],
            hybrid_index_topk=hybrid_index_dimensions["hybrid_index_topk"],
        )

    qk_pos_emb_head_dim = getattr(config, "qk_pos_emb_head_dim", None) or 0
    v_head_dim = getattr(config, "v_head_dim", None)
    if v_head_dim is None:
        v_head_dim = config.head_dim
    if not 0 <= qk_pos_emb_head_dim <= v_head_dim:
        raise ValueError(
            "DSV4 qk_pos_emb_head_dim must be within v_head_dim, got "
            f"{qk_pos_emb_head_dim} and {v_head_dim}"
        )
    return LayerAttentionConfig(
        layer_kind=info.layer_kind,
        logical_index=info.logical_index,
        compress_ratio=info.compress_ratio,
        q_lora_rank=config.q_lora_rank,
        v_head_dim=v_head_dim,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        qk_pos_emb_head_dim=qk_pos_emb_head_dim,
    )


def get_effective_mtp_layers(config: _HybridAttentionConfig) -> int:
    """Return the explicit MTP count or the next-N fallback.

    Positive explicit and fallback counts describe the same model boundary and
    must therefore agree when both are set.
    """
    mtp_num_layers = getattr(config, "mtp_num_layers", 0) or 0
    nextn_num_layers = getattr(config, "num_nextn_predict_layers", 0) or 0
    if (
        mtp_num_layers > 0
        and nextn_num_layers > 0
        and mtp_num_layers != nextn_num_layers
    ):
        raise ValueError(
            "mtp_num_layers and num_nextn_predict_layers must be equal when "
            f"both are positive, got {mtp_num_layers} and {nextn_num_layers}"
        )
    return mtp_num_layers if mtp_num_layers > 0 else nextn_num_layers


def resolve_hybrid_attention_layer(
    config: _HybridAttentionConfig,
    layer_number: int,
    is_mtp_layer: bool = False,
) -> HybridAttentionLayerInfo:
    """Resolve a runtime layer number to its zero-based logical identity.

    Decoder ``layer_number`` is physical and includes head empty layers. MTP
    ``layer_number`` is zero-based within the MTP block and never includes the
    head-empty offset.
    """
    if not isinstance(layer_number, int) or isinstance(layer_number, bool):
        raise TypeError(
            f"layer_number must be an integer, got {layer_number!r}"
        )

    if is_mtp_layer:
        effective_mtp_layers = get_effective_mtp_layers(config)
        if not 0 <= layer_number < effective_mtp_layers:
            raise IndexError(
                f"MTP layer_number {layer_number} is outside [0, {effective_mtp_layers})"
            )
        logical_index = config.num_hidden_layers + layer_number
    else:
        head_offset = getattr(config, "num_empty_layers_add_in_head", 0) or 0
        logical_index = layer_number - head_offset
        if not 0 <= logical_index < config.num_hidden_layers:
            raise IndexError(
                f"decoder layer_number {layer_number} resolves to logical index "
                f"{logical_index}, outside [0, {config.num_hidden_layers})"
            )

    ratios = config.csa_compress_ratios
    if ratios is None:
        raise ValueError(
            "csa_compress_ratios must be set for DSV4 hybrid attention"
        )
    if logical_index >= len(ratios):
        raise IndexError(
            f"logical layer index {logical_index} has no csa_compress_ratios entry "
            f"(length {len(ratios)})"
        )

    ratio = ratios[logical_index]
    if not isinstance(ratio, int) or isinstance(ratio, bool):
        raise ValueError(
            f"csa_compress_ratios[{logical_index}]={ratio!r} must be an integer"
        )
    if ratio == -2:
        layer_kind: HybridAttentionKind = "mla"
    elif ratio == 128:
        layer_kind = "hca"
    elif 2 <= ratio < 128:
        layer_kind = "csa"
    elif ratio == 0:
        layer_kind = "window"
    else:
        raise ValueError(
            f"csa_compress_ratios[{logical_index}]={ratio!r} does not identify "
            "an MLA, HCA, CSA, or window layer"
        )

    return HybridAttentionLayerInfo(logical_index, layer_kind, ratio)
