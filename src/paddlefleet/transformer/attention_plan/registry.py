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

"""Field-ownership registry: single source of truth for which attention
family reads which config field.

This is the extension point for future field registration (P2/P4): add
entries to ``_OWNER_FIELDS`` below -- no other module needs to change.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Literal

# ---------------------------------------------------------------------------
# (1) AttentionFieldRegistry -- field-ownership registry (single source of
#     truth, initial version)
# ---------------------------------------------------------------------------

Owner = Literal[
    "common",  # common fields, readable by all attention families
    "mla",  # MLA only (including dsv4 -2 layers)
    "dsv4",  # dsv4_hybrid variant only (CSA/HCA/window layers)
    "dsa",  # DSA indexer only
    "vha",  # VHA only
    "swa",  # SWA layer override only
    "business",  # cross-repo business fields, not read by PaddleFleet
    # (explicitly isolated, e.g. use_attn_merger)
    "alias",  # deprecated aliases (normalized by the Normalizer)
]


@dataclass(frozen=True)
class FieldSpec:
    """A single field-ownership declaration."""

    name: str
    owner: Owner
    type: str = "unknown"  # "int" / "float" / "bool" / "str" / "list"
    aliases: tuple[str, ...] = ()
    deprecated: bool = False
    applies_to: str = ""  # applicability description (human-readable +
    # input for the unconsumed-field check)


# The registry is organized by owner: one registry block per owner, one field
# per line. Entry format: (field name, type, applicability description), with
# an optional 4th dict element carrying extras such as aliases/deprecated.
# The first version registers only the governed fields (A/B/C/D series);
# full registration is extended in P2/P4.
_OWNER_FIELDS: dict[Owner, list[tuple]] = {
    "common": [
        # --- variant selection (A series) ---
        (
            "experimental_attention_variant",
            "str",
            "非 None 时抢占 MLA flag 与 attention_layer_type",
        ),
        (
            "multi_latent_attention",
            "bool",
            "仅当 experimental_attention_variant != dsv4_hybrid 时生效",
        ),
        ("layer_types", "list", "仅非 dsv4 路径逐层类型来源"),
        # --- RoPE basics (B series) ---
        ("rope_theta", "float", "唯一 RoPE base 主字段（B2 归一后）"),
        ("rope_type", "str", "MLA/DSA 路径的 rope|yarn 选择；dsv4 按层覆盖"),
        (
            "rotary_interleaved",
            "bool",
            "唯一 interleave 主开关；DSv4/CSA eager 路径现状忽略",
        ),
        ("rotary_percent", "float", "标准路径推导 rope 宽度（derived）"),
        # --- QK Norm ---
        (
            "use_qk_norm",
            "bool",
            "self/MLA 路径的 q/k LayerNorm 开关；变体：qk_l2_norm（L2 范数，"
            "MLA 不支持）、qk_norm_fusion（triton 融合）；dsv4 路径另读 "
            "qk_layernorm（默认 True）",
        ),
    ],
    "alias": [
        (
            "rotary_base",
            "float",
            "rope_theta 的废弃别名（仅 dsv4/kimi_k3 路径读取）。B2 "
            "(rope_config 分支) 合入后 __post_init__ 校验并剥离；当前 "
            "develop 上仍是普通字段，Normalizer 不做归一以免改变行为",
            {"deprecated": True},
        ),
    ],
    "mla": [
        ("q_lora_rank", "int", ""),
        ("kv_lora_rank", "int", ""),
        ("qk_nope_head_dim", "int", ""),
        (
            "qk_rope_head_dim",
            "int",
            "MLA 路径 rope 宽度；DSv4/CSA 层不读（V-ROPE-01）",
        ),
    ],
    "dsv4": [
        (
            "qk_pos_emb_head_dim",
            "int",
            "DSv4/CSA/HCA/window 层 rope 宽度；None → 0 即 RoPE 关闭（V-ROPE-02）",
        ),
        (
            "csa_compress_rotary_base",
            "float",
            "压缩层（ratio > 1）base；JSON 里常为字符串（B5 强转）",
        ),
        ("hca_rope_type", "str", "ratio == 128 层的 rope|yarn 覆盖"),
        ("csa_rope_type", "str", "2 <= ratio < 128 层的 rope|yarn 覆盖"),
        (
            "csa_compress_ratios",
            "list",
            "dsv4_hybrid 下的逐层类型来源（覆盖 layer_types，A2）",
        ),
        ("csa_window_size", "int", "CSA/window 层内部窗口"),
    ],
    "dsa": [
        (
            "dsa_index_n_heads",
            "int",
            "非 None 时 MLA core 切到 DSA",
            {"aliases": ("index_n_heads",)},
        ),
        ("dsa_index_head_dim", "int", "", {"aliases": ("index_head_dim",)}),
        ("dsa_index_topk", "int", "", {"aliases": ("index_topk",)}),
        (
            "dsa_indexer_rotary_interleaved",
            "bool",
            "indexer 独立布局开关（B4）",
        ),
    ],
    "vha": [
        (
            "use_vha_attention",
            "bool",
            "同一开关三种语义：self family 换整个类为 SelfAttentionVHA"
            "（gpt_layer_specs:230）；mla/MQA 与 dsv4 层只复用为 postmix"
            "（multi_latent_attention:667、dsv4_hybrid_attention:913）+ 可选 "
            "premix（use_vha_premix AND，dsv4:1659）；gdn/kda/gemma4 不读",
        ),
        (
            "use_vha_premix",
            "bool",
            "仅 dsv4 CSA/HCA/window 层生效，且需 use_vha_attention 同时为 True",
        ),
    ],
    "swa": [
        ("swa_qk_rope_head_dim", "int", ""),
        ("swa_rope_theta", "float", ""),
    ],
    "business": [
        ("use_attn_merger", "bool", "跨仓业务字段，PaddleFleet 不读取"),
    ],
}


def _build_field_registry() -> dict[str, FieldSpec]:
    registry: dict[str, FieldSpec] = {}
    for owner, entries in _OWNER_FIELDS.items():
        for entry in entries:
            name, type_, applies_to, *extra = entry
            spec = FieldSpec(
                name=name,
                owner=owner,
                type=type_,
                applies_to=applies_to,
                **(extra[0] if extra else {}),
            )
            registry[name] = spec
    return registry


ATTENTION_FIELD_REGISTRY: dict[str, FieldSpec] = _build_field_registry()

_ALIAS_TO_CANONICAL: dict[str, str] = {
    alias: spec.name
    for spec in ATTENTION_FIELD_REGISTRY.values()
    for alias in spec.aliases
}


def _explicitly_set(config: Any, name: str) -> bool:
    """Whether the field is explicitly set to a non-default value
    (comparison against the dataclass default).

    ``__post_init__`` derives some fields in place (e.g. ``swa_head_dim``),
    so this can only approximate "current value != dataclass default".
    In the warn-only phase this approximation only affects warning noise,
    not correctness.
    """
    if not hasattr(config, name):
        return False
    value = getattr(config, name)
    for f in dataclasses.fields(config):
        if f.name != name:
            continue
        if f.default is not dataclasses.MISSING:
            return value != f.default
        if f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            return value != f.default_factory()  # type: ignore[misc]
    # TransformerConfig is a dataclass, so this is normally unreachable;
    # treat non-dataclass fields as explicitly set.
    return True
