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

"""Attention Plan (the four normalization/validation/resolution/print steps).

After loading and before layer construction, run an explicit pipeline over the
flat ``TransformerConfig``:

    1. Normalizer  -- registry-driven lossless normalization (aliases, type
       coercion)
    2. Validator   -- combined validation of the A/B rule families (warn-only
       mode: warn but do not raise by default)
    3. Resolver    -- produce one ``AttentionExecutionPlan`` per layer
    4. Plan print  -- rank0 prints the ``[ATTN-PLAN]`` summary table + JSON
       dump to disk

This module is a sidecar: the construction chain of
gpt_layer_specs / TransformerBlock will consume the Plan produced
here.
The Resolver's plan method implements the "plan resolution" decision
tree replicates, branch by branch, the current priority order of
gpt_layer_specs.get_gpt_layer_local_spec; it is used only for observation
and snapshots and does not change any numerical behavior.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Literal

from paddlefleet.transformer.utils import is_layer_window_attention

__all__ = [
    "AttentionExecutionPlan",
    "AttentionPlanBundle",
    "ATTENTION_FIELD_REGISTRY",
    "normalize_attention_config",
    "resolve_attention_plan",
    "run_attention_plan",
]

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


# ---------------------------------------------------------------------------
# Normalizer (1) -- lossless conversion
# ---------------------------------------------------------------------------


@dataclass
class NormalizationReport:
    """Normalization result: records lossless conversions only, changing no
    semantics."""

    changes: list[dict] = field(
        default_factory=list
    )  # {field, old, new, reason}
    warnings: list[dict] = field(default_factory=list)  # {field, message}

    def to_json(self) -> dict:
        return {"changes": self.changes, "warnings": self.warnings}


_SPEC_PY_TYPES: dict[str, type | tuple[type, ...]] = {
    "int": int,
    "float": (int, float),  # int is an acceptable float source, coerced below
    "bool": bool,
    "str": str,
    "list": (list, tuple),
}

# Sentinel: coercion attempted but failed (warning already recorded).
_COERCE_FAILED = object()


def _coerce_value(
    name: str, spec_type: str, value: Any, report: NormalizationReport
) -> Any:
    """Coerce ``value`` to the declared ``spec_type``; warn-only on failure.

    Covers every mismatch between declared and actual type: numeric strings
    ("160000.0", "3"), string bools ("true"/"false"), float->int (integral
    only, lossy warned), int->str, and JSON-parseable list strings.
    """
    try:
        if spec_type in ("int", "float"):
            coerced: Any = float(value)
            if spec_type == "int":
                if coerced != int(coerced):
                    raise ValueError(f"non-integral value {value!r}")
                coerced = int(coerced)
        elif spec_type == "bool":
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in ("true", "1", "yes"):
                    return True
                if lowered in ("false", "0", "no"):
                    return False
                raise ValueError(f"non-boolean string {value!r}")
            return bool(value)
        elif spec_type == "str":
            return str(value)
        elif spec_type == "list":
            if isinstance(value, str):
                parsed = json.loads(value)
                if not isinstance(parsed, list):
                    raise ValueError(f"non-list JSON {value!r}")
                return parsed
            return list(value)
        else:  # "unknown" and anything undeclared: leave as-is
            return _COERCE_FAILED
    except (ValueError, TypeError):
        report.warnings.append(
            {
                "field": name,
                "message": (
                    f"'{name}' declared {spec_type} but got "
                    f"{type(value).__name__} {value!r} that cannot be coerced"
                ),
            }
        )
        return _COERCE_FAILED
    return coerced


def normalize_attention_config(
    config: Any, *, apply: bool = True
) -> NormalizationReport:
    """Lossless normalization driven by the registry declarations.

    Only two kinds of conversion are performed:
    - Alias normalization: when a legacy name (``index_*``) appears in the
      instance dict, map it to ``dsa_index_*`` (TransformerConfig has already
      been renamed via transform_rules; this covers direct-constructed /
      external configs);
    - Type coercion: fields declared float/int that receive numeric strings
      (e.g. ``"160000.0"``) are coerced; failure only warns (no raise in the
      warn-only phase).
    """
    report = NormalizationReport()

    # Alias normalization
    for old, canonical in _ALIAS_TO_CANONICAL.items():
        if old in getattr(config, "__dict__", {}):
            old_value = config.__dict__[old]
            if old_value is None:
                continue
            cur = getattr(config, canonical, None)
            if cur is None or cur == old_value:
                if apply and hasattr(config, canonical):
                    setattr(config, canonical, old_value)
                report.changes.append(
                    {
                        "field": old,
                        "old": old_value,
                        "new": canonical,
                        "reason": f"alias -> {canonical} (registry)",
                    }
                )
            else:
                report.warnings.append(
                    {
                        "field": old,
                        "message": (
                            f"legacy alias '{old}'={old_value!r} conflicts "
                            f"with '{canonical}'={cur!r}; alias ignored"
                        ),
                    }
                )

    # Type coercion (declared type vs actual type)
    for name, spec in ATTENTION_FIELD_REGISTRY.items():
        if not hasattr(config, name):
            continue
        value = getattr(config, name)
        if value is None or type(value) is _SPEC_PY_TYPES.get(spec.type):
            continue
        coerced = _coerce_value(name, spec.type, value, report)
        if coerced is _COERCE_FAILED:
            continue
        if apply:
            setattr(config, name, coerced)
        report.changes.append(
            {
                "field": name,
                "old": value,
                "new": coerced,
                "reason": f"{type(value).__name__} coerced to {spec.type} "
                "(registry type contract)",
            }
        )

    return report


# ---------------------------------------------------------------------------
# Validator (2) -- combined validation, warn-only mode
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    """A single validation result. Under warn-only (log_only) mode every
    severity is downgraded to warning."""

    rule_id: str  # e.g. "V-VAR-01"
    severity: Literal["error", "warning"]
    message: str
    remediation: str


def _validate(
    config: Any, layer_plans: list[AttentionExecutionPlan]
) -> list[Finding]:
    """Combined validation of the rule families. Returns a list of
    findings."""
    out: list[Finding] = []
    variant = getattr(config, "experimental_attention_variant", None)
    is_dsv4 = variant == "dsv4_hybrid"
    families = {p.family for p in layer_plans}
    decoder_plans = [p for p in layer_plans if not p.is_mtp]
    mtp_plans = [p for p in layer_plans if p.is_mtp]

    # VAR-01: dsv4_hybrid unconditionally takes over the
    # multi_latent_attention flag
    if is_dsv4 and getattr(config, "multi_latent_attention", False):
        out.append(
            Finding(
                "V-VAR-01",
                "error",
                "'multi_latent_attention=true' is set but "
                "experimental_attention_variant='dsv4_hybrid' takes precedence; "
                "per-layer types come from csa_compress_ratios and the flag "
                "never reaches the spec chain.",
                "Remove 'multi_latent_attention' (dsv4_hybrid already implies "
                "MLA layers via csa_compress_ratios == -2), or drop "
                "experimental_attention_variant.",
            )
        )

    # VAR-02: under dsv4_hybrid, layer_types is overridden by
    # csa_compress_ratios
    if is_dsv4 and getattr(config, "layer_types", None) is not None:
        out.append(
            Finding(
                "V-VAR-02",
                "error",
                "'layer_types' is set but is decorative under "
                "experimental_attention_variant='dsv4_hybrid': every layer type "
                "is rewritten from csa_compress_ratios.",
                "Remove 'layer_types' or switch off the dsv4_hybrid variant.",
            )
        )

    # VHA-01: use_vha_attention is read by nobody under gdn/kda/gemma4
    # (self swaps the class to SelfAttentionVHA; mla/dsv4 reuse it as
    # postmix/premix, which is a legitimate effect)
    _vha_unread = {"gdn", "kda", "gemma4"}
    if (
        getattr(config, "use_vha_attention", False)
        and families
        and families <= _vha_unread
    ):
        out.append(
            Finding(
                "V-VHA-01",
                "error",
                "'use_vha_attention=true' is not read by any resolved layer: "
                f"all layers are {sorted(families)} (self swaps in "
                "SelfAttentionVHA; mla/dsv4 reuse it as postmix/premix; "
                "gdn/kda/gemma4 never read it).",
                "Remove 'use_vha_attention' or use an attention family that "
                "supports it (self / mla / dsv4).",
            )
        )
    # VHA-02 (finer-grained): vha_premix is only read by dsv4 layers
    if getattr(config, "use_vha_premix", False) and not (families & {"dsv4"}):
        out.append(
            Finding(
                "V-VHA-02",
                "error",
                "'use_vha_premix=true' only applies to dsv4 "
                "(CSA/HCA/window) layers; no such layer is resolved.",
                "Remove 'use_vha_premix' or use "
                "experimental_attention_variant='dsv4_hybrid'.",
            )
        )

    # VAR-03: MTP layer types look only at the multi_latent_attention flag
    if mtp_plans and decoder_plans and not is_dsv4:
        mtp_families = {p.family for p in mtp_plans}
        decoder_families = {p.family for p in decoder_plans}
        if mtp_families - decoder_families:
            out.append(
                Finding(
                    "V-VAR-03",
                    "error",
                    f"MTP layers resolve to {sorted(mtp_families)} (driven only "
                    "by multi_latent_attention) while decoder layers resolve "
                    f"to {sorted(decoder_families)} (layer_types / "
                    "attention_layer_type); the two disagree.",
                    "Set multi_latent_attention to match the decoder layer "
                    "family, or accept homogeneous MTP layers.",
                )
            )

    # VAR-04: gemma4 combined with dsv4_hybrid; the gemma4 branch is checked
    # after the dsv4 rewrite
    layer_types = getattr(config, "layer_types", None)
    if (
        is_dsv4
        and isinstance(layer_types, (list, tuple))
        and "gemma4" in layer_types
    ):
        out.append(
            Finding(
                "V-VAR-04",
                "error",
                "'gemma4' appears in layer_types but the gemma4 branch is "
                "unreachable under experimental_attention_variant='dsv4_hybrid'.",
                "Remove 'gemma4' from layer_types or drop the dsv4_hybrid "
                "variant.",
            )
        )

    # MLA-01: MLA does not support GQA/MQA (num_key_value_heads is pinned);
    # the error surfaces late
    if "mla" in families and not is_dsv4:
        nkv = getattr(config, "num_key_value_heads", None)
        nq = getattr(config, "num_attention_heads", None)
        if nkv is not None and nq is not None and nkv != nq:
            out.append(
                Finding(
                    "V-MLA-01",
                    "error",
                    f"MLA pins num_key_value_heads={nq} (got {nkv}); the "
                    "mismatch is only surfaced late (shape error at build "
                    "time).",
                    "Set num_key_value_heads == num_attention_heads for MLA "
                    "models.",
                )
            )

    # VAR-05: hy_sparse's MLA eligibility check uses the pre-rewrite
    # attention_layer_type argument
    if getattr(config, "enable_hy_sparse_attention", False) and is_dsv4:
        if not getattr(config, "multi_latent_attention", False) and not any(
            p.family == "mla" for p in decoder_plans
        ):
            out.append(
                Finding(
                    "V-VAR-05",
                    "error",
                    "enable_hy_sparse_attention requires the pre-rewrite "
                    "attention_layer_type/MLA flag to say MLA, but under "
                    "dsv4_hybrid the eligibility check runs before the "
                    "per-layer rewrite, so the -2 (MLA) layers cannot satisfy "
                    "it via csa_compress_ratios alone.",
                    "Set multi_latent_attention=true alongside dsv4_hybrid "
                    "(the flag is otherwise ignored for spec selection), or "
                    "disable hy_sparse.",
                )
            )

    # ROPE-01: MLA dimension fields misconfigured onto DSv4/CSA layers
    if is_dsv4 and any(p.family == "dsv4" for p in decoder_plans):
        for mla_field in ("qk_rope_head_dim", "qk_nope_head_dim"):
            if _explicitly_set(config, mla_field):
                out.append(
                    Finding(
                        "V-ROPE-01",
                        "error",
                        f"'{mla_field}' is an MLA-only field; DSv4/CSA layers "
                        "read 'qk_pos_emb_head_dim' (currently "
                        f"{getattr(config, 'qk_pos_emb_head_dim', None)!r}).",
                        "Set qk_pos_emb_head_dim for the DSv4/CSA layers "
                        "instead.",
                    )
                )
                break

    # ROPE-02: under dsv4, qk_pos_emb_head_dim None -> RoPE width silently
    # becomes 0
    if is_dsv4 and getattr(config, "qk_pos_emb_head_dim", None) is None:
        out.append(
            Finding(
                "V-ROPE-02",
                "error",  # silently changes model behavior (RoPE disabled):
                # per the severity standard this is an error; warn-only mode
                # still prints just a W line until the observation period ends
                # and the blocking mode is enabled
                "'qk_pos_emb_head_dim' is None on the dsv4_hybrid path: "
                "DSv4/CSA layers resolve RoPE width to 0 (RoPE silently "
                "disabled).",
                "Set qk_pos_emb_head_dim explicitly (0 only if RoPE is "
                "intended to be off).",
            )
        )

    # ROPE-03: rotary_interleaved is hardcoded away on the DSv4/CSA eager path
    if is_dsv4 and getattr(config, "rotary_interleaved", False):
        out.append(
            Finding(
                "V-ROPE-03",
                "error",
                "'rotary_interleaved=true' is not implemented on the DSv4/CSA "
                "eager RoPE path (hardcoded non-interleaved layout); the "
                "config is silently ignored.",
                "Set rotary_interleaved=false, or wait for the DSv4 "
                "interleaved implementation.",
            )
        )

    # ROPE-04: dsa_indexer_rotary_interleaved only takes effect when a DSA
    # indexer exists
    if _explicitly_set(config, "dsa_indexer_rotary_interleaved") and not any(
        "indexer" in p.notes for p in layer_plans
    ):
        out.append(
            Finding(
                "V-ROPE-04",
                "warning",
                "'dsa_indexer_rotary_interleaved' is set but no DSA indexer "
                "is built under the resolved plan.",
                "Remove the field or enable the DSA indexer "
                "(dsa_index_n_heads).",
            )
        )

    return out


# ---------------------------------------------------------------------------
# (3) AttentionExecutionPlan (data contract)
# ---------------------------------------------------------------------------

# family only denotes the "attention computation structure"
# (self/mla/dsv4/gdn/kda/gemma4). VHA, gated_attention, hy_sparse and SWA
# are vertical capabilities that cut across families: VHA is expressed via
# qkv_layout=shared_kv; the rest go into capabilities / the swa column.
Family = Literal["self", "mla", "dsv4", "gdn", "kda", "gemma4", "cross"]
# Vertical capabilities (stackable, not mutually exclusive)
Capability = Literal[
    "vha",  # self family: swaps the whole class for SelfAttentionVHA
    "vha_postmix",  # mla/dsv4: reuses the flag, only adds a low-rank postmix
    # in the output head space
    "vha_premix",  # dsv4: structured premix for the Q up-projection
    # (requires vha_postmix as well)
    "gated_attention",
    "hy_sparse",
]
Core = Literal[
    "dot_product",  # standard backend.core_attention() (eager/sdpa
    # negotiated at runtime)
    "linear",  # GDN/KDA linear attention
    "dsa",  # DSAttention + DSA indexer
    "csa",  # CompressedSparseAttention (core_detail distinguishes
    # hca/csa/window/mqa)
    "mqa_latent",  # MQALatentAttention (latent MQA absorption path)
]
QkvLayout = Literal["mha", "gqa", "latent", "shared_kv", "linear"]
PositionType = Literal["none", "rope", "yarn", "mrope"]
PositionLayout = Literal[
    "STANDARD",  # standard partial rope (rope_first segment order)
    "MLA_EAGER",
    "MLA_FUSED_PAIR",
    "MLA_FUSED_INPLACE_INTERLEAVED",
    "DSV4_CSA_EAGER",
    "DSA_INDEXER",
]
SegmentOrder = Literal["nope_first", "rope_first"]


@dataclass(frozen=True)
class RopeSpec:
    """Full expansion of the position column (RoPE information must be shown
    completely)."""

    type: PositionType
    base: float
    dim: int
    layout: PositionLayout
    segment_order: SegmentOrder
    # True/False = the effective value; "ignored" = hardcoded on that path,
    # the config is disregarded (attached W line)
    interleave: Any
    yarn_params: dict = field(default_factory=dict)  # yarn params that
    # differ from defaults

    def render(self) -> str:
        text = (
            f"{self.type}(base={_fmt_num(self.base)}, dim={self.dim}, "
            f"layout={self.layout}, seg={self.segment_order}"
        )
        # An interleave ignored via a hardcoded path must also be shown
        # explicitly
        text += f", interleave={self.interleave}"
        for key, value in sorted(self.yarn_params.items()):
            text += f", {key}={_fmt_num(value)}"
        return text + ")"


def _fmt_num(value: Any) -> str:
    if isinstance(value, float) and value == int(value):
        return str(int(value))
    return str(value)


@dataclass(frozen=True)
class AttentionExecutionPlan:
    """Per-layer execution plan (one entry per layer).

    Differences from the original contract (recorded in the plan document's
    implementation notes):
    - Added ``index`` / ``layer_number`` / ``is_mtp`` / ``core_detail`` /
      ``notes``: snapshots and printing need layer positioning and core
      granularity (hca/csa/window/mqa);
    - ``precision`` / ``recompute`` are summary strings in the first version
      (to be structured during the P3 refactor);
    - Enums gained ``core=dot_product|linear`` and ``qkv_layout=linear``.
    """

    index: int  # logical layer index (decoder from 0, MTP right after)
    layer_number: int  # physical layer number passed to
    # get_gpt_layer_local_spec
    is_mtp: bool
    family: Family
    core: Core
    core_detail: str | None = None  # csa: hca|csa|window|mqa_full_causal
    swa: bool = False
    window_size: int | None = None
    qkv_layout: QkvLayout = "mha"
    position: RopeSpec | None = None
    precision: str = ""
    recompute: str = ""
    notes: tuple[str, ...] = ()
    # Dimension fields actually effective on this layer (mirrors of each
    # family's __init__ read points, the observation surface of C1-C3):
    # MLA reads q_lora_rank/qk_rope_head_dim etc., dsv4 -2 layers read
    # hybrid_mla_*, CSA/HCA/window layers read v_head_dim/qk_pos_emb_head_dim,
    # VHA reads vha_q_lora_rank, GDN/KDA read linear_*. SWA layers record the
    # values after swa_* overrides. Goes into JSON (print/dump/snapshot); the
    # summary table omits it to stay narrow.
    dims: dict = field(default_factory=dict)
    # Actual q/k norm type on this layer (observation surface of C4):
    # "none" | "LayerNorm" | "RMSNorm" | "L2" | "triton-rms"
    # (qk_norm_fusion fused kernel)
    qk_norm: str = "none"
    # Vertical capabilities (cross-family, stackable): vha / gated_attention /
    # hy_sparse. Orthogonal to family (computation structure): VHA is a
    # projection variant of self (qkv_layout=shared_kv); gating is a
    # gate_proj that self/mla/dsv4 can all carry.
    capabilities: tuple[str, ...] = ()

    def to_json(self) -> dict:
        """Serialization shared by printing, disk dump and snapshot."""
        d = dataclasses.asdict(self)
        d["position"] = (
            dataclasses.asdict(self.position)
            if self.position is not None
            else None
        )
        return d


@dataclass
class AttentionPlanBundle:
    """Resolve result for the whole config."""

    variant: str | None
    layers: list[AttentionExecutionPlan] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    normalization: NormalizationReport = field(
        default_factory=NormalizationReport
    )
    log_only: bool = True

    # -- Aggregate statistics (table header) --
    def family_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for p in self.layers:
            counts[p.family] = counts.get(p.family, 0) + 1
        return counts

    def rope_groups(self) -> int:
        """Number of layer groups with inconsistent RoPE base/type."""
        return len(
            {
                (p.position.type, p.position.base)
                for p in self.layers
                if p.position is not None
            }
        )

    def to_json(self) -> dict:
        return {
            "variant": self.variant,
            "log_only": self.log_only,
            "family_counts": self.family_counts(),
            "rope_groups": self.rope_groups(),
            "layers": [p.to_json() for p in self.layers],
            "findings": [dataclasses.asdict(f) for f in self.findings],
            "normalization": {
                "changes": self.normalization.changes,
                "warnings": self.normalization.warnings,
            },
        }

    def to_json_str(self) -> str:
        return json.dumps(self.to_json(), indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# (4) Printing (rank0 summary table)
# ---------------------------------------------------------------------------


def format_plan_lines(bundle: AttentionPlanBundle) -> list[str]:
    """Render the ``[ATTN-PLAN]`` summary table (each line already carries
    the prefix)."""
    lines: list[str] = []
    n_warn = len(bundle.findings)
    lines.append(
        f"[ATTN-PLAN] {len(bundle.layers)} layers resolved "
        f"(variant={bundle.variant or 'none'}, "
        f"rope_groups={bundle.rope_groups()}, "
        f"warnings={n_warn}{' [warn-only]' if bundle.log_only else ''})"
    )
    lines.append(
        "[ATTN-PLAN]   layer  family  core         swa        qkv_layout  norm  "
        "caps  position"
    )
    for p in bundle.layers:
        label = f"{p.index}" + ("m" if p.is_mtp else "")
        core = p.core + (f":{p.core_detail}" if p.core_detail else "")
        swa = f"swa({p.window_size})" if p.swa else "-"
        caps = "+".join(p.capabilities) if p.capabilities else "-"
        pos = p.position.render() if p.position is not None else "none"
        lines.append(
            f"[ATTN-PLAN]   {label:>5}  {p.family:<6}  {core:<12} {swa:<10} "
            f"{p.qkv_layout:<10} {p.qk_norm:<6} {caps:<6} {pos}"
        )
        for note in p.notes:
            lines.append(f"[ATTN-PLAN]          note: {note}")
    for f in bundle.findings:
        sev = "W" if bundle.log_only else f.severity
        lines.append(f"[ATTN-PLAN] {sev}  {f.rule_id}: {f.message}")
        lines.append(f"[ATTN-PLAN]        fix: {f.remediation}")
    return lines


def _is_rank0() -> bool:
    try:
        import paddle

        if not paddle.distributed.is_initialized():
            return True
        return paddle.distributed.get_rank() == 0
    except Exception:  # pragma: no cover - no paddle distributed environment
        return True


def _dump_plan_json(config: Any, bundle: AttentionPlanBundle) -> None:
    """Dump JSON to disk: reuses config_logger (skipped if
    config_logger_dir is not set)."""
    try:
        from paddlefleet.config_logger import (
            has_config_logger_enabled,
            log_config_to_disk,
        )

        if not has_config_logger_enabled(config):
            return
        if not _is_rank0():
            return
        log_config_to_disk(
            config,
            {"attention_plan": bundle.to_json()},
            prefix="attention_plan",
        )
    except Exception:
        # A dump failure must not block construction (sidecar constraint)
        pass


# ---------------------------------------------------------------------------
# (3) Resolver -- replicates gpt_layer_specs' current priority order
#     (byte-for-byte aligned, observation only)
# ---------------------------------------------------------------------------

_FAMILY_BY_LAYER_TYPE = {
    "full_attention": "self",
    "self_attention": "self",
    "linear_attention": "gdn",
    "gated_delta_net": "gdn",
    "kimi_delta_attention": "kda",
    "multi_latent_attention": "mla",
    "dsv4_hybrid_attention": "dsv4",
    "gemma4": "gemma4",
}


def _ratio_kind(ratio: int) -> tuple[str, str | None]:
    """csa_compress_ratios value -> (family, core_detail).

    Source: gpt_layer_specs._get_dsv4_hybrid_attention_layer_type and the
    ratio convention comments in hf_export.py (-2 -> MLA; -1 -> CSA
    full-causal MQA; 0 -> window; 128 -> HCA; [2,128) -> CSA).
    """
    if ratio == -2:
        return "mla", None
    if ratio == -1:
        return "dsv4", "mqa_full_causal"
    if ratio == 0:
        return "dsv4", "window"
    if ratio == 128:
        return "dsv4", "hca"
    if 2 <= ratio < 128:
        return "dsv4", "csa"
    raise ValueError(f"unknown csa_compress_ratio: {ratio!r}")


def _is_swa_layer(config: Any, logical_index: int) -> bool:
    """Standard-path SWA decision (mirror of attention.py:250-273, by
    logical layer index)."""
    sliding_window = getattr(config, "sliding_window", None)
    if not sliding_window:
        return False
    return is_layer_window_attention(
        sliding_window,
        getattr(config, "window_attn_skip_freq", None),
        logical_index,
    )


_YARN_DEFAULTS = (
    ("rotary_scaling_factor", 40),
    ("original_max_position_embeddings", 4096),
    ("mscale", 1.0),
    ("mscale_all_dim", 0.0),
)


def _yarn_extras(config: Any) -> dict:
    return {
        key: getattr(config, key, default)
        for key, default in _YARN_DEFAULTS
        if getattr(config, key, default) != default
    }


def _standard_rope_spec(config: Any, is_swa: bool) -> RopeSpec:
    """Position for the standard / VHA / gemma4 families."""
    pet = getattr(config, "position_embedding_type", "learned_absolute")
    if pet in ("learned_absolute", "none"):
        return RopeSpec(
            type="none",
            base=0.0,
            dim=0,
            layout="STANDARD",
            segment_order="rope_first",
            interleave=False,
        )
    pos_type = {"rope": "rope", "yarn": "yarn", "mrope": "mrope"}.get(
        pet, "rope"
    )
    base = getattr(config, "swa_rope_theta", None) if is_swa else None
    base = base or getattr(config, "rope_theta", 10000.0)
    head_dim = getattr(config, "swa_head_dim", None) if is_swa else None
    head_dim = head_dim or getattr(config, "head_dim", 0)
    percent = getattr(config, "rotary_percent", 1.0)
    dim = (
        int(head_dim * percent)
        if isinstance(percent, (int, float))
        else head_dim
    )
    return RopeSpec(
        type=pos_type,
        base=float(base),
        dim=int(dim),
        layout="STANDARD",
        segment_order="rope_first",
        interleave=bool(getattr(config, "rotary_interleaved", False)),
        yarn_params=_yarn_extras(config) if pos_type == "yarn" else {},
    )


def _mla_rope_spec(config: Any, is_dsv4_hybrid: bool, is_swa: bool) -> RopeSpec:
    """Position for the MLA family (incl. dsv4 -2 layers)
    (multi_latent_attention.py:476-505)."""
    pos_type = (
        "yarn" if getattr(config, "rope_type", "yarn") == "yarn" else "rope"
    )
    base = getattr(config, "rope_theta", 10000.0)
    if is_dsv4_hybrid:
        dim = getattr(config, "hybrid_mla_qk_rope_head_dim", None) or getattr(
            config, "qk_rope_head_dim", 0
        )
    elif is_swa and getattr(config, "swa_qk_rope_head_dim", None):
        dim = config.swa_qk_rope_head_dim
    else:
        dim = getattr(config, "qk_rope_head_dim", 0)
    return RopeSpec(
        type=pos_type,
        base=float(base),
        dim=int(dim),
        layout="MLA_EAGER",
        segment_order="nope_first",
        interleave=bool(getattr(config, "rotary_interleaved", False)),
        yarn_params=_yarn_extras(config) if pos_type == "yarn" else {},
    )


def _dsv4_rope_spec(config: Any, ratio: int) -> RopeSpec:
    """Position for DSv4/CSA/HCA/window layers
    (mirror of dsv4_hybrid_attention.py:720-771)."""
    if ratio == 128:
        per_type = getattr(config, "hca_rope_type", None)
    elif 2 <= ratio < 128:
        per_type = getattr(config, "csa_rope_type", None)
    else:
        per_type = None
    pos_type = per_type or ("yarn" if ratio > 1 else "rope")
    base = getattr(config, "rope_theta", 10000.0)
    if ratio > 1:
        base = getattr(config, "csa_compress_rotary_base", base)
        if isinstance(base, str):
            base = float(base)
    dim = getattr(config, "qk_pos_emb_head_dim", None) or 0
    return RopeSpec(
        type=pos_type,
        base=float(base),
        dim=int(dim),
        layout="DSV4_CSA_EAGER",
        segment_order="nope_first",
        interleave="ignored",  # the DSv4/CSA eager path hardcodes
        # non-interleaved
        yarn_params=_yarn_extras(config) if pos_type == "yarn" else {},
    )


def _precision_summary(config: Any) -> str:
    parts = []
    for key in ("params_dtype", "bf16", "fp8"):
        value = getattr(config, key, None)
        if value not in (None, False):
            parts.append(f"{key}={value}")
    return ",".join(parts)


def _recompute_summary(config: Any) -> str:
    modules = getattr(config, "recompute_modules", None)
    return str(modules) if modules else ""


def _qk_norm_of(config: Any, family: str, is_dsv4_hybrid: bool) -> str:
    """q/k norm type on this layer (mirror of the q_norm/k_norm selection in
    gpt_layer_specs).

    - self/vha: qk_l2_norm -> L2; use_qk_norm -> RMSNorm/LayerNorm,
      RMSNorm + qk_norm_fusion (head_dim=128 limit enforced inside the
      kernel) -> triton-rms
    - mla (incl. dsv4 -2 layers): use_qk_norm -> q_a/kv_a LayerNorm
      (RMSNorm-ified)
    - dsv4 CSA/HCA/window: qk_layernorm (defaults to True) -> RMSNorm-ified
    - gemma4: standard norm always on; gdn/kda: no qk norm (out_norm only)
    """
    if family in ("gdn", "kda"):
        return "none"
    rms = getattr(config, "normalization", None) == "RMSNorm"
    if family == "gemma4":
        return "RMSNorm" if rms else "LayerNorm"
    if family == "dsv4":
        if not getattr(config, "qk_layernorm", True):
            return "none"
        return "RMSNorm" if rms else "LayerNorm"
    if family == "mla":
        if not getattr(config, "use_qk_norm", False):
            return "none"
        return "RMSNorm" if rms else "LayerNorm"
    # self / vha
    if getattr(config, "qk_l2_norm", False):
        return "L2"
    if not getattr(config, "use_qk_norm", False):
        return "none"
    if rms and getattr(config, "qk_norm_fusion", False):
        return "triton-rms"
    return "RMSNorm" if rms else "LayerNorm"


def _dims_mla(config: Any, is_dsv4_hybrid: bool, is_swa: bool) -> dict:
    """MLA family dimensions (mirror of multi_latent_attention.py:334-398)."""
    if is_dsv4_hybrid:
        dims = {
            "q_lora_rank": config.hybrid_mla_q_lora_rank,
            "kv_lora_rank": config.hybrid_mla_kv_lora_rank,
            "qk_nope_head_dim": config.hybrid_mla_qk_nope_head_dim,
            "qk_rope_head_dim": config.hybrid_mla_qk_rope_head_dim,
            "v_head_dim": config.hybrid_mla_v_head_dim,
            "num_attention_heads": config.hybrid_mla_num_attention_heads,
            # num_key_value_heads is pinned to num_attention_heads
            "num_key_value_heads": config.hybrid_mla_num_attention_heads,
        }
        return dims
    dims = {
        "q_lora_rank": config.q_lora_rank,
        "kv_lora_rank": config.kv_lora_rank,
        "qk_nope_head_dim": config.qk_nope_head_dim,
        "qk_rope_head_dim": config.qk_rope_head_dim,
        "v_head_dim": config.v_head_dim,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_attention_heads,  # pinned
    }
    if is_swa:  # SWA layers swap in a whole different dimension set
        if getattr(config, "swa_qk_nope_head_dim", None) is not None:
            dims["qk_nope_head_dim"] = config.swa_qk_nope_head_dim
        if getattr(config, "swa_qk_rope_head_dim", None) is not None:
            dims["qk_rope_head_dim"] = config.swa_qk_rope_head_dim
    return dims


def _dims_dsv4(config: Any, ratio: int) -> dict:
    """DSv4/CSA/HCA/window layer dimensions
    (mirror of dsv4_hybrid_attention.py:687-708)."""
    return {
        "num_attention_heads": config.num_attention_heads,  # n_local_heads
        "v_head_dim": config.v_head_dim,
        "q_head_dim": config.v_head_dim,  # q_head_dim = v_head_dim
        "qk_pos_emb_head_dim": getattr(config, "qk_pos_emb_head_dim", None)
        or 0,
        "compress_ratio": ratio,
        "window_size": config.csa_window_size,
    }


def _dims_standard(config: Any, is_swa: bool) -> dict:
    """Standard self/gemma4 family dimensions
    (mirror of attention.py:275-303)."""
    if is_swa:  # SWA layers use the swa_* overrides (attention.py:275-280)
        return {
            "num_attention_heads": config.swa_num_attention_heads,
            "num_key_value_heads": config.swa_num_key_value_heads,
            "head_dim": config.swa_head_dim,
            "v_head_dim": config.swa_v_head_dim,
        }
    v_head_dim = getattr(config, "v_head_dim", None)
    return {
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": getattr(config, "num_key_value_heads", None),
        "head_dim": getattr(config, "head_dim", None),
        "v_head_dim": v_head_dim
        if isinstance(v_head_dim, int)
        else getattr(config, "head_dim", None),
    }


def _dims_vha(config: Any, is_swa: bool) -> dict:
    """VHA dimensions (mirror of attention.py:1197-1213)."""
    nq = getattr(config, "num_attention_heads", None)
    dims = {
        "vha_shared_kv": getattr(config, "vha_shared_kv", False),
        "vha_q_lora_rank": (
            config.swa_vha_q_lora_rank
            if is_swa and getattr(config, "swa_vha_q_lora_rank", None)
            else getattr(config, "vha_q_lora_rank", None)
        ),
        # num_attention_heads // 4 is silently derived by default
        "vha_postmix_rank": (
            config.swa_vha_postmix_rank
            if is_swa and getattr(config, "swa_vha_postmix_rank", None)
            else getattr(config, "vha_postmix_rank", None)
        )
        or (nq // 4 if isinstance(nq, int) else None),
    }
    dims.update(_dims_standard(config, is_swa))
    return dims


def _dims_linear(config: Any) -> dict:
    """GDN/KDA dimensions (mirror of the reads in gpt_layer_specs.py:284-291)."""
    return {
        "linear_conv_kernel_dim": getattr(config, "linear_conv_kernel_dim", 4),
        "linear_key_head_dim": getattr(config, "linear_key_head_dim", 128),
        "linear_value_head_dim": getattr(config, "linear_value_head_dim", 128),
        "linear_num_key_heads": getattr(config, "linear_num_key_heads", 16),
        "linear_num_value_heads": getattr(config, "linear_num_value_heads", 32),
    }


def _window_size_of(config: Any) -> Any:
    ws = getattr(config, "sliding_window", None)
    if isinstance(ws, (list, tuple)):
        ws = ws[0]
    return ws


def _resolve_dsv4_layer(
    config: Any, index: int, layer_number: int, is_mtp: bool, ratio: int
) -> AttentionExecutionPlan:
    family, core_detail = _ratio_kind(ratio)
    notes: list[str] = []
    capabilities: list[str] = []
    if getattr(config, "use_vha_attention", False):
        # Both the MLA(-2) layers and the CSA/HCA/window layers reuse this
        # flag for postmix (:667/:913)
        capabilities.append("vha_postmix")
    if (
        family == "dsv4"
        and getattr(config, "use_vha_attention", False)
        and getattr(config, "use_vha_premix", False)
    ):
        capabilities.append("vha_premix")
    if family == "mla":
        mode = getattr(config, "hybrid_mla_attention", "mha")
        if mode == "mqa_dsa":
            core = "mqa_latent"
            notes.append("indexer: DSA (hybrid_mla_attention=mqa_dsa)")
        elif mode == "mqa_full_causal":
            core = "mqa_latent"
            notes.append("no indexer (hybrid_mla_attention=mqa_full_causal)")
        else:
            core = "dot_product"
            notes.append("dense MHA on -2 layer (hybrid_mla_attention=mha)")
        qkv_layout = "latent" if core == "mqa_latent" else "mha"
        position = _mla_rope_spec(config, is_dsv4_hybrid=True, is_swa=False)
        dims = _dims_mla(config, is_dsv4_hybrid=True, is_swa=False)
        swa = False
        window_size = None
    else:
        core = "csa"
        qkv_layout = "mha"
        position = _dsv4_rope_spec(config, ratio)
        dims = _dims_dsv4(config, ratio)
        if core_detail == "window":
            swa = True
            window_size = getattr(config, "csa_window_size", None)
        else:
            swa = False
            window_size = None
        if core_detail == "csa" and not getattr(
            config, "csa_dense_mode", False
        ):
            notes.append("indexer: CSA")
    return AttentionExecutionPlan(
        index=index,
        layer_number=layer_number,
        is_mtp=is_mtp,
        family=family,  # type: ignore[arg-type]
        core=core,  # type: ignore[arg-type]
        core_detail=core_detail,
        swa=swa,
        window_size=window_size,
        qkv_layout=qkv_layout,  # type: ignore[arg-type]
        position=position,
        precision=_precision_summary(config),
        recompute=_recompute_summary(config),
        notes=tuple(notes),
        dims=dims,
        qk_norm=_qk_norm_of(config, family, is_dsv4_hybrid=True),
        capabilities=(
            *capabilities,
            *(
                ("gated_attention",)
                if getattr(config, "gated_attention", False)
                else ()
            ),
        ),
    )


def _resolve_standard_layer(
    config: Any,
    index: int,
    layer_number: int,
    is_mtp: bool,
    layer_type: str,
) -> AttentionExecutionPlan:
    family = _FAMILY_BY_LAYER_TYPE[layer_type]
    notes: list[str] = []
    capabilities: list[str] = []
    if family == "self" and getattr(config, "use_vha_attention", False):
        # VHA is not a separate family: it is a projection variant of self;
        # qkv_layout expresses the layout, capabilities records the
        # capability flag (vertical, cross-family).
        capabilities.append("vha")
    if family == "mla" and getattr(config, "use_vha_attention", False):
        # MLA/MQA does not swap the class; the flag is reused to only add an
        # output-head-space postmix (:667)
        capabilities.append("vha_postmix")
    if getattr(config, "gated_attention", False) and family in (
        "self",
        "mla",
        "dsv4",
    ):
        capabilities.append("gated_attention")
    if family in ("gdn", "kda"):
        core = "linear"
        qkv_layout = "linear"
    else:
        core = "dot_product"
        if "vha" in capabilities:
            qkv_layout = "shared_kv"
        else:
            nkv = getattr(config, "num_key_value_heads", None)
            nq = getattr(config, "num_attention_heads", None)
            qkv_layout = (
                "gqa"
                if nkv is not None and nq is not None and nkv != nq
                else "mha"
            )
    if "vha" in capabilities:
        is_swa = _is_swa_layer(config, index)
        dims = _dims_vha(config, is_swa)
    elif family in ("gdn", "kda"):
        dims = _dims_linear(config)
    elif family == "mla":
        dims = _dims_mla(
            config,
            is_dsv4_hybrid=False,
            is_swa=_is_swa_layer(config, index),
        )
    else:
        dims = _dims_standard(config, _is_swa_layer(config, index))
    if family == "mla":
        if getattr(config, "enable_hy_sparse_attention", False):
            core = "mqa_latent"
            qkv_layout = "latent"
            capabilities.append("hy_sparse")
            notes.append("hy_sparse swaps the class to MQASelfAttention")
        elif getattr(config, "dsa_index_n_heads", None) is not None:
            core = "dsa"
            notes.append("indexer: DSA")
            qkv_layout = "latent"
        else:
            qkv_layout = "latent"
        is_swa = _is_swa_layer(config, index)
        position = _mla_rope_spec(config, is_dsv4_hybrid=False, is_swa=is_swa)
        swa = is_swa
        window_size = _window_size_of(config) if is_swa else None
    else:
        is_swa = _is_swa_layer(config, index)
        position = _standard_rope_spec(config, is_swa)
        swa = is_swa
        window_size = _window_size_of(config) if is_swa else None
    return AttentionExecutionPlan(
        index=index,
        layer_number=layer_number,
        is_mtp=is_mtp,
        family=family,  # type: ignore[arg-type]
        core=core,  # type: ignore[arg-type]
        swa=swa,
        window_size=window_size,
        qkv_layout=qkv_layout,  # type: ignore[arg-type]
        position=position,
        precision=_precision_summary(config),
        recompute=_recompute_summary(config),
        notes=tuple(notes),
        dims=dims,
        qk_norm=_qk_norm_of(config, family, is_dsv4_hybrid=False),
        capabilities=tuple(capabilities),
    )


def _effective_mtp_layers(config: Any) -> int:
    n = getattr(config, "num_nextn_predict_layers", 0) or 0
    if not isinstance(n, int) or isinstance(n, bool):
        return 0
    return n


def resolve_attention_plan(
    config: Any, *, log_only: bool = True
) -> AttentionPlanBundle:
    """Resolve: produce one AttentionExecutionPlan per layer.

    The decision tree replicates the current priority order of
    ``gpt_layer_specs.get_gpt_layer_local_spec``: dsv4_hybrid (per-layer
    csa_compress_ratios) -> multi_latent_attention flag ->
    attention_layer_type (layer_types or default). The first version must be
    byte-for-byte aligned with current behavior (snapshot baseline); only
    afterwards is convergence allowed.
    """
    variant = getattr(config, "experimental_attention_variant", None)
    is_dsv4 = variant == "dsv4_hybrid"
    num_layers = getattr(config, "num_hidden_layers", 1)
    head_offset = getattr(config, "num_empty_layers_add_in_head", 0) or 0

    layers: list[AttentionExecutionPlan] = []
    if is_dsv4:
        ratios = getattr(config, "csa_compress_ratios", None) or []
        for i in range(num_layers):
            ratio = ratios[i] if i < len(ratios) else None
            if ratio is None:
                # Current behavior: _get_dsv4_hybrid_attention_layer_type
                # raises here; the sidecar just skips (construction-time
                # reporting already covers it, no duplication).
                continue
            layers.append(
                _resolve_dsv4_layer(
                    config, i, i + head_offset, False, int(ratio)
                )
            )
        for m in range(_effective_mtp_layers(config)):
            logical = num_layers + m
            ratio = ratios[logical] if logical < len(ratios) else None
            if ratio is None:
                continue
            layers.append(
                _resolve_dsv4_layer(config, logical, m, True, int(ratio))
            )
    else:
        layer_types = getattr(config, "layer_types", None)
        if layer_types is None:
            fallback = (
                "multi_latent_attention"
                if getattr(config, "multi_latent_attention", False)
                else "self_attention"
            )
            layer_types = [fallback] * num_layers
        for i, layer_type in enumerate(layer_types):
            # Synonym mapping in get_attention_spec
            # (gpt_layer_specs.py:200-203)
            layer_type = {
                "full_attention": "self_attention",
                "linear_attention": "gated_delta_net",
            }.get(layer_type, layer_type)
            if layer_type not in _FAMILY_BY_LAYER_TYPE:
                continue  # construction-time will raise; the sidecar does not
                # duplicate
            layers.append(
                _resolve_standard_layer(
                    config, i, i + head_offset, False, layer_type
                )
            )
        for m in range(_effective_mtp_layers(config)):
            # Current behavior: MTP looks only at the
            # multi_latent_attention flag (gpt_layer_specs:877)
            mtp_type = (
                "multi_latent_attention"
                if getattr(config, "multi_latent_attention", False)
                else "self_attention"
            )
            layers.append(
                _resolve_standard_layer(
                    config, num_layers + m, m, True, mtp_type
                )
            )

    bundle = AttentionPlanBundle(
        variant=variant, layers=layers, log_only=log_only
    )
    bundle.findings = _validate(config, layers)
    return bundle


# ---------------------------------------------------------------------------
# Pipeline entry point (1 -> 2 -> 3 -> 4)
# ---------------------------------------------------------------------------


def run_attention_plan(
    config: Any,
    *,
    log_only: bool = True,
    print_plan: bool = True,
) -> AttentionPlanBundle:
    """Run the full pipeline and return the Plan (sidecar: takes no part in
    construction decisions).

    Warn-only mode (``log_only=True``, the default): the Validator's errors
    are also emitted only as ``[ATTN-PLAN]`` warning lines, no raise --
    switch to hard errors after observing for one release cycle. With
    ``log_only=False``, error-severity findings raise directly (with
    migration-guidance text).
    """
    # (1) Normalizer (lossless conversion, applied directly to config)
    report = normalize_attention_config(config)

    # (3) Resolver ((2) Validator runs inside resolve, based on the resolved
    # layer structure)
    bundle = resolve_attention_plan(config, log_only=log_only)
    bundle.normalization = report

    # (4) Print + dump to disk
    if print_plan and _is_rank0():
        for line in format_plan_lines(bundle):
            print(line)
    _dump_plan_json(config, bundle)

    if not log_only:
        errors = [f for f in bundle.findings if f.severity == "error"]
        if errors:
            lines = [
                "Attention config validation failed "
                f"({len(errors)} error(s), warn-only mode off):"
            ]
            for f in errors:
                lines.append(f"  [{f.rule_id}] {f.message}")
                lines.append(f"    fix: {f.remediation}")
            raise ValueError("\n".join(lines))
    return bundle
