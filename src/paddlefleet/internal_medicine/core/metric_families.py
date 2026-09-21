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

"""Metric families: the sub-monitor unit of selection.

A single monitor owns far too many metrics to be a useful on/off switch —
``mhc_health`` emits 106 keys per layer, ``moe_health`` 56 plus two per expert.
Turning a whole monitor off to save payload throws away the handful of curves
that are actually being watched; keeping it on pays for all of them.

So each monitor's metrics are partitioned into *families*: `gate`, `mix`,
`gain`, `param`, ... — groups that answer the same kind of question. Selection
happens at ``(monitor, family)`` granularity.

``components`` is an orthogonal axis, not a family. ``mhc_health`` reports every
one of its families twice, once for the attention hyper-connection and once for
the MLP one; folding that into the family list would double the family count for
no gain, so it stays a separate dimension.

This taxonomy is shared with the visualisation layer's chip groups
(``tools/internal_medicine/viewer.py``) on purpose: "I turned off
``mhc_health:mix``" and "the 混合矩阵 group of charts is gone" have to be the
same sentence. ``tests/test_metric_families.py`` pins the two against each other
so they cannot drift.
"""

from __future__ import annotations

import re

__all__ = [
    "METRIC_TAXONOMY",
    "DEBUG_ONLY_FAMILIES",
    "FAMILY_OTHER",
    "UnknownFamilyError",
    "UnknownMonitorError",
    "classify",
    "components_of",
    "exclusions_for",
    "families_of",
    "parse_exclude",
    "parse_exclusions",
    "strip_prefixes",
    "validate_exclusions",
    "FamilySelection",
]

FAMILY_OTHER = "other"

# Attention-kind and branch prefixes are part of the key, not of the metric
# identity: `hca_h_res_cell0` and `mla_h_res_cell0` are the same quantity
# measured on two attention stacks and belong in the same family.
_ATTN_PREFIX_RE = re.compile(
    r"^(?:swa|full|mla|mqa|hca|csa|dsa|window)_(?P<base>.+)$"
)
_VHA_BRANCH_RE = re.compile(r"^(?:main|sparse)_(?P<base>.+)$")

# Families are matched in declaration order, first match wins. Order is
# therefore load-bearing wherever one pattern is a prefix of another — see the
# note on moe_health below.
METRIC_TAXONOMY: dict[str, dict] = {
    "mhc_health": {
        "components": ("attn", "mlp"),
        # `stream` and `cell` must precede `gate` / `mix`: they are the per-unit
        # slices of those same quantities and would otherwise be swallowed.
        "families": (
            ("stream", "按流门控值 h_pre / h_post_idx", r"^h_(pre|post)_idx\d"),
            ("gate", "门控强度 h_pre / h_post", r"^h_(pre|post)(_|$)"),
            ("cell", "混合矩阵元素 h_res_cell", r"^h_res_cell\d"),
            ("mix", "混合矩阵 h_res", r"^h_res(_|$)"),
            ("gain", "增益放大 amax / composite", r"amax_gain"),
            ("param", "可学习参数 α / bias", r"^(alpha|bias)_"),
            ("share", "残差占比", r"^branch_residual_share"),
        ),
    },
    "qk_stats": {
        "families": (
            ("sink", "sink head", r"^sink"),
            ("logit", "logits 量级", r"^(max|mean|attn_sink_logit)$"),
            ("entropy", "注意力熵", r"^entropy_"),
            ("qkv", "q / k / v 范数", r"^[qkv]_norm_"),
        ),
    },
    "massive_act": {
        "families": (
            (
                "channel",
                "通道量级分布",
                r"^(channel_(?!count_gt)|topk_channel_|massive_act_channel_count)",
            ),
            ("outlier", "超大通道计数", r"^channel_count_gt_"),
            (
                "module",
                "各模块输出量级",
                r"^(attn_out|ffn_or_moe_out|layer_input|post_ffn_residual)_",
            ),
            ("norm", "归一化后表示", r"^post_norm_"),
            ("overall", "整体激活量级", r"^activation_rms$"),
        ),
    },
    "ape_health": {
        "components": ("core", "indexer"),
        "families": (
            ("softmax", "位置 softmax 形状", r"^softmax_"),
            ("coverage", "位置覆盖", r"^(effective_positions|position_range)"),
            (
                "scale",
                "量级与数值健康",
                r"^(rms|centered_rms|non_finite_ratio)$",
            ),
        ),
    },
    "vha_health": {
        "components": ("main", "mtp"),
        "families": (
            ("mix", "postmix 混合矩阵", r"^postmix_(u|v|uv|offdiag)"),
            ("gain", "postmix 增益与扰动", r"^postmix_(amax_gain|delta_rel)"),
            ("head", "头间一致性", r"^(postmix_head_cos|head_out_norm)"),
        ),
    },
    "moe_health": {
        # `spectrum` and `act` must precede `shared`, or `shared_gate_stable_rank`
        # and `shared_act_*` would be swallowed by the `^shared_` catch.
        "families": (
            ("router", "路由打分质量", r"^router_"),
            ("balance", "负载均衡", r"^(assignment_load|gate_mass)_"),
            (
                "norm",
                "专家范数与 bias",
                r"^(expert_norm|expert_bias|score_sum|bias_affinity)",
            ),
            ("spectrum", "gate 谱性质", r"gate_(stable_rank|singular_entropy)"),
            (
                "act",
                "激活量级",
                r"_act_(abs_max|mean|norm)$|^routed_act|^shared_act",
            ),
            ("shared", "共享 vs 路由", r"^shared_"),
            ("expert", "按专家分布", r"^expert_(token|weight)_share"),
            ("card", "按卡负载（EP 组内）", r"^card_tokens_"),
        ),
    },
    "dsa_health": {
        # `crosslayer` must precede `select`, or `^select_` would swallow the
        # cross-layer IoU — which is the one family here that answers a question
        # about the stack rather than about a layer.
        "families": (
            ("indexer", "indexer 投影量级", r"^index_"),
            ("score", "打分分布与选择边界", r"^score_"),
            ("crosslayer", "跨层选择重合度", r"^select_iou"),
            ("select", "选择结构与稀疏率", r"^select_"),
            ("match", "与稠密 attn 的一致性", r"^attn_"),
        ),
    },
}


def strip_prefixes(sub: str) -> str:
    """Reduce a metric name to the form the family patterns match against.

    Drops the ``global_`` marker (the cross-layer aggregate of a quantity belongs
    in that quantity's family), then the attention kind, then the vha branch.
    """
    sub = sub.removeprefix("global_")
    m = _ATTN_PREFIX_RE.match(sub)
    if m:
        sub = m["base"]
    m = _VHA_BRANCH_RE.match(sub)
    if m:
        sub = m["base"]
    return sub


def classify(monitor: str, sub: str) -> tuple[str, str]:
    """``(family, component)`` for one metric name.

    A monitor with no taxonomy entry, or a name matching none of its patterns,
    yields ``FAMILY_OTHER``. Callers that gate collection must treat that as
    "always collect" — an unclassified metric is a gap in this table, and
    silently dropping it would be the worst possible failure mode.
    """
    spec = METRIC_TAXONOMY.get(monitor)
    if not spec:
        return FAMILY_OTHER, ""
    rest = strip_prefixes(sub)
    component = ""
    for candidate in spec.get("components", ()):
        if rest.startswith(candidate + "_"):
            component, rest = candidate, rest[len(candidate) + 1 :]
            break
    for key, _label, pattern in spec.get("families", ()):
        if re.search(pattern, rest):
            return key, component
    return FAMILY_OTHER, component


def families_of(monitor: str) -> tuple[str, ...]:
    """Declared family keys for one monitor, in declaration order."""
    spec = METRIC_TAXONOMY.get(monitor) or {}
    return tuple(key for key, _label, _pattern in spec.get("families", ()))


def components_of(monitor: str) -> tuple[str, ...]:
    spec = METRIC_TAXONOMY.get(monitor) or {}
    return tuple(spec.get("components", ()))


class FamilySelection:
    """Which families of one monitor to *skip*. Empty set means collect everything.

    Exclusion, not inclusion, is the primary model on purpose. The operational
    question is always "which of these is too expensive to keep on", never "which
    three do I want" — a whitelist of 27 families would have to be rewritten every
    time a monitor gains a metric, and forgetting to extend it loses data silently.

    It also makes the safe default fall out for free: a metric matching no family
    pattern is not in any exclusion set, so a hole in ``METRIC_TAXONOMY`` costs
    payload rather than deleting a curve.
    """

    __slots__ = ("monitor", "excluded")

    def __init__(self, monitor: str, excluded=()):
        self.monitor = monitor
        self.excluded = frozenset(excluded)

    def allows(self, sub: str) -> bool:
        if not self.excluded:
            return True
        family, _component = classify(self.monitor, sub)
        return family not in self.excluded

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        dropped = "+".join(sorted(self.excluded)) or "nothing"
        return f"FamilySelection({self.monitor}, excluding {dropped})"


class UnknownFamilyError(ValueError):
    """A config named a family the monitor does not declare."""


# Families kept out of a production run and collected only under debug mode.
#
# The criterion is *how a family's key count grows*, not how interesting it is:
# each family below emits one curve per structural unit — per expert, per
# hyper-connection cell, per input stream — so its footprint scales with depth x
# expert count while every other family stays constant. On a 43-layer mHC + MoE
# run these three are 1392 of 3732 keys per step (37.3%), the same share of the
# cross-rank payload and of the jsonl on disk.
#
# They are also exactly the families the diagnosis layer never reads: it works
# off the per-layer aggregates (`h_res_logits_max`, `h_post_mean`, ...), which
# stay on. So this default costs payload and nothing else — the reason `mix` and
# `gate` are split into an aggregate half and a per-unit half at all.
#
# Deliberately a code-side constant rather than a config string: the switch that
# selects it carries one bit, so there is nothing to get wrong in a yaml file,
# and changing what "debug" means becomes a reviewable diff.
DEBUG_ONLY_FAMILIES: dict[str, tuple[str, ...]] = {
    "moe_health": ("expert",),
    "mhc_health": ("cell", "stream"),
}


def exclusions_for(debug_mode: bool) -> dict[str, list[str]]:
    """``{monitor: [family, ...]}`` to switch off for this run.

    ``debug_mode=True`` collects everything; the production default drops
    ``DEBUG_ONLY_FAMILIES``. Returns a fresh dict so callers may mutate it.
    """
    if debug_mode:
        return {}
    return {
        monitor: list(families)
        for monitor, families in DEBUG_ONLY_FAMILIES.items()
    }


def _split_names(value) -> list[str]:
    if isinstance(value, str):
        return [n for n in re.split(r"[+,\s]+", value) if n]
    return [str(n).strip() for n in value if str(n).strip()]


def _validate(monitor: str, names: list[str], action: str) -> None:
    declared = families_of(monitor)
    if not declared:
        raise UnknownFamilyError(
            f"{monitor!r} declares no metric families, so {action} {names!r} is meaningless; "
            f"add it to METRIC_TAXONOMY first"
        )
    unknown = [n for n in names if n not in declared]
    if unknown:
        raise UnknownFamilyError(
            f"unknown {monitor!r} metric families {unknown!r}; declared families are {list(declared)!r}"
        )


def parse_exclude(monitor: str, exclude) -> FamilySelection:
    """Build a selection from the families to switch off.

    ``exclude`` accepts ``None`` (drop nothing), a ``"mix+param"`` string, or any
    iterable of names. Unknown names raise rather than being ignored: a typo here
    means a family stays on that someone believed was off, and the only symptom is
    a payload that did not shrink.
    """
    if exclude is None:
        return FamilySelection(monitor)
    names = _split_names(exclude)
    if not names or names == ["none"]:
        return FamilySelection(monitor)
    _validate(monitor, names, "excluding")
    return FamilySelection(monitor, names)


def parse_exclusions(spec) -> dict[str, list[str]]:
    """``"mhc_health:mix, moe_health:expert+act"`` → per-monitor family lists.

    A monitor written without a ``:`` part, or an absent/empty spec, yields no
    entry — i.e. nothing is switched off and behaviour is exactly what it was
    before families existed. Names are not checked here: only the backend knows
    its monitor registry, so both validations happen in ``validate_exclusions``.
    """
    if not spec:
        return {}
    entries = (
        spec.split(",")
        if isinstance(spec, str)
        else [str(entry) for entry in spec]
    )
    excluded: dict[str, list[str]] = {}
    for entry in entries:
        name, _, family_part = entry.strip().partition(":")
        name = name.strip()
        families = [n for n in re.split(r"[+\s]+", family_part) if n]
        if name and families:
            excluded.setdefault(name, []).extend(families)
    return excluded


class UnknownMonitorError(ValueError):
    """A config excluded families from a monitor the backend does not have."""


def validate_exclusions(excluded, known_monitors, backend: str = "") -> None:
    """Fail the run on a monitor name the backend never heard of.

    Family names are already fail-closed inside ``parse_exclude``, but a
    misspelled *monitor* name would otherwise be fail-open: the backend's setup
    loop only injects ``exclude_families`` for names it recognises, so
    ``"moe:expert"`` silently switches nothing off. The only symptom is a payload
    that did not shrink. Both backends therefore call this right after parsing,
    which also moves family validation ahead of their per-monitor
    ``try/except Exception`` — inside it, a family typo would degrade into a
    logged "failed to setup" instead of stopping the run.

    A name the backend knows but this run did not enable passes silently: the
    exclusion is simply inert, and sharing one exclusion string across configs
    that enable different monitors is legitimate.
    """
    label = f"[InternalMedicine/{backend}] " if backend else ""
    unknown = sorted(name for name in excluded if name not in known_monitors)
    if unknown:
        raise UnknownMonitorError(
            f"{label}exclude_families names unknown monitors {unknown!r}; known monitors are {sorted(known_monitors)!r}"
        )
    for name, families in excluded.items():
        parse_exclude(name, families)
