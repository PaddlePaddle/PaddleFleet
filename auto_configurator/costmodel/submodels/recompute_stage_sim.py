#!/usr/bin/env python3
"""
Stage-local recompute simulation helpers.

This module is intentionally self-contained and does not depend on repository
internal implementations other than a light dependency on `RecomputeGranularity`
when converting enum values to strings.

Goal:
- make block recompute semantics stage-local instead of using one global proxy;
- expose enough detail for debugging / RL reward shaping / paper figures;
- provide a single source of truth used by both compute_model and memory_model.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class StageRecomputePlan:
    stage_id: int
    stage_layers: int
    granularity: str
    method: str
    requested_num_layers: int
    checkpoint_span: int
    checkpoint_boundaries: List[int]
    block_lengths: List[int]
    num_checkpoints: int
    recomputed_layers: int
    recomputed_layer_flags: List[bool]
    recompute_fraction: float
    activation_keep_ratio: float
    activation_save_ratio: float
    extra_forward_fraction: float
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _granularity_to_str(value: Any) -> str:
    if value is None:
        return "none"
    if hasattr(value, "value"):
        return str(value.value).lower()
    return str(value).lower()


def _extract_recompute_tuple(rc: Any) -> Tuple[str, str, int, Tuple[str, ...]]:
    if rc is None:
        return ("none", "uniform", 1, tuple())
    granularity = _granularity_to_str(getattr(rc, "granularity", None))
    method = str(getattr(rc, "method", "uniform") or "uniform").lower()
    num_layers = int(getattr(rc, "num_layers", 1) or 1)
    modules = tuple(getattr(rc, "modules", tuple()) or tuple())
    return (granularity, method, max(1, num_layers), modules)


def build_stage_recompute_plan(
    stage_id: int,
    stage_layers: int,
    granularity: str,
    method: str,
    requested_num_layers: int,
) -> StageRecomputePlan:
    """
    Build a stage-local recompute plan.

    Semantics:
    - `checkpoint_boundaries` uses layer-boundary indexing in [0, L].
    - `recomputed_layer_flags[i]` means layer i is recomputed once during
      backward.
    - `requested_num_layers` is interpreted stage-locally.
        * uniform/full: ignored except for fallback book-keeping.
        * block/full: block span = min(requested_num_layers, stage_layers)
    """
    L = max(0, int(stage_layers or 0))
    gran = str(granularity or "none").lower()
    meth = str(method or "uniform").lower()
    n = max(1, int(requested_num_layers or 1))

    if L <= 0 or gran == "none":
        return StageRecomputePlan(
            stage_id=stage_id,
            stage_layers=L,
            granularity="none",
            method=meth,
            requested_num_layers=n,
            checkpoint_span=max(1, L),
            checkpoint_boundaries=[0, L] if L > 0 else [0],
            block_lengths=[L] if L > 0 else [],
            num_checkpoints=2 if L > 0 else 1,
            recomputed_layers=0,
            recomputed_layer_flags=[False] * L,
            recompute_fraction=0.0,
            activation_keep_ratio=1.0,
            activation_save_ratio=0.0,
            extra_forward_fraction=0.0,
            note="no recompute",
        )

    if gran == "selective":
        # The repository currently models selective recompute coarsely.
        # Keep a conservative approximation here. The main user complaint is
        # block/full, so the more detailed exact plan focuses there.
        target = max(1, min(L, int(round(0.30 * L))))
        flags = [False] * L
        for idx in range(target):
            flags[idx] = True
        keep_ratio = max(0.65, 1.0 - target / max(L, 1) * 0.35)
        return StageRecomputePlan(
            stage_id=stage_id,
            stage_layers=L,
            granularity=gran,
            method=meth,
            requested_num_layers=n,
            checkpoint_span=max(1, L),
            checkpoint_boundaries=[0, L],
            block_lengths=[L],
            num_checkpoints=2,
            recomputed_layers=target,
            recomputed_layer_flags=flags,
            recompute_fraction=target / max(L, 1),
            activation_keep_ratio=keep_ratio,
            activation_save_ratio=max(0.0, 1.0 - keep_ratio),
            extra_forward_fraction=target / max(L, 1),
            note="selective recompute approximated as partial full-recompute",
        )

    # full recompute family
    if meth == "uniform":
        # Save only stage input and stage output.
        boundaries = [0, L]
        flags = [False] + [True] * max(0, L - 1)
        recomputed_layers = max(0, L - 1)
        keep_ratio = min(1.0, max(2.0 / max(L + 1, 1), 0.04))
        return StageRecomputePlan(
            stage_id=stage_id,
            stage_layers=L,
            granularity=gran,
            method=meth,
            requested_num_layers=n,
            checkpoint_span=max(1, L),
            checkpoint_boundaries=boundaries,
            block_lengths=[L],
            num_checkpoints=len(boundaries),
            recomputed_layers=recomputed_layers,
            recomputed_layer_flags=flags,
            recompute_fraction=recomputed_layers / max(L, 1),
            activation_keep_ratio=keep_ratio,
            activation_save_ratio=max(0.0, 1.0 - keep_ratio),
            extra_forward_fraction=recomputed_layers / max(L, 1),
            note="uniform full recompute: keep stage boundaries only",
        )

    if meth == "block":
        # Stage-local exact semantics.
        span = max(1, min(n, L))
        boundaries = list(range(0, L, span))
        if not boundaries or boundaries[-1] != L:
            boundaries.append(L)
        if boundaries[0] != 0:
            boundaries.insert(0, 0)
        block_lengths = [boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1)]
        flags = [False] * L
        recomputed_layers = 0
        offset = 0
        for blen in block_lengths:
            # Inside a checkpoint block, the block input is saved, so all layers
            # except the first one in the block are recomputed once during
            # backward. Example block len=4 -> recompute layers 1,2,3.
            for inner in range(1, blen):
                if offset + inner < L:
                    flags[offset + inner] = True
                    recomputed_layers += 1
            offset += blen
        keep_ratio = min(1.0, max(len(boundaries) / max(L + 1, 1), 0.05))
        return StageRecomputePlan(
            stage_id=stage_id,
            stage_layers=L,
            granularity=gran,
            method=meth,
            requested_num_layers=n,
            checkpoint_span=span,
            checkpoint_boundaries=boundaries,
            block_lengths=block_lengths,
            num_checkpoints=len(boundaries),
            recomputed_layers=recomputed_layers,
            recomputed_layer_flags=flags,
            recompute_fraction=recomputed_layers / max(L, 1),
            activation_keep_ratio=keep_ratio,
            activation_save_ratio=max(0.0, 1.0 - keep_ratio),
            extra_forward_fraction=recomputed_layers / max(L, 1),
            note=(
                f"block full recompute with stage-local span={span}, "
                f"num_blocks={len(block_lengths)}"
            ),
        )

    if meth == "first_n":
        span = max(1, min(n, L))
        flags = [idx < span for idx in range(L)]
        recomputed_layers = sum(1 for v in flags if v)
        boundaries = [0, span, L] if span < L else [0, L]
        keep_ratio = min(1.0, max(len(boundaries) / max(L + 1, 1), 0.10))
        return StageRecomputePlan(
            stage_id=stage_id,
            stage_layers=L,
            granularity=gran,
            method=meth,
            requested_num_layers=n,
            checkpoint_span=span,
            checkpoint_boundaries=boundaries,
            block_lengths=[span, max(0, L - span)] if span < L else [L],
            num_checkpoints=len(boundaries),
            recomputed_layers=recomputed_layers,
            recomputed_layer_flags=flags,
            recompute_fraction=recomputed_layers / max(L, 1),
            activation_keep_ratio=keep_ratio,
            activation_save_ratio=max(0.0, 1.0 - keep_ratio),
            extra_forward_fraction=recomputed_layers / max(L, 1),
            note="first_n recompute",
        )

    # Fallback.
    return build_stage_recompute_plan(stage_id, L, gran, "uniform", n)


def build_stage_plans_from_recompute_configs(
    stage_layer_counts: Sequence[int],
    per_stage_recompute_configs: Sequence[Any],
) -> List[StageRecomputePlan]:
    plans: List[StageRecomputePlan] = []
    for sid, stage_layers in enumerate(stage_layer_counts):
        rc = per_stage_recompute_configs[sid] if sid < len(per_stage_recompute_configs) else None
        granularity, method, num_layers, _modules = _extract_recompute_tuple(rc)
        plans.append(
            build_stage_recompute_plan(
                sid,
                int(stage_layers),
                granularity,
                method,
                num_layers,
            )
        )
    return plans


def build_uniform_stage_plans(
    stage_layer_counts: Sequence[int],
    granularity: Any,
    method: Optional[str],
    num_layers: Optional[int],
) -> List[StageRecomputePlan]:
    gran = _granularity_to_str(granularity)
    meth = str(method or "uniform").lower()
    n = max(1, int(num_layers or 1))
    return [
        build_stage_recompute_plan(sid, int(stage_layers), gran, meth, n)
        for sid, stage_layers in enumerate(stage_layer_counts)
    ]


def plans_to_dicts(plans: Sequence[StageRecomputePlan]) -> List[Dict[str, Any]]:
    return [plan.to_dict() for plan in plans]


__all__ = [
    "StageRecomputePlan",
    "build_stage_recompute_plan",
    "build_stage_plans_from_recompute_configs",
    "build_uniform_stage_plans",
    "plans_to_dicts",
]
