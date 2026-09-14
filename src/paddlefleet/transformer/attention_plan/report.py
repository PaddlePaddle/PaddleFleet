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

"""Plan reporting (pipeline step 4): rank0 summary table rendering and the
JSON dump to disk (config_logger side channel).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from paddlefleet.transformer.attention_plan.plan import (
        AttentionPlanBundle,
    )

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
