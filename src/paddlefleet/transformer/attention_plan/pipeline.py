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

"""Pipeline entry point: Normalizer -> Resolver (Validator inside) ->
print + dump. Run from ``gpt_builders`` before layer construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from paddlefleet.transformer.attention_plan.normalize import (
    normalize_attention_config,
)
from paddlefleet.transformer.attention_plan.report import (
    _dump_plan_json,
    _is_rank0,
    format_plan_lines,
)
from paddlefleet.transformer.attention_plan.resolve import (
    resolve_attention_plan,
)

if TYPE_CHECKING:
    from paddlefleet.transformer.attention_plan.plan import (
        AttentionPlanBundle,
    )

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
