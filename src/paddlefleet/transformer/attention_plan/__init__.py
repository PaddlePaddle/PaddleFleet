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

The package is split by pipeline stage so future registry / rule / family
extensions land in their own module instead of growing one file:

- ``registry``  -- field-ownership registry (single source of truth)
- ``normalize`` -- step 1: alias normalization + type coercion
- ``validate``  -- step 2: A/B rule families, warn-only findings
- ``plan``      -- AttentionExecutionPlan / RopeSpec / bundle data contract
- ``report``    -- step 4: summary table + JSON dump
- ``resolve``   -- step 3: per-layer resolver (mirror of gpt_layer_specs)
- ``pipeline``  -- ``run_attention_plan`` entry point
"""

from __future__ import annotations

from paddlefleet.transformer.attention_plan.normalize import (
    _COERCE_FAILED,
    NormalizationReport,
    _coerce_value,
    normalize_attention_config,
)
from paddlefleet.transformer.attention_plan.pipeline import (
    run_attention_plan,
)
from paddlefleet.transformer.attention_plan.plan import (
    AttentionExecutionPlan,
    AttentionPlanBundle,
    RopeSpec,
    _fmt_num,
)
from paddlefleet.transformer.attention_plan.registry import (
    ATTENTION_FIELD_REGISTRY,
    FieldSpec,
    _explicitly_set,
)
from paddlefleet.transformer.attention_plan.report import (
    format_plan_lines,
)
from paddlefleet.transformer.attention_plan.resolve import (
    _effective_mtp_layers,
    _qk_norm_of,
    _ratio_kind,
    resolve_attention_plan,
)
from paddlefleet.transformer.attention_plan.validate import (
    Finding,
)

__all__ = [
    "AttentionExecutionPlan",
    "AttentionPlanBundle",
    "ATTENTION_FIELD_REGISTRY",
    "FieldSpec",
    "Finding",
    "NormalizationReport",
    "RopeSpec",
    "format_plan_lines",
    "normalize_attention_config",
    "resolve_attention_plan",
    "run_attention_plan",
    # internal names kept importable from the package root for the snapshot
    # tests (they exercise the Normalizer/Resolver internals directly)
    "_COERCE_FAILED",
    "_coerce_value",
    "_effective_mtp_layers",
    "_explicitly_set",
    "_fmt_num",
    "_qk_norm_of",
    "_ratio_kind",
]
