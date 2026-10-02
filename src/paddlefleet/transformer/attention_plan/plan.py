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

"""AttentionExecutionPlan data contract (pipeline step 3 output): the
per-layer plan, the RopeSpec position column and the whole-config bundle.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from paddlefleet.transformer.attention_plan.normalize import (
    NormalizationReport,
)

if TYPE_CHECKING:
    from paddlefleet.transformer.attention_plan.validate import Finding

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
