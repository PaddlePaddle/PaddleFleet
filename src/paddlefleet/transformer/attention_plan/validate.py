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

"""Validator (pipeline step 2): combined A/B rule-family validation,
warn-only mode by default.

Extension point: new rules are appended inside ``_validate`` (or a future
per-family rule registry); each returns a ``Finding``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from paddlefleet.transformer.attention_plan.registry import (
    _explicitly_set,
)

if TYPE_CHECKING:
    # plan imports validate at runtime (Finding in bundle.findings), so this
    # side must stay annotation-only to avoid a cycle
    from paddlefleet.transformer.attention_plan.plan import (
        AttentionExecutionPlan,
    )

# ---------------------------------------------------------------------------
# Validator (2) -- combined validation, warn-only mode
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    """A single validation result. Under warn-only (log_only) mode findings
    are printed with their real severity prefix (E/W) but error findings
    are not raised; only the report header gains the [warn-only] marker."""

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
