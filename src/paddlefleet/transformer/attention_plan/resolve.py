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

"""Resolver (pipeline step 3): produce one ``AttentionExecutionPlan`` per
layer, replicating branch-by-branch the priority order of
``gpt_layer_specs.get_gpt_layer_local_spec``.

Extension point: a new attention family adds an entry to
``_FAMILY_BY_LAYER_TYPE`` plus a ``_resolve_*_layer`` branch (and its
dims/rope mirrors).
"""

from __future__ import annotations

from typing import Any

from paddlefleet.transformer.attention_plan.plan import (
    AttentionExecutionPlan,
    AttentionPlanBundle,
    RopeSpec,
)
from paddlefleet.transformer.attention_plan.validate import Finding, _validate
from paddlefleet.transformer.utils import is_layer_window_attention

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
    # MultiLatentAttention inherits Attention.__init__: when the layer hits
    # the sliding-window pattern, rope_theta is swapped to swa_rope_theta
    # (attention.py:281) -- the swa_* qk dims are mirrored in _dims_mla.
    base = getattr(config, "rope_theta", 10000.0)
    if is_swa:
        base = getattr(config, "swa_rope_theta", None) or base
    if is_dsv4_hybrid:
        dim = getattr(config, "hybrid_mla_qk_rope_head_dim", None) or getattr(
            config, "qk_rope_head_dim", 0
        )
    else:
        dim = getattr(config, "qk_rope_head_dim", 0)
    if is_swa and getattr(config, "swa_qk_rope_head_dim", None):
        # attention.py:394-398 overrides qk_rope_head_dim after the
        # hybrid/plain split
        dim = config.swa_qk_rope_head_dim
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
        if is_swa:
            # multi_latent_attention.py:388-398: the swa_* overrides apply
            # after (and regardless of) the hybrid_mla_* split
            if getattr(config, "swa_qk_nope_head_dim", None) is not None:
                dims["qk_nope_head_dim"] = config.swa_qk_nope_head_dim
            if getattr(config, "swa_qk_rope_head_dim", None) is not None:
                dims["qk_rope_head_dim"] = config.swa_qk_rope_head_dim
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
        # The -2 layer is built as MultiLatentAttention, which inherits
        # Attention.__init__: sliding_window/window_attn_skip_freq can still
        # flag it as SWA, swapping in the swa_* dims/rope. Decide by the real
        # layer number (logical index), not by family.
        is_swa = _is_swa_layer(config, index)
        position = _mla_rope_spec(config, is_dsv4_hybrid=True, is_swa=is_swa)
        dims = _dims_mla(config, is_dsv4_hybrid=True, is_swa=is_swa)
        swa = is_swa
        window_size = _window_size_of(config) if is_swa else None
        if is_swa:
            notes.append("swa layer despite hybrid MLA (sliding_window hit)")
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
        if family in ("gdn", "kda"):
            # GatedDeltaNet / KimiDeltaAttention are standalone linear
            # attention implementations: they never read sliding_window and
            # have no is_swa, so the plan must not flag them.
            is_swa = False
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
    # Resolver-level errors: the real construction chain raises on these
    # inputs (get_gpt_layer_local_spec / get_gpt_decoder_layers_spec); the
    # sidecar must not silently drop the layer and print a truncated
    # "N layers resolved" -- record an error finding instead (raised when
    # log_only=False).
    resolver_findings: list[Finding] = []
    if is_dsv4:
        ratios = getattr(config, "csa_compress_ratios", None) or []
        for i in range(num_layers):
            ratio = ratios[i] if i < len(ratios) else None
            if ratio is None:
                # Current behavior: _get_dsv4_hybrid_attention_layer_type
                # raises here.
                resolver_findings.append(
                    Finding(
                        rule_id="V-RES-01",
                        severity="error",
                        message=(
                            f"csa_compress_ratios has no entry for layer {i} "
                            f"(len={len(ratios)}); construction raises "
                            "before any layer is built"
                        ),
                        remediation=(
                            "provide one csa_compress_ratios entry per layer "
                            "(incl. MTP: num_hidden_layers + "
                            "num_nextn_predict_layers)"
                        ),
                    )
                )
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
                resolver_findings.append(
                    Finding(
                        rule_id="V-RES-01",
                        severity="error",
                        message=(
                            f"csa_compress_ratios has no entry for MTP layer "
                            f"{logical} (len={len(ratios)}); construction "
                            "raises before any layer is built"
                        ),
                        remediation=(
                            "provide one csa_compress_ratios entry per layer "
                            "(incl. MTP: num_hidden_layers + "
                            "num_nextn_predict_layers)"
                        ),
                    )
                )
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
        elif len(layer_types) != num_layers:
            resolver_findings.append(
                Finding(
                    rule_id="V-RES-02",
                    severity="error",
                    message=(
                        f"layer_types has {len(layer_types)} entries but "
                        f"num_hidden_layers={num_layers}; the decoder builds "
                        f"{num_layers} layers and construction "
                        "fails/misbuilds on the mismatch"
                    ),
                    remediation=("make len(layer_types) == num_hidden_layers"),
                )
            )
        for i, layer_type in enumerate(layer_types):
            # Synonym mapping in get_attention_spec
            # (gpt_layer_specs.py:200-203)
            layer_type = {
                "full_attention": "self_attention",
                "linear_attention": "gated_delta_net",
            }.get(layer_type, layer_type)
            if layer_type not in _FAMILY_BY_LAYER_TYPE:
                # Current behavior: get_gpt_layer_local_spec raises
                # "Unknown attention_layer_type" here.
                resolver_findings.append(
                    Finding(
                        rule_id="V-RES-03",
                        severity="error",
                        message=(
                            f"unknown layer_types[{i}]={layer_type!r}; "
                            "construction raises 'Unknown "
                            "attention_layer_type' before any layer is built"
                        ),
                        remediation=(
                            "use one of the known types: "
                            + ", ".join(sorted(_FAMILY_BY_LAYER_TYPE))
                        ),
                    )
                )
                continue
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
    # Resolver errors first: they mean the printed layer list is incomplete.
    bundle.findings = resolver_findings + _validate(config, layers)
    return bundle
