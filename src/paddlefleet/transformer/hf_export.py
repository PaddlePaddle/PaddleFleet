# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""Fleet <-> open-source (HuggingFace) config bridge: naming map + model-fact helpers.

Single source of truth for the pieces of HF ``config.json`` export/import that are
**Fleet domain knowledge** -- i.e. facts about how a Fleet model's fields name and
structure map to the open-source (HF) side:

- naming map ``FLEET_HF_FIELD_MAPPING`` / ``ROPE_SCALING_KEYMAP`` and the derived
  ``HF_EXPORT_RULES`` / ``HF_IMPORT_RULES`` (+ ``rule_target``);
- rope structure: ``uses_yarn`` / ``pack_rope_scaling`` / ``unpack_rope_scaling``;
- window semantics: ``is_active_window`` / ``check_window_export_conflict`` /
  ``swa_aware_import_rules``;
- MTP per-layer trimming ``trim_mtp_layers``; mHC injection ``inject_mhc_from_provider``.

Pure Python, no ``paddle`` dependency. Sibling of ``TransformerConfig.transform_rules``
(inbound HF->Fleet rename table in ``transformer_config.py``); this module is the
outbound / bidirectional counterpart. The export *strategy* (whitelist/blacklist,
orchestration, trainer I/O) lives above this layer, in the caller (erniebot).
"""

# rope_scaling structural map: HF nested key -> Fleet flat field name.
ROPE_SCALING_KEYMAP = {
    "type": "rope_type",
    "factor": "rotary_scaling_factor",
    "original_max_position_embeddings": "original_max_position_embeddings",
    "beta_fast": "beta_fast",
    "beta_slow": "beta_slow",
    "mscale": "mscale",
    "mscale_all_dim": "mscale_all_dim",
}


# Single source of truth for bidirectional field mapping:
# (fleet_key, hf_key, Fleet -> HF converter, HF -> Fleet converter).
FLEET_HF_FIELD_MAPPING = [
    ("multi_latent_attention", "use_mla", None, None),
    ("gated_attention", "use_gated_attn", None, None),
    ("rotary_interleaved", "rope_interleave", None, None),
    ("csa_compress_ratios", "compress_ratios", None, None),
    ("csa_window_size", "sliding_window", None, None),
    ("csa_compress_rotary_base", "compress_rope_theta", None, None),
    (
        "params_dtype",
        "torch_dtype",
        lambda v: str(v).replace("paddle.", ""),
        lambda v: f"paddle.{v}"
        if isinstance(v, str) and not v.startswith("paddle.")
        else v,
    ),
    (
        "hidden_act",
        "hidden_act",
        lambda v: v.__name__ if callable(v) else v,
        None,
    ),
    ("window_attn_skip_freq", "hybrid_layer_pattern", None, None),
    (
        "multimax_modules",
        "multimax",
        lambda v: v[0] if isinstance(v, (list, tuple)) and v else v,
        lambda v: list(v)
        if isinstance(v, (list, tuple))
        else ([v] if v is not None else v),
    ),
    ("num_residual_streams", "hc_mult", None, None),
    ("mhc_sinkhorn_iterations", "hc_sinkhorn_iters", None, None),
]

# Fleet -> HF config.json rename / value rules.
HF_EXPORT_RULES = {
    fleet_key: (hf_key, export_fn) if export_fn else hf_key
    for fleet_key, hf_key, export_fn, _ in FLEET_HF_FIELD_MAPPING
}

# HF -> Fleet config.json reverse rename / value rules.
HF_IMPORT_RULES = {
    hf_key: (fleet_key, import_fn) if import_fn else fleet_key
    for fleet_key, hf_key, _, import_fn in FLEET_HF_FIELD_MAPPING
}


def rule_target(key, rules):
    """The HF-side name a raw source key maps to under ``rules``.

    Keys without a rename rule pass through unchanged (target == source).
    Used to translate a set of raw model_config keys into the HF names they
    appear as in the built config, so they can be unioned with the whitelist.
    """
    spec = rules.get(key)
    if spec is None:
        return key
    return spec[0] if isinstance(spec, tuple) else spec


def source_or_provider(raw, provider, key):
    """Value of ``key`` from the export source, falling back to the provider.

    Single provider-fallback accessor shared by the rope / mHC paths: the
    pristine export source (``raw``) wins; when it does not declare ``key`` the
    resolved provider config supplies the default. Returns ``None`` if neither
    has it.
    """
    val = raw.get(key, None)
    if val is None and provider is not None:
        val = getattr(provider, key, None)
    return val


# dsv4_hybrid per-layer compress-ratio conventions. Source of truth:
# paddlefleet/transformer/dsv4_hybrid_attention.py (``compress_ratio == 128`` ->
# HCA; ``2 <= ratio < 128`` -> CSA; ``-1``/``0`` -> window; ``-2`` -> MLA).
# Compressed layers (ratio > 1) default to YaRN.
HCA_COMPRESS_RATIO = 128
CSA_COMPRESS_RATIO_MIN = (
    2  # CSA range is [CSA_COMPRESS_RATIO_MIN, HCA_COMPRESS_RATIO)
)


def uses_yarn(raw, provider=None):
    """Whether the model actually applies YaRN RoPE on any attention layer.

    Decided from the model *structure*, not from ``rotary_scaling_factor``:
    those flat fields (``rotary_scaling_factor`` / ``original_max_position_embeddings``
    / ``beta_fast`` / ...) are YaRN-only parameters that sit in the config even
    when RoPE runs in plain ``"rope"`` mode, so keying off ``factor != 1.0``
    yields false positives.

    YaRN is used when either:

    - the global ``rope_type == "yarn"`` -- the plain MLA / DSA path reads this
      field directly (``multi_latent_attention.py`` / ``dsa_attention.py``); or
    - the model is ``dsv4_hybrid`` and has at least one *compressed* layer
      (``csa_compress_ratios`` entry > 1: HCA ratio 128 or CSA ratio in [2,128))
      whose per-type override does not force ``"rope"``. Compressed layers
      default to YaRN there (``dsv4_hybrid_attention.py``), independent of the
      global ``rope_type``; ``hca_rope_type`` / ``csa_rope_type`` can override
      HCA / CSA respectively. Window ratios (-1 / 0) and MLA (-2) never YaRN
      via compression.
    """

    def _pick(key):
        return source_or_provider(raw, provider, key)

    if _pick("rope_type") == "yarn":
        return True

    if _pick("experimental_attention_variant") == "dsv4_hybrid":
        ratios = _pick("csa_compress_ratios") or []
        hca_rope_type = _pick("hca_rope_type")
        csa_rope_type = _pick("csa_rope_type")
        for ratio in ratios:
            if not isinstance(ratio, (int, float)):
                continue
            if ratio == HCA_COMPRESS_RATIO:
                per_type = hca_rope_type
            elif CSA_COMPRESS_RATIO_MIN <= ratio < HCA_COMPRESS_RATIO:
                per_type = csa_rope_type
            else:
                continue  # window (-1 / 0) or MLA (-2): not YaRN via compression
            if per_type != "rope":
                return True
    return False


def pack_rope_scaling(raw, provider=None):
    """Fleet flat YARN fields -> HF nested ``rope_scaling`` dict; None when no YARN.

    Whether YaRN is active is decided structurally by :func:`uses_yarn` (global
    ``rope_type == "yarn"`` or a ``dsv4_hybrid`` compressed layer that defaults
    to YaRN), *not* from ``rotary_scaling_factor``. ``raw`` is the pristine
    export source; it carries the model_config values (``rotary_scaling_factor``
    -> ``factor``, ``original_max_position_embeddings``). ``provider`` is the
    resolved provider config, used to fill the YARN sub-fields the pristine
    source does not declare (``beta_fast`` / ``beta_slow`` / ...). When YaRN is
    active the ``type`` is normalized to ``"yarn"``.

    The neutral mscale defaults (``mscale`` == 1.0 / ``mscale_all_dim`` == 0.0)
    are omitted so the emitted dict stays minimal (type / factor /
    original_max_position_embeddings / beta_fast / beta_slow).
    """
    if not uses_yarn(raw, provider):
        return None

    out = {}
    for hf_key, fleet_key in ROPE_SCALING_KEYMAP.items():
        val = source_or_provider(raw, provider, fleet_key)
        if val is None:
            continue
        if hf_key == "mscale" and val == 1.0:
            continue
        if hf_key == "mscale_all_dim" and val == 0.0:
            continue
        out[hf_key] = val
    if out.get("type") in (None, "default", "rope"):
        out["type"] = "yarn"
    return out or None


def unpack_rope_scaling(rope_scaling):
    """HF nested rope_scaling dict -> Fleet flat fields dict."""
    if not rope_scaling or not isinstance(rope_scaling, dict):
        return {}
    return {
        fleet_key: rope_scaling[hf_key]
        for hf_key, fleet_key in ROPE_SCALING_KEYMAP.items()
        if hf_key in rope_scaling
    }


def is_active_window(sw):
    """Whether a window-size field declares an active (non-zero) window.

    Neutral naming: used both for the native SWA ``sliding_window`` field and
    the CSA ``csa_window_size`` field. A scalar is active when truthy; a
    per-layer list/tuple is active when any entry is truthy.
    """
    if sw in (None, 0, (), []):
        return False
    if isinstance(sw, (list, tuple)):
        return any(x for x in sw)
    return True


def check_window_export_conflict(raw):
    """SWA and CSA both target the HF ``sliding_window`` field; reject coexistence.

    The SWA path exports its native ``sliding_window`` directly, while the CSA
    path maps ``csa_window_size`` -> ``sliding_window`` (see ``HF_EXPORT_RULES``).
    Emitting both into the single HF ``sliding_window`` field is not supported
    yet, so a config declaring both active windows is rejected up front instead
    of silently letting the CSA rename overwrite the SWA value.
    """
    if is_active_window(raw.get("sliding_window")) and is_active_window(
        raw.get("csa_window_size")
    ):
        raise ValueError(
            "Both 'sliding_window' (SWA) and 'csa_window_size' (CSA) are set; "
            "exporting both to the HF 'sliding_window' field is not supported yet."
        )


SWA_MARKER_HF_KEYS = [
    "add_swa_attention_sink_bias",
    "swa_head_dim",
    "swa_v_head_dim",
    "swa_num_attention_heads",
    "swa_num_key_value_heads",
    "swa_rope_theta",
    "swa_qk_nope_head_dim",
    "swa_qk_rope_head_dim",
    "head_wise_swa_ratio",
]


def is_swa_config(hf_config):
    """Whether an HF config declares SWA-specific fields."""
    return any(
        key in SWA_MARKER_HF_KEYS or key.startswith("swa_") for key in hf_config
    )


def swa_aware_import_rules(hf_config, rules):
    """Drop the ``sliding_window`` -> ``csa_window_size`` rename for SWA configs.

    On export the HF ``sliding_window`` field has a single owner: SWA writes it
    natively, CSA writes it through the ``csa_window_size`` rename, and
    ``check_window_export_conflict`` rejects a config declaring both. Import has
    to infer that owner from the config alone, so a config carrying any SWA
    companion field keeps ``sliding_window`` as itself rather than reviving it as
    a CSA window.
    """
    if "sliding_window" not in rules or not is_swa_config(hf_config):
        return rules
    return {
        hf_key: spec
        for hf_key, spec in rules.items()
        if hf_key != "sliding_window"
    }


# Per-layer list fields whose trailing MTP-layer entries are trimmed on import.
# ``compress_ratios`` is the HF-side rename of ``csa_compress_ratios``; both are
# listed so the same key set works whether the input is an HF ``config.json``
# (renamed) or a Fleet-native ``model_config.json`` (original name).
MTP_TRIM_KEYS = (
    "window_attn_skip_freq",
    "csa_compress_ratios",
    "compress_ratios",
)


def trim_mtp_layers(out):
    """Drop trailing MTP-layer entries from per-layer list fields, in place.

    When ``num_nextn_predict_layers`` > 0 the per-layer lists carry extra
    trailing entries for the MTP layer(s), which the inference-side base config
    should not include. Trims each present list in ``MTP_TRIM_KEYS`` and zeroes
    ``num_nextn_predict_layers``.
    """
    mtp_layers = out.get("num_nextn_predict_layers", 0)
    if mtp_layers and mtp_layers > 0:
        for key in MTP_TRIM_KEYS:
            val = out.get(key)
            if isinstance(val, list) and len(val) > mtp_layers:
                out[key] = val[:-mtp_layers]
        out["num_nextn_predict_layers"] = 0


# mHC (Hyper-Connections) fields are Fleet TransformerConfig defaults that live
# only on the resolved provider (model.config at train time), not on the
# pristine export ``source``. When hyper-connections are active they are sourced
# off the provider here (Fleet names, later renamed to HF names via
# ``FLEET_HF_FIELD_MAPPING``); ``hc_eps`` has no config field (a module constant
# in hyper_connection.py) and is injected directly.
MHC_PROVIDER_FIELDS = ("num_residual_streams", "mhc_sinkhorn_iterations")
MHC_HC_EPS = 1e-6


def inject_mhc_from_provider(raw, provider):
    """Merge mHC fields off the resolved provider into ``raw`` (Fleet names).

    No-op unless ``provider.enable_hyper_connections`` is truthy. Uses the same
    provider-fallback semantics as the rope path (:func:`source_or_provider`,
    source wins), but applies it as an *early mutation* of ``raw`` -- rather
    than a pack-time read -- because these fields must flow through the rename
    rules (``num_residual_streams`` -> ``hc_mult`` /
    ``mhc_sinkhorn_iterations`` -> ``hc_sinkhorn_iters``) to land as top-level
    HF keys. ``hc_eps`` has no config field (a module constant in
    hyper_connection.py) and is injected directly.
    """
    if provider is None or not getattr(
        provider, "enable_hyper_connections", False
    ):
        return
    for name in MHC_PROVIDER_FIELDS:
        val = source_or_provider(raw, provider, name)
        if val is not None:
            raw.setdefault(name, val)
    raw.setdefault("hc_eps", MHC_HC_EPS)
