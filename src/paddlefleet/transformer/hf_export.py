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

import functools


def hidden_act_to_hf(act):
    """Fleet ``hidden_act`` (callable or str) -> HF ``hidden_act`` name.

    Reverses ``TransformerConfig._process_attribute`` (``transformer_config.py``),
    which turns an HF string into a callable: ``"gelu_pytorch_tanh"`` becomes
    ``functools.partial(F.gelu, approximate=True)``, ``"situ"`` a named function,
    and any other name ``n`` becomes ``getattr(F, n)`` (whose ``__name__`` is
    ``n``). A ``partial`` has no ``__name__``, so reading it directly would raise
    ``AttributeError``; the gelu-tanh partial is matched structurally instead.
    """
    if isinstance(act, str):
        return act
    if isinstance(act, functools.partial):
        func_name = getattr(act.func, "__name__", None)
        if func_name == "gelu" and act.keywords.get("approximate") is True:
            return "gelu_pytorch_tanh"
        if func_name is not None:
            return func_name
        return act
    return getattr(act, "__name__", act)


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
        lambda v: v.replace("paddle.", "") if isinstance(v, str) else v,
    ),
    (
        "hidden_act",
        "hidden_act",
        hidden_act_to_hf,
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
# Compressed layers (ratio > 1) default to YaRN; window layers stay plain RoPE;
# MLA (-2) layers are built on the plain MLA path and follow global ``rope_type``.
HCA_COMPRESS_RATIO = 128
CSA_COMPRESS_RATIO_MIN = (
    2  # CSA range is [CSA_COMPRESS_RATIO_MIN, HCA_COMPRESS_RATIO)
)
MLA_COMPRESS_RATIO = -2


def uses_yarn(raw, provider=None):
    """Whether the model actually applies YaRN RoPE on any attention layer.

    Decided from the model *structure*, not from ``rotary_scaling_factor``:
    those flat fields (``rotary_scaling_factor`` / ``original_max_position_embeddings``
    / ``beta_fast`` / ...) are YaRN-only parameters that sit in the config even
    when RoPE runs in plain ``"rope"`` mode, so keying off ``factor != 1.0``
    yields false positives.

    The global ``rope_type`` is authoritative only for **non-hybrid** models
    (plain MLA / DSA, which read it directly in ``multi_latent_attention.py`` /
    ``dsa_attention.py``). For ``dsv4_hybrid`` the RoPE mode is decided *per
    layer* (``dsv4_hybrid_attention.py``), so the global value must NOT
    short-circuit -- otherwise ``rope_type="yarn"`` with every layer overridden
    to plain RoPE would wrongly export a YaRN ``rope_scaling``:

    - HCA (ratio 128) / CSA (ratio in [2, 128)) layers default to YaRN and are
      overridden per type by ``hca_rope_type`` / ``csa_rope_type``; a layer uses
      YaRN unless its override forces ``"rope"``;
    - MLA (ratio -2) layers are built on the plain MLA path and follow the
      global ``rope_type``;
    - window (ratio -1 / 0) layers stay plain RoPE.
    """

    def _pick(key):
        return source_or_provider(raw, provider, key)

    global_rope_type = _pick("rope_type")

    if _pick("experimental_attention_variant") != "dsv4_hybrid":
        return global_rope_type == "yarn"

    ratios = _pick("csa_compress_ratios") or []
    hca_rope_type = _pick("hca_rope_type")
    csa_rope_type = _pick("csa_rope_type")
    for ratio in ratios:
        if not isinstance(ratio, (int, float)):
            continue
        if ratio == HCA_COMPRESS_RATIO:
            per_type = hca_rope_type  # default YaRN unless overridden to "rope"
        elif CSA_COMPRESS_RATIO_MIN <= ratio < HCA_COMPRESS_RATIO:
            per_type = csa_rope_type  # default YaRN unless overridden to "rope"
        elif ratio == MLA_COMPRESS_RATIO:
            if global_rope_type == "yarn":  # MLA follows global rope_type
                return True
            continue
        else:
            continue  # window (-1 / 0): plain RoPE
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
    """HF nested rope_scaling dict -> Fleet flat fields dict.

    The RoPE type accepts both the current HF canonical key ``rope_type`` and
    its legacy alias ``type`` (``ROPE_SCALING_KEYMAP`` only carries ``type``);
    ``rope_type`` wins when both are present. Without this a modern config such
    as ``{"rope_type": "yarn", "factor": 4}`` would drop the YaRN type and fall
    back to plain RoPE.
    """
    if not rope_scaling or not isinstance(rope_scaling, dict):
        return {}
    out = {
        fleet_key: rope_scaling[hf_key]
        for hf_key, fleet_key in ROPE_SCALING_KEYMAP.items()
        if hf_key in rope_scaling
    }
    if "rope_type" in rope_scaling:
        out["rope_type"] = rope_scaling["rope_type"]
    return out


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
    """Whether an HF config declares SWA-specific companion fields."""
    return any(
        key in SWA_MARKER_HF_KEYS or key.startswith("swa_") for key in hf_config
    )


# HF markers that identify a DSv4 CSA config, whose ``sliding_window`` is the
# CSA window (renamed to ``csa_window_size`` on import). ``compress_ratios`` is
# the HF name of ``csa_compress_ratios``; ``csa_compress_ratios`` covers a
# Fleet-native input; ``experimental_attention_variant == "dsv4_hybrid"`` is the
# structural flag. Any other ``sliding_window`` is a native SWA window.
CSA_MARKER_HF_KEYS = ("compress_ratios", "csa_compress_ratios")


def is_csa_config(hf_config):
    """Whether an HF config's ``sliding_window`` is a DSv4 CSA window.

    Only a dsv4_hybrid / compress-ratio config routes HF ``sliding_window`` to
    Fleet ``csa_window_size``. A standard HF ``sliding_window`` (e.g. Mistral's
    bare ``sliding_window=4096``) carries no CSA markers and must keep its
    native name on import.
    """
    if hf_config.get("experimental_attention_variant") == "dsv4_hybrid":
        return True
    return any(key in hf_config for key in CSA_MARKER_HF_KEYS)


def swa_aware_import_rules(hf_config, rules):
    """Keep the ``sliding_window`` -> ``csa_window_size`` rename only for CSA.

    On import HF ``sliding_window`` has two possible owners: DSv4 CSA, where it
    is the compressed-attention window and must become ``csa_window_size``, and
    everything else (native SWA -- Mistral's bare ``sliding_window``, or a Fleet
    SWA config carrying ``swa_*`` companions), which must keep its own name.
    The rename therefore applies *only* when the config carries CSA markers;
    otherwise ``sliding_window`` passes through unchanged, so a standard HF SWA
    config is no longer silently re-homed onto ``csa_window_size``.
    """
    if "sliding_window" not in rules or is_csa_config(hf_config):
        return rules
    return {
        hf_key: spec
        for hf_key, spec in rules.items()
        if hf_key != "sliding_window"
    }


# Per-layer list fields whose trailing MTP-layer entries are trimmed on import.
# Both Fleet-native and HF-renamed names are listed so the same key set works
# whether the input is an HF ``config.json`` or a Fleet ``model_config.json``:
# ``compress_ratios`` <- ``csa_compress_ratios`` and ``hybrid_layer_pattern`` <-
# ``window_attn_skip_freq``. Missing the HF name leaves an over-long list that
# fails ``TransformerConfig`` validation once ``num_nextn_predict_layers`` is 0.
MTP_TRIM_KEYS = (
    "window_attn_skip_freq",
    "hybrid_layer_pattern",
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
