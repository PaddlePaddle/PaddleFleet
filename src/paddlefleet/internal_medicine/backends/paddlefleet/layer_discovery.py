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

"""Layer discovery helpers for PaddleFleet monitors."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MonitorLayer:
    """A transformer layer together with its metric layer id and attention type."""

    idx: int
    layer: object
    is_mtp: bool = False
    # Attention kind tag for stacks that mix kinds across layers. Which config
    # field describes the mix decides the value space (see classify_attn_type):
    #   - KDA hybrid stack                 -> ``"kda"`` on the linear-attention
    #     layers, ``"mla"`` / ``"global"`` on the interleaved global ones
    #   - ``csa_compress_ratios`` present  -> ``"mla"`` / ``"mqa"`` /
    #     ``"window"`` / ``"csa"`` / ``"hca"``
    #   - ``sliding_window`` present       -> ``"swa"`` / ``"full"``
    #   - none of the above                -> ``None`` (homogeneous stack; keeps
    #     the legacy untagged metric keys)
    # Monitors prepend this to the metric name so statistics of different
    # attention kinds never mix in one chart.
    attn_type: str | None = None
    # Structural metadata needed to reproduce the real attention key set (see
    # ``qk_monitor``). Only available on ``csa_compress_ratios`` stacks.
    compress_ratio: int | None = None
    window_size: int | None = None
    # True when the layer runs a learned Lightning Indexer, i.e. the compressed
    # key set is top-k selected at runtime and cannot be reconstructed from
    # ``csa_compress_ratios`` alone.
    has_indexer: bool = False


# Sentinel values of ``config.csa_compress_ratios``; keep in sync with
# ``paddlefleet.transformer.transformer_config`` validation and
# ``paddlefleet.transformer.csa_attention``.
MLA_RATIO = -2
MQA_RATIO = -1
WINDOW_RATIO = 0
HCA_RATIO = 128

_RATIO_KINDS = {
    MLA_RATIO: "mla",
    MQA_RATIO: "mqa",
    WINDOW_RATIO: "window",
    HCA_RATIO: "hca",
}

# Both names are required because GDN (``gated_delta_net``, a selectable
# ``attention_layer_type``) shares ``A_log`` / ``dt_bias`` / ``conv1d`` /
# ``in_proj`` with KDA and must not match.
_KDA_ATTRS = ("f_b_proj", "gate_lower_bound")


def is_kda_layer(layer) -> bool:
    """True when this layer's token mixer is Kimi Delta Attention."""
    attn = get_attention_module(layer)
    if attn is None:
        return False
    # ``hasattr``, not truthiness: ``gate_lower_bound=None`` is a valid config
    # (it selects the unbounded softplus gate).
    return all(hasattr(attn, name) for name in _KDA_ATTRS)


def _flatten_model_chunks(model) -> list[object] | None:
    """Return flattened run_function entries from PaddleFleet VPP chunks."""
    chunks = getattr(model, "_model_chunks", None)
    if not chunks:
        return None

    layers = []
    for chunk in chunks:
        chunk_layers = get_decoder_layers(chunk)
        if chunk_layers is not None:
            layers.extend(chunk_layers)
    return layers if layers else None


def get_decoder_layers(model) -> list[object] | None:
    """Find PaddleFleet decoder layers, including VPP chunks and MTP wrappers."""
    candidates = [model]
    if hasattr(model, "_layers"):
        candidates.append(model._layers)
    if hasattr(model, "module"):
        candidates.append(model.module)

    seen = set()
    for candidate in candidates:
        if candidate is None:
            continue
        candidate_id = id(candidate)
        if candidate_id in seen:
            continue
        seen.add(candidate_id)

        chunk_layers = _flatten_model_chunks(candidate)
        if chunk_layers is not None:
            return chunk_layers
        if hasattr(candidate, "run_function"):
            return list(candidate.run_function)
        if hasattr(candidate, "decoder") and hasattr(
            candidate.decoder, "layers"
        ):
            return list(candidate.decoder.layers)
        if hasattr(candidate, "encoder") and hasattr(
            candidate.encoder, "layers"
        ):
            return list(candidate.encoder.layers)
        if hasattr(candidate, "layers"):
            return list(candidate.layers)
    return None


def is_mtp_wrapper(layer) -> bool:
    """Return True for wrapper layers that contain a real MTP transformer layer."""
    inner = getattr(layer, "transformer_layer", None)
    return inner is not None and inner is not layer


def unwrap_mtp_layer(layer):
    """Return the transformer layer to hook for a possible MTP wrapper."""
    return (
        getattr(layer, "transformer_layer", None)
        if is_mtp_wrapper(layer)
        else layer
    )


def get_attention_module(layer):
    """Return the layer's self-attention module, or ``None``."""
    attn = getattr(layer, "self_attn", None)
    if attn is None:
        attn = getattr(layer, "self_attention", None)
    return attn


def get_dsa_core(layer):
    """Return the layer's DeepSeek-Sparse-Attention core attention, or ``None``.

    DSA plugs into MLA as ``core_attention``, so there is no dedicated attention
    class to match on, and the indexer's API is not a discriminator either:
    ``CSAIndexer`` carries the same ``forward_before_topk`` / ``index_topk`` pair
    and returns the same ``(q, k, weights)`` arity. What differs is the *key
    axis* -- CSA scores compressed positions (``n_compressed ~ sq / ratio``)
    while DSA scores original tokens -- and nothing in the shapes says which,
    so the two must be told apart structurally or the metrics silently describe
    the wrong sequence.

    The CSA markers are therefore rejected explicitly: a compressor on the
    indexer, and the ``compressed_sparse_attn`` entry point on the core (the
    same attribute ``qk_monitor`` keys its sparse path off).
    """
    core = getattr(get_attention_module(layer), "core_attention", None)
    indexer = getattr(core, "indexer", None)
    if indexer is None:
        return None
    if not hasattr(indexer, "forward_before_topk") or not hasattr(
        indexer, "index_topk"
    ):
        return None
    if hasattr(indexer, "compressor") or hasattr(indexer, "compress_ratio"):
        return None
    if hasattr(core, "compressed_sparse_attn") or hasattr(
        core, "compress_ratio"
    ):
        return None
    return core


def is_dsa_layer(layer) -> bool:
    """True when this layer's core attention is DSA."""
    return get_dsa_core(layer) is not None


def _uses_compress_ratios(attn) -> bool:
    """True when the stack describes its per-layer attention kind via ratios.

    Signalled by ``config.experimental_attention_variant == "dsv4_hybrid"``,
    which is what makes ``config.csa_compress_ratios`` authoritative. Gating on
    the field (not on a model name) keeps stacks that carry no mix information
    on their existing untagged metric keys.
    """
    config = getattr(attn, "config", None)
    if config is None:
        return False
    return (
        getattr(config, "experimental_attention_variant", None) == "dsv4_hybrid"
    )


def _compress_ratio_meta(
    attn,
) -> tuple[str, int | None, int | None, bool] | None:
    """Classify one layer of a ``csa_compress_ratios`` stack.

    Returns ``(kind, compress_ratio, window_size, has_indexer)``, or ``None``
    when the stack is not ratio-described.

    Layers whose ratio selects sparse attention build a
    ``CompressedSparseAttention`` core that carries the per-layer
    ``compress_ratio``; ratio ``-2`` builds an ``MLASelfAttention`` instead,
    which exposes no ratio.
    """
    if not _uses_compress_ratios(attn):
        return None
    core = getattr(attn, "core_attention", None)
    ratio = getattr(core, "compress_ratio", None)
    if ratio is None:
        # No CSA core in a ratio-described stack -> this is the MLA branch (-2).
        return ("mla", MLA_RATIO, None, False)
    ratio = int(ratio)
    kind = _RATIO_KINDS.get(ratio)
    if kind is None:
        kind = "csa" if 2 <= ratio < HCA_RATIO else "unknown"
    window_size = getattr(core, "window_size", None)
    has_indexer = getattr(core, "indexer", None) is not None
    return (kind, ratio, window_size, has_indexer)


def attn_meta(layer) -> tuple[str | None, int | None, int | None, bool]:
    """Return ``(attn_type, compress_ratio, window_size, has_indexer)``.

    KDA layers are recognised by class; ratio-described stacks are classified
    from ``compress_ratio``; otherwise the ``is_swa`` flag is used (see
    :func:`classify_attn_type`).
    """
    attn = get_attention_module(layer)
    if attn is None:
        return (None, None, None, False)
    if is_kda_layer(layer):
        # KDA has no softmax logits, no compress ratio and no window; the tag is
        # the only piece of metadata that applies.
        return ("kda", None, None, False)
    by_ratio = _compress_ratio_meta(attn)
    if by_ratio is not None:
        return by_ratio
    is_swa = getattr(attn, "is_swa", None)
    if is_swa is None:
        return (None, None, None, False)
    return ("swa" if is_swa else "full", None, None, False)


def classify_attn_type(layer) -> str | None:
    """Classify a transformer layer's attention kind.

    Which config field describes the layer mix decides the value space; the
    mechanisms are independent and never both apply:

    1. **KDA hybrid stacks** (Kimi-Linear / Kimi-K3): a layer whose token mixer
       is Kimi Delta Attention (see :func:`is_kda_layer`) is tagged ``"kda"``. The interleaved global
       layers get their tag from :func:`iter_monitor_layers`, which is the only
       place that sees the whole stack — ``"mla"`` for ``MQALatentAttention``,
       ``"global"`` for anything else. Neither of the two mechanisms below fires
       on these stacks (no ``csa_compress_ratios``, no ``sliding_window``), so
       without this rule every layer would come back ``None`` and KDA metrics
       would share one chart with the global layers'.
    2. ``csa_compress_ratios`` (flagged by
       ``experimental_attention_variant="dsv4_hybrid"``): the per-layer kind
       implied by the ratio — ``"mla"`` (-2), ``"mqa"`` (-1), ``"window"`` (0),
       ``"csa"`` (2..127), ``"hca"`` (128). ``is_swa`` is useless here because
       it is derived from ``config.sliding_window``, which these configs do not
       set, so every layer would look like ``"full"``.
    3. ``config.sliding_window`` (+ ``window_attn_skip_freq``): the ``is_swa``
       flag that ``Attention.__init__`` computes per layer → ``"swa"`` /
       ``"full"``.

    Returns ``None`` when none of them apply (homogeneous stack), which callers
    treat as "do not tag metrics with attention kind" so those runs keep their
    existing metric key layout.
    """
    return attn_meta(layer)[0]


def _layer_config(layer):
    """Return the ``TransformerConfig`` of a layer or of the layer it wraps."""
    config = getattr(layer, "config", None)
    if config is None:
        config = getattr(
            getattr(layer, "transformer_layer", None), "config", None
        )
    return config


def _head_offset(layer) -> int:
    """``num_empty_layers_add_in_head`` of the stack this layer belongs to."""
    offset = (
        getattr(_layer_config(layer), "num_empty_layers_add_in_head", 0) or 0
    )
    return offset if isinstance(offset, int) else 0


def resolve_layer_idx(
    layer,
    local_idx: int,
    num_local_layers: int,
    pp_rank: int = 0,
    layer_offset: int = 0,
) -> int:
    """Resolve a PaddleFleet metric layer id without converting 0-based ids."""
    for attr in ("layer_idx", "layer_index", "idx"):
        value = getattr(layer, attr, None)
        if isinstance(value, int):
            return value
    layer_number = getattr(layer, "layer_number", None)
    if isinstance(layer_number, int):
        return layer_number - _head_offset(layer)
    return pp_rank * num_local_layers + layer_offset + local_idx


def _absolute_mtp_idx(wrapper, layer, mtp_local_idx: int) -> int | None:
    """Absolute metric id of an MTP layer, or ``None`` when undeterminable."""
    config = _layer_config(wrapper) or _layer_config(layer)
    num_hidden_layers = getattr(config, "num_hidden_layers", None)
    if not isinstance(num_hidden_layers, int) or num_hidden_layers <= 0:
        return None
    local = getattr(wrapper, "layer_number", None)
    if not isinstance(local, int) or local < 0:
        local = mtp_local_idx
    return num_hidden_layers + local


def _global_layer_tag(layer) -> str:
    """Tag for a non-KDA attention layer inside a KDA hybrid stack.

    KDA hybrids pair ``KimiDeltaAttention`` with ``MQALatentAttention`` for the
    periodic global layers (``gpt_layer_specs.py``), so name that case ``"mla"``.
    Anything else gets the neutral ``"global"`` rather than a guess.
    """
    attn = get_attention_module(layer)
    return "mla" if type(attn).__name__ == "MQALatentAttention" else "global"


def _retag_kda_hybrid(
    monitor_layers: list[MonitorLayer], all_layers: list[object]
) -> list[MonitorLayer]:
    """Give the global layers of a KDA hybrid stack an explicit tag.

    ``attn_meta`` only sees one layer, so it cannot tell a global layer of a KDA
    hybrid (which must be tagged, or its metrics share a chart with the KDA
    layers') from a layer of a homogeneous stack (which must stay untagged, or
    existing runs lose their metric keys). This is the only place that sees the
    whole stack, so the stack-level decision is made here. No-op unless the stack
    contains at least one KDA layer.

    Detection scans ``all_layers``, not the matched subset: a monitor that
    deliberately excludes KDA layers from its own predicate (``qk_stats`` has no
    logits to read there) must still tag its global layers the same way every
    other monitor does, or the same physical layer would carry different metric
    keys depending on which monitor emitted them.
    """
    if not any(is_kda_layer(layer) for layer in all_layers):
        return monitor_layers
    return [
        item
        if item.attn_type is not None
        else replace(item, attn_type=_global_layer_tag(item.layer))
        for item in monitor_layers
    ]


def iter_monitor_layers(
    layers: Iterable[object],
    matches: Callable[[object], bool],
    *,
    pp_rank: int = 0,
    layer_offset: int = 0,
) -> list[MonitorLayer]:
    """Return main + MTP layers that satisfy ``matches``.

    PaddleFleet MTP layers are wrappers whose real transformer layer lives at
    ``wrapper.transformer_layer``. Pipeline ``run_function`` also contains
    embedding, norm, empty, and LM head entries, so MTP metric ids must be
    assigned after matched main transformer layers rather than physical entries.

    Each returned ``MonitorLayer`` also carries an ``attn_type`` tag (see
    :func:`classify_attn_type`) so monitors that emit attention-related
    metrics can split window vs full statistics.
    """
    layers = list(layers)
    main_layers = [layer for layer in layers if not is_mtp_wrapper(layer)]
    mtp_wrappers = [layer for layer in layers if is_mtp_wrapper(layer)]
    matched_main_layers = [layer for layer in main_layers if matches(layer)]
    num_main_layers = len(matched_main_layers)
    monitor_layers: list[MonitorLayer] = []

    def _make(idx: int, layer: object, is_mtp: bool) -> MonitorLayer:
        kind, ratio, window_size, has_indexer = attn_meta(layer)
        return MonitorLayer(
            idx=idx,
            layer=layer,
            is_mtp=is_mtp,
            attn_type=kind,
            compress_ratio=ratio,
            window_size=window_size,
            has_indexer=has_indexer,
        )

    for local_idx, layer in enumerate(matched_main_layers):
        idx = resolve_layer_idx(
            layer,
            local_idx,
            num_main_layers,
            pp_rank=pp_rank,
            layer_offset=layer_offset,
        )
        monitor_layers.append(_make(idx, layer, False))

    for mtp_idx, wrapper in enumerate(mtp_wrappers):
        layer = unwrap_mtp_layer(wrapper)
        if not matches(layer):
            continue
        idx = _absolute_mtp_idx(wrapper, layer, mtp_idx)
        if idx is None:
            logger.warning(
                "Skipping MTP metric layer: num_hidden_layers is unavailable, so no "
                "layout-independent id can be assigned. Deriving one from the main layer ids "
                "present on this rank would silently mislabel it (that rule collapses to id 0 "
                "when no main layer matched), so no metric is emitted for this layer."
            )
            continue
        monitor_layers.append(_make(idx, layer, True))

    return _retag_kda_hybrid(monitor_layers, layers)
