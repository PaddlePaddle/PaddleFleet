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

from __future__ import annotations

import logging
import os
from itertools import chain

import paddle

from paddlefleet.accuracy_compatible_patch import (
    HAS_RECOMPUTE_STORE,
    RecomputeStore,
)
from paddlefleet.utils import use_dsv4_accuracy_compatible

logger = logging.getLogger(__name__)

g_has_print_recovery_log = False


def install_recompute_p2p_overlap(config):
    """Let the pp scheduler recompute the next backward chunk inside a p2p window.

    Reads ``config.p2p_overlap_recompute`` onto the process-global
    ``RecomputeStore``, which is where the scheduler looks; off means no span
    ever registers. Idempotent, so it is safe to call from every layer's
    constructor.
    """
    enabled = bool(getattr(config, "p2p_overlap_recompute", False))
    if enabled and use_dsv4_accuracy_compatible() and not HAS_RECOMPUTE_STORE:
        raise RuntimeError(
            "p2p_overlap_recompute requires a Paddle runtime with "
            "RecomputeStore support"
        )
    if enabled and config.recompute_granularity != "selective":
        raise ValueError(
            "p2p_overlap_recompute needs recompute_granularity='selective', "
            f"got {config.recompute_granularity!r}: there are no recompute "
            "spans to run early otherwise"
        )
    if enabled and getattr(config, "pipeline_model_parallel_size", 1) <= 1:
        raise ValueError(
            "p2p_overlap_recompute needs pipeline_model_parallel_size > 1: "
            "without pipeline parallel there is no p2p window to fill"
        )
    vpp_size = getattr(config, "virtual_pipeline_model_parallel_size", None)
    if enabled and (vpp_size is None or vpp_size <= 1):
        raise ValueError(
            "p2p_overlap_recompute needs "
            "virtual_pipeline_model_parallel_size > 1: the recompute store "
            "lifecycle is only established by the interleaved/VPP scheduler; "
            "ordinary pipeline parallel has no named chunk to run"
        )
    RecomputeStore.enabled = enabled


def keep_indexer_grad_path(hidden_states, config):
    """Keep a recompute segment differentiable when only the CSA Indexer trains.

    Every recompute wrapper in this repo is a PyLayer, so whether its output is
    differentiable depends only on its input tensors, not on the parameters used
    inside. With every backbone parameter frozen (``train_indexer_only``) the
    segment input has ``stop_gradient=True``, the segment output inherits it, and
    the indexer loss attached inside the segment silently never gets a backward
    pass. ``RecomputeWithoutOutput`` is worse than ``recompute``: it skips
    registering its recompute hook entirely when the hook tensor is detached
    (``tensor_parallel/random.py:590``), so there is not even a warning.

    Re-entering the autograd graph through a scalar anchor restores that path
    without changing any activation value. Only the first segment whose input is
    still detached pays for it: once a segment output is differentiable, the
    following ones short-circuit here.

    Call this on the input of every recompute segment that can contain a CSA
    Indexer: the base ``TransformerLayer`` layer-level segment, and
    ``DSv4HybridAttention``'s inner ``full_attn`` segment.
    """
    if not getattr(config, "train_indexer_only", False):
        return hidden_states
    if not isinstance(hidden_states, paddle.Tensor):
        return hidden_states
    if not hidden_states.stop_gradient or not paddle.is_grad_enabled():
        return hidden_states
    anchor = paddle.zeros([1], dtype=hidden_states.dtype)
    anchor.stop_gradient = False
    return hidden_states + anchor


def has_recovered():
    """has recovered"""
    recover_step = os.getenv("RECOVER_STEP")
    if recover_step is None:
        return True
    recover_step = int(recover_step)
    current_step = os.getenv("TRAINER_GLOBAL_STEP")
    if current_step is None:
        current_step = os.getenv("PDC_INIT_STEP")
        assert current_step is not None, (
            "TRAINER_GLOBAL_STEP or PDC_INIT_STEP should be specified"
        )
    current_step = int(current_step)
    if current_step > recover_step:
        global g_has_print_recovery_log
        if not g_has_print_recovery_log:
            logger.info(f"Recovery would be enabled in the step {current_step}")
            g_has_print_recovery_log = True
        return True
    else:
        return False


def need_recompute_in_block(layer_number, config, recompute_num_layers):
    assert recompute_num_layers is not None, (
        "recompute_num_layers cannot be none"
    )

    if recompute_num_layers < 0:
        return True

    total_num_hidden_layers = (
        config.num_empty_layers_add_in_head
        + config.num_hidden_layers
        + config.num_empty_layers_add_in_tail
    )
    vpp_size = (
        config.virtual_pipeline_model_parallel_size
        if config.virtual_pipeline_model_parallel_size
        else 1
    )
    parallel_size = config.pipeline_model_parallel_size * vpp_size
    # This count only covers decoder layers; the network split also includes
    # other units (embedding/MTP/heads/loss), so divisibility is not required.
    # ceil division lets the last chunk be unfilled, e.g. 15 layers pp=4 ->
    # chunks 4,4,4,3.
    chunk_size = (total_num_hidden_layers + parallel_size - 1) // parallel_size
    assert recompute_num_layers <= chunk_size
    layers = list(range(total_num_hidden_layers))
    recompute_layers = list(
        chain.from_iterable(
            [
                layers[i : i + recompute_num_layers]
                for i in range(0, len(layers), chunk_size)
            ]
        )
    )
    if layer_number in recompute_layers:
        return True
    return False


def _pipeline_chunk_size(config):
    """Return ``(total decoder layers, layers per pipeline chunk, rounded up)``."""
    total_num_hidden_layers = (
        config.num_empty_layers_add_in_head
        + config.num_hidden_layers
        + config.num_empty_layers_add_in_tail
    )
    vpp_size = (
        config.virtual_pipeline_model_parallel_size
        if config.virtual_pipeline_model_parallel_size
        else 1
    )
    parallel_size = config.pipeline_model_parallel_size * vpp_size
    chunk_size = (total_num_hidden_layers + parallel_size - 1) // parallel_size
    return total_num_hidden_layers, chunk_size


def explicit_mhc_recompute_blocks(config):
    """Return the explicit mHC block list, or ``None`` for uniform sizing.

    ``recompute_modules['mhc_block']`` accepts a *nested* layer list, one inner
    list per block::

        recompute_modules:
          mhc_block: [[3, 4, 5], [9, 10]]

    Layers outside every block do no mHC recompute. Layer ids use the
    ``logical_layer_index`` space, as everywhere else in ``recompute_modules``.
    Nesting is what separates "one block of layers 3-5" from "three one-layer
    blocks", so a flat list is rejected rather than guessed at.
    """
    _, layer_selector = _get_module_recompute_config("mhc_block", config)
    if not isinstance(layer_selector, (list, tuple)) or not layer_selector:
        return None
    nested = [isinstance(block, (list, tuple)) for block in layer_selector]
    if not any(nested):
        return None
    if not all(nested):
        raise ValueError(
            "recompute_modules['mhc_block'] mixes blocks with bare layer ids "
            f"({layer_selector!r}): give one inner list per block, e.g. "
            "[[3, 4, 5], [9, 10]]"
        )
    return tuple(tuple(block) for block in layer_selector)


def mhc_recompute_block_plan(layer_number, config, is_mtp_layer=False):
    """Return ``(block_id, is_block_end)`` for an mHC layer.

    ``(None, False)`` means this layer does no mHC recompute, which only an
    explicit block list can produce.

    Without an explicit list, blocks are split within each pipeline chunk so
    they never cross a stage; ``mhc_recompute_layer_num=None`` means one block
    per chunk. An explicit list is held to the same constraint by
    ``_validate_mhc_block_recompute``. MTP layers always form independent
    one-layer blocks, since their layer numbers are separate from the
    backbone's.
    """
    if is_mtp_layer:
        return ("mtp", int(layer_number)), True

    blocks = explicit_mhc_recompute_blocks(config)
    if blocks is not None:
        layer_id = logical_layer_index(config, layer_number)
        for index, block in enumerate(blocks):
            if layer_id in block:
                return ("explicit", index), layer_id == block[-1]
        return None, False

    _, chunk_size = _pipeline_chunk_size(config)
    chunk_index, index_in_chunk = divmod(layer_number, chunk_size)
    # The block ends on the last *real* mHC layer of the chunk, not the last
    # physical slot: tail EmptyLayers occupy chunk slots but never run
    # finalize_mhc_recompute_block, so counting them would leave the block's
    # is_block_end permanently False and the manager never discarded. The last
    # real layer is head_offset + num_hidden_layers - 1; in every chunk before
    # the one holding it the last real index is chunk_size - 1.
    head_offset = getattr(config, "num_empty_layers_add_in_head", 0) or 0
    last_real_layer = head_offset + config.num_hidden_layers - 1
    last_index_in_chunk = min(
        chunk_size - 1,
        last_real_layer - chunk_index * chunk_size,
    )

    block_size = config.mhc_recompute_layer_num or chunk_size
    block_in_chunk, index_in_block = divmod(index_in_chunk, block_size)
    is_block_end = (
        index_in_block == block_size - 1
        or index_in_chunk == last_index_in_chunk
    )
    return (chunk_index, block_in_chunk), is_block_end


def need_recompute_in_first_n(layer_number, config, recompute_num_layers):
    assert recompute_num_layers is not None, (
        "recompute_num_layers cannot be none"
    )
    total_num_hidden_layers = (
        config.num_empty_layers_add_in_head
        + config.num_hidden_layers
        + config.num_empty_layers_add_in_tail
    )
    vpp_size = (
        config.virtual_pipeline_model_parallel_size
        if config.virtual_pipeline_model_parallel_size
        else 1
    )
    parallel_size = config.pipeline_model_parallel_size * vpp_size
    # This count only covers decoder layers; the network split also includes
    # other units (embedding/MTP/heads/loss), so divisibility is not required.
    # ceil division lets the last chunk be unfilled, e.g. 15 layers pp=4 ->
    # chunks 4,4,4,3.
    chunk_size = (total_num_hidden_layers + parallel_size - 1) // parallel_size
    num_layers_in_each_stage = (
        total_num_hidden_layers + config.pipeline_model_parallel_size - 1
    ) // config.pipeline_model_parallel_size
    assert recompute_num_layers <= num_layers_in_each_stage, (
        "recompute_num_layers cannot be greater than num_layers_in_each_stage"
    )
    if vpp_size > 1:
        layers = range(total_num_hidden_layers)
        chunks = [
            layers[i * chunk_size : (i + 1) * chunk_size]
            for i in range(0, len(layers), chunk_size)
        ]
        recompute_layers = []
        for pp_stage in range(config.pipeline_model_parallel_size):
            recompute_layers_in_curr_stage = list(
                chain.from_iterable(
                    chunks[pp_stage :: config.pipeline_model_parallel_size]
                )
            )[:recompute_num_layers]
            recompute_layers += recompute_layers_in_curr_stage
    else:
        recompute_layers = []
        layers = list(range(total_num_hidden_layers))
        if config.pipeline_model_parallel_size > 1:
            for recompute_layer_id in range(recompute_num_layers):
                recompute_layers_in_curr_stage = list(
                    layers[recompute_layer_id::chunk_size]
                )
                recompute_layers += recompute_layers_in_curr_stage
        else:
            recompute_layers = list(
                range(
                    config.pipeline_model_parallel_size * recompute_num_layers
                )
            )
    if layer_number in recompute_layers:
        return True
    return False


RECOMPUTE_ALL_LAYERS = "all"
"""Dict value meaning "every layer" (``-1`` also works)."""

LAYER_AGNOSTIC_RECOMPUTE_MODULES = frozenset({"lm_head", "loss_fn"})
"""Single-instance modules: no layer number, so a layer list is rejected."""

REFINED_RECOMPUTE_MODULES = frozenset({"flash_attn", "moe_combine"})
"""RR modules: count-based selectors always use ``first_n``."""

BLOCK_SCOPED_RECOMPUTE_MODULES = frozenset({"mhc_block"})
"""Modules spanning layers: a flat layer selector is rejected, a nested one is
the explicit block list."""


def effective_mtp_layers(config):
    """MTP layer count the model actually builds."""
    nextn_num_layers = getattr(config, "num_nextn_predict_layers", 0) or 0
    if not isinstance(nextn_num_layers, int) or isinstance(
        nextn_num_layers, bool
    ):
        nextn_num_layers = 0
    return nextn_num_layers


def logical_layer_index(config, layer_number, is_mtp_layer=False):
    """Map a physical ``layer_number`` to the config-facing layer index."""
    if is_mtp_layer:
        return config.num_hidden_layers + layer_number
    head_offset = getattr(config, "num_empty_layers_add_in_head", 0) or 0
    return layer_number - head_offset


def _get_module_recompute_config(module_name, config):
    """Return whether ``module_name`` is configured, and its layer selector.

    The selector is the dict value in dict mode, or the shared
    ``config.recompute_num_layers`` in list mode.
    """
    recompute_modules = config.recompute_modules
    if recompute_modules is None:
        return False, None
    if isinstance(recompute_modules, dict):
        if module_name not in recompute_modules:
            return False, None
        return True, recompute_modules[module_name]
    if isinstance(recompute_modules, (list, tuple, set, frozenset)):
        if module_name not in recompute_modules:
            return False, None
        return True, config.recompute_num_layers
    raise ValueError(
        "recompute_modules must be a sequence or dict, got "
        f"{type(recompute_modules).__name__}"
    )


def normalize_recompute_layer_ids(layer_selector, module_name):
    """Validate an explicit layer-id selector and return it as a frozenset."""
    layer_ids = set()
    for layer_id in layer_selector:
        if isinstance(layer_id, bool) or not isinstance(layer_id, int):
            raise ValueError(
                f"recompute_modules['{module_name}'] layer ids must be ints, "
                f"got {layer_id!r}"
            )
        if layer_id < 0:
            raise ValueError(
                f"recompute_modules['{module_name}'] layer ids must be "
                f"non-negative, got {layer_id}"
            )
        layer_ids.add(layer_id)
    return frozenset(layer_ids)


def _selector_matches_layer(
    layer_selector,
    layer_number,
    config,
    module_name,
    defer_if_layer_unknown=False,
    is_mtp_layer=False,
):
    """Whether ``layer_selector`` selects ``layer_number``.

    Selectors: ``None`` / ``"all"`` / negative int mean every layer; a list of
    ints means those layer ids in the ``logical_layer_index`` space (the one
    ``csa_compress_ratios`` uses); a non-negative int is a layer count resolved
    through ``config.recompute_method`` over the physical layer number.

    ``layer_number`` is ``None`` for layer-agnostic modules and for MoE
    submodules before ``set_layer_number()``. A count then means every layer.
    A layer list raises, unless ``defer_if_layer_unknown`` says the caller will
    ask again with a real layer number.
    """
    if layer_selector is None or layer_selector == RECOMPUTE_ALL_LAYERS:
        return True

    if isinstance(layer_selector, (list, tuple, set, frozenset)):
        layer_ids = normalize_recompute_layer_ids(layer_selector, module_name)
        if layer_number is None:
            if defer_if_layer_unknown:
                return False
            raise ValueError(
                f"recompute_modules['{module_name}'] was given an explicit "
                f"layer list {sorted(layer_ids)}, but '{module_name}' has no "
                "layer number to filter on. Use "
                f"'{RECOMPUTE_ALL_LAYERS}' to enable it everywhere."
            )
        return (
            logical_layer_index(config, layer_number, is_mtp_layer) in layer_ids
        )

    if isinstance(layer_selector, bool) or not isinstance(layer_selector, int):
        raise ValueError(
            f"recompute_modules['{module_name}'] must be an int, a list of "
            f"layer ids, or '{RECOMPUTE_ALL_LAYERS}', got "
            f"{layer_selector!r}"
        )

    if layer_selector < 0:
        return True
    if layer_number is None:
        return True
    if config.recompute_method == "block":
        return need_recompute_in_block(layer_number, config, layer_selector)
    if config.recompute_method in ("first_n", None):
        return need_recompute_in_first_n(layer_number, config, layer_selector)
    raise ValueError(
        f"recompute_modules['{module_name}']={layer_selector} needs recompute_method to "
        f"be 'first_n' or 'block', got {config.recompute_method!r}"
    )


_logged_recompute_decisions = set()


def _log_recompute_decision(
    kind, module_name, layer_number, enabled, is_mtp_layer=False
):
    """Log one decision, deduped: MoE resolves its flags twice."""
    key = (kind, module_name, layer_number, is_mtp_layer)
    if key in _logged_recompute_decisions:
        return
    _logged_recompute_decisions.add(key)
    layer_text = "n/a" if layer_number is None else str(layer_number)
    if is_mtp_layer:
        layer_text = f"mtp{layer_text}"
    logger.info(
        f"[RECOMPUTE-DECISION] kind={kind} module={module_name} "
        f"layer={layer_text} enabled={enabled}"
    )


def module_needs_recompute(
    module_name,
    layer_number,
    config,
    defer_if_layer_unknown=False,
    is_mtp_layer=False,
):
    """Whether ``module_name`` should be recomputed on layer ``layer_number``.

    Single entry point for every ``recompute_modules`` lookup. Only meaningful
    under ``recompute_granularity == "selective"``; ``lm_head`` and ``loss_fn``
    keep ignoring the granularity, as they always did.

    ``layer_number`` is physical; ``is_mtp_layer`` routes layer lists through
    ``logical_layer_index`` so MTP layers do not collide with backbone layer 0.

    Pass ``defer_if_layer_unknown=True`` when ``layer_number=None`` just means
    "not yet known" and the caller will ask again: a layer list then resolves to
    False instead of raising. MoE submodules need this.
    """
    module_configured, layer_selector = _get_module_recompute_config(
        module_name, config
    )
    if not module_configured:
        # Queried on every layer, so logging these would bury the real ones.
        return False
    if module_name in LAYER_AGNOSTIC_RECOMPUTE_MODULES:
        # No layer to filter on; a layer list is rejected during validation.
        _log_recompute_decision("plain", module_name, layer_number, True)
        return True
    if module_name in BLOCK_SCOPED_RECOMPUTE_MODULES:
        # Per-layer participation comes from the block plan, not the selector.
        _log_recompute_decision("plain", module_name, layer_number, True)
        return True
    enabled = _selector_matches_layer(
        layer_selector,
        layer_number,
        config,
        module_name,
        defer_if_layer_unknown=defer_if_layer_unknown,
        is_mtp_layer=is_mtp_layer,
    )
    _log_recompute_decision(
        "plain", module_name, layer_number, enabled, is_mtp_layer
    )
    return enabled


def module_needs_refined_recompute(
    module_name, layer_number, config, is_mtp_layer=False
):
    """Whether ``module_name`` should use refined recompute (RR) on this layer.

    RR inverts the selector: selected layers keep the plain recompute path, RR
    runs on the rest. So ``"all"`` / ``None`` / a negative count disable RR,
    while ``0`` selects nothing and enables it everywhere; a list-mode entry
    carries no layer info and also enables it everywhere.

    Count-based selectors always resolve with ``first_n``; only ``moe_combine``
    rejects a different ``recompute_method``, and it does so itself.
    """
    module_configured, layer_selector = _get_module_recompute_config(
        module_name, config
    )
    if not module_configured:
        return False
    if not isinstance(config.recompute_modules, dict):
        _log_recompute_decision(
            "rr", module_name, layer_number, True, is_mtp_layer
        )
        return True
    if layer_selector is None or layer_selector == RECOMPUTE_ALL_LAYERS:
        _log_recompute_decision(
            "rr", module_name, layer_number, False, is_mtp_layer
        )
        return False
    if isinstance(layer_selector, (list, tuple, set, frozenset)):
        layer_ids = normalize_recompute_layer_ids(layer_selector, module_name)
        if layer_number is None:
            raise ValueError(
                f"recompute_modules['{module_name}'] was given an explicit "
                f"layer list but no layer number is available"
            )
        enabled = (
            logical_layer_index(config, layer_number, is_mtp_layer)
            not in layer_ids
        )
        _log_recompute_decision(
            "rr", module_name, layer_number, enabled, is_mtp_layer
        )
        return enabled
    if isinstance(layer_selector, bool) or not isinstance(layer_selector, int):
        raise ValueError(
            f"recompute_modules['{module_name}'] must be an int, a list of "
            f"layer ids, or '{RECOMPUTE_ALL_LAYERS}', got {layer_selector!r}"
        )
    if layer_selector < 0:
        # Same as "all"/None. Handled here because need_recompute_in_first_n
        # selects no layer for a negative count, which would invert into "RR
        # everywhere" -- the exact opposite.
        _log_recompute_decision(
            "rr", module_name, layer_number, False, is_mtp_layer
        )
        return False
    enabled = not need_recompute_in_first_n(
        layer_number, config, layer_selector
    )
    _log_recompute_decision(
        "rr", module_name, layer_number, enabled, is_mtp_layer
    )
    return enabled


def _validate_mhc_block_recompute(config):
    """Cross-field checks for ``recompute_modules=['mhc_block']``."""
    if "mhc_block" not in (config.recompute_modules or ()):
        return
    if not config.enable_hyper_connections:
        raise ValueError(
            "recompute_modules['mhc_block'] requires "
            "enable_hyper_connections=True: there is no mHC recompute to "
            "group otherwise."
        )
    if config.recompute_granularity != "selective":
        raise ValueError(
            "recompute_modules['mhc_block'] requires "
            "recompute_granularity='selective', got "
            f"{config.recompute_granularity!r}. Under 'full' the whole layer is "
            "replayed, so the block manager would collect each span twice in "
            "one micro-batch."
        )
    if "mhc_forward" in config.recompute_modules:
        raise ValueError(
            "recompute_modules cannot contain both 'mhc_block' and "
            "'mhc_forward': they are the block-scoped and half-layer-scoped "
            "versions of the same mechanism. 'mhc_block' strictly subsumes "
            "'mhc_forward'."
        )
    block_size = config.mhc_recompute_layer_num
    if block_size is not None and (
        isinstance(block_size, bool)
        or not isinstance(block_size, int)
        or block_size < 1
    ):
        raise ValueError(
            "mhc_recompute_layer_num must be a positive integer or None, got "
            f"{block_size!r}"
        )
    if block_size is not None:
        _, chunk_size = _pipeline_chunk_size(config)
        if block_size > chunk_size:
            raise ValueError(
                f"mhc_recompute_layer_num={block_size} exceeds the "
                f"{chunk_size} decoder layers per pipeline chunk. A recompute "
                "block cannot span a pipeline stage."
            )
    _validate_explicit_mhc_blocks(config, block_size)


def _validate_explicit_mhc_blocks(config, block_size):
    """Check the nested ``recompute_modules['mhc_block']`` block list."""
    blocks = explicit_mhc_recompute_blocks(config)
    if blocks is None:
        return
    if block_size is not None:
        raise ValueError(
            f"recompute_modules['mhc_block']={list(map(list, blocks))} lists "
            "the blocks explicitly, so mhc_recompute_layer_num="
            f"{block_size} has nothing to size. Drop one of the two."
        )

    seen = {}
    for index, block in enumerate(blocks):
        _validate_one_mhc_block(config, index, block, seen)


def _validate_one_mhc_block(config, index, block, seen):
    """Check one inner block list and record its layer ids in ``seen``."""
    where = f"recompute_modules['mhc_block'][{index}]"
    if not block:
        raise ValueError(f"{where} is empty")
    for layer_id in block:
        if isinstance(layer_id, bool) or not isinstance(layer_id, int):
            raise ValueError(f"{where} layer ids must be ints, got {block!r}")
    if list(block) != list(range(block[0], block[0] + len(block))):
        raise ValueError(
            f"{where}={list(block)} is not a run of consecutive layers. A "
            "block hooks its last layer's residual state and replays the rest "
            "off it, so a gap inside a block would keep the skipped layers' "
            "activations alive for the whole block instead of their own span."
        )
    if block[0] < 0 or block[-1] >= config.num_hidden_layers:
        raise ValueError(
            f"{where}={list(block)} is out of range for the "
            f"{config.num_hidden_layers} backbone layers (0-based, excluding "
            "empty head/tail layers; MTP layers always form their own blocks)"
        )
    for layer_id in block:
        if layer_id in seen:
            raise ValueError(
                f"{where} repeats layer {layer_id}, already in "
                f"recompute_modules['mhc_block'][{seen[layer_id]}]. A layer "
                "belongs to at most one block."
            )
        seen[layer_id] = index

    head_offset = getattr(config, "num_empty_layers_add_in_head", 0) or 0
    _, chunk_size = _pipeline_chunk_size(config)
    first_chunk = (block[0] + head_offset) // chunk_size
    last_chunk = (block[-1] + head_offset) // chunk_size
    if first_chunk != last_chunk:
        boundary = (first_chunk + 1) * chunk_size - head_offset
        raise ValueError(
            f"{where}={list(block)} crosses a pipeline chunk boundary at layer "
            f"{boundary}. A block cannot span a pipeline stage: the earlier "
            "stage would never see the boundary tensor its hook goes on.\n"
            + mhc_chunk_layout_text(config)
            + f"\nEach block must sit inside one of those spans; split "
            f"{list(block)} at layer {boundary}."
        )


def mhc_chunk_layout_text(config):
    """Human-readable map of which layer ids each pipeline chunk holds.

    Empty head/tail layers occupy chunk slots without being addressable, which
    is what makes an explicit block list easy to get wrong -- hence printing the
    layout in error messages instead of leaving the arithmetic to the reader.
    """
    head_offset = getattr(config, "num_empty_layers_add_in_head", 0) or 0
    tail_offset = getattr(config, "num_empty_layers_add_in_tail", 0) or 0
    total, chunk_size = _pipeline_chunk_size(config)
    spans = []
    for chunk_index in range(-(-total // chunk_size)):
        first = chunk_index * chunk_size - head_offset
        last = min(first + chunk_size - 1, config.num_hidden_layers - 1)
        if last >= max(first, 0):
            spans.append(f"chunk {chunk_index}: layers {max(first, 0)}-{last}")

    empties = []
    if head_offset:
        empties.append(f"{head_offset} empty head")
    if tail_offset:
        empties.append(f"{tail_offset} empty tail")
    note = ""
    if empties:
        note = (
            f" ({' + '.join(empties)} layers take up chunk slots without being "
            "addressable, which is why the first/last chunk is short)"
        )
    return (
        f"{config.num_hidden_layers} backbone layers, "
        f"pipeline_model_parallel_size={config.pipeline_model_parallel_size} x "
        "virtual_pipeline_model_parallel_size="
        f"{config.virtual_pipeline_model_parallel_size} => {chunk_size} layers "
        f"per chunk{note}:\n  " + "\n  ".join(spans)
    )


def make_refined_recompute(owner, point, supported=True):
    """Return ``(enabled, boundary)`` for one refined-recompute point.

    ``supported`` is a point-specific precondition only the call site knows.

    Two preconditions are checked here instead of per point: the layer must be
    under full recompute, or the boundary retains a frame no backward consumes;
    and there must be no virtual pipeline, which reorders replay against the
    first forward and would pair frames across chunks. A vetoed point warns
    rather than raises -- it only costs the speedup.
    """
    # Local import: recompute_utils is imported nearly everywhere, while
    # refined_recompute pulls in flash_attn and paddlefleet_ops.
    from paddlefleet.refined_recompute import AutoRefinedRecompute

    if not module_needs_refined_recompute(
        point,
        owner.layer_number,
        owner.config,
        is_mtp_layer=getattr(owner, "is_mtp_layer", False),
    ):
        return False, None

    vpp = getattr(owner.config, "virtual_pipeline_model_parallel_size", None)
    in_full = need_full_recompute(owner.layer_number, owner.config)
    if not (supported and in_full and (vpp is None or vpp <= 1)):
        logger.warning(
            f"[RECOMPUTE-DECISION] kind=rr module={point} "
            f"layer={owner.layer_number} enabled=False vetoed "
            f"supported={supported} full_recompute={in_full} "
            f"virtual_pipeline={vpp}"
        )
        return False, None
    return True, AutoRefinedRecompute(point)


def validate_recompute_modules(config):
    """Structural check of ``config.recompute_modules``, run from config init.

    Fails on malformed selectors and out-of-range layer ids at startup rather
    than deep inside a layer constructor.
    """
    recompute_modules = config.recompute_modules
    if recompute_modules is None:
        return
    if isinstance(recompute_modules, (list, tuple, set, frozenset)):
        for module_name in recompute_modules:
            if not isinstance(module_name, str):
                raise ValueError(
                    "recompute_modules entries must be str, got "
                    f"{module_name!r}"
                )
        _validate_mhc_block_recompute(config)
        return
    if not isinstance(recompute_modules, dict):
        raise ValueError(
            "recompute_modules must be a sequence or dict, got "
            f"{type(recompute_modules).__name__}"
        )
    _validate_mhc_block_recompute(config)

    # Layer lists live in the logical_layer_index space: backbone layers then
    # MTP layers. Empty head/tail layers hold no module and are not addressable.
    num_layer_ids = config.num_hidden_layers + effective_mtp_layers(config)
    for module_name, layer_selector in recompute_modules.items():
        if not isinstance(module_name, str):
            raise ValueError(
                f"recompute_modules keys must be str, got {module_name!r}"
            )
        if layer_selector is None or layer_selector == RECOMPUTE_ALL_LAYERS:
            continue
        if module_name in BLOCK_SCOPED_RECOMPUTE_MODULES:
            if explicit_mhc_recompute_blocks(config) is not None:
                # Already checked by _validate_mhc_block_recompute above.
                continue
            raise ValueError(
                f"recompute_modules['{module_name}'] does not support a flat "
                f"layer selector ({layer_selector!r}): '{module_name}' groups "
                "consecutive layers into a block and hooks the block's last "
                "residual state, so a flat list cannot say whether [3, 4, 5] "
                "is one block or three. Nest one list per block, e.g. "
                "[[3, 4, 5], [9, 10]]; or use "
                f"'{RECOMPUTE_ALL_LAYERS}' with mhc_recompute_layer_num for "
                "uniformly sized blocks over every layer."
            )
        if isinstance(layer_selector, (list, tuple, set, frozenset)):
            layer_ids = normalize_recompute_layer_ids(
                layer_selector, module_name
            )
            if module_name in LAYER_AGNOSTIC_RECOMPUTE_MODULES:
                raise ValueError(
                    f"recompute_modules['{module_name}'] does not support a "
                    f"layer list: '{module_name}' is not a per-layer module. "
                    f"Use '{RECOMPUTE_ALL_LAYERS}'."
                )
            out_of_range_layer_ids = [
                layer_id
                for layer_id in sorted(layer_ids)
                if layer_id >= num_layer_ids
            ]
            if out_of_range_layer_ids:
                raise ValueError(
                    f"recompute_modules['{module_name}'] layer ids "
                    f"{out_of_range_layer_ids} are "
                    f"out of range for {num_layer_ids} layer ids (0-based: "
                    f"backbone layers 0..{config.num_hidden_layers - 1} "
                    "excluding empty head/tail layers, then the MTP layers)"
                )
            continue
        if isinstance(layer_selector, bool) or not isinstance(
            layer_selector, int
        ):
            raise ValueError(
                f"recompute_modules['{module_name}'] must be an int, a list "
                f"of layer ids, or '{RECOMPUTE_ALL_LAYERS}', got "
                f"{layer_selector!r}"
            )
        if layer_selector >= 0 and (
            module_name not in REFINED_RECOMPUTE_MODULES
            and config.recompute_method not in ("first_n", "block")
        ):
            raise ValueError(
                f"recompute_modules['{module_name}']={layer_selector} is a "
                "layer count and "
                "needs recompute_method to be 'first_n' or 'block', got "
                f"{config.recompute_method!r}. Use a layer list to select "
                "layers explicitly."
            )


def need_full_recompute(layer_number, config):
    if config.recompute_granularity == "full":
        if config.recompute_method == "uniform":
            assert config.recompute_num_layers == 1, (
                "don't support recompute_method=uniform wihile recompute_num_layers != 1"
            )
            return True
        elif config.recompute_method == "first_n":
            return need_recompute_in_first_n(
                layer_number, config, config.recompute_num_layers
            )
        elif config.recompute_method == "block":
            return need_recompute_in_block(
                layer_number, config, config.recompute_num_layers
            )
    return False
