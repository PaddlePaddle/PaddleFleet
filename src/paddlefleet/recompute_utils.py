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
from paddle.distributed.fleet.meta_parallel.zero_bubble_utils import (
    RecomputeStore,
)

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
    assert total_num_hidden_layers % parallel_size == 0, (
        "num_hidden_layers must be divided by parallel_size"
    )
    chunk_size = int(total_num_hidden_layers / parallel_size)
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
    assert total_num_hidden_layers % parallel_size == 0, (
        "num_hidden_layers must be divided by parallel_size"
    )
    chunk_size = int(total_num_hidden_layers / parallel_size)
    num_layers_in_each_stage = (
        total_num_hidden_layers / config.pipeline_model_parallel_size
    )
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


def effective_mtp_layers(config):
    """MTP layer count the model actually builds."""
    mtp_num_layers = getattr(config, "mtp_num_layers", 0) or 0
    nextn_num_layers = getattr(config, "num_nextn_predict_layers", 0) or 0
    if not isinstance(mtp_num_layers, int) or isinstance(mtp_num_layers, bool):
        mtp_num_layers = 0
    if not isinstance(nextn_num_layers, int) or isinstance(
        nextn_num_layers, bool
    ):
        nextn_num_layers = 0
    if (
        mtp_num_layers > 0
        and nextn_num_layers > 0
        and mtp_num_layers != nextn_num_layers
    ):
        raise ValueError(
            "mtp_num_layers and num_nextn_predict_layers must be equal when "
            f"both are positive, got {mtp_num_layers} and {nextn_num_layers}"
        )
    return mtp_num_layers if mtp_num_layers > 0 else nextn_num_layers


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
        return
    if not isinstance(recompute_modules, dict):
        raise ValueError(
            "recompute_modules must be a sequence or dict, got "
            f"{type(recompute_modules).__name__}"
        )

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
