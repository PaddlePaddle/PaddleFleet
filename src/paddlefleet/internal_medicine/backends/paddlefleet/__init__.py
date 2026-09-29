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

"""PaddleFleet backend for internal_medicine."""

import logging

from ...core.metric_families import parse_exclusions, validate_exclusions
from .ape_monitor import PaddleAPEHealthMonitor, setup_ape_monitor
from .attn_update_monitor import (
    PaddleAttnUpdateMonitor,
    setup_attn_update_monitor,
)
from .base import PaddleProbe
from .dsa_monitor import PaddleDSAHealthMonitor, setup_dsa_monitor
from .gather import install_gather_fn
from .kda_monitor import PaddleKDAHealthMonitor, setup_kda_monitor
from .massive_activation_monitor import (
    PaddleMassiveActivationMonitor,
    setup_massive_activation_monitor,
)
from .mhc_monitor import PaddleMHCHealthMonitor, setup_mhc_monitor
from .mlp_update_monitor import PaddleMLPUpdateMonitor, setup_mlp_update_monitor
from .moe_monitor import PaddleMoEMonitor, setup_moe_monitor
from .qk_monitor import PaddleQKStatsMonitor, setup_qk_monitor
from .vha_monitor import PaddleVHAHealthMonitor, setup_vha_monitor

logger = logging.getLogger(__name__)

_MONITOR_MAP = {
    "ape_health": setup_ape_monitor,
    "qk_stats": setup_qk_monitor,
    "moe_health": setup_moe_monitor,
    "massive_act": setup_massive_activation_monitor,
    "mhc_health": setup_mhc_monitor,
    "vha_health": setup_vha_monitor,
    "attn_update": setup_attn_update_monitor,
    "mlp_update": setup_mlp_update_monitor,
    "kda_health": setup_kda_monitor,
    "dsa_health": setup_dsa_monitor,
}

_MODEL_MONITOR_ATTR = "_internal_medicine_paddlefleet_monitors"
_MONITOR_CONFIG_ATTR = "_internal_medicine_paddlefleet_config"


def _monitor_config(monitor_interval, verbose, options):
    return {
        "monitor_interval": monitor_interval,
        "verbose": verbose,
        "options": repr(sorted(options.items())),
    }


def _monitor_has_active_hooks(monitor):
    hooks = getattr(monitor, "hooks", None)
    if hooks:
        return True
    # mhc_health does not register forward hooks; it wraps ``compute_mappings``.
    # Fall back to its wrapped-method registry to detect an active setup.
    wrapped = getattr(monitor, "_wrapped", None)
    return bool(wrapped)


def _normalize_monitor_names(monitors):
    """Normalize list- or comma-separated monitor selections."""
    if monitors is None:
        return ["all"]
    if isinstance(monitors, str):
        monitors = [monitors]

    names = []
    for value in monitors:
        if not isinstance(value, str):
            raise TypeError("monitor names must be strings")
        names.extend(name.strip() for name in value.split(",") if name.strip())
    return list(dict.fromkeys(names))


def setup_monitors(
    model,
    monitors=None,
    monitor_dict=None,
    monitor_interval=1,
    verbose=False,
    exclude_families=None,
    **kwargs,
):
    """Setup all requested monitors on a PaddleFleet model.

    ``exclude_families`` names metric families to switch off, either as a spec
    string (``"mhc_health:mix, moe_health:expert+act"``) or as an already-parsed
    ``{monitor: [family, ...]}`` mapping. Families not named stay on, so omitting
    it reproduces the previous behaviour exactly. An unknown monitor or family
    name raises here, before any monitor is built — see ``core.metric_families``.
    """
    install_gather_fn()

    monitors = _normalize_monitor_names(monitors)
    if "all" in monitors:
        monitors = list(_MONITOR_MAP.keys())
    if monitor_dict is None:
        monitor_dict = {}
    excluded = (
        exclude_families
        if isinstance(exclude_families, dict)
        else parse_exclusions(exclude_families)
    )
    validate_exclusions(excluded, _MONITOR_MAP, backend="paddlefleet")

    model_monitor_dict = getattr(model, _MODEL_MONITOR_ATTR, None)
    if model_monitor_dict is None:
        model_monitor_dict = {}
        setattr(model, _MODEL_MONITOR_ATTR, model_monitor_dict)

    for name in monitors:
        if name not in _MONITOR_MAP:
            logger.warning(
                f"[InternalMedicine/paddlefleet] Unknown monitor: {name}, skipping"
            )
            continue
        options = kwargs.get(name, {})
        if name in excluded and "exclude_families" not in options:
            options = {**options, "exclude_families": excluded[name]}
        expected_config = _monitor_config(monitor_interval, verbose, options)
        existing_monitor = model_monitor_dict.get(name)
        if existing_monitor is not None:
            existing_config = getattr(
                existing_monitor, _MONITOR_CONFIG_ATTR, None
            )
            if (
                _monitor_has_active_hooks(existing_monitor)
                and existing_config == expected_config
            ):
                monitor_dict[name] = existing_monitor
                logger.info(
                    f"[InternalMedicine/paddlefleet] Monitor already enabled: {name}, skipping duplicate setup"
                )
                continue
            existing_monitor.remove_hooks()
            model_monitor_dict.pop(name, None)

        try:
            _MONITOR_MAP[name](
                model,
                monitor_dict=monitor_dict,
                monitor_interval=monitor_interval,
                verbose=verbose,
                **options,
            )
            monitor = monitor_dict.get(name)
            if monitor is not None:
                setattr(monitor, _MONITOR_CONFIG_ATTR, expected_config)
                model_monitor_dict[name] = monitor
            logger.info(
                f"[InternalMedicine/paddlefleet] Enabled monitor: {name}"
            )
        except Exception:
            partial_monitor = monitor_dict.pop(name, None)
            if partial_monitor is not None:
                partial_monitor.remove_hooks()
            model_monitor_dict.pop(name, None)
            logger.exception(
                f"[InternalMedicine/paddlefleet] Failed to setup {name}"
            )
            raise

    return model


__all__ = [
    "setup_monitors",
    "PaddleProbe",
    "PaddleQKStatsMonitor",
    "setup_qk_monitor",
    "PaddleAPEHealthMonitor",
    "setup_ape_monitor",
    "PaddleMoEMonitor",
    "setup_moe_monitor",
    "PaddleMassiveActivationMonitor",
    "setup_massive_activation_monitor",
    "PaddleMHCHealthMonitor",
    "setup_mhc_monitor",
    "PaddleVHAHealthMonitor",
    "setup_vha_monitor",
    "PaddleAttnUpdateMonitor",
    "setup_attn_update_monitor",
    "PaddleMLPUpdateMonitor",
    "setup_mlp_update_monitor",
    "PaddleKDAHealthMonitor",
    "setup_kda_monitor",
    "PaddleDSAHealthMonitor",
    "setup_dsa_monitor",
]
