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

"""APE health monitor for PaddleFleet compressor parameters."""

import logging

import paddle
from paddle import nn

from .ape_metrics import compute_ape_p0_metrics
from .base import PaddleProbe
from .layer_discovery import (
    get_attention_module,
    get_decoder_layers,
    iter_monitor_layers,
)

logger = logging.getLogger(__name__)

_APE_BRANCHES = ("core", "indexer")
_P0_METRICS = (
    "non_finite_ratio",
    "rms",
    "centered_rms",
    "position_range_p95",
    "softmax_entropy_norm_mean",
    "softmax_max_prob_p95",
    "effective_positions_mean",
)


class PaddleAPEHealthMonitor(PaddleProbe):
    """Monitor APE parameters without materializing them on CPU in hooks."""

    METRIC_PREFIX = "ape_health"
    MAX_AGGREGATED = {
        f"{branch}_{metric}"
        for branch in _APE_BRANCHES
        for metric in (
            "non_finite_ratio",
            "position_range_p95",
            "softmax_max_prob_p95",
        )
    }
    MIN_AGGREGATED = {
        f"{branch}_{metric}"
        for branch in _APE_BRANCHES
        for metric in ("softmax_entropy_norm_mean", "effective_positions_mean")
    }

    def __init__(
        self,
        log_per_layer: bool = True,
        log_global: bool = True,
        monitor_interval: int = 1,
        verbose: bool = False,
        sample_layers: list[int] | None = None,
        exclude_families=None,
    ):
        super().__init__(
            exclude_families=exclude_families,
            log_per_layer=log_per_layer,
            log_global=log_global,
            monitor_interval=monitor_interval,
            verbose=verbose,
        )
        self.sample_layers = set(sample_layers) if sample_layers else None
        self._failed_layers: set[tuple[int, str]] = set()

    @staticmethod
    def _compressors(attention) -> list[tuple[str, nn.Layer]]:
        """Return distinct core/indexer compressors that own an APE tensor."""
        core_attention = getattr(attention, "core_attention", None)
        if core_attention is None:
            return []
        candidates = [
            ("core", getattr(core_attention, "compressor", None)),
            (
                "indexer",
                getattr(
                    getattr(core_attention, "indexer", None), "compressor", None
                ),
            ),
        ]
        result = []
        seen = set()
        for branch, compressor in candidates:
            ape = getattr(compressor, "ape", None)
            if (
                compressor is None
                or not isinstance(ape, paddle.Tensor)
                or id(compressor) in seen
            ):
                continue
            seen.add(id(compressor))
            result.append((branch, compressor))
        return result

    def _find_targets(self, model):
        layers = get_decoder_layers(model)
        if not layers:
            return []

        def matches(layer):
            attention = get_attention_module(layer)
            return attention is not None and bool(self._compressors(attention))

        monitor_layers = iter_monitor_layers(
            layers, matches, pp_rank=self.pp_rank
        )
        self.mark_mtp_layers(item.idx for item in monitor_layers if item.is_mtp)
        targets = []
        for item in monitor_layers:
            if self.sample_layers and item.idx not in self.sample_layers:
                continue
            attention = get_attention_module(item.layer)
            targets.extend(
                (item.idx, branch, compressor)
                for branch, compressor in self._compressors(attention)
            )
        return targets

    def register_hooks(self, model: nn.Layer):
        targets = self._find_targets(model)
        if not targets:
            logger.info(
                "[PaddleAPEHealthMonitor] No APE compressors found; skipping."
            )
            return

        for layer_idx, branch, _compressor in targets:
            for metric_name in _P0_METRICS:
                self.declare_layer_metric(layer_idx, f"{branch}_{metric_name}")
        self.allocate_buffers()

        for layer_idx, branch, compressor in targets:
            hook = compressor.register_forward_pre_hook(
                self._make_ape_hook(layer_idx, branch)
            )
            self.hooks.append(hook)
        logger.info(
            "[PaddleAPEHealthMonitor] Registered %d APE hooks.", len(targets)
        )

    def _make_ape_hook(self, layer_idx: int, branch: str):
        def hook_fn(module, _inputs):
            if not module.training or not self._should_monitor():
                return
            try:
                with paddle.no_grad():
                    metrics = compute_ape_p0_metrics(module.ape)
                    for name, value in metrics.items():
                        self.record_layer_metric(
                            layer_idx, f"{branch}_{name}", value
                        )
            except Exception as exc:
                key = (layer_idx, branch)
                if self.verbose and key not in self._failed_layers:
                    logger.exception(
                        "[PaddleAPEHealthMonitor] Failed at layer %s (%s): %s",
                        layer_idx,
                        branch,
                        exc,
                    )
                    self._failed_layers.add(key)

        return hook_fn


def setup_ape_monitor(
    model,
    log_per_layer: bool = True,
    log_global: bool = True,
    monitor_interval: int = 1,
    verbose: bool = False,
    sample_layers: list[int] | None = None,
    monitor_dict: dict | None = None,
    exclude_families=None,
):
    monitor = PaddleAPEHealthMonitor(
        exclude_families=exclude_families,
        log_per_layer=log_per_layer,
        log_global=log_global,
        monitor_interval=monitor_interval,
        verbose=verbose,
        sample_layers=sample_layers,
    )
    monitor.register_hooks(model)
    if monitor_dict is not None:
        monitor_dict["ape_health"] = monitor
    return model
