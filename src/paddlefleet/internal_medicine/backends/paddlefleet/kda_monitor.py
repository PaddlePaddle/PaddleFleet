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

"""KDA (Kimi Delta Attention) health monitor for PaddleFleet.

Watches the three pathways that make KDA's fixed-capacity associative memory
work - decay (``alpha``), write (``beta``), read (the output gate) - plus one
attribution parameter. Eight metrics per layer; see ``docs/kda_health.md``.

Collection is plain ``forward_post_hook``s plus one parameter read. None of
``alpha`` / ``beta`` / the output gate is returned by any submodule
(``KimiDeltaAttention.forward`` keeps them as locals), but all three are
recoverable from the projections that produce them:

- ``f_b_proj`` output is the raw decay logit ``z``
- ``in_proj`` output packs ``[qkv | beta | gate]``; the last two are what we need
  (``gate`` only when ``use_full_rank_gate``, otherwise it comes from ``g_b_proj``)

Nothing here wraps a bound method, so there is no recompute-replay dedupe and no
``RecomputeWithoutOutput`` lifetime constraint to reason about. Hot-path
discipline: see ``.claude/skills/monitor-hook-perf-rules``.
"""

from __future__ import annotations

import logging

import paddle
from paddle import nn

from . import kda_metrics
from .base import PaddleProbe
from .layer_discovery import (
    get_attention_module,
    get_decoder_layers,
    is_kda_layer,
    iter_monitor_layers,
)

logger = logging.getLogger(__name__)


def _unwrap(output):
    """PaddleFleet linear layers return ``(out, bias)``; take the tensor."""
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


class PaddleKDAHealthMonitor(PaddleProbe):
    """Monitor the decay / write / read pathways of KDA layers."""

    METRIC_PREFIX = "kda_health"
    MAX_AGGREGATED: set[str] = set()
    MIN_AGGREGATED: set[str] = set(kda_metrics.MIN_METRICS)

    def __init__(
        self,
        log_per_layer: bool = True,
        log_global: bool = True,
        monitor_interval: int = 1,
        verbose: bool = False,
        sample_layers: list[int] | None = None,
    ):
        super().__init__(
            log_per_layer=log_per_layer,
            log_global=log_global,
            monitor_interval=monitor_interval,
            verbose=verbose,
        )
        self.sample_layers = set(sample_layers) if sample_layers else None
        self._failed_layers: set[int] = set()

    # ------------------------------------------------------------------
    # Setup: discover -> declare -> allocate -> attach
    # ------------------------------------------------------------------

    def _init_parallel_state(self):
        try:
            from paddlefleet.parallel_state import (
                get_pipeline_model_parallel_rank,
            )

            self.pp_rank = get_pipeline_model_parallel_rank()
        except Exception:
            pass

    def _find_targets(self, model):
        layers = get_decoder_layers(model)
        if not layers:
            return []

        monitor_layers = iter_monitor_layers(
            layers, is_kda_layer, pp_rank=self.pp_rank
        )
        mtp_layer_ids = [item.idx for item in monitor_layers if item.is_mtp]
        if mtp_layer_ids:
            self.mark_mtp_layers(mtp_layer_ids)

        targets = []
        for item in monitor_layers:
            if self.sample_layers and item.idx not in self.sample_layers:
                continue
            targets.append(
                (item.idx, get_attention_module(item.layer), item.attn_type)
            )
        return targets

    def register_hooks(self, model: nn.Layer):
        self._init_parallel_state()
        targets = self._find_targets(model)
        if not targets:
            logger.info("[PaddleKDAMonitor] No KDA layers found; skipping.")
            return

        for layer_idx, _attn, attn_type in targets:
            for name in kda_metrics.ALL_METRICS:
                self.declare_layer_metric(layer_idx, name, attn_type=attn_type)

        self.allocate_buffers()

        for layer_idx, attn, attn_type in targets:
            self.hooks.append(
                attn.f_b_proj.register_forward_post_hook(
                    self._make_decay_hook(layer_idx, attn, attn_type)
                )
            )
            self.hooks.append(
                attn.in_proj.register_forward_post_hook(
                    self._make_in_proj_hook(layer_idx, attn, attn_type)
                )
            )
            if not attn.use_full_rank_gate:
                # Low-rank output gate: it never passes through in_proj.
                self.hooks.append(
                    attn.g_b_proj.register_forward_post_hook(
                        self._make_gate_hook(layer_idx, attn, attn_type)
                    )
                )

        logger.info(
            f"[PaddleKDAMonitor] Registered {len(self.hooks)} hooks on {len(targets)} KDA layers."
        )

    # ------------------------------------------------------------------
    # Hooks (the hot path)
    # ------------------------------------------------------------------

    def _record(self, layer_idx: int, attn_type, stats: dict) -> None:
        for name, value in stats.items():
            self.record_layer_metric(
                layer_idx, name, value, attn_type=attn_type
            )

    def _log_failure(self, layer_idx: int, exc: Exception) -> None:
        if self.verbose and layer_idx not in self._failed_layers:
            logger.error(
                f"[PaddleKDAMonitor] Error at layer {layer_idx}: {exc}"
            )
            self._failed_layers.add(layer_idx)

    def _make_decay_hook(self, layer_idx: int, attn, attn_type):
        """``f_b_proj`` output is the raw decay logit ``z``; derive ``g`` from it.

        ``A_log`` is recorded here too: it is a parameter read, so it needs no hook
        of its own, and piggybacking keeps it on exactly the same cadence as the
        decay metrics it is meant to explain.
        """

        def hook(_layer, _input, output):
            if not self._should_monitor():
                return None
            try:
                z = _unwrap(output).detach()
                with paddle.no_grad():
                    g = kda_metrics.kda_log_decay(
                        z,
                        attn.A_log,
                        attn.dt_bias,
                        # Read the gate form off the layer, never hardcode -5.0:
                        # gate_lower_bound=None selects the unbounded softplus form.
                        safe_gate=attn.gate_lower_bound is not None,
                        lower_bound=attn.gate_lower_bound,
                        num_heads=attn.num_v_heads_local_tp,
                        head_dim=attn.value_head_dim,
                    )
                    self._record(
                        layer_idx, attn_type, kda_metrics.decay_gate_stats(g)
                    )
                    self._record(
                        layer_idx,
                        attn_type,
                        kda_metrics.param_stats(attn.A_log),
                    )
            except Exception as exc:
                self._log_failure(layer_idx, exc)
            return None

        return hook

    def _make_in_proj_hook(self, layer_idx: int, attn, attn_type):
        """Slice ``beta`` (and the full-rank output gate) out of ``in_proj``.

        Channel layout is ``[qkv | beta | gate]``, matching the ``split_sizes``
        that ``KimiDeltaAttention.forward`` builds. Widths are read off the layer
        so a TP shard slices its own local widths.
        """

        def hook(_layer, _input, output):
            if not self._should_monitor():
                return None
            try:
                qkvbz = _unwrap(output).detach()
                beta_start = attn.conv_dim_local_tp
                beta_end = beta_start + attn.num_v_heads_local_tp
                with paddle.no_grad():
                    beta = qkvbz[..., beta_start:beta_end]
                    self._record(
                        layer_idx,
                        attn_type,
                        kda_metrics.write_gate_stats(
                            beta, attn.num_v_heads_local_tp
                        ),
                    )
                    if attn.use_full_rank_gate:
                        gate = qkvbz[
                            ..., beta_end : beta_end + attn.v_dim_local_tp
                        ]
                        self._record(
                            layer_idx,
                            attn_type,
                            kda_metrics.read_gate_stats(
                                gate, attn.v_dim_local_tp
                            ),
                        )
            except Exception as exc:
                self._log_failure(layer_idx, exc)
            return None

        return hook

    def _make_gate_hook(self, layer_idx: int, attn, attn_type):
        """Low-rank output gate: ``g_b_proj`` output is the gate logit."""

        def hook(_layer, _input, output):
            if not self._should_monitor():
                return None
            try:
                with paddle.no_grad():
                    gate = _unwrap(output).detach()
                    self._record(
                        layer_idx,
                        attn_type,
                        kda_metrics.read_gate_stats(gate, attn.v_dim_local_tp),
                    )
            except Exception as exc:
                self._log_failure(layer_idx, exc)
            return None

        return hook


def setup_kda_monitor(
    model,
    log_per_layer: bool = True,
    log_global: bool = True,
    monitor_interval: int = 1,
    verbose: bool = False,
    sample_layers: list[int] | None = None,
    monitor_dict: dict | None = None,
):
    monitor = PaddleKDAHealthMonitor(
        log_per_layer=log_per_layer,
        log_global=log_global,
        monitor_interval=monitor_interval,
        verbose=verbose,
        sample_layers=sample_layers,
    )
    monitor.register_hooks(model)
    if monitor_dict is not None:
        monitor_dict["kda_health"] = monitor
    return model
