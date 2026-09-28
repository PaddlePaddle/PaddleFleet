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

"""mHC Health Monitor for Megatron-Bridge.

Monitors the three per-token mappings of every ``HyperConnectionModule`` in an
mHC (Manifold-Constrained Hyper-Connections) model:

- ``h_pre`` / ``h_post`` — the aggregation / expansion gates: mean and std.
- ``h_res``              — the doubly-stochastic residual-mixing matrix: the
  paper's ``amax_gain`` forward (max-abs row sum) and backward (max-abs column
  sum), computed both on this layer's ``h_res`` and on the running **composite
  mapping** (cumulative product of ``h_res`` across the layers local to this
  pipeline stage / VPP chunk).

Per hyper-connection module (a layer has two: ``attn`` and ``mlp``) we emit 8
mean-aggregated series, name-prefixed by component:

    {attn,mlp}_h_pre_mean   {attn,mlp}_h_pre_std
    {attn,mlp}_h_post_mean  {attn,mlp}_h_post_std
    {attn,mlp}_amax_gain_fwd            {attn,mlp}_amax_gain_bwd
    {attn,mlp}_composite_amax_gain_fwd  {attn,mlp}_composite_amax_gain_bwd

``h_pre`` is not part of ``HyperConnectionModule.forward``'s return, so we cannot
use a forward hook. Instead we wrap the module's ``compute_mappings`` bound method
to capture its real ``(h_pre, h_post, h_res)`` — no recompute. Everything is
detached and computed under ``no_grad`` (see the VRAM-safety notes on the hook).

The monitor is a hard no-op unless the model actually uses the mHC layer: if the
mHC classes cannot be imported, or no ``HyperConnectionTransformerLayer`` is
found, ``setup_mhc_monitor`` attaches nothing and registers no metrics.

Hot-path discipline (no D2H sync, no hook-time collectives, schema fixed at
registration): see ``.claude/skills/monitor-hook-perf-rules``.
"""

import logging

import torch
from torch import nn

from .base import TorchProbe
from .mhc_metrics import amax_gain, gate_stats

logger = logging.getLogger(__name__)


# mHC classes are optional: this monitor must be a no-op when they are absent
# (non-mHC model, or a Megatron build without hyper-connections). Bind to None
# on import failure and gate every code path on that.
try:
    from megatron.core.transformer.hyper_connection import HyperConnectionModule
    from megatron.core.transformer.transformer_layer import (
        HyperConnectionTransformerLayer,
    )
except Exception:  # pragma: no cover - environment dependent
    HyperConnectionModule = None
    HyperConnectionTransformerLayer = None


_METRIC_NAMES = (
    "h_pre_mean",
    "h_pre_std",
    "h_post_mean",
    "h_post_std",
    "amax_gain_fwd",
    "amax_gain_bwd",
    "composite_amax_gain_fwd",
    "composite_amax_gain_bwd",
)

# (component_name, layer attribute) — attn runs before mlp in the layer forward.
_COMPONENTS = (
    ("attn", "self_attention_hyper_connection"),
    ("mlp", "mlp_hyper_connection"),
)


class MHCHealthMonitor(TorchProbe):
    """Monitor the h_pre / h_post / h_res mappings of mHC hyper-connection layers."""

    METRIC_PREFIX = "mhc_health"
    # Every metric is a mean over tokens/batch (and over microbatches/ranks at
    # flush). No max/min series, so both classification sets stay empty. The
    # metric names end in _mean/_std/_fwd/_bwd — never a `_max`/`_min` boundary —
    # so training_logs' suffix classifier also keeps them as "mean".
    MAX_AGGREGATED: set[str] = set()
    MIN_AGGREGATED: set[str] = set()

    def __init__(
        self,
        log_per_layer: bool = True,
        log_global: bool = True,
        monitor_interval: int = 1,
        verbose: bool = False,
        hook_timing_enabled: bool = False,
        exclude_families=None,
    ):
        super().__init__(
            exclude_families=exclude_families,
            log_per_layer=log_per_layer,
            log_global=log_global,
            monitor_interval=monitor_interval,
            verbose=verbose,
            hook_timing_enabled=hook_timing_enabled,
        )
        # chunk_id -> running composite mapping [s*b, n, n], detached (no graph).
        self._composite: dict[int, torch.Tensor] = {}
        # (module, original_compute_mappings) pairs, for remove_hooks() restoration.
        self._wrapped: list[tuple[nn.Module, object]] = []

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _find_layer_stack(self, model: nn.Module):
        if hasattr(model, "module"):
            model = model.module
        if hasattr(model, "decoder") and hasattr(model.decoder, "layers"):
            return model.decoder.layers
        if hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
            return model.encoder.layers
        if hasattr(model, "layers"):
            return model.layers
        if hasattr(model, "language_model"):
            lm = model.language_model
            if hasattr(lm, "decoder") and hasattr(lm.decoder, "layers"):
                return lm.decoder.layers
        return None

    def _find_hc_modules(self, model: nn.Module, layer_offset: int = 0):
        """Return ``[(global_idx, component, hc_module)]`` for every mHC hc module.

        Empty (auto-skip) when the mHC classes are unavailable or the model has no
        ``HyperConnectionTransformerLayer``. Uses ``isinstance`` against the real
        classes rather than duck-typing, so a plain ``TransformerLayer`` (or an
        ``IdentityOp`` placeholder) is never matched.
        """
        if (
            HyperConnectionTransformerLayer is None
            or HyperConnectionModule is None
        ):
            return []
        layers = self._find_layer_stack(model)
        if layers is None:
            return []

        entries = []
        num_local = len(layers)
        for local_idx, layer in enumerate(layers):
            if not isinstance(layer, HyperConnectionTransformerLayer):
                continue
            global_idx = self._resolve_layer_idx(
                layer, local_idx, num_local, layer_offset
            )
            for comp, attr in _COMPONENTS:
                mod = getattr(layer, attr, None)
                if isinstance(mod, HyperConnectionModule):
                    entries.append((global_idx, comp, mod))
        return entries

    # ------------------------------------------------------------------
    # Three-phase setup: prepare (declare) -> allocate -> attach
    # ------------------------------------------------------------------

    def _prepare_layers(
        self, model: nn.Module, chunk_id: int, layer_offset: int = 0
    ):
        entries = self._find_hc_modules(model, layer_offset=layer_offset)
        if not entries:
            return []
        for global_idx, comp, _ in entries:
            for name in _METRIC_NAMES:
                self.declare_layer_metric(global_idx, f"{comp}_{name}")
        # The chunk root = the attn hc of the lowest-index layer; it is the first
        # hc module executed on this stage, so it resets the composite each forward.
        root_idx = min(gi for gi, comp, _ in entries if comp == "attn")
        return [
            (gi, comp, mod, chunk_id, comp == "attn" and gi == root_idx)
            for gi, comp, mod in entries
        ]

    def _attach_hooks(self, targets):
        for layer_idx, comp, mod, chunk_id, is_root in targets:
            orig = mod.compute_mappings
            mod.compute_mappings = self._make_capture(
                orig, layer_idx, comp, chunk_id, is_root
            )
            self._wrapped.append((mod, orig))
        logger.info(
            f"[MHCMonitor] Wrapped {len(self._wrapped)} hyper-connection modules."
        )

    def register_hooks(self, model: nn.Module):
        """Single-chunk convenience path. Prefer ``setup_mhc_monitor`` for VPP."""
        self._init_parallel_state()
        targets = self._prepare_layers(model, chunk_id=0)
        if not targets:
            return
        self.allocate_buffers(next(model.parameters()).device)
        self._attach_hooks(targets)

    def _init_parallel_state(self):
        try:
            from megatron.core import parallel_state

            if parallel_state.model_parallel_is_initialized():
                self.pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        except ImportError:
            pass

    def remove_hooks(self):
        # Restore the original bound methods and drop all cross-call state so the
        # monitor holds no module references or composite tensors after teardown.
        for mod, orig in self._wrapped:
            try:
                del mod.compute_mappings  # fall back to the class method
            except AttributeError:
                mod.compute_mappings = orig
        self._wrapped = []
        self._composite.clear()
        super().remove_hooks()

    def step(self):
        super().step()
        # Release the running composite between train steps so a [s*b, n, n]
        # buffer never sits idle in VRAM; the root reseeds it next forward, so
        # clearing is correctness-neutral.
        self._composite.clear()

    # ------------------------------------------------------------------
    # Capture wrapper (the hot path)
    # ------------------------------------------------------------------

    def _make_capture(
        self, orig, layer_idx: int, component: str, chunk_id: int, is_root: bool
    ):
        """Wrap ``compute_mappings`` to record metrics from its real return value.

        VRAM safety: the mappings arrive attached to the training autograd graph;
        we ``.detach()`` them and do all metric/composite math under ``no_grad`` so
        no stored tensor pins the graph through backward. The composite slot holds
        one detached ``[s*b, n, n]`` tensor per chunk (cloned on seed so it never
        aliases the model's ``h_res`` storage); ``step()`` clears it.
        """

        def wrapped(x):
            out = orig(
                x
            )  # the real mappings the model consumes — returned unchanged
            # Gate the whole capture: metrics are only recorded on monitored steps,
            # and the composite is reset at the chunk root on each such step, so
            # gating here keeps the composite self-consistent (root reset -> ordered
            # bmm builds within one monitored forward). _should_monitor() also
            # requires grad enabled, selecting the training (not a no-grad) forward.
            if not self._should_monitor():
                return out
            try:
                h_pre, h_post, h_res = out
                with torch.no_grad():
                    h_pre = h_pre.detach()
                    h_post = h_post.detach()
                    h_res = h_res.detach()

                    pre_mean, pre_std = gate_stats(h_pre)
                    post_mean, post_std = gate_stats(h_post)
                    self.record_layer_metric(
                        layer_idx, f"{component}_h_pre_mean", pre_mean
                    )
                    self.record_layer_metric(
                        layer_idx, f"{component}_h_pre_std", pre_std
                    )
                    self.record_layer_metric(
                        layer_idx, f"{component}_h_post_mean", post_mean
                    )
                    self.record_layer_metric(
                        layer_idx, f"{component}_h_post_std", post_std
                    )

                    self.record_layer_metric(
                        layer_idx,
                        f"{component}_amax_gain_fwd",
                        amax_gain(h_res, dim=-1),
                    )
                    self.record_layer_metric(
                        layer_idx,
                        f"{component}_amax_gain_bwd",
                        amax_gain(h_res, dim=-2),
                    )

                    # Composite mapping M_k = h_res_k @ M_{k-1} (per token).
                    s, b, n, _ = h_res.shape
                    hb = h_res.reshape(s * b, n, n)
                    prev = self._composite.get(chunk_id)
                    # Reset at the chunk root each forward; also self-heals on first
                    # fire / shape drift (variable s*b across microbatches). identity @
                    # h_res == h_res, so seed with h_res; clone() so the slot owns a
                    # fresh buffer and never pins the model's h_res storage.
                    if is_root or prev is None or prev.shape != hb.shape:
                        M = hb.clone()
                    else:
                        M = torch.bmm(hb, prev)  # fresh tensor, no graph
                    self._composite[chunk_id] = M

                    self.record_layer_metric(
                        layer_idx,
                        f"{component}_composite_amax_gain_fwd",
                        amax_gain(M, dim=-1),
                    )
                    self.record_layer_metric(
                        layer_idx,
                        f"{component}_composite_amax_gain_bwd",
                        amax_gain(M, dim=-2),
                    )
            except Exception as e:
                if self.verbose:
                    logger.error(
                        f"[MHCMonitor] Error layer {layer_idx}/{component}: {e}"
                    )
            return out

        return wrapped


def setup_mhc_monitor(
    model,
    log_per_layer: bool = True,
    log_global: bool = True,
    monitor_interval: int = 1,
    verbose: bool = False,
    hook_timing_enabled: bool = False,
    monitor_dict: dict | None = None,
    exclude_families=None,
):
    """Enable the mHC health monitor. No-op on any non-mHC model.

    Multi-chunk (VPP / interleaved 1F1B) safe: declares the schema across all
    chunks before ``allocate_buffers`` locks it, then attaches. Each model chunk
    gets its own composite slot (keyed by its enumerate index) so a later chunk's
    layers never contaminate an earlier chunk's running product.
    """
    # No-op guarantee #1: mHC classes unavailable -> touch nothing.
    if HyperConnectionTransformerLayer is None or HyperConnectionModule is None:
        logger.info(
            "[MHCMonitor] Hyper-connection classes unavailable; skipping."
        )
        return model

    monitor = MHCHealthMonitor(
        exclude_families=exclude_families,
        log_per_layer=log_per_layer,
        log_global=log_global,
        monitor_interval=monitor_interval,
        verbose=verbose,
        hook_timing_enabled=hook_timing_enabled,
    )
    models = [model] if not isinstance(model, list) else model
    monitor._init_parallel_state()

    chunk_targets = []
    layer_offset = 0
    for chunk_id, m in enumerate(models):
        targets = monitor._prepare_layers(
            m, chunk_id=chunk_id, layer_offset=layer_offset
        )
        chunk_targets.append(targets)
        stack = monitor._find_layer_stack(m)
        layer_offset += len(stack) if stack is not None else 0

    if any(chunk_targets):
        device = next((p.device for m in models for p in m.parameters()), None)
        assert device is not None, (
            "no parameters across model chunks; cannot pick a device"
        )
        monitor.allocate_buffers(device)
        for targets in chunk_targets:
            monitor._attach_hooks(targets)
        if monitor_dict is not None:
            monitor_dict["mhc_health"] = monitor
    else:
        # No-op guarantee #2: no HyperConnectionTransformerLayer found.
        logger.info("[MHCMonitor] No hyper-connection layers found; skipping.")

    logger.info(
        f"[MHCMonitor] Setup complete. Monitoring {len(monitor._wrapped)} hc modules."
    )
    return model
