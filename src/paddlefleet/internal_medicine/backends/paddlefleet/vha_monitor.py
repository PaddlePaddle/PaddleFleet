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

"""
VHA Health Monitor for PaddleFleet.

VHA (Virtual Head Attention) halves the physical query heads and sets ``H_kv = 2``,
then restores expressiveness with two strictly linear operations:

- **Q Premix** — a near-identity transform applied *per KV group*, expanding the
  halved physical query heads into ``H_v = H'_q × H_kv`` virtual heads to recover
  attention diversity. Present only when ``use_vha_premix`` is set.
- **Linear Postmix** — a low-rank identity-residual ``I + A Bᵗ`` (``r = H'_q``) on
  the head axis of the attention output, fusing cross-group inter-head features.
  PaddleFleet names the factors ``vha_postmix_U`` / ``vha_postmix_V``.

Both fold away at inference (Premix into ``q_proj``, Postmix into ``o_proj``),
leaving a plain GQA-2 model. These matrices therefore exist only during training,
and training is the only time their structure can be observed at all.

Neither is visible to the other monitors: ``qk_stats`` samples ``core_attention``
inputs (Postmix happens after that call), and ``massive_act`` only sees the
residual stream, where the Postmix effect arrives diluted by ``o_proj``.

``V`` is zero-initialised, so Postmix starts as the exact identity. The headline
metric is therefore ``postmix_delta_rel_*``: it starts at 0 and its trend says
whether the mixer is learning anything and whether it stays bounded.

Metrics produced (``{branch}`` is ``main`` or ``sparse`` — MQA stacks carry a
separate postmix parameter set for the block-sparse branch)::

    vha_health/layer_{i}/{branch}_postmix_uv_sigma_max
    vha_health/layer_{i}/{branch}_postmix_uv_eff_rank
    vha_health/layer_{i}/{branch}_postmix_offdiag_ratio
    vha_health/layer_{i}/{branch}_postmix_u_fro
    vha_health/layer_{i}/{branch}_postmix_v_fro
    vha_health/layer_{i}/{branch}_postmix_delta_rel_mean
    vha_health/layer_{i}/{branch}_postmix_delta_rel_max
    vha_health/layer_{i}/{branch}_postmix_amax_gain_max
    vha_health/layer_{i}/{branch}_head_out_norm_max
    vha_health/layer_{i}/{branch}_head_out_norm_min
    vha_health/layer_{i}/{branch}_head_out_norm_std
    vha_health/layer_{i}/{branch}_postmix_head_cos_mean
    vha_health/layer_{i}/premix_sigma_max          (only with use_vha_premix)
    vha_health/layer_{i}/premix_identity_dev       (square premix)
    vha_health/layer_{i}/premix_group_div_ratio    (square premix, H_kv > 1)
    vha_health/layer_{i}/premix_orth_dev           (non-square premix)
    vha_health/global_*

On mixed stacks the attention kind is prepended to the metric name exactly like
the other monitors, e.g. ``vha_health/layer_5/hca_main_postmix_delta_rel_max``.
"""

import logging

import paddle
from paddle import nn

from .base import PaddleProbe
from .layer_discovery import (
    get_attention_module,
    get_decoder_layers,
    iter_monitor_layers,
)
from .vha_metrics import (
    head_output_stats,
    postmix_delta_stats,
    postmix_operator_stats,
    premix_metric_names,
    premix_stats,
)

logger = logging.getLogger(__name__)

_BRANCHES = ("main", "sparse")

_POSTMIX_METRICS = (
    "postmix_uv_sigma_max",
    "postmix_uv_eff_rank",
    "postmix_offdiag_ratio",
    "postmix_u_fro",
    "postmix_v_fro",
    "postmix_delta_rel_mean",
    "postmix_delta_rel_max",
    "postmix_amax_gain_max",
    "head_out_norm_max",
    "head_out_norm_min",
    "head_out_norm_std",
    "postmix_head_cos_mean",
)

# Premix keys depend on the weight shape (square vs not, H_kv > 1), so the exact
# set comes from premix_metric_names() at registration time.
_PREMIX_METRICS = (
    "premix_sigma_max",
    "premix_identity_dev",
    "premix_group_div_ratio",
    "premix_orth_dev",
)

# Extrema keep extremum semantics across microbatches / layers / ranks; the rest
# are means. Every name below ends in `_max` / `_min`, so training_logs' suffix
# classifier reaches the same verdict on the full key — including the attn_type
# and branch prefixes it carries.
_MAX_METRICS = frozenset(
    {
        "postmix_uv_sigma_max",
        "postmix_delta_rel_max",
        "postmix_amax_gain_max",
        "head_out_norm_max",
        "premix_sigma_max",
    }
)
_MIN_METRICS = frozenset({"head_out_norm_min"})


def _has_postmix(attn) -> bool:
    return (
        getattr(attn, "use_vha_postmix", False)
        and getattr(attn, "vha_postmix_U", None) is not None
    )


def _premix_weight(attn):
    """Return the premix weight when premix is active, else ``None``."""
    if not getattr(attn, "use_vha_premix", False):
        return None
    return getattr(attn, "vha_premix_weight", None)


def _num_heads(factor: paddle.Tensor) -> int:
    """Head count implied by a postmix factor (``[nh, r]`` or ``[g, j, r]``)."""
    if factor.ndim == 2:
        return int(factor.shape[0])
    return int(factor.shape[0]) * int(factor.shape[1])


class PaddleVHAHealthMonitor(PaddleProbe):
    """Monitor the VHA premix / postmix structures of attention layers."""

    METRIC_PREFIX = "vha_health"
    MAX_AGGREGATED: set[str] = set()
    MIN_AGGREGATED: set[str] = set()

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
        # Branch-prefixed names, because record_layer_metric classifies on the
        # name it is handed.
        self.MAX_AGGREGATED = {
            f"{b}_{m}"
            for b in _BRANCHES
            for m in _MAX_METRICS
            if m in _POSTMIX_METRICS
        } | {m for m in _MAX_METRICS if m in _PREMIX_METRICS}
        self.MIN_AGGREGATED = {
            f"{b}_{m}" for b in _BRANCHES for m in _MIN_METRICS
        }
        # (module, original__apply_vha_postmix) pairs, for remove_hooks().
        self._wrapped: list[tuple[nn.Layer, object]] = []
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

        def matches(layer):
            attn = get_attention_module(layer)
            return attn is not None and _has_postmix(attn)

        monitor_layers = iter_monitor_layers(
            layers, matches, pp_rank=self.pp_rank
        )
        mtp_layer_ids = [item.idx for item in monitor_layers if item.is_mtp]
        if mtp_layer_ids:
            self.mark_mtp_layers(mtp_layer_ids)

        targets = []
        for item in monitor_layers:
            if self.sample_layers and item.idx not in self.sample_layers:
                continue
            attn = get_attention_module(item.layer)
            branches = ["main"]
            if getattr(attn, "sparse_vha_postmix_U", None) is not None:
                branches.append("sparse")
            targets.append((item.idx, attn, item.attn_type, branches))
        return targets

    def register_hooks(self, model: nn.Layer):
        self._init_parallel_state()
        targets = self._find_targets(model)
        if not targets:
            logger.info(
                "[PaddleVHAMonitor] No VHA postmix layers found; skipping."
            )
            return

        for layer_idx, attn, attn_type, branches in targets:
            for branch in branches:
                for name in _POSTMIX_METRICS:
                    self.declare_layer_metric(
                        layer_idx, f"{branch}_{name}", attn_type=attn_type
                    )
            premix = _premix_weight(attn)
            if premix is not None:
                for name in premix_metric_names(premix):
                    self.declare_layer_metric(
                        layer_idx, name, attn_type=attn_type
                    )

        self.allocate_buffers()

        for layer_idx, attn, attn_type, _branches in targets:
            orig = attn._apply_vha_postmix
            attn._apply_vha_postmix = self._make_capture(
                orig, attn, layer_idx, attn_type
            )
            self._wrapped.append((attn, orig))

        logger.info(
            f"[PaddleVHAMonitor] Wrapped {len(self._wrapped)} attention modules."
        )

    def remove_hooks(self):
        for attn, orig in self._wrapped:
            try:
                del attn._apply_vha_postmix
            except AttributeError:
                attn._apply_vha_postmix = orig
        self._wrapped = []
        super().remove_hooks()

    # ------------------------------------------------------------------
    # Capture wrapper (the hot path)
    # ------------------------------------------------------------------

    def _make_capture(self, orig, attn, layer_idx: int, attn_type: str | None):
        """Wrap ``_apply_vha_postmix`` and derive metrics from its real I/O.

        ``delta`` is ``out − attn_out``, so nothing is recomputed. Everything runs
        detached under ``no_grad`` and writes 0-dim GPU tensors, so no D2H sync
        and no autograd graph is pinned.

        Selective recompute (``recompute_modules=[..., "vha_postmix"]``) can replay
        this call during backward. The replay produces bit-identical tensors, so a
        second record leaves mean (sum and count both double), max, and min
        unchanged — no dedupe needed.
        """

        def wrapped(attn_out, *args, **kwargs):
            out = orig(attn_out, *args, **kwargs)
            if not self._should_monitor():
                return out
            try:
                u = kwargs.get("U", args[0] if len(args) >= 1 else None)
                v = kwargs.get("V", args[1] if len(args) >= 2 else None)
                if u is None or v is None:
                    branch = "main"
                    u, v = attn.vha_postmix_U, attn.vha_postmix_V
                else:
                    branch = (
                        "sparse"
                        if u is getattr(attn, "sparse_vha_postmix_U", None)
                        else "main"
                    )

                with paddle.no_grad():
                    stats = postmix_operator_stats(u, v)
                    stats.update(postmix_delta_stats(attn_out, out))
                    stats.update(head_output_stats(out, _num_heads(u)))
                    for name, value in stats.items():
                        self.record_layer_metric(
                            layer_idx,
                            f"{branch}_{name}",
                            value,
                            attn_type=attn_type,
                        )

                    premix = _premix_weight(attn)
                    if premix is not None and branch == "main":
                        for name, value in premix_stats(premix).items():
                            self.record_layer_metric(
                                layer_idx, name, value, attn_type=attn_type
                            )
            except Exception as e:
                if self.verbose and layer_idx not in self._failed_layers:
                    logger.error(
                        f"[PaddleVHAMonitor] Error at layer {layer_idx}: {e}"
                    )
                    self._failed_layers.add(layer_idx)
            return out

        return wrapped


def setup_vha_monitor(
    model,
    log_per_layer: bool = True,
    log_global: bool = True,
    monitor_interval: int = 1,
    verbose: bool = False,
    sample_layers: list[int] | None = None,
    monitor_dict: dict | None = None,
    exclude_families=None,
):
    monitor = PaddleVHAHealthMonitor(
        exclude_families=exclude_families,
        log_per_layer=log_per_layer,
        log_global=log_global,
        monitor_interval=monitor_interval,
        verbose=verbose,
        sample_layers=sample_layers,
    )
    monitor.register_hooks(model)
    if monitor_dict is not None:
        monitor_dict["vha_health"] = monitor
    return model
