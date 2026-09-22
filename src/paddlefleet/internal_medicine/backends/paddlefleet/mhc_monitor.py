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

"""mHC Health Monitor for PaddleFleet.

Per hyper-connection module (a layer has two: ``attn`` and ``mlp``) we emit 29
series, name-prefixed by component:

    {attn,mlp}_h_pre_mean   {attn,mlp}_h_pre_std
    {attn,mlp}_h_post_mean  {attn,mlp}_h_post_std
    {attn,mlp}_h_post_stream_concentration  {attn,mlp}_h_post_token_std
    {attn,mlp}_branch_residual_share  {attn,mlp}_branch_residual_share_max
    {attn,mlp}_amax_gain_fwd  {attn,mlp}_amax_gain_bwd
    {attn,mlp}_h_res_logits_min  {attn,mlp}_h_res_logits_max
    {attn,mlp}_h_res_logits_grad_min  {attn,mlp}_h_res_logits_grad_max
    {attn,mlp}_composite_amax_gain_fwd_max  {attn,mlp}_composite_amax_gain_bwd_max
    {attn,mlp}_h_pre_logits_min   {attn,mlp}_h_pre_logits_max
    {attn,mlp}_h_post_logits_min  {attn,mlp}_h_post_logits_max
    {attn,mlp}_alpha_pre  {attn,mlp}_alpha_post  {attn,mlp}_alpha_res
    {attn,mlp}_bias_pre_mean   {attn,mlp}_bias_pre_abs_max
    {attn,mlp}_bias_post_mean  {attn,mlp}_bias_post_abs_max
    {attn,mlp}_bias_res_mean   {attn,mlp}_bias_res_abs_max

The last 13 cover paper Eq. (7) — the ``alpha`` / ``bias`` / pre-sigmoid-logit
terms that feed Eq. (8) — and were added for the algorithm side. The nine
parameter series are step-level and are read once per flush, off the hot path.

Forward hooks are not enough, so three bound methods are wrapped instead (see
``mhc_metrics`` for what each metric means):

- ``compute_mappings`` — ``h_pre`` is not in ``forward``'s return value.
- ``_compute_h``       — the mixing logits only exist before the Sinkhorn
  projection, and wrapping here also sidesteps the ``use_fused_mhc`` fork.
- ``fused_h_res_h_post_bda`` — the only point where both update terms are
  visible.
"""

import logging

import paddle
from paddle import nn

from .base import PaddleProbe
from .layer_discovery import get_decoder_layers, iter_monitor_layers
from .mhc_metrics import (
    amax_gain,
    branch_residual_share,
    gate_logits_extrema,
    gate_stats,
    h_post_structure_stats,
    h_res_logits_extrema,
    mapping_param_stats,
)

logger = logging.getLogger(__name__)


# mHC classes are optional: this monitor must be a no-op when they are absent
# (non-mHC model, or a PaddleFleet build without hyper-connections). Bind to
# None on import failure and gate every code path on that.
try:
    from paddlefleet.transformer.hyper_connection import HyperConnectionModule
    from paddlefleet.transformer.transformer_layer import (
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
    "h_post_stream_concentration",
    "h_post_token_std",
    "branch_residual_share",
    "branch_residual_share_max",
    "amax_gain_fwd",
    "amax_gain_bwd",
    "h_res_logits_min",
    "h_res_logits_max",
    "h_res_logits_grad_min",
    "h_res_logits_grad_max",
    "composite_amax_gain_fwd_max",
    "composite_amax_gain_bwd_max",
    "h_pre_logits_min",
    "h_pre_logits_max",
    "h_post_logits_min",
    "h_post_logits_max",
    "alpha_pre",
    "alpha_post",
    "alpha_res",
    "bias_pre_mean",
    "bias_pre_abs_max",
    "bias_post_mean",
    "bias_post_abs_max",
    "bias_res_mean",
    "bias_res_abs_max",
)

# Read once per flush from the module parameters, not from the hot path.
_PARAM_METRICS = (
    "alpha_pre",
    "alpha_post",
    "alpha_res",
    "bias_pre_mean",
    "bias_pre_abs_max",
    "bias_post_mean",
    "bias_post_abs_max",
    "bias_res_mean",
    "bias_res_abs_max",
)


def _vector_metric_specs(n: int) -> tuple[tuple[str, str, int], ...]:
    """``(metric_name, elem_tag, size)`` for the per-element mapping series."""
    return (("h_res", "cell", n * n), ("h_pre", "idx", n), ("h_post", "idx", n))


# Extrema, not means: a worst case that a mean over 43 layers would bury. The
# `_max` / `_min` suffixes keep training_logs' classifier in agreement on the
# full key too.
_MAX_METRICS = frozenset(
    {
        "branch_residual_share_max",
        "h_res_logits_max",
        "h_res_logits_grad_max",
        "composite_amax_gain_fwd_max",
        "composite_amax_gain_bwd_max",
        "h_pre_logits_max",
        "h_post_logits_max",
        "bias_pre_abs_max",
        "bias_post_abs_max",
        "bias_res_abs_max",
    }
)
_MIN_METRICS = frozenset(
    {
        "h_res_logits_min",
        "h_res_logits_grad_min",
        "h_pre_logits_min",
        "h_post_logits_min",
    }
)

# (component_name, layer attribute) — attn runs before mlp in the layer forward.
_COMPONENTS = (
    ("attn", "self_attention_hyper_connection"),
    ("mlp", "mlp_hyper_connection"),
)
_COMPONENT_ORDER = {
    component: order for order, (component, _) in enumerate(_COMPONENTS)
}


def _iter_chunks(model):
    """Yield ``(chunk_id, chunk)`` pairs for VPP-aware discovery.

    PaddleFleet exposes VPP model chunks via ``_model_chunks`` on the pipeline
    layer; each chunk owns its own ``run_function`` and executes in its own
    forward pass, so discovery has to walk all of them.

    Falls back to a single chunk ``(0, model)`` when no VPP chunking is present.
    """
    candidates = [model]
    if hasattr(model, "_layers"):
        candidates.append(model._layers)
    if hasattr(model, "module"):
        candidates.append(model.module)
    for cand in candidates:
        if cand is None:
            continue
        chunks = getattr(cand, "_model_chunks", None)
        if chunks:
            yield from enumerate(chunks)
            return
    yield 0, model


class PaddleMHCHealthMonitor(PaddleProbe):
    """Monitor the h_pre / h_post / h_res mappings of mHC hyper-connection layers."""

    METRIC_PREFIX = "mhc_health"
    # Populated in __init__ with component-prefixed names, because
    # record_layer_metric classifies on the name it is handed.
    MAX_AGGREGATED: set[str] = set()
    MIN_AGGREGATED: set[str] = set()

    def __init__(
        self,
        log_per_layer: bool = True,
        log_global: bool = True,
        monitor_interval: int = 1,
        verbose: bool = False,
        exclude_families=None,
    ):
        self.MAX_AGGREGATED = {
            f"{comp}_{m}" for comp, _ in _COMPONENTS for m in _MAX_METRICS
        }
        self.MIN_AGGREGATED = {
            f"{comp}_{m}" for comp, _ in _COMPONENTS for m in _MIN_METRICS
        }
        super().__init__(
            exclude_families=exclude_families,
            log_per_layer=log_per_layer,
            log_global=log_global,
            monitor_interval=monitor_interval,
            verbose=verbose,
        )
        # (module, attribute name, original bound method) triples, for remove_hooks().
        self._wrapped: list[tuple[nn.Layer, str, object]] = []
        # (layer_idx, component) -> token-mean of the *operator* h_res^T [n, n],
        # for the composite product. Keyed rather than appended so the product is
        # built in layer order, not call order (see _record_composite).
        self._h_res_snapshot: dict[tuple[int, str], paddle.Tensor] = {}
        # MTP layers are off the main trunk and must not enter the product.
        self._mtp_layer_ids: set[int] = set()
        # (layer_idx, component, module) for the step-level parameter series.
        self._param_targets: list[tuple[int, str, nn.Layer]] = []
        # Set by the capture wrapper: whether this step's forward was monitored.
        # A plain bool assignment, so it costs nothing on the hot path, and it
        # keeps the parameter series on exactly the steps the rest are on
        # (`_should_monitor()` cannot answer that at flush time — `step()` has
        # already incremented `step_count` by then).
        self._captured_this_step = False
        # Gradient hooks see AMP-scaled activation gradients. The callback
        # finalizes these accumulators with this step's scale before flush.
        self._grad_metrics_finalized = False

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _find_hc_modules(self, chunk):
        """Return ``[(global_idx, comp, hc_module)]`` for one chunk.

        Empty (auto-skip) when the mHC classes are unavailable or the chunk has
        no ``HyperConnectionTransformerLayer``. Uses ``isinstance`` against the
        real classes rather than duck-typing, so a plain ``TransformerLayer``
        (or an ``IdentityOp`` placeholder) is never matched.
        """
        if (
            HyperConnectionTransformerLayer is None
            or HyperConnectionModule is None
        ):
            return []
        layers = get_decoder_layers(chunk)
        if not layers:
            return []

        def matches(layer):
            return isinstance(layer, HyperConnectionTransformerLayer)

        monitor_layers = iter_monitor_layers(
            layers, matches, pp_rank=self.pp_rank
        )
        entries: list[tuple[int, str, nn.Layer]] = []
        # is_mtp so we can mark them for `_mtp` suffix in metric keys.
        mtp_layer_ids = [item.idx for item in monitor_layers if item.is_mtp]
        if mtp_layer_ids:
            self.mark_mtp_layers(mtp_layer_ids)
            self._mtp_layer_ids.update(mtp_layer_ids)
        for item in monitor_layers:
            for comp, attr in _COMPONENTS:
                mod = getattr(item.layer, attr, None)
                if not isinstance(mod, HyperConnectionModule):
                    continue
                entries.append((item.idx, comp, mod))
        return entries

    # ------------------------------------------------------------------
    # Setup: prepare (declare) -> allocate -> attach
    # ------------------------------------------------------------------

    def _init_parallel_state(self):
        try:
            from paddlefleet.parallel_state import (
                get_pipeline_model_parallel_rank,
            )

            self.pp_rank = get_pipeline_model_parallel_rank()
        except Exception:
            pass

    def _prepare(self, model):
        """Discover all hc modules across chunks and declare metric schema.

        Returns a list of chunks' entries; buffers are still unallocated so
        every ``declare_layer_metric`` call remains legal.
        """
        self._init_parallel_state()
        all_targets: list[list[tuple[int, str, nn.Layer]]] = []
        for _chunk_id, chunk in _iter_chunks(model):
            entries = self._find_hc_modules(chunk)
            if not entries:
                continue
            for global_idx, comp, mod in entries:
                for name in _METRIC_NAMES:
                    self.declare_layer_metric(global_idx, f"{comp}_{name}")
                # `n` is static module metadata, so reading it at declare time
                # costs no hot-path sync and keeps the schema fixed (Rule 3).
                for name, tag, size in _vector_metric_specs(
                    int(getattr(mod, "n", 0) or 0)
                ):
                    if size > 0:
                        self.declare_layer_vector(
                            global_idx, f"{comp}_{name}", size, elem_tag=tag
                        )
            all_targets.append(entries)
        return all_targets

    def _attach(self, all_targets):
        for entries in all_targets:
            for layer_idx, comp, mod in entries:
                orig = mod.compute_mappings
                mod.compute_mappings = self._make_capture(orig, layer_idx, comp)
                self._wrapped.append((mod, "compute_mappings", orig))
                self._param_targets.append((layer_idx, comp, mod))
                orig_h = getattr(mod, "_compute_h", None)
                if orig_h is not None:
                    mod._compute_h = self._make_logits_capture(
                        orig_h, layer_idx, comp, mod
                    )
                    self._wrapped.append((mod, "_compute_h", orig_h))
                orig_bda = getattr(mod, "fused_h_res_h_post_bda", None)
                if orig_bda is None:
                    continue
                mod.fused_h_res_h_post_bda = self._make_bda_capture(
                    orig_bda, layer_idx, comp
                )
                self._wrapped.append((mod, "fused_h_res_h_post_bda", orig_bda))
        logger.info(
            f"[PaddleMHCMonitor] Wrapped {len(self._wrapped)} hyper-connection methods."
        )

    def register_hooks(self, model):
        """Discover, declare, allocate, and attach — the whole three-phase setup."""
        all_targets = self._prepare(model)
        if not any(all_targets):
            logger.info(
                "[PaddleMHCMonitor] No hyper-connection layers found; skipping."
            )
            return
        self.allocate_buffers()
        self._attach(all_targets)

    def remove_hooks(self):
        # Restore the original bound methods so the monitor holds no module
        # references after teardown. ``del mod.<attr>`` removes the instance
        # attribute and falls back to the class method; if that fails (e.g.
        # slots), we re-bind the captured original.
        for mod, attr, orig in self._wrapped:
            try:
                delattr(mod, attr)
            except AttributeError:
                setattr(mod, attr, orig)
        self._wrapped = []
        self._param_targets = []
        self._h_res_snapshot.clear()
        super().remove_hooks()

    # ------------------------------------------------------------------
    # Composite product (flush time, not hot path)
    # ------------------------------------------------------------------

    def _record_composite(self) -> None:
        """Record the gains of the layer-ordered cumulative product of h_res^T.

        The snapshots hold ``h_res^T``, i.e. the matrix ``apply_h_res`` actually
        applies to the stream vector (``mixed = h_res^T @ residual``), so the
        product ``A_k = h_res_k^T ... h_res_0^T`` is the real composite operator
        from the first layer up to layer ``k``. ``_fwd`` is its max-abs row sum
        (worst forward signal gain), ``_bwd`` its max-abs column sum.

        Scope is the layers this rank owns: the whole model under sharding (every
        rank holds all layers), but one stage under real pipeline parallelism,
        which we can neither detect nor correct while
        ``get_pipeline_model_parallel_{rank,world_size}`` are stubs in PaddleFleet
        (they return 0 / 1 unconditionally).
        """
        branches = sorted(
            (
                (idx, component, mat)
                for (idx, component), mat in self._h_res_snapshot.items()
                if idx not in self._mtp_layer_ids
            ),
            key=lambda item: (item[0], _COMPONENT_ORDER[item[1]]),
        )

        prefix = None
        for layer_idx, component, mat in branches:
            prefix = mat if prefix is None else paddle.matmul(mat, prefix)
            self.record_layer_metric(
                layer_idx,
                f"{component}_composite_amax_gain_fwd_max",
                amax_gain(prefix, axis=-1),
            )

        suffix = None
        for layer_idx, component, mat in reversed(branches):
            suffix = mat if suffix is None else paddle.matmul(suffix, mat)
            self.record_layer_metric(
                layer_idx,
                f"{component}_composite_amax_gain_bwd_max",
                amax_gain(suffix, axis=-2),
            )

    def finalize_composite_microbatch(self) -> None:
        """Record and release one microbatch's detached composite snapshots."""
        if not self._h_res_snapshot:
            return
        try:
            with paddle.no_grad():
                self._record_composite()
        except Exception as e:
            if self.verbose:
                logger.error(f"[PaddleMHCMonitor] Error composite: {e}")
        finally:
            self._h_res_snapshot.clear()

    def finalize_scaled_grad_metrics(self, scaler=None) -> None:
        """Remove AMP loss scaling from this step's logits-gradient extrema."""
        # `on_optimizer_begin` is the last hook before `optimizer_step()`, so it
        # is also where the Eq. (7) parameters still hold the values this step's
        # forward actually used. Called outside the guard below and idempotent
        # within a step, so the `_flush_buffers` fallback stays harmless.
        self.finalize_param_metrics()
        if self._grad_metrics_finalized:
            return
        scale = getattr(scaler, "_scale", None) if scaler is not None else None
        if scale is not None:
            scale = paddle.assign(scale).detach().astype("float32")
            for key in self._max_keys | self._min_keys:
                if "_h_res_logits_grad_" in key and self._gpu_cnt[key] > 0:
                    self._gpu_acc[key].divide_(scale)
        self._grad_metrics_finalized = True

    def finalize_param_metrics(self) -> None:
        """Record the step-level ``alpha`` / ``bias`` series from the parameters.

        Cold path: once per step, so nine tiny reductions per module instead of
        per microbatch. Skipped entirely when this step's forward was not
        monitored, which keeps these series sampled on the same steps as the
        activation ones under ``monitor_interval > 1`` — ``_should_monitor()``
        cannot answer that here, because ``step()`` increments ``step_count``
        before ``_flush_buffers`` runs.

        Read point is ``on_optimizer_begin`` (via ``finalize_scaled_grad_metrics``),
        i.e. *before* ``optimizer.step()``, so the values line up with the forward
        that produced this step's activation metrics. A caller that only drives
        ``step()`` (direct users, and PaddleFormers' ``on_step_end`` runs after
        the optimizer) falls back to the flush path and therefore logs the
        post-update value — one update later than the rest of the step's series.

        The parameters are replicated across TP ranks (``HyperConnectionModule``
        uses a plain ``nn.Linear`` and marks its params ``is_distributed=False``),
        so no collective is needed — every rank holds the same values.
        """
        if not self._captured_this_step:
            return
        try:
            for layer_idx, component, mod in self._param_targets:
                try:
                    with paddle.no_grad():
                        stats = mapping_param_stats(
                            mod.alpha_pre,
                            mod.alpha_post,
                            mod.alpha_res,
                            mod.bias,
                            int(mod.n),
                        )
                    for name, value in stats.items():
                        self.record_layer_metric(
                            layer_idx, f"{component}_{name}", value
                        )
                except Exception as e:
                    # Per module, so one module without the Eq. (7) parameters
                    # cannot silence the series for every other layer.
                    if self.verbose:
                        logger.error(
                            f"[PaddleMHCMonitor] Error mapping params {layer_idx}/{component}: {e}"
                        )
        finally:
            self._captured_this_step = False

    def _flush_buffers(self) -> None:
        # Direct users and non-AMP trainers may not emit on_optimizer_begin.
        # finalize_scaled_grad_metrics also covers the parameter series, so this
        # is the fallback read point for both.
        self.finalize_scaled_grad_metrics()
        self.finalize_composite_microbatch()
        super()._flush_buffers()
        self._grad_metrics_finalized = False

    # ------------------------------------------------------------------
    # Capture wrapper (the hot path)
    # ------------------------------------------------------------------

    def _make_capture(self, orig, layer_idx: int, component: str):
        """Wrap ``compute_mappings`` to record metrics from its real return value."""

        def wrapped(x):
            out = orig(
                x
            )  # the real mappings the model consumes — returned unchanged
            if not self._should_monitor():
                return out
            try:
                h_pre, h_post, h_res = out
                self._captured_this_step = True
                with paddle.no_grad():
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
                    for name, value in h_post_structure_stats(h_post).items():
                        self.record_layer_metric(
                            layer_idx, f"{component}_{name}", value
                        )

                    self.record_layer_metric(
                        layer_idx,
                        f"{component}_amax_gain_fwd",
                        amax_gain(h_res, axis=-2),
                    )
                    self.record_layer_metric(
                        layer_idx,
                        f"{component}_amax_gain_bwd",
                        amax_gain(h_res, axis=-1),
                    )

                    # Token-mean of every mapping element. One reduce feeds both
                    # the per-cell series and the composite product's snapshot.
                    n = int(h_res.shape[-1])
                    res_mean = (
                        h_res.astype("float32").reshape([-1, n, n]).mean(axis=0)
                    )
                    self.record_layer_vector(
                        layer_idx, f"{component}_h_res", res_mean.reshape([-1])
                    )
                    self.record_layer_vector(
                        layer_idx,
                        f"{component}_h_pre",
                        h_pre.astype("float32").reshape([-1, n]).mean(axis=0),
                    )
                    self.record_layer_vector(
                        layer_idx,
                        f"{component}_h_post",
                        h_post.astype("float32").reshape([-1, n]).mean(axis=0),
                    )

                    # Token-mean snapshot of the *operator* for the composite product.
                    self._h_res_snapshot[(layer_idx, component)] = (
                        res_mean.transpose([1, 0])
                    )
            except Exception as e:
                if self.verbose:
                    logger.error(
                        f"[PaddleMHCMonitor] Error layer {layer_idx}/{component}: {e}"
                    )
            return out

        return wrapped

    def _make_logits_capture(self, orig, layer_idx: int, component: str, mod):
        """Wrap ``_compute_h`` to record the saturation sentinels.

        Two families come from here: the residual-mixing logits (the real return
        value) and the pre-sigmoid ``H_pre`` / ``H_post`` logits, which
        ``_compute_h`` does not return and are therefore rebuilt from its own
        ``(proj, r)`` arguments plus ``mod``'s ``alpha`` / ``bias`` — see
        ``gate_logits_extrema``.
        """

        def wrapped(proj, r):
            out = orig(
                proj, r
            )  # the real mappings the model consumes — returned unchanged
            if not self._should_monitor():
                return out
            try:
                _h_pre, _h_post, h_res_logits = out
                if not h_res_logits.stop_gradient:

                    def record_grad_extrema(grad):
                        try:
                            grad_fp32 = grad.detach().astype("float32")
                            self.record_layer_metric(
                                layer_idx,
                                f"{component}_h_res_logits_grad_min",
                                grad_fp32.min(),
                            )
                            self.record_layer_metric(
                                layer_idx,
                                f"{component}_h_res_logits_grad_max",
                                grad_fp32.max(),
                            )
                            self._grad_metrics_finalized = False
                        except Exception as e:
                            if self.verbose:
                                logger.error(
                                    f"[PaddleMHCMonitor] Error logits gradient layer {layer_idx}/{component}: {e}"
                                )
                        return grad

                    h_res_logits.register_hook(record_grad_extrema)
                with paddle.no_grad():
                    for name, value in h_res_logits_extrema(
                        h_res_logits
                    ).items():
                        self.record_layer_metric(
                            layer_idx, f"{component}_{name}", value
                        )
                    # proj / r are the only route to the pre-sigmoid pre/post
                    # logits; a caller that does not supply them (a stub, or a
                    # future fused path that bypasses this signature) simply
                    # leaves those four accumulators empty.
                    if proj is not None and r is not None:
                        gate_logits = gate_logits_extrema(
                            proj,
                            r,
                            mod.alpha_pre,
                            mod.alpha_post,
                            mod.bias,
                            int(mod.n),
                        )
                        for name, value in gate_logits.items():
                            self.record_layer_metric(
                                layer_idx, f"{component}_{name}", value
                            )
            except Exception as e:
                if self.verbose:
                    logger.error(
                        f"[PaddleMHCMonitor] Error logits layer {layer_idx}/{component}: {e}"
                    )
            return out

        return wrapped

    def _make_bda_capture(self, orig, layer_idx: int, component: str):
        """Wrap ``fused_h_res_h_post_bda`` to record the branch/residual ratio.

        The only place both mHC update terms are available: the call receives
        ``h_res`` + ``original_residual`` and ``layer_output_with_bias``. Metrics
        come from the **arguments**, so the fused fast path and the sequential
        dropout path are covered identically. Arguments are read by keyword with a
        positional fallback — PaddleFleet passes keywords today, but relying on
        that alone would silently stop recording if it switched.
        """

        def wrapped(*args, **kwargs):
            out = orig(*args, **kwargs)
            if not self._should_monitor():
                return out
            try:

                def pick(name, pos):
                    if name in kwargs:
                        return kwargs[name]
                    return args[pos] if len(args) > pos else None

                h_res = pick("h_res", 0)
                original_residual = pick("original_residual", 1)
                h_post = pick("h_post", 2)
                layer_output_with_bias = pick("layer_output_with_bias", 3)
                if h_res is None or original_residual is None or h_post is None:
                    return out
                if isinstance(layer_output_with_bias, tuple):
                    layer_output, bias = layer_output_with_bias
                else:
                    layer_output, bias = layer_output_with_bias, None
                if layer_output is None:
                    return out

                with paddle.no_grad():
                    stats = branch_residual_share(
                        h_res, original_residual, h_post, layer_output, bias
                    )
                    for name, value in stats.items():
                        self.record_layer_metric(
                            layer_idx, f"{component}_{name}", value
                        )
            except Exception as e:
                if self.verbose:
                    logger.error(
                        f"[PaddleMHCMonitor] Error bda layer {layer_idx}/{component}: {e}"
                    )
            return out

        return wrapped


def setup_mhc_monitor(
    model,
    log_per_layer: bool = True,
    log_global: bool = True,
    monitor_interval: int = 1,
    verbose: bool = False,
    monitor_dict: dict | None = None,
    exclude_families=None,
):
    """Enable the mHC health monitor. No-op on any non-mHC model.

    Multi-chunk (VPP / interleaved 1F1B) safe: declares the schema across all
    chunks before ``allocate_buffers`` locks it, then attaches.
    """
    # No-op guarantee #1: mHC classes unavailable -> touch nothing.
    if HyperConnectionTransformerLayer is None or HyperConnectionModule is None:
        logger.info(
            "[PaddleMHCMonitor] Hyper-connection classes unavailable; skipping."
        )
        return model

    monitor = PaddleMHCHealthMonitor(
        exclude_families=exclude_families,
        log_per_layer=log_per_layer,
        log_global=log_global,
        monitor_interval=monitor_interval,
        verbose=verbose,
    )
    monitor.register_hooks(model)
    if monitor._wrapped and monitor_dict is not None:
        monitor_dict["mhc_health"] = monitor
    logger.info(
        f"[PaddleMHCMonitor] Setup complete. Monitoring {len(monitor._wrapped)} hc modules."
    )
    return model
