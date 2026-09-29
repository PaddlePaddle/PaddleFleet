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

"""Activation-gradient health monitor for PaddleFleet.

Answers "where does the gradient grow or die on the way back" with one L2 norm
per module output per layer. Three positions per layer -- the attention branch
output, the MLP/MoE branch output, and the decoder layer output (the residual
stream itself) -- times the three magnitudes in ``grad_metrics``.

Collection is a ``forward_post_hook`` that registers a tensor gradient hook on
the module's output. The forward hook measures nothing itself: it exists only to
get hold of the tensor whose gradient is wanted, so the forward path pays one
``register_hook`` per module and the reductions all happen during backward.

**AMP.** A gradient hook sees the *loss-scaled* activation gradient, so the raw
numbers are the scale (order 1e4, and it moves on every overflow) times the
quantity of interest. ``finalize_scaled_grad_metrics`` divides the scale back out
once per step from ``on_optimizer_begin`` -- before ``optimizer.step()``, while
``scaler._scale`` still holds the value this step's backward actually used.
Without that step the curves would jump by 2x every time the scaler halves.

**Sharding.** No collective runs here (hot-path discipline: see
``.claude/skills/monitor-hook-perf-rules``); the cross-rank reduction happens at
flush time in ``gather.py``. Under TP / SP / CP each rank therefore measures its
own shard: ``rms`` is the true global value when shards are equal-sized, while
``norm`` is the per-shard norm -- a fixed ``1/sqrt(num_shards)`` of the global
one, so it tracks the same trend but is not the number a global grad-norm clip
would report.

**Recompute.** ``_should_monitor`` is False while grad is disabled, so the
discarded first forward of a recomputed block registers nothing and the replayed
forward -- the one whose graph actually carries the backward -- is what gets
hooked.
"""

from __future__ import annotations

import logging

import paddle
from paddle import nn

from .base import PaddleProbe
from .grad_metrics import (
    GLOBAL_METRICS,
    MAX_AGGREGATED,
    METRICS,
    TOKEN_METRICS,
    grad_square_and_stats,
)
from .grad_reduce import (
    DATA_CANDIDATES,
    SHARD_CANDIDATES,
    GroupInfo,
    describe,
    plan_total_groups,
)
from .layer_discovery import get_decoder_layers, iter_monitor_layers

logger = logging.getLogger(__name__)


def _is_transformer_layer(layer) -> bool:
    """Same predicate ``massive_act`` uses, so both monitors see one layer set."""
    return (
        hasattr(layer, "self_attn")
        or hasattr(layer, "self_attention")
        or hasattr(layer, "input_layernorm")
    )


def _output_tensor(outputs):
    """First tensor of a PaddleFleet forward result, or ``None``."""
    if isinstance(outputs, paddle.Tensor):
        return outputs
    if isinstance(outputs, dict):
        return outputs.get("hidden_states")
    if isinstance(outputs, (tuple, list)) and outputs:
        return _output_tensor(outputs[0])
    return None


def _branch_modules(layer):
    """``[(position, module)]`` for the branches this layer actually has.

    Only positions that exist are returned, and the same list drives both the
    schema and the hooks -- so a layer without an ``mlp`` never declares a key it
    could not fill.
    """
    modules = []
    attn = getattr(layer, "self_attn", None)
    if attn is None:
        attn = getattr(layer, "self_attention", None)
    if attn is not None:
        modules.append(("attn_out", attn))
    ffn = getattr(layer, "mlp", None)
    if ffn is None:
        ffn = getattr(layer, "moe", None)
    if ffn is not None:
        modules.append(("ffn_or_moe_out", ffn))
    # The layer itself is the residual stream leaving this block.
    modules.append(("layer_out", layer))
    return modules


class ExactNormReducer:
    """Turns per-rank sums of squares into exact global norms, once per step.

    Why a collective at all: ``norm`` / ``rms`` are recorded per shard and then
    *averaged* across ranks, which is only an approximation of the global value
    (the arithmetic mean of per-rank RMS understates the true RMS by about half
    the squared cross-rank coefficient of variation). Summing squares instead is
    exact, but sums do not commute with sharding for free -- hence this.

    Where the collective goes: **flush time, one batched reduction per group**,
    never in a hook. A 43-layer stack with 3 positions and 6 microbatches would
    otherwise queue ~774 all_reduces per step, which is the exact failure mode
    ``monitor-hook-perf-rules`` was written about.

    Two independent sums come off the *same* local value:

    - **shard sum** over the groups that split one tensor -- ``cp`` always, and
      ``tp`` only under ``sequence_parallel`` (without SP the very same tensors
      are *replicated* across TP, so summing would multiply every squared norm by
      ``tp_size``). Feeds ``norm_mb``.
    - **total sum** over a group set that covers every participating rank
      *exactly once*. Feeds ``rms_global`` / ``norm_global``.

    The two are never chained. An earlier version reduced over ``cp`` and then
    continued on the same buffer over ``dp``, which silently assumed the two rank
    sets were orthogonal. On a ``sharding_first`` topology that assumption fails:
    with ``data_parallel_size=1`` the samples live on the sharding dimension, so
    ``_DATA_PARALLEL_GROUP`` spans ``cp`` and the context-parallel contribution
    got counted twice -- ``norm_global`` came out ``sqrt(cp)`` high (+41% at
    ``cp=2``) with nothing in the numbers to show it. Chaining was only ever
    correct by luck.

    So membership is *checked*, not assumed:

    - two mesh groups through this rank are orthogonal iff they intersect in
      exactly this rank; a candidate that overlaps an already-chosen group more
      than that is dropped rather than double counted,
    - the product of the chosen sizes must equal ``world_size / pp_size``. When it
      does not, ``norm_global`` is **suppressed** rather than emitted wrong: a
      partial cover makes it the norm of a sub-batch, while ``rms_global`` stays
      an unbiased estimate on any subset and keeps being reported.

    ``pp`` is never summed: its ranks own different layers, so their key sets
    differ and a reduction across them would deadlock on mismatched shapes.
    """

    def __init__(self, sequence_parallel: bool, verbose: bool = False):
        self.shard_groups: list = []  # same tensor, split across ranks
        self.total_groups: list = []  # covers every rank exactly once
        self.shard_factor = 1
        self.total_factor = 1
        self.enabled = False
        self.exact_total = False  # False -> norm_global is suppressed
        self.plan = "no distributed context"
        self._warned = False
        self._init_groups(sequence_parallel, verbose)

    def _init_groups(self, sequence_parallel: bool, verbose: bool) -> None:
        try:
            import paddle.distributed as dist

            if not dist.is_initialized():
                return
        except Exception:
            return
        try:
            from paddlefleet.process_groups_config import ProcessGroupCollection
        except (
            Exception
        ) as exc:  # pragma: no cover - depends on backend version
            self.plan = f"process groups unavailable ({exc})"
            return

        world = int(dist.get_world_size())
        found = {}
        for name in dict.fromkeys(SHARD_CANDIDATES + DATA_CANDIDATES + ("pp",)):
            info = self._resolve(ProcessGroupCollection, name)
            if info is not None:
                found[name] = info
        # ``pp`` is never summed, but its size says how much of the world one
        # rank's key set is expected to cover: pipeline ranks own different layers.
        pp_size = found["pp"].size if "pp" in found else 1
        target = max(1, world // max(1, pp_size))

        for name in SHARD_CANDIDATES:
            if name == "tp" and not sequence_parallel:
                continue
            info = found.get(name)
            if info is not None and info.size > 1:
                self.shard_groups.append(info)
                self.shard_factor *= info.size

        # Shard groups are candidates for the total too: the ranks that split one
        # tensor still hold part of the batch.
        pool = [
            found[name] for name in DATA_CANDIDATES if name in found
        ] + list(self.shard_groups)
        self.total_groups, self.total_factor = plan_total_groups(pool, target)
        self.exact_total = self.total_factor == target
        self.enabled = True
        self.plan = (
            f"shard[{describe(self.shard_groups)}] total[{describe(self.total_groups)}]"
            f" covering {self.total_factor}/{target}"
        )
        seen = (
            ", ".join(f"{name}={info.size}" for name, info in found.items())
            or "none"
        )
        if verbose or not self.exact_total:
            emit = logger.info if self.exact_total else logger.warning
            emit(
                f"[PaddleGradMonitor] exact-norm reduction {self.plan}; world={world}; groups seen: {seen}"
            )
        if not self.exact_total:
            logger.warning(
                "[PaddleGradMonitor] chosen groups cover %d of %d ranks, so norm_global is "
                "suppressed -- a partial cover would make it a sub-batch norm. rms_global is "
                "unbiased on any subset and keeps being reported.",
                self.total_factor,
                target,
            )

    @staticmethod
    def _resolve(collection, name) -> GroupInfo | None:
        """One named process group as a ``GroupInfo``, or ``None`` if absent.

        Membership comes from the group itself (``nranks`` / ``ranks``) rather than
        from a size helper, because the plan needs the rank *sets* to test
        orthogonality -- sizes alone cannot tell a nested group from a disjoint one.
        """
        try:
            pg = collection.use_mpu_process_groups(required_pgs=[name])
            group = getattr(pg, name, None)
            if group is None:
                return None
            return GroupInfo(
                name,
                group,
                getattr(group, "nranks", 1),
                getattr(group, "ranks", ()) or (),
            )
        except Exception:
            return None

    def reduce(self, sq_values, counts):
        """``(shard_sq, total_sq, total_count)`` from this rank's running sums.

        The two sums are **independent**, both taken from the same local value on
        its own copy. They are deliberately not chained: continuing the data
        reduction on top of the shard-reduced buffer would double count whenever a
        data group happens to span a shard group, which is exactly how
        ``norm_global`` came out ``sqrt(cp)`` high on a sharding-first topology.

        Each vector pair travels as one concatenated tensor, so the whole schema
        costs one reduction per group rather than one per key.

        With nothing to reduce (single rank, or a layout that shards nothing) this
        is the identity, so the exact metrics stay well defined and equal to their
        local counterparts rather than disappearing.
        """
        if not self.enabled or (
            not self.shard_groups and not self.total_groups
        ):
            return sq_values, sq_values, counts
        import paddle
        import paddle.distributed as dist

        n = int(sq_values.shape[0])
        local = paddle.concat([sq_values, counts])

        shard = local.clone()
        for info in self.shard_groups:
            dist.all_reduce(shard, group=info.group)

        total = local.clone()
        for info in self.total_groups:
            dist.all_reduce(total, group=info.group)

        return shard[:n], total[:n], total[n:]


class PaddleGradHealthMonitor(PaddleProbe):
    """L2 norm / RMS / abs-max of the activation gradient, per layer per module."""

    METRIC_PREFIX = "grad_health"
    MAX_AGGREGATED = set(MAX_AGGREGATED)
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
            log_per_layer=log_per_layer,
            log_global=log_global,
            monitor_interval=monitor_interval,
            verbose=verbose,
            exclude_families=exclude_families,
        )
        self.sample_layers = set(sample_layers) if sample_layers else None
        self._failed: set[tuple[int, str]] = set()
        # AMP: latched so the scale is divided out exactly once per step, and so a
        # step whose backward recorded nothing is left alone.
        self._grad_metrics_finalized = False
        # Exact-norm path. Keyed by ``(layer_idx, position, attn_type)``: a running
        # sum of squares on GPU plus the element and microbatch counts as plain
        # Python ints, so the hot path adds one ``add_`` and two integer
        # increments and still never syncs.
        self._sq_acc: dict[tuple, paddle.Tensor] = {}
        self._sq_numel: dict[tuple, int] = {}
        self._sq_micro: dict[tuple, int] = {}
        self._reducer: ExactNormReducer | None = None

    # ------------------------------------------------------------------
    # Setup: discover -> declare -> allocate -> attach
    # ------------------------------------------------------------------

    def _init_parallel_state(self) -> None:
        try:
            from paddlefleet.parallel_state import (
                get_pipeline_model_parallel_rank,
            )

            self.pp_rank = get_pipeline_model_parallel_rank()
        except Exception:
            pass

    def _find_targets(self, model):
        """``[(layer_idx, position, module, attn_type)]`` for every hooked module."""
        layers = get_decoder_layers(model)
        if not layers:
            return []

        monitor_layers = iter_monitor_layers(
            layers, _is_transformer_layer, pp_rank=self.pp_rank
        )
        mtp_layer_ids = [item.idx for item in monitor_layers if item.is_mtp]
        if mtp_layer_ids:
            self.mark_mtp_layers(mtp_layer_ids)

        targets = []
        for item in monitor_layers:
            if self.sample_layers and item.idx not in self.sample_layers:
                continue
            for position, module in _branch_modules(item.layer):
                targets.append((item.idx, position, module, item.attn_type))
        return targets

    def register_hooks(self, model: nn.Layer):
        self._init_parallel_state()
        targets = self._find_targets(model)
        if not targets:
            logger.info(
                "[PaddleGradMonitor] No transformer layers found; skipping."
            )
            return

        for layer_idx, position, _module, attn_type in targets:
            for metric in METRICS + GLOBAL_METRICS + TOKEN_METRICS:
                self.declare_layer_metric(
                    layer_idx, f"{position}_{metric}", attn_type=attn_type
                )

        self.allocate_buffers()

        # Exact-norm infrastructure: built after allocate so the GPU device is known.
        sp_on = self._detect_sp(model)
        self._reducer = ExactNormReducer(
            sequence_parallel=sp_on, verbose=self.verbose
        )
        for layer_idx, position, _module2, attn_type in targets:
            self._init_sq_accum(layer_idx, position, attn_type)

        for layer_idx, position, module, attn_type in targets:
            self.hooks.append(
                module.register_forward_post_hook(
                    self._make_output_hook(layer_idx, position, attn_type)
                )
            )

        layer_count = len({layer_idx for layer_idx, _p, _m, _a in targets})
        logger.info(
            f"[PaddleGradMonitor] Registered {len(self.hooks)} hooks across {layer_count} layers."
        )

    # ------------------------------------------------------------------
    # Hooks (the hot path)
    # ------------------------------------------------------------------

    def _log_failure(
        self, layer_idx: int, position: str, exc: Exception
    ) -> None:
        if self.verbose and (layer_idx, position) not in self._failed:
            logger.error(
                f"[PaddleGradMonitor] Error at layer {layer_idx}/{position}: {exc}"
            )
            self._failed.add((layer_idx, position))

    def _make_output_hook(
        self, layer_idx: int, position: str, attn_type: str | None
    ):
        """Forward hook: attach a gradient hook to this module's output tensor.

        ``stop_gradient`` outputs are skipped rather than guarded later: a tensor
        outside the graph will never call the hook, so registering one would only
        cost a closure per microbatch.
        """

        def hook_fn(module, _inputs, outputs):
            if not module.training or not self._should_monitor():
                return None
            try:
                tensor = _output_tensor(outputs)
                if tensor is None or tensor.stop_gradient:
                    return None
                tensor.register_hook(
                    self._make_grad_recorder(layer_idx, position, attn_type)
                )
            except Exception as exc:
                self._log_failure(layer_idx, position, exc)
            return None

        return hook_fn

    def _make_grad_recorder(
        self, layer_idx: int, position: str, attn_type: str | None
    ):
        """Gradient hook: reduce ``grad`` into the accumulators, return it unchanged."""
        slot = (layer_idx, position, attn_type)

        def record(grad):
            try:
                with paddle.no_grad():
                    stats, sum_sq, numel = grad_square_and_stats(grad)
                    for metric, value in stats.items():
                        self.record_layer_metric(
                            layer_idx,
                            f"{position}_{metric}",
                            value,
                            attn_type=attn_type,
                        )
                    buf = self._sq_acc.get(slot)
                    if buf is not None:
                        buf.add_(sum_sq)
                        self._sq_numel[slot] += numel
                        self._sq_micro[slot] += 1
                self._grad_metrics_finalized = False
            except Exception as exc:
                self._log_failure(layer_idx, position, exc)
            return grad

        return record

    # ------------------------------------------------------------------
    # Exact global norms (cold path, one batched collective per step)
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_sp(model) -> bool:
        """Whether activations are sequence-sharded inside the TP group.

        Read off the model config rather than guessed: it decides whether the TP
        group is summed at all, and getting it wrong scales every squared norm by
        ``tp_size``.
        """
        for attr in ("config", "_config"):
            config = getattr(model, attr, None)
            if config is not None and hasattr(config, "sequence_parallel"):
                return bool(config.sequence_parallel)
        layers = get_decoder_layers(model) or []
        for layer in layers:
            config = getattr(layer, "config", None)
            if config is not None and hasattr(config, "sequence_parallel"):
                return bool(config.sequence_parallel)
        return False

    def _init_sq_accum(
        self, layer_idx: int, position: str, attn_type: str | None
    ) -> None:
        slot = (layer_idx, position, attn_type)
        self._sq_acc[slot] = paddle.zeros((), dtype="float32")
        self._sq_numel[slot] = 0
        self._sq_micro[slot] = 0

    def _reset_sq_accum(self) -> None:
        for slot, buf in self._sq_acc.items():
            buf.zero_()
            self._sq_numel[slot] = 0
            self._sq_micro[slot] = 0

    # ------------------------------------------------------------------
    # AMP de-scaling (cold path, once per step)
    # ------------------------------------------------------------------

    def finalize_scaled_grad_metrics(self, scaler=None) -> None:
        """Divide this step's AMP loss scale out of every accumulator.

        Every metric this monitor owns is degree-1 homogeneous in the gradient, so
        one division fixes all of them -- and it is valid on the raw accumulator
        because both aggregations commute with a positive scale: ``sum(g_i)/S`` is
        the mean of ``g_i/S``, and ``max(g_i)/S`` is the max of ``g_i/S``. The
        scaler updates ``_scale`` in its own ``step``/``update``, i.e. once per
        optimizer step, so every microbatch folded into these sums shared it.

        Idempotent within a step: called from ``on_optimizer_begin`` when the
        trainer provides a scaler, with ``_flush_buffers`` as the fallback read
        point for direct users and non-AMP runs.
        """
        if self._grad_metrics_finalized:
            return
        # Before the division, so the exact norms get de-scaled with everything
        # else -- they are sqrt of a sum of squares, i.e. also degree-1 in g.
        self._emit_exact_norms()
        scale = getattr(scaler, "_scale", None) if scaler is not None else None
        if scale is not None:
            scale = paddle.assign(scale).detach().astype("float32")
            for key in self._mean_keys | self._max_keys:
                if self._gpu_cnt.get(key, 0) > 0:
                    self._gpu_acc[key].divide_(scale)
        self._grad_metrics_finalized = True

    def _emit_exact_norms(self) -> None:
        """One batched collective, then write the three exact series.

        Recorded through ``record_layer_metric`` like any other value, so family
        filtering, MTP tagging and the ``log_per_layer`` / global derivation all
        keep working. Each key is written once, so the mean accumulator divides by
        a count of 1 and the number survives the flush unchanged; and because
        every rank inside the reduction ends up with the same value, the cross-rank
        mean in ``gather.py`` is a no-op for the two global series.
        """
        slots = [
            slot
            for slot in sorted(self._sq_acc, key=str)
            if self._sq_micro[slot] > 0
        ]
        if not slots or self._reducer is None:
            self._reset_sq_accum()
            return
        try:
            with paddle.no_grad():
                local_sq = paddle.stack([self._sq_acc[slot] for slot in slots])
                counts = paddle.to_tensor(
                    [float(self._sq_numel[slot]) for slot in slots],
                    dtype="float32",
                )
                shard_sq, total_sq, total_count = self._reducer.reduce(
                    local_sq, counts
                )
                micro = paddle.to_tensor(
                    [float(self._sq_micro[slot]) for slot in slots],
                    dtype="float32",
                )
                # norm_mb: complete tensor of one microbatch, quadratic-mean over
                # the microbatches this rank saw (the accumulator holds their sum,
                # so the arithmetic mean of per-microbatch norms is not recoverable
                # -- and the quadratic mean is the more natural batch quantity).
                norm_mb = paddle.sqrt(shard_sq / micro.clip(min=1.0))
                rms_global = paddle.sqrt(total_sq / total_count.clip(min=1.0))
                norm_global = paddle.sqrt(total_sq)
                # ``norm_global`` only means "the whole batch" when the reduction
                # covered every rank exactly once; on a partial cover it would be
                # the norm of a sub-batch, so it is dropped rather than logged
                # wrong. ``rms_global`` is unbiased on any subset and stays.
                emit_total_norm = (
                    self._reducer.exact_total or not self._reducer.enabled
                )
                for i, (layer_idx, position, attn_type) in enumerate(slots):
                    series = [
                        ("norm_mb", norm_mb[i]),
                        ("rms_global", rms_global[i]),
                    ]
                    if emit_total_norm:
                        series.append(("norm_global", norm_global[i]))
                    for metric, value in series:
                        self.record_layer_metric(
                            layer_idx,
                            f"{position}_{metric}",
                            value,
                            attn_type=attn_type,
                        )
        except Exception as exc:
            if self.verbose and not getattr(self, "_exact_warned", False):
                logger.error(
                    f"[PaddleGradMonitor] exact-norm reduction failed: {exc}"
                )
                self._exact_warned = True
        finally:
            self._reset_sq_accum()

    def _flush_buffers(self) -> None:
        self.finalize_scaled_grad_metrics()
        super()._flush_buffers()
        self._grad_metrics_finalized = False


def setup_grad_monitor(
    model,
    log_per_layer: bool = True,
    log_global: bool = True,
    monitor_interval: int = 1,
    verbose: bool = False,
    sample_layers: list[int] | None = None,
    monitor_dict: dict | None = None,
    exclude_families=None,
):
    monitor = PaddleGradHealthMonitor(
        log_per_layer=log_per_layer,
        log_global=log_global,
        monitor_interval=monitor_interval,
        verbose=verbose,
        sample_layers=sample_layers,
        exclude_families=exclude_families,
    )
    monitor.register_hooks(model)
    if monitor_dict is not None:
        monitor_dict["grad_health"] = monitor
    return model
