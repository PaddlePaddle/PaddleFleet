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

"""MoE expert update monitoring for PaddleFleet.

Tracks the parameter increment of the expert MLPs,

    dW[l, e, m] = W[l, e, m]_t - W[l, e, m]_{t - delta},

for layer ``l``, local expert ``e`` and matrix ``m`` in gate / up / down, then
reduces it to per-layer statistics over the experts. This is the MoE counterpart
of ``attn_update``: same "read weights at step boundaries, keep the previous
reading as the base point" scheme, no forward hooks.

The headline metric is the RELATIVE update

    r[l, e, m] = ||dW[l, e, m]||_F / (||W[l, e, m]||_F + eps),

which is comparable across layers and experts in a way ``||dW||_F`` alone is not.
Each matrix also reports the stable rank ``||dW||_F^2 / ||dW||_2^2``, i.e. how
many directions the update actually spans.

Per the diagnostic intent the reduction order matters: every (layer, expert,
matrix) triple is measured on its own and only then summarised per layer, never
by concatenating experts into one matrix. The summaries are mean / median /
p10 / p90 / min / max, which is what separates the four failure modes this is
for -- one expert updating far too much (max), experts that barely move at all
(min, p10), uneven learning speed across experts (p90 - p10), and an update that
narrows onto a few singular directions (stable rank).

``sigma_1 / ||dW||_F`` is deliberately not reported: it equals
``1 / sqrt(stable_rank)`` exactly, so it carries no information the stable rank
does not already carry.

Cost and precision, measured on the 4B-A500M shapes (32 local experts per EP
rank, weight1 ``[32, 512, 1024]``, weight2 ``[32, 512, 512]``), one H800:

* Norms plus a batched power iteration for ``sigma_1``: 11.3 ms/layer, 0.20 s
  over 18 layers. This is the default tier.
* The full Gram + ``eigvalsh`` spectrum needed for the singular entropy:
  500.9 ms/layer, 9.02 s over 18 layers. Off unless ``log_spectrum=True``.

``sigma_1`` comes from a power iteration rather than an eigensolve, and its error
profile happens to favour the thing being detected: convergence goes as
``(sigma_2 / sigma_1)^(2k)``, so at 30 iterations a rank-dominated update
(``sigma_2 / sigma_1 = 0.25``) is exact to 5e-8 while a flat random update
(``0.99``) carries ~1.2% on ``sigma_1`` and ~2.4% on the stable rank. The
imprecise case is the one where the stable rank is large and its exact value does
not change the reading; the precise case is the collapse being monitored. The
error is one-sided -- an underestimated ``sigma_1`` inflates the stable rank.

Two limits worth knowing before reading the curves:

* The base snapshot is a clone in the parameter's own dtype, so for bf16 expert
  weights ``dW`` is exact with respect to what is readable, but it is quantised
  to the bf16 grid: an update below ~2^-9 of the weight magnitude is invisible
  and reads as ``r = 0``. A "sleeping expert" verdict therefore means "not moving
  in the representable weights over ``delta`` steps", which is what a longer
  ``sample_interval`` buys. Reading the optimizer's fp32 master weights instead
  would lift this, at the price of reaching into the optimizer.
* ``_mean``, ``_min`` and ``_max`` reduce correctly across EP ranks (each rank
  holds its own expert shard, and the expert count divides evenly across the
  group). ``_median`` / ``_p10`` / ``_p90`` do not: what lands in the log is the
  mean of the per-rank quantiles, not the global quantile. That is accepted here
  as a spread signal, the same trade the megatron-side spectral entropy makes.

A small update is not by itself a functional verdict: an expert routed few tokens
has little to learn from. Read these against the routed-token distribution that
``moe_health`` already reports (``assignment_load_cv``, ``assignment_load_min_frac``,
``assignment_load_max_min_ratio``, ``gate_mass_*``) rather than duplicating it
here. A true functional drift measure -- expert output displacement on a fixed
probe batch -- needs the old weights kept live and an extra expert forward, so it
is out of scope for a weights-only probe.
"""

import logging

import paddle

from .base import PaddleProbe
from .layer_discovery import get_decoder_layers, iter_monitor_layers
from .moe_monitor import (
    _expert_fc1_weight,
    _singular_value_entropy,
    _swiglu_gate_half,
)

logger = logging.getLogger(__name__)

# Expert projections monitored, in the order they appear in the MLP.
MATRIX_NAMES = ("gate", "up", "down")
# Per-layer summaries of a per-expert statistic.
_SPREAD_KEYS = ("mean", "median", "p10", "p90", "min", "max")
# Reduced summary set for the spectral statistics.
_RANK_KEYS = ("mean", "min", "max")
# Power-iteration steps for sigma_1. 30 keeps a rank-dominated update exact and a
# flat one inside ~1.2%; see the module docstring for the measured table.
_POWER_ITERS = 30
_EPS = 1e-12


def metric_names(
    log_spectrum: bool = False, with_shared: bool = True
) -> tuple[str, ...]:
    """Full per-layer metric schema, in declaration order.

    ``log_spectrum`` adds the singular-entropy keys, which are the only ones that
    need a full eigensolve. ``with_shared`` is off for a layer that has no shared
    expert, so it does not declare a key it will never record.
    """
    names: list[str] = [f"update_zmax_{key}" for key in ("max", "p90", "min")]
    for matrix in MATRIX_NAMES:
        names += [f"{matrix}_rel_update_{key}" for key in _SPREAD_KEYS]
        names.append(f"{matrix}_delta_norm_mean")
        names += [f"{matrix}_stable_rank_{key}" for key in _RANK_KEYS]
        if log_spectrum:
            names += [f"{matrix}_singular_entropy_{key}" for key in _RANK_KEYS]
        if with_shared:
            names.append(f"shared_{matrix}_rel_update")
    return tuple(names)


def _swiglu_up_half(fc1_weight):
    """The ungated "up" half of a fused SwiGLU fc1 weight.

    Counterpart of :func:`moe_monitor._swiglu_gate_half`: paddle lays fc1 out as
    ``[..., in, 2 * inter]`` and ``glu()`` applies SiLU to the first chunk, so the
    linear half is the second one.
    """
    return fc1_weight[..., fc1_weight.shape[-1] // 2 :]


def _expert_fc2_weight(moe_layer):
    """Routed-expert fc2 (down) weight with a leading expert dim, or None.

    Mirrors the layouts :func:`moe_monitor._expert_fc1_weight` handles:
    grouped-gemm ``weight2``, a fused ``down_proj.weight``, and the non-fused
    ``LayerList`` of per-expert MLPs (stacked here).
    """
    ggm = getattr(moe_layer, "grouped_gemm_experts", None)
    if ggm is not None and hasattr(ggm, "weight2"):
        return ggm.weight2
    experts = getattr(moe_layer, "experts", None)
    if experts is None:
        return None
    if hasattr(experts, "down_proj"):
        return experts.down_proj.weight
    per_expert = [
        e.down_proj.weight
        for e in experts
        if e is not None and hasattr(e, "down_proj")
    ]
    return paddle.stack(per_expert) if per_expert else None


def _shared_weights(moe_layer):
    """``(fc1, fc2)`` of the shared expert, either entry possibly None."""
    shared = getattr(moe_layer, "shared_experts", None)
    if shared is None:
        return None, None
    fc1 = getattr(getattr(shared, "up_gate_proj", None), "weight", None)
    fc2 = getattr(getattr(shared, "down_proj", None), "weight", None)
    return fc1, fc2


def _as_stack(weight):
    """Detached float32 view of a weight with a guaranteed leading expert axis.

    A fused grouped-gemm weight already carries ``[E, in, out]``; the shared
    expert is a plain ``[in, out]`` matrix and gets a length-1 axis so both go
    through the same batched code path.
    """
    matrices = weight.detach().astype("float32")
    return matrices if len(matrices.shape) > 2 else matrices.unsqueeze(0)


def _frobenius(matrices):
    """``||.||_F`` of every matrix in a ``[..., m, n]`` stack -> ``[...]``."""
    return paddle.sqrt((matrices * matrices).sum(axis=[-2, -1]))


def _deterministic_start(width: int):
    """Unit start vector for the power iteration, drawn without the global RNG.

    ``paddle.randn`` would consume the global generator and shift every
    downstream draw (dropout, router noise) on monitored steps only, which would
    make the run irreproducible against an unmonitored one. A fixed irrational
    stride gives a vector with no particular relation to the weight layout, which
    is all the iteration needs.
    """
    index = paddle.arange(width, dtype="float32")
    vector = paddle.sin(index * 0.7071067811865476 + 1.0).reshape([width, 1])
    return vector / paddle.linalg.norm(vector).clip(min=_EPS)


def _sigma_max(matrices, iters: int = _POWER_ITERS):
    """Largest singular value of every matrix in a ``[E, m, n]`` stack -> ``[E]``.

    Power iteration on ``A^T A``, which needs only matrix-vector products: at
    these shapes it is ~45x cheaper than the eigensolve that would give the whole
    spectrum, and the stable rank needs nothing but ``sigma_1``. Converges as
    ``(sigma_2 / sigma_1)^(2 * iters)``; error is one-sided (low), so the stable
    rank it feeds is biased high.
    """
    vector = paddle.broadcast_to(
        _deterministic_start(matrices.shape[-1]).unsqueeze(0),
        [matrices.shape[0], matrices.shape[-1], 1],
    )
    for _ in range(iters):
        vector = paddle.matmul(
            matrices, paddle.matmul(matrices, vector), transpose_x=True
        )
        vector = vector / paddle.linalg.norm(
            vector, axis=-2, keepdim=True
        ).clip(min=_EPS)
    return paddle.linalg.norm(paddle.matmul(matrices, vector), axis=-2).squeeze(
        -1
    )


def _spread(values):
    """The six per-layer summaries of a ``[E]`` per-expert statistic.

    ``_mean`` / ``_min`` / ``_max`` survive the cross-rank reduction exactly; the
    quantiles do not (see the module docstring).
    """
    return {
        "mean": values.mean(),
        "median": paddle.quantile(values, 0.5),
        "p10": paddle.quantile(values, 0.10),
        "p90": paddle.quantile(values, 0.90),
        "min": values.min(),
        "max": values.max(),
    }


def _gram(matrices):
    """Smaller-side Gram matrix of a ``[E, m, n]`` stack, for the eigensolve."""
    if matrices.shape[-2] <= matrices.shape[-1]:
        return paddle.matmul(matrices, matrices, transpose_y=True)
    return paddle.matmul(matrices, matrices, transpose_x=True)


def _find_moe_layers(model, pp_rank, mark_mtp):
    """``[(layer_idx, moe_module)]`` for every MoE layer visible on this rank.

    Same discovery contract as the MoE health monitor: the expert block hangs off
    ``layer.mlp`` on this model family, off ``layer.moe`` elsewhere, and a bare
    ``MoELayer`` is its own module.
    """

    def has_moe(layer):
        return (
            (hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"))
            or hasattr(layer, "moe")
            or hasattr(layer, "gate")
        )

    layers = get_decoder_layers(model)
    if layers is None:
        layers = [
            m
            for _name, m in model.named_sublayers()
            if m.__class__.__name__ == "MoELayer"
        ] or None
    if layers is None:
        return []

    monitor_layers = iter_monitor_layers(layers, has_moe, pp_rank=pp_rank)
    mark_mtp(item.idx for item in monitor_layers if item.is_mtp)
    found = []
    for item in monitor_layers:
        layer = item.layer
        module = None
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
            module = layer.mlp
        elif hasattr(layer, "moe"):
            module = layer.moe
        elif hasattr(layer, "gate"):
            module = layer
        if module is not None:
            found.append((item.idx, module))
    return found


class PaddleMLPUpdateMonitor(PaddleProbe):
    """Per-layer distribution of the expert MLP parameter increment.

    Weights only -- no forward hooks. The base point is the previous monitored
    reading, so the increment spans ``sample_interval`` steps (falling back to the
    shared ``monitor_interval``).

    Snapshots are taken from :meth:`collect_expert_norms`, which the trainer
    callback calls at step begin. That name is the callback's contract, not a
    description: it is the only step-begin slot, and it has to run there because
    ``FP8QuantWeightCallback`` clears the bf16 expert weights later in the same
    step. Reading them at step end would see the cleared storage.
    """

    METRIC_PREFIX = "mlp_update"

    def __init__(
        self,
        log_per_layer=True,
        log_global=True,
        monitor_interval=1,
        verbose=False,
        sample_interval=None,
        log_spectrum=False,
        spectrum_interval=1,
    ):
        interval = (
            monitor_interval
            if sample_interval is None
            else int(sample_interval)
        )
        if interval < 1:
            raise ValueError(
                f"sample_interval must be >= 1, got {sample_interval}"
            )
        if int(spectrum_interval) < 1:
            raise ValueError(
                f"spectrum_interval must be >= 1, got {spectrum_interval}"
            )
        super().__init__(
            log_per_layer=log_per_layer,
            log_global=log_global,
            monitor_interval=interval,
            verbose=verbose,
        )
        self.log_spectrum = bool(log_spectrum)
        # The eigensolve is ~45x the cost of the rest, so it rides a coarser clock:
        # the relative updates land on every sample, the spectrum every
        # ``spectrum_interval``-th one.
        self.spectrum_interval = int(spectrum_interval)
        self._samples = 0
        self._spectrum_due = False
        self._metric_names = metric_names(self.log_spectrum)
        self.MAX_AGGREGATED = {
            name for name in self._metric_names if name.endswith("_max")
        }
        self.MIN_AGGREGATED = {
            name for name in self._metric_names if name.endswith("_min")
        }
        # [(layer_idx, moe_module)] and layer_idx -> {"routed": (fc1, fc2), "shared": (fc1, fc2)}
        self._layers = []
        self._snapshots = {}

    def register_hooks(self, model):
        try:
            from paddlefleet.parallel_state import (
                get_pipeline_model_parallel_rank,
            )

            self.pp_rank = get_pipeline_model_parallel_rank()
        except Exception:
            pass

        self._layers = _find_moe_layers(
            model, self.pp_rank, self.mark_mtp_layers
        )
        if not self._layers:
            logger.warning("[PaddleMLPUpdateMonitor] No MoE layers found!")
            return

        for layer_idx, module in self._layers:
            shared_fc1, shared_fc2 = _shared_weights(module)
            has_shared = shared_fc1 is not None or shared_fc2 is not None
            for name in metric_names(self.log_spectrum, with_shared=has_shared):
                self.declare_layer_metric(layer_idx, name)
        self.allocate_buffers()

        logger.info(
            "[PaddleMLPUpdateMonitor] Tracking %s on %d MoE layers, delta=%d steps, spectrum=%s.",
            "/".join(MATRIX_NAMES),
            len(self._layers),
            self.monitor_interval,
            f"{self.log_spectrum} (every {self.spectrum_interval} samples)",
        )

    def _live_weights(self, module):
        """``{"routed": (fc1, fc2), "shared": (fc1, fc2)}`` of live parameters."""
        return {
            "routed": (_expert_fc1_weight(module), _expert_fc2_weight(module)),
            "shared": _shared_weights(module),
        }

    @staticmethod
    def _snapshot(group):
        """Clone a weight pair in its own dtype.

        Kept at the parameter's dtype rather than upcast: the increment is exact
        with respect to what is readable either way, and fp32 would double an
        already ~50 MB-per-layer resident cost for no extra resolution.
        """
        return tuple(
            None if weight is None else weight.detach().clone()
            for weight in group
        )

    @staticmethod
    def _matrix_deltas(live, base):
        """``[(name, delta, weight)]`` for gate / up / down on one expert group.

        Skips a pair whose shape moved between readings (resume, reshard) rather
        than differencing mismatched storage; the fresh snapshot rebases it.
        """
        pairs = []
        fc1_now, fc2_now = live
        fc1_base, fc2_base = base
        if (
            fc1_now is not None
            and fc1_base is not None
            and list(fc1_now.shape) == list(fc1_base.shape)
        ):
            now = _as_stack(fc1_now)
            delta = now - _as_stack(fc1_base)
            pairs.append(
                ("gate", _swiglu_gate_half(delta), _swiglu_gate_half(now))
            )
            pairs.append(("up", _swiglu_up_half(delta), _swiglu_up_half(now)))
        if (
            fc2_now is not None
            and fc2_base is not None
            and list(fc2_now.shape) == list(fc2_base.shape)
        ):
            now = _as_stack(fc2_now)
            pairs.append(("down", now - _as_stack(fc2_base), now))
        return pairs

    def _record_routed(self, layer_idx, name, delta, weight):
        """Record one projection's statistics; returns its per-expert ``r`` vector."""
        delta_norm = _frobenius(delta)
        relative = delta_norm / (_frobenius(weight) + _EPS)
        for key, value in _spread(relative).items():
            self.record_layer_metric(
                layer_idx, f"{name}_rel_update_{key}", value
            )
        self.record_layer_metric(
            layer_idx, f"{name}_delta_norm_mean", delta_norm.mean()
        )

        # A frozen expert has dW = 0, where the ratio is 0/0. Report the lower
        # bound 1 so the series stays inside [1, k] instead of planting a 0 that
        # reads as total collapse and wins every _min reduction; the sleeping
        # expert is already unambiguous in rel_update_min / _p10.
        sigma_max = _sigma_max(delta)
        stable_rank = paddle.where(
            delta_norm > 0.0,
            (delta_norm * delta_norm) / (sigma_max * sigma_max).clip(min=_EPS),
            paddle.ones_like(delta_norm),
        )
        spread = _spread(stable_rank)
        for key in _RANK_KEYS:
            self.record_layer_metric(
                layer_idx, f"{name}_stable_rank_{key}", spread[key]
            )

        if self._spectrum_due:
            sigma = paddle.sqrt(
                paddle.linalg.eigvalsh(_gram(delta)).clip(min=0.0)
            )
            spread = _spread(_singular_value_entropy(sigma))
            for key in _RANK_KEYS:
                self.record_layer_metric(
                    layer_idx, f"{name}_singular_entropy_{key}", spread[key]
                )
        return relative

    def _record_expert_score(self, layer_idx, relatives):
        """Expert-level summary ``S[e] = max_m z(r_m[e])`` over the layer's experts.

        Averaging the three projections would let one anomalous matrix be diluted
        by two healthy ones, and averaging raw norms would let whichever matrix
        has the largest scale set the verdict. Standardising each projection over
        the layer's experts first puts them on one scale; the max over ``m`` then
        keeps a single misbehaving projection visible.

        ``_max`` is the layer's most anomalous expert. ``_min`` is the expert that
        sits furthest below its peers on *every* projection at once, which is the
        sharpest "sleeping expert" reading available here -- a per-matrix
        ``rel_update_min`` can be low for one matrix by chance.

        The standardisation is over the experts this rank holds, not the global
        expert set, so under EP the score is relative to the local shard.
        """
        scores = None
        for relative in relatives:
            centred = relative - relative.mean()
            z = (
                centred / (relative.std() + _EPS)
                if relative.shape[0] > 1
                else paddle.zeros_like(centred)
            )
            scores = z if scores is None else paddle.maximum(scores, z)
        if scores is None:
            return
        self.record_layer_metric(layer_idx, "update_zmax_max", scores.max())
        self.record_layer_metric(
            layer_idx, "update_zmax_p90", paddle.quantile(scores, 0.90)
        )
        self.record_layer_metric(layer_idx, "update_zmax_min", scores.min())

    def _process_layer(self, layer_idx, module):
        live = self._live_weights(module)
        previous = self._snapshots.get(layer_idx)
        self._snapshots[layer_idx] = {
            key: self._snapshot(group) for key, group in live.items()
        }
        if previous is None:
            return
        relatives = [
            self._record_routed(layer_idx, name, delta, weight)
            for name, delta, weight in self._matrix_deltas(
                live["routed"], previous["routed"]
            )
        ]
        self._record_expert_score(layer_idx, relatives)
        for name, delta, weight in self._matrix_deltas(
            live["shared"], previous["shared"]
        ):
            relative = _frobenius(delta) / (_frobenius(weight) + _EPS)
            self.record_layer_metric(
                layer_idx, f"shared_{name}_rel_update", relative.mean()
            )

    def collect_expert_norms(self):
        """Step-begin entry point; see the class docstring for why it is this name."""
        if not self._buffers_allocated or not self._should_monitor():
            return
        measuring = bool(self._snapshots)
        self._spectrum_due = (
            self.log_spectrum
            and measuring
            and self._samples % self.spectrum_interval == 0
        )
        for layer_idx, module in self._layers:
            try:
                with paddle.no_grad():
                    self._process_layer(layer_idx, module)
            except Exception as exc:
                if self.verbose:
                    logger.error(
                        "[PaddleMLPUpdateMonitor] layer %s failed: %s",
                        layer_idx,
                        exc,
                    )
        # Counted only once a base point exists, so the first measurement -- not the
        # snapshot-only pass that precedes it -- is spectrum sample zero.
        if measuring:
            self._samples += 1

    def remove_hooks(self):
        super().remove_hooks()
        self._layers = []
        # The base snapshots are the monitor's whole resident cost; drop them.
        self._snapshots = {}


def setup_mlp_update_monitor(
    model,
    log_per_layer=True,
    log_global=True,
    monitor_interval=1,
    verbose=False,
    monitor_dict=None,
    sample_interval=None,
    log_spectrum=False,
    spectrum_interval=1,
):
    monitor = PaddleMLPUpdateMonitor(
        log_per_layer=log_per_layer,
        log_global=log_global,
        monitor_interval=monitor_interval,
        verbose=verbose,
        sample_interval=sample_interval,
        log_spectrum=log_spectrum,
        spectrum_interval=spectrum_interval,
    )
    monitor.register_hooks(model)
    logger.info(
        "[PaddleMLPUpdateMonitor] Setup complete on %d MoE layers.",
        len(monitor._layers),
    )
    if monitor_dict is not None:
        monitor_dict["mlp_update"] = monitor
    return model
