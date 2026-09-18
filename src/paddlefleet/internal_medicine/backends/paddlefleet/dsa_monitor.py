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

"""DSA (DeepSeek Sparse Attention) health monitor for PaddleFleet.

Answers three questions about a DSA layer, in increasing cost:

1. **Is the indexer healthy?** ``index_*`` — magnitudes of the three indexer
   projections, plus the sign balance of the per-head importance weights.
2. **What is it selecting?** ``score_*`` / ``select_*`` — separation between kept
   and dropped scores, realised sparsity, look-back distance, and whether the
   selection has degenerated into a sliding window.
3. **Does the selection match the trunk attention?** ``attn_*`` — the dense
   attention mass that survives the sparse mask (``attn_recall``), the
   ``KL(p_sparse || p_dense)`` that is its per-head logarithm, and whether the
   single largest logit is being dropped.
4. **Do layers select the same keys?** ``select_iou_prev`` — the cross-layer
   overlap that decides whether a shared index is worth trying.

Collection is one instance patch plus one forward post hook per layer. The patch
is on ``DSAIndexer.forward_before_topk`` because that is the only place the
indexer's ``q`` / ``k`` / ``weights`` exist: on the training path
``DSAttention.forward`` feeds them straight to the fused loss kernel and
``DSAIndexer.forward`` never runs, so a forward hook on the indexer would never
fire. The post hook on the ``DSAttention`` module then pairs those with the
trunk ``query`` / ``key`` from its own positional inputs.

Everything is measured on a subsample of query rows and attention heads --
reproducing the indexer scoring or the dense attention at full size would cost
as much as the layer. Hot-path discipline: see
``.claude/skills/monitor-hook-perf-rules``.
"""

from __future__ import annotations

import logging

import paddle
from paddle import nn

from . import dsa_metrics
from .base import PaddleProbe
from .layer_discovery import (
    get_decoder_layers,
    get_dsa_core,
    is_dsa_layer,
    iter_monitor_layers,
)

logger = logging.getLogger(__name__)


class _MethodPatch:
    """Handle for a monkey-patched bound method, shaped like a paddle hook."""

    def __init__(self, module, name: str, original):
        self._module = module
        self._name = name
        self._original = original

    def remove(self) -> None:
        setattr(self._module, self._name, self._original)


class PaddleDSAHealthMonitor(PaddleProbe):
    """Monitor the indexer, the selection, and its agreement with dense attention."""

    METRIC_PREFIX = "dsa_health"
    MAX_AGGREGATED = set(dsa_metrics.MAX_METRICS)
    MIN_AGGREGATED = set(dsa_metrics.MIN_METRICS)

    def __init__(
        self,
        log_per_layer: bool = True,
        log_global: bool = True,
        monitor_interval: int = 1,
        verbose: bool = False,
        exclude_families=None,
        row_samples: int = 32,
        head_samples: int = 8,
        local_window: int = 128,
        sample_layers: list[int] | None = None,
    ):
        super().__init__(
            log_per_layer=log_per_layer,
            log_global=log_global,
            monitor_interval=monitor_interval,
            verbose=verbose,
            exclude_families=exclude_families,
        )
        if int(row_samples) < 1:
            raise ValueError(
                f"[PaddleDSAMonitor] row_samples must be >= 1, got {row_samples}"
            )
        if int(head_samples) < 1:
            raise ValueError(
                f"[PaddleDSAMonitor] head_samples must be >= 1, got {head_samples}"
            )
        if int(local_window) < 1:
            raise ValueError(
                f"[PaddleDSAMonitor] local_window must be >= 1, got {local_window}"
            )
        self.row_samples = int(row_samples)
        self.head_samples = int(head_samples)
        self.local_window = int(local_window)
        self.sample_layers = set(sample_layers) if sample_layers else None
        self.cp_size = 1
        self._match_enabled = True
        self._stash: dict[int, tuple] = {}
        self._prev_selection: tuple[int, paddle.Tensor] | None = None
        self._failed_layers: set[int] = set()

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
        try:
            from paddlefleet.process_groups_config import ProcessGroupCollection
            from paddlefleet.utils import get_pg_size

            self.cp_size = get_pg_size(
                ProcessGroupCollection.use_mpu_process_groups(
                    required_pgs=["cp"]
                ).cp
            )
        except Exception:
            pass

    def _find_targets(self, model):
        layers = get_decoder_layers(model)
        if not layers:
            return []
        monitor_layers = iter_monitor_layers(
            layers, is_dsa_layer, pp_rank=self.pp_rank
        )
        mtp_layer_ids = [item.idx for item in monitor_layers if item.is_mtp]
        if mtp_layer_ids:
            self.mark_mtp_layers(mtp_layer_ids)
        targets = []
        for item in monitor_layers:
            if self.sample_layers and item.idx not in self.sample_layers:
                continue
            targets.append((item.idx, get_dsa_core(item.layer), item.attn_type))
        return targets

    def register_hooks(self, model: nn.Layer):
        self._init_parallel_state()
        targets = self._find_targets(model)
        if not targets:
            logger.info("[PaddleDSAMonitor] No DSA layers found; skipping.")
            return

        # ``match`` compares against the trunk attention, whose key is this
        # rank's sequence slice while the indexer key arrives all-gathered. There
        # is no way to build the global logits without gathering the trunk key
        # too, so under CP that family is dropped rather than reported wrong.
        # Everything else works off the indexer outputs alone and only needs the
        # global row ids, which ``position_offset`` supplies.
        self._match_enabled = self.cp_size <= 1
        if not self._match_enabled:
            logger.warning(
                "[PaddleDSAMonitor] CP=%d: dropping the 'match' family (trunk key "
                "is a sequence slice); indexer / score / select / crosslayer stay on.",
                self.cp_size,
            )

        skipped = () if self._match_enabled else dsa_metrics.MATCH_METRICS
        for layer_idx, _core, attn_type in targets:
            for name in dsa_metrics.ALL_METRICS:
                if name in skipped:
                    continue
                self.declare_layer_metric(layer_idx, name, attn_type=attn_type)

        self.allocate_buffers()

        for layer_idx, core, attn_type in targets:
            self.hooks.append(self._patch_indexer(layer_idx, core))
            self.hooks.append(
                core.register_forward_post_hook(
                    self._make_hook(layer_idx, core, attn_type)
                )
            )

        logger.info(
            f"[PaddleDSAMonitor] Registered {len(self.hooks)} hooks on {len(targets)} DSA layers "
            f"(rows={self.row_samples} heads={self.head_samples})."
        )

    def _patch_indexer(self, layer_idx: int, core):
        """Capture ``forward_before_topk``'s return and its ``position_offset``.

        The offset is a plain int the caller already computed
        (``MQALatentAttention._indexer_projections`` passes it positionally); it
        is the only thing that turns this rank's local row ids into the global
        positions the all-gathered indexer key is indexed by. Reading it here
        rather than re-deriving it from the CP rank keeps the monitor agnostic to
        ``cp_balance_mode``.
        """
        indexer = core.indexer
        original = indexer.forward_before_topk
        monitor = self

        def wrapped(*args, **kwargs):
            out = original(*args, **kwargs)
            if monitor._should_monitor():
                offset = kwargs.get("position_offset")
                if offset is None and len(args) >= 3:
                    offset = args[2]
                monitor._stash[layer_idx] = (out, int(offset or 0))
            return out

        indexer.forward_before_topk = wrapped
        return _MethodPatch(indexer, "forward_before_topk", original)

    # ------------------------------------------------------------------
    # Hooks (the hot path)
    # ------------------------------------------------------------------

    def _log_failure(self, layer_idx: int, exc: Exception) -> None:
        if layer_idx not in self._failed_layers:
            logger.error(
                f"[PaddleDSAMonitor] Error at layer {layer_idx}: {exc}"
            )
            self._failed_layers.add(layer_idx)

    def _make_hook(self, layer_idx: int, core, attn_type):
        def hook(_layer, inputs, _output):
            if not self._should_monitor():
                return None
            stashed = self._stash.pop(layer_idx, None)
            if stashed is None or len(inputs) < 2:
                return None
            try:
                with paddle.no_grad():
                    self._record_layer(
                        layer_idx,
                        core,
                        attn_type,
                        stashed,
                        inputs[0],
                        inputs[1],
                    )
            except Exception as exc:
                self._log_failure(layer_idx, exc)
            return None

        return hook

    def _record(self, layer_idx: int, attn_type, stats: dict) -> None:
        for name, value in stats.items():
            self.record_layer_metric(
                layer_idx, name, value, attn_type=attn_type
            )

    def _record_layer(
        self, layer_idx, core, attn_type, stashed, query, key
    ) -> None:
        (q_idx, k_idx, weights), position_offset = stashed
        self._record(
            layer_idx,
            attn_type,
            dsa_metrics.indexer_projection_stats(q_idx, k_idx, weights),
        )

        index_topk = int(core.indexer.index_topk)
        # ``rows`` index this rank's slice; ``rows + position_offset`` are the
        # global positions the all-gathered indexer key is laid out along. Under
        # CP=1 the offset is 0 and the two coincide.
        rows = dsa_metrics.sample_query_rows(
            q_idx.shape[1],
            index_topk,
            self.row_samples,
            position_offset=position_offset,
        )
        global_rows = rows + position_offset
        causal, dist = dsa_metrics.causal_geometry(global_rows, k_idx.shape[1])
        index_scores = dsa_metrics.index_scores_on_rows(
            q_idx, k_idx, weights, rows
        )
        sel = dsa_metrics.selection_mask(index_scores, causal, index_topk)

        self._record(
            layer_idx,
            attn_type,
            dsa_metrics.score_stats(index_scores, sel, causal),
        )
        self._record(
            layer_idx,
            attn_type,
            dsa_metrics.selection_stats(sel, causal, dist, self.local_window),
        )
        self._record_cross_layer(layer_idx, attn_type, sel)

        if not self._match_enabled:
            return
        heads = dsa_metrics.align_and_sample_heads(
            paddle.index_select(query, rows, axis=1), key, self.head_samples
        )
        if heads is None:
            return
        q_rows, key_sub = heads
        self._record(
            layer_idx,
            attn_type,
            dsa_metrics.dense_match_stats(
                q_rows, key_sub, sel, causal, core.softmax_scale
            ),
        )

    def _record_cross_layer(
        self, layer_idx: int, attn_type, sel: paddle.Tensor
    ) -> None:
        """Compare against the previously hooked layer of the same forward pass.

        ``layer_idx`` not increasing means a new pass started, so the cached mask
        belongs to the last layer of the previous microbatch and comparing
        against it would measure microbatch-to-microbatch drift instead of
        layer-to-layer agreement.
        """
        previous = self._prev_selection
        self._prev_selection = (layer_idx, sel)
        if (
            previous is None
            or previous[0] >= layer_idx
            or previous[1].shape != sel.shape
        ):
            return
        self._record(
            layer_idx, attn_type, dsa_metrics.selection_iou(sel, previous[1])
        )

    def _log_record_counts(self) -> None:
        """Log how many times each per-layer key was recorded this step.

        A mean metric divides by its own count, so recording the same microbatch
        twice leaves the *value* unchanged and only the count betrays it. That is
        not a hypothetical here: under ``recompute_granularity: full`` the layer
        runs twice, and ``MQALatentAttention._indexer_projections`` wraps the
        indexer call in an explicit ``paddle.enable_grad()`` (to keep the
        projections in the graph under ``dsa_indexer_loss_bwd_p2p_overlap``), so
        the grad-enabled guard in ``_should_monitor`` may not filter the no-grad
        pass out. The expected count is the number of microbatches per step.

        Verbose-only, and a plain python dict read — no D2H, nothing on the hot
        path. Called before ``super().step()`` because the flush resets counts.
        """
        per_metric: dict[str, set] = {}
        for key, count in self._gpu_cnt.items():
            if count == 0 or key not in self._layer_metric_keys:
                continue
            per_metric.setdefault(key.rsplit("/", 1)[-1], set()).add(count)
        if not per_metric:
            return
        distinct = sorted({c for counts in per_metric.values() for c in counts})
        logger.info(
            "[PaddleDSAMonitor] step=%d record counts=%s (expected: microbatches per step)",
            self.step_count,
            distinct,
        )
        if len(distinct) > 1:
            # Uneven counts mean some metric saw a different number of passes
            # than its siblings — worth the full dump rather than a summary.
            logger.info(
                "[PaddleDSAMonitor] uneven record counts per metric: %s",
                {
                    name: sorted(counts)
                    for name, counts in sorted(per_metric.items())
                },
            )

    def step(self):
        if self.verbose:
            self._log_record_counts()
        # A stash entry left behind means a patched indexer ran without its
        # module's post hook (a partial forward, or a layer skipped mid-pass);
        # carrying it into the next step would pair it with the wrong query.
        self._stash.clear()
        self._prev_selection = None
        super().step()


def setup_dsa_monitor(
    model,
    log_per_layer: bool = True,
    log_global: bool = True,
    monitor_interval: int = 1,
    verbose: bool = False,
    exclude_families=None,
    row_samples: int = 32,
    head_samples: int = 8,
    local_window: int = 128,
    sample_layers: list[int] | None = None,
    monitor_dict: dict | None = None,
):
    monitor = PaddleDSAHealthMonitor(
        log_per_layer=log_per_layer,
        log_global=log_global,
        monitor_interval=monitor_interval,
        verbose=verbose,
        exclude_families=exclude_families,
        row_samples=row_samples,
        head_samples=head_samples,
        local_window=local_window,
        sample_layers=sample_layers,
    )
    monitor.register_hooks(model)
    if monitor_dict is not None:
        monitor_dict["dsa_health"] = monitor
    return model
