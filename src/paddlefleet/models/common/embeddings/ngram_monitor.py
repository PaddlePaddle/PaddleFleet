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

"""Usage / collision monitor for N-gram Embedding tables.

Design notes (why these metrics and not the obvious ones):

* Cumulative "fraction of rows ever touched" saturates to ~1.0 within a few
  hundred steps (M = 2.4M rows vs. ~7.8M lookups per global step), so it is
  reported but is NOT the primary utilization signal.  The primary signals are
  the *windowed* hit ratio, the *effective* row count exp(H(p)) and the
  *staleness* distribution (steps since a row was last updated), which stay
  informative for the whole run.

* Two questions are deliberately kept apart.  "Is the hash function good?" is
  answered by the collision block, which normalises against an ideal uniform
  hash over *distinct* keys and is therefore independent of the data
  distribution.  "Is the load on the buckets even?" is answered by the usage
  block -- entropy / effective rows, Gini and the index of dispersion
  Var/Mean -- which deliberately *includes* the Zipf skew of natural text,
  because that skew is what actually decides whether a row ever gets enough
  gradient signal to learn anything.  Both views are reported per sub-table and
  for the two time horizons (current window, whole run).

* Raw in-batch collision rate is dominated by the sample size (birthday
  problem: E[collisions] ~ U^2 / 2M).  We therefore always report the ratio to
  the ideal-uniform-hash expectation ("lift"), which is sample-size invariant
  and measures hash quality, and we make the sequence-scoped variant the
  primary metric because its statistical population is fixed at S tokens.

* "All sub-tables collide" is the metric that validates the K-split design:
  under independent hash families its expectation is ~0, so any persistent
  non-zero value proves the splits are correlated and K is not buying the
  collision reduction it is supposed to buy.

* The full per-row hit-count vector is the primitive from which every usage
  scalar above is derived, so it is both summarised online (histogram over
  count buckets, count quantiles, Lorenz mass shares, Gini) and persisted in
  full into every checkpoint.  The summary is computed *after* the cross-rank
  reduction of the counters, hence needs no extra collective and is identical
  on every rank; the persisted vector is what makes the cumulative half of it
  survive a warm restart.

* Anything that is an exact algebraic function of another reported number is
  not reported: ``never_hit_ratio`` is ``1 - cum_hit_ratio``, the head-mass of
  the busiest 1% of rows is ``cum_mass_top01``, and the per-table excess rate
  carries no information beyond ``collide_key_rate`` plus the lift.
"""

from __future__ import annotations

import math

import numpy as np
import paddle
import paddle.distributed as dist

# Bucket edges of the per-row hit-count histogram: bucket i covers
# [_HIST_EDGES[i], _HIST_EDGES[i + 1]).  A power-of-4 ladder keeps the whole
# distribution inside a dozen scalars even when a single row is hit 1e8 times
# (at ratio 12.2 the mean row gets ~3e2 hits per 100-step window and ~3e5 hits
# over a full run, so both ends of the ladder are actually used).
_HIST_EDGES = (0, 1, 2, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 1 << 20)
_HIST_NAMES = (
    "n0",
    "n1",
    "n2_3",
    "n4_15",
    "n16_63",
    "n64_255",
    "n256_1k",
    "n1k_4k",
    "n4k_16k",
    "n16k_64k",
    "n64k_1m",
    "n1m_up",
)
# Quantiles of the per-row count, and the top-row fractions whose share of all
# lookups is reported (a three-point Lorenz curve).
_QUANTILES = (0.5, 0.9, 0.99, 0.999)
_QUANTILE_NAMES = ("p50", "p90", "p99", "p999")
_TOP_FRACS = (0.001, 0.01, 0.1)
_TOP_NAMES = ("top001", "top01", "top10")
_DIST_WIDTH = len(_HIST_EDGES) + len(_QUANTILES) + len(_TOP_FRACS) + 1

# Layout of the per-sub-table scalar block that is stacked once per analysis
# step and copied to the host in a single D2H.  ``win_`` refers to the current
# window, ``cum_`` to the whole run; ``_sq`` is sum of squared row counts, which
# yields the index of dispersion Var/Mean without a second pass.
_BASE_NAMES = (
    "win_lookups",
    "win_hit",
    "win_entropy",
    "win_max",
    "win_sq",
    "stale",
    "cum_lookups",
    "cum_hit",
    "cum_entropy",
    "cum_sq",
)
_BASE_WIDTH = len(_BASE_NAMES)


def _ideal_excess(n_keys: int, n_buckets: int) -> float:
    """Expected number of *excess* distinct keys under an ideal uniform hash.

    ``n_keys`` distinct keys thrown into ``n_buckets`` buckets occupy
    ``n_buckets * (1 - (1 - 1/n_buckets) ** n_keys)`` buckets in expectation, so
    ``n_keys - occupied`` of them are, in expectation, "excess": they land in a
    bucket some other distinct key already owns.  Written with ``expm1`` /
    ``log1p`` because ``1 / n_buckets`` is ~4e-7 here.
    """
    if n_keys <= 1 or n_buckets <= 1:
        return 0.0
    occupied = n_buckets * (-math.expm1(n_keys * math.log1p(-1.0 / n_buckets)))
    return max(float(n_keys) - occupied, 0.0)


def _resolve_reduce_group(mode: str):
    """Group over which per-rank observations must be summed.

    Returning ``None`` means "the default (world) group".  That is the correct
    choice whenever TP == PP == CP == 1, because then every rank -- including
    sharding and expert-parallel ranks -- holds a *distinct* slice of the global
    batch, and none of them holds a replica of another rank's tokens.
    """
    from paddlefleet import parallel_state as ps

    if mode == "world":
        return None
    if mode == "auto":
        try:
            replicated_input = (
                ps.get_tensor_model_parallel_world_size() == 1
                and ps.get_pipeline_model_parallel_world_size() == 1
                and ps.get_context_parallel_world_size() == 1
            )
        except Exception:
            replicated_input = True
        if replicated_input:
            return None
    return ps.get_data_parallel_group(
        check_initialized=False, with_context_parallel=True
    )


class NgramUsageMonitor:
    """Collects usage / collision statistics for the N-gram embedding tables.

    Cost model.  Per training step the monitor runs exactly two extra GPU ops on
    the already-computed bucket ids (one ``concat``, one ``bincount``) plus one
    in-place add.  Everything expensive (cross-rank reduction, entropy, collision
    analysis) runs once every ``ngram_monitor_interval`` steps.

    Statistical scopes.  ``batch`` scope pools all valid positions of one rank's
    micro-batch; ``seq`` scope pools the positions of a single sequence (keys and
    buckets are offset by the sequence index so that different sequences cannot
    collide with each other).  ``seq`` is the primary scope because its
    population size is fixed at S for the whole run, which makes the number
    comparable across steps and across configurations.

    This is deliberately a plain object rather than an ``nn.Layer``: the counters
    are integer tensors that must never be registered as sub-layer buffers (they
    would otherwise be visible to ``Layer.to()`` dtype casting, to the pipeline
    name mapping and to the checkpoint writer).  They are persisted separately by
    ``NgramMonitorCallback``.

    Args:
        config: transformer config carrying the ``ngram_monitor_*`` fields.
        table_sizes: row count (the modulus) of every sub-table, in the same flat
            order as ``NgramEmbedding.embedders``.
        table_orders: N-gram order of every sub-table, same order as above.
        vocab_size: base vocabulary size, used to bound the exact key range.
    """

    def __init__(self, config, table_sizes, table_orders, vocab_size):
        self.table_sizes = [int(m) for m in table_sizes]
        self.table_orders = [int(n) for n in table_orders]
        self.vocab_size = int(vocab_size)
        self.num_tables = len(self.table_sizes)

        def _cfg(name, default):
            return getattr(config, "ngram_monitor_" + name, default)

        self.interval = max(int(_cfg("interval", 100)), 1)
        self.usage_enabled = bool(_cfg("usage", True))
        self.collision_enabled = bool(_cfg("collision", True))
        self.collision_max_rows = int(_cfg("collision_max_rows", 0))
        self.signal_enabled = bool(_cfg("signal", True))
        self.signal_rows = max(int(_cfg("signal_rows", 1)), 1)
        self.per_table_metrics = bool(_cfg("per_table_metrics", True))
        self.distribution_enabled = bool(_cfg("distribution", True))
        self.stale_windows = max(int(_cfg("stale_windows", 10)), 1)
        self.reduce_group_mode = str(_cfg("reduce_group", "auto"))

        offsets, total = [], 0
        for m in self.table_sizes:
            offsets.append(total)
            total += m
        self._offsets = offsets
        self._total = total

        if self.usage_enabled:
            # A single flat buffer covers all sub-tables.  Index ``total`` is a
            # sentinel slot that absorbs masked-out (invalid) positions, so the
            # per-step update needs no boolean indexing.
            self.cum_count = paddle.zeros([total + 1], dtype="int64")
            self.delta_count = paddle.zeros([total + 1], dtype="int64")
            self.last_window = paddle.zeros([total + 1], dtype="int32")
            self.sentinel_index = paddle.full([1], total, dtype="int64")
            self._hist_edges = paddle.to_tensor(_HIST_EDGES, dtype="int64")

        # Number of completed analysis windows; also the "clock" of last_window.
        self._window_id = 0
        self._window_steps = 0
        # Steps already folded into cum_count.  Persisted, so that cumulative
        # per-step rates keep their meaning across a warm restart.
        self._cum_steps = 0
        self._global_step = 0
        self._analyze_now = False
        self._pending = []
        self._collect = []
        self._signal = None
        self._group = None
        self._group_resolved = False
        self.loaded_from = None

    # ---------------------------------------------------------------- plumbing
    @property
    def needs_keys(self) -> bool:
        """True when the embedding must also hand over the exact N-gram keys."""
        return self._analyze_now and self.collision_enabled

    @property
    def window_id(self) -> int:
        return self._window_id

    def begin_step(self, global_step: int) -> None:
        """Called once per optimizer step, before the forward pass."""
        self._global_step = int(global_step)
        self._analyze_now = self._global_step % self.interval == 0
        self._pending = []
        self._collect = []
        self._signal = None

    def _reduce_group(self):
        if not self._group_resolved:
            self._group = _resolve_reduce_group(self.reduce_group_mode)
            self._group_resolved = True
        return self._group

    @staticmethod
    def _group_size(group) -> int:
        if not dist.is_initialized():
            return 1
        return dist.get_world_size() if group is None else group.nranks

    @staticmethod
    def _pack(values):
        parts = []
        for v in values:
            if isinstance(v, paddle.Tensor):
                parts.append(v.astype("float64").reshape([1]))
            else:
                parts.append(paddle.to_tensor([float(v)], dtype="float64"))
        return paddle.concat(parts)

    def _key_range(self, order: int) -> int:
        return self.vocab_size**order

    # ----------------------------------------------------------------- observe
    def observe(self, index, hash_ids, valid_mask, true_key=None) -> None:
        """Record the lookups of one sub-table for the current step.

        Args:
            index: flat sub-table index.
            hash_ids: [B, S] bucket ids that the embedding lookup actually used.
            valid_mask: [B, S] bool; False where the N-gram window reaches before
                the start of the sequence (the artificially padded prefix).
            true_key: [B, S] exact (un-modded) polynomial key.  Only required
                when ``needs_keys`` is True.
        """
        if self.usage_enabled:
            flat = hash_ids + self._offsets[index]
            if valid_mask is not None:
                flat = paddle.where(valid_mask, flat, self.sentinel_index)
            self._pending.append(flat.reshape([-1]))
        if self._analyze_now and self.collision_enabled and true_key is not None:
            self._collect.append((index, hash_ids, true_key, valid_mask))

    def observe_flat(self, flat_index, valid_mask=None) -> None:
        """Record lookups that are already expressed as global row indices.

        The routed variant addresses one flat table, so it hands over the global
        row index directly instead of a (sub-table, bucket) pair.  ``flat_index``
        may have any shape; ``valid_mask`` only has to broadcast against it.
        """
        if not self.usage_enabled:
            return
        flat = flat_index
        if valid_mask is not None:
            flat = paddle.where(valid_mask, flat, self.sentinel_index)
        self._pending.append(flat.reshape([-1]))

    def commit(self) -> None:
        """Fold this step's lookups into the counters (one concat + one bincount)."""
        if not self._pending:
            return
        flat = (
            self._pending[0]
            if len(self._pending) == 1
            else paddle.concat(self._pending)
        )
        self.delta_count += paddle.bincount(flat, minlength=self._total + 1)
        self._window_steps += 1
        self._pending = []

    def observe_signal(self, word_emb, ngram_signal) -> None:
        """Record the magnitude of the N-gram signal relative to the word embedding."""
        if not (self._analyze_now and self.signal_enabled):
            return
        rows = min(self.signal_rows, word_emb.shape[0])
        with paddle.no_grad():
            w = word_emb[:rows].astype("float32")
            g = ngram_signal[:rows].astype("float32")
            self._signal = paddle.stack([(w * w).mean(), (g * g).mean()]).astype(
                "float64"
            )

    # ---------------------------------------------------------------- analysis
    def analyze(self):
        """Reduce across ranks and summarise.  Returns a flat metric dict or None."""
        if not self._analyze_now:
            return None
        self._analyze_now = False
        group = self._reduce_group()
        metrics = {}
        if self.signal_enabled and self._signal is not None:
            metrics.update(self._analyze_signal(group))
        if self.collision_enabled and self._collect:
            metrics.update(self._analyze_collision(group))
        # Usage last: it consumes (and zeroes) delta_count.
        if self.usage_enabled and self._window_steps > 0:
            metrics.update(self._analyze_usage(group))
        self._collect = []
        self._signal = None
        return metrics

    def _analyze_signal(self, group):
        vals = self._signal
        if dist.is_initialized():
            dist.all_reduce(vals, group=group)
            vals = vals / float(self._group_size(group))
        w2, g2 = (float(v) for v in vals.numpy())
        word_rms = math.sqrt(max(w2, 0.0))
        ngram_rms = math.sqrt(max(g2, 0.0))
        return {
            "word_emb_rms": word_rms,
            "ngram_signal_rms": ngram_rms,
            "ngram_over_word_rms": (ngram_rms / word_rms) if word_rms > 0 else 0.0,
        }

    def _fold_delta(self):
        """Sum the local window counts across ranks and fold them into cum_count.

        After this call ``cum_count`` and ``last_window`` are identical on every
        rank, and ``delta_count`` holds the *global* counts of the window.
        """
        delta = self.delta_count
        if dist.is_initialized():
            dist.all_reduce(delta, group=self._reduce_group())
        self.cum_count += delta
        self._window_id += 1
        self._cum_steps += self._window_steps
        paddle.assign(
            paddle.where(
                delta > 0,
                paddle.full_like(self.last_window, self._window_id),
                self.last_window,
            ),
            self.last_window,
        )
        return delta

    def sync_pending(self) -> None:
        """Collective; flush counts that have not been folded in yet.

        Must be called on every rank (e.g. right before checkpointing) so that
        the persisted cumulative table does not lose the current window.
        """
        if not self.usage_enabled or self._window_steps == 0:
            return
        self._fold_delta().zero_()
        self._window_steps = 0

    def _row_distribution(self, counts, m):
        """Shape of the per-row hit-count distribution of one sub-table.

        ``counts`` must already be summed across ranks, which makes this a
        purely local computation whose result is bit-identical on every rank --
        no extra collective is needed for any of the distribution metrics.

        Returns one flat float64 tensor laid out as
        ``[histogram(12), quantiles(4), top-row mass shares(3), gini]``:

        * histogram -- number of rows whose count falls in each bucket, i.e.
          the literal "how many table positions were hit k times" answer;
        * quantiles -- the count value at the p50/p90/p99/p999 row;
        * mass shares -- fraction of all lookups absorbed by the busiest
          0.1% / 1% / 10% of rows (a three-point Lorenz curve);
        * gini -- 0 when every row is used equally, 1 under total concentration.
        """
        c = counts.reshape([-1])
        bucket = paddle.searchsorted(self._hist_edges, c, right=True) - 1
        hist = paddle.bincount(bucket, minlength=len(_HIST_EDGES)).astype("float64")
        srt = paddle.sort(c).astype("float64")
        csum = paddle.cumsum(srt)
        total = csum[-1]
        denom = paddle.clip(total, min=1.0)
        q_idx = paddle.to_tensor(
            [min(int(q * (m - 1)), m - 1) for q in _QUANTILES], dtype="int64"
        )
        # Mass of the busiest k rows = total - csum[m - k - 1].
        t_idx = paddle.to_tensor(
            [max(m - max(int(f * m), 1) - 1, 0) for f in _TOP_FRACS], dtype="int64"
        )
        # With x sorted ascending, sum_i i*x_i == (m + 1) * total - sum(csum),
        # so the Gini coefficient needs no second pass over the rows.
        weighted = (m + 1) * total - csum.sum()
        gini = 2.0 * weighted / (m * denom) - float(m + 1) / m
        return paddle.concat(
            [
                hist[: len(_HIST_EDGES)],
                srt.gather(q_idx),
                (total - csum.gather(t_idx)) / denom,
                gini.reshape([1]),
            ]
        )

    @staticmethod
    def _entropy(counts, lookups):
        """Shannon entropy (nats) of the bucket-load distribution counts/sum."""
        p = counts / paddle.clip(lookups, min=1.0)
        return -(p * paddle.log(paddle.clip(p, min=1e-300))).sum()

    @staticmethod
    def _dispersion(sq, lookups, m):
        """Index of dispersion Var/Mean of the per-row counts.

        ``sq`` is sum(c^2), so Var/Mean == sum(c^2)/L - L/M.  For an ideal
        uniform hash over L *distinct* keys the counts are Binomial(L, 1/M) and
        this is ~1.0; values far above 1 mean the load is concentrated, either
        because the hash is bad or -- far more likely here -- because the token
        distribution itself is Zipf.  It is the cheapest single number that
        answers "are the buckets loaded evenly?".
        """
        if lookups <= 0:
            return 0.0
        return sq / lookups - lookups / m

    def _analyze_usage(self, group):
        """Windowed + cumulative row-usage statistics.

        ``delta_count`` is summed across ranks first, so ``cum_count``,
        ``last_window`` and every number derived below are identical on every
        rank; only rank 0 ever has to write them out.
        """
        delta = self._fold_delta()
        window_steps = self._window_steps
        self._window_steps = 0

        per_table = []
        for t, m in enumerate(self.table_sizes):
            o = self._offsets[t]
            d = delta[o : o + m].astype("float64")
            c = self.cum_count[o : o + m].astype("float64")
            lw = self.last_window[o : o + m]
            win_lookups = d.sum()
            cum_lookups = c.sum()
            age = self._window_id - lw
            parts = [
                paddle.stack(
                    [
                        win_lookups,
                        (d > 0).astype("float64").sum(),
                        self._entropy(d, win_lookups),
                        d.max(),
                        (d * d).sum(),
                        ((lw > 0) & (age > self.stale_windows))
                        .astype("float64")
                        .sum(),
                        cum_lookups,
                        (c > 0).astype("float64").sum(),
                        self._entropy(c, cum_lookups),
                        (c * c).sum(),
                    ]
                )
            ]
            if self.distribution_enabled:
                parts.append(self._row_distribution(delta[o : o + m], m))
                parts.append(self._row_distribution(self.cum_count[o : o + m], m))
            per_table.append(paddle.concat(parts) if len(parts) > 1 else parts[0])
        # One device-to-host copy for the whole analysis step.
        raw = paddle.stack(per_table).numpy()
        delta.zero_()
        return self._usage_metrics(raw, window_steps)

    def _distribution_metrics(self, raw, out):
        """Turn the stacked per-table distribution block into named metrics.

        Histograms are additive over sub-tables, so the global histogram is the
        row-count-weighted truth.  Quantiles, mass shares and Gini are not
        additive; the global value is the plain mean over sub-tables, which is
        the right summary here because the moduli differ by at most 2K rows.
        Per sub-table only the two cumulative shape numbers are exported --
        Gini and the top-1% mass share -- because the per-table window-side
        shape is noisy and the quantiles are already implied by the histogram.
        """
        rows = float(self._total)
        n_hist = len(_HIST_EDGES)
        for block, prefix in ((0, "win_"), (_DIST_WIDTH, "cum_")):
            base = _BASE_WIDTH + block
            hist = raw[:, base : base + n_hist].sum(axis=0)
            for name, value in zip(_HIST_NAMES, hist):
                out[prefix + "hist/" + name] = float(value) / rows
            names = (
                [prefix + q for q in _QUANTILE_NAMES]
                + [prefix + "mass_" + t for t in _TOP_NAMES]
                + [prefix + "gini"]
            )
            for offset, name in enumerate(names):
                out[name] = float(np.mean(raw[:, base + n_hist + offset]))
        if self.per_table_metrics:
            tail = _BASE_WIDTH + _DIST_WIDTH + n_hist + len(_QUANTILES)
            columns = {
                "cum_mass_top01": tail + _TOP_NAMES.index("top01"),
                "cum_gini": tail + len(_TOP_FRACS),
            }
            for name, col in columns.items():
                for t in range(len(self.table_sizes)):
                    out[f"t{t}/" + name] = float(raw[t][col])
        return out

    def _usage_metrics(self, raw, window_steps):
        """Name the stacked per-table block; see _BASE_NAMES for its layout."""
        out = {}
        rows = float(self._total)
        agg = raw[:, :_BASE_WIDTH].sum(axis=0)
        win_lookups, cum_lookups = float(agg[0]), float(agg[6])
        win_eff, cum_eff, win_disp, cum_disp, ideal_ratios = [], [], [], [], []
        win_entropy = cum_entropy = win_max = 0.0
        for t, m in enumerate(self.table_sizes):
            (l_w, hit_w, ent_w, max_w, sq_w, stale, l_c, hit_c, ent_c, sq_c) = (
                float(x) for x in raw[t][:_BASE_WIDTH]
            )
            eff_w = math.exp(min(ent_w, 40.0)) / m
            eff_c = math.exp(min(ent_c, 40.0)) / m
            disp_w = self._dispersion(sq_w, l_w, m)
            disp_c = self._dispersion(sq_c, l_c, m)
            win_eff.append(eff_w)
            cum_eff.append(eff_c)
            win_disp.append(disp_w)
            cum_disp.append(disp_c)
            win_max = max(win_max, max_w)
            ideal_t = 0.0
            if l_w > 0:
                ideal_t = min((l_w - _ideal_excess(int(l_w), m)) / m, 1.0)
                ideal_ratios.append(ideal_t)
                # Pooled entropy of the concatenation of the sub-tables:
                # H = sum_t (L_t/L) * (H_t - ln(L_t/L)).
                share = l_w / win_lookups
                win_entropy += share * (ent_w - math.log(share))
            if l_c > 0 and cum_lookups > 0:
                share = l_c / cum_lookups
                cum_entropy += share * (ent_c - math.log(share))
            if self.per_table_metrics:
                pre = f"t{t}/"
                out[pre + "win_hit_ratio"] = hit_w / m
                out[pre + "win_over_ideal"] = (
                    (hit_w / m / ideal_t) if ideal_t > 0 else 0.0
                )
                out[pre + "win_eff_rows_ratio"] = eff_w
                out[pre + "win_dispersion"] = disp_w
                out[pre + "win_max_over_mean"] = max_w * m / l_w if l_w > 0 else 0.0
                out[pre + "cum_hit_ratio"] = hit_c / m
                out[pre + "cum_eff_rows_ratio"] = eff_c
                out[pre + "cum_dispersion"] = disp_c
                # The literal per-sub-table cumulative bucket hit count; the
                # full per-row vector behind it lives in the checkpoint.
                out[pre + "cum_lookups"] = l_c
                out[pre + "stale_ratio"] = stale / m

        ideal = float(np.mean(ideal_ratios)) if ideal_ratios else 0.0
        win_hit, cum_hit, stale = float(agg[1]), float(agg[7]), float(agg[5])
        out["win_hit_ratio"] = win_hit / rows
        # Reference value for a perfectly uniform hash with the same number of
        # lookups; the ratio below is the sample-size-invariant skew signal.
        out["win_hit_ratio_ideal"] = ideal
        out["win_hit_over_ideal"] = (win_hit / rows / ideal) if ideal > 0 else 0.0
        out["win_eff_rows_ratio"] = float(np.mean(win_eff))
        out["win_entropy_nats"] = win_entropy
        out["win_dispersion"] = float(np.mean(win_disp))
        out["win_max_over_mean"] = (
            win_max * rows / win_lookups if win_lookups > 0 else 0.0
        )
        out["cum_hit_ratio"] = cum_hit / rows
        out["cum_eff_rows_ratio"] = float(np.mean(cum_eff))
        out["cum_entropy_nats"] = cum_entropy
        out["cum_dispersion"] = float(np.mean(cum_disp))
        out["stale_ratio"] = stale / rows
        out["lookups_per_step"] = win_lookups / max(window_steps, 1)
        out["cum_lookups_per_step"] = cum_lookups / max(self._cum_steps, 1)
        out["window_steps"] = float(window_steps)
        out["window_id"] = float(self._window_id)
        out["cum_steps"] = float(self._cum_steps)
        if self.distribution_enabled:
            self._distribution_metrics(raw, out)
        return {k: float(v) for k, v in out.items()}

    def _analyze_collision(self, group):
        by_order = {}
        for item in self._collect:
            by_order.setdefault(self.table_orders[item[0]], []).append(item)
        out = {}
        for order, items in sorted(by_order.items()):
            for scope in ("seq", "batch"):
                out.update(self._collision_one(order, items, scope, group))
        return out

    def _collision_one(self, order, items, scope, group):
        """Exact collision statistics for one N-gram order and one pooling scope.

        A collision means two *different* N-grams (different exact keys) landing
        in the same row.  Repeated occurrences of the *same* N-gram are not a
        collision, which is exactly why the un-modded key is needed: ``hash_ids``
        alone cannot tell the two situations apart.
        """
        n_tab = len(items)
        b, s = items[0][1].shape
        rows = b if self.collision_max_rows <= 0 else min(b, self.collision_max_rows)
        # Keep the sequence-scoped key inside int64.
        rows = max(min(rows, (2**62) // self._key_range(order)), 1)

        vm = items[0][3]
        if vm is None:
            mask = paddle.ones([rows * s], dtype="bool")
        else:
            mask = vm.expand([rows, s]) if vm.shape[0] == 1 else vm[:rows]
            mask = mask.reshape([-1])
        row_id = paddle.masked_select(
            paddle.arange(rows, dtype="int64")
            .unsqueeze(-1)
            .expand([rows, s])
            .reshape([-1]),
            mask,
        )
        key = paddle.masked_select(items[0][2][:rows].reshape([-1]), mask)
        n_tokens = int(key.shape[0])
        # Shape-determined, therefore identical on every rank: no collective is
        # skipped on one rank only.
        if n_tokens == 0:
            return {}
        n_scopes = rows if scope == "seq" else 1
        if scope == "seq":
            key = key + row_id * self._key_range(order)

        uk, first, tok = paddle.unique(key, return_index=True, return_counts=True)
        n_keys = int(uk.shape[0])
        tok = tok.astype("float64")
        row_first = row_id.gather(first) if scope == "seq" else None

        names, values = [], []
        collided = paddle.zeros([n_keys], dtype="int32")
        key_coll = paddle.zeros([], dtype="float64")
        tok_coll = paddle.zeros([], dtype="float64")
        excess, ideal = 0.0, 0.0
        for index, hash_ids, _k, _m in items:
            m_i = self.table_sizes[index]
            bucket = paddle.masked_select(
                hash_ids[:rows].reshape([-1]), mask
            ).gather(first)
            if scope == "seq":
                bucket = bucket + row_first * m_i
            ub, inv, cnt = paddle.unique(
                bucket, return_inverse=True, return_counts=True
            )
            flag = (cnt.gather(inv) > 1).astype("float64")
            collided += flag.astype("int32")
            key_coll = key_coll + flag.sum()
            tok_coll = tok_coll + (tok * flag).sum()
            excess_t = float(n_keys - int(ub.shape[0]))
            ideal_t = n_scopes * _ideal_excess(n_keys / n_scopes, m_i)
            excess += excess_t
            ideal += ideal_t
            if self.per_table_metrics:
                names += [f"t{index}/excess", f"t{index}/ideal"]
                values += [excess_t, ideal_t]

        hist = paddle.bincount(collided, minlength=n_tab + 1).astype("float64")
        names += [
            "key_coll",
            "tok_coll",
            "allk_keys",
            "any_keys",
            "any_tok",
            "n_keys",
            "n_tokens",
            "excess",
            "ideal",
            "n_scopes",
        ]
        values += [
            key_coll,
            tok_coll,
            hist[n_tab],
            float(n_keys) - hist[0],
            (tok * (collided > 0).astype("float64")).sum(),
            float(n_keys),
            float(n_tokens),
            excess,
            ideal,
            float(n_scopes),
        ]
        packed = self._pack(values)
        if dist.is_initialized():
            dist.all_reduce(packed, group=group)
        v = {k: float(x) for k, x in zip(names, packed.numpy())}
        return self._collision_metrics(order, scope, items, n_tab, v)

    def _collision_metrics(self, order, scope, items, n_tab, v):
        nk, nt = v["n_keys"], v["n_tokens"]
        if nk <= 0 or nt <= 0:
            return {}
        p = f"o{order}/{scope}/"
        rate_key = v["key_coll"] / (nk * n_tab)
        any_key = v["any_keys"] / nk
        # Under independent hash families the chance of escaping all n_tab tables
        # is (1 - rate)^n_tab; comparing against it is the only way to tell
        # whether the K splits are actually independent.
        indep_any = 1.0 - (1.0 - rate_key) ** n_tab
        out = {
            p + "collide_key_rate": rate_key,
            p + "collide_token_rate": v["tok_coll"] / (nt * n_tab),
            p + "excess_lift": (v["excess"] / v["ideal"]) if v["ideal"] > 0 else 0.0,
            p + f"all{n_tab}_key_rate": v["allk_keys"] / nk,
            p + "any_key_rate": any_key,
            p + "any_token_rate": v["any_tok"] / nt,
            p + "any_over_indep": (any_key / indep_any) if indep_any > 0 else 0.0,
            p + "distinct_key_ratio": nk / nt,
            p + "pool_size": nt / max(v["n_scopes"], 1.0),
        }
        if self.per_table_metrics:
            for index, _h, _k, _m in items:
                i_ = v[f"t{index}/ideal"]
                out[p + f"t{index}/excess_lift"] = (
                    (v[f"t{index}/excess"] / i_) if i_ > 0 else 0.0
                )
        return {k: float(val) for k, val in out.items()}

    # ------------------------------------------------------------- persistence
    def state_dict_for_save(self, full: bool = True):
        """Everything needed to continue the counters after a restart.

        ``cum_count`` is the per-row hit-count vector itself, laid out as the
        concatenation of the sub-tables in ``table_sizes`` order (plus one
        trailing sentinel slot), so an offline reader can recover the exact
        distribution per table position from the checkpoint alone.
        """
        state = {
            "version": np.array(2, dtype=np.int64),
            "window_id": np.array(self._window_id, dtype=np.int64),
            "cum_steps": np.array(self._cum_steps, dtype=np.int64),
            "global_step": np.array(self._global_step, dtype=np.int64),
            "table_sizes": np.array(self.table_sizes, dtype=np.int64),
            "table_orders": np.array(self.table_orders, dtype=np.int64),
        }
        if full and self.usage_enabled:
            state["cum_count"] = self.cum_count.numpy()
            state["last_window"] = self.last_window.numpy()
        return state

    def load_saved_state(self, state) -> bool:
        """Restore the counters; False means "incompatible, start cold"."""
        sizes = [int(x) for x in np.asarray(state["table_sizes"]).reshape([-1])]
        if sizes != self.table_sizes:
            return False
        if "table_orders" in state:
            orders = [int(x) for x in np.asarray(state["table_orders"]).reshape([-1])]
            if orders != self.table_orders:
                return False
        if self.usage_enabled and "cum_count" not in state:
            # A layout-only state file cannot restore the cumulative counts, and
            # adopting its window_id without them would silently turn every row
            # into "never hit" and every age into a huge staleness.  Refuse.
            return False
        self._window_id = int(np.asarray(state["window_id"]))
        self._window_steps = 0
        self._cum_steps = (
            int(np.asarray(state["cum_steps"])) if "cum_steps" in state else 0
        )
        if self.usage_enabled:
            paddle.assign(
                paddle.to_tensor(np.asarray(state["cum_count"]), dtype="int64"),
                self.cum_count,
            )
            paddle.assign(
                paddle.to_tensor(np.asarray(state["last_window"]), dtype="int32"),
                self.last_window,
            )
            self.delta_count.zero_()
        return True

    def verify_loaded_state(self) -> bool:
        """Collective; True when every rank restored the *same* history.

        ``output_dir`` is normally shared storage, so all ranks read one file.
        If it is not shared, each rank quietly continues from its own history
        and every cumulative metric becomes meaningless.  Two all_reduces at
        startup turn that into a visible warning.
        """
        if not dist.is_initialized():
            return True
        probe = paddle.to_tensor(
            [
                float(self._window_id),
                float(self._cum_steps),
                float(self.cum_count.sum()) if self.usage_enabled else 0.0,
            ],
            dtype="float64",
        )
        lo, hi = probe.clone(), probe.clone()
        group = self._reduce_group()
        dist.all_reduce(lo, op=dist.ReduceOp.MIN, group=group)
        dist.all_reduce(hi, op=dist.ReduceOp.MAX, group=group)
        return bool((lo == hi).all())
