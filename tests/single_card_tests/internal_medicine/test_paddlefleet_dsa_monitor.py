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

import importlib
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

importlib.import_module("_backend_env").skip_unless_backend("paddlefleet")

try:
    paddle = importlib.import_module("paddle")
    nn = importlib.import_module("paddle.nn")
    F = importlib.import_module("paddle.nn.functional")
except (
    Exception
) as exc:  # pragma: no cover - depends on optional backend install
    raise unittest.SkipTest(f"paddle backend unavailable: {exc}") from exc

dsa_metrics = importlib.import_module(
    "paddlefleet.internal_medicine.backends.paddlefleet.dsa_metrics"
)
dsa_monitor = importlib.import_module(
    "paddlefleet.internal_medicine.backends.paddlefleet.dsa_monitor"
)
layer_discovery = importlib.import_module(
    "paddlefleet.internal_medicine.backends.paddlefleet.layer_discovery"
)
_training_logs_mod = importlib.import_module(
    "paddlefleet.internal_medicine.core.training_logs"
)
training_logs = _training_logs_mod.training_logs
TrainingLogs = _training_logs_mod.TrainingLogs

PaddleDSAHealthMonitor = dsa_monitor.PaddleDSAHealthMonitor

SEQ = 16
IDX_HEADS = 2
IDX_DIM = 4
ATTN_HEADS = 2
ATTN_DIM = 4
TOPK = 4
SCALE = 0.5


class FakeIndexer(nn.Layer):
    """``DSAIndexer`` reduced to what the monitor reads: the API and the return."""

    def __init__(self, index_topk=TOPK):
        super().__init__()
        self.index_topk = index_topk
        self.next_output = None

    def forward_before_topk(
        self, hidden_states, q_latent, position_offset=0, cp_group=None
    ):
        return self.next_output


class FakeDSAttention(nn.Layer):
    """``DSAttention``: calls the indexer, then returns something query-shaped.

    ``position_offset`` is passed positionally, the way
    ``MQALatentAttention._indexer_projections`` does it.
    """

    def __init__(self, indexer=None, softmax_scale=SCALE, position_offset=0):
        super().__init__()
        self.indexer = indexer if indexer is not None else FakeIndexer()
        self.softmax_scale = softmax_scale
        self.position_offset = position_offset

    def forward(self, query, key, value, attention_mask=None, x=None, qr=None):
        self.indexer.forward_before_topk(x, qr, self.position_offset, None)
        return query


class FakeCSAIndexer(nn.Layer):
    """``CSAIndexer``: same ``forward_before_topk`` / ``index_topk`` pair as DSA's.

    Shaped after the real one, which is why it is a trap: it returns the same
    ``(q, k, weights)`` arity, but its ``k`` axis is compressed positions rather
    than original tokens, so every causal / distance reading would silently be
    against the wrong sequence.
    """

    def __init__(self, index_topk=TOPK, compress_ratio=4):
        super().__init__()
        self.index_topk = index_topk
        self.compress_ratio = compress_ratio
        self.compressor = nn.Linear(4, 4)

    def forward_before_topk(self, x, qr, **kwargs):
        raise AssertionError("the DSA monitor must never reach a CSA indexer")


class FakeCompressedSparseAttention(nn.Layer):
    """``CompressedSparseAttention``: an indexer, plus the CSA entry point."""

    def __init__(self):
        super().__init__()
        self.indexer = FakeCSAIndexer()
        self.compress_ratio = 4
        self.softmax_scale = SCALE

    def compressed_sparse_attn(self, *args, **kwargs):
        raise AssertionError("not reachable from the DSA monitor")


class FakeAttention(nn.Layer):
    def __init__(self, core):
        super().__init__()
        self.core_attention = core


class FakeLayer(nn.Layer):
    def __init__(self, core):
        super().__init__()
        self.self_attn = FakeAttention(core)


def _model(cores):
    return SimpleNamespace(
        decoder=SimpleNamespace(
            layers=nn.LayerList([FakeLayer(c) for c in cores])
        )
    )


def _indexer_inputs(seed=0):
    paddle.seed(seed)
    q = paddle.randn([1, SEQ, IDX_HEADS, IDX_DIM], dtype="float32")
    k = paddle.randn([1, SEQ, IDX_DIM], dtype="float32")
    weights = paddle.rand([1, SEQ, IDX_HEADS], dtype="float32")
    return q, k, weights


def _attn_inputs(seed=1):
    paddle.seed(seed)
    query = paddle.randn([1, SEQ, ATTN_HEADS, ATTN_DIM], dtype="float32")
    key = paddle.randn([1, SEQ, ATTN_HEADS, ATTN_DIM], dtype="float32")
    return query, key


def _one_hot_indexer_output(scores_per_position):
    """Indexer output whose row score at position ``t`` is ``IDX_HEADS * relu(v[t])``."""
    q = paddle.zeros([1, SEQ, IDX_HEADS, IDX_DIM], dtype="float32")
    q[:, :, :, 0] = 1.0
    k = paddle.zeros([1, SEQ, IDX_DIM], dtype="float32")
    k[0, :, 0] = scores_per_position
    weights = paddle.ones([1, SEQ, IDX_HEADS], dtype="float32")
    return q, k, weights


def _fire(core, indexer_out, query, key):
    core.indexer.next_output = indexer_out
    core(query, key, query, x=None, qr=None)


def _reference_last_row(indexer_out, query, key, index_topk):
    """Recall / selection for the single row ``SEQ - 1``, computed without the masks.

    Deliberately a different formulation from ``dsa_metrics``: direct indexing of
    the last row instead of a causal mask, so an error in the mask construction
    cannot cancel out.
    """
    q, k, weights = indexer_out
    row = SEQ - 1
    scores = paddle.einsum("hd,td->ht", q[0, row], k[0])
    index_scores = (weights[0, row].unsqueeze(-1) * F.relu(scores)).sum(axis=0)
    selected = paddle.topk(index_scores, k=index_topk)[1]
    logits = paddle.einsum("hd,thd->ht", query[0, row], key[0]) * SCALE
    probs = F.softmax(logits, axis=-1)
    recall = paddle.index_select(probs, selected, axis=-1).sum(axis=-1)
    return float(recall.mean()), selected


class DSADiscoveryTest(unittest.TestCase):
    def test_a_dsa_core_is_found_by_its_indexer_api(self):
        self.assertTrue(
            layer_discovery.is_dsa_layer(FakeLayer(FakeDSAttention()))
        )

    def test_a_plain_attention_is_not_dsa(self):
        self.assertFalse(
            layer_discovery.is_dsa_layer(FakeLayer(nn.Linear(4, 4)))
        )

    def test_a_csa_indexer_is_not_mistaken_for_dsa(self):
        # `CSAIndexer` has the same forward_before_topk / index_topk pair, so the
        # indexer API alone is not a discriminator. Its key axis is compressed
        # positions, which would make every causal and distance reading wrong.
        self.assertFalse(
            layer_discovery.is_dsa_layer(
                FakeLayer(FakeCompressedSparseAttention())
            )
        )

    def test_a_csa_indexer_hung_on_a_dsa_shaped_core_is_still_rejected(self):
        # Belt and braces: reject on the indexer's own markers too, not only on
        # the core's `compressed_sparse_attn` entry point.
        self.assertFalse(
            layer_discovery.is_dsa_layer(
                FakeLayer(FakeDSAttention(indexer=FakeCSAIndexer()))
            )
        )

    def test_the_dsa_monitor_registers_nothing_on_a_csa_stack(self):
        monitor = PaddleDSAHealthMonitor()
        monitor.register_hooks(_model([FakeCompressedSparseAttention()]))
        self.assertEqual(monitor.hooks, [])


class DSAMonitorTest(unittest.TestCase):
    """One sampled row (``row_samples=1`` picks ``SEQ - 1``) so values are exact."""

    def setUp(self):
        training_logs.reset()

    def tearDown(self):
        training_logs.reset()

    def _monitor(self, cores, **kwargs):
        options = {
            "row_samples": 1,
            "head_samples": ATTN_HEADS,
            "local_window": 2,
        }
        options.update(kwargs)
        monitor = PaddleDSAHealthMonitor(**options)
        monitor.register_hooks(_model(cores))
        return monitor

    def test_every_declared_metric_is_emitted_per_layer_and_globally(self):
        core = FakeDSAttention()
        monitor = self._monitor([core])
        # One instance patch on the indexer + one post hook on the module.
        self.assertEqual(len(monitor.hooks), 2)

        _fire(core, _indexer_inputs(), *_attn_inputs())
        monitor.step()

        latest = training_logs.get_latest(prefix="dsa_health")
        # `select_iou_prev` needs a preceding layer; everything else must be here.
        for name in set(dsa_metrics.ALL_METRICS) - {"select_iou_prev"}:
            self.assertIn(f"dsa_health/layer_0/{name}", latest, name)
            self.assertIn(f"dsa_health/global_{name}", latest, name)
        self.assertNotIn("dsa_health/layer_0/select_iou_prev", latest)

    def test_recall_matches_an_independent_computation(self):
        core = FakeDSAttention()
        monitor = self._monitor([core])
        indexer_out, (query, key) = _indexer_inputs(), _attn_inputs()

        _fire(core, indexer_out, query, key)
        monitor.step()

        want, _selected = _reference_last_row(indexer_out, query, key, TOPK)
        got = training_logs.get_latest(prefix="dsa_health")[
            "dsa_health/layer_0/attn_recall"
        ]
        self.assertAlmostEqual(got, want, places=5)

    def test_kl_is_the_negative_log_of_recall_per_row(self):
        # p_sparse is p_dense renormalised over the selected set, so the
        # per-(head, row) KL is exactly -log(recall). It is reported separately
        # because the *aggregate* is not: mean(-log r) >= -log(mean r), and the
        # gap is what a single badly-served head costs.
        core = FakeDSAttention()
        monitor = self._monitor(
            [core], head_samples=1
        )  # one head, one row -> no averaging
        _fire(core, _indexer_inputs(), *_attn_inputs())
        monitor.step()

        latest = training_logs.get_latest(prefix="dsa_health")
        self.assertAlmostEqual(
            latest["dsa_health/layer_0/attn_kl"],
            -math.log(latest["dsa_health/layer_0/attn_recall"]),
            places=5,
        )

    def test_the_reported_kl_is_at_least_the_log_of_the_reported_recall(self):
        # Jensen, across the heads that get averaged. A violation would mean the
        # two metrics are not measuring the same selection.
        core = FakeDSAttention()
        monitor = self._monitor([core])
        _fire(core, _indexer_inputs(), *_attn_inputs())
        monitor.step()

        latest = training_logs.get_latest(prefix="dsa_health")
        self.assertGreaterEqual(
            latest["dsa_health/layer_0/attn_kl"] + 1e-6,
            -math.log(latest["dsa_health/layer_0/attn_recall"]),
        )

    def test_selecting_everything_gives_perfect_recall_and_zero_kl(self):
        core = FakeDSAttention(indexer=FakeIndexer(index_topk=SEQ))
        monitor = self._monitor([core])
        _fire(core, _indexer_inputs(), *_attn_inputs())
        monitor.step()

        latest = training_logs.get_latest(prefix="dsa_health")
        self.assertAlmostEqual(
            latest["dsa_health/layer_0/attn_recall"], 1.0, places=5
        )
        self.assertAlmostEqual(
            latest["dsa_health/layer_0/attn_kl"], 0.0, places=5
        )
        self.assertAlmostEqual(
            latest["dsa_health/layer_0/select_ratio"], 1.0, places=5
        )
        self.assertAlmostEqual(
            latest["dsa_health/layer_0/score_topk_mass"], 1.0, places=5
        )

    def test_realised_sparsity_is_topk_over_the_causal_length(self):
        core = FakeDSAttention()
        monitor = self._monitor([core])
        _fire(core, _indexer_inputs(), *_attn_inputs())
        monitor.step()

        latest = training_logs.get_latest(prefix="dsa_health")
        self.assertAlmostEqual(
            latest["dsa_health/layer_0/select_ratio"], TOPK / SEQ, places=5
        )

    def test_identical_selections_across_two_layers_give_iou_one(self):
        shared = _indexer_inputs()
        cores = [FakeDSAttention(), FakeDSAttention()]
        monitor = self._monitor(cores)
        query, key = _attn_inputs()
        for core in cores:  # same indexer inputs -> same selection
            _fire(core, shared, query, key)
        monitor.step()

        latest = training_logs.get_latest(prefix="dsa_health")
        # Only the second layer has a predecessor within the pass.
        self.assertNotIn("dsa_health/layer_0/select_iou_prev", latest)
        self.assertAlmostEqual(
            latest["dsa_health/layer_1/select_iou_prev"], 1.0, places=5
        )

    def test_disjoint_selections_across_two_layers_give_iou_zero(self):
        cores = [FakeDSAttention(), FakeDSAttention()]
        monitor = self._monitor(cores)
        query, key = _attn_inputs()
        # q on channel 0 only and k carrying its score on channel 0 makes the
        # index score of position t exactly `IDX_HEADS * relu(v[t])`, so `v`
        # dictates the ranking: increasing picks the last TOPK, decreasing the
        # first TOPK, and the two selections are disjoint by construction.
        rising = paddle.arange(1, SEQ + 1, dtype="float32")
        for core, v in ((cores[0], rising), (cores[1], rising.flip([0]))):
            _fire(core, _one_hot_indexer_output(v), query, key)
        monitor.step()

        latest = training_logs.get_latest(prefix="dsa_health")
        self.assertAlmostEqual(
            latest["dsa_health/layer_1/select_iou_prev"], 0.0, places=6
        )
        # Sanity: the two layers really did look in opposite directions.
        self.assertGreater(latest["dsa_health/layer_0/select_local_ratio"], 0.0)

    def test_monitor_interval_zero_collects_nothing(self):
        core = FakeDSAttention()
        monitor = self._monitor([core], monitor_interval=0)
        _fire(core, _indexer_inputs(), *_attn_inputs())
        monitor.step()
        self.assertEqual(training_logs.get_latest(prefix="dsa_health"), {})

    def test_remove_hooks_detaches_the_patch_as_well_as_the_hook(self):
        core = FakeDSAttention()
        monitor = self._monitor([core])
        monitor.remove_hooks()
        self.assertEqual(monitor.hooks, [])

        _fire(core, _indexer_inputs(), *_attn_inputs())
        monitor.step()
        self.assertEqual(training_logs.get_latest(prefix="dsa_health"), {})

    def test_no_dsa_layer_is_a_clean_noop(self):
        monitor = PaddleDSAHealthMonitor()
        monitor.register_hooks(
            SimpleNamespace(
                decoder=SimpleNamespace(layers=nn.LayerList([nn.Linear(4, 4)]))
            )
        )
        monitor.step()
        self.assertEqual(monitor.hooks, [])
        self.assertEqual(training_logs.get_latest(prefix="dsa_health"), {})

    def test_a_failing_layer_is_logged_once_and_does_not_raise(self):
        core = FakeDSAttention()
        monitor = self._monitor([core])
        # Shapes the metric math cannot use: the hook must swallow it.
        _fire(
            core,
            (paddle.zeros([1, 2]), paddle.zeros([1, 2]), paddle.zeros([1, 2])),
            *_attn_inputs(),
        )
        monitor.step()
        self.assertEqual(monitor._failed_layers, {0})

    def test_min_and_max_metric_naming_contract(self):
        # training_logs picks the cross-rank reduction from the key name. It must
        # agree with MAX_AGGREGATED / MIN_AGGREGATED or a min gets averaged.
        for name in dsa_metrics.MIN_METRICS:
            key = f"dsa_health/layer_0/{name}"
            self.assertTrue(TrainingLogs._is_min_metric(key), key)
        for name in dsa_metrics.MAX_METRICS:
            key = f"dsa_health/layer_0/{name}"
            self.assertTrue(TrainingLogs._is_max_metric(key), key)
        for name in (
            set(dsa_metrics.ALL_METRICS)
            - set(dsa_metrics.MAX_METRICS)
            - set(dsa_metrics.MIN_METRICS)
        ):
            key = f"dsa_health/layer_0/{name}"
            self.assertFalse(TrainingLogs._is_max_metric(key), key)
            self.assertFalse(TrainingLogs._is_min_metric(key), key)


class DSASamplingTest(unittest.TestCase):
    def test_rows_start_where_the_topk_rule_starts_to_bite(self):
        # Below index_topk valid keys every causal position is selected, so those
        # rows would pin recall at 1 and hide everything.
        rows = dsa_metrics.sample_query_rows(1024, 256, 8)
        self.assertGreaterEqual(int(rows.min()), 256)
        self.assertEqual(int(rows.max()), 1023)
        self.assertEqual(rows.shape[0], 8)

    def test_a_sequence_shorter_than_topk_falls_back_to_the_second_half(self):
        rows = dsa_metrics.sample_query_rows(64, 256, 4)
        self.assertGreaterEqual(int(rows.min()), 32)
        self.assertEqual(int(rows.max()), 63)

    def test_grouped_kv_heads_are_matched_by_index(self):
        query = paddle.randn([1, 8, 4, 2])
        key = paddle.randn([1, 8, 2, 2])
        q_sub, k_sub = dsa_metrics.align_and_sample_heads(query, key, 4)
        self.assertEqual(q_sub.shape[2], 4)
        self.assertEqual(k_sub.shape[2], 4)
        # Query head h maps to kv head h // 2.
        self.assertTrue(bool((k_sub[0, :, 1] == key[0, :, 0]).all()))
        self.assertTrue(bool((k_sub[0, :, 2] == key[0, :, 1]).all()))

    def test_indivisible_head_counts_skip_the_comparison(self):
        query = paddle.randn([1, 8, 3, 2])
        key = paddle.randn([1, 8, 2, 2])
        self.assertIsNone(dsa_metrics.align_and_sample_heads(query, key, 3))


if __name__ == "__main__":
    unittest.main()


class DSAContextParallelTest(unittest.TestCase):
    """CP: rows are a local slice, the indexer key is already global."""

    def setUp(self):
        training_logs.reset()

    def tearDown(self):
        training_logs.reset()

    def test_rows_are_local_but_the_cut_is_on_the_global_position(self):
        # A CP rank whose slice starts past index_topk has real history on every
        # one of its rows, so none are skipped.
        rows = dsa_metrics.sample_query_rows(1024, 256, 8, position_offset=4096)
        self.assertEqual(int(rows.min()), 0)
        self.assertEqual(int(rows.max()), 1023)
        # Rank 0 still skips its first index_topk rows.
        rows0 = dsa_metrics.sample_query_rows(1024, 256, 8, position_offset=0)
        self.assertGreaterEqual(int(rows0.min()), 256)

    def test_causal_geometry_uses_the_global_row_ids(self):
        # Local row 3 of the slice starting at 100 sees 104 keys, not 4.
        local = paddle.to_tensor([3], dtype="int64")
        causal, dist = dsa_metrics.causal_geometry(local + 100, seq_len_k=256)
        self.assertEqual(int(causal.sum()), 104)
        self.assertEqual(int(dist.max()), 103)

    def test_the_offset_is_read_off_the_indexer_call(self):
        core = FakeDSAttention(position_offset=64)
        monitor = PaddleDSAHealthMonitor(row_samples=1, head_samples=ATTN_HEADS)
        monitor.register_hooks(_model([core]))
        core.indexer.next_output = _indexer_inputs()
        # Call the patched indexer directly: going through the module would let
        # the post hook consume the stash before this can look at it.
        core.indexer.forward_before_topk(None, None, 64, None)
        self.assertEqual(monitor._stash[0][1], 64)
        # Same value when the caller passes it by keyword.
        monitor._stash.clear()
        core.indexer.forward_before_topk(None, None, position_offset=128)
        self.assertEqual(monitor._stash[0][1], 128)
        monitor.remove_hooks()

    def test_cp_drops_only_the_match_family(self):
        core = FakeDSAttention()
        monitor = PaddleDSAHealthMonitor(row_samples=1, head_samples=ATTN_HEADS)
        # Stand in for a real CP group: _init_parallel_state would otherwise
        # overwrite cp_size from the (absent) process group.
        monitor._init_parallel_state = lambda: setattr(monitor, "cp_size", 4)
        monitor.register_hooks(_model([core]))
        self.assertFalse(monitor._match_enabled)
        # Hooks still attached — the monitor is not a no-op under CP.
        self.assertEqual(len(monitor.hooks), 2)

        _fire(core, _indexer_inputs(), *_attn_inputs())
        monitor.step()

        latest = training_logs.get_latest(prefix="dsa_health")
        for name in dsa_metrics.MATCH_METRICS:
            self.assertNotIn(f"dsa_health/layer_0/{name}", latest, name)
        for name in (
            dsa_metrics.INDEXER_METRICS
            + dsa_metrics.SCORE_METRICS
            + dsa_metrics.SELECT_METRICS
        ):
            self.assertIn(f"dsa_health/layer_0/{name}", latest, name)
        monitor.remove_hooks()

    def test_cp_one_keeps_the_match_family(self):
        core = FakeDSAttention()
        monitor = PaddleDSAHealthMonitor(row_samples=1, head_samples=ATTN_HEADS)
        monitor.register_hooks(_model([core]))
        self.assertTrue(monitor._match_enabled)
        _fire(core, _indexer_inputs(), *_attn_inputs())
        monitor.step()
        latest = training_logs.get_latest(prefix="dsa_health")
        for name in dsa_metrics.MATCH_METRICS:
            self.assertIn(f"dsa_health/layer_0/{name}", latest, name)
        monitor.remove_hooks()

    def test_verbose_logs_the_record_count_per_step(self):
        """The count is the only way double counting shows up (mean hides it)."""
        core = FakeDSAttention()
        monitor = PaddleDSAHealthMonitor(
            row_samples=1, head_samples=ATTN_HEADS, verbose=True
        )
        monitor.register_hooks(_model([core]))
        # Two microbatches.
        _fire(core, _indexer_inputs(), *_attn_inputs())
        _fire(core, _indexer_inputs(seed=2), *_attn_inputs())

        with self.assertLogs(dsa_monitor.logger, level="INFO") as captured:
            monitor.step()
        counts = [line for line in captured.output if "record counts" in line]
        self.assertTrue(counts, captured.output)
        self.assertIn("[2]", counts[0])  # one entry per microbatch, not two
        monitor.remove_hooks()
