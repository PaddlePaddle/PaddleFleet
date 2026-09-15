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
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

importlib.import_module("_backend_env").skip_unless_backend("paddlefleet")

try:
    paddle = importlib.import_module("paddle")
    nn = importlib.import_module("paddle.nn")
except (
    Exception
) as exc:  # pragma: no cover - depends on optional backend install
    raise unittest.SkipTest(f"paddle backend unavailable: {exc}") from exc

kda_metrics = importlib.import_module(
    "paddlefleet.internal_medicine.backends.paddlefleet.kda_metrics"
)
kda_monitor = importlib.import_module(
    "paddlefleet.internal_medicine.backends.paddlefleet.kda_monitor"
)
layer_discovery = importlib.import_module(
    "paddlefleet.internal_medicine.backends.paddlefleet.layer_discovery"
)
_training_logs_mod = importlib.import_module(
    "paddlefleet.internal_medicine.core.training_logs"
)
training_logs = _training_logs_mod.training_logs
TrainingLogs = _training_logs_mod.TrainingLogs

PaddleKDAHealthMonitor = kda_monitor.PaddleKDAHealthMonitor

HV = 4  # value heads
DK = 8  # key/value head dim
CONV_DIM = 3 * HV * DK  # q | k | v, all local to this fake rank


class FakeLinear(nn.Layer):
    """Stand-in for a PaddleFleet parallel linear: returns ``(out, bias)``."""

    def __init__(self, out_width):
        super().__init__()
        self.out_width = out_width
        self.next_output = None

    def forward(self, x):
        return self.next_output, None


class FakeKDAAttention(nn.Layer):
    """Stand-in for ``KimiDeltaAttention``: only the attributes the monitor reads.

    Discovery is an attribute probe (``f_b_proj`` + ``gate_lower_bound``), so
    carrying the attributes the monitor reads is enough to be found.
    """

    def __init__(self, use_full_rank_gate=True, gate_lower_bound=-5.0):
        super().__init__()
        self.num_v_heads_local_tp = HV
        self.value_head_dim = DK
        self.key_head_dim = DK
        self.conv_dim_local_tp = CONV_DIM
        self.v_dim_local_tp = HV * DK
        self.use_full_rank_gate = use_full_rank_gate
        self.gate_lower_bound = gate_lower_bound
        self.A_log = paddle.zeros([HV], dtype="float32")
        self.dt_bias = paddle.zeros([HV * DK], dtype="float32")
        self.in_proj = FakeLinear(
            CONV_DIM + HV + (HV * DK if use_full_rank_gate else 0)
        )
        self.f_a_proj = FakeLinear(DK)
        self.f_b_proj = FakeLinear(HV * DK)
        if not use_full_rank_gate:
            self.g_a_proj = FakeLinear(DK)
            self.g_b_proj = FakeLinear(HV * DK)


class FakeLayer(nn.Layer):
    def __init__(self, attn):
        super().__init__()
        self.self_attn = attn


def _model(layers):
    return SimpleNamespace(decoder=SimpleNamespace(layers=nn.LayerList(layers)))


def _fire(attn, z, beta_logit, gate_logit=None, tokens=6):
    """Drive the fake projections so the monitor's hooks see real tensors."""
    attn.f_b_proj.next_output = z
    qkv = paddle.zeros([1, tokens, CONV_DIM], dtype="float32")
    parts = [qkv, beta_logit]
    if attn.use_full_rank_gate:
        parts.append(gate_logit)
    attn.in_proj.next_output = paddle.concat(parts, axis=-1)
    attn.in_proj(None)
    attn.f_b_proj(None)
    if not attn.use_full_rank_gate:
        attn.g_b_proj.next_output = gate_logit
        attn.g_b_proj(None)


class KDAGateMathTest(unittest.TestCase):
    """kda_log_decay must reproduce the layer's own gate, both branches."""

    def _reference(self, z, A_log, dt_bias, safe_gate, lower_bound):
        # Transcribed from kimi_delta_attention.kda_gate.
        x = z.astype("float32").reshape([-1, HV, DK]) + dt_bias.astype(
            "float32"
        ).reshape([HV, DK])
        a = A_log.astype("float32").exp().reshape([HV, 1])
        if safe_gate:
            return lower_bound * nn.functional.sigmoid(a * x)
        return -a * nn.functional.softplus(x)

    def _check(self, safe_gate, lower_bound):
        paddle.seed(0)
        z = paddle.randn([1, 5, HV * DK], dtype="float32") * 3.0
        A_log = paddle.to_tensor([0.0, 0.7, -0.5, 1.2], dtype="float32")
        dt_bias = paddle.randn([HV * DK], dtype="float32")
        got = kda_metrics.kda_log_decay(
            z, A_log, dt_bias, safe_gate, lower_bound, num_heads=HV, head_dim=DK
        )
        want = self._reference(z, A_log, dt_bias, safe_gate, lower_bound)
        self.assertLess(float((got - want).abs().max()), 1e-5)

    def test_matches_lower_bounded_branch(self):
        self._check(safe_gate=True, lower_bound=-5.0)

    def test_matches_unbounded_softplus_branch(self):
        # gate_lower_bound=None配置: g has no floor, so the monitor must not
        # silently assume -5.0.
        self._check(safe_gate=False, lower_bound=None)

    def test_lower_bounded_gate_stays_in_range(self):
        z = paddle.linspace(-50.0, 50.0, HV * DK).reshape([1, 1, HV * DK])
        g = kda_metrics.kda_log_decay(
            z, paddle.zeros([HV]), None, True, -5.0, num_heads=HV, head_dim=DK
        )
        self.assertGreaterEqual(float(g.min()), -5.0)
        self.assertLessEqual(float(g.max()), 0.0)

    def test_safe_gate_without_bound_is_rejected(self):
        with self.assertRaises(ValueError):
            kda_metrics.kda_log_decay(
                paddle.zeros([1, 1, HV * DK]),
                paddle.zeros([HV]),
                None,
                True,
                None,
                HV,
                DK,
            )


class KDAMetricsTest(unittest.TestCase):
    def test_channel_spread_separates_collapse_from_diversity(self):
        # The whole point of alpha_channel_spread: two tensors with the SAME mean,
        # one collapsed onto a single time scale and one spread across channels.
        collapsed = paddle.full([10, HV, DK], -2.0, dtype="float32")
        spread = paddle.concat(
            [
                paddle.full([10, HV, DK // 2], -1.0),
                paddle.full([10, HV, DK // 2], -3.0),
            ],
            axis=-1,
        )
        a = kda_metrics.decay_gate_stats(collapsed)
        b = kda_metrics.decay_gate_stats(spread)
        self.assertAlmostEqual(
            float(a["alpha_log_mean"]), float(b["alpha_log_mean"]), places=5
        )
        self.assertAlmostEqual(float(a["alpha_channel_spread"]), 0.0, places=6)
        self.assertGreater(float(b["alpha_channel_spread"]), 0.9)

    def test_token_spread_is_zero_for_an_input_independent_gate(self):
        # A gate frozen into a constant decay: identical across tokens.
        frozen = paddle.tile(paddle.randn([1, HV, DK]), [12, 1, 1])
        stats = kda_metrics.decay_gate_stats(frozen)
        self.assertAlmostEqual(
            float(stats["alpha_token_spread"]), 0.0, places=5
        )

        varying = paddle.randn([12, HV, DK])
        self.assertGreater(
            float(kda_metrics.decay_gate_stats(varying)["alpha_token_spread"]),
            0.0,
        )

    def test_channel_min_averages_tokens_before_taking_the_extremum(self):
        # Constraint 3: a single outlier token must not pin the reading. One token
        # dips to -5 on one channel; the channel mean over 10 tokens stays near 0.
        g = paddle.zeros([10, HV, DK], dtype="float32").numpy()
        g[0, 0, 0] = -5.0
        stats = kda_metrics.decay_gate_stats(paddle.to_tensor(g))
        self.assertAlmostEqual(
            float(stats["alpha_log_channel_min"]), -0.5, places=5
        )

    def test_single_token_token_spread_is_defined(self):
        # std over a length-1 axis is undefined; the helper must return 0, not NaN.
        stats = kda_metrics.decay_gate_stats(paddle.randn([1, HV, DK]))
        self.assertEqual(float(stats["alpha_token_spread"]), 0.0)

    def test_beta_head_min_catches_dead_heads_that_the_mean_hides(self):
        # 4 heads, 1 dead: the mean barely moves, the per-head min collapses.
        logit = paddle.zeros([1, 20, HV], dtype="float32").numpy()
        logit[:, :, :] = 0.0  # sigmoid(0) = 0.5 everywhere
        logit[:, :, 0] = -20.0  # head 0 stops writing
        stats = kda_metrics.write_gate_stats(paddle.to_tensor(logit), HV)
        self.assertAlmostEqual(
            float(stats["beta_mean"]), 0.375, places=4
        )  # (3*0.5 + 0) / 4
        self.assertLess(float(stats["beta_head_min"]), 1e-6)
        # The mean alone would look like normal drift off 0.5; the min does not.
        self.assertGreater(float(stats["beta_mean"]), 0.3)

    def test_read_gate_mean_is_the_sigmoid_aperture(self):
        stats = kda_metrics.read_gate_stats(
            paddle.zeros([1, 5, HV * DK]), HV * DK
        )
        self.assertAlmostEqual(float(stats["out_gate_mean"]), 0.5, places=6)

    def test_param_stats_reads_A_log(self):
        stats = kda_metrics.param_stats(
            paddle.to_tensor([0.0, 2.0], dtype="float32")
        )
        self.assertAlmostEqual(float(stats["A_log_mean"]), 1.0, places=6)


class FakeGDNAttention(nn.Layer):
    """Gated Delta Net shape: shares A_log/dt_bias/conv1d with KDA, no f_b_proj.

    GDN is a selectable ``attention_layer_type`` in paddlefleet, so the probe has
    to discriminate against it rather than keying off a generic decay attribute.
    """

    def __init__(self):
        super().__init__()
        self.A_log = paddle.zeros([HV], dtype="float32")
        self.dt_bias = paddle.zeros([HV * DK], dtype="float32")
        self.conv1d = nn.Conv1D(CONV_DIM, CONV_DIM, 4, groups=CONV_DIM)
        self.in_proj = FakeLinear(CONV_DIM + HV)


class KDADiscoveryTest(unittest.TestCase):
    def test_kda_layer_is_tagged_and_others_are_not(self):
        self.assertEqual(
            layer_discovery.classify_attn_type(FakeLayer(FakeKDAAttention())),
            "kda",
        )
        self.assertTrue(
            layer_discovery.is_kda_layer(FakeLayer(FakeKDAAttention()))
        )
        self.assertFalse(
            layer_discovery.is_kda_layer(FakeLayer(nn.Linear(4, 4)))
        )

    def test_gdn_shaped_layer_is_not_mistaken_for_kda(self):
        self.assertFalse(
            layer_discovery.is_kda_layer(FakeLayer(FakeGDNAttention()))
        )

    def test_unbounded_gate_config_is_still_kda(self):
        # gate_lower_bound=None selects the softplus branch; hasattr, not
        # truthiness, is what the probe must use.
        self.assertTrue(
            layer_discovery.is_kda_layer(
                FakeLayer(FakeKDAAttention(gate_lower_bound=None))
            )
        )

    def test_global_layers_of_a_hybrid_stack_get_tagged(self):
        # A KDA/global mix must not leave the global layers untagged, or their
        # metrics land in the same chart as the KDA layers'.
        layers = [FakeLayer(FakeKDAAttention()), FakeLayer(nn.Linear(4, 4))]
        items = layer_discovery.iter_monitor_layers(layers, lambda layer: True)
        tags = [item.attn_type for item in items]
        self.assertEqual(tags[0], "kda")
        self.assertEqual(tags[1], "global")

    def test_homogeneous_stack_keeps_untagged_keys(self):
        # No KDA anywhere: tags must stay None so existing runs keep their keys.
        items = layer_discovery.iter_monitor_layers(
            [FakeLayer(nn.Linear(4, 4))], lambda layer: True
        )
        self.assertEqual([item.attn_type for item in items], [None])


class KDAMonitorTest(unittest.TestCase):
    def setUp(self):
        training_logs.reset()

    def tearDown(self):
        training_logs.reset()

    @staticmethod
    def _inputs(tokens=6, beta_fill=0.0, gate_fill=0.0):
        z = paddle.randn([1, tokens, HV * DK], dtype="float32")
        beta = paddle.full([1, tokens, HV], beta_fill, dtype="float32")
        gate = paddle.full([1, tokens, HV * DK], gate_fill, dtype="float32")
        return z, beta, gate

    def test_full_rank_gate_records_all_eight_metrics(self):
        attn = FakeKDAAttention(use_full_rank_gate=True)
        monitor = PaddleKDAHealthMonitor()
        monitor.register_hooks(_model([FakeLayer(attn)]))
        # Two hooks only: f_b_proj and in_proj.
        self.assertEqual(len(monitor.hooks), 2)

        _fire(attn, *self._inputs())
        monitor.step()

        latest = training_logs.get_latest(prefix="kda_health")
        for name in kda_metrics.ALL_METRICS:
            self.assertIn(f"kda_health/layer_0/kda_{name}", latest, name)
            self.assertIn(f"kda_health/global_kda_{name}", latest, name)

    def test_low_rank_gate_uses_a_third_hook(self):
        attn = FakeKDAAttention(use_full_rank_gate=False)
        monitor = PaddleKDAHealthMonitor()
        monitor.register_hooks(_model([FakeLayer(attn)]))
        self.assertEqual(len(monitor.hooks), 3)

        _fire(attn, *self._inputs(gate_fill=0.0))
        monitor.step()

        latest = training_logs.get_latest(prefix="kda_health")
        self.assertAlmostEqual(
            latest["kda_health/layer_0/kda_out_gate_mean"], 0.5, places=5
        )

    def test_layers_are_tagged_kda(self):
        # The attn_type tag is what keeps KDA metrics out of the global layers'
        # charts; without it every key would be untagged.
        attn = FakeKDAAttention()
        monitor = PaddleKDAHealthMonitor()
        monitor.register_hooks(_model([FakeLayer(attn)]))
        _fire(attn, *self._inputs())
        monitor.step()

        keys = training_logs.get_latest(prefix="kda_health")
        self.assertTrue(any("/kda_alpha_log_mean" in k for k in keys), keys)

    def test_unbounded_gate_config_still_records(self):
        # gate_lower_bound=None must not raise and must not be read as -5.0.
        attn = FakeKDAAttention(gate_lower_bound=None)
        monitor = PaddleKDAHealthMonitor(verbose=True)
        monitor.register_hooks(_model([FakeLayer(attn)]))
        _fire(attn, *self._inputs())
        monitor.step()

        latest = training_logs.get_latest(prefix="kda_health")
        self.assertIn("kda_health/layer_0/kda_alpha_log_mean", latest)
        self.assertEqual(monitor._failed_layers, set())

    def test_monitor_interval_gates_collection(self):
        attn = FakeKDAAttention()
        monitor = PaddleKDAHealthMonitor(monitor_interval=0)
        monitor.register_hooks(_model([FakeLayer(attn)]))
        _fire(attn, *self._inputs())
        monitor.step()
        self.assertEqual(training_logs.get_latest(prefix="kda_health"), {})

    def test_remove_hooks_detaches_everything(self):
        attn = FakeKDAAttention()
        monitor = PaddleKDAHealthMonitor()
        monitor.register_hooks(_model([FakeLayer(attn)]))
        monitor.remove_hooks()
        self.assertEqual(monitor.hooks, [])

        _fire(attn, *self._inputs())
        monitor.step()
        self.assertEqual(training_logs.get_latest(prefix="kda_health"), {})

    def test_no_kda_layer_is_a_clean_noop(self):
        plain = SimpleNamespace(
            decoder=SimpleNamespace(layers=nn.LayerList([nn.Linear(4, 4)]))
        )
        monitor = PaddleKDAHealthMonitor()
        monitor.register_hooks(plain)
        monitor.step()
        self.assertEqual(monitor.hooks, [])
        self.assertEqual(training_logs.get_latest(prefix="kda_health"), {})

    def test_min_metric_naming_contract(self):
        # training_logs picks the reduction from the key name, independently of
        # MIN_AGGREGATED. The two must agree or cross-rank aggregation averages
        # what should be a min.
        for name in kda_metrics.MIN_METRICS:
            key = f"kda_health/layer_0/kda_{name}"
            self.assertTrue(TrainingLogs._is_min_metric(key), key)
            self.assertFalse(TrainingLogs._is_max_metric(key), key)
        for name in set(kda_metrics.ALL_METRICS) - set(kda_metrics.MIN_METRICS):
            key = f"kda_health/layer_0/kda_{name}"
            self.assertFalse(TrainingLogs._is_min_metric(key), key)
            self.assertFalse(TrainingLogs._is_max_metric(key), key)


if __name__ == "__main__":
    unittest.main()
