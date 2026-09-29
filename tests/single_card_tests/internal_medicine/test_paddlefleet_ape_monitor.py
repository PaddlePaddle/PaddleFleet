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
import unittest
from types import SimpleNamespace

try:
    paddle = importlib.import_module("paddle")
except Exception as exc:  # pragma: no cover - optional backend
    raise unittest.SkipTest(f"paddle backend unavailable: {exc}") from exc

ape_metrics = importlib.import_module(
    "paddlefleet.internal_medicine.backends.paddlefleet.ape_metrics"
)
ape_monitor = importlib.import_module(
    "paddlefleet.internal_medicine.backends.paddlefleet.ape_monitor"
)
training_logs = importlib.import_module(
    "paddlefleet.internal_medicine.core.training_logs"
).training_logs


class PaddleAPEMetricsTest(unittest.TestCase):
    def test_uniform_ape_has_uniform_position_distribution(self):
        metrics = ape_metrics.compute_ape_p0_metrics(
            paddle.zeros([4, 3], dtype="float32")
        )

        self.assertAlmostEqual(float(metrics["non_finite_ratio"]), 0.0)
        self.assertAlmostEqual(float(metrics["rms"]), 0.0)
        self.assertAlmostEqual(float(metrics["centered_rms"]), 0.0)
        self.assertAlmostEqual(float(metrics["position_range_p95"]), 0.0)
        self.assertAlmostEqual(
            float(metrics["softmax_entropy_norm_mean"]), 1.0, places=6
        )
        self.assertAlmostEqual(
            float(metrics["softmax_max_prob_p95"]), 0.25, places=6
        )
        self.assertAlmostEqual(
            float(metrics["effective_positions_mean"]), 4.0, places=6
        )

    def test_non_finite_values_are_counted_and_do_not_poison_metrics(self):
        ape = paddle.to_tensor(
            [[float("nan"), 1.0], [float("inf"), 1.0]], dtype="float32"
        )
        metrics = ape_metrics.compute_ape_p0_metrics(ape)

        self.assertAlmostEqual(float(metrics["non_finite_ratio"]), 0.5)
        self.assertTrue(
            all(bool(paddle.isfinite(value)) for value in metrics.values())
        )

    def test_metric_function_requires_rank_two_ape(self):
        with self.assertRaises(ValueError):
            ape_metrics.compute_ape_p0_metrics(
                paddle.zeros([2, 2, 1], dtype="float32")
            )


class PaddleAPEHealthMonitorTest(unittest.TestCase):
    def tearDown(self):
        training_logs.reset()

    def test_p0_schema_and_aggregation_rules(self):
        monitor = ape_monitor.PaddleAPEHealthMonitor()
        for branch in ("core", "indexer"):
            for metric in ape_monitor._P0_METRICS:
                monitor.declare_layer_metric(0, f"{branch}_{metric}")

        self.assertIn("core_non_finite_ratio", monitor.MAX_AGGREGATED)
        self.assertIn("indexer_position_range_p95", monitor.MAX_AGGREGATED)
        self.assertIn("core_softmax_entropy_norm_mean", monitor.MIN_AGGREGATED)
        self.assertIn(
            "indexer_effective_positions_mean", monitor.MIN_AGGREGATED
        )

    def test_setup_discovers_core_and_indexer_ape(self):
        class Compressor(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.ape = self.create_parameter(
                    shape=[4, 3],
                    default_initializer=paddle.nn.initializer.Constant(0.0),
                )

            def forward(self, value):
                return value

        core_compressor = Compressor()
        indexer_compressor = Compressor()
        core_attention = SimpleNamespace(
            compressor=core_compressor,
            indexer=SimpleNamespace(compressor=indexer_compressor),
        )
        model = SimpleNamespace(
            layers=[
                SimpleNamespace(
                    self_attn=SimpleNamespace(core_attention=core_attention)
                )
            ]
        )
        monitor_dict = {}

        ape_monitor.setup_ape_monitor(model, monitor_dict=monitor_dict)
        monitor = monitor_dict["ape_health"]
        self.assertEqual(len(monitor.hooks), 2)

        value = paddle.ones([1], dtype="float32")
        core_compressor(value)
        indexer_compressor(value)
        monitor.step()

        latest = training_logs.get_latest(prefix="ape_health")
        self.assertAlmostEqual(latest["ape_health/layer_0/core_rms"], 0.0)
        self.assertAlmostEqual(
            latest["ape_health/layer_0/indexer_effective_positions_mean"], 4.0
        )
