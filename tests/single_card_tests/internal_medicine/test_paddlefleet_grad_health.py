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

"""grad_health: gradient magnitude math, schema, AMP de-scaling, end-to-end.

The monitor measures ``dL/d(activation)``, so every test here drives a real
backward rather than calling a metric function on a synthetic "gradient" -- the
part that can break is the hook wiring, not the arithmetic.
"""

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
except (
    Exception
) as exc:  # pragma: no cover - depends on optional backend install
    raise unittest.SkipTest(f"paddle backend unavailable: {exc}") from exc

grad_metrics = importlib.import_module(
    "paddlefleet.internal_medicine.backends.paddlefleet.grad_metrics"
)
grad_monitor = importlib.import_module(
    "paddlefleet.internal_medicine.backends.paddlefleet.grad_monitor"
)
training_logs = importlib.import_module(
    "paddlefleet.internal_medicine.core.training_logs"
).training_logs

PaddleGradHealthMonitor = grad_monitor.PaddleGradHealthMonitor

WIDTH = 4


class FakeBranch(nn.Layer):
    """Deterministic stand-in for attention / MLP: a fixed scale of its input."""

    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, x):
        return x * self.factor


class FakeLayer(nn.Layer):
    """Two residual branches, both hooked, plus the layer output itself."""

    def __init__(self, idx, attn_factor=0.5, mlp_factor=0.25):
        super().__init__()
        self.idx = idx
        self.self_attn = FakeBranch(attn_factor)
        self.mlp = FakeBranch(mlp_factor)

    def forward(self, x):
        x = x + self.self_attn(x)
        return x + self.mlp(x)


def _model(layers):
    return SimpleNamespace(decoder=SimpleNamespace(layers=nn.LayerList(layers)))


def _monitor(layers, **kwargs):
    monitor = PaddleGradHealthMonitor(**kwargs)
    monitor.register_hooks(_model(layers))
    return monitor


def _run_backward(layers, grad_seed):
    """Forward the stack, then backprop ``grad_seed`` as ``dL/d(final output)``."""
    x = paddle.ones([2, WIDTH], dtype="float32")
    x.stop_gradient = False
    out = x
    for layer in layers:
        out = layer(out)
    (out * grad_seed).sum().backward()


class GradMagnitudeMathTest(unittest.TestCase):
    def test_norm_is_the_l2_norm(self):
        grad = paddle.to_tensor([[3.0, 4.0], [0.0, 0.0]])
        self.assertAlmostEqual(
            float(grad_metrics.grad_magnitude_stats(grad)["norm"]),
            5.0,
            places=5,
        )

    def test_rms_is_the_norm_over_sqrt_numel(self):
        stats = grad_metrics.grad_magnitude_stats(paddle.randn([3, 5, 7]))
        self.assertAlmostEqual(
            float(stats["rms"]),
            float(stats["norm"]) / math.sqrt(3 * 5 * 7),
            places=4,
        )

    def test_abs_max_sees_a_negative_spike(self):
        grad = paddle.to_tensor([[0.1, -9.0], [0.2, 0.3]])
        self.assertAlmostEqual(
            float(grad_metrics.grad_magnitude_stats(grad)["abs_max"]),
            9.0,
            places=5,
        )

    def test_every_hooked_position_is_a_declared_position(self):
        """``_branch_modules`` and ``POSITIONS`` must not drift apart.

        ``MAX_AGGREGATED`` is built from ``POSITIONS``, so a position the monitor
        emits but that list does not carry would silently become a mean.
        """
        positions = [
            position
            for position, _module in grad_monitor._branch_modules(FakeLayer(0))
        ]
        self.assertEqual(positions, list(grad_metrics.POSITIONS))


class GradMonitorSchemaTest(unittest.TestCase):
    def setUp(self):
        training_logs.reset()

    def tearDown(self):
        training_logs.reset()

    def test_schema_covers_every_position_times_every_metric(self):
        monitor = _monitor([FakeLayer(0), FakeLayer(1)], log_global=False)
        expected = {
            f"grad_health/layer_{idx}/{position}_{metric}"
            for idx in (0, 1)
            for position in grad_metrics.POSITIONS
            for metric in grad_metrics.METRICS
            + grad_metrics.GLOBAL_METRICS
            + grad_metrics.TOKEN_METRICS
        }
        self.assertEqual(monitor._mean_keys | monitor._max_keys, expected)
        self.assertEqual(len(monitor.hooks), 6)  # 3 positions x 2 layers

    def test_one_square_accumulator_per_hooked_module(self):
        """The exact path keeps its own state; it must cover exactly the hooks."""
        monitor = _monitor([FakeLayer(0), FakeLayer(1)], log_global=False)
        self.assertEqual(len(monitor._sq_acc), len(monitor.hooks))
        self.assertEqual(
            sorted(monitor._sq_acc, key=str),
            sorted(
                (
                    (idx, position, None)
                    for idx in (0, 1)
                    for position in grad_metrics.POSITIONS
                ),
                key=str,
            ),
        )

    def test_only_abs_max_is_max_aggregated(self):
        """Spike detectors reduce by max; every magnitude series reduces by mean.

        Averaging a spike detector across microbatches / ranks / layers is exactly
        the wrong reduction -- the one loud microbatch is the signal.
        """
        monitor = _monitor([FakeLayer(0)], log_global=False)
        self.assertEqual(
            monitor._max_keys,
            {
                f"grad_health/layer_0/{position}_{metric}"
                for position in grad_metrics.POSITIONS
                for metric in grad_metrics.MAX_METRICS
            },
        )
        self.assertEqual(
            set(grad_metrics.MAX_METRICS),
            {"abs_max", "token_norm_max", "token_norm_ratio"},
        )

    def test_sample_layers_restricts_both_schema_and_hooks(self):
        monitor = _monitor(
            [FakeLayer(0), FakeLayer(1)], log_global=False, sample_layers=[1]
        )
        self.assertEqual(len(monitor.hooks), 3)
        self.assertTrue(
            all(
                "/layer_1/" in key
                for key in monitor._mean_keys | monitor._max_keys
            )
        )

    def test_a_model_without_layers_registers_nothing(self):
        monitor = PaddleGradHealthMonitor()
        monitor.register_hooks(SimpleNamespace())
        self.assertEqual(monitor.hooks, [])


class GradMonitorAmpTest(unittest.TestCase):
    """The loss scale must not reach the curves; see ``finalize_scaled_grad_metrics``."""

    def setUp(self):
        training_logs.reset()

    def tearDown(self):
        training_logs.reset()

    def _norm_after(self, finalize_calls, scale=8.0):
        layers = [FakeLayer(0)]
        monitor = _monitor(layers, log_global=False)
        _run_backward(layers, paddle.full([2, WIDTH], 2.0))
        for _ in range(finalize_calls):
            monitor.finalize_scaled_grad_metrics(
                SimpleNamespace(_scale=paddle.to_tensor(scale))
            )
        monitor.step()
        return training_logs.get_latest(prefix="grad_health")[
            "grad_health/layer_0/layer_out_norm"
        ]

    def test_the_loss_scale_is_divided_out(self):
        expected = float(paddle.linalg.norm(paddle.full([2, WIDTH], 2.0))) / 8.0
        self.assertAlmostEqual(self._norm_after(1), expected, places=4)

    def test_finalizing_twice_in_one_step_divides_once(self):
        """``on_optimizer_begin`` and the ``_flush_buffers`` fallback both fire."""
        expected = float(paddle.linalg.norm(paddle.full([2, WIDTH], 2.0))) / 8.0
        self.assertAlmostEqual(self._norm_after(2), expected, places=4)

    def test_a_run_without_a_scaler_is_left_alone(self):
        layers = [FakeLayer(0)]
        monitor = _monitor(layers, log_global=False)
        seed = paddle.full([2, WIDTH], 2.0)
        _run_backward(layers, seed)
        monitor.finalize_scaled_grad_metrics(None)
        monitor.step()
        latest = training_logs.get_latest(prefix="grad_health")
        self.assertAlmostEqual(
            latest["grad_health/layer_0/layer_out_norm"],
            float(paddle.linalg.norm(seed)),
            places=4,
        )

    def test_the_next_step_is_de_scaled_again(self):
        """The latch has to reset at flush, or step 2 keeps the raw scaled value."""
        layers = [FakeLayer(0)]
        monitor = _monitor(layers, log_global=False)
        seed = paddle.full([2, WIDTH], 2.0)
        for _ in range(2):
            _run_backward(layers, seed)
            monitor.finalize_scaled_grad_metrics(
                SimpleNamespace(_scale=paddle.to_tensor(8.0))
            )
            monitor.step()
        latest = training_logs.get_latest(prefix="grad_health")
        self.assertAlmostEqual(
            latest["grad_health/layer_0/layer_out_norm"],
            float(paddle.linalg.norm(seed)) / 8.0,
            places=4,
        )


class GradMonitorEndToEndTest(unittest.TestCase):
    def setUp(self):
        training_logs.reset()

    def tearDown(self):
        training_logs.reset()

    def test_layer_out_norm_is_the_norm_of_the_incoming_gradient(self):
        """The last layer's output gradient is exactly the seed, so this is exact."""
        layers = [FakeLayer(0)]
        monitor = _monitor(layers, log_global=False)
        seed = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0]])
        _run_backward(layers, seed)
        monitor.step()

        latest = training_logs.get_latest(prefix="grad_health")
        expected = float(paddle.linalg.norm(seed))
        self.assertAlmostEqual(
            latest["grad_health/layer_0/layer_out_norm"], expected, places=4
        )
        self.assertAlmostEqual(
            latest["grad_health/layer_0/layer_out_rms"],
            expected / math.sqrt(2 * WIDTH),
            places=4,
        )
        self.assertAlmostEqual(
            latest["grad_health/layer_0/layer_out_abs_max"], 4.0, places=4
        )

    def test_every_position_of_every_layer_reports(self):
        layers = [FakeLayer(0), FakeLayer(1)]
        monitor = _monitor(layers, log_global=False)
        _run_backward(layers, paddle.ones([2, WIDTH]))
        monitor.step()

        latest = training_logs.get_latest(prefix="grad_health")
        for idx in (0, 1):
            for position in grad_metrics.POSITIONS:
                for metric in grad_metrics.METRICS:
                    key = f"grad_health/layer_{idx}/{position}_{metric}"
                    self.assertIn(key, latest)
                    self.assertGreater(latest[key], 0.0)

    def test_the_gradient_grows_towards_the_input_of_a_residual_stack(self):
        """Each block multiplies the backward signal by (1+attn)(1+mlp) > 1.

        Not a property of the monitor, but it is the property these curves exist
        to show, and it pins the layer-to-layer direction of what gets recorded.
        """
        layers = [FakeLayer(0), FakeLayer(1)]
        monitor = _monitor(layers, log_global=False)
        _run_backward(layers, paddle.ones([2, WIDTH]))
        monitor.step()

        latest = training_logs.get_latest(prefix="grad_health")
        self.assertGreater(
            latest["grad_health/layer_0/layer_out_norm"],
            latest["grad_health/layer_1/layer_out_norm"],
        )
