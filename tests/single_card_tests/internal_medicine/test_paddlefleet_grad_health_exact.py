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

"""grad_health exact norms: what the collective buys, pinned on one rank.

The reduction itself needs several ranks to exercise, but every property that
can go wrong in the *arithmetic* is visible single-process: with nothing to
reduce, ``ExactNormReducer`` is the identity, so the three exact series must
collapse onto closed-form expressions in the per-microbatch norms. Those
expressions are the contract -- they are what stays true once the collective
starts summing shards and data ranks.
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
except (
    Exception
) as exc:  # pragma: no cover - depends on optional backend install
    raise unittest.SkipTest(f"paddle backend unavailable: {exc}") from exc

_fixtures = importlib.import_module("test_paddlefleet_grad_health")
grad_metrics = importlib.import_module(
    "paddlefleet.internal_medicine.backends.paddlefleet.grad_metrics"
)
grad_monitor = importlib.import_module(
    "paddlefleet.internal_medicine.backends.paddlefleet.grad_monitor"
)
training_logs = importlib.import_module(
    "paddlefleet.internal_medicine.core.training_logs"
).training_logs

FakeLayer = _fixtures.FakeLayer
WIDTH = _fixtures.WIDTH
NUMEL = 2 * WIDTH  # the fixture drives a [2, WIDTH] tensor


def _run(seeds):
    """Backprop each seed as its own microbatch; return (metrics, per-mb norms)."""
    layers = [FakeLayer(0)]
    monitor = _fixtures._monitor(layers, log_global=False)
    for seed in seeds:
        _fixtures._run_backward(layers, seed)
    monitor.step()
    norms = [float(paddle.linalg.norm(s)) for s in seeds]
    return training_logs.get_latest(prefix="grad_health"), norms


def _key(metric):
    return f"grad_health/layer_0/layer_out_{metric}"


class ExactNormSingleRankTest(unittest.TestCase):
    def setUp(self):
        training_logs.reset()

    def tearDown(self):
        training_logs.reset()

    def test_with_nothing_to_reduce_the_exact_series_equal_the_local_ones(self):
        """A layout that shards nothing must not change any number.

        This is the continuity property the whole design rests on: the same key
        means the same thing whether or not the run has SP / CP / DP to sum over.
        """
        latest, norms = _run(
            [paddle.to_tensor([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0]])]
        )
        for metric in ("norm_mb", "norm_global"):
            self.assertAlmostEqual(latest[_key(metric)], norms[0], places=4)
        self.assertAlmostEqual(
            latest[_key("rms_global")], latest[_key("rms")], places=6
        )

    def test_the_five_series_are_five_different_averages(self):
        """Two microbatches separate every definition from every other."""
        seeds = [paddle.full([2, WIDTH], 1.0), paddle.full([2, WIDTH], 3.0)]
        latest, (n0, n1) = _run(seeds)
        sq = n0**2 + n1**2
        self.assertAlmostEqual(latest[_key("norm")], (n0 + n1) / 2, places=4)
        self.assertAlmostEqual(
            latest[_key("norm_mb")], math.sqrt(sq / 2), places=4
        )
        self.assertAlmostEqual(
            latest[_key("norm_global")], math.sqrt(sq), places=4
        )
        self.assertAlmostEqual(
            latest[_key("rms")], (n0 + n1) / 2 / math.sqrt(NUMEL), places=4
        )
        self.assertAlmostEqual(
            latest[_key("rms_global")], math.sqrt(sq / (2 * NUMEL)), places=4
        )

    def test_the_approximate_rms_never_exceeds_the_exact_one(self):
        """Arithmetic mean <= quadratic mean, with equality only when all equal.

        The gap is the whole reason the exact series exists, so pin its sign.
        """
        latest, _ = _run(
            [paddle.full([2, WIDTH], 1.0), paddle.full([2, WIDTH], 5.0)]
        )
        self.assertLess(latest[_key("rms")], latest[_key("rms_global")])
        latest, _ = _run(
            [paddle.full([2, WIDTH], 2.0), paddle.full([2, WIDTH], 2.0)]
        )
        self.assertAlmostEqual(
            latest[_key("rms")], latest[_key("rms_global")], places=5
        )

    def test_the_loss_scale_is_divided_out_of_the_exact_series_too(self):
        """They are sqrt of a sum of squares, i.e. also degree-1 in the gradient."""
        layers = [FakeLayer(0)]
        monitor = _fixtures._monitor(layers, log_global=False)
        seed = paddle.full([2, WIDTH], 2.0)
        _fixtures._run_backward(layers, seed)
        monitor.finalize_scaled_grad_metrics(
            SimpleNamespace(_scale=paddle.to_tensor(8.0))
        )
        monitor.step()
        latest = training_logs.get_latest(prefix="grad_health")
        expected = float(paddle.linalg.norm(seed)) / 8.0
        self.assertAlmostEqual(latest[_key("norm_global")], expected, places=4)
        self.assertAlmostEqual(latest[_key("norm_mb")], expected, places=4)

    def test_the_accumulators_reset_between_steps(self):
        """A leaked sum of squares would make norm_global grow without bound."""
        layers = [FakeLayer(0)]
        monitor = _fixtures._monitor(layers, log_global=False)
        seed = paddle.full([2, WIDTH], 2.0)
        seen = []
        for _ in range(3):
            _fixtures._run_backward(layers, seed)
            monitor.step()
            seen.append(
                training_logs.get_latest(prefix="grad_health")[
                    _key("norm_global")
                ]
            )
        for value in seen[1:]:
            self.assertAlmostEqual(value, seen[0], places=5)

    def test_a_step_without_a_backward_emits_no_exact_series(self):
        layers = [FakeLayer(0)]
        monitor = _fixtures._monitor(layers, log_global=False)
        monitor.step()
        latest = training_logs.get_latest(prefix="grad_health")
        self.assertNotIn(_key("norm_global"), latest)


class ReducerPlanTest(unittest.TestCase):
    def test_without_a_distributed_context_the_reducer_is_the_identity(self):
        reducer = grad_monitor.ExactNormReducer(sequence_parallel=False)
        self.assertFalse(reducer.enabled)
        sq = paddle.to_tensor([4.0, 9.0])
        counts = paddle.to_tensor([2.0, 3.0])
        shard, total, total_count = reducer.reduce(sq, counts)
        for got, want in ((shard, sq), (total, sq), (total_count, counts)):
            self.assertTrue(bool(paddle.all(got == want)))

    def test_tp_is_only_summed_under_sequence_parallel(self):
        """Without SP the same tensor is replicated across TP; summing inflates it."""
        self.assertEqual(
            grad_monitor.ExactNormReducer(sequence_parallel=False).shard_factor,
            1,
        )
        self.assertEqual(
            grad_monitor.ExactNormReducer(sequence_parallel=True).shard_factor,
            1,
        )


if __name__ == "__main__":
    unittest.main()
