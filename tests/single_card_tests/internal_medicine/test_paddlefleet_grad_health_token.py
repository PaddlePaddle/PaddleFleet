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

"""grad_health token axis: telling "one loud token" from "one runaway cell".

``abs_max`` returns the largest single element, which is the same number whether
a whole token carries a huge gradient or one cell ran away -- and those two want
opposite fixes (look at the data vs look at the numerics). These tests pin the
property that makes the token series worth its keys: the two cases must separate.
"""

import importlib
import math
import sys
import unittest
from pathlib import Path

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
training_logs = importlib.import_module(
    "paddlefleet.internal_medicine.core.training_logs"
).training_logs

FakeLayer = _fixtures.FakeLayer
WIDTH = _fixtures.WIDTH

TOKENS = 64
CHANNELS = 8
QUIET = 0.01


def _grad(loud_tokens=(), loud_cells=(), loud=1.0):
    """A [TOKENS, CHANNELS] gradient with hand-placed outliers."""
    value = paddle.full([TOKENS, CHANNELS], QUIET, dtype="float32").numpy()
    for t in loud_tokens:
        value[t] = loud
    for t, c in loud_cells:
        value[t][c] = loud
    return paddle.to_tensor(value)


class TokenStatsMathTest(unittest.TestCase):
    def test_the_median_ignores_loss_masked_tokens(self):
        """Masked rows are exact zeros; letting them into the median broke this.

        Measured 2000-19000 on a real 8k run before the fix, because the median sat
        at ~0 and the ratio was really reporting the masked fraction. The peakiness
        must now be the same number whatever share of the batch is masked.
        """
        for masked in (0, TOKENS // 2, int(TOKENS * 0.9)):
            value = paddle.full(
                [TOKENS, CHANNELS], QUIET, dtype="float32"
            ).numpy()
            value[:masked] = 0.0
            value[TOKENS - 1] = 1.0
            stats = grad_metrics.grad_token_stats(paddle.to_tensor(value))
            self.assertAlmostEqual(
                float(stats["token_norm_ratio"]),
                1.0 / QUIET,
                places=2,
                msg=f"masked={masked}",
            )

    def test_zero_ratio_reports_the_masked_share_on_its_own(self):
        value = paddle.full([TOKENS, CHANNELS], QUIET, dtype="float32").numpy()
        value[: TOKENS // 4] = 0.0
        stats = grad_metrics.grad_token_stats(paddle.to_tensor(value))
        self.assertAlmostEqual(float(stats["token_zero_ratio"]), 0.25, places=6)

    def test_an_all_masked_microbatch_stays_finite(self):
        """``nanmedian`` of nothing is NaN; a NaN here would poison the step."""
        stats = grad_metrics.grad_token_stats(paddle.zeros([TOKENS, CHANNELS]))
        for name, value in stats.items():
            self.assertFalse(bool(paddle.isnan(value)), msg=name)
        self.assertAlmostEqual(float(stats["token_zero_ratio"]), 1.0, places=6)

    def test_the_four_series_match_closed_form(self):
        stats = grad_metrics.grad_token_stats(_grad(loud_tokens=(5, 17, 40)))
        self.assertAlmostEqual(
            float(stats["token_norm_max"]), math.sqrt(CHANNELS), places=5
        )
        self.assertAlmostEqual(
            float(stats["token_norm_ratio"]), 1.0 / QUIET, places=3
        )
        self.assertAlmostEqual(
            float(stats["token_outlier_ratio"]), 3 / TOKENS, places=6
        )

    def test_a_uniform_gradient_has_no_peak_and_no_outliers(self):
        stats = grad_metrics.grad_token_stats(
            paddle.full([TOKENS, CHANNELS], 0.5)
        )
        self.assertAlmostEqual(float(stats["token_norm_ratio"]), 1.0, places=5)
        self.assertAlmostEqual(
            float(stats["token_outlier_ratio"]), 0.0, places=6
        )

    def test_one_loud_token_and_one_loud_cell_are_told_apart(self):
        """The reason this family exists: ``abs_max`` cannot separate these.

        The two token series split the phenomenon in two: ``token_norm_ratio``
        answers *how big* (100x vs 35x here, because a loud cell only lifts its
        token's norm by one channel's worth) and ``token_outlier_ratio`` answers
        *how many* -- one token either way, so it is deliberately equal.
        """
        by_token, _sq, _n = grad_metrics.grad_square_and_stats(
            _grad(loud_tokens=(5,))
        )
        by_cell, _sq2, _n2 = grad_metrics.grad_square_and_stats(
            _grad(loud_cells=((5, 3),))
        )
        self.assertAlmostEqual(
            float(by_token["abs_max"]), float(by_cell["abs_max"]), places=6
        )
        self.assertGreater(
            float(by_token["token_norm_ratio"]),
            float(by_cell["token_norm_ratio"]) * 2,
        )
        self.assertAlmostEqual(
            float(by_token["token_outlier_ratio"]),
            float(by_cell["token_outlier_ratio"]),
            places=6,
        )

    def test_p99_sits_between_the_median_and_the_max(self):
        stats = grad_metrics.grad_token_stats(
            _grad(loud_tokens=tuple(range(3)))
        )
        self.assertLessEqual(
            float(stats["token_norm_p99"]),
            float(stats["token_norm_max"]) + 1e-6,
        )
        self.assertGreater(
            float(stats["token_norm_p99"]), QUIET * math.sqrt(CHANNELS)
        )

    def test_the_leading_axes_are_flattened_not_interpreted(self):
        """``[B, S, H]`` and ``[B*S, H]`` describe the same tokens."""
        flat = _grad(loud_tokens=(1, 2))
        nested = flat.reshape([2, TOKENS // 2, CHANNELS])
        for name, value in grad_metrics.grad_token_stats(flat).items():
            self.assertAlmostEqual(
                float(value),
                float(grad_metrics.grad_token_stats(nested)[name]),
                places=6,
                msg=name,
            )


class TokenStatsThroughTheMonitorTest(unittest.TestCase):
    def setUp(self):
        training_logs.reset()

    def tearDown(self):
        training_logs.reset()

    def test_every_token_series_reaches_the_log(self):
        layers = [FakeLayer(0)]
        monitor = _fixtures._monitor(layers, log_global=False)
        _fixtures._run_backward(layers, paddle.ones([2, WIDTH]))
        monitor.step()
        latest = training_logs.get_latest(prefix="grad_health")
        for metric in grad_metrics.TOKEN_METRICS:
            self.assertIn(f"grad_health/layer_0/layer_out_{metric}", latest)

    def test_a_uniform_seed_yields_a_flat_token_profile(self):
        """Every token gets the same gradient here, so peakiness must be 1."""
        layers = [FakeLayer(0)]
        monitor = _fixtures._monitor(layers, log_global=False)
        _fixtures._run_backward(layers, paddle.ones([2, WIDTH]))
        monitor.step()
        latest = training_logs.get_latest(prefix="grad_health")
        self.assertAlmostEqual(
            latest["grad_health/layer_0/layer_out_token_norm_ratio"],
            1.0,
            places=4,
        )
        self.assertAlmostEqual(
            latest["grad_health/layer_0/layer_out_token_outlier_ratio"],
            0.0,
            places=6,
        )

    def test_the_loud_microbatch_survives_the_step(self):
        """``token_norm_ratio`` is max-aggregated, so a spike is not averaged away.

        Driven with ``TOKENS`` rows rather than the shared 2-row fixture: the
        statistic is a ratio to the *median*, so with only two tokens the outlier
        drags the median halfway up and the ratio saturates near 2 no matter how
        loud the spike is. Real runs have thousands of tokens per microbatch, and
        this is the property that needs pinning.
        """
        layers = [FakeLayer(0)]
        monitor = _fixtures._monitor(layers, log_global=False)
        quiet = paddle.ones([TOKENS, WIDTH])
        loud = quiet.numpy()
        loud[0] = 100.0
        for seed in (quiet, paddle.to_tensor(loud)):
            x = paddle.ones([TOKENS, WIDTH])
            x.stop_gradient = False
            (layers[0](x) * seed).sum().backward()
        monitor.step()
        latest = training_logs.get_latest(prefix="grad_health")
        self.assertGreater(
            latest["grad_health/layer_0/layer_out_token_norm_ratio"], 50.0
        )
        # And the contrast: the fraction is mean-aggregated, so the quiet
        # microbatch's zero halves it -- one outlier token out of 2 x TOKENS seen.
        self.assertAlmostEqual(
            latest["grad_health/layer_0/layer_out_token_outlier_ratio"],
            1 / (2 * TOKENS),
            places=5,
        )


if __name__ == "__main__":
    unittest.main()
