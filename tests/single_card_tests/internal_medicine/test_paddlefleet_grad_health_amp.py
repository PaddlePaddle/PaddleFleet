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

"""grad_health under AMP: the loss scale must not reach the curves.

Split from ``test_paddlefleet_grad_health`` (whose fixtures this reuses) because
these cases are about the step lifecycle rather than a metric's value: a gradient
hook sees the *scaled* gradient, and ``finalize_scaled_grad_metrics`` is the only
thing standing between that and the logged number.
"""

import importlib
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
training_logs = importlib.import_module(
    "paddlefleet.internal_medicine.core.training_logs"
).training_logs

FakeLayer = _fixtures.FakeLayer
WIDTH = _fixtures.WIDTH
SCALE = 8.0


class GradHealthAmpTest(unittest.TestCase):
    def setUp(self):
        training_logs.reset()

    def tearDown(self):
        training_logs.reset()

    def _run(self, finalize_calls, steps=1):
        """Drive ``steps`` full steps, finalizing ``finalize_calls`` times in each."""
        layers = [FakeLayer(0)]
        monitor = _fixtures._monitor(layers, log_global=False)
        seed = paddle.full([2, WIDTH], 2.0)
        for _ in range(steps):
            _fixtures._run_backward(layers, seed)
            for _ in range(finalize_calls):
                # Minimal GradScaler stand-in: only ``_scale`` is ever read.
                monitor.finalize_scaled_grad_metrics(
                    SimpleNamespace(_scale=paddle.to_tensor(SCALE))
                )
            monitor.step()
        return training_logs.get_latest(prefix="grad_health")[
            "grad_health/layer_0/layer_out_norm"
        ]

    def _raw_norm(self):
        return float(paddle.linalg.norm(paddle.full([2, WIDTH], 2.0)))

    def test_the_loss_scale_is_divided_out(self):
        self.assertAlmostEqual(self._run(1), self._raw_norm() / SCALE, places=4)

    def test_finalizing_twice_in_one_step_divides_once(self):
        """``on_optimizer_begin`` and the ``_flush_buffers`` fallback both fire."""
        self.assertAlmostEqual(self._run(2), self._raw_norm() / SCALE, places=4)

    def test_the_next_step_is_de_scaled_again(self):
        """The latch has to reset at flush, or step 2 keeps the raw scaled value."""
        self.assertAlmostEqual(
            self._run(1, steps=2), self._raw_norm() / SCALE, places=4
        )

    def test_a_run_without_a_scaler_is_left_alone(self):
        """Non-AMP runs and direct users must not have their values touched."""
        self.assertAlmostEqual(self._run(0), self._raw_norm(), places=4)


if __name__ == "__main__":
    unittest.main()
