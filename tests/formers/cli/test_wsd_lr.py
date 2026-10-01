# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""Behavior tests for ``get_wsd_schedule_with_warmup``.

Module under test:
    paddlefleet.cli.train.ernie_pretrain.src.lr_schedulers.wsd_lr

Independent oracle
------------------
The production factory returns a ``paddle.optimizer.lr.LambdaDecay`` whose
emitted learning rate at ``step`` is ``base_lr * lr_lambda(step)``. We verify
the *actual* learning rate (base_lr multiplied by the production ``lr_lambda``
closure) against values derived BY HAND from the piecewise WSD definition and
hard-coded as literal expectations. The oracle is never produced by calling the
scheduler under test.

Let ``lr`` be the base learning rate, ``W`` warmup steps, ``T`` total steps,
``S`` steady steps, ``m`` min_lr, and fixed ``base = 0.05``. The closed form:

    step < W (linear warmup, ramp ignores min_lr):
        out = step / max(1, W)

    W <= step < S (steady / stable plateau):
        out = 1.0

    step >= S (decay, with num_decay = T - S, p = (step - S)/max(1, num_decay)):
        half_life: ratio = base**p
                   norm  = (ratio - base) / (1 - base)
                   out   = norm * (1 - m/lr) + m/lr
        1-sqrt:    ratio = 1 - sqrt(p)
                   out   = ratio * (1 - m/lr) + m/lr

Chosen non-degenerate anchors (multiplier -> effective lr), reduced so the
decay lands on clean progress fractions:

    A) lr=2.0, W=10, T=100, S=80, half_life, m=0:
        step   0 -> ramp 0.0                 -> 0.0    (warmup start)
        step   5 -> ramp 0.5                 -> 1.0    (warmup mid)
        step   9 -> ramp 0.9                 -> 1.8    (warmup, still < peak)
        step  10 -> steady 1.0               -> 2.0    (warmup->steady seam)
        step  40 -> steady 1.0               -> 2.0    (steady interior)
        step  79 -> steady 1.0               -> 2.0    (last steady step)
        step  80 -> p=0,   norm=1.0          -> 2.0    (steady->decay seam)
        step  90 -> p=1/2, norm=0.18274399.. -> 0.365488.. (mid decay)
        step 100 -> p=1,   norm=0.0          -> 0.0    (schedule end)

    B) lr=4.0, W=0, T=200, S=100, 1-sqrt, m=0.4 (floor m/lr = 0.1):
        step   0 -> steady 1.0               -> 4.0    (W=0 => no warmup)
        step  99 -> steady 1.0               -> 4.0    (last steady step)
        step 100 -> p=0,   ratio=1           -> 4.0    (decay seam)
        step 125 -> p=1/4, ratio=0.5 -> 0.55 -> 2.2    (mid decay)
        step 200 -> p=1,   ratio=0   -> 0.1  -> 0.4    (== min_lr floor)

    C) lr=1.0, W=0, T=1000, default S (=0.9*T=900), half_life, m=0:
        step 899 -> steady 1.0               -> 1.0    (default S not yet hit)
        step 950 -> p=1/2, norm=0.18274399.. -> 0.18274399.. (decay w/ S=900)

Paddle is an optional heavy dependency and the production module imports it at
top level; every test therefore skips when the import fails (this environment
has no paddle installed).
"""

import math
import unittest

try:
    from paddle.optimizer.lr import LambdaDecay

    from paddlefleet.cli.train.ernie_pretrain.src.lr_schedulers.wsd_lr import (
        get_wsd_schedule_with_warmup,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle (or the module) unavailable in this env
    LambdaDecay = None
    get_wsd_schedule_with_warmup = None
    _IMPORT_ERROR = exc


class WsdScheduleWithWarmupTest(unittest.TestCase):
    """Verify the actual LR emitted at warmup, steady and decay points."""

    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(
                f"paddle/wsd_lr import unavailable: {_IMPORT_ERROR!r}"
            )

    def _lr_at(self, scheduler, step):
        """Actual learning rate the LambdaDecay scheduler yields at ``step``.

        LambdaDecay.get_lr() returns ``base_lr * lr_lambda(last_epoch)``; we
        reproduce that composition explicitly so every anchor below is a real
        end-to-end LR check (base_lr scaling included), not a bare factor check.
        """
        return scheduler.base_lr * scheduler.lr_lambda(step)

    def test_scheduler_is_lambda_decay_carrying_base_lr(self):
        """Structural anchor: factory yields a LambdaDecay holding base_lr."""
        scheduler = get_wsd_schedule_with_warmup(
            learning_rate=2.0,
            num_warmup_steps=10,
            num_training_steps=100,
            num_steady_steps=80,
        )
        self.assertIsInstance(scheduler, LambdaDecay)
        self.assertAlmostEqual(scheduler.base_lr, 2.0, places=12)

    def test_warmup_is_linear_ramp_scaled_by_base_lr(self):
        """Warmup ramps linearly; base_lr=2.0 => steps 0/5/9 -> 0.0/1.0/1.8."""
        scheduler = get_wsd_schedule_with_warmup(
            learning_rate=2.0,
            num_warmup_steps=10,
            num_training_steps=100,
            num_steady_steps=80,
        )
        self.assertAlmostEqual(self._lr_at(scheduler, 0), 0.0, places=12)
        self.assertAlmostEqual(self._lr_at(scheduler, 5), 1.0, places=12)
        self.assertAlmostEqual(self._lr_at(scheduler, 9), 1.8, places=12)
        # The ramp has not yet reached the peak one step before warmup ends.
        self.assertLess(self._lr_at(scheduler, 9), self._lr_at(scheduler, 10))

    def test_steady_phase_is_flat_at_peak(self):
        """Steady window [W, S) holds the peak lr (=base_lr) at every step."""
        scheduler = get_wsd_schedule_with_warmup(
            learning_rate=2.0,
            num_warmup_steps=10,
            num_training_steps=100,
            num_steady_steps=80,
        )
        # Warmup->steady seam (step == W leaves the warmup branch).
        self.assertAlmostEqual(self._lr_at(scheduler, 10), 2.0, places=12)
        self.assertAlmostEqual(self._lr_at(scheduler, 40), 2.0, places=12)
        # Last step still inside the steady window (step == S-1).
        self.assertAlmostEqual(self._lr_at(scheduler, 79), 2.0, places=12)

    def test_half_life_decay_multiple_points(self):
        """half_life decay across p=0, p=1/2, p=1 with hand-derived values."""
        scheduler = get_wsd_schedule_with_warmup(
            learning_rate=2.0,
            num_warmup_steps=10,
            num_training_steps=100,
            num_steady_steps=80,
            decay_function="half_life",
            min_lr=0.0,
        )
        # Steady->decay seam: p=0 => base**0=1 => norm=(1-0.05)/0.95=1 => peak.
        self.assertAlmostEqual(self._lr_at(scheduler, 80), 2.0, places=12)
        # Mid decay p=1/2: norm=(sqrt(0.05)-0.05)/0.95, scaled by base_lr 2.0.
        expected_mid = 2.0 * ((math.sqrt(0.05) - 0.05) / (1.0 - 0.05))
        self.assertAlmostEqual(expected_mid, 0.3654879952631136, places=12)
        self.assertAlmostEqual(
            self._lr_at(scheduler, 90), expected_mid, places=12
        )
        # Schedule end p=1: base**1=0.05 => norm=0 => lr collapses to 0.
        self.assertAlmostEqual(self._lr_at(scheduler, 100), 0.0, places=12)
        # Decay is strictly monotonically decreasing across the three points.
        self.assertGreater(
            self._lr_at(scheduler, 80), self._lr_at(scheduler, 90)
        )
        self.assertGreater(
            self._lr_at(scheduler, 90), self._lr_at(scheduler, 100)
        )

    def test_one_sqrt_decay_with_min_lr_floor(self):
        """1-sqrt decay honors the min_lr floor; p=0/1/4/1 -> 4.0/2.2/0.4."""
        scheduler = get_wsd_schedule_with_warmup(
            learning_rate=4.0,
            num_warmup_steps=0,
            num_training_steps=200,
            num_steady_steps=100,
            decay_function="1-sqrt",
            min_lr=0.4,
        )
        # W=0 => step 0 is already steady; steady holds peak base_lr.
        self.assertAlmostEqual(self._lr_at(scheduler, 0), 4.0, places=12)
        self.assertAlmostEqual(self._lr_at(scheduler, 99), 4.0, places=12)
        # Decay seam p=0: ratio=1-sqrt(0)=1 => still peak.
        self.assertAlmostEqual(self._lr_at(scheduler, 100), 4.0, places=12)
        # Mid decay p=1/4: ratio=1-0.5=0.5 => 0.5*(1-0.1)+0.1=0.55 => 4*0.55.
        self.assertAlmostEqual(self._lr_at(scheduler, 125), 2.2, places=12)
        # End p=1: ratio=0 => floor m/lr=0.1 => effective lr == min_lr 0.4.
        self.assertAlmostEqual(self._lr_at(scheduler, 200), 0.4, places=12)

    def test_default_num_steady_steps_is_ninety_percent(self):
        """Unset num_steady_steps defaults to 0.9 * num_training_steps (=900)."""
        scheduler = get_wsd_schedule_with_warmup(
            learning_rate=1.0,
            num_warmup_steps=0,
            num_training_steps=1000,
        )
        # step 899 is still steady only if the default steady boundary is 900;
        # a smaller default would have entered decay here (lr < 1.0).
        self.assertAlmostEqual(self._lr_at(scheduler, 899), 1.0, places=12)
        # step 950 sits at p=(950-900)/(1000-900)=1/2 of the default half_life
        # decay; the value pins both the 900 boundary and num_decay=100.
        expected_mid = (math.sqrt(0.05) - 0.05) / (1.0 - 0.05)
        self.assertAlmostEqual(expected_mid, 0.1827439976315568, places=12)
        self.assertAlmostEqual(
            self._lr_at(scheduler, 950), expected_mid, places=12
        )

    def test_invalid_decay_function_only_raises_inside_decay(self):
        """An invalid decay_function is inert until the decay branch runs."""
        scheduler = get_wsd_schedule_with_warmup(
            learning_rate=1.0,
            num_warmup_steps=10,
            num_training_steps=100,
            num_steady_steps=80,
            decay_function="not-a-real-decay",
        )
        # Warmup and steady branches return before touching decay_function.
        self.assertAlmostEqual(scheduler.lr_lambda(5), 0.5, places=12)
        self.assertAlmostEqual(scheduler.lr_lambda(50), 1.0, places=12)
        # Reaching the decay phase surfaces the ValueError from the bad name.
        with self.assertRaises(ValueError):
            scheduler.lr_lambda(90)

    def test_num_cycles_is_currently_inert(self):
        """num_cycles is accepted but unused: it does not alter any emitted lr.

        This asserts the *current* production behavior (the parameter is dead in
        wsd_lr.py, unlike cosine_lr.py where num_cycles drives the schedule).
        It is documented here so a future wiring-in of num_cycles updates this
        expectation rather than silently changing the schedule shape.
        """
        common = {
            "learning_rate": 2.0,
            "num_warmup_steps": 10,
            "num_training_steps": 100,
            "num_steady_steps": 80,
            "decay_function": "half_life",
        }
        sched_a = get_wsd_schedule_with_warmup(num_cycles=0.5, **common)
        sched_b = get_wsd_schedule_with_warmup(num_cycles=3.0, **common)
        for step in (5, 40, 80, 90, 100):
            self.assertAlmostEqual(
                sched_a.lr_lambda(step),
                sched_b.lr_lambda(step),
                places=12,
                msg=f"num_cycles unexpectedly affected lr at step {step}",
            )


if __name__ == "__main__":
    unittest.main()
