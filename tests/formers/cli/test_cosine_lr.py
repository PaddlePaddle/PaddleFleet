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

"""Behavior tests for ``get_cosine_schedule_with_warmup``.

Module under test:
    paddlefleet.cli.train.ernie_pretrain.src.lr_schedulers.cosine_lr

Independent oracle
------------------
The production factory returns a ``paddle.optimizer.lr.LambdaDecay`` whose
emitted learning rate at ``step`` is ``base_lr * lr_lambda(step)``. We verify
the *actual* learning rate (base_lr multiplied by the production ``lr_lambda``
closure) against learning-rate values that were derived BY HAND from the
piecewise definition below and hard-coded as literal expectations. The oracle
is never produced by calling the scheduler itself.

Let ``lr`` be the base learning rate, ``W`` warmup steps, ``T`` total steps,
``c`` num_cycles, ``m`` min_lr. The closed form is:

    step < W (linear warmup, ramp ignores min_lr):
        lr_out = lr * (step / max(1, W))

    step >= W (cosine decay with min_lr floor):
        p     = (step - W) / max(1, T - W)
        ratio = max(0, 0.5 * (1 + cos(pi * c * 2 * p)))
        lr_out = lr * (ratio * (1 - m/lr) + m/lr)

Chosen non-degenerate anchors (base lr = 2e-4 unless noted), all reduced to
exact rationals so cos hits {1, 0.5, 0, -1} and the expected value is a clean
literal:

    W=100, T=1000, c=0.5, m=0:
        step   30 -> ramp   0.30              -> 6.0e-5   (warmup ramp)
        step   75 -> ramp   0.75              -> 1.5e-4   (warmup ramp)
        step  100 -> p=0,   cos0=1,  ratio=1  -> 2.0e-4   (warmup->decay seam)
        step  400 -> p=1/3, cos60=.5, ratio=.75 -> 1.5e-4 (mid decay)
        step  550 -> p=1/2, cos90=0, ratio=.5 -> 1.0e-4   (mid decay)
        step 1000 -> p=1,   cos180=-1,ratio=0 -> 0.0      (schedule end)

    W=100, T=1000, c=0.5, m=2e-5 (floor = 0.1 of base):
        step 1000 -> ratio=0  -> lr*0.1              = 2.0e-5 (== min_lr)
        step  400 -> ratio=.75 -> lr*(.75*.9+.1)=.775 = 1.55e-4
        step    5 -> ramp 0.05 -> 1.0e-5  (BELOW min_lr: ramp ignores m)

    W=0, T=1000, m=0, mid step 500:
        c=0.5 -> p=.5, cos90=0,  ratio=.5 -> 1.0e-4
        c=1.0 -> p=.5, cos180=-1,ratio=0  -> 0.0   (distinguishes num_cycles)

Paddle is an optional heavy dependency and the production module imports it at
top level; every test therefore skips when the import fails.
"""

import unittest

try:
    from paddlefleet.cli.train.ernie_pretrain.src.lr_schedulers.cosine_lr import (
        get_cosine_schedule_with_warmup,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle (or the module) unavailable in this env
    get_cosine_schedule_with_warmup = None
    _IMPORT_ERROR = exc


class CosineScheduleWithWarmupTest(unittest.TestCase):
    """Verify actual LR emitted at multiple non-degenerate schedule points."""

    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(
                f"paddle/cosine_lr import unavailable: {_IMPORT_ERROR!r}"
            )
        # Anchor the LambdaDecay semantics we rely on: emitted lr at a step is
        # base_lr * lr_lambda(step). Assert structure once so the per-step math
        # below is a genuine actual-LR check rather than a bare factor check.
        from paddle.optimizer.lr import LambdaDecay

        self.LambdaDecay = LambdaDecay

    def _lr_at(self, scheduler, step):
        """Actual learning rate the LambdaDecay scheduler yields at ``step``."""
        return scheduler.base_lr * scheduler.lr_lambda(step)

    def test_scheduler_is_lambda_decay_carrying_base_lr(self):
        """Structural anchor: LambdaDecay stores base_lr used by get_lr()."""
        scheduler = get_cosine_schedule_with_warmup(
            learning_rate=2e-4,
            num_warmup_steps=100,
            num_training_steps=1000,
        )
        self.assertIsInstance(scheduler, self.LambdaDecay)
        self.assertAlmostEqual(scheduler.base_lr, 2e-4, places=12)

    def test_warmup_linear_ramp_midpoints(self):
        """Warmup ramp at steps 30 and 75 -> 6.0e-5 and 1.5e-4 (not step 0)."""
        scheduler = get_cosine_schedule_with_warmup(
            learning_rate=2e-4,
            num_warmup_steps=100,
            num_training_steps=1000,
        )
        self.assertAlmostEqual(self._lr_at(scheduler, 30), 6.0e-5, places=12)
        self.assertAlmostEqual(self._lr_at(scheduler, 75), 1.5e-4, places=12)

    def test_warmup_to_decay_seam_reaches_peak(self):
        """At step==W the decay branch starts at p=0 -> peak lr 2.0e-4."""
        scheduler = get_cosine_schedule_with_warmup(
            learning_rate=2e-4,
            num_warmup_steps=100,
            num_training_steps=1000,
        )
        self.assertAlmostEqual(self._lr_at(scheduler, 100), 2.0e-4, places=12)

    def test_mid_decay_cosine_values(self):
        """Mid-decay at p=1/3 (1.5e-4) and p=1/2 (1.0e-4); interior of decay."""
        scheduler = get_cosine_schedule_with_warmup(
            learning_rate=2e-4,
            num_warmup_steps=100,
            num_training_steps=1000,
        )
        self.assertAlmostEqual(self._lr_at(scheduler, 400), 1.5e-4, places=12)
        self.assertAlmostEqual(self._lr_at(scheduler, 550), 1.0e-4, places=12)

    def test_schedule_end_floor_is_zero_without_min_lr(self):
        """At step==T with min_lr=0 the cosine bottoms out at exactly 0."""
        scheduler = get_cosine_schedule_with_warmup(
            learning_rate=2e-4,
            num_warmup_steps=100,
            num_training_steps=1000,
        )
        self.assertAlmostEqual(self._lr_at(scheduler, 1000), 0.0, places=12)

    def test_min_lr_floor_and_interpolation(self):
        """min_lr=2e-5: end == min_lr exactly; mid-decay interpolates to 1.55e-4."""
        scheduler = get_cosine_schedule_with_warmup(
            learning_rate=2e-4,
            num_warmup_steps=100,
            num_training_steps=1000,
            min_lr=2e-5,
        )
        self.assertAlmostEqual(self._lr_at(scheduler, 1000), 2.0e-5, places=12)
        self.assertAlmostEqual(self._lr_at(scheduler, 400), 1.55e-4, places=12)

    def test_warmup_ramp_ignores_min_lr_floor(self):
        """Documented subtlety: early warmup dips BELOW min_lr (ramp ignores m).

        At step 5 the ramp yields 1.0e-5, strictly below the 2.0e-5 floor that
        the decay tail respects. This verifies the actual value and the branch
        divergence rather than assuming min_lr is a global lower bound.
        """
        scheduler = get_cosine_schedule_with_warmup(
            learning_rate=2e-4,
            num_warmup_steps=100,
            num_training_steps=1000,
            min_lr=2e-5,
        )
        lr_at_5 = self._lr_at(scheduler, 5)
        self.assertAlmostEqual(lr_at_5, 1.0e-5, places=12)
        self.assertLess(lr_at_5, 2e-5)

    def test_num_cycles_changes_mid_decay_value(self):
        """At the same step, num_cycles=0.5 -> 1.0e-4 but 1.0 -> 0.0 (distinct)."""
        half = get_cosine_schedule_with_warmup(
            learning_rate=2e-4,
            num_warmup_steps=0,
            num_training_steps=1000,
            num_cycles=0.5,
        )
        full = get_cosine_schedule_with_warmup(
            learning_rate=2e-4,
            num_warmup_steps=0,
            num_training_steps=1000,
            num_cycles=1.0,
        )
        lr_half = self._lr_at(half, 500)
        lr_full = self._lr_at(full, 500)
        self.assertAlmostEqual(lr_half, 1.0e-4, places=12)
        self.assertAlmostEqual(lr_full, 0.0, places=12)
        self.assertNotAlmostEqual(lr_half, lr_full, places=8)

    @unittest.expectedFailure
    def test_post_schedule_stays_at_floor_CONTRACT(self):
        """Docstring says lr decreases 'to 0'; beyond T it must stay at floor.

        Expected-failure: with num_cycles=0.5 and no clamp on ``progress``, at
        step 1450 (p=1.5, cos(1.5*pi)=0) the actual lr rebounds to 1.0e-4 after
        having reached 0 at step 1000. Asserting the intended monotone-to-floor
        contract exposes this rebound. No production code is modified.
        """
        scheduler = get_cosine_schedule_with_warmup(
            learning_rate=2e-4,
            num_warmup_steps=100,
            num_training_steps=1000,
        )
        self.assertAlmostEqual(self._lr_at(scheduler, 1450), 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
