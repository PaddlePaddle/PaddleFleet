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

"""Behaviour tests for paddlefleet.transformers.optimization LR schedulers.

These tests exercise the REAL schedulers and compare their emitted learning
rate against learning-rate values derived independently BY HAND (closed-form
warmup / cosine / linear formulas evaluated off-line, hard-coded as literals).
The scheduler's own ``get_lr`` is never used to build the expected values.

No-card (CPU) execution: the schedulers subclass ``paddle.optimizer.lr.
LRScheduler`` which advances ``last_epoch`` on every ``step()`` and recomputes
``get_lr`` on the current ``last_epoch``. All logic here is pure Python math
runnable on CPU; no accelerator is required.
"""

import unittest

from paddlefleet.transformers.optimization import (
    CosineAnnealingWithWarmupDecay,
    LinearAnnealingWithWarmupDecay,
    is_integer,
)


def advance_to(scheduler, target_last_epoch):
    """Step the real scheduler until ``last_epoch == target_last_epoch``.

    A freshly constructed LRScheduler (last_epoch=-1) runs one internal
    ``step()`` in ``__init__`` and lands on ``last_epoch == 0``. Each further
    ``step()`` increments ``last_epoch`` by one. We drive the genuine
    ``step()`` loop rather than mutating state so the production advance path
    stays on the validation chain.
    """
    while scheduler.last_epoch < target_last_epoch:
        scheduler.step()
    return float(scheduler.get_lr())


class TestIsInteger(unittest.TestCase):
    """`is_integer` is a thin ``isinstance(x, int)`` predicate."""

    def test_plain_int_is_integer(self):
        self.assertTrue(is_integer(5))
        self.assertTrue(is_integer(-3))
        self.assertTrue(is_integer(0))

    def test_float_is_not_integer(self):
        self.assertFalse(is_integer(3.14))
        # A float that is numerically whole is still NOT an int instance.
        self.assertFalse(is_integer(4.0))

    def test_bool_is_integer_subclass(self):
        # bool subclasses int in Python, so this predicate reports True.
        self.assertTrue(is_integer(True))
        self.assertTrue(is_integer(False))

    def test_non_numeric_is_not_integer(self):
        self.assertFalse(is_integer("5"))
        self.assertFalse(is_integer(None))
        self.assertFalse(is_integer([1]))


# ---------------------------------------------------------------------------
# Independent (hand-derived) reference constants.
#
# Warmup branch (warmup_step > 0 and last_epoch <= warmup_step):
#       lr = max_lr * last_epoch / warmup_step        (note: min_lr is ignored,
#                                                       warmup starts from 0)
# Cosine decay branch (warmup_step < last_epoch <= decay_step):
#       r     = (last_epoch - warmup_step) / (decay_step - warmup_step)
#       coeff = 0.5 * (cos(pi * r) + 1)
#       lr    = min_lr + coeff * (max_lr - min_lr)
# Linear decay branch:
#       coeff = 1 - r
#       lr    = min_lr + coeff * (max_lr - min_lr)
# Post-decay (last_epoch > decay_step):  lr = min_lr
#
# cos(pi/4)  =  0.70710678118654752  -> cosine coeff @ r=0.25 = 0.85355339059
# cos(pi/2)  =  0.0                  -> cosine coeff @ r=0.50 = 0.50000000000
# cos(3pi/4) = -0.70710678118654752  -> cosine coeff @ r=0.75 = 0.14644660941
# ---------------------------------------------------------------------------
COS_COEFF_QUARTER = 0.8535533905932738
COS_COEFF_HALF = 0.5
COS_COEFF_THREE_QUARTER = 0.14644660940672624


class TestCosineAnnealingWithWarmupDecay(unittest.TestCase):
    """Cosine warmup-decay schedule, verified at many non-degenerate steps."""

    def test_warmup_is_linear_ramp_from_zero(self):
        # max_lr=1, min_lr=0, warmup=100, decay=1000.
        # lr = last_epoch / 100 across the whole warmup ramp.
        sched = CosineAnnealingWithWarmupDecay(
            max_lr=1.0, min_lr=0.0, warmup_step=100, decay_step=1000
        )
        self.assertAlmostEqual(advance_to(sched, 0), 0.0, places=7)
        sched = CosineAnnealingWithWarmupDecay(
            max_lr=1.0, min_lr=0.0, warmup_step=100, decay_step=1000
        )
        self.assertAlmostEqual(advance_to(sched, 25), 0.25, places=7)
        sched = CosineAnnealingWithWarmupDecay(
            max_lr=1.0, min_lr=0.0, warmup_step=100, decay_step=1000
        )
        self.assertAlmostEqual(advance_to(sched, 50), 0.50, places=7)
        sched = CosineAnnealingWithWarmupDecay(
            max_lr=1.0, min_lr=0.0, warmup_step=100, decay_step=1000
        )
        self.assertAlmostEqual(advance_to(sched, 75), 0.75, places=7)
        # End of warmup (last_epoch == warmup_step) reaches max_lr.
        sched = CosineAnnealingWithWarmupDecay(
            max_lr=1.0, min_lr=0.0, warmup_step=100, decay_step=1000
        )
        self.assertAlmostEqual(advance_to(sched, 100), 1.0, places=7)

    def test_cosine_decay_shape_at_multiple_steps(self):
        # After warmup, lr follows the cosine coefficient (max_lr=1, min_lr=0).
        # r = (le-100)/900.  le=325 -> r=0.25, le=550 -> r=0.5, le=775 -> r=0.75.
        for le, expected in (
            (325, COS_COEFF_QUARTER),
            (550, COS_COEFF_HALF),
            (775, COS_COEFF_THREE_QUARTER),
        ):
            sched = CosineAnnealingWithWarmupDecay(
                max_lr=1.0, min_lr=0.0, warmup_step=100, decay_step=1000
            )
            self.assertAlmostEqual(advance_to(sched, le), expected, places=6)

    def test_reaches_min_lr_at_and_after_decay_step(self):
        # le=1000 hits r=1.0 -> coeff 0 -> min_lr; le>1000 short-circuits to min_lr.
        sched = CosineAnnealingWithWarmupDecay(
            max_lr=1.0, min_lr=0.0, warmup_step=100, decay_step=1000
        )
        self.assertAlmostEqual(advance_to(sched, 1000), 0.0, places=6)
        sched = CosineAnnealingWithWarmupDecay(
            max_lr=1.0, min_lr=0.0, warmup_step=100, decay_step=1000
        )
        self.assertAlmostEqual(advance_to(sched, 1010), 0.0, places=7)

    def test_nonzero_min_lr_and_scaled_amplitude(self):
        # max_lr=5, min_lr=1: decay lr = 1 + 4*coeff.
        sched = CosineAnnealingWithWarmupDecay(
            max_lr=5.0, min_lr=1.0, warmup_step=100, decay_step=1000
        )
        self.assertAlmostEqual(
            advance_to(sched, 550), 1.0 + 4.0 * COS_COEFF_HALF, places=6
        )
        sched = CosineAnnealingWithWarmupDecay(
            max_lr=5.0, min_lr=1.0, warmup_step=100, decay_step=1000
        )
        self.assertAlmostEqual(
            advance_to(sched, 775),
            1.0 + 4.0 * COS_COEFF_THREE_QUARTER,
            places=6,
        )
        # Past decay -> exactly min_lr, not 0.
        sched = CosineAnnealingWithWarmupDecay(
            max_lr=5.0, min_lr=1.0, warmup_step=100, decay_step=1000
        )
        self.assertAlmostEqual(advance_to(sched, 1005), 1.0, places=6)

    def test_warmup_ignores_min_lr_floor(self):
        # Documents real behaviour: warmup ramps from 0, so during early warmup
        # the lr is BELOW min_lr. le=10 -> 5*10/100 = 0.5, which is < min_lr=1.0.
        sched = CosineAnnealingWithWarmupDecay(
            max_lr=5.0, min_lr=1.0, warmup_step=100, decay_step=1000
        )
        lr = advance_to(sched, 10)
        self.assertAlmostEqual(lr, 0.5, places=7)
        self.assertLess(lr, 1.0)

    def test_no_warmup_starts_at_max_and_cosine_decays(self):
        # warmup_step=0 disables the warmup branch; r = le/1000 from step 0.
        for le, expected in (
            (0, 1.0),
            (250, COS_COEFF_QUARTER),
            (500, COS_COEFF_HALF),
            (750, COS_COEFF_THREE_QUARTER),
            (1000, 0.0),
        ):
            sched = CosineAnnealingWithWarmupDecay(
                max_lr=1.0, min_lr=0.0, warmup_step=0, decay_step=1000
            )
            self.assertAlmostEqual(advance_to(sched, le), expected, places=6)

    def test_constructor_records_configuration(self):
        sched = CosineAnnealingWithWarmupDecay(
            max_lr=2.0, min_lr=0.1, warmup_step=30, decay_step=300
        )
        self.assertEqual(sched.max_lr, 2.0)
        self.assertEqual(sched.min_lr, 0.1)
        self.assertEqual(sched.warmup_step, 30)
        self.assertEqual(sched.decay_step, 300)


class TestLinearAnnealingWithWarmupDecay(unittest.TestCase):
    """Linear warmup-decay schedule, verified at many non-degenerate steps."""

    def test_warmup_is_linear_ramp_from_zero(self):
        sched = LinearAnnealingWithWarmupDecay(
            max_lr=1.0, min_lr=0.0, warmup_step=100, decay_step=1000
        )
        self.assertAlmostEqual(advance_to(sched, 0), 0.0, places=7)
        sched = LinearAnnealingWithWarmupDecay(
            max_lr=1.0, min_lr=0.0, warmup_step=100, decay_step=1000
        )
        self.assertAlmostEqual(advance_to(sched, 40), 0.40, places=7)
        sched = LinearAnnealingWithWarmupDecay(
            max_lr=1.0, min_lr=0.0, warmup_step=100, decay_step=1000
        )
        self.assertAlmostEqual(advance_to(sched, 100), 1.0, places=7)

    def test_linear_decay_shape_at_multiple_steps(self):
        # After warmup lr = coeff = 1 - r, r = (le-100)/900.
        # le=325 -> r=0.25 -> 0.75, le=550 -> 0.5, le=775 -> r=0.75 -> 0.25.
        for le, expected in ((325, 0.75), (550, 0.5), (775, 0.25)):
            sched = LinearAnnealingWithWarmupDecay(
                max_lr=1.0, min_lr=0.0, warmup_step=100, decay_step=1000
            )
            self.assertAlmostEqual(advance_to(sched, le), expected, places=6)

    def test_reaches_min_lr_at_and_after_decay_step(self):
        sched = LinearAnnealingWithWarmupDecay(
            max_lr=1.0, min_lr=0.0, warmup_step=100, decay_step=1000
        )
        self.assertAlmostEqual(advance_to(sched, 1000), 0.0, places=6)
        sched = LinearAnnealingWithWarmupDecay(
            max_lr=1.0, min_lr=0.0, warmup_step=100, decay_step=1000
        )
        self.assertAlmostEqual(advance_to(sched, 1010), 0.0, places=7)

    def test_nonzero_min_lr_and_scaled_amplitude(self):
        # max_lr=5, min_lr=1: decay lr = 1 + 4*(1-r).
        for le, expected in ((325, 4.0), (550, 3.0), (775, 2.0)):
            sched = LinearAnnealingWithWarmupDecay(
                max_lr=5.0, min_lr=1.0, warmup_step=100, decay_step=1000
            )
            self.assertAlmostEqual(advance_to(sched, le), expected, places=6)
        sched = LinearAnnealingWithWarmupDecay(
            max_lr=5.0, min_lr=1.0, warmup_step=100, decay_step=1000
        )
        self.assertAlmostEqual(advance_to(sched, 1005), 1.0, places=6)

    def test_warmup_ignores_min_lr_floor(self):
        sched = LinearAnnealingWithWarmupDecay(
            max_lr=5.0, min_lr=1.0, warmup_step=100, decay_step=1000
        )
        lr = advance_to(sched, 10)
        self.assertAlmostEqual(lr, 0.5, places=7)
        self.assertLess(lr, 1.0)

    def test_no_warmup_starts_at_max_and_linearly_decays(self):
        for le, expected in (
            (0, 1.0),
            (250, 0.75),
            (500, 0.5),
            (750, 0.25),
            (1000, 0.0),
        ):
            sched = LinearAnnealingWithWarmupDecay(
                max_lr=1.0, min_lr=0.0, warmup_step=0, decay_step=1000
            )
            self.assertAlmostEqual(advance_to(sched, le), expected, places=6)

    def test_constructor_records_configuration(self):
        sched = LinearAnnealingWithWarmupDecay(
            max_lr=2.0, min_lr=0.1, warmup_step=30, decay_step=300
        )
        self.assertEqual(sched.max_lr, 2.0)
        self.assertEqual(sched.min_lr, 0.1)
        self.assertEqual(sched.warmup_step, 30)
        self.assertEqual(sched.decay_step, 300)


class TestCosineVsLinearDecayAreDistinct(unittest.TestCase):
    """Guard against either decay collapsing into the other's shape.

    Cosine and linear agree at the midpoint (r=0.5 -> 0.5) but must differ at
    r=0.25 and r=0.75. Comparing the two real schedulers at the same off-mid
    steps would fail if one were (mis)implemented with the other's coefficient.
    """

    def test_quarter_and_three_quarter_points_differ(self):
        cos = CosineAnnealingWithWarmupDecay(
            max_lr=1.0, min_lr=0.0, warmup_step=0, decay_step=1000
        )
        lin = LinearAnnealingWithWarmupDecay(
            max_lr=1.0, min_lr=0.0, warmup_step=0, decay_step=1000
        )
        cos_q = advance_to(cos, 250)
        lin_q = advance_to(lin, 250)
        # Hand-derived: cosine 0.85355..., linear 0.75.
        self.assertAlmostEqual(cos_q, COS_COEFF_QUARTER, places=6)
        self.assertAlmostEqual(lin_q, 0.75, places=6)
        self.assertGreater(cos_q - lin_q, 0.10)

        cos = CosineAnnealingWithWarmupDecay(
            max_lr=1.0, min_lr=0.0, warmup_step=0, decay_step=1000
        )
        lin = LinearAnnealingWithWarmupDecay(
            max_lr=1.0, min_lr=0.0, warmup_step=0, decay_step=1000
        )
        cos_tq = advance_to(cos, 750)
        lin_tq = advance_to(lin, 750)
        # Hand-derived: cosine 0.14645..., linear 0.25.
        self.assertAlmostEqual(cos_tq, COS_COEFF_THREE_QUARTER, places=6)
        self.assertAlmostEqual(lin_tq, 0.25, places=6)
        self.assertGreater(lin_tq - cos_tq, 0.10)

    def test_midpoint_coincides(self):
        cos = CosineAnnealingWithWarmupDecay(
            max_lr=1.0, min_lr=0.0, warmup_step=0, decay_step=1000
        )
        lin = LinearAnnealingWithWarmupDecay(
            max_lr=1.0, min_lr=0.0, warmup_step=0, decay_step=1000
        )
        self.assertAlmostEqual(advance_to(cos, 500), 0.5, places=6)
        self.assertAlmostEqual(advance_to(lin, 500), 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
