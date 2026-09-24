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

"""Behavior tests for GCCallback.

GCCallback controls Python's cyclic garbage collector during training:
- on_train_begin: disables gc up front when gc_interval > 0, so that no
  implicit collection runs mid-step.
- on_step_end: triggers an explicit gc.collect() only on steps that are exact
  multiples of gc_interval (and only when gc_interval > 0).

gc.collect / gc.disable are collaborators, so mocking / observing them is
appropriate here. Tests save and restore the process-wide gc-enabled flag via
addCleanup so a failing assertion cannot leak a disabled collector into other
tests. Expected trigger steps are hand-derived, not read back from the code.
"""

import gc
import unittest
from types import SimpleNamespace
from unittest import mock

try:
    from paddlefleet.cli.train.ernie_pretrain.src.callbacks.gc_callback import (
        GCCallback,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # local env has no paddle; skip honestly
    GCCallback = None
    _IMPORT_ERROR = exc


@unittest.skipUnless(
    GCCallback is not None,
    "GCCallback import failed (paddle not installed in this env): "
    f"{_IMPORT_ERROR}",
)
class GCCallbackBehaviorTest(unittest.TestCase):
    def setUp(self):
        # Restore the global gc-enabled flag after every test, even on failure.
        was_enabled = gc.isenabled()

        def _restore():
            if was_enabled:
                gc.enable()
            else:
                gc.disable()

        self.addCleanup(_restore)
        self.callback = GCCallback()

    @staticmethod
    def _args(gc_interval):
        return SimpleNamespace(gc_interval=gc_interval)

    @staticmethod
    def _state(global_step):
        return SimpleNamespace(global_step=global_step)

    # --- on_train_begin: gc-enabled flag transitions -----------------------

    def test_on_train_begin_disables_gc_when_interval_positive(self):
        gc.enable()
        self.assertTrue(gc.isenabled())  # precondition
        with mock.patch.object(gc, "collect") as collect:
            self.callback.on_train_begin(
                self._args(10), self._state(0), object()
            )
        # gc turned off, and train-begin must not itself collect.
        self.assertFalse(gc.isenabled())
        collect.assert_not_called()

    def test_on_train_begin_keeps_gc_enabled_when_interval_zero(self):
        gc.enable()
        self.callback.on_train_begin(self._args(0), self._state(0), object())
        self.assertTrue(gc.isenabled())

    def test_on_train_begin_leaves_disabled_flag_untouched_when_interval_zero(
        self,
    ):
        # interval == 0 means "no action": a disabled collector stays disabled
        # (rules out an implementation that unconditionally re-enables gc).
        gc.disable()
        self.callback.on_train_begin(self._args(0), self._state(0), object())
        self.assertFalse(gc.isenabled())

    def test_on_train_begin_treats_negative_interval_as_disabled_guard(self):
        # Guard is strictly `> 0`, so a negative interval must not disable gc.
        gc.enable()
        self.callback.on_train_begin(self._args(-1), self._state(0), object())
        self.assertTrue(gc.isenabled())

    # --- on_step_end: gc.collect() firing schedule -------------------------

    def test_on_step_end_collects_on_matching_step(self):
        with mock.patch.object(gc, "collect") as collect:
            self.callback.on_step_end(self._args(10), self._state(10), object())
        self.assertEqual(collect.call_count, 1)

    def test_on_step_end_skips_non_matching_step(self):
        with mock.patch.object(gc, "collect") as collect:
            self.callback.on_step_end(self._args(10), self._state(5), object())
        collect.assert_not_called()

    def test_on_step_end_interval_zero_never_collects(self):
        # gc_interval == 0 must short-circuit before the modulo, so there is
        # no collect and no ZeroDivisionError even on a "round" step.
        with mock.patch.object(gc, "collect") as collect:
            self.callback.on_step_end(self._args(0), self._state(10), object())
        collect.assert_not_called()

    def test_on_step_end_collects_on_step_zero(self):
        # step 0 % interval == 0, so step 0 with a positive interval fires.
        with mock.patch.object(gc, "collect") as collect:
            self.callback.on_step_end(self._args(5), self._state(0), object())
        self.assertEqual(collect.call_count, 1)

    def test_on_step_end_collects_only_at_interval_multiples(self):
        interval = 4
        triggered = []
        with mock.patch.object(gc, "collect") as collect:
            for step in range(1, 13):  # steps 1..12
                before = collect.call_count
                self.callback.on_step_end(
                    self._args(interval), self._state(step), object()
                )
                if collect.call_count > before:
                    triggered.append(step)
        # Hand-derived: multiples of 4 within 1..12 are 4, 8, 12.
        self.assertEqual(triggered, [4, 8, 12])
        self.assertEqual(collect.call_count, 3)

    def test_on_step_end_interval_one_collects_every_step(self):
        # interval == 1 => every step is a multiple => collect each call.
        with mock.patch.object(gc, "collect") as collect:
            for step in range(1, 6):
                self.callback.on_step_end(
                    self._args(1), self._state(step), object()
                )
        self.assertEqual(collect.call_count, 5)


if __name__ == "__main__":
    unittest.main()
