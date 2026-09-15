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

"""Behavior tests for the ``_Timer`` state machine in paddlefleet.timers.

Scope (see sibling test_timers_2.py which concurrently covers the Timers
manager aggregation and log-string formatting):
  * _Timer.start / stop guards and their exact assertion messages
  * elapsed_ accumulation arithmetic across repeated start/stop cycles
  * reset() clearing both elapsed_ and started_
  * elapsed(reset=...) stop -> read -> optional reset -> restart control flow,
    verified through the observable start_time the timer keeps running with

The wall-clock is replaced by a deterministic fake clock (patched onto
paddlefleet.timers.time.time) so every expected value below is hand-derived
arithmetic on controlled timestamps; no >0-only magnitude assertions are made.
_Timer.start/stop call paddle.device.synchronize() unless the active device is
CPU, so each test pins the device to "cpu" and restores it in tearDown; this
keeps the pure control-flow test off the GPU sync path without mocking any of
the code under test.
"""

import os
import sys
import unittest
from unittest.mock import patch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "src"))
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Honest capability probe: timers.py imports paddle unconditionally (for the
# cuda-sync guard). Only a genuine missing dependency (ImportError, which
# ModuleNotFoundError subclasses) is allowed to skip; any other error must
# surface as a real failure rather than a fake pass.
try:
    import paddlefleet.timers as timers_mod

    _IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - depends on environment
    timers_mod = None
    _IMPORT_ERROR = exc


class _FakeClock:
    """Deterministic monotonic stand-in for time.time()."""

    def __init__(self, start):
        self.now = float(start)

    def __call__(self):
        return self.now

    def advance(self, delta):
        self.now += float(delta)


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddlefleet.timers import failed (missing dependency): {_IMPORT_ERROR}",
)
class TimerStateMachineTest(unittest.TestCase):
    """Controlled-clock tests of the _Timer start/stop/reset/elapsed logic."""

    def setUp(self):
        import paddle

        # Pin to CPU so start()/stop() skip paddle.device.synchronize();
        # restore the original device so we do not leak global state.
        self._orig_device = paddle.device.get_device()
        self.addCleanup(paddle.device.set_device, self._orig_device)
        paddle.device.set_device("cpu")

    def _new_timer(self, clock, name="probe"):
        # Construct under the patched clock so start_time is controlled too.
        with patch.object(timers_mod.time, "time", clock):
            return timers_mod._Timer(name)

    def test_init_state(self):
        clock = _FakeClock(1000.0)
        timer = self._new_timer(clock, name="init")
        self.assertEqual(timer.name, "init")
        self.assertEqual(timer.elapsed_, 0.0)
        self.assertFalse(timer.started_)
        # __init__ samples the clock for start_time.
        self.assertEqual(timer.start_time, 1000.0)

    def test_single_start_stop_accumulates_exact_delta(self):
        clock = _FakeClock(100.0)
        timer = self._new_timer(clock)
        with patch.object(timers_mod.time, "time", clock):
            timer.start()
            self.assertTrue(timer.started_)
            self.assertEqual(timer.start_time, 100.0)
            clock.advance(3.0)
            timer.stop()
        # elapsed_ += 103.0 - 100.0
        self.assertEqual(timer.elapsed_, 3.0)
        self.assertFalse(timer.started_)

    def test_repeated_cycles_sum_deltas(self):
        clock = _FakeClock(100.0)
        timer = self._new_timer(clock)
        with patch.object(timers_mod.time, "time", clock):
            timer.start()  # start_time = 100.0
            clock.advance(3.0)
            timer.stop()  # elapsed_ = 3.0
            clock.advance(7.0)  # idle gap must NOT be counted
            timer.start()  # start_time = 110.0
            clock.advance(4.0)
            timer.stop()  # elapsed_ = 3.0 + 4.0
        self.assertEqual(timer.elapsed_, 7.0)
        self.assertFalse(timer.started_)

    def test_double_start_raises_with_name_in_message(self):
        clock = _FakeClock(100.0)
        timer = self._new_timer(clock, name="dup")
        with patch.object(timers_mod.time, "time", clock):
            timer.start()
            with self.assertRaises(AssertionError) as ctx:
                timer.start()
        self.assertEqual(str(ctx.exception), "dup timer has already started")
        # The failed second start must not have perturbed the running state.
        self.assertTrue(timer.started_)
        self.assertEqual(timer.start_time, 100.0)

    def test_stop_without_start_raises_with_name_in_message(self):
        clock = _FakeClock(100.0)
        timer = self._new_timer(clock, name="cold")
        with patch.object(timers_mod.time, "time", clock):
            with self.assertRaises(AssertionError) as ctx:
                timer.stop()
        self.assertEqual(str(ctx.exception), "cold timer is not started.")
        self.assertEqual(timer.elapsed_, 0.0)
        self.assertFalse(timer.started_)

    def test_reset_after_accumulation_clears_elapsed(self):
        clock = _FakeClock(100.0)
        timer = self._new_timer(clock)
        with patch.object(timers_mod.time, "time", clock):
            timer.start()
            clock.advance(5.0)
            timer.stop()
            self.assertEqual(timer.elapsed_, 5.0)
            timer.reset()
        self.assertEqual(timer.elapsed_, 0.0)
        self.assertFalse(timer.started_)

    def test_reset_while_running_drops_started_flag(self):
        clock = _FakeClock(100.0)
        timer = self._new_timer(clock)
        with patch.object(timers_mod.time, "time", clock):
            timer.start()
            clock.advance(2.0)
            timer.reset()  # reset does not stop/accumulate, just clears
        self.assertEqual(timer.elapsed_, 0.0)
        self.assertFalse(timer.started_)

    def test_elapsed_stopped_with_reset_returns_and_clears(self):
        clock = _FakeClock(100.0)
        timer = self._new_timer(clock)
        with patch.object(timers_mod.time, "time", clock):
            timer.start()
            clock.advance(7.0)
            timer.stop()  # elapsed_ = 7.0, not running
            value = timer.elapsed(reset=True)
        self.assertEqual(value, 7.0)
        self.assertEqual(timer.elapsed_, 0.0)
        self.assertFalse(timer.started_)

    def test_elapsed_stopped_without_reset_preserves_elapsed(self):
        clock = _FakeClock(100.0)
        timer = self._new_timer(clock)
        with patch.object(timers_mod.time, "time", clock):
            timer.start()
            clock.advance(7.0)
            timer.stop()  # elapsed_ = 7.0
            value = timer.elapsed(reset=False)
        self.assertEqual(value, 7.0)
        self.assertEqual(timer.elapsed_, 7.0)
        self.assertFalse(timer.started_)

    def test_elapsed_while_running_no_reset_restarts_from_current_time(self):
        clock = _FakeClock(100.0)
        timer = self._new_timer(clock)
        with patch.object(timers_mod.time, "time", clock):
            timer.start()  # start_time = 100.0
            clock.advance(5.0)  # now = 105.0
            # elapsed(): stop -> elapsed_ = 5.0; no reset; restart at now=105.0
            value = timer.elapsed(reset=False)
            self.assertEqual(value, 5.0)
            self.assertEqual(timer.elapsed_, 5.0)
            self.assertTrue(timer.started_)
            self.assertEqual(timer.start_time, 105.0)
            # A later stop must add only the post-restart delta.
            clock.advance(3.0)  # now = 108.0
            timer.stop()  # elapsed_ = 5.0 + (108.0 - 105.0)
        self.assertEqual(timer.elapsed_, 8.0)
        self.assertFalse(timer.started_)

    def test_elapsed_while_running_with_reset_restarts_and_zeroes(self):
        clock = _FakeClock(100.0)
        timer = self._new_timer(clock)
        with patch.object(timers_mod.time, "time", clock):
            timer.start()  # start_time = 100.0
            clock.advance(5.0)  # now = 105.0
            # elapsed(): stop -> elapsed_=5.0; reset -> elapsed_=0.0;
            # then restart at now=105.0 because it was running.
            value = timer.elapsed(reset=True)
            self.assertEqual(value, 5.0)
            self.assertEqual(timer.elapsed_, 0.0)
            self.assertTrue(timer.started_)
            self.assertEqual(timer.start_time, 105.0)
            clock.advance(4.0)  # now = 109.0
            timer.stop()  # elapsed_ = 0.0 + (109.0 - 105.0)
        self.assertEqual(timer.elapsed_, 4.0)
        self.assertFalse(timer.started_)


if __name__ == "__main__":
    unittest.main()
