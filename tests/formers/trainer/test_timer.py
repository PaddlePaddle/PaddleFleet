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

"""Behavior unit tests for paddlefleet.trainer.plugins.timer.

These tests drive the real timer classes with a monkeypatched, fully
controllable time source so that every elapsed value is exact and asserted
against an independently hand-derived expectation (never merely ``> 0``).
Device synchronization and the tensorboard writer are treated as external
collaborators; the timing / accumulation / logging logic under test stays real.

Paddle is required to import the module under test; when it is unavailable the
suites skip (with the concrete ImportError recorded) rather than fabricate a
pass. This is a no-card (CPU) suite exercising control logic only.
"""

import unittest
from unittest.mock import patch

try:
    import paddle  # noqa: F401

    from paddlefleet.trainer.plugins.timer import (
        RuntimeTimer,
        Timers,
        _Timer,
        disable_timers,
        get_timers,
        set_timers,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - environment dependent
    _IMPORT_ERROR = exc

_SKIP_REASON = "requires paddle/paddlefleet.trainer.plugins.timer: {}".format(
    _IMPORT_ERROR
)


class _FakeClock:
    """Deterministic stand-in for time.time with an explicit settable value."""

    def __init__(self, start=0.0):
        self.now = float(start)

    def __call__(self):
        return self.now

    def set(self, value):
        self.now = float(value)


class _RecordingWriter:
    """Minimal tensorboard-like writer capturing add_scalar arguments."""

    def __init__(self):
        self.scalars = []

    def add_scalar(self, tag, value, iteration):
        self.scalars.append((tag, value, iteration))


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class _TimerTestBase(unittest.TestCase):
    """Shared fixture: a controllable clock and a CPU device by default."""

    def setUp(self):
        self.clock = _FakeClock(0.0)
        clock_patcher = patch("time.time", self.clock)
        clock_patcher.start()
        self.addCleanup(clock_patcher.stop)

        device_patcher = patch("paddle.device.get_device", return_value="cpu")
        device_patcher.start()
        self.addCleanup(device_patcher.stop)

    def run_interval(self, timer, start_t, stop_t):
        """Start at start_t and stop at stop_t on the fake clock."""
        self.clock.set(start_t)
        timer.start()
        self.clock.set(stop_t)
        timer.stop()


class TestTimer(_TimerTestBase):
    """Behavior of the low-level _Timer: start/stop/reset/elapsed accounting."""

    def test_init_uses_clock_and_zeroes_state(self):
        self.clock.set(123.5)
        timer = _Timer("phase")
        self.assertEqual(timer.name, "phase")
        self.assertEqual(timer.elapsed_, 0.0)
        self.assertFalse(timer.started_)
        self.assertEqual(timer.start_time, 123.5)

    def test_start_records_time_sets_flag_and_skips_cpu_sync(self):
        timer = _Timer("p")
        with patch("paddle.device.synchronize") as sync:
            self.clock.set(10.0)
            timer.start()
        self.assertTrue(timer.started_)
        self.assertEqual(timer.start_time, 10.0)
        sync.assert_not_called()  # "cpu" in device name -> no synchronize

    def test_start_twice_raises(self):
        timer = _Timer("p")
        self.clock.set(1.0)
        timer.start()
        with self.assertRaises(AssertionError):
            timer.start()

    def test_stop_without_start_raises(self):
        timer = _Timer("p")
        with self.assertRaises(AssertionError):
            timer.stop()

    def test_stop_accumulates_exact_interval(self):
        timer = _Timer("p")
        self.run_interval(timer, 100.0, 102.5)
        self.assertEqual(timer.elapsed_, 2.5)
        self.assertFalse(timer.started_)

    def test_multiple_intervals_accumulate(self):
        timer = _Timer("p")
        self.run_interval(timer, 10.0, 13.0)  # +3.0
        self.run_interval(timer, 20.0, 24.5)  # +4.5
        self.assertEqual(timer.elapsed_, 7.5)

    def test_reset_clears_elapsed_and_flag(self):
        timer = _Timer("p")
        self.run_interval(timer, 0.0, 5.0)
        self.assertEqual(timer.elapsed_, 5.0)
        timer.reset()
        self.assertEqual(timer.elapsed_, 0.0)
        self.assertFalse(timer.started_)

    def test_elapsed_reset_true_returns_and_clears(self):
        timer = _Timer("p")
        self.run_interval(timer, 0.0, 5.0)
        value = timer.elapsed(reset=True)
        self.assertEqual(value, 5.0)
        self.assertEqual(timer.elapsed_, 0.0)

    def test_elapsed_reset_false_returns_and_keeps(self):
        timer = _Timer("p")
        self.run_interval(timer, 0.0, 5.0)
        value = timer.elapsed(reset=False)
        self.assertEqual(value, 5.0)
        self.assertEqual(timer.elapsed_, 5.0)

    def test_elapsed_on_fresh_timer_is_zero_without_side_effects(self):
        timer = _Timer("p")
        value = timer.elapsed(reset=True)
        self.assertEqual(value, 0.0)
        self.assertFalse(timer.started_)
        self.assertEqual(timer.elapsed_, 0.0)

    def test_elapsed_while_running_reset_true_restarts(self):
        timer = _Timer("p")
        self.clock.set(0.0)
        timer.start()
        self.clock.set(5.0)
        value = timer.elapsed(reset=True)
        # stop commits 5-0; value captured before reset; timer restarts at t=5.
        self.assertEqual(value, 5.0)
        self.assertEqual(timer.elapsed_, 0.0)
        self.assertTrue(timer.started_)
        self.assertEqual(timer.start_time, 5.0)
        # A further 2.0s interval accumulates from the restart point.
        self.clock.set(7.0)
        timer.stop()
        self.assertEqual(timer.elapsed_, 2.0)

    def test_elapsed_while_running_reset_false_commits_and_continues(self):
        timer = _Timer("p")
        self.clock.set(0.0)
        timer.start()
        self.clock.set(5.0)
        value = timer.elapsed(reset=False)
        self.assertEqual(value, 5.0)
        self.assertEqual(timer.elapsed_, 5.0)
        self.assertTrue(timer.started_)
        self.assertEqual(timer.start_time, 5.0)
        # A further 3.0s stacks onto the committed 5.0s (no double counting).
        self.clock.set(8.0)
        timer.stop()
        self.assertEqual(timer.elapsed_, 8.0)

    def test_non_cpu_device_triggers_synchronize_on_start_and_stop(self):
        timer = _Timer("p")
        with (
            patch("paddle.device.get_device", return_value="gpu:0"),
            patch("paddle.device.synchronize") as sync,
        ):
            self.clock.set(0.0)
            timer.start()
            self.clock.set(1.0)
            timer.stop()
        self.assertEqual(sync.call_count, 2)  # once in start, once in stop
        self.assertEqual(timer.elapsed_, 1.0)


class TestRuntimeTimer(_TimerTestBase):
    """Behavior of RuntimeTimer: renaming, accounting, log format and reset."""

    def test_init_wraps_named_timer(self):
        rt = RuntimeTimer("rt")
        self.assertIsInstance(rt.timer, _Timer)
        self.assertEqual(rt.timer.name, "rt")

    def test_start_overrides_name_and_starts(self):
        rt = RuntimeTimer("rt")
        self.clock.set(0.0)
        rt.start("phase_a")
        self.assertEqual(rt.timer.name, "phase_a")
        self.assertTrue(rt.timer.started_)

    def test_start_stop_accumulates_exact(self):
        rt = RuntimeTimer("rt")
        self.clock.set(0.0)
        rt.start("phase_a")
        self.clock.set(3.0)
        rt.stop()
        self.assertEqual(rt.timer.elapsed_, 3.0)

    def test_log_formats_elapsed_and_resets(self):
        rt = RuntimeTimer("rt")
        self.clock.set(0.0)
        rt.start("myphase")
        self.clock.set(2.5)
        rt.stop()
        with patch("time.strftime", return_value="2026-01-01 00:00:00"):
            result = rt.log()
        self.assertEqual(
            result, "[timelog] myphase: 2.50s (2026-01-01 00:00:00) "
        )
        self.assertEqual(rt.timer.elapsed_, 0.0)
        self.assertFalse(rt.timer.started_)

    def test_log_while_running_stops_and_reports(self):
        rt = RuntimeTimer("rt")
        self.clock.set(0.0)
        rt.start("running_phase")
        self.clock.set(
            4.0
        )  # deliberately not stopped: log() handles live timer
        with patch("time.strftime", return_value="2026-01-01 00:00:00"):
            result = rt.log()
        self.assertEqual(
            result, "[timelog] running_phase: 4.00s (2026-01-01 00:00:00) "
        )
        self.assertFalse(rt.timer.started_)
        self.assertEqual(rt.timer.elapsed_, 0.0)


class TestTimers(_TimerTestBase):
    """Behavior of the Timers group: registry, write/log/info accounting."""

    def _make_timer(self, timers, name, start_t, stop_t):
        with patch("paddle.is_compiled_with_cuda", return_value=False):
            timer = timers(name)
        self.run_interval(timer, start_t, stop_t)
        return timer

    def test_init_empty(self):
        timers = Timers()
        self.assertEqual(timers.timers, {})

    def test_call_creates_named_timer(self):
        timers = Timers()
        with patch("paddle.is_compiled_with_cuda", return_value=False):
            timer = timers("a")
        self.assertIsInstance(timer, _Timer)
        self.assertEqual(timer.name, "a")
        self.assertIs(timers.timers["a"], timer)

    def test_call_returns_same_instance_for_same_name(self):
        timers = Timers()
        with patch("paddle.is_compiled_with_cuda", return_value=False):
            first = timers("shared")
            second = timers("shared")
        self.assertIs(first, second)

    def test_call_use_event_without_cuda_falls_back_to_timer(self):
        timers = Timers()
        with patch("paddle.is_compiled_with_cuda", return_value=False):
            timer = timers("evt", use_event=True)
        self.assertIsInstance(timer, _Timer)

    def test_write_records_normalized_value_tag_and_iteration(self):
        timers = Timers()
        timer = self._make_timer(timers, "a", 0.0, 4.0)
        writer = _RecordingWriter()
        timers.write(["a"], writer, iteration=7, normalizer=2.0)
        # value = elapsed(4.0) / normalizer(2.0) = 2.0; "timers/" prefix; reset.
        self.assertEqual(writer.scalars, [("timers/a", 2.0, 7)])
        self.assertEqual(timer.elapsed_, 0.0)  # default reset=True

    def test_write_reset_false_preserves_elapsed(self):
        timers = Timers()
        timer = self._make_timer(timers, "a", 0.0, 4.0)
        writer = _RecordingWriter()
        timers.write(["a"], writer, iteration=1, normalizer=1.0, reset=False)
        self.assertEqual(writer.scalars, [("timers/a", 4.0, 1)])
        self.assertEqual(timer.elapsed_, 4.0)

    def test_write_multiple_names_record_each_value(self):
        timers = Timers()
        self._make_timer(timers, "a", 0.0, 1.0)
        self._make_timer(timers, "b", 0.0, 3.0)
        writer = _RecordingWriter()
        timers.write(["a", "b"], writer, iteration=2, normalizer=1.0)
        self.assertEqual(
            writer.scalars,
            [("timers/a", 1.0, 2), ("timers/b", 3.0, 2)],
        )

    def test_write_rejects_nonpositive_normalizer(self):
        timers = Timers()
        writer = _RecordingWriter()
        with self.assertRaises(AssertionError):
            timers.write([], writer, iteration=1, normalizer=0.0)
        with self.assertRaises(AssertionError):
            timers.write([], writer, iteration=1, normalizer=-1.0)

    def test_log_scales_to_ms_and_sorts_by_value_desc(self):
        timers = Timers()
        self._make_timer(timers, "a", 0.0, 1.0)  # 1000 ms
        self._make_timer(timers, "b", 0.0, 3.0)  # 3000 ms
        self._make_timer(timers, "c", 0.0, 2.0)  # 2000 ms
        result = timers.log(["a", "b", "c"], normalizer=1.0)
        self.assertEqual(
            result,
            "time (ms) | b : 3000.00 | c : 2000.00 | a : 1000.00",
        )

    def test_log_applies_normalizer(self):
        timers = Timers()
        self._make_timer(timers, "a", 0.0, 4.0)
        result = timers.log(["a"], normalizer=2.0)
        # 4.0s * 1000 / 2.0 = 2000.0 ms
        self.assertEqual(result, "time (ms) | a : 2000.00")

    def test_log_rejects_nonpositive_normalizer(self):
        timers = Timers()
        with self.assertRaises(AssertionError):
            timers.log([], normalizer=0.0)

    def test_info_returns_ms_dict_sorted_by_name_asc(self):
        timers = Timers()
        self._make_timer(timers, "charlie", 0.0, 3.0)
        self._make_timer(timers, "alpha", 0.0, 1.0)
        self._make_timer(timers, "bravo", 0.0, 2.0)
        result = timers.info(["charlie", "alpha", "bravo"], normalizer=1.0)
        self.assertEqual(list(result.keys()), ["alpha", "bravo", "charlie"])
        self.assertEqual(
            result,
            {"alpha": 1000.0, "bravo": 2000.0, "charlie": 3000.0},
        )

    def test_info_default_reset_false_preserves_elapsed(self):
        timers = Timers()
        timer = self._make_timer(timers, "a", 0.0, 2.0)
        result = timers.info(["a"], normalizer=1.0)
        self.assertEqual(result, {"a": 2000.0})
        self.assertEqual(timer.elapsed_, 2.0)  # info default reset=False

    def test_info_reset_true_clears_elapsed(self):
        timers = Timers()
        timer = self._make_timer(timers, "a", 0.0, 2.0)
        timers.info(["a"], normalizer=1.0, reset=True)
        self.assertEqual(timer.elapsed_, 0.0)

    def test_info_applies_normalizer(self):
        timers = Timers()
        self._make_timer(timers, "a", 0.0, 4.0)
        result = timers.info(["a"], normalizer=4.0)
        self.assertEqual(result, {"a": 1000.0})  # 4.0*1000/4.0

    def test_info_rejects_nonpositive_normalizer(self):
        timers = Timers()
        with self.assertRaises(AssertionError):
            timers.info([], normalizer=-1.0)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestGlobalTimers(unittest.TestCase):
    """Global timer registry transitions via the real module state."""

    def setUp(self):
        import paddlefleet.trainer.plugins.timer as timer_mod

        self.timer_mod = timer_mod
        original = timer_mod._GLOBAL_TIMERS
        # Restore even on assertion failure to avoid cross-test pollution.
        self.addCleanup(setattr, timer_mod, "_GLOBAL_TIMERS", original)

    def test_set_then_get_returns_timers_instance(self):
        self.timer_mod._GLOBAL_TIMERS = None
        self.assertIsNone(get_timers())
        set_timers()
        self.assertIsInstance(get_timers(), Timers)

    def test_disable_resets_registry_to_none(self):
        set_timers()
        self.assertIsInstance(get_timers(), Timers)
        disable_timers()
        self.assertIsNone(get_timers())


if __name__ == "__main__":
    unittest.main()
