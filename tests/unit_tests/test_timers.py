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

import time
import unittest

from fleet.core.timers import DummyTimer, Timer, Timers


class TestTimer(unittest.TestCase):
    """Test cases for Timer class."""

    def test_timer_initialization(self):
        """Test timer initialization."""
        timer = Timer("test_timer")
        self.assertEqual(timer.name, "test_timer")
        self.assertEqual(timer._elapsed, 0.0)
        self.assertEqual(timer._active_time, 0.0)
        self.assertFalse(timer._started)

    def test_timer_start_stop(self):
        """Test timer start and stop functionality."""
        timer = Timer("test_timer")

        # Start the timer
        timer.start()
        self.assertTrue(timer._started)

        # Sleep for a short duration
        time.sleep(0.1)

        # Stop the timer
        timer.stop()
        self.assertFalse(timer._started)
        self.assertGreater(timer._elapsed, 0.0)
        self.assertGreater(timer._active_time, 0.0)

    def test_timer_start_already_started(self):
        """Test that starting an already started timer raises an error."""
        timer = Timer("test_timer")
        timer.start()

        with self.assertRaises(AssertionError):
            timer.start()

        timer.stop()

    def test_timer_stop_not_started(self):
        """Test that stopping a non-started timer raises an error."""
        timer = Timer("test_timer")

        with self.assertRaises(AssertionError):
            timer.stop()

    def test_timer_reset(self):
        """Test timer reset functionality."""
        timer = Timer("test_timer")
        timer.start()
        time.sleep(0.1)
        timer.stop()

        elapsed_before = timer._elapsed
        active_time_before = timer._active_time
        self.assertGreater(elapsed_before, 0.0)

        timer.reset()
        self.assertEqual(timer._elapsed, 0.0)
        self.assertFalse(timer._started)
        # active_time should not be reset
        self.assertEqual(timer._active_time, active_time_before)

    def test_timer_elapsed(self):
        """Test timer elapsed method."""
        timer = Timer("test_timer")
        timer.start()
        time.sleep(0.1)
        timer.stop()

        elapsed = timer.elapsed(reset=True)
        self.assertGreater(elapsed, 0.0)
        # After reset, elapsed should be 0
        self.assertEqual(timer._elapsed, 0.0)

    def test_timer_elapsed_without_reset(self):
        """Test timer elapsed method without reset."""
        timer = Timer("test_timer")
        timer.start()
        time.sleep(0.1)
        timer.stop()

        elapsed_before = timer._elapsed
        elapsed = timer.elapsed(reset=False)
        self.assertGreater(elapsed, 0.0)
        # Should not reset
        self.assertEqual(timer._elapsed, elapsed_before)

    def test_timer_elapsed_while_running(self):
        """Test timer elapsed method while timer is running."""
        timer = Timer("test_timer")
        timer.start()
        time.sleep(0.1)

        elapsed = timer.elapsed(reset=True)
        self.assertGreater(elapsed, 0.0)
        # Timer should be restarted
        self.assertTrue(timer._started)

        timer.stop()

    def test_timer_active_time(self):
        """Test timer active_time method."""
        timer = Timer("test_timer")

        # First run
        timer.start()
        time.sleep(0.1)
        timer.stop()
        first_active_time = timer.active_time()

        # Reset elapsed time
        timer.reset()

        # Second run
        timer.start()
        time.sleep(0.1)
        timer.stop()
        second_active_time = timer.active_time()

        # active_time should accumulate
        self.assertGreater(second_active_time, first_active_time)

    def test_timer_multiple_cycles(self):
        """Test timer through multiple start-stop cycles."""
        timer = Timer("test_timer")

        for i in range(3):
            timer.start()
            time.sleep(0.05)
            timer.stop()

        # Elapsed time should be accumulated
        self.assertGreater(timer._elapsed, 0.1)
        self.assertGreater(timer._active_time, 0.1)


class TestDummyTimer(unittest.TestCase):
    """Test cases for DummyTimer class."""

    def test_dummy_timer_initialization(self):
        """Test dummy timer initialization."""
        timer = DummyTimer()
        self.assertEqual(timer.name, "dummy timer")

    def test_dummy_timer_start_stop_reset(self):
        """Test dummy timer start, stop, and reset do nothing."""
        timer = DummyTimer()

        # These should not raise errors
        timer.start()
        timer.stop()
        timer.reset()

    def test_dummy_timer_elapsed_raises(self):
        """Test that dummy timer elapsed raises an exception."""
        timer = DummyTimer()

        with self.assertRaises(Exception) as context:
            timer.elapsed()

        self.assertIn("dummy timer should not be used", str(context.exception))

    def test_dummy_timer_active_time_raises(self):
        """Test that dummy timer active_time raises an exception."""
        timer = DummyTimer()

        with self.assertRaises(Exception) as context:
            timer.active_time()

        self.assertIn("active timer should not be used", str(context.exception))


class TestTimers(unittest.TestCase):
    """Test cases for Timers class."""

    def test_timers_initialization(self):
        """Test Timers initialization."""
        timers = Timers(log_level=2, log_option="max")
        self.assertEqual(timers._log_level, 2)
        self.assertEqual(timers._log_option, "max")
        self.assertEqual(len(timers._timers), 0)

    def test_timers_invalid_log_option(self):
        """Test that invalid log option raises an error."""
        with self.assertRaises(AssertionError):
            Timers(log_level=2, log_option="invalid")

    def test_timers_valid_log_options(self):
        """Test all valid log options."""
        for option in ["max", "minmax", "all"]:
            timers = Timers(log_level=2, log_option=option)
            self.assertEqual(timers._log_option, option)

    def test_timers_create_timer(self):
        """Test creating a timer."""
        timers = Timers(log_level=2, log_option="max")
        timer = timers("test_timer", log_level=1)

        self.assertIsInstance(timer, Timer)
        self.assertEqual(timer.name, "test_timer")
        self.assertIn("test_timer", timers._timers)
        self.assertEqual(timers._log_levels["test_timer"], 1)

    def test_timers_create_timer_default_log_level(self):
        """Test creating a timer with default log level."""
        timers = Timers(log_level=2, log_option="max")
        timer = timers("test_timer")

        self.assertIsInstance(timer, Timer)
        self.assertEqual(timers._log_levels["test_timer"], 2)

    def test_timers_get_existing_timer(self):
        """Test getting an existing timer."""
        timers = Timers(log_level=2, log_option="max")
        timer1 = timers("test_timer", log_level=1)
        timer2 = timers("test_timer", log_level=1)

        # Should return the same timer
        self.assertIs(timer1, timer2)

    def test_timers_get_existing_timer_mismatched_log_level(self):
        """Test getting existing timer with mismatched log level raises error."""
        timers = Timers(log_level=2, log_option="max")
        timers("test_timer", log_level=1)

        with self.assertRaises(AssertionError):
            timers("test_timer", log_level=2)

    def test_timers_log_level_too_high(self):
        """Test that log level higher than max raises error."""
        timers = Timers(log_level=2, log_option="max")

        with self.assertRaises(AssertionError):
            timers("test_timer", log_level=3)

    def test_timers_return_dummy_timer(self):
        """Test that dummy timer is returned for log level higher than threshold."""
        timers = Timers(log_level=1, log_option="max")
        timer = timers("test_timer", log_level=2)

        # Should return dummy timer
        self.assertIsInstance(timer, DummyTimer)
        # Should not be added to timers dict
        self.assertNotIn("test_timer", timers._timers)

    def test_timers_multiple_timers(self):
        """Test creating multiple timers."""
        timers = Timers(log_level=2, log_option="max")

        timer1 = timers("timer1", log_level=1)
        timer2 = timers("timer2", log_level=2)
        timer3 = timers("timer3", log_level=1)

        self.assertEqual(len(timers._timers), 3)
        self.assertIn("timer1", timers._timers)
        self.assertIn("timer2", timers._timers)
        self.assertIn("timer3", timers._timers)

    def test_timers_workflow(self):
        """Test a complete workflow with Timers."""
        timers = Timers(log_level=2, log_option="max")

        # Create and use timer1
        timer1 = timers("operation1", log_level=1)
        timer1.start()
        time.sleep(0.1)
        timer1.stop()

        # Create and use timer2
        timer2 = timers("operation2", log_level=2)
        timer2.start()
        time.sleep(0.05)
        timer2.stop()

        # Verify both timers have elapsed time
        self.assertGreater(timer1._elapsed, 0.0)
        self.assertGreater(timer2._elapsed, 0.0)

        # Verify timer1 has more elapsed time
        self.assertGreater(timer1._elapsed, timer2._elapsed)

    def test_timers_with_barrier_group(self):
        """Test timer with barrier group."""
        timer = Timer("test_timer")
        barrier_group = None  # Use default group
        timer.set_barrier_group(barrier_group)

        self.assertEqual(timer._barrier_group, barrier_group)


class TestTimersIntegration(unittest.TestCase):
    """Integration tests for Timers functionality."""

    def test_nested_timing(self):
        """Test nested timing operations."""
        timers = Timers(log_level=2, log_option="max")

        outer_timer = timers("outer", log_level=1)
        inner_timer = timers("inner", log_level=2)

        outer_timer.start()
        time.sleep(0.05)

        inner_timer.start()
        time.sleep(0.05)
        inner_timer.stop()

        time.sleep(0.05)
        outer_timer.stop()

        # Outer timer should have more elapsed time
        self.assertGreater(outer_timer._elapsed, inner_timer._elapsed)

    def test_timer_precision(self):
        """Test timer precision."""
        timer = Timer("precision_test")

        timer.start()
        time.sleep(0.1)
        timer.stop()

        # Should be approximately 0.1 seconds (with some tolerance)
        self.assertAlmostEqual(timer._elapsed, 0.1, delta=0.02)


if __name__ == "__main__":
    unittest.main()
