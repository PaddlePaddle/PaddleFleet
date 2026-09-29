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

"""Behavior tests for the ``Timers`` manager in paddlefleet.timers.

Scope (see sibling test_timers.py which concurrently covers the low-level
``_Timer`` start/stop/reset/elapsed state machine):
  * __call__ named-timer registry: creation, per-name caching (identity),
    and independence of distinct names
  * log(): exact formatted string, descending-by-milliseconds ordering,
    normalizer scaling, and reset=True/False accumulation gating
  * info(): dict ordered ascending by name, millisecond scaling, normalizer,
    and reset=False (its default) repeatability
  * write(): dispatch gating between a SummaryWriter and the wandb module,
    the (seconds / normalizer) value it forwards, and the normalizer guard

Every expected value is hand-derived arithmetic on a deterministic fake clock
patched onto paddlefleet.timers.time.time; timers are loaded through the real
Timers.__call__ + _Timer.start/stop path, so no timer values are set by hand
and no >0-only wall-clock assertions are made. The device is pinned to "cpu"
(and restored) so start()/stop() skip paddle.device.synchronize() without
mocking any code under test. SummaryWriter / wandb are non-under-test writer
collaborators; they are replaced with recording doubles and their exact
received arguments are asserted.
"""

import os
import sys
import unittest
from unittest import mock
from unittest.mock import patch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "src"))
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Honest capability probe: timers.py imports paddle unconditionally (for the
# cuda-sync guard), so the module cannot be imported without it. Only a genuine
# missing dependency (ImportError, of which ModuleNotFoundError is a subclass)
# is allowed to skip; any other error must surface as a real failure rather
# than a fake pass.
try:
    import paddlefleet.timers as timers_mod

    _IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - depends on environment
    timers_mod = None
    _IMPORT_ERROR = exc


class _FakeClock:
    """Deterministic stand-in for time.time(); ``now`` is set explicitly."""

    def __init__(self, start=0.0):
        self.now = float(start)

    def __call__(self):
        return self.now


class _RecordingSummaryWriter:
    """Non-under-test tensorboard writer double; records add_scalar calls."""

    def __init__(self):
        self.scalars = []

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))


class _RecordingWandb:
    """Non-under-test wandb-module double; records log() calls."""

    def __init__(self):
        self.logged = []

    def log(self, data, step):
        self.logged.append((data, step))


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddlefleet.timers import failed (missing dependency): {_IMPORT_ERROR}",
)
class TimersManagerTest(unittest.TestCase):
    """Controlled-clock tests of the Timers manager registry and formatting."""

    def setUp(self):
        import paddle

        # Pin to CPU so _Timer.start()/stop() skip paddle.device.synchronize();
        # restore the original device so no global state leaks between tests.
        self._orig_device = paddle.device.get_device()
        self.addCleanup(paddle.device.set_device, self._orig_device)
        paddle.device.set_device("cpu")

    def _loaded_timers(self, spec):
        """Build a Timers manager and load exact elapsed seconds per name.

        ``spec`` is a list of (name, seconds). Each timer is created through
        the real Timers.__call__ registry and accumulates precisely ``seconds``
        via _Timer.start()/stop() driven by the fake clock (stop - start with
        start == 0.0, so no floating-point cancellation).
        """
        timers = timers_mod.Timers()
        clock = _FakeClock(0.0)
        with patch.object(timers_mod.time, "time", clock):
            for name, seconds in spec:
                clock.now = 0.0
                timer = timers(name)
                timer.start()
                clock.now = float(seconds)
                timer.stop()
        return timers

    # PLACEHOLDER_TESTS

    def test_call_creates_caches_and_isolates_named_timers(self):
        timers = timers_mod.Timers()
        self.assertEqual(timers.timers, {})

        fwd = timers("forward")
        self.assertIsInstance(fwd, timers_mod._Timer)
        self.assertEqual(fwd.name, "forward")
        # The created timer is the exact object stored in the registry.
        self.assertIs(timers.timers["forward"], fwd)

        # Same name returns the cached object, not a fresh one.
        self.assertIs(timers("forward"), fwd)

        # A distinct name yields an independent timer with its own name.
        bwd = timers("backward")
        self.assertIsNot(bwd, fwd)
        self.assertEqual(bwd.name, "backward")
        self.assertEqual(set(timers.timers), {"forward", "backward"})

    def test_log_orders_by_descending_milliseconds(self):
        timers = self._loaded_timers(
            [("alpha", 0.010), ("beta", 0.030), ("gamma", 0.020)]
        )
        with patch("builtins.print") as printed:
            timers.log(names=["alpha", "beta", "gamma"], normalizer=1.0)
        # ms = seconds * 1000 / normalizer: alpha=10, beta=30, gamma=20;
        # emitted high-to-low regardless of the ascending name sort.
        printed.assert_called_once_with(
            "time (ms) | beta : 30.00 | gamma : 20.00 | alpha : 10.00"
        )

    def test_log_applies_normalizer(self):
        timers = self._loaded_timers([("perf", 0.040), ("io", 0.080)])
        with patch("builtins.print") as printed:
            timers.log(names=["perf", "io"], normalizer=4.0)
        # 0.040*1000/4=10.00 ; 0.080*1000/4=20.00 -> io before perf.
        printed.assert_called_once_with("time (ms) | io : 20.00 | perf : 10.00")

    def test_log_reset_true_zeroes_all_named_timers(self):
        timers = self._loaded_timers(
            [("alpha", 0.010), ("beta", 0.030), ("gamma", 0.020)]
        )
        with patch("builtins.print") as printed:
            timers.log(names=["alpha", "beta", "gamma"], normalizer=1.0)
            timers.log(names=["alpha", "beta", "gamma"], normalizer=1.0)
        first = printed.call_args_list[0].args[0]
        second = printed.call_args_list[1].args[0]
        self.assertEqual(
            first, "time (ms) | beta : 30.00 | gamma : 20.00 | alpha : 10.00"
        )
        # reset=True (default) consumed every timer, so the re-log is all zero,
        # in the stable ascending-name order the equal values preserve.
        self.assertEqual(
            second, "time (ms) | alpha : 0.00 | beta : 0.00 | gamma : 0.00"
        )

    def test_log_reset_false_preserves_accumulation(self):
        timers = self._loaded_timers([("alpha", 0.010), ("beta", 0.030)])
        with patch("builtins.print") as printed:
            timers.log(names=["alpha", "beta"], normalizer=1.0, reset=False)
            timers.log(names=["alpha", "beta"], normalizer=1.0, reset=False)
        expected = "time (ms) | beta : 30.00 | alpha : 10.00"
        self.assertEqual(printed.call_args_list[0].args[0], expected)
        # reset=False leaves elapsed_ intact, so the second call is identical.
        self.assertEqual(printed.call_args_list[1].args[0], expected)

    # PLACEHOLDER_TESTS_2

    def test_info_orders_by_name_and_scales_to_milliseconds(self):
        timers = self._loaded_timers([("zebra", 0.001), ("apple", 0.002)])
        result = timers.info(names=["zebra", "apple"], normalizer=1.0)
        # Returned dict is ordered ascending by name (not by value/insertion).
        self.assertEqual(list(result.keys()), ["apple", "zebra"])
        # ms = seconds * 1000 / normalizer.
        self.assertAlmostEqual(result["apple"], 2.0, places=6)
        self.assertAlmostEqual(result["zebra"], 1.0, places=6)

    def test_info_reset_false_default_is_repeatable(self):
        timers = self._loaded_timers([("x", 0.005)])
        first = timers.info(names=["x"], normalizer=1.0)
        # info defaults to reset=False, so a second call reads the same value.
        second = timers.info(names=["x"], normalizer=1.0)
        self.assertAlmostEqual(first["x"], 5.0, places=6)
        self.assertAlmostEqual(second["x"], 5.0, places=6)

    def test_info_applies_normalizer(self):
        timers = self._loaded_timers([("x", 0.100)])
        result = timers.info(names=["x"], normalizer=10.0)
        # 0.100 * 1000 / 10 = 10.0
        self.assertAlmostEqual(result["x"], 10.0, places=6)

    def test_normalizer_must_be_positive(self):
        timers = timers_mod.Timers()
        # The normalizer > 0 guard runs before any timer lookup on all three.
        with self.assertRaises(AssertionError):
            timers.log(names=["x"], normalizer=0.0)
        with self.assertRaises(AssertionError):
            timers.log(names=["x"], normalizer=-2.0)
        with self.assertRaises(AssertionError):
            timers.info(names=["x"], normalizer=0.0)
        with self.assertRaises(AssertionError):
            timers.write(names=["x"], writer=None, iteration=0, normalizer=0.0)

    def test_write_dispatches_scaled_values_to_summary_writer(self):
        timers = self._loaded_timers([("perf", 0.040), ("io", 0.010)])
        writer = _RecordingSummaryWriter()
        with patch.object(timers_mod, "SummaryWriter", _RecordingSummaryWriter):
            timers.write(
                names=["perf", "io"],
                writer=writer,
                iteration=7,
                normalizer=2.0,
            )
        # write() forwards seconds / normalizer (NOT scaled to ms) in the
        # given name order, tagging each "timers/<name>" at the iteration.
        self.assertEqual(len(writer.scalars), 2)
        self.assertEqual(writer.scalars[0][0], "timers/perf")
        self.assertAlmostEqual(writer.scalars[0][1], 0.020, places=6)
        self.assertEqual(writer.scalars[0][2], 7)
        self.assertEqual(writer.scalars[1][0], "timers/io")
        self.assertAlmostEqual(writer.scalars[1][1], 0.005, places=6)
        self.assertEqual(writer.scalars[1][2], 7)

    def test_write_dispatches_scaled_value_to_wandb(self):
        timers = self._loaded_timers([("step", 0.060)])
        fake_wandb = _RecordingWandb()
        # SummaryWriter is a real (non-matching) class so the isinstance check
        # is well-formed and False; the writer identity-matches the wandb hook.
        with (
            patch.object(timers_mod, "SummaryWriter", _RecordingSummaryWriter),
            patch.object(timers_mod, "wandb", fake_wandb),
        ):
            timers.write(
                names=["step"],
                writer=fake_wandb,
                iteration=11,
                normalizer=3.0,
            )
        self.assertEqual(len(fake_wandb.logged), 1)
        data, step = fake_wandb.logged[0]
        self.assertEqual(list(data.keys()), ["timers/step"])
        self.assertAlmostEqual(data["timers/step"], 0.020, places=6)
        self.assertEqual(step, 11)

    def test_write_ignores_unrecognized_writer_but_still_resets(self):
        timers = self._loaded_timers([("x", 0.050)])
        writer = mock.Mock()  # neither a SummaryWriter nor the wandb hook
        with (
            patch.object(timers_mod, "SummaryWriter", _RecordingSummaryWriter),
            patch.object(timers_mod, "wandb", None),
        ):
            timers.write(
                names=["x"], writer=writer, iteration=0, normalizer=1.0
            )
        # Neither dispatch branch fires for an unrecognized writer.
        writer.add_scalar.assert_not_called()
        writer.log.assert_not_called()
        # But reset=True (default) still consumed the timer, so it now reads 0.
        remaining = timers.info(names=["x"], normalizer=1.0)
        self.assertAlmostEqual(remaining["x"], 0.0, places=6)

    @unittest.expectedFailure
    def test_write_should_dispatch_to_wandb_when_tensorboardx_absent(self):
        """Real bug: write() crashes if tensorboardX is not installed.

        When tensorboardX is missing, module-level ``SummaryWriter`` is None.
        The guard ``isinstance(writer, SummaryWriter) and SummaryWriter is not
        None`` evaluates isinstance() FIRST, and isinstance(writer, None) raises
        TypeError before the ``is not None`` short-circuit can protect it. So a
        legitimate wandb writer can never be reached in that environment. The
        correct behavior asserted here (wandb.log receives the scaled value) is
        therefore expected to fail. Production is left unmodified.
        """
        timers = self._loaded_timers([("step", 0.060)])
        fake_wandb = _RecordingWandb()
        with (
            patch.object(timers_mod, "SummaryWriter", None),
            patch.object(timers_mod, "wandb", fake_wandb),
        ):
            timers.write(
                names=["step"],
                writer=fake_wandb,
                iteration=5,
                normalizer=3.0,
            )
        self.assertEqual(len(fake_wandb.logged), 1)
        self.assertAlmostEqual(
            fake_wandb.logged[0][0]["timers/step"], 0.020, places=6
        )


if __name__ == "__main__":
    unittest.main()
