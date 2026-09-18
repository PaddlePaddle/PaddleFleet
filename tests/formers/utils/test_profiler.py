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

# Behavior test for paddlefleet.utils.profiler.
#
# Scope of evidence (无卡 / CPU): this file verifies the pure-Python option
# parsing contract of ProfilerOptions and the schedule/state DECISION logic of
# add_profiler_step (when it starts vs. steps vs. stops, how batch_range /
# timer_only / record_shapes / profile_path flow into the constructed profiler,
# the step-id progression, the option-caching decision and the exit decision).
# paddle.profiler.Profiler is a genuine external collaborator that requires a
# device to actually profile; it is replaced by a distinguishable recording
# stub so the module's own control flow is exercised and observed. This file
# therefore does NOT verify real profiling numerics, timeline export contents,
# or any GPU behavior -- only the CPU-executable decisions of this module.

import unittest
from unittest import mock

import paddlefleet.utils.profiler as profiler_module
from paddlefleet.utils.profiler import ProfilerOptions, add_profiler_step


class TestProfilerOptions(unittest.TestCase):
    # Expected values below are hand-derived from the documented defaults and
    # the parsing rules, never produced by calling the code under test.

    def test_defaults_applied_when_only_batch_range_given(self):
        opts = ProfilerOptions("batch_range=[10,20]")
        self.assertEqual(opts["batch_range"], [10, 20])
        self.assertEqual(opts["state"], "All")
        self.assertEqual(opts["sorted_key"], "total")
        self.assertEqual(opts["tracer_option"], "Default")
        self.assertEqual(opts["profile_path"], "/tmp/profile")
        self.assertIs(opts["exit_on_finished"], True)
        self.assertIs(opts["timer_only"], True)
        self.assertIs(opts["record_shapes"], False)

    def test_batch_range_parsed(self):
        opts = ProfilerOptions("batch_range=[50,60]")
        self.assertEqual(opts["batch_range"], [50, 60])

    def test_batch_range_start_ge_end_keeps_default(self):
        # start > end and start == end are both rejected (end must be > start),
        # so the default window is retained.
        self.assertEqual(
            ProfilerOptions("batch_range=[60,50]")["batch_range"], [10, 20]
        )
        self.assertEqual(
            ProfilerOptions("batch_range=[30,30]")["batch_range"], [10, 20]
        )

    def test_batch_range_negative_start_keeps_default(self):
        # A negative start fails the start >= 0 guard, default is retained.
        self.assertEqual(
            ProfilerOptions("batch_range=[-1,10]")["batch_range"], [10, 20]
        )

    def test_batch_range_three_values_are_all_stored(self):
        # The guard only checks the first two entries; a third value is kept
        # verbatim rather than being truncated to two.
        self.assertEqual(
            ProfilerOptions("batch_range=[1,2,3]")["batch_range"], [1, 2, 3]
        )

    def test_exit_on_finished_truthy_variants(self):
        # Only these case-insensitive tokens are truthy for exit_on_finished.
        for token in ("true", "TRUE", "yes", "t", "1"):
            with self.subTest(token=token):
                self.assertIs(
                    ProfilerOptions(f"exit_on_finished={token}")[
                        "exit_on_finished"
                    ],
                    True,
                )
        for token in ("false", "no", "0", "2", "maybe"):
            with self.subTest(token=token):
                self.assertIs(
                    ProfilerOptions(f"exit_on_finished={token}")[
                        "exit_on_finished"
                    ],
                    False,
                )

    def test_string_options_stored_verbatim(self):
        opts = ProfilerOptions(
            "batch_range=[1,2];state=GPU;sorted_key=max;"
            "tracer_option=OpDetail;profile_path=/tmp/my_profile"
        )
        self.assertEqual(opts["state"], "GPU")
        self.assertEqual(opts["sorted_key"], "max")
        self.assertEqual(opts["tracer_option"], "OpDetail")
        self.assertEqual(opts["profile_path"], "/tmp/my_profile")

    def test_timer_only_and_record_shapes_kept_as_raw_strings(self):
        # These two options are stored exactly as the raw string token (unlike
        # exit_on_finished, which is coerced to bool). add_profiler_step later
        # normalizes them via str(...) == str(True); the parsed value here is a
        # str, not a bool.
        opts = ProfilerOptions(
            "batch_range=[1,2];timer_only=False;record_shapes=True"
        )
        self.assertEqual(opts["timer_only"], "False")
        self.assertIsInstance(opts["timer_only"], str)
        self.assertEqual(opts["record_shapes"], "True")
        self.assertIsInstance(opts["record_shapes"], str)

    def test_multiple_options_split_on_semicolon(self):
        opts = ProfilerOptions("batch_range=[50,60];state=CPU;sorted_key=ave")
        self.assertEqual(opts["batch_range"], [50, 60])
        self.assertEqual(opts["state"], "CPU")
        self.assertEqual(opts["sorted_key"], "ave")

    def test_all_whitespace_is_stripped_before_parsing(self):
        opts = ProfilerOptions("batch_range = [50, 60] ; state = GPU")
        self.assertEqual(opts["batch_range"], [50, 60])
        self.assertEqual(opts["state"], "GPU")

    def test_getitem_unknown_key_raises_value_error(self):
        opts = ProfilerOptions("batch_range=[10,20]")
        with self.assertRaises(ValueError):
            opts["nonexistent_key"]

    def test_non_string_input_rejected(self):
        # The constructor asserts a str input.
        with self.assertRaises(AssertionError):
            ProfilerOptions(["batch_range=[10,20]"])


class _FakeProfiler:
    """Distinguishable recording stub for paddle.profiler.Profiler.

    It records the constructor kwargs and the exact order of lifecycle calls so
    the test can assert which decision add_profiler_step made at each step. It
    performs no real profiling; profiling numerics are out of scope here.
    """

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.events = []
        _FakeProfiler.instances.append(self)

    def start(self):
        self.events.append("start")

    def step(self):
        self.events.append("step")

    def stop(self):
        self.events.append("stop")

    def summary(self, **kwargs):
        self.events.append("summary")


class TestAddProfilerStepDecisions(unittest.TestCase):
    def setUp(self):
        # Save and reset the module globals so each test starts from a known
        # state and never leaks into sibling tests.
        self._saved = (
            profiler_module._profiler_step_id,
            profiler_module._profiler_options,
            profiler_module._prof,
        )
        profiler_module._profiler_step_id = 0
        profiler_module._profiler_options = None
        profiler_module._prof = None
        _FakeProfiler.instances = []

        # export_chrome_tracing is called at construction with profile_path and
        # its return value becomes on_trace_ready. Replace it with a sentinel so
        # we can assert profile_path flows through.
        self._export_calls = []

        def _fake_export(path):
            self._export_calls.append(path)
            return ("on_trace_ready", path)

        exit_patcher = mock.patch.object(profiler_module.sys, "exit")
        prof_patcher = mock.patch.object(
            profiler_module.profiler, "Profiler", _FakeProfiler
        )
        export_patcher = mock.patch.object(
            profiler_module.profiler, "export_chrome_tracing", _fake_export
        )
        self.mock_exit = exit_patcher.start()
        prof_patcher.start()
        export_patcher.start()
        self.addCleanup(exit_patcher.stop)
        self.addCleanup(prof_patcher.stop)
        self.addCleanup(export_patcher.stop)

    def tearDown(self):
        (
            profiler_module._profiler_step_id,
            profiler_module._profiler_options,
            profiler_module._prof,
        ) = self._saved
        _FakeProfiler.instances = []

    def test_none_options_is_noop(self):
        add_profiler_step(None)
        self.assertEqual(profiler_module._profiler_step_id, 0)
        self.assertIsNone(profiler_module._profiler_options)
        self.assertIsNone(profiler_module._prof)
        self.assertEqual(_FakeProfiler.instances, [])
        self.mock_exit.assert_not_called()

    def test_first_call_starts_profiler_and_wires_config(self):
        add_profiler_step(
            "batch_range=[1,3];profile_path=/tmp/xyz;exit_on_finished=false"
        )
        # Exactly one profiler is constructed and started (no step yet).
        self.assertEqual(len(_FakeProfiler.instances), 1)
        prof = _FakeProfiler.instances[0]
        self.assertEqual(prof.events, ["start"])
        # batch_range flows into the scheduler window.
        self.assertEqual(prof.kwargs["scheduler"], (1, 3))
        # timer_only default True -> True; record_shapes default False -> False.
        self.assertIs(prof.kwargs["timer_only"], True)
        self.assertIs(prof.kwargs["record_shapes"], False)
        # profile_path flows into export_chrome_tracing and its result becomes
        # on_trace_ready.
        self.assertEqual(self._export_calls, ["/tmp/xyz"])
        self.assertEqual(
            prof.kwargs["on_trace_ready"], ("on_trace_ready", "/tmp/xyz")
        )
        # Step id advanced by one; not yet at the stop step.
        self.assertEqual(profiler_module._profiler_step_id, 1)
        self.assertIsNotNone(profiler_module._prof)
        self.mock_exit.assert_not_called()

    def test_timer_only_and_record_shapes_string_coercion(self):
        add_profiler_step(
            "batch_range=[1,9];timer_only=False;record_shapes=True;"
            "exit_on_finished=false"
        )
        prof = _FakeProfiler.instances[0]
        # "False"/"True" strings are coerced via str(x) == str(True).
        self.assertIs(prof.kwargs["timer_only"], False)
        self.assertIs(prof.kwargs["record_shapes"], True)

    def test_step_progression_starts_once_then_steps_until_stop(self):
        opts = "batch_range=[1,3];exit_on_finished=false"
        # call 1: step_id 0 -> start;            step_id -> 1
        # call 2: step_id 1 -> step; 1 != 3;     step_id -> 2
        # call 3: step_id 2 -> step; 2 != 3;     step_id -> 3
        # call 4: step_id 3 -> step; 3 == 3 stop; step_id -> 4
        for _ in range(4):
            add_profiler_step(opts)

        self.assertEqual(len(_FakeProfiler.instances), 1)
        prof = _FakeProfiler.instances[0]
        self.assertEqual(
            prof.events, ["start", "step", "step", "step", "stop", "summary"]
        )
        self.assertEqual(profiler_module._profiler_step_id, 4)
        # After stop the module clears its handle.
        self.assertIsNone(profiler_module._prof)
        # exit_on_finished=false -> no process exit.
        self.mock_exit.assert_not_called()

    def test_stop_triggers_exit_when_exit_on_finished_true(self):
        opts = "batch_range=[1,2]"  # exit_on_finished defaults to True
        # call 1: step_id 0 -> start;            step_id -> 1
        # call 2: step_id 1 -> step; 1 != 2;     step_id -> 2
        # call 3: step_id 2 -> step; 2 == 2 stop -> sys.exit(0); step_id -> 3
        self.mock_exit.assert_not_called()
        add_profiler_step(opts)
        add_profiler_step(opts)
        self.mock_exit.assert_not_called()
        add_profiler_step(opts)

        prof = _FakeProfiler.instances[0]
        self.assertEqual(
            prof.events, ["start", "step", "step", "stop", "summary"]
        )
        self.mock_exit.assert_called_once_with(0)

    def test_options_are_parsed_once_and_cached(self):
        # First call fixes the options; a later call with a different string
        # must not re-parse (the cached batch_range is retained).
        add_profiler_step("batch_range=[1,5];exit_on_finished=false")
        add_profiler_step("batch_range=[100,200];exit_on_finished=false")
        self.assertEqual(
            profiler_module._profiler_options["batch_range"], [1, 5]
        )
        # Still a single profiler wired to the first window.
        self.assertEqual(len(_FakeProfiler.instances), 1)
        self.assertEqual(_FakeProfiler.instances[0].kwargs["scheduler"], (1, 5))

    @unittest.expectedFailure
    def test_profiler_not_recreated_after_window_completes(self):
        # BUG (src/paddlefleet/utils/profiler.py:139-146): once the window ends,
        # _prof is set to None but _profiler_step_id keeps advancing past
        # batch_range[1]. With exit_on_finished=false the next call sees
        # _prof is None and constructs+starts a brand new profiler that can
        # never reach its stop condition again. Correct behavior: the profiler
        # should be created exactly once and not restart after completion.
        opts = "batch_range=[1,3];exit_on_finished=false"
        for _ in range(6):
            add_profiler_step(opts)
        # Correct expectation (currently fails: the buggy code builds two).
        self.assertEqual(len(_FakeProfiler.instances), 1)


if __name__ == "__main__":
    unittest.main()
