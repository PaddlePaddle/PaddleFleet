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

"""Behavior tests for ``paddlefleet.utils.perf_utils``.

These exercise the real profiling / timing helpers. Expected values are
hand-derived from each function's documented contract and are NEVER produced by
calling the function under test. The NVTX / profiler-control collaborators
(``paddle.base.core.nvprof_*``) and the CUDA memory queries
(``paddle.device.cuda.*``) are external, non-under-test dependencies; they are
mocked with distinguishable responses so the byte->GB conversion, metric wiring,
context-manager ordering and profiler state machine can be verified on a
no-card (CPU) host. Real GPU memory readings and real NVTX side effects are not
validated here. Module globals (``_DEBUG_INFO``, ``_PROFILER_ENABLED``) are
saved and restored so tests do not pollute each other.
"""

import contextlib
import io
import unittest
from unittest.mock import patch

import paddle

from paddlefleet.utils import perf_utils
from paddlefleet.utils.perf_utils import (
    add_record_event,
    memory_info,
    pop_record_event,
    push_record_event,
    register_profile_hook,
    switch_profile,
)

_GIB = 1024 * 1024 * 1024


class MemoryInfoTest(unittest.TestCase):
    def test_memory_info_converts_bytes_to_gigabytes(self):
        # Each metric is fed a distinct, binary-exact byte count so a wrong
        # divisor, wrong .3f formatting, or a swapped label is caught. Expected
        # string is derived by hand: bytes / 1024**3 formatted with 3 decimals.
        with (
            patch(
                "paddle.device.cuda.memory_allocated",
                return_value=int(1.5 * _GIB),
            ),
            patch(
                "paddle.device.cuda.memory_reserved",
                return_value=int(2.25 * _GIB),
            ),
            patch(
                "paddle.device.cuda.max_memory_allocated",
                return_value=int(3.125 * _GIB),
            ),
            patch(
                "paddle.device.cuda.max_memory_reserved",
                return_value=int(0.5 * _GIB),
            ),
        ):
            result = memory_info()
        self.assertEqual(
            result,
            "memory_allocated=1.500 GB, memory_reserved=2.250; "
            "max_memory_allocated=3.125 GB, max_memory_reserved=0.500 GB",
        )


class RegisterProfileHookTest(unittest.TestCase):
    def _guard_debug_info(self):
        orig = perf_utils._DEBUG_INFO
        self.addCleanup(setattr, perf_utils, "_DEBUG_INFO", orig)

    def test_register_profile_hook_fires_hooks_on_forward(self):
        # A real 3-node tree (Sequential + 2 Linear). Recursion must register
        # pre/post hooks on EVERY node, so one full forward emits exactly one
        # enter + one leave line per node. Counting proves the recursion reached
        # the children, not just the root.
        self._guard_debug_info()
        model = paddle.nn.Sequential(
            paddle.nn.Linear(4, 3), paddle.nn.Linear(3, 2)
        )
        register_profile_hook(model, debug="verbose")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            model(paddle.ones([2, 4]))
        out = buf.getvalue()
        self.assertEqual(out.count("[Enter"), 3)
        self.assertEqual(out.count("[Leave"), 3)
        self.assertIn("Enter Sequential forward", out)
        self.assertIn("Leave Sequential forward", out)
        self.assertEqual(out.count("Enter Linear forward"), 2)
        self.assertEqual(out.count("Leave Linear forward"), 2)

    def test_register_profile_hook_list_registers_each(self):
        # A list argument must register hooks on every element. If only the
        # first were registered, the second forward would emit nothing and the
        # enter count would be 1 instead of 2.
        self._guard_debug_info()
        m1 = paddle.nn.Linear(4, 3)
        m2 = paddle.nn.Linear(3, 2)
        register_profile_hook([m1, m2], debug="verbose")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            y = m1(paddle.ones([1, 4]))
            m2(y)
        out = buf.getvalue()
        self.assertEqual(out.count("Enter Linear forward"), 2)
        self.assertEqual(out.count("Leave Linear forward"), 2)

    def test_register_profile_hook_debug_arg_wiring(self):
        # debug=None must NOT overwrite the existing global; an explicit value
        # must replace it. This observes the ``if debug is not None`` branch
        # effect rather than merely asserting no exception.
        self._guard_debug_info()
        perf_utils._DEBUG_INFO = "sentinel-original"
        model = paddle.nn.Linear(2, 2)
        register_profile_hook(model, debug=None)
        self.assertEqual(perf_utils._DEBUG_INFO, "sentinel-original")
        register_profile_hook(model, debug="memory")
        self.assertEqual(perf_utils._DEBUG_INFO, "memory")


class RecordEventTest(unittest.TestCase):
    def test_add_record_event_disabled_is_passthrough(self):
        # With the profiler disabled the context manager must run the body and
        # touch no NVTX collaborator at all.
        ran = []
        with (
            patch("paddle.base.core.nvprof_nvtx_push") as push,
            patch("paddle.base.core.nvprof_nvtx_pop") as pop,
            patch.object(perf_utils, "_PROFILER_ENABLED", False),
        ):
            with add_record_event("region"):
                ran.append(True)
            push.assert_not_called()
            pop.assert_not_called()
        self.assertEqual(ran, [True])

    def test_add_record_event_enabled_pushes_then_pops(self):
        # Enabled: push happens before the body, pop after it, with the exact
        # event name. Ordering is captured so a push/pop swap is rejected.
        calls = []
        body_ran = []
        with (
            patch.object(perf_utils, "_PROFILER_ENABLED", True),
            patch(
                "paddle.base.core.nvprof_nvtx_push",
                side_effect=lambda n: calls.append(("push", n)),
            ),
            patch(
                "paddle.base.core.nvprof_nvtx_pop",
                side_effect=lambda: calls.append(("pop",)),
            ),
        ):
            with add_record_event("region"):
                body_ran.append(True)
                self.assertEqual(calls, [("push", "region")])
        self.assertEqual(body_ran, [True])
        self.assertEqual(calls, [("push", "region"), ("pop",)])

    @unittest.expectedFailure
    def test_add_record_event_pops_on_exception(self):
        # PRODUCTION BUG: add_record_event pushes an NVTX range then yields, but
        # pops only on the normal path (no try/finally). If the body raises, the
        # range is never popped and the profiler's push/pop stack is left
        # unbalanced. Correct behavior is a balanced push+pop even on exception.
        # This asserts the correct contract and is expected to fail until the
        # production code wraps the pop in try/finally. Production is NOT
        # modified here.
        calls = []
        with (
            patch.object(perf_utils, "_PROFILER_ENABLED", True),
            patch(
                "paddle.base.core.nvprof_nvtx_push",
                side_effect=lambda n: calls.append("push"),
            ),
            patch(
                "paddle.base.core.nvprof_nvtx_pop",
                side_effect=lambda: calls.append("pop"),
            ),
        ):
            with self.assertRaises(ValueError):
                with add_record_event("boom"):
                    raise ValueError("boom")
        self.assertEqual(calls, ["push", "pop"])

    def test_push_and_pop_record_event_respect_profiler_flag(self):
        # Disabled: no NVTX calls. Enabled: exactly one push with the event name
        # and one pop.
        with (
            patch("paddle.base.core.nvprof_nvtx_push") as push,
            patch("paddle.base.core.nvprof_nvtx_pop") as pop,
            patch.object(perf_utils, "_PROFILER_ENABLED", False),
        ):
            push_record_event("x")
            pop_record_event()
            push.assert_not_called()
            pop.assert_not_called()

        with (
            patch("paddle.base.core.nvprof_nvtx_push") as push,
            patch("paddle.base.core.nvprof_nvtx_pop") as pop,
            patch.object(perf_utils, "_PROFILER_ENABLED", True),
        ):
            push_record_event("region-A")
            pop_record_event()
            push.assert_called_once_with("region-A")
            pop.assert_called_once_with()


class SwitchProfileTest(unittest.TestCase):
    def _guard_profiler_flag(self):
        orig = perf_utils._PROFILER_ENABLED
        self.addCleanup(setattr, perf_utils, "_PROFILER_ENABLED", orig)

    def test_switch_profile_start_enables_and_pushes(self):
        # At iter_id == start: synchronize + nvprof_start fire, the flag flips
        # to True, and a single range is pushed under the default name
        # "iter_{start}". No stop, no pop.
        self._guard_profiler_flag()
        perf_utils._PROFILER_ENABLED = False
        calls = []
        with (
            patch("paddle.device.cuda.synchronize") as sync,
            patch("paddle.base.core.nvprof_start") as start_fn,
            patch("paddle.base.core.nvprof_stop") as stop_fn,
            patch(
                "paddle.base.core.nvprof_nvtx_push",
                side_effect=lambda n: calls.append(("push", n)),
            ),
            patch(
                "paddle.base.core.nvprof_nvtx_pop",
                side_effect=lambda: calls.append(("pop",)),
            ),
        ):
            switch_profile(iter_id=5, start=5, end=10)
        self.assertTrue(perf_utils._PROFILER_ENABLED)
        sync.assert_called_once_with()
        start_fn.assert_called_once_with()
        stop_fn.assert_not_called()
        self.assertEqual(calls, [("push", "iter_5")])

    def test_switch_profile_end_disables_and_pops(self):
        # At iter_id == end: the open range is popped, the flag flips to False
        # and nvprof_stop fires. No new range is started or pushed.
        self._guard_profiler_flag()
        perf_utils._PROFILER_ENABLED = True
        calls = []
        with (
            patch("paddle.device.cuda.synchronize"),
            patch("paddle.base.core.nvprof_start") as start_fn,
            patch("paddle.base.core.nvprof_stop") as stop_fn,
            patch(
                "paddle.base.core.nvprof_nvtx_push",
                side_effect=lambda n: calls.append(("push", n)),
            ),
            patch(
                "paddle.base.core.nvprof_nvtx_pop",
                side_effect=lambda: calls.append(("pop",)),
            ),
        ):
            switch_profile(iter_id=10, start=5, end=10)
        self.assertFalse(perf_utils._PROFILER_ENABLED)
        stop_fn.assert_called_once_with()
        start_fn.assert_not_called()
        self.assertEqual(calls, [("pop",)])

    def test_switch_profile_middle_rotates_range(self):
        # Between start and end the previous range is closed then a fresh range
        # is opened (pop THEN push), the explicit event name is used, and the
        # enabled flag is left untouched.
        self._guard_profiler_flag()
        perf_utils._PROFILER_ENABLED = True
        calls = []
        with (
            patch("paddle.device.cuda.synchronize"),
            patch("paddle.base.core.nvprof_start") as start_fn,
            patch("paddle.base.core.nvprof_stop") as stop_fn,
            patch(
                "paddle.base.core.nvprof_nvtx_push",
                side_effect=lambda n: calls.append(("push", n)),
            ),
            patch(
                "paddle.base.core.nvprof_nvtx_pop",
                side_effect=lambda: calls.append(("pop",)),
            ),
        ):
            switch_profile(iter_id=7, start=5, end=10, event_name="mid")
        self.assertEqual(calls, [("pop",), ("push", "mid")])
        self.assertTrue(perf_utils._PROFILER_ENABLED)
        start_fn.assert_not_called()
        stop_fn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
