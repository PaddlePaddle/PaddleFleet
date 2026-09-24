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

"""Behavior unit tests for paddlefleet.training.global_vars.

Scope: the timers build helper (``_set_timers``) and the non-asserting
profile-timers / training-logs accessors, plus how ``destroy_global_vars`` and
``unset_global_variables`` clear that slice of module globals. The get/set_args
lifecycle is intentionally left to the sibling ``test_global_vars.py`` and is
not duplicated here.
"""

import os
import sys
import unittest

_REPO_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")
)
if os.path.isdir(_REPO_SRC) and _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)

try:
    import paddlefleet.training.global_vars as gv
    from paddlefleet.timers import Timers

    _IMPORT_ERROR = None
except ImportError as exc:  # honest: no paddle -> the package import fails
    gv = None
    Timers = None
    _IMPORT_ERROR = exc

_SKIP_REASON = (
    "paddlefleet.training.global_vars and its paddle dependency are not "
    f"importable in this environment ({_IMPORT_ERROR!r}); the control logic "
    "under test was not executed."
)


class _GlobalVarsIsolation(unittest.TestCase):
    """Snapshot every module global touched here; restore it in tearDown.

    Antipattern #11: mutating module-level globals without restoring them
    poisons sibling tests in the same process. We capture the real originals
    and reinstate them even if a test fails.
    """

    def setUp(self):
        self._snapshot = {
            "_GLOBAL_ARGS": gv._GLOBAL_ARGS,
            "_GLOBAL_TIMERS": gv._GLOBAL_TIMERS,
            "_GLOBAL_PROFILE_TIMERS": gv._GLOBAL_PROFILE_TIMERS,
            "_GLOBAL_TRAINING_LOGS": gv._GLOBAL_TRAINING_LOGS,
        }
        # Clean slate so each test observes the production logic, not leftovers.
        gv._GLOBAL_ARGS = None
        gv._GLOBAL_TIMERS = None
        gv._GLOBAL_PROFILE_TIMERS = None
        gv._GLOBAL_TRAINING_LOGS = None

    def tearDown(self):
        for name, value in self._snapshot.items():
            setattr(gv, name, value)


@unittest.skipUnless(gv is not None, _SKIP_REASON)
class TestProfileTimersAccessors(_GlobalVarsIsolation):
    """get_profile_timers is a plain accessor with no initialization guard."""

    def test_returns_none_when_unset_without_raising(self):
        # Unlike get_args/get_timers, get_profile_timers has no
        # _ensure_var_is_initialized guard: unset must yield None, not raise.
        self.assertIsNone(gv.get_profile_timers())

    def test_getter_reflects_latest_setter_by_identity(self):
        first = object()
        second = object()
        gv.set_profile_timers(first)
        self.assertIs(gv.get_profile_timers(), first)
        # Overwrite: the getter reads the live global, latest write wins.
        gv.set_profile_timers(second)
        self.assertIs(gv.get_profile_timers(), second)
        self.assertIsNot(gv.get_profile_timers(), first)


@unittest.skipUnless(gv is not None, _SKIP_REASON)
class TestTrainingLogsAccessors(_GlobalVarsIsolation):
    """get_global_training_logs mirrors the profile-timers accessor contract."""

    def test_returns_none_when_unset_without_raising(self):
        self.assertIsNone(gv.get_global_training_logs())

    def test_getter_reflects_latest_setter_by_identity(self):
        first = object()
        second = object()
        gv.set_global_training_logs(first)
        self.assertIs(gv.get_global_training_logs(), first)
        gv.set_global_training_logs(second)
        self.assertIs(gv.get_global_training_logs(), second)
        self.assertIsNot(gv.get_global_training_logs(), first)


@unittest.skipUnless(gv is not None, _SKIP_REASON)
class TestTimersBuildHelper(_GlobalVarsIsolation):
    """_set_timers builds a real Timers; get_timers guards initialization."""

    def test_get_timers_raises_exact_message_when_unset(self):
        with self.assertRaises(AssertionError) as ctx:
            gv.get_timers()
        # _ensure_var_is_initialized(var, "timers") formats
        # f"{name} is not initialized." -> "timers is not initialized."
        self.assertEqual(str(ctx.exception), "timers is not initialized.")

    def test_set_timers_builds_fresh_empty_timers(self):
        gv._set_timers()
        built = gv.get_timers()
        # Genuine collaborator: _set_timers constructs Timers(), whose registry
        # starts empty (Timers.__init__ sets self.timers = {}).
        self.assertIsInstance(built, Timers)
        self.assertEqual(built.timers, {})
        # get_timers returns the exact object _set_timers stored.
        self.assertIs(built, gv._GLOBAL_TIMERS)

    def test_set_timers_rejects_second_initialization(self):
        gv._set_timers()
        with self.assertRaises(AssertionError) as ctx:
            gv._set_timers()
        # _ensure_var_is_not_initialized(var, "timers") formats
        # f"{name} is already initialized." -> "timers is already initialized."
        self.assertEqual(str(ctx.exception), "timers is already initialized.")


@unittest.skipUnless(gv is not None, _SKIP_REASON)
class TestSetGlobalVariablesBuildsTimers(_GlobalVarsIsolation):
    """set_global_variables must run the _set_timers build helper."""

    def test_timers_usable_after_set_global_variables(self):
        args = object()
        # Before: no timers registered, so get_timers guards and raises.
        with self.assertRaises(AssertionError):
            gv.get_timers()
        gv.set_global_variables(args)
        built = gv.get_timers()
        # set_global_variables delegated to _set_timers, yielding a real,
        # freshly-constructed Timers rather than merely flipping a flag.
        self.assertIsInstance(built, Timers)
        self.assertEqual(built.timers, {})


@unittest.skipUnless(gv is not None, _SKIP_REASON)
class TestDestroyAndUnsetClearTimersSlice(_GlobalVarsIsolation):
    """destroy_global_vars and unset_global_variables both clear the slice."""

    def _arrange_populated_slice(self):
        profile = object()
        logs = object()
        gv._set_timers()
        gv.set_profile_timers(profile)
        gv.set_global_training_logs(logs)
        # Preconditions genuinely populated via production setters (not asserting
        # an already-empty container).
        self.assertIsInstance(gv.get_timers(), Timers)
        self.assertIs(gv.get_profile_timers(), profile)
        self.assertIs(gv.get_global_training_logs(), logs)

    def test_destroy_global_vars_clears_slice(self):
        self._arrange_populated_slice()
        gv.destroy_global_vars()
        self.assertIsNone(gv.get_profile_timers())
        self.assertIsNone(gv.get_global_training_logs())
        with self.assertRaises(AssertionError) as ctx:
            gv.get_timers()
        self.assertEqual(str(ctx.exception), "timers is not initialized.")

    def test_unset_global_variables_clears_slice(self):
        self._arrange_populated_slice()
        gv.unset_global_variables()
        self.assertIsNone(gv.get_profile_timers())
        self.assertIsNone(gv.get_global_training_logs())
        with self.assertRaises(AssertionError) as ctx:
            gv.get_timers()
        self.assertEqual(str(ctx.exception), "timers is not initialized.")


if __name__ == "__main__":
    unittest.main()
