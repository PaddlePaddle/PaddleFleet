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

"""Behavior tests for the args/ensure/lifecycle logic in
paddlefleet.training.global_vars.

Scope (see sibling test_global_vars_2.py which concurrently covers the
timers/tensorboard/wandb getters):
  * get_args / set_args
  * _ensure_var_is_initialized / _ensure_var_is_not_initialized
  * the "already initialized" / "not initialized" guards in
    set_global_variables
  * the unset_global_variables / destroy_global_vars lifecycle for args

global_vars mutates MODULE-LEVEL globals; every global this file touches is
snapshotted in setUp and restored in tearDown so tests cannot leak state
into each other or into the concurrent sibling.
"""

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "src"))
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Honest capability probe: global_vars -> paddlefleet.timers -> paddle.
# Only a genuine missing-dependency (ImportError) is allowed to skip; any
# other error must surface as a real failure rather than a fake pass.
try:
    import paddlefleet.training.global_vars as gv

    _IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - depends on environment
    gv = None
    _IMPORT_ERROR = exc

# Module globals that this file reads or mutates (directly or via the
# lifecycle helpers, which clear all four).
_TOUCHED_GLOBALS = (
    "_GLOBAL_ARGS",
    "_GLOBAL_TIMERS",
    "_GLOBAL_PROFILE_TIMERS",
    "_GLOBAL_TRAINING_LOGS",
)


@unittest.skipUnless(
    gv is not None,
    f"paddlefleet.training.global_vars not importable: {_IMPORT_ERROR}",
)
class TestGlobalVarsArgsLifecycle(unittest.TestCase):
    """Args get/set + ensure guards + unset/destroy lifecycle."""

    def setUp(self):
        # Snapshot every global we may touch, then establish a known clean
        # precondition (uninitialized) for each test.
        self._snapshot = {name: getattr(gv, name) for name in _TOUCHED_GLOBALS}
        for name in _TOUCHED_GLOBALS:
            setattr(gv, name, None)

    def tearDown(self):
        # Restore the exact pre-test module state, even if the test failed.
        for name, value in self._snapshot.items():
            setattr(gv, name, value)

    def test_set_args_writes_module_global_and_get_args_returns_same_object(
        self,
    ):
        sentinel = {"lr": 0.001, "name": "run-A"}
        gv.set_args(sentinel)
        # set_args must write the module-level global itself...
        self.assertIs(gv._GLOBAL_ARGS, sentinel)
        # ...and get_args must return that very object, not a copy/wrapper.
        self.assertIs(gv.get_args(), sentinel)

    def test_get_args_before_initialization_raises_with_exact_message(self):
        # setUp left _GLOBAL_ARGS = None -> guard must fire.
        with self.assertRaises(AssertionError) as ctx:
            gv.get_args()
        self.assertEqual(str(ctx.exception), "args is not initialized.")

    def test_ensure_var_is_initialized_uses_is_not_none_not_truthiness(self):
        # 0 is falsy but is NOT None -> must pass the guard. This would fail
        # if the guard were `assert var` instead of `assert var is not None`.
        self.assertIsNone(gv._ensure_var_is_initialized(0, "widget"))
        self.assertIsNone(gv._ensure_var_is_initialized("", "widget"))
        # None -> AssertionError carrying the supplied name.
        with self.assertRaises(AssertionError) as ctx:
            gv._ensure_var_is_initialized(None, "widget")
        self.assertEqual(str(ctx.exception), "widget is not initialized.")

    def test_ensure_var_is_not_initialized_uses_is_none_not_truthiness(self):
        # None -> passes (still uninitialized).
        self.assertIsNone(gv._ensure_var_is_not_initialized(None, "widget"))
        # 0 is falsy but NOT None -> must be treated as already-initialized.
        # This would wrongly pass if the guard used `assert not var`.
        with self.assertRaises(AssertionError) as ctx:
            gv._ensure_var_is_not_initialized(0, "widget")
        self.assertEqual(str(ctx.exception), "widget is already initialized.")

    def test_set_args_overwrites_previous_value_without_any_guard(self):
        first = {"tag": "first"}
        second = {"tag": "second"}
        gv.set_args(first)
        # Unlike set_global_variables, set_args has no re-init guard: a second
        # call must silently replace the first object.
        gv.set_args(second)
        self.assertIs(gv._GLOBAL_ARGS, second)
        self.assertIs(gv.get_args(), second)

    def test_set_global_variables_rejects_none_and_leaves_args_unset(self):
        with self.assertRaises(AssertionError):
            gv.set_global_variables(None)
        # The `assert args is not None` fires before anything is stored.
        self.assertIsNone(gv._GLOBAL_ARGS)
        with self.assertRaises(AssertionError) as ctx:
            gv.get_args()
        self.assertEqual(str(ctx.exception), "args is not initialized.")

    def test_set_global_variables_rejects_reinitialization_and_keeps_existing(
        self,
    ):
        existing = {"tag": "already-here"}
        gv.set_args(existing)
        new = {"tag": "rejected"}
        with self.assertRaises(AssertionError) as ctx:
            gv.set_global_variables(new)
        # Guard message and, crucially, the existing object is untouched
        # because the guard fires before set_args runs.
        self.assertEqual(str(ctx.exception), "args is already initialized.")
        self.assertIs(gv._GLOBAL_ARGS, existing)

    def test_unset_global_variables_clears_args_to_uninitialized(self):
        gv.set_args({"tag": "live"})
        gv.unset_global_variables()
        self.assertIsNone(gv._GLOBAL_ARGS)
        # Real proof it was cleared: the read guard fires again.
        with self.assertRaises(AssertionError) as ctx:
            gv.get_args()
        self.assertEqual(str(ctx.exception), "args is not initialized.")

    def test_unset_global_variables_is_idempotent(self):
        gv.set_args({"tag": "live"})
        gv.unset_global_variables()
        # A second unset on already-clear state must not raise.
        gv.unset_global_variables()
        self.assertIsNone(gv._GLOBAL_ARGS)

    def test_destroy_global_vars_clears_args_to_uninitialized(self):
        gv.set_args({"tag": "live"})
        gv.destroy_global_vars()
        self.assertIsNone(gv._GLOBAL_ARGS)
        with self.assertRaises(AssertionError) as ctx:
            gv.get_args()
        self.assertEqual(str(ctx.exception), "args is not initialized.")


if __name__ == "__main__":
    unittest.main()
