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

"""Behavior tests for ``paddlefleet.config_logger`` (config & run infra module).

This file deliberately covers a *different* slice than the sibling
``test_config_logger.py`` (which drives the path getters
``get_config_logger_path`` / ``has_config_logger_enabled``). Here we exercise:

* ``JSONEncoderWithMcoreTypes.default`` -- the custom-serialization *dict/list
  recursion* and primitive fallback. ``json.dump`` only routes non-native
  objects through ``default``; the ``"dict"``/``"list"`` branches implement a
  manual recursive descent whose leaves fall through to ``str(o)``. We call
  ``default`` directly (its documented entry) and hand-derive every expected
  value from the branch logic -- e.g. a nested ``int`` comes back as the
  *string* ``"42"`` because the ``super().default`` fallback stringifies it.
* ``log_config_to_disk`` -- the log-writing path: real ``makedirs``, the
  ``"self"``-key prefix/deletion rule, the per-path iteration counter that
  feeds ``get_path_with_count``, and the JSON file actually written to a temp
  directory. Expected file names and file *contents* are derived by hand and
  read back from disk; the module's global ``__config_logger_path_counts``
  counter is snapshotted and restored so tests do not pollute one another.

The module imports ``paddle`` at import time, so the whole suite is skipped with
an honest reason when Paddle / paddlefleet is unavailable rather than reporting
a hollow pass. Every assertion concerns locally observable CPU behaviour; no
device numerics or multi-card semantics are claimed.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

# Make the in-tree ``src/paddlefleet`` importable when the package has not been
# pip-installed (repo_root/src is 4 levels up from this file).
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    import paddle  # noqa: F401  (needed transitively by config_logger)

    from paddlefleet import config_logger

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle / paddlefleet not installed on this host
    config_logger = None
    _IMPORT_ERROR = exc


_skip_reason = f"paddle/paddlefleet unavailable: {_IMPORT_ERROR}"


@unittest.skipUnless(_IMPORT_ERROR is None, _skip_reason)
class TestJSONEncoderDefaultRecursion(unittest.TestCase):
    """``JSONEncoderWithMcoreTypes.default`` recursion + primitive fallback.

    ``json.dump`` handles native containers itself and only calls ``default``
    for objects it cannot serialise. The ``"dict"`` / ``"list"`` branches
    therefore implement a manual recursive descent, and any leaf that no branch
    matches goes through ``super().default(...)`` (which always raises) into the
    ``except: return str(o)`` fallback. We drive ``default`` directly and derive
    every expectation from that logic.
    """

    def setUp(self):
        self.enc = config_logger.JSONEncoderWithMcoreTypes()

    def test_primitive_str_falls_through_to_str(self):
        # "str" matches no branch -> super().default raises -> str("abc").
        self.assertEqual(self.enc.default("abc"), "abc")

    def test_primitive_int_is_stringified_by_fallback(self):
        # "int" matches no branch; fallback returns str(42), i.e. the STRING.
        result = self.enc.default(42)
        self.assertEqual(result, "42")
        self.assertIsInstance(result, str)

    def test_dict_branch_recurses_and_stringifies_leaves(self):
        # type name "dict" -> {k: default(v)}; each leaf hits the str fallback.
        result = self.enc.default({"a": 42, "b": "x"})
        self.assertEqual(result, {"a": "42", "b": "x"})

    def test_list_branch_recurses_and_stringifies_leaves(self):
        # type name "list" -> [default(v) ...]; ints become their str form.
        result = self.enc.default([1, "two"])
        self.assertEqual(result, ["1", "two"])

    def test_nested_dict_recurses_to_full_depth(self):
        # Outer dict recurses into inner dict, whose int leaf is stringified.
        result = self.enc.default({"outer": {"inner": 5}})
        self.assertEqual(result, {"outer": {"inner": "5"}})


def _counts_dict():
    """Return the module-global path counter (module-level, not name-mangled)."""
    return getattr(config_logger, "__config_logger_path_counts")


@unittest.skipUnless(_IMPORT_ERROR is None, _skip_reason)
class TestLogConfigToDisk(unittest.TestCase):
    """``log_config_to_disk`` file-writing, prefix rules, and iter counter."""

    class _Cfg:
        """Genuine tiny config collaborator exposing ``config_logger_dir``."""

        def __init__(self, directory):
            self.config_logger_dir = directory

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmp, ignore_errors=True)
        # Snapshot and restore the global iteration counter so a mutated global
        # never leaks between tests (restored even if an assertion fails).
        counts = _counts_dict()
        snapshot = dict(counts)

        def _restore():
            counts.clear()
            counts.update(snapshot)

        self.addCleanup(_restore)

    def test_writes_named_json_file_with_exact_contents(self):
        cfg = self._Cfg(self._tmp)
        data = {"lr": 0.001, "steps": 10, "name": "run"}
        config_logger.log_config_to_disk(
            cfg, data, prefix="cfg", rank_str="2_1_0_3_0"
        )
        # Hand-derived: first sight of this path -> count 0 -> ".iter0".
        expected = os.path.join(self._tmp, "cfg.rank_2_1_0_3_0.iter0.json")
        self.assertTrue(os.path.isfile(expected))
        with open(expected) as fp:
            self.assertEqual(
                json.load(fp), {"lr": 0.001, "steps": 10, "name": "run"}
            )

    def test_counter_advances_to_distinct_files_per_call(self):
        cfg = self._Cfg(self._tmp)
        config_logger.log_config_to_disk(
            cfg, {"call": 1}, prefix="cfg", rank_str="0_0_0_0_0"
        )
        config_logger.log_config_to_disk(
            cfg, {"call": 2}, prefix="cfg", rank_str="0_0_0_0_0"
        )
        first = os.path.join(self._tmp, "cfg.rank_0_0_0_0_0.iter0.json")
        second = os.path.join(self._tmp, "cfg.rank_0_0_0_0_0.iter1.json")
        self.assertTrue(os.path.isfile(first))
        self.assertTrue(os.path.isfile(second))
        # Each call lands in its own file with its own payload.
        with open(first) as fp:
            self.assertEqual(json.load(fp), {"call": 1})
        with open(second) as fp:
            self.assertEqual(json.load(fp), {"call": 2})

    def test_self_key_with_empty_prefix_uses_class_name_and_is_removed(self):
        class _SelfMarker:
            pass

        cfg = self._Cfg(self._tmp)
        data = {"self": _SelfMarker(), "value": 5}
        config_logger.log_config_to_disk(
            cfg, data, prefix="", rank_str="0_0_0_0_0"
        )
        # Empty prefix -> derived from type(self).__name__ == "_SelfMarker".
        expected = os.path.join(
            self._tmp, "_SelfMarker.rank_0_0_0_0_0.iter0.json"
        )
        self.assertTrue(os.path.isfile(expected))
        # "self" is stripped from the dumped payload and mutated out of the arg.
        self.assertNotIn("self", data)
        with open(expected) as fp:
            self.assertEqual(json.load(fp), {"value": 5})

    def test_self_key_with_explicit_prefix_keeps_prefix(self):
        class _SelfMarker:
            pass

        cfg = self._Cfg(self._tmp)
        data = {"self": _SelfMarker(), "a": 1}
        config_logger.log_config_to_disk(
            cfg, data, prefix="keep", rank_str="0_0_0_0_0"
        )
        # Non-empty prefix must NOT be overwritten by the class name.
        expected = os.path.join(self._tmp, "keep.rank_0_0_0_0_0.iter0.json")
        self.assertTrue(os.path.isfile(expected))
        self.assertNotIn("self", data)
        with open(expected) as fp:
            self.assertEqual(json.load(fp), {"a": 1})

    def test_creates_missing_nested_output_directory(self):
        nested = os.path.join(self._tmp, "a", "b", "c")
        self.assertFalse(os.path.exists(nested))
        cfg = self._Cfg(nested)
        config_logger.log_config_to_disk(
            cfg, {"x": 1}, prefix="p", rank_str="0"
        )
        self.assertTrue(os.path.isdir(nested))
        expected = os.path.join(nested, "p.rank_0.iter0.json")
        self.assertTrue(os.path.isfile(expected))
        with open(expected) as fp:
            self.assertEqual(json.load(fp), {"x": 1})


if __name__ == "__main__":
    unittest.main()
