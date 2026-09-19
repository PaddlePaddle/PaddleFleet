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

"""Behavior tests for the CPU-observable pure logic in
paddlefleet.config_logger.

Covered entry points:
  * get_config_logger_path        -- getattr with "" default
  * has_config_logger_enabled     -- "path != ''" gate
  * get_path_count                -- per-path monotonic counter (0,1,2,...)
  * get_path_with_count           -- "{path}.iter{count}" formatting
  * JSONEncoderWithMcoreTypes.default -- recursive stringify branches
  * log_config_to_disk            -- real temp-dir JSON write, self-key
                                     removal, prefix derivation, None guard

paddle is imported at module import time by config_logger, so the whole
module is unimportable on a CPU-only box without paddle. Imports are guarded
and every test is gated behind skipUnless with an honest reason -- a skip
here means "paddle was not installed", never a faked pass.

get_path_count / get_path_with_count / log_config_to_disk mutate the
MODULE-LEVEL counter dict paddlefleet.config_logger.__config_logger_path_counts.
It is snapshotted in setUp and restored via addCleanup so tests cannot leak
counter state into each other. The attribute is reached through getattr with a
string literal on purpose: writing the dunder-prefixed name directly inside a
class body would trigger Python name mangling.
"""

import copy
import dataclasses
import json
import os
import sys
import tempfile
import unittest
from collections import OrderedDict
from types import SimpleNamespace

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

# Module-level global (no name mangling outside a class body) holding the
# per-path counter that several functions under test mutate in place.
_COUNTS_ATTR = "__config_logger_path_counts"

try:
    import paddle

    from paddlefleet import config_logger
    from paddlefleet.config_logger import (
        JSONEncoderWithMcoreTypes,
        get_config_logger_path,
        get_path_count,
        get_path_with_count,
        has_config_logger_enabled,
        log_config_to_disk,
    )

    _IMPORT_OK = True
    _IMPORT_REASON = ""
except ImportError as exc:  # pragma: no cover - environment dependent
    _IMPORT_OK = False
    _IMPORT_REASON = (
        "paddle/paddlefleet not importable on this CPU-only env: "
        f"{type(exc).__name__}: {exc}"
    )
    config_logger = None


@unittest.skipUnless(_IMPORT_OK, _IMPORT_REASON)
class TestConfigLogger(unittest.TestCase):
    """CPU behavior tests for paddlefleet.config_logger pure logic."""

    def setUp(self):
        # Snapshot and restore the module-level counter dict so per-path
        # counter mutations never leak between tests. Reached via getattr with
        # a string to dodge class-body name mangling of the dunder name.
        snapshot = copy.deepcopy(getattr(config_logger, _COUNTS_ATTR))
        self.addCleanup(setattr, config_logger, _COUNTS_ATTR, snapshot)

    # ---- get_config_logger_path -------------------------------------------

    def test_get_config_logger_path_missing_attr_returns_empty_string(self):
        # No config_logger_dir attribute -> getattr default "" is returned.
        config = SimpleNamespace()
        self.assertEqual(get_config_logger_path(config), "")

    def test_get_config_logger_path_returns_configured_dir(self):
        config = SimpleNamespace(config_logger_dir="/tmp/pf_logs")
        self.assertEqual(get_config_logger_path(config), "/tmp/pf_logs")

    # ---- has_config_logger_enabled ----------------------------------------

    def test_has_config_logger_enabled_false_when_empty_string(self):
        # Empty string is the disabled sentinel -> False (not merely falsy).
        config = SimpleNamespace(config_logger_dir="")
        self.assertIs(has_config_logger_enabled(config), False)

    def test_has_config_logger_enabled_false_when_attr_missing(self):
        config = SimpleNamespace()
        self.assertIs(has_config_logger_enabled(config), False)

    def test_has_config_logger_enabled_true_when_set(self):
        config = SimpleNamespace(config_logger_dir="/tmp/pf_logs")
        self.assertIs(has_config_logger_enabled(config), True)

    # ---- get_path_count ----------------------------------------------------

    def test_get_path_count_returns_prior_occurrence_count(self):
        # A fresh path yields the number of prior calls: 0, then 1, then 2.
        path = "/tmp/pf_count_fresh_path_a"
        self.assertEqual(get_path_count(path), 0)
        self.assertEqual(get_path_count(path), 1)
        self.assertEqual(get_path_count(path), 2)

    def test_get_path_count_is_independent_per_path(self):
        # Distinct paths keep independent counters.
        path_a = "/tmp/pf_count_indep_a"
        path_b = "/tmp/pf_count_indep_b"
        self.assertEqual(get_path_count(path_a), 0)
        self.assertEqual(get_path_count(path_b), 0)
        self.assertEqual(get_path_count(path_a), 1)
        self.assertEqual(get_path_count(path_b), 1)
        self.assertEqual(get_path_count(path_a), 2)

    # ---- get_path_with_count ----------------------------------------------

    def test_get_path_with_count_appends_iteration_suffix(self):
        # Exact formatting "{path}.iter{count}" with the counter advancing.
        path = "/tmp/pf_with_count_path"
        self.assertEqual(
            get_path_with_count(path), "/tmp/pf_with_count_path.iter0"
        )
        self.assertEqual(
            get_path_with_count(path), "/tmp/pf_with_count_path.iter1"
        )
        self.assertEqual(
            get_path_with_count(path), "/tmp/pf_with_count_path.iter2"
        )

    # ---- JSONEncoderWithMcoreTypes.default --------------------------------
    #
    # When default() is invoked directly on a container, it takes the custom
    # recursive branch: every child is passed back through default(). A plain
    # int has no matching type-name branch, so it falls through to the
    # try/except tail where json's base default() raises and str(o) is
    # returned. Hence ints inside these containers come back as strings. The
    # expected values below are hand-written literals, not recomputed.

    def test_default_dict_recursively_stringifies_int_values(self):
        encoder = JSONEncoderWithMcoreTypes()
        result = encoder.default({"a": 1, "b": 2})
        self.assertEqual(result, {"a": "1", "b": "2"})

    def test_default_nested_dict_recurses(self):
        encoder = JSONEncoderWithMcoreTypes()
        result = encoder.default({"outer": {"inner": 7}})
        self.assertEqual(result, {"outer": {"inner": "7"}})

    def test_default_list_recursively_stringifies_elements(self):
        encoder = JSONEncoderWithMcoreTypes()
        result = encoder.default([1, 2, 3])
        self.assertEqual(result, ["1", "2", "3"])

    def test_default_ordered_dict_takes_dict_branch(self):
        encoder = JSONEncoderWithMcoreTypes()
        result = encoder.default(OrderedDict([("x", 1), ("y", 2)]))
        # OrderedDict shares the dict branch; the comprehension yields a plain
        # dict with stringified values.
        self.assertEqual(result, {"x": "1", "y": "2"})

    def test_default_dataclass_uses_asdict_and_preserves_types(self):
        # A dataclass instance hits dataclasses.asdict, which (unlike the
        # container branches) preserves the original field types.
        @dataclasses.dataclass
        class _SimpleConfig:
            name: str = "test"
            value: int = 42

        encoder = JSONEncoderWithMcoreTypes()
        result = encoder.default(_SimpleConfig())
        self.assertEqual(result, {"name": "test", "value": 42})

    def test_default_paddle_dtype_returns_str_form(self):
        encoder = JSONEncoderWithMcoreTypes()
        result = encoder.default(paddle.float32)
        self.assertEqual(result, "paddle.float32")

    # ---- log_config_to_disk -----------------------------------------------

    def test_log_config_to_disk_writes_json_at_derived_path(self):
        # rank_str is passed explicitly so parallel_state is not consulted;
        # the fresh tempdir keeps the path counter at iter0, letting the exact
        # output filename be derived by hand.
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SimpleNamespace(config_logger_dir=tmpdir)
            data = {"k": "v"}
            log_config_to_disk(config, data, prefix="pfx", rank_str="1_2_3_4_5")

            expected = os.path.join(tmpdir, "pfx.rank_1_2_3_4_5.iter0.json")
            self.assertEqual(
                os.listdir(tmpdir), ["pfx.rank_1_2_3_4_5.iter0.json"]
            )
            with open(expected, "r") as fp:
                self.assertEqual(json.load(fp), {"k": "v"})

    def test_log_config_to_disk_strips_self_and_uses_class_name_prefix(self):
        # With prefix="" and a "self" entry, the prefix becomes the class name
        # of that entry and "self" is removed from the dumped payload.
        class _Widget:
            pass

        with tempfile.TemporaryDirectory() as tmpdir:
            config = SimpleNamespace(config_logger_dir=tmpdir)
            data = {"self": _Widget(), "x": "y"}
            log_config_to_disk(config, data, prefix="", rank_str="0_0_0_0_0")

            expected = os.path.join(tmpdir, "_Widget.rank_0_0_0_0_0.iter0.json")
            self.assertEqual(
                os.listdir(tmpdir), ["_Widget.rank_0_0_0_0_0.iter0.json"]
            )
            with open(expected, "r") as fp:
                self.assertEqual(json.load(fp), {"x": "y"})
            # The function mutates the caller's dict in place.
            self.assertNotIn("self", data)
            self.assertEqual(data, {"x": "y"})

    def test_log_config_to_disk_none_path_raises_assertion_error(self):
        # config_logger_dir=None makes get_config_logger_path return None,
        # tripping the "assert path is not None" guard.
        config = SimpleNamespace(config_logger_dir=None)
        with self.assertRaises(AssertionError):
            log_config_to_disk(config, {"k": "v"}, rank_str="0_0_0_0_0")


if __name__ == "__main__":
    unittest.main()
