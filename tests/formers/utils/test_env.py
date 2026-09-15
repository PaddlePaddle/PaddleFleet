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

"""Behavior tests for paddlefleet.utils.env.

These exercise the real env-parsing / path-resolution helpers in
``src/paddlefleet/utils/env.py``. Expected values are hand-derived from the
documented contract (never produced by calling the function under test), env
state is patched with ``patch.dict`` and always restored, and filesystem
effects are checked on real temp dirs rather than mocked away.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

from paddlefleet.utils.env import (
    PADDLE_WEIGHTS_INDEX_NAME,
    PADDLE_WEIGHTS_NAME,
    PREFIX_CHECKPOINT_DIR,
    SAFE_OPTIMIZER_INDEX_NAME,
    SAFE_OPTIMIZER_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    _get_bool_env,
    _get_pf_home,
    _get_sub_home,
    _get_user_home,
    _re_checkpoint,
)

# Probe keys unlikely to collide with anything already in the environment.
BOOL_KEY = "PF_TEST_BOOL_ENV_PROBE"


class TestGetBoolEnv(unittest.TestCase):
    """_get_bool_env(key, default) -> value.lower() in ("true", "1").

    Independent contract: only the lowercased strings "true" and "1" are
    truthy; every other string (including the Python-truthy "false") is False.
    """

    def test_true_tokens_are_truthy(self):
        # Hand-derived: these lowercase to "true"/"1", the only truthy tokens.
        for value in ("true", "True", "TRUE", "tRuE", "1"):
            with self.subTest(value=value):
                with patch.dict(os.environ, {BOOL_KEY: value}):
                    # default is deliberately "false" so a True result can only
                    # come from consuming the env value, not the default.
                    self.assertIs(_get_bool_env(BOOL_KEY, "false"), True)

    def test_other_strings_are_falsy_and_env_overrides_default(self):
        # default "true" would be truthy if ignored; a present env value must
        # override it, and none of these tokens are "true"/"1".
        for value in (
            "false",
            "False",
            "FALSE",
            "0",
            "2",
            "10",  # not "1"
            "yes",
            "no",
            "on",
            "off",
            "",  # explicit empty overrides default -> "" not in list
            "megatron",  # string-truthiness pitfall: non-empty but not a flag
            "hf",
        ):
            with self.subTest(value=value):
                with patch.dict(os.environ, {BOOL_KEY: value}):
                    self.assertIs(_get_bool_env(BOOL_KEY, "true"), False)

    def test_no_whitespace_stripping(self):
        # Contract does membership on the raw lowercased value, no strip().
        for value in (" true", "true ", " 1", "1 "):
            with self.subTest(value=value):
                with patch.dict(os.environ, {BOOL_KEY: value}):
                    self.assertIs(_get_bool_env(BOOL_KEY, "false"), False)

    def test_default_used_when_unset(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(BOOL_KEY, None)
            self.assertIs(_get_bool_env(BOOL_KEY, "true"), True)
            self.assertIs(_get_bool_env(BOOL_KEY, "True"), True)
            self.assertIs(_get_bool_env(BOOL_KEY, "1"), True)
            self.assertIs(_get_bool_env(BOOL_KEY, "false"), False)
            self.assertIs(_get_bool_env(BOOL_KEY, "0"), False)
            self.assertIs(_get_bool_env(BOOL_KEY, "megatron"), False)

    def test_string_false_default_is_not_python_truthy(self):
        # The pitfall this helper guards against: bool("false") is True in
        # Python, but the flag parser must treat "false" as disabled.
        self.assertTrue(bool("false"))
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(BOOL_KEY, None)
            self.assertIs(_get_bool_env(BOOL_KEY, "false"), False)


class TestGetUserHome(unittest.TestCase):
    def test_reads_home_env(self):
        # On POSIX, expanduser("~") resolves via $HOME; the helper must reflect
        # whatever HOME points at. os.path.expanduser is stdlib (independent).
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"HOME": tmp}):
                self.assertEqual(_get_user_home(), tmp)


class TestGetPfHome(unittest.TestCase):
    def test_returns_existing_dir_from_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"PF_HOME": tmp}):
                self.assertEqual(_get_pf_home(), tmp)

    def test_returns_nonexistent_path_without_creating(self):
        # PF_HOME pointing at a not-yet-existing path is returned verbatim and
        # must NOT be created as a side effect.
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "not_here_yet")
            self.assertFalse(os.path.exists(missing))
            with patch.dict(os.environ, {"PF_HOME": missing}):
                self.assertEqual(_get_pf_home(), missing)
            self.assertFalse(os.path.exists(missing))

    def test_raises_when_pf_home_is_file(self):
        with tempfile.NamedTemporaryFile() as f:
            with patch.dict(os.environ, {"PF_HOME": f.name}):
                with self.assertRaises(RuntimeError):
                    _get_pf_home()

    def test_default_when_unset(self):
        # Independent expected: os.path.join(HOME, ".paddlefleet").
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"HOME": tmp}):
                os.environ.pop("PF_HOME", None)
                expected = os.path.join(tmp, ".paddlefleet")
                self.assertEqual(_get_pf_home(), expected)


class TestGetSubHome(unittest.TestCase):
    def test_creates_and_returns_subdir(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = _get_sub_home("models", parent_home=tmp)
            expected = os.path.join(tmp, "models")
            self.assertEqual(result, expected)
            # Real side effect: the directory is created on disk.
            self.assertTrue(os.path.isdir(expected))

    def test_idempotent_when_dir_already_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = _get_sub_home("datasets", parent_home=tmp)
            second = _get_sub_home("datasets", parent_home=tmp)
            self.assertEqual(first, second)
            self.assertEqual(first, os.path.join(tmp, "datasets"))
            self.assertTrue(os.path.isdir(first))

    def test_creates_nested_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = _get_sub_home(os.path.join("a", "b"), parent_home=tmp)
            expected = os.path.join(tmp, "a", "b")
            self.assertEqual(result, expected)
            self.assertTrue(os.path.isdir(expected))


class TestCheckpointRegex(unittest.TestCase):
    def test_matches_and_captures_step(self):
        m = _re_checkpoint.match("checkpoint-1000")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "1000")

        m0 = _re_checkpoint.match("checkpoint-0")
        self.assertIsNotNone(m0)
        self.assertEqual(m0.group(1), "0")

    def test_rejects_non_checkpoint_names(self):
        for name in (
            "checkpoint",  # no step
            "checkpoint-",  # needs >= 1 digit
            "checkpoint-12a",  # anchored end, trailing non-digit
            "checkpoint-1.5",  # only \d+ allowed
            "xcheckpoint-1",  # anchored start
            "checkpoint-1-2",
            "other-100",
        ):
            with self.subTest(name=name):
                self.assertIsNone(_re_checkpoint.match(name))

    def test_regex_is_built_from_prefix_constant(self):
        # The compiled pattern must derive from PREFIX_CHECKPOINT_DIR, so a
        # freshly built name using that constant should match and capture.
        name = "{}-77".format(PREFIX_CHECKPOINT_DIR)
        m = _re_checkpoint.match(name)
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "77")


class TestWeightNameConventions(unittest.TestCase):
    """Index filenames follow the "<base>.index.json" convention.

    This is a load-bearing invariant for checkpoint sharding code, not just a
    literal restatement: a mismatched index name would break shard discovery.
    """

    def test_index_names_derive_from_base_names(self):
        self.assertEqual(
            SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME + ".index.json"
        )
        self.assertEqual(
            SAFE_OPTIMIZER_INDEX_NAME, SAFE_OPTIMIZER_NAME + ".index.json"
        )
        self.assertEqual(
            PADDLE_WEIGHTS_INDEX_NAME, PADDLE_WEIGHTS_NAME + ".index.json"
        )

    def test_base_names_have_expected_extensions(self):
        self.assertTrue(SAFE_WEIGHTS_NAME.endswith(".safetensors"))
        self.assertTrue(PADDLE_WEIGHTS_NAME.endswith(".pdparams"))


if __name__ == "__main__":
    unittest.main()
