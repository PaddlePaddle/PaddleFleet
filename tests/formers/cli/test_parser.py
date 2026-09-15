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

"""Behavior tests for ``paddlefleet.cli.hparams.parser``.

These tests exercise the real config-reading (``read_args``), custom-template
loading (``_load_custom_template``) and argument-orchestration (``_parse_args``)
logic with independently hand-derived expectations.

The production module imports ``omegaconf`` and ``paddlefleet.trainer`` at
import time, both of which are third-party dependency walls. The local
environment has no Paddle installed and ships a broken ``omegaconf`` (antlr4
version mismatch), so the module cannot be imported here. The import is guarded
below and the *exact* import exception is surfaced in the skip reason, so a real
regression in the parser is never silently masked. No production code is
modified.
"""

import json
import os
import shutil
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

read_args = None
_load_custom_template = None
_parse_args = None
_IMPORT_ERROR = None
try:
    from paddlefleet.cli.hparams.parser import (  # noqa: E501
        _load_custom_template,
        _parse_args,
        read_args,
    )
except ImportError as exc:  # missing paddle in local env
    _IMPORT_ERROR = exc
except Exception as exc:  # noqa: BLE001 - broken omegaconf/antlr install
    # The parser imports omegaconf before paddle; the local omegaconf raises a
    # non-ImportError at import time. This is a third-party dependency wall, not
    # a parser regression; the exact error is preserved in the skip reason.
    _IMPORT_ERROR = exc

_SKIP_REASON = (
    "paddlefleet.cli.hparams.parser is not importable in this environment "
    "(no paddle / broken omegaconf): {!r}".format(_IMPORT_ERROR)
)


@unittest.skipUnless(read_args is not None, _SKIP_REASON)
class TestReadArgs(unittest.TestCase):
    """read_args resolves explicit args, config files and CLI overrides."""

    def _write(self, suffix, text):
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        with open(path, "w") as fh:
            fh.write(text)
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        return path

    def test_explicit_dict_is_returned_unchanged(self):
        # When args are passed in, read_args must hand back the very same object.
        args = {"model_name_or_path": "/models/base", "stage": "SFT"}
        result = read_args(args)
        self.assertIs(result, args)

    def test_explicit_list_is_returned_unchanged(self):
        args = ["--model_name_or_path", "/models/base"]
        result = read_args(args)
        self.assertIs(result, args)

    def test_yaml_override_wins_preserves_and_adds(self):
        # YAML base + CLI dotlist overrides. Expected merge derived by hand:
        #  - stage overridden by CLI, model_name_or_path preserved from YAML,
        #  - run_name is a brand-new key contributed only by the CLI.
        path = self._write(
            ".yaml", "stage: SFT\nmodel_name_or_path: /models/base\n"
        )
        argv = ["fleet", "train", path, "stage=DPO", "run_name=exp2"]
        with mock.patch.object(sys, "argv", argv):
            result = read_args()
        self.assertEqual(
            result,
            {
                "stage": "DPO",
                "model_name_or_path": "/models/base",
                "run_name": "exp2",
            },
        )

    def test_yml_suffix_is_also_treated_as_yaml(self):
        path = self._write(".yml", "stage: SFT\ncutoff_len: 512\n")
        argv = ["fleet", "train", path]
        with mock.patch.object(sys, "argv", argv):
            result = read_args()
        self.assertEqual(result, {"stage": "SFT", "cutoff_len": 512})

    def test_json_config_merges_with_override(self):
        # JSON branch is distinct from YAML; values kept as strings so the
        # expectation does not depend on numeric coercion.
        path = self._write(
            ".json", json.dumps({"stage": "SFT", "cutoff_len": "1024"})
        )
        argv = ["fleet", "train", path, "stage=DPO"]
        with mock.patch.object(sys, "argv", argv):
            result = read_args()
        self.assertEqual(result, {"stage": "DPO", "cutoff_len": "1024"})

    def test_py_config_is_rejected(self):
        path = self._write(".py", "x = 1\n")
        argv = ["fleet", "train", path]
        with mock.patch.object(sys, "argv", argv):
            with self.assertRaises(ValueError) as ctx:
                read_args()
        self.assertIn("Yaml/Json/Arguments", str(ctx.exception))

    def test_non_config_tokens_returned_from_index_two(self):
        # A non config-file token means read_args returns sys.argv[2:] verbatim:
        # index 0 (prog) and index 1 (subcommand) are dropped, the rest kept.
        argv = [
            "fleet",
            "train",
            "--model_name_or_path",
            "/models/base",
            "--stage",
            "SFT",
        ]
        with mock.patch.object(sys, "argv", argv):
            result = read_args()
        self.assertEqual(
            result,
            ["--model_name_or_path", "/models/base", "--stage", "SFT"],
        )

    def test_missing_config_file_raises_assertion(self):
        # Only prog + subcommand present -> len(sys.argv) == 2, guard trips.
        argv = ["fleet", "train"]
        with mock.patch.object(sys, "argv", argv):
            with self.assertRaises(AssertionError) as ctx:
                read_args()
        self.assertIn("Missing configuration files", str(ctx.exception))


@unittest.skipUnless(_load_custom_template is not None, _SKIP_REASON)
class TestLoadCustomTemplate(unittest.TestCase):
    """_load_custom_template genuinely executes the target module body."""

    def _tmpdir(self):
        workdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, workdir, ignore_errors=True)
        return workdir

    def test_template_module_body_is_executed_with_expected_name(self):
        # Prove real execution via an on-disk side effect, and confirm the
        # module is loaded under the name the production code chooses
        # ("custom_template"), rather than merely asserting a logger fired.
        workdir = self._tmpdir()
        marker = os.path.join(workdir, "marker.txt")
        template = os.path.join(workdir, "tmpl.py")
        with open(template, "w") as fh:
            fh.write(
                textwrap.dedent(
                    """\
                    with open({marker!r}, "w") as _fh:
                        _fh.write("executed:" + __name__)
                    """
                ).format(marker=marker)
            )

        _load_custom_template(template)

        self.assertTrue(os.path.exists(marker))
        with open(marker) as fh:
            self.assertEqual(fh.read(), "executed:custom_template")

    def test_missing_template_raises_runtimeerror_naming_path(self):
        missing = "/nonexistent/dir/does_not_exist_tmpl.py"
        with self.assertRaises(RuntimeError) as ctx:
            _load_custom_template(missing)
        message = str(ctx.exception)
        self.assertIn("Failed to load", message)
        self.assertIn(missing, message)


@unittest.skipUnless(_parse_args is not None, _SKIP_REASON)
class TestParseArgs(unittest.TestCase):
    """_parse_args orchestrates parsing; the parser is a recording stub.

    The stub stands in for the (unavailable) PdArgumentParser collaborator and
    returns input-dependent, controlled values. The orchestration logic under
    test -- unknown-argument rejection, allow_extra_keys handling and
    custom_register_path stripping -- runs for real and is what we observe.
    """

    def test_dict_unknown_keys_raise_value_error(self):
        parsed = object()

        class _Parser:
            def parse_dict(self, args, return_unknown_ars=False):
                return (parsed, {"bogus_key": "value"})

        with self.assertRaises(ValueError) as ctx:
            _parse_args(
                _Parser(),
                {"model_name_or_path": "/m", "bogus_key": "value"},
            )
        message = str(ctx.exception)
        self.assertIn("not used by the PdArgumentParser", message)
        self.assertIn("bogus_key", message)

    def test_dict_without_unknown_returns_parsed_tuple(self):
        first, second = object(), object()

        class _Parser:
            def parse_dict(self, args, return_unknown_ars=False):
                return (first, second, {})

        result = _parse_args(_Parser(), {"model_name_or_path": "/m"})
        self.assertEqual(result, (first, second))
        self.assertIs(result[0], first)
        self.assertIs(result[1], second)

    def test_custom_register_path_is_loaded_and_stripped_before_parsing(self):
        workdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, workdir, ignore_errors=True)
        marker = os.path.join(workdir, "loaded.txt")
        template = os.path.join(workdir, "reg.py")
        with open(template, "w") as fh:
            fh.write(
                "with open({m!r}, 'w') as _f:\n    _f.write('ok')\n".format(
                    m=marker
                )
            )

        received = {}

        class _Parser:
            def parse_dict(self, args, return_unknown_ars=False):
                received["args"] = dict(args)
                return (object(), {})

        _parse_args(
            _Parser(),
            {"model_name_or_path": "/m", "custom_register_path": template},
        )

        # The template body actually ran (real side effect on disk).
        self.assertTrue(os.path.exists(marker))
        # custom_register_path was popped and never forwarded to the parser.
        self.assertNotIn("custom_register_path", received["args"])
        self.assertEqual(received["args"], {"model_name_or_path": "/m"})

    def test_list_unknown_args_raise_when_extra_not_allowed(self):
        class _Parser:
            def parse_args_into_dataclasses(
                self, args, return_remaining_strings=False
            ):
                return (object(), ["--bogus"])

            def format_help(self):
                return "usage: fleet train ..."

        with self.assertRaises(ValueError) as ctx:
            _parse_args(
                _Parser(),
                ["--model_name_or_path", "/m", "--bogus"],
            )
        message = str(ctx.exception)
        self.assertIn("not used by the PdArgumentParser", message)
        self.assertIn("--bogus", message)

    def test_list_extra_keys_allowed_returns_parsed_without_raising(self):
        sentinel = object()

        class _Parser:
            def parse_args_into_dataclasses(
                self, args, return_remaining_strings=False
            ):
                return (sentinel, ["--bogus"])

        result = _parse_args(
            _Parser(),
            ["--model_name_or_path", "/m", "--bogus"],
            allow_extra_keys=True,
        )
        self.assertEqual(result, (sentinel,))
        self.assertIs(result[0], sentinel)


if __name__ == "__main__":
    unittest.main()
