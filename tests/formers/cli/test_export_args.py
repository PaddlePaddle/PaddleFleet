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

"""Behavior tests for ExportArguments driven through its real CLI consumer.

ExportArguments only declares ``copy_tokenizer: bool = field(default=True)``.
Merely constructing the dataclass and reading the field back would just echo
the value the test set (self-assign/self-read), so instead these tests drive
the field through ``PdArgumentParser`` -- the parser the export command
actually uses (see ``paddlefleet/cli/hparams/parser.py`` ``_EXPORT_ARGS``).
That exercises the real argument synthesis: a ``bool`` field defaulting to
True gets a ``--copy_tokenizer`` option (``type=strtobool``, ``nargs="?"``,
``const=True``) plus an auto-generated ``--no_copy_tokenizer`` complement.
Expected values are hand-derived from those documented semantics, not read
back from the dataclass.
"""

import unittest

try:
    from paddlefleet.cli.hparams.export_args import ExportArguments
    from paddlefleet.trainer.argparser import PdArgumentParser

    _IMPORT_ERROR = None
except ImportError as exc:  # local env has no paddle / omegaconf
    ExportArguments = None
    PdArgumentParser = None
    _IMPORT_ERROR = exc


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddlefleet CLI parser dependencies unavailable: {_IMPORT_ERROR}",
)
class ExportArgumentsParsingTest(unittest.TestCase):
    """Verify copy_tokenizer flows correctly through PdArgumentParser."""

    def _parse(self, argv):
        # A fresh parser per call: PdArgumentParser mutates dataclass field
        # types in place during argument synthesis, so reuse could mask bugs.
        parser = PdArgumentParser([ExportArguments])
        (export_args,) = parser.parse_args_into_dataclasses(
            args=argv, look_for_args_file=False
        )
        self.assertIsInstance(export_args, ExportArguments)
        return export_args

    def test_default_enables_tokenizer_copy(self):
        # No CLI value: the field default (True) must reach the dataclass.
        # This resolves through argparse default handling, not a direct set.
        parsed = self._parse([])
        self.assertIs(parsed.copy_tokenizer, True)

    def test_no_flag_complement_disables_copy(self):
        # --no_copy_tokenizer exists ONLY because the default is True; the
        # parser synthesizes it as store_false on the same dest. If that
        # synthesis were dropped, this arg would be unknown and parsing would
        # fail, so the flag both parses and flips the value to False.
        parsed = self._parse(["--no_copy_tokenizer"])
        self.assertIs(parsed.copy_tokenizer, False)

    def test_explicit_false_value_is_consumed(self):
        # The provided value must actually be parsed via strtobool, not
        # ignored in favor of the True default.
        parsed = self._parse(["--copy_tokenizer", "false"])
        self.assertIs(parsed.copy_tokenizer, False)

    def test_strtobool_accepts_multiple_truthy_falsy_spellings(self):
        # Independent expectations from strtobool's accepted vocabulary:
        # yes/true/t/y/1 -> True; no/false/f/n/0 -> False (case-insensitive).
        truthy = ["yes", "true", "T", "y", "1"]
        falsy = ["no", "false", "F", "n", "0"]
        for token in truthy:
            self.assertIs(
                self._parse(["--copy_tokenizer", token]).copy_tokenizer,
                True,
                msg=f"{token!r} should parse as True",
            )
        for token in falsy:
            self.assertIs(
                self._parse(["--copy_tokenizer", token]).copy_tokenizer,
                False,
                msg=f"{token!r} should parse as False",
            )

    def test_bare_flag_uses_const_true(self):
        # nargs="?" with const=True: --copy_tokenizer given with no value
        # resolves to the const, True. Paired with the "0" case below this
        # distinguishes const from the value path (both are exercised).
        self.assertIs(self._parse(["--copy_tokenizer"]).copy_tokenizer, True)
        self.assertIs(
            self._parse(["--copy_tokenizer", "0"]).copy_tokenizer, False
        )

    def test_invalid_truthy_value_is_rejected(self):
        # strtobool raises ArgumentTypeError on unrecognized tokens, which
        # argparse turns into a parser error (SystemExit). A silently accepted
        # value would be a real bug, so we assert the rejection contract.
        with self.assertRaises(SystemExit):
            self._parse(["--copy_tokenizer", "maybe"])


if __name__ == "__main__":
    unittest.main()
