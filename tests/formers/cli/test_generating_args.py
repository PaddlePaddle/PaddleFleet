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

"""Behavior tests for the generating-args CLI surface.

These tests drive ``GeneratingArguments`` through its real consumer,
``PdArgumentParser`` (the same parser used by ``paddlefleet.cli.hparams.parser``
to build train/eval/server argument sets). Rather than assigning a dataclass
field and reading it straight back, we hand values to the parser as raw command
line strings and check that the parser (a) wires every field to a ``--flag``,
(b) applies the declared field type so a string ``"2048"`` becomes ``int``
2048 rather than the string, and (c) reproduces the boolean ``--no_stream``
complement that argparse-based parsing relies on. Expected values are derived
by hand from the argument grammar, not read from the dataclass definition.
"""

import unittest

# The local environment has NO paddle installed. Both the parser and the
# dataclass live behind lazy-import packages that may transitively require
# paddle (or omegaconf) at import time, so guard the imports honestly and skip
# with a real reason when the dependency chain cannot be satisfied.
_IMPORT_ERROR = None
try:
    from paddlefleet.cli.hparams.generating_args import (
        GeneratingArguments,
        StreamOptions,
    )
    from paddlefleet.trainer import PdArgumentParser
except ImportError as exc:  # pragma: no cover - depends on local env
    GeneratingArguments = None
    StreamOptions = None
    PdArgumentParser = None
    _IMPORT_ERROR = exc


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddlefleet CLI imports unavailable (no paddle in local env): {_IMPORT_ERROR}",
)
class TestGeneratingArgumentsThroughParser(unittest.TestCase):
    """Drive GeneratingArguments through the real PdArgumentParser consumer."""

    def _parse(self, argv):
        """Build the parser fresh and return the parsed GeneratingArguments.

        ``return_remaining_strings=False`` makes the parser raise if any token
        is not consumed, so a mistyped/unwired flag surfaces as an error rather
        than being silently dropped.
        """
        parser = PdArgumentParser([GeneratingArguments])
        (parsed,) = parser.parse_args_into_dataclasses(
            args=list(argv), look_for_args_file=False
        )
        self.assertIsInstance(parsed, GeneratingArguments)
        return parsed

    def test_empty_argv_yields_hand_derived_defaults(self):
        """With no flags the parser must expose each field's declared default.

        Values below are written out by hand from the argument grammar; if a
        default were changed or a field dropped from the CLI surface, the exact
        comparison fails.
        """
        args = self._parse([])
        self.assertEqual(args.max_new_tokens, 1024)
        self.assertEqual(args.min_tokens, 0)
        self.assertEqual(args.temperature, 0.95)
        self.assertEqual(args.top_p, 0.7)
        self.assertEqual(args.frequency_penalty, 0.0)
        self.assertEqual(args.presence_penalty, 0.0)
        self.assertEqual(args.repetition_penalty, 1.0)
        self.assertIs(args.stream, True)
        self.assertIs(args.enable_thinking, False)
        # stream_options has no scalar default; it must arrive as None, not a
        # constructed StreamOptions or an empty object.
        self.assertIsNone(args.stream_options)

    def test_numeric_strings_are_converted_to_declared_types(self):
        """CLI tokens are strings; the parser must coerce them via field types.

        This is the behavior a self-assign test cannot show: `--max_new_tokens
        2048` must land as the int 2048, not the string "2048".
        """
        args = self._parse(
            [
                "--max_new_tokens",
                "2048",
                "--min_tokens",
                "10",
                "--temperature",
                "0.5",
                "--top_p",
                "0.9",
                "--frequency_penalty",
                "0.25",
                "--presence_penalty",
                "0.3",
                "--repetition_penalty",
                "1.2",
            ]
        )
        self.assertEqual(args.max_new_tokens, 2048)
        self.assertIsInstance(args.max_new_tokens, int)
        self.assertNotIsInstance(args.max_new_tokens, str)
        self.assertEqual(args.min_tokens, 10)
        self.assertIsInstance(args.min_tokens, int)
        self.assertEqual(args.temperature, 0.5)
        self.assertIsInstance(args.temperature, float)
        self.assertEqual(args.top_p, 0.9)
        self.assertEqual(args.frequency_penalty, 0.25)
        self.assertEqual(args.presence_penalty, 0.3)
        self.assertEqual(args.repetition_penalty, 1.2)
        # Untouched fields keep their defaults.
        self.assertIs(args.stream, True)
        self.assertIs(args.enable_thinking, False)

    def test_stream_default_true_adds_no_stream_complement(self):
        """`stream` defaults True, so the parser must emit a `--no_stream` flag.

        Order matters in the parser: the positive `--stream` (default True) is
        added before the `--no_stream` store_false complement, so bare parsing
        keeps True while `--no_stream` flips it. This complement only exists
        because the default is True.
        """
        self.assertIs(self._parse([]).stream, True)
        self.assertIs(self._parse(["--no_stream"]).stream, False)
        # `--stream false` goes through strtobool.
        self.assertIs(self._parse(["--stream", "false"]).stream, False)
        # `--stream` with no value uses the const (True).
        self.assertIs(self._parse(["--stream"]).stream, True)

    def test_enable_thinking_bool_parsing(self):
        """`enable_thinking` defaults False and parses truthy strings.

        Because the default is False (not True) there is no `--no_enable_thinking`
        complement; the flag alone selects the const True, and explicit values
        route through strtobool.
        """
        self.assertIs(self._parse([]).enable_thinking, False)
        self.assertIs(self._parse(["--enable_thinking"]).enable_thinking, True)
        self.assertIs(
            self._parse(["--enable_thinking", "true"]).enable_thinking, True
        )
        self.assertIs(
            self._parse(["--enable_thinking", "false"]).enable_thinking, False
        )
        # There must be no auto-generated negative complement for a False
        # default, so "--no_enable_thinking" is never wired to a flag and stays
        # unconsumed. ``_parse`` uses ``return_remaining_strings=False``, so the
        # custom parser rejects the leftover token by raising ``ValueError``
        # (not ``SystemExit``); see ``common_parse`` in
        # src/paddlefleet/trainer/argparser.py (around line 295), which raises
        # "Some specified arguments are not used by the PdArgumentParser: [...]".
        with self.assertRaises(ValueError) as ctx:
            self._parse(["--no_enable_thinking"])
        self.assertIn("not used", str(ctx.exception))

    def test_combined_flags_are_consumed_independently(self):
        """A mixed invocation must set each field from its own token."""
        args = self._parse(
            [
                "--max_new_tokens",
                "512",
                "--min_tokens",
                "5",
                "--temperature",
                "0.8",
                "--top_p",
                "0.95",
                "--repetition_penalty",
                "1.1",
                "--no_stream",
                "--enable_thinking",
            ]
        )
        self.assertEqual(args.max_new_tokens, 512)
        self.assertEqual(args.min_tokens, 5)
        self.assertEqual(args.temperature, 0.8)
        self.assertEqual(args.top_p, 0.95)
        self.assertEqual(args.repetition_penalty, 1.1)
        self.assertIs(args.stream, False)
        self.assertIs(args.enable_thinking, True)
        # Fields not named on the command line retain their declared defaults.
        self.assertEqual(args.frequency_penalty, 0.0)
        self.assertEqual(args.presence_penalty, 0.0)


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddlefleet CLI imports unavailable (no paddle in local env): {_IMPORT_ERROR}",
)
class TestStreamOptions(unittest.TestCase):
    """StreamOptions exists to expose its __dict__ as URL parameters.

    Per the class docstring, the whole point of the camel-cased attributes is
    that ``__dict__`` can be turned directly into request parameters, so the
    full attribute set (not just individual reads) is the contract worth
    pinning down.
    """

    def test_default_dict_matches_url_parameter_contract(self):
        opts = StreamOptions()
        # Full mapping derived by hand from __init__; a renamed/dropped/added
        # attribute would break the URL-parameter payload and this comparison.
        self.assertEqual(
            vars(opts),
            {
                "count": 20,
                "ranked": "newest",
                "unreadOnly": False,
                "newerThan": None,
                "_max_count": 100,
                "continuation": None,
            },
        )

    def test_max_count_argument_only_changes_max_count(self):
        """The single constructor parameter must land in _max_count alone.

        The other five attributes are fixed in __init__ regardless of the
        argument, so passing max_count must not disturb them.
        """
        opts = StreamOptions(max_count=50)
        self.assertEqual(opts._max_count, 50)
        # Everything else stays at the fixed defaults.
        self.assertEqual(opts.count, 20)
        self.assertEqual(opts.ranked, "newest")
        self.assertIs(opts.unreadOnly, False)
        self.assertIsNone(opts.newerThan)
        self.assertIsNone(opts.continuation)


if __name__ == "__main__":
    unittest.main()
