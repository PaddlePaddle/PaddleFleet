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

"""Behavior tests for ServerArguments driven through its real consumer,
PdArgumentParser (the same parser used by _parse_server_args/get_server_args).

These tests exercise CLI-style parsing: string tokens must be converted to the
declared field types, distinct ports must not be cross-wired, and the boolean
field must follow strtobool semantics. Expected values are hand-derived from the
field defaults and the parser's documented conversion rules, not read back from
a value the test just assigned.
"""

import unittest
from argparse import ArgumentTypeError

# Import real production APIs. Importing the paddlefleet package triggers its
# package __init__ (parallel_state, etc.), which imports paddle. The local
# environment has NO paddle installed, so guard the import honestly and skip
# rather than swallow unrelated failures.
try:
    from paddlefleet.cli.hparams.server_args import ServerArguments
    from paddlefleet.trainer.argparser import PdArgumentParser, strtobool

    _IMPORT_ERROR = None
except ImportError as exc:  # only ImportError -> genuine missing dependency
    ServerArguments = None
    PdArgumentParser = None
    strtobool = None
    _IMPORT_ERROR = exc


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddlefleet import requires paddle, unavailable locally: {_IMPORT_ERROR}",
)
class TestServerArgumentsThroughParser(unittest.TestCase):
    """Drive ServerArguments through PdArgumentParser (its real consumer)."""

    def _parse(self, argv):
        # Fresh parser per call so argparse state cannot leak between cases.
        parser = PdArgumentParser(ServerArguments)
        (server_args,) = parser.parse_args_into_dataclasses(
            args=argv, look_for_args_file=False
        )
        self.assertIsInstance(server_args, ServerArguments)
        return server_args

    def test_defaults_flow_through_parser(self):
        """Empty argv: the parser must seed argparse defaults from the field
        defaults, yielding the declared typed values (not their string forms)."""
        args = self._parse([])
        # Ports are ints, not the string forms of the defaults.
        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.port, 8188)
        self.assertIs(type(args.port), int)
        self.assertEqual(args.metrics_port, 8001)
        self.assertEqual(args.engine_worker_queue_port, 8002)
        # None-default optional strings stay None (not the string "None").
        self.assertIsNone(args.quantization)
        self.assertIsNone(args.load_choices)
        self.assertIsNone(args.tool_call_parser)
        # Bool default is a real bool False.
        self.assertIs(args.enable_mm, False)
        # Float default keeps float type and value.
        self.assertIs(type(args.gpu_memory_utilization), float)
        self.assertEqual(args.gpu_memory_utilization, 0.9)

    def test_string_tokens_converted_to_declared_types(self):
        """CLI tokens arrive as strings; the parser must convert them per the
        field type hints. If conversion were skipped, port would be "9090"."""
        args = self._parse(
            [
                "--port",
                "9090",
                "--max_model_len",
                "4096",
                "--max_num_batched_tokens",
                "512",
                "--gpu_memory_utilization",
                "0.95",
                "--kv_cache_ratio",
                "0.5",
            ]
        )
        self.assertEqual(args.port, 9090)
        self.assertIs(type(args.port), int)
        self.assertNotEqual(args.port, "9090")

        self.assertEqual(args.max_model_len, 4096)
        self.assertIs(type(args.max_model_len), int)

        self.assertEqual(args.max_num_batched_tokens, 512)
        self.assertIs(type(args.max_num_batched_tokens), int)

        self.assertEqual(args.gpu_memory_utilization, 0.95)
        self.assertIs(type(args.gpu_memory_utilization), float)

        self.assertEqual(args.kv_cache_ratio, 0.5)
        self.assertIs(type(args.kv_cache_ratio), float)

    def test_distinct_ports_are_not_crosswired(self):
        """Three distinct port values must each land in their own field. Equal
        defaults would hide a swap; distinct values expose mis-wiring."""
        args = self._parse(
            [
                "--port",
                "1111",
                "--metrics_port",
                "2222",
                "--engine_worker_queue_port",
                "3333",
            ]
        )
        self.assertEqual(args.port, 1111)
        self.assertEqual(args.metrics_port, 2222)
        self.assertEqual(args.engine_worker_queue_port, 3333)
        # All three distinct: no field received another's value.
        self.assertEqual(
            len({args.port, args.metrics_port, args.engine_worker_queue_port}),
            3,
        )

    def test_optional_string_fields_override(self):
        """None-default string fields must accept and store the given string."""
        args = self._parse(
            [
                "--quantization",
                "wint4",
                "--load_choices",
                "default_v1",
                "--tool_call_parser",
                "ernie",
                "--reasoning_parser",
                "custom_parser",
            ]
        )
        self.assertEqual(args.quantization, "wint4")
        self.assertEqual(args.load_choices, "default_v1")
        self.assertEqual(args.tool_call_parser, "ernie")
        self.assertEqual(args.reasoning_parser, "custom_parser")

    def test_limit_mm_per_prompt_kept_as_raw_string(self):
        """limit_mm_per_prompt is declared str, so a passed value is kept
        verbatim as text (the field does not parse it into a dict)."""
        raw = "{'image': 10, 'video': 3}"
        args = self._parse(["--limit_mm_per_prompt", raw])
        self.assertEqual(args.limit_mm_per_prompt, raw)
        self.assertIs(type(args.limit_mm_per_prompt), str)

    def test_enable_mm_bool_follows_strtobool_semantics(self):
        """The bool field uses strtobool with nargs='?'/const=True. Each form
        maps to an independently known boolean."""
        # Bare flag -> const True.
        self.assertIs(self._parse(["--enable_mm"]).enable_mm, True)
        # Explicit truthy / falsy tokens routed through strtobool.
        self.assertIs(self._parse(["--enable_mm", "true"]).enable_mm, True)
        self.assertIs(self._parse(["--enable_mm", "yes"]).enable_mm, True)
        self.assertIs(self._parse(["--enable_mm", "1"]).enable_mm, True)
        self.assertIs(self._parse(["--enable_mm", "false"]).enable_mm, False)
        self.assertIs(self._parse(["--enable_mm", "0"]).enable_mm, False)
        self.assertIs(self._parse(["--enable_mm", "no"]).enable_mm, False)

    def test_use_warmup_is_int_flag_not_bool(self):
        """use_warmup is declared int (0/1), so it must round-trip as an int,
        not be coerced through the boolean path."""
        args = self._parse(["--use_warmup", "1"])
        self.assertEqual(args.use_warmup, 1)
        self.assertIs(type(args.use_warmup), int)
        self.assertNotIsInstance(args.use_warmup, bool)


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddlefleet import requires paddle, unavailable locally: {_IMPORT_ERROR}",
)
class TestStrtobool(unittest.TestCase):
    """strtobool is the real converter the parser installs for bool fields."""

    def test_truthy_and_falsy_tokens(self):
        for token in ("yes", "true", "t", "y", "1", "TRUE", "Yes"):
            self.assertIs(strtobool(token), True, token)
        for token in ("no", "false", "f", "n", "0", "FALSE", "No"):
            self.assertIs(strtobool(token), False, token)

    def test_bool_passthrough(self):
        self.assertIs(strtobool(True), True)
        self.assertIs(strtobool(False), False)

    def test_invalid_token_raises(self):
        with self.assertRaises(ArgumentTypeError):
            strtobool("maybe")


if __name__ == "__main__":
    unittest.main()
