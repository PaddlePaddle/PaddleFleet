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

"""Behavior tests for ``paddlefleet.cli.train.dpo.data_config.DataConfig``.

``DataConfig`` is a plain ``@dataclass`` of field defaults with no
``__post_init__`` (no self-contained normalization or validation). Assigning a
field and reading it straight back would only exercise dataclass machinery (the
"配置自赋值后读回" antipattern), so these tests instead drive the config through
its real consumer: the production ``PdArgumentParser`` transform. That parser is
what turns CLI/argv tokens into a typed ``DataConfig`` instance, exercising the
declared defaults, per-field type coercion, boolean parsing via ``strtobool``
and the generated ``--no_<field>`` explicit-off complements. Every expected
value is hand-derived from the field declarations in ``data_config.py``.

The whole ``paddlefleet`` package imports ``paddle`` at import time, so tests
skip (with an honest reason) when Paddle / paddlefleet is unavailable; they run
for real on any CPU where Paddle is installed.
"""

import unittest

try:
    from paddlefleet.cli.train.dpo.data_config import DataConfig
    from paddlefleet.trainer.argparser import PdArgumentParser

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle / paddlefleet not installed
    DataConfig = None
    PdArgumentParser = None
    _IMPORT_ERROR = exc


class DpoDataConfigParsingTest(unittest.TestCase):
    """Drive DataConfig through the real PdArgumentParser transform."""

    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"paddlefleet/paddle unavailable: {_IMPORT_ERROR!r}")

    def _parse(self, argv):
        """Parse ``argv`` (CLI tokens) into a DataConfig via the real parser.

        Returns the parsed dataclass and argparse's remaining-strings list so
        callers can assert on both consumed and unrecognized tokens.
        """
        parser = PdArgumentParser((DataConfig,))
        parsed = parser.parse_args_into_dataclasses(
            args=argv,
            return_remaining_strings=True,
            look_for_args_file=False,
        )
        return parsed[0], parsed[-1]

    def test_parser_surfaces_declared_defaults(self):
        """Empty argv yields the exact defaults declared in data_config.py."""
        args, remaining = self._parse([])
        self.assertEqual(remaining, [])
        # None-defaulted fields must stay None, not become the string "None".
        self.assertIsNone(args.dataset_name_or_path)
        self.assertIsNone(args.train_dataset_type)
        self.assertIsNone(args.train_dataset_path)
        self.assertIsNone(args.train_dataset_prob)
        self.assertIsNone(args.input_dir)
        self.assertIsNone(args.task_name)
        self.assertIsNone(args.pad_to_multiple_of)
        self.assertIsNone(args.chat_template)
        # Explicit string defaults.
        self.assertEqual(args.eval_dataset_type, "erniekit")
        self.assertEqual(args.eval_dataset_path, "examples/data/sft-eval.jsonl")
        self.assertEqual(args.dataset_type, "iterable")
        self.assertEqual(args.split, "950,50")
        self.assertEqual(args.mix_strategy, "concat")
        # eval_dataset_prob is declared ``str`` -> a numeric-looking default
        # must remain a string, never float 1.0.
        self.assertIsInstance(args.eval_dataset_prob, str)
        self.assertEqual(args.eval_dataset_prob, "1.0")
        self.assertNotEqual(args.eval_dataset_prob, 1.0)
        # int default kept as int.
        self.assertIsInstance(args.num_samples_each_epoch, int)
        self.assertEqual(args.num_samples_each_epoch, 6000000)

    def test_parser_bool_defaults_are_real_bools(self):
        """Boolean defaults resolve to Python bools with the declared value."""
        args, _ = self._parse([])
        # Declared default True.
        self.assertIs(args.use_template, True)
        self.assertIs(args.encode_one_turn, True)
        self.assertIs(args.greedy_intokens, True)
        self.assertIs(args.random_shuffle, True)
        # Declared default False.
        self.assertIs(args.packing, False)
        self.assertIs(args.eval_with_do_generation, False)
        self.assertIs(args.save_generation_output, False)
        self.assertIs(args.lazy, False)
        self.assertIs(args.pad_to_max_length, False)
        self.assertIs(args.autoregressive, False)
        self.assertIs(args.use_pose_convert, False)

    def test_parser_coerces_int_fields_from_strings(self):
        """Int-typed fields are coerced from their CLI string tokens."""
        args, remaining = self._parse(
            [
                "--num_samples_each_epoch",
                "128",
                "--pad_to_multiple_of",
                "16",
            ]
        )
        self.assertEqual(remaining, [])
        self.assertIsInstance(args.num_samples_each_epoch, int)
        self.assertNotIsInstance(args.num_samples_each_epoch, str)
        self.assertEqual(args.num_samples_each_epoch, 128)
        # pad_to_multiple_of defaults to None but is int-typed: a provided
        # token becomes an int, not the string "16".
        self.assertIsInstance(args.pad_to_multiple_of, int)
        self.assertEqual(args.pad_to_multiple_of, 16)

    def test_parser_keeps_string_typed_fields_as_str(self):
        """str-typed fields keep numeric/comma tokens verbatim as strings."""
        args, remaining = self._parse(
            [
                "--train_dataset_path",
                "./sft-1.jsonl,./sft-2.jsonl",
                "--train_dataset_prob",
                "0.8,0.2",
                "--train_dataset_type",
                "erniekit,erniekit",
                "--eval_dataset_prob",
                "0.5",
                "--split",
                "800,200",
            ]
        )
        self.assertEqual(remaining, [])
        # Multi-source comma-joined values survive intact (no splitting).
        self.assertEqual(args.train_dataset_path, "./sft-1.jsonl,./sft-2.jsonl")
        self.assertEqual(args.train_dataset_prob, "0.8,0.2")
        self.assertEqual(args.train_dataset_type, "erniekit,erniekit")
        self.assertEqual(args.split, "800,200")
        # A bare float-looking token stays a string, never becomes 0.5.
        self.assertIsInstance(args.eval_dataset_prob, str)
        self.assertEqual(args.eval_dataset_prob, "0.5")
        self.assertNotEqual(args.eval_dataset_prob, 0.5)

    def test_parser_coerces_bool_values_true_and_false(self):
        """Boolean fields accept true/false tokens and a bare flag."""
        on, _ = self._parse(["--packing", "True", "--lazy", "true"])
        self.assertIs(on.packing, True)
        self.assertIs(on.lazy, True)
        off, _ = self._parse(["--packing", "False", "--lazy", "0"])
        # bool False, not the string "False".
        self.assertIs(off.packing, False)
        self.assertNotEqual(off.packing, "False")
        self.assertIs(off.lazy, False)
        # bare flag with no value resolves to True via the argparse const.
        bare, _ = self._parse(["--packing"])
        self.assertIs(bare.packing, True)

    def test_parser_no_complement_only_for_default_true_bools(self):
        """Default-True bools get a --no_<field> switch; default-False do not."""
        args, remaining = self._parse(
            [
                "--no_use_template",
                "--no_encode_one_turn",
                "--no_greedy_intokens",
                "--no_random_shuffle",
            ]
        )
        self.assertEqual(remaining, [])
        self.assertIs(args.use_template, False)
        self.assertIs(args.encode_one_turn, False)
        self.assertIs(args.greedy_intokens, False)
        self.assertIs(args.random_shuffle, False)
        # packing defaults to False, so no --no_packing switch is generated;
        # the token is left unrecognized rather than flipping the field.
        packing_args, packing_remaining = self._parse(["--no_packing"])
        self.assertIn("--no_packing", packing_remaining)
        self.assertIs(packing_args.packing, False)


if __name__ == "__main__":
    unittest.main()
