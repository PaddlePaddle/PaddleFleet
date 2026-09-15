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

"""Behavior tests for ``paddlefleet.cli.hparams.data_args.DataArguments``.

``DataArguments`` is a plain ``@dataclass`` with field defaults and no
``__post_init__`` (no self-contained normalization or validation). Rather than
assigning a field and reading it straight back (the "配置自赋值后读回"
antipattern), these tests drive the fields through their real consumer: the
production ``PdArgumentParser`` transform entry used by
``paddlefleet.cli.hparams.parser``. That parser is what turns CLI/argv tokens
into a typed ``DataArguments`` instance, so it exercises the declared defaults,
per-field type coercion, boolean parsing via ``strtobool`` and the generated
``--no_<field>`` explicit-off complements. Expected values are hand-derived
from the field declarations in ``data_args.py``.

The whole ``paddlefleet`` package imports ``paddle`` at import time, so tests
skip when Paddle (and therefore the package) is unavailable; they run for real
on any CPU where Paddle is installed.
"""

import unittest

try:
    from paddlefleet.cli.hparams.data_args import DataArguments
    from paddlefleet.trainer.argparser import PdArgumentParser

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle / paddlefleet not installed
    DataArguments = None
    PdArgumentParser = None
    _IMPORT_ERROR = exc


class DataArgumentsParsingTest(unittest.TestCase):
    """Drive DataArguments through the real PdArgumentParser transform."""

    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"paddlefleet/paddle unavailable: {_IMPORT_ERROR!r}")

    def _parse(self, argv):
        """Parse ``argv`` (a list of CLI tokens) into a DataArguments.

        Returns the parsed dataclass and argparse's remaining-strings list so
        callers can assert on both consumed and unrecognized tokens.
        """
        parser = PdArgumentParser((DataArguments,))
        parsed = parser.parse_args_into_dataclasses(
            args=argv,
            return_remaining_strings=True,
            look_for_args_file=False,
        )
        return parsed[0], parsed[-1]

    def test_parser_surfaces_declared_defaults(self):
        """Empty argv yields the defaults declared in data_args.py."""
        args, remaining = self._parse([])
        self.assertEqual(remaining, [])
        self.assertEqual(args.dataset_type, "iterable")
        self.assertIsNone(args.input_dir)
        self.assertEqual(args.split, "950,50")
        self.assertIsNone(args.train_dataset_type)
        self.assertIsNone(args.train_dataset_path)
        self.assertIsNone(args.train_dataset_prob)
        self.assertEqual(args.eval_dataset_type, "erniekit")
        self.assertEqual(args.eval_dataset_path, "examples/data/sft-eval.jsonl")
        # eval_dataset_prob is declared ``str`` -> a numeric-looking default
        # must stay a string, not become a float.
        self.assertIsInstance(args.eval_dataset_prob, str)
        self.assertEqual(args.eval_dataset_prob, "1.0")
        self.assertIsInstance(args.max_seq_len, int)
        self.assertEqual(args.max_seq_len, 4096)
        self.assertEqual(args.max_prompt_len, 2048)
        self.assertEqual(args.buffer_size, 500)
        self.assertEqual(args.mix_strategy, "concat")
        self.assertEqual(args.template_backend, "custom")
        self.assertIsNone(args.template)
        self.assertEqual(args.data_impl, "mmap")
        self.assertEqual(args.truncation_strategy, "delete")
        self.assertEqual(args.dataset_output_dir, "./dataset_output")
        self.assertEqual(args.packing_interval, 1000)
        self.assertEqual(args.num_samples_each_epoch, 6000000)
        self.assertIsNone(args.data_cache)
        self.assertIsNone(args.new_special_tokens_path)
        self.assertIsNone(args.custom_register_path)
        self.assertIsNone(args.packed_idx_cache_dir)
        self.assertIsNone(args.processor_use_fast)

    def test_parser_bool_defaults_are_real_bools(self):
        """Boolean defaults are Python bools, not truthy strings."""
        args, _ = self._parse([])
        # default False
        self.assertIs(args.packing, False)
        self.assertIs(args.padding_free, False)
        self.assertIs(args.split_multi_turn, False)
        self.assertIs(args.make_offline_data, False)
        self.assertIs(args.eval_with_do_generation, False)
        self.assertIs(args.share_folder, False)
        self.assertIs(args.warmup_only_rank0, False)
        # default True
        self.assertIs(args.random_shuffle, True)
        self.assertIs(args.greedy_intokens, True)
        self.assertIs(args.use_template, True)
        self.assertIs(args.encode_one_turn, True)
        self.assertIs(args.skip_warmup, True)
        self.assertIs(args.truncate_packing, True)
        self.assertIs(args.binpacking, True)

    def test_parser_coerces_int_fields_from_strings(self):
        """Int-typed fields are coerced from their CLI string tokens."""
        args, remaining = self._parse(
            [
                "--max_seq_len",
                "8192",
                "--buffer_size",
                "512",
                "--packing_interval",
                "4",
                "--num_samples_each_epoch",
                "42",
            ]
        )
        self.assertEqual(remaining, [])
        self.assertIsInstance(args.max_seq_len, int)
        self.assertNotIsInstance(args.max_seq_len, str)
        self.assertEqual(args.max_seq_len, 8192)
        self.assertEqual(args.buffer_size, 512)
        self.assertEqual(args.packing_interval, 4)
        self.assertEqual(args.num_samples_each_epoch, 42)

    def test_parser_keeps_string_typed_fields_as_str(self):
        """str-typed fields keep numeric-looking values as strings."""
        args, _ = self._parse(
            ["--eval_dataset_prob", "0.8,0.2", "--split", "800,200"]
        )
        self.assertIsInstance(args.eval_dataset_prob, str)
        self.assertEqual(args.eval_dataset_prob, "0.8,0.2")
        self.assertEqual(args.split, "800,200")
        # a bare float-looking token stays a string, never becomes 1.0
        args2, _ = self._parse(["--eval_dataset_prob", "1.0"])
        self.assertIsInstance(args2.eval_dataset_prob, str)
        self.assertEqual(args2.eval_dataset_prob, "1.0")
        self.assertNotEqual(args2.eval_dataset_prob, 1.0)

    def test_parser_coerces_bool_values_true_and_false(self):
        """Boolean fields accept true/false tokens and a bare flag."""
        on, _ = self._parse(["--packing", "True", "--padding_free", "true"])
        self.assertIs(on.packing, True)
        self.assertIs(on.padding_free, True)
        off, _ = self._parse(["--packing", "False", "--padding_free", "0"])
        # bool False, not the string "False"
        self.assertIs(off.packing, False)
        self.assertNotEqual(off.packing, "False")
        self.assertIs(off.padding_free, False)
        # bare flag with no value resolves to True via argparse const
        bare, _ = self._parse(["--packing"])
        self.assertIs(bare.packing, True)

    def test_parser_no_complement_only_for_default_true_bools(self):
        """Default-True bools get a --no_<field> switch; default-False do not."""
        args, remaining = self._parse(
            ["--no_random_shuffle", "--no_skip_warmup", "--no_use_template"]
        )
        self.assertEqual(remaining, [])
        self.assertIs(args.random_shuffle, False)
        self.assertIs(args.skip_warmup, False)
        self.assertIs(args.use_template, False)
        # packing defaults to False, so no --no_packing switch is generated;
        # the token is left unrecognized rather than flipping the field.
        packing_args, packing_remaining = self._parse(["--no_packing"])
        self.assertIn("--no_packing", packing_remaining)
        self.assertIs(packing_args.packing, False)


if __name__ == "__main__":
    unittest.main()
