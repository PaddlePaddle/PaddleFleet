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

"""Behavior tests for ``paddlefleet.cli.hparams.preprocess_args``.

The preprocess-argument dataclasses carry real ``__post_init__`` normalization
and validation logic: ``UtteranceProcessorArguments`` back-fills ``tokenizer``
from ``tokenizer_name``, ``CoarseProcessorArguments`` lower-cases and validates
``video_frames_sample``, and ``InputIdsMassageArguments`` parses the paired
``adaptive_max_imgtoken_*`` CSV strings into typed lists. ``End2EndProcessor
Arguments`` inherits all of these through a cooperative ``super().__post_init__()``
chain, so a single construction must trigger every parent's transform.

Rather than assigning a field and reading the same field straight back (the
"配置自赋值后读回" antipattern), these tests observe transformations where the
output differs from the input, and drive the dataclasses through their real
consumer -- the production ``PdArgumentParser`` -- which turns raw CLI/argv
tokens into typed instances and runs ``__post_init__`` on the result. Expected
values are hand-derived from the field declarations and the ``__post_init__``
bodies in ``preprocess_args.py``; none are produced by calling the code under
test.

The whole ``paddlefleet`` package imports ``paddle`` at import time, so the
tests skip when Paddle (and therefore the package) is unavailable; they run for
real on any CPU where Paddle is installed.
"""

import unittest

try:
    from paddlefleet.cli.hparams.preprocess_args import (
        CoarseProcessorArguments,
        End2EndProcessorArguments,
        End2EndProcessorArgumentsHelper,
        ImageModificationProcessorArguments,
        InputIdsMassageArguments,
        UtteranceProcessorArguments,
    )
    from paddlefleet.trainer.argparser import PdArgumentParser

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle / paddlefleet not installed
    CoarseProcessorArguments = None
    End2EndProcessorArguments = None
    End2EndProcessorArgumentsHelper = None
    ImageModificationProcessorArguments = None
    InputIdsMassageArguments = None
    UtteranceProcessorArguments = None
    PdArgumentParser = None
    _IMPORT_ERROR = exc


class PreprocessArgumentsPostInitTest(unittest.TestCase):
    """Exercise the ``__post_init__`` normalization/validation directly.

    Each case feeds an input whose normalized form is distinguishable from the
    raw input, so a missing or broken transform changes the observed value.
    """

    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"paddlefleet/paddle unavailable: {_IMPORT_ERROR!r}")

    def test_tokenizer_name_backfills_tokenizer(self):
        """tokenizer_name is copied into tokenizer only when tokenizer is unset."""
        args = UtteranceProcessorArguments(tokenizer_name="/models/tok-A")
        # tokenizer was declared None; __post_init__ derived it from a *different*
        # field, so this is a real transform rather than a self read-back.
        self.assertEqual(args.tokenizer, "/models/tok-A")
        self.assertEqual(args.tokenizer_name, "/models/tok-A")

    def test_explicit_tokenizer_wins_over_name(self):
        """An explicit tokenizer must not be clobbered by tokenizer_name."""
        args = UtteranceProcessorArguments(
            tokenizer="/models/explicit",
            tokenizer_name="/models/from-name",
        )
        self.assertEqual(args.tokenizer, "/models/explicit")
        self.assertEqual(args.tokenizer_name, "/models/from-name")

    def test_video_frames_sample_is_lowercased(self):
        """A mixed-case accepted value is normalized to lower case."""
        args = CoarseProcessorArguments(video_frames_sample="LEADING")
        self.assertEqual(args.video_frames_sample, "leading")

    def test_video_frames_sample_rejects_unknown_value(self):
        """An out-of-set value fails validation with AssertionError."""
        with self.assertRaises(AssertionError):
            CoarseProcessorArguments(video_frames_sample="diagonal")

    def test_adaptive_imgtoken_pair_parsed_to_typed_lists(self):
        """Paired CSV strings become int / float lists; whitespace is trimmed."""
        args = InputIdsMassageArguments(
            adaptive_max_imgtoken_option="  1,2,3 ",
            adaptive_max_imgtoken_rate="0.1,0.2,0.3",
        )
        self.assertEqual(args.adaptive_max_imgtoken_option, [1, 2, 3])
        self.assertEqual(args.adaptive_max_imgtoken_rate, [0.1, 0.2, 0.3])
        # element types matter: options are ints, rates are floats.
        self.assertTrue(
            all(isinstance(v, int) for v in args.adaptive_max_imgtoken_option)
        )
        self.assertTrue(
            all(isinstance(v, float) for v in args.adaptive_max_imgtoken_rate)
        )

    def test_adaptive_imgtoken_requires_both_to_parse(self):
        """With only one of the pair set, neither is parsed (the ``and`` guard)."""
        args = InputIdsMassageArguments(adaptive_max_imgtoken_rate="0.1,0.2")
        # rate stays the raw CSV string; option stays None -> no list conversion.
        self.assertEqual(args.adaptive_max_imgtoken_rate, "0.1,0.2")
        self.assertIsNone(args.adaptive_max_imgtoken_option)

    def test_end2end_runs_every_parent_transform(self):
        """A single End2End construction must fire the whole super() chain.

        If any subclass dropped its ``super().__post_init__()`` call, one of
        these downstream transforms would silently not run, so asserting all of
        them together pins the cooperative MRO wiring.
        """
        args = End2EndProcessorArguments(
            tokenizer_name="/models/e2e-tok",
            video_frames_sample="RAND",
            adaptive_max_imgtoken_option="4,5",
            adaptive_max_imgtoken_rate="0.5,0.5",
        )
        self.assertEqual(
            args.tokenizer, "/models/e2e-tok"
        )  # Utterance transform
        self.assertEqual(args.video_frames_sample, "rand")  # Coarse transform
        self.assertEqual(  # InputIdsMassage transform
            args.adaptive_max_imgtoken_option, [4, 5]
        )
        self.assertEqual(args.adaptive_max_imgtoken_rate, [0.5, 0.5])


class PreprocessArgumentsParsingTest(unittest.TestCase):
    """Drive the dataclasses through the real PdArgumentParser consumer."""

    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"paddlefleet/paddle unavailable: {_IMPORT_ERROR!r}")

    def _parse(self, dtype, argv):
        """Parse ``argv`` CLI tokens into ``dtype`` and return (instance, remaining)."""
        parser = PdArgumentParser((dtype,))
        parsed = parser.parse_args_into_dataclasses(
            args=argv,
            return_remaining_strings=True,
            look_for_args_file=False,
        )
        return parsed[0], parsed[-1]

    def test_defaults_surface_through_parser(self):
        """Empty argv yields the declared defaults after __post_init__."""
        args, remaining = self._parse(End2EndProcessorArguments, [])
        self.assertEqual(remaining, [])
        # Utterance
        self.assertIsNone(args.tokenizer)
        self.assertIsNone(args.tokenizer_name)
        # Coarse (ints stay ints; default sample already lower-case)
        self.assertEqual(args.video_fps, 2)
        self.assertEqual(args.video_min_frames, 16)
        self.assertEqual(args.video_max_frames, 480)
        self.assertEqual(args.video_target_frames, -1)
        self.assertEqual(args.video_frames_sample, "middle")
        # InputIdsMassage
        self.assertEqual(args.im_prefix_length, 64)
        self.assertTrue(args.use_pic_id)
        self.assertEqual(args.prompt_dir, "./")
        self.assertEqual(args.spatial_conv_size, 2)
        self.assertEqual(args.chat_template, "ernie_vl")
        self.assertIsNone(args.adaptive_max_imgtoken_option)
        self.assertIsNone(args.adaptive_max_imgtoken_rate)
        # ImageModification
        self.assertEqual(args.image_token_len, 64)
        self.assertEqual(args.image_dtype, "uint8")
        self.assertFalse(args.render_timestamp)
        # Helper
        self.assertFalse(args.load_args_from_api)

    def test_parser_coerces_types_and_runs_post_init(self):
        """CLI string tokens are typed by argparse then normalized by __post_init__."""
        args, remaining = self._parse(
            End2EndProcessorArguments,
            [
                "--tokenizer_name",
                "/cli/tok",
                "--video_fps",
                "7",
                "--video_frames_sample",
                "MIDDLE",
                "--image_token_len",
                "128",
                "--adaptive_max_imgtoken_option",
                "1,2,3",
                "--adaptive_max_imgtoken_rate",
                "0.1,0.2,0.3",
            ],
        )
        self.assertEqual(remaining, [])
        # tokenizer back-filled from the parsed tokenizer_name.
        self.assertEqual(args.tokenizer, "/cli/tok")
        # "7" arrives as a string on argv but is coerced to int by the parser.
        self.assertIsInstance(args.video_fps, int)
        self.assertEqual(args.video_fps, 7)
        self.assertIsInstance(args.image_token_len, int)
        self.assertEqual(args.image_token_len, 128)
        # "MIDDLE" survives argparse as-is then is lower-cased by __post_init__.
        self.assertEqual(args.video_frames_sample, "middle")
        # str-typed CSV fields are parsed into typed lists by __post_init__.
        self.assertEqual(args.adaptive_max_imgtoken_option, [1, 2, 3])
        self.assertEqual(args.adaptive_max_imgtoken_rate, [0.1, 0.2, 0.3])

    def test_parser_generates_no_complement_for_true_bool(self):
        """--no_use_pic_id (generated for the True-default bool) turns it off."""
        args, remaining = self._parse(
            End2EndProcessorArguments, ["--no_use_pic_id"]
        )
        self.assertEqual(remaining, [])
        self.assertFalse(args.use_pic_id)
        # A separate parse without the flag keeps the declared True default.
        default_args, _ = self._parse(End2EndProcessorArguments, [])
        self.assertTrue(default_args.use_pic_id)

    def test_parser_propagates_validation_failure(self):
        """An invalid video_frames_sample must raise from __post_init__, not pass."""
        with self.assertRaises(AssertionError):
            self._parse(
                End2EndProcessorArguments,
                ["--video_frames_sample", "invalid"],
            )


if __name__ == "__main__":
    unittest.main()
