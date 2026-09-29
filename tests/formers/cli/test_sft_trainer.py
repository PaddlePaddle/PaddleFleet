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

"""Behavior tests for paddlefleet.cli.train.sft.sft_trainer.SFTTrainer.

These tests exercise real method bodies of SFTTrainer with hand-derived
expected values:
  * log(): perplexity is exp(loss) / exp(eval_loss); expected values come from
    the standard library math.exp (an implementation independent from the
    production code, which uses numpy.exp), and delegation to the parent
    Trainer.log is observed via a spy.
  * _prepare_dataset(): the None guard and the skip_prepare_dataset short
    circuit are verified for content / identity.
  * __init__(): the do_generation validation guards are verified to reject a
    missing gen_args / data_args.

The production module imports paddle at import time. This environment has no
paddle installed, so the import is guarded and, when it fails, the whole
test class is skipped with an honest reason instead of being silently mocked.
"""

import math
import unittest
from unittest.mock import patch

try:
    import paddle  # noqa: F401

    from paddlefleet.cli.train.sft.sft_trainer import SFTTrainer
    from paddlefleet.trainer import Trainer

    _IMPORT_ERROR = None
except ImportError as exc:  # honest: no paddle/paddlefleet runtime available
    SFTTrainer = None
    Trainer = None
    _IMPORT_ERROR = exc

_SKIP_REASON = (
    "paddle / paddlefleet runtime not importable in this environment: "
    f"{_IMPORT_ERROR!r}"
)


def _bare_trainer():
    """Return a real SFTTrainer instance without running the heavy __init__.

    log() and _prepare_dataset() depend only on their own method bodies (and,
    for log(), on the parent Trainer.log reached through super()), not on any
    attribute assigned by __init__, which would otherwise require a real model,
    tokenizer and dataset. Using __new__ keeps the instance a genuine
    SFTTrainer so that super(SFTTrainer, self) resolves to Trainer.
    """
    return SFTTrainer.__new__(SFTTrainer)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class SFTTrainerLogTest(unittest.TestCase):
    def test_log_adds_ppl_from_loss_and_delegates(self):
        captured = {}

        def spy_log(self, logs, **kwargs):
            # Capture a snapshot of what the parent actually received.
            captured["logs"] = dict(logs)
            captured["kwargs"] = kwargs

        trainer = _bare_trainer()
        logs = {"loss": 2.0}
        with patch.object(Trainer, "log", spy_log):
            SFTTrainer.log(trainer, logs, foo="bar")

        expected_ppl = math.exp(2.0)  # independent reference (stdlib, not np)
        self.assertAlmostEqual(logs["ppl"], expected_ppl, places=9)
        self.assertAlmostEqual(logs["ppl"], 7.38905609893065, places=9)
        self.assertNotIn("eval_ppl", logs)
        # Delegation: parent Trainer.log received the mutated dict + kwargs.
        self.assertEqual(captured["logs"]["loss"], 2.0)
        self.assertAlmostEqual(captured["logs"]["ppl"], expected_ppl, places=9)
        self.assertEqual(captured["kwargs"], {"foo": "bar"})

    def test_log_adds_eval_ppl_from_eval_loss(self):
        captured = {}

        def spy_log(self, logs, **kwargs):
            captured["logs"] = dict(logs)

        trainer = _bare_trainer()
        logs = {"eval_loss": 1.5}
        with patch.object(Trainer, "log", spy_log):
            SFTTrainer.log(trainer, logs)

        expected = math.exp(1.5)
        self.assertAlmostEqual(logs["eval_ppl"], expected, places=9)
        self.assertAlmostEqual(logs["eval_ppl"], 4.4816890703380645, places=9)
        self.assertNotIn("ppl", logs)
        self.assertAlmostEqual(captured["logs"]["eval_ppl"], expected, places=9)

    def test_log_adds_both_ppls_independently(self):
        captured = {}

        def spy_log(self, logs, **kwargs):
            captured["logs"] = dict(logs)

        trainer = _bare_trainer()
        logs = {"loss": 0.0, "eval_loss": 1.0}
        with patch.object(Trainer, "log", spy_log):
            SFTTrainer.log(trainer, logs)

        # exp(0) == 1.0 and exp(1) == e; a swap of the two source keys would
        # be caught because the expected values differ.
        self.assertAlmostEqual(logs["ppl"], 1.0, places=12)
        self.assertAlmostEqual(logs["eval_ppl"], math.e, places=9)
        self.assertIn("ppl", captured["logs"])
        self.assertIn("eval_ppl", captured["logs"])

    def test_log_without_loss_keys_adds_nothing_but_delegates(self):
        captured = {}

        def spy_log(self, logs, **kwargs):
            captured["called"] = True
            captured["logs"] = dict(logs)

        trainer = _bare_trainer()
        logs = {"grad_norm": 3.5}
        with patch.object(Trainer, "log", spy_log):
            SFTTrainer.log(trainer, logs)

        # No perplexity key should be injected when neither loss key is present.
        self.assertEqual(logs, {"grad_norm": 3.5})
        self.assertTrue(captured.get("called"))
        self.assertEqual(captured["logs"], {"grad_norm": 3.5})


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class SFTTrainerPrepareDatasetTest(unittest.TestCase):
    def test_none_dataset_raises_value_error(self):
        trainer = _bare_trainer()
        with self.assertRaises(ValueError) as ctx:
            SFTTrainer._prepare_dataset(
                trainer,
                None,
                tokenizer=None,
                dataset_text_field="text",
                max_seq_len=8,
                formatting_func=None,
            )
        # Confirm it is the None guard, not some unrelated ValueError.
        self.assertIn("should not be None", str(ctx.exception))

    def test_skip_prepare_returns_same_object(self):
        trainer = _bare_trainer()
        sentinel = object()
        result = SFTTrainer._prepare_dataset(
            trainer,
            sentinel,
            tokenizer=None,
            dataset_text_field="text",
            max_seq_len=8,
            formatting_func=None,
            skip_prepare_dataset=True,
        )
        # skip_prepare_dataset must short-circuit and hand back the exact object.
        self.assertIs(result, sentinel)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class SFTTrainerDoGenerationGuardTest(unittest.TestCase):
    def test_do_generation_requires_gen_args(self):
        # do_generation=True with gen_args=None must be rejected by the first
        # assertion, before any model/dataset work.
        with self.assertRaises(AssertionError):
            SFTTrainer(do_generation=True, gen_args=None, data_args=object())

    def test_do_generation_requires_data_args(self):
        # gen_args present but data_args=None must be rejected by the second
        # assertion.
        with self.assertRaises(AssertionError):
            SFTTrainer(do_generation=True, gen_args=object(), data_args=None)


if __name__ == "__main__":
    unittest.main()
