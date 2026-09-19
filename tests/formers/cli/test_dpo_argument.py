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

"""Behavior tests for ``paddlefleet.cli.train.dpo.dpo_argument``.

Three of the classes (``DPOConfig``, ``DPOModelArgument``, ``DPODataArgument``)
are plain ``@dataclass`` argument holders. For those the meaningful contract is
the set of declared field defaults, so the tests assert the exact default
values hand-derived from the field declarations in ``dpo_argument.py`` and
``data_config.py`` (not shape/type/existence only).

``DPOTrainingArguments`` is the only class carrying real logic: its
``__post_init__`` rewrites a group of training flags when ``autotuner_benchmark``
is set, gates the logging override behind ``not disable_tqdm``, and forces
``num_train_epochs == 1`` whenever ``max_steps > 0``. Those branches are
exercised with distinguishing inputs (including the negative cases that a
missing guard would let through), so removing or weakening a branch fails a
test rather than passing silently.

The whole ``paddlefleet`` package imports ``paddle`` at import time, so every
test skips with an honest reason when Paddle (and therefore the package) is
unavailable; they run for real on any environment where Paddle is installed.
"""

import shutil
import tempfile
import unittest

try:
    from paddlefleet.cli.train.dpo.data_config import DataConfig
    from paddlefleet.cli.train.dpo.dpo_argument import (
        DPOConfig,
        DPODataArgument,
        DPOModelArgument,
        DPOTrainingArguments,
    )
    from paddlefleet.trainer.trainer_utils import IntervalStrategy

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle / package not installed in this env
    DataConfig = None
    DPOConfig = DPODataArgument = DPOModelArgument = DPOTrainingArguments = None
    IntervalStrategy = None
    _IMPORT_ERROR = exc

_SKIP_REASON = (
    "paddlefleet.cli.train.dpo.dpo_argument imports paddle at module load; "
    f"import unavailable here: {_IMPORT_ERROR}"
)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class DPOConfigDefaultsTest(unittest.TestCase):
    """DPOConfig declared field defaults (hand-derived from dpo_argument.py)."""

    def test_default_values(self):
        cfg = DPOConfig()
        self.assertEqual(cfg.beta, 0.1)
        self.assertEqual(cfg.simpo_gamma, 0.5)
        self.assertEqual(cfg.label_smoothing, 0.0)
        self.assertEqual(cfg.loss_type, "sigmoid")
        self.assertEqual(cfg.pref_loss_ratio, 1.0)
        self.assertEqual(cfg.sft_loss_ratio, 0.0)
        self.assertEqual(cfg.dpop_lambda, 50)
        self.assertEqual(cfg.ref_model_update_steps, -1)
        self.assertIs(cfg.reference_free, False)
        self.assertIs(cfg.lora, False)
        self.assertEqual(cfg.offset_alpha, 0.0)
        self.assertIs(cfg.normalize_logps, False)
        self.assertIs(cfg.ignore_eos_token, False)

    def test_constructor_wires_overrides(self):
        # A couple of non-default values to confirm the constructor stores what
        # it is given rather than always returning the defaults above.
        cfg = DPOConfig(beta=0.3, loss_type="ipo", reference_free=True)
        self.assertEqual(cfg.beta, 0.3)
        self.assertEqual(cfg.loss_type, "ipo")
        self.assertIs(cfg.reference_free, True)
        # Untouched fields keep their declared defaults.
        self.assertEqual(cfg.simpo_gamma, 0.5)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class DPOModelArgumentDefaultsTest(unittest.TestCase):
    """DPOModelArgument declared field defaults."""

    def test_default_values(self):
        arg = DPOModelArgument()
        self.assertIsNone(arg.model_name_or_path)
        self.assertIsNone(arg.tokenizer_name_or_path)
        self.assertEqual(arg.download_hub, "aistudio")
        self.assertIs(arg.flash_mask, False)
        self.assertIsNone(arg.weight_quantize_algo)
        self.assertIs(arg.use_attn_mask_startend_row_indices, True)
        self.assertEqual(arg.lora_rank, 8)
        self.assertIsNone(arg.lora_path)
        self.assertIs(arg.rslora, False)
        self.assertEqual(arg.lora_plus_scale, 1.0)
        self.assertEqual(arg.lora_alpha, -1)
        self.assertIs(arg.rslora_plus, False)
        self.assertEqual(arg._attn_implementation, "flashmask")


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class DPODataArgumentDefaultsTest(unittest.TestCase):
    """DPODataArgument defaults, including values inherited from DataConfig."""

    def test_own_field_defaults(self):
        arg = DPODataArgument()
        self.assertEqual(arg.max_seq_len, 4096)
        self.assertEqual(arg.max_prompt_len, 2048)
        self.assertEqual(arg.num_samples_each_epoch, 6000000)
        self.assertEqual(arg.buffer_size, 1000)

    def test_subclasses_data_config_and_inherits_defaults(self):
        self.assertTrue(issubclass(DPODataArgument, DataConfig))
        arg = DPODataArgument()
        # Fields declared on DataConfig, not re-declared on DPODataArgument.
        self.assertEqual(arg.eval_dataset_type, "erniekit")
        self.assertEqual(arg.dataset_type, "iterable")
        self.assertEqual(arg.split, "950,50")
        self.assertEqual(arg.mix_strategy, "concat")
        self.assertEqual(arg.eval_dataset_prob, "1.0")
        self.assertIs(arg.use_template, True)
        self.assertIs(arg.packing, False)
        self.assertIs(arg.greedy_intokens, True)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class DPOTrainingArgumentsBehaviorTest(unittest.TestCase):
    """__post_init__ behavior of DPOTrainingArguments."""

    def setUp(self):
        self.output_dir = tempfile.mkdtemp(prefix="dpo_targs_")
        self.addCleanup(shutil.rmtree, self.output_dir, ignore_errors=True)

    def _build(self, **kwargs):
        kwargs.setdefault("output_dir", self.output_dir)
        return DPOTrainingArguments(**kwargs)

    def test_dpo_specific_field_defaults(self):
        # No autotuner and default max_steps (-1): neither __post_init__ branch
        # fires, so the DPO-specific field defaults are observable as declared.
        args = self._build()
        self.assertEqual(args.num_of_gpus, -1)
        self.assertIs(args.unified_checkpoint, True)
        self.assertEqual(args.unified_checkpoint_config, "")
        self.assertIs(args.autotuner_benchmark, False)
        self.assertIs(args.use_intermediate_api, False)
        self.assertEqual(args.num_hidden_layers, 2)

    def test_autotuner_benchmark_rewrites_training_flags(self):
        args = self._build(autotuner_benchmark=True, disable_tqdm=False)
        # The autotuner branch pins this exact flag set.
        self.assertEqual(args.num_train_epochs, 1)
        self.assertEqual(args.max_steps, 5)
        self.assertIs(args.do_train, True)
        self.assertIs(args.do_export, False)
        self.assertIs(args.do_predict, False)
        self.assertIs(args.do_eval, False)
        self.assertIs(args.overwrite_output_dir, True)
        self.assertIs(args.load_best_model_at_end, False)
        self.assertEqual(args.report_to, [])
        self.assertEqual(args.save_strategy, IntervalStrategy.NO)
        self.assertEqual(args.evaluation_strategy, IntervalStrategy.NO)
        # disable_tqdm is False, so the guarded logging override also fires.
        self.assertEqual(args.logging_steps, 1)
        self.assertEqual(args.logging_strategy, IntervalStrategy.STEPS)

    def test_autotuner_benchmark_respects_disable_tqdm(self):
        # logging_steps=7 makes the default (not 1) distinguishable; the branch
        # `if not self.disable_tqdm` must NOT force logging_steps to 1 here.
        args = self._build(
            autotuner_benchmark=True, disable_tqdm=True, logging_steps=7
        )
        # The unconditional autotuner overrides still apply.
        self.assertEqual(args.max_steps, 5)
        self.assertEqual(args.save_strategy, IntervalStrategy.NO)
        # The tqdm-guarded logging override does not.
        self.assertNotEqual(args.logging_steps, 1)

    def test_max_steps_positive_forces_single_epoch(self):
        # max_steps > 0 overrides num_train_epochs to 1 regardless of the value
        # the user passed.
        args = self._build(
            autotuner_benchmark=False, num_train_epochs=3, max_steps=100
        )
        self.assertEqual(args.num_train_epochs, 1)
        self.assertEqual(args.max_steps, 100)

    def test_num_train_epochs_preserved_when_max_steps_not_set(self):
        # autotuner off and max_steps <= 0: neither branch fires, so the user
        # value survives. Guards the `if self.max_steps > 0` condition against
        # an unconditional override.
        args = self._build(
            autotuner_benchmark=False, num_train_epochs=3, max_steps=-1
        )
        self.assertEqual(args.num_train_epochs, 3)


if __name__ == "__main__":
    unittest.main()
