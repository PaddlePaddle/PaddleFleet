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
"""Accuracy targets select the clipping recipe and preserve its threshold.

Both sides of an alignment run must declare matching thresholds. The target
must not silently override an explicit user value or the documented default.
"""

import shutil
import tempfile
import unittest

from paddlefleet.trainer import Trainer, TrainingArguments
from formers.trainer.trainer_utils import (
    RegressionModelConfig,
    RegressionPretrainedModel,
)


class TestAccuracyTargetGradClip(unittest.TestCase):
    def setUp(self):
        self.output_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.output_dir, ignore_errors=True)

    def _max_grad_norm(self, accuracy_target, **kwargs):
        config = RegressionModelConfig()
        config.use_accuracy_compatible = accuracy_target
        model = RegressionPretrainedModel(config)
        args = TrainingArguments(
            self.output_dir, report_to=[], bf16=True, **kwargs
        )
        return Trainer(model=model, args=args).args.max_grad_norm

    def test_megatron_target_keeps_default_threshold(self):
        self.assertEqual(self._max_grad_norm("megatron"), 1.0)

    def test_bare_true_keeps_default_threshold(self):
        self.assertEqual(self._max_grad_norm(True), 1.0)

    def test_explicit_threshold_is_preserved(self):
        self.assertEqual(
            self._max_grad_norm("megatron", max_grad_norm=5.0), 5.0
        )

    def test_already_off_is_left_alone(self):
        """``MinimaxV2.5_EP2.yaml`` spells this out; it must not error."""
        self.assertEqual(
            self._max_grad_norm("megatron", max_grad_norm=0.0), 0.0
        )

    def test_hf_target_keeps_clipping(self):
        """torch's reference clips, so zeroing this would drop the aligned step."""
        self.assertEqual(self._max_grad_norm("hf"), 1.0)
        self.assertEqual(self._max_grad_norm("hf", max_grad_norm=5.0), 5.0)

    def test_default_target_keeps_clipping(self):
        """A run that targets nothing keeps the user's threshold."""
        self.assertEqual(self._max_grad_norm(False), 1.0)

    def test_untouched_config_keeps_clipping(self):
        """``PretrainedConfig`` defaults the field to ``False``, i.e. off."""
        config = RegressionModelConfig()
        self.assertIs(config.use_accuracy_compatible, False)
        model = RegressionPretrainedModel(config)
        args = TrainingArguments(self.output_dir, report_to=[], bf16=True)
        self.assertEqual(
            Trainer(model=model, args=args).args.max_grad_norm, 1.0
        )


if __name__ == "__main__":
    unittest.main()
