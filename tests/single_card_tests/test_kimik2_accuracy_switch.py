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
"""``use_kimik2_accuracy`` and its single runtime read point.

The switch must default to off so the other alignment targets (GLM45Air,
MinimaxV2.5, DSV4) keep their numerics, and enabling it must reach
``paddlefleet.utils.use_kimik2_accuracy_compatible``, which the CE / MoE / wgrad
paths read. The published state is process-global, so every case restores it.
"""

import types
import unittest

from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.transformers.configuration_utils import LlmMetaConfig
from paddlefleet.utils import (
    set_kimik2_accuracy_compatible,
    use_kimik2_accuracy_compatible,
)


class TestKimiK2AccuracySwitch(unittest.TestCase):
    def setUp(self):
        previous = use_kimik2_accuracy_compatible()
        self.addCleanup(set_kimik2_accuracy_compatible, previous)
        set_kimik2_accuracy_compatible(False)

    def test_defaults_to_off(self):
        config = TransformerConfig()
        self.assertFalse(config.use_kimik2_accuracy)
        self.assertFalse(use_kimik2_accuracy_compatible())

    def test_enabling_publishes_runtime_switch(self):
        config = TransformerConfig(use_kimik2_accuracy=True)
        self.assertTrue(config.use_kimik2_accuracy)
        self.assertTrue(use_kimik2_accuracy_compatible())

    def test_checkpoint_config_key_publishes_before_layers_are_built(self):
        """A ``config.json`` carrying the switch arrives through the attribute
        copy at ``AutoConfig.from_pretrained`` time, i.e. after
        ``__post_init__``. ``LanguageLoss.__init__`` reads the runtime switch
        while building layers, so that path has to publish too."""
        config = TransformerConfig()
        self.assertFalse(use_kimik2_accuracy_compatible())
        config._process_attribute("use_kimik2_accuracy", True)
        self.assertTrue(config.use_kimik2_accuracy)
        self.assertTrue(use_kimik2_accuracy_compatible())

    def test_training_args_enable_it(self):
        """YAML / CLI reach the config through ``set_llm_config`` after
        ``__post_init__``, so the funnel has to publish too."""
        config = types.SimpleNamespace()
        LlmMetaConfig.set_llm_config(
            config, types.SimpleNamespace(use_kimik2_accuracy=True)
        )
        self.assertIs(config.use_kimik2_accuracy, True)
        self.assertTrue(use_kimik2_accuracy_compatible())

    def test_training_args_default_is_off(self):
        config = types.SimpleNamespace()
        LlmMetaConfig.set_llm_config(config, types.SimpleNamespace())
        self.assertIs(config.use_kimik2_accuracy, False)
        self.assertFalse(use_kimik2_accuracy_compatible())

    def test_unset_args_keep_the_checkpoint_value(self):
        config = types.SimpleNamespace(use_kimik2_accuracy=True)
        LlmMetaConfig.set_llm_config(config, types.SimpleNamespace())
        self.assertIs(config.use_kimik2_accuracy, True)


if __name__ == "__main__":
    unittest.main()
