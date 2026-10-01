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

import dataclasses
import json
import tempfile
import types
import unittest
from pathlib import Path

from paddlefleet.transformers.configuration_utils import LlmMetaConfig
from paddlefleet.transformers.deepseek_v32.configuration import (
    DeepseekV32Config,
)
from paddlefleet.transformers.glm_moe_dsa.configuration import GlmMoeDsaConfig


class TestDsaConfigContract(unittest.TestCase):
    def test_training_arguments_expose_dsa_layout_fields(self):
        from paddlefleet.trainer import TrainingArguments

        fields = {
            field.name: field for field in dataclasses.fields(TrainingArguments)
        }

        self.assertIsNone(fields["dsa_indexer_topk_freq"].default)
        self.assertIsNone(fields["dsa_indexer_skip_topk_offset"].default)
        self.assertIsNone(fields["dsa_indexer_types"].default)
        self.assertIsNone(fields["dsa_index_share_for_mtp_iteration"].default)

    def test_llm_meta_defaults_and_explicit_values(self):
        defaults = LlmMetaConfig._get_defaults()
        self.assertEqual(defaults["dsa_indexer_topk_freq"], 1)
        self.assertEqual(defaults["dsa_indexer_skip_topk_offset"], 0)
        self.assertIsNone(defaults["dsa_indexer_types"])
        self.assertFalse(defaults["dsa_index_share_for_mtp_iteration"])

        config = types.SimpleNamespace()
        LlmMetaConfig.set_llm_config(
            config,
            types.SimpleNamespace(
                dsa_indexer_topk_freq=4,
                dsa_indexer_skip_topk_offset=1,
                dsa_indexer_types=["full", "shared"],
                dsa_index_share_for_mtp_iteration=True,
            ),
        )
        self.assertEqual(config.dsa_indexer_topk_freq, 4)
        self.assertEqual(config.dsa_indexer_skip_topk_offset, 1)
        self.assertEqual(config.dsa_indexer_types, ["full", "shared"])
        self.assertTrue(config.dsa_index_share_for_mtp_iteration)

    def test_overrides_replace_checkpoint_values_and_survive_save_load(self):
        overrides = {
            "dsa_indexer_topk_freq": 4,
            "dsa_indexer_skip_topk_offset": 1,
            "dsa_indexer_types": ["full", "full", "full", "shared"],
            "dsa_index_share_for_mtp_iteration": True,
        }
        aliases = {
            "dsa_indexer_topk_freq": "index_topk_freq",
            "dsa_indexer_skip_topk_offset": "index_skip_topk_offset",
            "dsa_indexer_types": "indexer_types",
            "dsa_index_share_for_mtp_iteration": "index_share_for_mtp_iteration",
        }
        for config_class in (GlmMoeDsaConfig, DeepseekV32Config):
            for has_checkpoint_layout in (False, True):
                with self.subTest(
                    model=config_class.model_type,
                    has_checkpoint_layout=has_checkpoint_layout,
                ):
                    original = (
                        {
                            "index_topk_freq": 1,
                            "index_skip_topk_offset": 0,
                            "indexer_types": ["full"] * 4,
                            "index_share_for_mtp_iteration": False,
                        }
                        if has_checkpoint_layout
                        else {}
                    )
                    config = config_class(
                        num_hidden_layers=4,
                        num_nextn_predict_layers=1,
                        **original,
                    )
                    LlmMetaConfig.set_llm_config(
                        config,
                        types.SimpleNamespace(
                            num_nextn_predict_layers=1, **overrides
                        ),
                    )
                    # Unspecified CLI values must retain the loaded layout.
                    LlmMetaConfig.set_llm_config(
                        config,
                        types.SimpleNamespace(
                            num_nextn_predict_layers=1,
                            **dict.fromkeys(overrides),
                        ),
                    )
                    with tempfile.TemporaryDirectory() as directory:
                        config.save_pretrained(directory)
                        saved = json.loads(
                            (Path(directory) / "config.json").read_text()
                        )
                        loaded = config_class.from_pretrained(directory)
                    for internal, official in aliases.items():
                        expected = overrides[internal]
                        self.assertEqual(getattr(config, official), expected)
                        self.assertEqual(saved[official], expected)
                        self.assertNotIn(internal, saved)
                        self.assertEqual(getattr(loaded, internal), expected)


if __name__ == "__main__":
    unittest.main()
