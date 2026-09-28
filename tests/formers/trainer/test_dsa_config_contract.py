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
import types
import unittest

from paddlefleet.trainer import TrainingArguments
from paddlefleet.transformers.configuration_utils import LlmMetaConfig


class TestDsaConfigContract(unittest.TestCase):
    def test_training_arguments_expose_dsa_layout_fields(self):
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


if __name__ == "__main__":
    unittest.main()
