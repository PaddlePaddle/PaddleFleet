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
"""Official GLM-5.2 HF config.json DSA fields map onto TransformerConfig."""

from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace
from unittest import TestCase

from paddle.distributed.fleet.meta_parallel import zero_bubble_utils

if not hasattr(zero_bubble_utils, "RecomputeStore"):

    class RecomputeStore:  # paddle nightly used by this venv is older than upstream develop
        pass

    zero_bubble_utils.RecomputeStore = RecomputeStore

from paddlefleet.transformer.transformer_config import TransformerConfig


class TestGlm52OfficialDsaHfFields(TestCase):
    _OFFICIAL_TO_INTERNAL = {
        "index_topk_freq": "dsa_indexer_topk_freq",
        "index_skip_topk_offset": "dsa_indexer_skip_topk_offset",
        "indexer_types": "dsa_indexer_types",
        "index_share_for_mtp_iteration": "dsa_index_share_for_mtp_iteration",
    }

    def test_transform_rules_map_official_glm52_dsa_keys(self):
        names = {item.name for item in fields(TransformerConfig)}
        for official, internal in self._OFFICIAL_TO_INTERNAL.items():
            self.assertIn(internal, names)
            self.assertEqual(
                TransformerConfig.transform_rules[official], internal
            )

    def test_register_attributes_copies_official_keys_onto_internal_fields(
        self,
    ):
        cfg = object.__new__(TransformerConfig)
        src = SimpleNamespace(
            index_topk_freq=4,
            index_skip_topk_offset=3,
            indexer_types=["full", "full", "full", "shared"],
            index_share_for_mtp_iteration=True,
        )
        TransformerConfig.register_attributes(cfg, src)
        self.assertEqual(cfg.dsa_indexer_topk_freq, 4)
        self.assertEqual(cfg.dsa_indexer_skip_topk_offset, 3)
        self.assertEqual(
            cfg.dsa_indexer_types, ["full", "full", "full", "shared"]
        )
        self.assertIs(cfg.dsa_index_share_for_mtp_iteration, True)
