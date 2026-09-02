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

    def test_defaults_keep_always_on_indexer(self):
        config = TransformerConfig(num_hidden_layers=4)
        self.assertEqual(config.dsa_indexer_topk_freq, 1)
        self.assertEqual(config.dsa_indexer_skip_topk_offset, 0)
        self.assertIsNone(config.dsa_indexer_types)
        self.assertIs(config.dsa_index_share_for_mtp_iteration, False)

    def test_official_nondefault_values_are_accepted(self):
        config = TransformerConfig(
            num_hidden_layers=4,
            num_nextn_predict_layers=1,
            dsa_indexer_topk_freq=4,
            dsa_indexer_skip_topk_offset=3,
            dsa_indexer_types=["full", "full", "full", "shared"],
            dsa_index_share_for_mtp_iteration=True,
        )
        self.assertEqual(config.dsa_indexer_topk_freq, 4)
        self.assertEqual(config.dsa_indexer_skip_topk_offset, 3)
        self.assertEqual(
            config.dsa_indexer_types, ["full", "full", "full", "shared"]
        )
        self.assertIs(config.dsa_index_share_for_mtp_iteration, True)

    def test_invalid_dsa_indexer_fields_raise(self):
        with self.assertRaisesRegex(
            ValueError, "dsa_indexer_topk_freq must be >= 1"
        ):
            TransformerConfig(dsa_indexer_topk_freq=0)
        with self.assertRaisesRegex(
            ValueError, "dsa_indexer_skip_topk_offset must be >= 0"
        ):
            TransformerConfig(dsa_indexer_skip_topk_offset=-1)
        with self.assertRaisesRegex(ValueError, "dsa_indexer_types length"):
            TransformerConfig(num_hidden_layers=2, dsa_indexer_types=["full"])
        with self.assertRaisesRegex(ValueError, "must be 'full' or 'shared'"):
            TransformerConfig(
                num_hidden_layers=1, dsa_indexer_types=["unknown"]
            )
        with self.assertRaisesRegex(
            ValueError, "dsa_index_share_for_mtp_iteration=True requires"
        ):
            TransformerConfig(
                num_hidden_layers=2,
                num_nextn_predict_layers=0,
                dsa_index_share_for_mtp_iteration=True,
            )
        with self.assertRaisesRegex(ValueError, r"dsa_indexer_types\[0\]"):
            TransformerConfig(
                num_hidden_layers=2, dsa_indexer_types=["shared", "full"]
            )

    def test_shared_layer_skips_indexer_construction(self):
        from paddlefleet.transformer.dsa_attention import (
            resolve_dsa_indexer_layout,
        )

        config = TransformerConfig(
            hidden_size=64,
            num_attention_heads=2,
            num_hidden_layers=4,
            dsa_indexer_types=["full", "full", "full", "shared"],
        )
        _, skip_topk, index_share, source_layer = resolve_dsa_indexer_layout(
            config, 3
        )
        self.assertTrue(skip_topk)
        self.assertTrue(index_share)
        self.assertEqual(source_layer, 2)

    def test_first_decoder_from_official_types_owns_indexer(self):
        from paddlefleet.transformer.dsa_attention import (
            resolve_dsa_indexer_layout,
        )

        config = TransformerConfig(
            hidden_size=64,
            num_attention_heads=2,
            num_hidden_layers=4,
            dsa_indexer_types=["full", "full", "full", "shared"],
        )
        indexer_type, skip_topk, index_share, source_layer = (
            resolve_dsa_indexer_layout(config, 0)
        )
        self.assertEqual(indexer_type, "full")
        self.assertFalse(skip_topk)
        # Later shared layers reuse this layer, so it must publish top-k.
        self.assertTrue(index_share)
        self.assertEqual(source_layer, 0)

    def test_defaults_do_not_skip_indexer(self):
        from paddlefleet.transformer.dsa_attention import (
            resolve_dsa_indexer_layout,
        )

        config = TransformerConfig(
            hidden_size=64, num_attention_heads=2, num_hidden_layers=4
        )
        indexer_type, skip_topk, index_share, source_layer = (
            resolve_dsa_indexer_layout(config, 3)
        )
        self.assertEqual(indexer_type, "full")
        self.assertFalse(skip_topk)
        self.assertFalse(index_share)
        self.assertEqual(source_layer, 3)

    def test_mtp_layer_reuses_last_decoder_when_share_flag_is_set(self):
        from paddlefleet.transformer.dsa_attention import (
            resolve_dsa_indexer_layout,
        )

        config = TransformerConfig(
            hidden_size=64,
            num_attention_heads=2,
            num_hidden_layers=4,
            num_nextn_predict_layers=1,
            dsa_indexer_types=["full", "full", "full", "shared"],
            dsa_index_share_for_mtp_iteration=True,
        )
        indexer_type, skip_topk, index_share, source_layer = (
            resolve_dsa_indexer_layout(config, 0, is_mtp_layer=True)
        )
        self.assertEqual(indexer_type, "shared")
        self.assertTrue(skip_topk)
        self.assertTrue(index_share)
        self.assertEqual(source_layer, 3)

    def test_last_decoder_publishes_topk_when_mtp_share_is_set(self):
        from paddlefleet.transformer.dsa_attention import (
            resolve_dsa_indexer_layout,
        )

        config = TransformerConfig(
            hidden_size=64,
            num_attention_heads=2,
            num_hidden_layers=4,
            num_nextn_predict_layers=1,
            dsa_indexer_types=["full", "full", "full", "shared"],
            dsa_index_share_for_mtp_iteration=True,
        )
        indexer_type, skip_topk, index_share, source_layer = (
            resolve_dsa_indexer_layout(config, 2)
        )
        self.assertEqual(indexer_type, "full")
        self.assertFalse(skip_topk)
        self.assertTrue(index_share)
        self.assertEqual(source_layer, 2)
