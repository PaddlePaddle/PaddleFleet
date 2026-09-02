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
        with self.assertRaisesRegex(
            ValueError, "dsa_indexer_topk_freq must be a positive int"
        ):
            TransformerConfig(dsa_indexer_topk_freq=True)
        with self.assertRaisesRegex(
            ValueError,
            "dsa_indexer_skip_topk_offset must be a non-negative int",
        ):
            TransformerConfig(dsa_indexer_skip_topk_offset=1.5)
        with self.assertRaisesRegex(
            ValueError, "dsa_indexer_types must be None or a list of strings"
        ):
            TransformerConfig(num_hidden_layers=1, dsa_indexer_types="full")

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
        # Later shared layers reuse the last preceding full layer (2), not
        # layer 0, so this producer does not publish a holder key.
        self.assertFalse(index_share)
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
        # Last decoder is shared, so MTP must read that layer's producer (2),
        # not the last decoder's own index (3).
        self.assertEqual(source_layer, 2)

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

    def test_periodic_skip_helpers_and_freq_layout(self):
        from paddlefleet.transformer.dsa_attention import (
            is_dsa_skip_topk_layer,
            resolve_dsa_indexer_layout,
            source_dsa_compute_layer,
        )

        self.assertFalse(
            is_dsa_skip_topk_layer(1, skip_topk_offset=0, topk_freq=4)
        )
        self.assertTrue(
            is_dsa_skip_topk_layer(2, skip_topk_offset=0, topk_freq=4)
        )
        self.assertEqual(
            source_dsa_compute_layer(2, skip_topk_offset=0, topk_freq=4), 1
        )
        with self.assertRaisesRegex(ValueError, "1-indexed"):
            is_dsa_skip_topk_layer(0, 0, 4)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            is_dsa_skip_topk_layer(1, -1, 4)
        with self.assertRaisesRegex(ValueError, "positive"):
            is_dsa_skip_topk_layer(1, 0, 0)

        config = TransformerConfig(
            hidden_size=64,
            num_attention_heads=2,
            num_hidden_layers=4,
            dsa_indexer_topk_freq=4,
            dsa_indexer_skip_topk_offset=0,
        )
        layouts = [
            resolve_dsa_indexer_layout(config, layer) for layer in range(4)
        ]
        self.assertEqual(
            [(item[1], item[2], item[3]) for item in layouts],
            [
                (False, True, 0),
                (True, True, 0),
                (True, True, 0),
                (True, True, 0),
            ],
        )

    def test_unsupported_indexer_type_and_missing_full_source_raise(self):
        from paddlefleet.transformer.dsa_attention import (
            resolve_dsa_indexer_layout,
        )

        config = TransformerConfig(
            hidden_size=64, num_attention_heads=2, num_hidden_layers=2
        )
        config.dsa_indexer_types = ["full", "sparse"]
        with self.assertRaisesRegex(ValueError, "Unsupported DSA indexer type"):
            resolve_dsa_indexer_layout(config, 1)

        config.dsa_indexer_types = ["shared", "shared"]
        with self.assertRaisesRegex(ValueError, "no preceding full indexer"):
            resolve_dsa_indexer_layout(config, 1)

    def test_producer_and_consumer_holder_keys_match_for_legal_share_layouts(
        self,
    ):
        from paddlefleet.transformer.dsa_attention import (
            resolve_dsa_indexer_layout,
        )

        periodic = TransformerConfig(
            hidden_size=64,
            num_attention_heads=2,
            num_hidden_layers=4,
            num_nextn_predict_layers=1,
            dsa_indexer_topk_freq=4,
            dsa_index_share_for_mtp_iteration=True,
        )
        published = {
            layer: resolve_dsa_indexer_layout(periodic, layer)
            for layer in range(4)
        }
        mtp = resolve_dsa_indexer_layout(periodic, 0, is_mtp_layer=True)
        self.assertFalse(published[0][1])
        self.assertTrue(published[0][2])
        self.assertEqual(published[0][3], 0)
        self.assertEqual(published[1][3], 0)
        self.assertTrue(mtp[1])
        self.assertEqual(mtp[3], 0)

        official = TransformerConfig(
            hidden_size=64,
            num_attention_heads=2,
            num_hidden_layers=4,
            num_nextn_predict_layers=1,
            dsa_indexer_types=["full", "full", "full", "shared"],
            dsa_index_share_for_mtp_iteration=True,
        )
        last_full = resolve_dsa_indexer_layout(official, 2)
        last_decoder = resolve_dsa_indexer_layout(official, 3)
        official_mtp = resolve_dsa_indexer_layout(
            official, 0, is_mtp_layer=True
        )
        self.assertFalse(last_full[1])
        self.assertTrue(last_full[2])
        self.assertEqual(last_full[3], 2)
        self.assertTrue(last_decoder[1])
        self.assertEqual(last_decoder[3], 2)
        self.assertTrue(official_mtp[1])
        self.assertEqual(official_mtp[3], 2)

    def test_head_empty_offset_uses_logical_decoder_index(self):
        from paddlefleet.transformer.dsa_attention import (
            resolve_dsa_indexer_layout,
        )

        config = TransformerConfig(
            hidden_size=64,
            num_attention_heads=2,
            num_hidden_layers=4,
            num_empty_layers_add_in_head=2,
            dsa_indexer_types=["full", "shared", "full", "shared"],
        )
        first = resolve_dsa_indexer_layout(config, 2)
        second = resolve_dsa_indexer_layout(config, 3)
        self.assertEqual(first[0], "full")
        self.assertFalse(first[1])
        self.assertTrue(first[2])
        self.assertEqual(first[3], 0)
        self.assertEqual(second[0], "shared")
        self.assertTrue(second[1])
        self.assertEqual(second[3], 0)
        with self.assertRaises(IndexError):
            resolve_dsa_indexer_layout(config, 0)

    def test_index_share_holder_is_created_on_config_when_mask_is_none(self):
        from paddlefleet.transformer.dsa_attention import DSAttention

        attn = DSAttention.__new__(DSAttention)
        attn.config = TransformerConfig(
            hidden_size=64, num_attention_heads=2, num_hidden_layers=4
        )
        holder = attn._get_index_share_topk_holder(None)
        self.assertEqual(holder, {})
        self.assertIs(getattr(attn.config, DSAttention._HOLDER_ATTR), holder)
        holder[2] = "indices"
        cloned_mask = object()
        self.assertEqual(
            attn._get_index_share_topk_holder(cloned_mask)[2], "indices"
        )

    def test_shared_consumer_reads_producer_holder_key(self):
        from paddlefleet.transformer.dsa_attention import DSAttention

        config = TransformerConfig(
            hidden_size=64,
            num_attention_heads=2,
            num_hidden_layers=4,
            dsa_indexer_topk_freq=4,
        )
        producer = DSAttention.__new__(DSAttention)
        producer.config = config
        producer.layer_number = 0
        producer.skip_topk = False
        producer.index_share = True
        producer.source_layer = 0
        consumer = DSAttention.__new__(DSAttention)
        consumer.config = config
        consumer.layer_number = 1
        consumer.skip_topk = True
        consumer.index_share = True
        consumer.source_layer = 0
        holder = producer._get_index_share_topk_holder(object())
        holder[producer.layer_number] = "topk"
        self.assertEqual(
            consumer._get_index_share_topk_holder(object())[
                consumer.source_layer
            ],
            "topk",
        )

    def test_source_layer_at_or_before_skip_offset_is_itself(self):
        from paddlefleet.transformer.dsa_attention import (
            source_dsa_compute_layer,
        )

        self.assertEqual(
            source_dsa_compute_layer(3, skip_topk_offset=3, topk_freq=4), 3
        )
        self.assertEqual(
            source_dsa_compute_layer(2, skip_topk_offset=3, topk_freq=4), 2
        )

    def test_logical_mtp_layer_keeps_physical_index(self):
        from paddlefleet.transformer.dsa_attention import (
            decoder_dsa_logical_layer,
        )

        config = TransformerConfig(
            hidden_size=64,
            num_attention_heads=2,
            num_hidden_layers=4,
            num_empty_layers_add_in_head=2,
        )
        self.assertEqual(
            decoder_dsa_logical_layer(config, 7, is_mtp_layer=True), 7
        )

    def test_producer_rejects_layer_outside_indexer_types(self):
        from paddlefleet.transformer.dsa_attention import (
            decoder_dsa_topk_producer_layer,
        )

        config = TransformerConfig(
            hidden_size=64,
            num_attention_heads=2,
            num_hidden_layers=4,
            dsa_indexer_types=["full", "full", "full", "shared"],
        )
        with self.assertRaisesRegex(ValueError, "outside dsa_indexer_types"):
            decoder_dsa_topk_producer_layer(config, 4)

    def test_periodic_producer_without_indexer_types(self):
        from paddlefleet.transformer.dsa_attention import (
            decoder_dsa_topk_producer_layer,
        )

        config = TransformerConfig(
            hidden_size=64,
            num_attention_heads=2,
            num_hidden_layers=4,
            dsa_indexer_topk_freq=4,
            dsa_indexer_skip_topk_offset=0,
        )
        self.assertEqual(decoder_dsa_topk_producer_layer(config, 0), 0)
        self.assertEqual(decoder_dsa_topk_producer_layer(config, 3), 0)

    def test_negative_layer_does_not_publish_shared_topk(self):
        from paddlefleet.transformer.dsa_attention import (
            _decoder_layer_publishes_shared_topk,
        )

        config = TransformerConfig(
            hidden_size=64, num_attention_heads=2, num_hidden_layers=4
        )
        self.assertFalse(_decoder_layer_publishes_shared_topk(config, -1))

    def test_mtp_shared_indexer_requires_a_decoder(self):
        from paddlefleet.transformer.dsa_attention import (
            resolve_dsa_indexer_layout,
        )

        config = TransformerConfig(
            hidden_size=64, num_attention_heads=2, num_hidden_layers=4
        )
        config.num_hidden_layers = 0
        config.dsa_index_share_for_mtp_iteration = True
        with self.assertRaisesRegex(ValueError, "preceding decoder layer"):
            resolve_dsa_indexer_layout(config, 0, is_mtp_layer=True)

    def test_skip_consumer_holder_does_not_contain_source_until_producer_runs(
        self,
    ):
        from paddlefleet.transformer.dsa_attention import DSAttention

        attn = DSAttention.__new__(DSAttention)
        attn.config = TransformerConfig(
            hidden_size=64, num_attention_heads=2, num_hidden_layers=4
        )
        attn.source_layer = 0
        holder = attn._get_index_share_topk_holder(None)
        self.assertEqual(holder, {})
        self.assertNotIn(attn.source_layer, holder)
