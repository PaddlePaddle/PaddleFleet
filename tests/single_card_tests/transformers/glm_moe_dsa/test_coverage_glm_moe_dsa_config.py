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
"""Single card tests for ``GlmMoeDsaConfig``.

The constructor reconciles three RoPE spellings that the official
GLM-5.2 ``config.json``, the Fleet providers and the HF converters each
expect:

* ``rope_parameters`` is the official nested field and stays the source
  of truth for ``rope_theta`` / ``rope_type``;
* ``rope_scaling`` stays ``None`` unless the caller passes the legacy
  flat dict, so ``from_json_file`` keeps agreeing with a default build;
* ``rotary_base`` / ``rope_type`` are derived aliases and must not be
  serialized back out.

Config objects only, so this needs no device and no checkpoint.
"""

import json
import tempfile
import unittest
from pathlib import Path

from paddlefleet.transformers.glm_moe_dsa.configuration import (
    GlmMoeDsaConfig,
)


def _small(**kwargs):
    """Build a small config; size fields are irrelevant to these paths."""
    kwargs.setdefault("vocab_size", 256)
    kwargs.setdefault("hidden_size", 24)
    return GlmMoeDsaConfig(**kwargs)


class GlmMoeDsaConfigIdentityTests(unittest.TestCase):
    def test_model_type_and_attribute_map(self):
        self.assertEqual(GlmMoeDsaConfig.model_type, "glm_moe_dsa")
        self.assertEqual(
            GlmMoeDsaConfig.keys_to_ignore_at_inference, ["past_key_values"]
        )
        # ``rotary_interleaved`` is the Fleet-facing alias; the stored
        # attribute keeps the official ``indexer_rope_interleave`` name.
        self.assertEqual(
            GlmMoeDsaConfig.attribute_map["rotary_interleaved"],
            "indexer_rope_interleave",
        )
        self.assertEqual(
            GlmMoeDsaConfig.attribute_map["num_classes"], "num_labels"
        )

    def test_moe_and_attention_defaults(self):
        config = GlmMoeDsaConfig()
        self.assertEqual(config.num_hidden_layers, 46)
        self.assertEqual(config.n_routed_experts, 128)
        self.assertEqual(config.n_shared_experts, 1)
        self.assertEqual(config.num_experts_per_tok, 8)
        self.assertEqual(config.first_k_dense_replace, 1)
        self.assertEqual(config.moe_intermediate_size, 1408)
        self.assertEqual(config.scoring_func, "sigmoid")
        self.assertEqual(config.topk_method, "noaux_tc")
        self.assertTrue(config.norm_topk_prob)
        self.assertTrue(config.seq_aux)
        self.assertTrue(config.using_flex_token)
        self.assertTrue(config.use_qk_norm)
        self.assertFalse(config.use_fp8)
        self.assertFalse(config.fp32_residual_connection)
        self.assertFalse(config.disable_ffn_model_parallel)
        self.assertEqual(config.moe_subbatch_token_num_before_dispatch, 0)
        self.assertEqual(config.pp_seg_method, "layer:Glm4MoeDecoderLayer")
        self.assertIsNone(config.sliding_window)
        self.assertFalse(config.fd_fallback)

    def test_optional_switches_are_passed_through(self):
        config = _small(sliding_window=512, fd_fallback=True)
        self.assertEqual(config.sliding_window, 512)
        self.assertTrue(config.fd_fallback)


class GlmMoeDsaConfigRopeTests(unittest.TestCase):
    def test_default_build_derives_nested_rope_from_top_level_theta(self):
        config = _small()
        self.assertIsNone(config.rope_scaling)
        self.assertEqual(
            config.rope_parameters,
            {"rope_type": "default", "rope_theta": 10000.0},
        )
        self.assertEqual(config.rope_theta, 10000.0)
        self.assertEqual(config.rotary_base, 10000.0)
        # "default" is renamed to the Fleet spelling, other types are kept.
        self.assertEqual(config.rope_type, "rope")
        self.assertFalse(config.rope_interleave)
        self.assertFalse(config.indexer_rope_interleave)
        self.assertFalse(config.rotary_interleaved)
        self.assertEqual(config.partial_rotary_factor, 0.5)

    def test_nested_rope_parameters_win_over_top_level_theta(self):
        config = _small(
            rope_theta=10000.0,
            rope_parameters={"rope_theta": 8_000_000},
        )
        self.assertEqual(config.rope_parameters["rope_theta"], 8_000_000)
        self.assertEqual(config.rope_theta, 8_000_000)
        self.assertEqual(config.rotary_base, 8_000_000)
        self.assertEqual(config.rope_type, "rope")
        self.assertIsNone(config.rope_scaling)

    def test_nested_partial_rotary_factor_is_dropped_not_applied(self):
        # The nested value is derived from the top-level field, so it must
        # not leak back into the config or the serialized nest.
        config = _small(
            partial_rotary_factor=0.25,
            rope_parameters={
                "rope_theta": 8_000_000,
                "partial_rotary_factor": 0.75,
            },
        )
        self.assertEqual(config.partial_rotary_factor, 0.25)
        self.assertNotIn("partial_rotary_factor", config.rope_parameters)

    def test_non_default_rope_type_is_preserved(self):
        config = _small(
            rope_parameters={
                "rope_type": "yarn",
                "rope_theta": 8_000_000,
                "factor": 2.0,
            }
        )
        self.assertEqual(config.rope_type, "yarn")
        self.assertEqual(config.rope_parameters["factor"], 2.0)
        self.assertEqual(config.rotary_base, 8_000_000)

    def test_legacy_flat_rope_scaling_is_upgraded_in_place(self):
        config = _small(
            partial_rotary_factor=0.25,
            rope_scaling={
                "type": "linear",
                "factor": 2.0,
                "partial_rotary_factor": 0.75,
            },
        )
        # BC: a "type" field is mirrored onto "rope_type" ...
        self.assertEqual(config.rope_scaling["type"], "linear")
        self.assertEqual(config.rope_scaling["rope_type"], "linear")
        self.assertEqual(config.rope_type, "linear")
        # ... the flat dict also seeds the nested official field ...
        self.assertEqual(config.rope_parameters["rope_type"], "linear")
        self.assertEqual(config.rope_theta, config.rotary_base)
        # ... and the derived factor is stripped from both views.
        self.assertNotIn("partial_rotary_factor", config.rope_scaling)
        self.assertNotIn("partial_rotary_factor", config.rope_parameters)
        self.assertEqual(config.partial_rotary_factor, 0.25)

    def test_interleave_flags_are_independent(self):
        fleet_layout = _small(rope_interleave=True)
        self.assertTrue(fleet_layout.rope_interleave)
        self.assertFalse(fleet_layout.rotary_interleaved)

        indexer_layout = _small(indexer_rope_interleave=True)
        self.assertFalse(indexer_layout.rope_interleave)
        self.assertTrue(indexer_layout.indexer_rope_interleave)
        self.assertTrue(indexer_layout.rotary_interleaved)

    def test_official_indexer_field_reads_back_through_the_alias(self):
        config = GlmMoeDsaConfig.from_dict({"indexer_rope_interleave": True})
        self.assertTrue(config.indexer_rope_interleave)
        self.assertTrue(config.rotary_interleaved)


class GlmMoeDsaConfigRoundTripTests(unittest.TestCase):
    def test_from_dict_reproduces_to_dict(self):
        config = _small(rope_theta=8_000_000)
        restored = GlmMoeDsaConfig.from_dict(config.to_dict())
        self.assertEqual(restored.to_dict(), config.to_dict())
        self.assertEqual(restored.rope_parameters, config.rope_parameters)
        self.assertIsNone(restored.rope_scaling)

    def test_json_file_round_trip_keeps_rope_scaling_none(self):
        config = _small()
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "config.json")
            config.to_json_file(path)
            loaded = GlmMoeDsaConfig.from_json_file(path)
        self.assertIsNone(loaded.rope_scaling)
        self.assertEqual(loaded.to_dict(), config.to_dict())

    def test_save_pretrained_keeps_official_field_and_drops_aliases(self):
        config = _small(rope_theta=8_000_000)
        with tempfile.TemporaryDirectory() as tmp:
            config.save_pretrained(tmp)
            saved = json.loads((Path(tmp) / "config.json").read_text())
            reloaded = GlmMoeDsaConfig.from_pretrained(tmp)

        self.assertIn("rope_parameters", saved)
        self.assertEqual(saved["rope_parameters"]["rope_theta"], 8_000_000)
        self.assertNotIn("partial_rotary_factor", saved["rope_parameters"])
        # Derived aliases are registered as unsavable.
        self.assertNotIn("rotary_base", saved)
        self.assertNotIn("rope_type", saved)
        self.assertEqual(reloaded.rope_theta, 8_000_000)
        self.assertEqual(reloaded.rotary_base, 8_000_000)
        self.assertEqual(reloaded.rope_type, "rope")
        self.assertIsNone(reloaded.rope_scaling)


if __name__ == "__main__":
    unittest.main()
