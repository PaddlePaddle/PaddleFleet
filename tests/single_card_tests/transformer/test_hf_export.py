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

"""Unit tests for the Fleet<->HF config bridge (paddlefleet.transformer.hf_export).

Pure-Python (no paddle): covers the naming map / rules, structural YaRN
detection, rope_scaling pack/unpack, window semantics, MTP trimming, mHC
injection, and the provider-fallback accessor.
"""

import functools
import unittest
from types import SimpleNamespace

from paddlefleet.transformer.hf_export import (
    FLEET_HF_FIELD_MAPPING,
    HF_EXPORT_RULES,
    HF_IMPORT_RULES,
    check_window_export_conflict,
    hidden_act_to_hf,
    inject_mhc_from_provider,
    is_active_window,
    is_csa_config,
    pack_rope_scaling,
    rule_target,
    source_or_provider,
    swa_aware_import_rules,
    trim_mtp_layers,
    unpack_rope_scaling,
    uses_yarn,
)


class TestRulesAndMapping(unittest.TestCase):
    def test_export_import_rules_are_inverse_on_renames(self):
        for fleet_key, spec in HF_EXPORT_RULES.items():
            hf_key = spec[0] if isinstance(spec, tuple) else spec
            self.assertIn(hf_key, HF_IMPORT_RULES)
            back = HF_IMPORT_RULES[hf_key]
            back_key = back[0] if isinstance(back, tuple) else back
            self.assertEqual(back_key, fleet_key)

    def test_mapping_entry_count(self):
        self.assertEqual(len(FLEET_HF_FIELD_MAPPING), len(HF_EXPORT_RULES))

    def test_rule_target(self):
        self.assertEqual(
            rule_target("csa_window_size", HF_EXPORT_RULES), "sliding_window"
        )
        self.assertEqual(
            rule_target("unknown_key", HF_EXPORT_RULES), "unknown_key"
        )

    def test_value_converters(self):
        # params_dtype -> torch_dtype strips the "paddle." prefix on export.
        spec = HF_EXPORT_RULES["params_dtype"]
        self.assertEqual(spec[0], "torch_dtype")
        self.assertEqual(spec[1]("paddle.bfloat16"), "bfloat16")
        # multimax_modules list -> scalar on export.
        spec = HF_EXPORT_RULES["multimax_modules"]
        self.assertEqual(spec[1](["lm_head"]), "lm_head")

    def test_dtype_import_returns_canonical_string(self):
        # HF torch_dtype "bfloat16" must import to a bare Paddle-accepted
        # dtype string, NOT "paddle.bfloat16" (create_parameter rejects it).
        spec = HF_IMPORT_RULES["torch_dtype"]
        self.assertEqual(spec[0], "params_dtype")
        self.assertEqual(spec[1]("bfloat16"), "bfloat16")
        # A Fleet-native value still normalises to the bare name.
        self.assertEqual(spec[1]("paddle.bfloat16"), "bfloat16")

    def test_hidden_act_export_handles_partial(self):
        # gelu_pytorch_tanh round-trips through functools.partial (no __name__).
        act_fn = HF_EXPORT_RULES["hidden_act"][1]
        partial = functools.partial(
            lambda x, approximate=False: x, approximate=True
        )
        partial.func.__name__ = "gelu"
        self.assertEqual(hidden_act_to_hf(partial), "gelu_pytorch_tanh")
        # exposed via the export rule too.
        self.assertEqual(act_fn(partial), "gelu_pytorch_tanh")

    def test_hidden_act_export_named_callable(self):
        def silu(x):
            return x

        self.assertEqual(hidden_act_to_hf(silu), "silu")

    def test_hidden_act_export_passthrough_string(self):
        self.assertEqual(hidden_act_to_hf("relu"), "relu")


class TestSourceOrProvider(unittest.TestCase):
    def test_source_wins(self):
        prov = SimpleNamespace(k=2)
        self.assertEqual(source_or_provider({"k": 1}, prov, "k"), 1)

    def test_provider_fallback(self):
        prov = SimpleNamespace(k=2)
        self.assertEqual(source_or_provider({}, prov, "k"), 2)

    def test_none_when_absent(self):
        self.assertIsNone(source_or_provider({}, None, "k"))


class TestUsesYarn(unittest.TestCase):
    def test_global_rope_type_yarn(self):
        self.assertTrue(uses_yarn({"rope_type": "yarn"}))

    def test_plain_rope_not_yarn(self):
        # leftover rotary_scaling_factor must NOT trigger yarn on plain rope.
        self.assertFalse(
            uses_yarn({"rope_type": "rope", "rotary_scaling_factor": 16})
        )

    def test_dsv4_hybrid_hca_defaults_yarn(self):
        self.assertTrue(
            uses_yarn(
                {
                    "rope_type": "rope",
                    "experimental_attention_variant": "dsv4_hybrid",
                    "csa_compress_ratios": [128, -2],
                }
            )
        )

    def test_dsv4_hybrid_hca_override_rope(self):
        self.assertFalse(
            uses_yarn(
                {
                    "rope_type": "rope",
                    "experimental_attention_variant": "dsv4_hybrid",
                    "csa_compress_ratios": [128, -2],
                    "hca_rope_type": "rope",
                }
            )
        )

    def test_dsv4_hybrid_csa_layer_defaults_yarn(self):
        self.assertTrue(
            uses_yarn(
                {
                    "rope_type": "rope",
                    "experimental_attention_variant": "dsv4_hybrid",
                    "csa_compress_ratios": [64, -2],
                }
            )
        )

    def test_dsv4_hybrid_no_compressed_layer(self):
        self.assertFalse(
            uses_yarn(
                {
                    "rope_type": "rope",
                    "experimental_attention_variant": "dsv4_hybrid",
                    "csa_compress_ratios": [-2, 0, -2],
                }
            )
        )

    def test_dsv4_hybrid_global_yarn_but_all_layers_overridden_rope(self):
        # Global rope_type="yarn" must NOT short-circuit for dsv4_hybrid: with
        # HCA/CSA forced to "rope" and only a window layer, no layer uses YaRN.
        self.assertFalse(
            uses_yarn(
                {
                    "rope_type": "yarn",
                    "experimental_attention_variant": "dsv4_hybrid",
                    "csa_compress_ratios": [128, 64, 0],
                    "hca_rope_type": "rope",
                    "csa_rope_type": "rope",
                }
            )
        )

    def test_dsv4_hybrid_mla_layer_follows_global_yarn(self):
        # An MLA (-2) layer is built on the plain MLA path and follows the
        # global rope_type, so global "yarn" activates YaRN through it.
        self.assertTrue(
            uses_yarn(
                {
                    "rope_type": "yarn",
                    "experimental_attention_variant": "dsv4_hybrid",
                    "csa_compress_ratios": [-2, 0],
                    "hca_rope_type": "rope",
                    "csa_rope_type": "rope",
                }
            )
        )

    def test_dsv4_hybrid_mla_layer_global_rope_not_yarn(self):
        # MLA layer with a non-yarn global rope_type stays plain RoPE.
        self.assertFalse(
            uses_yarn(
                {
                    "rope_type": "rope",
                    "experimental_attention_variant": "dsv4_hybrid",
                    "csa_compress_ratios": [-2, 0],
                }
            )
        )

    def test_provider_fallback_rope_type(self):
        prov = SimpleNamespace(rope_type="yarn")
        self.assertTrue(uses_yarn({}, prov))


class TestRopeScaling(unittest.TestCase):
    def test_pack_yarn(self):
        rs = pack_rope_scaling(
            {
                "rope_type": "yarn",
                "rotary_scaling_factor": 4.0,
                "original_max_position_embeddings": 4096,
                "beta_fast": 32,
                "beta_slow": 1,
            }
        )
        self.assertEqual(rs["type"], "yarn")
        self.assertEqual(rs["factor"], 4.0)
        self.assertEqual(rs["beta_fast"], 32)

    def test_pack_none_when_no_yarn(self):
        self.assertIsNone(
            pack_rope_scaling(
                {"rope_type": "rope", "rotary_scaling_factor": 16}
            )
        )

    def test_pack_normalizes_type_to_yarn(self):
        # dsv4_hybrid compressed layer -> yarn active; global type "rope" normalized.
        rs = pack_rope_scaling(
            {
                "rope_type": "rope",
                "experimental_attention_variant": "dsv4_hybrid",
                "csa_compress_ratios": [128],
                "rotary_scaling_factor": 8,
            }
        )
        self.assertEqual(rs["type"], "yarn")

    def test_pack_omits_neutral_mscale(self):
        rs = pack_rope_scaling(
            {"rope_type": "yarn", "mscale": 1.0, "mscale_all_dim": 0.0}
        )
        self.assertNotIn("mscale", rs)
        self.assertNotIn("mscale_all_dim", rs)

    def test_unpack(self):
        flat = unpack_rope_scaling(
            {"type": "yarn", "factor": 4.0, "beta_fast": 32}
        )
        self.assertEqual(flat["rope_type"], "yarn")
        self.assertEqual(flat["rotary_scaling_factor"], 4.0)
        self.assertEqual(flat["beta_fast"], 32)

    def test_unpack_canonical_rope_type(self):
        # Current HF configs use "rope_type" (not the legacy "type" alias);
        # the YaRN type must survive import.
        flat = unpack_rope_scaling({"rope_type": "yarn", "factor": 4})
        self.assertEqual(flat["rope_type"], "yarn")
        self.assertEqual(flat["rotary_scaling_factor"], 4)

    def test_unpack_rope_type_wins_over_legacy_type(self):
        flat = unpack_rope_scaling({"rope_type": "yarn", "type": "linear"})
        self.assertEqual(flat["rope_type"], "yarn")

    def test_unpack_empty(self):
        self.assertEqual(unpack_rope_scaling(None), {})


class TestWindow(unittest.TestCase):
    def test_is_active_window(self):
        self.assertFalse(is_active_window(0))
        self.assertFalse(is_active_window(None))
        self.assertFalse(is_active_window([0, 0]))
        self.assertTrue(is_active_window(128))
        self.assertTrue(is_active_window([0, 128, 0]))

    def test_conflict_raises(self):
        with self.assertRaises(ValueError):
            check_window_export_conflict(
                {"sliding_window": 4096, "csa_window_size": 128}
            )

    def test_no_conflict_ok(self):
        check_window_export_conflict({"sliding_window": 4096})  # no raise
        check_window_export_conflict({"csa_window_size": 128})  # no raise

    def test_swa_aware_import_rules_drops_sliding_window_for_swa(self):
        rules = {"sliding_window": "csa_window_size", "other": "x"}
        swa_cfg = {"swa_num_attention_heads": 8}
        out = swa_aware_import_rules(swa_cfg, rules)
        self.assertNotIn("sliding_window", out)
        self.assertIn("other", out)

    def test_swa_aware_import_rules_native_sliding_window_kept(self):
        # Standard HF SWA (Mistral: bare sliding_window, no CSA markers) must
        # keep its native name rather than be re-homed to csa_window_size.
        rules = {"sliding_window": "csa_window_size", "other": "x"}
        out = swa_aware_import_rules({"sliding_window": 4096}, rules)
        self.assertNotIn("sliding_window", out)
        self.assertIn("other", out)

    def test_swa_aware_import_rules_keeps_rename_for_csa(self):
        rules = {"sliding_window": "csa_window_size", "other": "x"}
        out = swa_aware_import_rules({"compress_ratios": [128, 64]}, rules)
        self.assertEqual(out["sliding_window"], "csa_window_size")
        self.assertIn("other", out)

    def test_swa_aware_import_rules_keeps_rename_for_dsv4_variant(self):
        rules = {"sliding_window": "csa_window_size"}
        out = swa_aware_import_rules(
            {"experimental_attention_variant": "dsv4_hybrid"}, rules
        )
        self.assertIn("sliding_window", out)

    def test_is_csa_config(self):
        self.assertTrue(is_csa_config({"compress_ratios": [128]}))
        self.assertTrue(
            is_csa_config({"experimental_attention_variant": "dsv4_hybrid"})
        )
        self.assertFalse(is_csa_config({"sliding_window": 4096}))


class TestMtpTrim(unittest.TestCase):
    def test_trim(self):
        out = {
            "num_nextn_predict_layers": 1,
            "window_attn_skip_freq": [1, 2, 3],
            "csa_compress_ratios": [4, 5, 6],
        }
        trim_mtp_layers(out)
        self.assertEqual(out["window_attn_skip_freq"], [1, 2])
        self.assertEqual(out["csa_compress_ratios"], [4, 5])
        self.assertEqual(out["num_nextn_predict_layers"], 0)

    def test_trim_hf_renamed_keys(self):
        # HF-side names: hybrid_layer_pattern <- window_attn_skip_freq and
        # compress_ratios <- csa_compress_ratios must also be trimmed.
        out = {
            "num_nextn_predict_layers": 1,
            "hybrid_layer_pattern": [1, 2, 3],
            "compress_ratios": [4, 5, 6],
        }
        trim_mtp_layers(out)
        self.assertEqual(out["hybrid_layer_pattern"], [1, 2])
        self.assertEqual(out["compress_ratios"], [4, 5])
        self.assertEqual(out["num_nextn_predict_layers"], 0)

    def test_no_trim_when_no_mtp(self):
        out = {"num_nextn_predict_layers": 0, "csa_compress_ratios": [4, 5, 6]}
        trim_mtp_layers(out)
        self.assertEqual(out["csa_compress_ratios"], [4, 5, 6])


class TestMhcInjection(unittest.TestCase):
    def test_inject(self):
        prov = SimpleNamespace(
            enable_hyper_connections=True,
            num_residual_streams=4,
            mhc_sinkhorn_iterations=3,
        )
        raw = {}
        inject_mhc_from_provider(raw, prov)
        self.assertEqual(raw["num_residual_streams"], 4)
        self.assertEqual(raw["mhc_sinkhorn_iterations"], 3)
        self.assertEqual(raw["hc_eps"], 1e-6)

    def test_disabled_noop(self):
        prov = SimpleNamespace(
            enable_hyper_connections=False, num_residual_streams=4
        )
        raw = {}
        inject_mhc_from_provider(raw, prov)
        self.assertEqual(raw, {})

    def test_source_value_wins(self):
        prov = SimpleNamespace(
            enable_hyper_connections=True,
            num_residual_streams=4,
            mhc_sinkhorn_iterations=3,
        )
        raw = {"num_residual_streams": 9}
        inject_mhc_from_provider(raw, prov)
        self.assertEqual(raw["num_residual_streams"], 9)


if __name__ == "__main__":
    unittest.main()
