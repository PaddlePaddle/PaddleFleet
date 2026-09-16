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

"""Behavior tests for MoE AOA (weight-conversion) config generation.

Scope: 配置与运行基础设施. These tests drive the real production entry points
(``gen_aoa_config`` / ``gen_inv_aoa_config`` and the statement generators) and
check that config values are actually *consumed* into the generated conversion
statements -- alias resolution, defaults, explicit-off, bool distinctions and
layer-index propagation -- rather than reading dataclass fields back. Expected
statement strings are written independently by hand, not produced by calling the
production helpers. CPU-only, no Paddle / accelerator required.
"""

import unittest

from paddlefleet.transformers.aoa_config_base import (
    MoEAOAConfigGenerator,
    MoEAOAConfigParams,
)


class _Config:
    """Attribute-only config double.

    ``_extract_params`` reads ``num_hidden_layers`` / ``num_attention_heads`` /
    ``num_key_value_heads`` as direct attributes and everything else through
    ``getattr``/``hasattr``. It only consults ``fd_fallback`` when the config
    exposes a ``.get`` method, so this attribute-only double deliberately lacks
    ``.get`` to exercise the "no dict interface" extraction branch.
    """

    _REQUIRED = {
        "num_hidden_layers": 0,
        "num_attention_heads": 0,
        "num_key_value_heads": 0,
    }

    def __init__(self, **kwargs):
        for key, value in {**self._REQUIRED, **kwargs}.items():
            setattr(self, key, value)


class _DictConfig(_Config):
    """Config double that additionally exposes dict-style ``.get``.

    Presence of ``.get`` is what enables the ``fd_fallback`` extraction branch
    in ``_extract_params``; comparing this against ``_Config`` isolates that
    behavior.
    """

    def get(self, key, default=None):
        return getattr(self, key, default)


class TestExtractParamsAliases(unittest.TestCase):
    """_extract_params resolves config aliases / precedence / type coercion."""

    def test_n_routed_experts_takes_precedence_over_num_experts(self):
        # Both present: production must prefer n_routed_experts, and the chosen
        # count must reach the grouped-GEMM consumer (5 experts, not 99).
        config = _Config(
            num_hidden_layers=1,
            num_attention_heads=8,
            num_key_value_heads=2,
            n_routed_experts=5,
            num_experts=99,
            moe_expert_fusion=True,
        )
        params = MoEAOAConfigGenerator._extract_params(config)
        self.assertEqual(params.num_experts, 5)

        gemm = MoEAOAConfigGenerator._get_grouped_gemm_statements(params)
        left = gemm[0].split(" -> ")[0]
        self.assertEqual(len(left.split(",")), 5)

    def test_num_experts_used_when_no_n_routed_experts(self):
        config = _Config(
            num_hidden_layers=1,
            num_attention_heads=8,
            num_key_value_heads=2,
            num_experts=7,
        )
        params = MoEAOAConfigGenerator._extract_params(config)
        self.assertEqual(params.num_experts, 7)

    def test_missing_experts_defaults_to_zero(self):
        config = _Config(
            num_hidden_layers=1, num_attention_heads=8, num_key_value_heads=2
        )
        params = MoEAOAConfigGenerator._extract_params(config)
        self.assertEqual(params.num_experts, 0)

    def test_fd_fallback_requires_dict_get_interface(self):
        # Same fd_fallback=True attribute, but only the config exposing .get
        # actually turns it on; the attribute-only config stays False.
        common = {
            "num_hidden_layers": 1,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "fd_fallback": True,
        }
        with_get = MoEAOAConfigGenerator._extract_params(_DictConfig(**common))
        without_get = MoEAOAConfigGenerator._extract_params(_Config(**common))
        self.assertTrue(with_get.fd_fallback)
        self.assertFalse(without_get.fd_fallback)

    def test_num_head_empty_layers_alias_and_falsy(self):
        base = {
            "num_hidden_layers": 1,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
        }
        aliased = MoEAOAConfigGenerator._extract_params(
            _Config(num_empty_layers_add_in_head=3, **base)
        )
        zeroed = MoEAOAConfigGenerator._extract_params(
            _Config(num_empty_layers_add_in_head=0, **base)
        )
        absent = MoEAOAConfigGenerator._extract_params(_Config(**base))
        self.assertEqual(aliased.num_head_empty_layers, 3)
        self.assertEqual(zeroed.num_head_empty_layers, 0)
        self.assertEqual(absent.num_head_empty_layers, 0)

    def test_num_nextn_predict_layers_none_coerced_to_zero(self):
        params = MoEAOAConfigGenerator._extract_params(
            _Config(
                num_hidden_layers=1,
                num_attention_heads=8,
                num_key_value_heads=2,
                num_nextn_predict_layers=None,
            )
        )
        self.assertEqual(params.num_nextn_predict_layers, 0)


class TestModelPrefixPropagation(unittest.TestCase):
    """model_prefix selection propagates into generated statement targets."""

    def test_default_prefix_is_model_dot(self):
        prefix = MoEAOAConfigGenerator._get_model_prefix(_Config())
        self.assertEqual(prefix, "model.")

    def test_base_model_class_uses_empty_prefix_end_to_end(self):
        # When the generator IS its own base_model_class the prefix is "" and
        # that must reach the target names of the basic-weight statements.
        class _BaseGen(MoEAOAConfigGenerator):
            pass

        _BaseGen.base_model_class = _BaseGen
        self.assertEqual(_BaseGen._get_model_prefix(_Config()), "")

        result = _BaseGen.gen_aoa_config(
            _Config(
                num_hidden_layers=1,
                num_attention_heads=8,
                num_key_value_heads=2,
            )
        )
        stmts = result["aoa_statements"]
        self.assertIn("model.norm.weight -> norm.weight", stmts)
        self.assertIn(
            "model.embed_tokens.weight -> embedding.embed_tokens.weight", stmts
        )


class TestBasicWeightStatements(unittest.TestCase):
    """norm / embedding / lm_head mapping, incl. tie_word_embeddings bool."""

    def test_untied_lm_head_maps_from_lm_head(self):
        params = MoEAOAConfigParams(tie_word_embeddings=False)
        stmts = MoEAOAConfigGenerator._get_basic_weight_statements(params)
        self.assertEqual(len(stmts), 3)
        self.assertEqual(stmts[0], "model.norm.weight -> model.norm.weight")
        self.assertEqual(
            stmts[1],
            "model.embed_tokens.weight -> model.embedding.embed_tokens.weight",
        )
        self.assertEqual(stmts[2], "lm_head.weight -> model.lm_head.weight")

    def test_tied_lm_head_maps_from_embed_tokens(self):
        params = MoEAOAConfigParams(tie_word_embeddings=True)
        stmts = MoEAOAConfigGenerator._get_basic_weight_statements(params)
        self.assertEqual(len(stmts), 3)
        # The only difference from the untied case is the lm_head source.
        self.assertEqual(
            stmts[2],
            "model.embed_tokens.weight -> model.lm_head.weight",
        )


class TestAttentionStatements(unittest.TestCase):
    """Standard-vs-MLA switch and head-count propagation into fused_qkv."""

    def test_standard_attention_embeds_head_counts_in_order(self):
        # Distinct 12 / 3 catch a head <-> kv-group swap in the fused_qkv spec.
        params = MoEAOAConfigParams(
            num_attention_heads=12,
            num_key_value_heads=3,
            attention_bias=False,
        )
        stmts = MoEAOAConfigGenerator._get_attention_statements(
            params, 0, "model.layers.0", "model.layers.0"
        )
        self.assertEqual(len(stmts), 1)
        self.assertEqual(
            stmts[0],
            "model.layers.0.self_attn.q_proj.weight^T, "
            "model.layers.0.self_attn.k_proj.weight^T, "
            "model.layers.0.self_attn.v_proj.weight^T -> "
            "model.layers.0.self_attn.qkv_proj.weight, fused_qkv, "
            "num_heads=12, num_key_value_groups=3",
        )

    def test_attention_bias_adds_bias_row_only_when_enabled(self):
        on = MoEAOAConfigParams(
            num_attention_heads=12, num_key_value_heads=3, attention_bias=True
        )
        off = MoEAOAConfigParams(
            num_attention_heads=12, num_key_value_heads=3, attention_bias=False
        )
        on_stmts = MoEAOAConfigGenerator._get_attention_statements(
            on, 0, "model.layers.0", "model.layers.0"
        )
        off_stmts = MoEAOAConfigGenerator._get_attention_statements(
            off, 0, "model.layers.0", "model.layers.0"
        )
        self.assertEqual(len(on_stmts), 2)
        self.assertEqual(len(off_stmts), 1)
        self.assertIn("self_attn.qkv_proj.bias", on_stmts[1])
        self.assertIn("axis=0", on_stmts[1])

    def test_mla_replaces_fused_qkv_with_compressed_projections(self):
        params = MoEAOAConfigParams(
            multi_latent_attention=True, use_qk_norm=False
        )
        stmts = MoEAOAConfigGenerator._get_attention_statements(
            params, 0, "model.layers.0", "model.layers.0"
        )
        self.assertEqual(len(stmts), 4)
        joined = "\n".join(stmts)
        self.assertNotIn("fused_qkv", joined)
        self.assertIn("kv_a_proj_with_mqa", joined)
        self.assertIn("kv_b_proj", joined)
        self.assertIn("q_a_proj", joined)
        self.assertIn("q_b_proj", joined)

    def test_mla_qk_norm_adds_two_layernorm_rows(self):
        no_norm = MoEAOAConfigParams(
            multi_latent_attention=True, use_qk_norm=False
        )
        with_norm = MoEAOAConfigParams(
            multi_latent_attention=True, use_qk_norm=True
        )
        base = MoEAOAConfigGenerator._get_attention_statements(
            no_norm, 0, "model.layers.0", "model.layers.0"
        )
        extra = MoEAOAConfigGenerator._get_attention_statements(
            with_norm, 0, "model.layers.0", "model.layers.0"
        )
        self.assertEqual(len(extra) - len(base), 2)
        joined = "\n".join(extra)
        self.assertIn("q_a_layernorm.weight", joined)
        self.assertIn("kv_a_layernorm.weight", joined)

    def test_mla_indexer_added_only_for_positive_index_n_heads(self):
        without = MoEAOAConfigParams(
            multi_latent_attention=True, index_n_heads=0
        )
        with_idx = MoEAOAConfigParams(
            multi_latent_attention=True, index_n_heads=4
        )
        base = MoEAOAConfigGenerator._get_attention_statements(
            without, 0, "model.layers.0", "model.layers.0"
        )
        idx = MoEAOAConfigGenerator._get_attention_statements(
            with_idx, 0, "model.layers.0", "model.layers.0"
        )
        # 3 indexer projections + 2 k_norm rows.
        self.assertEqual(len(idx) - len(base), 5)
        joined = "\n".join(idx)
        for weight_name in (
            "indexer.wq_b",
            "indexer.wk",
            "indexer.weights_proj",
        ):
            self.assertIn(weight_name, joined)
        self.assertIn("indexer.k_norm.weight", joined)


class TestRoutedExpertStatements(unittest.TestCase):
    """using_sonic_moe changes transpose (^T) and fusion axis for experts."""

    def test_non_sonic_transposes_and_fuses_on_axis_1(self):
        params = MoEAOAConfigParams(using_sonic_moe=False)
        stmts = MoEAOAConfigGenerator._get_routed_expert_statements(
            params, "model.layers.0", "model.layers.0"
        )
        self.assertEqual(len(stmts), 2)
        self.assertEqual(
            stmts[0],
            "model.layers.0.mlp.experts.$EXPERT_ID.down_proj.weight^T -> "
            "model.layers.0.mlp.experts.$EXPERT_ID.down_proj.weight",
        )
        self.assertIn("^T", stmts[1])
        self.assertTrue(stmts[1].endswith("axis=1"))

    def test_sonic_moe_skips_transpose_and_fuses_on_axis_0(self):
        params = MoEAOAConfigParams(using_sonic_moe=True)
        stmts = MoEAOAConfigGenerator._get_routed_expert_statements(
            params, "model.layers.0", "model.layers.0"
        )
        self.assertEqual(len(stmts), 2)
        self.assertEqual(
            stmts[0],
            "model.layers.0.mlp.experts.$EXPERT_ID.down_proj.weight -> "
            "model.layers.0.mlp.experts.$EXPERT_ID.down_proj.weight",
        )
        self.assertNotIn("^T", "\n".join(stmts))
        self.assertTrue(stmts[1].endswith("axis=0"))


class TestMoEExpertStatements(unittest.TestCase):
    """Shared-expert rows appear only when has_shared_experts is True."""

    def test_shared_experts_present_and_absent(self):
        with_shared = MoEAOAConfigParams(has_shared_experts=True)
        without_shared = MoEAOAConfigParams(has_shared_experts=False)
        a = MoEAOAConfigGenerator._get_moe_expert_statements(
            with_shared, "model.layers.0", "model.layers.0"
        )
        b = MoEAOAConfigGenerator._get_moe_expert_statements(
            without_shared, "model.layers.0", "model.layers.0"
        )
        self.assertIn("shared_experts", "\n".join(a))
        self.assertNotIn("shared_experts", "\n".join(b))
        # Shared experts contribute exactly two rows (down + fused up/gate).
        self.assertEqual(len(a) - len(b), 2)

    def test_gate_weight_cast_to_float32(self):
        params = MoEAOAConfigParams(has_shared_experts=False)
        stmts = MoEAOAConfigGenerator._get_moe_expert_statements(
            params, "model.layers.0", "model.layers.0"
        )
        gate_stmt = next(
            s for s in stmts if s.endswith("gate.weight, dtype='float32'")
        )
        self.assertIn("model.layers.0.mlp.gate.weight ->", gate_stmt)


class TestGroupedGemmStatements(unittest.TestCase):
    """Grouped-GEMM branch selection and per-expert weight consolidation."""

    def test_disabled_when_no_fusion_no_sonic_no_fp8_no_fallback(self):
        params = MoEAOAConfigParams(
            num_hidden_layers=1,
            num_experts=2,
            moe_expert_fusion=False,
            using_sonic_moe=False,
            fp8=False,
            fd_fallback=False,
        )
        self.assertEqual(
            MoEAOAConfigGenerator._get_grouped_gemm_statements(params), []
        )

    def test_fusion_consolidates_all_experts_into_grouped_weights(self):
        params = MoEAOAConfigParams(
            num_hidden_layers=2,
            first_k_dense_replace=0,
            num_experts=3,
            moe_expert_fusion=True,
            model_prefix="model.",
        )
        stmts = MoEAOAConfigGenerator._get_grouped_gemm_statements(params)
        # Two layers x (weight1, weight2).
        self.assertEqual(len(stmts), 4)
        self.assertEqual(
            stmts[0],
            "model.layers.0.mlp.experts.0.up_gate_proj.weight,"
            "model.layers.0.mlp.experts.1.up_gate_proj.weight,"
            "model.layers.0.mlp.experts.2.up_gate_proj.weight -> "
            "model.layers.0.mlp.grouped_gemm_experts.weight1, axis=0",
        )
        self.assertIn("grouped_gemm_experts.weight2", stmts[1])
        # Second layer targets layers.1.
        self.assertIn(
            "model.layers.1.mlp.grouped_gemm_experts.weight1", stmts[2]
        )

    def test_fd_fallback_uses_plain_expert_targets_not_grouped(self):
        params = MoEAOAConfigParams(
            num_hidden_layers=1,
            first_k_dense_replace=0,
            num_experts=2,
            moe_expert_fusion=False,
            using_sonic_moe=False,
            fp8=False,
            fd_fallback=True,
            model_prefix="model.",
        )
        stmts = MoEAOAConfigGenerator._get_grouped_gemm_statements(params)
        self.assertEqual(len(stmts), 2)
        joined = "\n".join(stmts)
        self.assertNotIn("grouped_gemm_experts", joined)
        self.assertIn("model.layers.0.mlp.experts.up_gate_proj, axis=0", joined)
        self.assertIn("model.layers.0.mlp.experts.down_proj, axis=0", joined)

    def test_fp8_alone_enables_grouped_gemm_forward(self):
        # fp8=True (no fusion / no sonic) must NOT fall through to fallback; it
        # produces grouped_gemm consolidation.
        params = MoEAOAConfigParams(
            num_hidden_layers=1,
            first_k_dense_replace=0,
            num_experts=2,
            moe_expert_fusion=False,
            using_sonic_moe=False,
            fp8=True,
            fd_fallback=False,
            model_prefix="model.",
        )
        stmts = MoEAOAConfigGenerator._get_grouped_gemm_statements(params)
        self.assertEqual(len(stmts), 2)
        self.assertIn("grouped_gemm_experts.weight1", stmts[0])

    def test_fp8_alone_produces_no_inverse_ungrouping(self):
        # Documented asymmetry: forward fp8-only emits grouped_gemm weights, but
        # the inverse ungroup guard requires (fusion or sonic) and not fp8, so
        # fp8-only yields nothing to un-group on save-back.
        params = MoEAOAConfigParams(
            num_experts=2,
            moe_expert_fusion=False,
            using_sonic_moe=False,
            fp8=True,
            fd_fallback=False,
            model_prefix="model.",
        )
        inv = MoEAOAConfigGenerator._get_inv_grouped_gemm_layer_statements(
            params, "model.layers.0"
        )
        self.assertEqual(inv, [])


class TestLayerRangeAndOffset(unittest.TestCase):
    """first_k_dense_replace / num_head_empty_layers / MTP index handling."""

    def test_dense_layers_only_below_first_k_dense_replace(self):
        none_dense = MoEAOAConfigParams(first_k_dense_replace=0)
        self.assertEqual(
            MoEAOAConfigGenerator._get_dense_layer_statements(none_dense), []
        )
        two_dense = MoEAOAConfigParams(
            first_k_dense_replace=2,
            num_attention_heads=8,
            num_key_value_heads=2,
        )
        stmts = MoEAOAConfigGenerator._get_dense_layer_statements(two_dense)
        joined = "\n".join(stmts)
        # Dense source layers are 0 and 1 only (reversed order).
        self.assertIn("model.layers.1.input_layernorm.weight", joined)
        self.assertIn("model.layers.0.input_layernorm.weight", joined)
        self.assertIn("fused_ffn", joined)

    def test_num_head_empty_layers_offsets_target_index(self):
        # Source dense layer 0 must map to target layers.2 when empty-head
        # offset is 2 (attention weights land on the offset side).
        params = MoEAOAConfigParams(
            first_k_dense_replace=1,
            num_head_empty_layers=2,
            num_attention_heads=8,
            num_key_value_heads=2,
            model_prefix="model.",
        )
        stmts = MoEAOAConfigGenerator._get_dense_layer_statements(params)
        self.assertIn(
            "model.layers.0.input_layernorm.weight -> "
            "model.layers.2.input_layernorm.weight",
            stmts,
        )

    def test_mtp_layers_gated_and_suffixed(self):
        no_mtp = MoEAOAConfigParams(num_nextn_predict_layers=0)
        self.assertEqual(
            MoEAOAConfigGenerator._get_mtp_layer_statements(no_mtp), []
        )
        params = MoEAOAConfigParams(
            num_hidden_layers=4, num_nextn_predict_layers=2
        )
        stmts = MoEAOAConfigGenerator._get_mtp_layer_statements(params)
        joined = "\n".join(stmts)
        # MTP layers occupy indices 4 and 5.
        self.assertIn("model.layers.4.eh_proj.weight", joined)
        self.assertIn("model.layers.5.eh_proj.weight", joined)
        self.assertIn("enorm.weight", joined)
        self.assertIn("hnorm.weight", joined)

    def test_moe_layer_mtp_gets_transformer_layer_suffix(self):
        params = MoEAOAConfigParams(
            num_hidden_layers=1,
            num_nextn_predict_layers=1,
            first_k_dense_replace=0,
            num_attention_heads=8,
            num_key_value_heads=2,
            model_prefix="model.",
        )
        stmts = MoEAOAConfigGenerator._get_moe_layer_statements(params)
        joined = "\n".join(stmts)
        # Layer index 1 is an MTP layer -> target prefix carries .transformer_layer.
        self.assertIn(
            "model.layers.1.transformer_layer.input_layernorm.weight", joined
        )
        # Plain hidden layer 0 has no such suffix.
        self.assertIn(
            "model.layers.0.input_layernorm.weight -> "
            "model.layers.0.input_layernorm.weight",
            joined,
        )


class TestBuildAndEntryPoint(unittest.TestCase):
    """_build_aoa_config assembly + gen_aoa_config real entry propagation."""

    def test_extra_statements_appended_at_end(self):
        params = MoEAOAConfigParams(
            num_hidden_layers=1,
            num_attention_heads=8,
            num_key_value_heads=2,
            num_experts=2,
            extra_statements=["custom.a -> custom.a", "custom.b -> custom.b"],
        )
        result = MoEAOAConfigGenerator._build_aoa_config(params)
        stmts = result["aoa_statements"]
        self.assertEqual(
            stmts[-2:], ["custom.a -> custom.a", "custom.b -> custom.b"]
        )

    def test_gen_aoa_config_propagates_config_through_entry(self):
        # Real entry: alias (n_routed_experts), tie flag, head counts and
        # fusion must all show up correctly in the assembled statements.
        config = _Config(
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=2,
            n_routed_experts=4,
            first_k_dense_replace=1,
            tie_word_embeddings=True,
            moe_expert_fusion=True,
        )
        stmts = MoEAOAConfigGenerator.gen_aoa_config(config)["aoa_statements"]

        self.assertIn(
            "model.embed_tokens.weight -> model.lm_head.weight", stmts
        )
        self.assertTrue(
            any("num_heads=8, num_key_value_groups=2" in s for s in stmts)
        )
        gemm = [s for s in stmts if "grouped_gemm_experts.weight1" in s]
        self.assertTrue(gemm)
        self.assertEqual(len(gemm[0].split(" -> ")[0].split(",")), 4)


class TestInverseConfig(unittest.TestCase):
    """Inverse (PaddleFleet -> HuggingFace) config generation behaviors."""

    def test_inverse_tied_drops_lm_head_source(self):
        tied = MoEAOAConfigParams(tie_word_embeddings=True)
        untied = MoEAOAConfigParams(tie_word_embeddings=False)
        tied_stmts = MoEAOAConfigGenerator._get_inv_basic_weight_statements(
            tied
        )
        untied_stmts = MoEAOAConfigGenerator._get_inv_basic_weight_statements(
            untied
        )
        self.assertEqual(tied_stmts[-1], "model.lm_head.weight -> _")
        self.assertEqual(
            untied_stmts[-1], "model.lm_head.weight -> lm_head.weight"
        )

    def test_inverse_standard_attention_unfuses_qkv(self):
        params = MoEAOAConfigParams(
            num_attention_heads=8,
            num_key_value_heads=2,
            attention_bias=False,
        )
        stmts = MoEAOAConfigGenerator._get_inv_standard_attention_statements(
            params, "model.layers.0", "model.layers.0"
        )
        # qkv_proj un-fuse row plus per-projection transpose rows.
        self.assertIn(
            "model.layers.0.self_attn.qkv_proj.weight -> "
            "model.layers.0.self_attn.q_proj.weight, "
            "model.layers.0.self_attn.k_proj.weight, "
            "model.layers.0.self_attn.v_proj.weight, fused_qkv, "
            "num_heads=8, num_key_value_groups=2",
            stmts,
        )
        for x in ("q", "k", "v"):
            self.assertIn(
                f"model.layers.0.self_attn.{x}_proj.weight^T -> "
                f"model.layers.0.self_attn.{x}_proj.weight",
                stmts,
            )

    def test_inverse_routed_experts_sonic_vs_non_sonic(self):
        non_sonic = MoEAOAConfigParams(using_sonic_moe=False)
        sonic = MoEAOAConfigParams(using_sonic_moe=True)
        ns = MoEAOAConfigGenerator._get_inv_routed_expert_statements(
            non_sonic, "model.layers.0", "model.layers.0"
        )
        so = MoEAOAConfigGenerator._get_inv_routed_expert_statements(
            sonic, "model.layers.0", "model.layers.0"
        )
        # Non-sonic un-fuses on axis=1 and adds three transpose-back rows.
        self.assertTrue(ns[0].endswith("axis=1"))
        self.assertEqual(len(ns), 4)
        self.assertTrue(any("^T" in s for s in ns[1:]))
        # Sonic un-fuses on axis=0 and skips the transpose-back rows.
        self.assertTrue(so[0].endswith("axis=0"))
        self.assertEqual(len(so), 1)


if __name__ == "__main__":
    unittest.main()
