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

"""Behavior tests for the ERNIE MoE pretrain configuration.

Module under test:
``paddlefleet.cli.train.ernie_pretrain.models.ernie.configuration`` -- the
``ErnieMoEConfig`` (a ``PretrainedConfig`` subclass, ``model_type == "ernie"``)
plus the two module-level registries ``ERNIE_PRETRAINED_INIT_CONFIGURATION`` and
``ERNIE_PRETRAINED_RESOURCE_FILES_MAP``. This is the "configuration and runtime
infrastructure" layer: each field value is a contract that downstream model
construction consumes, so the tests verify content/identity and the derived
logic the constructor actually computes -- not just that attributes exist.

What these tests actually verify (all expected values hand-derived by reading
the constructor signature and body, never produced by the code under test):

* Documented default field values callers rely on; a silent change flips the
  model architecture.
* Derived / forced behaviour the subclass computes rather than merely stores:
  ``moe_layer_end_index == -1`` back-fills to ``num_hidden_layers - 1``;
  ``use_recompute_attn=True`` force-disables ``use_recompute`` (flips a caller's
  ``True`` to ``False``); ``routed_scaling_factor`` is renamed to
  ``scaling_factor``; ``insert_empty_layer=None`` becomes ``[]``.
* The nested-dict merge for ``fp8_configs``: a partial override reaches the
  target leaf while its siblings and untouched top-level keys survive.
* Validation contracts raised from ``__init__`` / ``__setattr__``
  (``use_async_a2a`` requires ``use_quant_before_a2a``; a non-list
  ``insert_empty_layer`` is rejected).
* Attribute-map aliasing is functional (``n_embd`` reads back ``hidden_size``),
  proving the alias maps to the real field rather than only living in a dict.
* ``to_json_string(use_diff=False)`` round-trips real stored values.
* The two registries carry the exact documented values / object identity.

A real production defect is captured with ``expectedFailure``: the
``pp_no_recompute_layer`` type check asserts ``isinstance(insert_empty_layer,
list)`` instead of ``isinstance(pp_no_recompute_layer, list)``, so a non-list
``pp_no_recompute_layer`` is never rejected. Production code is NOT modified.

These tests run on CPU. Importing the production module pulls the ``paddlefleet``
package, which requires Paddle; when Paddle is absent the import raises
``ImportError`` and every test skips (recorded honestly, never silently passed).
"""

import json
import unittest

try:
    from paddlefleet.cli.train.ernie_pretrain.models.ernie.configuration import (
        ERNIE_PRETRAINED_INIT_CONFIGURATION,
        ERNIE_PRETRAINED_RESOURCE_FILES_MAP,
        ErnieMoEConfig,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # Paddle backend not installed in this env.
    ERNIE_PRETRAINED_INIT_CONFIGURATION = None
    ERNIE_PRETRAINED_RESOURCE_FILES_MAP = None
    ErnieMoEConfig = None
    _IMPORT_ERROR = exc


class _ConfigTestBase(unittest.TestCase):
    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(
                "paddlefleet ernie configuration import failed "
                "(Paddle dependency unavailable): {!r}".format(_IMPORT_ERROR)
            )


class TestClassLevelContract(_ConfigTestBase):
    """Class-level constants select the architecture family and aliases."""

    def test_model_type_identifies_ernie(self):
        # Registry / dispatch key, a hard string contract not a free label.
        self.assertEqual(ErnieMoEConfig.model_type, "ernie")

    def test_pretrained_init_configuration_is_the_registry_object(self):
        # The class must expose the module registry itself, not a copy, so
        # from_pretrained("ernie/tiny-random-ernie") resolves the same dict.
        self.assertIs(
            ErnieMoEConfig.pretrained_init_configuration,
            ERNIE_PRETRAINED_INIT_CONFIGURATION,
        )

    def test_attribute_map_exact_pairs(self):
        # These aliases are consumed by convert_to_legacy_config on load; a
        # wrong target field would silently drop a legacy config value.
        self.assertEqual(
            ErnieMoEConfig.attribute_map,
            {
                "n_positions": "max_position_embeddings",
                "n_embd": "hidden_size",
                "n_layer": "num_hidden_layers",
                "n_head": "num_attention_heads",
                "n_inner": "intermediate_size",
                "activation_function": "hidden_act",
                "moe_fuse_experts": "moe_expert_fusion",
            },
        )


class TestDefaults(_ConfigTestBase):
    """Constructor defaults, hand-derived from the signature."""

    def test_core_dimension_defaults(self):
        config = ErnieMoEConfig()
        self.assertEqual(config.vocab_size, 32000)
        self.assertEqual(config.hidden_size, 768)
        self.assertEqual(config.intermediate_size, 11008)
        self.assertEqual(config.max_position_embeddings, 32768)
        self.assertEqual(config.num_hidden_layers, 2)
        self.assertEqual(config.num_attention_heads, 2)
        self.assertEqual(config.rms_norm_eps, 1e-6)
        self.assertEqual(config.initializer_range, 0.02)

    def test_boolean_feature_defaults(self):
        # Identity checks so a truthy non-bool could not masquerade as the
        # documented boolean default.
        config = ErnieMoEConfig()
        self.assertIs(config.use_cache, False)
        self.assertIs(config.use_recompute, False)
        self.assertIs(config.use_recompute_attn, False)
        self.assertIs(config.use_flash_attn, True)
        self.assertIs(config.use_mem_eff_attn, False)
        self.assertIs(config.use_rmsnorm, True)
        self.assertIs(config.fuse_rms_norm, True)
        self.assertIs(config.fuse_ln, False)
        self.assertIs(config.use_bias, False)

    def test_special_token_defaults_reach_base_attributes(self):
        # These are forwarded into super().__init__; reading them back proves
        # the parent constructor genuinely ran and stored them.
        config = ErnieMoEConfig()
        self.assertEqual(config.pad_token_id, 0)
        self.assertEqual(config.bos_token_id, 1)
        self.assertEqual(config.eos_token_id, 2)

    def test_tie_word_embeddings_overrides_base_default(self):
        # ErnieMoEConfig injects tie_word_embeddings=False into kwargs when the
        # caller omits it, overriding the base PretrainedConfig behaviour.
        self.assertIs(ErnieMoEConfig().tie_word_embeddings, False)

    def test_moe_layer_end_index_default_backfills(self):
        # Default num_hidden_layers=2 and moe_layer_end_index=-1 sentinel ->
        # num_hidden_layers - 1 == 1.
        self.assertEqual(ErnieMoEConfig().moe_layer_end_index, 1)


class TestCustomValues(_ConfigTestBase):
    """Caller-supplied values reach the stored attributes."""

    def test_custom_dimensions_override_defaults(self):
        config = ErnieMoEConfig(
            vocab_size=65536,
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
        )
        self.assertEqual(config.vocab_size, 65536)
        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.num_hidden_layers, 32)
        self.assertEqual(config.num_attention_heads, 32)

    def test_routed_scaling_factor_is_renamed_to_scaling_factor(self):
        # The constructor stores routed_scaling_factor under the attribute
        # ``scaling_factor``; the original name is not kept.
        config = ErnieMoEConfig(routed_scaling_factor=2.5)
        self.assertEqual(config.scaling_factor, 2.5)
        # Default is None when the caller omits it.
        self.assertIsNone(ErnieMoEConfig().scaling_factor)


class TestUseMoeProperty(_ConfigTestBase):
    """``use_moe`` is computed as ``moe_num_experts > 0``."""

    def test_zero_experts_is_not_moe(self):
        self.assertIs(ErnieMoEConfig(moe_num_experts=0).use_moe, False)

    def test_positive_experts_is_moe(self):
        self.assertIs(ErnieMoEConfig(moe_num_experts=8).use_moe, True)

    def test_moe_params_are_stored(self):
        config = ErnieMoEConfig(
            moe_num_experts=8,
            moe_layer_interval=4,
            num_experts_per_tok=2,
            router_aux_loss_coef=0.01,
            moe_k=2,
        )
        self.assertEqual(config.moe_num_experts, 8)
        self.assertEqual(config.moe_layer_interval, 4)
        self.assertEqual(config.num_experts_per_tok, 2)
        self.assertEqual(config.router_aux_loss_coef, 0.01)
        self.assertEqual(config.moe_k, 2)


class TestMoeLayerEndIndex(_ConfigTestBase):
    """The -1 sentinel back-fills to num_hidden_layers - 1; else pass-through."""

    def test_sentinel_backfills_from_num_hidden_layers(self):
        config = ErnieMoEConfig(num_hidden_layers=10, moe_layer_end_index=-1)
        self.assertEqual(config.moe_layer_end_index, 9)

    def test_explicit_value_is_kept(self):
        config = ErnieMoEConfig(num_hidden_layers=10, moe_layer_end_index=5)
        self.assertEqual(config.moe_layer_end_index, 5)


class TestRecomputeInteraction(_ConfigTestBase):
    """use_recompute_attn force-disables use_recompute inside __init__."""

    def test_recompute_attn_flips_caller_true_to_false(self):
        # Caller explicitly asks for both; the constructor must win and turn
        # use_recompute off (a passthrough would leave it True).
        config = ErnieMoEConfig(use_recompute_attn=True, use_recompute=True)
        self.assertIs(config.use_recompute, False)
        self.assertIs(config.use_recompute_attn, True)

    def test_recompute_left_enabled_without_attn(self):
        config = ErnieMoEConfig(use_recompute_attn=False, use_recompute=True)
        self.assertIs(config.use_recompute, True)


class TestFp8ConfigMerge(_ConfigTestBase):
    """update_nested_dict deep-merges caller fp8 overrides onto defaults."""

    def test_fp8_configs_full_defaults(self):
        config = ErnieMoEConfig()
        self.assertEqual(config.fp8_configs["quant_scheme"], "DelayedScaling")
        self.assertEqual(config.fp8_configs["recipe"]["format"], "hybrid")
        self.assertEqual(config.fp8_configs["recipe"]["amax_history_len"], 1024)
        self.assertIs(config.fp8_configs["recipe"]["calibrating"], True)
        # A representative layer flag defaults to True.
        self.assertIs(config.fp8_configs["layers"]["attn_fc1_linear"], True)

    def test_partial_recipe_override_merges_not_replaces(self):
        # Overriding one leaf must reach it while siblings and untouched
        # top-level keys survive the deep merge.
        config = ErnieMoEConfig(
            fp8_configs={"recipe": {"amax_history_len": 512}}
        )
        self.assertEqual(config.fp8_configs["recipe"]["amax_history_len"], 512)
        # Sibling leaf inside the same nested dict is preserved.
        self.assertEqual(config.fp8_configs["recipe"]["format"], "hybrid")
        # Untouched top-level key is preserved.
        self.assertEqual(config.fp8_configs["quant_scheme"], "DelayedScaling")

    def test_fp8_mem_configs_defaults(self):
        config = ErnieMoEConfig()
        self.assertIs(config.fp8_mem_configs["shared_expert"], False)
        self.assertIs(config.fp8_mem_configs["dequant_input"], False)
        self.assertIs(config.fp8_mem_configs["recompute_fwd_gate_up"], False)
        self.assertIs(
            config.fp8_mem_configs["offline_quant_expert_weight"], False
        )

    def test_fp8_fused_ops_configs_defaults(self):
        config = ErnieMoEConfig()
        self.assertIs(config.fp8_fused_ops_configs["stack_quant"], False)
        self.assertIs(config.fp8_fused_ops_configs["swiglu_probs_bwd"], False)
        # split_group_gemm is the one default that ships enabled.
        self.assertIs(config.fp8_fused_ops_configs["split_group_gemm"], True)


class TestValidationContracts(_ConfigTestBase):
    """Assertions raised from __init__ reject inconsistent configurations."""

    def test_async_a2a_without_quant_before_a2a_raises(self):
        with self.assertRaises(AssertionError):
            ErnieMoEConfig(use_async_a2a=True, use_quant_before_a2a=False)

    def test_async_a2a_with_quant_before_a2a_ok(self):
        config = ErnieMoEConfig(use_async_a2a=True, use_quant_before_a2a=True)
        self.assertIs(config.use_async_a2a, True)
        self.assertIs(config.use_quant_before_a2a, True)

    def test_insert_empty_layer_none_becomes_empty_list(self):
        self.assertEqual(
            ErnieMoEConfig(insert_empty_layer=None).insert_empty_layer, []
        )

    def test_insert_empty_layer_list_is_kept(self):
        self.assertEqual(
            ErnieMoEConfig(insert_empty_layer=[1, 2]).insert_empty_layer, [1, 2]
        )

    def test_insert_empty_layer_non_list_raises(self):
        with self.assertRaises(AssertionError):
            ErnieMoEConfig(insert_empty_layer="invalid")

    @unittest.expectedFailure
    def test_pp_no_recompute_layer_non_list_should_raise(self):
        # PRODUCTION BUG (configuration.py, the pp_no_recompute_layer guard):
        # the check reads ``assert isinstance(insert_empty_layer, list)`` but
        # the message says "pp_no_recompute_layer should be a list". Since
        # insert_empty_layer has already been coerced to a list a few lines
        # above, the assertion is always satisfied and a non-list
        # pp_no_recompute_layer is never rejected. The CORRECT behaviour is to
        # raise AssertionError here; this test asserts the correct behaviour and
        # is marked expectedFailure until the production check is fixed to
        # validate pp_no_recompute_layer itself. Production code is not touched.
        with self.assertRaises(AssertionError):
            ErnieMoEConfig(pp_no_recompute_layer="not-a-list")


class TestAttributeAliasing(_ConfigTestBase):
    """attribute_map is functionally wired via __getattr__/__setattr__."""

    def test_alias_reads_back_the_real_field(self):
        # n_embd is an alias of hidden_size; a custom hidden_size must be
        # observable through the alias, proving the map targets the real field.
        config = ErnieMoEConfig(hidden_size=1234)
        self.assertEqual(config.n_embd, 1234)
        self.assertEqual(config.n_layer, config.num_hidden_layers)
        self.assertEqual(config.n_head, config.num_attention_heads)


class TestJsonSerialization(_ConfigTestBase):
    """to_json_string emits real stored values as parseable JSON."""

    def test_full_json_roundtrips_values(self):
        config = ErnieMoEConfig(hidden_size=512, vocab_size=1000)
        parsed = json.loads(config.to_json_string(use_diff=False))
        self.assertIsInstance(parsed, dict)
        # Content, not mere presence: the custom values must survive.
        self.assertEqual(parsed["hidden_size"], 512)
        self.assertEqual(parsed["vocab_size"], 1000)
        self.assertEqual(parsed["model_type"], "ernie")


class TestPretrainedInitConfiguration(_ConfigTestBase):
    """The tiny-random-ernie preset carries exact documented values."""

    def test_tiny_random_ernie_exact_values(self):
        cfg = ERNIE_PRETRAINED_INIT_CONFIGURATION["ernie/tiny-random-ernie"]
        self.assertEqual(cfg["hidden_size"], 768)
        self.assertEqual(cfg["num_attention_heads"], 2)
        self.assertEqual(cfg["num_hidden_layers"], 2)
        self.assertEqual(cfg["vocab_size"], 32000)
        self.assertEqual(cfg["intermediate_size"], 11008)
        # The preset caps positions at 2048 -- distinct from the class default
        # of 32768; verifying the exact value guards that difference.
        self.assertEqual(cfg["max_position_embeddings"], 2048)
        self.assertEqual(cfg["rms_norm_eps"], 1e-06)
        self.assertEqual(cfg["model_type"], "ernie")
        self.assertIs(cfg["use_flash_attn"], True)
        self.assertIs(cfg["use_cache"], False)


class TestPretrainedResourceFilesMap(_ConfigTestBase):
    """The resource map points at the exact published weight URL."""

    def test_model_state_url_is_exact(self):
        self.assertEqual(
            ERNIE_PRETRAINED_RESOURCE_FILES_MAP["model_state"][
                "facebookresearch/tiny-random-ernie"
            ],
            "https://bj.bcebos.com/paddleformers/models/community/"
            "facebookresearch/tiny-random-ernie/model_state.pdparams",
        )


if __name__ == "__main__":
    unittest.main()
