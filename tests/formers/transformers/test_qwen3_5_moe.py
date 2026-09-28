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

"""Behavior tests for the Qwen3_5-MoE model and its configuration.

Modules under test:
  * 配置与运行基础设施 (config): Qwen3_5MoEConfig / Qwen3_5MoETextConfig /
    Qwen3_5MoEVisionConfig -- default values, MoE hyper-parameters, RoPE
    parameter derivation, and the outer-config attribute delegation that
    forwards text-config fields through __getattribute__ / __setattr__.
  * 模型层 (model): Qwen3_5MoEForConditionalGeneration -- the IS-A contract
    with Qwen3_5ForConditionalGeneration.

Environment: 无卡 / CPU-only. Every test here exercises real production
constructors with independently hand-derived expected values. Building the
actual transformer (Qwen3_5MoEForConditionalGeneration.__new__ ->
build_qwen3_5_model) needs paddle on a GPU plus distributed/parallel state,
so that path is marked skip rather than faked.
"""

import unittest

from paddlefleet.transformers.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
)
from paddlefleet.transformers.qwen3_5_moe.configuration import (
    Qwen3_5MoEConfig,
    Qwen3_5MoETextConfig,
    Qwen3_5MoEVisionConfig,
)
from paddlefleet.transformers.qwen3_5_moe.modeling import (
    Qwen3_5MoEForConditionalGeneration,
)


class TestQwen3_5MoETextConfigDefaults(unittest.TestCase):
    """Scalar defaults and precise typing of Qwen3_5MoETextConfig."""

    def test_moe_and_structural_defaults(self):
        config = Qwen3_5MoETextConfig()
        # MoE-specific hyper-parameters.
        self.assertEqual(config.num_experts, 60)
        self.assertEqual(config.num_experts_per_tok, 4)
        self.assertEqual(config.moe_intermediate_size, 1408)
        self.assertEqual(config.decoder_sparse_step, 1)
        # Backbone structural defaults.
        self.assertEqual(config.vocab_size, 151936)
        self.assertEqual(config.hidden_size, 2048)
        self.assertEqual(config.intermediate_size, 5632)
        self.assertEqual(config.num_hidden_layers, 24)
        self.assertEqual(config.num_attention_heads, 32)
        self.assertEqual(config.num_key_value_heads, 16)
        self.assertEqual(config.max_position_embeddings, 128000)
        self.assertEqual(config.rms_norm_eps, 1e-06)
        self.assertEqual(config.rope_theta, 1000000.0)
        self.assertEqual(config.attention_dropout, 0.0)
        self.assertEqual(config.hidden_act, "silu")
        self.assertIsNone(config.head_dim)

    def test_boolean_and_string_defaults_are_exact(self):
        # A truthy-string regression (e.g. use_cache="") would slip past a
        # loose assertTrue but must fail assertIs(..., True).
        config = Qwen3_5MoETextConfig()
        self.assertIs(config.use_cache, True)
        self.assertIs(config.attention_bias, False)
        self.assertEqual(config.model_type, "qwen3_vl_moe_text")
        self.assertIsInstance(config.model_type, str)

    def test_explicit_overrides_are_respected(self):
        config = Qwen3_5MoETextConfig(
            num_experts=8,
            num_experts_per_tok=2,
            hidden_size=128,
            num_hidden_layers=4,
            attention_bias=True,
        )
        self.assertEqual(config.num_experts, 8)
        self.assertEqual(config.num_experts_per_tok, 2)
        self.assertEqual(config.hidden_size, 128)
        self.assertEqual(config.num_hidden_layers, 4)
        self.assertIs(config.attention_bias, True)


class TestQwen3_5MoETextConfigDerived(unittest.TestCase):
    """Fields the constructor normalizes or derives from other inputs."""

    def test_mlp_only_layers_none_becomes_empty_list(self):
        # None must be normalized to [], and an explicit list preserved
        # verbatim (distinguishable content, not just "is a list").
        self.assertEqual(Qwen3_5MoETextConfig().mlp_only_layers, [])
        self.assertEqual(
            Qwen3_5MoETextConfig(mlp_only_layers=[1, 3]).mlp_only_layers,
            [1, 3],
        )

    def test_num_key_value_heads_none_falls_back_to_attention_heads(self):
        # Backward-compat branch: None -> num_attention_heads.
        config = Qwen3_5MoETextConfig(
            num_attention_heads=12, num_key_value_heads=None
        )
        self.assertEqual(config.num_key_value_heads, 12)
        # An explicit value must NOT be overwritten by the fallback.
        config2 = Qwen3_5MoETextConfig(
            num_attention_heads=12, num_key_value_heads=4
        )
        self.assertEqual(config2.num_key_value_heads, 4)

    def test_layer_types_default_is_full_attention_per_layer(self):
        # Independent expectation: exactly num_hidden_layers copies of
        # "full_attention". Length AND content are checked so a wrong-length
        # or wrong-label regression is caught.
        for num_layers in (2, 5, 24):
            config = Qwen3_5MoETextConfig(num_hidden_layers=num_layers)
            self.assertEqual(
                config.layer_types, ["full_attention"] * num_layers
            )

    def test_layer_types_explicit_preserved(self):
        explicit = ["full_attention", "full_attention", "full_attention"]
        config = Qwen3_5MoETextConfig(num_hidden_layers=3, layer_types=explicit)
        self.assertEqual(config.layer_types, explicit)


class TestQwen3_5MoETextConfigRope(unittest.TestCase):
    """rope_theta must reach the derived rope_parameters, and the mrope
    alias must be rewritten to the 'default' rope type."""

    def test_default_rope_parameters_populated(self):
        config = Qwen3_5MoETextConfig()
        # standardize_rope_params derives this dict; a downstream RoPE init
        # reads it, so it must be populated, not left as None.
        self.assertEqual(config.rope_parameters["rope_type"], "default")
        self.assertEqual(config.rope_parameters["rope_theta"], 1000000.0)

    def test_custom_rope_theta_flows_into_rope_parameters(self):
        config = Qwen3_5MoETextConfig(rope_theta=50000.0)
        self.assertEqual(config.rope_theta, 50000.0)
        self.assertEqual(config.rope_parameters["rope_theta"], 50000.0)
        self.assertEqual(config.rope_parameters["rope_type"], "default")

    def test_mrope_rope_type_downgraded_to_default(self):
        # The constructor rewrites a "mrope" rope_type to "default"; a plain
        # "default" is a control that must be left untouched.
        downgraded = Qwen3_5MoETextConfig(
            rope_parameters={"rope_type": "mrope"}
        )
        self.assertEqual(downgraded.rope_parameters["rope_type"], "default")

        untouched = Qwen3_5MoETextConfig(
            rope_parameters={"rope_type": "default"}
        )
        self.assertEqual(untouched.rope_parameters["rope_type"], "default")


class TestQwen3_5MoEConfigComposition(unittest.TestCase):
    """The outer Qwen3_5MoEConfig delegates text-config fields via
    __getattribute__ / __setattr__, while an exclusion list keeps a few keys
    (model_type, dtype, ...) on the outer object."""

    def test_model_type_is_not_delegated_to_text_config(self):
        # model_type is on the delegation exclusion list, so the outer config
        # reports its own class-level "qwen3_5_moe" rather than the text
        # sub-config's distinct "qwen3_vl_moe_text".
        config = Qwen3_5MoEConfig()
        self.assertEqual(config.model_type, "qwen3_5_moe")
        self.assertEqual(config.text_config.model_type, "qwen3_vl_moe_text")
        self.assertNotEqual(config.model_type, config.text_config.model_type)

    def test_text_field_is_delegated_to_text_config(self):
        # A text-only field with no counterpart on the outer object must be
        # served from text_config.
        config = Qwen3_5MoEConfig()
        self.assertEqual(config.num_experts, 60)
        self.assertEqual(config.num_experts, config.text_config.num_experts)
        self.assertEqual(config.hidden_size, 2048)
        self.assertEqual(config.hidden_size, config.text_config.hidden_size)

    def test_setattr_delegates_to_text_config(self):
        # Writing hidden_size must land on text_config: __getattribute__
        # always reads text_config first, so reading back the new value
        # proves the write was delegated (a write to the outer object would
        # be shadowed and read back as the old 2048).
        config = Qwen3_5MoEConfig()
        config.hidden_size = 999
        self.assertEqual(config.text_config.hidden_size, 999)
        self.assertEqual(config.hidden_size, 999)

    def test_outer_only_fields_stay_on_outer(self):
        config = Qwen3_5MoEConfig()
        self.assertEqual(config.image_token_id, 151655)
        self.assertEqual(config.video_token_id, 151656)
        self.assertEqual(config.vision_start_token_id, 151652)
        self.assertEqual(config.vision_end_token_id, 151653)

    def test_text_config_dict_builds_subconfig_and_delegates(self):
        config = Qwen3_5MoEConfig(
            text_config={"num_experts": 8, "hidden_size": 128}
        )
        self.assertIsInstance(config.text_config, Qwen3_5MoETextConfig)
        self.assertEqual(config.num_experts, 8)
        self.assertEqual(config.hidden_size, 128)

    def test_bare_kwargs_flow_into_default_text_config(self):
        # With text_config=None the outer config forwards **kwargs to build
        # the text config, so a MoE field passed at the top level is honored.
        config = Qwen3_5MoEConfig(num_experts=8)
        self.assertEqual(config.num_experts, 8)
        self.assertEqual(config.text_config.num_experts, 8)


class TestQwen3_5MoEVisionConfig(unittest.TestCase):
    """Vision sub-config defaults and overrides."""

    def test_vision_defaults(self):
        config = Qwen3_5MoEVisionConfig()
        self.assertEqual(config.model_type, "qwen3_5_moe")
        self.assertEqual(config.base_config_key, "vision_config")
        self.assertEqual(config.depth, 27)
        self.assertEqual(config.hidden_size, 1152)
        self.assertEqual(config.intermediate_size, 4304)
        self.assertEqual(config.num_heads, 16)
        self.assertEqual(config.patch_size, 16)
        self.assertEqual(config.spatial_merge_size, 2)
        self.assertEqual(config.temporal_patch_size, 2)
        self.assertEqual(config.out_hidden_size, 3584)
        self.assertEqual(config.normalization, "LayerNorm")

    def test_vision_overrides(self):
        config = Qwen3_5MoEVisionConfig(
            depth=4, hidden_size=64, num_heads=2, out_hidden_size=128
        )
        self.assertEqual(config.depth, 4)
        self.assertEqual(config.hidden_size, 64)
        self.assertEqual(config.num_heads, 2)
        self.assertEqual(config.out_hidden_size, 128)

    def test_config_default_vision_config_is_vision_subconfig(self):
        config = Qwen3_5MoEConfig()
        self.assertIsInstance(config.vision_config, Qwen3_5MoEVisionConfig)
        self.assertEqual(config.vision_config.depth, 27)

    def test_config_vision_dict_builds_subconfig(self):
        config = Qwen3_5MoEConfig(vision_config={"depth": 5, "num_heads": 3})
        self.assertIsInstance(config.vision_config, Qwen3_5MoEVisionConfig)
        self.assertEqual(config.vision_config.depth, 5)
        self.assertEqual(config.vision_config.num_heads, 3)


class TestQwen3_5MoEModelClass(unittest.TestCase):
    """Structural contract of the model entry class."""

    def test_is_subclass_of_qwen3_5_conditional_generation(self):
        # The MoE variant is defined as a subclass of the shared qwen3_5
        # conditional-generation class; both names resolve through the same
        # lazy package, so this is a genuine (not module-identity-fragile)
        # IS-A relationship.
        self.assertTrue(
            issubclass(
                Qwen3_5MoEForConditionalGeneration,
                Qwen3_5ForConditionalGeneration,
            )
        )
        self.assertIn(
            Qwen3_5ForConditionalGeneration,
            Qwen3_5MoEForConditionalGeneration.__mro__,
        )

    def test_moe_class_overrides_new_to_pass_have_criterion(self):
        # The subclass exists to thread have_criterion through __new__, so it
        # must define its own __new__ (not merely inherit the parent's) and
        # expose have_criterion on the signature.
        import inspect

        self.assertIn("__new__", Qwen3_5MoEForConditionalGeneration.__dict__)
        params = inspect.signature(
            Qwen3_5MoEForConditionalGeneration.__new__
        ).parameters
        self.assertIn("have_criterion", params)
        self.assertEqual(params["have_criterion"].default, True)

    @unittest.skip(
        "Instantiation calls build_qwen3_5_model, which needs paddle on GPU "
        "plus distributed/parallel state; not runnable on no-card CPU."
    )
    def test_instantiation_builds_model(self):
        config = Qwen3_5MoEConfig(
            text_config={
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 32,
                "vocab_size": 256,
            }
        )
        model = Qwen3_5MoEForConditionalGeneration(config, have_criterion=True)
        self.assertIsNotNone(model)


class TestQwen3_5MoEConfigKnownBug(unittest.TestCase):
    """Documents a confirmed defect in Qwen3_5MoEConfig.__init__.

    The docstring types both sub-configs as ``Union[PreTrainedConfig, dict]``,
    but the constructor only handles the ``dict`` and ``None`` cases
    (configuration.py lines ~299-310). When a sub-config *object* is passed,
    neither branch runs, so ``self.text_config`` / ``self.vision_config`` is
    never assigned and later attribute access raises AttributeError. The
    caller-supplied object is silently dropped.

    These tests assert the CORRECT contract (a passed object is honored) and
    are marked expectedFailure so they surface the bug without turning CI red
    and without touching production code.
    """

    @unittest.expectedFailure
    def test_text_config_object_should_be_accepted(self):
        text = Qwen3_5MoETextConfig(num_experts=7, hidden_size=128)
        config = Qwen3_5MoEConfig(text_config=text)
        self.assertEqual(config.text_config.num_experts, 7)
        self.assertEqual(config.hidden_size, 128)

    @unittest.expectedFailure
    def test_vision_config_object_should_be_accepted(self):
        vision = Qwen3_5MoEVisionConfig(depth=6)
        config = Qwen3_5MoEConfig(vision_config=vision)
        self.assertEqual(config.vision_config.depth, 6)


if __name__ == "__main__":
    unittest.main()
