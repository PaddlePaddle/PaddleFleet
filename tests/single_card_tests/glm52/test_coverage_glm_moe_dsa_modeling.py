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
"""Single card tests for ``paddlefleet.transformers.glm_moe_dsa.modeling``.

These tests exercise the top-level imports, the ``__all__`` export list,
the AOA generators under both DSA field spellings, and the
``_gen_inv_aoa_config`` expert-ID expansion logic. None of them need a
distributed group or a real checkpoint.
"""

import logging
import types
import unittest
from unittest.mock import patch

from paddlefleet.transformers.aoa_config_base import MoEAOAConfigGenerator
from paddlefleet.transformers.glm_moe_dsa import modeling


class ModelingExportsTests(unittest.TestCase):
    def test_all_contains_expected_classes(self):
        self.assertIn("GlmMoeDsaForCausalLM", modeling.__all__)
        self.assertIn("GlmMoeDsaForCausalLMPipe", modeling.__all__)
        # PreTrainedModel and Provider are internal building blocks.
        self.assertNotIn("GlmMoeDsaPreTrainedModel", modeling.__all__)
        self.assertNotIn("GlmMoeDsaModelProvider", modeling.__all__)

    def test_module_logger_is_named_after_module(self):
        self.assertIsInstance(modeling.logger, logging.Logger)
        self.assertEqual(
            modeling.logger.name,
            "paddlefleet.transformers.glm_moe_dsa.modeling",
        )


def _provider_config(**kwargs):
    """Minimal provider-shaped config for AOA statement generation."""
    values = {
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "n_routed_experts": 3,
        "first_k_dense_replace": 1,
        "multi_latent_attention": True,
        "use_qk_norm": True,
        "dsa_index_n_heads": 2,
        "dsa_indexer_types": ["full", "shared"],
    }
    values.update(kwargs)
    return types.SimpleNamespace(**values)


def _hf_config(**kwargs):
    """``_provider_config`` with the HF config.json spelling of the DSA fields."""
    config = _provider_config(**kwargs)
    config.index_n_heads = vars(config).pop("dsa_index_n_heads")
    config.indexer_types = vars(config).pop("dsa_indexer_types")
    return config


class GenAoaConfigTests(unittest.TestCase):
    def test_forward_statements_are_well_formed(self):
        aoa = modeling.GlmMoeDsaPreTrainedModel._gen_aoa_config(
            _provider_config()
        )
        self.assertEqual(list(aoa), ["aoa_statements"])
        statements = aoa["aoa_statements"]
        self.assertTrue(statements)
        for statement in statements:
            self.assertIsInstance(statement, str)
            self.assertIn("->", statement)
        self.assertIn(
            "model.embed_tokens.weight -> model.embedding.embed_tokens.weight",
            statements,
        )

    def test_forward_direction_keeps_expert_wildcards(self):
        # Only the inverse direction needs concrete expert IDs: the lexer
        # expands forward wildcards from the official input keys.
        aoa = modeling.GlmMoeDsaPreTrainedModel._gen_aoa_config(
            _provider_config()
        )
        self.assertTrue(
            any("$EXPERT_ID" in s for s in aoa["aoa_statements"]),
        )


class DsaIndexerSpellingTests(unittest.TestCase):
    # The trainer loads and exports through ``model.config``, the provider.
    GENERATORS = (
        modeling.GlmMoeDsaPreTrainedModel._gen_aoa_config,
        modeling.GlmMoeDsaPreTrainedModel._gen_inv_aoa_config,
    )

    def test_provider_spelling_maps_the_same_indexers_as_hf(self):
        for generate in self.GENERATORS:
            statements = generate(_provider_config())["aoa_statements"]

            self.assertEqual(
                statements, generate(_hf_config())["aoa_statements"]
            )
            self.assertTrue(
                any(
                    "model.layers.0.self_attn.indexer.wq_b.weight" in s
                    for s in statements
                )
            )
            # Layer 1 is ``shared`` and owns no indexer weights.
            self.assertFalse(
                any(
                    "model.layers.1.self_attn.indexer." in s for s in statements
                )
            )

    def test_zero_index_heads_maps_no_indexer(self):
        for generate in self.GENERATORS:
            statements = generate(_provider_config(dsa_index_n_heads=0))[
                "aoa_statements"
            ]

            self.assertFalse(any(".indexer." in s for s in statements))


class GenInvAoaConfigTests(unittest.TestCase):
    def test_expert_wildcards_expand_once_per_routed_expert(self):
        hf_spelled = _provider_config(
            index_n_heads=2, indexer_types=["full", "shared"]
        )
        base = MoEAOAConfigGenerator.gen_inv_aoa_config(hf_spelled)
        templates = [s for s in base["aoa_statements"] if "$EXPERT_ID" in s]
        self.assertTrue(templates)

        expanded = modeling.GlmMoeDsaPreTrainedModel._gen_inv_aoa_config(
            _provider_config()
        )["aoa_statements"]

        self.assertFalse(any("$EXPERT_ID" in s for s in expanded))
        self.assertEqual(
            len(expanded),
            len(base["aoa_statements"]) + len(templates) * (3 - 1),
        )
        for template in templates:
            for expert_id in range(3):
                self.assertIn(
                    template.replace("$EXPERT_ID", str(expert_id)), expanded
                )
        # Statements without expert IDs keep their original relative order.
        self.assertEqual(
            [s for s in expanded if ".experts." not in s],
            [s for s in base["aoa_statements"] if ".experts." not in s],
        )

    def test_num_experts_is_the_fallback_expert_count(self):
        config = types.SimpleNamespace(
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_experts=2,
            first_k_dense_replace=1,
            multi_latent_attention=True,
        )
        expanded = modeling.GlmMoeDsaPreTrainedModel._gen_inv_aoa_config(
            config
        )["aoa_statements"]
        self.assertTrue(any(".experts.0." in s for s in expanded))
        self.assertTrue(any(".experts.1." in s for s in expanded))
        self.assertFalse(any(".experts.2." in s for s in expanded))

    def test_zero_experts_drops_the_wildcard_statements(self):
        expanded = modeling.GlmMoeDsaPreTrainedModel._gen_inv_aoa_config(
            _provider_config(n_routed_experts=0)
        )["aoa_statements"]
        self.assertFalse(any("$EXPERT_ID" in s for s in expanded))
        self.assertFalse(any(".mlp.experts." in s for s in expanded))
        # Non-expert rules (norm, embeddings, MLA, indexer) still survive.
        self.assertTrue(any("indexer" in s for s in expanded))


class _FakeModel:
    """Stands in for the GPTModel that the provider would build."""


class _RecordingProvider:
    """Records the config and loss_fn the entrypoint hands over."""

    last = None

    def __init__(self, config):
        self.config = config
        self.loss_fn = "unset"
        _RecordingProvider.last = self

    def provide(self, loss_fn=None):
        self.loss_fn = loss_fn
        return _FakeModel()


def _fake_from_config(config, **kwargs):
    return _RecordingProvider(config)


def _parallel_config(**kwargs):
    """Config with unset (0 / negative) parallel degrees."""
    values = {
        "tensor_model_parallel_size": 0,
        "context_parallel_size": -3,
        "pipeline_model_parallel_size": 0,
        "virtual_pipeline_model_parallel_size": 0,
        "expert_model_parallel_size": 0,
    }
    values.update(kwargs)
    return types.SimpleNamespace(**values)


class CausalLmEntrypointTests(unittest.TestCase):
    def setUp(self):
        _RecordingProvider.last = None
        patcher = patch.object(
            modeling.GlmMoeDsaModelProvider,
            "from_config",
            _fake_from_config,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_parallel_degrees_are_clamped_and_dsa_switches_forced(self):
        config = _parallel_config()
        model = modeling.GlmMoeDsaForCausalLM(config)

        self.assertIsInstance(model, _FakeModel)
        self.assertEqual(config.tensor_model_parallel_size, 1)
        self.assertEqual(config.context_parallel_size, 1)
        self.assertEqual(config.pipeline_model_parallel_size, 1)
        self.assertEqual(config.virtual_pipeline_model_parallel_size, 1)
        self.assertEqual(config.expert_model_parallel_size, 1)
        self.assertTrue(config.fuse_rms_norm)
        self.assertTrue(config.multi_latent_attention)

    def test_larger_degrees_are_left_alone(self):
        config = _parallel_config(
            tensor_model_parallel_size=4,
            virtual_pipeline_model_parallel_size=2,
        )
        modeling.GlmMoeDsaForCausalLM(config)
        self.assertEqual(config.tensor_model_parallel_size, 4)
        self.assertEqual(config.virtual_pipeline_model_parallel_size, 2)

    def test_provider_builds_the_model_and_hooks_are_attached(self):
        config = _parallel_config()
        model = modeling.GlmMoeDsaForCausalLM(config)
        provider = _RecordingProvider.last

        self.assertIs(provider.config, config)
        self.assertIsNone(provider.loss_fn)
        self.assertIs(model.config_to_save, config)
        self.assertIs(model.is_fleet, True)
        self.assertIs(
            model._gen_aoa_config.__func__,
            modeling.GlmMoeDsaPreTrainedModel._gen_aoa_config.__func__,
        )
        self.assertIs(
            model._gen_inv_aoa_config.__func__,
            modeling.GlmMoeDsaPreTrainedModel._gen_inv_aoa_config.__func__,
        )
        self.assertIs(
            model.build_muon_param_info_map.__func__,
            modeling.GlmMoeDsaPreTrainedModel.build_muon_param_info_map.__func__,
        )

    def test_dpo_config_arms_the_criterion_layer(self):
        config = _parallel_config(dpo_config={"beta": 0.1})
        with patch.object(
            modeling, "CriterionLayerPipe", return_value="criterion"
        ) as criterion:
            modeling.GlmMoeDsaForCausalLM(config)
        criterion.assert_called_once_with(config, use_infohub=True)
        self.assertEqual(_RecordingProvider.last.loss_fn, "criterion")

    def test_plain_entrypoint_does_not_stamp_architectures(self):
        config = _parallel_config()
        modeling.GlmMoeDsaForCausalLM(config)
        self.assertFalse(hasattr(config, "architectures"))


class CausalLmPipeEntrypointTests(unittest.TestCase):
    def setUp(self):
        _RecordingProvider.last = None
        patcher = patch.object(
            modeling.GlmMoeDsaModelProvider,
            "from_config",
            _fake_from_config,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_pipe_shares_the_provider_and_switches(self):
        config = _parallel_config()
        model = modeling.GlmMoeDsaForCausalLMPipe(config)

        self.assertIsInstance(model, _FakeModel)
        self.assertIs(_RecordingProvider.last.config, config)
        self.assertEqual(config.tensor_model_parallel_size, 1)
        self.assertEqual(config.expert_model_parallel_size, 1)
        self.assertTrue(config.fuse_rms_norm)
        self.assertTrue(config.multi_latent_attention)
        self.assertIs(model.config_to_save, config)
        self.assertIs(model.is_fleet, True)

    def test_pipe_stamps_the_non_pipe_architecture_name(self):
        config = _parallel_config()
        modeling.GlmMoeDsaForCausalLMPipe(config)
        self.assertEqual(config.architectures, ["GlmMoeDsaForCausalLM"])

    def test_pipe_keeps_an_existing_architecture_list(self):
        config = _parallel_config(architectures=["SomethingElse"])
        modeling.GlmMoeDsaForCausalLMPipe(config)
        self.assertEqual(config.architectures, ["SomethingElse"])


if __name__ == "__main__":
    unittest.main()
