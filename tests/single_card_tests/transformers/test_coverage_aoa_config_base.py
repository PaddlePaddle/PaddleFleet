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

"""Single-card coverage for the shared MoE AOA config generator.

Focus areas: the last-stage MTP embedding copy (and its inverse drop), reading
the DSA fields under either the HF or the provider spelling, and the DSA
indexer branches that decide whether a layer owns a full indexer, shares one,
or is rejected outright.
"""

import unittest

from paddlefleet.transformers.aoa_config_base import (
    MoEAOAConfigGenerator,
    MoEAOAConfigParams,
)

_PREFIX = "model.layers.0"
_OFFSET = "model.layers.0"


class _StubConfig:
    """Minimal model config understood by ``_extract_params``."""

    def __init__(self, **overrides):
        self.num_hidden_layers = 4
        self.num_attention_heads = 8
        self.num_key_value_heads = 2
        self.num_experts = 4
        for name, value in overrides.items():
            setattr(self, name, value)


def _mtp_params(**overrides):
    defaults = {
        "num_hidden_layers": 4,
        "num_head_empty_layers": 1,
        "num_nextn_predict_layers": 2,
    }
    defaults.update(overrides)
    return MoEAOAConfigParams(**defaults)


def _mla_params(**overrides):
    defaults = {
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "multi_latent_attention": True,
        "index_n_heads": 8,
    }
    defaults.update(overrides)
    return MoEAOAConfigParams(**defaults)


class TestBasicWeightMtpEmbedCopy(unittest.TestCase):
    def test_magic_send_copies_the_table_into_every_mtp_layer(self):
        params = _mtp_params(enable_mtp_magic_send=True)

        statements = MoEAOAConfigGenerator._get_basic_weight_statements(params)

        # mtp index = num_hidden_layers + num_head_empty_layers + mtp_i
        self.assertEqual(
            statements,
            [
                "model.norm.weight -> model.norm.weight",
                "model.embed_tokens.weight -> "
                "model.embedding.embed_tokens.weight",
                "model.embed_tokens.weight -> model.layers.5.mtp_embed.weight",
                "model.embed_tokens.weight -> model.layers.6.mtp_embed.weight",
                "lm_head.weight -> model.lm_head.weight",
            ],
        )

    def test_no_copy_without_magic_send(self):
        params = _mtp_params()

        statements = MoEAOAConfigGenerator._get_basic_weight_statements(params)

        self.assertEqual([s for s in statements if "mtp_embed" in s], [])
        self.assertEqual(len(statements), 3)

    def test_no_copy_when_there_are_no_mtp_layers(self):
        params = _mtp_params(
            num_nextn_predict_layers=0, enable_mtp_magic_send=True
        )

        statements = MoEAOAConfigGenerator._get_basic_weight_statements(params)

        self.assertEqual([s for s in statements if "mtp_embed" in s], [])

    def test_copy_index_includes_the_head_empty_layer_offset(self):
        params = _mtp_params(
            num_hidden_layers=2,
            num_head_empty_layers=3,
            num_nextn_predict_layers=1,
            enable_mtp_magic_send=True,
            tie_word_embeddings=True,
        )

        statements = MoEAOAConfigGenerator._get_basic_weight_statements(params)

        self.assertIn(
            "model.embed_tokens.weight -> model.layers.5.mtp_embed.weight",
            statements,
        )
        self.assertIn(
            "model.embed_tokens.weight -> model.lm_head.weight", statements
        )


class TestExtractDsaParams(unittest.TestCase):
    def test_provider_spelling_is_read(self):
        indexer_types = ["full", "shared", "full", "shared"]

        params = MoEAOAConfigGenerator._extract_params(
            _StubConfig(dsa_index_n_heads=8, dsa_indexer_types=indexer_types)
        )

        self.assertEqual(params.index_n_heads, 8)
        self.assertEqual(params.indexer_types, indexer_types)

    def test_unset_provider_index_heads_means_no_indexer(self):
        # TransformerConfig.dsa_index_n_heads defaults to None.
        params = MoEAOAConfigGenerator._extract_params(
            _StubConfig(dsa_index_n_heads=None)
        )

        self.assertEqual(params.index_n_heads, 0)


class TestIndexerTypeForLayer(unittest.TestCase):
    def test_mtp_layers_always_own_a_full_indexer(self):
        params = _mla_params(indexer_types=["full", "shared"])

        self.assertEqual(
            MoEAOAConfigGenerator._indexer_type_for_layer(params, 2), "full"
        )

    def test_decoder_layers_read_their_own_entry(self):
        params = _mla_params(indexer_types=["full", "shared"])

        self.assertEqual(
            MoEAOAConfigGenerator._indexer_type_for_layer(params, 0), "full"
        )
        self.assertEqual(
            MoEAOAConfigGenerator._indexer_type_for_layer(params, 1), "shared"
        )

    def test_missing_indexer_types_default_to_full(self):
        params = _mla_params(indexer_types=None)

        self.assertEqual(
            MoEAOAConfigGenerator._indexer_type_for_layer(params, 0), "full"
        )

    def test_short_indexer_types_default_to_full(self):
        params = _mla_params(num_hidden_layers=4, indexer_types=["shared"])

        self.assertEqual(
            MoEAOAConfigGenerator._indexer_type_for_layer(params, 1), "full"
        )

    def test_entry_is_used_when_layer_count_is_unknown(self):
        params = _mla_params(num_hidden_layers=0, indexer_types=["shared"])

        self.assertEqual(
            MoEAOAConfigGenerator._indexer_type_for_layer(params, 0), "shared"
        )


class TestMlaIndexerStatements(unittest.TestCase):
    def test_full_indexer_adds_projections_and_norms(self):
        params = _mla_params(indexer_types=["full", "shared"])

        statements = MoEAOAConfigGenerator._get_mla_attention_statements(
            params, 0, _PREFIX, _OFFSET
        )

        self.assertEqual(len(statements), 9)
        for name in ("wq_b", "wk", "weights_proj"):
            self.assertIn(
                f"{_PREFIX}.self_attn.indexer.{name}.weight^T -> "
                f"{_OFFSET}.self_attn.core_attention.indexer.{name}.weight",
                statements,
            )
        self.assertEqual(
            len([s for s in statements if "indexer.k_norm" in s]), 2
        )

    def test_shared_indexer_returns_only_the_projection_statements(self):
        params = _mla_params(indexer_types=["full", "shared"])

        statements = MoEAOAConfigGenerator._get_mla_attention_statements(
            params, 1, _PREFIX, _OFFSET
        )

        self.assertEqual(len(statements), 4)
        self.assertEqual([s for s in statements if "indexer" in s], [])

    def test_unknown_indexer_type_is_rejected(self):
        params = _mla_params(indexer_types=["sparse", "full"])

        with self.assertRaises(ValueError) as caught:
            MoEAOAConfigGenerator._get_mla_attention_statements(
                params, 0, _PREFIX, _OFFSET
            )

        message = str(caught.exception)
        self.assertIn("Unsupported indexer type 'sparse'", message)
        self.assertIn("for layer 0", message)

    def test_indexer_is_skipped_without_index_heads(self):
        params = _mla_params(index_n_heads=0, indexer_types=["sparse"])

        statements = MoEAOAConfigGenerator._get_mla_attention_statements(
            params, 0, _PREFIX, _OFFSET
        )

        self.assertEqual(len(statements), 4)


class TestInverseMlaIndexerStatements(unittest.TestCase):
    def test_full_indexer_maps_paddle_names_back(self):
        params = _mla_params(indexer_types=["full", "shared"])

        statements = MoEAOAConfigGenerator._get_inv_mla_attention_statements(
            params, 0, _PREFIX, _OFFSET
        )

        self.assertEqual(len(statements), 9)
        for name in ("wq_b", "wk", "weights_proj"):
            self.assertIn(
                f"{_OFFSET}.self_attn.core_attention.indexer.{name}"
                f".weight^T -> {_PREFIX}.self_attn.indexer.{name}.weight",
                statements,
            )
        self.assertIn(
            f"{_OFFSET}.self_attn.core_attention.indexer.k_norm.bias -> "
            f"{_PREFIX}.self_attn.indexer.k_norm.bias",
            statements,
        )

    def test_shared_indexer_returns_only_the_projection_statements(self):
        params = _mla_params(indexer_types=["full", "shared"])

        statements = MoEAOAConfigGenerator._get_inv_mla_attention_statements(
            params, 1, _PREFIX, _OFFSET
        )

        self.assertEqual(len(statements), 4)
        self.assertEqual([s for s in statements if "indexer" in s], [])

    def test_unknown_indexer_type_is_rejected(self):
        params = _mla_params(indexer_types=["bogus", "full"])

        with self.assertRaises(ValueError) as caught:
            MoEAOAConfigGenerator._get_inv_mla_attention_statements(
                params, 0, _PREFIX, _OFFSET
            )

        self.assertIn("Unsupported indexer type 'bogus'", str(caught.exception))

    def test_attention_dispatch_selects_the_mla_branch(self):
        params = _mla_params(indexer_types=["shared", "shared"])

        statements = MoEAOAConfigGenerator._get_inv_attention_statements(
            params, 1, _PREFIX, _OFFSET
        )

        self.assertEqual(
            statements,
            MoEAOAConfigGenerator._get_inv_mla_attention_statements(
                params, 1, _PREFIX, _OFFSET
            ),
        )


class TestInverseBasicWeights(unittest.TestCase):
    def test_magic_send_drops_every_mtp_embedding(self):
        params = _mtp_params(enable_mtp_magic_send=True)

        statements = MoEAOAConfigGenerator._get_inv_basic_weight_statements(
            params
        )

        self.assertEqual(
            statements,
            [
                "model.norm.weight -> model.norm.weight",
                "model.embedding.embed_tokens.weight -> "
                "model.embed_tokens.weight",
                "model.lm_head.weight -> lm_head.weight",
                "model.layers.5.mtp_embed.weight -> _",
                "model.layers.6.mtp_embed.weight -> _",
            ],
        )

    def test_nothing_is_dropped_without_magic_send(self):
        params = _mtp_params(tie_word_embeddings=True)

        statements = MoEAOAConfigGenerator._get_inv_basic_weight_statements(
            params
        )

        self.assertEqual([s for s in statements if "mtp_embed" in s], [])
        self.assertIn("model.lm_head.weight -> _", statements)


class TestGeneratedConfigEndToEnd(unittest.TestCase):
    def test_forward_config_carries_the_mtp_embedding_copy(self):
        config = _StubConfig(
            num_nextn_predict_layers=1, enable_mtp_magic_send=True
        )

        statements = MoEAOAConfigGenerator.gen_aoa_config(config)[
            "aoa_statements"
        ]

        self.assertIn(
            "model.embed_tokens.weight -> model.layers.4.mtp_embed.weight",
            statements,
        )

    def test_inverse_config_drops_the_mtp_embedding(self):
        config = _StubConfig(
            num_nextn_predict_layers=1, enable_mtp_magic_send=True
        )

        statements = MoEAOAConfigGenerator.gen_inv_aoa_config(config)[
            "aoa_statements"
        ]

        self.assertIn("model.layers.4.mtp_embed.weight -> _", statements)

    def test_accuracy_compatible_mode_alone_maps_no_mtp_embedding(self):
        # ``mtp_embed`` is built only under magic send; mapping it otherwise
        # names a parameter the model does not have.
        config = _StubConfig(
            num_nextn_predict_layers=1, use_accuracy_compatible="hf"
        )

        for generate in (
            MoEAOAConfigGenerator.gen_aoa_config,
            MoEAOAConfigGenerator.gen_inv_aoa_config,
        ):
            statements = generate(config)["aoa_statements"]
            self.assertEqual([s for s in statements if "mtp_embed" in s], [])


if __name__ == "__main__":
    unittest.main()
