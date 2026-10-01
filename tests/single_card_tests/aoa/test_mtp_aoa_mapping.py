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

import unittest
from types import SimpleNamespace

from paddlefleet.transformers.aoa_config_base import MoEAOAConfigGenerator


def _config(**overrides):
    values = {
        "num_hidden_layers": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "num_empty_layers_add_in_head": 1,
        "num_nextn_predict_layers": 2,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TestMtpAoaMapping(unittest.TestCase):
    def test_magic_send_copies_the_embedding_table_to_each_mtp_layer(self):
        statements = MoEAOAConfigGenerator.gen_aoa_config(
            _config(enable_mtp_magic_send=True)
        )["aoa_statements"]

        self.assertEqual(
            [item for item in statements if "mtp_embed" in item],
            [
                "model.embed_tokens.weight -> model.layers.5.mtp_embed.weight",
                "model.embed_tokens.weight -> model.layers.6.mtp_embed.weight",
            ],
        )

    def test_non_magic_send_does_not_claim_an_mtp_embedding(self):
        for config in (
            _config(),
            _config(enable_mtp_magic_send=True, num_nextn_predict_layers=0),
        ):
            with self.subTest(config=config):
                for generate in (
                    MoEAOAConfigGenerator.gen_aoa_config,
                    MoEAOAConfigGenerator.gen_inv_aoa_config,
                ):
                    statements = generate(config)["aoa_statements"]
                    self.assertEqual(
                        [item for item in statements if "mtp_embed" in item], []
                    )

    def test_inverse_magic_send_drops_each_mtp_embedding(self):
        statements = MoEAOAConfigGenerator.gen_inv_aoa_config(
            _config(enable_mtp_magic_send=True)
        )["aoa_statements"]

        self.assertEqual(
            [item for item in statements if "mtp_embed" in item],
            [
                "model.layers.5.mtp_embed.weight -> _",
                "model.layers.6.mtp_embed.weight -> _",
            ],
        )

    def test_provider_expert_alias_and_magic_send_are_extracted(self):
        for fields, expected in (
            ({}, 0),
            ({"num_experts": None}, 0),
            ({"num_experts": 8}, 8),
            ({"n_routed_experts": None, "num_experts": 8}, 8),
            ({"n_routed_experts": 0, "num_experts": 8}, 0),
            ({"n_routed_experts": 4, "num_experts": 8}, 4),
        ):
            with self.subTest(fields=fields):
                params = MoEAOAConfigGenerator._extract_params(
                    _config(enable_mtp_magic_send=True, **fields)
                )
                self.assertEqual(params.num_experts, expected)
                self.assertTrue(params.enable_mtp_magic_send)


if __name__ == "__main__":
    unittest.main()
