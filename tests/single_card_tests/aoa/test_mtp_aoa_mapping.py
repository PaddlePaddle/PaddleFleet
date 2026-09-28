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

from paddlefleet.transformers.aoa_config_base import (
    MoEAOAConfigGenerator,
    MoEAOAConfigParams,
)


def _params(**overrides):
    values = {
        "num_hidden_layers": 4,
        "num_head_empty_layers": 1,
        "num_nextn_predict_layers": 2,
    }
    values.update(overrides)
    return MoEAOAConfigParams(**values)


class TestMtpAoaMapping(unittest.TestCase):
    def test_magic_send_copies_the_embedding_table_to_each_mtp_layer(self):
        statements = MoEAOAConfigGenerator._get_basic_weight_statements(
            _params(enable_mtp_magic_send=True)
        )

        self.assertIn(
            "model.embed_tokens.weight -> model.layers.5.mtp_embed.weight",
            statements,
        )
        self.assertIn(
            "model.embed_tokens.weight -> model.layers.6.mtp_embed.weight",
            statements,
        )

    def test_non_magic_send_does_not_claim_an_mtp_embedding(self):
        forward = MoEAOAConfigGenerator._get_basic_weight_statements(_params())
        inverse = MoEAOAConfigGenerator._get_inv_basic_weight_statements(
            _params()
        )

        self.assertEqual([item for item in forward if "mtp_embed" in item], [])
        self.assertEqual([item for item in inverse if "mtp_embed" in item], [])

    def test_inverse_magic_send_drops_each_mtp_embedding(self):
        statements = MoEAOAConfigGenerator._get_inv_basic_weight_statements(
            _params(enable_mtp_magic_send=True)
        )

        self.assertIn("model.layers.5.mtp_embed.weight -> _", statements)
        self.assertIn("model.layers.6.mtp_embed.weight -> _", statements)

    def test_provider_expert_alias_and_magic_send_are_extracted(self):
        class Config:
            num_hidden_layers = 1
            num_attention_heads = 2
            num_key_value_heads = 2
            n_routed_experts = None
            num_experts = 8
            enable_mtp_magic_send = True

        params = MoEAOAConfigGenerator._extract_params(Config())

        self.assertEqual(params.num_experts, 8)
        self.assertTrue(params.enable_mtp_magic_send)


if __name__ == "__main__":
    unittest.main()
