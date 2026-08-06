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

import types
import unittest
from unittest import mock

from paddlefleet.models.gpt import gpt_layer_specs
from paddlefleet.transformer.transformer_config import TransformerConfig


def _make_spec_config(**overrides):
    values = {
        "num_hidden_layers": 4,
        "layer_types": [
            "kimi_delta_attention",
            "kimi_delta_attention",
            "multi_latent_attention",
            "multi_latent_attention",
        ],
        "multi_latent_attention": True,
        "moe_layer_freq": [0, 1, 0, 1],
        "num_empty_layers_add_in_head": 2,
        "n_routed_experts": 8,
        "moe_expert_fusion": True,
        "use_qk_norm": True,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


class TestGPTMixedAttentionDecoderSpec(unittest.TestCase):
    def test_attention_schedule_and_moe_pattern_are_orthogonal(self):
        """Every attention type must combine with the matching Dense/MoE slot."""
        config = _make_spec_config()
        built_specs = [object() for _ in range(config.num_hidden_layers)]
        with mock.patch.object(
            gpt_layer_specs,
            "get_gpt_layer_local_spec",
            side_effect=built_specs,
        ) as build_layer:
            result = gpt_layer_specs.get_gpt_decoder_layers_spec(
                config, normalization="RMSNorm", qk_l2_norm=True
            )

        self.assertEqual(result, built_specs)
        calls = [call.kwargs for call in build_layer.call_args_list]
        self.assertEqual(
            [call["attention_layer_type"] for call in calls],
            config.layer_types,
        )
        self.assertEqual(
            [call["num_experts"] for call in calls], [None, 8, None, 8]
        )
        self.assertEqual(
            [call["moe_expert_fusion"] for call in calls],
            [False, True, False, True],
        )
        self.assertEqual([call["layer_number"] for call in calls], [2, 3, 4, 5])
        self.assertTrue(all(call["use_qk_norm"] for call in calls))
        self.assertTrue(all(call["qk_l2_norm"] for call in calls))

    def test_unset_layer_types_preserves_homogeneous_mla_default(self):
        config = _make_spec_config(layer_types=None)
        with mock.patch.object(
            gpt_layer_specs,
            "get_gpt_layer_local_spec",
            side_effect=[object() for _ in range(config.num_hidden_layers)],
        ) as build_layer:
            gpt_layer_specs.get_gpt_decoder_layers_spec(config)

        self.assertEqual(
            [
                call.kwargs["attention_layer_type"]
                for call in build_layer.call_args_list
            ],
            ["multi_latent_attention"] * config.num_hidden_layers,
        )

    def test_layer_types_length_must_match_num_hidden_layers(self):
        config = _make_spec_config(layer_types=["kimi_delta_attention"])
        with self.assertRaisesRegex(ValueError, "layer_types must contain 4"):
            gpt_layer_specs.get_gpt_decoder_layers_spec(config)

    def test_transformer_config_declares_kimi_generic_switches(self):
        base = TransformerConfig(hidden_size=64, num_attention_heads=4)
        enabled = TransformerConfig(
            hidden_size=64,
            num_attention_heads=4,
            latent_moe_use_norm=True,
            mla_use_nope=True,
        )

        self.assertFalse(base.latent_moe_use_norm)
        self.assertFalse(base.mla_use_nope)
        self.assertTrue(enabled.latent_moe_use_norm)
        self.assertTrue(enabled.mla_use_nope)


if __name__ == "__main__":
    unittest.main()
