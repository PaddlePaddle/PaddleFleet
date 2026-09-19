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

"""Structural unit tests for the HyperBody decoder layer specs.

These only inspect the ``LayerSpec`` structure produced by
``get_hyperbody_decoder_layer_specs`` / ``get_hyperbody_decoder_block_spec``,
so they run on a single card without any distributed / RNG initialization.
"""

import unittest

from paddlefleet.models.hyperbody_decoder.layer_specs import (
    _is_moe_layer,
    get_hyperbody_decoder_block_spec,
    get_hyperbody_decoder_layer_specs,
)
from paddlefleet.transformer.enums import AttnMaskType
from paddlefleet.transformer.transformer_block import LayerNormImpl
from paddlefleet.transformer.transformer_config import TransformerConfig


def _make_config(**overrides) -> TransformerConfig:
    """A tiny DeepSeekV2-Lite-shaped config: layer 0 dense, rest MoE."""
    kwargs = {
        "num_hidden_layers": 4,
        "hidden_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "intermediate_size": 128,
        "moe_layer_freq": [0, 1, 1, 1],
        "n_routed_experts": 4,
        "moe_intermediate_size": 64,
        "num_experts_per_tok": 2,
        "n_shared_experts": 1,
        "normalization": "RMSNorm",
    }
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


class TestHyperBodyDecoderLayerSpecs(unittest.TestCase):
    def test_spec_count_matches_num_layers(self):
        config = _make_config()
        specs = get_hyperbody_decoder_layer_specs(config)
        self.assertEqual(len(specs), config.num_hidden_layers)

    def test_layer0_dense_rest_moe(self):
        config = _make_config()
        specs = get_hyperbody_decoder_layer_specs(config)
        mlp_classes = [s.sublayers_spec.mlp.layer.__name__ for s in specs]
        self.assertEqual(mlp_classes[0], "MLP")
        self.assertTrue(all(name == "MoELayer" for name in mlp_classes[1:]))

    def test_is_moe_layer_reads_list(self):
        config = _make_config()
        flags = [
            _is_moe_layer(config, i) for i in range(config.num_hidden_layers)
        ]
        self.assertEqual(flags, [False, True, True, True])

    def test_all_layers_causal(self):
        config = _make_config()
        specs = get_hyperbody_decoder_layer_specs(config)
        for s in specs:
            self.assertEqual(
                s.sublayers_spec.self_attn.extra_kwargs["attn_mask_type"],
                AttnMaskType.causal,
            )

    def test_block_spec_final_norm_is_stock_norm(self):
        # layer_norm must be the stock LayerNormImpl -- a None would make
        # TransformerBlock silently drop the final norm.
        config = _make_config()
        block = get_hyperbody_decoder_block_spec(config)
        self.assertIs(block.layer_norm, LayerNormImpl)
        self.assertEqual(len(block.layer_specs), config.num_hidden_layers)

    def test_int_moe_layer_freq_rejected(self):
        # Passing an int would silently take the `i % N` path (Paddle) which
        # disagrees with Megatron's per-layer list semantics -> must raise.
        config = _make_config(moe_layer_freq=2)
        with self.assertRaises(TypeError):
            _is_moe_layer(config, 0)

    def test_length_mismatch_rejected(self):
        config = _make_config(moe_layer_freq=[0, 1])
        with self.assertRaises(ValueError):
            _is_moe_layer(config, 0)

    def test_multi_latent_attention_rejected(self):
        # HyperBody decoder is pure MHA; MLA must be refused early.
        config = _make_config(multi_latent_attention=True)
        with self.assertRaises(ValueError):
            get_hyperbody_decoder_layer_specs(config)

    def test_non_binary_moe_layer_freq_rejected(self):
        # A non-0/1 entry would be silently coerced to a MoE layer by bool();
        # it must be rejected instead.
        config = _make_config(moe_layer_freq=[0, 2, 1, 1])
        with self.assertRaises(ValueError):
            _is_moe_layer(config, 1)

    def test_dsv4_hybrid_attention_variant_rejected(self):
        # experimental_attention_variant="dsv4_hybrid" would route to the DSV4
        # hybrid attention path, not pure MHA -> must be refused early.
        config = _make_config()
        config.experimental_attention_variant = "dsv4_hybrid"
        with self.assertRaises(ValueError):
            get_hyperbody_decoder_layer_specs(config)

    def test_vha_attention_rejected(self):
        # use_vha_attention=True would route self_attention to SelfAttentionVHA,
        # not pure MHA -> must be refused early.
        config = _make_config()
        config.use_vha_attention = True
        with self.assertRaises(ValueError):
            get_hyperbody_decoder_layer_specs(config)


if __name__ == "__main__":
    unittest.main()
