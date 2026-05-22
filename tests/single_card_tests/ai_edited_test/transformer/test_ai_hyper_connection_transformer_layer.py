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
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

import unittest

import paddle

from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddlefleet.transformer.hyper_connection import HyperConnectionModule
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.transformer.transformer_layer import (
    HyperConnectionTransformerLayer,
    TransformerLayer,
    TransformerLayerSublayersSpec,
)
from paddlefleet.utils import init_method_normal, scaled_init_method_normal

# Initialize CUDA RNG tracker for tensor parallel layers
model_parallel_cuda_manual_seed(42, tp_rank=0, ep_rank=0, etp_rank=0)


def _make_config(**overrides):
    defaults = {
        "num_hidden_layers": 2,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 16,
        "use_bias": False,
        "hidden_dropout_prob": 0.0,
        "normalization": "RMSNorm",
        "rms_norm_eps": 1e-5,
        "sequence_parallel": False,
        "tensor_model_parallel_size": 1,
        "context_parallel_size": 1,
        "recompute_granularity": None,
        "recompute_method": None,
        "recompute_num_layers": None,
        "recompute_modules": None,
        "block_attention_residuals": False,
        "attn_res_block_size": 1,
        "attention_dropout": 0.0,
        "bias_dropout_fusion": False,
        "apply_rope_fusion": False,
        "apply_query_key_layer_scaling": False,
        "sliding_window": None,
        "softmax_type": "vanilla",
        "gated_linear_unit": False,
        "bias_activation_fusion": False,
        "gated_attention": False,
        "num_nextn_predict_layers": 0,
        "mtp_load_weight_only": False,
        "init_method": init_method_normal(0.02),
        "output_layer_init_method": scaled_init_method_normal(0.02, 1, 2.0),
        "fp16": False,
        "bf16": False,
        "masked_softmax_fusion": False,
        "attention_softmax_in_fp32": True,
        "softmax_scale": None,
        "multi_latent_attention": False,
        "rotary_interleaved": False,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _make_hc_config(**overrides):
    """Create a config with hyper-connections enabled."""
    hc_defaults = {
        "enable_hyper_connections": True,
        "num_residual_streams": 4,
        "mhc_sinkhorn_iterations": 5,
        "mhc_init_gating_factor": 0.01,
    }
    hc_defaults.update(overrides)
    return _make_config(**hc_defaults)


def _make_hc_layer(config, layer_number=1):
    spec = get_gpt_layer_local_spec(config)
    return HyperConnectionTransformerLayer(
        config=config,
        sublayers_spec=spec.sublayers_spec,
        layer_number=layer_number,
    )


class TestHyperConnectionModule(unittest.TestCase):
    """Tests for HyperConnectionModule."""

    def setUp(self):
        self.config = _make_hc_config()
        self.n = self.config.num_residual_streams
        self.C = self.config.hidden_size
        self.module = HyperConnectionModule(config=self.config, layer_number=1)

    def test_construction(self):
        self.assertEqual(self.module.n, self.n)
        self.assertEqual(self.module.hidden_size, self.C)
        self.assertIsNotNone(self.module.mapping_proj)

    def test_forward_output_shapes(self):
        """forward() should return (aggregated, h_res, h_post) with correct shapes."""
        B, S = 2, 4
        x = paddle.randn([B, S, self.n * self.C])
        aggregated, h_res, h_post = self.module(x)
        self.assertEqual(list(aggregated.shape), [B, S, self.C])
        self.assertEqual(list(h_res.shape), [B, S, self.n, self.n])
        self.assertEqual(list(h_post.shape), [B, S, self.n])

    def test_fused_h_res_h_post_bda_shape(self):
        """fused_h_res_h_post_bda should return shape [..., n*C]."""
        B, S = 2, 4
        x = paddle.randn([B, S, self.n * self.C])
        aggregated, h_res, h_post = self.module(x)

        # Simulate layer output (attn/mlp output, bias=None)
        layer_output = paddle.randn([B, S, self.C])
        layer_output_with_bias = (layer_output, None)

        result = self.module.fused_h_res_h_post_bda(
            h_res=h_res,
            original_residual=x,
            h_post=h_post,
            layer_output_with_bias=layer_output_with_bias,
            dropout_prob=0.0,
            training=False,
            fused=False,
        )
        self.assertEqual(list(result.shape), [B, S, self.n * self.C])

    def test_fused_h_res_h_post_bda_with_bias(self):
        """fused_h_res_h_post_bda should handle non-None bias correctly."""
        B, S = 2, 4
        x = paddle.randn([B, S, self.n * self.C])
        aggregated, h_res, h_post = self.module(x)

        layer_output = paddle.randn([B, S, self.C])
        bias = paddle.randn([self.C])
        layer_output_with_bias = (layer_output, bias)

        result = self.module.fused_h_res_h_post_bda(
            h_res=h_res,
            original_residual=x,
            h_post=h_post,
            layer_output_with_bias=layer_output_with_bias,
            dropout_prob=0.0,
            training=False,
            fused=False,
        )
        self.assertEqual(list(result.shape), [B, S, self.n * self.C])


class TestHyperConnectionTransformerLayerConstruction(unittest.TestCase):
    """Tests for HyperConnectionTransformerLayer constructor."""

    def test_basic_construction(self):
        config = _make_hc_config()
        layer = _make_hc_layer(config)
        self.assertIsInstance(layer, HyperConnectionTransformerLayer)
        self.assertIsInstance(layer, TransformerLayer)

    def test_has_hyper_connection_sublayers(self):
        config = _make_hc_config()
        layer = _make_hc_layer(config)
        self.assertIsInstance(
            layer.self_attention_hyper_connection, HyperConnectionModule
        )
        self.assertIsInstance(layer.mlp_hyper_connection, HyperConnectionModule)

    def test_raises_without_hc_spec(self):
        """Should raise if sublayers_spec has IdentityOp for hyper connections."""
        config = _make_hc_config()
        spec = TransformerLayerSublayersSpec()  # all IdentityOp
        with self.assertRaises(AssertionError):
            HyperConnectionTransformerLayer(
                config=config,
                sublayers_spec=spec,
                layer_number=1,
            )

    def test_raises_with_block_attention_residuals(self):
        """mHC is incompatible with block_attention_residuals."""
        config = _make_hc_config(block_attention_residuals=True)
        with self.assertRaises(AssertionError):
            _make_hc_layer(config)

    def test_layer_number(self):
        config = _make_hc_config()
        layer = _make_hc_layer(config, layer_number=3)
        self.assertEqual(layer.layer_number, 3)


class TestHyperConnectionTransformerLayerForward(unittest.TestCase):
    """Tests for HyperConnectionTransformerLayer forward pass."""

    def setUp(self):
        self.config = _make_hc_config()
        self.layer = _make_hc_layer(self.config)
        self.layer.eval()
        self.n = self.config.num_residual_streams
        self.C = self.config.hidden_size

    def test_forward_output_shape(self):
        """Forward should produce hidden_states with shape [S, B, n*C] (seq-first)."""
        B, S = 2, 4
        # Transformer layer uses seq-first: [S, B, n*C]
        hidden_states = paddle.randn([S, B, self.n * self.C])
        dict_args = {
            "hidden_states": hidden_states,
            "attention_mask": None,
        }
        result = self.layer.forward(dict_args)
        self.assertIn("hidden_states", result)
        self.assertEqual(
            list(result["hidden_states"].shape), [S, B, self.n * self.C]
        )

    def test_forward_no_nan(self):
        """Output should not contain NaN values."""
        B, S = 2, 4
        hidden_states = paddle.randn([S, B, self.n * self.C])
        dict_args = {
            "hidden_states": hidden_states,
            "attention_mask": None,
        }
        result = self.layer.forward(dict_args)
        self.assertFalse(paddle.isnan(result["hidden_states"]).any().item())

    def test_forward_with_rotary_pos_emb(self):
        """Forward should work with rotary_pos_emb provided."""
        B, S = 2, 4
        head_dim = self.config.head_dim
        hidden_states = paddle.randn([S, B, self.n * self.C])
        rotary_pos_emb = paddle.randn([B, S, head_dim])
        dict_args = {
            "hidden_states": hidden_states,
            "attention_mask": None,
            "rotary_pos_emb": rotary_pos_emb,
        }
        result = self.layer.forward(dict_args)
        self.assertEqual(
            list(result["hidden_states"].shape), [S, B, self.n * self.C]
        )

    def test_forward_deterministic_in_eval(self):
        """Two forward passes with same input should produce same output in eval mode."""
        B, S = 2, 4
        hidden_states = paddle.randn([S, B, self.n * self.C])
        dict_args1 = {
            "hidden_states": hidden_states.clone(),
            "attention_mask": None,
        }
        dict_args2 = {
            "hidden_states": hidden_states.clone(),
            "attention_mask": None,
        }
        result1 = self.layer.forward(dict_args1)
        result2 = self.layer.forward(dict_args2)
        self.assertTrue(
            paddle.allclose(result1["hidden_states"], result2["hidden_states"])
        )


class TestHyperConnectionTransformerLayerRecompute(unittest.TestCase):
    """Tests for HyperConnectionTransformerLayer with selective recompute."""

    def test_selective_recompute_mlp(self):
        config = _make_hc_config(
            recompute_granularity="selective",
            recompute_modules=["mlp"],
        )
        layer = _make_hc_layer(config)
        self.assertTrue(layer.recompute_mlp)

    def test_selective_recompute_norm(self):
        config = _make_hc_config(
            recompute_granularity="selective",
            recompute_modules=["norm"],
        )
        layer = _make_hc_layer(config)
        self.assertTrue(layer.recompute_input_layernorm)
        self.assertTrue(layer.recompute_post_attention_layernorm)


class TestGetGptLayerSpecWithHC(unittest.TestCase):
    """Tests that get_gpt_layer_local_spec produces correct spec when mHC is enabled."""

    def test_spec_uses_hc_transformer_layer(self):
        config = _make_hc_config()
        spec = get_gpt_layer_local_spec(config)
        self.assertEqual(spec.layer, HyperConnectionTransformerLayer)

    def test_spec_without_hc_uses_base_layer(self):
        config = _make_config()
        spec = get_gpt_layer_local_spec(config)
        self.assertEqual(spec.layer, TransformerLayer)

    def test_spec_sublayers_have_hc_modules(self):
        config = _make_hc_config()
        spec = get_gpt_layer_local_spec(config)
        from paddle.distributed.fleet.meta_parallel import LayerSpec

        self.assertIsInstance(
            spec.sublayers_spec.self_attention_hyper_connection, LayerSpec
        )
        self.assertEqual(
            spec.sublayers_spec.self_attention_hyper_connection.layer,
            HyperConnectionModule,
        )
        self.assertIsInstance(
            spec.sublayers_spec.mlp_hyper_connection, LayerSpec
        )
        self.assertEqual(
            spec.sublayers_spec.mlp_hyper_connection.layer,
            HyperConnectionModule,
        )


if __name__ == "__main__":
    unittest.main()
