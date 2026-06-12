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
"""Tests to cover high_precision_compressor and high_precision_mhc paths."""

import os
import sys
import types
import unittest
from unittest.mock import patch

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

import paddle
from paddle import nn

from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddlefleet.transformer.csa_attention import (
    Compressor,
    CompressorSublayersSpec,
)
from paddlefleet.transformer.hyper_connection import HyperConnectionModule
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.utils import init_method_normal, scaled_init_method_normal

model_parallel_cuda_manual_seed(42, tp_rank=0, ep_rank=0, etp_rank=0)


# ---------- Helpers ----------


class _Lin(nn.Layer):
    def __init__(self, input_size, output_size, **kwargs):
        super().__init__()
        self.weight = self.create_parameter(
            shape=[output_size, input_size],
            dtype="bfloat16",
            default_initializer=nn.initializer.Normal(std=0.02),
        )

    def forward(self, x):
        return paddle.matmul(x, self.weight.T), None


class _Norm(nn.Layer):
    def __init__(self, hidden_size=None, **kwargs):
        super().__init__()
        self.variance_epsilon = 1e-5
        self.weight = self.create_parameter(
            shape=[hidden_size],
            dtype="float32",
            default_initializer=nn.initializer.Constant(1.0),
        )

    def forward(self, x):
        return (
            x
            * paddle.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5)
            * self.weight.cast(x.dtype)
        )


def _make_compressor_config(**overrides):
    defaults = {
        "hidden_size": 64,
        "qk_pos_emb_head_dim": 0,
        "init_method": None,
        "init_method_std": 0.02,
        "rms_norm_eps": 1e-5,
        "high_precision_compressor": True,
    }
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def _make_hc_config(**overrides):
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
        "enable_hyper_connections": True,
        "num_residual_streams": 4,
        "mhc_sinkhorn_iterations": 5,
        "mhc_init_gating_factor": 0.01,
        "high_precision_mhc": True,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


# ==============================================================================
# Tests for Compressor high_precision_compressor path
# ==============================================================================


class TestCompressorHighPrecision(unittest.TestCase):
    """Cover csa_attention.py lines 1175-1182 and 1327-1333."""

    def setUp(self):
        self.config = _make_compressor_config(high_precision_compressor=True)
        self.head_dim = 64
        self.ratio = 128
        spec = CompressorSublayersSpec(
            linear_wkv=_Lin, linear_wgate=_Lin, norm=_Norm
        )
        paddle.seed(42)
        self.compressor = Compressor(
            config=self.config,
            sublayers_spec=spec,
            compress_ratio=self.ratio,
            head_dim=self.head_dim,
            rotate=False,
            rotary_pos_emb=None,
        )
        self.compressor.eval()

    def test_high_precision_forward(self):
        """high_precision_compressor=True should use float32 matmul path."""
        b, sq = 1, 128
        x = paddle.randn([b, sq, 64], dtype="bfloat16")
        result = self.compressor(x)
        self.assertIsNotNone(result)
        self.assertEqual(result.dtype, paddle.bfloat16)
        self.assertFalse(paddle.isnan(result).any().item())

    def test_low_precision_forward(self):
        """high_precision_compressor=False should use standard linear + norm path."""
        config = _make_compressor_config(high_precision_compressor=False)
        spec = CompressorSublayersSpec(
            linear_wkv=_Lin, linear_wgate=_Lin, norm=_Norm
        )
        compressor = Compressor(
            config=config,
            sublayers_spec=spec,
            compress_ratio=128,
            head_dim=64,
            rotate=False,
            rotary_pos_emb=None,
        )
        compressor.eval()
        x = paddle.randn([1, 128, 64], dtype="bfloat16")
        result = compressor(x)
        self.assertIsNotNone(result)
        self.assertFalse(paddle.isnan(result).any().item())


# ==============================================================================
# Tests for HyperConnectionModule accuracy-compatible + sequential paths
# ==============================================================================


class TestHyperConnectionAccuracyCompatibleKernel(unittest.TestCase):
    """Cover hyper_connection.py lines 333,335,345-346."""

    def setUp(self):
        self.n = 4
        self.C = 64

    @patch(
        "paddlefleet.transformer.hyper_connection._ACCURACY_COMPATIBLE_KERNEL",
        True,
    )
    def test_projection_accuracy_compatible_low_precision(self):
        """_use_accuracy_compatible_kernel=True with high_precision_mhc=False."""
        config = _make_hc_config(high_precision_mhc=False)
        module = HyperConnectionModule(config=config, layer_number=1)
        module = paddle.amp.decorate(
            models=module, level="O2", dtype="bfloat16"
        )
        x = paddle.randn([2, 4, self.n * self.C]).astype("bfloat16")
        # This calls _projection_and_get_norm internally
        contracted, h_res, h_post = module(x)
        self.assertFalse(
            paddle.isnan(contracted.astype("float32")).any().item()
        )


class TestHyperConnectionSequentialPath(unittest.TestCase):
    """Cover hyper_connection.py lines 722-726 (sequential path with dropout)."""

    def setUp(self):
        self.n = 4
        self.C = 64

    @patch(
        "paddlefleet.transformer.hyper_connection._ACCURACY_COMPATIBLE_KERNEL",
        True,
    )
    def test_bda_sequential_path_with_bias(self):
        """Sequential path triggered by accuracy_compatible_kernel=True, covers 722-726."""
        config = _make_hc_config(high_precision_mhc=True)
        module = HyperConnectionModule(config=config, layer_number=1)
        module = paddle.amp.decorate(
            models=module, level="O2", dtype="bfloat16"
        )
        B, S = 2, 4
        x = paddle.randn([B, S, self.n * self.C]).astype("bfloat16")
        _, h_res, h_post = module(x)
        layer_output = paddle.randn([B, S, self.C]).astype("bfloat16")
        bias = paddle.randn([self.C]).astype("bfloat16")

        result = module.fused_h_res_h_post_bda(
            h_res=h_res,
            original_residual=x,
            h_post=h_post,
            layer_output_with_bias=(layer_output, bias),
            dropout_prob=0.0,
            training=False,
            fused=False,
        )
        self.assertFalse(paddle.isnan(result.astype("float32")).any().item())
        self.assertEqual(list(result.shape), [B, S, self.n * self.C])


if __name__ == "__main__":
    unittest.main()
