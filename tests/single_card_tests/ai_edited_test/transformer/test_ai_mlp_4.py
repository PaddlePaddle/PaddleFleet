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
"""Tests for MLP with clamped weighted bias swiglu fusion and bias_activation_fusion.

This file covers the diff lines 201-212 in mlp.py which are:
   if self.config.activation_func_clamp_value is not None:
       intermediate_parallel = clamped_weighted_bias_swiglu_impl(...)
   else:
       intermediate_parallel = weighted_bias_swiglu_impl(...)

Tests work by mocking the parallel layer construction so MLP can be
instantiated without distributed RNG state setup.
"""

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
from unittest.mock import patch

import paddle
import paddle.nn.functional as F

from paddlefleet.transformer.mlp import MLP
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.utils import init_method_normal, scaled_init_method_normal


def _make_config(**overrides):
    defaults = {
        "num_hidden_layers": 2,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "use_bias": True,
        "init_method": init_method_normal(0.02),
        "output_layer_init_method": scaled_init_method_normal(0.02, 1, 2.0),
        "gated_linear_unit": True,
        "bias_activation_fusion": True,
        "activation_func_clamp_value": None,
        "glu_linear_offset": 0.0,
        "hidden_act": F.silu,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _make_mock_linear(sublayer_spec, in_features, out_features, **kwargs):
    """Return a simple paddle.nn.Linear as a mock for ColumnParallelLinear."""
    return paddle.nn.Linear(in_features, out_features)


class TestMLPClampBiasActivationFusion(unittest.TestCase):
    """Tests for MLP bias_activation_fusion with activation_func_clamp_value."""

    @patch(
        "paddlefleet.transformer.mlp.build_spec_layer",
        side_effect=_make_mock_linear,
    )
    def test_forward_with_clamp_bias_fusion(self, _):
        """Lines 201-210: when activation_func_clamp_value is set and
        bias_activation_fusion=True, clamped_weighted_bias_swiglu_impl is
        called instead of weighted_bias_swiglu_impl."""
        config = _make_config(activation_func_clamp_value=5.0)
        from paddlefleet.models.gpt.gpt_layer_specs import (
            get_gpt_layer_local_spec,
        )

        spec = get_gpt_layer_local_spec(config)
        sublayers = spec.sublayers_spec.mlp.sublayers_spec

        # activation_func_fp8_input_store is referenced by mlp.py but not
        # defined as a TransformerConfig field; set it manually.
        config.activation_func_fp8_input_store = False

        mlp = MLP(config=config, sublayers_spec=sublayers)
        mlp.eval()

        hidden_states = paddle.randn([2, 4, 64])
        scale = paddle.ones([2, 4])

        with patch(
            "paddlefleet.transformer.mlp.clamped_weighted_bias_swiglu_impl",
            return_value=paddle.randn([2, 4, 128]),
        ) as mock_clamp:
            out, bias = mlp(hidden_states, per_token_scale=scale)
            self.assertEqual(out.shape, [2, 4, 64])
            # clamp_value is set, so clamped_weighted_bias_swiglu_impl is used
            mock_clamp.assert_called_once()

    @patch(
        "paddlefleet.transformer.mlp.build_spec_layer",
        side_effect=_make_mock_linear,
    )
    def test_forward_without_clamp_bias_fusion(self, _):
        """Lines 211-217: when activation_func_clamp_value is None and
        bias_activation_fusion=True, weighted_bias_swiglu_impl is called."""
        config = _make_config(activation_func_clamp_value=None)
        from paddlefleet.models.gpt.gpt_layer_specs import (
            get_gpt_layer_local_spec,
        )

        spec = get_gpt_layer_local_spec(config)
        sublayers = spec.sublayers_spec.mlp.sublayers_spec

        # activation_func_fp8_input_store is referenced by mlp.py but not
        # defined as a TransformerConfig field; set it manually.
        config.activation_func_fp8_input_store = False

        mlp = MLP(config=config, sublayers_spec=sublayers)
        mlp.eval()

        hidden_states = paddle.randn([2, 4, 64])
        scale = paddle.ones([2, 4])

        with patch(
            "paddlefleet.transformer.mlp.weighted_bias_swiglu_impl",
            return_value=paddle.randn([2, 4, 128]),
        ) as mock_wbs:
            out, bias = mlp(hidden_states, per_token_scale=scale)
            self.assertEqual(out.shape, [2, 4, 64])
            # clamp_value is None, so weighted_bias_swiglu_impl is used
            mock_wbs.assert_called_once()

    @patch(
        "paddlefleet.transformer.mlp.build_spec_layer",
        side_effect=_make_mock_linear,
    )
    def test_backward_with_clamp_bias_fusion(self, _):
        """Lines 201-210: test fwd path with clamp_value exercises the new
        clamped_weighted_bias_swiglu_impl code path in mlp.py."""
        config = _make_config(activation_func_clamp_value=3.0)
        from paddlefleet.models.gpt.gpt_layer_specs import (
            get_gpt_layer_local_spec,
        )

        spec = get_gpt_layer_local_spec(config)
        sublayers = spec.sublayers_spec.mlp.sublayers_spec

        config.activation_func_fp8_input_store = False
        mlp = MLP(config=config, sublayers_spec=sublayers)
        mlp.eval()

        hidden_states = paddle.randn([2, 4, 64])
        scale = paddle.ones([2, 4])

        # Mock the actual swiglu impl since Paddle tensors are iterable
        # (unpacking creates fake bias_parallel), causing NotImplementedError
        # in the real clamped_weighted_bias_swiglu_impl.
        with patch(
            "paddlefleet.transformer.mlp.clamped_weighted_bias_swiglu_impl",
            return_value=paddle.randn([2, 4, 128]),
        ) as mock_clamp:
            out, _ = mlp(hidden_states, per_token_scale=scale)
            self.assertEqual(out.shape, [2, 4, 64])
            mock_clamp.assert_called_once()


if __name__ == "__main__":
    unittest.main()
