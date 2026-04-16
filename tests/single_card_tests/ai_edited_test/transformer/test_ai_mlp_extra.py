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
from unittest.mock import MagicMock, patch

import paddle
import paddle.nn.functional as F

from paddlefleet.transformer.mlp import MLP, MLPSublayersSpec
from paddlefleet.transformer.transformer_config import TransformerConfig


class FakeColumnParallel(paddle.nn.Layer):
    def __init__(self, in_f, out_f, **kwargs):
        super().__init__()
        self.in_features = in_f
        self.out_features = out_f
        self.linear = paddle.nn.Linear(in_f, out_f)
        self.skip_bias_add = kwargs.get("skip_bias_add", False)
        self.gather_output = kwargs.get("gather_output", False)
        self.input_is_parallel = kwargs.get("input_is_parallel", False)
        self.is_expert = kwargs.get("is_expert", False)
        self.tp_group = kwargs.get("tp_group", None)
        self.use_bias = kwargs.get("use_bias", False)
        self._bias_val = kwargs.get("bias", False)

    def forward(self, x, per_token_scale=None):
        # Always re-create linear to match actual input size, keep out_features
        in_f = x.shape[-1]
        if in_f != self.linear.weight.shape[0]:
            self.linear = paddle.nn.Linear(in_f, self.out_features)
        out = self.linear(x)
        bias = self.linear.bias if self._bias_val else None
        return out, bias

    def backward_dw(self):
        pass


def _make_config(**overrides):
    defaults = {
        "hidden_size": 64,
        "num_attention_heads": 2,
        "gated_linear_unit": False,
        "hidden_act": F.gelu,
        "intermediate_size": 256,
        "use_bias": False,
        "bias_activation_fusion": False,
        "params_dtype": paddle.float32,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _make_build_side_effect(in_f, inter_f, hidden_f, use_bias=False):
    """Return a side_effect that produces correctly-sized FakeColumnParallel
    objects: first for up_gate_proj, second for down_proj."""
    call_idx = [0]

    def _side_effect(*a, **kw):
        idx = call_idx[0]
        call_idx[0] += 1
        if idx == 0:
            return FakeColumnParallel(in_f, inter_f, use_bias=use_bias)
        else:
            return FakeColumnParallel(inter_f, hidden_f, use_bias=use_bias)

    return _side_effect


class TestMLPConstruction(unittest.TestCase):
    """Test MLP construction paths."""

    @patch("paddlefleet.transformer.mlp.build_spec_layer")
    def test_glu_doubles_intermediate(self, mock_build):
        mock_build.return_value = FakeColumnParallel(64, 512)
        config = _make_config(gated_linear_unit=True, intermediate_size=256)
        spec = MLPSublayersSpec(up_gate_proj=MagicMock(), down_proj=MagicMock())
        mlp = MLP(config, spec)
        self.assertIsNotNone(mlp.up_gate_proj)
        self.assertIsNotNone(mlp.down_proj)
        self.assertIsNotNone(mlp.hidden_act)

    @patch("paddlefleet.transformer.mlp.build_spec_layer")
    def test_expert_requires_intermediate_size(self, mock_build):
        mock_build.side_effect = lambda *a, **kw: FakeColumnParallel(64, 256)
        config = _make_config()
        config.intermediate_size = None
        config.moe_intermediate_size = None
        spec = MLPSublayersSpec(up_gate_proj=MagicMock(), down_proj=MagicMock())
        with self.assertRaises(ValueError):
            MLP(config, spec, is_expert=True)

    @patch("paddlefleet.transformer.mlp.build_spec_layer")
    def test_no_intermediate_size_warns(self, mock_build):
        mock_build.side_effect = lambda *a, **kw: FakeColumnParallel(64, 256)
        config = _make_config(intermediate_size=256)
        config.moe_intermediate_size = None
        spec = MLPSublayersSpec(up_gate_proj=MagicMock(), down_proj=MagicMock())
        with self.assertWarns(DeprecationWarning):
            MLP(config, spec)


class TestMLPForwardBiasActivationFusion(unittest.TestCase):
    """Test bias_activation_fusion paths."""

    @patch("paddlefleet.transformer.mlp.build_spec_layer")
    @patch("paddlefleet.transformer.mlp.bias_gelu_impl")
    def test_gelu_fusion_without_glu(self, mock_gelu, mock_build):
        mock_gelu.return_value = paddle.zeros([2, 256])
        mock_build.side_effect = _make_build_side_effect(
            64, 256, 64, use_bias=True
        )
        config = _make_config(
            gated_linear_unit=False,
            bias_activation_fusion=True,
            use_bias=True,
        )
        spec = MLPSublayersSpec(up_gate_proj=MagicMock(), down_proj=MagicMock())
        mlp = MLP(config, spec)
        x = paddle.randn([2, 64])
        out, bias = mlp(x)
        self.assertEqual(out.shape, [2, 64])
        mock_gelu.assert_called()

    @patch("paddlefleet.transformer.mlp.build_spec_layer")
    @patch("paddlefleet.transformer.mlp.bias_swiglu_impl")
    def test_swiglu_fusion_with_glu(self, mock_swiglu, mock_build):
        mock_swiglu.return_value = paddle.zeros([2, 256])
        mock_build.side_effect = _make_build_side_effect(64, 512, 64)
        config = _make_config(
            gated_linear_unit=True,
            bias_activation_fusion=True,
            hidden_act=F.silu,
        )
        spec = MLPSublayersSpec(up_gate_proj=MagicMock(), down_proj=MagicMock())
        mlp = MLP(config, spec)
        x = paddle.randn([2, 64])
        out, bias = mlp(x)
        self.assertEqual(out.shape, [2, 64])
        mock_swiglu.assert_called()


class TestMLPForwardNonFusion(unittest.TestCase):
    """Test non-fusion forward paths."""

    @patch("paddlefleet.transformer.mlp.build_spec_layer")
    def test_glu_activation(self, mock_build):
        mock_build.side_effect = _make_build_side_effect(64, 512, 64)
        config = _make_config(
            gated_linear_unit=True,
            bias_activation_fusion=False,
            hidden_act=F.silu,
            activation_func_clamp_value=None,
        )
        spec = MLPSublayersSpec(up_gate_proj=MagicMock(), down_proj=MagicMock())
        mlp = MLP(config, spec)
        x = paddle.randn([2, 64])
        out, bias = mlp(x)
        self.assertEqual(out.shape, [2, 64])

    @patch("paddlefleet.transformer.mlp.build_spec_layer")
    def test_glu_activation_with_clamp(self, mock_build):
        mock_build.side_effect = _make_build_side_effect(64, 512, 64)
        config = _make_config(
            gated_linear_unit=True,
            bias_activation_fusion=False,
            hidden_act=F.silu,
            activation_func_clamp_value=1.0,
            glu_linear_offset=0.0,
        )
        spec = MLPSublayersSpec(up_gate_proj=MagicMock(), down_proj=MagicMock())
        mlp = MLP(config, spec)
        x = paddle.randn([2, 64])
        out, bias = mlp(x)
        self.assertEqual(out.shape, [2, 64])

    @patch("paddlefleet.transformer.mlp.build_spec_layer")
    def test_non_glu_activation(self, mock_build):
        mock_build.side_effect = _make_build_side_effect(64, 256, 64)
        config = _make_config(
            gated_linear_unit=False,
            bias_activation_fusion=False,
            hidden_act=F.gelu,
        )
        spec = MLPSublayersSpec(up_gate_proj=MagicMock(), down_proj=MagicMock())
        mlp = MLP(config, spec)
        x = paddle.randn([2, 64])
        out, bias = mlp(x)
        self.assertEqual(out.shape, [2, 64])


class TestMLPForwardPerTokenScale(unittest.TestCase):
    """Test per_token_scale in forward."""

    @patch("paddlefleet.transformer.mlp.build_spec_layer")
    @patch("paddlefleet.transformer.mlp.weighted_bias_swiglu_impl")
    def test_fusion_with_per_token_scale(self, mock_wbs, mock_build):
        mock_wbs.return_value = paddle.zeros([2, 256])
        mock_build.side_effect = _make_build_side_effect(64, 512, 64)
        config = _make_config(
            gated_linear_unit=True,
            bias_activation_fusion=True,
            hidden_act=F.silu,
        )
        config.activation_func_fp8_input_store = False
        spec = MLPSublayersSpec(up_gate_proj=MagicMock(), down_proj=MagicMock())
        mlp = MLP(config, spec)
        x = paddle.randn([2, 64])
        scale = paddle.randn([2])
        out, bias = mlp(x, per_token_scale=scale)
        self.assertEqual(out.shape, [2, 64])

    @patch("paddlefleet.transformer.mlp.build_spec_layer")
    def test_non_fusion_with_per_token_scale(self, mock_build):
        mock_build.side_effect = _make_build_side_effect(64, 256, 64)
        config = _make_config(
            gated_linear_unit=False,
            bias_activation_fusion=False,
            hidden_act=F.gelu,
        )
        spec = MLPSublayersSpec(up_gate_proj=MagicMock(), down_proj=MagicMock())
        mlp = MLP(config, spec)
        x = paddle.randn([2, 64])
        scale = paddle.randn([2])
        out, bias = mlp(x, per_token_scale=scale)
        self.assertEqual(out.shape, [2, 64])


class TestMLPInputSize(unittest.TestCase):
    """Test custom input_size parameter."""

    @patch("paddlefleet.transformer.mlp.build_spec_layer")
    def test_custom_input_size(self, mock_build):
        mock_build.side_effect = lambda *a, **kw: FakeColumnParallel(
            32, 128, skip_bias_add=True
        )
        config = _make_config(intermediate_size=128)
        spec = MLPSublayersSpec(up_gate_proj=MagicMock(), down_proj=MagicMock())
        mlp = MLP(config, spec, input_size=32)
        self.assertEqual(mlp.input_size, 32)

    @patch("paddlefleet.transformer.mlp.build_spec_layer")
    def test_custom_hidden_size(self, mock_build):
        mock_build.side_effect = lambda *a, **kw: FakeColumnParallel(
            64, 128, input_is_parallel=True, skip_bias_add=True
        )
        config = _make_config(intermediate_size=128)
        spec = MLPSublayersSpec(up_gate_proj=MagicMock(), down_proj=MagicMock())
        mlp = MLP(config, spec, hidden_size=32)
        self.assertEqual(mlp.hidden_size, 32)


class TestMLPBackwardDw(unittest.TestCase):
    """Test backward_dw method."""

    @patch("paddlefleet.transformer.mlp.build_spec_layer")
    def test_backward_dw(self, mock_build):
        mock_down = MagicMock()
        mock_up = MagicMock()
        call_count = [0]

        def build_side_effect(*a, **kw):
            call_count[0] += 1
            if "down" in str(kw) or call_count[0] == 2:
                return mock_down
            return mock_up

        mock_build.side_effect = build_side_effect

        config = _make_config()
        spec = MLPSublayersSpec(up_gate_proj=MagicMock(), down_proj=MagicMock())
        mlp = MLP(config, spec)
        mlp.backward_dw()
        mock_up.backward_dw.assert_called_once()
        mock_down.backward_dw.assert_called_once()


if __name__ == "__main__":
    unittest.main()
