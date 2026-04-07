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
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import unittest

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import base
from paddle.base import core

from paddlefleet.fusions.fused_bias_swiglu import swiglu
from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from paddlefleet.ops import fused_swiglu_bwd
from paddlefleet.transformer.mlp import MLP
from paddlefleet.transformer.transformer_config import TransformerConfig


class TestParallelMLP(unittest.TestCase):
    transformer_config = TransformerConfig(
        num_hidden_layers=2,
        hidden_size=12,
        intermediate_size=48,
        num_attention_heads=4,
        use_bias=True,
    )
    expected_num_weights = 1212

    def setUp(self):
        self.mlp = MLP(
            self.transformer_config,
            get_gpt_layer_local_spec().sublayers_spec.mlp.sublayers_spec,
        )

    def test_constructor(self):
        assert isinstance(self.mlp, MLP)

        num_weights = sum([p.numel() for p in self.mlp.parameters()])
        assert num_weights == self.expected_num_weights

    def test_forward_backward(self):
        mlp = self.mlp
        # [sequence length, batch size, hidden size]
        hidden_states = paddle.ones((32, 12, mlp.config.hidden_size))
        hidden_states.stop_gradient = False

        # add 0.0 to make hidden_states non-leaf
        output, output_bias = mlp(hidden_states + 0.0)
        assert output.shape[0] == 32
        assert output.shape[1] == 12
        assert output.shape[2] == mlp.config.hidden_size
        assert output.dtype == paddle.float32
        assert output_bias.shape[0] == mlp.config.hidden_size

        paddle.autograd.backward((output, output_bias))
        assert hidden_states.grad is not None


class TestBiasFusedGatedMLP(TestParallelMLP):
    transformer_config = TransformerConfig(
        num_hidden_layers=2,
        hidden_size=12,
        intermediate_size=48,
        num_attention_heads=4,
        bias_activation_fusion=True,
        gated_linear_unit=True,
        hidden_act=F.silu,
        use_bias=True,
    )
    expected_num_weights = 1836


class TestBiasFusedSwiGLURegression(unittest.TestCase):
    """Covers the MLP bias-fused SwiGLU branch used by silu-gated GLU."""

    def setUp(self):
        if not paddle.is_compiled_with_cuda():
            self.skipTest("CUDA is not available")
        np.random.seed(2026)
        paddle.seed(2026)

    @staticmethod
    def _reference_swiglu(y):
        gate, value = paddle.chunk(y, 2, axis=-1)
        return F.silu(gate) * value

    @staticmethod
    def _reference_swiglu_grad(g, y):
        gate, value = paddle.chunk(y, 2, axis=-1)
        gate_sigmoid = paddle.sigmoid(gate)
        gate_silu = F.silu(gate)
        gate_grad = g * gate_sigmoid * (1 + gate * (1 - gate_sigmoid)) * value
        value_grad = g * gate_silu
        return paddle.concat([gate_grad, value_grad], axis=-1)

    def test_bias_fused_swiglu_forward_matches_chunk_silu(self):
        dtype = (
            "bfloat16"
            if core.is_bfloat16_supported(base.CUDAPlace(0))
            else "float16"
        )
        hidden_states = paddle.randn([8, 32], dtype=dtype)
        bias = paddle.randn([32], dtype=dtype)
        fused_input = hidden_states + bias

        fused_out = swiglu(fused_input)
        ref_out = self._reference_swiglu(fused_input)

        np.testing.assert_allclose(
            fused_out.astype("float32").numpy(),
            ref_out.astype("float32").numpy(),
            rtol=0,
            atol=1e-6,
        )

    def test_bias_fused_swiglu_backward_matches_chunk_silu(self):
        fused_input = paddle.randn([8, 32], dtype="float32")
        grad_output = paddle.randn([8, 16], dtype="float32")

        fused_grad = fused_swiglu_bwd(grad_output, fused_input)
        ref_grad = self._reference_swiglu_grad(grad_output, fused_input)

        np.testing.assert_allclose(
            fused_grad.numpy(),
            ref_grad.numpy(),
            rtol=1e-4,
            atol=1e-4,
        )


if __name__ == "__main__":
    unittest.main()
