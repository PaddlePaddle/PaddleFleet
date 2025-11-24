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

import unittest

import paddle

from paddlefleet.transformer.attention import (
    SelfAttention,
    SelfAttentionSublayersSpec,
)
from paddlefleet.transformer.dot_product_attention import DotProductAttention
from paddlefleet.transformer.enums import AttnMaskType
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.utils import (
    init_method_normal,
    scaled_init_method_normal,
)


class BiasedLinear(paddle.nn.Layer):
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__()
        self.linear = paddle.nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x), self.linear.bias


class RMSNorm(paddle.nn.Layer):
    def __init__(self, hidden_size, eps, **kwargs):
        super().__init__()
        self.weight = paddle.nn.Parameter(paddle.zeros([hidden_size]))
        self.eps = eps

    def forward(self, x):
        d_norm = paddle.rsqrt(x.pow(2).mean(axis=-1, keepdim=True) + self.eps)
        return x * d_norm * self.weight


class TestSelfAttention(unittest.TestCase):
    def setUp(self):
        self.config = TransformerConfig(
            num_hidden_layers=1,
            hidden_size=128,
            num_attention_heads=4,
        )

        # TODO(liangshuhao): make these args formal
        self.config.num_key_value_heads = self.config.num_attention_heads
        self.config.head_dim = (
            self.config.hidden_size // self.config.num_attention_heads
        )
        self.config.softmax_scale = None
        self.config.use_bias = True
        self.config.no_rope_freq = None
        self.config.recompute_granularity = None
        self.config.fused_single_qkv_rope = False
        self.config.rotary_interleaved = False
        self.config.multi_latent_attention = False
        self.config.init_method = init_method_normal(0.02)
        self.config.output_layer_init_method = scaled_init_method_normal(
            0.02, 1, 2.0
        )
        self.config.rms_norm_eps = 1e-5
        self.config.context_parallel_size = 1
        self.config.apply_query_key_layer_scaling = False
        self.config.sliding_window = None
        self.config.window_attn_skip_freq = None
        self.config.fp16 = False
        self.config.bf16 = False
        self.config.masked_softmax_fusion = False
        self.config.attention_softmax_in_fp32 = True
        self.config.attention_dropout = 0.1
        self.config.softmax_type = "vanilla"

        self.self_attn = SelfAttention(
            self.config,
            SelfAttentionSublayersSpec(
                qkv_proj=BiasedLinear,
                core_attention=DotProductAttention,
                o_proj=BiasedLinear,
                q_layernorm=RMSNorm,
                k_layernorm=RMSNorm,
            ),
            attn_mask_type=AttnMaskType.causal,
            layer_number=1,
        )

    def test_self_attention(self):
        config = self.self_attn.config
        sequence_length = 127
        micro_batch_size = 2
        hidden_size = self.self_attn.config.hidden_size

        hidden_states = paddle.randn(
            (micro_batch_size, sequence_length, hidden_size),
        )
        rotary_pos_emb = paddle.randn(
            (1, sequence_length, 1, self.config.head_dim)
        )

        output, bias = self.self_attn(
            hidden_states, attention_mask=None, rotary_pos_emb=rotary_pos_emb
        )

        # Check if output and bias have the correct shape
        assert output.shape[0] == micro_batch_size
        assert output.shape[1] == sequence_length
        assert output.shape[2] == config.hidden_size
        assert bias.shape[0] == config.hidden_size


if __name__ == "__main__":
    unittest.main()
