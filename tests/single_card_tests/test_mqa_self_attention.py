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

import dataclasses
import os
import unittest

os.environ["FLAGS_cudnn_deterministic"] = "True"

import numpy as np
import paddle
from paddle.distributed.fleet.meta_parallel import LayerSpec

from paddlefleet.fusions.fused_bias_dropout import get_bias_dropout_add
from paddlefleet.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddlefleet.transformer.dot_product_attention import DotProductAttention
from paddlefleet.transformer.enums import AttnMaskType
from paddlefleet.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSublayersSpec,
    MQASelfAttention,
)
from paddlefleet.transformer.paddle_norm import WrappedPaddleNorm
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.transformer.transformer_layer import (
    HySparseTransformerLayer,
    TransformerLayerSublayersSpec,
)


class TestMQASelfAttention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        paddle.seed(2026)
        model_parallel_cuda_manual_seed(2026)

        cls.batch_size = 2
        cls.seq_len = 4096

        cls.config = TransformerConfig(
            hidden_size=1536,
            head_dim=128,
            num_attention_heads=4,
            num_key_value_heads=4,
            gated_attention=True,
            qk_rope_head_dim=64,
            qk_nope_head_dim=192,
            v_head_dim=256,
            kv_lora_rank=192,
            rope_theta=5000000,
            use_qk_norm=True,
            multi_latent_attention=True,
            rope_type="rope",
            add_swa_attention_sink_bias=False,
            sliding_window=[128, 128],
            window_attn_skip_freq=2,
            enable_hy_sparse_attention=True,
        )

        cls.sublayer_spec = MLASelfAttentionSublayersSpec(
            core_attention=DotProductAttention,
            o_proj=RowParallelLinear,
            gate_proj=ColumnParallelLinear,
            q_a_proj=ColumnParallelLinear,
            q_b_proj=ColumnParallelLinear,
            kv_a_proj_with_mqa=ColumnParallelLinear,
            kv_b_proj=ColumnParallelLinear,
            q_a_layernorm=WrappedPaddleNorm,
            kv_a_layernorm=WrappedPaddleNorm,
        )

    def test_forward_backward(self):
        # use larger kv_lora_rank to force eager path
        config = dataclasses.replace(self.config, kv_lora_rank=512)

        mla = MLASelfAttention(
            config,
            self.sublayer_spec,
            layer_number=0,
            attn_mask_type=AttnMaskType.causal,
        )
        mla = paddle.amp.decorate(mla, level="O2", dtype="bfloat16")

        mqa = MQASelfAttention(
            config,
            self.sublayer_spec,
            layer_number=0,
            attn_mask_type=AttnMaskType.causal,
        )
        mqa = paddle.amp.decorate(mqa, level="O2", dtype="bfloat16")
        mqa.set_state_dict(mla.state_dict())

        hidden_states = paddle.randn(
            [self.batch_size, self.seq_len, config.hidden_size],
            dtype="bfloat16",
        )
        hidden_states.stop_gradient = False

        # Run MLA
        mla_out, _ = mla(
            hidden_states=hidden_states,
            attention_mask=None,
        )
        output_grad = paddle.randn_like(mla_out) * 1e-2
        mla_out.backward(output_grad)
        mla_input_grad = hidden_states.grad

        hidden_states = hidden_states.detach()
        hidden_states.stop_gradient = False

        # Run MQA
        mqa_out, _ = mqa(
            hidden_states=hidden_states,
            attention_mask=None,
        )
        mqa_out.backward(output_grad)
        mqa_input_grad = hidden_states.grad

        # Compare
        np.testing.assert_allclose(
            mla_out.float(), mqa_out.float(), atol=2e-3, rtol=2e-3
        )
        np.testing.assert_allclose(
            mla_input_grad.float(), mqa_input_grad.float(), atol=2e-3, rtol=2e-3
        )
        for mla_param, mqa_param in zip(mla.parameters(), mqa.parameters()):
            np.testing.assert_allclose(
                mla_param.grad.float(),
                mqa_param.grad.float(),
                atol=2e-3,
                rtol=2e-3,
            )

    def test_kv_sharing(self):
        layer_spec = TransformerLayerSublayersSpec(
            self_attn=LayerSpec(
                layer=MQASelfAttention,
                sublayers_spec=self.sublayer_spec,
            ),
            self_attn_bda=get_bias_dropout_add,
        )
        full_layer = HySparseTransformerLayer(
            self.config, layer_spec, layer_number=0
        )
        full_layer.self_attn.attn_mask_type = AttnMaskType.causal
        full_layer = paddle.amp.decorate(
            full_layer, level="O2", dtype="bfloat16"
        )
        full_layer.full_recompute = True

        swa_layer = HySparseTransformerLayer(
            self.config, layer_spec, layer_number=1
        )
        swa_layer.self_attn.attn_mask_type = AttnMaskType.causal
        swa_layer = paddle.amp.decorate(swa_layer, level="O2", dtype="bfloat16")

        hidden_states = paddle.randn(
            [self.batch_size, self.seq_len, self.config.hidden_size],
            dtype="bfloat16",
        )
        attn_mask_startend_row_indices = paddle.full(
            [self.batch_size, 1, self.seq_len, 1], self.seq_len, dtype="int32"
        )

        out_dict = full_layer(
            {
                "hidden_states": hidden_states,
                "attn_mask_startend_row_indices": attn_mask_startend_row_indices,
            }
        )

        self.assertTrue("shared_key" in out_dict)
        self.assertTrue("shared_block_indices" in out_dict)

        out_dict = swa_layer(out_dict)

        self.assertTrue("shared_key" in out_dict)
        self.assertFalse(out_dict["shared_key"].stop_gradient)


if __name__ == "__main__":
    unittest.main()
