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

import paddle
from paddle.distributed.fleet.meta_parallel import build_spec_layer

from paddlefleet.models.gpt.gpt_layer_specs import get_attention_spec
from paddlefleet.transformer.csa_attention import CompressedSparseAttention
from paddlefleet.transformer.dsv4_hybrid_attention import (
    DSv4HybridSelfAttention,
)
from paddlefleet.transformer.enums import AttnMaskType
from paddlefleet.transformer.transformer_config import TransformerConfig

_SEED = 42


def _make_config(
    num_layers=4,
    hidden_size=256,
    num_attention_heads=8,
    v_head_dim=32,
    qk_pos_emb_head_dim=16,
    q_lora_rank=64,
    o_groups=4,
    o_lora_rank=32,
    csa_compress_ratios=None,
    csa_window_size=16,
    dsa_index_n_heads=4,
    dsa_index_head_dim=32,
    dsa_index_topk=8,
    dsa_indexer_loss_coeff=1.0,
):
    if csa_compress_ratios is None:
        csa_compress_ratios = [0, 4, 128, 4]

    return TransformerConfig(
        num_hidden_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        params_dtype=paddle.bfloat16,
        bf16=True,
        use_bias=False,
        multi_latent_attention=True,
        experimental_attention_variant="dsv4_hybrid",
        q_lora_rank=q_lora_rank,
        kv_lora_rank=v_head_dim - qk_pos_emb_head_dim,
        qk_nope_head_dim=v_head_dim - qk_pos_emb_head_dim,
        qk_rope_head_dim=qk_pos_emb_head_dim,
        qk_pos_emb_head_dim=qk_pos_emb_head_dim,
        v_head_dim=v_head_dim,
        o_groups=o_groups,
        o_lora_rank=o_lora_rank,
        rope_type="rope",
        rotary_base=10000.0,
        rotary_percent=1.0,
        normalization="RMSNorm",
        use_qk_norm=True,
        csa_compress_ratios=csa_compress_ratios,
        csa_window_size=csa_window_size,
        dsa_index_n_heads=dsa_index_n_heads,
        dsa_index_head_dim=dsa_index_head_dim,
        dsa_index_topk=dsa_index_topk,
        dsa_indexer_loss_coeff=dsa_indexer_loss_coeff,
        dsa_indexer_use_sparse_loss=False,
        dsa_indexer_rotary_interleaved=False,
        apply_rope_fusion=False,
        attention_dropout=0.0,
        attention_softmax_in_fp32=True,
        masked_softmax_fusion=False,
        softmax_type="vanilla",
    )


def _build_attention(config, layer_number):
    spec = get_attention_spec(
        config=config,
        attention_layer_type="dsv4_hybrid_attention",
        attn_mask_type=AttnMaskType.causal,
    )
    return build_spec_layer(spec, config=config, layer_number=layer_number)


class TestDSv4HybridAttentionConstructor(unittest.TestCase):
    def test_basic_construction(self):
        paddle.seed(_SEED)
        config = _make_config()
        attn = _build_attention(config, layer_number=1)

        self.assertIsInstance(attn, DSv4HybridSelfAttention)
        self.assertTrue(hasattr(attn, "linear_q_down_proj"))
        self.assertTrue(hasattr(attn, "linear_q_up_proj"))
        self.assertTrue(hasattr(attn, "linear_kv_proj"))
        self.assertTrue(hasattr(attn, "o_proj"))
        self.assertTrue(hasattr(attn, "linear_o_group_proj"))
        self.assertTrue(hasattr(attn, "core_attention"))
        self.assertTrue(hasattr(attn, "q_layernorm"))
        self.assertTrue(hasattr(attn, "kv_layernorm"))

    def test_q_head_dim_equals_v_head_dim(self):
        paddle.seed(_SEED)
        config = _make_config()
        attn = _build_attention(config, layer_number=1)

        self.assertEqual(attn.q_head_dim, config.v_head_dim)

    def test_rope_base_varies_with_compress_ratio(self):
        paddle.seed(_SEED)
        ratios = [0, 4, 128, 4]
        config = _make_config(csa_compress_ratios=ratios)

        for layer_number, ratio in enumerate(ratios, start=1):
            attn = _build_attention(config, layer_number=layer_number)
            self.assertIsInstance(
                attn.core_attention, CompressedSparseAttention
            )
            self.assertEqual(attn.core_attention.compress_ratio, ratio)

            expected_base = (
                config.csa_compress_rotary_base
                if ratio > 1
                else config.rotary_base
            )
            dim = config.qk_pos_emb_head_dim
            expected_inv_freq = 1.0 / (
                expected_base
                ** (paddle.arange(0, dim, 2, dtype="float32") / dim)
            )
            self.assertTrue(
                paddle.allclose(
                    attn.rotary_pos_emb.inv_freq.cast("float32"),
                    expected_inv_freq,
                    rtol=1e-5,
                    atol=1e-5,
                ).item()
            )

    def test_o_group_proj_shape(self):
        paddle.seed(_SEED)
        o_groups = 4
        o_lora_rank = 32
        config = _make_config(o_groups=o_groups, o_lora_rank=o_lora_rank)
        attn = _build_attention(config, layer_number=1)

        expected_out = o_groups * o_lora_rank
        expected_in = (
            config.v_head_dim * config.num_attention_heads
        ) // o_groups
        self.assertEqual(
            list(attn.linear_o_group_proj.shape), [expected_out, expected_in]
        )
        self.assertFalse(attn.linear_o_group_proj.stop_gradient)


class TestDSv4HybridAttentionForwardBackward(unittest.TestCase):
    def setUp(self):
        paddle.seed(_SEED)
        self.config = _make_config(dsa_indexer_loss_coeff=1.0)

    def test_forward_output_shape(self):
        batch_size = 2
        seq_len = 64

        for layer_number in [1, 2, 3, 4]:
            attn = _build_attention(self.config, layer_number=layer_number)
            hidden = paddle.randn(
                [batch_size, seq_len, self.config.hidden_size],
                dtype=paddle.bfloat16,
            )

            output, bias = attn(hidden_states=hidden, attention_mask=None)

            self.assertEqual(
                list(output.shape),
                [batch_size, seq_len, self.config.hidden_size],
            )
            self.assertEqual(output.dtype, paddle.bfloat16)
            self.assertTrue(
                paddle.isfinite(output.cast("float32")).all().item()
            )
            self.assertIsNone(bias)

    def test_backward_gradient_flow(self):
        batch_size = 2
        seq_len = 64

        for layer_number in [1, 2]:
            attn = _build_attention(self.config, layer_number=layer_number)
            attn.train()
            hidden = paddle.randn(
                [batch_size, seq_len, self.config.hidden_size],
                dtype=paddle.bfloat16,
            )
            hidden.stop_gradient = False

            output, _ = attn(hidden_states=hidden, attention_mask=None)
            loss = output.cast("float32").sum()
            loss.backward()

            self.assertIsNotNone(hidden.grad)
            self.assertTrue(
                paddle.isfinite(hidden.grad.cast("float32")).all().item()
            )
            for name, param in attn.named_parameters():
                if not param.stop_gradient:
                    self.assertIsNotNone(
                        param.grad, f"No gradient for parameter {name}"
                    )
                    self.assertTrue(
                        paddle.isfinite(param.grad.cast("float32"))
                        .all()
                        .item(),
                        f"Non-finite gradient for parameter {name}",
                    )

    def test_eval_mode(self):
        batch_size = 2
        seq_len = 64
        attn = _build_attention(self.config, layer_number=1)
        attn.eval()
        hidden = paddle.randn(
            [batch_size, seq_len, self.config.hidden_size],
            dtype=paddle.bfloat16,
        )

        with paddle.no_grad():
            output, bias = attn(hidden_states=hidden, attention_mask=None)

        self.assertEqual(
            list(output.shape), [batch_size, seq_len, self.config.hidden_size]
        )
        self.assertTrue(paddle.isfinite(output.cast("float32")).all().item())
        self.assertIsNone(bias)

    def test_different_seq_lengths(self):
        batch_size = 2
        attn = _build_attention(self.config, layer_number=2)

        for seq_len in [32, 64, 128]:
            hidden = paddle.randn(
                [batch_size, seq_len, self.config.hidden_size],
                dtype=paddle.bfloat16,
            )
            output, _ = attn(hidden_states=hidden, attention_mask=None)
            self.assertEqual(
                list(output.shape),
                [batch_size, seq_len, self.config.hidden_size],
            )
            self.assertTrue(
                paddle.isfinite(output.cast("float32")).all().item()
            )


class TestDSv4HybridQKV(unittest.TestCase):
    def setUp(self):
        paddle.seed(_SEED)
        self.config = _make_config(dsa_indexer_loss_coeff=0.0)

    def test_qkv_shapes(self):
        batch_size = 2
        seq_len = 64
        attn = _build_attention(self.config, layer_number=1)
        hidden = paddle.randn(
            [batch_size, seq_len, self.config.hidden_size],
            dtype=paddle.bfloat16,
        )

        q, k, v, q_compressed, kv_compressed = attn.get_query_key_value_tensors(
            hidden
        )

        self.assertEqual(
            list(q.shape),
            [
                batch_size,
                seq_len,
                self.config.num_attention_heads,
                self.config.v_head_dim,
            ],
        )
        self.assertEqual(
            list(k.shape), [batch_size, seq_len, 1, self.config.v_head_dim]
        )
        self.assertEqual(
            list(v.shape), [batch_size, seq_len, 1, self.config.v_head_dim]
        )
        self.assertEqual(
            list(q_compressed.shape),
            [batch_size, seq_len, self.config.q_lora_rank],
        )
        self.assertEqual(list(kv_compressed.shape), list(hidden.shape))

    def test_key_equals_value(self):
        batch_size = 2
        seq_len = 64
        attn = _build_attention(self.config, layer_number=1)
        hidden = paddle.randn(
            [batch_size, seq_len, self.config.hidden_size],
            dtype=paddle.bfloat16,
        )

        _, key, value, _, _ = attn.get_query_key_value_tensors(hidden)
        self.assertTrue(
            paddle.equal_all(key.cast("float32"), value.cast("float32")).item()
        )


if __name__ == "__main__":
    unittest.main()
