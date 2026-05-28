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

from paddlefleet.models.gpt.gpt_layer_specs import (
    get_attention_spec,
    get_gpt_decoder_layers_spec,
    get_gpt_layer_local_spec,
    get_gpt_mtp_layers_spec,
)
from paddlefleet.tilelang_ops import csa_sparse_attn
from paddlefleet.transformer.csa_attention import (
    CompressedSparseAttention,
    _resolve_csa_indexer_attn_topk_effective,
    _resolve_csa_indexer_loss_topk_effective,
    _resolve_csa_tilelang_switch,
    get_compress_topk_idxs,
    get_window_topk_idxs,
    unfused_compressed_sparse_attn,
)
from paddlefleet.transformer.dsa_attention import (
    fused_qk_topk_naive,
)
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
    rope_type="rope",
    apply_rope_fusion=False,
    multi_latent_attention=True,
    num_nextn_predict_layers=0,
    csa_tilelang_backend=None,
    csa_tilelang_enable_indexer=None,
    csa_tilelang_enable_sparse_attn=None,
):
    if csa_compress_ratios is None:
        csa_compress_ratios = [0, 4, 128, 4]

    return TransformerConfig(
        num_hidden_layers=num_layers,
        num_nextn_predict_layers=num_nextn_predict_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        params_dtype=paddle.bfloat16,
        bf16=True,
        use_bias=False,
        multi_latent_attention=multi_latent_attention,
        experimental_attention_variant="dsv4_hybrid",
        q_lora_rank=q_lora_rank,
        kv_lora_rank=v_head_dim - qk_pos_emb_head_dim,
        qk_nope_head_dim=v_head_dim - qk_pos_emb_head_dim,
        qk_rope_head_dim=qk_pos_emb_head_dim,
        qk_pos_emb_head_dim=qk_pos_emb_head_dim,
        v_head_dim=v_head_dim,
        o_groups=o_groups,
        o_lora_rank=o_lora_rank,
        rope_type=rope_type,
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
        apply_rope_fusion=apply_rope_fusion,
        attention_dropout=0.0,
        attention_softmax_in_fp32=True,
        masked_softmax_fusion=False,
        softmax_type="vanilla",
        csa_tilelang_backend=csa_tilelang_backend,
        csa_tilelang_enable_indexer=csa_tilelang_enable_indexer,
        csa_tilelang_enable_sparse_attn=csa_tilelang_enable_sparse_attn,
    )


def _build_attention(config, layer_number):
    spec = get_attention_spec(
        config=config,
        attention_layer_type="dsv4_hybrid_attention",
        attn_mask_type=AttnMaskType.causal,
    )
    return build_spec_layer(spec, config=config, layer_number=layer_number)


class TestDSv4HybridConfigAndSpec(unittest.TestCase):
    def test_gpt_layer_local_spec_routes_to_dsv4_hybrid_attention(self):
        config = _make_config()
        spec = get_gpt_layer_local_spec(
            config=config,
            multi_latent_attention=False,
            normalization=config.normalization,
        )

        self_attn_spec = spec.sublayers_spec.self_attn
        self.assertIs(self_attn_spec.layer, DSv4HybridSelfAttention)

    def test_config_validation_errors(self):
        with self.assertRaisesRegex(
            ValueError, "csa_compress_ratios to be set"
        ):
            TransformerConfig(
                num_hidden_layers=1,
                hidden_size=256,
                num_attention_heads=8,
                params_dtype=paddle.bfloat16,
                bf16=True,
                multi_latent_attention=True,
                experimental_attention_variant="dsv4_hybrid",
            )

        with self.assertRaisesRegex(ValueError, "must equal num_hidden_layers"):
            _make_config(num_layers=2, csa_compress_ratios=[0])

        with self.assertRaisesRegex(ValueError, "is invalid"):
            _make_config(num_layers=1, csa_compress_ratios=[2])

    def test_csa_tilelang_backend_switches_and_overrides(self):
        paddle_config = _make_config()
        self.assertFalse(
            _resolve_csa_tilelang_switch(
                paddle_config, "csa_tilelang_enable_indexer"
            )
        )
        self.assertFalse(
            _resolve_csa_tilelang_switch(
                paddle_config, "csa_tilelang_enable_sparse_attn"
            )
        )

        tilelang_config = _make_config(
            csa_tilelang_backend="attention_paddle_compat"
        )
        self.assertTrue(
            _resolve_csa_tilelang_switch(
                tilelang_config, "csa_tilelang_enable_indexer"
            )
        )
        self.assertTrue(
            _resolve_csa_tilelang_switch(
                tilelang_config, "csa_tilelang_enable_sparse_attn"
            )
        )

        override_config = _make_config(
            csa_tilelang_backend="attention_paddle_compat",
            csa_tilelang_enable_indexer=False,
            csa_tilelang_enable_sparse_attn=False,
        )
        self.assertFalse(
            _resolve_csa_tilelang_switch(
                override_config, "csa_tilelang_enable_indexer"
            )
        )
        self.assertFalse(
            _resolve_csa_tilelang_switch(
                override_config, "csa_tilelang_enable_sparse_attn"
            )
        )

        with self.assertRaisesRegex(
            ValueError, "csa_tilelang_enable_indexer=True requires"
        ):
            _make_config(csa_tilelang_enable_indexer=True)

        with self.assertRaisesRegex(
            ValueError, "csa_tilelang_enable_sparse_attn=True requires"
        ):
            _make_config(csa_tilelang_enable_sparse_attn=True)

    def test_phase2_loss_topk_does_not_expand_attention_topk(self):
        config = _make_config(
            dsa_index_topk=2,
        )
        n_compressed = 8

        self.assertEqual(
            _resolve_csa_indexer_loss_topk_effective(
                config, config.dsa_index_topk, n_compressed
            ),
            n_compressed,
        )
        self.assertEqual(
            _resolve_csa_indexer_attn_topk_effective(
                config.dsa_index_topk, n_compressed
            ),
            config.dsa_index_topk,
        )

        config.dsa_indexer_use_sparse_loss = True
        self.assertEqual(
            _resolve_csa_indexer_loss_topk_effective(
                config, config.dsa_index_topk, n_compressed
            ),
            config.dsa_index_topk,
        )


class TestCSAIndexHelpers(unittest.TestCase):
    def test_window_and_compress_indices(self):
        window = get_window_topk_idxs(
            window_size=3,
            batch_size=2,
            seqlen=4,
        )
        self.assertEqual(list(window.shape), [2, 4, 3])
        self.assertEqual(
            window.numpy().tolist()[0],
            [[0, -1, -1], [0, 1, -1], [0, 1, 2], [1, 2, 3]],
        )

        compressed = get_compress_topk_idxs(
            ratio=4,
            batch_size=2,
            seqlen=8,
            offset=8,
        )
        self.assertEqual(list(compressed.shape), [2, 8, 2])
        self.assertEqual(
            compressed.numpy().tolist()[0],
            [
                [-1, -1],
                [-1, -1],
                [-1, -1],
                [8, -1],
                [8, -1],
                [8, -1],
                [8, -1],
                [8, 9],
            ],
        )

    def test_fused_qk_topk_naive_with_mask(self):
        q = paddle.ones([1, 2, 1, 2], dtype="bfloat16")
        k = paddle.to_tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype="bfloat16")
        weights = paddle.ones([1, 2, 1], dtype="float32")
        mask = paddle.to_tensor(
            [[[0.0, float("-inf")], [0.0, 0.0]]], dtype="float32"
        )

        index_scores, topk_indices = fused_qk_topk_naive(q, k, weights, 2, mask)

        self.assertEqual(list(index_scores.shape), [1, 2, 2])
        self.assertEqual(list(topk_indices.shape), [1, 2, 2])
        self.assertEqual(topk_indices.numpy().tolist()[0][0][0], 0)


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

        for layer_number, ratio in enumerate(ratios):
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

    def test_mtp_layer_uses_nextn_compress_ratio(self):
        ratios = [0, 4, 128, 4, 128]
        config = _make_config(
            num_layers=4,
            num_nextn_predict_layers=1,
            csa_compress_ratios=ratios,
        )
        spec = get_attention_spec(
            config=config,
            attention_layer_type="dsv4_hybrid_attention",
            attn_mask_type=AttnMaskType.causal,
            is_mtp_layer=True,
        )
        attn = build_spec_layer(spec, config=config, layer_number=0)

        self.assertEqual(
            attn.core_attention.compress_ratio, ratios[config.num_hidden_layers]
        )

    def test_non_dense_mtp_spec_uses_mtp_attention_ratio(self):
        ratios = [0, 4, 128, 4, 128]
        config = _make_config(
            num_layers=4,
            num_nextn_predict_layers=1,
            csa_compress_ratios=ratios,
        )
        decoder_specs = get_gpt_decoder_layers_spec(
            config=config,
            normalization=config.normalization,
        )
        mtp_specs = get_gpt_mtp_layers_spec(config=config, spec=decoder_specs)
        mtp_self_attn_spec = mtp_specs[
            0
        ].sublayers_spec.transformer_layer.sublayers_spec.self_attn
        attn = build_spec_layer(
            mtp_self_attn_spec,
            config=config,
            layer_number=0,
        )

        self.assertEqual(
            attn.core_attention.compress_ratio, ratios[config.num_hidden_layers]
        )

    def test_yarn_rope_construction(self):
        config = _make_config(rope_type="yarn")
        attn = _build_attention(config, layer_number=1)
        freqs, mscale = attn.rotary_pos_emb(8, packed_seq=False)

        self.assertEqual(
            list(freqs.shape), [1, 8, 1, config.qk_pos_emb_head_dim]
        )
        self.assertIsInstance(mscale, float)

    def test_unsupported_rope_type_raises(self):
        config = _make_config(rope_type="invalid")
        with self.assertRaisesRegex(ValueError, "Unsupported rope_type"):
            _build_attention(config, layer_number=1)

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


class TestDSv4HybridFusedSparseAttention(unittest.TestCase):
    def test_fused_matches_unfused_forward_backward(self):
        old_flag = paddle.get_flags(["FLAGS_cudnn_deterministic"])[
            "FLAGS_cudnn_deterministic"
        ]
        paddle.set_flags({"FLAGS_cudnn_deterministic": 0})
        try:
            paddle.seed(_SEED)
            batch_size = 1
            seq_len = 128
            num_heads = 16
            head_dim = 128
            topk = 64
            softmax_scale = head_dim**-0.5

            query = paddle.randn(
                [batch_size, seq_len, num_heads, head_dim],
                dtype=paddle.bfloat16,
            )
            kv_full = paddle.randn(
                [batch_size, seq_len, head_dim], dtype=paddle.bfloat16
            )
            attn_sink = paddle.randn([num_heads], dtype=paddle.float32)
            topk_idxs = (
                paddle.arange(topk, dtype="int32")
                .reshape([1, 1, topk])
                .expand([batch_size, seq_len, topk])
            )

            query.stop_gradient = False
            kv_full.stop_gradient = False
            attn_sink.stop_gradient = False
            fused_out = csa_sparse_attn(
                query, kv_full, attn_sink, topk_idxs, softmax_scale
            )
            fused_loss = fused_out.cast("float32").sum()
            fused_loss.backward()
            fused_query_grad = query.grad.clone()
            fused_kv_grad = kv_full.grad.clone()
            fused_attn_sink_grad = attn_sink.grad.clone()

            query.clear_gradient()
            kv_full.clear_gradient()
            attn_sink.clear_gradient()
            unfused_out = unfused_compressed_sparse_attn(
                query, kv_full, attn_sink, topk_idxs, softmax_scale
            )
            unfused_loss = unfused_out.cast("float32").sum()
            unfused_loss.backward()

            self.assertTrue(
                paddle.allclose(
                    fused_out.cast("float32"),
                    unfused_out.cast("float32"),
                    rtol=1e-2,
                    atol=1e-2,
                ).item()
            )
            self.assertTrue(
                paddle.allclose(
                    fused_query_grad.cast("float32"),
                    query.grad.cast("float32"),
                    rtol=1e-2,
                    atol=1e-2,
                ).item()
            )
            self.assertTrue(
                paddle.allclose(
                    fused_kv_grad.cast("float32"),
                    kv_full.grad.cast("float32"),
                    rtol=1e-2,
                    atol=1e-2,
                ).item()
            )
            self.assertTrue(
                paddle.allclose(
                    fused_attn_sink_grad.cast("float32"),
                    attn_sink.grad.cast("float32"),
                    rtol=1e-2,
                    atol=1e-2,
                ).item()
            )
        finally:
            paddle.set_flags({"FLAGS_cudnn_deterministic": old_flag})


class TestDSv4HybridAttentionForwardBackward(unittest.TestCase):
    def setUp(self):
        paddle.seed(_SEED)
        self.config = _make_config(dsa_indexer_loss_coeff=1.0)

    def test_forward_output_shape(self):
        batch_size = 2
        seq_len = 64

        for layer_number in [0, 1, 2, 3]:
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

        for layer_number in [0, 1]:
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
            used_params = [
                name
                for name, param in attn.named_parameters()
                if not param.stop_gradient and param.grad is not None
            ]
            self.assertGreater(len(used_params), 0)
            for name, param in attn.named_parameters():
                if not param.stop_gradient and param.grad is not None:
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
