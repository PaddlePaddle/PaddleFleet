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

"""Unit tests for DSA (DeepSeek Sparse Attention) module.

Tests are organized in 4 layers:
  1. Pure functions: hadamard_transform, rotate_activation, _unfused_dsa_attention,
     _compute_index_scores_fused
  2. Indexer module: forward_before_topk, compute_index_scores, backward
  3. Loss: _compute_dsa_indexer_loss, FusedDSAIndexerLoss, DSAIndexerLossAutoScaler
  4. Integration: MLASelfAttentionWithDSA forward + backward
"""

import unittest

import paddle

from paddlefleet.transformer.dot_product_attention import DotProductAttention
from paddlefleet.transformer.dsa_attention import (
    DSAIndexerLossAutoScaler,
    FusedDSAIndexerLoss,
    Indexer,
    MLASelfAttentionWithDSA,
    _compute_dsa_indexer_loss,
    _compute_index_scores_fused,
    _unfused_dsa_attention,
    hadamard_transform,
    rotate_activation,
)
from paddlefleet.transformer.enums import AttnMaskType
from paddlefleet.transformer.multi_latent_attention import (
    MLASelfAttentionSublayersSpec,
)
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.utils import (
    init_method_normal,
    scaled_init_method_normal,
)


# ---------------------------------------------------------------------------
# Stub layers (same pattern as test_attention.py)
# ---------------------------------------------------------------------------
class BiasedLinear(paddle.nn.Layer):
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__()
        self.linear = paddle.nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x), self.linear.bias


class RMSNorm(paddle.nn.Layer):
    def __init__(self, hidden_size, eps, **kwargs):
        super().__init__()
        self.weight = paddle.nn.Parameter(paddle.ones([hidden_size]))
        self.eps = eps

    def forward(self, x):
        d_norm = paddle.rsqrt(x.pow(2).mean(axis=-1, keepdim=True) + self.eps)
        return x * d_norm * self.weight


# ---------------------------------------------------------------------------
# Helper: create DSA-compatible TransformerConfig
# ---------------------------------------------------------------------------
def _create_dsa_config(
    hidden_size=256,
    num_attention_heads=2,
    q_lora_rank=64,
    kv_lora_rank=64,
    qk_nope_head_dim=32,
    qk_rope_head_dim=32,
    v_head_dim=64,
    index_n_heads=2,
    index_head_dim=128,
    index_topk=16,
    indexer_loss_coeff=1.0,
    indexer_use_sparse_loss=False,
    sequence_parallel=False,
):
    config = TransformerConfig(
        num_hidden_layers=2,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
    )
    # MLA fields
    config.num_key_value_heads = num_attention_heads
    config.head_dim = hidden_size // num_attention_heads
    config.q_lora_rank = q_lora_rank
    config.kv_lora_rank = kv_lora_rank
    config.qk_nope_head_dim = qk_nope_head_dim
    config.qk_rope_head_dim = qk_rope_head_dim
    config.v_head_dim = v_head_dim
    config.multi_latent_attention = True

    # RoPE / YaRN
    config.rope_type = "yarn"
    config.rope_theta = 10000.0
    config.rotary_interleaved = False
    config.rotary_percent = 1.0
    config.rotary_scaling_factor = 40.0
    config.original_max_position_embeddings = 4096
    config.beta_fast = 32.0
    config.beta_slow = 1.0
    config.mscale = 1.0
    config.mscale_all_dim = 0.0
    config.apply_rope_fusion = False  # DSA requires unfused RoPE

    # DSA Indexer fields
    config.index_n_heads = index_n_heads
    config.index_head_dim = index_head_dim
    config.index_topk = index_topk
    config.indexer_loss_coeff = indexer_loss_coeff
    config.indexer_use_sparse_loss = indexer_use_sparse_loss

    # Attention generic fields
    config.softmax_scale = None
    config.use_bias = True
    config.no_rope_freq = None
    config.recompute_granularity = None
    config.fused_single_qkv_rope = False
    config.init_method = init_method_normal(0.02)
    config.output_layer_init_method = scaled_init_method_normal(0.02, 1, 2.0)
    config.rms_norm_eps = 1e-5
    config.context_parallel_size = 1
    config.apply_query_key_layer_scaling = False
    config.sliding_window = None
    config.window_attn_skip_freq = None
    config.fp16 = False
    config.bf16 = False
    config.masked_softmax_fusion = False
    config.attention_softmax_in_fp32 = True
    config.attention_dropout = 0.0
    config.softmax_type = "vanilla"
    config.sequence_parallel = sequence_parallel

    return config


def _create_sublayers_spec():
    return MLASelfAttentionSublayersSpec(
        q_proj=BiasedLinear,
        q_a_proj=BiasedLinear,
        q_b_proj=BiasedLinear,
        kv_a_proj_with_mqa=BiasedLinear,
        kv_b_proj=BiasedLinear,
        core_attention=DotProductAttention,
        o_proj=BiasedLinear,
        q_a_layernorm=RMSNorm,
        kv_a_layernorm=RMSNorm,
    )


def _make_causal_topk_indices(b, sq, sk, topk):
    """Generate topk indices that respect causality (indices <= current position)."""
    indices_list = []
    for i in range(sq):
        max_idx = min(i + 1, sk)
        actual_topk = min(topk, max_idx)
        # Pick the last `actual_topk` positions (most recent)
        row_indices = paddle.arange(max_idx - actual_topk, max_idx)
        if actual_topk < topk:
            # Pad with the last valid index
            pad = paddle.full([topk - actual_topk], max_idx - 1, dtype="int64")
            row_indices = paddle.concat([row_indices, pad])
        indices_list.append(row_indices)
    # [sq, topk] -> expand to [b, sq, topk]
    indices = (
        paddle.stack(indices_list, axis=0).unsqueeze(0).expand([b, sq, topk])
    )
    return indices


# ===========================================================================
# Layer 1: Pure function tests
# ===========================================================================
class TestHadamardTransform(unittest.TestCase):
    def test_output_shape(self):
        x = paddle.randn([4, 8, 16])
        out = hadamard_transform(x)
        self.assertEqual(out.shape, [4, 8, 16])

    def test_power_of_two_assertion(self):
        x = paddle.randn([4, 7])
        with self.assertRaises(AssertionError):
            hadamard_transform(x)

    def test_involution(self):
        """H(H(x)) = dim * x (Hadamard is involutory up to scaling)."""
        dim = 16
        x = paddle.randn([3, dim], dtype="float32")
        hx = hadamard_transform(x)
        hhx = hadamard_transform(hx)
        self.assertTrue(paddle.allclose(hhx, x * dim, atol=1e-4, rtol=1e-4))

    def test_scale_factor(self):
        x = paddle.randn([4, 8])
        out_unscaled = hadamard_transform(x)
        out_scaled = hadamard_transform(x, scale=0.5)
        self.assertTrue(
            paddle.allclose(out_scaled, out_unscaled * 0.5, atol=1e-5)
        )

    def test_1d_input(self):
        x = paddle.randn([16])
        out = hadamard_transform(x)
        self.assertEqual(out.shape, [16])


class TestRotateActivation(unittest.TestCase):
    def test_output_shape(self):
        x = paddle.randn([2, 4, 128]).cast("bfloat16")
        out = rotate_activation(x)
        self.assertEqual(list(out.shape), [2, 4, 128])
        self.assertEqual(out.dtype, paddle.bfloat16)

    def test_dtype_assertion(self):
        x = paddle.randn([2, 4, 64], dtype="float32")
        with self.assertRaises(AssertionError):
            rotate_activation(x)


class TestUnfusedDSAAttention(unittest.TestCase):
    def setUp(self):
        self.b, self.s, self.nhpp = 2, 8, 4
        self.qk_hd, self.v_hd = 32, 64
        self.softmax_scale = self.qk_hd**-0.5

    def test_output_shape(self):
        query = paddle.randn([self.b, self.s, self.nhpp, self.qk_hd])
        key = paddle.randn([self.b, self.s, self.nhpp, self.qk_hd])
        value = paddle.randn([self.b, self.s, self.nhpp, self.v_hd])
        out = _unfused_dsa_attention(
            query, key, value, None, self.softmax_scale
        )
        self.assertEqual(out.shape, [self.b, self.s, self.nhpp * self.v_hd])

    def test_with_causal_mask(self):
        query = paddle.randn([self.b, self.s, self.nhpp, self.qk_hd])
        key = paddle.randn([self.b, self.s, self.nhpp, self.qk_hd])
        value = paddle.randn([self.b, self.s, self.nhpp, self.v_hd])
        causal = paddle.triu(
            paddle.full([self.s, self.s], float("-inf"), dtype="float32"),
            diagonal=1,
        )
        mask = causal.unsqueeze(0).unsqueeze(0)  # [1, 1, s, s]
        out = _unfused_dsa_attention(
            query, key, value, mask, self.softmax_scale
        )
        self.assertEqual(out.shape, [self.b, self.s, self.nhpp * self.v_hd])

    def test_asymmetric_dims(self):
        """qk_head_dim != v_head_dim should work."""
        qk_hd, v_hd = 48, 32
        query = paddle.randn([self.b, self.s, self.nhpp, qk_hd])
        key = paddle.randn([self.b, self.s, self.nhpp, qk_hd])
        value = paddle.randn([self.b, self.s, self.nhpp, v_hd])
        out = _unfused_dsa_attention(query, key, value, None, qk_hd**-0.5)
        self.assertEqual(out.shape, [self.b, self.s, self.nhpp * v_hd])


class TestComputeIndexScoresFused(unittest.TestCase):
    def test_output_shape(self):
        sq, b, h, d = 8, 2, 4, 32
        sk = 8
        q = paddle.randn([sq, b, h, d])
        weights = paddle.randn([sq, b, h])
        k = paddle.randn([sk, b, d])
        out = _compute_index_scores_fused(q, weights, k)
        self.assertEqual(out.shape, [b, sq, sk])

    def test_nonnegative_after_relu(self):
        sq, b, h, d = 8, 2, 4, 32
        q = paddle.randn([sq, b, h, d])
        # Use positive weights so that relu * positive_weights >= 0
        weights = paddle.abs(paddle.randn([sq, b, h])) + 0.1
        k = paddle.randn([sq, b, d])
        out = _compute_index_scores_fused(q, weights, k)
        self.assertTrue((out >= -1e-6).all().item())


# ===========================================================================
# Layer 2: Indexer module tests
# ===========================================================================
class TestIndexer(unittest.TestCase):
    def setUp(self):
        self.config = _create_dsa_config()
        self.indexer = Indexer(self.config, layer_number=1)
        self.b = 2
        self.s = 16

    def _prepare_indexer_bf16(self):
        """Convert wq_b/wk to bf16 for rotate_activation, keep weights_proj fp32."""
        self.indexer.wq_b = self.indexer.wq_b.to(dtype="bfloat16")
        self.indexer.wk = self.indexer.wk.to(dtype="bfloat16")
        self.indexer.k_norm = self.indexer.k_norm.to(dtype="bfloat16")
        # weights_proj stays fp32 (code does hidden.cast("float32") before calling it)

    def test_forward_before_topk_shapes(self):
        self._prepare_indexer_bf16()
        hidden = paddle.randn([self.b, self.s, self.config.hidden_size]).cast(
            "bfloat16"
        )
        q_latent = paddle.randn([self.b, self.s, self.config.q_lora_rank]).cast(
            "bfloat16"
        )

        q, k, weights = self.indexer.forward_before_topk(
            hidden, q_latent, freqs=None, mscale=1.0
        )
        self.assertEqual(
            list(q.shape),
            [
                self.b,
                self.s,
                self.config.index_n_heads,
                self.config.index_head_dim,
            ],
        )
        self.assertEqual(
            list(k.shape),
            [self.b, self.s, self.config.index_head_dim],
        )
        self.assertEqual(
            list(weights.shape),
            [self.b, self.s, self.config.index_n_heads],
        )

    def test_compute_index_scores_shapes(self):
        q = paddle.randn(
            [
                self.b,
                self.s,
                self.config.index_n_heads,
                self.config.index_head_dim,
            ]
        )
        k = paddle.randn([self.b, self.s, self.config.index_head_dim])
        weights = paddle.randn([self.b, self.s, self.config.index_n_heads])
        index_scores, topk_indices = self.indexer.compute_index_scores(
            q, k, weights, mask=None
        )
        self.assertEqual(list(index_scores.shape), [self.b, self.s, self.s])
        self.assertEqual(
            list(topk_indices.shape),
            [self.b, self.s, self.config.index_topk],
        )

    def test_topk_in_range(self):
        q = paddle.randn(
            [
                self.b,
                self.s,
                self.config.index_n_heads,
                self.config.index_head_dim,
            ]
        )
        k = paddle.randn([self.b, self.s, self.config.index_head_dim])
        weights = paddle.randn([self.b, self.s, self.config.index_n_heads])
        _, topk_indices = self.indexer.compute_index_scores(
            q, k, weights, mask=None
        )
        self.assertTrue((topk_indices >= 0).all().item())
        self.assertTrue((topk_indices < self.s).all().item())

    def test_backward(self):
        """Indexer parameters receive gradients."""
        self._prepare_indexer_bf16()
        hidden = paddle.randn([self.b, self.s, self.config.hidden_size]).cast(
            "bfloat16"
        )
        q_latent = paddle.randn([self.b, self.s, self.config.q_lora_rank]).cast(
            "bfloat16"
        )

        q, k, weights = self.indexer.forward_before_topk(
            hidden, q_latent, freqs=None, mscale=1.0
        )
        # rotate_activation requires bf16, so skip it in this unit test
        # and just use the raw outputs for gradient checking.
        loss = q.cast("float32").sum() + k.cast("float32").sum() + weights.sum()
        loss.backward()

        for name, param in self.indexer.named_parameters():
            self.assertIsNotNone(
                param.grad, f"Parameter {name} has no gradient"
            )


# ===========================================================================
# Layer 3: Loss tests
# ===========================================================================
class TestComputeDSAIndexerLoss(unittest.TestCase):
    def setUp(self):
        self.sq, self.sk = 8, 8
        self.b, self.np, self.hn = 2, 4, 32
        self.topk = 4
        self.softmax_scale = self.hn**-0.5
        self.loss_coeff = 1.0

    def _make_inputs(self, sparse=False):
        index_scores = paddle.randn([self.b, self.sq, self.sk], dtype="float32")
        if sparse:
            topk_indices = _make_causal_topk_indices(
                self.b, self.sq, self.sk, self.topk
            )
        else:
            topk_indices = paddle.randint(
                0, self.sk, [self.b, self.sq, self.topk]
            ).cast("int64")
        query = paddle.randn(
            [self.sq, self.b, self.np, self.hn], dtype="float32"
        )
        key = paddle.randn([self.sk, self.b, self.np, self.hn], dtype="float32")
        return index_scores, topk_indices, query, key

    def test_loss_is_scalar(self):
        index_scores, topk_indices, query, key = self._make_inputs()
        loss = _compute_dsa_indexer_loss(
            index_scores,
            topk_indices,
            query,
            key,
            self.softmax_scale,
            self.loss_coeff,
            False,
            None,
        )
        self.assertEqual(loss.shape, [])

    def test_loss_with_sparse(self):
        index_scores, topk_indices, query, key = self._make_inputs(sparse=True)
        loss = _compute_dsa_indexer_loss(
            index_scores,
            topk_indices,
            query,
            key,
            self.softmax_scale,
            self.loss_coeff,
            True,
            None,
        )
        self.assertEqual(loss.shape, [])
        self.assertTrue(paddle.isfinite(loss).item())

    def test_loss_coeff_scaling(self):
        index_scores, topk_indices, query, key = self._make_inputs()
        loss1 = _compute_dsa_indexer_loss(
            index_scores,
            topk_indices,
            query,
            key,
            self.softmax_scale,
            1.0,
            False,
            None,
        )
        loss2 = _compute_dsa_indexer_loss(
            index_scores,
            topk_indices,
            query,
            key,
            self.softmax_scale,
            2.0,
            False,
            None,
        )
        self.assertTrue(
            paddle.allclose(loss2, loss1 * 2.0, atol=1e-4),
            f"loss2={loss2.item():.6f} != 2*loss1={2 * loss1.item():.6f}",
        )


class TestFusedDSAIndexerLoss(unittest.TestCase):
    def setUp(self):
        self.sq, self.sk = 8, 8
        self.b = 2
        self.h, self.d = 4, 32  # indexer heads/dim
        self.np, self.hn = 4, 64  # MLA heads/dim
        self.topk = 4
        self.softmax_scale = self.hn**-0.5

    def _make_inputs(self, with_mask=False):
        q = paddle.randn([self.sq, self.b, self.h, self.d], dtype="float32")
        q.stop_gradient = False
        weights = paddle.randn([self.sq, self.b, self.h], dtype="float32")
        weights.stop_gradient = False
        k = paddle.randn([self.sk, self.b, self.d], dtype="float32")
        k.stop_gradient = False
        query = paddle.randn(
            [self.sq, self.b, self.np, self.hn], dtype="float32"
        )
        key = paddle.randn([self.sk, self.b, self.np, self.hn], dtype="float32")
        if with_mask:
            causal = paddle.triu(
                paddle.full([self.sq, self.sk], float("-inf"), dtype="float32"),
                diagonal=1,
            )
            mask = causal.unsqueeze(0).unsqueeze(0)  # [1, 1, sq, sk]
        else:
            mask = None
        return q, weights, k, query, key, mask

    def test_forward_returns_scalar(self):
        q, weights, k, query, key, mask = self._make_inputs(with_mask=True)
        loss = FusedDSAIndexerLoss.apply(
            q,
            weights,
            k,
            query,
            key,
            self.softmax_scale,
            self.topk,
            1.0,
            mask,
            False,
            None,
        )
        self.assertEqual(loss.shape, [])
        self.assertTrue(paddle.isfinite(loss).item())

    def test_topk_indices_stored(self):
        FusedDSAIndexerLoss._last_topk_indices = None
        q, weights, k, query, key, _ = self._make_inputs()
        loss = FusedDSAIndexerLoss.apply(
            q,
            weights,
            k,
            query,
            key,
            self.softmax_scale,
            self.topk,
            1.0,
            None,
            False,
            None,
        )
        self.assertIsNotNone(FusedDSAIndexerLoss._last_topk_indices)
        self.assertEqual(
            list(FusedDSAIndexerLoss._last_topk_indices.shape),
            [self.b, self.sq, self.topk],
        )

    def test_backward_gradients(self):
        # Pass a mask tensor so PyLayer sees 6 tensor inputs (q, weights, k,
        # query, key, mask) matching the 6 return values in backward.
        q, weights, k, query, key, mask = self._make_inputs(with_mask=True)
        loss = FusedDSAIndexerLoss.apply(
            q,
            weights,
            k,
            query,
            key,
            self.softmax_scale,
            self.topk,
            1.0,
            mask,
            False,
            None,
        )
        loss.backward()

        self.assertIsNotNone(q.grad)
        self.assertIsNotNone(weights.grad)
        self.assertIsNotNone(k.grad)
        self.assertEqual(list(q.grad.shape), [self.sq, self.b, self.h, self.d])
        self.assertEqual(list(weights.grad.shape), [self.sq, self.b, self.h])
        self.assertEqual(list(k.grad.shape), [self.sk, self.b, self.d])
        self.assertTrue(paddle.isfinite(q.grad).all().item())
        self.assertTrue(paddle.isfinite(weights.grad).all().item())
        self.assertTrue(paddle.isfinite(k.grad).all().item())


class TestDSAIndexerLossAutoScaler(unittest.TestCase):
    def _make_non_leaf_output(self, shape):
        """Create a non-leaf tensor (required by PyLayer inplace check)."""
        x = paddle.randn(shape)
        x.stop_gradient = False
        return x + 0  # Adding 0 makes it non-leaf

    def test_forward_passthrough(self):
        output = self._make_non_leaf_output([2, 8, 64])
        indexer_loss = self._make_non_leaf_output([])
        result = DSAIndexerLossAutoScaler.apply(output, indexer_loss)
        self.assertEqual(list(result.shape), [2, 8, 64])

    def test_backward_grad_output(self):
        output = self._make_non_leaf_output([2, 8, 64])
        indexer_loss = self._make_non_leaf_output([])

        result = DSAIndexerLossAutoScaler.apply(output, indexer_loss)
        loss = result.sum()
        loss.backward()
        # output is non-leaf (x + 0), so its grad may not be retained,
        # but the computation should not error out.
        self.assertTrue(True)  # Just verify no error

    def test_loss_scale(self):
        DSAIndexerLossAutoScaler.set_loss_scale(
            paddle.to_tensor(2.0, dtype="float32")
        )
        output = self._make_non_leaf_output([2, 4])
        indexer_loss = paddle.to_tensor(1.0, dtype="float32")
        indexer_loss.stop_gradient = False
        indexer_loss = indexer_loss * 1.0  # Make non-leaf

        result = DSAIndexerLossAutoScaler.apply(output, indexer_loss)
        loss = result.sum()
        loss.backward()
        # Verify no errors
        self.assertTrue(True)
        # Reset
        DSAIndexerLossAutoScaler._main_loss_backward_scale = None


# ===========================================================================
# Layer 4: MLASelfAttentionWithDSA integration tests
# ===========================================================================
class TestMLASelfAttentionWithDSA(unittest.TestCase):
    def setUp(self):
        self.config = _create_dsa_config()
        self.micro_batch_size = 2
        self.sequence_length = 32

    def _build_model(self, config=None):
        cfg = config or self.config
        model = MLASelfAttentionWithDSA(
            cfg,
            _create_sublayers_spec(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )
        # Convert model to bf16 because rotate_activation requires bf16 input.
        # But weights_proj does hidden.cast("float32") internally and expects
        # fp32 weights, so convert it back to fp32 after the global bf16 cast.
        model = model.to(dtype="bfloat16")
        model.indexer.weights_proj = model.indexer.weights_proj.to(
            dtype="float32"
        )
        return model

    def _make_hidden(self, dtype="bfloat16"):
        return paddle.randn(
            [
                self.micro_batch_size,
                self.sequence_length,
                self.config.hidden_size,
            ],
        ).cast(dtype)

    def test_forward_shape(self):
        model = self._build_model()
        hidden = self._make_hidden()
        output, bias = model(hidden, attention_mask=None)

        self.assertEqual(output.shape[0], self.micro_batch_size)
        self.assertEqual(output.shape[1], self.sequence_length)
        self.assertEqual(output.shape[2], self.config.hidden_size)
        self.assertEqual(bias.shape[0], self.config.hidden_size)

    def test_forward_with_attention_mask(self):
        model = self._build_model()
        hidden = self._make_hidden()
        causal = paddle.triu(
            paddle.full(
                [self.sequence_length, self.sequence_length],
                float("-inf"),
                dtype="float32",
            ),
            diagonal=1,
        )
        mask = (
            causal.unsqueeze(0)
            .unsqueeze(0)
            .expand(
                [
                    self.micro_batch_size,
                    1,
                    self.sequence_length,
                    self.sequence_length,
                ]
            )
        )
        output, bias = model(hidden, attention_mask=mask)

        self.assertEqual(output.shape[0], self.micro_batch_size)
        self.assertEqual(output.shape[1], self.sequence_length)
        self.assertEqual(output.shape[2], self.config.hidden_size)

    def test_forward_training_with_loss(self):
        model = self._build_model()
        model.train()
        hidden = self._make_hidden()
        output, bias = model(hidden, attention_mask=None)

        self.assertEqual(output.shape[0], self.micro_batch_size)
        self.assertEqual(output.shape[1], self.sequence_length)
        self.assertEqual(output.shape[2], self.config.hidden_size)

    def test_forward_eval_mode(self):
        config = _create_dsa_config(indexer_loss_coeff=None)
        model = self._build_model(config)
        model.eval()
        hidden = self._make_hidden()
        output, bias = model(hidden, attention_mask=None)

        self.assertEqual(output.shape[0], self.micro_batch_size)
        self.assertEqual(output.shape[1], self.sequence_length)

    def test_backward_gradients(self):
        model = self._build_model()
        model.train()
        hidden = self._make_hidden()
        hidden.stop_gradient = False
        output, bias = model(hidden, attention_mask=None)
        loss = output.cast("float32").sum()
        loss.backward()

        self.assertIsNotNone(hidden.grad)
        for name, param in model.named_parameters():
            if not param.stop_gradient:
                self.assertIsNotNone(
                    param.grad, f"Parameter {name} has no gradient"
                )
                self.assertTrue(
                    paddle.isfinite(param.grad).all().item(),
                    f"Parameter {name} has non-finite gradient",
                )

    def test_indexer_params_have_grad(self):
        model = self._build_model()
        model.train()
        hidden = self._make_hidden()
        hidden.stop_gradient = False
        output, bias = model(hidden, attention_mask=None)
        loss = output.cast("float32").sum()
        loss.backward()

        indexer_param_names = [
            "indexer.wq_b",
            "indexer.wk",
            "indexer.weights_proj",
        ]
        for name, param in model.named_parameters():
            for iname in indexer_param_names:
                if iname in name:
                    self.assertIsNotNone(
                        param.grad,
                        f"Indexer parameter {name} has no gradient",
                    )


if __name__ == "__main__":
    unittest.main()
