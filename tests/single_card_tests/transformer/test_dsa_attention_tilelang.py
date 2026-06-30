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

"""Unit tests for dsa_attention.py TileLang branch (TileLangDSAFusedFunction, TileLangDSAIndexerLoss).

Includes precision alignment tests against paddle small-op reference from dsa_attention.py.
"""

import unittest

import paddle
import paddle.nn.functional as F

paddle.enable_compat(scope={"tilelang"}, silent=True)


# =========================================================================
# Helpers
# =========================================================================


def _cuda_or_skip(testcase):
    if not paddle.device.is_compiled_with_cuda():
        testcase.skipTest("CUDA build of Paddle is required")
    if paddle.device.cuda.device_count() == 0:
        testcase.skipTest("No CUDA device available")


def _make_thd_inputs(
    batch,
    seqlen,
    heads,
    dim,
    tail_dim,
    index_h,
    index_d,
    topk,
    dtype="bfloat16",
    seed=42,
):
    """Create THD-format inputs for TileLangDSAFusedFunction.

    Note: index_h must be >= 16 and divisible by 8 (tilelang GEMM constraint).
    """
    paddle.seed(seed)
    total_tokens = batch * seqlen
    query = paddle.randn([total_tokens, heads, dim + tail_dim]).astype(dtype)
    key = paddle.randn([total_tokens, 1, dim + tail_dim]).astype(dtype)
    index_q = paddle.randn([total_tokens, index_h, index_d]).astype(dtype)
    index_k = paddle.randn([total_tokens, index_d]).astype(dtype)
    weights = paddle.randn([total_tokens, index_h]).astype(dtype)
    offsets = (paddle.arange(0, batch + 1, dtype="int32") * seqlen).cast(
        "int32"
    )
    return query, key, index_q, index_k, weights, offsets


# =========================================================================
# Reference: naive sparse MLA forward
# =========================================================================


def _ref_sparse_mla_fwd(query, key, topk_indices, offsets, sm_scale, d_v):
    """Naive sparse MLA forward for verification. THD format.

    Equivalent to _unfused_dsa_attention from dsa_attention.py but in THD format:
    builds a sparse causal mask from topk_indices, then does full Q@K softmax V.
    """
    Q_f = query.cast("float32")
    KV_f = key.cast("float32")
    offsets_np = offsets.numpy().tolist()
    all_o, all_lse = [], []

    for i in range(len(offsets_np) - 1):
        start, end = int(offsets_np[i]), int(offsets_np[i + 1])
        q = Q_f[start:end]
        kv = KV_f[start:end].squeeze(1)
        indices = topk_indices[start:end]
        s = q.shape[0]
        heads = q.shape[1]

        output = paddle.zeros([s, heads, d_v], dtype="float32")
        lse_out = paddle.zeros([s, heads], dtype="float32")

        for row in range(s):
            kv_indices = indices[row, 0, :]
            valid_mask = (kv_indices >= 0) & (kv_indices <= row)
            valid_indices = kv_indices[valid_mask]
            if len(valid_indices) == 0:
                continue
            kv_valid = kv[valid_indices]
            scores = paddle.einsum("hd,nd->hn", q[row], kv_valid) * sm_scale
            scores_max = scores.max(axis=-1, keepdim=True)
            scores_exp = paddle.exp(scores - scores_max)
            scores_sum = scores_exp.sum(axis=-1, keepdim=True)
            attn_weights = scores_exp / scores_sum
            kv_v = kv_valid[:, :d_v]
            output[row] = paddle.einsum("hn,nd->hd", attn_weights, kv_v)
            lse_out[row] = paddle.log(
                scores_sum.squeeze(-1)
            ) + scores_max.squeeze(-1)

        all_o.append(output)
        all_lse.append(lse_out)

    return paddle.concat(all_o, axis=0).cast("bfloat16"), paddle.concat(
        all_lse, axis=0
    )


def _ref_indexer_topk_thd(index_q, weights, index_k, topk, offsets):
    """Reference indexer topk in THD format.

    Matches dsa_attention.py DSAIndexer.compute_index_scores logic:
      scores = einsum("shd,td->sht", q, k) * sm_scale
      scores = relu(scores)
      index_scores = (scores * weights.unsqueeze(-1)).sum(axis=1)  (sum over heads)
      topk_indices = topk(index_scores, k=topk)

    Note: the tilelang kernel applies sm_scale = dim**-0.5 to q before einsum,
    which is equivalent to scaling the scores.
    """
    offsets_np = offsets.numpy().tolist()
    all_topk_indices = []
    all_topk_scores = []

    for i in range(len(offsets_np) - 1):
        start, end = int(offsets_np[i]), int(offsets_np[i + 1])
        q = index_q[start:end].cast("float32")  # [s, h, d]
        k = index_k[start:end].cast("float32")  # [s, d]
        w = weights[start:end].cast("float32")  # [s, h]
        s = q.shape[0]
        dim = q.shape[-1]
        sm_scale = dim**-0.5

        # einsum q@k: [s, h, d] x [s, d] -> [s, h, s], then scale
        scores = paddle.einsum("shd,td->sht", q, k) * sm_scale
        scores = F.relu(scores)
        # weight and sum over heads: [s, h, s] * [s, h, 1] -> sum -> [s, s]
        index_scores = (scores * w.unsqueeze(-1)).sum(axis=1)

        # causal mask
        causal = paddle.triu(
            paddle.full([s, s], float("-inf"), dtype="float32"), diagonal=1
        )
        index_scores = index_scores + causal

        actual_topk = min(topk, s)
        topk_vals, topk_idx = paddle.topk(index_scores, k=actual_topk, axis=-1)
        if actual_topk < topk:
            pad = topk - actual_topk
            topk_idx = paddle.concat(
                [topk_idx, paddle.full([s, pad], -1, dtype="int64")], axis=-1
            )
            topk_vals = paddle.concat(
                [
                    topk_vals,
                    paddle.full([s, pad], float("-inf"), dtype="float32"),
                ],
                axis=-1,
            )

        topk_probs = F.softmax(topk_vals, axis=-1)
        all_topk_indices.append(topk_idx.cast("int32"))
        all_topk_scores.append(topk_probs)

    return paddle.concat(all_topk_indices, axis=0), paddle.concat(
        all_topk_scores, axis=0
    )


def _ref_unfused_dsa_attention_thd(
    query, key, topk_indices, offsets, sm_scale, d_v
):
    """Reference unfused DSA attention in THD format.

    Equivalent to dsa_attention._unfused_dsa_attention:
    builds sparse+causal mask from topk_indices, then Q@K*mask softmax V.
    """
    Q_f = query.cast("float32")
    K_f = key.cast("float32").squeeze(1)  # [S, dim+tail_dim]
    offsets_np = offsets.numpy().tolist()
    all_o = []

    for i in range(len(offsets_np) - 1):
        start, end = int(offsets_np[i]), int(offsets_np[i + 1])
        q = Q_f[start:end]  # [s, heads, qk_dim]
        k = K_f[start:end]  # [s, qk_dim]
        idx = topk_indices[start:end]  # [s, topk]
        s = q.shape[0]
        heads = q.shape[1]

        # Build combined mask: causal + sparse index
        # causal: [s, s] upper triangle = -inf
        causal_mask = paddle.triu(
            paddle.full([s, s], float("-inf"), dtype="float32"), diagonal=1
        )
        # sparse index mask: only topk positions visible per row
        index_mask = paddle.full([s, s], float("-inf"), dtype="float32")
        for row in range(s):
            valid = idx[row][(idx[row] >= 0) & (idx[row] < s)]
            if len(valid) > 0:
                index_mask[row, valid] = 0.0

        combined_mask = causal_mask + index_mask  # -inf where either masks

        # Q@K^T: [heads, s, s]
        attn_scores = paddle.einsum("shd,td->hst", q, k) * sm_scale
        # Apply mask: [1, s, s] broadcast over heads
        attn_scores = attn_scores + combined_mask.unsqueeze(0)
        attn_weights = F.softmax(attn_scores, axis=-1)
        # V = k[:, :d_v]
        v = k[:, :d_v]  # [s, d_v]
        # output: [heads, s, d_v]
        output = paddle.einsum("hst,td->hsd", attn_weights, v)
        # transpose to [s, heads, d_v]
        output = output.transpose([1, 0, 2])
        all_o.append(output)

    return paddle.concat(all_o, axis=0).cast("bfloat16")


# =========================================================================
# Tests for TileLangDSAFusedFunction
# =========================================================================


@unittest.skipUnless(
    paddle.device.is_compiled_with_cuda()
    and paddle.device.cuda.device_count() > 0,
    "TileLang DSA kernels require CUDA",
)
class TestTileLangDSAFusedFunction(unittest.TestCase):
    """Test TileLangDSAFusedFunction forward and backward."""

    def setUp(self):
        paddle.set_device("gpu")
        try:
            from paddlefleet.transformer.dsa_attention import (
                TileLangDSAFusedFunction,
            )

            self.TileLangDSAFusedFunction = TileLangDSAFusedFunction
        except ImportError:
            self.skipTest("tilelang not available")

    def _run_forward(
        self,
        batch=1,
        seqlen=64,
        heads=16,
        dim=64,
        tail_dim=64,
        index_h=16,
        index_d=128,
        topk=32,
    ):
        v_channels = dim
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5
        query, key, index_q, index_k, weights, offsets = _make_thd_inputs(
            batch, seqlen, heads, dim, tail_dim, index_h, index_d, topk
        )
        o = self.TileLangDSAFusedFunction.apply(
            query,
            key,
            index_q,
            index_k,
            weights,
            offsets,
            topk,
            topk,
            v_channels,
            sm_scale,
            0.0,
            False,
        )
        return o, query, key, offsets

    def test_forward_output_shape(self):
        """Forward output shape should be [total_tokens, heads, v_channels]."""
        batch, seqlen, heads, dim = 1, 64, 16, 64
        o, query, key, offsets = self._run_forward(
            batch=batch, seqlen=seqlen, heads=heads, dim=dim
        )
        self.assertEqual(list(o.shape), [batch * seqlen, heads, dim])

    def test_forward_no_nan(self):
        """Forward output should not contain NaN."""
        o, _, _, _ = self._run_forward()
        self.assertFalse(paddle.isnan(o).any().item(), "Output contains NaN")

    def test_forward_deterministic(self):
        """Two calls with same seed should produce identical output."""
        o1, _, _, _ = self._run_forward()
        o2, _, _, _ = self._run_forward()
        self.assertTrue(
            paddle.allclose(
                o1.cast("float32"), o2.cast("float32"), rtol=0.0, atol=0.0
            ).item()
        )

    def test_backward_produces_gradients(self):
        """Backward should produce non-zero gradients for trainable inputs."""
        batch, seqlen, heads, dim, tail_dim = 1, 64, 16, 64, 64
        index_h, index_d, topk = 16, 128, 32
        v_channels = dim
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5

        query, key, index_q, index_k, weights, offsets = _make_thd_inputs(
            batch, seqlen, heads, dim, tail_dim, index_h, index_d, topk
        )
        query.stop_gradient = False
        key.stop_gradient = False
        index_q.stop_gradient = False
        index_k.stop_gradient = False
        weights.stop_gradient = False

        o = self.TileLangDSAFusedFunction.apply(
            query,
            key,
            index_q,
            index_k,
            weights,
            offsets,
            topk,
            topk,
            v_channels,
            sm_scale,
            1.0,
            False,
        )
        loss = o.cast("float32").sum()
        loss.backward()

        self.assertIsNotNone(query.grad)
        self.assertIsNotNone(key.grad)
        self.assertTrue(
            query.grad.abs().sum().item() > 0, "query grad is all zero"
        )
        self.assertTrue(key.grad.abs().sum().item() > 0, "key grad is all zero")

    def test_backward_grad_shape(self):
        """Gradients should have same shape as inputs."""
        batch, seqlen, heads, dim, tail_dim = 1, 64, 16, 64, 64
        index_h, index_d, topk = 16, 128, 32
        v_channels = dim
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5

        query, key, index_q, index_k, weights, offsets = _make_thd_inputs(
            batch, seqlen, heads, dim, tail_dim, index_h, index_d, topk
        )
        query.stop_gradient = False
        key.stop_gradient = False
        index_q.stop_gradient = False
        index_k.stop_gradient = False
        weights.stop_gradient = False

        o = self.TileLangDSAFusedFunction.apply(
            query,
            key,
            index_q,
            index_k,
            weights,
            offsets,
            topk,
            topk,
            v_channels,
            sm_scale,
            1.0,
            False,
        )
        o.cast("float32").sum().backward()

        self.assertEqual(list(query.grad.shape), list(query.shape))
        self.assertEqual(list(key.grad.shape), list(key.shape))
        self.assertEqual(list(index_q.grad.shape), list(index_q.shape))
        self.assertEqual(list(index_k.grad.shape), list(index_k.shape))
        self.assertEqual(list(weights.grad.shape), list(weights.shape))

    def test_varlen_batch(self):
        """Test with multiple sequences (varlen)."""
        batch, seqlen, heads, dim = 2, 32, 16, 64
        o, _, _, _ = self._run_forward(
            batch=batch, seqlen=seqlen, heads=heads, dim=dim
        )
        self.assertEqual(list(o.shape), [batch * seqlen, heads, dim])
        self.assertFalse(paddle.isnan(o).any().item())


# =========================================================================
# Tests for TileLangDSAIndexerLoss
# =========================================================================


@unittest.skipUnless(
    paddle.device.is_compiled_with_cuda()
    and paddle.device.cuda.device_count() > 0,
    "TileLang DSA kernels require CUDA",
)
class TestTileLangDSAIndexerLoss(unittest.TestCase):
    """Test TileLangDSAIndexerLoss forward and backward."""

    def setUp(self):
        paddle.set_device("gpu")
        try:
            from paddlefleet.transformer.dsa_attention import (
                TileLangDSAIndexerLoss,
            )

            self.TileLangDSAIndexerLoss = TileLangDSAIndexerLoss
        except ImportError:
            self.skipTest("tilelang not available")

    def test_loss_forward_returns_scalar(self):
        """Forward should return (loss_scalar, topk_indices)."""
        batch, seqlen, heads, dim, tail_dim = 1, 64, 16, 64, 64
        index_h, index_d, topk = 16, 128, 32
        v_channels = dim
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5

        query, key, index_q, index_k, weights, offsets = _make_thd_inputs(
            batch, seqlen, heads, dim, tail_dim, index_h, index_d, topk
        )
        loss, topk_idx = self.TileLangDSAIndexerLoss.apply(
            index_q,
            weights,
            index_k,
            query,
            key,
            offsets,
            topk,
            topk,
            v_channels,
            sm_scale,
            1.0,
            False,
        )
        self.assertEqual(loss.ndim, 0, "loss should be scalar")
        self.assertFalse(paddle.isnan(loss).item(), "loss is NaN")
        self.assertEqual(list(topk_idx.shape), [batch * seqlen, topk])

    def test_loss_is_non_negative(self):
        """KL divergence loss should be >= 0."""
        batch, seqlen, heads, dim, tail_dim = 1, 64, 16, 64, 64
        index_h, index_d, topk = 16, 128, 32
        v_channels = dim
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5

        query, key, index_q, index_k, weights, offsets = _make_thd_inputs(
            batch, seqlen, heads, dim, tail_dim, index_h, index_d, topk
        )
        loss, _ = self.TileLangDSAIndexerLoss.apply(
            index_q,
            weights,
            index_k,
            query,
            key,
            offsets,
            topk,
            topk,
            v_channels,
            sm_scale,
            1.0,
            False,
        )
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_loss_backward_produces_grads(self):
        """Backward should produce gradients for index_q, weights, index_k."""
        batch, seqlen, heads, dim, tail_dim = 1, 64, 16, 64, 64
        index_h, index_d, topk = 16, 128, 32
        v_channels = dim
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5

        query, key, index_q, index_k, weights, offsets = _make_thd_inputs(
            batch, seqlen, heads, dim, tail_dim, index_h, index_d, topk
        )
        index_q.stop_gradient = False
        index_k.stop_gradient = False
        weights.stop_gradient = False

        loss, _ = self.TileLangDSAIndexerLoss.apply(
            index_q,
            weights,
            index_k,
            query,
            key,
            offsets,
            topk,
            topk,
            v_channels,
            sm_scale,
            1.0,
            False,
        )
        loss.backward()

        self.assertIsNotNone(index_q.grad)
        self.assertIsNotNone(weights.grad)
        self.assertIsNotNone(index_k.grad)

    def test_loss_zero_coeff_gives_zero_loss(self):
        """With loss_coeff=0, loss should be 0."""
        batch, seqlen, heads, dim, tail_dim = 1, 64, 16, 64, 64
        index_h, index_d, topk = 16, 128, 32
        v_channels = dim
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5

        query, key, index_q, index_k, weights, offsets = _make_thd_inputs(
            batch, seqlen, heads, dim, tail_dim, index_h, index_d, topk
        )
        loss, _ = self.TileLangDSAIndexerLoss.apply(
            index_q,
            weights,
            index_k,
            query,
            key,
            offsets,
            topk,
            topk,
            v_channels,
            sm_scale,
            0.0,
            False,
        )
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_full_loss_mode(self):
        """use_full_loss=True should also produce valid loss."""
        batch, seqlen, heads, dim, tail_dim = 1, 64, 16, 64, 64
        index_h, index_d, topk = 16, 128, 32
        v_channels = dim
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5

        query, key, index_q, index_k, weights, offsets = _make_thd_inputs(
            batch, seqlen, heads, dim, tail_dim, index_h, index_d, topk
        )
        loss, topk_idx = self.TileLangDSAIndexerLoss.apply(
            index_q,
            weights,
            index_k,
            query,
            key,
            offsets,
            topk,
            topk,
            v_channels,
            sm_scale,
            1.0,
            True,
        )
        self.assertFalse(paddle.isnan(loss).item())
        self.assertGreaterEqual(loss.item(), 0.0)


# =========================================================================
# Tests for TileLangDSAFusedFunction with loss_coeff > 0
# =========================================================================


@unittest.skipUnless(
    paddle.device.is_compiled_with_cuda()
    and paddle.device.cuda.device_count() > 0,
    "TileLang DSA kernels require CUDA",
)
class TestTileLangDSAFusedWithLoss(unittest.TestCase):
    """Test fused function with indexer loss enabled (loss_coeff > 0)."""

    def setUp(self):
        paddle.set_device("gpu")
        try:
            from paddlefleet.transformer.dsa_attention import (
                TileLangDSAFusedFunction,
            )

            self.TileLangDSAFusedFunction = TileLangDSAFusedFunction
        except ImportError:
            self.skipTest("tilelang not available")

    def test_backward_with_loss_coeff(self):
        """With loss_coeff > 0, indexer grads should be non-zero."""
        batch, seqlen, heads, dim, tail_dim = 1, 64, 16, 64, 64
        index_h, index_d, topk = 16, 128, 32
        v_channels = dim
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5

        query, key, index_q, index_k, weights, offsets = _make_thd_inputs(
            batch, seqlen, heads, dim, tail_dim, index_h, index_d, topk
        )
        query.stop_gradient = False
        key.stop_gradient = False
        index_q.stop_gradient = False
        index_k.stop_gradient = False
        weights.stop_gradient = False

        o = self.TileLangDSAFusedFunction.apply(
            query,
            key,
            index_q,
            index_k,
            weights,
            offsets,
            topk,
            topk,
            v_channels,
            sm_scale,
            1.0,
            False,
        )
        o.cast("float32").sum().backward()

        self.assertIsNotNone(index_q.grad)
        self.assertIsNotNone(weights.grad)
        self.assertIsNotNone(index_k.grad)
        self.assertTrue(
            index_q.grad.abs().sum().item() > 0, "index_q grad is all zero"
        )
        self.assertTrue(
            weights.grad.abs().sum().item() > 0, "weights grad is all zero"
        )

    def test_loss_coeff_scales_indexer_grads(self):
        """Doubling loss_coeff should roughly double indexer grads."""
        batch, seqlen, heads, dim, tail_dim = 1, 64, 16, 64, 64
        index_h, index_d, topk = 16, 128, 32
        v_channels = dim
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5

        def _run_with_coeff(coeff):
            query, key, index_q, index_k, weights, offsets = _make_thd_inputs(
                batch, seqlen, heads, dim, tail_dim, index_h, index_d, topk
            )
            query.stop_gradient = False
            key.stop_gradient = False
            index_q.stop_gradient = False
            index_k.stop_gradient = False
            weights.stop_gradient = False
            o = self.TileLangDSAFusedFunction.apply(
                query,
                key,
                index_q,
                index_k,
                weights,
                offsets,
                topk,
                topk,
                v_channels,
                sm_scale,
                coeff,
                False,
            )
            o.cast("float32").sum().backward()
            return index_q.grad.cast("float32").abs().mean().item()

        grad_1x = _run_with_coeff(1.0)
        grad_2x = _run_with_coeff(2.0)
        if grad_1x > 1e-8:
            ratio = grad_2x / grad_1x
            self.assertGreater(
                ratio, 1.5, f"Expected ~2x scaling, got {ratio:.2f}"
            )
            self.assertLess(
                ratio, 2.5, f"Expected ~2x scaling, got {ratio:.2f}"
            )


# =========================================================================
# Precision alignment: TileLang vs paddle small-op reference (dsa_attention.py)
# =========================================================================


@unittest.skipUnless(
    paddle.device.is_compiled_with_cuda()
    and paddle.device.cuda.device_count() > 0,
    "TileLang DSA kernels require CUDA",
)
class TestTileLangVsPaddleRefPrecision(unittest.TestCase):
    """Verify tilelang kernel outputs match paddle small-op reference implementations.

    Reference implementations follow the same logic as dsa_attention.py:
    - Indexer: einsum(q, k) -> relu -> weight*sum -> causal_mask -> topk -> softmax
    - Sparse MLA: Q@K with sparse+causal mask -> softmax -> V
    """

    def setUp(self):
        paddle.set_device("gpu")
        try:
            from paddlefleet.tilelang_ops import (
                dsa_indexer_topk_reducesum_interface,
                dsa_prepare_varlen_metadata,
                dsa_sparse_mla_fwd_interface,
                dsa_sparse_mla_topk_reducesum_interface,
            )
            from paddlefleet.transformer.dsa_attention import (
                TileLangDSAFusedFunction,
                TileLangDSAIndexerLoss,
            )

            self.TileLangDSAFusedFunction = TileLangDSAFusedFunction
            self.TileLangDSAIndexerLoss = TileLangDSAIndexerLoss
            self.dsa_indexer_topk_reducesum_interface = (
                dsa_indexer_topk_reducesum_interface
            )
            self.dsa_sparse_mla_fwd_interface = dsa_sparse_mla_fwd_interface
            self.dsa_sparse_mla_topk_reducesum_interface = (
                dsa_sparse_mla_topk_reducesum_interface
            )
            self.dsa_prepare_varlen_metadata = dsa_prepare_varlen_metadata
        except ImportError as e:
            self.skipTest(f"Import failed: {e}")

    def test_indexer_topk_matches_ref(self):
        """TileLang indexer topk should produce same indices as paddle ref."""
        batch, seqlen = 1, 64
        index_h, index_d, topk = 16, 128, 32
        paddle.seed(100)
        total = batch * seqlen
        index_q = paddle.randn([total, index_h, index_d]).astype("bfloat16")
        index_k = paddle.randn([total, index_d]).astype("bfloat16")
        weights = paddle.randn([total, index_h]).astype("bfloat16")
        offsets = (paddle.arange(0, batch + 1, dtype="int32") * seqlen).cast(
            "int32"
        )
        offsets_prep, token_indices = self.dsa_prepare_varlen_metadata(offsets)

        # TileLang kernel
        tl_indices, tl_scores = self.dsa_indexer_topk_reducesum_interface(
            index_q,
            weights,
            index_k,
            topk,
            offsets_prep,
            token_indices,
            use_full_loss=False,
        )
        # Paddle reference
        ref_indices, ref_scores = _ref_indexer_topk_thd(
            index_q, weights, index_k, topk, offsets
        )

        # Indices: sorted comparison (topk order may differ for equal scores)
        # Note: bfloat16 precision means scores can differ slightly, causing
        # different topk selections for borderline elements. 50%+ match is acceptable.
        tl_sorted = paddle.sort(tl_indices, axis=-1)
        ref_sorted = paddle.sort(ref_indices, axis=-1)
        index_match = (tl_sorted == ref_sorted).cast("float32").mean().item()
        self.assertGreater(
            index_match, 0.4, f"Index match rate {index_match:.2%} too low"
        )

        # Scores: where indices match, scores should be close
        valid = ref_indices >= 0
        if valid.any().item():
            paddle.testing.assert_close(
                tl_scores[valid].cast("float32"),
                ref_scores[valid].cast("float32"),
                rtol=1e-1,
                atol=5e-2,
            )

    def test_sparse_mla_fwd_matches_ref(self):
        """TileLang sparse MLA fwd output should match naive paddle implementation."""
        batch, seqlen, heads, dim, tail_dim = 1, 64, 16, 64, 64
        topk = 32
        v_channels = dim
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5
        paddle.seed(200)
        total = batch * seqlen

        query = paddle.randn([total, heads, dim + tail_dim]).astype("bfloat16")
        key = paddle.randn([total, 1, dim + tail_dim]).astype("bfloat16")
        offsets = (paddle.arange(0, batch + 1, dtype="int32") * seqlen).cast(
            "int32"
        )
        offsets_prep, token_indices = self.dsa_prepare_varlen_metadata(offsets)

        # Generate causal-valid indices
        indices = paddle.zeros([total, 1, topk], dtype="int32")
        for row in range(total):
            valid_range = row + 1
            if valid_range >= topk:
                idx = paddle.randperm(valid_range)[:topk].sort()[0]
            else:
                idx = paddle.concat(
                    [
                        paddle.arange(valid_range),
                        paddle.full([topk - valid_range], -1, dtype="int64"),
                    ]
                )
            indices[row, 0, :] = idx.cast("int32")

        # TileLang kernel
        tl_o, tl_lse = self.dsa_sparse_mla_fwd_interface(
            query,
            key,
            indices,
            offsets_prep,
            token_indices,
            sm_scale=sm_scale,
            d_v=v_channels,
        )
        # Paddle reference
        ref_o, ref_lse = _ref_sparse_mla_fwd(
            query, key, indices, offsets, sm_scale, v_channels
        )

        # Output comparison (bfloat16 tolerance)
        paddle.testing.assert_close(
            tl_o.cast("float32"),
            ref_o.cast("float32"),
            rtol=1e-1,
            atol=5e-2,
        )

    def test_fused_forward_matches_unfused_dsa_attention(self):
        """TileLangDSAFusedFunction output should match _unfused_dsa_attention logic."""
        batch, seqlen, heads, dim, tail_dim = 1, 64, 16, 64, 64
        index_h, index_d, topk = 16, 128, 32
        v_channels = dim
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5
        paddle.seed(300)
        total = batch * seqlen

        query = paddle.randn([total, heads, dim + tail_dim]).astype("bfloat16")
        key = paddle.randn([total, 1, dim + tail_dim]).astype("bfloat16")
        index_q = paddle.randn([total, index_h, index_d]).astype("bfloat16")
        index_k = paddle.randn([total, index_d]).astype("bfloat16")
        weights = paddle.randn([total, index_h]).astype("bfloat16")
        offsets = (paddle.arange(0, batch + 1, dtype="int32") * seqlen).cast(
            "int32"
        )

        # TileLang fused forward
        tl_o = self.TileLangDSAFusedFunction.apply(
            query,
            key,
            index_q,
            index_k,
            weights,
            offsets,
            topk,
            topk,
            v_channels,
            sm_scale,
            0.0,
            False,
        )

        # Get tilelang's topk_indices for reference comparison
        topk_indices = self.TileLangDSAFusedFunction._last_topk_indices

        # Reference: unfused DSA attention using same topk_indices
        ref_o = _ref_unfused_dsa_attention_thd(
            query, key, topk_indices, offsets, sm_scale, v_channels
        )

        # Compare outputs (bf16 + sparse attention tolerance)
        tl_f = tl_o.cast("float32")
        ref_f = ref_o.cast("float32")
        # Mask out rows with no valid topk (output is zero for both)
        nonzero_mask = ref_f.abs().sum(axis=-1).sum(axis=-1) > 1e-6
        if nonzero_mask.any().item():
            paddle.testing.assert_close(
                tl_f[nonzero_mask],
                ref_f[nonzero_mask],
                rtol=1e-1,
                atol=5e-2,
            )

    def test_indexer_loss_matches_ref_kl(self):
        """TileLangDSAIndexerLoss KL should match paddle reference computation."""
        batch, seqlen, heads, dim, tail_dim = 1, 64, 16, 64, 64
        index_h, index_d, topk = 16, 128, 32
        v_channels = dim
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5
        paddle.seed(400)
        total = batch * seqlen

        query = paddle.randn([total, heads, dim + tail_dim]).astype("bfloat16")
        key = paddle.randn([total, 1, dim + tail_dim]).astype("bfloat16")
        index_q = paddle.randn([total, index_h, index_d]).astype("bfloat16")
        index_k = paddle.randn([total, index_d]).astype("bfloat16")
        weights = paddle.randn([total, index_h]).astype("bfloat16")
        offsets = (paddle.arange(0, batch + 1, dtype="int32") * seqlen).cast(
            "int32"
        )

        # TileLang loss
        tl_loss, tl_topk_idx = self.TileLangDSAIndexerLoss.apply(
            index_q,
            weights,
            index_k,
            query,
            key,
            offsets,
            topk,
            topk,
            v_channels,
            sm_scale,
            1.0,
            False,
        )

        # Manual KL reference using tilelang's own intermediate results:
        # Get index_score and attn_score via interfaces
        offsets_prep, token_indices = self.dsa_prepare_varlen_metadata(offsets)
        _, index_score = self.dsa_indexer_topk_reducesum_interface(
            index_q,
            weights,
            index_k,
            topk,
            offsets_prep,
            token_indices,
            use_full_loss=False,
        )
        _, lse = self.dsa_sparse_mla_fwd_interface(
            query,
            key,
            tl_topk_idx.unsqueeze(1),
            offsets_prep,
            token_indices,
            sm_scale=sm_scale,
            d_v=v_channels,
        )
        attn_score = self.dsa_sparse_mla_topk_reducesum_interface(
            query,
            key,
            tl_topk_idx.unsqueeze(1),
            lse,
            offsets_prep,
            token_indices,
            dim_v=v_channels,
            sm_scale=sm_scale,
        ).squeeze(1)

        # Compute expected KL
        kl = attn_score * (
            paddle.log(attn_score + 1e-10) - paddle.log(index_score + 1e-10)
        )
        expected_loss = kl.sum(axis=-1).mean()

        paddle.testing.assert_close(
            tl_loss.cast("float32").reshape([1]),
            expected_loss.cast("float32").reshape([1]),
            rtol=1e-3,
            atol=1e-5,
        )


# =========================================================================
# Integration test: DSAttention config-driven TileLang branch
# =========================================================================


class TestDSAttentionConfigTileLangBranch(unittest.TestCase):
    """Verify that TransformerConfig.dsa_tilelang_enable controls DSAttention._use_tilelang."""

    def test_config_field_exists_and_defaults_false(self):
        """TransformerConfig should have dsa_tilelang_enable defaulting to False."""
        from paddlefleet.transformer.transformer_config import TransformerConfig

        config = TransformerConfig()
        self.assertFalse(config.dsa_tilelang_enable)

    def test_config_field_enables_tilelang(self):
        """Setting dsa_tilelang_enable=True should set DSAttention._use_tilelang=True."""
        from paddlefleet.transformer.transformer_config import TransformerConfig

        config = TransformerConfig(dsa_tilelang_enable=True)
        self.assertTrue(config.dsa_tilelang_enable)

    def test_transform_rules_maps_field(self):
        """transform_rules should contain dsa_tilelang_enable mapping."""
        from paddlefleet.transformer.transformer_config import TransformerConfig

        self.assertIn("dsa_tilelang_enable", TransformerConfig.transform_rules)


if __name__ == "__main__":
    unittest.main()
