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

"""
Test MLA (Multi-Latent Attention) with FlashMask.

This test file covers the MLA + flashmask code path in dot_product_attention.py,
specifically the handling of different query/key head_dim vs value head_dim cases.
"""

import math
import unittest

import numpy as np
import paddle

from paddlefleet.transformer.dot_product_attention import DotProductAttention
from paddlefleet.transformer.enums import AttnMaskType
from paddlefleet.transformer.transformer_config import TransformerConfig


class TestMLAFlashMaskWithBackward(unittest.TestCase):
    """Test backward pass for MLA with FlashMask."""

    def setUp(self):
        paddle.seed(42)
        np.random.seed(42)

    def _create_config(
        self,
        hidden_size: int = 256,
        num_attention_heads: int = 4,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        bf16: bool = True,
    ):
        """Create a TransformerConfig for MLA testing."""
        config = TransformerConfig(
            num_hidden_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
        )

        config.qk_nope_head_dim = qk_nope_head_dim
        config.qk_rope_head_dim = qk_rope_head_dim
        config.v_head_dim = v_head_dim
        config.head_dim = qk_nope_head_dim + qk_rope_head_dim

        config.num_key_value_heads = num_attention_heads
        config.softmax_scale = None
        config.use_bias = True
        config.context_parallel_size = 1
        config.apply_query_key_layer_scaling = False
        config.sliding_window = None
        config.window_attn_skip_freq = None
        config.fp16 = False
        config.bf16 = bf16
        config.masked_softmax_fusion = False
        config.attention_softmax_in_fp32 = True
        config.attention_dropout = 0.0
        config.softmax_type = "vanilla"

        return config

    def _create_attn_mask_startend_row_indices(
        self, batch_size: int, num_heads: int, seq_len: int, causal: bool = True
    ):
        """Create attention mask startend row indices for flashmask."""
        if causal:
            start_indices = np.zeros(
                (batch_size, 1, seq_len, 1), dtype=np.int32
            )
            end_indices = np.arange(1, seq_len + 1, dtype=np.int32).reshape(
                1, 1, seq_len, 1
            )
            end_indices = np.broadcast_to(
                end_indices, (batch_size, 1, seq_len, 1)
            )
            indices = np.concatenate([start_indices, end_indices], axis=-1)
        else:
            start_indices = np.zeros(
                (batch_size, 1, seq_len, 1), dtype=np.int32
            )
            end_indices = np.full(
                (batch_size, 1, seq_len, 1), seq_len, dtype=np.int32
            )
            indices = np.concatenate([start_indices, end_indices], axis=-1)

        return paddle.to_tensor(indices)

    def test_backward_with_padding(self):
        """Test backward pass when q_head_dim != v_head_dim (requires padding)."""
        config = self._create_config(
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,  # v_head_dim < q_head_dim
        )

        attention = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )

        batch_size = 2
        seq_len = 16
        num_heads = 4
        q_head_dim = 192
        v_head_dim = 128

        # Create tensors with requires_grad=True
        query = paddle.randn(
            (batch_size, seq_len, num_heads, q_head_dim), dtype=paddle.bfloat16
        )
        query.stop_gradient = False
        key = paddle.randn(
            (batch_size, seq_len, num_heads, q_head_dim), dtype=paddle.bfloat16
        )
        key.stop_gradient = False
        value = paddle.randn(
            (batch_size, seq_len, num_heads, v_head_dim), dtype=paddle.bfloat16
        )
        value.stop_gradient = False

        attn_mask_startend_row_indices = (
            self._create_attn_mask_startend_row_indices(
                batch_size, num_heads, seq_len, causal=True
            )
        )

        # Forward pass
        output = attention(
            query=query,
            key=key,
            value=value,
            attention_mask=None,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            attn_mask_type=AttnMaskType.causal,
        )

        # Backward pass
        grad_output = paddle.randn_like(output)
        output.backward(grad_output)

        # Check gradients exist and have correct shapes
        self.assertIsNotNone(query.grad)
        self.assertIsNotNone(key.grad)
        self.assertIsNotNone(value.grad)

        self.assertEqual(query.grad.shape, query.shape)
        self.assertEqual(key.grad.shape, key.shape)
        self.assertEqual(value.grad.shape, value.shape)

    def test_backward_without_padding(self):
        """Test backward pass when q_head_dim == v_head_dim (no padding)."""
        config = self._create_config(
            qk_nope_head_dim=96,
            qk_rope_head_dim=32,
            v_head_dim=128,  # v_head_dim == q_head_dim
        )

        attention = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )

        batch_size = 2
        seq_len = 16
        num_heads = 4
        head_dim = 128

        query = paddle.randn(
            (batch_size, seq_len, num_heads, head_dim), dtype=paddle.bfloat16
        )
        query.stop_gradient = False
        key = paddle.randn(
            (batch_size, seq_len, num_heads, head_dim), dtype=paddle.bfloat16
        )
        key.stop_gradient = False
        value = paddle.randn(
            (batch_size, seq_len, num_heads, head_dim), dtype=paddle.bfloat16
        )
        value.stop_gradient = False

        attn_mask_startend_row_indices = (
            self._create_attn_mask_startend_row_indices(
                batch_size, num_heads, seq_len, causal=True
            )
        )

        output = attention(
            query=query,
            key=key,
            value=value,
            attention_mask=None,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            attn_mask_type=AttnMaskType.causal,
        )

        grad_output = paddle.randn_like(output)
        output.backward(grad_output)

        self.assertIsNotNone(query.grad)
        self.assertIsNotNone(key.grad)
        self.assertIsNotNone(value.grad)

        self.assertEqual(query.grad.shape, query.shape)
        self.assertEqual(key.grad.shape, key.shape)
        self.assertEqual(value.grad.shape, value.shape)


def _make_mla_config(q_head_dim, v_head_dim, num_heads):
    """A minimal MLA-style config with ``q_head_dim != v_head_dim``."""
    config = TransformerConfig(
        num_hidden_layers=1,
        hidden_size=256,
        num_attention_heads=num_heads,
    )
    config.qk_rope_head_dim = 64
    config.qk_nope_head_dim = q_head_dim - config.qk_rope_head_dim
    config.v_head_dim = v_head_dim
    config.head_dim = q_head_dim
    config.num_key_value_heads = num_heads
    config.softmax_scale = None
    config.use_bias = True
    config.context_parallel_size = 1
    config.apply_query_key_layer_scaling = False
    config.sliding_window = None
    config.window_attn_skip_freq = None
    config.fp16 = False
    config.bf16 = True
    config.masked_softmax_fusion = False
    config.attention_softmax_in_fp32 = True
    config.attention_dropout = 0.0
    config.softmax_type = "vanilla"
    return config


def _reference_full_attention(query, key, value):
    """Dense fp32 attention with no masking, for numerical comparison.

    Matches the kernel's default scale (``1 / sqrt(q_head_dim)``) and the
    flashmask indices used below, which leave every KV column visible.
    """
    q = query.astype(paddle.float32).transpose([0, 2, 1, 3])
    k = key.astype(paddle.float32).transpose([0, 2, 1, 3])
    v = value.astype(paddle.float32).transpose([0, 2, 1, 3])
    scores = paddle.matmul(q, k, transpose_y=True) / math.sqrt(query.shape[-1])
    probs = paddle.nn.functional.softmax(scores, axis=-1)
    return paddle.matmul(probs, v).transpose([0, 2, 1, 3])


class _SpyKernel:
    """Stand-in for the fused kernel that records the shapes it was handed.

    The value padding / output truncation around the kernel call is plain
    tensor bookkeeping, so it can be exercised without a GPU by swapping the
    kernel for this spy. It returns an output shaped like the *padded* value
    (``q_head_dim`` last dim), which is what a real kernel produces and what
    the caller must truncate back to ``v_head_dim``.
    """

    def __init__(self):
        self.value_shape = None

    def __call__(self, query=None, key=None, value=None, *args, **kwargs):
        if query is None:
            query, key, value = args[0], args[1], args[2]
        self.value_shape = list(value.shape)
        bsz, q_len, num_heads, _ = query.shape
        return paddle.zeros(
            [bsz, q_len, num_heads, value.shape[-1]], dtype=value.dtype
        )


class _StubKVCache:
    """Minimal ``past_key_values`` stub: concatenates along the seq axis."""

    def __init__(self, key_cache, value_cache):
        self.key_cache = key_cache
        self.value_cache = value_cache

    def update(self, key, value, layer_idx):
        self.key_cache = paddle.concat([self.key_cache, key], axis=1)
        self.value_cache = paddle.concat([self.value_cache, value], axis=1)
        return self.key_cache, self.value_cache


class TestMLAFlashMaskDecodeAsymmetricHeadDim(unittest.TestCase):
    """Decode-time padding when ``q_len == 1`` but ``kv_len > 1``.

    Regression guard: the value padding used to be allocated with the *query*
    seq len, which silently matches during training (``q_len == kv_len``) and
    breaks the concat as soon as a KV cache makes the two differ. The other
    tests in this file only cover the symmetric case.
    """

    BATCH_SIZE = 2
    NUM_HEADS = 4
    Q_HEAD_DIM = 192
    V_HEAD_DIM = 128
    CACHE_LEN = 16

    def setUp(self):
        paddle.seed(42)
        np.random.seed(42)
        self.kv_len = self.CACHE_LEN + 1

    def _decode_qkv(self):
        """One query token against a ``CACHE_LEN``-token history."""
        query = paddle.randn(
            (self.BATCH_SIZE, 1, self.NUM_HEADS, self.Q_HEAD_DIM),
            dtype=paddle.bfloat16,
        )
        key = paddle.randn(
            (self.BATCH_SIZE, 1, self.NUM_HEADS, self.Q_HEAD_DIM),
            dtype=paddle.bfloat16,
        )
        value = paddle.randn(
            (self.BATCH_SIZE, 1, self.NUM_HEADS, self.V_HEAD_DIM),
            dtype=paddle.bfloat16,
        )
        return query, key, value

    def _cache(self):
        key_cache = paddle.randn(
            (self.BATCH_SIZE, self.CACHE_LEN, self.NUM_HEADS, self.Q_HEAD_DIM),
            dtype=paddle.bfloat16,
        )
        value_cache = paddle.randn(
            (self.BATCH_SIZE, self.CACHE_LEN, self.NUM_HEADS, self.V_HEAD_DIM),
            dtype=paddle.bfloat16,
        )
        return _StubKVCache(key_cache, value_cache)

    def _row_indices(self, seq_len):
        """Flashmask indices that mask nothing (full attention).

        For ``causal=False`` with 2 columns the convention is
        ``[down_left_start, up_right_end]``: rows ``[down_left_start:, j]`` and
        rows ``[:up_right_end, j]`` are masked. ``[seq_len, 0]`` therefore
        leaves every KV column visible -- see Paddle's
        ``test/test_flashmask_ci/generate_startend_row_indices.py``
        (``generate_non_causal_mask``).
        """
        down_left_start = np.full(
            (self.BATCH_SIZE, 1, seq_len, 1), seq_len, dtype=np.int32
        )
        up_right_end = np.zeros(
            (self.BATCH_SIZE, 1, seq_len, 1), dtype=np.int32
        )
        return paddle.to_tensor(
            np.concatenate([down_left_start, up_right_end], axis=-1)
        )

    def test_facade_flashmask_pads_value_by_its_own_seq_len(self):
        """``flash_mask_facade.flashmask_attention`` with q_len=1, kv_len>1."""
        from paddlefleet_ops import flash_mask_facade

        query, _, _ = self._decode_qkv()
        key = paddle.randn(
            (self.BATCH_SIZE, self.kv_len, self.NUM_HEADS, self.Q_HEAD_DIM),
            dtype=paddle.bfloat16,
        )
        value = paddle.randn(
            (self.BATCH_SIZE, self.kv_len, self.NUM_HEADS, self.V_HEAD_DIM),
            dtype=paddle.bfloat16,
        )

        spy = _SpyKernel()
        original = flash_mask_facade._flashmask_attention
        flash_mask_facade._flashmask_attention = spy
        try:
            out = flash_mask_facade.flashmask_attention(
                query,
                key,
                value,
                startend_row_indices=self._row_indices(self.kv_len),
                causal=False,
            )
        finally:
            flash_mask_facade._flashmask_attention = original

        # The concat must widen value along head_dim while keeping its own
        # kv_len -- not the query's q_len of 1.
        self.assertEqual(
            spy.value_shape,
            [self.BATCH_SIZE, self.kv_len, self.NUM_HEADS, self.Q_HEAD_DIM],
        )
        # Output is truncated back to the original v_head_dim.
        self.assertEqual(
            out.shape,
            [self.BATCH_SIZE, 1, self.NUM_HEADS, self.V_HEAD_DIM],
        )

    def test_dot_product_attention_decode_with_kv_cache(self):
        """The flashmask branch of ``DotProductAttention`` during decode."""
        import paddlefleet.transformer.dot_product_attention as dpa

        config = _make_mla_config(
            self.Q_HEAD_DIM, self.V_HEAD_DIM, self.NUM_HEADS
        )

        attention = dpa.DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        attention.eval()

        query, key, value = self._decode_qkv()
        spy = _SpyKernel()
        original = dpa.flashmask_attention
        dpa.flashmask_attention = spy
        try:
            output = attention(
                query=query,
                key=key,
                value=value,
                attention_mask=None,
                attn_mask_startend_row_indices=self._row_indices(self.kv_len),
                attn_mask_type=AttnMaskType.causal,
                past_key_values=self._cache(),
                layer_idx=0,
                use_cache=True,
            )
        finally:
            dpa.flashmask_attention = original

        # value came from the cache (kv_len rows), padded to q_head_dim.
        self.assertEqual(
            spy.value_shape,
            [self.BATCH_SIZE, self.kv_len, self.NUM_HEADS, self.Q_HEAD_DIM],
        )
        # Output keeps the query's single row and the original v_head_dim.
        self.assertEqual(
            output.shape,
            [self.BATCH_SIZE, 1, self.NUM_HEADS * self.V_HEAD_DIM],
        )


@unittest.skipUnless(
    paddle.device.cuda.device_count() > 0,
    "decode-time flashmask correctness needs a real GPU kernel",
)
class TestMLAFlashMaskDecodeCorrectness(unittest.TestCase):
    """Numerical check of the decode path on a real GPU kernel.

    Companion to ``TestMLAFlashMaskDecodeAsymmetricHeadDim``, which only pins
    down shapes with a stubbed kernel. Here the real kernel runs with
    ``q_len=1``, ``kv_len>1`` and ``q_head_dim != v_head_dim``, so a wrong
    padding (e.g. zeros in the wrong rows) shows up as a numerical mismatch
    rather than just a shape error.

    Two head-dim pairs are covered:
      * ``(192, 128)`` -- handled natively by FA4, padded on FA2/FA3.
      * ``(128, 64)``  -- padded on every version, so the padding branch is
        exercised even when FA4 is dispatched.
    """

    BATCH_SIZE = 2
    NUM_HEADS = 4
    CACHE_LEN = 16
    HEAD_DIM_PAIRS = ((192, 128), (128, 64))

    def setUp(self):
        paddle.seed(42)
        np.random.seed(42)
        self.kv_len = self.CACHE_LEN + 1

    def _row_indices(self, seq_len):
        """Flashmask indices that mask nothing (full attention).

        ``causal=False`` with 2 columns means ``[down_left_start,
        up_right_end]``; ``[seq_len, 0]`` masks no rows. See Paddle's
        ``test/test_flashmask_ci/generate_startend_row_indices.py``.
        """
        down_left_start = np.full(
            (self.BATCH_SIZE, 1, seq_len, 1), seq_len, dtype=np.int32
        )
        up_right_end = np.zeros(
            (self.BATCH_SIZE, 1, seq_len, 1), dtype=np.int32
        )
        return paddle.to_tensor(
            np.concatenate([down_left_start, up_right_end], axis=-1)
        )

    def _assert_close(self, actual, expected):
        """bf16 tolerance: the kernel accumulates in fp32, inputs are bf16."""
        np.testing.assert_allclose(
            actual.astype(paddle.float32).numpy(),
            expected.numpy(),
            rtol=1e-2,
            atol=2e-2,
        )

    def test_facade_flashmask_decode_matches_dense_reference(self):
        """Facade output must match dense attention for q_len=1, kv_len>1."""
        from paddlefleet_ops import flash_mask_facade

        for q_head_dim, v_head_dim in self.HEAD_DIM_PAIRS:
            with self.subTest(q_head_dim=q_head_dim, v_head_dim=v_head_dim):
                query = paddle.randn(
                    (self.BATCH_SIZE, 1, self.NUM_HEADS, q_head_dim),
                    dtype=paddle.bfloat16,
                )
                key = paddle.randn(
                    (
                        self.BATCH_SIZE,
                        self.kv_len,
                        self.NUM_HEADS,
                        q_head_dim,
                    ),
                    dtype=paddle.bfloat16,
                )
                value = paddle.randn(
                    (
                        self.BATCH_SIZE,
                        self.kv_len,
                        self.NUM_HEADS,
                        v_head_dim,
                    ),
                    dtype=paddle.bfloat16,
                )

                out = flash_mask_facade.flashmask_attention(
                    query,
                    key,
                    value,
                    startend_row_indices=self._row_indices(self.kv_len),
                    causal=False,
                )

                self.assertEqual(
                    out.shape,
                    [self.BATCH_SIZE, 1, self.NUM_HEADS, v_head_dim],
                )
                self._assert_close(
                    out, _reference_full_attention(query, key, value)
                )

    def test_dot_product_attention_decode_matches_dense_reference(self):
        """Same check through ``DotProductAttention`` with a KV cache."""
        for q_head_dim, v_head_dim in self.HEAD_DIM_PAIRS:
            with self.subTest(q_head_dim=q_head_dim, v_head_dim=v_head_dim):
                config = _make_mla_config(
                    q_head_dim, v_head_dim, self.NUM_HEADS
                )
                attention = DotProductAttention(
                    config=config,
                    layer_number=1,
                    attn_mask_type=AttnMaskType.causal,
                    attention_type="self",
                )
                attention.eval()

                query = paddle.randn(
                    (self.BATCH_SIZE, 1, self.NUM_HEADS, q_head_dim),
                    dtype=paddle.bfloat16,
                )
                new_key = paddle.randn(
                    (self.BATCH_SIZE, 1, self.NUM_HEADS, q_head_dim),
                    dtype=paddle.bfloat16,
                )
                new_value = paddle.randn(
                    (self.BATCH_SIZE, 1, self.NUM_HEADS, v_head_dim),
                    dtype=paddle.bfloat16,
                )
                cache = _StubKVCache(
                    paddle.randn(
                        (
                            self.BATCH_SIZE,
                            self.CACHE_LEN,
                            self.NUM_HEADS,
                            q_head_dim,
                        ),
                        dtype=paddle.bfloat16,
                    ),
                    paddle.randn(
                        (
                            self.BATCH_SIZE,
                            self.CACHE_LEN,
                            self.NUM_HEADS,
                            v_head_dim,
                        ),
                        dtype=paddle.bfloat16,
                    ),
                )

                output = attention(
                    query=query,
                    key=new_key,
                    value=new_value,
                    attention_mask=None,
                    attn_mask_startend_row_indices=self._row_indices(
                        self.kv_len
                    ),
                    attn_mask_type=AttnMaskType.causal,
                    past_key_values=cache,
                    layer_idx=0,
                    use_cache=True,
                )

                # The cache stub appended the new token, so the reference sees
                # exactly what the kernel saw.
                expected = _reference_full_attention(
                    query, cache.key_cache, cache.value_cache
                )
                self.assertEqual(
                    output.shape,
                    [self.BATCH_SIZE, 1, self.NUM_HEADS * v_head_dim],
                )
                self._assert_close(
                    output.reshape(
                        [self.BATCH_SIZE, 1, self.NUM_HEADS, v_head_dim]
                    ),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
