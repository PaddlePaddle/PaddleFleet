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

import numpy as np
import paddle
from paddle import nn


def _is_fa4_supported():
    """sink_impl only supports FlashMask V4 (Blackwell SM100, FA4 cute kernel).

    FA2 / FA3 dispatch paths have been removed from sink_impl, so the only
    requirement is a Blackwell GPU with the FA4 cute interface importable.
    """
    if not paddle.is_compiled_with_cuda():
        return False
    try:
        cap = paddle.device.cuda.get_device_capability()
    except Exception:
        return False
    if cap[0] != 10:
        return False
    try:
        from paddlefleet_ops.flash_mask.cute.interface import (  # noqa: F401
            _flash_attn_bwd,
            _flash_attn_fwd,
        )
    except Exception:
        return False
    return True


if not _is_fa4_supported():
    raise unittest.SkipTest(
        "sink_impl only supports FlashMask V4 (FA4); requires Blackwell GPU "
        "(SM100) with paddlefleet_ops FA4 cute kernel available. "
        "FA2 / FA3 are not supported."
    )


from paddlefleet.transformer.sink_impl import (
    sink_attention,
)


def gen_dense_mask_from_startend_row_indices(
    attn_mask_startend_row_indices: paddle.Tensor,
    dtype: paddle.dtype = paddle.bfloat16,
    is_causal: bool | None = None,
):
    """Recover a 4-D dense attention mask from FlashMask's
    ``startend_row_indices`` representation.
    """
    if (
        attn_mask_startend_row_indices is not None
        and attn_mask_startend_row_indices.ndim == 3
    ):
        attn_mask_startend_row_indices = (
            attn_mask_startend_row_indices.unsqueeze(-1)
        )
    if (
        attn_mask_startend_row_indices is not None
        and attn_mask_startend_row_indices.shape[-1] == 1
    ):
        is_causal = True
    if (
        attn_mask_startend_row_indices is not None
        and attn_mask_startend_row_indices.shape[-1] == 4
    ):
        is_causal = False

    if is_causal is None:
        raise ValueError(
            "The `is_causal` argument must be specified when recovering the "
            "dense attention mask from the column-wise sparse attention mask "
            "row indices."
        )

    batch_size, num_head, seq_len, bound_num = (
        attn_mask_startend_row_indices.shape
    )
    has_end = (is_causal and bound_num == 2) or (
        (not is_causal) and bound_num == 4
    )

    attention_mask = paddle.ones([seq_len, seq_len], dtype="bool").expand(
        [batch_size, num_head, seq_len, seq_len]
    )
    if is_causal:
        attention_mask = paddle.tril(attention_mask)

    base = (
        paddle.arange(seq_len, dtype="int32")
        .unsqueeze(1)
        .expand([batch_size, num_head, -1, seq_len])
    )

    mask_indices = attn_mask_startend_row_indices.transpose([0, 1, 3, 2])

    downstart_mask_indices = mask_indices[:, :, 0:1, :]
    downstart_mask_indices = downstart_mask_indices.expand(
        [batch_size, num_head, seq_len, -1]
    )
    lower_tri = base < downstart_mask_indices
    if has_end:
        downend_mask_indices = mask_indices[:, :, 1:2, :]
        downend_mask_indices = downend_mask_indices.expand(
            [batch_size, num_head, seq_len, -1]
        )
        lower_tri = paddle.logical_or(lower_tri, base >= downend_mask_indices)

    attention_mask = paddle.logical_and(attention_mask, lower_tri)

    if not is_causal:
        if has_end:
            upstart_mask_indices = mask_indices[:, :, 2:3, :]
            upstart_mask_indices = upstart_mask_indices.expand(
                [batch_size, num_head, seq_len, -1]
            )
            upend_mask_indices = mask_indices[:, :, 3:4, :]
            upend_mask_indices = upend_mask_indices.expand(
                [batch_size, num_head, seq_len, -1]
            )
            upper_tri = base >= upend_mask_indices
            upper_tri = paddle.logical_or(
                upper_tri, base < upstart_mask_indices
            )
        else:
            upend_mask_indices = mask_indices[:, :, 1:2, :]
            upend_mask_indices = upend_mask_indices.expand(
                [batch_size, num_head, seq_len, -1]
            )
            upper_tri = base >= upend_mask_indices

        attention_mask = paddle.logical_and(attention_mask, upper_tri)

    attention_mask = paddle.scale(
        x=attention_mask.astype(dtype),
        scale=1000000.0,
        bias=-1.0,
        bias_after_scale=False,
    )
    return attention_mask


def eager_attention_forward(module, query, key, value, scaling, **kwargs):
    """
    Minimal eager-path implementation: naive matmul softmax for the smoke test only.
    Inputs are [B, H, S, D]; returns (attn_output[B, S, H*D], None).
    """
    k_t = paddle.transpose(key, perm=[0, 1, 3, 2])
    attn_weights = paddle.matmul(query, k_t) * scaling
    probs = nn.functional.softmax(
        attn_weights, axis=-1, dtype=attn_weights.dtype
    )
    out = paddle.matmul(probs, value)
    out = paddle.transpose(out, perm=[0, 2, 1, 3]).contiguous()
    out = paddle.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])
    return out, None


def sdpa_attention_forward(
    module, query, key, value, scaling, is_causal=False, sink=None, **kwargs
):
    """
    SDPA-style attention entry. Inputs are [B, H, S, D]; forwards to
    sink_attention (which expects [B, S, H, D]). When sink is None,
    falls back to native SDPA.
    """
    q = paddle.transpose(query, perm=[0, 2, 1, 3])
    k = paddle.transpose(key, perm=[0, 2, 1, 3])
    v = paddle.transpose(value, perm=[0, 2, 1, 3])
    if sink is None:
        out = paddle.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=is_causal,
            training=True,
        )
    else:
        out = sink_attention(
            q,
            k,
            v,
            sink,
            startend_row_indices=None,
            softmax_scale=scaling,
            causal=is_causal,
        )
    out = paddle.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])
    return out, None


def flashmask_attention_forward(
    module,
    query,
    key,
    value,
    scaling,
    attn_mask_startend_row_indices,
    sink=None,
    is_causal=False,
    **kwargs,
):
    """FlashMask-style attention entry that routes through sink_attention."""
    q = paddle.transpose(query, perm=[0, 2, 1, 3])
    k = paddle.transpose(key, perm=[0, 2, 1, 3])
    v = paddle.transpose(value, perm=[0, 2, 1, 3])
    out = sink_attention(
        q,
        k,
        v,
        sink,
        startend_row_indices=attn_mask_startend_row_indices,
        softmax_scale=scaling,
        causal=is_causal,
    )
    out = paddle.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])
    return out, None


# Minimal dict-based registry of attention implementations, used by the tests
# below to drive each path through a single dispatch table.
ALL_ATTENTION_FUNCTIONS = {
    "eager": eager_attention_forward,
    "sdpa": sdpa_attention_forward,
    "flashmask": flashmask_attention_forward,
}


def flashmask_to_densemask(
    startend_row_indices, num_key_value_groups, dtype, causal=True
):
    """
    Helper function to convert the sparse `startend_row_indices` format, used by FlashMask,
    into a dense attention mask tensor that can be used by naive attention implementations.
    """
    bz, num_head, seq_len, bound_num = startend_row_indices.shape
    m = paddle.zeros((bz, num_head, seq_len, seq_len), dtype=dtype)
    has_end = (causal and bound_num == 2) or ((not causal) and bound_num == 4)

    # Iterate through batch, heads, and sequence length to build the dense mask
    for bi in range(bz):
        for hi in range(num_head):
            for j in range(seq_len):  # j represents the key/column index
                downstart = startend_row_indices[bi, hi, j, 0].item()
                if has_end:
                    downend = startend_row_indices[bi, hi, j, 1].item()
                    m[bi, hi, downstart:downend, j] = -np.inf
                else:
                    m[bi, hi, downstart:, j] = -np.inf

                if causal:
                    # For causal attention, mask out all future tokens
                    m[bi, hi, j + 1 :, j] = -np.inf
                else:
                    # For non-causal, use the provided upper bounds
                    if has_end:
                        upstart = startend_row_indices[bi, hi, j, 2].item()
                        upend = startend_row_indices[bi, hi, j, 3].item()
                        m[bi, hi, upstart:upend, j] = -np.inf
                    else:
                        upend = startend_row_indices[bi, hi, j, 1].item()
                        m[bi, hi, :upend, j] = -np.inf

    # If using Grouped-Query Attention (GQA), the mask for KV heads must be
    # expanded to match the number of Query heads.
    if num_key_value_groups > 1:
        m = m.unsqueeze(2).expand(
            [bz, num_head, num_key_value_groups, seq_len, seq_len]
        )
        num_q_heads = num_head * num_key_value_groups
        m = m.reshape([bz, num_q_heads, seq_len, seq_len])

    # The final mask shape is [B, H, S, S] to match the attention weights matrix.
    return m


class TestGenDenseMask(unittest.TestCase):
    """Tests for gen_dense_mask_from_startend_row_indices."""

    def test_causal_1_bound(self):
        """Test causal mask with single start bound (ndim=3 input)."""
        batch, num_head, seq_len = 1, 2, 8
        # Create startend_row_indices with shape [B, H, S] (ndim=3)
        indices = paddle.zeros([batch, num_head, seq_len], dtype="int32")
        # Set start indices at or below diagonal
        for j in range(seq_len):
            indices[:, :, j] = j

        mask = gen_dense_mask_from_startend_row_indices(
            indices, dtype=paddle.float32
        )
        self.assertEqual(mask.shape, [batch, num_head, seq_len, seq_len])
        # For causal with start at diagonal, should be a standard causal mask
        # Check that upper triangle is masked (negative values)
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                self.assertTrue(mask[0, 0, i, j].item() < 0)

    def test_causal_2_bounds(self):
        """Test causal mask with start and end bounds."""
        batch, num_head, seq_len = 1, 2, 8
        # Shape [B, H, S, 2]: start and end
        indices = paddle.zeros([batch, num_head, seq_len, 2], dtype="int32")
        for j in range(seq_len):
            indices[:, :, j, 0] = j  # start at diagonal
            indices[:, :, j, 1] = min(j + 3, seq_len)  # end 3 after

        mask = gen_dense_mask_from_startend_row_indices(
            indices, dtype=paddle.float32, is_causal=True
        )
        self.assertEqual(mask.shape, [batch, num_head, seq_len, seq_len])

    def test_non_causal_2_bounds(self):
        """Test non-causal mask with 2 bounds (down_start + up_end)."""
        batch, num_head, seq_len = 1, 2, 8
        # Shape [B, H, S, 2]: down_start and up_end
        indices = paddle.zeros([batch, num_head, seq_len, 2], dtype="int32")
        for j in range(seq_len):
            indices[:, :, j, 0] = max(0, j - 2)  # down_start
            indices[:, :, j, 1] = max(0, j - 1)  # up_end

        mask = gen_dense_mask_from_startend_row_indices(
            indices, dtype=paddle.float32, is_causal=False
        )
        self.assertEqual(mask.shape, [batch, num_head, seq_len, seq_len])

    def test_non_causal_4_bounds(self):
        """Test non-causal mask with 4 bounds."""
        batch, num_head, seq_len = 1, 2, 8
        # Shape [B, H, S, 4]: down_start, down_end, up_start, up_end
        indices = paddle.zeros([batch, num_head, seq_len, 4], dtype="int32")
        for j in range(seq_len):
            indices[:, :, j, 0] = j  # down_start
            indices[:, :, j, 1] = min(j + 2, seq_len)  # down_end
            indices[:, :, j, 2] = max(0, j - 2)  # up_start
            indices[:, :, j, 3] = j  # up_end

        mask = gen_dense_mask_from_startend_row_indices(
            indices, dtype=paddle.float32, is_causal=False
        )
        self.assertEqual(mask.shape, [batch, num_head, seq_len, seq_len])

    def test_raises_without_is_causal(self):
        """Test that ValueError is raised when is_causal is not specified and can't be inferred."""
        batch, num_head, seq_len = 1, 2, 8
        # 2 bounds could be causal or non-causal, so is_causal must be given
        indices = paddle.zeros([batch, num_head, seq_len, 2], dtype="int32")
        with self.assertRaises(ValueError):
            gen_dense_mask_from_startend_row_indices(
                indices, dtype=paddle.float32
            )


class TestAttentionInterface(unittest.TestCase):
    """
    Unit tests for the high-level attention dispatch helpers defined above.
    This class tests both the callability and numerical correctness of different
    attention implementations (e.g., sdpa, flashmask) against a naive reference.
    """

    def gen_random_flashmask(self, bz, num_head, seqlen, has_end, causal):
        """Generates a random sparse mask in the FlashMask format [start, end]."""
        mask_num = 1
        if not causal:
            mask_num *= 2
        if has_end:
            mask_num *= 2

        m = np.random.randint(0, seqlen, (bz, num_head, seqlen, mask_num))
        diag = np.arange(seqlen).reshape((1, 1, seqlen))

        # Ensure start index is not after the diagonal
        m[:, :, :, 0] = np.maximum(diag, m[:, :, :, 0])

        if not causal:
            if has_end:
                raise NotImplementedError
            # Ensure end index is after the start index
            m[:, :, :, 1] = np.minimum(diag + 1, m[:, :, :, 1])
        else:
            m[:, :, :, 0] = diag  # For causal, start is always the diagonal
            if has_end:
                m[:, :, :, 1] = m[:, :, :, 0] + np.random.randint(
                    1, seqlen, m[:, :, :, 0].shape
                )
                m[:, :, :, 1] = np.minimum(seqlen, m[:, :, :, 1])

        return paddle.to_tensor(m, dtype="int32")

    def setUp(self):
        """Set up common parameters and tensors for all tests in this class."""
        paddle.seed(92)  # Set a fixed seed for reproducibility
        self.batch_size = 1
        self.seq_len = 1024
        self.num_heads = 64
        self.head_dim = 64

        self.scaling = self.head_dim**-0.5
        self.training = True
        self.dtype = "bfloat16"

        # Tensors are created in the [batch, seq_len, num_heads, head_dim] layout.
        # This setup configures a Multi-Head Attention (MHA) scenario because the
        # number of heads for key and value is the same as for query.
        self.query = paddle.rand(
            [self.batch_size, self.seq_len, self.num_heads, self.head_dim],
            dtype=self.dtype,
        )
        self.key = paddle.rand(
            [self.batch_size, self.seq_len, self.num_heads, self.head_dim],
            dtype=self.dtype,
        )
        self.value = paddle.rand(
            [self.batch_size, self.seq_len, self.num_heads, self.head_dim],
            dtype=self.dtype,
        )
        self.sink = paddle.rand([self.num_heads], dtype=self.dtype)

        # Flashmask is generated based on the number of attention heads.
        self.startend_row_indices = self.gen_random_flashmask(
            self.batch_size,
            self.num_heads,
            self.seq_len,
            has_end=False,
            causal=False,
        )

    def assert_tensor_close(self, a, b, atol=1e-2, rtol=1e-2):
        """
        Assert that two tensors are close within specified tolerances.
        Converts tensors to float32 before comparison for better stability.
        """
        # Cast to float32 to avoid precision issues with bfloat16 during comparison
        a = a.to("float32")
        b = b.to("float32")
        self.assertTrue(
            paddle.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True),
            f"Tensors are not close.\n"
            f"Max Abs Error: {paddle.max(paddle.abs(a - b)).item()}\n"
            f"Max Rel Error: {paddle.max(paddle.abs(a - b) / (paddle.abs(b) + 1e-9)).item()}",
        )

    def naive_attn_sink(
        self,
        query: paddle.Tensor,
        key: paddle.Tensor,
        value: paddle.Tensor,
        sink: paddle.Tensor,
        attention_mask: paddle.Tensor | None,
        scaling: float,
        dropout: float = 0.0,
        num_key_value_groups: int = 1,
        **kwargs,
    ):
        """
        A naive reference implementation of attention with a 'sink' mechanism.
        This serves as the ground truth for correctness validation. It follows the
        standard attention formula step-by-step.
        """
        # Step 1: Reshape tensors from [B, S, H, D] to [B, H, S, D] for matrix multiplication
        query_states = paddle.transpose(query, perm=[0, 2, 1, 3])
        key_states = paddle.transpose(key, perm=[0, 2, 1, 3])
        value_states = paddle.transpose(value, perm=[0, 2, 1, 3])

        # Step 2: Transpose key for matmul: [B, H, S, D] -> [B, H, D, S]
        key_states = paddle.transpose(key_states, perm=[0, 1, 3, 2])

        # Step 3: Calculate attention scores (Query @ Key^T) and apply scaling
        attn_weights = paddle.matmul(query_states, key_states) * scaling

        # Step 4: Apply the attention mask if provided
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-1]]
            attn_weights = attn_weights + causal_mask

        # Step 5: Prepare and concatenate the sink logits. The sink is a special token
        # that every other token can attend to, preventing the attention from collapsing.
        sinks = sink.reshape(shape=[1, -1, 1, 1]).expand(
            shape=[query_states.shape[0], -1, query_states.shape[-2], -1]
        )
        combined_logits = paddle.cat(x=[attn_weights, sinks], axis=-1)

        # Step 6: Apply softmax over the combined logits (scores + sink)
        combined_logits = combined_logits - paddle.max(
            combined_logits, axis=-1, keepdim=True
        )
        probs = nn.functional.softmax(
            combined_logits, axis=-1, dtype=combined_logits.dtype
        )

        # Step 7: Separate the attention probabilities from the sink probabilities
        scores = probs[..., :-1]

        # Step 8: Apply dropout to the scores
        attn_weights = nn.functional.dropout(scores, p=dropout, training=True)

        # Step 9: Compute the weighted sum of values (Scores @ Value)
        attn_output = paddle.matmul(attn_weights, value_states)

        # Step 10: Reshape the output back to [B, S, H, D] and flatten the head dimension
        attn_output = paddle.transpose(
            attn_output, perm=[0, 2, 1, 3]
        ).contiguous()
        attn_output = paddle.reshape(
            x=attn_output,
            shape=[0, 0, attn_output.shape[2] * attn_output.shape[3]],
        )

        return attn_output

    def test_forward_calls_correct_function(self):
        """
        A simple 'smoke test' to ensure that all attention interfaces can be
        called with the configured tensors without raising an error.
        """
        # Test the basic eager implementation
        eager_interface = ALL_ATTENTION_FUNCTIONS["eager"]
        eager_interface(
            self,
            self.query.transpose([0, 2, 1, 3]),
            self.key.transpose([0, 2, 1, 3]),
            self.value.transpose([0, 2, 1, 3]),
            scaling=self.scaling,
        )

        # Test the SDPA implementation (without and with sink)
        sdpa_interface = ALL_ATTENTION_FUNCTIONS["sdpa"]
        sdpa_interface(
            self,
            self.query.transpose([0, 2, 1, 3]),
            self.key.transpose([0, 2, 1, 3]),
            self.value.transpose([0, 2, 1, 3]),
            scaling=self.scaling,
        )
        sdpa_interface(
            self,
            self.query.transpose([0, 2, 1, 3]),
            self.key.transpose([0, 2, 1, 3]),
            self.value.transpose([0, 2, 1, 3]),
            sink=self.sink,
            scaling=self.scaling,
        )

        # Test the FlashMask implementation with its specific arguments
        flashmask_interface = ALL_ATTENTION_FUNCTIONS["flashmask"]
        flashmask_interface(
            self,
            self.query.transpose([0, 2, 1, 3]),
            self.key.transpose([0, 2, 1, 3]),
            self.value.transpose([0, 2, 1, 3]),
            scaling=self.scaling,
            attn_mask_startend_row_indices=self.startend_row_indices,
            sink=self.sink,
        )

    def test_correctness(self):
        """
        Verifies the numerical correctness of optimized attention implementations
        against the naive reference implementation.
        """
        # --- Test 1: SDPA with Causal Mask and Sink ---

        # Get the output from the optimized SDPA implementation
        sdpa_interface = ALL_ATTENTION_FUNCTIONS["sdpa"]
        sdpa_output, _ = sdpa_interface(
            self,
            self.query.transpose([0, 2, 1, 3]),
            self.key.transpose([0, 2, 1, 3]),
            self.value.transpose([0, 2, 1, 3]),
            sink=self.sink,
            scaling=self.scaling,
            is_causal=True,
        )

        # Create the ground truth dense causal mask for the naive implementation
        causal_mask = paddle.triu(
            paddle.full(
                shape=[self.seq_len, self.seq_len],
                fill_value=float("-inf"),
                dtype=self.dtype,
            ),
            diagonal=1,
        )
        causal_mask = (
            causal_mask.unsqueeze(0)
            .unsqueeze(0)
            .expand(shape=[self.batch_size, self.num_heads, -1, -1])
        )

        # Get the output from the naive reference implementation
        eager_output_causal = self.naive_attn_sink(
            self.query,
            self.key,
            self.value,
            self.sink,
            causal_mask,
            self.scaling,
        )

        # Compare the results from the optimized and naive implementations
        self.assert_tensor_close(sdpa_output, eager_output_causal)

        # --- Test 2: FlashMask with Non-Causal Mask and Sink ---

        # Get the output from the optimized FlashMask implementation
        flashmask_interface = ALL_ATTENTION_FUNCTIONS["flashmask"]
        flashmask_output, _ = flashmask_interface(
            self,
            self.query.transpose([0, 2, 1, 3]),
            self.key.transpose([0, 2, 1, 3]),
            self.value.transpose([0, 2, 1, 3]),
            scaling=self.scaling,
            attn_mask_startend_row_indices=self.startend_row_indices,
            sink=self.sink,
            is_causal=False,
        )

        # Create the ground truth dense mask from the FlashMask sparse format
        dense_mask = flashmask_to_densemask(
            self.startend_row_indices,
            num_key_value_groups=1,
            dtype=self.dtype,
            causal=False,
        )
        # Get the output from the naive reference implementation
        eager_output_flashmask = self.naive_attn_sink(
            self.query,
            self.key,
            self.value,
            self.sink,
            dense_mask,
            self.scaling,
        )

        # Compare the results
        self.assert_tensor_close(flashmask_output, eager_output_flashmask)


class TestBackward(unittest.TestCase):
    """Tests for backward pass through FlashMaskSinkPyLayer."""

    def setUp(self):
        paddle.seed(42)
        self.batch_size = 1
        self.seq_len = 1024
        self.num_heads = 8
        self.head_dim = 64
        self.dtype = "bfloat16"
        self.scaling = self.head_dim**-0.5

    def _make_tensors(self, num_kv_heads=None, stop_gradient_sink=False):
        """Create tensors with gradients enabled."""
        if num_kv_heads is None:
            num_kv_heads = self.num_heads

        query = paddle.rand(
            [self.batch_size, self.seq_len, self.num_heads, self.head_dim],
            dtype=self.dtype,
        )
        key = paddle.rand(
            [self.batch_size, self.seq_len, num_kv_heads, self.head_dim],
            dtype=self.dtype,
        )
        value = paddle.rand(
            [self.batch_size, self.seq_len, num_kv_heads, self.head_dim],
            dtype=self.dtype,
        )
        sink = paddle.rand([self.num_heads], dtype=self.dtype)

        query.stop_gradient = False
        key.stop_gradient = False
        value.stop_gradient = False
        sink.stop_gradient = stop_gradient_sink

        return query, key, value, sink

    def test_backward_causal_mha(self):
        """Test backward pass with causal MHA (covers FA backward dispatch, LSE compat, dtype cast)."""
        query, key, value, sink = self._make_tensors()

        out = sink_attention(
            query,
            key,
            value,
            sink,
            causal=True,
            softmax_scale=self.scaling,
        )
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(query.grad)
        self.assertIsNotNone(key.grad)
        self.assertIsNotNone(value.grad)
        self.assertIsNotNone(sink.grad)
        self.assertEqual(query.grad.shape, query.shape)
        self.assertEqual(key.grad.shape, key.shape)
        self.assertEqual(value.grad.shape, value.shape)

    def test_backward_sink_stop_gradient(self):
        """Test backward when sink has stop_gradient=True returns None for sink grad."""
        query, key, value, sink = self._make_tensors(stop_gradient_sink=True)

        out = sink_attention(
            query,
            key,
            value,
            sink,
            causal=True,
            softmax_scale=self.scaling,
        )
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(query.grad)
        self.assertIsNotNone(key.grad)
        self.assertIsNotNone(value.grad)
        # sink should have no gradient since stop_gradient=True
        self.assertIsNone(sink.grad)

    def test_backward_flashmask(self):
        """Test backward with FlashMask path (covers flashmask backward dispatch)."""
        query, key, value, sink = self._make_tensors()

        # Generate non-causal flashmask indices
        indices = np.zeros(
            (self.batch_size, self.num_heads, self.seq_len, 2), dtype="int32"
        )
        diag = np.arange(self.seq_len).reshape((1, 1, self.seq_len))
        indices[:, :, :, 0] = diag  # down_start at diagonal
        indices[:, :, :, 1] = 0  # up_end at 0
        startend_row_indices = paddle.to_tensor(indices, dtype="int32")

        out = sink_attention(
            query,
            key,
            value,
            sink,
            startend_row_indices=startend_row_indices,
            causal=False,
            softmax_scale=self.scaling,
        )
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(query.grad)
        self.assertIsNotNone(key.grad)
        self.assertIsNotNone(value.grad)
        self.assertIsNotNone(sink.grad)

    def test_backward_flashmask_sink_stop_gradient(self):
        """Test backward FlashMask path with sink stop_gradient=True."""
        query, key, value, sink = self._make_tensors(stop_gradient_sink=True)

        indices = np.zeros(
            (self.batch_size, self.num_heads, self.seq_len, 2), dtype="int32"
        )
        diag = np.arange(self.seq_len).reshape((1, 1, self.seq_len))
        indices[:, :, :, 0] = diag
        indices[:, :, :, 1] = 0
        startend_row_indices = paddle.to_tensor(indices, dtype="int32")

        out = sink_attention(
            query,
            key,
            value,
            sink,
            startend_row_indices=startend_row_indices,
            causal=False,
            softmax_scale=self.scaling,
        )
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(query.grad)
        self.assertIsNone(sink.grad)

    def test_backward_gqa(self):
        """Test backward with GQA (num_q_heads > num_kv_heads) computes correct grad shapes."""
        num_kv_heads = self.num_heads // 4  # 2 KV heads for 8 Q heads
        query, key, value, sink = self._make_tensors(num_kv_heads=num_kv_heads)

        out = sink_attention(
            query,
            key,
            value,
            sink,
            causal=True,
            softmax_scale=self.scaling,
        )
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(query.grad)
        self.assertIsNotNone(key.grad)
        self.assertIsNotNone(value.grad)
        self.assertEqual(key.grad.shape, key.shape)
        self.assertEqual(value.grad.shape, value.shape)


def _is_sm90():
    """Check if the GPU is Hopper (SM90), required for FA3.

    FA3 kernels target Hopper only; on Blackwell (SM100) FA3 will not run.
    """
    if not paddle.is_compiled_with_cuda():
        return False
    try:
        cap = paddle.device.cuda.get_device_capability()
        return cap[0] == 9
    except Exception:
        return False


def _is_sm100():
    """Check if the GPU is Blackwell (SM100), required for FA4."""
    if not paddle.is_compiled_with_cuda():
        return False
    try:
        cap = paddle.device.cuda.get_device_capability()
        return cap[0] == 10
    except Exception:
        return False


class TestFA3Path(unittest.TestCase):
    """Tests for the FA3 code path on Hopper GPUs."""

    @classmethod
    def setUpClass(cls):
        cls.has_fa3 = (
            hasattr(paddle.base.libpaddle.pir.ops, "flash_attn_v3")
            and _is_sm90()
        )

    def setUp(self):
        if not self.has_fa3:
            self.skipTest("flash_attn_v3 requires Hopper GPU (SM90)")
        paddle.seed(42)
        self.batch_size = 1
        self.seq_len = 1024
        self.num_heads = 8
        self.head_dim = 64
        self.dtype = "bfloat16"
        self.scaling = self.head_dim**-0.5
        # Set FA version to 3
        paddle.base.framework.set_flags({"FLAGS_flash_attn_version": 3})

    def tearDown(self):
        # Restore default
        paddle.base.framework.set_flags({"FLAGS_flash_attn_version": 2})

    def test_fa3_forward_causal(self):
        """Test FA3 causal forward produces correct output shape."""
        query = paddle.rand(
            [self.batch_size, self.seq_len, self.num_heads, self.head_dim],
            dtype=self.dtype,
        )
        key = paddle.rand(
            [self.batch_size, self.seq_len, self.num_heads, self.head_dim],
            dtype=self.dtype,
        )
        value = paddle.rand(
            [self.batch_size, self.seq_len, self.num_heads, self.head_dim],
            dtype=self.dtype,
        )
        sink = paddle.rand([self.num_heads], dtype=self.dtype)

        out = sink_attention(
            query,
            key,
            value,
            sink,
            causal=True,
            softmax_scale=self.scaling,
        )
        self.assertEqual(
            out.shape,
            [self.batch_size, self.seq_len, self.num_heads, self.head_dim],
        )

    def test_fa3_backward(self):
        """Test FA3 backward produces gradients for all inputs."""
        query = paddle.rand(
            [self.batch_size, self.seq_len, self.num_heads, self.head_dim],
            dtype=self.dtype,
        )
        key = paddle.rand(
            [self.batch_size, self.seq_len, self.num_heads, self.head_dim],
            dtype=self.dtype,
        )
        value = paddle.rand(
            [self.batch_size, self.seq_len, self.num_heads, self.head_dim],
            dtype=self.dtype,
        )
        sink = paddle.rand([self.num_heads], dtype=self.dtype)
        query.stop_gradient = False
        key.stop_gradient = False
        value.stop_gradient = False
        sink.stop_gradient = False

        out = sink_attention(
            query,
            key,
            value,
            sink,
            causal=True,
            softmax_scale=self.scaling,
        )
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(query.grad)
        self.assertIsNotNone(key.grad)
        self.assertIsNotNone(value.grad)

    def test_fa3_flashmask_forward(self):
        """Test FlashMask with FA3 forward produces correct output shape."""
        has_flashmask_v2 = hasattr(
            paddle.base.libpaddle.pir.ops, "flashmask_attention_v2"
        ) or hasattr(paddle.nn.functional, "flashmask_attention")
        if not has_flashmask_v2:
            self.skipTest("flashmask_attention not available for FA3")

        query = paddle.rand(
            [self.batch_size, self.seq_len, self.num_heads, self.head_dim],
            dtype=self.dtype,
        )
        key = paddle.rand(
            [self.batch_size, self.seq_len, self.num_heads, self.head_dim],
            dtype=self.dtype,
        )
        value = paddle.rand(
            [self.batch_size, self.seq_len, self.num_heads, self.head_dim],
            dtype=self.dtype,
        )
        sink = paddle.rand([self.num_heads], dtype=self.dtype)

        # Non-causal flashmask indices
        indices = np.zeros(
            (self.batch_size, self.num_heads, self.seq_len, 2), dtype="int32"
        )
        diag = np.arange(self.seq_len).reshape((1, 1, self.seq_len))
        indices[:, :, :, 0] = diag
        indices[:, :, :, 1] = 0
        startend_row_indices = paddle.to_tensor(indices, dtype="int32")

        out = sink_attention(
            query,
            key,
            value,
            sink,
            startend_row_indices=startend_row_indices,
            causal=False,
            softmax_scale=self.scaling,
        )
        self.assertEqual(
            out.shape,
            [self.batch_size, self.seq_len, self.num_heads, self.head_dim],
        )

    def test_fa3_flashmask_backward(self):
        """Test FlashMask with FA3 backward produces gradients."""
        has_flashmask_v2_grad = hasattr(
            paddle.base.libpaddle.pir.ops, "flashmask_attention_v2_grad"
        )
        if not has_flashmask_v2_grad:
            self.skipTest("flashmask_attention_v2_grad not available")

        query = paddle.rand(
            [self.batch_size, self.seq_len, self.num_heads, self.head_dim],
            dtype=self.dtype,
        )
        key = paddle.rand(
            [self.batch_size, self.seq_len, self.num_heads, self.head_dim],
            dtype=self.dtype,
        )
        value = paddle.rand(
            [self.batch_size, self.seq_len, self.num_heads, self.head_dim],
            dtype=self.dtype,
        )
        sink = paddle.rand([self.num_heads], dtype=self.dtype)
        query.stop_gradient = False
        key.stop_gradient = False
        value.stop_gradient = False
        sink.stop_gradient = False

        indices = np.zeros(
            (self.batch_size, self.num_heads, self.seq_len, 2), dtype="int32"
        )
        diag = np.arange(self.seq_len).reshape((1, 1, self.seq_len))
        indices[:, :, :, 0] = diag
        indices[:, :, :, 1] = 0
        startend_row_indices = paddle.to_tensor(indices, dtype="int32")

        out = sink_attention(
            query,
            key,
            value,
            sink,
            startend_row_indices=startend_row_indices,
            causal=False,
            softmax_scale=self.scaling,
        )
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(query.grad)
        self.assertIsNotNone(key.grad)


class TestFA4Path(unittest.TestCase):
    """Tests for the FA4 code path on Blackwell GPUs (SM100, e.g. B100)."""

    @classmethod
    def setUpClass(cls):
        cls.has_fa4 = _is_sm100()
        if cls.has_fa4:
            try:
                from paddlefleet.ops.flash_mask.cute.interface import (  # noqa: F401
                    _flash_attn_bwd,
                    _flash_attn_fwd,
                )
            except Exception:
                cls.has_fa4 = False

    def setUp(self):
        if not self.has_fa4:
            self.skipTest("FA4 requires Blackwell GPU (SM100)")
        paddle.seed(42)
        self.batch_size = 1
        self.seq_len = 1024
        self.num_heads = 8
        self.head_dim = 64
        self.dtype = "bfloat16"
        self.scaling = self.head_dim**-0.5
        paddle.base.framework.set_flags({"FLAGS_flash_attn_version": 4})

    def tearDown(self):
        paddle.base.framework.set_flags({"FLAGS_flash_attn_version": 2})

    def test_fa4_forward_causal(self):
        """Test FA4 causal forward produces correct output shape."""
        query = paddle.rand(
            [self.batch_size, self.seq_len, self.num_heads, self.head_dim],
            dtype=self.dtype,
        )
        key = paddle.rand(
            [self.batch_size, self.seq_len, self.num_heads, self.head_dim],
            dtype=self.dtype,
        )
        value = paddle.rand(
            [self.batch_size, self.seq_len, self.num_heads, self.head_dim],
            dtype=self.dtype,
        )
        sink = paddle.rand([self.num_heads], dtype=self.dtype)

        out = sink_attention(
            query,
            key,
            value,
            sink,
            causal=True,
            softmax_scale=self.scaling,
        )
        self.assertEqual(
            out.shape,
            [self.batch_size, self.seq_len, self.num_heads, self.head_dim],
        )

    def test_fa4_backward(self):
        """Test FA4 backward produces gradients for all inputs."""
        query = paddle.rand(
            [self.batch_size, self.seq_len, self.num_heads, self.head_dim],
            dtype=self.dtype,
        )
        key = paddle.rand(
            [self.batch_size, self.seq_len, self.num_heads, self.head_dim],
            dtype=self.dtype,
        )
        value = paddle.rand(
            [self.batch_size, self.seq_len, self.num_heads, self.head_dim],
            dtype=self.dtype,
        )
        sink = paddle.rand([self.num_heads], dtype=self.dtype)
        query.stop_gradient = False
        key.stop_gradient = False
        value.stop_gradient = False
        sink.stop_gradient = False

        out = sink_attention(
            query,
            key,
            value,
            sink,
            causal=True,
            softmax_scale=self.scaling,
        )
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(query.grad)
        self.assertIsNotNone(key.grad)
        self.assertIsNotNone(value.grad)

    def test_fa4_flashmask_forward(self):
        """Test FlashMask with FA4 forward produces correct output shape."""
        query = paddle.rand(
            [self.batch_size, self.seq_len, self.num_heads, self.head_dim],
            dtype=self.dtype,
        )
        key = paddle.rand(
            [self.batch_size, self.seq_len, self.num_heads, self.head_dim],
            dtype=self.dtype,
        )
        value = paddle.rand(
            [self.batch_size, self.seq_len, self.num_heads, self.head_dim],
            dtype=self.dtype,
        )
        sink = paddle.rand([self.num_heads], dtype=self.dtype)

        indices = np.zeros(
            (self.batch_size, self.num_heads, self.seq_len, 2), dtype="int32"
        )
        diag = np.arange(self.seq_len).reshape((1, 1, self.seq_len))
        indices[:, :, :, 0] = diag
        indices[:, :, :, 1] = 0
        startend_row_indices = paddle.to_tensor(indices, dtype="int32")

        out = sink_attention(
            query,
            key,
            value,
            sink,
            startend_row_indices=startend_row_indices,
            causal=False,
            softmax_scale=self.scaling,
        )
        self.assertEqual(
            out.shape,
            [self.batch_size, self.seq_len, self.num_heads, self.head_dim],
        )

    def test_fa4_flashmask_backward(self):
        """Test FlashMask with FA4 backward produces gradients."""
        query = paddle.rand(
            [self.batch_size, self.seq_len, self.num_heads, self.head_dim],
            dtype=self.dtype,
        )
        key = paddle.rand(
            [self.batch_size, self.seq_len, self.num_heads, self.head_dim],
            dtype=self.dtype,
        )
        value = paddle.rand(
            [self.batch_size, self.seq_len, self.num_heads, self.head_dim],
            dtype=self.dtype,
        )
        sink = paddle.rand([self.num_heads], dtype=self.dtype)
        query.stop_gradient = False
        key.stop_gradient = False
        value.stop_gradient = False
        sink.stop_gradient = False

        indices = np.zeros(
            (self.batch_size, self.num_heads, self.seq_len, 2), dtype="int32"
        )
        diag = np.arange(self.seq_len).reshape((1, 1, self.seq_len))
        indices[:, :, :, 0] = diag
        indices[:, :, :, 1] = 0
        startend_row_indices = paddle.to_tensor(indices, dtype="int32")

        out = sink_attention(
            query,
            key,
            value,
            sink,
            startend_row_indices=startend_row_indices,
            causal=False,
            softmax_scale=self.scaling,
        )
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(query.grad)
        self.assertIsNotNone(key.grad)


class TestLSEShapeCompat(unittest.TestCase):
    """Test LSE shape compatibility when kernel returns rounded sequence length."""

    def test_lse_rounded_shape(self):
        """Test that rounded LSE shape is handled correctly in forward."""
        from unittest.mock import patch

        if not _is_sm100():
            self.skipTest("FA4 requires Blackwell GPU (SM100)")

        paddle.seed(42)
        batch, seq, heads, dim = 1, 100, 4, 64
        dtype = "bfloat16"

        query = paddle.rand([batch, seq, heads, dim], dtype=dtype)
        key = paddle.rand([batch, seq, heads, dim], dtype=dtype)
        value = paddle.rand([batch, seq, heads, dim], dtype=dtype)
        sink = paddle.rand([heads], dtype=dtype)

        # Mock the FA4 forward op to return a rounded LSE
        mock_out = paddle.rand([batch, seq, heads, dim], dtype=dtype)
        # Simulate a rounded LSE (e.g., 128 instead of 100)
        rounded_seq = 128
        mock_lse = paddle.rand([batch, heads, rounded_seq], dtype="float32")

        with patch(
            "paddlefleet.transformer.sink_impl._flash_attn_fwd",
            return_value=(mock_out, mock_lse),
        ):
            out = sink_attention(
                query, key, value, sink, causal=True, softmax_scale=dim**-0.5
            )
            self.assertEqual(out.shape, [batch, seq, heads, dim])


# Standard entry point to run the tests when the script is executed directly
if __name__ == "__main__":
    unittest.main()
