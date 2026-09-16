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

"""Unit tests for GatedDeltaNet module."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import nn

from paddlefleet.transformer.gated_delta_net import (
    GatedDeltaNet,
    GatedDeltaNetSublayersSpec,
    _l2norm,
    paddle_chunk_gated_delta_rule,
)
from paddlefleet.transformer.paddle_norm import WrappedPaddleNorm
from paddlefleet.transformer.transformer_config import TransformerConfig

# ---- Local stand-in layers (no fleet / TP required) ----


class BiasedLinear(nn.Layer):
    """Simple linear layer that returns (output, bias), matching ColumnParallel/RowParallel API."""

    def __init__(self, in_features, out_features, **kwargs):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x), self.linear.bias

    def backward_dw(self):
        pass


class NoBiasLinear(nn.Layer):
    """Linear layer without bias that returns (output, None)."""

    def __init__(self, in_features, out_features, **kwargs):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias_attr=False)

    def forward(self, x):
        return self.linear(x), None

    def backward_dw(self):
        pass


class SimpleRMSNorm(nn.Layer):
    """Minimal RMSNorm for testing."""

    def __init__(self, normalized_shape, eps=1e-5, **kwargs):
        super().__init__()
        self.weight = self.create_parameter(
            shape=[normalized_shape],
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.eps = eps

    def forward(self, x):
        x_float = x.astype(paddle.float32)
        rms = paddle.rsqrt(
            x_float.pow(2).mean(axis=-1, keepdim=True) + self.eps
        )
        return (x_float * rms * self.weight.astype(paddle.float32)).astype(
            x.dtype
        )


# ---- Fake ProcessGroupCollection for single-GPU testing ----


class _FakeGroup:
    """Fake process group that reports world_size=1."""

    ranks = [0]
    nranks = 1


class _FakePGCollection:
    """Minimal stand-in for ProcessGroupCollection (TP=1)."""

    def __init__(self):
        self.tp = _FakeGroup()


# ---- Test dimensions ----
HIDDEN_SIZE = 64
NUM_KEY_HEADS = 4
NUM_VALUE_HEADS = 4
KEY_HEAD_DIM = 16
VALUE_HEAD_DIM = 16
CONV_KERNEL_DIM = 4
MICRO_BATCH_SIZE = 2
SEQ_LENGTH = 32


class TestPaddleChunkGatedDeltaRule(unittest.TestCase):
    """Test the deterministic paddle_chunk_gated_delta_rule function."""

    def test_output_shape(self):
        """Output shape must match [batch, seq_len, num_heads, v_head_dim]."""
        batch, seq_len, num_heads, k_dim, v_dim = 2, 32, 4, 16, 16
        query = paddle.randn([batch, seq_len, num_heads, k_dim])
        key = paddle.randn([batch, seq_len, num_heads, k_dim])
        value = paddle.randn([batch, seq_len, num_heads, v_dim])
        g = paddle.randn([batch, seq_len, num_heads]) * 0.1
        beta = paddle.rand([batch, seq_len, num_heads])

        out, state = paddle_chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            chunk_size=16,
            output_final_state=False,
        )

        self.assertEqual(list(out.shape), [batch, seq_len, num_heads, v_dim])
        self.assertIsNone(state)
        self.assertEqual(out.dtype, query.dtype)

    def test_output_final_state(self):
        """When output_final_state=True, state should be returned."""
        batch, seq_len, num_heads, k_dim, v_dim = 1, 16, 2, 8, 8
        query = paddle.randn([batch, seq_len, num_heads, k_dim])
        key = paddle.randn([batch, seq_len, num_heads, k_dim])
        value = paddle.randn([batch, seq_len, num_heads, v_dim])
        g = paddle.randn([batch, seq_len, num_heads]) * 0.1
        beta = paddle.rand([batch, seq_len, num_heads])

        out, state = paddle_chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            chunk_size=16,
            output_final_state=True,
        )

        self.assertIsNotNone(state)
        self.assertEqual(list(state.shape), [batch, num_heads, k_dim, v_dim])

    def test_backward(self):
        """Gradients must flow through the chunked gated delta rule."""
        batch, seq_len, num_heads, k_dim, v_dim = 2, 32, 4, 16, 16
        query = paddle.randn([batch, seq_len, num_heads, k_dim])
        key = paddle.randn([batch, seq_len, num_heads, k_dim])
        value = paddle.randn([batch, seq_len, num_heads, v_dim])
        query.stop_gradient = False
        key.stop_gradient = False
        value.stop_gradient = False

        g = paddle.randn([batch, seq_len, num_heads]) * 0.1
        beta = paddle.rand([batch, seq_len, num_heads])

        out, _ = paddle_chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            chunk_size=16,
        )
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(query.grad)
        self.assertIsNotNone(key.grad)
        self.assertIsNotNone(value.grad)
        self.assertTrue(paddle.isfinite(query.grad).all().item())
        self.assertTrue(paddle.isfinite(key.grad).all().item())
        self.assertTrue(paddle.isfinite(value.grad).all().item())

    def test_seq_len_not_divisible_by_chunk(self):
        """Sequence length not divisible by chunk_size should still work (padding)."""
        batch, seq_len, num_heads, k_dim, v_dim = 1, 37, 2, 8, 8
        query = paddle.randn([batch, seq_len, num_heads, k_dim])
        key = paddle.randn([batch, seq_len, num_heads, k_dim])
        value = paddle.randn([batch, seq_len, num_heads, v_dim])
        g = paddle.randn([batch, seq_len, num_heads]) * 0.1
        beta = paddle.rand([batch, seq_len, num_heads])

        out, _ = paddle_chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            chunk_size=16,
        )
        self.assertEqual(list(out.shape), [batch, seq_len, num_heads, v_dim])
        self.assertTrue(paddle.isfinite(out).all().item())


class TestL2Norm(unittest.TestCase):
    """Test the _l2norm helper function."""

    def test_output_shape(self):
        x = paddle.randn([2, 4, 3, 16])
        y = _l2norm(x)
        self.assertEqual(list(y.shape), list(x.shape))
        self.assertEqual(y.dtype, x.dtype)

    def test_normalization(self):
        """After L2 norm, mean of squared values along last dim should be ~1."""
        x = paddle.randn([4, 8, 32])
        y = _l2norm(x)
        mean_sq = y.astype(paddle.float32).pow(2).sum(-1)
        assert paddle.allclose(
            mean_sq, paddle.ones_like(mean_sq), atol=1e-4, rtol=1e-4
        ).item()


class TestGatedDeltaNet(unittest.TestCase):
    """Test the full GatedDeltaNet module (single-GPU, no TP)."""

    def setUp(self):
        self.config = TransformerConfig(
            hidden_size=HIDDEN_SIZE,
            num_attention_heads=NUM_KEY_HEADS,
            num_hidden_layers=2,
            hidden_act=F.silu,
            rms_norm_eps=1e-5,
            normalization="RMSNorm",
            sequence_parallel=False,
            deterministic_mode=True,
        )

        sublayers_spec = GatedDeltaNetSublayersSpec(
            in_proj=NoBiasLinear,
            out_norm=SimpleRMSNorm,
            out_proj=NoBiasLinear,
        )

        self.gdn = GatedDeltaNet(
            config=self.config,
            sublayers_spec=sublayers_spec,
            layer_number=1,
            bias=False,
            conv_bias=False,
            conv_init=1.0,
            use_qk_l2norm=True,
            A_init_range=(1, 16),
            pg_collection=_FakePGCollection(),
            conv_kernel_dim=CONV_KERNEL_DIM,
            key_head_dim=KEY_HEAD_DIM,
            value_head_dim=VALUE_HEAD_DIM,
            num_key_heads=NUM_KEY_HEADS,
            num_value_heads=NUM_VALUE_HEADS,
        )

    def test_constructor(self):
        """GatedDeltaNet should instantiate with the correct sub-modules."""
        self.assertIsInstance(self.gdn, GatedDeltaNet)
        self.assertTrue(hasattr(self.gdn, "in_proj"))
        self.assertTrue(hasattr(self.gdn, "conv1d"))
        self.assertTrue(hasattr(self.gdn, "dt_bias"))
        self.assertTrue(hasattr(self.gdn, "A_log"))
        self.assertTrue(hasattr(self.gdn, "out_norm"))
        self.assertTrue(hasattr(self.gdn, "out_proj"))

        sublayers_spec = GatedDeltaNetSublayersSpec(
            in_proj=NoBiasLinear,
            out_norm=WrappedPaddleNorm,
            out_proj=NoBiasLinear,
        )

        gdn = GatedDeltaNet(
            config=self.config,
            sublayers_spec=sublayers_spec,
            layer_number=1,
            bias=False,
            conv_bias=False,
            conv_init=1.0,
            use_qk_l2norm=True,
            A_init_range=(1, 16),
            pg_collection=_FakePGCollection(),
            conv_kernel_dim=CONV_KERNEL_DIM,
            key_head_dim=KEY_HEAD_DIM,
            value_head_dim=VALUE_HEAD_DIM,
            num_key_heads=NUM_KEY_HEADS,
            num_value_heads=NUM_VALUE_HEADS,
        )

    def test_sharded_state_dict(self):
        """Check sharded_state_dict() completeness."""
        sharded_sd = self.gdn.sharded_state_dict()
        self.assertEqual(
            len(sharded_sd), 6
        )  # 13 from GatedDeltaNetSublayersSpec

    def test_parameter_shapes(self):
        """Verify key parameter shapes."""
        # conv1d: depthwise conv with groups=conv_dim
        conv_dim = (
            KEY_HEAD_DIM * NUM_KEY_HEADS * 2 + VALUE_HEAD_DIM * NUM_VALUE_HEADS
        )
        self.assertEqual(
            list(self.gdn.conv1d.weight.shape),
            [conv_dim, 1, CONV_KERNEL_DIM],
        )
        # dt_bias and A_log
        self.assertEqual(list(self.gdn.dt_bias.shape), [NUM_VALUE_HEADS])
        self.assertEqual(list(self.gdn.A_log.shape), [NUM_VALUE_HEADS])

    def test_forward_output_shape(self):
        """Forward output shape should match [batch, seq_len, hidden_size]."""
        hidden_states = paddle.randn(
            [MICRO_BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE]
        )
        attention_mask = None

        output, output_bias = self.gdn(hidden_states, attention_mask)

        self.assertEqual(output.ndim, 3)
        self.assertEqual(output.shape[0], MICRO_BATCH_SIZE)
        self.assertEqual(output.shape[1], SEQ_LENGTH)
        self.assertEqual(output.shape[2], HIDDEN_SIZE)
        self.assertEqual(output.dtype, hidden_states.dtype)

    def test_forward_output_finite(self):
        """Forward output should contain no NaN or Inf values."""
        hidden_states = paddle.randn(
            [MICRO_BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE]
        )

        output, _ = self.gdn(hidden_states, attention_mask=None)

        self.assertTrue(
            paddle.isfinite(output).all().item(),
            "Output contains NaN or Inf",
        )

    def test_backward_all_grads(self):
        """All parameters in the forward path should receive gradients."""
        hidden_states = paddle.randn(
            [MICRO_BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE]
        )
        hidden_states.stop_gradient = False

        output, output_bias = self.gdn(hidden_states, attention_mask=None)
        loss = output.sum()
        loss.backward()

        # Check input gradients
        self.assertIsNotNone(hidden_states.grad)
        self.assertTrue(paddle.isfinite(hidden_states.grad).all().item())

        # Check all parameter gradients
        params_with_grad = 0
        no_grad_params = []
        for name, param in self.gdn.named_parameters():
            if param.grad is None:
                no_grad_params.append(name)
            else:
                params_with_grad += 1
                self.assertEqual(
                    list(param.shape),
                    list(param.grad.shape),
                    f"Gradient shape mismatch for {name}",
                )
                self.assertTrue(
                    paddle.isfinite(param.grad).all().item(),
                    f"Non-finite gradients for {name}",
                )

        self.assertGreater(
            params_with_grad, 0, "No parameters received gradients"
        )
        if no_grad_params:
            print(f"  [WARNING] Parameters without gradients: {no_grad_params}")

    def test_packed_seq_not_supported(self):
        """Packed sequence should raise NotImplementedError."""
        hidden_states = paddle.randn(
            [MICRO_BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE]
        )

        with self.assertRaises(NotImplementedError):
            self.gdn(
                hidden_states, attention_mask=None, packed_seq_params="dummy"
            )


class TestGatedDeltaNetWithBias(unittest.TestCase):
    """Test GatedDeltaNet with bias enabled in linear layers and conv."""

    def setUp(self):
        self.config = TransformerConfig(
            hidden_size=HIDDEN_SIZE,
            num_attention_heads=NUM_KEY_HEADS,
            num_hidden_layers=2,
            hidden_act=F.silu,
            rms_norm_eps=1e-5,
            normalization="RMSNorm",
            sequence_parallel=False,
            deterministic_mode=True,
        )

        sublayers_spec = GatedDeltaNetSublayersSpec(
            in_proj=BiasedLinear,
            out_norm=SimpleRMSNorm,
            out_proj=BiasedLinear,
        )

        self.gdn = GatedDeltaNet(
            config=self.config,
            sublayers_spec=sublayers_spec,
            layer_number=1,
            bias=True,
            conv_bias=True,
            conv_init=0.5,
            use_qk_l2norm=True,
            A_init_range=(1, 16),
            pg_collection=_FakePGCollection(),
            conv_kernel_dim=CONV_KERNEL_DIM,
            key_head_dim=KEY_HEAD_DIM,
            value_head_dim=VALUE_HEAD_DIM,
            num_key_heads=NUM_KEY_HEADS,
            num_value_heads=NUM_VALUE_HEADS,
        )

    def test_forward_backward(self):
        """Forward and backward with bias should work correctly."""
        hidden_states = paddle.randn(
            [MICRO_BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE]
        )
        hidden_states.stop_gradient = False

        output, output_bias = self.gdn(hidden_states, attention_mask=None)

        self.assertEqual(output.shape[0], MICRO_BATCH_SIZE)
        self.assertEqual(output.shape[1], SEQ_LENGTH)
        self.assertEqual(output.shape[2], HIDDEN_SIZE)
        self.assertTrue(paddle.isfinite(output).all().item())

        loss = output.sum()
        loss.backward()

        self.assertIsNotNone(hidden_states.grad)

        # Conv1d bias should have gradient
        self.assertIsNotNone(self.gdn.conv1d.bias)
        self.assertIsNotNone(
            self.gdn.conv1d.bias.grad,
            "conv1d.bias should receive gradient",
        )


class TestGatedDeltaNetGQA(unittest.TestCase):
    """Test GatedDeltaNet with GQA (num_value_heads > num_key_heads)."""

    def setUp(self):
        self.config = TransformerConfig(
            hidden_size=HIDDEN_SIZE,
            num_attention_heads=NUM_KEY_HEADS,
            num_hidden_layers=2,
            hidden_act=F.silu,
            rms_norm_eps=1e-5,
            normalization="RMSNorm",
            sequence_parallel=False,
            deterministic_mode=True,
        )

        sublayers_spec = GatedDeltaNetSublayersSpec(
            in_proj=NoBiasLinear,
            out_norm=SimpleRMSNorm,
            out_proj=NoBiasLinear,
        )

        # GQA: 8 value heads, 4 key heads => repeat factor 2
        self.gdn = GatedDeltaNet(
            config=self.config,
            sublayers_spec=sublayers_spec,
            layer_number=1,
            bias=False,
            conv_bias=False,
            use_qk_l2norm=True,
            A_init_range=(1, 16),
            pg_collection=_FakePGCollection(),
            conv_kernel_dim=CONV_KERNEL_DIM,
            key_head_dim=KEY_HEAD_DIM,
            value_head_dim=VALUE_HEAD_DIM,
            num_key_heads=4,
            num_value_heads=8,
        )

    def test_forward_shape(self):
        """GQA should produce correct output shape."""
        hidden_states = paddle.randn(
            [MICRO_BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE]
        )
        output, _ = self.gdn(hidden_states, attention_mask=None)

        self.assertEqual(output.shape[0], MICRO_BATCH_SIZE)
        self.assertEqual(output.shape[1], SEQ_LENGTH)
        self.assertEqual(output.shape[2], HIDDEN_SIZE)
        self.assertTrue(paddle.isfinite(output).all().item())

    def test_backward(self):
        """Backward through GQA should produce finite gradients for all params."""
        hidden_states = paddle.randn(
            [MICRO_BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE]
        )
        hidden_states.stop_gradient = False

        output, _ = self.gdn(hidden_states, attention_mask=None)
        output.sum().backward()

        self.assertIsNotNone(hidden_states.grad)
        for name, param in self.gdn.named_parameters():
            if param.grad is not None:
                self.assertTrue(
                    paddle.isfinite(param.grad).all().item(),
                    f"Non-finite gradient for {name}",
                )


def _recurrent_gated_delta_reference(
    query, key, value, g, beta, initial_state=None
):
    """Independent, sequential reference for the gated delta rule.

    Implements the token-at-a-time recurrence in plain numpy/float64 -- a
    different algorithm from the chunk-parallel ``paddle_chunk_gated_delta_rule``
    under test, so a shared bug cannot pass both. The recurrence per (batch,
    head) is::

        S <- S * exp(g_t)                        # scalar decay of the state
        kv_mem = (S * k_t[:, None]).sum(axis=0)  # k_t^T S  over the key axis
        delta  = (v_t - kv_mem) * beta_t
        S <- S + outer(k_t, delta)
        o_t = (S * q_scaled_t[:, None]).sum(axis=0)

    Only ``query`` is scaled by ``1/sqrt(k_head_dim)``, matching production. No
    L2 norm is applied (callers pass raw tensors here). Returns ``(out, state)``
    with ``out`` shaped ``[b, s, h, v_dim]`` and ``state`` the final ``[b, h,
    k_dim, v_dim]`` recurrent state.
    """
    query = np.asarray(query, dtype=np.float64)
    key = np.asarray(key, dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)

    b, s, h, k_dim = query.shape
    v_dim = value.shape[-1]
    scale = 1.0 / (k_dim**0.5)
    q_scaled = query * scale

    out = np.zeros((b, s, h, v_dim), dtype=np.float64)
    final_state = np.zeros((b, h, k_dim, v_dim), dtype=np.float64)
    for bi in range(b):
        for hi in range(h):
            if initial_state is None:
                state = np.zeros((k_dim, v_dim), dtype=np.float64)
            else:
                state = np.asarray(initial_state, dtype=np.float64)[
                    bi, hi
                ].copy()
            for t in range(s):
                state = state * np.exp(g[bi, t, hi])
                k_t = key[bi, t, hi]
                v_t = value[bi, t, hi]
                kv_mem = (state * k_t[:, None]).sum(axis=0)
                delta = (v_t - kv_mem) * beta[bi, t, hi]
                state = state + k_t[:, None] * delta[None, :]
                out[bi, t, hi] = (state * q_scaled[bi, t, hi][:, None]).sum(
                    axis=0
                )
            final_state[bi, hi] = state
    return out, final_state


class TestPaddleChunkGatedDeltaRuleNumeric(unittest.TestCase):
    """Numeric correctness of paddle_chunk_gated_delta_rule vs an independent
    sequential reference (existing tests only check shapes/finiteness)."""

    def _fixed_inputs(self, batch, seq_len, heads, k_dim, v_dim, chunk_size):
        paddle.seed(20260916)
        query = paddle.randn([batch, seq_len, heads, k_dim], dtype="float32")
        key = paddle.randn([batch, seq_len, heads, k_dim], dtype="float32")
        value = paddle.randn([batch, seq_len, heads, v_dim], dtype="float32")
        # Keep g negative so the recurrence stays bounded over the sequence.
        g = (
            -paddle.abs(paddle.randn([batch, seq_len, heads], dtype="float32"))
            * 0.1
        )
        beta = paddle.rand([batch, seq_len, heads], dtype="float32")
        return query, key, value, g, beta, chunk_size

    def test_matches_sequential_reference(self):
        """Chunked output must equal the sequential recurrence element-wise."""
        query, key, value, g, beta, chunk_size = self._fixed_inputs(
            batch=2, seq_len=8, heads=3, k_dim=4, v_dim=6, chunk_size=16
        )
        out, _ = paddle_chunk_gated_delta_rule(
            query, key, value, g=g, beta=beta, chunk_size=chunk_size
        )
        ref, _ = _recurrent_gated_delta_reference(
            query.numpy(), key.numpy(), value.numpy(), g.numpy(), beta.numpy()
        )
        # Reference must carry real signal so a zeroed/degenerate output fails.
        self.assertGreater(float(np.abs(ref).max()), 1e-2)
        np.testing.assert_allclose(out.numpy(), ref, rtol=2e-2, atol=1e-3)
        # Negative control: a scaled copy must NOT satisfy the same tolerance,
        # proving the check is magnitude-sensitive (not cosine-only).
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(
                out.numpy(), ref * 2.0, rtol=2e-2, atol=1e-3
            )

    def test_single_token_closed_form(self):
        """For seq_len=1 the output is beta * (q_scaled . k) * v exactly."""
        k_dim, v_dim = 4, 8
        paddle.seed(7)
        query = paddle.randn([1, 1, 1, k_dim], dtype="float32")
        key = paddle.randn([1, 1, 1, k_dim], dtype="float32")
        value = paddle.randn([1, 1, 1, v_dim], dtype="float32")
        g = paddle.randn(
            [1, 1, 1], dtype="float32"
        )  # irrelevant: state starts 0
        beta = paddle.rand([1, 1, 1], dtype="float32")

        out, _ = paddle_chunk_gated_delta_rule(
            query, key, value, g=g, beta=beta, chunk_size=1
        )
        scale = 1.0 / (k_dim**0.5)
        q = query.numpy()[0, 0, 0] * scale
        k = key.numpy()[0, 0, 0]
        v = value.numpy()[0, 0, 0]
        expected = float(beta.numpy()[0, 0, 0]) * float(np.dot(q, k)) * v
        np.testing.assert_allclose(
            out.numpy()[0, 0, 0], expected, rtol=1e-4, atol=1e-4
        )

    def test_final_state_matches_reference(self):
        """output_final_state must return the true recurrent state values."""
        query, key, value, g, beta, chunk_size = self._fixed_inputs(
            batch=1, seq_len=8, heads=2, k_dim=4, v_dim=4, chunk_size=16
        )
        _, state = paddle_chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            chunk_size=chunk_size,
            output_final_state=True,
        )
        _, ref_state = _recurrent_gated_delta_reference(
            query.numpy(), key.numpy(), value.numpy(), g.numpy(), beta.numpy()
        )
        self.assertGreater(float(np.abs(ref_state).max()), 1e-2)
        np.testing.assert_allclose(
            state.numpy(), ref_state, rtol=2e-2, atol=1e-3
        )

    def test_initial_state_is_consumed(self):
        """A non-zero initial_state must change the output and match the
        reference seeded with the same state."""
        query, key, value, g, beta, chunk_size = self._fixed_inputs(
            batch=1, seq_len=8, heads=2, k_dim=4, v_dim=4, chunk_size=16
        )
        b, _, h, k_dim = query.shape
        v_dim = value.shape[-1]
        paddle.seed(99)
        initial = paddle.randn([b, h, k_dim, v_dim], dtype="float32")

        out_zero, _ = paddle_chunk_gated_delta_rule(
            query, key, value, g=g, beta=beta, chunk_size=chunk_size
        )
        out_init, _ = paddle_chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            chunk_size=chunk_size,
            initial_state=initial,
        )
        # Consumption: seeding the state must actually move the output.
        self.assertFalse(
            np.allclose(
                out_zero.numpy(), out_init.numpy(), rtol=1e-3, atol=1e-4
            )
        )
        ref, _ = _recurrent_gated_delta_reference(
            query.numpy(),
            key.numpy(),
            value.numpy(),
            g.numpy(),
            beta.numpy(),
            initial_state=initial.numpy(),
        )
        np.testing.assert_allclose(out_init.numpy(), ref, rtol=2e-2, atol=1e-3)


class TestL2NormIndependentReference(unittest.TestCase):
    """Exact numeric checks for _l2norm (existing tests only assert unit norm)."""

    def test_matches_independent_numpy_formula(self):
        """_l2norm(x) == x * rsqrt(sum(x^2, -1) + 1e-6), computed independently."""
        x = paddle.to_tensor(
            [[[3.0, 4.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]], dtype="float32"
        )
        out = _l2norm(x)
        xn = x.numpy().astype(np.float64)
        ref = xn / np.sqrt((xn**2).sum(axis=-1, keepdims=True) + 1e-6)
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-5, atol=1e-6)

    def test_normalizes_each_last_axis_row_independently(self):
        """Rows with different magnitudes must each be scaled by their own norm,
        not by a shared/global norm."""
        x = paddle.to_tensor([[[3.0, 4.0], [30.0, 40.0]]], dtype="float32")
        out = _l2norm(x).numpy()
        # 3-4-5 triangle: both rows point the same direction after per-row norm.
        expected_dir = np.array([0.6, 0.8])
        np.testing.assert_allclose(
            out[0, 0], expected_dir, rtol=1e-4, atol=1e-4
        )
        np.testing.assert_allclose(
            out[0, 1], expected_dir, rtol=1e-4, atol=1e-4
        )

    def test_operates_on_last_axis_of_4d(self):
        x = paddle.randn([2, 3, 4, 5], dtype="float32")
        out = _l2norm(x)
        self.assertEqual(list(out.shape), [2, 3, 4, 5])
        xn = x.numpy().astype(np.float64)
        ref = xn / np.sqrt((xn**2).sum(axis=-1, keepdims=True) + 1e-6)
        np.testing.assert_allclose(out.numpy(), ref, rtol=1e-4, atol=1e-5)

    def test_zero_input_is_finite_via_eps(self):
        """The 1e-6 epsilon keeps an all-zero vector from producing NaN/Inf."""
        x = paddle.zeros([1, 1, 4], dtype="float32")
        out = _l2norm(x)
        self.assertTrue(paddle.isfinite(out).all().item())
        np.testing.assert_allclose(out.numpy(), np.zeros([1, 1, 4]), atol=0.0)


def _build_gdn(
    perform_initialization=False, conv_init=None, A_init_range=(1, 16)
):
    """Construct a single-GPU (TP=1) GatedDeltaNet with the local stand-ins."""
    config = TransformerConfig(
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_KEY_HEADS,
        num_hidden_layers=2,
        hidden_act=F.silu,
        rms_norm_eps=1e-5,
        normalization="RMSNorm",
        sequence_parallel=False,
        deterministic_mode=True,
        perform_initialization=perform_initialization,
    )
    sublayers_spec = GatedDeltaNetSublayersSpec(
        in_proj=NoBiasLinear,
        out_norm=SimpleRMSNorm,
        out_proj=NoBiasLinear,
    )
    return GatedDeltaNet(
        config=config,
        sublayers_spec=sublayers_spec,
        layer_number=1,
        bias=False,
        conv_bias=False,
        conv_init=conv_init,
        use_qk_l2norm=True,
        A_init_range=A_init_range,
        pg_collection=_FakePGCollection(),
        conv_kernel_dim=CONV_KERNEL_DIM,
        key_head_dim=KEY_HEAD_DIM,
        value_head_dim=VALUE_HEAD_DIM,
        num_key_heads=NUM_KEY_HEADS,
        num_value_heads=NUM_VALUE_HEADS,
    )


class TestGatedDeltaNetPaddingMask(unittest.TestCase):
    """_build_padding_mask branch behavior and its forward-path effect
    (not covered by existing tests)."""

    def setUp(self):
        self.gdn = _build_gdn()

    def test_2d_all_valid_returns_none(self):
        mask = paddle.ones([2, SEQ_LENGTH], dtype="float32")
        self.assertIsNone(
            self.gdn._build_padding_mask(mask, None, 2, SEQ_LENGTH)
        )

    def test_2d_mask_with_padding_content(self):
        mask = paddle.to_tensor([[1.0, 1.0, 0.0, 0.0]], dtype="float32")
        out = self.gdn._build_padding_mask(mask, None, 1, 4)
        self.assertIsNotNone(out)
        self.assertEqual(list(out.shape), [1, 4, 1])
        np.testing.assert_array_equal(
            out.numpy(),
            np.array([[[1.0], [1.0], [0.0], [0.0]]], dtype=np.float32),
        )

    def test_4d_block_causal_reduced_via_diagonal(self):
        # A [b, 1, s, s] mask; only the causal diagonal encodes per-token
        # validity. Padded token 3 has a zero diagonal entry.
        m = paddle.zeros([1, 1, 4, 4], dtype="float32")
        eye = paddle.to_tensor([1.0, 1.0, 1.0, 0.0], dtype="float32")
        m = m + paddle.diag(eye).reshape([1, 1, 4, 4])
        out = self.gdn._build_padding_mask(m, None, 1, 4)
        np.testing.assert_array_equal(
            out.numpy(),
            np.array([[[1.0], [1.0], [1.0], [0.0]]], dtype=np.float32),
        )

    def test_startend_row_indices_derives_validity(self):
        # indices come from [:, 0, :, 0]; valid iff index > position. A constant
        # boundary of 2 makes positions 0,1 valid (2>0, 2>1) and positions 2,3
        # padding (2>2, 2>3 both false) -> [1,1,0,0]. A partially-invalid mask
        # is required so the all-valid ``None`` fast path is not taken.
        idx = paddle.to_tensor([2, 2, 2, 2], dtype="int32").reshape(
            [1, 1, 4, 1]
        )
        out = self.gdn._build_padding_mask(None, idx, 1, 4)
        np.testing.assert_array_equal(
            out.numpy(),
            np.array([[[1.0], [1.0], [0.0], [0.0]]], dtype=np.float32),
        )

    def test_unresolvable_mask_raises_value_error(self):
        # 2D mask whose length disagrees with seq_len and no startend indices:
        # nothing can be derived, so the entry point must raise, not guess.
        mask = paddle.ones([1, 5], dtype="float32")
        with self.assertRaises(ValueError):
            self.gdn._build_padding_mask(mask, None, 1, 4)

    def test_all_valid_mask_matches_no_mask_forward(self):
        """An all-ones mask hits the None fast path, so forward output is
        identical to passing attention_mask=None."""
        hidden = paddle.randn([MICRO_BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE])
        mask = paddle.ones([MICRO_BATCH_SIZE, SEQ_LENGTH], dtype="float32")
        out_none, _ = self.gdn(hidden, attention_mask=None)
        out_mask, _ = self.gdn(hidden, attention_mask=mask)
        np.testing.assert_allclose(
            out_none.numpy(), out_mask.numpy(), rtol=1e-6, atol=1e-6
        )

    def test_padding_mask_changes_forward_output(self):
        """A mask with real padding must alter the output (mask is consumed)."""
        hidden = paddle.randn([MICRO_BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE])
        mask = paddle.ones([MICRO_BATCH_SIZE, SEQ_LENGTH], dtype="float32")
        mask[:, SEQ_LENGTH // 2 :] = 0.0
        out_none, _ = self.gdn(hidden, attention_mask=None)
        out_mask, _ = self.gdn(hidden, attention_mask=mask)
        self.assertFalse(
            np.allclose(
                out_none.numpy(), out_mask.numpy(), rtol=1e-4, atol=1e-4
            )
        )
        self.assertTrue(paddle.isfinite(out_mask).all().item())


class TestGatedDeltaNetParameterInit(unittest.TestCase):
    """reset_parameters initialization ranges and A_init_range validation."""

    def test_dt_bias_initialized_to_one(self):
        gdn = _build_gdn(perform_initialization=True, conv_init=0.02)
        np.testing.assert_allclose(
            gdn.dt_bias.numpy(),
            np.ones([NUM_VALUE_HEADS], dtype=np.float32),
            atol=0.0,
        )

    def test_A_log_within_log_of_init_range(self):
        gdn = _build_gdn(
            perform_initialization=True, conv_init=0.02, A_init_range=(1, 16)
        )
        a_log = gdn.A_log.numpy()
        # A ~ Uniform(1, 16) => A_log = log(A) in [log 1, log 16] = [0, ~2.7726].
        self.assertGreaterEqual(float(a_log.min()), -1e-5)
        self.assertLessEqual(float(a_log.max()), np.log(16.0) + 1e-5)

    def test_conv_weight_within_conv_init_bounds(self):
        conv_init = 0.05
        gdn = _build_gdn(perform_initialization=True, conv_init=conv_init)
        w = gdn.conv1d.weight.numpy()
        self.assertLessEqual(float(np.abs(w).max()), conv_init + 1e-6)
        # Uniform init should not collapse to a single constant.
        self.assertGreater(float(w.std()), 0.0)

    def test_A_init_range_start_greater_than_end_raises(self):
        with self.assertRaises(AssertionError):
            _build_gdn(A_init_range=(5, 3))

    def test_A_init_range_negative_start_raises(self):
        with self.assertRaises(AssertionError):
            _build_gdn(A_init_range=(-1, 2))


class TestGatedDeltaNetShardedStateDictImportFallback(unittest.TestCase):
    """sharded_state_dict returns {} when the flex_checkpoint import fails."""

    def test_returns_empty_dict_when_import_unavailable(self):
        gdn = _build_gdn()
        with mock.patch.dict(
            "sys.modules",
            {"paddle.distributed.flex_checkpoint.dcp.sharded_weight": None},
        ):
            result = gdn.sharded_state_dict()
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
