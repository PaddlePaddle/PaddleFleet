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

"""Unit tests for the KimiDeltaAttention module."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import paddle
import paddle.nn.functional as F
from paddle import nn

from paddlefleet.transformer import kimi_delta_attention as kda_mod
from paddlefleet.transformer.kimi_delta_attention import (
    KimiDeltaAttention,
    KimiDeltaAttentionSublayersSpec,
    kda_gate,
    paddle_chunk_kda,
)
from paddlefleet.transformer.transformer_config import TransformerConfig

# ---- Local stand-in layers (no fleet / TP required) ----


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

    def __init__(self, normalized_shape, eps=1e-5, norm_eps=None, **kwargs):
        super().__init__()
        self.weight = self.create_parameter(
            shape=[normalized_shape],
            default_initializer=nn.initializer.Constant(1.0),
        )
        # get_norm_extra_args passes norm_eps; out_norm inputs are small so the
        # eps actually matters for matching the reference implementation.
        self.eps = norm_eps if norm_eps is not None else eps

    def forward(self, x):
        x_float = x.astype(paddle.float32)
        rms = paddle.rsqrt(
            x_float.pow(2).mean(axis=-1, keepdim=True) + self.eps
        )
        return (x_float * rms * self.weight.astype(paddle.float32)).astype(
            x.dtype
        )


class _FakeGroup:
    ranks = [0]
    nranks = 1


class _FakePGCollection:
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
GATE_LOWER_BOUND = -5.0


def _naive_recurrent_kda(query, key, value, g, beta, initial_state=None):
    """Token-by-token fp32 reference for the KDA recurrence.

    S_t = S_{t-1} * exp(g_t) + beta_t * k_t (v_t - k_t^T S_t)^T,  o_t = q_t S_t
    """
    batch, seq_len, num_k_heads, k_dim = query.shape
    num_v_heads, v_dim = value.shape[2], value.shape[3]
    group = num_v_heads // num_k_heads
    scale = k_dim**-0.5
    query = paddle.repeat_interleave(query, group, axis=2) * scale
    key = paddle.repeat_interleave(key, group, axis=2)
    state = paddle.zeros([batch, num_v_heads, k_dim, v_dim], dtype="float32")
    if initial_state is not None:
        state = state + initial_state
    outs = []
    for t in range(seq_len):
        q_t, k_t, v_t = query[:, t], key[:, t], value[:, t]
        state = state * g[:, t].unsqueeze(-1).exp()
        delta = v_t - (k_t.unsqueeze(-1) * state).sum(-2)
        state = state + (beta[:, t].unsqueeze(-1) * k_t).unsqueeze(
            -1
        ) * delta.unsqueeze(-2)
        outs.append((q_t.unsqueeze(-1) * state).sum(-2))
    return paddle.stack(outs, axis=1), state


class TestPaddleChunkKda(unittest.TestCase):
    """Test the paddle-native chunked KDA against the recurrent reference."""

    def _inputs(
        self,
        batch=2,
        seq_len=SEQ_LENGTH,
        num_k_heads=NUM_KEY_HEADS,
        num_v_heads=NUM_VALUE_HEADS,
        k_dim=KEY_HEAD_DIM,
        v_dim=VALUE_HEAD_DIM,
    ):
        paddle.seed(0)
        return {
            "query": paddle.randn([batch, seq_len, num_k_heads, k_dim]),
            "key": paddle.randn([batch, seq_len, num_k_heads, k_dim]),
            "value": paddle.randn([batch, seq_len, num_v_heads, v_dim]),
            "g": paddle.randn([batch, seq_len, num_v_heads, k_dim]),
            "beta": paddle.randn([batch, seq_len, num_v_heads]),
        }

    def test_matches_recurrent_reference(self):
        """Chunked and recurrent forms must agree in fp32."""
        for chunk_size in (32, 64):
            x = self._inputs()
            g_decay = kda_gate(
                x["g"],
                paddle.zeros([NUM_VALUE_HEADS]),
                None,
                safe_gate=True,
                lower_bound=GATE_LOWER_BOUND,
            )
            out, state = paddle_chunk_kda(
                **x,
                A_log=paddle.zeros([NUM_VALUE_HEADS]),
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                safe_gate=True,
                lower_bound=GATE_LOWER_BOUND,
                output_final_state=True,
                chunk_size=chunk_size,
            )
            ref, ref_state = _naive_recurrent_kda(
                x["query"],
                x["key"],
                x["value"],
                g_decay,
                F.sigmoid(x["beta"]),
            )
            self.assertEqual(list(out.shape), list(ref.shape))
            assert paddle.allclose(out, ref, atol=1e-4, rtol=1e-4).item()
            assert paddle.allclose(
                state, ref_state, atol=1e-4, rtol=1e-4
            ).item()

    def test_gva_and_padding(self):
        """hv > h (GVA) and a sequence length that is not a chunk multiple."""
        x = self._inputs(seq_len=50, num_k_heads=2, num_v_heads=4)
        ref_g = kda_gate(x["g"], paddle.zeros([4]), None)
        out, _ = paddle_chunk_kda(
            **x,
            A_log=paddle.zeros([4]),
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
        )
        ref, _ = _naive_recurrent_kda(
            x["query"], x["key"], x["value"], ref_g, F.sigmoid(x["beta"])
        )
        self.assertEqual(list(out.shape), [2, 50, 4, VALUE_HEAD_DIM])
        assert paddle.allclose(out, ref, atol=1e-4, rtol=1e-4).item()

    def test_initial_state_and_layout(self):
        """initial_state is honoured and state_v_first transposes the output."""
        x = self._inputs()
        h0 = paddle.randn([2, NUM_VALUE_HEADS, KEY_HEAD_DIM, VALUE_HEAD_DIM])
        common = {
            "A_log": paddle.zeros([NUM_VALUE_HEADS]),
            "use_gate_in_kernel": True,
            "use_beta_sigmoid_in_kernel": True,
            "output_final_state": True,
        }
        out, state = paddle_chunk_kda(**x, initial_state=h0, **common)
        ref, ref_state = _naive_recurrent_kda(
            x["query"],
            x["key"],
            x["value"],
            kda_gate(x["g"], paddle.zeros([NUM_VALUE_HEADS]), None),
            F.sigmoid(x["beta"]),
            initial_state=h0,
        )
        assert paddle.allclose(out, ref, atol=1e-4, rtol=1e-4).item()
        assert paddle.allclose(state, ref_state, atol=1e-4, rtol=1e-4).item()

        _, state_vf = paddle_chunk_kda(
            **x, initial_state=h0, state_v_first=True, **common
        )
        assert paddle.equal_all(state_vf, state.transpose([0, 1, 3, 2])).item()

    def test_backward(self):
        """Gradients must reach every input and stay finite."""
        x = self._inputs()
        A_log = paddle.zeros([NUM_VALUE_HEADS])
        dt_bias = paddle.zeros([NUM_VALUE_HEADS * KEY_HEAD_DIM])
        for t in (*x.values(), A_log, dt_bias):
            t.stop_gradient = False
        out, _ = paddle_chunk_kda(
            **x,
            A_log=A_log,
            dt_bias=dt_bias,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=True,
            lower_bound=GATE_LOWER_BOUND,
        )
        out.pow(2).mean().backward()
        for name, t in [*x.items(), ("A_log", A_log), ("dt_bias", dt_bias)]:
            self.assertIsNotNone(t.grad, f"{name} has no gradient")
            assert paddle.isfinite(t.grad).all().item(), (
                f"{name} grad not finite"
            )

    def test_causality(self):
        """Perturbing position t must not change outputs before t."""
        x = self._inputs()
        common = {
            "A_log": paddle.zeros([NUM_VALUE_HEADS]),
            "use_gate_in_kernel": True,
            "use_beta_sigmoid_in_kernel": True,
        }
        t = 20
        out_a, _ = paddle_chunk_kda(**x, **common)
        x2 = dict(x)
        x2["value"] = x["value"] + F.one_hot(
            paddle.to_tensor([t]), num_classes=SEQ_LENGTH
        ).reshape([1, SEQ_LENGTH, 1, 1])
        out_b, _ = paddle_chunk_kda(**x2, **common)
        diff = (out_a - out_b).abs().max(axis=[0, 2, 3])
        self.assertLess(float(diff[:t].max()), 1e-6)
        self.assertGreater(float(diff[t]), 1e-4)


def _build_kda(use_full_rank_gate=True, sequence_parallel=False, **overrides):
    config = TransformerConfig(
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_KEY_HEADS,
        num_hidden_layers=2,
        hidden_act=F.silu,
        rms_norm_eps=1e-5,
        normalization="RMSNorm",
        sequence_parallel=sequence_parallel,
        deterministic_mode=True,
    )
    spec = KimiDeltaAttentionSublayersSpec(
        in_proj=NoBiasLinear,
        f_a_proj=NoBiasLinear,
        f_b_proj=NoBiasLinear,
        g_a_proj=NoBiasLinear,
        g_b_proj=NoBiasLinear,
        out_norm=SimpleRMSNorm,
        out_proj=NoBiasLinear,
    )
    kwargs = {
        "config": config,
        "sublayers_spec": spec,
        "layer_number": 1,
        "bias": False,
        "conv_bias": False,
        "conv_init": 1.0,
        "use_qk_l2norm": True,
        "A_init_range": (1, 16),
        "pg_collection": _FakePGCollection(),
        "conv_kernel_dim": CONV_KERNEL_DIM,
        "key_head_dim": KEY_HEAD_DIM,
        "value_head_dim": VALUE_HEAD_DIM,
        "num_key_heads": NUM_KEY_HEADS,
        "num_value_heads": NUM_VALUE_HEADS,
        "gate_lora_rank": VALUE_HEAD_DIM,
        "use_full_rank_gate": use_full_rank_gate,
        "gate_lower_bound": GATE_LOWER_BOUND,
    }
    kwargs.update(overrides)
    return KimiDeltaAttention(**kwargs)


class TestKimiDeltaAttention(unittest.TestCase):
    """Test the full KimiDeltaAttention module (single-GPU, no TP)."""

    def setUp(self):
        self.kda = _build_kda()

    def test_constructor(self):
        for name in [
            "in_proj",
            "f_a_proj",
            "f_b_proj",
            "conv1d",
            "dt_bias",
            "A_log",
            "out_norm",
            "out_proj",
        ]:
            self.assertTrue(hasattr(self.kda, name), name)
        # full-rank output gate is folded into in_proj
        self.assertFalse(hasattr(self.kda, "g_a_proj"))
        self.assertTrue(
            hasattr(_build_kda(use_full_rank_gate=False), "g_a_proj")
        )

    def test_parameter_shapes(self):
        qk_dim = KEY_HEAD_DIM * NUM_KEY_HEADS
        v_dim = VALUE_HEAD_DIM * NUM_VALUE_HEADS
        conv_dim = qk_dim * 2 + v_dim
        self.assertEqual(
            list(self.kda.conv1d.weight.shape), [conv_dim, 1, CONV_KERNEL_DIM]
        )
        # dt_bias is per-channel for KDA, A_log per-head
        self.assertEqual(list(self.kda.dt_bias.shape), [v_dim])
        self.assertEqual(list(self.kda.A_log.shape), [NUM_VALUE_HEADS])
        self.assertEqual(
            self.kda.in_proj_dim, qk_dim * 2 + v_dim * 2 + NUM_VALUE_HEADS
        )
        self.assertEqual(
            _build_kda(use_full_rank_gate=False).in_proj_dim,
            qk_dim * 2 + v_dim + NUM_VALUE_HEADS,
        )

    def test_conv_weight_stays_fp32_in_o2(self):
        original_dtype = paddle.get_default_dtype()
        try:
            paddle.set_default_dtype("bfloat16")
            kda = _build_kda()
        finally:
            paddle.set_default_dtype(original_dtype)

        self.assertEqual(kda.conv1d.weight.dtype, paddle.float32)
        self.assertEqual(kda.in_proj.linear.weight.dtype, paddle.bfloat16)
        paddle.amp.decorate(models=kda, level="O2", dtype="bfloat16")
        self.assertEqual(kda.conv1d.weight.dtype, paddle.float32)

        x = paddle.randn(
            [MICRO_BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE], dtype="bfloat16"
        )
        x.stop_gradient = False
        with patch.object(
            kda_mod.fla.ops.kda,
            "chunk_kda",
            wraps=paddle_chunk_kda,
        ):
            out, _ = kda(hidden_states=x, attention_mask=None)
            self.assertEqual(out.dtype, paddle.bfloat16)
            out.astype(paddle.float32).square().mean().backward()
        self.assertEqual(kda.conv1d.weight.grad.dtype, paddle.float32)
        self.assertEqual(x.grad.dtype, paddle.bfloat16)

    def test_forward_shape(self):
        x = paddle.randn([MICRO_BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE])
        out, out_bias = self.kda(hidden_states=x, attention_mask=None)
        self.assertEqual(
            list(out.shape), [MICRO_BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE]
        )
        self.assertIsNone(out_bias)
        assert paddle.isfinite(out).all().item()

    def test_forward_always_calls_fla_chunk_kda(self):
        """Deterministic mode must not select the removed eager fallback."""
        self.assertTrue(self.kda.config.deterministic_mode)
        x = paddle.randn([MICRO_BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE])

        with patch.object(
            kda_mod.fla.ops.kda,
            "chunk_kda",
            wraps=paddle_chunk_kda,
        ) as mock_chunk_kda:
            out, _ = self.kda(hidden_states=x, attention_mask=None)

        mock_chunk_kda.assert_called_once()
        call_kwargs = mock_chunk_kda.call_args.kwargs
        self.assertTrue(call_kwargs["use_gate_in_kernel"])
        self.assertTrue(call_kwargs["use_beta_sigmoid_in_kernel"])
        self.assertTrue(call_kwargs["safe_gate"])
        self.assertEqual(call_kwargs["lower_bound"], GATE_LOWER_BOUND)
        self.assertEqual(
            list(out.shape), [MICRO_BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE]
        )

    def test_forward_low_rank_gate(self):
        kda = _build_kda(use_full_rank_gate=False)
        x = paddle.randn([MICRO_BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE])
        out, _ = kda(hidden_states=x, attention_mask=None)
        self.assertEqual(
            list(out.shape), [MICRO_BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE]
        )
        assert paddle.isfinite(out).all().item()

    def test_backward(self):
        x = paddle.randn([MICRO_BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE])
        x.stop_gradient = False
        out, _ = self.kda(hidden_states=x, attention_mask=None)
        out.pow(2).mean().backward()
        self.assertIsNotNone(x.grad)
        assert paddle.isfinite(x.grad).all().item()
        for name, param in self.kda.named_parameters():
            self.assertIsNotNone(param.grad, f"{name} has no gradient")
            assert paddle.isfinite(param.grad).all().item(), name

    def test_unsupported_inputs(self):
        x = paddle.randn([1, SEQ_LENGTH, HIDDEN_SIZE])
        with self.assertRaises(NotImplementedError):
            self.kda(hidden_states=x, attention_mask=None, packed_seq_params={})

    def test_rejects_invalid_head_config(self):
        """Head counts / dims that would silently break the TP splits."""
        for overrides in (
            {"num_key_heads": 3},  # not a divisor of num_value_heads
            {"value_head_dim": KEY_HEAD_DIM * 2},  # gate dim mismatch
        ):
            with self.assertRaises(ValueError):
                _build_kda(**overrides)

    def test_rejects_sharded_gate_a_proj(self):
        """f_a_proj must be replicated: f_b_proj consumes the full rank."""

        class ShardedLinear(NoBiasLinear):
            output_size_per_partition = 1

        with self.assertRaises(ValueError):
            _build_kda(
                sublayers_spec=KimiDeltaAttentionSublayersSpec(
                    in_proj=NoBiasLinear,
                    f_a_proj=ShardedLinear,
                    f_b_proj=NoBiasLinear,
                    out_norm=SimpleRMSNorm,
                    out_proj=NoBiasLinear,
                )
            )


if __name__ == "__main__":
    unittest.main()
