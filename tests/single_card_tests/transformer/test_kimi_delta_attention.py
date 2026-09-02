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

import inspect
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import nn

from paddlefleet.models.gpt import GPTConfig
from paddlefleet.transformer import kimi_delta_attention as kda_mod
from paddlefleet.transformer.kimi_delta_attention import (
    HAVE_FLA,
    KimiDeltaAttention,
    KimiDeltaAttentionSublayersSpec,
    build_cu_seqlens,
    kda_gate,
    paddle_chunk_kda,
)
from paddlefleet.transformer.paddle_norm import RMSNorm
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
        self.cp = _FakeGroup()


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


def _build_kda(
    use_full_rank_gate=True,
    sequence_parallel=False,
    hidden_act=F.silu,
    deterministic_mode=True,
    out_norm=SimpleRMSNorm,
    config_overrides=None,
    **overrides,
):
    config = TransformerConfig(
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_KEY_HEADS,
        num_hidden_layers=2,
        hidden_act=hidden_act,
        rms_norm_eps=1e-5,
        normalization="RMSNorm",
        sequence_parallel=sequence_parallel,
        deterministic_mode=deterministic_mode,
        **(config_overrides or {}),
    )
    spec = KimiDeltaAttentionSublayersSpec(
        in_proj=NoBiasLinear,
        f_a_proj=NoBiasLinear,
        f_b_proj=NoBiasLinear,
        g_a_proj=NoBiasLinear,
        g_b_proj=NoBiasLinear,
        out_norm=out_norm,
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

    def test_fp32_params_stay_fp32_in_o2(self):
        """conv1d / dt_bias / A_log / out_norm are pinned to fp32 under amp O2."""
        original_dtype = paddle.get_default_dtype()
        try:
            paddle.set_default_dtype("bfloat16")
            # The real RMSNorm sizes its weight from config.params_dtype, which
            # KDA pins to fp32; SimpleRMSNorm ignores it and would be born
            # bfloat16, making the out_norm assertion a false pass.
            kda = _build_kda(out_norm=RMSNorm)
        finally:
            paddle.set_default_dtype(original_dtype)

        fp32_names = ["conv1d.weight", "dt_bias", "A_log", "out_norm.weight"]

        def assert_pinned():
            params = dict(kda.named_parameters())
            for name in fp32_names:
                self.assertEqual(
                    params[name].dtype, paddle.float32, f"{name} left fp32"
                )

        assert_pinned()
        self.assertEqual(kda.in_proj.linear.weight.dtype, paddle.bfloat16)
        paddle.amp.decorate(models=kda, level="O2", dtype="bfloat16")
        assert_pinned()

        x = paddle.randn(
            [MICRO_BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE], dtype="bfloat16"
        )
        x.stop_gradient = False
        # Deterministic mode, so this runs the paddle-native conv + recurrence
        # and needs no fla kernels.
        out, _ = kda(hidden_states=x, attention_mask=None)
        self.assertEqual(out.dtype, paddle.bfloat16)
        out.astype(paddle.float32).square().mean().backward()
        grads = dict(kda.named_parameters())
        for name in fp32_names:
            self.assertEqual(
                grads[name].grad.dtype, paddle.float32, f"grad[{name}]"
            )
        self.assertEqual(x.grad.dtype, paddle.bfloat16)

    @unittest.skipUnless(
        HAVE_FLA, "paddlefleet_ops fla kernels are not available"
    )
    def test_fused_backend_logged_once_per_process(self):
        """Every layer picks the same backend, so don't log it per layer."""
        with (
            patch.object(kda_mod, "_FUSED_KERNEL_LOGGED", False),
            self.assertLogs(kda_mod.logger, level="INFO") as logs,
        ):
            _build_kda(deterministic_mode=False)
            _build_kda(deterministic_mode=False)
        self.assertEqual(len(logs.output), 1, logs.output)

    def test_forward_shape(self):
        x = paddle.randn([MICRO_BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE])
        out, out_bias = self.kda(hidden_states=x, attention_mask=None)
        self.assertEqual(
            list(out.shape), [MICRO_BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE]
        )
        self.assertIsNone(out_bias)
        assert paddle.isfinite(out).all().item()

    def test_deterministic_mode_uses_paddle_fallback(self):
        """Deterministic mode routes to the paddle-native chunked recurrence."""
        self.assertTrue(self.kda.config.deterministic_mode)
        self.assertFalse(self.kda.use_fused_kernels)
        x = paddle.randn([MICRO_BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE])

        with patch.object(
            kda_mod, "chunk_kda", wraps=kda_mod.chunk_kda
        ) as mock_chunk_kda:
            out, _ = self.kda(hidden_states=x, attention_mask=None)

        mock_chunk_kda.assert_not_called()
        self.assertEqual(
            list(out.shape), [MICRO_BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE]
        )

    @unittest.skipUnless(
        HAVE_FLA, "paddlefleet_ops fla kernels are not available"
    )
    def test_fused_path_calls_fla_chunk_kda(self):
        """Non-deterministic mode folds the gate/beta/l2norm into the kernel."""
        kda = _build_kda(deterministic_mode=False)
        self.assertTrue(kda.use_fused_kernels)
        x = paddle.randn([MICRO_BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE])

        with patch.object(
            kda_mod, "chunk_kda", wraps=kda_mod.chunk_kda
        ) as mock_chunk_kda:
            out, _ = kda(hidden_states=x, attention_mask=None)

        mock_chunk_kda.assert_called_once()
        call_kwargs = mock_chunk_kda.call_args.kwargs
        self.assertEqual(call_kwargs["beta"].dtype, paddle.float32)
        self.assertTrue(call_kwargs["use_qk_l2norm_in_kernel"])
        self.assertTrue(call_kwargs["use_gate_in_kernel"])
        self.assertTrue(call_kwargs["use_beta_sigmoid_in_kernel"])
        self.assertTrue(call_kwargs["safe_gate"])
        self.assertEqual(call_kwargs["lower_bound"], GATE_LOWER_BOUND)
        self.assertEqual(
            list(out.shape), [MICRO_BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE]
        )

    def test_short_conv_always_uses_silu(self):
        """KDA short convolution uses the official SiLU, not hidden_act."""
        kda = _build_kda(hidden_act=F.relu)
        self.assertFalse(hasattr(kda, "act_fn"))
        self.assertFalse(hasattr(kda, "activation"))
        x = paddle.randn([MICRO_BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE])

        with patch.object(kda_mod.F, "silu", wraps=kda_mod.F.silu) as mock_silu:
            kda(hidden_states=x, attention_mask=None)

        mock_silu.assert_called_once()

    @unittest.skipUnless(
        HAVE_FLA, "paddlefleet_ops fla kernels are not available"
    )
    def test_fused_sigmoid_gated_norm_contract(self):
        """Output gating delegates to FLA RMSNorm with sigmoid and Fleet eps."""
        kda = _build_kda(deterministic_mode=False)
        hidden_states = paddle.randn(
            [MICRO_BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE]
        )
        core_attn_out = paddle.randn(
            [MICRO_BATCH_SIZE, SEQ_LENGTH, NUM_VALUE_HEADS, VALUE_HEAD_DIM]
        )
        fused_shape = [
            MICRO_BATCH_SIZE * SEQ_LENGTH * NUM_VALUE_HEADS,
            VALUE_HEAD_DIM,
        ]
        expected = paddle.randn(fused_shape)

        with (
            patch.object(
                kda_mod,
                "causal_conv1d",
                side_effect=lambda x, **kwargs: (x, None),
            ),
            patch.object(
                kda_mod,
                "chunk_kda",
                return_value=(core_attn_out, None),
            ),
            patch.object(
                kda_mod,
                "rms_norm_gated",
                return_value=expected,
            ) as mock_rms_norm_gated,
        ):
            actual, _ = kda(hidden_states=hidden_states, attention_mask=None)

        self.assertEqual(
            list(actual.shape), [MICRO_BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE]
        )
        args = mock_rms_norm_gated.call_args.args
        kwargs = mock_rms_norm_gated.call_args.kwargs
        self.assertEqual(list(args[0].shape), fused_shape)
        self.assertEqual(list(args[1].shape), fused_shape)
        self.assertIs(args[2], kda.out_norm.weight)
        self.assertIsNone(args[3])
        self.assertEqual(kwargs["activation"], "sigmoid")
        self.assertEqual(kwargs["eps"], kda.config.rms_norm_eps)

    @unittest.skipUnless(
        HAVE_FLA, "paddlefleet_ops fla kernels are not available"
    )
    def test_fused_sigmoid_gated_norm_forward_backward(self):
        """Fused RMSNorm+sigmoid matches an independent FP32 oracle."""
        shape = [2, 7, NUM_VALUE_HEADS, VALUE_HEAD_DIM]
        for dtype, rtol, atol in (
            ("float32", 1e-5, 1e-5),
            ("bfloat16", 1e-2, 1e-2),
        ):
            with self.subTest(dtype=dtype):
                paddle.seed(2026)
                kda = _build_kda(deterministic_mode=False)
                kda.out_norm.weight.set_value(
                    paddle.linspace(0.5, 1.5, VALUE_HEAD_DIM, dtype="float32")
                )

                x = paddle.randn(shape, dtype=dtype)
                gate = paddle.randn(shape, dtype=dtype)
                x.stop_gradient = False
                gate.stop_gradient = False
                x_ref = x.detach().clone()
                gate_ref = gate.detach().clone()
                weight_ref = kda.out_norm.weight.detach().clone()
                x_ref.stop_gradient = False
                gate_ref.stop_gradient = False
                weight_ref.stop_gradient = False

                actual = kda_mod.rms_norm_gated(
                    x.reshape([-1, VALUE_HEAD_DIM]),
                    gate.reshape([-1, VALUE_HEAD_DIM]),
                    kda.out_norm.weight,
                    None,
                    activation="sigmoid",
                    eps=kda.config.rms_norm_eps,
                ).reshape(shape)
                x_ref_fp32 = x_ref.astype("float32")
                expected = x_ref_fp32 * paddle.rsqrt(
                    x_ref_fp32.square().mean(axis=-1, keepdim=True)
                    + kda.config.rms_norm_eps
                )
                expected = (
                    expected
                    * weight_ref.astype("float32")
                    * F.sigmoid(gate_ref.astype("float32"))
                ).astype(dtype)

                output_gradient = paddle.randn(shape, dtype=dtype)
                (
                    actual.astype("float32") * output_gradient.astype("float32")
                ).sum().backward()
                (
                    expected.astype("float32")
                    * output_gradient.astype("float32")
                ).sum().backward()

                for actual_value, expected_value in (
                    (actual, expected),
                    (x.grad, x_ref.grad),
                    (gate.grad, gate_ref.grad),
                    (kda.out_norm.weight.grad, weight_ref.grad),
                ):
                    paddle.testing.assert_close(
                        actual_value.astype("float32"),
                        expected_value.astype("float32"),
                        rtol=rtol,
                        atol=atol,
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


class TestKdaParameterInitialization(unittest.TestCase):
    """reset_parameters() must reproduce fla's KDA init, not the neutral zeros.

    Locks the pieces the init rewrite introduced: the log-uniform dt draw behind
    dt_bias, its inverse-softplus mapping and floor, the two A_log branches, the
    perform_initialization=False escape hatch, and the new argument validation.
    """

    def _softplus_dt(self, kda):
        """Recover the drawn dt from dt_bias (softplus is the exact inverse)."""
        return F.softplus(kda.dt_bias.astype("float32")).numpy()

    def test_dt_bias_is_inverse_softplus_of_log_uniform_dt(self):
        dt_min, dt_max = 0.001, 0.1
        kda = _build_kda(dt_init_range=(dt_min, dt_max), dt_init_floor=1e-4)
        dt = self._softplus_dt(kda)

        # dt_init_floor is below dt_min here, so the clamp is a no-op and the
        # draw must land strictly inside the requested range.
        self.assertTrue((dt >= dt_min - 1e-6).all(), dt.min())
        self.assertTrue((dt <= dt_max + 1e-6).all(), dt.max())
        # Not the old constant-0 init: softplus(0) = log(2) is far above dt_max.
        self.assertGreater(float(np.abs(kda.dt_bias.numpy()).min()), 0.0)
        # log-uniform, so the draw spans decades rather than clustering at one end
        self.assertGreater(dt.max() / dt.min(), 2.0)

    def test_dt_init_floor_clamps_the_draw(self):
        """A range fully below the floor collapses dt onto the floor exactly."""
        floor = 0.05
        kda = _build_kda(dt_init_range=(1e-8, 2e-8), dt_init_floor=floor)
        dt = self._softplus_dt(kda)
        np.testing.assert_allclose(dt, np.full_like(dt, floor), rtol=1e-5)

    def test_dt_bias_is_reproducible_under_a_fixed_seed(self):
        paddle.seed(1234)
        first = _build_kda().dt_bias.numpy().copy()
        paddle.seed(1234)
        second = _build_kda().dt_bias.numpy().copy()
        np.testing.assert_allclose(first, second, rtol=0, atol=0)

    def test_bounded_gate_starts_from_zero_a_log(self):
        """exp(A_log) is only the sigmoid slope here, so 1 is the neutral start."""
        kda = _build_kda(gate_lower_bound=GATE_LOWER_BOUND)
        np.testing.assert_allclose(
            kda.A_log.numpy(), np.zeros([NUM_VALUE_HEADS], dtype="float32")
        )

    def test_softplus_gate_draws_a_log_from_a_init_range(self):
        """Without a lower bound exp(A_log) is the decay rate: uniform draw."""
        low, high = 2.0, 16.0
        kda = _build_kda(gate_lower_bound=None, A_init_range=(low, high))
        A = np.exp(kda.A_log.numpy())
        self.assertTrue((A >= low - 1e-4).all(), A.min())
        self.assertTrue((A <= high + 1e-4).all(), A.max())

    def test_initial_bounded_gate_keeps_the_recurrent_state(self):
        """The point of the fix: dt_bias=0 decayed ~0.08/step, fla's init ~0.95."""
        kda = _build_kda(gate_lower_bound=GATE_LOWER_BOUND)
        g = paddle.zeros(
            [1, 1, NUM_VALUE_HEADS, VALUE_HEAD_DIM], dtype="float32"
        )
        decay = kda_gate(
            g,
            kda.A_log,
            kda.dt_bias,
            safe_gate=True,
            lower_bound=GATE_LOWER_BOUND,
        ).exp()
        self.assertGreater(float(decay.min()), 0.5)
        self.assertLess(float(decay.max()), 1.0)

    def test_no_initialization_when_perform_initialization_is_false(self):
        """Checkpoint loading path: the placeholder zeros must survive."""
        kda = _build_kda(
            config_overrides={"perform_initialization": False},
            gate_lower_bound=None,
        )
        v_dim = VALUE_HEAD_DIM * NUM_VALUE_HEADS
        np.testing.assert_allclose(
            kda.dt_bias.numpy(), np.zeros([v_dim], dtype="float32")
        )
        np.testing.assert_allclose(
            kda.A_log.numpy(), np.zeros([NUM_VALUE_HEADS], dtype="float32")
        )

    def test_rejects_invalid_dt_init_arguments(self):
        """Runtime ValueError, not assert: the checks must survive python -O."""
        for overrides in (
            {"dt_init_range": (0.0, 0.1)},  # dt_min must be > 0
            {"dt_init_range": (-0.1, 0.1)},  # dt_min must be > 0
            {"dt_init_range": (0.1, 0.001)},  # dt_min must be <= dt_max
            {"dt_init_floor": 0.0},  # floor must be > 0
            {"dt_init_floor": -1e-4},  # floor must be > 0
        ):
            with self.assertRaises(ValueError, msg=str(overrides)):
                _build_kda(**overrides)

    def test_dt_init_arguments_are_not_positional_before_pg_collection(self):
        """The new kwargs are appended, so old positional calls still bind."""
        params = list(inspect.signature(KimiDeltaAttention.__init__).parameters)
        self.assertLess(
            params.index("pg_collection"), params.index("dt_init_range")
        )
        self.assertLess(
            params.index("gate_lower_bound"), params.index("dt_init_range")
        )
        self.assertLess(
            params.index("dt_init_range"), params.index("dt_init_floor")
        )
        self.assertEqual(params[-1], "dt_init_floor")


@unittest.skipUnless(HAVE_FLA, "paddlefleet_ops fla kernels are not available")
class TestFusedKernels(unittest.TestCase):
    """The fused triton path must agree with the paddle native fallback."""

    def test_fused_matches_native(self):
        native = _build_kda()
        fused = _build_kda(deterministic_mode=False)
        self.assertFalse(native.use_fused_kernels)
        self.assertTrue(fused.use_fused_kernels)

        native_params = dict(native.named_parameters())
        with paddle.no_grad():
            for name, param in fused.named_parameters():
                param.set_value(native_params[name])

        paddle.seed(0)
        x = paddle.randn([2, SEQ_LENGTH, HIDDEN_SIZE])
        outs = {}
        for tag, kda in (("native", native), ("fused", fused)):
            xi = x.clone()
            xi.stop_gradient = False
            out, _ = kda(hidden_states=xi, attention_mask=None)
            out.pow(2).mean().backward()
            outs[tag] = (out, xi.grad, dict(kda.named_parameters()))

        def rel(a, b):
            return float((a - b).norm() / b.norm())

        o_f, gx_f, p_f = outs["fused"]
        o_n, gx_n, p_n = outs["native"]
        # Both run the same math but the triton kernel uses TF32 matmuls, so the
        # floor is ~2e-3 rather than round-off (see check_paddle_kda_module.py).
        self.assertLess(rel(o_f, o_n), 5e-3)
        self.assertLess(rel(gx_f, gx_n), 5e-3)
        for name in p_n:
            err = rel(p_f[name].grad, p_n[name].grad)
            # The gate parameters sit behind softplus/sigmoid, which amplifies it
            tol = 5e-2 if name in ("A_log", "dt_bias") else 2e-2
            self.assertLess(err, tol, f"grad[{name}] rel_err={err:.3e}")


def _startend_row_indices(docs_per_row, seq_len):
    """[b, 1, s, 1] where each entry is the exclusive end of its document."""
    rows = []
    for lens in docs_per_row:
        row, end = [], 0
        for length in lens:
            end += length
            row += [end] * length
        assert len(row) == seq_len, (len(row), seq_len)
        rows.append(row)
    return paddle.to_tensor(rows, dtype="int32").reshape(
        [len(rows), 1, seq_len, 1]
    )


class TestBuildCuSeqlens(unittest.TestCase):
    """build_cu_seqlens flattens a [b, 1, s, 1] mask into packed segment starts."""

    def test_no_mask(self):
        self.assertIsNone(build_cu_seqlens(None, 2, SEQ_LENGTH))
        self.assertEqual(
            build_cu_seqlens(
                None, 2, SEQ_LENGTH, keep_single_segment=True
            ).tolist(),
            [0, 2 * SEQ_LENGTH],
        )

    def test_single_segment(self):
        """A lone document covering everything needs no mask at all."""
        indices = _startend_row_indices([[SEQ_LENGTH]], SEQ_LENGTH)
        self.assertIsNone(build_cu_seqlens(indices, 1, SEQ_LENGTH))
        # context parallel always needs one to slice with
        self.assertEqual(
            build_cu_seqlens(
                indices, 1, SEQ_LENGTH, keep_single_segment=True
            ).tolist(),
            [0, SEQ_LENGTH],
        )

    def test_document_boundaries(self):
        indices = _startend_row_indices([[12, 20], [8, 24]], SEQ_LENGTH)
        self.assertEqual(
            build_cu_seqlens(indices, 2, SEQ_LENGTH).tolist(),
            [0, 12, SEQ_LENGTH, SEQ_LENGTH + 8, 2 * SEQ_LENGTH],
        )

    def test_row_seam_is_a_boundary(self):
        """Rows ending at the same value must not be merged when flattened."""
        indices = _startend_row_indices([[SEQ_LENGTH]] * 2, SEQ_LENGTH)
        self.assertEqual(
            build_cu_seqlens(indices, 2, SEQ_LENGTH).tolist(),
            [0, SEQ_LENGTH, 2 * SEQ_LENGTH],
        )

    def test_rejects_bad_shapes(self):
        indices = _startend_row_indices([[12, 20]], SEQ_LENGTH)
        # [b, 1, s, 2]: a start/end mask cannot be reduced to column 0
        with self.assertRaises(ValueError):
            build_cu_seqlens(
                paddle.concat([indices, indices], axis=-1), 1, SEQ_LENGTH
            )
        # a head-wise mask cannot be honoured by a linear recurrence
        with self.assertRaises(ValueError):
            build_cu_seqlens(
                paddle.concat([indices, indices], axis=1), 1, SEQ_LENGTH
            )
        with self.assertRaises(ValueError):
            build_cu_seqlens(indices, 1, SEQ_LENGTH + 1)


@unittest.skipUnless(HAVE_FLA, "paddlefleet_ops fla kernels are not available")
class TestVarlen(unittest.TestCase):
    """Packed variable-length input must equal running each document alone."""

    DOCS = [[12, 20], [8, 24]]

    def setUp(self):
        self.kda = _build_kda(deterministic_mode=False)
        paddle.seed(0)
        self.x = paddle.randn([len(self.DOCS), SEQ_LENGTH, HIDDEN_SIZE])

    def _forward_backward(self, x, indices=None):
        """Returns (out, grad_x, {param: grad}) for a fresh backward."""
        self.kda.clear_gradients()
        xi = x.clone()
        xi.stop_gradient = False
        out, _ = self.kda(
            hidden_states=xi, attn_mask_startend_row_indices=indices
        )
        # sum (not mean) so the packed loss equals the sum of the per-doc losses
        out.pow(2).sum().backward()
        return (
            out.numpy(),
            xi.grad.numpy(),
            {n: p.grad.numpy() for n, p in self.kda.named_parameters()},
        )

    def _rel(self, a, b):
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))

    def test_matches_per_document(self):
        indices = _startend_row_indices(self.DOCS, SEQ_LENGTH)
        out, grad_x, grads = self._forward_backward(self.x, indices)

        # Reference: every document on its own, gradients summed
        self.kda.clear_gradients()
        ref_out = np.zeros_like(out)
        ref_grad_x = np.zeros_like(grad_x)
        for row, lens in enumerate(self.DOCS):
            offset = 0
            for length in lens:
                seg = self.x[row : row + 1, offset : offset + length]
                si = seg.clone()
                si.stop_gradient = False
                o, _ = self.kda(hidden_states=si)
                o.pow(2).sum().backward()
                ref_out[row, offset : offset + length] = o.numpy()[0]
                ref_grad_x[row, offset : offset + length] = si.grad.numpy()[0]
                offset += length
        ref_grads = {n: p.grad.numpy() for n, p in self.kda.named_parameters()}

        # Not bit-exact: the packed run feeds [b*s, h] to the projections while
        # the reference feeds [L, h], so the GEMMs accumulate in a different
        # order. Measured ~1.4e-5 (out) / 6e-5 (grad_x) / 1.8e-4 (dt_bias).
        self.assertLess(self._rel(out, ref_out), 1e-4, "output mismatch")
        self.assertLess(self._rel(grad_x, ref_grad_x), 5e-4, "grad_x mismatch")
        for name in ref_grads:
            err = self._rel(grads[name], ref_grads[name])
            self.assertLess(err, 1e-3, f"grad[{name}] rel_err={err:.3e}")

    def test_row_seam_is_a_boundary(self):
        """One document per row must reproduce plain batched execution."""
        docs = [[SEQ_LENGTH]] * len(self.DOCS)
        indices = _startend_row_indices(docs, SEQ_LENGTH)
        packed = self._forward_backward(self.x, indices)
        batched = self._forward_backward(self.x)
        self.assertLess(self._rel(packed[0], batched[0]), 5e-4)
        self.assertLess(self._rel(packed[1], batched[1]), 5e-3)

    def test_precomputed_cu_seqlens_matches_mask(self):
        """The embedding hands cu_seqlens down; that must change nothing."""
        indices = _startend_row_indices(self.DOCS, SEQ_LENGTH)
        cu_seqlens = build_cu_seqlens(indices, len(self.DOCS), SEQ_LENGTH)
        self.assertIsNotNone(cu_seqlens)
        from_mask = self._forward_backward(self.x, indices)
        self.kda.clear_gradients()
        xi = self.x.clone()
        xi.stop_gradient = False
        out, _ = self.kda(hidden_states=xi, cu_seqlens=cu_seqlens)
        out.pow(2).sum().backward()
        np.testing.assert_array_equal(out.numpy(), from_mask[0])
        np.testing.assert_array_equal(xi.grad.numpy(), from_mask[1])

    def test_native_path_rejects_varlen(self):
        native = _build_kda()
        self.assertFalse(native.use_fused_kernels)
        indices = _startend_row_indices(self.DOCS, SEQ_LENGTH)
        with self.assertRaises(NotImplementedError):
            native(
                hidden_states=self.x,
                attn_mask_startend_row_indices=indices,
            )


@unittest.skipUnless(HAVE_FLA, "paddlefleet_ops fla kernels are not available")
class TestContextParallelGuards(unittest.TestCase):
    """The CP preconditions must reject unsupported layouts before any collective.

    cp_size is set after construction so a single card can reach the checks.
    """

    def _kda(self, cp_size=4, deterministic_mode=False):
        kda = _build_kda(deterministic_mode=deterministic_mode)
        kda.cp_size = cp_size
        kda.config.context_parallel_size = cp_size
        kda.config.cp_balance_mode = "contiguous_allgather"
        return kda

    def test_dynamic_seq_len_enters_cp_path(self):
        """Without max_sequence_length, CP path passes all guards and reaches build_cp_context."""
        kda = self._kda()
        # Ensure max_sequence_length is not set (dynamic length scenario)
        if hasattr(kda.config, "max_sequence_length"):
            delattr(kda.config, "max_sequence_length")

        # Sentinel to prove execution reached build_cp_context (past all guards)
        class _ReachedBuildCpContext(Exception):
            pass

        with (
            patch.object(
                kda_mod, "build_cp_context", side_effect=_ReachedBuildCpContext
            ),
            self.assertRaises(_ReachedBuildCpContext),
        ):
            kda(hidden_states=paddle.randn([1, SEQ_LENGTH, HIDDEN_SIZE]))

    def test_requires_batch_one(self):
        kda = self._kda()
        with self.assertRaises(NotImplementedError):
            kda(hidden_states=paddle.randn([2, SEQ_LENGTH, HIDDEN_SIZE]))

    def test_requires_contiguous_balance_mode(self):
        kda = self._kda()
        kda.config.cp_balance_mode = "dualchunk_allgather"
        with self.assertRaises(NotImplementedError):
            kda(hidden_states=paddle.randn([1, SEQ_LENGTH, HIDDEN_SIZE]))

    def test_requires_fused_kernels(self):
        kda = self._kda(deterministic_mode=True)
        self.assertFalse(kda.use_fused_kernels)
        with self.assertRaises(NotImplementedError):
            kda(hidden_states=paddle.randn([1, SEQ_LENGTH, HIDDEN_SIZE]))


def _build_gpt_embedding(config):
    """GPTEmbedding with a plain nn.Embedding and no rope (no fleet init needed)."""
    from paddlefleet.models.gpt.gpt_embedding import GPTEmbedding

    mock_spec = MagicMock(rope_embedding=None)

    class _Emb(nn.Layer):
        def __init__(self, v, h):
            super().__init__()
            self.embed_tokens = nn.Embedding(v, h)
            self.reduce_scatter_embeddings = (
                self.scatter_to_sequence_parallel
            ) = self.sequence_parallel = False

        @property
        def embedding_weight(self):
            return self.embed_tokens.weight

        def forward(self, input_ids, position_ids=None):
            out = self.embed_tokens(input_ids)
            if config.sequence_parallel:
                # real embeddings hand back [s/tp, b, h] under SP
                tp = config.tensor_model_parallel_size
                out = out.transpose([1, 0, 2])[: out.shape[1] // tp]
            return out

    emb_layer = _Emb(config.vocab_size, config.hidden_size)
    with (
        patch(
            "paddlefleet.models.gpt.gpt_embedding.build_spec_layer",
            side_effect=lambda s, *a, **kw: emb_layer
            if s is mock_spec.language_embedding
            else None,
        ),
        patch(
            "paddlefleet.models.gpt.gpt_embedding.mark_context_parallel_parameter_disable_scale_grad"
        ),
    ):
        return GPTEmbedding(
            sublayers_spec=mock_spec,
            config=config,
            vocab_size=config.vocab_size,
            max_sequence_length=SEQ_LENGTH,
            position_embedding_type="rope",
        )


class TestCuSeqlensFromEmbedding(unittest.TestCase):
    """The embedding builds cu_seqlens once per step, but only for KDA models."""

    BATCH = 2
    DOCS = [[12, 20], [8, 24]]

    def _run(
        self,
        with_mask=True,
        cp_world_size=1,
        seq_len=SEQ_LENGTH,
        **cfg_kwargs,
    ):
        """Returns (preproc_output, mask, cp_scatter_mock)."""
        cfg_kwargs.setdefault("experimental_dataflow", True)
        config = GPTConfig(
            vocab_size=128,
            hidden_size=HIDDEN_SIZE,
            num_attention_heads=NUM_KEY_HEADS,
            num_hidden_layers=2,
            **cfg_kwargs,
        )
        emb = _build_gpt_embedding(config)
        indices = (
            _startend_row_indices(self.DOCS, seq_len) if with_mask else None
        )
        with (
            patch(
                "paddlefleet.models.gpt.gpt_embedding.get_context_parallel_world_size",
                return_value=cp_world_size,
            ),
            patch("paddlefleet.models.gpt.gpt_embedding.ScatterOp") as scatter,
            patch(
                "paddlefleet.models.gpt.gpt_embedding.ContextParallelScatterOp"
            ) as cp_scatter,
        ):
            scatter.apply = lambda x: x
            # Stand in for scatter_contiguous: rank 0's floor-divided slice.
            cp_scatter.apply = MagicMock(
                side_effect=lambda x, axis=1, **kw: x[
                    :, : x.shape[axis] // cp_world_size
                ]
            )
            out = emb.forward(
                {
                    "input_ids": paddle.randint(0, 128, [self.BATCH, seq_len]),
                    "attn_mask_startend_row_indices": indices,
                }
            )
        return out, indices, cp_scatter.apply

    def test_built_for_kda_layers(self):
        out, indices, _ = self._run(
            layer_types=["kimi_delta_attention", "self_attention"]
        )
        expected = build_cu_seqlens(indices, self.BATCH, SEQ_LENGTH)
        np.testing.assert_array_equal(
            out["cu_seqlens"].numpy(), expected.numpy()
        )

    def test_absent_without_kda_layers(self):
        out, _, _ = self._run(layer_types=["self_attention"] * 2)
        self.assertNotIn("cu_seqlens", out)
        out, _, _ = self._run()  # layer_types unset
        self.assertNotIn("cu_seqlens", out)

    def test_built_without_experimental_dataflow(self):
        """The old dataflow needs it too; the mask still has the MTP tail."""
        out, indices, _ = self._run(
            layer_types=["kimi_delta_attention", "self_attention"],
            experimental_dataflow=False,
        )
        expected = build_cu_seqlens(indices, self.BATCH, SEQ_LENGTH)
        np.testing.assert_array_equal(
            out["cu_seqlens"].numpy(), expected.numpy()
        )

    def test_no_mask_only_builds_one_for_context_parallel(self):
        """Without document boundaries only CP needs a (shared) cu_seqlens."""
        out, _, _ = self._run(
            with_mask=False,
            layer_types=["kimi_delta_attention", "self_attention"],
        )
        self.assertNotIn("cu_seqlens", out)

        out, _, cp_scatter = self._run(
            with_mask=False,
            cp_world_size=2,
            layer_types=["kimi_delta_attention", "self_attention"],
        )
        # hidden_states is this rank's shard, but cu_seqlens stays global
        cp_scatter.assert_called_once()
        self.assertEqual(out["hidden_states"].shape[1], SEQ_LENGTH // 2)
        self.assertEqual(
            out["cu_seqlens"].tolist(), [0, self.BATCH * SEQ_LENGTH]
        )

    def test_sequence_parallel_layout(self):
        """Under SP hidden_states is [s/tp, b, h]; the length scales back up."""
        out, indices, _ = self._run(
            layer_types=["kimi_delta_attention", "self_attention"],
            sequence_parallel=True,
            tensor_model_parallel_size=2,
        )
        self.assertEqual(
            out["hidden_states"].shape,
            [SEQ_LENGTH // 2, self.BATCH, HIDDEN_SIZE],
        )
        expected = build_cu_seqlens(indices, self.BATCH, SEQ_LENGTH)
        np.testing.assert_array_equal(
            out["cu_seqlens"].numpy(), expected.numpy()
        )

    def test_mtp_tail_is_sliced_off_the_mask(self):
        """MTP shortens the backbone, so the mask tail must be dropped here."""
        out, indices, _ = self._run(
            layer_types=["kimi_delta_attention", "self_attention"],
            experimental_dataflow=False,
            num_nextn_predict_layers=1,
        )
        backbone = SEQ_LENGTH - 1
        # the embedding concatenates the MTP depths along axis 0; every decoder
        # layer splits them off again, so cu_seqlens covers the backbone only
        self.assertEqual(
            out["hidden_states"].shape, [2 * self.BATCH, backbone, HIDDEN_SIZE]
        )
        expected = build_cu_seqlens(
            indices[:, :, :backbone, :], self.BATCH, backbone
        )
        np.testing.assert_array_equal(
            out["cu_seqlens"].numpy(), expected.numpy()
        )


class TestGatedNormRecompute(unittest.TestCase):
    """Selective recompute of the gated RMSNorm via RecomputeWithoutOutput.

    The gated norm output is dropped after the forward and re-run in backward,
    so the result must stay bit-exact while the norm is executed a second time.
    """

    def _forward_backward(self, kda, seed=1):
        paddle.seed(seed)
        x = paddle.randn([MICRO_BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE])
        x.stop_gradient = False
        out, _ = kda(hidden_states=x, attention_mask=None)
        out.pow(2).mean().backward()
        return (
            out.numpy(),
            x.grad.numpy(),
            {n: p.grad.numpy() for n, p in kda.named_parameters()},
        )

    def test_flag_follows_recompute_modules(self):
        """The flag is on only for selective recompute listing rms_norm_gated."""
        on = _build_kda(
            config_overrides={
                "recompute_granularity": "selective",
                "recompute_modules": ["rms_norm_gated"],
            }
        )
        self.assertTrue(on.recompute_rms_norm_gated)

        # selective, but a different module is requested
        other = _build_kda(
            config_overrides={
                "recompute_granularity": "selective",
                "recompute_modules": ["gated_attn"],
            }
        )
        self.assertFalse(other.recompute_rms_norm_gated)

        # no recompute configured at all
        self.assertFalse(_build_kda().recompute_rms_norm_gated)

    def test_layer_range_restricts_recompute(self):
        """recompute_num_layers / dict values must scope recompute by layer.

        With the test config (num_hidden_layers=2, pp=vpp=1) first_n/block with
        a count of 1 selects only layer 0, so layer 1 must stay off instead of
        every KDA layer recomputing.
        """

        def flag(layer_number, modules, num_layers=None, method="first_n"):
            overrides = {
                "recompute_granularity": "selective",
                "recompute_modules": modules,
                "recompute_method": method,
            }
            if num_layers is not None:
                overrides["recompute_num_layers"] = num_layers
            kda = _build_kda(
                layer_number=layer_number, config_overrides=overrides
            )
            return kda.recompute_rms_norm_gated

        # list + recompute_num_layers=1, first_n -> only layer 0
        self.assertTrue(flag(0, ["rms_norm_gated"], num_layers=1))
        self.assertFalse(flag(1, ["rms_norm_gated"], num_layers=1))

        # block behaves the same for a single-stage config
        self.assertTrue(
            flag(0, ["rms_norm_gated"], num_layers=1, method="block")
        )
        self.assertFalse(
            flag(1, ["rms_norm_gated"], num_layers=1, method="block")
        )

        # dict form: the value is the per-module layer count
        self.assertTrue(flag(0, {"rms_norm_gated": 1}))
        self.assertFalse(flag(1, {"rms_norm_gated": 1}))

        # list without a count still recomputes every layer
        self.assertTrue(flag(1, ["rms_norm_gated"]))

    def test_invalid_recompute_method_never_resolves_silently(self):
        """An invalid recompute_method must raise, never silently pick first_n.

        The config assert is stripped under ``python -O``; the resolver's raise
        is what still catches it there.
        """
        with self.assertRaises((AssertionError, ValueError)):
            _build_kda(
                layer_number=0,
                config_overrides={
                    "recompute_granularity": "selective",
                    "recompute_modules": ["rms_norm_gated"],
                    "recompute_num_layers": 1,
                    "recompute_method": "uniform",
                },
            )

    def test_dict_layer_count_without_method_rejected_at_startup(self):
        """A dict layer count needs first_n/block, checked at config init."""
        with self.assertRaises(ValueError):
            _build_kda(
                layer_number=0,
                config_overrides={
                    "recompute_granularity": "selective",
                    "recompute_modules": {"rms_norm_gated": 1},
                    "recompute_method": None,
                },
            )

    def test_method_none_resolves_as_first_n(self):
        """recompute_method=None is a first_n alias, like every other module."""

        def flag(layer_number):
            return _build_kda(
                layer_number=layer_number,
                config_overrides={
                    "recompute_granularity": "selective",
                    "recompute_modules": ["rms_norm_gated"],
                    "recompute_num_layers": 1,
                    "recompute_method": None,
                },
            ).recompute_rms_norm_gated

        self.assertTrue(flag(0))
        self.assertFalse(flag(1))

    def test_dict_selectors(self):
        """rms_norm_gated accepts every selector the shared resolver does."""

        def flag(layer_number, spec, method=None):
            return _build_kda(
                layer_number=layer_number,
                config_overrides={
                    "recompute_granularity": "selective",
                    "recompute_modules": {"rms_norm_gated": spec},
                    "recompute_method": method,
                },
            ).recompute_rms_norm_gated

        # Explicit layer list: only the listed global 0-based ids, no method
        # needed.
        self.assertFalse(flag(0, [1]))
        self.assertTrue(flag(1, [1]))
        self.assertTrue(flag(0, [0, 1]))

        # "all" / None / a negative count mean every layer.
        for spec in ("all", None, -1):
            self.assertTrue(flag(0, spec))
            self.assertTrue(flag(1, spec))

        # A layer count still honours recompute_method.
        self.assertTrue(flag(0, 1, method="first_n"))
        self.assertFalse(flag(1, 1, method="first_n"))
        self.assertTrue(flag(0, 1, method="block"))
        self.assertFalse(flag(1, 1, method="block"))

    def test_non_list_sequence_entry(self):
        """A tuple entry behaves like a list instead of silently disabling."""
        kda = _build_kda(
            layer_number=0,
            config_overrides={
                "recompute_granularity": "selective",
                "recompute_modules": ("rms_norm_gated",),
            },
        )
        self.assertTrue(kda.recompute_rms_norm_gated)

    def test_out_of_range_layer_id_rejected_at_startup(self):
        """num_hidden_layers=2 here, so layer id 5 is out of range."""
        with self.assertRaises(ValueError):
            _build_kda(
                layer_number=0,
                config_overrides={
                    "recompute_granularity": "selective",
                    "recompute_modules": {"rms_norm_gated": [0, 5]},
                },
            )

    def _assert_matches_baseline(self, deterministic):
        paddle.seed(0)
        baseline = _build_kda(deterministic_mode=deterministic)
        baseline.train()
        o0, gx0, g0 = self._forward_backward(baseline)

        paddle.seed(0)
        recomputed = _build_kda(deterministic_mode=deterministic)
        recomputed.recompute_rms_norm_gated = True
        recomputed.train()
        o1, gx1, g1 = self._forward_backward(recomputed)

        np.testing.assert_array_equal(o0, o1, err_msg="output")
        np.testing.assert_array_equal(gx0, gx1, err_msg="grad_x")
        for name in g0:
            np.testing.assert_array_equal(g0[name], g1[name], err_msg=name)

    def test_matches_baseline_native(self):
        """Paddle-native fallback: recompute is bit-exact with the baseline."""
        self._assert_matches_baseline(deterministic=True)

    @unittest.skipUnless(
        HAVE_FLA, "paddlefleet_ops fla kernels are not available"
    )
    def test_matches_baseline_fused(self):
        """Fused FLA kernel: recompute is bit-exact with the baseline."""
        self._assert_matches_baseline(deterministic=False)

    def test_reruns_gated_norm_in_backward(self):
        """With the flag on and training, the gated norm runs twice."""
        kda = _build_kda()
        kda.recompute_rms_norm_gated = True
        kda.train()
        with patch.object(kda, "_gated_norm", wraps=kda._gated_norm) as spy:
            self._forward_backward(kda)
        # once in forward, once when the backward hook recomputes it
        self.assertEqual(spy.call_count, 2)

    def test_no_recompute_without_flag(self):
        """Baseline path runs the gated norm exactly once."""
        kda = _build_kda()
        kda.train()
        with patch.object(kda, "_gated_norm", wraps=kda._gated_norm) as spy:
            self._forward_backward(kda)
        self.assertEqual(spy.call_count, 1)

    def test_no_recompute_in_eval(self):
        """Recompute is skipped outside training even when the flag is set."""
        kda = _build_kda()
        kda.recompute_rms_norm_gated = True
        kda.eval()
        with patch.object(kda, "_gated_norm", wraps=kda._gated_norm) as spy:
            self._forward_backward(kda)
        self.assertEqual(spy.call_count, 1)


if __name__ == "__main__":
    unittest.main()
