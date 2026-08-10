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
        kda.config.max_sequence_length = SEQ_LENGTH * cp_size
        return kda

    def test_shard_must_be_one_over_cp_size(self):
        kda = self._kda()
        kda.config.max_sequence_length = SEQ_LENGTH  # not SEQ_LENGTH * cp_size
        with self.assertRaises(ValueError):
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


if __name__ == "__main__":
    unittest.main()
