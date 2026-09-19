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

from paddlefleet.transformer.attention import (
    CrossAttention,
    CrossAttentionSublayersSpec,
    SelfAttention,
    SelfAttentionSublayersSpec,
)
from paddlefleet.transformer.dot_product_attention import DotProductAttention
from paddlefleet.transformer.enums import AttnMaskType
from paddlefleet.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSublayersSpec,
)
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.utils import (
    init_method_normal,
    scaled_init_method_normal,
)


class BiasedLinear(paddle.nn.Layer):
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__()
        self.linear = paddle.nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x), self.linear.bias


class RMSNorm(paddle.nn.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        hidden_size = kwargs.get("normalized_shape", kwargs.get("hidden_size"))
        eps = kwargs.get("norm_eps", kwargs.get("eps"))
        self.weight = paddle.nn.Parameter(paddle.zeros([hidden_size]))
        self.eps = eps

    def forward(self, x):
        d_norm = paddle.rsqrt(x.pow(2).mean(axis=-1, keepdim=True) + self.eps)
        return x * d_norm * self.weight


class TestSelfAttention(unittest.TestCase):
    def setUp(self):
        self.config = TransformerConfig(
            num_hidden_layers=1,
            hidden_size=128,
            num_attention_heads=4,
        )

        # TODO(liangshuhao): make these args formal
        self.config.num_key_value_heads = self.config.num_attention_heads
        self.config.head_dim = (
            self.config.hidden_size // self.config.num_attention_heads
        )
        self.config.softmax_scale = None
        self.config.use_bias = True
        self.config.no_rope_freq = None
        self.config.recompute_granularity = None
        self.config.fused_single_qkv_rope = False
        self.config.rotary_interleaved = False
        self.config.multi_latent_attention = False
        self.config.init_method = init_method_normal(0.02)
        self.config.output_layer_init_method = scaled_init_method_normal(
            0.02, 1, 2.0
        )
        self.config.rms_norm_eps = 1e-5
        self.config.context_parallel_size = 1
        self.config.apply_query_key_layer_scaling = False
        self.config.sliding_window = None
        self.config.window_attn_skip_freq = None
        self.config.fp16 = False
        self.config.bf16 = False
        self.config.masked_softmax_fusion = False
        self.config.attention_softmax_in_fp32 = True
        self.config.attention_dropout = 0.1
        self.config.softmax_type = "vanilla"

        self.self_attn = SelfAttention(
            self.config,
            SelfAttentionSublayersSpec(
                qkv_proj=BiasedLinear,
                core_attention=DotProductAttention,
                o_proj=BiasedLinear,
                q_norm=RMSNorm,
                k_norm=RMSNorm,
            ),
            attn_mask_type=AttnMaskType.causal,
            layer_number=1,
        )

    def test_self_attention(self):
        config = self.self_attn.config
        sequence_length = 127
        micro_batch_size = 2
        hidden_size = self.self_attn.config.hidden_size

        hidden_states = paddle.randn(
            (micro_batch_size, sequence_length, hidden_size),
        )
        rotary_pos_emb = paddle.randn(
            (1, sequence_length, 1, self.config.head_dim)
        )

        output, bias = self.self_attn(
            hidden_states, attention_mask=None, rotary_pos_emb=rotary_pos_emb
        )

        # Check if output and bias have the correct shape
        assert output.shape[0] == micro_batch_size
        assert output.shape[1] == sequence_length
        assert output.shape[2] == config.hidden_size
        assert bias.shape[0] == config.hidden_size


class TestSelfAttentionQKNormPerLayer(unittest.TestCase):
    """Test SelfAttention with qk_norm_type='per_layer'."""

    def setUp(self):
        self.config = TransformerConfig(
            num_hidden_layers=1,
            hidden_size=128,
            num_attention_heads=4,
        )
        self.config.num_key_value_heads = self.config.num_attention_heads
        self.config.head_dim = (
            self.config.hidden_size // self.config.num_attention_heads
        )
        self.config.softmax_scale = None
        self.config.use_bias = True
        self.config.no_rope_freq = None
        self.config.recompute_granularity = None
        self.config.fused_single_qkv_rope = False
        self.config.rotary_interleaved = False
        self.config.multi_latent_attention = False
        self.config.init_method = init_method_normal(0.02)
        self.config.output_layer_init_method = scaled_init_method_normal(
            0.02, 1, 2.0
        )
        self.config.rms_norm_eps = 1e-5
        self.config.context_parallel_size = 1
        self.config.apply_query_key_layer_scaling = False
        self.config.sliding_window = None
        self.config.window_attn_skip_freq = None
        self.config.fp16 = False
        self.config.bf16 = False
        self.config.masked_softmax_fusion = False
        self.config.attention_softmax_in_fp32 = True
        self.config.attention_dropout = 0.1
        self.config.softmax_type = "vanilla"
        self.config.qk_norm_type = "per_layer"

        self.self_attn = SelfAttention(
            self.config,
            SelfAttentionSublayersSpec(
                qkv_proj=BiasedLinear,
                core_attention=DotProductAttention,
                o_proj=BiasedLinear,
                q_norm=RMSNorm,
                k_norm=RMSNorm,
            ),
            attn_mask_type=AttnMaskType.causal,
            layer_number=1,
        )

    def test_self_attention_qk_norm_per_layer(self):
        config = self.self_attn.config
        sequence_length = 127
        micro_batch_size = 2
        hidden_size = self.self_attn.config.hidden_size

        hidden_states = paddle.randn(
            (micro_batch_size, sequence_length, hidden_size),
        )
        rotary_pos_emb = paddle.randn(
            (1, sequence_length, 1, self.config.head_dim)
        )

        output, bias = self.self_attn(
            hidden_states, attention_mask=None, rotary_pos_emb=rotary_pos_emb
        )

        # Check if output and bias have the correct shape
        assert output.shape[0] == micro_batch_size
        assert output.shape[1] == sequence_length
        assert output.shape[2] == config.hidden_size
        assert bias.shape[0] == config.hidden_size


class TestMLASelfAttention(unittest.TestCase):
    def setUp(self):
        self.config = TransformerConfig(
            num_hidden_layers=1,
            hidden_size=128,
            num_attention_heads=1,
        )

        self.config.num_key_value_heads = self.config.num_attention_heads
        self.config.head_dim = (
            self.config.hidden_size // self.config.num_attention_heads
        )
        self.config.softmax_scale = None
        self.config.use_bias = True
        self.config.no_rope_freq = None
        self.config.recompute_granularity = None
        self.config.fused_single_qkv_rope = False
        self.config.rotary_interleaved = False
        self.config.multi_latent_attention = True
        self.config.init_method = init_method_normal(0.02)
        self.config.output_layer_init_method = scaled_init_method_normal(
            0.02, 1, 2.0
        )
        self.config.rms_norm_eps = 1e-5
        self.config.context_parallel_size = 1
        self.config.apply_query_key_layer_scaling = False
        self.config.sliding_window = None
        self.config.window_attn_skip_freq = None
        self.config.fp16 = False
        self.config.bf16 = False
        self.config.masked_softmax_fusion = False
        self.config.attention_softmax_in_fp32 = True
        self.config.attention_dropout = 0.1
        self.config.softmax_type = "vanilla"

    def test_self_attention(self):
        self.self_attn = MLASelfAttention(
            self.config,
            MLASelfAttentionSublayersSpec(
                q_proj=BiasedLinear,
                q_a_proj=BiasedLinear,
                q_b_proj=BiasedLinear,
                kv_a_proj_with_mqa=BiasedLinear,
                kv_b_proj=BiasedLinear,
                core_attention=DotProductAttention,
                o_proj=BiasedLinear,
                q_a_layernorm=RMSNorm,
                kv_a_layernorm=RMSNorm,
            ),
            attn_mask_type=AttnMaskType.causal,
            layer_number=1,
        )
        config = self.self_attn.config
        sequence_length = 127
        micro_batch_size = 2
        hidden_size = self.self_attn.config.hidden_size

        hidden_states = paddle.randn(
            (micro_batch_size, sequence_length, hidden_size),
        )

        output, bias = self.self_attn(
            hidden_states,
            attention_mask=None,
        )

        # Check if output and bias have the correct shape
        assert output.shape[0] == micro_batch_size
        assert output.shape[1] == sequence_length
        assert output.shape[2] == config.hidden_size
        assert bias.shape[0] == config.hidden_size

    def test_self_attention_sp(self):
        self.config.sequence_parallel = True
        self.self_attn = MLASelfAttention(
            self.config,
            MLASelfAttentionSublayersSpec(
                q_proj=BiasedLinear,
                q_a_proj=BiasedLinear,
                q_b_proj=BiasedLinear,
                kv_a_proj_with_mqa=BiasedLinear,
                kv_b_proj=BiasedLinear,
                core_attention=DotProductAttention,
                o_proj=BiasedLinear,
                q_a_layernorm=RMSNorm,
                kv_a_layernorm=RMSNorm,
            ),
            attn_mask_type=AttnMaskType.causal,
            layer_number=1,
        )
        config = self.self_attn.config
        sequence_length = 127
        micro_batch_size = 2
        hidden_size = self.self_attn.config.hidden_size

        hidden_states = paddle.randn(
            (micro_batch_size, sequence_length, hidden_size),
        )

        output, bias = self.self_attn(
            hidden_states,
            attention_mask=None,
        )

        # Check if output and bias have the correct shape
        assert output.shape[0] == micro_batch_size
        assert output.shape[1] == sequence_length
        assert output.shape[2] == config.hidden_size
        assert bias.shape[0] == config.hidden_size


class TestGatedSelfAttention(unittest.TestCase):
    """Test SelfAttention with gated_attention=True (forward and backward)."""

    def _make_config(self, gqa=False):
        config = TransformerConfig(
            num_hidden_layers=1,
            hidden_size=128,
            num_attention_heads=4,
        )
        config.num_key_value_heads = 2 if gqa else config.num_attention_heads
        config.head_dim = config.hidden_size // config.num_attention_heads
        config.softmax_scale = None
        config.use_bias = True
        config.no_rope_freq = None
        config.recompute_granularity = None
        config.fused_single_qkv_rope = False
        config.rotary_interleaved = False
        config.multi_latent_attention = False
        config.init_method = init_method_normal(0.02)
        config.output_layer_init_method = scaled_init_method_normal(
            0.02, 1, 2.0
        )
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
        config.gated_attention = True
        return config

    def _build_attn(self, config):
        return SelfAttention(
            config,
            SelfAttentionSublayersSpec(
                qkv_proj=BiasedLinear,
                core_attention=DotProductAttention,
                o_proj=BiasedLinear,
                q_norm=RMSNorm,
                k_norm=RMSNorm,
            ),
            attn_mask_type=AttnMaskType.causal,
            layer_number=1,
        )

    def test_gated_attention_forward_shape(self):
        """Gated attention output should have the same shape as standard attention."""
        config = self._make_config()
        attn = self._build_attn(config)

        seq_len, batch_size = 64, 2
        hidden_states = paddle.randn((batch_size, seq_len, config.hidden_size))
        rotary_pos_emb = paddle.randn((1, seq_len, 1, config.head_dim))

        output, bias = attn(
            hidden_states, attention_mask=None, rotary_pos_emb=rotary_pos_emb
        )

        self.assertEqual(
            output.shape, [batch_size, seq_len, config.hidden_size]
        )
        self.assertEqual(bias.shape[0], config.hidden_size)
        self.assertTrue(
            paddle.all(paddle.isfinite(output)).item(),
            "Output contains NaN or Inf",
        )

    def test_gated_attention_backward(self):
        """Gated attention should produce valid gradients for all parameters."""
        config = self._make_config()
        attn = self._build_attn(config)

        seq_len, batch_size = 32, 2
        hidden_states = paddle.randn((batch_size, seq_len, config.hidden_size))
        hidden_states.stop_gradient = False
        rotary_pos_emb = paddle.randn((1, seq_len, 1, config.head_dim))

        output, bias = attn(
            hidden_states, attention_mask=None, rotary_pos_emb=rotary_pos_emb
        )
        loss = output.sum()
        loss.backward()

        # Check input gradient exists and is finite
        self.assertIsNotNone(hidden_states.grad)
        self.assertTrue(
            paddle.all(paddle.isfinite(hidden_states.grad)).item(),
            "Input gradient contains NaN or Inf",
        )

        # Check all parameter gradients exist and are finite
        for name, param in attn.named_parameters():
            self.assertIsNotNone(
                param.grad, f"Parameter {name} has no gradient"
            )
            self.assertTrue(
                paddle.all(paddle.isfinite(param.grad)).item(),
                f"Parameter {name} gradient contains NaN or Inf",
            )

    def test_gated_attention_gqa(self):
        """Gated attention should work with grouped query attention (GQA)."""
        config = self._make_config(gqa=True)
        attn = self._build_attn(config)

        seq_len, batch_size = 32, 2
        hidden_states = paddle.randn((batch_size, seq_len, config.hidden_size))
        hidden_states.stop_gradient = False
        rotary_pos_emb = paddle.randn((1, seq_len, 1, config.head_dim))

        output, bias = attn(
            hidden_states, attention_mask=None, rotary_pos_emb=rotary_pos_emb
        )
        loss = output.sum()
        loss.backward()

        self.assertEqual(
            output.shape, [batch_size, seq_len, config.hidden_size]
        )
        self.assertIsNotNone(hidden_states.grad)
        self.assertTrue(
            paddle.all(paddle.isfinite(output)).item(),
            "GQA gated attention output contains NaN or Inf",
        )

    def test_gate_has_effect(self):
        """Verify that the gate actually modulates the output (not a no-op)."""
        config_gated = self._make_config()
        config_ungated = self._make_config()
        config_ungated.gated_attention = False

        paddle.manual_seed(42)
        attn_gated = self._build_attn(config_gated)
        paddle.manual_seed(42)
        attn_ungated = self._build_attn(config_ungated)

        seq_len, batch_size = 32, 2
        paddle.manual_seed(123)
        hidden_states = paddle.randn(
            (batch_size, seq_len, config_gated.hidden_size)
        )
        rotary_pos_emb = paddle.randn((1, seq_len, 1, config_gated.head_dim))

        out_gated, _ = attn_gated(
            hidden_states, attention_mask=None, rotary_pos_emb=rotary_pos_emb
        )
        out_ungated, _ = attn_ungated(
            hidden_states, attention_mask=None, rotary_pos_emb=rotary_pos_emb
        )

        # Outputs should differ because gated has extra gate projection
        self.assertFalse(
            paddle.allclose(out_gated, out_ungated, atol=1e-6).item(),
            "Gated and ungated outputs should differ",
        )


class TestMLAUseVarlenSelfAttention(TestMLASelfAttention):
    def setUp(self):
        super().setUp()
        self.config.flashmask_use_varlen = True


def _build_attention_config(**overrides):
    """Build a fully-populated TransformerConfig for attention construction.

    Mirrors the attribute set the existing ``TestSelfAttention.setUp`` relies
    on so the added behavior tests exercise the real constructor. ``overrides``
    are applied last, so callers can flip a single knob (recompute flags, kv
    head count, gated_attention, ...) and observe the resulting behavior.
    """
    num_hidden_layers = overrides.pop("num_hidden_layers", 1)
    num_attention_heads = overrides.pop("num_attention_heads", 4)
    hidden_size = overrides.pop("hidden_size", 128)
    config = TransformerConfig(
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
    )
    config.num_key_value_heads = num_attention_heads
    config.head_dim = hidden_size // num_attention_heads
    config.softmax_scale = None
    config.use_bias = True
    config.no_rope_freq = None
    config.recompute_granularity = None
    config.fused_single_qkv_rope = False
    config.rotary_interleaved = False
    config.multi_latent_attention = False
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
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


class TestSelfAttentionRecomputeConfig(unittest.TestCase):
    """SelfAttention constructor consumes recompute_* config into flags.

    Each test flips a single recompute knob and observes the resulting
    attribute, so a constructor that ignored the config (or wired the wrong
    module) would be caught rather than passing on a bare ``is True`` check.
    """

    def _build(self, **overrides):
        config = _build_attention_config(**overrides)
        return SelfAttention(
            config,
            SelfAttentionSublayersSpec(
                qkv_proj=BiasedLinear,
                core_attention=DotProductAttention,
                o_proj=BiasedLinear,
            ),
            attn_mask_type=AttnMaskType.causal,
            layer_number=1,
        )

    def test_core_attn_recompute_off_by_default(self):
        attn = self._build()
        self.assertFalse(attn.recompute_core_attention)
        self.assertFalse(attn.use_rr_flash_attention)

    def test_selective_core_attn_enables_core_recompute_only(self):
        attn = self._build(
            recompute_granularity="selective",
            recompute_modules=["core_attn"],
        )
        # core_attn listed -> core recompute on; flash_attn absent -> RR off.
        self.assertTrue(attn.recompute_core_attention)
        self.assertFalse(attn.use_rr_flash_attention)

    def test_flash_attn_module_enables_refined_recompute(self):
        attn = self._build(
            recompute_granularity="selective",
            recompute_modules=["core_attn", "flash_attn"],
        )
        self.assertTrue(attn.recompute_core_attention)
        self.assertTrue(attn.use_rr_flash_attention)

    def test_block_method_selects_layers_by_num_layers(self):
        # num_hidden_layers=2 -> chunk covers layers [0, 1]; layer_number=1.
        # recompute_num_layers=1 selects only layer 0, =2 selects both. The
        # boundary proves need_recompute_in_block actually consumes the count.
        attn_excluded = self._build(
            num_hidden_layers=2,
            recompute_granularity="selective",
            recompute_modules=["core_attn"],
            recompute_method="block",
            recompute_num_layers=1,
        )
        self.assertFalse(attn_excluded.recompute_core_attention)

        attn_included = self._build(
            num_hidden_layers=2,
            recompute_granularity="selective",
            recompute_modules=["core_attn"],
            recompute_method="block",
            recompute_num_layers=2,
        )
        self.assertTrue(attn_included.recompute_core_attention)

    def test_set_for_recompute_input_layernorm_not_implemented(self):
        attn = self._build()
        with self.assertRaises(NotImplementedError):
            attn.set_for_recompute_input_layernorm()


class TestSelfAttentionQKVProjectionLayout(unittest.TestCase):
    """get_query_key_value_tensors emits the documented per-group QKV layout.

    The reference is the fused projection recomputed independently (x @ W + b
    read off the real qkv_proj weights) then sliced by hand according to the
    per-group interleaved layout. This distinguishes the correct
    ``[Q | K | V]`` (and gated ``[Q | Gate | K | V]``) grouping from a plain
    ``[all_Q | all_K | all_V]`` split, a Q/K/V swap, or a wrong group stride --
    none of which a shape-only check would catch.
    """

    def _build(self, **overrides):
        config = _build_attention_config(**overrides)
        attn = SelfAttention(
            config,
            SelfAttentionSublayersSpec(
                qkv_proj=BiasedLinear,
                core_attention=DotProductAttention,
                o_proj=BiasedLinear,
                q_norm=None,
                k_norm=None,
            ),
            attn_mask_type=AttnMaskType.causal,
            layer_number=1,
        )
        attn.eval()
        return attn

    def _reference_mixed_qkv(self, attn, hidden_states):
        # paddle.nn.Linear computes x @ weight + bias with weight [in, out].
        weight = attn.qkv_proj.linear.weight.numpy()
        bias = attn.qkv_proj.linear.bias.numpy()
        return hidden_states.numpy() @ weight + bias

    def test_split_qkv_true_matches_independent_layout(self):
        attn = self._build()
        batch, seq = 2, 4
        heads, head_dim = 4, 32
        group_dim = 3 * head_dim  # Q | K | V per group, one head per group
        hidden_states = paddle.randn([batch, seq, 128], dtype="float32")

        query, key, value = attn.get_query_key_value_tensors(
            hidden_states, split_qkv=True
        )
        self.assertEqual(query.shape, [batch, seq, heads, head_dim])
        self.assertEqual(key.shape, [batch, seq, heads, head_dim])
        self.assertEqual(value.shape, [batch, seq, heads, head_dim])

        mixed = self._reference_mixed_qkv(attn, hidden_states)
        exp_q = np.stack(
            [
                mixed[:, :, g * group_dim : g * group_dim + head_dim]
                for g in range(heads)
            ],
            axis=2,
        )
        exp_k = np.stack(
            [
                mixed[
                    :,
                    :,
                    g * group_dim + head_dim : g * group_dim + 2 * head_dim,
                ]
                for g in range(heads)
            ],
            axis=2,
        )
        exp_v = np.stack(
            [
                mixed[
                    :,
                    :,
                    g * group_dim + 2 * head_dim : g * group_dim + 3 * head_dim,
                ]
                for g in range(heads)
            ],
            axis=2,
        )
        np.testing.assert_allclose(query.numpy(), exp_q, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(key.numpy(), exp_k, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(value.numpy(), exp_v, rtol=1e-5, atol=1e-5)

    def test_split_qkv_false_returns_unsplit_with_arg_list(self):
        attn = self._build()
        batch, seq = 2, 4
        groups, group_dim = 4, 96
        hidden_states = paddle.randn([batch, seq, 128], dtype="float32")

        mixed_qkv, split_arg_list = attn.get_query_key_value_tensors(
            hidden_states, split_qkv=False
        )
        # q_dim, k head_dim, v head_dim for one head per group.
        self.assertEqual(split_arg_list, [32, 32, 32])
        self.assertEqual(sum(split_arg_list), group_dim)
        self.assertEqual(mixed_qkv.shape, [batch, seq, groups, group_dim])

        ref = self._reference_mixed_qkv(attn, hidden_states)
        ref = ref.reshape(batch, seq, groups, group_dim)
        np.testing.assert_allclose(mixed_qkv.numpy(), ref, rtol=1e-5, atol=1e-5)

    def test_gated_qkv_layout_places_gate_between_q_and_k(self):
        attn = self._build(gated_attention=True)
        batch, seq = 2, 4
        heads, head_dim = 4, 32
        group_dim = 4 * head_dim  # Q | Gate | K | V per group
        hidden_states = paddle.randn([batch, seq, 128], dtype="float32")

        result = attn.get_query_key_value_tensors(hidden_states, split_qkv=True)
        self.assertEqual(len(result), 4)
        query, key, value, gate = result
        self.assertEqual(query.shape, [batch, seq, heads, head_dim])
        self.assertEqual(gate.shape, [batch, seq, heads * head_dim])

        mixed = self._reference_mixed_qkv(attn, hidden_states)
        exp_q = np.stack(
            [
                mixed[:, :, g * group_dim : g * group_dim + head_dim]
                for g in range(heads)
            ],
            axis=2,
        )
        exp_k = np.stack(
            [
                mixed[
                    :,
                    :,
                    g * group_dim + 2 * head_dim : g * group_dim + 3 * head_dim,
                ]
                for g in range(heads)
            ],
            axis=2,
        )
        exp_v = np.stack(
            [
                mixed[
                    :,
                    :,
                    g * group_dim + 3 * head_dim : g * group_dim + 4 * head_dim,
                ]
                for g in range(heads)
            ],
            axis=2,
        )
        # Gate sits at columns [head_dim, 2*head_dim) of each group and is
        # flattened head-major into [batch, seq, heads * head_dim].
        exp_gate = np.concatenate(
            [
                mixed[
                    :,
                    :,
                    g * group_dim + head_dim : g * group_dim + 2 * head_dim,
                ]
                for g in range(heads)
            ],
            axis=-1,
        )
        np.testing.assert_allclose(query.numpy(), exp_q, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(key.numpy(), exp_k, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(value.numpy(), exp_v, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(gate.numpy(), exp_gate, rtol=1e-5, atol=1e-5)


class TestCrossAttentionContracts(unittest.TestCase):
    """CrossAttention construction and its documented rejection contracts.

    CrossAttention is otherwise untested in this file. These lock in the
    supported (MHA-only) construction and the three explicit failure modes,
    using assertRaises on the exact exception type rather than swallowing.
    """

    def _spec(self):
        return CrossAttentionSublayersSpec(
            linear_q=BiasedLinear,
            linear_kv=BiasedLinear,
            core_attention=DotProductAttention,
            o_proj=BiasedLinear,
        )

    def _build(self, **overrides):
        config = _build_attention_config(**overrides)
        return CrossAttention(
            config,
            self._spec(),
            attn_mask_type=AttnMaskType.padding,
            layer_number=1,
        )

    def test_mha_construction_wires_projections(self):
        attn = self._build()
        self.assertEqual(attn.attention_type, "cross")
        self.assertIsNotNone(attn.linear_q)
        self.assertIsNotNone(attn.linear_kv)
        # Equal head counts -> equal query/key projection sizes (asserted in
        # the constructor); expose that the invariant actually held.
        self.assertEqual(attn.query_projection_size, attn.key_projection_size)

    def test_group_query_attention_rejected(self):
        with self.assertRaises(ValueError):
            self._build(num_key_value_heads=2)

    def test_split_qkv_false_rejected(self):
        attn = self._build()
        hidden_states = paddle.randn([2, 4, 128], dtype="float32")
        key_value_states = paddle.randn([2, 4, 128], dtype="float32")
        with self.assertRaises(AssertionError):
            attn.get_query_key_value_tensors(
                hidden_states, key_value_states, split_qkv=False
            )

    def test_backward_dw_not_available(self):
        attn = self._build()
        # backward_dw is a SelfAttention-only weight-update hook.
        with self.assertRaises(AttributeError):
            attn.backward_dw()


if __name__ == "__main__":
    unittest.main()
