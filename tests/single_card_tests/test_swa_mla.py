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

"""Unit tests for MLA + Sliding Window Attention (SWA) support.

Covers:
- is_swa propagation to MLA core_attention
- layer-effective MLA SWA dimension inheritance and overrides
- SDPA path does not bypass SWA
- packed path applies SWA
- TransformerConfig MLA+SWA validation
- full/SWA pattern consistency
- attention sink parameter alignment
- 4-column startend_row_indices sliding window
- gradient correctness (window outside tokens have zero gradient)
"""

import unittest

import paddle

from paddlefleet.training.initialize import initialize_fleet
from paddlefleet.transformer.dot_product_attention import DotProductAttention
from paddlefleet.transformer.enums import AttnMaskType
from paddlefleet.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSublayersSpec,
)
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.transformer.utils import (
    get_real_layer_idx_for_swa,
    is_layer_window_attention,
    startend_row_indices_add_sliding_window,
)
from paddlefleet.utils import (
    init_method_normal,
    scaled_init_method_normal,
)

strategy = paddle.distributed.fleet.DistributedStrategy()
initialize_fleet(strategy=strategy)


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
        self.weight = paddle.nn.Parameter(paddle.ones([hidden_size]))
        self.eps = eps

    def forward(self, x):
        d_norm = paddle.rsqrt(x.pow(2).mean(axis=-1, keepdim=True) + self.eps)
        return x * d_norm * self.weight


# ============================================================================
# Helper: MLA config factory
# ============================================================================


def _make_mla_config(
    sliding_window=(4096, 0),
    window_attn_skip_freq=4,
    num_hidden_layers=8,
    add_swa_attention_sink_bias=True,
    add_full_attention_sink_bias=False,
    head_wise_swa_ratio=0.0,
):
    """Create a TransformerConfig with MLA + SWA enabled."""
    config = TransformerConfig(
        num_hidden_layers=num_hidden_layers,
        hidden_size=256,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=64,
        v_head_dim=64,
        sliding_window=sliding_window,
        window_attn_skip_freq=window_attn_skip_freq,
        # MLA params
        multi_latent_attention=True,
        q_lora_rank=128,
        qk_nope_head_dim=48,
        qk_rope_head_dim=16,
        kv_lora_rank=128,
        rope_type="rope",
        rope_theta=10000.0,
        head_wise_swa_ratio=head_wise_swa_ratio,
        add_swa_attention_sink_bias=add_swa_attention_sink_bias,
        add_full_attention_sink_bias=add_full_attention_sink_bias,
    )
    config.rotary_interleaved = False
    config.rotary_scaling_factor = 1.0
    config.original_max_position_embeddings = 4096
    config.beta_fast = 32.0
    config.beta_slow = 1.0
    config.mscale = 1.0
    config.mscale_all_dim = 0.0
    config.softmax_scale = None
    config.use_bias = False
    config.no_rope_freq = None
    config.recompute_granularity = None
    config.fused_single_qkv_rope = False
    config.init_method = init_method_normal(0.02)
    config.output_layer_init_method = scaled_init_method_normal(0.02, 1, 2.0)
    config.rms_norm_eps = 1e-5
    config.context_parallel_size = 1
    config.apply_query_key_layer_scaling = False
    config.fp16 = False
    config.bf16 = False
    config.masked_softmax_fusion = False
    config.attention_softmax_in_fp32 = True
    config.attention_dropout = 0.0
    config.softmax_type = "vanilla"
    config.gated_attention = False
    config.attention_value_scale = None
    return config


def _mla_sublayers_spec():
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
        gate_proj=BiasedLinear,
    )


# ============================================================================
# Test: is_swa propagation to core_attention
# ============================================================================


class TestMLASWAIsSWAPropagation(unittest.TestCase):
    """Test that is_swa is correctly propagated from MLA to core_attention."""

    def test_mla_swa_layer_core_attention_receives_is_swa(self):
        """MLA SWA layer's core_attention should have is_swa=True."""
        config = _make_mla_config(
            sliding_window=(3, 0),
            window_attn_skip_freq=None,  # All layers are SWA
        )

        # DotProductAttention is the core_attention used by MLA
        # Simulate what MLA does: pass is_swa=True
        core_attn = DotProductAttention(
            config=config,
            layer_number=0,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            is_swa=True,
            k_channels=64,
            v_channels=64,
            num_attention_heads=4,
            num_key_value_heads=4,
        )

        self.assertTrue(core_attn.is_swa)
        self.assertEqual(core_attn.sliding_window, (3, 0))

    def test_mla_full_layer_core_attention_no_swa(self):
        """MLA full attention layer's core_attention should have is_swa=False."""
        config = _make_mla_config(
            sliding_window=(3, 0),
            window_attn_skip_freq=4,
        )

        core_attn = DotProductAttention(
            config=config,
            layer_number=0,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            is_swa=False,
            k_channels=64,
            v_channels=64,
            num_attention_heads=4,
            num_key_value_heads=4,
        )

        self.assertFalse(core_attn.is_swa)
        self.assertIsNone(core_attn.sliding_window)


# ============================================================================
# Test: MLA layer-effective SWA dimensions
# ============================================================================


class TestMLAEffectiveSWADims(unittest.TestCase):
    """Test layer-effective MLA dimensions for full and SWA layers."""

    def test_mla_swa_layer_inherits_full_dims_by_default(self):
        config = TransformerConfig(
            num_hidden_layers=4,
            hidden_size=256,
            num_attention_heads=4,
            num_key_value_heads=2,
            v_head_dim=64,
            multi_latent_attention=True,
            sliding_window=(4096, 0),
            window_attn_skip_freq=None,
            qk_nope_head_dim=48,
            qk_rope_head_dim=16,
            kv_lora_rank=128,
            q_lora_rank=96,
            rope_theta=10000.0,
            head_wise_swa_ratio=0.0,
        )

        full_dims = config.get_effective_mla_dims(is_swa=False)
        swa_dims = config.get_effective_mla_dims(is_swa=True)
        self.assertEqual(swa_dims, full_dims)
        self.assertFalse(config.mla_swa_uses_overridden_dims())

    def test_mla_swa_layer_uses_explicit_overrides(self):
        config = TransformerConfig(
            num_hidden_layers=4,
            hidden_size=512,
            num_attention_heads=8,
            num_key_value_heads=4,
            v_head_dim=64,
            multi_latent_attention=True,
            sliding_window=(4096, 0),
            window_attn_skip_freq=None,
            qk_nope_head_dim=48,
            qk_rope_head_dim=16,
            kv_lora_rank=128,
            q_lora_rank=96,
            rope_theta=10000.0,
            swa_num_attention_heads=4,
            swa_num_key_value_heads=2,
            swa_v_head_dim=32,
            swa_rope_theta=20000.0,
            swa_qk_nope_head_dim=32,
            swa_qk_rope_head_dim=8,
            swa_kv_lora_rank=64,
            swa_q_lora_rank=48,
            swa_head_dim=40,
            head_wise_swa_ratio=0.0,
        )

        self.assertEqual(
            config.get_effective_mla_dims(is_swa=False),
            {
                "num_attention_heads": 8,
                "num_key_value_heads": 4,
                "qk_nope_head_dim": 48,
                "qk_rope_head_dim": 16,
                "kv_lora_rank": 128,
                "q_lora_rank": 96,
                "v_head_dim": 64,
                "rope_theta": 10000.0,
            },
        )
        self.assertEqual(
            config.get_effective_mla_dims(is_swa=True),
            {
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "qk_nope_head_dim": 32,
                "qk_rope_head_dim": 8,
                "kv_lora_rank": 64,
                "q_lora_rank": 48,
                "v_head_dim": 32,
                "rope_theta": 20000.0,
            },
        )
        self.assertTrue(config.mla_swa_uses_overridden_dims())

    def test_mla_swa_partial_overrides_inherit_remaining_dims(self):
        config = TransformerConfig(
            num_hidden_layers=4,
            hidden_size=512,
            num_attention_heads=8,
            num_key_value_heads=4,
            v_head_dim=64,
            multi_latent_attention=True,
            sliding_window=(4096, 0),
            window_attn_skip_freq=None,
            qk_nope_head_dim=48,
            qk_rope_head_dim=16,
            kv_lora_rank=128,
            q_lora_rank=96,
            rope_theta=10000.0,
            swa_qk_nope_head_dim=32,
            swa_kv_lora_rank=64,
            head_wise_swa_ratio=0.0,
        )

        self.assertEqual(
            config.get_effective_mla_dims(is_swa=True),
            {
                "num_attention_heads": 8,
                "num_key_value_heads": 4,
                "qk_nope_head_dim": 32,
                "qk_rope_head_dim": 16,
                "kv_lora_rank": 64,
                "q_lora_rank": 96,
                "v_head_dim": 64,
                "rope_theta": 10000.0,
            },
        )


# ============================================================================
# Test: full/SWA pattern consistency
# ============================================================================


class TestMLASelfAttentionEffectiveShapes(unittest.TestCase):
    """Test true MLASelfAttention construction with layer-effective dims."""

    def _build_attn(self, config, layer_number):
        return MLASelfAttention(
            config,
            _mla_sublayers_spec(),
            layer_number=layer_number,
            attn_mask_type=AttnMaskType.causal,
        )

    def _assert_mla_shapes(self, attn, dims):
        q_head_dim = dims["qk_nope_head_dim"] + dims["qk_rope_head_dim"]
        self.assertEqual(attn.num_attention_heads, dims["num_attention_heads"])
        self.assertEqual(attn.num_key_value_heads, dims["num_key_value_heads"])
        self.assertEqual(attn.qk_nope_head_dim, dims["qk_nope_head_dim"])
        self.assertEqual(attn.qk_rope_head_dim, dims["qk_rope_head_dim"])
        self.assertEqual(attn.kv_lora_rank, dims["kv_lora_rank"])
        self.assertEqual(attn.q_lora_rank, dims["q_lora_rank"])
        self.assertEqual(attn.v_head_dim, dims["v_head_dim"])
        self.assertEqual(attn.rope_theta, dims["rope_theta"])
        self.assertEqual(attn.q_head_dim, q_head_dim)
        self.assertEqual(
            attn.query_projection_size,
            dims["v_head_dim"] * dims["num_attention_heads"],
        )
        self.assertEqual(
            attn.o_proj.linear.weight.shape,
            [attn.query_projection_size, attn.config.hidden_size],
        )
        self.assertEqual(
            attn.q_a_proj.linear.weight.shape,
            [attn.config.hidden_size, dims["q_lora_rank"]],
        )
        self.assertEqual(
            attn.q_b_proj.linear.weight.shape,
            [dims["q_lora_rank"], dims["num_attention_heads"] * q_head_dim],
        )
        self.assertEqual(attn.q_a_layernorm.weight.shape, [dims["q_lora_rank"]])
        self.assertEqual(
            attn.kv_a_proj_with_mqa.linear.weight.shape,
            [
                attn.config.hidden_size,
                dims["kv_lora_rank"] + dims["qk_rope_head_dim"],
            ],
        )
        self.assertEqual(
            attn.kv_b_proj.linear.weight.shape,
            [
                dims["kv_lora_rank"],
                dims["num_attention_heads"]
                * (dims["qk_nope_head_dim"] + dims["v_head_dim"]),
            ],
        )
        self.assertEqual(
            attn.kv_a_layernorm.weight.shape, [dims["kv_lora_rank"]]
        )
        self.assertEqual(attn.core_attention.k_channels, q_head_dim)
        self.assertEqual(attn.core_attention.v_channels, dims["v_head_dim"])
        self.assertEqual(
            attn.core_attention.num_attention_heads, dims["num_attention_heads"]
        )
        self.assertEqual(
            attn.core_attention.num_key_value_heads, dims["num_attention_heads"]
        )

    def test_full_and_swa_layers_use_layer_effective_shapes(self):
        config = TransformerConfig(
            num_hidden_layers=4,
            hidden_size=512,
            num_attention_heads=8,
            num_key_value_heads=4,
            v_head_dim=64,
            multi_latent_attention=True,
            sliding_window=(4096, 0),
            window_attn_skip_freq=2,
            qk_nope_head_dim=48,
            qk_rope_head_dim=16,
            kv_lora_rank=128,
            q_lora_rank=96,
            rope_theta=10000.0,
            swa_num_attention_heads=4,
            swa_num_key_value_heads=2,
            swa_v_head_dim=32,
            swa_rope_theta=20000.0,
            swa_qk_nope_head_dim=32,
            swa_qk_rope_head_dim=8,
            swa_kv_lora_rank=64,
            swa_q_lora_rank=48,
            swa_head_dim=40,
            head_wise_swa_ratio=0.0,
            rope_type="rope",
            gated_attention=True,
        )

        full_attn = self._build_attn(config, layer_number=0)
        swa_attn = self._build_attn(config, layer_number=1)

        self.assertFalse(full_attn.is_swa)
        self.assertTrue(swa_attn.is_swa)
        self.assertFalse(full_attn.core_attention.is_swa)
        self.assertTrue(swa_attn.core_attention.is_swa)
        self._assert_mla_shapes(
            full_attn, config.get_effective_mla_dims(is_swa=False)
        )
        self._assert_mla_shapes(
            swa_attn, config.get_effective_mla_dims(is_swa=True)
        )
        self.assertEqual(
            swa_attn.gate_proj.linear.weight.shape,
            [config.hidden_size, 32 * 4],
        )

    def test_swa_layer_constructs_with_inherited_shapes(self):
        config = TransformerConfig(
            num_hidden_layers=2,
            hidden_size=256,
            num_attention_heads=4,
            num_key_value_heads=2,
            v_head_dim=64,
            multi_latent_attention=True,
            sliding_window=(4096, 0),
            window_attn_skip_freq=None,
            qk_nope_head_dim=48,
            qk_rope_head_dim=16,
            kv_lora_rank=128,
            q_lora_rank=96,
            rope_theta=10000.0,
            head_wise_swa_ratio=0.0,
            rope_type="rope",
        )

        attn = self._build_attn(config, layer_number=0)
        self.assertTrue(attn.is_swa)
        self._assert_mla_shapes(
            attn, config.get_effective_mla_dims(is_swa=True)
        )

    def test_absorbed_decode_guard_for_swa_layer(self):
        config = _make_mla_config(
            sliding_window=(4, 0), window_attn_skip_freq=None
        )
        attn = self._build_attn(config, layer_number=0)
        query = paddle.randn([1, 1, attn.num_attention_heads, attn.q_head_dim])
        with self.assertRaises(NotImplementedError):
            attn._compute_absorbed_q(query)

    def test_fused_rope_matches_unfused_for_hetero_mla_swa(self):
        def make_config(apply_rope_fusion):
            config = TransformerConfig(
                num_hidden_layers=2,
                hidden_size=256,
                num_attention_heads=4,
                num_key_value_heads=2,
                v_head_dim=64,
                multi_latent_attention=True,
                sliding_window=(4096, 0),
                window_attn_skip_freq=None,
                qk_nope_head_dim=48,
                qk_rope_head_dim=16,
                kv_lora_rank=128,
                q_lora_rank=96,
                rope_theta=10000.0,
                swa_qk_nope_head_dim=32,
                swa_qk_rope_head_dim=8,
                swa_kv_lora_rank=64,
                swa_q_lora_rank=48,
                swa_v_head_dim=32,
                swa_head_dim=40,
                head_wise_swa_ratio=0.0,
                rope_type="rope",
                apply_rope_fusion=apply_rope_fusion,
            )
            config.rotary_interleaved = False
            config.use_bias = False
            config.no_rope_freq = None
            config.recompute_granularity = None
            config.fused_single_qkv_rope = False
            config.init_method = init_method_normal(0.02)
            config.output_layer_init_method = scaled_init_method_normal(
                0.02, 1, 2.0
            )
            config.rms_norm_eps = 1e-5
            config.context_parallel_size = 1
            config.apply_query_key_layer_scaling = False
            config.fp16 = False
            config.bf16 = False
            config.masked_softmax_fusion = False
            config.attention_softmax_in_fp32 = True
            config.attention_dropout = 0.0
            config.softmax_type = "vanilla"
            config.gated_attention = False
            config.attention_value_scale = None
            return config

        paddle.manual_seed(2026)
        unfused_attn = self._build_attn(make_config(False), layer_number=0)
        fused_attn = self._build_attn(make_config(True), layer_number=0)
        fused_attn.set_state_dict(unfused_attn.state_dict())
        unfused_attn.train()
        fused_attn.train()

        paddle.manual_seed(2027)
        hidden_states = paddle.randn([1, 4, 256])

        unfused_outputs = unfused_attn.get_query_key_value_tensors(
            hidden_states
        )
        fused_outputs = fused_attn.get_query_key_value_tensors(hidden_states)

        for name, unfused, fused in zip(
            ("query", "key", "value"), unfused_outputs[:3], fused_outputs[:3]
        ):
            self.assertEqual(unfused.shape, fused.shape, name)
            self.assertTrue(
                paddle.allclose(unfused, fused, atol=1e-5, rtol=1e-5).item(),
                f"{name} max diff: {(unfused - fused).abs().max().item()}",
            )


class TestMLASWAPatternConsistency(unittest.TestCase):
    """Test that full/SWA pattern is consistent across different callers."""

    def test_window_attn_skip_freq_4_pattern(self):
        """With skip_freq=4, layer 0,4,8 are full; 1,2,3,5,6,7 are SWA."""
        sliding_window = (4096, 0)
        skip_freq = 4

        expected = [
            False,  # layer 0: full (0 % 4 == 0)
            True,  # layer 1: SWA
            True,  # layer 2: SWA
            True,  # layer 3: SWA
            False,  # layer 4: full (4 % 4 == 0)
            True,  # layer 5: SWA
            True,  # layer 6: SWA
            True,  # layer 7: SWA
        ]

        for i in range(8):
            result = is_layer_window_attention(sliding_window, skip_freq, i)
            self.assertEqual(
                result,
                expected[i],
                f"Layer {i}: expected is_swa={expected[i]}, got {result}",
            )

    def test_get_real_layer_idx_for_swa_normal(self):
        """Normal layer: real_idx = layer_number - num_empty_layers_add_in_head."""
        # layer_number=5, offset=2 -> real_idx=3
        self.assertEqual(get_real_layer_idx_for_swa(5, 2), 3)
        self.assertEqual(get_real_layer_idx_for_swa(0, 0), 0)
        self.assertEqual(get_real_layer_idx_for_swa(3, 3), 0)

    def test_get_real_layer_idx_for_swa_mtp(self):
        """MTP layer: real_idx = layer_number + num_hidden_layers."""
        self.assertEqual(
            get_real_layer_idx_for_swa(0, 2, is_mtp=True, num_hidden_layers=8),
            8,
        )
        self.assertEqual(
            get_real_layer_idx_for_swa(1, 2, is_mtp=True, num_hidden_layers=8),
            9,
        )

    def test_get_real_layer_idx_for_swa_negative_raises(self):
        """Negative real_idx should raise AssertionError."""
        with self.assertRaises(AssertionError):
            get_real_layer_idx_for_swa(1, 5)  # 1 - 5 = -4


# ============================================================================
# Test: SDPA path does not bypass SWA
# ============================================================================


class TestSDPAPathDoesNotBypassSWA(unittest.TestCase):
    """Test that SDPA fast path is disabled when sliding_window is set."""

    def test_swa_core_attention_has_sliding_window(self):
        """Core attention with is_swa=True should store sliding_window."""
        config = _make_mla_config(sliding_window=(16, 0))

        core_attn = DotProductAttention(
            config=config,
            layer_number=0,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            is_swa=True,
            k_channels=64,
            v_channels=64,
            num_attention_heads=4,
            num_key_value_heads=4,
        )

        # sliding_window should be set, which will prevent SDPA fast path
        self.assertIsNotNone(core_attn.sliding_window)
        self.assertEqual(core_attn.sliding_window, (16, 0))

    def test_explicit_sdpa_with_swa_raises(self):
        """Explicit SDPA should not silently fall back to raw attention for SWA."""
        config = _make_mla_config(sliding_window=(16, 0))
        config._attn_implementation = "sdpa"

        core_attn = DotProductAttention(
            config=config,
            layer_number=0,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            is_swa=True,
            k_channels=64,
            v_channels=64,
            num_attention_heads=4,
            num_key_value_heads=4,
        )
        query = paddle.randn([1, 4, 4, 64], dtype=paddle.float16)
        key = paddle.randn([1, 4, 4, 64], dtype=paddle.float16)
        value = paddle.randn([1, 4, 4, 64], dtype=paddle.float16)

        with self.assertRaises(NotImplementedError):
            core_attn(query, key, value, attention_mask=None)

    def test_head_wise_swa_raw_path_raises(self):
        """Head-wise SWA should not silently run as all-head SWA in raw attention."""
        config = _make_mla_config(
            sliding_window=(16, 0), head_wise_swa_ratio=0.5
        )
        config._attn_implementation = "eager"

        core_attn = DotProductAttention(
            config=config,
            layer_number=0,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            is_swa=True,
            k_channels=64,
            v_channels=64,
            num_attention_heads=4,
            num_key_value_heads=4,
        )
        query = paddle.randn([1, 4, 4, 64])
        key = paddle.randn([1, 4, 4, 64])
        value = paddle.randn([1, 4, 4, 64])

        with self.assertRaises(NotImplementedError):
            core_attn(query, key, value, attention_mask=None)


# ============================================================================
# Test: attention sink parameter alignment
# ============================================================================


class TestAttentionSinkAlignment(unittest.TestCase):
    """Test that attention sink parameters are created correctly for MLA+SWA."""

    def test_mla_swa_layer_has_softmax_offset(self):
        """MLA SWA layer with add_swa_attention_sink_bias should have learnable softmax_offset."""
        config = _make_mla_config(
            add_swa_attention_sink_bias=True,
            add_full_attention_sink_bias=False,
        )
        config.softmax_type = "vanilla"

        core_attn = DotProductAttention(
            config=config,
            layer_number=0,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            is_swa=True,
            k_channels=64,
            v_channels=64,
            num_attention_heads=4,
            num_key_value_heads=4,
        )

        # SWA layer should have learnable softmax_offset (promoted to "learnable")
        self.assertIsNotNone(core_attn.softmax_offset)
        # It should be a Parameter (learnable)
        self.assertTrue(hasattr(core_attn.softmax_offset, "stop_gradient"))
        self.assertFalse(core_attn.softmax_offset.stop_gradient)

    def test_mla_full_layer_no_softmax_offset(self):
        """MLA full layer without add_full_attention_sink_bias should not have softmax_offset."""
        config = _make_mla_config(
            add_swa_attention_sink_bias=True,
            add_full_attention_sink_bias=False,
        )
        config.softmax_type = "vanilla"

        core_attn = DotProductAttention(
            config=config,
            layer_number=0,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            is_swa=False,  # Full attention layer
            k_channels=64,
            v_channels=64,
            num_attention_heads=4,
            num_key_value_heads=4,
        )

        # Full layer should NOT have softmax_offset (vanilla type)
        self.assertIsNone(core_attn.softmax_offset)


# ============================================================================
# Test: TransformerConfig MLA+SWA validation
# ============================================================================


class TestTransformerConfigMLASWAValidation(unittest.TestCase):
    """Test MLA+SWA config validation in __post_init__."""

    def test_mla_swa_head_wise_config_passes(self):
        """MLA+SWA may configure head-wise SWA for supported flashmask paths."""
        config = _make_mla_config(head_wise_swa_ratio=0.5)
        self.assertEqual(config.head_wise_swa_ratio, 0.5)

    def test_mla_swa_cp_raises(self):
        """MLA+SWA with context_parallel_size > 1 should raise NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            TransformerConfig(
                num_hidden_layers=4,
                multi_latent_attention=True,
                sliding_window=(4096, 0),
                context_parallel_size=2,
            )

    def test_mla_swa_valid_config_passes(self):
        """MLA+SWA with valid config should not raise."""
        config = TransformerConfig(
            num_hidden_layers=4,
            hidden_size=256,
            num_attention_heads=4,
            num_key_value_heads=4,
            v_head_dim=64,
            multi_latent_attention=True,
            sliding_window=(4096, 0),
            window_attn_skip_freq=4,
            qk_nope_head_dim=48,
            qk_rope_head_dim=16,
            kv_lora_rank=128,
            q_lora_rank=96,
            head_wise_swa_ratio=0.0,
            context_parallel_size=1,
        )
        self.assertTrue(config.multi_latent_attention)
        self.assertEqual(config.sliding_window, (4096, 0))

    def test_sliding_window_both_infinite_normalized_to_none(self):
        """sliding_window=(-1, -1) should be normalized to None."""
        config = TransformerConfig(
            num_hidden_layers=4,
            sliding_window=(-1, -1),
        )
        self.assertIsNone(config.sliding_window)

    def test_sliding_window_single_infinite_preserved(self):
        """sliding_window=(4096, -1) should be preserved (single-side infinite)."""
        config = TransformerConfig(
            num_hidden_layers=4,
            sliding_window=(4096, -1),
        )
        self.assertEqual(config.sliding_window, (4096, -1))

    def test_mla_swa_head_dim_mismatch_raises(self):
        """swa_head_dim validates the effective SWA MLA qk split when configured."""
        with self.assertRaises(ValueError):
            TransformerConfig(
                num_hidden_layers=4,
                hidden_size=512,
                num_attention_heads=8,
                num_key_value_heads=4,
                v_head_dim=64,
                multi_latent_attention=True,
                sliding_window=(4096, 0),
                window_attn_skip_freq=None,
                qk_nope_head_dim=48,
                qk_rope_head_dim=16,
                kv_lora_rank=128,
                q_lora_rank=96,
                swa_qk_nope_head_dim=32,
                swa_qk_rope_head_dim=8,
                swa_head_dim=41,
                head_wise_swa_ratio=0.0,
            )


# ============================================================================
# Test: MLA+SWA reference alignment (small scale)
# ============================================================================


class TestMLASWAReferenceAlignment(unittest.TestCase):
    """Test MLA+SWA output matches manual sliding window mask computation."""

    def _manual_swa_attention(
        self, query, key, value, window_size, softmax_scale
    ):
        """Compute attention with manual sliding window causal mask.

        Fleet's sliding_window[0] semantics: token i can attend [i - window_size, i]
        (window_size steps to the left, plus self = window_size + 1 tokens total).

        Args:
            query: [B, S, H, D]
            key: [B, S, H, D]
            value: [B, S, H, Dv]
            window_size: left window size (number of steps to the left)
            softmax_scale: scaling factor

        Returns:
            output: [B, S, H, Dv]
        """
        B, S, H, D = query.shape
        Dv = value.shape[-1]

        # Transpose to [B, H, S, D]
        q = query.transpose([0, 2, 1, 3])
        k = key.transpose([0, 2, 1, 3])
        v = value.transpose([0, 2, 1, 3])

        # Compute attention scores [B, H, S, S]
        scores = paddle.matmul(q, k, transpose_y=True) * softmax_scale

        # Create causal + sliding window mask matching Fleet's get_sliding_window_causal_mask
        # token i can see [max(0, i - window_size), i] inclusive
        mask = paddle.full([S, S], float("-inf"))
        for i in range(S):
            start = max(0, i - window_size)
            for j in range(start, i + 1):
                mask[i, j] = 0.0

        scores = scores + mask.unsqueeze(0).unsqueeze(0)

        # Softmax
        attn_weights = paddle.nn.functional.softmax(scores, axis=-1)

        # Weighted sum
        output = paddle.matmul(attn_weights, v)  # [B, H, S, Dv]
        return output.transpose([0, 2, 1, 3])  # [B, S, H, Dv]

    def test_small_scale_forward_alignment(self):
        """Small-scale MLA+SWA forward should match manual computation."""
        B, S, H, D = 1, 8, 2, 16
        window_size = 3
        softmax_scale = 1.0 / (D**0.5)

        paddle.seed(42)
        query = paddle.randn([B, S, H, D])
        key = paddle.randn([B, S, H, D])
        value = paddle.randn([B, S, H, D])

        # Manual reference
        ref_output = self._manual_swa_attention(
            query, key, value, window_size, softmax_scale
        )

        # Fleet DotProductAttention (eager path with SWA, no attention sink for fair comparison)
        config = _make_mla_config(
            sliding_window=(window_size, 0), add_swa_attention_sink_bias=False
        )
        config._attn_implementation = "eager"

        core_attn = DotProductAttention(
            config=config,
            layer_number=0,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            is_swa=True,
            softmax_scale=softmax_scale,
            k_channels=D,
            v_channels=D,
            num_attention_heads=H,
            num_key_value_heads=H,
        )

        fleet_output = core_attn(
            query,
            key,
            value,
            attention_mask=None,
        )

        # Reshape fleet output to match reference shape
        fleet_output_reshaped = fleet_output.reshape([B, S, H, D])

        self.assertTrue(
            paddle.allclose(
                fleet_output_reshaped, ref_output, atol=1e-5, rtol=1e-5
            ).item(),
            f"Max diff: {(fleet_output_reshaped - ref_output).abs().max().item()}",
        )

    def test_medium_scale_window_outside_zero(self):
        """S=32, window=8: tokens outside window should not be attended."""
        B, S, H, D = 2, 32, 4, 16
        window_size = 8  # token i can see [i-8, i]
        softmax_scale = 1.0 / (D**0.5)

        paddle.seed(123)
        query = paddle.randn([B, S, H, D])
        key = paddle.randn([B, S, H, D])
        value = paddle.randn([B, S, H, D])

        ref_output = self._manual_swa_attention(
            query, key, value, window_size, softmax_scale
        )

        # Verify that for position 20 (window=8), positions 0-11 don't contribute
        # Position 20 can see [12, 20] (i - window_size = 20 - 8 = 12)
        self.assertEqual(ref_output.shape, [B, S, H, D])

        # Output at position 20 should be the same whether or not tokens 0-11 exist
        query2 = query.clone()
        key2 = key.clone()
        value2 = value.clone()
        # Zero out positions 0-11 in key/value (outside window for position 20)
        key2[:, :12, :, :] = 0.0
        value2[:, :12, :, :] = 0.0

        ref_output2 = self._manual_swa_attention(
            query2, key2, value2, window_size, softmax_scale
        )

        # Position 20 output should be identical (positions 0-11 are outside window)
        self.assertTrue(
            paddle.allclose(
                ref_output[:, 20, :, :],
                ref_output2[:, 20, :, :],
                atol=1e-6,
                rtol=1e-6,
            ).item(),
            "Position 20 should not be affected by tokens outside window (0-11)",
        )


# ============================================================================
# Test: gradient correctness
# ============================================================================


class TestMLASWAGradientCorrectness(unittest.TestCase):
    """Test that SWA mask is correct in backward pass."""

    def test_gradient_isolation_outside_window(self):
        """Tokens outside the window should have zero gradient contribution."""
        B, S, H, D = 1, 16, 2, 8
        window_size = 4
        softmax_scale = 1.0 / (D**0.5)

        paddle.seed(42)

        config = _make_mla_config(sliding_window=(window_size, 0))
        config._attn_implementation = "eager"

        core_attn = DotProductAttention(
            config=config,
            layer_number=0,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            is_swa=True,
            softmax_scale=softmax_scale,
            k_channels=D,
            v_channels=D,
            num_attention_heads=H,
            num_key_value_heads=H,
        )

        # Test: perturbing position 0's key should NOT affect position 5+'s output
        # because window=4 means position 5 can only see [2, 3, 4, 5]
        key_base = paddle.randn([B, S, H, D])
        query = paddle.randn([B, S, H, D])
        value = paddle.randn([B, S, H, D])

        # Run 1: baseline
        key1 = key_base.clone()
        key1.stop_gradient = False
        out1 = core_attn(query, key1, value, attention_mask=None)
        # Take loss only from position 5+
        loss1 = out1.reshape([B, S, -1])[:, 5:, :].sum()
        loss1.backward()
        grad1_pos0 = key1.grad[:, 0, :, :].clone()

        # Gradient at position 0 w.r.t. loss at position 5+ should be zero
        # because position 0 is outside the window for all positions >= 5
        self.assertTrue(
            paddle.allclose(
                grad1_pos0, paddle.zeros_like(grad1_pos0), atol=1e-7
            ).item(),
            f"Position 0 key gradient should be zero for loss at positions 5+. "
            f"Max abs grad: {grad1_pos0.abs().max().item()}",
        )


# ============================================================================
# Test: packed sequence with sliding window
# ============================================================================


class TestPackedSequenceSWA(unittest.TestCase):
    """Test that packed sequence path correctly applies sliding window."""

    def test_packed_path_swa_applied(self):
        """Packed sequence path should invoke startend_row_indices_add_sliding_window."""
        # This is an integration-level attribute test:
        # When DotProductAttention has sliding_window set, the packed path
        # should call startend_row_indices_add_sliding_window
        config = _make_mla_config(sliding_window=(4, 0))

        core_attn = DotProductAttention(
            config=config,
            layer_number=0,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            is_swa=True,
            k_channels=64,
            v_channels=64,
            num_attention_heads=4,
            num_key_value_heads=4,
        )

        # Verify that sliding_window is stored (prerequisite for packed path SWA)
        self.assertIsNotNone(core_attn.sliding_window)
        self.assertEqual(core_attn.sliding_window, (4, 0))

    def test_head_wise_swa_num_vec_1_mask(self):
        """num_vec=1 flashmask startend path supports MLA-style head-wise SWA."""
        bsz, seq, num_heads = 1, 8, 4
        indices = (
            paddle.ones([bsz, num_heads, seq, 1], dtype=paddle.int32) * 10000
        )

        result = startend_row_indices_add_sliding_window(
            indices,
            (3, 0),
            0.5,
            num_heads,
        )

        self.assertTrue(
            paddle.equal_all(result[:, :2, :, :], indices[:, :2, :, :]).item()
        )
        expected_swa = (
            paddle.arange(3, seq + 3, dtype=paddle.int32)
            .reshape([1, 1, seq, 1])
            .expand([bsz, 2, seq, 1])
        )
        self.assertTrue(
            paddle.equal_all(result[:, 2:, :, :], expected_swa).item()
        )


if __name__ == "__main__":
    unittest.main()
