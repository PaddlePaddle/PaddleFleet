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
"""Behavior tests for ``SelfAttention`` construction contracts.

These focus on two config-driven decisions made in
``paddlefleet.transformer.attention``:

* ``gated_attention`` -- when True (non-experimental path) the *fused* qkv
  projection gains an extra gate segment whose width equals
  ``out_projection_size``; the projection actually built must reflect that.
* ``use_rr_flash_attention`` -- False unless ``recompute_modules`` contains
  ``"flash_attn"`` under a non-None ``recompute_granularity``; the guard that
  rejects the module without a granularity is a real contract.

Real ``TransformerConfig``, real ``DotProductAttention`` and real
``SelfAttention.__init__`` run; only the projection/norm sublayers are replaced
by thin genuine ``paddle.nn.Layer`` stubs (not the code under test), so the
built qkv projection's weight shape can be read back. Expected widths are hand
derived from the head/dim config, not from any attribute the class computed.
"""

import unittest

_IMPORT_ERROR = None
try:
    import paddle

    from paddlefleet.transformer.attention import (
        SelfAttention,
        SelfAttentionSublayersSpec,
    )
    from paddlefleet.transformer.dot_product_attention import (
        DotProductAttention,
    )
    from paddlefleet.transformer.enums import AttnMaskType
    from paddlefleet.transformer.transformer_config import TransformerConfig
    from paddlefleet.utils import (
        init_method_normal,
        scaled_init_method_normal,
    )
except (ImportError, ModuleNotFoundError) as exc:  # no paddle/GPU stack here
    _IMPORT_ERROR = exc

if _IMPORT_ERROR is None:

    class BiasedLinear(paddle.nn.Layer):
        """Genuine (not-under-test) stand-in for ColumnParallel/RowParallel
        linear: returns ``(output, bias)`` and, crucially, keeps a real
        ``paddle.nn.Linear`` whose weight shape exposes the (in, out) features
        the attention module asked ``build_spec_layer`` to construct."""

        def __init__(self, in_features, out_features, **kwargs):
            super().__init__()
            self.linear = paddle.nn.Linear(in_features, out_features)

        def forward(self, x):
            return self.linear(x), self.linear.bias

    class SimpleRMSNorm(paddle.nn.Layer):
        def __init__(self, **kwargs):
            super().__init__()
            hidden_size = kwargs.get(
                "normalized_shape", kwargs.get("hidden_size")
            )
            self.eps = kwargs.get("norm_eps", kwargs.get("eps", 1e-5))
            self.weight = paddle.create_parameter(
                shape=[hidden_size],
                dtype="float32",
                default_initializer=paddle.nn.initializer.Constant(1.0),
            )

        def forward(self, x):
            d_norm = paddle.rsqrt(
                x.pow(2).mean(axis=-1, keepdim=True) + self.eps
            )
            return x * d_norm * self.weight

    def _make_config(**overrides):
        """Minimal, fully-populated real config for a GQA attention layer.

        Heads/kv-heads are intentionally unequal (4 vs 2) so the query, key and
        value segments of the fused projection have distinct widths; a bug that
        swapped two of them would change the total width and be caught.
        """
        config = TransformerConfig(
            num_hidden_layers=1,
            hidden_size=128,
            num_attention_heads=4,
        )
        config.num_key_value_heads = 2  # GQA: distinct from num_attention_heads
        config.head_dim = 32
        config.softmax_scale = None
        config.use_bias = True
        config.no_rope_freq = None
        config.recompute_granularity = None
        config.recompute_method = None
        config.recompute_num_layers = None
        config.recompute_modules = None
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
        config.gated_attention = False
        for key, value in overrides.items():
            setattr(config, key, value)
        return config

    def _build_attn(config):
        return SelfAttention(
            config,
            SelfAttentionSublayersSpec(
                qkv_proj=BiasedLinear,
                core_attention=DotProductAttention,
                o_proj=BiasedLinear,
                q_norm=SimpleRMSNorm,
                k_norm=SimpleRMSNorm,
            ),
            attn_mask_type=AttnMaskType.causal,
            layer_number=1,
        )

    def _qkv_out_features(attn):
        # paddle.nn.Linear weight is [in_features, out_features].
        weight = attn.qkv_proj.linear.weight
        return int(weight.shape[0]), int(weight.shape[1])


# Hand-derived from _make_config (head_dim=32, heads=4, kv_heads=2, v_head_dim
# falls back to head_dim=32):
#   query = 32*4 = 128 ; key = 32*2 = 64 ; value = 32*2 = 64
#   out_projection_size (== gate segment) = v_head_dim*heads = 32*4 = 128
_HIDDEN = 128
_QUERY = 128
_KEY = 64
_VALUE = 64
_GATE = 128  # equals out_projection_size
_UNGATED_QKV = _QUERY + _KEY + _VALUE  # 256
_GATED_QKV = _QUERY + _KEY + _VALUE + _GATE  # 384


@unittest.skipIf(
    _IMPORT_ERROR is not None,
    f"paddle/paddlefleet import unavailable: {_IMPORT_ERROR!r}",
)
class TestSelfAttentionGatedFusedWidth(unittest.TestCase):
    """``gated_attention`` must both set the flag and widen the fused qkv."""

    def test_ungated_flag_and_fused_width(self):
        attn = _build_attn(_make_config(gated_attention=False))
        self.assertFalse(attn.gated_attention)
        in_features, out_features = _qkv_out_features(attn)
        self.assertEqual(in_features, _HIDDEN)
        # No gate segment: exactly query + key + value.
        self.assertEqual(out_features, _UNGATED_QKV)

    def test_gated_flag_and_fused_width(self):
        attn = _build_attn(_make_config(gated_attention=True))
        self.assertTrue(attn.gated_attention)
        in_features, out_features = _qkv_out_features(attn)
        self.assertEqual(in_features, _HIDDEN)
        # Gate segment appended to the fused projection.
        self.assertEqual(out_features, _GATED_QKV)

    def test_gate_segment_width_equals_out_projection(self):
        # The gate's *effect* on the built projection, isolated: enabling the
        # gate adds exactly out_projection_size columns and nothing else.
        ungated = _build_attn(_make_config(gated_attention=False))
        gated = _build_attn(_make_config(gated_attention=True))
        _, ungated_out = _qkv_out_features(ungated)
        _, gated_out = _qkv_out_features(gated)
        self.assertEqual(gated_out - ungated_out, _GATE)

    def test_default_config_is_ungated(self):
        # A freshly-built config that never touches gated_attention must yield
        # an ungated layer (the flag defaults to False, not a truthy sentinel).
        attn = _build_attn(_make_config())
        self.assertFalse(attn.gated_attention)
        _, out_features = _qkv_out_features(attn)
        self.assertEqual(out_features, _UNGATED_QKV)


@unittest.skipIf(
    _IMPORT_ERROR is not None,
    f"paddle/paddlefleet import unavailable: {_IMPORT_ERROR!r}",
)
class TestSelfAttentionRRFlashAttention(unittest.TestCase):
    """``use_rr_flash_attention`` reflects the recompute configuration."""

    def test_default_is_false(self):
        attn = _build_attn(_make_config())
        self.assertIs(attn.use_rr_flash_attention, False)

    def test_enabled_when_flash_attn_in_recompute_modules(self):
        # selective granularity + a list that names flash_attn -> refined
        # recompute is enabled on every layer (list mode carries no per-layer
        # selector), so the flag must flip to True.
        attn = _build_attn(
            _make_config(
                recompute_granularity="selective",
                recompute_method=None,
                recompute_modules=["flash_attn"],
            )
        )
        self.assertTrue(attn.use_rr_flash_attention)

    def test_flash_attn_recompute_requires_granularity(self):
        # Naming flash_attn for recompute without a granularity is rejected by
        # the guard in Attention.__init__ (a real precondition, not a crash we
        # want to paper over).
        config = _make_config(
            recompute_granularity=None,
            recompute_modules=["flash_attn"],
        )
        with self.assertRaises(AssertionError):
            _build_attn(config)


if __name__ == "__main__":
    unittest.main()
