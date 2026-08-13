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

"""Tests for Mixture-of-Heads (MoH) routing in CSAIndexer and TransformerConfig."""

import types
import unittest

import paddle
from paddle import nn
from paddle.distributed.fleet.meta_parallel import LayerSpec

from paddlefleet.transformer.csa_attention import (
    Compressor,
    CompressorSublayersSpec,
    CSAIndexer,
    CSAIndexerSublayersSpec,
)
from paddlefleet.transformer.transformer_config import TransformerConfig

# =========================================================================
# Helpers
# =========================================================================


class _Linear(nn.Layer):
    """Linear layer with weight shape [in_size, out_size] to match
    linear_bf16_fp32 which does x @ weight directly (no transpose)."""

    def __init__(self, in_size, out_size, **kwargs):
        super().__init__()
        self.weight = self.create_parameter(
            shape=[in_size, out_size],
            dtype="float32",
            default_initializer=nn.initializer.Normal(std=0.02),
        )

    def forward(self, x):
        return (
            paddle.matmul(x.cast("float32"), self.weight).cast(x.dtype),
            None,
        )


class _Norm(nn.Layer):
    def __init__(self, hidden_size=None, **kwargs):
        super().__init__()
        size = hidden_size or 1
        self.weight = self.create_parameter(
            shape=[size],
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.eps = 1e-5

    def forward(self, x, **kwargs):
        x_f32 = x.cast("float32")
        normed = (
            x_f32
            * paddle.rsqrt(x_f32.square().mean(-1, keepdim=True) + self.eps)
            * self.weight
        )
        return normed.cast(x.dtype)


def _make_indexer_config(
    hidden_size=256,
    pos_dim=16,
    index_n_heads=8,
    index_head_dim=32,
    index_topk=4,
    q_lora_rank=128,
    use_moh=False,
    num_activated_heads=None,
):
    """Minimal namespace config for constructing a bare CSAIndexer."""
    return types.SimpleNamespace(
        hidden_size=hidden_size,
        qk_pos_emb_head_dim=pos_dim,
        init_method=None,
        init_method_std=0.02,
        rms_norm_eps=1e-5,
        num_hidden_layers=1,
        use_fp8_qat=False,
        swa_high_precision_norm=False,
        use_fast_hadamard=False,
        high_precision_rope=False,
        q_lora_rank=q_lora_rank,
        dsa_index_n_heads=index_n_heads,
        dsa_index_head_dim=index_head_dim,
        dsa_index_topk=index_topk,
        use_moh=use_moh,
        num_activated_heads=num_activated_heads,
    )


def _build_indexer(config, compress_ratio=2, rotary_pos_emb=None):
    indexer_comp_spec = CompressorSublayersSpec(
        linear_wkv=_Linear,
        linear_wgate=_Linear,
        norm=_Norm,
    )
    indexer_spec = CSAIndexerSublayersSpec(
        linear_wq_b=_Linear,
        linear_weights_proj=_Linear,
        compressor=LayerSpec(Compressor, sublayers_spec=indexer_comp_spec),
    )
    return CSAIndexer(
        config=config,
        sublayers_spec=indexer_spec,
        compress_ratio=compress_ratio,
        rotary_pos_emb=rotary_pos_emb,
    )


# =========================================================================
# CSAIndexer MoH tests
# =========================================================================


class TestCSAIndexerMoHConstruction(unittest.TestCase):
    """MoH buffers exist iff ``config.use_moh`` and match ``dsa_index_n_heads``."""

    def test_disabled_by_default(self):
        cfg = _make_indexer_config()
        indexer = _build_indexer(cfg)
        self.assertFalse(indexer.use_moh)
        # No MoH buffers registered.
        buffer_names = {name for name, _ in indexer.named_buffers()}
        self.assertNotIn("indexer_moh_bias", buffer_names)
        self.assertNotIn("local_tokens_per_indexer_moh", buffer_names)

    def test_enabled_registers_buffers_with_head_shape(self):
        cfg = _make_indexer_config(
            index_n_heads=64, num_activated_heads=8, use_moh=True
        )
        indexer = _build_indexer(cfg)
        self.assertTrue(indexer.use_moh)
        self.assertEqual(indexer.num_activated_heads, 8)
        # Both buffers shaped by index_n_heads.
        self.assertEqual(list(indexer.indexer_moh_bias.shape), [64])
        self.assertEqual(indexer.indexer_moh_bias.dtype, paddle.float32)
        self.assertEqual(list(indexer.local_tokens_per_indexer_moh.shape), [64])
        # Only ``indexer_moh_bias`` is persistable -- the counter is a
        # non-persistent trainer scratch buffer.
        state_dict_keys = set(indexer.state_dict().keys())
        self.assertIn("indexer_moh_bias", state_dict_keys)
        self.assertNotIn("local_tokens_per_indexer_moh", state_dict_keys)


class TestCSAIndexerMoHRouting(unittest.TestCase):
    """MoH selects exactly ``num_activated_heads`` heads per token."""

    def setUp(self):
        paddle.seed(42)
        # index_n_heads=32 exceeds the tilelang H>=16 pad path and keeps the
        # top-k routing meaningful (activate half of the heads).
        self.b = 2
        self.sq = 8
        self.index_n_heads = 64
        self.num_activated_heads = 8
        self.cfg = _make_indexer_config(
            index_n_heads=self.index_n_heads,
            index_head_dim=32,
            num_activated_heads=self.num_activated_heads,
            use_moh=True,
        )
        self.indexer = _build_indexer(self.cfg, compress_ratio=2)

    def _inputs(self):
        x = paddle.randn([self.b, self.sq, self.cfg.hidden_size]).astype(
            "bfloat16"
        )
        qr = paddle.randn([self.b, self.sq, self.cfg.q_lora_rank]).astype(
            "bfloat16"
        )
        return x, qr

    def test_output_shapes_use_activated_head_count(self):
        self.indexer.eval()
        x, qr = self._inputs()
        q, k, weights = self.indexer.forward_before_topk(x, qr)
        # Head axis of q/weights is narrowed to ``num_activated_heads``. K is
        # shared across query heads, so its shape is untouched.
        head_pad = max(self.num_activated_heads, 16)
        self.assertEqual(q.shape[2], head_pad)
        self.assertEqual(weights.shape[-1], head_pad)
        self.assertEqual(q.shape[:2], [self.b, self.sq])
        self.assertEqual(q.shape[3], self.cfg.dsa_index_head_dim)
        self.assertEqual(len(k.shape), 3)

    def test_training_updates_token_counter(self):
        self.indexer.train()
        x, qr = self._inputs()
        self.indexer.forward_before_topk(x, qr)
        # Each token activates exactly ``num_activated_heads`` heads, so the
        # total counted mass is ``b * sq * num_activated_heads`` regardless of
        # how it is distributed across heads.
        total = self.indexer.local_tokens_per_indexer_moh.sum().item()
        self.assertEqual(
            int(total), self.b * self.sq * self.num_activated_heads
        )
        # Per-head counts cannot exceed the number of tokens.
        max_per_head = self.indexer.local_tokens_per_indexer_moh.max().item()
        self.assertLessEqual(int(max_per_head), self.b * self.sq)

    def test_eval_does_not_update_counter(self):
        self.indexer.eval()
        x, qr = self._inputs()
        self.indexer.forward_before_topk(x, qr)
        self.assertEqual(
            self.indexer.local_tokens_per_indexer_moh.sum().item(), 0.0
        )

    def test_bias_shifts_selection(self):
        """A large positive bias on one head forces every token to activate it."""
        self.indexer.eval()
        x, qr = self._inputs()

        # Baseline: which heads did head 0 receive without bias? Reset the
        # counter (training path) then measure via a training-mode pass.
        self.indexer.train()
        self.indexer.local_tokens_per_indexer_moh.zero_()
        self.indexer.forward_before_topk(x, qr)
        base_head0 = int(self.indexer.local_tokens_per_indexer_moh[0].item())

        # Force head 0 selected for every token by pumping its bias.
        self.indexer.local_tokens_per_indexer_moh.zero_()
        self.indexer.indexer_moh_bias[:] = 0.0
        self.indexer.indexer_moh_bias[0] = 1e9
        self.indexer.forward_before_topk(x, qr)
        forced_head0 = int(self.indexer.local_tokens_per_indexer_moh[0].item())

        self.assertEqual(forced_head0, self.b * self.sq)
        self.assertGreaterEqual(forced_head0, base_head0)


# =========================================================================
# TransformerConfig MoH validation tests
# =========================================================================


class TestTransformerConfigMoH(unittest.TestCase):
    """``use_moh`` / ``num_activated_heads`` plumbing and __post_init__ checks."""

    @staticmethod
    def _config(**overrides):
        kwargs = {
            "num_hidden_layers": 2,
            "hidden_size": 256,
            "num_attention_heads": 8,
            "multi_latent_attention": True,
            "experimental_attention_variant": "dsv4_hybrid",
            "csa_compress_ratios": [0, 4],
            "csa_window_size": 16,
            "q_lora_rank": 64,
            "kv_lora_rank": 16,
            "qk_nope_head_dim": 16,
            "qk_rope_head_dim": 16,
            "qk_pos_emb_head_dim": 16,
            "v_head_dim": 32,
            "dsa_index_n_heads": 64,
            "dsa_index_head_dim": 32,
            "dsa_index_topk": 8,
            "csa_indexer_backend": "unfused",
            "csa_sparse_attn_backend": "unfused",
        }
        kwargs.update(overrides)
        return TransformerConfig(**kwargs)

    def test_defaults_are_off(self):
        config = self._config()
        self.assertFalse(config.use_moh)
        self.assertIsNone(config.num_activated_heads)

    def test_valid_moh_config_is_accepted(self):
        config = self._config(use_moh=True, num_activated_heads=8)
        self.assertTrue(config.use_moh)
        self.assertEqual(config.num_activated_heads, 8)

    def test_hf_field_names_map_through_transform_rules(self):
        rules = TransformerConfig.transform_rules
        self.assertEqual(rules["use_moh"], "use_moh")
        self.assertEqual(rules["num_activated_heads"], "num_activated_heads")

    def test_moh_without_num_activated_heads_raises(self):
        with self.assertRaisesRegex(ValueError, "num_activated_heads"):
            self._config(use_moh=True)

    def test_num_activated_heads_without_moh_raises(self):
        # A stray num_activated_heads is read by nothing, so a typo'd switch
        # must not look like it worked.
        with self.assertRaisesRegex(ValueError, "without use_moh=True"):
            self._config(num_activated_heads=8)

    def test_non_positive_num_activated_heads_raises(self):
        for bad in (0, -1):
            with (
                self.subTest(num_activated_heads=bad),
                self.assertRaisesRegex(ValueError, "positive"),
            ):
                self._config(use_moh=True, num_activated_heads=bad)

    def test_bool_num_activated_heads_raises(self):
        # ``True`` would otherwise sneak through as 1.
        with self.assertRaisesRegex(ValueError, "positive"):
            self._config(use_moh=True, num_activated_heads=True)

    def test_num_activated_heads_over_index_heads_raises(self):
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            self._config(
                use_moh=True, dsa_index_n_heads=8, num_activated_heads=9
            )

    def test_num_activated_heads_equal_index_heads_warns(self):
        # Legal but pointless: every head is activated, so the routing is a
        # no-op that still pays for the top-k and gather.
        with self.assertLogs(
            "paddlefleet.transformer.transformer_config", level="WARNING"
        ) as captured:
            config = self._config(
                use_moh=True, dsa_index_n_heads=8, num_activated_heads=8
            )
        self.assertEqual(config.num_activated_heads, 8)
        self.assertTrue(
            any("num_activated_heads" in line for line in captured.output)
        )

    def test_moh_without_index_heads_raises(self):
        with self.assertRaisesRegex(ValueError, "dsa_index_n_heads"):
            self._config(
                use_moh=True,
                dsa_index_n_heads=None,
                num_activated_heads=4,
            )

    def test_moh_outside_dsv4_hybrid_raises(self):
        # Only the dsv4-hybrid stack builds a CSAIndexer, so MoH is inert
        # anywhere else.
        with self.assertRaisesRegex(ValueError, "dsv4_hybrid"):
            self._config(
                experimental_attention_variant=None,
                csa_compress_ratios=None,
                use_moh=True,
                num_activated_heads=4,
            )

    def test_moh_with_dense_mode_raises(self):
        # csa_dense_mode drops the CSAIndexer, so there is no head to route.
        with self.assertRaisesRegex(ValueError, "CSAIndexer"):
            self._config(
                csa_dense_mode=True, use_moh=True, num_activated_heads=4
            )

    def test_moh_without_csa_layer_raises(self):
        # ratios of only 0 (window) / 128 (HCA) build no CSAIndexer either.
        with self.assertRaisesRegex(ValueError, "CSAIndexer"):
            self._config(
                csa_compress_ratios=[0, 128],
                use_moh=True,
                num_activated_heads=4,
            )


if __name__ == "__main__":
    unittest.main()
