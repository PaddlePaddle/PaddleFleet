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

"""HySparse inference (native KV cache) adaptation.

Covers the three pieces that make prefill + incremental decode work for a
HySparse stack:

* :class:`DynamicKVCache` gained a cross-layer ``shared_k`` slot so a full layer
  can publish its compressed KV latent across decode steps.
* The full (MLA) layer seeds its own KV cache during the block-score prefill and
  routes decode through ``core_attention``.
* The SWA (MQA) layer caches its own absorbed K/V, maps the sliding window into
  cache-local coordinates at decode, and runs window-only (the block-sparse
  branch needs whole-row logits that decode cannot produce).
"""

import os
import unittest

os.environ["FLAGS_cudnn_deterministic"] = "True"

import numpy as np
import paddle
from paddle.distributed.fleet.meta_parallel import LayerSpec

from paddlefleet.fusions.fused_bias_dropout import get_bias_dropout_add
from paddlefleet.generation.greedy_generator import DynamicKVCache
from paddlefleet.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddlefleet.transformer.dot_product_attention import DotProductAttention
from paddlefleet.transformer.enums import AttnMaskType
from paddlefleet.transformer.multi_latent_attention import (
    MLASelfAttentionSublayersSpec,
    MQASelfAttention,
)
from paddlefleet.transformer.paddle_norm import WrappedPaddleNorm
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.transformer.transformer_layer import (
    HySparseTransformerLayer,
    TransformerLayerSublayersSpec,
)

WINDOW = 128
BLOCK_B = 64


def _cuda_or_skip(testcase):
    if not paddle.device.is_compiled_with_cuda():
        testcase.skipTest("CUDA build of Paddle is required")
    if paddle.device.cuda.device_count() == 0:
        testcase.skipTest("No CUDA device available")


def _hysparse_backend_or_skip(testcase):
    """Skip unless the HySparse backends (FA4 FlashMask + cuDNN DSA) can run."""
    _cuda_or_skip(testcase)
    try:
        import paddlefleet_ops

        if not paddlefleet_ops.is_flash_mask_available():
            testcase.skipTest("FlashMask (FA4) backend not available")
        from paddlefleet.cudnn_ops import is_dsa_available

        if not is_dsa_available():
            testcase.skipTest("cuDNN DSA backend not available")
    except (ImportError, RuntimeError):
        testcase.skipTest("HySparse FA4/DSA backend import failed")


class TestDynamicKVCacheSharedSlot(unittest.TestCase):
    """The cross-layer ``shared_k`` slot and the decode/prefill discriminator."""

    def test_update_shared_accumulates_along_seq(self):
        cache = DynamicKVCache(num_layers=2)
        prefill = paddle.randn([2, 8, 1, 16])
        full = cache.update_shared(prefill, 0)
        self.assertEqual(full.shape, [2, 8, 1, 16])

        step = paddle.randn([2, 1, 1, 16])
        full = cache.update_shared(step, 0)
        self.assertEqual(full.shape, [2, 9, 1, 16])
        self.assertEqual(cache.get_shared(0).shape, [2, 9, 1, 16])
        # Untouched layers keep an empty slot.
        self.assertIsNone(cache.get_shared(1))
        # The shared latent is never window-truncated: the block-sparse branch
        # gathers blocks from the whole document.
        np.testing.assert_allclose(
            cache.get_shared(0)[:, :8].numpy(), prefill.numpy()
        )

    def test_has_layer_cache_marks_decode(self):
        cache = DynamicKVCache(
            num_layers=1, swa_layers=[True], window_size=WINDOW
        )
        self.assertFalse(cache.has_layer_cache(0))
        cache.update(
            paddle.randn([1, 4, 8]), paddle.randn([1, 4, 8]), layer_idx=0
        )
        self.assertTrue(cache.has_layer_cache(0))

    def test_reset_clears_shared_slot(self):
        cache = DynamicKVCache(num_layers=1)
        cache.update_shared(paddle.randn([1, 4, 1, 8]), 0)
        cache.update(
            paddle.randn([1, 4, 8]), paddle.randn([1, 4, 8]), layer_idx=0
        )
        cache.reset()
        self.assertIsNone(cache.get_shared(0))
        self.assertFalse(cache.has_layer_cache(0))
        self.assertEqual(cache.get_seq_len(0), 0)


class TestSWADecodeWindowMapping(unittest.TestCase):
    """Decode over the truncated SWA cache == full-sequence windowed attention.

    The SWA layer keeps only the trailing ``window_size`` absorbed K/V entries,
    so at decode the sliding window has to be expressed in *cache-local* columns
    (``[kv_s - window_size, kv_s)``) instead of absolute positions. This checks
    that mapping against the same kernel run over the whole sequence.
    """

    def test_decode_window_matches_full_sequence(self):
        _cuda_or_skip(self)
        from paddlefleet.tilelang_ops.hysparse import (
            sliding_window_mqa_attention,
        )
        from paddlefleet.transformer.multi_latent_attention import (
            build_hysparse_valid_range,
        )

        paddle.seed(2026)
        b, h, dk, dv = 2, 4, 576, 512
        prompt_len = 320
        total = prompt_len + 1
        sm_scale = dk**-0.5

        q = paddle.randn([b, total, h, dk], dtype="bfloat16")
        k = paddle.randn([b, total, dk], dtype="bfloat16")
        v = paddle.randn([b, total, dv], dtype="bfloat16")

        # Reference: the whole sequence in one pass; query rows are independent
        # so the last row is exactly what a decode step must reproduce.
        valid_range = build_hysparse_valid_range(
            None, total, b, window_size=WINDOW
        )
        ref, _ = sliding_window_mqa_attention(
            q, k, v, valid_range, sm_scale=sm_scale, block_B=BLOCK_B
        )

        # Cache path: prefill, then one decode step over the truncated cache.
        cache = DynamicKVCache(
            num_layers=1, swa_layers=[True], window_size=WINDOW
        )
        cache.update(k[:, :prompt_len], v[:, :prompt_len], 0)
        cached_k, cached_v = cache.update(
            k[:, prompt_len:], v[:, prompt_len:], 0
        )
        kv_s = cached_k.shape[1]
        # window_size entries kept after prefill + the new token.
        self.assertEqual(kv_s, WINDOW + 1)

        bos = max(0, kv_s - WINDOW)
        decode_valid_range = paddle.concat(
            [
                paddle.full([b, 1, 1], bos, dtype="int32"),
                paddle.full([b, 1, 1], kv_s, dtype="int32"),
            ],
            axis=-1,
        )
        out, _ = sliding_window_mqa_attention(
            q[:, -1:].contiguous(),
            cached_k.contiguous(),
            cached_v.contiguous(),
            decode_valid_range,
            sm_scale=sm_scale,
            block_B=BLOCK_B,
        )

        np.testing.assert_allclose(
            out[:, 0].astype("float32").numpy(),
            ref[:, -1].astype("float32").numpy(),
            atol=2e-2,
            rtol=2e-2,
        )


class TestHySparseLayerPrefillDecode(unittest.TestCase):
    """A full + SWA layer pair through prefill and incremental decode."""

    BATCH = 2
    PROMPT = 384

    @classmethod
    def setUpClass(cls):
        paddle.seed(2026)
        model_parallel_cuda_manual_seed(2026)
        cls.config = TransformerConfig(
            hidden_size=1536,
            head_dim=128,
            num_attention_heads=4,
            num_key_value_heads=4,
            gated_attention=True,
            qk_rope_head_dim=64,
            qk_nope_head_dim=192,
            v_head_dim=256,
            kv_lora_rank=192,
            rope_theta=5000000,
            use_qk_norm=True,
            multi_latent_attention=True,
            rope_type="rope",
            add_swa_attention_sink_bias=False,
            sliding_window=[WINDOW, WINDOW],
            window_attn_skip_freq=2,
            enable_hy_sparse_attention=True,
            hy_sparse_block_size=BLOCK_B,
        )
        cls.sublayer_spec = MLASelfAttentionSublayersSpec(
            core_attention=DotProductAttention,
            o_proj=RowParallelLinear,
            gate_proj=ColumnParallelLinear,
            q_a_proj=ColumnParallelLinear,
            q_b_proj=ColumnParallelLinear,
            kv_a_proj_with_mqa=ColumnParallelLinear,
            kv_b_proj=ColumnParallelLinear,
            q_a_layernorm=WrappedPaddleNorm,
            kv_a_layernorm=WrappedPaddleNorm,
        )

    def _build_layers(self):
        layer_spec = TransformerLayerSublayersSpec(
            self_attn=LayerSpec(
                layer=MQASelfAttention,
                sublayers_spec=self.sublayer_spec,
            ),
            self_attn_bda=get_bias_dropout_add,
        )
        layers = []
        # layer 0 -> full attention (produces shared KV + block indices),
        # layer 1 -> SWA / MQA (consumes them).
        for layer_number in (0, 1):
            layer = HySparseTransformerLayer(
                self.config, layer_spec, layer_number=layer_number
            )
            layer.self_attn.attn_mask_type = AttnMaskType.causal
            layer = paddle.amp.decorate(layer, level="O2", dtype="bfloat16")
            # eval() matters: training mode ignores position_ids when building
            # RoPE, which would place every decode token at position 0.
            layer.eval()
            layers.append(layer)
        self.assertFalse(layers[0].self_attn.is_swa)
        self.assertTrue(layers[1].self_attn.is_mqa)
        return layers

    @paddle.no_grad()
    def test_prefill_then_decode(self):
        _hysparse_backend_or_skip(self)
        full_layer, swa_layer = self._build_layers()
        b, prompt_len = self.BATCH, self.PROMPT
        hidden_size = self.config.hidden_size
        latent_dim = self.config.kv_lora_rank + self.config.qk_rope_head_dim

        cache = DynamicKVCache(
            num_layers=2, swa_layers=[False, True], window_size=WINDOW
        )

        # ---- Prefill ----
        dict_args = {
            "hidden_states": paddle.randn(
                [b, prompt_len, hidden_size], dtype="bfloat16"
            ),
            "attn_mask_startend_row_indices": paddle.full(
                [b, 1, prompt_len, 1], prompt_len, dtype="int32"
            ),
            "position_ids": paddle.arange(prompt_len, dtype="int64")
            .unsqueeze(0)
            .expand([b, prompt_len]),
            "past_key_values": cache,
            "use_cache": True,
        }
        dict_args = full_layer(dict_args)
        self.assertIsNotNone(dict_args["shared_block_indices"])
        dict_args = swa_layer(dict_args)

        self.assertEqual(cache.get_seq_len(0), prompt_len)
        self.assertEqual(cache.get_seq_len(1), prompt_len)
        # Full layer: own multi-head KV kept in full, shared latent published.
        self.assertEqual(cache.k[0].shape[1], prompt_len)
        self.assertEqual(
            cache.get_shared(0).shape, [b, prompt_len, 1, latent_dim]
        )
        self.assertIsNone(cache.get_shared(1))
        # SWA layer: own absorbed K/V truncated to the window.
        self.assertEqual(cache.k[1].shape, [b, WINDOW, latent_dim])
        self.assertEqual(
            cache.v[1].shape, [b, WINDOW, self.config.kv_lora_rank]
        )

        # ---- Decode ----
        for step in range(3):
            dict_args = {
                "hidden_states": paddle.randn(
                    [b, 1, hidden_size], dtype="bfloat16"
                ),
                "position_ids": paddle.full(
                    [b, 1], prompt_len + step, dtype="int64"
                ),
                "past_key_values": cache,
                "use_cache": True,
            }
            dict_args = full_layer(dict_args)
            # Decode cannot score blocks one token at a time -> window-only SWA.
            self.assertIsNone(dict_args["shared_block_indices"])
            dict_args = swa_layer(dict_args)

            self.assertEqual(
                dict_args["hidden_states"].shape, [b, 1, hidden_size]
            )
            self.assertTrue(
                paddle.isfinite(
                    dict_args["hidden_states"].astype("float32")
                ).all()
            )
            expected_len = prompt_len + step + 1
            self.assertEqual(cache.get_seq_len(0), expected_len)
            self.assertEqual(cache.get_seq_len(1), expected_len)
            self.assertEqual(cache.k[0].shape[1], expected_len)
            self.assertEqual(cache.k[1].shape[1], WINDOW)
            self.assertEqual(
                cache.get_shared(0).shape,
                [b, expected_len, 1, latent_dim],
            )

    @paddle.no_grad()
    def test_full_layer_decode_matches_no_cache(self):
        """The full layer's cached decode step reproduces a cache-less rerun.

        Prefill scores blocks through the FA4 dense kernel while decode attends
        over the cached KV with SDPA; both are the same dense causal attention,
        so the last position must agree.
        """
        _hysparse_backend_or_skip(self)
        full_layer, _ = self._build_layers()
        b, prompt_len = self.BATCH, self.PROMPT
        hidden_size = self.config.hidden_size

        prompt = paddle.randn([b, prompt_len, hidden_size], dtype="bfloat16")
        next_token = paddle.randn([b, 1, hidden_size], dtype="bfloat16")
        total = prompt_len + 1
        full_hidden = paddle.concat([prompt, next_token], axis=1)

        def run_prefill(hidden, cache):
            seq_len = hidden.shape[1]
            return full_layer(
                {
                    "hidden_states": hidden,
                    "attn_mask_startend_row_indices": paddle.full(
                        [b, 1, seq_len, 1], seq_len, dtype="int32"
                    ),
                    "position_ids": paddle.arange(seq_len, dtype="int64")
                    .unsqueeze(0)
                    .expand([b, seq_len]),
                    "past_key_values": cache,
                    "use_cache": cache is not None,
                }
            )

        # Reference: one dense pass over prompt + next token.
        ref = run_prefill(full_hidden, None)["hidden_states"][:, -1]

        cache = DynamicKVCache(
            num_layers=2, swa_layers=[False, True], window_size=WINDOW
        )
        run_prefill(prompt, cache)
        decoded = full_layer(
            {
                "hidden_states": next_token,
                "position_ids": paddle.full([b, 1], prompt_len, dtype="int64"),
                "past_key_values": cache,
                "use_cache": True,
            }
        )["hidden_states"][:, 0]

        self.assertEqual(cache.get_seq_len(0), total)
        np.testing.assert_allclose(
            decoded.astype("float32").numpy(),
            ref.astype("float32").numpy(),
            atol=3e-2,
            rtol=3e-2,
        )


if __name__ == "__main__":
    unittest.main()
