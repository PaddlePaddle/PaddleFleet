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

"""
Unit tests for generation module components.
This test file imports only the necessary components without full paddlefleet.
"""

import os
import sys

# Add src to path for direct import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import unittest
from types import SimpleNamespace

import paddle

from paddlefleet.generation.csa_cache import CSADynamicCache
from paddlefleet.generation.greedy_generator import (
    DynamicKVCache,
    GreedyGenerator,
    _uses_dsv4_hybrid_attention,
)


class TestDynamicKVCache(unittest.TestCase):
    """Test cases for DynamicKVCache."""

    def test_initialization(self):
        """Test cache initialization."""
        cache = DynamicKVCache(num_layers=4)

        self.assertEqual(len(cache.k), 4)
        self.assertEqual(len(cache.v), 4)

        for i in range(4):
            self.assertIsNone(cache.k[i])
            self.assertIsNone(cache.v[i])

    def test_basic_update(self):
        """Test basic KV cache update."""
        cache = DynamicKVCache(num_layers=2)

        # First update
        k1 = paddle.randn([1, 4, 8, 64], dtype="bfloat16")
        v1 = paddle.randn([1, 4, 8, 64], dtype="bfloat16")
        returned_k, returned_v = cache.update(k1, v1, 0)

        self.assertIsNotNone(returned_k)
        self.assertIsNotNone(returned_v)

        # Should be the same as input (first update)
        self.assertTrue(
            paddle.allclose(returned_k.cast("float32"), k1.cast("float32"))
        )
        self.assertTrue(
            paddle.allclose(returned_v.cast("float32"), v1.cast("float32"))
        )

    def test_second_update_concat(self):
        """Test that second update concatenates."""
        cache = DynamicKVCache(num_layers=2)

        # First update
        k1 = paddle.randn([1, 4, 8, 64], dtype="bfloat16")
        v1 = paddle.randn([1, 4, 8, 64], dtype="bfloat16")
        cache.update(k1, v1, 0)

        # Second update (different length)
        k2 = paddle.randn([1, 2, 8, 64], dtype="bfloat16")
        v2 = paddle.randn([1, 2, 8, 64], dtype="bfloat16")
        returned_k, returned_v = cache.update(k2, v2, 0)

        # Should be concatenated
        self.assertEqual(returned_k.shape[1], 6)  # 4 + 2
        self.assertEqual(returned_v.shape[1], 6)

    def test_get_seq_len(self):
        """Test get_seq_len method.

        Note: get_seq_len has a fallback that returns the first non-empty layer's
        sequence length when the requested layer is empty. This is by design since
        all layers should have the same sequence length during inference.
        """
        cache = DynamicKVCache(num_layers=2)

        self.assertEqual(cache.get_seq_len(0), 0)
        self.assertEqual(cache.get_seq_len(1), 0)

        # Update layer 0
        k = paddle.randn([1, 5, 8, 64], dtype="bfloat16")
        v = paddle.randn([1, 5, 8, 64], dtype="bfloat16")
        cache.update(k, v, 0)

        # Layer 0 has 5 tokens
        self.assertEqual(cache.get_seq_len(0), 5)
        # Layer 1 is empty, but fallback returns layer 0's length (by design)
        self.assertEqual(cache.get_seq_len(1), 5)

    def test_reset(self):
        """Test reset functionality."""
        cache = DynamicKVCache(num_layers=3)

        # Update a layer
        k = paddle.randn([1, 3, 8, 64], dtype="bfloat16")
        v = paddle.randn([1, 3, 8, 64], dtype="bfloat16")
        cache.update(k, v, 0)

        # Reset
        cache.reset()

        # All layers should be None
        for i in range(3):
            self.assertIsNone(cache.k[i])
            self.assertIsNone(cache.v[i])

    def test_multiple_layers(self):
        """Test that different layers have independent caches.

        Note: get_seq_len has a fallback that returns the first non-empty layer's
        sequence length when the requested layer is empty.
        """
        cache = DynamicKVCache(num_layers=4)

        # Update layer 0
        k0 = paddle.randn([1, 3, 8, 64], dtype="bfloat16")
        v0 = paddle.randn([1, 3, 8, 64], dtype="bfloat16")
        cache.update(k0, v0, 0)

        # Update layer 2
        k2 = paddle.randn([1, 5, 8, 64], dtype="bfloat16")
        v2 = paddle.randn([1, 5, 8, 64], dtype="bfloat16")
        cache.update(k2, v2, 2)

        # Check layer-specific cache lengths
        self.assertEqual(cache.get_seq_len(0), 3)  # Layer 0 has 3 tokens
        self.assertEqual(cache.get_seq_len(2), 5)  # Layer 2 has 5 tokens
        # Layers 1 and 3 are empty, fallback returns first non-empty (layer 0 = 3)
        self.assertEqual(cache.get_seq_len(1), 3)
        self.assertEqual(cache.get_seq_len(3), 3)


class TestCSADynamicCacheProtocol(unittest.TestCase):
    """CSADynamicCache must also satisfy the standard ``update`` protocol.

    A model that interleaves CSA/HCA layers with standard-attention layers
    shares one cache object; the standard layers still call
    ``update(k, v, layer_idx)`` / ``get_seq_len``.
    """

    def test_standard_update_and_seq_len(self):
        cache = CSADynamicCache(num_layers=3)
        self.assertEqual(cache.get_seq_len(0), 0)

        k1 = paddle.randn([1, 4, 8, 64], dtype="bfloat16")
        v1 = paddle.randn([1, 4, 8, 64], dtype="bfloat16")
        k, v = cache.update(k1, v1, 0)
        self.assertEqual(k.shape[1], 4)
        self.assertEqual(cache.get_seq_len(0), 4)

        k2 = paddle.randn([1, 2, 8, 64], dtype="bfloat16")
        v2 = paddle.randn([1, 2, 8, 64], dtype="bfloat16")
        k, v = cache.update(k2, v2, 0)
        self.assertEqual(k.shape[1], 6)
        self.assertEqual(v.shape[1], 6)

    def test_csa_state_is_per_layer(self):
        cache = CSADynamicCache(num_layers=4)
        s0 = cache.get_csa_state(0)
        s1 = cache.get_csa_state(1)
        self.assertIsNot(s0, s1)
        self.assertEqual(s0.raw_seq_len(), 0)
        self.assertEqual(s0.n_compressed, 0)

    def test_reset(self):
        cache = CSADynamicCache(num_layers=2)
        k = paddle.randn([1, 3, 8, 64], dtype="bfloat16")
        cache.update(k, k, 0)
        cache.get_csa_state(1).append_raw(
            paddle.randn([1, 1, 32], dtype="bfloat16")
        )
        cache.reset()
        self.assertEqual(cache.get_seq_len(0), 0)
        self.assertEqual(cache.get_csa_state(1).raw_seq_len(), 0)


class TestGeneratorCacheSelection(unittest.TestCase):
    """GreedyGenerator picks the cache class from the model config."""

    @staticmethod
    def _fake_model(cfg_kwargs):
        base = {
            "num_hidden_layers": 4,
            "sequence_parallel": False,
            "apply_rope_fusion": False,
            "recompute_granularity": None,
            "num_empty_layers_add_in_head": 0,
            "num_empty_layers_add_in_tail": 0,
        }
        base.update(cfg_kwargs)
        cfg = SimpleNamespace(**base)
        return SimpleNamespace(config=cfg)

    def test_detects_dsv4_variant(self):
        cfg = SimpleNamespace(
            experimental_attention_variant="dsv4_hybrid",
            csa_compress_ratios=None,
        )
        self.assertTrue(_uses_dsv4_hybrid_attention(cfg))

    def test_detects_via_compress_ratios(self):
        cfg = SimpleNamespace(
            experimental_attention_variant=None,
            csa_compress_ratios=[0, 4, 128, 4],
        )
        self.assertTrue(_uses_dsv4_hybrid_attention(cfg))

    def test_standard_config_not_detected(self):
        cfg = SimpleNamespace(
            experimental_attention_variant=None,
            csa_compress_ratios=None,
        )
        self.assertFalse(_uses_dsv4_hybrid_attention(cfg))

    def test_generator_selects_csa_cache(self):
        model = self._fake_model(
            {
                "experimental_attention_variant": "dsv4_hybrid",
                "csa_compress_ratios": [0, 4, 128, 4],
            }
        )
        gen = GreedyGenerator(model)
        self.assertIsInstance(gen.cache, CSADynamicCache)

    def test_generator_selects_standard_cache(self):
        model = self._fake_model(
            {
                "experimental_attention_variant": None,
                "csa_compress_ratios": None,
            }
        )
        gen = GreedyGenerator(model)
        self.assertIsInstance(gen.cache, DynamicKVCache)


class TestSWACacheInit(unittest.TestCase):
    """Test SWA layer detection and DynamicKVCache initialization in GreedyGenerator."""

    def _make_generator_with_cfg(
        self,
        num_hidden_layers=4,
        sliding_window=None,
        window_attn_skip_freq=None,
        num_empty_layers_add_in_head=0,
        num_empty_layers_add_in_tail=0,
    ):
        from unittest.mock import MagicMock

        from paddlefleet.generation.greedy_generator import GreedyGenerator

        model = MagicMock()
        cfg = MagicMock()
        cfg.num_hidden_layers = num_hidden_layers
        cfg.sequence_parallel = False
        cfg.apply_rope_fusion = False
        cfg.recompute_granularity = None
        cfg.sliding_window = sliding_window
        cfg.window_attn_skip_freq = window_attn_skip_freq
        cfg.num_empty_layers_add_in_head = num_empty_layers_add_in_head
        cfg.num_empty_layers_add_in_tail = num_empty_layers_add_in_tail
        cfg.head_wise_swa_ratio = 0.0  # Ensure this doesn't interfere
        # MagicMock auto-creates truthy attributes, which would make the
        # DSv4-Hybrid (CSA/HCA) detection fire and select CSADynamicCache.
        cfg.experimental_attention_variant = None
        cfg.csa_compress_ratios = None
        model.config = cfg
        return GreedyGenerator(model)

    def test_no_sliding_window(self):
        """No sliding_window: all swa_layers should be False."""
        gen = self._make_generator_with_cfg(
            num_hidden_layers=4, sliding_window=None
        )
        self.assertEqual(len(gen.cache.swa_layers), 4)
        self.assertTrue(all(not x for x in gen.cache.swa_layers))
        self.assertIsNone(gen.cache.window_size)

    def test_sliding_window_no_skip_freq(self):
        """sliding_window set, no skip_freq: all layers use SWA."""
        gen = self._make_generator_with_cfg(
            num_hidden_layers=4,
            sliding_window=(512, 512),
            window_attn_skip_freq=None,
        )
        self.assertTrue(all(gen.cache.swa_layers))
        self.assertEqual(gen.cache.window_size, 512)

    def test_sliding_window_with_int_skip_freq(self):
        """skip_freq=2: every 2nd layer (0,2,...) skips SWA."""
        gen = self._make_generator_with_cfg(
            num_hidden_layers=4,
            sliding_window=(256, 256),
            window_attn_skip_freq=2,
        )
        # layer % 2 != 0 => SWA: layers 1,3 are True; 0,2 are False
        self.assertEqual(gen.cache.swa_layers, [False, True, False, True])

    def test_sliding_window_with_list_skip_freq(self):
        """skip_freq as list: per-layer control."""
        gen = self._make_generator_with_cfg(
            num_hidden_layers=4,
            sliding_window=(128, 128),
            window_attn_skip_freq=[0, 1, 1, 0],
        )
        self.assertEqual(gen.cache.swa_layers, [False, True, True, False])

    def test_empty_layers_increase_total(self):
        """Empty layers in head/tail increase total cache layers."""
        gen = self._make_generator_with_cfg(
            num_hidden_layers=2,
            sliding_window=None,
            num_empty_layers_add_in_head=1,
            num_empty_layers_add_in_tail=1,
        )
        # total = 2 + 1 + 1 = 4
        self.assertEqual(len(gen.cache.swa_layers), 4)
        self.assertEqual(len(gen.cache.k), 4)

    def _make_generator_with_head_wise_swa(self, head_wise_swa_ratio):
        from unittest.mock import MagicMock

        from paddlefleet.generation.greedy_generator import GreedyGenerator

        model = MagicMock()
        cfg = MagicMock()
        cfg.num_hidden_layers = 4
        cfg.sequence_parallel = False
        cfg.apply_rope_fusion = False
        cfg.recompute_granularity = None
        cfg.sliding_window = (512, 512)
        cfg.window_attn_skip_freq = None
        cfg.num_empty_layers_add_in_head = 0
        cfg.num_empty_layers_add_in_tail = 0
        cfg.head_wise_swa_ratio = head_wise_swa_ratio
        # MagicMock auto-creates truthy attributes, which would make the
        # DSv4-Hybrid (CSA/HCA) detection fire and select CSADynamicCache.
        cfg.experimental_attention_variant = None
        cfg.csa_compress_ratios = None
        model.config = cfg
        return GreedyGenerator(model)

    def test_head_wise_swa_ratio_disables_window_size(self):
        """head_wise_swa_ratio in (0, 1) should set window_size to None."""
        with self.assertRaises(ValueError):
            self._make_generator_with_head_wise_swa(0.5)

    def test_head_wise_swa_ratio_zero_preserves_window_size(self):
        """head_wise_swa_ratio=0 should preserve window_size."""
        gen = self._make_generator_with_head_wise_swa(0.0)
        self.assertEqual(gen.cache.window_size, 512)

    def test_head_wise_swa_ratio_one_preserves_window_size(self):
        """head_wise_swa_ratio=1.0 (boundary) should preserve window_size."""
        gen = self._make_generator_with_head_wise_swa(1.0)
        self.assertEqual(gen.cache.window_size, 512)


class TestGreedyGeneratorEosStop(unittest.TestCase):
    """Test eos_token_id handling in GreedyGenerator.generate (mocked model)."""

    def _make_generator(self, token_sequence):
        """Create a GreedyGenerator with a fake model that yields given tokens."""
        from unittest.mock import MagicMock

        from paddlefleet.generation.greedy_generator import (
            DynamicKVCache,
            GreedyGenerator,
        )

        # token_sequence: list of int, tokens the model will output in order
        self._call_idx = 0
        seq = token_sequence

        def fake_forward(inputs):
            # Return logits where argmax gives the desired token
            vocab_size = 100
            logits = paddle.zeros([1, 1, vocab_size], dtype="float32")
            tok_id = seq[min(self._call_idx, len(seq) - 1)]
            logits[0, 0, tok_id] = 10.0
            self._call_idx += 1
            return logits

        model = MagicMock()
        model.side_effect = fake_forward
        model.config = MagicMock()
        model.config.num_hidden_layers = 1
        model.config.sequence_parallel = False
        model.config.apply_rope_fusion = False
        model.config.recompute_granularity = None
        model.config.num_empty_layers_add_in_head = 0
        model.config.num_empty_layers_add_in_tail = 0

        gen = object.__new__(GreedyGenerator)
        gen.model = model
        gen.cache = DynamicKVCache(num_layers=1)
        return gen

    def test_eos_int_stops_early(self):
        """eos_token_id as int should stop generation."""
        # Sequence: 5, 5, 3(eos), 5, 5 — should stop at step 3
        gen = self._make_generator([5, 5, 3, 5, 5])
        input_ids = paddle.to_tensor([[1, 2]], dtype="int64")
        out = gen.generate(input_ids, max_new_tokens=10, eos_token_id=3)
        generated = out[0, 2:].tolist()
        # Should contain 5, 5, 3 then stop
        self.assertEqual(generated, [5, 5, 3])

    def test_eos_list_single_token_stops(self):
        """eos_token_id as list of single-token lists (e.g. [[3],[7]]) should stop."""
        gen = self._make_generator([5, 5, 7, 5, 5])
        input_ids = paddle.to_tensor([[1, 2]], dtype="int64")
        out = gen.generate(
            input_ids, max_new_tokens=10, eos_token_id=[[3], [7]]
        )
        generated = out[0, 2:].tolist()
        self.assertEqual(generated, [5, 5, 7])

    def test_eos_list_multi_token_not_early_stop(self):
        """Multi-token stop sequences in list should not trigger early stop."""
        # [[10, 20]] is a multi-token stop — should NOT stop generation
        gen = self._make_generator([10, 5, 5, 5, 5])
        input_ids = paddle.to_tensor([[1, 2]], dtype="int64")
        out = gen.generate(input_ids, max_new_tokens=5, eos_token_id=[[10, 20]])
        generated = out[0, 2:].tolist()
        # All 5 tokens generated (no early stop)
        self.assertEqual(len(generated), 5)

    def test_eos_none_generates_max(self):
        """No eos_token_id should generate max_new_tokens."""
        gen = self._make_generator([5] * 10)
        input_ids = paddle.to_tensor([[1, 2]], dtype="int64")
        out = gen.generate(input_ids, max_new_tokens=5, eos_token_id=None)
        generated = out[0, 2:].tolist()
        self.assertEqual(len(generated), 5)


class TestGenerateUseCacheIsCausal(unittest.TestCase):
    """Test that DotProductAttention flashmask branch sets is_causal correctly with KV cache."""

    def _call_attention_forward(self, q_len):
        """Call DotProductAttention.forward entering flashmask+KV cache path."""
        from unittest.mock import MagicMock, patch

        from paddlefleet.transformer.dot_product_attention import (
            DotProductAttention,
        )
        from paddlefleet.transformer.enums import AttnMaskType

        config = MagicMock()
        config.gpt_model_use_experimental_version = False
        config.context_parallel_size = 1
        config.head_dim = 64
        config.num_attention_heads = 4
        config.num_key_value_heads = 4
        config.apply_query_key_layer_scaling = False
        config.fp16 = False
        config.bf16 = True
        config.masked_softmax_fusion = False
        config.attention_softmax_in_fp32 = True
        config.attention_dropout = 0.0
        config.sliding_window = None
        config.window_attn_skip_freq = None
        config.softmax_type = "vanilla"
        config.perform_initialization = True
        config.params_dtype = "bfloat16"
        config.init_method = MagicMock()
        config._attn_implementation = "flash"
        config.flashmask_use_varlen = False
        config.experimental_dataflow = False

        attn = DotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )

        B, H, D = 1, 4, 64
        query = paddle.randn([B, q_len, H, D]).cast("bfloat16")
        key = paddle.randn([B, q_len, H, D]).cast("bfloat16")
        value = paddle.randn([B, q_len, H, D]).cast("bfloat16")
        startend = paddle.zeros([B, 1, q_len, 2], dtype="int32")

        past_kv = MagicMock()
        past_kv.update = MagicMock(return_value=(key, value))

        with patch(
            "paddlefleet.transformer.dot_product_attention.flashmask_attention"
        ) as mock_fm:
            mock_fm.return_value = paddle.randn([B, q_len, H, D]).cast(
                "bfloat16"
            )
            attn.forward(
                query=query,
                key=key,
                value=value,
                attention_mask=None,
                attn_mask_startend_row_indices=startend,
                attn_mask_type=AttnMaskType.causal,
                past_key_values=past_kv,
                layer_idx=0,
                use_cache=True,
            )
            past_kv.update.assert_called_once()
            _, kwargs = mock_fm.call_args
            return kwargs["causal"]

    def test_decode_q_len_1_is_causal_false(self):
        """Decode step (q_len==1) should set is_causal=False."""
        self.assertFalse(self._call_attention_forward(q_len=1))

    def test_prefill_q_len_gt_1_is_causal_true(self):
        """Prefill step (q_len>1) should set is_causal=True."""
        self.assertTrue(self._call_attention_forward(q_len=4))


class TestGreedyGeneratorDebugMode(unittest.TestCase):
    """Cover all _DEBUG branches in GreedyGenerator.generate by forcing GREEDY_DEBUG=1."""

    def _make_debug_generator(self, token_sequence):
        """Return a GreedyGenerator whose fake model yields token_sequence,
        with _DEBUG patched to True in the greedy_generator module."""
        from unittest.mock import MagicMock

        from paddlefleet.generation.greedy_generator import (
            DynamicKVCache,
            GreedyGenerator,
        )

        self._call_idx = 0
        seq = token_sequence

        def fake_forward(inputs):
            vocab_size = 100
            logits = paddle.zeros([1, 1, vocab_size], dtype="float32")
            tok_id = seq[min(self._call_idx, len(seq) - 1)]
            logits[0, 0, tok_id] = 10.0
            self._call_idx += 1
            return logits

        model = MagicMock()
        model.side_effect = fake_forward
        model.config = MagicMock()
        model.config.num_hidden_layers = 1
        model.config.sequence_parallel = False
        model.config.apply_rope_fusion = False
        model.config.recompute_granularity = None
        model.config.num_empty_layers_add_in_head = 0
        model.config.num_empty_layers_add_in_tail = 0

        gen = object.__new__(GreedyGenerator)
        gen.model = model
        gen.cache = DynamicKVCache(num_layers=1)
        return gen

    def test_debug_mode_runs_without_error(self):
        """With _DEBUG=True all debug log branches execute without raising."""
        import paddlefleet.generation.greedy_generator as _m

        orig = _m._DEBUG
        try:
            _m._DEBUG = True
            gen = self._make_debug_generator([5, 6, 7])
            input_ids = paddle.to_tensor([[1, 2]], dtype="int64")
            out = gen.generate(input_ids, max_new_tokens=4, eos_token_id=None)
            self.assertEqual(out.shape[0], 1)
            self.assertEqual(out.shape[1], 2 + 4)
        finally:
            _m._DEBUG = orig

    def test_debug_mode_eos_stops_early(self):
        """_DEBUG=True still stops correctly at eos token."""
        import paddlefleet.generation.greedy_generator as _installed

        orig = getattr(_installed, "_DEBUG", False)
        try:
            _installed._DEBUG = True
            gen = self._make_debug_generator([5, 5, 3, 5, 5])
            input_ids = paddle.to_tensor([[1, 2]], dtype="int64")
            out = gen.generate(input_ids, max_new_tokens=10, eos_token_id=3)
            generated = out[0, 2:].tolist()
            self.assertEqual(generated, [5, 5, 3])
        finally:
            _installed._DEBUG = orig


class TestNoCacheDebugLogSteps(unittest.TestCase):
    """Cover the _log_this_step debug-logging branches in _generate_no_cache
    (reached via generate(..., no_cache=True)) by forcing _DEBUG=True.

    The default _DEBUG is False, so the input/logits logging blocks guarded by
    `_log_this_step` are otherwise never executed by the test suite.
    """

    def _make_debug_generator(
        self, token_sequence, batch_size=1, vocab_size=100
    ):
        from unittest.mock import MagicMock

        from paddlefleet.generation.greedy_generator import (
            DynamicKVCache,
            GreedyGenerator,
        )

        self._call_idx = 0
        seq = token_sequence
        bsz = batch_size

        def fake_forward(inputs):
            logits = paddle.zeros([bsz, 1, vocab_size], dtype="float32")
            tok_id = seq[min(self._call_idx, len(seq) - 1)]
            logits[:, 0, tok_id] = 10.0
            self._call_idx += 1
            return logits

        model = MagicMock()
        model.side_effect = fake_forward
        model.config = MagicMock()
        model.config.num_hidden_layers = 1
        model.config.sequence_parallel = False
        model.config.apply_rope_fusion = False
        model.config.recompute_granularity = None
        model.config.num_empty_layers_add_in_head = 0
        model.config.num_empty_layers_add_in_tail = 0

        gen = object.__new__(GreedyGenerator)
        gen.model = model
        gen.cache = DynamicKVCache(num_layers=1)
        return gen

    def test_no_cache_debug_covers_prefill_and_decode_logging(self):
        """_DEBUG=True + no_cache: prefill (step 0) and decode (step>0) logging
        branches run without error. max_new_tokens=5 => steps 0..4, so both the
        prefill and decode `_tag` values and both sides of the `step < 4` gate
        (True for steps 0-3, False for step 4) are exercised.
        """
        import paddlefleet.generation.greedy_generator as _m

        orig = _m._DEBUG
        try:
            _m._DEBUG = True
            gen = self._make_debug_generator([5, 6, 7, 8, 9])
            input_ids = paddle.to_tensor([[1, 2]], dtype="int64")
            out = gen.generate(input_ids, max_new_tokens=5, no_cache=True)
            self.assertEqual(out.shape[0], 1)
            self.assertEqual(out.shape[1], 2 + 5)
        finally:
            _m._DEBUG = orig

    def test_no_cache_debug_with_log_probs_and_eos(self):
        """_DEBUG=True + no_cache still returns correct log-probs and honors eos
        while running the logits-logging branch (shape/min-max-mean/top-5)."""
        import paddlefleet.generation.greedy_generator as _m

        orig = _m._DEBUG
        try:
            _m._DEBUG = True
            gen = self._make_debug_generator([5, 6, 3, 7])
            input_ids = paddle.to_tensor([[1, 2]], dtype="int64")
            generated, log_probs = gen.generate(
                input_ids,
                max_new_tokens=10,
                eos_token_id=3,
                no_cache=True,
                return_log_probs=True,
            )
            num_new = generated.shape[1] - input_ids.shape[1]
            self.assertEqual(num_new, 3)
            self.assertEqual(len(log_probs[0]), 3)
        finally:
            _m._DEBUG = orig


class TestReturnLogProbs(unittest.TestCase):
    """Unit tests for the return_log_probs feature in GreedyGenerator.generate."""

    def _make_generator(self, token_sequence, batch_size=1, vocab_size=100):
        """Create a GreedyGenerator backed by a fake model.

        The fake model always emits logits such that argmax gives
        ``token_sequence[call_idx]``.  Works for any batch_size (same token
        for every batch element).
        """
        from unittest.mock import MagicMock

        from paddlefleet.generation.greedy_generator import (
            DynamicKVCache,
            GreedyGenerator,
        )

        self._call_idx = 0
        seq = token_sequence
        bsz = batch_size

        def fake_forward(inputs):
            # logits shape: [B, seq_len, vocab]
            logits = paddle.zeros([bsz, 1, vocab_size], dtype="float32")
            tok_id = seq[min(self._call_idx, len(seq) - 1)]
            logits[:, 0, tok_id] = 10.0
            self._call_idx += 1
            return logits

        model = MagicMock()
        model.side_effect = fake_forward
        model.config = MagicMock()
        model.config.num_hidden_layers = 1
        model.config.sequence_parallel = False
        model.config.apply_rope_fusion = False
        model.config.recompute_granularity = None
        model.config.num_empty_layers_add_in_head = 0
        model.config.num_empty_layers_add_in_tail = 0

        gen = object.__new__(GreedyGenerator)
        gen.model = model
        gen.cache = DynamicKVCache(num_layers=1)
        return gen

    # ------------------------------------------------------------------
    # Return-type tests
    # ------------------------------------------------------------------

    def test_return_type_without_log_probs(self):
        """return_log_probs=False should return a plain Tensor."""
        gen = self._make_generator([5, 6, 7])
        input_ids = paddle.to_tensor([[1, 2]], dtype="int64")
        out = gen.generate(input_ids, max_new_tokens=3)
        self.assertIsInstance(out, paddle.Tensor)

    def test_return_type_with_log_probs(self):
        """return_log_probs=True should return (Tensor, list)."""
        gen = self._make_generator([5, 6, 7])
        input_ids = paddle.to_tensor([[1, 2]], dtype="int64")
        out = gen.generate(input_ids, max_new_tokens=3, return_log_probs=True)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        generated, log_probs = out
        self.assertIsInstance(generated, paddle.Tensor)
        self.assertIsInstance(log_probs, list)

    # ------------------------------------------------------------------
    # Correctness tests
    # ------------------------------------------------------------------

    def test_log_probs_are_non_positive(self):
        """Log-softmax values must be <= 0."""
        gen = self._make_generator([5, 6, 7])
        input_ids = paddle.to_tensor([[1, 2]], dtype="int64")
        _, log_probs = gen.generate(
            input_ids, max_new_tokens=3, return_log_probs=True
        )
        for lp in log_probs[0]:
            self.assertLessEqual(lp, 0.0)

    def test_log_probs_length_equals_generated_tokens(self):
        """Number of log-probs == number of generated tokens (no eos)."""
        max_new = 4
        gen = self._make_generator([5, 6, 7, 8])
        input_ids = paddle.to_tensor([[1, 2]], dtype="int64")
        generated, log_probs = gen.generate(
            input_ids, max_new_tokens=max_new, return_log_probs=True
        )
        # generated shape: [1, prompt_len + max_new]
        num_new = generated.shape[1] - input_ids.shape[1]
        self.assertEqual(len(log_probs[0]), num_new)

    def test_log_probs_length_with_eos(self):
        """Log-probs stop accumulating after eos (inclusive of eos step)."""
        # Sequence: 5, 3(eos), 6, 7 — generation stops after token 3
        gen = self._make_generator([5, 3, 6, 7])
        input_ids = paddle.to_tensor([[1, 2]], dtype="int64")
        generated, log_probs = gen.generate(
            input_ids,
            max_new_tokens=10,
            eos_token_id=3,
            return_log_probs=True,
        )
        # Tokens generated: 5, 3 → 2 tokens, 2 log-probs
        num_new = generated.shape[1] - input_ids.shape[1]
        self.assertEqual(num_new, 2)
        self.assertEqual(len(log_probs[0]), 2)

    def test_log_probs_dominant_token_is_high(self):
        """The log-prob of the chosen (dominant) token should be close to 0."""
        # logits: chosen token = 10.0, others = 0 → softmax ≈ 1 → log ≈ 0
        gen = self._make_generator([42])
        input_ids = paddle.to_tensor([[1]], dtype="int64")
        _, log_probs = gen.generate(
            input_ids, max_new_tokens=1, return_log_probs=True
        )
        # log_softmax of ~1 prob is close to 0
        self.assertGreater(log_probs[0][0], -0.5)

    # ------------------------------------------------------------------
    # Batch tests
    # ------------------------------------------------------------------

    def test_batch_log_probs_shape(self):
        """With batch_size=2 the outer list has 2 elements."""
        gen = self._make_generator([5, 6, 7], batch_size=2)
        input_ids = paddle.to_tensor([[1, 2], [3, 4]], dtype="int64")
        _, log_probs = gen.generate(
            input_ids, max_new_tokens=3, return_log_probs=True
        )
        self.assertEqual(len(log_probs), 2)

    def test_batch_each_element_is_list_of_floats(self):
        """Each per-batch log-prob collection must be a list of float."""
        gen = self._make_generator([5, 6, 7], batch_size=2)
        input_ids = paddle.to_tensor([[1, 2], [3, 4]], dtype="int64")
        _, log_probs = gen.generate(
            input_ids, max_new_tokens=3, return_log_probs=True
        )
        for per_seq in log_probs:
            self.assertIsInstance(per_seq, list)
            for lp in per_seq:
                self.assertIsInstance(lp, float)

    def test_batch_log_probs_not_collected_after_eos(self):
        """After a sequence hits eos, no further log-probs are appended for it."""
        # Both batch elements share the same fake forward (same token), so
        # both hit eos=3 at step 2 (tokens: 5, 3).
        gen = self._make_generator([5, 3, 6, 7], batch_size=2)
        input_ids = paddle.to_tensor([[1, 2], [1, 2]], dtype="int64")
        generated, log_probs = gen.generate(
            input_ids,
            max_new_tokens=10,
            eos_token_id=3,
            return_log_probs=True,
        )
        # Both sequences hit eos at step 2 → 2 log-probs each
        for per_seq in log_probs:
            self.assertEqual(len(per_seq), 2)

    # ------------------------------------------------------------------
    # Consistency: log-prob values match manual computation
    # ------------------------------------------------------------------

    def test_log_probs_value_consistency(self):
        """log_probs[0][0] must equal log_softmax of the first-step logits."""
        vocab_size = 100
        chosen_tok = 42
        logit_val = 10.0

        gen = self._make_generator(
            [chosen_tok], batch_size=1, vocab_size=vocab_size
        )
        input_ids = paddle.to_tensor([[1]], dtype="int64")
        _, log_probs = gen.generate(
            input_ids, max_new_tokens=1, return_log_probs=True
        )

        # Build the same logits and compute expected log-prob manually
        manual_logits = paddle.zeros([vocab_size], dtype="float32")
        manual_logits[chosen_tok] = logit_val
        expected_lp = float(
            paddle.nn.functional.log_softmax(manual_logits, axis=-1)[
                chosen_tok
            ].item()
        )

        self.assertAlmostEqual(log_probs[0][0], expected_lp, places=4)

    def test_log_probs_are_pre_temperature_distribution(self):
        """output_log_probs reflect the post-repetition-penalty raw distribution,
        NOT the temperature/top-k/top-p sampling distribution.

        Contract: with temperature=2.0 and top_k=5 the returned log-prob for
        the chosen token must still equal log_softmax over the *un-scaled*
        (pre-temperature) logits, not log_softmax over the temperature-divided
        logits actually used for sampling.
        """
        vocab_size = 50
        chosen_tok = 10
        logit_val = 8.0

        # Build a generator whose single call returns known logits
        from unittest.mock import MagicMock

        from paddlefleet.generation.greedy_generator import (
            DynamicKVCache,
            GreedyGenerator,
        )

        raw_logits_snapshot = []

        def fake_forward(inputs):
            logits = paddle.zeros([1, 1, vocab_size], dtype="float32")
            logits[0, 0, chosen_tok] = logit_val
            raw_logits_snapshot.append(logits[0, 0].clone())
            return logits

        model = MagicMock()
        model.side_effect = fake_forward
        model.config = MagicMock()
        model.config.num_hidden_layers = 1
        model.config.sequence_parallel = False
        model.config.apply_rope_fusion = False
        model.config.recompute_granularity = None
        model.config.num_empty_layers_add_in_head = 0
        model.config.num_empty_layers_add_in_tail = 0

        gen = object.__new__(GreedyGenerator)
        gen.model = model
        gen.cache = DynamicKVCache(num_layers=1)

        input_ids = paddle.to_tensor([[1]], dtype="int64")
        # Pin multinomial to always return chosen_tok so the test is
        # deterministic even though temperature/top_k enable sampling.
        # The entire temperature-scaling and top-k-filtering path still runs;
        # only the final random draw is fixed.
        with unittest.mock.patch(
            "paddle.multinomial",
            return_value=paddle.to_tensor([[chosen_tok]], dtype="int64"),
        ):
            _, log_probs = gen.generate(
                input_ids,
                max_new_tokens=1,
                return_log_probs=True,
                temperature=2.0,
                top_k=5,
            )

        # Expected: log_softmax over raw (pre-temperature) logits
        raw = raw_logits_snapshot[0].cast("float32")
        expected_pre_temp = float(
            paddle.nn.functional.log_softmax(raw, axis=-1)[chosen_tok].item()
        )
        # Sanity: log_softmax over temperature-divided logits would differ
        expected_post_temp = float(
            paddle.nn.functional.log_softmax(raw / 2.0, axis=-1)[
                chosen_tok
            ].item()
        )
        actual = log_probs[0][0]

        # The returned value matches the pre-temperature distribution
        self.assertAlmostEqual(actual, expected_pre_temp, places=4)
        # And it is *not* equal to the post-temperature distribution
        # (they differ because temperature != 1; if somehow they are equal
        # the test is vacuous, so we assert they differ first)
        if abs(expected_pre_temp - expected_post_temp) > 1e-4:
            self.assertNotAlmostEqual(actual, expected_post_temp, places=4)


class TestReturnLogProbsNoCache(unittest.TestCase):
    """Unit tests for return_log_probs in the no_cache=True branch."""

    def _make_generator(self, token_sequence, batch_size=1, vocab_size=100):
        """Create a GreedyGenerator backed by a fake model for no_cache mode.

        In no_cache mode the model is called with the full sequence each step,
        returning logits of shape [B, seq_len, vocab].  We make the last-position
        logit peak at ``token_sequence[call_idx]``.
        """
        from unittest.mock import MagicMock

        from paddlefleet.generation.greedy_generator import (
            DynamicKVCache,
            GreedyGenerator,
        )

        self._call_idx = 0
        seq = token_sequence
        bsz = batch_size

        def fake_forward(inputs):
            cur_len = inputs["input_ids"].shape[1]
            logits = paddle.zeros([bsz, cur_len, vocab_size], dtype="float32")
            tok_id = seq[min(self._call_idx, len(seq) - 1)]
            logits[:, -1, tok_id] = 10.0
            self._call_idx += 1
            return logits

        model = MagicMock()
        model.side_effect = fake_forward
        model.config = MagicMock()
        model.config.num_hidden_layers = 1
        model.config.sequence_parallel = False
        model.config.apply_rope_fusion = False
        model.config.recompute_granularity = None
        model.config.num_empty_layers_add_in_head = 0
        model.config.num_empty_layers_add_in_tail = 0

        gen = object.__new__(GreedyGenerator)
        gen.model = model
        gen.cache = DynamicKVCache(num_layers=1)
        return gen

    def test_return_type_without_log_probs(self):
        """no_cache + return_log_probs=False should return a plain Tensor."""
        gen = self._make_generator([5, 6, 7])
        input_ids = paddle.to_tensor([[1, 2]], dtype="int64")
        out = gen.generate(input_ids, max_new_tokens=3, no_cache=True)
        self.assertIsInstance(out, paddle.Tensor)

    def test_return_type_with_log_probs(self):
        """no_cache + return_log_probs=True should return (Tensor, list)."""
        gen = self._make_generator([5, 6, 7])
        input_ids = paddle.to_tensor([[1, 2]], dtype="int64")
        out = gen.generate(
            input_ids, max_new_tokens=3, no_cache=True, return_log_probs=True
        )
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        generated, log_probs = out
        self.assertIsInstance(generated, paddle.Tensor)
        self.assertIsInstance(log_probs, list)

    def test_log_probs_are_non_positive(self):
        """Log-softmax values must be <= 0 in no_cache mode."""
        gen = self._make_generator([5, 6, 7])
        input_ids = paddle.to_tensor([[1, 2]], dtype="int64")
        _, log_probs = gen.generate(
            input_ids, max_new_tokens=3, no_cache=True, return_log_probs=True
        )
        for lp in log_probs[0]:
            self.assertLessEqual(lp, 0.0)

    def test_log_probs_length_equals_generated_tokens(self):
        """Number of log-probs == max_new_tokens when no eos in no_cache mode."""
        max_new = 4
        gen = self._make_generator([5, 6, 7, 8])
        input_ids = paddle.to_tensor([[1, 2]], dtype="int64")
        generated, log_probs = gen.generate(
            input_ids,
            max_new_tokens=max_new,
            no_cache=True,
            return_log_probs=True,
        )
        num_new = generated.shape[1] - input_ids.shape[1]
        self.assertEqual(len(log_probs[0]), num_new)

    def test_log_probs_length_with_eos(self):
        """Log-probs list length equals number of tokens generated before eos stop."""
        # tokens: 5, 3(eos) → stops after 2 tokens
        gen = self._make_generator([5, 3, 6, 7])
        input_ids = paddle.to_tensor([[1, 2]], dtype="int64")
        generated, log_probs = gen.generate(
            input_ids,
            max_new_tokens=10,
            eos_token_id=3,
            no_cache=True,
            return_log_probs=True,
        )
        num_new = generated.shape[1] - input_ids.shape[1]
        self.assertEqual(num_new, 2)
        self.assertEqual(len(log_probs[0]), 2)

    def test_log_probs_dominant_token_is_high(self):
        """Chosen token has logit=10, others=0 → log-prob should be close to 0."""
        gen = self._make_generator([42])
        input_ids = paddle.to_tensor([[1]], dtype="int64")
        _, log_probs = gen.generate(
            input_ids, max_new_tokens=1, no_cache=True, return_log_probs=True
        )
        self.assertGreater(log_probs[0][0], -0.5)

    def test_batch_log_probs_shape(self):
        """With batch_size=2 in no_cache mode the outer list has 2 elements."""
        gen = self._make_generator([5, 6, 7], batch_size=2)
        input_ids = paddle.to_tensor([[1, 2], [3, 4]], dtype="int64")
        _, log_probs = gen.generate(
            input_ids, max_new_tokens=3, no_cache=True, return_log_probs=True
        )
        self.assertEqual(len(log_probs), 2)

    def test_batch_each_element_is_list_of_floats(self):
        """Each per-batch log-prob in no_cache mode must be a list of float."""
        gen = self._make_generator([5, 6, 7], batch_size=2)
        input_ids = paddle.to_tensor([[1, 2], [3, 4]], dtype="int64")
        _, log_probs = gen.generate(
            input_ids, max_new_tokens=3, no_cache=True, return_log_probs=True
        )
        for per_seq in log_probs:
            self.assertIsInstance(per_seq, list)
            for lp in per_seq:
                self.assertIsInstance(lp, float)

    def test_log_probs_value_consistency(self):
        """log_probs[0][0] must equal log_softmax of the first-step logits in no_cache."""
        vocab_size = 100
        chosen_tok = 42
        logit_val = 10.0

        gen = self._make_generator(
            [chosen_tok], batch_size=1, vocab_size=vocab_size
        )
        input_ids = paddle.to_tensor([[1]], dtype="int64")
        _, log_probs = gen.generate(
            input_ids, max_new_tokens=1, no_cache=True, return_log_probs=True
        )

        manual_logits = paddle.zeros([vocab_size], dtype="float32")
        manual_logits[chosen_tok] = logit_val
        expected_lp = float(
            paddle.nn.functional.log_softmax(manual_logits, axis=-1)[
                chosen_tok
            ].item()
        )
        self.assertAlmostEqual(log_probs[0][0], expected_lp, places=4)


class TestNoCacheEosListBranch(unittest.TestCase):
    """Cover the `isinstance(eos_token_id, list)` branch in _generate_no_cache."""

    def _make_generator(self, token_sequence, batch_size=1, vocab_size=100):
        from unittest.mock import MagicMock

        from paddlefleet.generation.greedy_generator import (
            DynamicKVCache,
            GreedyGenerator,
        )

        self._call_idx = 0
        seq = token_sequence
        bsz = batch_size

        def fake_forward(inputs):
            cur_len = inputs["input_ids"].shape[1]
            logits = paddle.zeros([bsz, cur_len, vocab_size], dtype="float32")
            tok_id = seq[min(self._call_idx, len(seq) - 1)]
            logits[:, -1, tok_id] = 10.0
            self._call_idx += 1
            return logits

        model = MagicMock()
        model.side_effect = fake_forward
        model.config = MagicMock()
        model.config.num_hidden_layers = 1
        model.config.sequence_parallel = False
        model.config.apply_rope_fusion = False
        model.config.recompute_granularity = None
        model.config.num_empty_layers_add_in_head = 0
        model.config.num_empty_layers_add_in_tail = 0

        gen = object.__new__(GreedyGenerator)
        gen.model = model
        gen.cache = DynamicKVCache(num_layers=1)
        return gen

    def test_eos_list_single_token_stops(self):
        """no_cache: eos_token_id=[[3],[7]] should stop on token 7."""
        gen = self._make_generator([5, 5, 7, 5, 5])
        input_ids = paddle.to_tensor([[1, 2]], dtype="int64")
        out = gen.generate(
            input_ids, max_new_tokens=10, eos_token_id=[[3], [7]], no_cache=True
        )
        generated = out[0, 2:].tolist()
        self.assertEqual(generated, [5, 5, 7])

    def test_eos_list_flat_int_stops(self):
        """no_cache: eos_token_id=[3, 7] (flat ints) should stop on token 3."""
        gen = self._make_generator([5, 3, 9, 9])
        input_ids = paddle.to_tensor([[1, 2]], dtype="int64")
        out = gen.generate(
            input_ids, max_new_tokens=10, eos_token_id=[3, 7], no_cache=True
        )
        generated = out[0, 2:].tolist()
        self.assertEqual(generated, [5, 3])

    def test_eos_list_multi_token_no_early_stop(self):
        """no_cache: multi-token stop seq [[10,20]] should NOT trigger early stop."""
        gen = self._make_generator([10, 5, 5, 5, 5])
        input_ids = paddle.to_tensor([[1, 2]], dtype="int64")
        out = gen.generate(
            input_ids, max_new_tokens=5, eos_token_id=[[10, 20]], no_cache=True
        )
        generated = out[0, 2:].tolist()
        self.assertEqual(len(generated), 5)

    def test_eos_list_batch_all_done(self):
        """no_cache: both batch elements hit eos from list → early stop."""
        gen = self._make_generator([5, 7, 9, 9], batch_size=2)
        input_ids = paddle.to_tensor([[1, 2], [3, 4]], dtype="int64")
        out = gen.generate(
            input_ids, max_new_tokens=10, eos_token_id=[[7]], no_cache=True
        )
        # Both hit token 7 at step 2 → generated 2 tokens: [5, 7]
        generated = out[0, 2:].tolist()
        self.assertEqual(generated, [5, 7])

    def test_eos_list_with_log_probs(self):
        """no_cache: list eos + return_log_probs works together."""
        gen = self._make_generator([5, 5, 3, 9, 9])
        input_ids = paddle.to_tensor([[1, 2]], dtype="int64")
        generated, log_probs = gen.generate(
            input_ids,
            max_new_tokens=10,
            eos_token_id=[[3], [7]],
            no_cache=True,
            return_log_probs=True,
        )
        num_new = generated.shape[1] - input_ids.shape[1]
        self.assertEqual(num_new, 3)  # 5, 5, 3(eos)
        self.assertEqual(len(log_probs[0]), 3)


class TestResolveLogprobStartLen(unittest.TestCase):
    """Unit tests for the ``logprob_start_len`` normalisation helper."""

    def setUp(self):
        from paddlefleet.generation.greedy_generator import (
            _resolve_logprob_start_len,
        )

        self.resolve = _resolve_logprob_start_len

    def test_none_defaults_to_prompt_len(self):
        """None means "generated tokens only", i.e. start at prompt_len."""
        self.assertEqual(self.resolve(None, 7), 7)

    def test_zero_is_clamped_to_one(self):
        """Position 0 has no preceding context, so it can never be scored."""
        self.assertEqual(self.resolve(0, 7), 1)

    def test_positive_value_passes_through(self):
        self.assertEqual(self.resolve(1, 7), 1)
        self.assertEqual(self.resolve(4, 7), 4)

    def test_value_beyond_prompt_len_passes_through(self):
        """Start positions inside the generated region are legal."""
        self.assertEqual(self.resolve(9, 7), 9)

    def test_negative_raises(self):
        with self.assertRaises(ValueError):
            self.resolve(-1, 7)


class _LogprobStartLenMixin:
    """Shared ``logprob_start_len`` assertions for both generate paths.

    Subclasses set ``no_cache``. Every fake model here returns logits for the
    *whole* input so the prompt-scoring slice is exercised; the KV-cache path
    is driven with a full-length prefill exactly like the real model.
    """

    no_cache = False
    prompt = [[1, 2, 3, 4]]  # prompt_len = 4
    max_new_tokens = 3

    def _make_generator(self, token_sequence, batch_size=1, vocab_size=100):
        """Fake model whose last-position argmax follows ``token_sequence``."""
        from unittest.mock import MagicMock

        from paddlefleet.generation.greedy_generator import (
            DynamicKVCache,
            GreedyGenerator,
        )

        call_idx = {"i": 0}
        seq = token_sequence
        bsz = batch_size

        def fake_forward(inputs):
            cur_len = inputs["input_ids"].shape[1]
            logits = paddle.zeros([bsz, cur_len, vocab_size], dtype="float32")
            tok_id = seq[min(call_idx["i"], len(seq) - 1)]
            logits[:, -1, tok_id] = 10.0
            call_idx["i"] += 1
            return logits

        model = MagicMock()
        model.side_effect = fake_forward
        model.config = MagicMock()
        model.config.num_hidden_layers = 1
        model.config.sequence_parallel = False
        model.config.apply_rope_fusion = False
        model.config.recompute_granularity = None
        model.config.num_empty_layers_add_in_head = 0
        model.config.num_empty_layers_add_in_tail = 0

        gen = object.__new__(GreedyGenerator)
        gen.model = model
        gen.cache = DynamicKVCache(num_layers=1)
        return gen

    def _run(self, batch_size=1, **kwargs):
        input_ids = paddle.to_tensor(self.prompt * batch_size, dtype="int64")
        gen = self._make_generator([5, 6, 7], batch_size=batch_size)
        return gen.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            return_log_probs=True,
            no_cache=self.no_cache,
            **kwargs,
        )

    # -- length semantics -------------------------------------------------

    def test_default_scores_generated_tokens_only(self):
        """Omitting logprob_start_len keeps the pre-existing behaviour."""
        generated, log_probs = self._run()
        num_new = generated.shape[1] - len(self.prompt[0])
        self.assertEqual(num_new, self.max_new_tokens)
        self.assertEqual(len(log_probs[0]), self.max_new_tokens)

    def test_start_zero_includes_prompt(self):
        """start=0 scores prompt positions 1.. plus every generated token."""
        prompt_len = len(self.prompt[0])
        _, log_probs = self._run(logprob_start_len=0)
        self.assertEqual(
            len(log_probs[0]), (prompt_len - 1) + self.max_new_tokens
        )

    def test_start_zero_and_one_are_equivalent(self):
        """Position 0 is unscoreable, so 0 and 1 must agree exactly."""
        _, lp_zero = self._run(logprob_start_len=0)
        _, lp_one = self._run(logprob_start_len=1)
        self.assertEqual(lp_zero, lp_one)

    def test_mid_prompt_start(self):
        """A start inside the prompt drops only the positions before it."""
        prompt_len = len(self.prompt[0])
        start = 2
        _, log_probs = self._run(logprob_start_len=start)
        self.assertEqual(
            len(log_probs[0]), (prompt_len - start) + self.max_new_tokens
        )

    def test_start_equal_prompt_len_matches_default(self):
        """start=prompt_len is the explicit spelling of the default."""
        prompt_len = len(self.prompt[0])
        _, lp_default = self._run()
        _, lp_explicit = self._run(logprob_start_len=prompt_len)
        self.assertEqual(lp_default, lp_explicit)

    def test_start_beyond_prompt_skips_generated_tokens(self):
        """A start inside the generated region skips the leading steps."""
        prompt_len = len(self.prompt[0])
        _, log_probs = self._run(logprob_start_len=prompt_len + 1)
        self.assertEqual(len(log_probs[0]), self.max_new_tokens - 1)

    def test_generated_tail_is_shared_across_starts(self):
        """Widening the window only prepends; the generated tail is stable."""
        _, lp_default = self._run()
        _, lp_full = self._run(logprob_start_len=0)
        self.assertEqual(lp_full[0][-self.max_new_tokens :], lp_default[0])

    # -- shape / type -----------------------------------------------------

    def test_single_flat_list_per_batch_element(self):
        """Prompt and generated scores share one flat list of floats."""
        _, log_probs = self._run(batch_size=2, logprob_start_len=0)
        expected = (len(self.prompt[0]) - 1) + self.max_new_tokens
        self.assertEqual(len(log_probs), 2)
        for per_seq in log_probs:
            self.assertIsInstance(per_seq, list)
            self.assertEqual(len(per_seq), expected)
            for lp in per_seq:
                self.assertIsInstance(lp, float)
                self.assertLessEqual(lp, 0.0)

    def test_return_type_is_two_tuple(self):
        """Both paths return ``(generated, log_probs)`` -- never a 3-tuple."""
        out = self._run(logprob_start_len=0)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        self.assertIsInstance(out[0], paddle.Tensor)
        self.assertIsInstance(out[1], list)

    def test_ignored_without_return_log_probs(self):
        """logprob_start_len is inert when log-probs are not requested."""
        input_ids = paddle.to_tensor(self.prompt, dtype="int64")
        gen = self._make_generator([5, 6, 7])
        out = gen.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            logprob_start_len=0,
            no_cache=self.no_cache,
        )
        self.assertIsInstance(out, paddle.Tensor)

    def test_negative_start_raises(self):
        input_ids = paddle.to_tensor(self.prompt, dtype="int64")
        gen = self._make_generator([5, 6, 7])
        with self.assertRaises(ValueError):
            gen.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                return_log_probs=True,
                logprob_start_len=-1,
                no_cache=self.no_cache,
            )

    def test_eos_truncation_still_applies(self):
        """Prompt scores are kept while the generated part stops at eos."""
        prompt_len = len(self.prompt[0])
        input_ids = paddle.to_tensor(self.prompt, dtype="int64")
        gen = self._make_generator([5, 3, 6, 7])
        generated, log_probs = gen.generate(
            input_ids,
            max_new_tokens=10,
            eos_token_id=3,
            return_log_probs=True,
            logprob_start_len=0,
            no_cache=self.no_cache,
        )
        num_new = generated.shape[1] - prompt_len
        self.assertEqual(num_new, 2)  # 5, 3(eos)
        self.assertEqual(len(log_probs[0]), (prompt_len - 1) + 2)

    # -- value / alignment ------------------------------------------------

    def _make_positional_generator(self, table):
        """Fake model whose logits are ``table[position_ids]``.

        Position-dependent logits make the prompt-scoring alignment
        observable: reading ``logits[p]`` instead of ``logits[p - 1]`` to score
        position ``p`` produces a different number, so an off-by-one fails
        instead of silently returning a plausible value.
        """
        from unittest.mock import MagicMock

        from paddlefleet.generation.greedy_generator import (
            DynamicKVCache,
            GreedyGenerator,
        )

        def fake_forward(inputs):
            bsz = inputs["input_ids"].shape[0]
            positions = inputs["position_ids"][0].reshape([-1])
            rows = paddle.index_select(table, positions, axis=0)
            return rows.unsqueeze(0).expand(
                [bsz, rows.shape[0], table.shape[1]]
            )

        model = MagicMock()
        model.side_effect = fake_forward
        model.config = MagicMock()
        model.config.num_hidden_layers = 1
        model.config.sequence_parallel = False
        model.config.apply_rope_fusion = False
        model.config.recompute_granularity = None
        model.config.num_empty_layers_add_in_head = 0
        model.config.num_empty_layers_add_in_tail = 0

        gen = object.__new__(GreedyGenerator)
        gen.model = model
        gen.cache = DynamicKVCache(num_layers=1)
        return gen

    def test_prompt_log_prob_values_match_manual(self):
        """Prompt scores equal ``log_softmax(logits[p - 1])[token_at_p]``."""
        paddle.seed(20260818)
        vocab_size = 16
        table = paddle.randn([32, vocab_size], dtype="float32")
        prompt_len = len(self.prompt[0])
        input_ids = paddle.to_tensor(self.prompt, dtype="int64")

        gen = self._make_positional_generator(table)
        _, log_probs = gen.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            return_log_probs=True,
            logprob_start_len=0,
            no_cache=self.no_cache,
        )

        for p in range(1, prompt_len):
            row = paddle.nn.functional.log_softmax(table[p - 1], axis=-1)
            expected = float(row[int(input_ids[0, p].item())].item())
            self.assertAlmostEqual(log_probs[0][p - 1], expected, places=5)

    def test_prompt_log_probs_are_not_off_by_one(self):
        """Guard the alignment explicitly: ``logits[p]`` must not be used."""
        paddle.seed(20260818)
        vocab_size = 16
        table = paddle.randn([32, vocab_size], dtype="float32")
        prompt_len = len(self.prompt[0])
        input_ids = paddle.to_tensor(self.prompt, dtype="int64")

        gen = self._make_positional_generator(table)
        _, log_probs = gen.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            return_log_probs=True,
            logprob_start_len=0,
            no_cache=self.no_cache,
        )

        for p in range(1, prompt_len):
            shifted = paddle.nn.functional.log_softmax(table[p], axis=-1)
            wrong = float(shifted[int(input_ids[0, p].item())].item())
            self.assertNotAlmostEqual(log_probs[0][p - 1], wrong, places=3)

    def test_mid_prompt_start_is_a_suffix_of_full_prompt_scores(self):
        """Slicing the window must not change the surviving prompt values."""
        paddle.seed(20260818)
        table = paddle.randn([32, 16], dtype="float32")
        prompt_len = len(self.prompt[0])
        input_ids = paddle.to_tensor(self.prompt, dtype="int64")

        full = self._make_positional_generator(table).generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            return_log_probs=True,
            logprob_start_len=0,
            no_cache=self.no_cache,
        )[1]
        partial = self._make_positional_generator(table).generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            return_log_probs=True,
            logprob_start_len=prompt_len - 1,
            no_cache=self.no_cache,
        )[1]
        # full holds prompt positions 1..prompt_len-1, partial only the last
        self.assertAlmostEqual(partial[0][0], full[0][prompt_len - 2], places=6)


class TestLogprobStartLenCached(_LogprobStartLenMixin, unittest.TestCase):
    """``logprob_start_len`` on the default KV-cache path."""

    no_cache = False


class TestLogprobStartLenNoCache(_LogprobStartLenMixin, unittest.TestCase):
    """``logprob_start_len`` on the no_cache (full-recompute) path."""

    no_cache = True


class TestLogprobStartLenCacheParity(unittest.TestCase):
    """The two generate paths must return the same prompt log-probs.

    The fake model is deterministic given the position ids, and the prompt is
    scored from the one full-length prefill both paths perform, so the prompt
    portion must agree exactly rather than only up to bf16 noise.
    """

    prompt = [[1, 2, 3, 4, 5]]
    max_new_tokens = 3

    def test_prompt_scores_match_across_paths(self):
        mixin = _LogprobStartLenMixin()
        paddle.seed(7)
        table = paddle.randn([32, 16], dtype="float32")
        prompt_len = len(self.prompt[0])
        input_ids = paddle.to_tensor(self.prompt, dtype="int64")

        results = []
        for no_cache in (False, True):
            _, log_probs = mixin._make_positional_generator(table).generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                return_log_probs=True,
                logprob_start_len=0,
                no_cache=no_cache,
            )
            results.append(log_probs[0][: prompt_len - 1])

        self.assertEqual(len(results[0]), prompt_len - 1)
        for cached, full in zip(*results):
            self.assertAlmostEqual(cached, full, places=6)


if __name__ == "__main__":
    print("Running greedy generator unit tests...")
    unittest.main(verbosity=2)
