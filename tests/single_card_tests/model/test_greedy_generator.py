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

import paddle

from paddlefleet.generation.greedy_generator import DynamicKVCache


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


if __name__ == "__main__":
    print("Running greedy generator unit tests...")
    unittest.main(verbosity=2)
