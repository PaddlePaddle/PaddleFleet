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

"""Unit tests for CSADynamicCache.has_layer_cache."""

import unittest

import paddle

from paddlefleet.generation.csa_cache import CSADynamicCache


class TestHasLayerCache(unittest.TestCase):
    """Test has_layer_cache correctly identifies primed vs unprimed layers."""

    def setUp(self):
        self.num_layers = 4
        self.cache = CSADynamicCache(self.num_layers)

    def test_fresh_cache_returns_false(self):
        """A freshly created cache has no primed layers."""
        for i in range(self.num_layers):
            self.assertFalse(self.cache.has_layer_cache(i))

    def test_true_after_raw_kv_set(self):
        """CSA layer is primed once raw_kv is populated (via append_raw)."""
        layer_idx = 1
        st = self.cache.get_csa_state(layer_idx)
        st.append_raw(paddle.zeros([2, 1, 64]))
        self.assertTrue(self.cache.has_layer_cache(layer_idx))
        # Other layers remain unprimed.
        self.assertFalse(self.cache.has_layer_cache(0))
        self.assertFalse(self.cache.has_layer_cache(2))

    def test_true_after_std_k_set(self):
        """Standard-attention layer is primed once k is populated (via update)."""
        layer_idx = 2
        k = paddle.zeros([2, 5, 64])
        v = paddle.zeros([2, 5, 64])
        self.cache.update(k, v, layer_idx)
        self.assertTrue(self.cache.has_layer_cache(layer_idx))
        # Other layers remain unprimed.
        self.assertFalse(self.cache.has_layer_cache(0))
        self.assertFalse(self.cache.has_layer_cache(1))

    def test_true_when_both_raw_kv_and_k_set(self):
        """Layer with both raw_kv and k set still returns True."""
        layer_idx = 0
        st = self.cache.get_csa_state(layer_idx)
        st.append_raw(paddle.zeros([1, 1, 32]))
        st.update_std(paddle.zeros([1, 1, 32]), paddle.zeros([1, 1, 32]))
        self.assertTrue(self.cache.has_layer_cache(layer_idx))

    def test_compressed_kv_alone_does_not_prime(self):
        """Setting only compressed_kv (without raw_kv or k) is not enough."""
        layer_idx = 3
        st = self.cache.get_csa_state(layer_idx)
        st.append_compressed(paddle.zeros([1, 1, 64]))
        self.assertFalse(self.cache.has_layer_cache(layer_idx))

    def test_reset_clears_primed_state(self):
        """After cache.reset(), all layers return False."""
        st = self.cache.get_csa_state(0)
        st.append_raw(paddle.zeros([1, 1, 64]))
        self.assertTrue(self.cache.has_layer_cache(0))
        self.cache.reset()
        self.assertFalse(self.cache.has_layer_cache(0))


if __name__ == "__main__":
    unittest.main()
