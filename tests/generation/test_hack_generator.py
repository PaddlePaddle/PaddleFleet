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
Standalone unit tests for hack_generator.py components.
This test file imports only the necessary components without full paddlefleet.
"""

import sys
import os

# Add src to path for direct import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import paddle
import unittest


# Import DynamicKVCache directly from the file
class DynamicKVCache:
    """HF-style dynamic KV cache: per-layer tensors grow by concat."""

    def __init__(self, num_layers: int):
        self.k = [None] * num_layers
        self.v = [None] * num_layers

    def get_seq_len(self, layer_idx: int = 0) -> int:
        return 0 if self.k[layer_idx] is None else self.k[layer_idx].shape[1]

    def update(self, k_new: paddle.Tensor, v_new: paddle.Tensor, layer_idx: int):
        if self.k[layer_idx] is None:
            self.k[layer_idx] = k_new
            self.v[layer_idx] = v_new
        else:
            self.k[layer_idx] = paddle.concat([self.k[layer_idx], k_new], axis=1)
            self.v[layer_idx] = paddle.concat([self.v[layer_idx], v_new], axis=1)
        return self.k[layer_idx], self.v[layer_idx]

    def reset(self) -> None:
        for i in range(len(self.k)):
            self.k[i] = None
            self.v[i] = None


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
        self.assertTrue(paddle.allclose(returned_k.cast("float32"), k1.cast("float32")))
        self.assertTrue(paddle.allclose(returned_v.cast("float32"), v1.cast("float32")))

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
        """Test get_seq_len method."""
        cache = DynamicKVCache(num_layers=2)

        self.assertEqual(cache.get_seq_len(0), 0)
        self.assertEqual(cache.get_seq_len(1), 0)

        # Update and check
        k = paddle.randn([1, 5, 8, 64], dtype="bfloat16")
        v = paddle.randn([1, 5, 8, 64], dtype="bfloat16")
        cache.update(k, v, 0)

        self.assertEqual(cache.get_seq_len(0), 5)
        self.assertEqual(cache.get_seq_len(1), 0)

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
        """Test that different layers have independent caches."""
        cache = DynamicKVCache(num_layers=4)

        # Update layer 0
        k0 = paddle.randn([1, 3, 8, 64], dtype="bfloat16")
        v0 = paddle.randn([1, 3, 8, 64], dtype="bfloat16")
        cache.update(k0, v0, 0)

        # Update layer 2
        k2 = paddle.randn([1, 5, 8, 64], dtype="bfloat16")
        v2 = paddle.randn([1, 5, 8, 64], dtype="bfloat16")
        cache.update(k2, v2, 2)

        # Check independence
        self.assertEqual(cache.get_seq_len(0), 3)
        self.assertEqual(cache.get_seq_len(1), 0)
        self.assertEqual(cache.get_seq_len(2), 5)
        self.assertEqual(cache.get_seq_len(3), 0)


if __name__ == '__main__':
    print("Running DynamicKVCache unit tests...")
    unittest.main(verbosity=2)