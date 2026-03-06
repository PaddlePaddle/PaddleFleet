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

import paddle

from paddlefleet.training.initialize import initialize_fleet
from paddlefleet.transformer.transformer_config import TransformerConfig

strategy = paddle.distributed.fleet.DistributedStrategy()
initialize_fleet(strategy=strategy)


class TestMoeLayerFreqAndFirstKDenseReplace(unittest.TestCase):
    """Tests for the moe_layer_freq / first_k_dense_replace logic in TransformerConfig.__post_init__."""

    def test_both_none_defaults_moe_layer_freq_to_1(self):
        """When both first_k_dense_replace and moe_layer_freq are None, moe_layer_freq defaults to 1."""
        config = TransformerConfig(
            first_k_dense_replace=None,
            moe_layer_freq=None,
            num_hidden_layers=12,
        )
        self.assertEqual(config.moe_layer_freq, 1)

    def test_first_k_dense_replace_with_list_moe_layer_freq_raises(self):
        """When first_k_dense_replace is set and moe_layer_freq is a list (not int), should raise ValueError."""
        with self.assertRaises(ValueError):
            TransformerConfig(
                first_k_dense_replace=2,
                moe_layer_freq=[1, 0, 1, 0],
                num_hidden_layers=4,
            )

    def test_first_k_dense_replace_with_int_moe_layer_freq(self):
        """When first_k_dense_replace is set and moe_layer_freq is an int,
        it should generate a pattern based on the frequency."""
        config = TransformerConfig(
            first_k_dense_replace=2,
            moe_layer_freq=2,
            num_hidden_layers=8,
        )
        # first 2 layers are dense (0), remaining layers follow pattern: 1 if (i % 2 == 0) else 0
        # Pattern for range(8): i=0->1, i=1->0, i=2->1, i=3->0, i=4->1, i=5->0, i=6->1, i=7->0
        expected = [0, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        self.assertEqual(config.moe_layer_freq, expected)

    def test_first_k_dense_replace_with_moe_layer_freq_none(self):
        """When first_k_dense_replace is set and moe_layer_freq is None,
        the remaining layers should all be MoE (all 1s)."""
        config = TransformerConfig(
            first_k_dense_replace=3,
            moe_layer_freq=None,
            num_hidden_layers=8,
        )
        # both-None check won't trigger since first_k_dense_replace is set,
        # moe_layer_freq stays None (falsy).
        # else branch: moe_layer_pattern = [1] * (8 - 3) = [1, 1, 1, 1, 1]
        # final = [0, 0, 0] + [1, 1, 1, 1, 1]
        expected = [0, 0, 0, 1, 1, 1, 1, 1]
        self.assertEqual(config.moe_layer_freq, expected)

    def test_first_k_dense_replace_with_moe_layer_freq_zero(self):
        """When first_k_dense_replace is set and moe_layer_freq is 0 (falsy int),
        the pattern should fall into the else branch producing all 1s for non-dense layers."""
        config = TransformerConfig(
            first_k_dense_replace=4,
            moe_layer_freq=0,
            num_hidden_layers=10,
        )
        # moe_layer_freq=0 is falsy, so moe_layer_pattern = [1] * (10 - 4) = [1, 1, 1, 1, 1, 1]
        expected = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        self.assertEqual(config.moe_layer_freq, expected)

    def test_first_k_dense_replace_with_moe_layer_freq_3(self):
        """When first_k_dense_replace is set with moe_layer_freq=3,
        every 3rd layer (index % 3 == 0) should be MoE."""
        config = TransformerConfig(
            first_k_dense_replace=1,
            moe_layer_freq=3,
            num_hidden_layers=7,
        )
        # first 1 layer is dense
        # Pattern for range(7): i=0->1, i=1->0, i=2->0, i=3->1, i=4->0, i=5->0, i=6->1
        expected = [0, 1, 0, 0, 1, 0, 0, 1]
        self.assertEqual(config.moe_layer_freq, expected)

    def test_first_k_dense_replace_equals_num_hidden_layers(self):
        """Edge case: first_k_dense_replace equals num_hidden_layers,
        all layers should be dense (all 0s)."""
        config = TransformerConfig(
            first_k_dense_replace=6,
            moe_layer_freq=0,
            num_hidden_layers=6,
        )
        # moe_layer_freq=0 is falsy, so moe_layer_pattern = [1] * (6 - 6) = []
        expected = [0, 0, 0, 0, 0, 0]
        self.assertEqual(config.moe_layer_freq, expected)

    def test_only_moe_layer_freq_int_no_first_k_dense(self):
        """When only moe_layer_freq is set (as int) and first_k_dense_replace is None,
        moe_layer_freq should remain as the integer value."""
        config = TransformerConfig(
            first_k_dense_replace=None,
            moe_layer_freq=2,
            num_hidden_layers=8,
        )
        self.assertEqual(config.moe_layer_freq, 2)

    def test_only_first_k_dense_replace_no_moe_layer_freq(self):
        """When first_k_dense_replace is set and moe_layer_freq is not specified (defaults to None),
        should generate the correct pattern."""
        config = TransformerConfig(
            first_k_dense_replace=2,
            num_hidden_layers=6,
        )
        # moe_layer_freq=None => both-None check won't trigger because first_k_dense_replace is set
        # After the None-None check, moe_layer_freq is still None
        # first_k_dense_replace is truthy => enter the block
        # moe_layer_freq is None (falsy) => moe_layer_pattern = [1] * (6-2) = [1,1,1,1]
        expected = [0, 0, 1, 1, 1, 1]
        self.assertEqual(config.moe_layer_freq, expected)

    def test_first_k_dense_replace_1_moe_layer_freq_1(self):
        """first_k_dense_replace=1, moe_layer_freq=1: first layer dense, rest all MoE."""
        config = TransformerConfig(
            first_k_dense_replace=1,
            moe_layer_freq=1,
            num_hidden_layers=5,
        )
        # moe_layer_freq=1 is truthy int, pattern: 1 if (i % 1 == 0) else 0 => all 1s
        expected = [0, 1, 1, 1, 1, 1]
        self.assertEqual(config.moe_layer_freq, expected)


if __name__ == "__main__":
    unittest.main()
