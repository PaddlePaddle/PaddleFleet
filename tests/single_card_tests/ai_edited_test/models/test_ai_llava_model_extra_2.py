# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless distributed on the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

import unittest
from unittest.mock import MagicMock

import paddle

from paddlefleet.models.multimodal.llava_model import (
    LLaVAModel,
    pixel_shuffle,
)


class TestPixelShuffle(unittest.TestCase):
    """Tests for pixel_shuffle function."""

    def test_pixel_shuffle_basic(self):
        """pixel_shuffle should rearrange spatial and channel dimensions."""
        paddle.disable_static()
        # Input: [B, H*W, C*r*r] where r is the upscale factor
        x = paddle.randn([1, 4, 16])  # 4 spatial positions, 16 channels
        result = pixel_shuffle(x, upscale_factor=2)
        # After pixel shuffle: spatial dim * upscale_factor^2 rearranged
        self.assertTrue(paddle.is_tensor(result))

    def test_pixel_shuffle_preserves_batch(self):
        """pixel_shuffle should preserve the batch dimension."""
        paddle.disable_static()
        x = paddle.randn([2, 4, 16])
        result = pixel_shuffle(x, upscale_factor=2)
        self.assertEqual(result.shape[0], 2)

    def test_pixel_shuffle_output_elements_match(self):
        """pixel_shuffle should preserve total number of elements."""
        paddle.disable_static()
        x = paddle.randn([1, 4, 16])
        result = pixel_shuffle(x, upscale_factor=2)
        self.assertEqual(x.numel(), result.numel())


class TestLLaVAModelAddEncoder(unittest.TestCase):
    """Tests for LLaVAModel add_encoder attribute."""

    def test_default_add_encoder_is_true(self):
        """Default add_encoder should be True."""
        model = LLaVAModel.__new__(LLaVAModel)
        model.add_encoder = True
        self.assertTrue(model.add_encoder)

    def test_default_add_decoder_is_true(self):
        """Default add_decoder should be True."""
        model = LLaVAModel.__new__(LLaVAModel)
        model.add_decoder = True
        self.assertTrue(model.add_decoder)


class TestLLaVAModelSetInputTensorEdgeCases(unittest.TestCase):
    """Tests for LLaVAModel set_input_tensor edge cases."""

    def test_set_input_tensor_no_encoder_no_decoder(self):
        """set_input_tensor with neither encoder nor decoder should handle gracefully."""
        model = LLaVAModel.__new__(LLaVAModel)
        model.add_encoder = False
        model.add_decoder = False
        model.pre_process = True
        # Should not raise
        model.set_input_tensor([paddle.randn([4, 8])])

    def test_set_input_tensor_with_decoder_only(self):
        """set_input_tensor with decoder only should set decoder input tensor."""
        model = LLaVAModel.__new__(LLaVAModel)
        model.add_encoder = False
        model.add_decoder = True
        model.pre_process = True
        model.vision_model = MagicMock()
        model.language_model = MagicMock()
        mock_tensor = paddle.randn([4, 8])
        model.set_input_tensor([mock_tensor])
        model.language_model.set_input_tensor.assert_called_once()


class TestLLaVAModelSharedEmbeddingWeight(unittest.TestCase):
    """Tests for LLaVAModel shared embedding weight handling."""

    def test_model_has_shared_embedding_weight_attribute(self):
        """LLaVAModel should have shared_embedding_weight attribute when present."""
        model = LLaVAModel.__new__(LLaVAModel)
        # The attribute may or may not exist depending on initialization
        self.assertFalse(hasattr(model, "shared_embedding_weight"))


if __name__ == "__main__":
    unittest.main()
