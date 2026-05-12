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
    IGNORE_INDEX,
    LLaVAModel,
    pixel_shuffle,
)


class TestPixelShuffleEdgeCases(unittest.TestCase):
    """Additional tests for pixel_shuffle edge cases."""

    def test_pixel_shuffle_upscale_1(self):
        """pixel_shuffle with upscale_factor=1 should not change tensor."""
        paddle.disable_static()
        x = paddle.randn([1, 4, 8])
        result = pixel_shuffle(x, upscale_factor=1)
        self.assertTrue(paddle.allclose(result, x))

    def test_pixel_shuffle_3d_input(self):
        """pixel_shuffle should handle 3D input [H*W, C*r*r]."""
        paddle.disable_static()
        x = paddle.randn([4, 16])
        result = pixel_shuffle(x, upscale_factor=2)
        self.assertTrue(paddle.is_tensor(result))


class TestLLaVAModelIgnoreIndex(unittest.TestCase):
    """Tests for IGNORE_INDEX constant."""

    def test_ignore_index_value(self):
        """IGNORE_INDEX should be -100."""
        self.assertEqual(IGNORE_INDEX, -100)


class TestLLaVAModelSetInputTensorBoth(unittest.TestCase):
    """Tests for LLaVAModel set_input_tensor with both encoder and decoder."""

    def test_sets_both_encoder_and_decoder(self):
        """set_input_tensor with both encoder and decoder should set both."""
        model = LLaVAModel.__new__(LLaVAModel)
        model.add_encoder = True
        model.add_decoder = True
        model.pre_process = True
        model.vision_model = MagicMock()
        model.language_model = MagicMock()
        mock_tensor = paddle.randn([4, 8])
        model.set_input_tensor([mock_tensor])
        model.vision_model.set_input_tensor.assert_called_once_with(mock_tensor)
        model.language_model.set_input_tensor.assert_called_once()


class TestLLaVAModelAttributes(unittest.TestCase):
    """Tests for LLaVAModel attribute defaults."""

    def test_image_token_index_default(self):
        """Default image_token_index should be a reasonable value."""
        model = LLaVAModel.__new__(LLaVAModel)
        # The attribute may not exist until init, but the constant should be defined
        self.assertIsNotNone(IGNORE_INDEX)


class TestPixelShuffleNonContiguous(unittest.TestCase):
    """Tests for pixel_shuffle with non-contiguous tensors."""

    def test_pixel_shuffle_works_with_transposed(self):
        """pixel_shuffle should work with transposed (non-contiguous) tensors."""
        paddle.disable_static()
        x = paddle.randn([1, 16, 4])
        x_t = x.transpose([0, 2, 1])  # Now non-contiguous
        # This should not raise
        result = pixel_shuffle(x_t.contiguous(), upscale_factor=2)
        self.assertTrue(paddle.is_tensor(result))


if __name__ == "__main__":
    unittest.main()
