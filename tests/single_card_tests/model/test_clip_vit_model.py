# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import unittest

import paddle
import pytest

from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from paddlefleet.models.vision.clip_vit_model import (
    CLIPViTModel,
    get_num_image_embeddings,
)
from paddlefleet.transformer.transformer_config import TransformerConfig


class TestCLIPViTModel(unittest.TestCase):
    """Test CLIP ViT model."""

    def setUp(self):
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            use_cpu_initialization=True,
        )
        transformer_layer_spec = get_gpt_layer_local_spec()
        self.model = CLIPViTModel(
            transformer_config,
            transformer_layer_spec,
            img_h=336,
            img_w=336,
            patch_dim=14,
        )

    def test_constructor(self):
        assert isinstance(self.model, CLIPViTModel)

        num_weights = sum([p.numel() for p in self.model.parameters()])

        assert num_weights == 174464

    def test_set_input_tensor(self):
        # [s, b, h] expected to the transformer.
        expected_shape = (577, 2, 64)
        input_tensor = paddle.zeros(expected_shape)

        self.model.set_input_tensor(input_tensor)
        assert self.model.decoder.input_tensor.shape == list(expected_shape)

    def test_forward(self):
        img = paddle.zeros((2, 3, 336, 336))

        out = self.model.forward(img)
        assert out.shape == [2, 577, 64]


@pytest.mark.internal
@pytest.mark.parametrize(
    "vision_model,pixel_shuffle,tile_tags,expected",
    [
        ("clip", False, False, 1024),
        ("internvit300M", False, False, 1024),
        ("clip", True, False, 256),
        ("internvit300M", True, True, 262),
    ],
)
def test_get_num_image_embeddings(
    vision_model, pixel_shuffle, tile_tags, expected
):
    assert (
        get_num_image_embeddings(
            448,
            448,
            14,
            vision_model,
            True,
            1,
            pixel_shuffle,
            tile_tags,
            0,
            "nemotron5",
        )
        == expected
    )


if __name__ == "__main__":
    unittest.main()
