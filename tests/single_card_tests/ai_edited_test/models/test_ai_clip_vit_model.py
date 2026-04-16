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
from unittest.mock import MagicMock, patch


class TestGetNumImageEmbeddingsClip(unittest.TestCase):
    """Test get_num_image_embeddings with clip vision model type."""

    def test_clip_with_class_token(self):
        from paddlefleet.models.vision.clip_vit_model import (
            get_num_image_embeddings,
        )

        result = get_num_image_embeddings(
            img_h=336,
            img_w=336,
            patch_dim=14,
            vision_model_type="clip",
            disable_vision_class_token=False,
            class_token_len=1,
            pixel_shuffle=False,
        )
        # num_patches = (336/14) * (336/14) = 24*24 = 576
        # 576 + 1 = 577
        self.assertEqual(result, 577)

    def test_clip_without_class_token(self):
        from paddlefleet.models.vision.clip_vit_model import (
            get_num_image_embeddings,
        )

        result = get_num_image_embeddings(
            img_h=336,
            img_w=336,
            patch_dim=14,
            vision_model_type="clip",
            disable_vision_class_token=True,
            class_token_len=1,
            pixel_shuffle=False,
        )
        self.assertEqual(result, 576)

    def test_clip_with_pixel_shuffle(self):
        from paddlefleet.models.vision.clip_vit_model import (
            get_num_image_embeddings,
        )

        result = get_num_image_embeddings(
            img_h=336,
            img_w=336,
            patch_dim=14,
            vision_model_type="clip",
            disable_vision_class_token=False,
            class_token_len=1,
            pixel_shuffle=True,
        )
        # (576+1) * 0.5^2 = 577 * 0.25 = 144.25 -> int = 144
        self.assertEqual(result, 144)


class TestGetNumImageEmbeddingsSiglip(unittest.TestCase):
    """Test get_num_image_embeddings with siglip vision model type."""

    def test_siglip_no_class_token(self):
        from paddlefleet.models.vision.clip_vit_model import (
            get_num_image_embeddings,
        )

        result = get_num_image_embeddings(
            img_h=336,
            img_w=336,
            patch_dim=14,
            vision_model_type="siglip",
            disable_vision_class_token=False,
            class_token_len=1,
            pixel_shuffle=False,
        )
        # SigLIP never keeps class token
        self.assertEqual(result, 576)


class TestGetNumImageEmbeddingsInternViT(unittest.TestCase):
    """Test get_num_image_embeddings with internvit vision model types."""

    def test_internvit_with_class_token(self):
        from paddlefleet.models.vision.clip_vit_model import (
            get_num_image_embeddings,
        )

        result = get_num_image_embeddings(
            img_h=224,
            img_w=224,
            patch_dim=14,
            vision_model_type="internvit",
            disable_vision_class_token=False,
            class_token_len=1,
            pixel_shuffle=False,
        )
        # (224/14)^2 + 1 = 256 + 1 = 257
        self.assertEqual(result, 257)

    def test_internvit300m(self):
        from paddlefleet.models.vision.clip_vit_model import (
            get_num_image_embeddings,
        )

        result = get_num_image_embeddings(
            img_h=336,
            img_w=336,
            patch_dim=14,
            vision_model_type="internvit300M",
            disable_vision_class_token=True,
            class_token_len=1,
            pixel_shuffle=False,
        )
        self.assertEqual(result, 576)


class TestGetNumImageEmbeddingsRadio(unittest.TestCase):
    """Test get_num_image_embeddings with radio vision model type."""

    def test_radio_with_class_token(self):
        from paddlefleet.models.vision.clip_vit_model import (
            get_num_image_embeddings,
        )

        result = get_num_image_embeddings(
            img_h=224,
            img_w=224,
            patch_dim=16,
            vision_model_type="radio",
            disable_vision_class_token=False,
            class_token_len=8,
            pixel_shuffle=False,
        )
        # (224/16)^2 + 8 = 196 + 8 = 204
        self.assertEqual(result, 204)


class TestGetNumImageEmbeddingsCRadioG(unittest.TestCase):
    """Test get_num_image_embeddings with cradio-g vision model type."""

    def test_cradio_g(self):
        from paddlefleet.models.vision.clip_vit_model import (
            get_num_image_embeddings,
        )

        result = get_num_image_embeddings(
            img_h=224,
            img_w=224,
            patch_dim=16,
            vision_model_type="cradio-g",
            disable_vision_class_token=False,
            class_token_len=1,
            pixel_shuffle=False,
        )
        # cradio-g forces class_token_len=8
        self.assertEqual(result, 204)


class TestGetNumImageEmbeddingsTileTags(unittest.TestCase):
    """Test get_num_image_embeddings with tile tags."""

    def test_tile_tags_llama3p1(self):
        from paddlefleet.models.vision.clip_vit_model import (
            get_num_image_embeddings,
        )

        result = get_num_image_embeddings(
            img_h=336,
            img_w=336,
            patch_dim=14,
            vision_model_type="clip",
            disable_vision_class_token=False,
            class_token_len=1,
            pixel_shuffle=False,
            use_tile_tags=True,
            tokenizer_type="llama3p1",
        )
        self.assertGreater(result, 577)

    def test_tile_tags_qwen2p0(self):
        from paddlefleet.models.vision.clip_vit_model import (
            get_num_image_embeddings,
        )

        result = get_num_image_embeddings(
            img_h=336,
            img_w=336,
            patch_dim=14,
            vision_model_type="clip",
            disable_vision_class_token=False,
            class_token_len=1,
            pixel_shuffle=False,
            use_tile_tags=True,
            tokenizer_type="qwen2p0",
        )
        self.assertGreater(result, 577)

    def test_tile_tags_unknown_tokenizer_raises(self):
        from paddlefleet.models.vision.clip_vit_model import (
            get_num_image_embeddings,
        )

        with self.assertRaises(ValueError):
            get_num_image_embeddings(
                img_h=336,
                img_w=336,
                patch_dim=14,
                vision_model_type="clip",
                disable_vision_class_token=False,
                class_token_len=1,
                pixel_shuffle=False,
                use_tile_tags=True,
                tokenizer_type="unknown",
            )


class TestGetNumImageEmbeddingsErrorCases(unittest.TestCase):
    """Test get_num_image_embeddings error cases."""

    def test_unknown_vision_model_raises(self):
        from paddlefleet.models.vision.clip_vit_model import (
            get_num_image_embeddings,
        )

        with self.assertRaises(NotImplementedError):
            get_num_image_embeddings(
                img_h=336,
                img_w=336,
                patch_dim=14,
                vision_model_type="unknown_type",
                disable_vision_class_token=False,
                class_token_len=1,
                pixel_shuffle=False,
            )

    def test_max_num_tiles_over_100_raises(self):
        from paddlefleet.models.vision.clip_vit_model import (
            get_num_image_embeddings,
        )

        with self.assertRaises(ValueError):
            get_num_image_embeddings(
                img_h=336,
                img_w=336,
                patch_dim=14,
                vision_model_type="clip",
                disable_vision_class_token=False,
                class_token_len=1,
                pixel_shuffle=False,
                use_tile_tags=True,
                tokenizer_type="llama3p1",
                max_num_tiles=200,
            )


class TestCLIPViTModelInit(unittest.TestCase):
    """Test CLIPViTModel initialization."""

    @patch(
        "paddlefleet.models.vision.clip_vit_model.has_config_logger_enabled",
        return_value=False,
    )
    @patch("paddlefleet.models.vision.clip_vit_model.build_spec_layer")
    @patch("paddlefleet.models.vision.clip_vit_model.TransformerBlock")
    def test_init_clip(self, mock_tb, mock_build, mock_log):
        from paddlefleet.models.vision.clip_vit_model import CLIPViTModel

        mock_config = MagicMock()
        mock_config.hidden_size = 768
        mock_config.rms_norm_eps = 1e-5
        mock_config.params_dtype = "float32"
        mock_tb.return_value = MagicMock()
        mock_build.return_value = MagicMock()

        model = CLIPViTModel(
            transformer_config=mock_config,
            transformer_layer_spec=MagicMock(),
            model_subtype="clip",
            patch_dim=14,
            img_h=336,
            img_w=336,
        )
        self.assertEqual(model.visual_hidden_size, 768)
        self.assertEqual(model.num_patches, 576)
        self.assertTrue(model.add_class_token)

    @patch(
        "paddlefleet.models.vision.clip_vit_model.has_config_logger_enabled",
        return_value=False,
    )
    @patch("paddlefleet.models.vision.clip_vit_model.build_spec_layer")
    @patch("paddlefleet.models.vision.clip_vit_model.TransformerBlock")
    def test_init_siglip_no_class_token(self, mock_tb, mock_build, mock_log):
        from paddlefleet.models.vision.clip_vit_model import CLIPViTModel

        mock_config = MagicMock()
        mock_config.hidden_size = 768
        mock_config.rms_norm_eps = 1e-5
        mock_config.params_dtype = "float32"
        mock_tb.return_value = MagicMock()
        mock_build.return_value = MagicMock()

        model = CLIPViTModel(
            transformer_config=mock_config,
            transformer_layer_spec=MagicMock(),
            model_subtype="siglip",
            patch_dim=14,
            img_h=336,
            img_w=336,
            add_class_token=False,
            class_token_len=0,
        )
        self.assertFalse(model.add_class_token)
        self.assertEqual(model.class_token_len, 0)

    @patch(
        "paddlefleet.models.vision.clip_vit_model.has_config_logger_enabled",
        return_value=False,
    )
    @patch("paddlefleet.models.vision.clip_vit_model.build_spec_layer")
    @patch("paddlefleet.models.vision.clip_vit_model.TransformerBlock")
    def test_unsupported_model_type_raises(self, mock_tb, mock_build, mock_log):
        from paddlefleet.models.vision.clip_vit_model import CLIPViTModel

        mock_config = MagicMock()
        mock_config.hidden_size = 768
        mock_config.rms_norm_eps = 1e-5
        mock_config.params_dtype = "float32"

        with self.assertRaises(AssertionError):
            CLIPViTModel(
                transformer_config=mock_config,
                transformer_layer_spec=MagicMock(),
                model_subtype="unsupported",
            )


if __name__ == "__main__":
    unittest.main()
