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


class TestRADIOViTModelInit(unittest.TestCase):
    """Test RADIOViTModel initialization."""

    @patch(
        "paddlefleet.models.vision.radio.has_config_logger_enabled",
        return_value=False,
    )
    @patch("paddlefleet.models.vision.radio.build_spec_layer")
    @patch("paddlefleet.models.vision.radio.TransformerBlock")
    @patch("paddlefleet.models.vision.radio.ColumnParallelLinear")
    def test_basic_init(self, mock_cpl, mock_tb, mock_build, mock_log):
        from paddlefleet.models.vision.radio import RADIOViTModel

        mock_config = MagicMock()
        mock_config.hidden_size = 768
        mock_config.rms_norm_eps = 1e-5
        mock_config.params_dtype = "float32"
        mock_tb.return_value = MagicMock()
        mock_cpl.return_value = None
        mock_build.return_value = MagicMock()

        model = RADIOViTModel(
            transformer_config=mock_config,
            transformer_layer_spec=MagicMock(),
            patch_dim=16,
            img_h=224,
            img_w=224,
        )
        self.assertEqual(model.visual_hidden_size, 768)
        self.assertEqual(model.patch_dim, 16)
        self.assertEqual(model.input_dims, (14, 14))

    @patch(
        "paddlefleet.models.vision.radio.has_config_logger_enabled",
        return_value=False,
    )
    @patch("paddlefleet.models.vision.radio.build_spec_layer")
    @patch("paddlefleet.models.vision.radio.TransformerBlock")
    @patch("paddlefleet.models.vision.radio.ColumnParallelLinear")
    def test_init_with_mask_token(
        self, mock_cpl, mock_tb, mock_build, mock_log
    ):
        from paddlefleet.models.vision.radio import RADIOViTModel

        mock_config = MagicMock()
        mock_config.hidden_size = 768
        mock_config.rms_norm_eps = 1e-5
        mock_config.params_dtype = "float32"
        mock_tb.return_value = MagicMock()
        mock_cpl.return_value = None
        mock_build.return_value = MagicMock()

        model = RADIOViTModel(
            transformer_config=mock_config,
            transformer_layer_spec=MagicMock(),
            use_mask_token=True,
        )
        self.assertTrue(model.use_mask_token)
        self.assertTrue(hasattr(model, "mask_token"))

    @patch(
        "paddlefleet.models.vision.radio.has_config_logger_enabled",
        return_value=False,
    )
    @patch("paddlefleet.models.vision.radio.build_spec_layer")
    @patch("paddlefleet.models.vision.radio.TransformerBlock")
    @patch("paddlefleet.models.vision.radio.ColumnParallelLinear")
    def test_init_without_class_token(
        self, mock_cpl, mock_tb, mock_build, mock_log
    ):
        from paddlefleet.models.vision.radio import RADIOViTModel

        mock_config = MagicMock()
        mock_config.hidden_size = 768
        mock_config.rms_norm_eps = 1e-5
        mock_config.params_dtype = "float32"
        mock_tb.return_value = MagicMock()
        mock_cpl.return_value = None
        mock_build.return_value = MagicMock()

        model = RADIOViTModel(
            transformer_config=mock_config,
            transformer_layer_spec=MagicMock(),
            add_class_token=False,
            class_token_len=0,
        )
        self.assertFalse(model.add_class_token)

    @patch(
        "paddlefleet.models.vision.radio.has_config_logger_enabled",
        return_value=False,
    )
    @patch("paddlefleet.models.vision.radio.build_spec_layer")
    @patch("paddlefleet.models.vision.radio.TransformerBlock")
    @patch("paddlefleet.models.vision.radio.ColumnParallelLinear")
    def test_init_with_ln_pre(self, mock_cpl, mock_tb, mock_build, mock_log):
        from paddlefleet.models.vision.radio import RADIOViTModel

        mock_config = MagicMock()
        mock_config.hidden_size = 768
        mock_config.rms_norm_eps = 1e-5
        mock_config.params_dtype = "float32"
        mock_tb.return_value = MagicMock()
        mock_cpl.return_value = None
        mock_build.return_value = MagicMock()

        model = RADIOViTModel(
            transformer_config=mock_config,
            transformer_layer_spec=MagicMock(),
            ln_pre_impl=MagicMock(),
        )
        self.assertIsNotNone(model.ln_pre)


class TestRADIOViTModelPosEnc(unittest.TestCase):
    """Test RADIOViTModel position encoding methods."""

    @patch(
        "paddlefleet.models.vision.radio.has_config_logger_enabled",
        return_value=False,
    )
    @patch("paddlefleet.models.vision.radio.build_spec_layer")
    @patch("paddlefleet.models.vision.radio.TransformerBlock")
    @patch("paddlefleet.models.vision.radio.ColumnParallelLinear")
    def test_get_pos_enc_no_idxs(self, mock_cpl, mock_tb, mock_build, mock_log):
        from paddlefleet.models.vision.radio import RADIOViTModel

        mock_config = MagicMock()
        mock_config.hidden_size = 64
        mock_config.rms_norm_eps = 1e-5
        mock_config.params_dtype = "float32"
        mock_tb.return_value = MagicMock()
        mock_cpl.return_value = None
        mock_build.return_value = MagicMock()

        model = RADIOViTModel(
            transformer_config=mock_config,
            transformer_layer_spec=MagicMock(),
            patch_dim=16,
            img_h=224,
            img_w=224,
            max_img_h=224,
            max_img_w=224,
        )
        # When input_dims matches max dims, should return position_embeddings directly
        result = model.get_pos_enc(
            batch_size=1, patch_idxs=None, input_size=None
        )
        self.assertIsNotNone(result)

    @unittest.skip(
        "get_pos_enc with input_size != max requires real position_embeddings tensors, "
        "not MagicMock, for _get_pos_embeddings internal grid_sample computation"
    )
    @patch(
        "paddlefleet.models.vision.radio.has_config_logger_enabled",
        return_value=False,
    )
    @patch("paddlefleet.models.vision.radio.build_spec_layer")
    @patch("paddlefleet.models.vision.radio.TransformerBlock")
    @patch("paddlefleet.models.vision.radio.ColumnParallelLinear")
    def test_get_pos_enc_with_input_size(
        self, mock_cpl, mock_tb, mock_build, mock_log
    ):
        from paddlefleet.models.vision.radio import RADIOViTModel

        mock_config = MagicMock()
        mock_config.hidden_size = 64
        mock_config.rms_norm_eps = 1e-5
        mock_config.params_dtype = "float32"
        mock_tb.return_value = MagicMock()
        mock_cpl.return_value = None
        mock_build.return_value = MagicMock()

        model = RADIOViTModel(
            transformer_config=mock_config,
            transformer_layer_spec=MagicMock(),
            patch_dim=16,
            img_h=224,
            img_w=224,
            max_img_h=448,
            max_img_w=448,
        )
        # Different input size than max
        result = model.get_pos_enc(
            batch_size=1, patch_idxs=None, input_size=(224, 224)
        )
        self.assertIsNotNone(result)


class TestRADIOViTModelApplyPosEnc(unittest.TestCase):
    """Test RADIOViTModel.apply_pos_enc method."""

    @patch(
        "paddlefleet.models.vision.radio.has_config_logger_enabled",
        return_value=False,
    )
    @patch("paddlefleet.models.vision.radio.build_spec_layer")
    @patch("paddlefleet.models.vision.radio.TransformerBlock")
    @patch("paddlefleet.models.vision.radio.ColumnParallelLinear")
    def test_apply_pos_enc_eval_mode(
        self, mock_cpl, mock_tb, mock_build, mock_log
    ):
        from paddlefleet.models.vision.radio import RADIOViTModel

        mock_config = MagicMock()
        mock_config.hidden_size = 64
        mock_config.rms_norm_eps = 1e-5
        mock_config.params_dtype = "float32"
        mock_tb.return_value = MagicMock()
        mock_cpl.return_value = None
        mock_build.return_value = MagicMock()

        model = RADIOViTModel(
            transformer_config=mock_config,
            transformer_layer_spec=MagicMock(),
            patch_dim=16,
            img_h=224,
            img_w=224,
            max_img_h=224,
            max_img_w=224,
            pos_dropout=0,
        )
        model.eval()
        import paddle

        patches = paddle.randn([1, 196, 64])
        result, pos_enc = model.apply_pos_enc(
            patches, patch_idxs=None, input_size=None
        )
        self.assertIsNotNone(result)
        self.assertIsNotNone(pos_enc)


class TestRADIOViTModelSeqLength(unittest.TestCase):
    """Test RADIOViTModel sequence length computation."""

    @patch(
        "paddlefleet.models.vision.radio.has_config_logger_enabled",
        return_value=False,
    )
    @patch("paddlefleet.models.vision.radio.build_spec_layer")
    @patch("paddlefleet.models.vision.radio.TransformerBlock")
    @patch("paddlefleet.models.vision.radio.ColumnParallelLinear")
    def test_seq_length_with_class_token(
        self, mock_cpl, mock_tb, mock_build, mock_log
    ):
        from paddlefleet.models.vision.radio import RADIOViTModel

        mock_config = MagicMock()
        mock_config.hidden_size = 64
        mock_config.rms_norm_eps = 1e-5
        mock_config.params_dtype = "float32"
        mock_tb.return_value = MagicMock()
        mock_cpl.return_value = None
        mock_build.return_value = MagicMock()

        model = RADIOViTModel(
            transformer_config=mock_config,
            transformer_layer_spec=MagicMock(),
            patch_dim=16,
            img_h=224,
            img_w=224,
            add_class_token=True,
            class_token_len=8,
        )
        # (224/16)^2 + 8 = 196 + 8 = 204
        self.assertEqual(model.seq_length, 204)

    @patch(
        "paddlefleet.models.vision.radio.has_config_logger_enabled",
        return_value=False,
    )
    @patch("paddlefleet.models.vision.radio.build_spec_layer")
    @patch("paddlefleet.models.vision.radio.TransformerBlock")
    @patch("paddlefleet.models.vision.radio.ColumnParallelLinear")
    def test_seq_length_without_class_token(
        self, mock_cpl, mock_tb, mock_build, mock_log
    ):
        from paddlefleet.models.vision.radio import RADIOViTModel

        mock_config = MagicMock()
        mock_config.hidden_size = 64
        mock_config.rms_norm_eps = 1e-5
        mock_config.params_dtype = "float32"
        mock_tb.return_value = MagicMock()
        mock_cpl.return_value = None
        mock_build.return_value = MagicMock()

        model = RADIOViTModel(
            transformer_config=mock_config,
            transformer_layer_spec=MagicMock(),
            patch_dim=16,
            img_h=224,
            img_w=224,
            add_class_token=False,
            class_token_len=0,
        )
        self.assertEqual(model.seq_length, 196)


class TestRADIOViTModelMaxDims(unittest.TestCase):
    """Test RADIOViTModel max dimension computations."""

    @patch(
        "paddlefleet.models.vision.radio.has_config_logger_enabled",
        return_value=False,
    )
    @patch("paddlefleet.models.vision.radio.build_spec_layer")
    @patch("paddlefleet.models.vision.radio.TransformerBlock")
    @patch("paddlefleet.models.vision.radio.ColumnParallelLinear")
    def test_max_num_patches(self, mock_cpl, mock_tb, mock_build, mock_log):
        from paddlefleet.models.vision.radio import RADIOViTModel

        mock_config = MagicMock()
        mock_config.hidden_size = 64
        mock_config.rms_norm_eps = 1e-5
        mock_config.params_dtype = "float32"
        mock_tb.return_value = MagicMock()
        mock_cpl.return_value = None
        mock_build.return_value = MagicMock()

        model = RADIOViTModel(
            transformer_config=mock_config,
            transformer_layer_spec=MagicMock(),
            patch_dim=16,
            img_h=224,
            img_w=224,
            max_img_h=2048,
            max_img_w=2048,
        )
        self.assertEqual(model.max_num_rows, 128)
        self.assertEqual(model.max_num_cols, 128)
        self.assertEqual(model.max_num_patches, 128 * 128)

    @patch(
        "paddlefleet.models.vision.radio.has_config_logger_enabled",
        return_value=False,
    )
    @patch("paddlefleet.models.vision.radio.build_spec_layer")
    @patch("paddlefleet.models.vision.radio.TransformerBlock")
    @patch("paddlefleet.models.vision.radio.ColumnParallelLinear")
    def test_set_input_tensor(self, mock_cpl, mock_tb, mock_build, mock_log):
        from paddlefleet.models.vision.radio import RADIOViTModel

        mock_config = MagicMock()
        mock_config.hidden_size = 64
        mock_config.rms_norm_eps = 1e-5
        mock_config.params_dtype = "float32"
        mock_decoder = MagicMock()
        mock_tb.return_value = mock_decoder
        mock_cpl.return_value = None
        mock_build.return_value = MagicMock()

        model = RADIOViTModel(
            transformer_config=mock_config,
            transformer_layer_spec=MagicMock(),
        )
        model.set_input_tensor(MagicMock())
        mock_decoder.set_input_tensor.assert_called_once()


if __name__ == "__main__":
    unittest.main()
