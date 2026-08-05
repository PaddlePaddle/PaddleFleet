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

import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import paddle

from paddlefleet.models.qwen3_vl.embedding import AccuracyCompatiblePatchProjection
from paddlefleet.models.qwen3_vl.embedding import VisionEmbedding, VisionEmbeddingSpec
from paddlefleet.transformer.paddle_norm import LayerNorm
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSublayersSpec,
)


class TestAccuracyCompatiblePatchProjection(unittest.TestCase):
    def _make_layer(self, accuracy_compatible):
        layer = AccuracyCompatiblePatchProjection(
            3,
            8,
            kernel_size=[2, 4, 4],
            stride=[2, 4, 4],
            bias=True,
            accuracy_compatible=accuracy_compatible,
        )
        layer.eval()
        return layer

    def test_accuracy_path_matches_declared_linear_projection(self):
        paddle.seed(1234)
        layer = self._make_layer(accuracy_compatible=True)
        pixel_values = paddle.randn([5, 3, 2, 4, 4])
        actual = layer(pixel_values)
        rows = pixel_values.reshape([5, -1]).astype(layer.weight.dtype)
        weight = layer.weight.reshape([8, -1]).transpose([1, 0])
        expected = paddle.nn.functional.linear(rows, weight, layer.bias).reshape(
            [5, 8, 1, 1, 1]
        )
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())

    def test_accuracy_path_preserves_conv_parameter_shapes(self):
        layer = self._make_layer(accuracy_compatible=True)
        self.assertEqual(list(layer.weight.shape), [8, 3, 2, 4, 4])
        self.assertEqual(list(layer.bias.shape), [8])

    def test_accuracy_path_rejects_non_patch_geometry(self):
        layer = self._make_layer(accuracy_compatible=True)
        with self.assertRaisesRegex(ValueError, "one non-overlapping patch per row"):
            layer(paddle.randn([5, 3, 2, 8, 4]))


class TestAccuracyCompatiblePositionInterpolation(unittest.TestCase):
    def _make_embedding(self, *, accuracy_mode, enabled):
        config = TransformerConfig(
            num_hidden_layers=2, hidden_size=8, num_attention_heads=2
        )
        config.spatial_merge_size = 2
        config.patch_size = 4
        config.temporal_patch_size = 2
        config.in_channels = 3
        config.num_position_embeddings = 25
        config.use_accuracy_compatible = accuracy_mode
        with patch.dict(
            os.environ,
            {"PADDLEFLEET_ACCURACY_POSITION_INTERPOLATION": "1" if enabled else "0"},
        ):
            embedding = VisionEmbedding(config, VisionEmbeddingSpec())
        embedding.pos_embed.to(dtype="bfloat16")
        return embedding

    def test_requires_accuracy_mode_and_explicit_opt_in(self):
        self.assertFalse(
            self._make_embedding(accuracy_mode=False, enabled=True).accuracy_compatible_position_interpolation
        )
        self.assertFalse(
            self._make_embedding(accuracy_mode=True, enabled=False).accuracy_compatible_position_interpolation
        )
        self.assertTrue(
            self._make_embedding(accuracy_mode=True, enabled=True).accuracy_compatible_position_interpolation
        )

    def test_accuracy_path_keeps_interpolation_output_in_float32(self):
        embedding = self._make_embedding(accuracy_mode=True, enabled=True)
        grid_thw = paddle.to_tensor([[1, 4, 4]], dtype="int64")
        actual = embedding.fast_pos_embed_interpolate(grid_thw)
        self.assertEqual(actual.dtype, paddle.float32)
        self.assertEqual(list(actual.shape), [16, 8])

    def test_default_path_preserves_embedding_dtype(self):
        embedding = self._make_embedding(accuracy_mode=True, enabled=False)
        grid_thw = paddle.to_tensor([[1, 4, 4]], dtype="int64")
        actual = embedding.fast_pos_embed_interpolate(grid_thw)
        self.assertEqual(actual.dtype, paddle.bfloat16)
        self.assertEqual(list(actual.shape), [16, 8])


class TestLayerNormDtypeCast(unittest.TestCase):
    """LayerNorm.forward() must cast output to weight dtype."""

    def test_float16_input_cast_to_weight_dtype(self):
        config = TransformerConfig(
            num_hidden_layers=2, hidden_size=16, num_attention_heads=4
        )
        norm = LayerNorm(config=config)
        out = norm(paddle.randn([2, 16]).astype("float16"))
        self.assertEqual(out.dtype, norm.weight.dtype)

    def test_bfloat16_input_cast_to_weight_dtype(self):
        config = TransformerConfig(
            num_hidden_layers=2, hidden_size=16, num_attention_heads=4
        )
        norm = LayerNorm(config=config)
        out = norm(paddle.randn([2, 16]).astype("bfloat16"))
        self.assertEqual(out.dtype, paddle.float32)


class TestTransformerLayerMTP(unittest.TestCase):
    """TransformerLayer.forward() rotary trim/restore and mask handling."""

    def _make_layer(self, experimental_dataflow=True):
        config = TransformerConfig(
            num_hidden_layers=4,
            hidden_size=16,
            num_attention_heads=4,
            num_nextn_predict_layers=2,
            tensor_model_parallel_size=1,
            mtp_load_weight_only=False,
        )
        config.sequence_parallel = False
        config.experimental_dataflow = experimental_dataflow
        config.block_attention_residuals = False
        config.gpt_model_use_experimental_version = False
        config.recompute_granularity = None
        layer = TransformerLayer(
            config=config,
            sublayers_spec=TransformerLayerSublayersSpec(),
            layer_number=1,
        )
        layer.eval()
        return layer

    def test_rotary_pos_emb_trim_and_restore(self):
        layer = self._make_layer()
        B, S, H, n = 2, 8, 16, 2
        rotary = paddle.randn([B, S + 4, H])
        cos = paddle.randn([B, S + 4, H])
        sin = paddle.randn([B, S + 4, H])

        result = layer(
            {
                "hidden_states": paddle.randn([B * (n + 1), S, H]),
                "rotary_pos_emb": rotary,
                "rotary_pos_cos": cos,
                "rotary_pos_sin": sin,
            }
        )

        np.testing.assert_array_equal(
            result["rotary_pos_emb"].numpy(), rotary.numpy()
        )
        np.testing.assert_array_equal(
            result["rotary_pos_cos"].numpy(), cos.numpy()
        )
        np.testing.assert_array_equal(
            result["rotary_pos_sin"].numpy(), sin.numpy()
        )

    def test_non_experimental_dataflow_mask_concat(self):
        layer = self._make_layer(experimental_dataflow=False)
        B, S, H, n = 2, 8, 16, 2
        mask = paddle.randint(0, 10, [B * (n + 1), 1, S + n, 1])

        result = layer(
            {
                "hidden_states": paddle.randn([B * (n + 1), S, H]),
                "attn_mask_startend_row_indices": mask,
            }
        )

        self.assertEqual(
            list(result["attn_mask_startend_row_indices"].shape),
            [B * (n + 1), 1, S + n, 1],
        )

    def test_experimental_dataflow_mask_preserved(self):
        layer = self._make_layer(experimental_dataflow=True)
        B, S, H, n = 2, 8, 16, 2
        mask = paddle.randint(0, 10, [B * (n + 1), 1, S, 1])

        result = layer(
            {
                "hidden_states": paddle.randn([B * (n + 1), S, H]),
                "attn_mask_startend_row_indices": mask,
            }
        )

        np.testing.assert_array_equal(
            result["attn_mask_startend_row_indices"].numpy(), mask.numpy()
        )

    def test_dense_attention_mask_drops_collator_mtp_lookahead(self):
        layer = self._make_layer(experimental_dataflow=False)
        B, S, H, n = 2, 8, 16, 2
        mask = paddle.tril(paddle.ones([B, 1, S + n, S + n]))

        result = layer(
            {
                "hidden_states": paddle.randn([B * (n + 1), S, H]),
                "attention_mask": mask,
            }
        )

        expected = mask[:, :, :S, :S]
        self.assertEqual(list(result["attention_mask"].shape), [B, 1, S, S])
        np.testing.assert_array_equal(result["attention_mask"].numpy(), expected.numpy())


class TestMTPLayerForward(unittest.TestCase):
    """MultiTokenPredictionLayer.forward() via mock to bypass distributed init."""

    def _call_mtp_forward(
        self,
        sequence_parallel=False,
        tp_size=1,
        train_mtp_only=False,
        experimental_version=False,
    ):
        from paddlefleet.transformer.multi_token_prediction import (
            MultiTokenPredictionLayer,
        )

        config = TransformerConfig(
            num_hidden_layers=4,
            hidden_size=16,
            num_attention_heads=4,
            num_nextn_predict_layers=2,
            tensor_model_parallel_size=tp_size,
            mtp_load_weight_only=False,
        )
        config.sequence_parallel = sequence_parallel
        config.train_mtp_only = train_mtp_only
        config.recompute_granularity = None
        config.gpt_model_use_experimental_version = experimental_version

        mock = MagicMock(spec=MultiTokenPredictionLayer)
        mock.config = config
        mock.layer_number = 0
        mock.training = False
        mock._proj_and_transformer_layer = lambda **kw: kw.get(
            "hidden_states", kw.get("decoder_input")
        )
        return mock, MultiTokenPredictionLayer

    def test_rotary_trim_restore_no_sp(self):
        mock, MTP = self._call_mtp_forward()
        B, S, H, n = 2, 8, 16, 2
        rotary = paddle.randn([B, S + 4, H])
        cos = paddle.randn([B, S + 4, H])
        sin = paddle.randn([B, S + 4, H])

        result = MTP.forward(
            mock,
            {
                "hidden_states": paddle.randn([B * (n + 1), S, H]),
                "rotary_pos_emb": rotary,
                "rotary_pos_cos": cos,
                "rotary_pos_sin": sin,
            },
        )

        np.testing.assert_array_equal(
            result["rotary_pos_emb"].numpy(), rotary.numpy()
        )
        np.testing.assert_array_equal(
            result["rotary_pos_cos"].numpy(), cos.numpy()
        )
        np.testing.assert_array_equal(
            result["rotary_pos_sin"].numpy(), sin.numpy()
        )

    def test_rotary_trim_restore_sp(self):
        mock, MTP = self._call_mtp_forward(sequence_parallel=True, tp_size=2)
        B, H, n = 2, 16, 2
        local_seq = 12
        main_seq_len = local_seq // (n + 1) * 2  # 8
        rotary = paddle.randn([main_seq_len + 4, B, H])

        result = MTP.forward(
            mock,
            {
                "hidden_states": paddle.randn([local_seq, B, H]),
                "rotary_pos_emb": rotary,
            },
        )

        np.testing.assert_array_equal(
            result["rotary_pos_emb"].numpy(), rotary.numpy()
        )

    def test_mask_slicing_experimental_vs_non(self):
        B, S, H, n = 2, 8, 16, 2
        mtp_mask = paddle.randint(0, 10, [B * (n + 1), n, S, 4])

        for exp in (True, False):
            mock, MTP = self._call_mtp_forward(experimental_version=exp)
            result = MTP.forward(
                mock,
                {
                    "hidden_states": paddle.randn([B * (n + 1), S, H]),
                    "mtp_startend_row_indices_all": mtp_mask,
                },
            )
            self.assertIn("mtp_startend_row_indices_all", result)

    def test_train_mtp_only_mode(self):
        mock, MTP = self._call_mtp_forward(train_mtp_only=True)
        B, S, H, n = 2, 8, 16, 2
        rotary = paddle.randn([B, S + 4, H])

        result = MTP.forward(
            mock,
            {
                "hidden_states": paddle.randn([B * (n + 1), S, H]),
                "rotary_pos_emb": rotary,
            },
        )

        np.testing.assert_array_equal(
            result["rotary_pos_emb"].numpy(), rotary.numpy()
        )


if __name__ == "__main__":
    unittest.main()
