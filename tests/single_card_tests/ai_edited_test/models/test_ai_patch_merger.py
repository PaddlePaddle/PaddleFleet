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


class TestQwen3VLVisionPatchMergerSpec(unittest.TestCase):
    """Test Qwen3VLVisionPatchMergerSpec dataclass."""

    def test_defaults(self):
        from paddlefleet.models.qwen3_vl.patch_merger import (
            Qwen3VLVisionPatchMergerSpec,
        )

        spec = Qwen3VLVisionPatchMergerSpec()
        self.assertIsNotNone(spec.norm)


class TestQwen3VLVisionPathMergerInit(unittest.TestCase):
    """Test Qwen3VLVisionPathMerger initialization."""

    @patch("paddlefleet.models.qwen3_vl.patch_merger.build_layer")
    def test_basic_init(self, mock_build):
        from paddlefleet.models.qwen3_vl.patch_merger import (
            Qwen3VLVisionPathMerger,
        )

        mock_config = MagicMock()
        mock_config.hidden_size = 1024
        mock_config.out_hidden_size = 128
        mock_config.spatial_merge_size = 2
        mock_config.params_dtype = "float32"

        mock_build.return_value = MagicMock()
        spec = MagicMock()

        merger = Qwen3VLVisionPathMerger(
            config=mock_config, sublayers_spec=spec
        )
        expected_hidden = 1024 * (2**2)  # context_dim * spatial_merge_size^2
        self.assertEqual(merger.hidden_size, expected_hidden)

    @patch("paddlefleet.models.qwen3_vl.patch_merger.build_layer")
    def test_init_with_explicit_dims(self, mock_build):
        from paddlefleet.models.qwen3_vl.patch_merger import (
            Qwen3VLVisionPathMerger,
        )

        mock_config = MagicMock()
        mock_config.hidden_size = 1024
        mock_config.out_hidden_size = 128
        mock_config.spatial_merge_size = 2
        mock_config.params_dtype = "float32"

        mock_build.return_value = MagicMock()
        spec = MagicMock()

        merger = Qwen3VLVisionPathMerger(
            config=mock_config,
            sublayers_spec=spec,
            dim=256,
            context_dim=512,
        )
        expected_hidden = 512 * (2**2)
        self.assertEqual(merger.hidden_size, expected_hidden)


class TestQwen3VLVisionPathMergerInitWithPostshuffleNorm(unittest.TestCase):
    """Test Qwen3VLVisionPathMerger with postshuffle norm."""

    @patch("paddlefleet.models.qwen3_vl.patch_merger.build_layer")
    def test_postshuffle_norm(self, mock_build):
        from paddlefleet.models.qwen3_vl.patch_merger import (
            Qwen3VLVisionPathMerger,
        )

        mock_config = MagicMock()
        mock_config.hidden_size = 1024
        mock_config.out_hidden_size = 128
        mock_config.spatial_merge_size = 2
        mock_config.params_dtype = "float32"

        mock_build.return_value = MagicMock()
        spec = MagicMock()

        merger = Qwen3VLVisionPathMerger(
            config=mock_config,
            sublayers_spec=spec,
            use_postshuffle_norm=True,
        )
        self.assertTrue(merger.use_postshuffle_norm)


class TestQwen3VLVisionPathMergerForwardDictInput(unittest.TestCase):
    """Test Qwen3VLVisionPathMerger forward with dict input."""

    @patch("paddlefleet.models.qwen3_vl.patch_merger.build_layer")
    def test_forward_with_dict(self, mock_build):
        import paddle

        from paddlefleet.models.qwen3_vl.patch_merger import (
            Qwen3VLVisionPathMerger,
        )

        mock_config = MagicMock()
        mock_config.hidden_size = 64
        mock_config.out_hidden_size = 16
        mock_config.spatial_merge_size = 2
        mock_config.params_dtype = "float32"

        # norm should return a tensor; mlp should return (tensor, bias)
        mock_norm = MagicMock()
        mock_norm.return_value = paddle.randn([4, 256])
        mock_mlp = MagicMock()
        mock_mlp.return_value = (paddle.randn([4, 16]), None)
        mock_build.side_effect = [mock_norm, mock_mlp]

        spec = MagicMock()
        merger = Qwen3VLVisionPathMerger(
            config=mock_config, sublayers_spec=spec
        )

        x = {"hidden_states": paddle.randn([1, 4, 4, 64])}
        result, _ = merger(x)
        self.assertIsNotNone(result)


class TestQwen3VLVisionPathMergerForwardTensorInput(unittest.TestCase):
    """Test Qwen3VLVisionPathMerger forward with tensor input."""

    @patch("paddlefleet.models.qwen3_vl.patch_merger.build_layer")
    def test_forward_with_tensor(self, mock_build):
        import paddle

        from paddlefleet.models.qwen3_vl.patch_merger import (
            Qwen3VLVisionPathMerger,
        )

        mock_config = MagicMock()
        mock_config.hidden_size = 64
        mock_config.out_hidden_size = 16
        mock_config.spatial_merge_size = 2
        mock_config.params_dtype = "float32"

        # norm should return a tensor; mlp should return (tensor, bias)
        mock_norm = MagicMock()
        mock_norm.return_value = paddle.randn([4, 256])
        mock_mlp = MagicMock()
        mock_mlp.return_value = (paddle.randn([4, 16]), None)
        mock_build.side_effect = [mock_norm, mock_mlp]

        spec = MagicMock()
        merger = Qwen3VLVisionPathMerger(
            config=mock_config, sublayers_spec=spec
        )

        x = paddle.randn([4, 256])
        result, _ = merger(x)
        self.assertIsNotNone(result)


class TestQwen3VLVisionPathMergerForwardWithBias(unittest.TestCase):
    """Test Qwen3VLVisionPathMerger forward when bias is not None."""

    @patch("paddlefleet.models.qwen3_vl.patch_merger.build_layer")
    def test_forward_adds_bias(self, mock_build):
        import paddle

        from paddlefleet.models.qwen3_vl.patch_merger import (
            Qwen3VLVisionPathMerger,
        )

        mock_config = MagicMock()
        mock_config.hidden_size = 64
        mock_config.out_hidden_size = 16
        mock_config.spatial_merge_size = 2
        mock_config.params_dtype = "float32"

        mock_norm = MagicMock()
        mock_norm.return_value = paddle.randn([4, 256])
        mock_mlp = MagicMock()
        output = paddle.randn([4, 16])
        bias = paddle.randn([16])
        mock_mlp.return_value = (output, bias)
        mock_build.side_effect = [mock_norm, mock_mlp]

        spec = MagicMock()
        merger = Qwen3VLVisionPathMerger(
            config=mock_config, sublayers_spec=spec
        )

        x = paddle.randn([4, 256])
        result, _ = merger(x)
        self.assertIsNotNone(result)


class TestQwen3VLVisionPathMergerPostshuffleNormForward(unittest.TestCase):
    """Test Qwen3VLVisionPathMerger forward with postshuffle norm."""

    @patch("paddlefleet.models.qwen3_vl.patch_merger.build_layer")
    def test_forward_postshuffle(self, mock_build):
        import paddle

        from paddlefleet.models.qwen3_vl.patch_merger import (
            Qwen3VLVisionPathMerger,
        )

        mock_config = MagicMock()
        mock_config.hidden_size = 64
        mock_config.out_hidden_size = 16
        mock_config.spatial_merge_size = 2
        mock_config.params_dtype = "float32"

        mock_norm = MagicMock()
        mock_norm.return_value = paddle.randn([4, 256])
        mock_mlp = MagicMock()
        mock_mlp.return_value = (paddle.randn([4, 16]), None)
        mock_build.side_effect = [mock_norm, mock_mlp]

        spec = MagicMock()
        merger = Qwen3VLVisionPathMerger(
            config=mock_config,
            sublayers_spec=spec,
            use_postshuffle_norm=True,
        )

        x = paddle.randn([4, 256])
        result, _ = merger(x)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
