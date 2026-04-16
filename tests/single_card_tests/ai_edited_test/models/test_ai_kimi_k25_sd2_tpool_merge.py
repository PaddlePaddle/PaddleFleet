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


class TestKimiK25VisionSd2TpoolMergerInit(unittest.TestCase):
    """Test KimiK25VisionSd2TpoolMerger initialization."""

    def test_basic_init(self):
        from paddlefleet.models.kimi_k25.sd2_tpool_merge import (
            KimiK25VisionSd2TpoolMerger,
        )

        mock_config = MagicMock()
        mock_config.merge_kernel_size = (2, 2)

        merger = KimiK25VisionSd2TpoolMerger(config=mock_config)
        self.assertEqual(merger.merge_kernel_size, (2, 2))


class TestKimiK25VisionSd2TpoolMergerForward(unittest.TestCase):
    """Test KimiK25VisionSd2TpoolMerger forward method."""

    def test_forward_single_temporal(self):
        import paddle

        from paddlefleet.models.kimi_k25.sd2_tpool_merge import (
            KimiK25VisionSd2TpoolMerger,
        )

        mock_config = MagicMock()
        mock_config.merge_kernel_size = (2, 2)

        merger = KimiK25VisionSd2TpoolMerger(config=mock_config)
        # 1 temporal frame, 4x4 spatial, 64 hidden
        hidden_states = paddle.randn([1, 1 * 4 * 4, 64])
        grid_thws = paddle.to_tensor([[1, 4, 4]])
        result = merger(
            {"hidden_states": hidden_states, "grid_thws": grid_thws}
        )
        self.assertIn("hidden_states", result)
        # After merging (2,2) -> (2,2) spatial, new_height=2, new_width=2
        self.assertIsInstance(result["hidden_states"], list)

    def test_forward_multiple_temporal(self):
        import paddle

        from paddlefleet.models.kimi_k25.sd2_tpool_merge import (
            KimiK25VisionSd2TpoolMerger,
        )

        mock_config = MagicMock()
        mock_config.merge_kernel_size = (2, 2)

        merger = KimiK25VisionSd2TpoolMerger(config=mock_config)
        # 2 temporal frames, 4x4 spatial, 64 hidden
        hidden_states = paddle.randn([1, 2 * 4 * 4, 64])
        grid_thws = paddle.to_tensor([[2, 4, 4]])
        result = merger(
            {"hidden_states": hidden_states, "grid_thws": grid_thws}
        )
        self.assertIn("hidden_states", result)

    def test_forward_preserves_other_keys(self):
        import paddle

        from paddlefleet.models.kimi_k25.sd2_tpool_merge import (
            KimiK25VisionSd2TpoolMerger,
        )

        mock_config = MagicMock()
        mock_config.merge_kernel_size = (2, 2)

        merger = KimiK25VisionSd2TpoolMerger(config=mock_config)
        hidden_states = paddle.randn([1, 16, 64])
        grid_thws = paddle.to_tensor([[1, 4, 4]])
        dict_args = {
            "hidden_states": hidden_states,
            "grid_thws": grid_thws,
            "attention_mask": paddle.ones([1, 16]),
        }
        result = merger(dict_args)
        self.assertIn("attention_mask", result)


class TestKimiK25VisionPatchMergerSpec(unittest.TestCase):
    """Test KimiK25VisionPatchMergerSpec dataclass."""

    def test_defaults(self):
        from paddlefleet.models.kimi_k25.sd2_tpool_merge import (
            KimiK25VisionPatchMergerSpec,
        )

        spec = KimiK25VisionPatchMergerSpec()
        self.assertIsNotNone(spec.norm)


class TestKimiK25VisionPathMergerInit(unittest.TestCase):
    """Test KimiK25VisionPathMerger initialization."""

    @patch("paddlefleet.models.kimi_k25.sd2_tpool_merge.build_spec_layer")
    def test_basic_init(self, mock_build):
        from paddlefleet.models.kimi_k25.sd2_tpool_merge import (
            KimiK25VisionPathMerger,
        )

        mock_config = MagicMock()
        mock_config.merge_kernel_size = (2, 2)
        mock_config.projector_ln_eps = 1e-5
        mock_config.mm_hidden_size = 1024
        mock_config.text_hidden_size = 4096
        mock_config.params_dtype = "float32"
        mock_build.return_value = MagicMock()

        spec = MagicMock()
        merger = KimiK25VisionPathMerger(
            config=mock_config, sublayers_spec=spec
        )
        expected_hidden = 1024 * 2 * 2  # mm_hidden_size * kernel_h * kernel_w
        self.assertEqual(merger.hidden_size, expected_hidden)

    @patch("paddlefleet.models.kimi_k25.sd2_tpool_merge.build_spec_layer")
    def test_kernel_size_computation(self, mock_build):
        from paddlefleet.models.kimi_k25.sd2_tpool_merge import (
            KimiK25VisionPathMerger,
        )

        mock_config = MagicMock()
        mock_config.merge_kernel_size = (4, 2)
        mock_config.projector_ln_eps = 1e-5
        mock_config.mm_hidden_size = 512
        mock_config.text_hidden_size = 2048
        mock_config.params_dtype = "float32"
        mock_build.return_value = MagicMock()

        spec = MagicMock()
        merger = KimiK25VisionPathMerger(
            config=mock_config, sublayers_spec=spec
        )
        expected_hidden = 512 * 4 * 2
        self.assertEqual(merger.hidden_size, expected_hidden)


class TestKimiK25VisionPathMergerForwardListInput(unittest.TestCase):
    """Test KimiK25VisionPathMerger forward with list input."""

    @patch("paddlefleet.models.kimi_k25.sd2_tpool_merge.build_spec_layer")
    def test_forward_with_list(self, mock_build):
        import paddle

        from paddlefleet.models.kimi_k25.sd2_tpool_merge import (
            KimiK25VisionPathMerger,
        )

        mock_config = MagicMock()
        mock_config.merge_kernel_size = (2, 2)
        mock_config.projector_ln_eps = 1e-5
        mock_config.mm_hidden_size = 64
        mock_config.text_hidden_size = 128
        mock_config.params_dtype = "float32"

        # pre_norm should return a tensor (passthrough), proj returns (tensor, bias)
        mock_norm = MagicMock()
        mock_norm.return_value = paddle.randn([4, 256])
        mock_proj = MagicMock()
        mock_proj.return_value = (paddle.randn([4, 128]), None)
        mock_build.side_effect = [mock_norm, mock_proj]

        spec = MagicMock()
        merger = KimiK25VisionPathMerger(
            config=mock_config, sublayers_spec=spec
        )
        # hidden_size = 64 * 2 * 2 = 256
        item = paddle.randn([4, 256])
        dict_args = {"hidden_states": [item]}
        result = merger(dict_args)
        self.assertIn("hidden_states", result)


class TestKimiK25VisionPathMergerForwardTensorInput(unittest.TestCase):
    """Test KimiK25VisionPathMerger forward with tensor input."""

    @patch("paddlefleet.models.kimi_k25.sd2_tpool_merge.build_spec_layer")
    def test_forward_with_tensor(self, mock_build):
        import paddle

        from paddlefleet.models.kimi_k25.sd2_tpool_merge import (
            KimiK25VisionPathMerger,
        )

        mock_config = MagicMock()
        mock_config.merge_kernel_size = (2, 2)
        mock_config.projector_ln_eps = 1e-5
        mock_config.mm_hidden_size = 64
        mock_config.text_hidden_size = 128
        mock_config.params_dtype = "float32"

        # pre_norm should return a tensor; proj returns (tensor, bias)
        mock_norm = MagicMock()
        mock_norm.return_value = paddle.randn([2, 4, 4, 256])
        mock_proj = MagicMock()
        mock_proj.return_value = (paddle.randn([2, 4, 128]), None)
        mock_build.side_effect = [mock_norm, mock_proj]

        spec = MagicMock()
        merger = KimiK25VisionPathMerger(
            config=mock_config, sublayers_spec=spec
        )
        # B=2, N=4, N_k=4, C=64 -> hidden_size = 64 * 4 = 256
        item = paddle.randn([2, 4, 4, 64])
        dict_args = {"hidden_states": item}
        result = merger(dict_args)
        self.assertIn("hidden_states", result)


class TestKimiK25VisionPathMergerForwardPreservesKeys(unittest.TestCase):
    """Test that merger preserves original dict keys."""

    @patch("paddlefleet.models.kimi_k25.sd2_tpool_merge.build_spec_layer")
    def test_preserves_keys(self, mock_build):
        import paddle

        from paddlefleet.models.kimi_k25.sd2_tpool_merge import (
            KimiK25VisionPathMerger,
        )

        mock_config = MagicMock()
        mock_config.merge_kernel_size = (2, 2)
        mock_config.projector_ln_eps = 1e-5
        mock_config.mm_hidden_size = 64
        mock_config.text_hidden_size = 128
        mock_config.params_dtype = "float32"

        mock_norm = MagicMock()
        mock_norm.return_value = paddle.randn([4, 256])
        mock_proj = MagicMock()
        mock_proj.return_value = (paddle.randn([4, 128]), None)
        mock_build.side_effect = [mock_norm, mock_proj]

        spec = MagicMock()
        merger = KimiK25VisionPathMerger(
            config=mock_config, sublayers_spec=spec
        )
        item = paddle.randn([4, 256])
        dict_args = {"hidden_states": [item], "extra_key": "extra_value"}
        result = merger(dict_args)
        self.assertEqual(result["extra_key"], "extra_value")


if __name__ == "__main__":
    unittest.main()
