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


class TestMultimodalProjectorInitMLP(unittest.TestCase):
    """Test MultimodalProjector initialization with MLP type."""

    @patch("paddlefleet.models.vision.multimodal_projector.MLP")
    def test_init_mlp_type(self, mock_mlp):
        from paddlefleet.models.vision.multimodal_projector import (
            MultimodalProjector,
        )

        mock_config = MagicMock()
        mock_config.hidden_size = 4096
        mock_config.init_method = MagicMock()
        mock_config.add_bias_linear = False
        mock_mlp_instance = MagicMock()
        mock_mlp.return_value = mock_mlp_instance

        spec = MagicMock()
        projector = MultimodalProjector(
            config=mock_config,
            sublayers_spec=spec,
            projector_type="mlp",
            input_size=1024,
        )
        self.assertEqual(projector.projector_type, "mlp")
        mock_mlp.assert_called_once()


class TestMultimodalProjectorInitAffine(unittest.TestCase):
    """Test MultimodalProjector initialization with affine type."""

    @patch("paddlefleet.models.vision.multimodal_projector.build_layer")
    def test_init_affine_type(self, mock_build):
        from paddlefleet.models.vision.multimodal_projector import (
            MultimodalProjector,
        )

        mock_config = MagicMock()
        mock_config.hidden_size = 4096
        mock_config.init_method = MagicMock()
        mock_config.add_bias_linear = True
        mock_build.return_value = MagicMock()

        spec = MagicMock()
        spec.linear_fc1 = MagicMock()
        projector = MultimodalProjector(
            config=mock_config,
            sublayers_spec=spec,
            projector_type="affine",
            input_size=1024,
        )
        self.assertEqual(projector.projector_type, "affine")
        mock_build.assert_called_once()


class TestMultimodalProjectorInitUnsupported(unittest.TestCase):
    """Test MultimodalProjector with unsupported projector type."""

    def test_unsupported_type_raises(self):
        from paddlefleet.models.vision.multimodal_projector import (
            MultimodalProjector,
        )

        mock_config = MagicMock()
        mock_config.hidden_size = 4096

        with self.assertRaises(Exception):  # noqa: B017
            MultimodalProjector(
                config=mock_config,
                sublayers_spec=MagicMock(),
                projector_type="unsupported",
                input_size=1024,
            )


class TestMultimodalProjectorInitNoneSpec(unittest.TestCase):
    """Test MultimodalProjector with None sublayers_spec."""

    def test_none_spec_raises(self):
        from paddlefleet.models.vision.multimodal_projector import (
            MultimodalProjector,
        )

        mock_config = MagicMock()
        mock_config.hidden_size = 4096

        with self.assertRaises(AssertionError):
            MultimodalProjector(
                config=mock_config,
                sublayers_spec=None,
                projector_type="mlp",
                input_size=1024,
            )


class TestMultimodalProjectorForward(unittest.TestCase):
    """Test MultimodalProjector forward pass."""

    @patch("paddlefleet.models.vision.multimodal_projector.MLP")
    def test_forward_returns_encoder_output(self, mock_mlp):
        import paddle

        from paddlefleet.models.vision.multimodal_projector import (
            MultimodalProjector,
        )

        mock_config = MagicMock()
        mock_config.hidden_size = 4096
        mock_config.init_method = MagicMock()
        mock_config.add_bias_linear = False

        mock_encoder = MagicMock()
        mock_output = paddle.randn([1, 10, 4096])
        mock_encoder.return_value = (mock_output, None)
        mock_mlp.return_value = mock_encoder

        spec = MagicMock()
        projector = MultimodalProjector(
            config=mock_config,
            sublayers_spec=spec,
            projector_type="mlp",
            input_size=1024,
        )
        hidden = paddle.randn([1, 10, 1024])
        result = projector.forward(hidden)
        self.assertIsNotNone(result)

    @patch("paddlefleet.models.vision.multimodal_projector.MLP")
    def test_forward_adds_bias(self, mock_mlp):
        import paddle

        from paddlefleet.models.vision.multimodal_projector import (
            MultimodalProjector,
        )

        mock_config = MagicMock()
        mock_config.hidden_size = 4096
        mock_config.init_method = MagicMock()
        mock_config.add_bias_linear = False

        mock_encoder = MagicMock()
        mock_output = paddle.randn([1, 10, 4096])
        mock_bias = paddle.randn([1, 10, 4096])
        mock_encoder.return_value = (mock_output, mock_bias)
        mock_mlp.return_value = mock_encoder

        spec = MagicMock()
        projector = MultimodalProjector(
            config=mock_config,
            sublayers_spec=spec,
            projector_type="mlp",
            input_size=1024,
        )
        hidden = paddle.randn([1, 10, 1024])
        result = projector.forward(hidden)
        self.assertIsNotNone(result)


class TestMultimodalProjectorAffineForward(unittest.TestCase):
    """Test MultimodalProjector forward with affine type."""

    @patch("paddlefleet.models.vision.multimodal_projector.build_layer")
    def test_affine_forward(self, mock_build):
        import paddle

        from paddlefleet.models.vision.multimodal_projector import (
            MultimodalProjector,
        )

        mock_config = MagicMock()
        mock_config.hidden_size = 4096
        mock_config.init_method = MagicMock()
        mock_config.add_bias_linear = False

        mock_encoder = MagicMock()
        mock_output = paddle.randn([1, 10, 4096])
        mock_encoder.return_value = (mock_output, None)
        mock_build.return_value = mock_encoder

        spec = MagicMock()
        spec.linear_fc1 = MagicMock()
        projector = MultimodalProjector(
            config=mock_config,
            sublayers_spec=spec,
            projector_type="affine",
            input_size=1024,
        )
        hidden = paddle.randn([1, 10, 1024])
        result = projector.forward(hidden)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
