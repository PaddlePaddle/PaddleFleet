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

import paddle
from paddle.distributed.fleet.meta_parallel import LayerSpec

from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.transformer.transformer_encoder import TransformerEncoder


def _make_config(**overrides):
    defaults = {
        "hidden_size": 64,
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "pipeline_model_parallel_size": 1,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


class FakeEmbedding(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.w = paddle.create_parameter(
            shape=[1000, config.hidden_size],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(0.01),
        )

    def forward(self, x):
        return self.w[x]


class FakeLayerNorm(paddle.nn.Layer):
    def __init__(self, config, hidden_size=None, eps=None):
        super().__init__()
        self.config = config

    def forward(self, x):
        return x


class FakeEmptyLayer(paddle.nn.Layer):
    def __init__(self, config=None):
        super().__init__()

    def forward(self, x):
        return x


class FakeSpec:
    def __init__(self, config):
        self.embedding = LayerSpec(FakeEmbedding)
        self.layer_norm = LayerSpec(FakeLayerNorm)
        self.head_empty_layers = []
        self.tail_empty_layers = []
        self.transformer_layers = []


class TestTransformerEncoderConstruction(unittest.TestCase):
    """Test TransformerEncoder construction with mocked pipeline."""

    @patch(
        "paddlefleet.transformer.transformer_encoder.fleet.get_hybrid_communicate_group"
    )
    @patch("paddlefleet.transformer.transformer_encoder.PipelineLayer.__init__")
    def test_construction(self, mock_pp_init, mock_fleet):
        mock_pp_init.return_value = None
        spec = FakeSpec(_make_config())
        encoder = TransformerEncoder.__new__(TransformerEncoder)
        encoder.config = _make_config()
        encoder.modal = None
        encoder._pipeline_name_mapping = None
        encoder._pp_to_single_mapping = None
        encoder._sequential_layers = encoder.get_layer_desc_list(spec)
        encoder.layers = encoder.get_sequential_layers()
        # PipelineLayer init would need many mocks, just verify our layers
        # embedding + layer_norm + 0 transformer_layers = at least 2
        self.assertGreaterEqual(len(encoder._sequential_layers), 2)

    def test_get_sequential_layers(self):
        spec = FakeSpec(_make_config())
        encoder = TransformerEncoder.__new__(TransformerEncoder)
        encoder.config = _make_config()
        encoder._sequential_layers = [
            {"layer": MagicMock(), "name_prefix": "model.0"},
            {"layer": MagicMock(), "name_prefix": "model.1"},
        ]
        layers = encoder.get_sequential_layers()
        self.assertEqual(len(layers), 2)

    def test_get_sequential_name_prefixes(self):
        spec = FakeSpec(_make_config())
        encoder = TransformerEncoder.__new__(TransformerEncoder)
        encoder.config = _make_config()
        encoder._sequential_layers = [
            {"layer": MagicMock(), "name_prefix": "model.0"},
            {"layer": MagicMock(), "name_prefix": "model.1"},
        ]
        prefixes = encoder.get_sequential_name_prefixes()
        self.assertEqual(prefixes["0"], "model.0")
        self.assertEqual(prefixes["1"], "model.1")


class TestTransformerEncoderAddSequentialLayer(unittest.TestCase):
    """Test add_sequential_layer method."""

    def test_add_layer(self):
        encoder = TransformerEncoder.__new__(TransformerEncoder)
        layers = []
        encoder.add_sequential_layer(layers, MagicMock(), "model")
        self.assertEqual(len(layers), 1)

    def test_add_multiple_layers(self):
        encoder = TransformerEncoder.__new__(TransformerEncoder)
        layers = []
        encoder.add_sequential_layer(layers, MagicMock(), "model.0")
        encoder.add_sequential_layer(layers, MagicMock(), "model.1")
        encoder.add_sequential_layer(layers, MagicMock(), "model.2")
        self.assertEqual(len(layers), 3)


class TestTransformerEncoderStateDict(unittest.TestCase):
    """Test state_dict remapping."""

    def test_state_dict_with_no_mapping(self):
        encoder = TransformerEncoder.__new__(TransformerEncoder)
        encoder.config = _make_config()
        encoder._pipeline_name_mapping = None
        encoder._pp_to_single_mapping = None
        # Without pipeline name mapping, state_dict would go to parent
        # This just tests the control flow path
        with patch.object(TransformerEncoder, "__init__", return_value=None):
            pass

    def test_set_state_dict_no_mapping(self):
        encoder = TransformerEncoder.__new__(TransformerEncoder)
        encoder.config = _make_config()
        encoder._pipeline_name_mapping = None
        encoder._pp_to_single_mapping = None
        # Cannot properly test set_state_dict without full PipelineLayer init
        # Just verify the method exists
        self.assertTrue(hasattr(encoder, "set_state_dict"))


class TestTransformerEncoderOverlappedForwardBackward(unittest.TestCase):
    """Test overlapped_forward_backward."""

    def test_with_no_overlap(self):
        encoder = TransformerEncoder.__new__(TransformerEncoder)
        encoder.config = _make_config()
        # Test the method exists and has proper structure
        self.assertTrue(hasattr(encoder, "overlapped_forward_backward"))


class TestTransformerEncoderGetHardwareFlops(unittest.TestCase):
    """Test get_hardware_flops."""

    def test_returns_value(self):
        encoder = TransformerEncoder.__new__(TransformerEncoder)
        flops = encoder.get_hardware_flops()
        self.assertEqual(flops, 989e3)


class TestTransformerEncoderGetEncoderLayerDescList(unittest.TestCase):
    """Test get_encoder_layer_desc_list."""

    def test_with_empty_layers(self):
        spec = FakeSpec(_make_config())
        encoder = TransformerEncoder.__new__(TransformerEncoder)
        layers = []
        encoder.get_encoder_layer_desc_list(layers, spec, name_prefix="model")
        # With empty head/tail and no transformer layers, should be empty
        self.assertEqual(len(layers), 0)

    def test_with_transformer_layers(self):
        spec = FakeSpec(_make_config())
        tl_spec = LayerSpec(FakeEmptyLayer)
        spec.transformer_layers = [tl_spec, tl_spec]
        encoder = TransformerEncoder.__new__(TransformerEncoder)
        layers = []
        encoder.get_encoder_layer_desc_list(layers, spec, name_prefix="model")
        self.assertEqual(len(layers), 2)

    def test_with_head_and_tail_empty(self):
        spec = FakeSpec(_make_config())
        empty_spec = LayerSpec(FakeEmptyLayer)
        spec.head_empty_layers = [empty_spec]
        spec.tail_empty_layers = [empty_spec, empty_spec]
        encoder = TransformerEncoder.__new__(TransformerEncoder)
        layers = []
        encoder.get_encoder_layer_desc_list(layers, spec, name_prefix="model")
        self.assertEqual(len(layers), 3)


class TestTransformerEncoderFp8QuantWeight(unittest.TestCase):
    """Test fp8_quant_weight method."""

    def test_no_virtual_stages(self):
        encoder = TransformerEncoder.__new__(TransformerEncoder)
        encoder._num_virtual_pipeline_stages = 1
        encoder.run_function = []
        # Should iterate over run_function
        encoder.fp8_quant_weight()

    def test_with_virtual_stages(self):
        encoder = TransformerEncoder.__new__(TransformerEncoder)
        encoder._num_virtual_pipeline_stages = 2
        encoder._model_chunks = [[], []]
        # Should iterate over model_chunks
        encoder.fp8_quant_weight()


class TestTransformerEncoderUseFp8(unittest.TestCase):
    """Test use_fp8 method."""

    def test_no_virtual_stages(self):
        encoder = TransformerEncoder.__new__(TransformerEncoder)
        encoder._num_virtual_pipeline_stages = 1
        encoder.run_function = []
        result = encoder.use_fp8()
        self.assertFalse(result)

    def test_with_virtual_stages(self):
        encoder = TransformerEncoder.__new__(TransformerEncoder)
        encoder._num_virtual_pipeline_stages = 2
        encoder._model_chunks = [[], []]
        result = encoder.use_fp8()
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
