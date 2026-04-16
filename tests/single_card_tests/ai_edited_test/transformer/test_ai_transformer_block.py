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

from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.transformer.paddle_norm import LayerNorm as LayerNormImpl
from paddlefleet.transformer.transformer_block import (
    TransformerBlock,
    TransformerBlockSublayersSpec,
    _get_block_sublayers_spec,
)
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.transformer.transformer_layer import TransformerLayer
from paddlefleet.utils import WrappedTensor


def _make_config(**overrides):
    defaults = {
        "hidden_size": 128,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "cpu_offloading": False,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


class SimpleLayer(paddle.nn.Layer):
    def __init__(self, config, layer_number=1, **kwargs):
        super().__init__()
        self.config = config
        self.layer_number = layer_number
        self.linear = paddle.nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states, attention_mask, context=None, **kwargs):
        out = self.linear(hidden_states)
        return out, context


class TestTransformerBlockSublayersSpec(unittest.TestCase):
    """Test TransformerBlockSublayersSpec dataclass."""

    def test_default_values(self):
        spec = TransformerBlockSublayersSpec()
        self.assertIsNone(spec.layer_specs)
        self.assertIsNone(spec.layer_norm)

    def test_custom_values(self):
        spec = TransformerBlockSublayersSpec(
            layer_specs=[LayerSpec(SimpleLayer)],
            layer_norm=MagicMock(),
        )
        self.assertEqual(len(spec.layer_specs), 1)


class TestGetBlockSublayersSpec(unittest.TestCase):
    """Test _get_block_sublayers_spec helper."""

    def test_from_transformer_block_sublayers_spec(self):
        spec = TransformerBlockSublayersSpec(layer_specs=[], layer_norm=None)
        result = _get_block_sublayers_spec(_make_config(), spec)
        self.assertIsInstance(result, TransformerBlockSublayersSpec)

    def test_from_layer_spec_transformer_block(self):
        # Use real TransformerBlock class so issubclass works in source code
        mock_sublayers = TransformerBlockSublayersSpec(
            layer_specs=[], layer_norm=None
        )
        # Create a proper LayerSpec with TransformerBlock
        layer_spec = LayerSpec(TransformerBlock, sublayers_spec=mock_sublayers)
        result = _get_block_sublayers_spec(_make_config(), layer_spec)
        self.assertEqual(result, mock_sublayers)

    def test_from_layer_spec_transformer_layer(self):
        # Use real TransformerLayer class so issubclass works in source code
        config = _make_config(num_hidden_layers=3)
        layer_spec = LayerSpec(TransformerLayer)
        result = _get_block_sublayers_spec(config, layer_spec)
        self.assertIsInstance(result, TransformerBlockSublayersSpec)
        self.assertEqual(len(result.layer_specs), 3)

    def test_from_invalid_spec_raises(self):
        with self.assertRaises(Exception):  # noqa: B017
            _get_block_sublayers_spec(_make_config(), "invalid_spec")

    @patch("paddlefleet.transformer.transformer_block.TransformerLayer")
    def test_from_layer_spec_unknown_layer_raises(self, mock_tl):
        mock_tl.__name__ = "UnknownLayer"
        layer_spec = LayerSpec(mock_tl)
        with self.assertRaises(Exception):  # noqa: B017
            _get_block_sublayers_spec(_make_config(), layer_spec)


class TestTransformerBlockConstruction(unittest.TestCase):
    """Test TransformerBlock __init__ and layer building."""

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    def test_basic_construction(self, mock_pg):
        mock_pg.return_value = MagicMock()
        config = _make_config(num_hidden_layers=2)
        spec = TransformerBlockSublayersSpec(
            layer_specs=[LayerSpec(SimpleLayer)] * 2,
            layer_norm=None,
        )
        block = TransformerBlock(config, spec)
        self.assertEqual(len(block.layers), 2)
        self.assertIsNone(block.norm)
        self.assertTrue(block.post_layer_norm)
        self.assertTrue(block.pre_process)
        self.assertTrue(block.post_process)

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    def test_norm_built_when_conditions_met(self, mock_pg):
        mock_pg.return_value = MagicMock()
        config = _make_config(num_hidden_layers=1)
        spec = TransformerBlockSublayersSpec(
            layer_specs=[LayerSpec(SimpleLayer)],
            layer_norm=LayerNormImpl,
        )
        block = TransformerBlock(config, spec)
        self.assertIsNotNone(block.norm)

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    def test_norm_none_when_post_process_false(self, mock_pg):
        mock_pg.return_value = MagicMock()
        config = _make_config(num_hidden_layers=1)
        spec = TransformerBlockSublayersSpec(
            layer_specs=[LayerSpec(SimpleLayer)],
            layer_norm=MagicMock(),
        )
        block = TransformerBlock(config, spec, post_process=False)
        self.assertIsNone(block.norm)

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    def test_norm_none_when_post_layer_norm_false(self, mock_pg):
        mock_pg.return_value = MagicMock()
        config = _make_config(num_hidden_layers=1)
        spec = TransformerBlockSublayersSpec(
            layer_specs=[LayerSpec(SimpleLayer)],
            layer_norm=MagicMock(),
        )
        block = TransformerBlock(config, spec, post_layer_norm=False)
        self.assertIsNone(block.norm)

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    def test_vp_stage_raises_assertion(self, mock_pg):
        mock_pg.return_value = MagicMock()
        config = _make_config()
        spec = TransformerBlockSublayersSpec(
            layer_specs=[LayerSpec(SimpleLayer)],
            layer_norm=None,
        )
        with self.assertRaises(AssertionError):
            TransformerBlock(config, spec, vp_stage=1)


class TestTransformerBlockGetLayer(unittest.TestCase):
    """Test _get_layer method."""

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    def test_get_layer_returns_correct_layer(self, mock_pg):
        mock_pg.return_value = MagicMock()
        config = _make_config(num_hidden_layers=3)
        spec = TransformerBlockSublayersSpec(
            layer_specs=[LayerSpec(SimpleLayer)] * 3,
            layer_norm=None,
        )
        block = TransformerBlock(config, spec)
        layer = block._get_layer(1)
        self.assertIsNotNone(layer)


class TestTransformerBlockForward(unittest.TestCase):
    """Test TransformerBlock.forward."""

    @patch("paddlefleet.transformer.transformer_block.tensor_parallel")
    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    def test_forward_basic(self, mock_pg, mock_tp):
        mock_pg.return_value = MagicMock()
        mock_rng = MagicMock()
        mock_rng.__enter__ = MagicMock(return_value=None)
        mock_rng.__exit__ = MagicMock(return_value=None)
        mock_tp.get_cuda_rng_tracker.return_value.fork.return_value = mock_rng
        config = _make_config(num_hidden_layers=1)
        spec = TransformerBlockSublayersSpec(
            layer_specs=[LayerSpec(SimpleLayer)],
            layer_norm=None,
        )
        block = TransformerBlock(config, spec)
        hidden = paddle.randn([2, 4, 128])
        mask = paddle.ones([1, 1, 4, 4])
        result = block(hidden, mask)
        self.assertIsNotNone(result)

    @patch("paddlefleet.transformer.transformer_block.tensor_parallel")
    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    def test_forward_wrapped_tensor(self, mock_pg, mock_tp):
        mock_pg.return_value = MagicMock()
        mock_rng = MagicMock()
        mock_rng.__enter__ = MagicMock(return_value=None)
        mock_rng.__exit__ = MagicMock(return_value=None)
        mock_tp.get_cuda_rng_tracker.return_value.fork.return_value = mock_rng
        config = _make_config(num_hidden_layers=1)
        spec = TransformerBlockSublayersSpec(
            layer_specs=[LayerSpec(SimpleLayer)],
            layer_norm=None,
        )
        block = TransformerBlock(config, spec)
        hidden = paddle.randn([2, 4, 128])
        wrapped = WrappedTensor(hidden)
        mask = paddle.ones([1, 1, 4, 4])
        result = block(wrapped, mask)
        self.assertIsNotNone(result)

    @patch("paddlefleet.transformer.transformer_block.tensor_parallel")
    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    def test_forward_with_norm(self, mock_pg, mock_tp):
        mock_pg.return_value = MagicMock()

        class NormLayer(paddle.nn.Layer):
            def __init__(self, **kwargs):
                super().__init__()
                self.w = paddle.create_parameter(
                    shape=[128],
                    dtype="float32",
                    default_initializer=paddle.nn.initializer.Constant(1.0),
                )

            def forward(self, x):
                return x * self.w

        norm_spec = type("N", (), {"__name__": "NormLayer"})()
        mock_rng = MagicMock()
        mock_rng.__enter__ = MagicMock(return_value=None)
        mock_rng.__exit__ = MagicMock(return_value=None)
        mock_tp.get_cuda_rng_tracker.return_value.fork.return_value = mock_rng

        config = _make_config(num_hidden_layers=1)
        spec = TransformerBlockSublayersSpec(
            layer_specs=[LayerSpec(SimpleLayer)],
            layer_norm=norm_spec,
        )
        # Use __new__ to bypass init and test norm path directly
        block = TransformerBlock.__new__(TransformerBlock)
        paddle.nn.Layer.__init__(block)
        block.config = config
        block.pg_collection = mock_pg.return_value
        block.sublayers_spec = spec
        block.post_layer_norm = True
        block.pre_process = True
        block.post_process = True
        block.input_tensor = None
        block.config.cpu_offloading = False
        block.config._cpu_offloading_context = None
        block.layers = paddle.nn.LayerList([SimpleLayer(config)])
        block.norm = NormLayer()
        hidden = paddle.randn([2, 4, 128])
        mask = paddle.ones([1, 1, 4, 4])
        result = block(hidden, mask)
        self.assertIsNotNone(result)


class TestTransformerBlockSetInputTensor(unittest.TestCase):
    """Test set_input_tensor method."""

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    def test_set_input_tensor(self, mock_pg):
        mock_pg.return_value = MagicMock()
        config = _make_config(num_hidden_layers=1)
        spec = TransformerBlockSublayersSpec(
            layer_specs=[LayerSpec(SimpleLayer)],
            layer_norm=None,
        )
        block = TransformerBlock(config, spec)
        tensor = paddle.randn([2, 4, 128])
        block.set_input_tensor(tensor)
        self.assertIs(block.input_tensor, tensor)


if __name__ == "__main__":
    unittest.main()
