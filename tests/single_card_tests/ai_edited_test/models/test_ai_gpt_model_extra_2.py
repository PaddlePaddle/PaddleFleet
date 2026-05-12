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
from unittest.mock import MagicMock, patch

from paddlefleet.models.gpt.gpt_model import (
    GPTModel,
    GPTSublayersSpec,
)


class TestGPTSublayersSpecDefaults(unittest.TestCase):
    """Tests for GPTSublayersSpec default values."""

    def test_all_fields_default_to_none(self):
        """All fields should default to None."""
        spec = GPTSublayersSpec()
        self.assertIsNone(spec.embedding)
        self.assertIsNone(spec.head_empty_layers)
        self.assertIsNone(spec.transformer_layers)
        self.assertIsNone(spec.tail_empty_layers)
        self.assertIsNone(spec.mtp)
        self.assertIsNone(spec.layer_norm)
        self.assertIsNone(spec.lm_head)
        self.assertIsNone(spec.mtp_lm_head)
        self.assertIsNone(spec.mtp_loss)


class TestGPTModelAddSequentialLayer(unittest.TestCase):
    """Tests for GPTModel.add_sequential_layer."""

    def test_add_sequential_layer_appends_dict(self):
        """add_sequential_layer should append a dict with layer and name_prefix."""
        with patch.object(GPTModel, "__init__", lambda self, *a, **kw: None):
            model = GPTModel.__new__(GPTModel)
            layers = []
            mock_desc = MagicMock()
            model.add_sequential_layer(layers, mock_desc, "test_prefix")
            self.assertEqual(len(layers), 1)
            self.assertEqual(layers[0]["layer"], mock_desc)
            self.assertEqual(layers[0]["name_prefix"], "test_prefix")


class TestGPTModelGetSequentialLayers(unittest.TestCase):
    """Tests for GPTModel.get_sequential_layers."""

    def test_extracts_layer_only(self):
        """get_sequential_layers should return only layer objects."""
        with patch.object(GPTModel, "__init__", lambda self, *a, **kw: None):
            model = GPTModel.__new__(GPTModel)
            mock_layer1 = MagicMock()
            mock_layer2 = MagicMock()
            model._sequential_layers = [
                {"layer": mock_layer1, "name_prefix": "a"},
                {"layer": mock_layer2, "name_prefix": "b"},
            ]
            result = model.get_sequential_layers()
            self.assertEqual(result, [mock_layer1, mock_layer2])


class TestGPTModelGetNamePrefixes(unittest.TestCase):
    """Tests for GPTModel.get_sequential_name_prefixes."""

    def test_returns_index_to_prefix_mapping(self):
        """get_sequential_name_prefixes should map indices to prefixes."""
        with patch.object(GPTModel, "__init__", lambda self, *a, **kw: None):
            model = GPTModel.__new__(GPTModel)
            model._sequential_layers = [
                {"layer": MagicMock(), "name_prefix": "embed"},
                {"layer": MagicMock(), "name_prefix": "layer.0"},
            ]
            result = model.get_sequential_name_prefixes()
            self.assertEqual(result["0"], "embed")
            self.assertEqual(result["1"], "layer.0")


class TestGPTModelGetHardwareFlops(unittest.TestCase):
    """Tests for GPTModel.get_hardware_flops."""

    def test_returns_expected_value(self):
        """get_hardware_flops should return 989e3."""
        with patch.object(GPTModel, "__init__", lambda self, *a, **kw: None):
            model = GPTModel.__new__(GPTModel)
            result = model.get_hardware_flops()
            self.assertEqual(result, 989e3)


class TestGPTModelFP8QuantWeight(unittest.TestCase):
    """Tests for GPTModel.fp8_quant_weight."""

    def test_calls_fp8_quant_on_transformer_layers(self):
        """fp8_quant_weight should call fp8_quant_weight on TransformerLayer instances."""
        with patch.object(GPTModel, "__init__", lambda self, *a, **kw: None):
            model = GPTModel.__new__(GPTModel)
            model._num_virtual_pipeline_stages = 1
            from paddlefleet.transformer.transformer_layer import (
                TransformerLayer,
            )

            mock_layer = MagicMock(spec=TransformerLayer)
            other_layer = MagicMock()
            model.run_function = [mock_layer, other_layer]
            model.fp8_quant_weight(batch_mode=False, quant_transpose=True)
            mock_layer.fp8_quant_weight.assert_called_once_with(
                batch_mode=False, quant_transpose=True
            )
            other_layer.fp8_quant_weight.assert_not_called()


class TestGPTModelUseFP8(unittest.TestCase):
    """Tests for GPTModel.use_fp8."""

    def test_returns_false_when_no_fp8(self):
        """use_fp8 should return False when no TransformerLayer uses fp8."""
        with patch.object(GPTModel, "__init__", lambda self, *a, **kw: None):
            model = GPTModel.__new__(GPTModel)
            model._num_virtual_pipeline_stages = 1
            mock_layer = MagicMock()
            mock_layer.use_fp8.return_value = False
            model.run_function = [mock_layer]
            result = model.use_fp8()
            self.assertFalse(result)


class TestGPTModelGetWeightOnlyParams(unittest.TestCase):
    """Tests for GPTModel._get_weight_only_params."""

    def test_returns_params_with_weight_only_mtp_flag(self):
        """_get_weight_only_params should return params with is_weight_only_mtp flag."""
        with patch.object(GPTModel, "__init__", lambda self, *a, **kw: None):
            model = GPTModel.__new__(GPTModel)
            mock_param1 = MagicMock()
            mock_param1.is_weight_only_mtp = True
            mock_param2 = MagicMock()
            mock_param2.is_weight_only_mtp = False

            with patch.object(
                model,
                "state_dict",
                return_value={"a": mock_param1, "b": mock_param2},
            ):
                result = model._get_weight_only_params()
                self.assertEqual(len(result), 1)
                self.assertEqual(result[0], mock_param1)


if __name__ == "__main__":
    unittest.main()
