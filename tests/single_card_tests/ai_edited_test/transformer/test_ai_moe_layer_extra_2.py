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
from unittest.mock import MagicMock

import paddle

from paddlefleet.transformer.moe.moe_layer import (
    MoELayer,
    MoESublayers,
)


class TestMoESublayersDefaults(unittest.TestCase):
    """Tests for MoESublayers default values."""

    def test_all_fields_default_to_none(self):
        """All fields should default to None."""
        spec = MoESublayers()
        self.assertIsNone(spec.mlp_spec)


class TestMoELayerSetInputTensor(unittest.TestCase):
    """Tests for MoELayer set_layer_number and use_fp8."""

    def test_set_layer_number(self):
        """Test set_layer_number stores the layer number."""
        layer = MoELayer.__new__(MoELayer)
        paddle.nn.Layer.__init__(layer)
        layer.set_layer_number(5)
        self.assertEqual(layer.layer_number, 5)


class TestMoELayerUseFP8Method(unittest.TestCase):
    """Tests for MoELayer.use_fp8 method."""

    def test_use_fp8_false_when_no_fp8_format(self):
        """use_fp8 should return False when no fp8_format is configured."""
        layer = MoELayer.__new__(MoELayer)
        paddle.nn.Layer.__init__(layer)
        layer.config = MagicMock()
        layer.config.fp8_format = None
        self.assertFalse(layer.use_fp8())


class TestMoELayerUseFP8(unittest.TestCase):
    """Tests for MoELayer use_fp8 attribute."""

    def test_use_fp8_false_by_default(self):
        """use_fp8 should return False when no fp8 is configured."""
        layer = MoELayer.__new__(MoELayer)
        paddle.nn.Layer.__init__(layer)
        layer.config = MagicMock()
        layer.config.fp8_format = None
        # Default should not use fp8
        self.assertFalse(layer.use_fp8())


class TestMoELayerMarkForwardOnly(unittest.TestCase):
    """Tests for MoELayer mark_forward_only method if it exists."""

    def test_has_forward_method(self):
        """MoELayer should have a forward method."""
        self.assertTrue(hasattr(MoELayer, "forward"))


if __name__ == "__main__":
    unittest.main()
