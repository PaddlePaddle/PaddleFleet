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


class TestMoELayerForwardMethod(unittest.TestCase):
    """Tests for MoELayer method existence."""

    def test_has_forward_method(self):
        """MoELayer should have a forward method."""
        self.assertTrue(hasattr(MoELayer, "forward"))

    def test_has_set_layer_number_method(self):
        """MoELayer should have a set_layer_number method."""
        self.assertTrue(hasattr(MoELayer, "set_layer_number"))

    def test_has_use_fp8_method(self):
        """MoELayer should have a use_fp8 method."""
        self.assertTrue(hasattr(MoELayer, "use_fp8"))


if __name__ == "__main__":
    unittest.main()
