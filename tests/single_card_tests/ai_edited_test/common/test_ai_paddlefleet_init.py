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


class TestPaddleFleetInit(unittest.TestCase):
    """Tests for paddlefleet top-level __init__.py."""

    def test_import_paddlefleet(self):
        """Test that paddlefleet can be imported."""
        import paddlefleet

        self.assertIsNotNone(paddlefleet)

    def test_all_exports(self):
        """Test that __all__ contains expected entries."""
        from paddlefleet import __all__

        expected_exports = [
            "ops",
            "training",
            "parallel_state",
            "Timers",
            "LayerSpec",
            "__version__",
            "__package_name__",
            "__description__",
            "__license__",
            "__contact_names__",
            "__contact_emails__",
            "__homepage__",
            "__repository_url__",
            "__download_url__",
            "__keywords__",
        ]
        for export in expected_exports:
            self.assertIn(export, __all__)

    def test_mpu_alias(self):
        """Test that mpu is an alias for parallel_state."""
        import paddlefleet

        self.assertIs(paddlefleet.mpu, paddlefleet.parallel_state)

    def test_timers_import(self):
        """Test that Timers can be imported from top-level package."""
        from paddlefleet import Timers

        self.assertIsNotNone(Timers)

    def test_layer_spec_import(self):
        """Test that LayerSpec can be imported from top-level package."""
        from paddlefleet import LayerSpec

        self.assertIsNotNone(LayerSpec)

    def test_ops_import(self):
        """Test that ops can be imported from top-level package."""
        from paddlefleet import ops

        self.assertIsNotNone(ops)

    def test_training_import(self):
        """Test that training can be imported from top-level package."""
        from paddlefleet import training

        self.assertIsNotNone(training)

    def test_parallel_state_import(self):
        """Test that parallel_state can be imported from top-level package."""
        from paddlefleet import parallel_state

        self.assertIsNotNone(parallel_state)

    def test_version_export(self):
        """Test that __version__ is accessible."""
        from paddlefleet import __version__

        self.assertIsNotNone(__version__)
        self.assertIsInstance(__version__, str)

    def test_package_info_exports(self):
        """Test that package metadata fields are accessible."""
        from paddlefleet import (
            __contact_emails__,
            __contact_names__,
            __description__,
            __download_url__,
            __homepage__,
            __keywords__,
            __license__,
            __package_name__,
            __repository_url__,
        )

        self.assertEqual(__package_name__, "paddlefleet")
        self.assertEqual(__contact_names__, "PaddlePaddle")
        self.assertEqual(__contact_emails__, "Paddle-better@baidu.com")
        self.assertIsInstance(__description__, str)
        self.assertIsInstance(__homepage__, str)
        self.assertIsInstance(__repository_url__, str)
        self.assertIsInstance(__download_url__, str)
        self.assertIsInstance(__keywords__, str)
        self.assertIsInstance(__license__, tuple)

    def test_spec_utils_import(self):
        """Test that spec_utils module is importable."""
        from paddlefleet.spec_utils import LayerSpec

        self.assertIsNotNone(LayerSpec)


if __name__ == "__main__":
    unittest.main()
