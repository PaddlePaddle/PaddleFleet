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

"""Unit tests for the HyperEncoder package re-exports."""

import unittest

import paddle  # noqa: F401


class TestPackageExports(unittest.TestCase):
    def test_all_names_are_exported(self):
        from paddlefleet.models import hyperencoder

        for name in hyperencoder.__all__:
            self.assertTrue(
                hasattr(hyperencoder, name),
                f"{name} listed in __all__ but not importable",
            )

    def test_symbols_are_callable(self):
        from paddlefleet.models.hyperencoder import (
            AudioEncoderConv,
            ImageEncoderConv,
            MlpProjector,
            PatchEmbed,
            get_abs_pos_1d,
            get_abs_pos_2d,
            get_hyperencoder_block_spec,
            get_hyperencoder_layer_specs,
        )

        for sym in (
            AudioEncoderConv,
            ImageEncoderConv,
            MlpProjector,
            PatchEmbed,
            get_abs_pos_1d,
            get_abs_pos_2d,
            get_hyperencoder_block_spec,
            get_hyperencoder_layer_specs,
        ):
            self.assertTrue(callable(sym))


if __name__ == "__main__":
    unittest.main()
