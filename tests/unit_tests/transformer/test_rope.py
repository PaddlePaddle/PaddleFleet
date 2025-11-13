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

import unittest

import paddle

from fleet.core.models.common.embeddings import (
    RotaryEmbedding,
    YarnRotaryEmbedding,
)


class TestRotaryEmbedding(unittest.TestCase):
    def setUp(self):
        self.kv_channels = 8
        self.rotary_percent = 1.0
        self.rope = RotaryEmbedding(self.kv_channels, self.rotary_percent)

    def test_forward(self):
        output = self.rope(64)
        assert output.shape[0] == 64
        assert output.shape[1] == 1
        assert output.shape[2] == 1
        assert output.shape[3] == self.kv_channels
        assert output.dtype == paddle.float32
        assert output.place.is_gpu_place()


class TestYarnRotaryEmbedding(unittest.TestCase):
    def setUp(self):
        self.kv_channels = 8
        self.rotary_percent = 1.0
        self.rope = YarnRotaryEmbedding(self.kv_channels, self.rotary_percent)

    def test_forward(self):
        output, mscale = self.rope(64)
        assert output.shape[0] == 64
        assert output.shape[1] == 1
        assert output.shape[2] == 1
        assert output.shape[3] == self.kv_channels
        assert output.dtype == paddle.float32
        assert output.place.is_gpu_place()
        assert mscale == 1.0


if __name__ == "__main__":
    unittest.main()
