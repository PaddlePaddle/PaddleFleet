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

import paddle


class TestFusedSwigluScaleForward(unittest.TestCase):
    """Tests for fused_swiglu_scale_forward."""

    def test_forward_cpu_fallback(self):
        """Test fused_swiglu_scale_forward CPU fallback path."""
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_forward,
        )

        x = paddle.randn([4, 16])
        scale = paddle.ones([4, 1])
        result = fused_swiglu_scale_forward(x, scale)
        self.assertEqual(result.shape, [4, 8])

    def test_forward_with_1d_scale(self):
        """Test with 1D scale tensor."""
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_forward,
        )

        x = paddle.randn([4, 16])
        scale = paddle.ones([4])
        result = fused_swiglu_scale_forward(x, scale)
        self.assertEqual(result.shape, [4, 8])

    def test_forward_scale_broadcast(self):
        """Test scale broadcasting."""
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_forward,
        )

        x = paddle.randn([4, 16])
        scale = paddle.full([4, 1], 2.0)
        result = fused_swiglu_scale_forward(x, scale)
        self.assertEqual(result.shape, [4, 8])


class TestFusedSwigluScaleBackward(unittest.TestCase):
    """Tests for fused_swiglu_scale_backward."""

    def test_backward_cpu_fallback(self):
        """Test fused_swiglu_scale_backward CPU fallback path."""
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_backward,
        )

        x = paddle.randn([4, 16])
        scale = paddle.ones([4, 1])
        out_grad = paddle.randn([4, 8])
        d_x, d_scale = fused_swiglu_scale_backward(x, scale, out_grad)
        self.assertEqual(d_x.shape, [4, 16])
        self.assertEqual(d_scale.shape, [4, 1])

    def test_backward_shapes(self):
        """Test backward output shapes match inputs."""
        from paddlefleet.fusions.fused_swiglu_scale import (
            fused_swiglu_scale_backward,
        )

        x = paddle.randn([2, 32])
        scale = paddle.ones([2, 1])
        out_grad = paddle.randn([2, 16])
        d_x, d_scale = fused_swiglu_scale_backward(x, scale, out_grad)
        self.assertEqual(d_x.shape, [2, 32])
        self.assertEqual(d_scale.shape, [2, 1])


if __name__ == "__main__":
    unittest.main()
