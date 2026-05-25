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


class TestSwiGLUFunctions(unittest.TestCase):
    """Tests for SwiGLU activation functions."""

    def test_swiglu_forward(self):
        """Test swiglu forward computation."""
        from paddlefleet.fusions.fused_bias_swiglu import swiglu

        y = paddle.randn([4, 16])
        result = swiglu(y)
        self.assertEqual(result.shape, [4, 8])

    def test_bias_swiglu_forward(self):
        """Test bias_swiglu forward computation."""
        from paddlefleet.fusions.fused_bias_swiglu import bias_swiglu

        y = paddle.randn([4, 16])
        bias = paddle.randn([16])
        result = bias_swiglu(y, bias)
        self.assertEqual(result.shape, [4, 8])

    def test_weighted_swiglu_forward(self):
        """Test weighted_swiglu forward computation."""
        from paddlefleet.fusions.fused_bias_swiglu import weighted_swiglu

        y = paddle.randn([4, 16])
        weights = paddle.randn([4, 1])
        result = weighted_swiglu(y, weights)
        self.assertEqual(result.shape, [4, 8])

    def test_bias_swiglu_impl_2d(self):
        """Test bias_swiglu_impl with 2D input and bias."""
        from paddlefleet.fusions.fused_bias_swiglu import bias_swiglu_impl

        y = paddle.randn([4, 16])
        bias = paddle.randn([16])
        result = bias_swiglu_impl(y, bias)
        self.assertEqual(result.shape, [4, 8])

    def test_bias_swiglu_impl_2d_no_bias(self):
        """Test bias_swiglu_impl with 2D input without bias."""
        from paddlefleet.fusions.fused_bias_swiglu import bias_swiglu_impl

        y = paddle.randn([4, 16])
        result = bias_swiglu_impl(y, None)
        self.assertEqual(result.shape, [4, 8])

    def test_bias_swiglu_impl_3d(self):
        """Test bias_swiglu_impl with 3D input."""
        from paddlefleet.fusions.fused_bias_swiglu import bias_swiglu_impl

        y = paddle.randn([2, 4, 16])
        bias = paddle.randn([16])
        result = bias_swiglu_impl(y, bias)
        self.assertEqual(result.shape, [2, 4, 8])

    def test_weighted_bias_swiglu_impl_no_bias(self):
        """Test weighted_bias_swiglu_impl without bias."""
        from paddlefleet.fusions.fused_bias_swiglu import (
            weighted_bias_swiglu_impl,
        )

        y = paddle.randn([4, 16])
        weights = paddle.randn([4, 1])
        result = weighted_bias_swiglu_impl(y, None, weights)
        self.assertEqual(result.shape, [4, 8])

    def test_weighted_bias_swiglu_impl_with_bias_raises(self):
        """Test weighted_bias_swiglu_impl with bias raises NotImplementedError."""
        from paddlefleet.fusions.fused_bias_swiglu import (
            weighted_bias_swiglu_impl,
        )

        y = paddle.randn([4, 16])
        bias = paddle.randn([16])
        weights = paddle.randn([4, 1])
        with self.assertRaises(NotImplementedError):
            weighted_bias_swiglu_impl(y, bias, weights)


if __name__ == "__main__":
    unittest.main()
