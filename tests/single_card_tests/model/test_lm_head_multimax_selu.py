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

"""Targeted unit tests for the multimax SeLU function used by GPTLMHead.

Covers the pure-math layer of the multimax feature:
- SeLU(x, ranges=0, ts=0) is the identity (resume safety / step-0 invariant).
- SeLU is element-wise.
- SeLU does NOT mutate the input tensor (must clone for autograd correctness).
- SeLU has the expected value at simple analytic points.
"""

import os
import sys
import unittest

# Load the in-repo source ahead of any installed copy.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
sys.path.insert(0, _REPO_ROOT)

import paddle  # noqa: E402


class TestSeLU(unittest.TestCase):
    """Math-level tests for the multimax SeLU activation."""

    @classmethod
    def setUpClass(cls):
        from paddlefleet.models.gpt.lm_head import SeLU

        cls.SeLU = staticmethod(SeLU)

    def test_zero_params_is_identity(self):
        """ranges=ts=0 -> SeLU should be the identity (init-time invariant)."""
        x = paddle.randn([2, 3, 5], dtype="float32")
        ranges = paddle.zeros([4], dtype="float32")
        ts = paddle.zeros([4], dtype="float32")
        y = self.SeLU(x, ranges, ts)
        self.assertEqual(list(y.shape), list(x.shape))
        # Identity at init guarantees resume from checkpoints lacking these
        # params produces bit-identical logits at step 0.
        self.assertTrue(
            paddle.allclose(y, x, atol=1e-6).item(),
            "SeLU(x, 0, 0) must equal x exactly",
        )

    def test_input_not_mutated(self):
        """SeLU must clone before in-place +=; mutating x would corrupt autograd."""
        x = paddle.randn([4, 7], dtype="float32")
        x_orig = x.clone()
        ranges = paddle.to_tensor([0.0, 0.0, 0.0, 0.0], dtype="float32")
        ts = paddle.to_tensor([0.5, 0.5, 0.1, 0.1], dtype="float32")
        _ = self.SeLU(x, ranges, ts)
        # x must be unchanged after the call
        self.assertTrue(
            paddle.allclose(x, x_orig, atol=0.0).item(),
            "SeLU must not mutate its input tensor",
        )

    def test_elementwise_shape_preserved(self):
        """Output shape == input shape across various rank tensors."""
        ranges = paddle.zeros([4], dtype="float32")
        ts = paddle.to_tensor([0.1, 0.2, 0.3, 0.4], dtype="float32")
        for shape in [[8], [3, 16], [2, 4, 32], [1, 2, 3, 64]]:
            x = paddle.randn(shape, dtype="float32")
            y = self.SeLU(x, ranges, ts)
            self.assertEqual(list(y.shape), shape)

    def test_inside_window_no_modulation(self):
        """For a value strictly inside [ranges[1], ranges[0]] = (-1, 1) and
        away from the squared cutoffs at ranges[2..3]=0, SeLU should equal x.

        Linear part: ts[0]*relu(ranges[0]-x) is 0 when x>ranges[0],
                     ts[1]*relu(x-ranges[1]) is 0 when x<ranges[1].
        Quadratic part: ts[2]*relu(ranges[2]-x)**2 is 0 when x>ranges[2],
                        ts[3]*relu(x-ranges[3])**2 is 0 when x<ranges[3].
        Set ranges[0]=+inf, ranges[1]=-inf, ranges[2]=+inf, ranges[3]=-inf
        -> all four ReLU args are negative -> SeLU is identity for any x.
        """
        x = paddle.to_tensor([-2.0, 0.0, 2.0], dtype="float32")
        ranges = paddle.to_tensor(
            [1e30, -1e30, 1e30, -1e30], dtype="float32"
        )
        ts = paddle.to_tensor([1.0, 1.0, 1.0, 1.0], dtype="float32")
        # NOTE: ranges[0]=+inf -> relu(inf - x) = inf, so this WOULD blow up.
        # Construct the opposite: choose ranges such that all relu(...) are 0.
        # relu(ranges[0] - x) = 0 iff ranges[0] <= x  -> use ranges[0] = -1e30
        # relu(x - ranges[1]) = 0 iff x <= ranges[1]  -> use ranges[1] = +1e30
        # relu(ranges[2] - x) = 0 iff ranges[2] <= x  -> use ranges[2] = -1e30
        # relu(x - ranges[3]) = 0 iff x <= ranges[3]  -> use ranges[3] = +1e30
        ranges = paddle.to_tensor(
            [-1e30, 1e30, -1e30, 1e30], dtype="float32"
        )
        y = self.SeLU(x, ranges, ts)
        self.assertTrue(paddle.allclose(y, x, atol=0.0).item())

    def test_known_linear_modulation(self):
        """With ts=[1,0,0,0] and ranges=[1,0,0,0], SeLU(x) = x + relu(1 - x).

        For x = 0:   1 + relu(1) = 0 + 1 = 1
        For x = 0.5: 0.5 + relu(0.5) = 1.0
        For x = 2:   2 + relu(-1) = 2 + 0 = 2
        """
        x = paddle.to_tensor([0.0, 0.5, 2.0], dtype="float32")
        ranges = paddle.to_tensor([1.0, 0.0, 0.0, 0.0], dtype="float32")
        ts = paddle.to_tensor([1.0, 0.0, 0.0, 0.0], dtype="float32")
        y = self.SeLU(x, ranges, ts)
        expected = paddle.to_tensor([1.0, 1.0, 2.0], dtype="float32")
        self.assertTrue(
            paddle.allclose(y, expected, atol=1e-6).item(),
            f"got {y.numpy().tolist()}, expected {expected.numpy().tolist()}",
        )

    def test_known_quadratic_modulation(self):
        """With ts=[0,0,1,0] and ranges=[0,0,1,0], SeLU(x) = x + relu(1 - x)**2.

        For x = -1: -1 + relu(2)**2 = -1 + 4 = 3
        For x = 0:   0 + relu(1)**2 = 0 + 1 = 1
        For x = 2:   2 + relu(-1)**2 = 2 + 0 = 2
        """
        x = paddle.to_tensor([-1.0, 0.0, 2.0], dtype="float32")
        ranges = paddle.to_tensor([0.0, 0.0, 1.0, 0.0], dtype="float32")
        ts = paddle.to_tensor([0.0, 0.0, 1.0, 0.0], dtype="float32")
        y = self.SeLU(x, ranges, ts)
        expected = paddle.to_tensor([3.0, 1.0, 2.0], dtype="float32")
        self.assertTrue(
            paddle.allclose(y, expected, atol=1e-6).item(),
            f"got {y.numpy().tolist()}, expected {expected.numpy().tolist()}",
        )


if __name__ == "__main__":
    unittest.main()
