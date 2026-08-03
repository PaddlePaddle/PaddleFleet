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
"""Coverage for ThreePathCloneAlignMG added in commit 80a72f9. It is a
three-way differentiable identity clone whose backward sums the incoming
gradients in the MG-aligned order (dispatcher + shared) + router."""

import os
import sys

_test_file = os.path.abspath(__file__)
_repo_root = _test_file
for _ in range(10):
    _repo_root = os.path.dirname(_repo_root)
    if os.path.isdir(os.path.join(_repo_root, "src", "paddlefleet")):
        break
sys.path.insert(0, _repo_root)
sys.path.insert(0, os.path.join(_repo_root, "src"))
for _mod in list(sys.modules.keys()):
    if _mod == "paddlefleet" or _mod.startswith("paddlefleet."):
        del sys.modules[_mod]

import unittest

import numpy as np
import paddle

from paddlefleet.transformer.moe.moe_layer import ThreePathCloneAlignMG


class TestThreePathCloneAlignMG(unittest.TestCase):
    def test_forward_returns_three_equal_clones(self):
        x = paddle.randn([3, 5])
        a, b, c = ThreePathCloneAlignMG.apply(x)
        for out in (a, b, c):
            self.assertEqual(out.shape, [3, 5])
            np.testing.assert_array_equal(out.numpy(), x.numpy())

    def test_backward_sums_all_three_paths(self):
        x = paddle.randn([3, 5])
        x.stop_gradient = False
        a, b, c = ThreePathCloneAlignMG.apply(x)
        # distinct weights so the summed gradient is 1 + 2 + 3 = 6.
        (a * 1.0 + b * 2.0 + c * 3.0).sum().backward()
        self.assertEqual(x.grad.shape, [3, 5])
        np.testing.assert_allclose(
            x.grad.numpy(), np.full([3, 5], 6.0, dtype="float32"), atol=1e-6
        )

    def test_backward_only_one_path_used(self):
        # Only the router path contributes to the loss; the other two clones
        # still feed zero grads into the backward sum.
        x = paddle.randn([2, 4])
        x.stop_gradient = False
        a, _b, _c = ThreePathCloneAlignMG.apply(x)
        a.sum().backward()
        np.testing.assert_allclose(
            x.grad.numpy(), np.ones([2, 4], dtype="float32"), atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
