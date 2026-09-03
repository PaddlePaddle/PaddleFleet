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
"""Coverage for the ``use_accuracy_compatible`` (eager) SwiGLU paths added in
commit 80a72f9. These eager paths must be numerically equivalent to the fused
paths while exercising the pure-python fallback branches."""

import os
import sys

# Walk up to find the repo root.
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

from paddlefleet.fusions.fused_bias_swiglu import (
    BiasSwiGLUFunction,
    SwiGLUFunction,
    WeightedSwiGLUFunction,
    bias_swiglu,
    bias_swiglu_eager,
    bias_swiglu_impl,
    swiglu,
    swiglu_back,
    swiglu_back_eager,
    swiglu_eager,
    weighted_bias_swiglu_impl,
    weighted_swiglu_back,
    weighted_swiglu_back_eager,
)


class TestEagerForwardEquivalence(unittest.TestCase):
    """Eager forward helpers must match their fused counterparts."""

    def test_swiglu_eager_matches_swiglu(self):
        y = paddle.randn([4, 16])
        out = swiglu_eager(y)
        self.assertEqual(out.shape, [4, 8])
        np.testing.assert_allclose(
            out.numpy(), swiglu(y).numpy(), rtol=1e-5, atol=1e-5
        )

    def test_bias_swiglu_eager_matches_bias_swiglu(self):
        y = paddle.randn([4, 16])
        b = paddle.randn([16])
        np.testing.assert_allclose(
            bias_swiglu_eager(y, b).numpy(),
            bias_swiglu(y, b).numpy(),
            rtol=1e-5,
            atol=1e-5,
        )


class TestEagerBackwardEquivalence(unittest.TestCase):
    """Eager backward helpers must match the native-grad-backed versions."""

    def test_swiglu_back_eager_matches_native(self):
        paddle.seed(0)
        g = paddle.randn([2, 4])
        y = paddle.randn([2, 8])
        np.testing.assert_allclose(
            swiglu_back_eager(g, y).numpy(),
            swiglu_back(g, y).numpy(),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_weighted_swiglu_back_eager_matches_native(self):
        paddle.seed(0)
        g = paddle.randn([2, 4])
        y = paddle.randn([2, 8])
        w = paddle.randn([2, 1])
        ig_e, wg_e = weighted_swiglu_back_eager(g, y, w)
        ig_s, wg_s = weighted_swiglu_back(g, y, w)
        self.assertEqual(ig_e.shape, [2, 8])
        self.assertEqual(wg_e.shape, [2, 1])
        np.testing.assert_allclose(
            ig_e.numpy(), ig_s.numpy(), rtol=1e-4, atol=1e-4
        )
        np.testing.assert_allclose(
            wg_e.numpy(), wg_s.numpy(), rtol=1e-4, atol=1e-4
        )


class TestPyLayerAccuracyCompatible(unittest.TestCase):
    """PyLayer forward+backward through the use_accuracy_compatible branch."""

    def test_swiglu_function_eager_fwd_bwd(self):
        x = paddle.randn([4, 16]).astype("float32")
        x.stop_gradient = False
        out = SwiGLUFunction.apply(x, False, False, None, True)
        self.assertEqual(out.shape, [4, 8])
        np.testing.assert_allclose(
            out.numpy(), swiglu_eager(x).numpy(), rtol=1e-5, atol=1e-5
        )
        out.sum().backward()
        self.assertEqual(x.grad.shape, [4, 16])

    def test_bias_swiglu_function_eager_fwd_bwd(self):
        x = paddle.randn([4, 16]).astype("float32")
        x.stop_gradient = False
        b = paddle.randn([16]).astype("float32")
        b.stop_gradient = False
        out = BiasSwiGLUFunction.apply(x, b, False, False, None, True)
        self.assertEqual(out.shape, [4, 8])
        np.testing.assert_allclose(
            out.numpy(),
            bias_swiglu_eager(x, b).numpy(),
            rtol=1e-5,
            atol=1e-5,
        )
        out.sum().backward()
        self.assertEqual(x.grad.shape, [4, 16])
        self.assertIsNotNone(b.grad)

    def test_weighted_swiglu_function_eager_fwd_bwd(self):
        x = paddle.randn([4, 16]).astype("float32")
        x.stop_gradient = False
        w = paddle.ones([4, 1]).astype("float32")
        w.stop_gradient = False
        out = WeightedSwiGLUFunction.apply(x, w, False, None, True)
        self.assertEqual(out.shape, [4, 8])
        # weights are all ones => equals plain eager swiglu
        np.testing.assert_allclose(
            out.numpy(), swiglu_eager(x).numpy(), rtol=1e-5, atol=1e-5
        )
        out.sum().backward()
        self.assertEqual(x.grad.shape, [4, 16])
        self.assertEqual(w.grad.shape, [4, 1])

    def test_weighted_swiglu_function_eager_dtype_preserved(self):
        x = paddle.randn([3, 8]).astype("float32")
        w = paddle.randn([3, 1]).astype("float32")
        out = WeightedSwiGLUFunction.apply(x, w, False, None, True)
        self.assertEqual(out.dtype, paddle.float32)
        self.assertEqual(out.shape, [3, 4])


class TestImplAccuracyCompatible(unittest.TestCase):
    """bias_swiglu_impl / weighted_bias_swiglu_impl accuracy-compatible path."""

    def test_bias_swiglu_impl_eager_with_bias(self):
        inp = paddle.randn([4, 16])
        inp.stop_gradient = False
        b = paddle.randn([16])
        b.stop_gradient = False
        out = bias_swiglu_impl(inp, b, use_accuracy_compatible=True)
        self.assertEqual(out.shape, [4, 8])
        grads = paddle.grad([out.sum()], [inp, b])
        self.assertEqual(grads[0].shape, [4, 16])
        # BiasSwiGLUFunction returns the same tensor for input and bias grads.
        self.assertIsNotNone(grads[1])

    def test_bias_swiglu_impl_eager_no_bias_3d(self):
        inp = paddle.randn([2, 4, 16])
        inp.stop_gradient = False
        out = bias_swiglu_impl(inp, None, use_accuracy_compatible=True)
        self.assertEqual(out.shape, [2, 4, 8])
        grads = paddle.grad([out.sum()], [inp])
        self.assertEqual(grads[0].shape, [2, 4, 16])

    def test_weighted_bias_swiglu_impl_eager_no_bias(self):
        inp = paddle.randn([4, 16])
        inp.stop_gradient = False
        w = paddle.randn([4, 1])
        w.stop_gradient = False
        out = weighted_bias_swiglu_impl(
            inp, None, w, use_accuracy_compatible=True
        )
        self.assertEqual(out.shape, [4, 8])
        grads = paddle.grad([out.sum()], [inp, w])
        self.assertEqual(grads[0].shape, [4, 16])
        self.assertEqual(grads[1].shape, [4, 1])


if __name__ == "__main__":
    unittest.main()
