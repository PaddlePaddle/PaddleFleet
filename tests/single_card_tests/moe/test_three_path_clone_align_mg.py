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
"""Behavior tests for ``ThreePathCloneAlignMG`` in ``transformer/moe/moe_layer``.

The MoE block, when the three-path topology is enabled, fans ``hidden_states``
out to three consumers via::

    router, dispatcher, shared = ThreePathCloneAlignMG.apply(hidden_states)

so the layer must satisfy two contracts:

* **forward** hands each consumer an independent, differentiable *copy* of the
  input (identity clone), not a shared alias -- three separate autograd
  consumers are what let the block control the gradient-accumulation order.
* **backward** returns the *sum* of the three incoming cotangents as the single
  gradient w.r.t. ``hidden_states``. For an identity clone used in three places
  the chain rule gives ``dL/dx = g_router + g_dispatcher + g_shared``; the
  layer computes ``(g_dispatcher + g_shared) + g_router`` to match the MG
  accumulation *order*. Order only affects float rounding, so it is not
  separately asserted here; the summation *value* is.

Expected values below are hand-derived from the identity-clone + sum definition
and are independent of the layer's implementation. The math is pure elementwise
clone/add with no device-specific numerics, so the tests run on CPU.
"""

import os
import sys
import unittest

import numpy as np

# Resolve the repo so ``paddlefleet`` imports whether or not it is installed:
# tests/single_card_tests/moe/<file> -> up four levels is the repo root.
_HERE = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
)
for _entry in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

try:
    import paddle

    from paddlefleet.transformer.moe.moe_layer import ThreePathCloneAlignMG

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle / paddlefleet unavailable in this env
    paddle = None
    ThreePathCloneAlignMG = None
    _IMPORT_ERROR = exc


@unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddle/paddlefleet not importable: {_IMPORT_ERROR!r}",
)
class TestThreePathCloneAlignMG(unittest.TestCase):
    """Real forward/backward behavior of the three-way identity clone."""

    def setUp(self):
        # Pure clone + add: no device kernels involved, so pin to CPU and
        # restore the caller's device afterwards to avoid cross-test bleed.
        self._orig_device = paddle.get_device()
        paddle.set_device("cpu")
        self.addCleanup(paddle.set_device, self._orig_device)

    def test_forward_produces_three_independent_clones(self):
        # Distinguishable content so a value-equality check is meaningful.
        x = paddle.to_tensor(np.arange(6, dtype="float32").reshape(2, 3))
        outs = ThreePathCloneAlignMG.apply(x)
        self.assertEqual(len(outs), 3)
        router, dispatcher, shared = outs

        # Each path is an exact copy of the input.
        for out in (router, dispatcher, shared):
            self.assertEqual(out.shape, x.shape)
            np.testing.assert_array_equal(out.numpy(), x.numpy())

        # Clones must be distinct tensor objects, not aliases of the input or
        # of one another; returning ``x`` three times would collapse the three
        # autograd consumers the topology relies on.
        self.assertIsNot(router, x)
        self.assertIsNot(dispatcher, x)
        self.assertIsNot(shared, x)
        self.assertIsNot(router, dispatcher)
        self.assertIsNot(router, shared)
        self.assertIsNot(dispatcher, shared)

    def test_forward_keeps_paths_differentiable(self):
        # A differentiable input must yield differentiable clones, otherwise
        # the downstream backward sum can never be driven.
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        x.stop_gradient = False
        router, dispatcher, shared = ThreePathCloneAlignMG.apply(x)
        for out in (router, dispatcher, shared):
            self.assertFalse(out.stop_gradient)

    def test_backward_sums_all_three_paths(self):
        # Give every path a distinct per-element upstream coefficient so that
        # dropping or under-counting any single path changes x.grad.
        x = paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        x.stop_gradient = False
        router, dispatcher, shared = ThreePathCloneAlignMG.apply(x)

        c_router = paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        c_dispatcher = paddle.to_tensor(
            [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]
        )
        c_shared = paddle.to_tensor(
            [[100.0, 200.0, 300.0], [400.0, 500.0, 600.0]]
        )

        # dL/d(router)=c_router, etc.; each output is an identity clone of x,
        # so dL/dx = c_router + c_dispatcher + c_shared, elementwise.
        loss = (
            router * c_router + dispatcher * c_dispatcher + shared * c_shared
        ).sum()
        loss.backward()

        expected = np.array(
            [[111.0, 222.0, 333.0], [444.0, 555.0, 666.0]], dtype="float32"
        )
        self.assertIsNotNone(x.grad)
        # Integer-valued and exactly representable in float32, so the summation
        # order cannot perturb the result: an exact comparison is valid.
        np.testing.assert_array_equal(x.grad.numpy(), expected)

    def test_backward_unused_paths_contribute_zero(self):
        # Only the dispatcher path feeds the loss; router and shared are
        # returned but unconsumed. Their materialized-zero cotangents must not
        # perturb the gradient, and the used path must route correctly, so
        # x.grad equals exactly the dispatcher coefficient.
        x = paddle.to_tensor([[2.0, 3.0], [5.0, 7.0]])
        x.stop_gradient = False
        _router, dispatcher, _shared = ThreePathCloneAlignMG.apply(x)

        c_dispatcher = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        (dispatcher * c_dispatcher).sum().backward()

        self.assertIsNotNone(x.grad)
        np.testing.assert_array_equal(x.grad.numpy(), c_dispatcher.numpy())


if __name__ == "__main__":
    unittest.main()
