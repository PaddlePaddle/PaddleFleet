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
"""Accumulation order of the three MoE-input gradient branches.

``ThreePathCloneAlignMG`` sums the router / dispatcher / shared-expert gradients
in an order that is selected by the alignment switches. bf16 addition is not
associative, so the order is observable: Megatron accumulates the shared branch
last, PaddleFleet historically accumulated the router last. The switch state is
process-global, so every case restores it.
"""

import unittest

import paddle

from paddlefleet.transformer.moe.moe_layer import ThreePathCloneAlignMG
from paddlefleet.utils import (
    set_kimik2_accuracy_compatible,
    use_kimik2_accuracy_compatible,
)


def _branch_grads(seed=0):
    """Three distinct bf16 gradients whose accumulation order is observable."""
    paddle.seed(seed)
    shape = [64, 32]
    g_router = (paddle.randn(shape) * 1e-3).astype("bfloat16")
    g_dispatcher = paddle.randn(shape).astype("bfloat16")
    g_shared = (paddle.randn(shape) * 1e-2).astype("bfloat16")
    return g_router, g_dispatcher, g_shared


def _accumulate(g_router, g_dispatcher, g_shared, shared_last):
    if shared_last:
        return (g_dispatcher + g_router) + g_shared
    return (g_dispatcher + g_shared) + g_router


def _input_grad(g_router, g_dispatcher, g_shared):
    """Run the clone's backward and return the gradient of its input."""
    x = paddle.zeros(g_router.shape, dtype="bfloat16")
    x.stop_gradient = False
    router, dispatcher, shared = ThreePathCloneAlignMG.apply(x)
    paddle.autograd.backward(
        [router, dispatcher, shared], [g_router, g_dispatcher, g_shared]
    )
    return x.grad


class TestThreePathGradAccumOrder(unittest.TestCase):
    def setUp(self):
        previous = use_kimik2_accuracy_compatible()
        self.addCleanup(set_kimik2_accuracy_compatible, previous)
        set_kimik2_accuracy_compatible(False)
        self.grads = _branch_grads()
        shared_last = _accumulate(*self.grads, shared_last=True)
        router_last = _accumulate(*self.grads, shared_last=False)
        # Guard the premise: if bf16 rounding made the two orders agree the test
        # below would pass for the wrong reason.
        self.assertFalse(
            bool(paddle.all(shared_last == router_last)),
            "the crafted gradients do not expose the accumulation order",
        )
        self.shared_last = shared_last
        self.router_last = router_last

    def test_default_keeps_router_last(self):
        grad = _input_grad(*self.grads)
        self.assertTrue(bool(paddle.all(grad == self.router_last)))

    def test_kimik2_switch_selects_shared_last(self):
        set_kimik2_accuracy_compatible(True)
        grad = _input_grad(*self.grads)
        self.assertTrue(bool(paddle.all(grad == self.shared_last)))


if __name__ == "__main__":
    unittest.main()
