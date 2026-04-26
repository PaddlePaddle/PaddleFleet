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

import unittest
from types import SimpleNamespace

import paddle

from paddlefleet.transformer.moe.moe_utils import AddAuxiliaryLoss


class TestAddAuxiliaryLoss(unittest.TestCase):
    def test_backward_propagates_required_x_and_aux_grads(self):
        x = paddle.ones([2, 3], dtype="float32")
        x.stop_gradient = False
        aux_loss = paddle.to_tensor([3.0], dtype="float32")
        aux_loss.stop_gradient = False

        out = AddAuxiliaryLoss.apply(x, aux_loss)
        out.sum().backward()

        self.assertEqual(x.grad.numpy().tolist(), [[1.0] * 3, [1.0] * 3])
        self.assertEqual(aux_loss.grad.numpy().tolist(), [1.0])

    def test_backward_respects_stop_gradient_flags(self):
        ctx = SimpleNamespace()
        x = paddle.ones([2, 3], dtype="float32")
        x.stop_gradient = True
        aux_loss = paddle.to_tensor([3.0], dtype="float32")
        aux_loss.stop_gradient = True

        out = AddAuxiliaryLoss.forward(ctx, x, aux_loss)
        grad_x, grad_loss = AddAuxiliaryLoss.backward(
            ctx, paddle.ones_like(out)
        )

        self.assertIsNone(grad_x)
        self.assertIsNone(grad_loss)

    def test_forward_requires_scalar_aux_loss(self):
        ctx = SimpleNamespace()
        with self.assertRaises(AssertionError):
            AddAuxiliaryLoss.forward(
                ctx,
                paddle.ones([2, 3], dtype="float32"),
                paddle.ones([2], dtype="float32"),
            )


if __name__ == "__main__":
    unittest.main()
