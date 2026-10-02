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
"""Tests for the PyLayer ``save_for_backward`` container shim.

The shim exists because saving a list or tuple through ``save_for_backward``
misbehaves while ``saved_tensors_hooks`` is installed, and real layers do save
lists that mix tensors with ``None`` and ints. What must hold is that the shim is
transparent: whatever structure a layer saved is the structure its backward gets
back, with or without hooks active, and gradients are unchanged.
"""

from __future__ import annotations

import unittest

import paddle
from paddle.autograd import PyLayer

from paddlefleet.activation_offload import install_pylayer_shim

# What backward actually received, per invocation. Asserting inside backward
# would swallow the failure into an autograd error, so it is recorded here and
# checked by the test body.
_SEEN: list = []


class _ContainerSaver(PyLayer):
    """Saves a list holding non-Tensors, a tuple, and a top-level tensor."""

    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward([x, None, 7], (y,), x)
        return x * 2.0 + y

    @staticmethod
    def backward(ctx, dout):
        _SEEN.append(ctx.saved_tensor())
        return dout * 2.0, dout


class _NothingSaved(PyLayer):
    """Calls ``saved_tensor()`` without ever calling ``save_for_backward``."""

    @staticmethod
    def forward(ctx, x):
        return x * 3.0

    @staticmethod
    def backward(ctx, dout):
        _SEEN.append(ctx.saved_tensor())
        return dout * 3.0


def _identity_hooks():
    """saved_tensors_hooks that change nothing, to isolate the shim itself."""
    return paddle.autograd.saved_tensors_hooks(lambda t: t, lambda t: t)


class TestPyLayerShim(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install_pylayer_shim()

    def setUp(self):
        _SEEN.clear()
        paddle.seed(46)

    def _run(self):
        x = paddle.to_tensor([1.0, 2.0, 3.0], stop_gradient=False)
        y = paddle.to_tensor([0.5, 0.5, 0.5], stop_gradient=False)
        out = _ContainerSaver.apply(x, y)
        out.sum().backward()
        return x, y, out

    def _assert_structure(self, saved, x, y):
        self.assertEqual(len(saved), 3, "three arguments were saved")
        first, second, third = saved
        # A list must come back a list, a tuple a tuple.
        self.assertIsInstance(first, list)
        self.assertIsInstance(second, tuple)
        # Non-Tensor elements survive verbatim.
        self.assertEqual(len(first), 3)
        self.assertIsNone(first[1])
        self.assertEqual(first[2], 7)
        # Tensor elements keep their values.
        self.assertTrue(paddle.equal_all(first[0], x))
        self.assertTrue(paddle.equal_all(second[0], y))
        self.assertTrue(paddle.equal_all(third, x))

    def test_containers_round_trip_without_hooks(self):
        x, y, _ = self._run()
        self.assertEqual(len(_SEEN), 1)
        self._assert_structure(_SEEN[0], x, y)

    def test_containers_round_trip_under_saved_tensors_hooks(self):
        # This is the case the shim exists for: saving a container while hooks
        # are installed.
        with _identity_hooks():
            x, y, _ = self._run()
        self.assertEqual(len(_SEEN), 1)
        self._assert_structure(_SEEN[0], x, y)

    def test_gradients_are_unaffected_by_the_shim(self):
        x, y, out = self._run()
        ref_x, ref_y = x.grad.numpy().copy(), y.grad.numpy().copy()
        _SEEN.clear()
        with _identity_hooks():
            x2, y2, out2 = self._run()
        self.assertEqual(out.numpy().tolist(), out2.numpy().tolist())
        self.assertEqual(ref_x.tolist(), x2.grad.numpy().tolist())
        self.assertEqual(ref_y.tolist(), y2.grad.numpy().tolist())

    def test_saved_tensor_without_save_for_backward(self):
        # No layout was recorded, so the shim must hand back whatever the
        # original implementation returns instead of trying to rebuild.
        x = paddle.to_tensor([1.0, 2.0], stop_gradient=False)
        _NothingSaved.apply(x).sum().backward()
        self.assertEqual(len(_SEEN), 1)
        self.assertIsNone(_SEEN[0])

    def test_install_is_idempotent(self):
        before = paddle.autograd.py_layer.PyLayerContext.save_for_backward
        install_pylayer_shim()
        install_pylayer_shim()
        after = paddle.autograd.py_layer.PyLayerContext.save_for_backward
        self.assertIs(before, after, "re-installing must not re-wrap")


if __name__ == "__main__":
    unittest.main()
