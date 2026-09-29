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

"""Behavior tests for ernie_pretrain models.utils.

Module under test:
    paddlefleet.cli.train.ernie_pretrain.models.utils

Scope and oracle policy
-----------------------
Every expected value below is hand-derived from the documented contract, not
read back from the code under test:

* ``global_training_logs_enabled`` -- ``get_global_training_logs`` is an
  isolated collaborator (patched with distinguishable responses). We verify
  the real ``isinstance(...) or logs.is_enabled()`` logic: a dict short-circuits
  (``is_enabled`` must NOT be consulted) and a non-dict's ``is_enabled()`` return
  value is actually consumed (True AND False cases).
* ``detach_and_requires_grad_`` -- content preservation, ``stop_gradient``
  propagation per argument, ``None`` pass-through, and detachment (distinct
  objects with a broken graph).
* ``FakeClone`` -- forward reproduces input content; backward is the identity on
  a *non-uniform* upstream gradient (a scaling bug would be caught).
* ``FakeGather`` -- forward is a row gather honoring index order; the empty-index
  branch returns a 0-row tensor; backward routes each upstream row back to its
  source row and zeros the unselected rows (checked with a non-uniform upstream).
* ``manual_backward`` -- the first-forward path returns ``(None, out)`` with
  ``out`` carrying ``f``'s real output; the replay path returns a backward closure
  whose gradients equal the hand-derived Jacobian applied to a non-uniform
  upstream gradient.
* ``inplace_offload`` -- content is preserved across the in-place buffer swap.

``FusedUnpermutation`` is intentionally NOT covered: it hard-depends on the
optional ``moe_permutation`` extension (imported at module load and, per the
production warning, frequently absent), so a genuine numeric oracle cannot be
built in a no-card CPU environment.

Paddle is an optional heavy dependency imported at module top level, so every
test skips with an honest reason when the import fails. The local environment
has no paddle installed.
"""

import unittest
from unittest import mock

import numpy as np

try:
    import paddle

    from paddlefleet.cli.train.ernie_pretrain.models import utils as utils_mod
    from paddlefleet.cli.train.ernie_pretrain.models.utils import (
        FakeClone,
        FakeGather,
        detach_and_requires_grad_,
        global_training_logs_enabled,
        inplace_offload,
        manual_backward,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle (or the module) unavailable in this env
    paddle = None
    utils_mod = None
    FakeClone = FakeGather = None
    detach_and_requires_grad_ = global_training_logs_enabled = None
    inplace_offload = manual_backward = None
    _IMPORT_ERROR = exc


class GlobalTrainingLogsEnabledTest(unittest.TestCase):
    """global_training_logs_enabled: real isinstance/or logic over a collaborator."""

    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"paddle/utils import unavailable: {_IMPORT_ERROR!r}")

    def test_dict_logs_short_circuit_without_calling_is_enabled(self):
        # A dict must be treated as enabled purely via isinstance; the `or`
        # must short-circuit so is_enabled() is never consulted.
        class _DictWithTrap(dict):
            def is_enabled(self):  # pragma: no cover - must not run
                raise AssertionError("is_enabled must not be called for a dict")

        with mock.patch.object(
            utils_mod, "get_global_training_logs", return_value=_DictWithTrap()
        ):
            self.assertIs(global_training_logs_enabled(), True)

    def test_non_dict_logs_consume_is_enabled_return_value(self):
        # For a non-dict collaborator the boolean actually returned by
        # is_enabled() must be propagated verbatim (both branches).
        class _Logs:
            def __init__(self, enabled):
                self._enabled = enabled

            def is_enabled(self):
                return self._enabled

        with mock.patch.object(
            utils_mod, "get_global_training_logs", return_value=_Logs(True)
        ):
            self.assertIs(global_training_logs_enabled(), True)

        with mock.patch.object(
            utils_mod, "get_global_training_logs", return_value=_Logs(False)
        ):
            self.assertIs(global_training_logs_enabled(), False)


class DetachAndRequiresGradTest(unittest.TestCase):
    """detach_and_requires_grad_: content, stop_gradient, None, detachment."""

    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"paddle/utils import unavailable: {_IMPORT_ERROR!r}")
        paddle.set_device("cpu")

    def test_preserves_content_and_per_arg_stop_gradient(self):
        trainable = paddle.to_tensor([1.0, 2.0, 3.0])
        trainable.stop_gradient = False
        frozen = paddle.to_tensor([4.0, 5.0])
        frozen.stop_gradient = True

        out = detach_and_requires_grad_(trainable, frozen)

        self.assertEqual(len(out), 2)
        # stop_gradient copied argument-by-argument (order preserved).
        self.assertFalse(out[0].stop_gradient)
        self.assertTrue(out[1].stop_gradient)
        # Values are unchanged.
        np.testing.assert_array_equal(out[0].numpy(), [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(out[1].numpy(), [4.0, 5.0])
        # Results are detached copies, not the original tensors.
        self.assertIsNot(out[0], trainable)
        self.assertIsNot(out[1], frozen)

    def test_none_arguments_pass_through_in_place(self):
        x = paddle.to_tensor([7.0])
        x.stop_gradient = False
        out = detach_and_requires_grad_(None, x, None)
        self.assertEqual(len(out), 3)
        self.assertIsNone(out[0])
        self.assertIsNone(out[2])
        self.assertFalse(out[1].stop_gradient)
        np.testing.assert_array_equal(out[1].numpy(), [7.0])


class FakeCloneTest(unittest.TestCase):
    """FakeClone: forward reproduces input; backward is the identity."""

    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"paddle/utils import unavailable: {_IMPORT_ERROR!r}")
        paddle.set_device("cpu")

    def test_forward_reproduces_input_content(self):
        x = paddle.arange(6, dtype="float32").reshape([2, 3])
        out = FakeClone.apply(x)
        np.testing.assert_array_equal(out.numpy(), x.numpy())

    def test_backward_is_identity_on_nonuniform_grad(self):
        # A non-uniform upstream gradient distinguishes the identity from any
        # scaling/permuting backward; sum() would only give a vector of ones.
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        x.stop_gradient = False
        out = FakeClone.apply(x)
        upstream = paddle.to_tensor([0.5, -2.0, 4.0])
        paddle.autograd.backward([out], [upstream])
        self.assertIsNotNone(x.grad)
        np.testing.assert_array_equal(x.grad.numpy(), [0.5, -2.0, 4.0])


class FakeGatherTest(unittest.TestCase):
    """FakeGather: ordered row gather + scatter-back backward."""

    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"paddle/utils import unavailable: {_IMPORT_ERROR!r}")
        paddle.set_device("cpu")

    def test_forward_gathers_rows_in_index_order(self):
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        # Index order (2, 0) must be honored -- not sorted.
        indices = paddle.to_tensor([2, 0], dtype="int64")
        out = FakeGather.apply(x, indices)
        np.testing.assert_array_equal(out.numpy(), [[5.0, 6.0], [1.0, 2.0]])

    def test_forward_empty_indices_returns_zero_row_tensor(self):
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        indices = paddle.to_tensor([], dtype="int64")
        out = FakeGather.apply(x, indices)
        self.assertEqual(list(out.shape), [0, 2])

    def test_backward_routes_rows_and_zeros_unselected(self):
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        x.stop_gradient = False
        indices = paddle.to_tensor([2, 0], dtype="int64")
        out = FakeGather.apply(x, indices)
        # Non-uniform per-row upstream: out row0<-x row2, out row1<-x row0.
        upstream = paddle.to_tensor([[10.0, 20.0], [30.0, 40.0]])
        paddle.autograd.backward([out], [upstream])
        self.assertIsNotNone(x.grad)
        expected = np.array(
            [[30.0, 40.0], [0.0, 0.0], [10.0, 20.0]], dtype="float32"
        )
        np.testing.assert_array_equal(x.grad.numpy(), expected)


class ManualBackwardTest(unittest.TestCase):
    """manual_backward: first-forward output and replay-path gradients."""

    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"paddle/utils import unavailable: {_IMPORT_ERROR!r}")
        paddle.set_device("cpu")

    def test_first_forward_returns_none_fn_and_real_output(self):
        def scale3(t):
            return t * 3.0

        x = paddle.to_tensor([1.0, 2.0, 3.0])
        x.stop_gradient = False
        bwd_f, out = manual_backward(scale3, True, x)

        self.assertIsNone(bwd_f)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 1)
        # out carries f's real result, not merely some tuple.
        np.testing.assert_allclose(out[0].numpy(), [3.0, 6.0, 9.0], rtol=1e-6)

    def test_replay_path_gradients_match_jacobian(self):
        # f(x) = 3x  ->  dx = 3 * upstream. Non-uniform upstream distinguishes
        # the true Jacobian from an identity or a wrong scale.
        def scale3(t):
            return t * 3.0

        x = paddle.to_tensor([1.0, 2.0, 3.0])
        x.stop_gradient = False
        bwd_f, out = manual_backward(scale3, False, x)

        self.assertIsNotNone(bwd_f)
        self.assertIsInstance(out, tuple)
        np.testing.assert_allclose(out[0].numpy(), [3.0, 6.0, 9.0], rtol=1e-6)

        upstream = paddle.to_tensor([0.5, -2.0, 4.0])
        grads = bwd_f(upstream)
        self.assertEqual(len(grads), 1)
        np.testing.assert_allclose(
            grads[0].numpy(), [1.5, -6.0, 12.0], rtol=1e-6
        )


class InplaceOffloadTest(unittest.TestCase):
    """inplace_offload: values survive the in-place buffer swap."""

    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"paddle/utils import unavailable: {_IMPORT_ERROR!r}")
        paddle.set_device("cpu")

    def test_content_preserved_after_offload(self):
        x = paddle.arange(6, dtype="float32").reshape([2, 3])
        before = x.numpy().copy()
        inplace_offload(x)
        np.testing.assert_array_equal(x.numpy(), before)


if __name__ == "__main__":
    unittest.main()
