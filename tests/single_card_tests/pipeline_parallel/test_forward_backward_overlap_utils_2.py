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

"""CPU-only behavior tests for the pure logic in
``paddlefleet.pipeline_parallel.pp_utils.forward_backward_overlap_utils``.

Only the device-independent, single-process helpers are exercised here:

* ``detach_and_requires_grad`` -- detaches while preserving container type,
  element order, per-element ``stop_gradient`` and tensor *values*.
* ``clone_and_clear_dataptr`` -- structural clone via ``FakeClone`` that drops
  ``None`` / non-tensor entries and preserves shape/dtype/order. Because
  ``FakeClone.forward`` returns ``paddle.empty_like``, the clone's *contents*
  are uninitialised and are deliberately NOT asserted.
* ``ScheduleNode`` forward/backward and ``ScheduleChunk`` composition -- real
  eager autograd on CPU with hand-derived gradients (no mocked kernels).

The pipeline scheduling that crosses ranks (the real send/recv overlap driven
by ``first_forward`` RNG/AMP capture and the recompute path) requires a real
process group and GPU AMP state; faking those to assert trivia would prove
nothing, so they are intentionally left to multi-card tests.

Expected gradients are derived by hand from the closed-form of each ``fwd_func``
(e.g. d(x*2)/dx = 2), never by re-running the production code as its own oracle.
"""

import unittest

try:
    import numpy as np
    import paddle

    from paddlefleet.pipeline_parallel.pp_utils.forward_backward_overlap_utils import (
        ScheduleChunk,
        ScheduleNode,
        clone_and_clear_dataptr,
        detach_and_requires_grad,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # honest capability probe
    _IMPORT_ERROR = exc

PADDLE_AVAILABLE = _IMPORT_ERROR is None
SKIP_REASON = (
    "paddle / paddlefleet forward_backward_overlap_utils import failed: "
    f"{_IMPORT_ERROR!r}"
    if _IMPORT_ERROR is not None
    else ""
)


@unittest.skipUnless(PADDLE_AVAILABLE, SKIP_REASON)
class TestDetachAndRequiresGrad(unittest.TestCase):
    """``detach_and_requires_grad`` container/stop_gradient/value contract."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_single_tensor_preserves_value_and_flag_false(self):
        x = paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        x.stop_gradient = False
        out = detach_and_requires_grad(x)
        # A fresh detached tensor, not the same object.
        self.assertIsNot(out, x)
        self.assertFalse(out.stop_gradient)
        np.testing.assert_array_equal(out.numpy(), x.numpy())

    def test_single_tensor_preserves_flag_true(self):
        x = paddle.to_tensor([[7.0, 8.0]])
        x.stop_gradient = True
        out = detach_and_requires_grad(x)
        self.assertTrue(out.stop_gradient)
        np.testing.assert_array_equal(out.numpy(), x.numpy())

    def test_tuple_preserves_order_flags_and_values(self):
        # Distinguishable shapes/values so a swap or inverted flag is caught.
        a = paddle.to_tensor([[1.0, 2.0, 3.0]])
        a.stop_gradient = False
        b = paddle.to_tensor([[10.0, 20.0]])
        b.stop_gradient = True
        out = detach_and_requires_grad((a, b))
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        self.assertFalse(out[0].stop_gradient)
        self.assertTrue(out[1].stop_gradient)
        np.testing.assert_array_equal(out[0].numpy(), a.numpy())
        np.testing.assert_array_equal(out[1].numpy(), b.numpy())

    def test_list_stays_list(self):
        x = paddle.to_tensor([1.0, 2.0])
        x.stop_gradient = False
        out = detach_and_requires_grad([x])
        self.assertIsInstance(out, list)
        self.assertFalse(out[0].stop_gradient)
        np.testing.assert_array_equal(out[0].numpy(), x.numpy())

    def test_nested_tuple_structure_and_values(self):
        x = paddle.to_tensor([[1.0, 2.0]])
        x.stop_gradient = False
        y = paddle.to_tensor([[3.0, 4.0, 5.0]])
        y.stop_gradient = True
        out = detach_and_requires_grad(((x,), y))
        self.assertIsInstance(out, tuple)
        self.assertIsInstance(out[0], tuple)
        np.testing.assert_array_equal(out[0][0].numpy(), x.numpy())
        self.assertFalse(out[0][0].stop_gradient)
        np.testing.assert_array_equal(out[1].numpy(), y.numpy())
        self.assertTrue(out[1].stop_gradient)

    def test_dict_preserves_keys_flags_and_values(self):
        x = paddle.to_tensor([[1.0, 2.0]])
        x.stop_gradient = True
        y = paddle.to_tensor([[9.0]])
        y.stop_gradient = False
        out = detach_and_requires_grad({"a": x, "b": y})
        self.assertIsInstance(out, dict)
        self.assertEqual(set(out), {"a", "b"})
        self.assertTrue(out["a"].stop_gradient)
        self.assertFalse(out["b"].stop_gradient)
        np.testing.assert_array_equal(out["a"].numpy(), x.numpy())
        np.testing.assert_array_equal(out["b"].numpy(), y.numpy())

    def test_none_entry_passes_through_in_container(self):
        # A bare None inside a tuple is not a Tensor, so it is returned as-is.
        out = detach_and_requires_grad((None,))
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 1)
        self.assertIsNone(out[0])


@unittest.skipUnless(PADDLE_AVAILABLE, SKIP_REASON)
class TestCloneAndClearDataptr(unittest.TestCase):
    """``clone_and_clear_dataptr`` filtering / structure / shape contract.

    ``FakeClone.forward`` returns ``paddle.empty_like`` so the *values* are
    uninitialised; only shape, dtype, ordering, container type and the
    None/non-tensor filtering are verifiable and asserted here.
    """

    def setUp(self):
        paddle.set_device("cpu")

    def test_single_tensor_matches_shape_and_dtype(self):
        x = paddle.zeros([2, 3], dtype="float32")
        out = clone_and_clear_dataptr(x)
        self.assertIsInstance(out, paddle.Tensor)
        self.assertIsNot(out, x)
        self.assertEqual(list(out.shape), [2, 3])
        self.assertEqual(out.dtype, x.dtype)

    def test_tuple_filters_none_and_keeps_order(self):
        # Distinct shapes make an order swap or wrong-element survival visible;
        # the None must be dropped, shrinking length from 3 to 2.
        t0 = paddle.zeros([2, 3], dtype="float32")
        t1 = paddle.zeros([4, 5], dtype="int64")
        out = clone_and_clear_dataptr((t0, None, t1))
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        self.assertEqual(list(out[0].shape), [2, 3])
        self.assertEqual(out[0].dtype, paddle.float32)
        self.assertEqual(list(out[1].shape), [4, 5])
        self.assertEqual(out[1].dtype, paddle.int64)

    def test_list_stays_list_and_filters_non_tensor(self):
        t0 = paddle.zeros([1, 7], dtype="float32")
        out = clone_and_clear_dataptr([t0, "not_a_tensor"])
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 1)
        self.assertEqual(list(out[0].shape), [1, 7])

    def test_dict_filters_none_and_preserves_surviving_keys(self):
        d = {
            "keep": paddle.zeros([3, 2], dtype="float32"),
            "drop": None,
        }
        out = clone_and_clear_dataptr(d)
        self.assertIsInstance(out, dict)
        self.assertEqual(set(out), {"keep"})
        self.assertEqual(list(out["keep"].shape), [3, 2])

    def test_clear_dataptr_flag_still_returns_filtered_structure(self):
        # Exercises the clear_dataptr=True branch: the pointer-clearing loop
        # must run without altering the filtered container structure.
        t0 = paddle.zeros([2, 2], dtype="float32")
        out = clone_and_clear_dataptr((t0, None), clear_dataptr=True)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 1)
        self.assertIsInstance(out[0], paddle.Tensor)


@unittest.skipUnless(PADDLE_AVAILABLE, SKIP_REASON)
class TestScheduleNodeForwardBackward(unittest.TestCase):
    """``ScheduleNode`` non-recompute forward/backward with hand-derived math."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_forward_runs_fwd_func_on_detached_inputs(self):
        x = paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        x.stop_gradient = False

        def fwd_func(inputs, **kwargs):
            return inputs * 2

        node = ScheduleNode(fwd_func=fwd_func)
        out = node.forward(x)
        # Forward returns the real fwd_func output (not the FakeClone), so its
        # value is fully determined: 2 * x.
        np.testing.assert_array_equal(
            out.numpy(), np.array([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]])
        )
        # Inputs are stored detached (new object) but value-preserving.
        self.assertIsNot(node.inputs, x)
        np.testing.assert_array_equal(node.inputs.numpy(), x.numpy())
        self.assertFalse(node.inputs.stop_gradient)

    def test_backward_with_output_grad_is_two_times_upstream(self):
        x = paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        x.stop_gradient = False

        def fwd_func(inputs, **kwargs):
            return inputs * 2

        node = ScheduleNode(fwd_func=fwd_func)
        node.forward(x)
        # Non-uniform upstream so a transpose/scale error would surface.
        g = paddle.to_tensor([[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]])
        grads = node.backward(output_grad=g)
        self.assertIsInstance(grads, tuple)
        self.assertEqual(len(grads), 1)
        # d(2x)/dx = 2, so grad = 2 * upstream.
        np.testing.assert_allclose(
            grads[0].numpy(),
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            rtol=1e-6,
            atol=1e-6,
        )
        # States are cleared after backward.
        self.assertIsNone(node.inputs)
        self.assertIsNone(node.outputs)

    def test_scale_loss_factor_divides_forward_output(self):
        x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
        x.stop_gradient = False

        def fwd_func(inputs, **kwargs):
            return inputs.sum()

        scaled = ScheduleNode(fwd_func=fwd_func)
        scaled.scale_loss_factor = 4.0
        scaled_out = scaled.forward(x)
        # sum = 10, divided by 4 -> 2.5.
        self.assertAlmostEqual(float(scaled_out.numpy()), 2.5, places=6)

        plain = ScheduleNode(fwd_func=fwd_func)
        plain_out = plain.forward(x)
        # Without the factor the same input yields 10, proving it was consumed.
        self.assertAlmostEqual(float(plain_out.numpy()), 10.0, places=6)

    def test_labels_branch_forward_and_backward(self):
        # labels != None routes through the labels branch AND keeps the output
        # data (clear_dataptr = labels is None = False).
        x = paddle.to_tensor([[1.0, 2.0, 3.0]])
        x.stop_gradient = False
        labels = paddle.to_tensor([[2.0, 3.0, 4.0]])
        labels.stop_gradient = True

        def fwd_func(inputs, labels, **kwargs):
            return inputs * labels

        node = ScheduleNode(fwd_func=fwd_func)
        node.labels = labels
        out = node.forward(x)
        np.testing.assert_array_equal(out.numpy(), np.array([[2.0, 6.0, 12.0]]))
        g = paddle.to_tensor([[1.0, 1.0, 1.0]])
        grads = node.backward(output_grad=g)
        # d(x*labels)/dx = labels, upstream is ones -> grad == labels.
        np.testing.assert_allclose(
            grads[0].numpy(),
            np.array([[2.0, 3.0, 4.0]]),
            rtol=1e-6,
            atol=1e-6,
        )
        self.assertIsNone(node.labels)


@unittest.skipUnless(PADDLE_AVAILABLE, SKIP_REASON)
class TestScheduleChunk(unittest.TestCase):
    """``ScheduleChunk`` node validation and forward/backward composition."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_accepts_schedule_node(self):
        # A chunk of valid ScheduleNodes must construct without error.
        node = ScheduleNode(fwd_func=lambda x, **kw: x)
        chunk = ScheduleChunk([node])
        self.assertEqual(chunk.nodes, [node])

    def test_accepts_nested_chunk(self):
        inner = ScheduleChunk([ScheduleNode(fwd_func=lambda x, **kw: x)])
        outer = ScheduleChunk([inner])
        self.assertIs(outer.nodes[0], inner)

    def test_rejects_non_node(self):
        with self.assertRaises(AssertionError):
            ScheduleChunk(["not_a_node"])

    def test_forward_composes_nodes_in_order(self):
        x = paddle.to_tensor([[1.0, 2.0]])
        x.stop_gradient = False
        n1 = ScheduleNode(fwd_func=lambda inputs, **kw: inputs + 1)
        n2 = ScheduleNode(fwd_func=lambda inputs, **kw: inputs * 3)
        chunk = ScheduleChunk([n1, n2])
        out = chunk.forward(x)
        # (x + 1) * 3, applied in order.
        np.testing.assert_array_equal(out.numpy(), np.array([[6.0, 9.0]]))

    def test_backward_chains_reversed(self):
        x = paddle.to_tensor([[1.0, 2.0]])
        x.stop_gradient = False
        n1 = ScheduleNode(fwd_func=lambda inputs, **kw: inputs + 1)
        n2 = ScheduleNode(fwd_func=lambda inputs, **kw: inputs * 3)
        chunk = ScheduleChunk([n1, n2])
        chunk.forward(x)
        g = paddle.to_tensor([[0.5, 2.0]])
        grad = chunk.backward(g)
        # d/dx of ((x+1)*3) = 3, so grad w.r.t. input = 3 * upstream.
        if isinstance(grad, (tuple, list)):
            grad_arr = grad[0].numpy()
        else:
            grad_arr = grad.numpy()
        np.testing.assert_allclose(
            grad_arr, np.array([[1.5, 6.0]]), rtol=1e-6, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
