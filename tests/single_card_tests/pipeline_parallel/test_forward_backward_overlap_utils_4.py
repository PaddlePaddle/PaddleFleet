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

"""CPU-only behavior tests for the device-independent pure logic in
``paddlefleet.pipeline_parallel.pp_utils.forward_backward_overlap_utils``.

Only the pieces whose correctness is local bookkeeping / branch selection /
autograd wiring are exercised here. These run on a single CPU process with no
process group:

* ``detach_and_requires_grad`` - structure walk that detaches tensors, copies
  ``stop_gradient`` across, recurses into nested lists, and passes non-tensors
  (and ``None``) through unchanged; the dict branch keeps ``None`` values.
* ``FakeClone`` - a PyLayer whose forward returns a *fresh* ``empty_like``
  buffer (shape/dtype preserved, contents intentionally NOT preserved so the
  original data pointer can later be cleared) and whose backward is the
  identity on the upstream gradient.
* ``clone_and_clear_dataptr`` - drops ``None`` / non-tensor entries from lists,
  drops non-tensor values from dicts, and preserves the container kind
  (tuple->tuple, list->list, dict->dict, single tensor->tensor). Element
  *values* are not asserted because ``FakeClone`` deliberately produces
  uninitialized buffers.
* ``ScheduleNode.forward`` / ``ScheduleNode.backward`` (the non-recompute
  path) - the returned tensor is the real forward value, ``self.inputs`` holds
  a detached copy, ``scale_loss_factor`` divides the output, the ``labels``
  branch feeds a second positional argument, and ``backward`` runs a real
  ``paddle.autograd.backward`` returning the per-input gradients.
* ``ScheduleChunk`` - node-type validation, forward composition across nodes,
  and reversed backward composition end-to-end.

The recompute branch (``first_forward`` / ``use_recompute``) is intentionally
NOT covered: it depends on a live AMP dygraph tracer and on
``custom_state_manager`` state functions, and verifying it meaningfully would
require replacing the unit under test with mocks. That RNG/AMP-preservation
behaviour belongs to a real training run, not a CPU pure-logic unit test.

Reference values are derived by hand with numpy and do not call any production
helper, so the test and the code under test never share a source of truth.
"""

import unittest

try:
    import numpy as np
    import paddle

    from paddlefleet.pipeline_parallel.pp_utils.forward_backward_overlap_utils import (
        FakeClone,
        ScheduleChunk,
        ScheduleNode,
        clone_and_clear_dataptr,
        detach_and_requires_grad,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # honest dependency probe
    np = None
    paddle = None
    FakeClone = None
    ScheduleChunk = None
    ScheduleNode = None
    clone_and_clear_dataptr = None
    detach_and_requires_grad = None
    _IMPORT_ERROR = exc

_HAS_DEPS = _IMPORT_ERROR is None
_SKIP_REASON = (
    f"paddle / paddlefleet not importable on this CPU host: {_IMPORT_ERROR!r}"
)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestDetachAndRequiresGrad(unittest.TestCase):
    """``detach_and_requires_grad`` structure walk and stop_gradient copy."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_single_tensor_detaches_and_copies_stop_gradient(self):
        src = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
        src.stop_gradient = False
        out = detach_and_requires_grad(src)
        # A fresh tensor object, detached from the graph, but same contents.
        self.assertIsNot(out, src)
        self.assertTrue(out.stop_gradient is False)
        np.testing.assert_array_equal(out.numpy(), [[1.0, 2.0], [3.0, 4.0]])

    def test_stop_gradient_true_is_preserved(self):
        src = paddle.to_tensor([5.0, 6.0], dtype="float32")
        src.stop_gradient = True
        out = detach_and_requires_grad(src)
        self.assertTrue(out.stop_gradient is True)
        np.testing.assert_array_equal(out.numpy(), [5.0, 6.0])

    def test_list_preserves_order_content_and_passes_through_nontensors(self):
        t0 = paddle.to_tensor([1.0, 2.0], dtype="float32")
        t0.stop_gradient = False
        result = detach_and_requires_grad([t0, "flag", 7, None])
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 4)
        self.assertIsNot(result[0], t0)
        self.assertFalse(result[0].stop_gradient)
        np.testing.assert_array_equal(result[0].numpy(), [1.0, 2.0])
        # Non-tensor / None entries are forwarded unchanged, in place.
        self.assertEqual(result[1], "flag")
        self.assertEqual(result[2], 7)
        self.assertIsNone(result[3])

    def test_tuple_input_yields_tuple(self):
        a = paddle.to_tensor([1.0], dtype="float32")
        b = paddle.to_tensor([2.0], dtype="float32")
        result = detach_and_requires_grad((a, b))
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        np.testing.assert_array_equal(result[0].numpy(), [1.0])
        np.testing.assert_array_equal(result[1].numpy(), [2.0])

    def test_nested_list_is_recursed(self):
        outer = paddle.to_tensor([1.0], dtype="float32")
        inner0 = paddle.to_tensor([2.0], dtype="float32")
        inner1 = paddle.to_tensor([3.0], dtype="float32")
        result = detach_and_requires_grad([outer, [inner0, inner1]])
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[1], list)
        np.testing.assert_array_equal(result[0].numpy(), [1.0])
        np.testing.assert_array_equal(result[1][0].numpy(), [2.0])
        np.testing.assert_array_equal(result[1][1].numpy(), [3.0])

    def test_dict_keeps_none_values_and_detaches_tensors(self):
        t = paddle.to_tensor([9.0, 8.0], dtype="float32")
        t.stop_gradient = False
        result = detach_and_requires_grad({"x": t, "y": None})
        self.assertIsInstance(result, dict)
        self.assertEqual(set(result), {"x", "y"})
        # Unlike clone_and_clear_dataptr, the dict branch retains None.
        self.assertIsNone(result["y"])
        self.assertIsNot(result["x"], t)
        self.assertFalse(result["x"].stop_gradient)
        np.testing.assert_array_equal(result["x"].numpy(), [9.0, 8.0])


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestFakeClone(unittest.TestCase):
    """``FakeClone`` forward allocates a fresh buffer; backward is identity."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_forward_returns_fresh_buffer_same_shape_and_dtype(self):
        x = paddle.to_tensor([[1.0, 2.0, 3.0]], dtype="float32")
        y = FakeClone.apply(x)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.dtype, x.dtype)
        # empty_like allocates new storage; it must not alias the input, since
        # the caller may later clear the input's data pointer.
        self.assertNotEqual(y.data_ptr(), x.data_ptr())

    def test_static_backward_passes_gradient_through_unchanged(self):
        grad = paddle.to_tensor([[1.5, -2.0], [0.25, 4.0]], dtype="float32")
        out = FakeClone.backward(None, grad)
        self.assertIs(out, grad)

    def test_apply_backward_is_identity_gradient(self):
        x = paddle.to_tensor([[2.0, -1.0], [0.0, 3.0]], dtype="float32")
        x.stop_gradient = False
        y = FakeClone.apply(x)
        upstream = paddle.to_tensor([[7.0, 5.0], [-3.0, 2.0]], dtype="float32")
        paddle.autograd.backward([y], [upstream])
        self.assertIsNotNone(x.grad)
        # d(FakeClone)/dx is identity, so the input grad equals the upstream.
        np.testing.assert_array_equal(x.grad.numpy(), [[7.0, 5.0], [-3.0, 2.0]])


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestCloneAndClearDataptr(unittest.TestCase):
    """``clone_and_clear_dataptr`` filtering and container-kind preservation."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_list_drops_none_and_nontensor_preserves_shapes(self):
        t1 = paddle.to_tensor([[1.0, 2.0, 3.0]], dtype="float32")
        t2 = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
        result = clone_and_clear_dataptr([t1, None, "x", t2])
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        # Only the two real tensors survive, in order; shapes/dtypes pass
        # through empty_like (contents are intentionally uninitialized).
        self.assertEqual(result[0].shape, [1, 3])
        self.assertEqual(result[1].shape, [2, 2])
        self.assertEqual(result[0].dtype, t1.dtype)
        self.assertEqual(result[1].dtype, t2.dtype)

    def test_tuple_input_yields_tuple(self):
        t1 = paddle.to_tensor([1.0, 2.0], dtype="float32")
        t2 = paddle.to_tensor([3.0], dtype="float32")
        result = clone_and_clear_dataptr((t1, t2))
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].shape, [2])
        self.assertEqual(result[1].shape, [1])

    def test_dict_drops_none_and_nontensor_values(self):
        t = paddle.to_tensor([[5.0, 6.0]], dtype="float32")
        result = clone_and_clear_dataptr({"a": t, "b": None, "c": 5})
        self.assertIsInstance(result, dict)
        self.assertEqual(set(result), {"a"})
        self.assertEqual(result["a"].shape, [1, 2])
        self.assertEqual(result["a"].dtype, t.dtype)

    def test_single_tensor_returns_single_tensor(self):
        t = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
        result = clone_and_clear_dataptr(t)
        self.assertIsInstance(result, paddle.Tensor)
        self.assertEqual(result.shape, [2, 2])
        self.assertEqual(result.dtype, t.dtype)
        self.assertNotEqual(result.data_ptr(), t.data_ptr())


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestScheduleNodeForward(unittest.TestCase):
    """``ScheduleNode.forward`` (non-recompute) value and state contract."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_forward_returns_real_value_and_stores_detached_input(self):
        x = paddle.to_tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32"
        )
        x.stop_gradient = False
        node = ScheduleNode(fwd_func=lambda inputs, **kw: inputs * 3)
        out = node.forward(x)
        # Returned value is the genuine forward output, not the fake clone.
        np.testing.assert_array_equal(out.numpy(), np.asarray(x.numpy()) * 3.0)
        # self.inputs is a detached copy carrying the same content/flag.
        self.assertIsNot(node.inputs, x)
        self.assertFalse(node.inputs.stop_gradient)
        np.testing.assert_array_equal(node.inputs.numpy(), x.numpy())
        # self.outputs is a fresh (fake-cloned) buffer of matching shape.
        self.assertEqual(node.outputs.shape, [2, 3])

    def test_scale_loss_factor_divides_output(self):
        x = paddle.to_tensor([[2.0, 4.0], [6.0, 8.0]], dtype="float32")
        x.stop_gradient = False
        node = ScheduleNode(fwd_func=lambda inputs, **kw: inputs * 3)
        node.scale_loss_factor = 2.0
        out = node.forward(x)
        # (x * 3) / 2 computed independently.
        np.testing.assert_allclose(
            out.numpy(), np.asarray(x.numpy()) * 3.0 / 2.0, rtol=1e-6, atol=0
        )

    def test_labels_branch_feeds_second_positional_argument(self):
        x = paddle.to_tensor([[1.0, 1.0], [1.0, 1.0]], dtype="float32")
        x.stop_gradient = False
        labels = paddle.to_tensor([[10.0, 20.0], [30.0, 40.0]], dtype="float32")
        node = ScheduleNode(fwd_func=lambda inp, lbl, **kw: inp + lbl)
        node.labels = labels
        out = node.forward(x)
        np.testing.assert_array_equal(
            out.numpy(), np.asarray(x.numpy()) + np.asarray(labels.numpy())
        )


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestScheduleNodeBackward(unittest.TestCase):
    """``ScheduleNode.backward`` runs real autograd and returns input grads."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_backward_with_output_grad_scales_by_local_jacobian(self):
        x = paddle.to_tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32"
        )
        x.stop_gradient = False
        node = ScheduleNode(fwd_func=lambda inputs, **kw: inputs * 3)
        node.forward(x)
        upstream = paddle.ones([2, 3], dtype="float32")
        grads = node.backward(output_grad=upstream)
        self.assertIsInstance(grads, tuple)
        self.assertEqual(len(grads), 1)
        # out = in * 3 -> d(in) = upstream * 3 = 3 everywhere.
        np.testing.assert_array_equal(
            grads[0].numpy(), np.full((2, 3), 3.0, dtype=np.float32)
        )
        # State is reset after backward.
        self.assertIsNone(node.inputs)
        self.assertIsNone(node.outputs)

    def test_backward_with_nonuniform_upstream(self):
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
        x.stop_gradient = False
        node = ScheduleNode(fwd_func=lambda inputs, **kw: inputs * 3)
        node.forward(x)
        upstream = paddle.to_tensor([[1.0, 2.0], [0.5, 4.0]], dtype="float32")
        grads = node.backward(output_grad=(upstream,))
        # d(in) = upstream * 3 element-wise.
        np.testing.assert_allclose(
            grads[0].numpy(),
            np.asarray(upstream.numpy()) * 3.0,
            rtol=1e-6,
            atol=0,
        )

    def test_backward_scalar_loss_without_output_grad(self):
        x = paddle.to_tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32"
        )
        x.stop_gradient = False
        node = ScheduleNode(fwd_func=lambda inputs, **kw: inputs.sum())
        node.forward(x)
        grads = node.backward()
        self.assertIsInstance(grads, tuple)
        self.assertEqual(len(grads), 1)
        # d(sum(x))/dx = ones, with the implicit scalar seed of 1.0.
        np.testing.assert_array_equal(
            grads[0].numpy(), np.ones((2, 3), dtype=np.float32)
        )


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestScheduleChunk(unittest.TestCase):
    """``ScheduleChunk`` validation and forward/backward composition."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_rejects_non_node_members(self):
        with self.assertRaises(AssertionError):
            ScheduleChunk([object()])

    def test_accepts_nested_chunk(self):
        inner = ScheduleChunk(
            [ScheduleNode(fwd_func=lambda inputs, **kw: inputs)]
        )
        outer = ScheduleChunk([inner])
        self.assertEqual(outer.nodes, [inner])

    def test_forward_composes_nodes_in_order(self):
        x = paddle.to_tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32"
        )
        x.stop_gradient = False
        node1 = ScheduleNode(fwd_func=lambda inputs, **kw: inputs * 2)
        node2 = ScheduleNode(fwd_func=lambda inputs, **kw: inputs + 1)
        chunk = ScheduleChunk([node1, node2])
        out = chunk.forward(x)
        # Independent reference: (x * 2) + 1.
        np.testing.assert_array_equal(
            out.numpy(), np.asarray(x.numpy()) * 2.0 + 1.0
        )

    def test_backward_composes_in_reverse_end_to_end(self):
        x = paddle.to_tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32"
        )
        x.stop_gradient = False
        node_a = ScheduleNode(fwd_func=lambda inputs, **kw: inputs * 2)
        node_b = ScheduleNode(fwd_func=lambda inputs, **kw: inputs.sum())
        chunk = ScheduleChunk([node_a, node_b])
        chunk.forward(x)
        grads = chunk.backward(output_grad=None)
        self.assertIsInstance(grads, tuple)
        self.assertEqual(len(grads), 1)
        # loss = sum(x * 2) -> d(x) = 2 everywhere.
        np.testing.assert_array_equal(
            grads[0].numpy(), np.full((2, 3), 2.0, dtype=np.float32)
        )


if __name__ == "__main__":
    unittest.main()
