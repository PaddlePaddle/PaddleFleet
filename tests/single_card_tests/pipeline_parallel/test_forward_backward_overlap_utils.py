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

Only the local, device-independent bookkeeping is exercised here:

* ``ScheduleChunk._check_nodes_valid`` - node-type gating rule.
* ``ScheduleChunk.forward`` / ``ScheduleChunk.backward`` - the *ordering*
  contract: forward runs its nodes front-to-back and threads each node's
  output into the next; backward runs them strictly back-to-front. A real
  ``ScheduleNode`` subclass whose per-node compute is replaced by an
  order-recording spy is used so the chunk chaining logic (the unit under
  test) runs unmodified while the heavy per-node autograd path is isolated.
* ``ScheduleNode.__init__`` / ``ScheduleNode._reset_states`` - default state
  and the exact set of fields cleared on reset.
* ``ScheduleNode.forward`` (no-recompute path) - the returned value is the real
  ``fwd_func`` applied to the *detached* inputs, inputs are stored detached
  with stop_gradient preserved, and the retained outputs keep the shape.
* ``FakeClone`` - forward preserves shape/dtype into a distinct tensor;
  backward passes the upstream gradient through unchanged (identity).
* ``detach_and_requires_grad`` - structure, per-tensor stop_gradient, value
  preservation, and None / non-tensor pass-through.
* ``clone_and_clear_dataptr`` - the None / non-tensor filtering and container
  shape (clone *content* is intentionally uninitialised ``empty_like`` memory
  and is therefore never compared).
* ``dict_to_tuple_helper`` - dict flattening order and per-tensor ``key``
  labelling, plus non-dict pass-through.

The recompute / AMP / RNG-preservation path in ``ScheduleNode.first_forward``
and the actual distributed pipeline schedule that consumes these chunks are
NOT covered here: they require a real device, real forward subgraphs and a
real pipeline process group. Faking those would only prove local orchestration,
not the scheduled cross-rank execution.

Expected values (gradient pass-through, ``2*x+1`` forward, node visit order,
``key`` labels) are hand-derived from the contract, not produced by calling the
code under test, so the test and the production code do not share a source of
truth.
"""

import unittest

try:
    import paddle

    from paddlefleet.pipeline_parallel.pp_utils.forward_backward_overlap_utils import (
        FakeClone,
        ScheduleChunk,
        ScheduleNode,
        clone_and_clear_dataptr,
        detach_and_requires_grad,
    )
    from paddlefleet.pipeline_parallel.pp_utils.utils import (
        dict_to_tuple_helper,
    )

    _IMPORT_ERROR = None

    class _RecordingNode(ScheduleNode):
        """A genuine ``ScheduleNode`` whose per-node forward/backward are
        replaced by lightweight order-recording spies.

        It stays a real ``ScheduleNode`` so it passes
        ``ScheduleChunk._check_nodes_valid``; only the heavy per-node compute
        is stubbed, leaving the ``ScheduleChunk`` chaining/ordering logic (the
        unit under test) to run unmodified.
        """

        def __init__(self, tag, log):
            super().__init__(lambda x: x, name=str(tag))
            self._tag = tag
            self._log = log
            self.received = None

        def forward(self, inputs=(), *, is_first_fwd=False, **kwargs):
            self.received = inputs
            self._log.append(("fwd", self._tag))
            return [*list(inputs), self._tag]

        def backward(self, output_grad=None, scaler=None):
            self.received = output_grad
            self._log.append(("bwd", self._tag))
            return [*list(output_grad), self._tag]

except (ImportError, ModuleNotFoundError) as exc:  # honest dependency probe
    paddle = None
    FakeClone = None
    ScheduleChunk = None
    ScheduleNode = None
    clone_and_clear_dataptr = None
    detach_and_requires_grad = None
    dict_to_tuple_helper = None
    _RecordingNode = None
    _IMPORT_ERROR = exc

_HAS_DEPS = _IMPORT_ERROR is None
_SKIP_REASON = (
    f"paddle / paddlefleet not importable on this CPU host: {_IMPORT_ERROR!r}"
)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestScheduleChunkNodeValidation(unittest.TestCase):
    """`ScheduleChunk._check_nodes_valid` accepts only ScheduleNode /
    ScheduleChunk and stores the node list verbatim."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_empty_chunk_has_no_nodes(self):
        chunk = ScheduleChunk([])
        self.assertEqual(chunk.nodes, [])

    def test_schedule_node_is_accepted_and_stored_by_identity(self):
        node = ScheduleNode(lambda x: x, name="n")
        chunk = ScheduleChunk([node])
        self.assertEqual(len(chunk.nodes), 1)
        self.assertIs(chunk.nodes[0], node)

    def test_nested_chunk_is_accepted(self):
        inner = ScheduleChunk([])
        outer = ScheduleChunk([inner])
        self.assertEqual(len(outer.nodes), 1)
        self.assertIs(outer.nodes[0], inner)

    def test_str_node_rejected(self):
        with self.assertRaises(AssertionError):
            ScheduleChunk(["not_a_node"])

    def test_int_node_rejected(self):
        with self.assertRaises(AssertionError):
            ScheduleChunk([42])

    def test_one_bad_node_among_valid_rejected(self):
        # A valid node must not mask an invalid sibling.
        with self.assertRaises(AssertionError):
            ScheduleChunk([ScheduleNode(lambda x: x), object()])


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestScheduleChunkOrdering(unittest.TestCase):
    """`ScheduleChunk.forward` chains nodes front-to-back and threads each
    node's output into the next; `ScheduleChunk.backward` runs them strictly
    back-to-front. Recording spies expose both the visit order and the value
    actually handed to each node."""

    def setUp(self):
        paddle.set_device("cpu")

    def _chunk(self):
        log = []
        nodes = [_RecordingNode(t, log) for t in ("a", "b", "c")]
        return ScheduleChunk(nodes), nodes, log

    def test_forward_runs_in_order_and_threads_outputs(self):
        chunk, nodes, log = self._chunk()
        result = chunk.forward([])
        # Each node appends its tag to the accumulator it received.
        self.assertEqual(result, ["a", "b", "c"])
        self.assertEqual(log, [("fwd", "a"), ("fwd", "b"), ("fwd", "c")])
        # Threading direction: node i sees the output produced by node i-1.
        self.assertEqual(nodes[0].received, [])
        self.assertEqual(nodes[1].received, ["a"])
        self.assertEqual(nodes[2].received, ["a", "b"])

    def test_backward_runs_reversed_and_threads_grads(self):
        chunk, nodes, log = self._chunk()
        result = chunk.backward([])
        # Reversed traversal: c, then b, then a.
        self.assertEqual(result, ["c", "b", "a"])
        self.assertEqual(log, [("bwd", "c"), ("bwd", "b"), ("bwd", "a")])
        self.assertEqual(nodes[2].received, [])
        self.assertEqual(nodes[1].received, ["c"])
        self.assertEqual(nodes[0].received, ["c", "b"])

    def test_empty_forward_returns_input_unchanged(self):
        chunk = ScheduleChunk([])
        obj = ("x", 1)
        self.assertIs(chunk.forward(obj), obj)

    def test_empty_backward_returns_grad_unchanged(self):
        chunk = ScheduleChunk([])
        grad = ("g", 2)
        self.assertIs(chunk.backward(grad), grad)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestScheduleNodeState(unittest.TestCase):
    """`ScheduleNode` default state and the exact fields `_reset_states`
    clears."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_defaults(self):
        node = ScheduleNode(lambda x: x, name="node")
        self.assertEqual(node.name, "node")
        self.assertIsNone(node.inputs)
        self.assertIsNone(node.outputs)
        self.assertIsNone(node.labels)
        self.assertIsNone(node.scale_loss_factor)
        self.assertFalse(node.use_recompute)

    def test_default_name_is_empty_string(self):
        node = ScheduleNode(lambda x: x)
        self.assertEqual(node.name, "")

    def test_reset_clears_io_and_loss_fields(self):
        node = ScheduleNode(lambda x: x)
        node.inputs = "some_input"
        node.outputs = "some_output"
        node.labels = "some_labels"
        node.scale_loss_factor = 2.0
        node.use_recompute = True

        node._reset_states()

        self.assertIsNone(node.inputs)
        self.assertIsNone(node.outputs)
        self.assertIsNone(node.labels)
        self.assertIsNone(node.scale_loss_factor)
        # _reset_states deliberately does NOT touch use_recompute: a node keeps
        # its recompute configuration across forward/backward cycles.
        self.assertTrue(node.use_recompute)

    def test_reset_is_idempotent(self):
        node = ScheduleNode(lambda x: x)
        node.inputs = "x"
        node._reset_states()
        node._reset_states()
        self.assertIsNone(node.inputs)
        self.assertIsNone(node.outputs)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestScheduleNodeForwardCPU(unittest.TestCase):
    """`ScheduleNode.forward` (no-recompute default path): the returned value
    is the real fwd_func applied to the *detached* inputs, and the node retains
    the detached inputs and a shape-preserving clone of the outputs."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_returns_fwd_func_of_detached_inputs(self):
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        x.stop_gradient = False
        node = ScheduleNode(lambda t: t * 2.0 + 1.0)

        out = node.forward(x)

        # Hand-derived: 2*[1,2,3]+1 == [3,5,7]. The RETURNED tensor is the real
        # forward result (cloning happens only on the retained self.outputs).
        self.assertEqual(out.numpy().tolist(), [3.0, 5.0, 7.0])

    def test_retains_detached_inputs_preserving_values_and_flag(self):
        x = paddle.to_tensor([4.0, 5.0])
        x.stop_gradient = False
        node = ScheduleNode(lambda t: t + 1.0)

        node.forward(x)

        # Inputs are stored detached (a distinct tensor) with values and the
        # stop_gradient flag preserved.
        self.assertIsNot(node.inputs, x)
        self.assertFalse(node.inputs.stop_gradient)
        self.assertEqual(node.inputs.numpy().tolist(), [4.0, 5.0])

    def test_retained_outputs_preserve_shape(self):
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        x.stop_gradient = False
        labels = paddle.zeros([2, 2], dtype="float32")
        # With labels set the output is a retained loss tensor, so production
        # takes the clear_dataptr=False branch (``clear_dataptr = labels is
        # None``). self.outputs is then a shape-preserving FakeClone
        # (empty_like) whose content is uninitialised and is not compared. On
        # the default (no-labels) path the output dataptr is released and the
        # shape would instead collapse to [].
        node = ScheduleNode(lambda t, lbl: t * 3.0)
        node.labels = labels

        node.forward(x)

        self.assertIsInstance(node.outputs, paddle.Tensor)
        self.assertEqual(list(node.outputs.shape), [2, 2])


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestFakeClone(unittest.TestCase):
    """`FakeClone` forward preserves shape/dtype into a distinct tensor;
    backward is the identity on the upstream gradient."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_forward_preserves_shape_and_dtype_distinct_tensor(self):
        x = paddle.ones([5, 10, 20], dtype="float32")
        out = FakeClone.apply(x)
        self.assertEqual(list(out.shape), [5, 10, 20])
        self.assertEqual(out.dtype, x.dtype)
        # empty_like returns a new tensor, not the same storage.
        self.assertIsNot(out, x)

    def test_backward_passes_upstream_gradient_through_unchanged(self):
        x = paddle.ones([2, 3], dtype="float32")
        x.stop_gradient = False
        out = FakeClone.apply(x)
        upstream = paddle.to_tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32"
        )

        paddle.autograd.backward([out], [upstream])

        # FakeClone.backward returns grad_output verbatim, so the input grad is
        # exactly the (non-uniform) upstream gradient - not a doubled or zeroed
        # variant.
        self.assertIsNotNone(x.grad)
        self.assertEqual(
            x.grad.numpy().tolist(), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        )


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestDetachAndRequiresGrad(unittest.TestCase):
    """`detach_and_requires_grad` preserves container structure, per-tensor
    stop_gradient and tensor values, and passes None / non-tensors through."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_single_tensor_detaches_and_preserves_values(self):
        t = paddle.to_tensor([1.0, 2.0, 3.0])
        t.stop_gradient = False
        out = detach_and_requires_grad(t)
        self.assertIsInstance(out, paddle.Tensor)
        self.assertIsNot(out, t)
        self.assertFalse(out.stop_gradient)
        self.assertEqual(out.numpy().tolist(), [1.0, 2.0, 3.0])

    def test_single_tensor_stop_gradient_true_is_preserved(self):
        t = paddle.to_tensor([7.0, 8.0])
        t.stop_gradient = True
        out = detach_and_requires_grad(t)
        self.assertTrue(out.stop_gradient)
        self.assertEqual(out.numpy().tolist(), [7.0, 8.0])

    def test_tuple_preserves_type_flags_values_and_passthrough(self):
        a = paddle.to_tensor([1.0])
        a.stop_gradient = False
        b = paddle.to_tensor([2.0])
        b.stop_gradient = True
        out = detach_and_requires_grad((a, b, None, 42))
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 4)
        self.assertFalse(out[0].stop_gradient)
        self.assertTrue(out[1].stop_gradient)
        self.assertEqual(out[0].numpy().tolist(), [1.0])
        self.assertEqual(out[1].numpy().tolist(), [2.0])
        self.assertIsNone(out[2])
        self.assertEqual(out[3], 42)

    def test_list_input_returns_list(self):
        a = paddle.to_tensor([1.0])
        out = detach_and_requires_grad([a])
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 1)

    def test_nested_tuple_structure_is_preserved(self):
        inner = paddle.to_tensor([9.0])
        outer = paddle.to_tensor([5.0])
        out = detach_and_requires_grad(((inner,), outer))
        self.assertIsInstance(out, tuple)
        self.assertIsInstance(out[0], tuple)
        self.assertEqual(out[0][0].numpy().tolist(), [9.0])
        self.assertEqual(out[1].numpy().tolist(), [5.0])

    def test_dict_preserves_keys_values_and_none(self):
        data = {"a": paddle.to_tensor([3.0]), "b": None}
        out = detach_and_requires_grad(data)
        self.assertIsInstance(out, dict)
        self.assertEqual(set(out), {"a", "b"})
        self.assertEqual(out["a"].numpy().tolist(), [3.0])
        self.assertIsNone(out["b"])


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestCloneAndClearDataptr(unittest.TestCase):
    """`clone_and_clear_dataptr` drops None / non-tensor entries and keeps the
    container type. The clone content is uninitialised ``empty_like`` memory,
    so only the surviving entries and their shapes are asserted."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_tuple_of_tensors_kept_with_shapes(self):
        t1 = paddle.ones([2, 3])
        t2 = paddle.ones([4])
        out = clone_and_clear_dataptr((t1, t2))
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        self.assertEqual(list(out[0].shape), [2, 3])
        self.assertEqual(list(out[1].shape), [4])

    def test_none_entry_is_dropped(self):
        t1 = paddle.ones([2, 3])
        out = clone_and_clear_dataptr((t1, None))
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 1)
        self.assertEqual(list(out[0].shape), [2, 3])

    def test_non_tensor_and_none_both_dropped(self):
        t1 = paddle.ones([5])
        out = clone_and_clear_dataptr((t1, 42, None))
        self.assertEqual(len(out), 1)
        self.assertEqual(list(out[0].shape), [5])

    def test_list_input_returns_list(self):
        out = clone_and_clear_dataptr([paddle.ones([2, 2])])
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 1)

    def test_dict_keeps_only_tensor_valued_keys(self):
        data = {"a": paddle.ones([2, 3]), "b": None}
        out = clone_and_clear_dataptr(data)
        self.assertIsInstance(out, dict)
        self.assertEqual(set(out), {"a"})
        self.assertEqual(list(out["a"].shape), [2, 3])


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestDictToTupleHelper(unittest.TestCase):
    """`dict_to_tuple_helper` flattens a dict into a tuple in insertion order,
    labelling each tensor with its ``key`` (list values get ``"<key> <idx>"``),
    and passes non-dict inputs through unchanged."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_dict_flattens_in_order_with_key_labels(self):
        t1 = paddle.ones([1])
        t2 = paddle.ones([1])
        t3 = paddle.ones([1])
        out = dict_to_tuple_helper({"a": t1, "b": [t2, t3]})
        self.assertIsInstance(out, tuple)
        # Order follows dict insertion: single "a", then the two "b" elements.
        self.assertEqual(len(out), 3)
        self.assertIs(out[0], t1)
        self.assertIs(out[1], t2)
        self.assertIs(out[2], t3)
        self.assertEqual(out[0].key, "a")
        self.assertEqual(out[1].key, "b 0")
        self.assertEqual(out[2].key, "b 1")

    def test_tuple_input_passthrough_identity(self):
        payload = (paddle.ones([1]), paddle.ones([1]))
        out = dict_to_tuple_helper(payload)
        self.assertIs(out, payload)

    def test_single_tensor_passthrough_identity(self):
        t = paddle.ones([2])
        out = dict_to_tuple_helper(t)
        self.assertIs(out, t)


if __name__ == "__main__":
    unittest.main()
