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
``paddlefleet.pipeline_parallel.utils``.

Only the pieces whose correctness is local (pass-through contracts, attribute
wiring, the abstract-plan protocol, and the autograd forward/backward of
``ScheduleNode.default_backward_func``) are exercised here:

* ``NoopScheduleNode.forward`` / ``.backward`` - both return the *same* object
  they were handed, unchanged, for tensors, tuples and lists.
* ``ScheduleNode.__init__`` - stores ``forward_func`` / ``stream`` / ``event`` /
  ``name`` verbatim, wires ``backward_func`` to the bound
  ``default_backward_func`` when none is given (and to the supplied callable
  otherwise), starts with ``inputs``/``outputs`` cleared, and asserts that
  ``free_input=True`` is rejected.
* ``ScheduleNode.default_backward_func`` - the real autograd path. Gradients
  are derived BY HAND from the local forward function: for ``f(x) = 2*x`` with
  upstream ``g`` the input grad is ``2*g``; for a scalar ``sum`` output with no
  upstream it is all-ones; multi-input collection and the ``None`` input slot
  are checked; the ``len(outputs) == len(output_grad)`` guard is asserted.
* ``ScheduleNode._reset_states`` - the exact set of fields cleared.
* ``AbstractSchedulePlan`` - the ABC cannot be instantiated, a subclass missing
  ``run`` cannot either, and a concrete subclass's ``run`` override is callable
  with the documented signature.

Expected numbers are hand-derived from the contract, never produced by calling
the code under test. The heavy paths that need a real device are deliberately
NOT covered: ``ScheduleNode.forward`` / ``.backward`` / ``_forward`` /
``_backward`` all run inside ``paddle.cuda`` stream / nvtx contexts, and
``set_streams`` allocates a ``paddle.cuda.Stream``; those require a GPU and a
real pipeline schedule and cannot be honestly validated on a CPU host.
"""

import unittest

import numpy as np

try:
    import paddle

    from paddlefleet.pipeline_parallel.utils import (
        AbstractSchedulePlan,
        NoopScheduleNode,
        ScheduleNode,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # honest dependency probe
    paddle = None
    AbstractSchedulePlan = None
    NoopScheduleNode = None
    ScheduleNode = None
    _IMPORT_ERROR = exc

_HAS_DEPS = _IMPORT_ERROR is None
_SKIP_REASON = (
    f"paddle / paddlefleet not importable on this CPU host: {_IMPORT_ERROR!r}"
)


def _stream_event_sentinels():
    """Two distinct, non-CUDA placeholder objects.

    ``ScheduleNode.__init__`` only *stores* ``stream`` / ``event`` (it never
    touches them), so plain sentinels are enough to exercise the constructor
    without a GPU. They are intentionally distinct so a swap would be visible.
    """
    return object(), object()


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestNoopScheduleNode(unittest.TestCase):
    """`NoopScheduleNode` forwards and backwards its argument unchanged."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_forward_returns_same_tensor_object(self):
        node = NoopScheduleNode()
        t = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
        out = node.forward(t)
        self.assertIs(out, t)  # no copy, no wrap: identical object

    def test_forward_returns_same_container_object(self):
        node = NoopScheduleNode()
        data = ([1, 2], "x", 3.5)
        out = node.forward(data)
        self.assertIs(out, data)
        self.assertEqual(out, ([1, 2], "x", 3.5))

    def test_backward_returns_same_tensor_object(self):
        node = NoopScheduleNode()
        g = paddle.to_tensor([[5.0, 6.0], [7.0, 8.0]], dtype="float32")
        out = node.backward(g)
        self.assertIs(out, g)

    def test_backward_returns_same_list_object(self):
        node = NoopScheduleNode()
        grads = [1.0, 2.0, 3.0]
        out = node.backward(grads)
        self.assertIs(out, grads)
        self.assertEqual(out, [1.0, 2.0, 3.0])


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestScheduleNodeInit(unittest.TestCase):
    """`ScheduleNode.__init__` attribute wiring and the free_input guard."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_stores_fields_verbatim(self):
        stream, event = _stream_event_sentinels()

        def fwd(x):
            return x

        node = ScheduleNode(
            forward_func=fwd, stream=stream, event=event, name="my_node"
        )
        self.assertIs(node.forward_func, fwd)
        self.assertIs(node.stream, stream)
        self.assertIs(node.event, event)
        self.assertEqual(node.name, "my_node")
        self.assertFalse(node.free_input)
        self.assertIsNone(node.inputs)
        self.assertIsNone(node.outputs)

    def test_default_name(self):
        stream, event = _stream_event_sentinels()
        node = ScheduleNode(
            forward_func=lambda x: x, stream=stream, event=event
        )
        self.assertEqual(node.name, "schedule_node")

    def test_default_backward_func_is_bound_default(self):
        stream, event = _stream_event_sentinels()
        node = ScheduleNode(
            forward_func=lambda x: x, stream=stream, event=event
        )
        # When no backward_func is given, the bound default must be wired in.
        self.assertIs(
            node.backward_func.__func__, ScheduleNode.default_backward_func
        )
        self.assertIs(node.backward_func.__self__, node)

    def test_custom_backward_func_overrides_default(self):
        stream, event = _stream_event_sentinels()

        def custom_backward(outputs, output_grad):
            return output_grad

        node = ScheduleNode(
            forward_func=lambda x: x,
            stream=stream,
            event=event,
            backward_func=custom_backward,
        )
        self.assertIs(node.backward_func, custom_backward)
        self.assertIsNot(
            getattr(node.backward_func, "__func__", None),
            ScheduleNode.default_backward_func,
        )

    def test_free_input_true_is_rejected(self):
        stream, event = _stream_event_sentinels()
        with self.assertRaises(AssertionError):
            ScheduleNode(
                forward_func=lambda x: x,
                stream=stream,
                event=event,
                free_input=True,
            )


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestScheduleNodeResetStates(unittest.TestCase):
    """`ScheduleNode._reset_states` clears exactly ``inputs`` and ``outputs``."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_reset_clears_inputs_and_outputs(self):
        stream, event = _stream_event_sentinels()
        node = ScheduleNode(
            forward_func=lambda x: x, stream=stream, event=event
        )
        node.inputs = [object()]
        node.outputs = object()
        node._reset_states()
        self.assertIsNone(node.inputs)
        self.assertIsNone(node.outputs)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestScheduleNodeDefaultBackward(unittest.TestCase):
    """`ScheduleNode.default_backward_func` runs real autograd and returns the
    input gradients. Expected gradients are hand-derived from the local
    forward function, not read back from the production code.

    ``default_backward_func`` uses ``self.inputs`` to collect ``.grad`` and then
    calls ``_reset_states``; it is called here directly (it is itself the unit
    under test) rather than through ``ScheduleNode.backward``, whose surrounding
    ``paddle.cuda`` stream/nvtx context requires a GPU.
    """

    def setUp(self):
        paddle.set_device("cpu")

    def _make_node(self):
        stream, event = _stream_event_sentinels()
        return ScheduleNode(
            forward_func=lambda x: x, stream=stream, event=event
        )

    def test_single_input_with_upstream_grad(self):
        # f(x) = 2*x  =>  dx = 2 * upstream (non-uniform upstream on purpose).
        x = paddle.to_tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32"
        )
        x.stop_gradient = False
        output = x * 2.0
        upstream = paddle.to_tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32"
        )
        expected = 2.0 * np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32
        )

        node = self._make_node()
        node.inputs = [x]
        grad = node.default_backward_func(output, upstream)

        np.testing.assert_allclose(grad.numpy(), expected, rtol=1e-6, atol=0.0)
        np.testing.assert_allclose(
            x.grad.numpy(), expected, rtol=1e-6, atol=0.0
        )
        # default_backward_func resets state after collecting the gradient.
        self.assertIsNone(node.inputs)
        self.assertIsNone(node.outputs)

    def test_scalar_output_without_upstream_grad(self):
        # output = sum(3*x) is scalar; with no upstream, autograd seeds 1.0,
        # so d/dx sum(3*x) = 3 everywhere.
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
        x.stop_gradient = False
        output = (x * 3.0).sum()
        expected = np.full((2, 2), 3.0, dtype=np.float32)

        node = self._make_node()
        node.inputs = [x]
        grad = node.default_backward_func(output, None)

        np.testing.assert_allclose(grad.numpy(), expected, rtol=1e-6, atol=0.0)

    def test_single_element_list_output_without_upstream_grad(self):
        # The None-grad branch unwraps a one-element list output before backward.
        x = paddle.to_tensor([[2.0, -1.0]], dtype="float32")
        x.stop_gradient = False
        output = (x * x).sum()  # d/dx sum(x^2) = 2*x
        expected = 2.0 * np.array([[2.0, -1.0]], dtype=np.float32)

        node = self._make_node()
        node.inputs = [x]
        grad = node.default_backward_func([output], None)

        np.testing.assert_allclose(grad.numpy(), expected, rtol=1e-6, atol=0.0)

    def test_multiple_inputs_with_none_slot(self):
        # inputs = [x, None]: grad tuple keeps the None slot; only x is in graph.
        x = paddle.to_tensor([[2.0, 3.0]], dtype="float32")
        x.stop_gradient = False
        output = x * 5.0  # dx = 5 * upstream
        upstream = paddle.to_tensor([[2.0, 3.0]], dtype="float32")
        expected = 5.0 * np.array([[2.0, 3.0]], dtype=np.float32)

        node = self._make_node()
        node.inputs = [x, None]
        grad = node.default_backward_func(output, upstream)

        self.assertIsInstance(grad, tuple)
        self.assertEqual(len(grad), 2)
        np.testing.assert_allclose(
            grad[0].numpy(), expected, rtol=1e-6, atol=0.0
        )
        self.assertIsNone(grad[1])

    def test_length_mismatch_between_outputs_and_grads_is_rejected(self):
        # The guard fires before any backward: 2 outputs vs 1 grad.
        a = paddle.to_tensor([1.0], dtype="float32")
        b = paddle.to_tensor([2.0], dtype="float32")
        g = paddle.to_tensor([1.0], dtype="float32")

        node = self._make_node()
        node.inputs = [a, b]
        with self.assertRaises(AssertionError):
            node.default_backward_func((a, b), (g,))


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestAbstractSchedulePlan(unittest.TestCase):
    """`AbstractSchedulePlan` is an ABC requiring a ``run`` implementation."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_cannot_instantiate_abstract_base(self):
        with self.assertRaises(TypeError):
            AbstractSchedulePlan()

    def test_subclass_without_run_cannot_instantiate(self):
        class MissingRun(AbstractSchedulePlan):
            pass

        with self.assertRaises(TypeError):
            MissingRun()

    def test_concrete_subclass_run_is_callable(self):
        class ConcretePlan(AbstractSchedulePlan):
            @staticmethod
            def run(
                f_schedule_plan,
                b_schedule_plan,
                grad=None,
                pre_forward=None,
                pre_backward=None,
                post_forward=None,
                post_backward=None,
            ):
                # Return the routed arguments so the override is observably real.
                return ("ran", f_schedule_plan, b_schedule_plan, grad)

        plan = ConcretePlan()
        self.assertEqual(
            plan.run("fwd_plan", "bwd_plan", grad=7),
            ("ran", "fwd_plan", "bwd_plan", 7),
        )


if __name__ == "__main__":
    unittest.main()
