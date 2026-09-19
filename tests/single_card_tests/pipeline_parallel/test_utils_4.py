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

"""CPU-only behavior tests for device-independent pure logic in
``paddlefleet.pipeline_parallel.utils``.

This file targets a slice that is DISTINCT from the sibling ``test_utils.py`` /
``test_utils_2.py`` / ``test_utils_3.py`` tests (which already cover the
``is_pp_*`` / ``is_vp_*`` predicates, ``get_pp_*_rank`` neighbour lookups,
``NoopScheduleNode``, ``ScheduleNode.__init__`` / ``_reset_states`` /
``default_backward_func``, ``AbstractSchedulePlan``, ``stream_acquire_context``
and the ``set_streams`` registry). Here we exercise only:

* ``make_viewless`` - the thin wrapper that forwards ``inp=e``,
  ``requires_grad=e.requires_grad`` and ``keep_graph=True`` to the genuine
  (NOT-under-test) collaborator ``make_viewless_tensor`` and returns its result.
  One test runs the real collaborator end-to-end on a non-view tensor (whose
  documented contract is identity pass-through); another spies on the
  collaborator to assert the exact forwarded arguments AND that the wrapper
  returns the collaborator's result verbatim (so a swapped/dropped kwarg or a
  discarded return value is caught).
* ``ScheduleNode.get_grad`` - collects ``.grad`` off ``self.inputs`` after a
  real CPU autograd backward, mapping a ``None`` input slot to a ``None`` grad
  and unwrapping the single-input case to a bare tensor. Gradients are derived
  by hand from the elementwise forward math, never from the production output.
* ``ScheduleNode.get_output`` - returns the stored ``self.output`` object by
  identity.

The stream-wrapped ``ScheduleNode._forward`` / ``_backward`` need real CUDA
streams / nvtx and are out of scope on a CPU host.
"""

import unittest
from unittest import mock

import numpy as np

try:
    import paddle

    import paddlefleet.pipeline_parallel.utils as pp_utils

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # honest dependency probe
    paddle = None
    pp_utils = None
    _IMPORT_ERROR = exc

_HAS_DEPS = _IMPORT_ERROR is None
_SKIP_REASON = (
    f"paddle / paddlefleet not importable on this CPU host: {_IMPORT_ERROR!r}"
)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestMakeViewless(unittest.TestCase):
    """``make_viewless`` forwards to ``make_viewless_tensor`` and returns it."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_non_view_tensor_returned_as_is(self):
        # A freshly built tensor is not a view, so the real (unmocked)
        # ``make_viewless_tensor`` contract is to return the input object
        # unchanged. Hand-reasoned expectation: identity + values preserved.
        t = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
        t.stop_gradient = False
        self.assertFalse(t._is_view())  # precondition for the identity path

        result = pp_utils.make_viewless(t)

        self.assertIs(result, t)
        np.testing.assert_array_equal(
            result.numpy(), np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        )

    def test_forwards_requires_grad_and_keep_graph_and_returns_result(self):
        # ``make_viewless_tensor`` is a genuine NOT-under-test collaborator
        # (defined in paddlefleet.utils). Spy on it to prove the wrapper passes
        # the tensor, the tensor's *actual* requires_grad value, and
        # keep_graph=True, and hands back exactly what the collaborator returns.
        for stop_gradient, expect_requires_grad in (
            (False, True),
            (True, False),
        ):
            with self.subTest(stop_gradient=stop_gradient):
                e = paddle.to_tensor([5.0, 6.0], dtype="float32")
                e.stop_gradient = stop_gradient
                # A distinguishable marker unrelated to the input, so a wrapper
                # that dropped the return value could not accidentally pass.
                marker = paddle.to_tensor([-99.0], dtype="float32")
                captured = {}

                def fake_make_viewless_tensor(inp, requires_grad, keep_graph):
                    captured["inp"] = inp
                    captured["requires_grad"] = requires_grad
                    captured["keep_graph"] = keep_graph
                    return marker

                with mock.patch.object(
                    pp_utils,
                    "make_viewless_tensor",
                    side_effect=fake_make_viewless_tensor,
                ) as spy:
                    out = pp_utils.make_viewless(e)

                spy.assert_called_once()
                self.assertIs(captured["inp"], e)
                self.assertEqual(
                    captured["requires_grad"], expect_requires_grad
                )
                self.assertIs(captured["keep_graph"], True)
                self.assertIs(out, marker)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestScheduleNodeGetGrad(unittest.TestCase):
    """``ScheduleNode.get_grad`` collects ``.grad`` off ``self.inputs``.

    Gradients are populated by a real CPU autograd backward and compared to
    values derived by hand from the elementwise forward math.
    """

    def setUp(self):
        paddle.set_device("cpu")

    @staticmethod
    def _make_node():
        # stream/event/forward_func are stored but untouched by get_grad;
        # free_input must stay False (asserted in __init__).
        return pp_utils.ScheduleNode(
            forward_func=lambda *a: a, stream=None, event=None
        )

    def test_multiple_inputs_collect_hand_derived_grads(self):
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        y = paddle.to_tensor([4.0, 5.0, 6.0])
        x.stop_gradient = False
        y.stop_gradient = False
        # f = sum(2*x + 3*y) => df/dx = 2, df/dy = 3 elementwise.
        loss = (x * 2.0 + y * 3.0).sum()
        loss.backward()

        node = self._make_node()
        node.inputs = [x, y]
        grads = node.get_grad()

        self.assertIsInstance(grads, tuple)
        self.assertEqual(len(grads), 2)
        np.testing.assert_allclose(
            grads[0].numpy(), [2.0, 2.0, 2.0], rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            grads[1].numpy(), [3.0, 3.0, 3.0], rtol=1e-6, atol=1e-6
        )

    def test_none_input_slot_maps_to_none_grad(self):
        x = paddle.to_tensor([1.0, 2.0])
        x.stop_gradient = False
        loss = (x * 5.0).sum()  # df/dx = 5 elementwise
        loss.backward()

        node = self._make_node()
        node.inputs = [x, None]  # second collaborator absent
        grads = node.get_grad()

        self.assertIsInstance(grads, tuple)
        self.assertEqual(len(grads), 2)
        np.testing.assert_allclose(
            grads[0].numpy(), [5.0, 5.0], rtol=1e-6, atol=1e-6
        )
        self.assertIsNone(grads[1])

    def test_single_input_returns_bare_tensor(self):
        x = paddle.to_tensor([2.0, 3.0, 4.0])
        x.stop_gradient = False
        loss = (x * x).sum()  # df/dx = 2*x
        loss.backward()

        node = self._make_node()
        node.inputs = [x]
        grad = node.get_grad()

        # Single input => unwrapped to a bare tensor, not a 1-tuple.
        self.assertIsInstance(grad, paddle.Tensor)
        np.testing.assert_allclose(
            grad.numpy(), [4.0, 6.0, 8.0], rtol=1e-6, atol=1e-6
        )


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestScheduleNodeGetOutput(unittest.TestCase):
    """``ScheduleNode.get_output`` returns the stored ``self.output`` object."""

    def setUp(self):
        paddle.set_device("cpu")

    @staticmethod
    def _make_node():
        return pp_utils.ScheduleNode(
            forward_func=lambda *a: a, stream=None, event=None
        )

    def test_returns_stored_output_by_identity(self):
        node = self._make_node()
        sentinel = paddle.to_tensor([7.0, 8.0])
        node.output = sentinel
        self.assertIs(node.get_output(), sentinel)

    def test_returns_stored_tuple_output_by_identity(self):
        node = self._make_node()
        payload = (paddle.to_tensor([1.0]), "meta", 42)
        node.output = payload
        returned = node.get_output()
        self.assertIs(returned, payload)


if __name__ == "__main__":
    unittest.main()
