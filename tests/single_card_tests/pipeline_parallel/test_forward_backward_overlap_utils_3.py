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
``paddlefleet.pipeline_parallel.pp_utils.forward_backward_overlap_utils``
and its ``utils`` collaborator.

Covered (real code, CPU, no mocks):
  * ``detach_and_requires_grad`` -- value preservation, container-type
    preservation, recursion and the (mis-named) *stop_gradient preservation*
    contract: it copies ``input.stop_gradient`` rather than forcing grad on.
  * ``FakeClone`` -- forward produces a same-shape/dtype tensor whose *values
    are intentionally uninitialised* (``paddle.empty_like``), while backward is
    an exact identity.  We therefore assert the gradient path, NOT the value.
  * ``clone_and_clear_dataptr`` -- container-type preservation and the
    None / non-Tensor filtering rule, plus the retained gradient path.
  * ``dict_to_tuple_helper`` / ``convert_tensor_dict_to_tuple`` -- dict->tuple
    flattening, per-tensor ``.key`` tagging, list-value index suffixing, and
    identity passthrough for non-dict inputs.
  * ``ScheduleChunk`` -- node validation and ordered forward composition.
  * ``ScheduleNode`` -- non-recompute forward returns the *real* fwd output
    (not the fake clone) and stores detached inputs, plus a real
    forward+backward gradient check against an independent derivative.

NOT covered (honest): ``ScheduleNode.first_forward`` preserves fleet RNG state
(``get_rng_state_tracker``), the recompute ``custom_state_manager`` state, and
the dygraph AMP-tracer level/dtype.  Exercising its AMP-level mapping requires a
real fleet recompute + AMP tracer environment; mocking those collaborators would
only test the mocks (anti-pattern 12/1), so it is left to an environment where
recompute actually runs rather than faked here.

Expected values are derived independently (by hand / numpy), never by calling
the code under test to produce its own reference.
"""

import unittest

import numpy as np

try:
    import paddle

    from paddlefleet.pipeline_parallel.pp_utils import (
        forward_backward_overlap_utils as fbo,
    )
    from paddlefleet.pipeline_parallel.pp_utils.utils import (
        dict_to_tuple_helper,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # honest dependency probe
    paddle = None
    fbo = None
    dict_to_tuple_helper = None
    _IMPORT_ERROR = exc

_HAS_DEPS = _IMPORT_ERROR is None
_SKIP_REASON = (
    f"paddle / paddlefleet not importable on this CPU host: {_IMPORT_ERROR!r}"
)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestDetachAndRequiresGrad(unittest.TestCase):
    """``detach_and_requires_grad`` preserves values, container type, nesting
    and each tensor's ``stop_gradient`` flag (it does NOT force grad on)."""

    def setUp(self):
        paddle.set_device("cpu")

    def _tensor(self, values, stop_gradient):
        t = paddle.to_tensor(values, dtype="float32")
        t.stop_gradient = stop_gradient
        return t

    def test_single_tensor_preserves_values_and_stop_gradient(self):
        for stop_gradient in (True, False):
            src = self._tensor(
                [[1.0, -2.0, 3.5], [4.0, 5.0, -6.0]], stop_gradient
            )
            out = fbo.detach_and_requires_grad(src)
            self.assertIsInstance(out, paddle.Tensor)
            self.assertIsNot(out, src)  # a detached copy, not the same handle
            np.testing.assert_array_equal(out.numpy(), src.numpy())
            # The contract is preservation of the flag, not requires_grad=True.
            self.assertEqual(out.stop_gradient, stop_gradient)

    def test_list_preserves_type_and_detaches_each(self):
        a = self._tensor([1.0, 2.0], stop_gradient=False)
        b = self._tensor([[3.0, 4.0]], stop_gradient=True)
        out = fbo.detach_and_requires_grad([a, b])
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 2)
        np.testing.assert_array_equal(out[0].numpy(), a.numpy())
        np.testing.assert_array_equal(out[1].numpy(), b.numpy())
        self.assertFalse(out[0].stop_gradient)
        self.assertTrue(out[1].stop_gradient)

    def test_tuple_preserves_type(self):
        a = self._tensor([7.0, 8.0, 9.0], stop_gradient=False)
        out = fbo.detach_and_requires_grad((a,))
        self.assertIsInstance(out, tuple)
        np.testing.assert_array_equal(out[0].numpy(), a.numpy())

    def test_nested_and_non_tensor_passthrough(self):
        a = self._tensor([1.0], stop_gradient=False)
        b = self._tensor([2.0, 3.0], stop_gradient=True)
        out = fbo.detach_and_requires_grad([a, [b], "raw", 42])
        self.assertIsInstance(out, list)
        self.assertIsInstance(out[1], list)  # nested container recursed
        np.testing.assert_array_equal(out[0].numpy(), a.numpy())
        np.testing.assert_array_equal(out[1][0].numpy(), b.numpy())
        self.assertEqual(out[2], "raw")  # non-tensor passed through unchanged
        self.assertEqual(out[3], 42)

    def test_dict_preserves_keys_values_and_none(self):
        a = self._tensor([[1.0, 2.0]], stop_gradient=False)
        b = self._tensor([3.0], stop_gradient=True)
        out = fbo.detach_and_requires_grad({"x": a, "y": b, "z": None})
        self.assertEqual(set(out.keys()), {"x", "y", "z"})
        np.testing.assert_array_equal(out["x"].numpy(), a.numpy())
        np.testing.assert_array_equal(out["y"].numpy(), b.numpy())
        self.assertFalse(out["x"].stop_gradient)
        self.assertTrue(out["y"].stop_gradient)
        self.assertIsNone(out["z"])


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestFakeClone(unittest.TestCase):
    """``FakeClone`` forward returns an uninitialised same-shape/dtype tensor
    (``empty_like``); backward is an exact identity.  Only shape/dtype and the
    gradient are contractual -- the forward *values* are deliberately garbage."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_forward_shape_and_dtype_only(self):
        x = paddle.to_tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32"
        )
        y = fbo.FakeClone.apply(x)
        self.assertIsInstance(y, paddle.Tensor)
        self.assertEqual(list(y.shape), list(x.shape))
        self.assertEqual(y.dtype, x.dtype)
        # Intentionally NOT asserting y == x: empty_like is uninitialised.

    def test_backward_is_identity(self):
        x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0], dtype="float32")
        x.stop_gradient = False
        upstream = paddle.to_tensor([10.0, -20.0, 30.0, -40.0], dtype="float32")
        y = fbo.FakeClone.apply(x)
        paddle.autograd.backward([y], [upstream])
        self.assertIsNotNone(x.grad)
        # d(out)/d(in) == identity, so the upstream gradient passes through 1:1.
        np.testing.assert_array_equal(x.grad.numpy(), upstream.numpy())


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestCloneAndClearDataptr(unittest.TestCase):
    """``clone_and_clear_dataptr`` preserves container type, drops ``None`` and
    non-Tensor entries (keeping order of the survivors), and retains the
    gradient path via ``FakeClone``."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_list_filters_none_and_nontensor_preserving_order(self):
        a = paddle.zeros([2, 3], dtype="float32")
        b = paddle.zeros([4, 5], dtype="float32")
        out = fbo.clone_and_clear_dataptr([a, None, b, 7])
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 2)  # None and the int are filtered out
        # Distinguishable shapes prove which survivors were kept, and in order.
        self.assertEqual(list(out[0].shape), [2, 3])
        self.assertEqual(list(out[1].shape), [4, 5])

    def test_tuple_type_preserved(self):
        a = paddle.zeros([2, 2], dtype="float32")
        b = paddle.zeros([3, 1], dtype="float32")
        out = fbo.clone_and_clear_dataptr((a, b))
        self.assertIsInstance(out, tuple)
        self.assertEqual([list(o.shape) for o in out], [[2, 2], [3, 1]])

    def test_dict_filters_and_keeps_keys(self):
        a = paddle.zeros([2, 3], dtype="float32")
        out = fbo.clone_and_clear_dataptr({"a": a, "b": None, "c": 5})
        self.assertEqual(set(out.keys()), {"a"})  # None and int dropped
        self.assertEqual(list(out["a"].shape), [2, 3])

    def test_single_tensor_retains_gradient_path(self):
        x = paddle.to_tensor([1.0, 2.0, 3.0], dtype="float32")
        x.stop_gradient = False
        upstream = paddle.to_tensor([5.0, 6.0, 7.0], dtype="float32")
        out = fbo.clone_and_clear_dataptr(x)
        self.assertIsInstance(out, paddle.Tensor)
        paddle.autograd.backward([out], [upstream])
        np.testing.assert_array_equal(x.grad.numpy(), upstream.numpy())


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestDictToTupleHelper(unittest.TestCase):
    """``dict_to_tuple_helper`` flattens a dict into an ordered tuple, tagging
    each tensor with a ``.key`` (list values get ``"<key> <idx>"``), and returns
    non-dict inputs unchanged (identity)."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_dict_becomes_tuple_with_keys(self):
        t1 = paddle.zeros([2], dtype="float32")
        t2 = paddle.zeros([3], dtype="float32")
        out = dict_to_tuple_helper({"a": t1, "b": t2})
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        self.assertIs(out[0], t1)  # same tensor objects, in insertion order
        self.assertIs(out[1], t2)
        self.assertEqual(t1.key, "a")
        self.assertEqual(t2.key, "b")

    def test_dict_list_values_get_indexed_keys(self):
        t1 = paddle.zeros([2], dtype="float32")
        t2 = paddle.zeros([2], dtype="float32")
        out = dict_to_tuple_helper({"a": [t1, t2]})
        self.assertEqual(len(out), 2)
        self.assertIs(out[0], t1)
        self.assertIs(out[1], t2)
        self.assertEqual(t1.key, "a 0")
        self.assertEqual(t2.key, "a 1")

    def test_non_dict_passthrough_identity(self):
        t1 = paddle.zeros([2], dtype="float32")
        pair = (t1,)
        self.assertIs(dict_to_tuple_helper(pair), pair)
        self.assertIs(dict_to_tuple_helper(t1), t1)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestScheduleChunk(unittest.TestCase):
    """``ScheduleChunk`` validates its members and composes node forwards in
    order."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_check_nodes_valid_rejects_non_node(self):
        with self.assertRaises(AssertionError):
            fbo.ScheduleChunk([object()])

    def test_accepts_nodes_and_nested_chunks(self):
        node = fbo.ScheduleNode(lambda t: t, name="n")
        inner = fbo.ScheduleChunk([node])
        outer = fbo.ScheduleChunk([node, inner])
        self.assertEqual(len(outer.nodes), 2)

    def test_forward_composes_nodes_in_order(self):
        # f(t)=t*2 then g(t)=t+3  =>  chunk(x) == x*2 + 3 (order-sensitive).
        node_f = fbo.ScheduleNode(lambda t: t * 2, name="f")
        node_g = fbo.ScheduleNode(lambda t: t + 3, name="g")
        chunk = fbo.ScheduleChunk([node_f, node_g])
        x_np = np.array([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]], dtype=np.float32)
        out = chunk.forward(paddle.to_tensor(x_np))
        np.testing.assert_allclose(
            out.numpy(), x_np * 2 + 3, rtol=1e-6, atol=1e-6
        )


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestScheduleNode(unittest.TestCase):
    """``ScheduleNode`` non-recompute path (``is_first_fwd=False``,
    ``use_recompute=False``): forward returns the *real* fwd_func output while
    storing detached inputs, and backward yields the true input gradient."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_forward_returns_real_output_and_stores_detached_inputs(self):
        x_np = np.array([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]], dtype=np.float32)
        x = paddle.to_tensor(x_np)
        x.stop_gradient = False
        node = fbo.ScheduleNode(lambda t: t * t, name="sq")
        out = node.forward(x)
        # The value returned is the genuine fwd output, not the fake clone.
        np.testing.assert_allclose(
            out.numpy(), x_np * x_np, rtol=1e-6, atol=1e-6
        )
        # Inputs are stored detached, with stop_gradient preserved.
        self.assertIsNotNone(node.inputs)
        self.assertIsNot(node.inputs, x)
        self.assertFalse(node.inputs.stop_gradient)
        np.testing.assert_array_equal(node.inputs.numpy(), x_np)
        # A plain (non-first) forward must not flip use_recompute on.
        self.assertFalse(node.use_recompute)

    def test_forward_backward_gradient_matches_independent_derivative(self):
        x_np = np.array([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]], dtype=np.float32)
        g_np = np.array([[7.0, 8.0, 9.0], [1.0, 2.0, 3.0]], dtype=np.float32)
        x = paddle.to_tensor(x_np)
        x.stop_gradient = False
        node = fbo.ScheduleNode(lambda t: t * t, name="sq")
        node.forward(x)
        grads = node.backward(paddle.to_tensor(g_np))
        self.assertEqual(len(grads), 1)
        # d(t*t)/dt = 2t, chained with the supplied upstream grad g.
        np.testing.assert_allclose(
            grads[0].numpy(), 2.0 * x_np * g_np, rtol=1e-5, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
