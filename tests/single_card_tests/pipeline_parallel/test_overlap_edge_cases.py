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

Only the pieces whose correctness is local (branch selection, container
reshaping, autograd wiring) are exercised here:

* ``detach_and_requires_grad`` - detaches tensors, *preserves* each tensor's
  ``stop_gradient`` flag (it does not force grad on despite the name), passes
  non-tensors through unchanged in tuples/lists, and maps ``None`` dict values
  to ``None``.
* ``clone_and_clear_dataptr`` - rebuilds the container while *dropping* ``None``
  and non-tensor entries (unlike ``detach_and_requires_grad``, which keeps
  them). The cloned payload comes from ``FakeClone`` (``empty_like``), so its
  numeric content is uninitialised and is deliberately never asserted.
* ``FakeClone`` (a ``paddle.autograd.PyLayer``) - forward yields an
  ``empty_like`` tensor (shape/dtype preserved, content undefined) and backward
  is the identity, so the input gradient equals the upstream gradient exactly.
* ``ScheduleNode.forward`` / ``ScheduleNode.backward`` on the default
  ``use_recompute == False`` path - the returned forward value is the real
  computed output, ``scale_loss_factor`` divides it, and the backward gradient
  handed back for ``f(x) = x * 2`` is ``2 * upstream`` (scale-sensitive, from an
  independently written expectation), after which ``_reset_states`` clears the
  node.

Expected numbers are derived by hand / with independent numpy, never by calling
the code under test. This file does NOT exercise pipeline scheduling across
ranks, recompute/AMP state restoration, or any real process-group behavior;
those require a real multi-card pipeline and are out of scope for a CPU unit.
"""

import unittest

import numpy as np

try:
    import paddle

    from paddlefleet.pipeline_parallel.pp_utils.forward_backward_overlap_utils import (
        FakeClone,
        ScheduleNode,
        clone_and_clear_dataptr,
        detach_and_requires_grad,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # honest dependency probe
    paddle = None
    FakeClone = None
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
    """``detach_and_requires_grad`` container handling and flag preservation."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_single_tensor_preserves_value_and_stop_gradient(self):
        # else-branch: a bare tensor is detached to a distinct object whose
        # value is unchanged and whose stop_gradient flag is copied verbatim.
        for flag in (True, False):
            x = paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            x.stop_gradient = flag
            out = detach_and_requires_grad(x)
            self.assertIsNot(out, x)
            self.assertEqual(out.stop_gradient, flag)
            np.testing.assert_array_equal(
                out.numpy(), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
            )

    def test_tuple_passes_non_tensor_through_unchanged(self):
        # tuple-branch: tensors are detached (value + flag kept); the plain int
        # is returned as the *same* object, and the tuple type is preserved.
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        x.stop_gradient = False
        out = detach_and_requires_grad((x, 42))
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        self.assertIs(out[1], 42)
        self.assertFalse(out[0].stop_gradient)
        np.testing.assert_array_equal(out[0].numpy(), [1.0, 2.0, 3.0])

    def test_nested_list_recurses_and_preserves_content(self):
        # list-branch recurses into the inner list; structure, values and the
        # stop_gradient flag survive the round trip.
        x = paddle.to_tensor([[7.0, 8.0]])
        x.stop_gradient = True
        out = detach_and_requires_grad([[x], 5])
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 2)
        self.assertIsInstance(out[0], list)
        self.assertEqual(out[1], 5)
        self.assertTrue(out[0][0].stop_gradient)
        np.testing.assert_array_equal(out[0][0].numpy(), [[7.0, 8.0]])

    def test_dict_maps_none_to_none_and_detaches_tensor(self):
        # dict-branch: None value stays None; the tensor value is detached with
        # its value and stop_gradient flag intact.
        x = paddle.to_tensor([9.0, 10.0])
        x.stop_gradient = False
        out = detach_and_requires_grad({"a": x, "b": None})
        self.assertEqual(set(out), {"a", "b"})
        self.assertIsNone(out["b"])
        self.assertIsNot(out["a"], x)
        self.assertFalse(out["a"].stop_gradient)
        np.testing.assert_array_equal(out["a"].numpy(), [9.0, 10.0])

    def test_dict_non_tensor_value_raises(self):
        # Documented asymmetry with the tuple/list branch: the dict branch calls
        # ``.detach()`` on every non-None value, so a plain int raises
        # AttributeError instead of passing through. Locked with assertRaises so
        # a future change of this behavior is noticed.
        x = paddle.to_tensor([1.0])
        with self.assertRaises(AttributeError):
            detach_and_requires_grad({"a": x, "b": 42})


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestCloneAndClearDataptr(unittest.TestCase):
    """``clone_and_clear_dataptr`` filtering and container-shape contract.

    The cloned payload is produced by ``FakeClone`` (``empty_like``), so element
    *content* is uninitialised and is never compared here; only which entries
    survive, the container type, and per-entry shape are asserted.
    """

    def setUp(self):
        paddle.set_device("cpu")

    def test_list_drops_none_and_non_tensor_entries(self):
        t1 = paddle.zeros([2, 3], dtype="float32")
        out = clone_and_clear_dataptr([t1, None, 42], clear_dataptr=True)
        self.assertIsInstance(out, list)
        # Only the single real tensor survives the ``is not None and Tensor``
        # filter -> length collapses from 3 to 1.
        self.assertEqual(len(out), 1)
        # clear_dataptr=True releases the clone's storage, so its shape
        # collapses to []; it is still a distinct object from the source.
        self.assertEqual(list(out[0].shape), [])
        self.assertIsNot(out[0], t1)

    def test_tuple_type_preserved_and_shapes_kept(self):
        t1 = paddle.zeros([2, 3], dtype="float32")
        t2 = paddle.zeros([3, 4], dtype="float32")
        out = clone_and_clear_dataptr((t1, t2))
        self.assertIsInstance(out, tuple)
        self.assertEqual([list(t.shape) for t in out], [[2, 3], [3, 4]])

    def test_dict_drops_none_valued_keys(self):
        t1 = paddle.zeros([5], dtype="float32")
        out = clone_and_clear_dataptr({"a": t1, "b": None}, clear_dataptr=True)
        self.assertIsInstance(out, dict)
        # ``b`` is dropped because its value is None; ``a`` survives, but with
        # clear_dataptr=True its storage is released so its shape collapses to
        # [].
        self.assertEqual(set(out), {"a"})
        self.assertEqual(list(out["a"].shape), [])

    def test_single_tensor_returns_distinct_same_shape_tensor(self):
        t = paddle.zeros([1, 3, 4], dtype="float32")
        out = clone_and_clear_dataptr(t)
        self.assertIsInstance(out, paddle.Tensor)
        self.assertIsNot(out, t)
        self.assertEqual(list(out.shape), [1, 3, 4])
        self.assertEqual(out.dtype, t.dtype)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestFakeClone(unittest.TestCase):
    """``FakeClone`` PyLayer: shape/dtype-preserving forward, identity backward.

    ``forward`` returns ``empty_like`` so the output *content* is undefined and
    is never asserted. The load-bearing contract is the backward: it returns the
    upstream gradient unchanged, so the input gradient equals it exactly.
    """

    def setUp(self):
        paddle.set_device("cpu")

    def test_forward_preserves_shape_and_dtype(self):
        for shape in ([100], [2, 3, 4], [1, 3, 8, 8]):
            t = paddle.zeros(shape, dtype="float32")
            out = FakeClone.apply(t)
            self.assertEqual(list(out.shape), shape)
            self.assertEqual(out.dtype, t.dtype)
            self.assertIsNot(out, t)

    def test_backward_is_identity_on_upstream_gradient(self):
        # Independent expectation: FakeClone.backward returns grad_output as-is,
        # so d(input) == upstream. A non-uniform, distinguishable upstream would
        # expose any reordering or scaling.
        x = paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        x.stop_gradient = False
        upstream = paddle.to_tensor([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]])
        out = FakeClone.apply(x)
        out.backward(upstream)
        self.assertIsNotNone(x.grad)
        np.testing.assert_allclose(
            x.grad.numpy(),
            [[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]],
            rtol=1e-6,
            atol=1e-6,
        )


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestScheduleNodeForwardBackward(unittest.TestCase):
    """``ScheduleNode`` on the default (non-recompute) forward/backward path."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_forward_returns_real_computed_output(self):
        # No labels, no scaling: the returned value is exactly the fwd_func
        # output over the detached inputs (sum of 1..6 == 21).
        node = ScheduleNode(fwd_func=lambda inputs, **kw: inputs.sum())
        x = paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        x.stop_gradient = False
        out = node.forward(x)
        np.testing.assert_allclose(out.numpy(), 21.0, rtol=1e-6, atol=1e-6)
        # The detached input is stored and remains grad-tracking.
        self.assertIsNotNone(node.inputs)
        self.assertFalse(node.inputs.stop_gradient)

    def test_forward_applies_scale_loss_factor_with_labels(self):
        # With labels the two-arg fwd_func runs, then the result is divided by
        # scale_loss_factor: (sum(1..6)=21 + sum([10,20])=30) / 2 == 25.5.
        node = ScheduleNode(
            fwd_func=lambda inputs, labels, **kw: inputs.sum() + labels.sum()
        )
        node.labels = paddle.to_tensor([10.0, 20.0])
        node.scale_loss_factor = 2.0
        x = paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        x.stop_gradient = False
        out = node.forward(x)
        np.testing.assert_allclose(out.numpy(), 25.5, rtol=1e-6, atol=1e-6)

    def test_backward_returns_scaled_input_gradient_and_resets(self):
        # fwd_func computes t = inputs * 2 and returns it as a 1-tuple. Backward
        # with a non-uniform upstream must yield d(inputs) = 2 * upstream, an
        # expectation written independently of the code under test.
        node = ScheduleNode(fwd_func=lambda inputs, **kw: (inputs * 2,))
        x = paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        x.stop_gradient = False
        result = node.forward(x)
        np.testing.assert_allclose(
            result[0].numpy(),
            [[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]],
            rtol=1e-6,
            atol=1e-6,
        )

        upstream = paddle.to_tensor([[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]])
        grads = node.backward(output_grad=upstream)
        self.assertIsInstance(grads, tuple)
        self.assertEqual(len(grads), 1)
        np.testing.assert_allclose(
            grads[0].numpy(),
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            rtol=1e-6,
            atol=1e-6,
        )
        # _reset_states clears the node after backward.
        self.assertIsNone(node.inputs)
        self.assertIsNone(node.outputs)
        self.assertIsNone(node.labels)
        self.assertIsNone(node.scale_loss_factor)


if __name__ == "__main__":
    unittest.main()
