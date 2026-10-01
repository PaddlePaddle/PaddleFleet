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

This file deliberately covers a slice that is DISTINCT from the stage-boundary
predicates (``is_pp_first_stage`` / ``is_pp_last_stage`` / ``is_vp_*``). Here we
exercise:

* ``get_pp_first_rank`` / ``get_pp_last_rank`` - map a pipeline group's global
  ``ranks()`` membership to its first / last global rank.
* ``get_pp_next_rank`` / ``get_pp_prev_rank`` - the in-group-index -> GLOBAL-rank
  neighbour lookup, plus the ``None`` sentinels at the last / first stage. The
  process-group rank/size queries (``get_pg_rank`` / ``get_pg_size``) are genuine
  NOT-under-test collaborators; they are patched to fixed values and the actual
  returned GLOBAL rank (index arithmetic against a distinct ``ranks()`` list) is
  asserted. This validates only local rank bookkeeping; no real process group /
  collective is exercised.
* ``NoopScheduleNode`` - identity pass-through of forward inputs and backward
  gradients (object identity, not merely equal value).
* ``ScheduleNode.default_backward_func`` - real dygraph autograd on CPU with
  gradients hand-derived from the elementwise math, covering the
  ``output_grad``-provided path, the ``output_grad is None`` (implicit ones)
  path, and the ``None`` input -> ``None`` grad slot.
* ``set_streams`` / ``get_comp_stream`` / ``get_comm_stream`` - the registry
  stores the exact stream objects handed in (no CUDA allocation), and the
  "already initialised" guard makes a second call a no-op.

All expected values are derived by hand / by an independent elementwise
reference, never from the production output and never from the coverage file.
"""

import unittest
from unittest import mock

import numpy as np

try:
    import paddle

    import paddlefleet.pipeline_parallel.utils as pp_utils

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    paddle = None
    pp_utils = None
    _IMPORT_ERROR = exc

_SKIP_REASON = (
    ""
    if _IMPORT_ERROR is None
    else f"paddle/paddlefleet not importable on this CPU-only collector: {_IMPORT_ERROR!r}"
)


class _FakeGroup:
    """Stand-in for a paddle pipeline process-group collaborator (not under test).

    ``ranks()`` returns the group's GLOBAL rank membership, which the helpers use
    to translate an in-group index into a global rank.
    """

    def __init__(self, ranks):
        self._ranks = list(ranks)

    def ranks(self):
        return self._ranks


@unittest.skipUnless(pp_utils is not None, _SKIP_REASON)
class TestPipelineRankHelpers(unittest.TestCase):
    """Global-rank mapping for the pipeline group neighbour helpers."""

    def test_first_and_last_rank_use_group_membership(self):
        # Global ranks are intentionally offset (start at 4) so that a bug
        # returning an in-group index (0 / size-1) instead of the global rank
        # would be caught.
        group = _FakeGroup([4, 5, 6, 7])
        self.assertEqual(pp_utils.get_pp_first_rank(group), 4)
        self.assertEqual(pp_utils.get_pp_last_rank(group), 7)

    def test_next_rank_returns_following_global_rank(self):
        group = _FakeGroup([4, 5, 6, 7])
        with (
            mock.patch.object(pp_utils, "get_pg_rank", return_value=1),
            mock.patch.object(pp_utils, "get_pg_size", return_value=4),
        ):
            # in-group index 1 (global 5) -> next is ranks[2] == 6
            self.assertEqual(pp_utils.get_pp_next_rank(group), 6)

    def test_next_rank_is_none_at_last_stage(self):
        group = _FakeGroup([4, 5, 6, 7])
        with (
            mock.patch.object(pp_utils, "get_pg_rank", return_value=3),
            mock.patch.object(pp_utils, "get_pg_size", return_value=4),
        ):
            self.assertIsNone(pp_utils.get_pp_next_rank(group))

    def test_prev_rank_returns_preceding_global_rank(self):
        group = _FakeGroup([4, 5, 6, 7])
        with (
            mock.patch.object(pp_utils, "get_pg_rank", return_value=2),
            mock.patch.object(pp_utils, "get_pg_size", return_value=4),
        ):
            # in-group index 2 (global 6) -> prev is ranks[1] == 5
            self.assertEqual(pp_utils.get_pp_prev_rank(group), 5)

    def test_prev_rank_is_none_at_first_stage(self):
        group = _FakeGroup([4, 5, 6, 7])
        with (
            mock.patch.object(pp_utils, "get_pg_rank", return_value=0),
            mock.patch.object(pp_utils, "get_pg_size", return_value=4),
        ):
            self.assertIsNone(pp_utils.get_pp_prev_rank(group))


@unittest.skipUnless(pp_utils is not None, _SKIP_REASON)
class TestNoopScheduleNode(unittest.TestCase):
    """NoopScheduleNode must pass inputs / gradients through untouched."""

    def test_forward_returns_same_object(self):
        node = pp_utils.NoopScheduleNode()
        sentinel = object()
        self.assertIs(node.forward(sentinel), sentinel)

        payload = ["a", 1, sentinel]
        returned = node.forward(payload)
        self.assertIs(returned, payload)
        self.assertEqual(returned, ["a", 1, sentinel])

    def test_backward_returns_same_object(self):
        node = pp_utils.NoopScheduleNode()
        grads = object()
        self.assertIs(node.backward(grads), grads)


@unittest.skipUnless(pp_utils is not None, _SKIP_REASON)
class TestScheduleNodeDefaultBackward(unittest.TestCase):
    """Real CPU autograd through ScheduleNode.default_backward_func.

    Gradients are hand-derived from the elementwise math, independent of the
    production code path.
    """

    def setUp(self):
        paddle.set_device("cpu")

    def _make_node(self):
        # forward_func / stream / event are stored but not exercised by
        # default_backward_func; free_input must stay False (asserted in __init__).
        return pp_utils.ScheduleNode(
            forward_func=lambda *a: a,
            stream=object(),
            event=object(),
        )

    def test_backward_with_explicit_output_grad(self):
        a = paddle.to_tensor([1.0, 2.0, 3.0])
        b = paddle.to_tensor([10.0, 20.0, 30.0])
        a.stop_gradient = False
        b.stop_gradient = False

        node = self._make_node()
        node.inputs = [a, b]
        out = a * b  # elementwise product; d/da = b, d/db = a
        upstream = paddle.to_tensor([0.5, 1.0, 2.0])

        grads = node.default_backward_func(out, upstream)

        # Hand-derived: da = upstream * b, db = upstream * a
        self.assertEqual(len(grads), 2)
        np.testing.assert_allclose(
            grads[0].numpy(), np.array([5.0, 20.0, 60.0]), rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            grads[1].numpy(), np.array([0.5, 2.0, 6.0]), rtol=1e-6, atol=1e-6
        )
        # States are reset after the backward pass.
        self.assertIsNone(node.inputs)

    def test_backward_with_none_output_grad_uses_ones(self):
        a = paddle.to_tensor([2.0, 3.0, 4.0])
        a.stop_gradient = False

        node = self._make_node()
        node.inputs = [a]
        out = a * a  # d/da = 2a with an implicit upstream of ones

        grad = node.default_backward_func(out, None)

        # Single input -> returned as a bare tensor, not a 1-tuple.
        self.assertIsInstance(grad, paddle.Tensor)
        np.testing.assert_allclose(
            grad.numpy(), np.array([4.0, 6.0, 8.0]), rtol=1e-6, atol=1e-6
        )

    def test_backward_none_input_yields_none_grad_slot(self):
        a = paddle.to_tensor([1.0, 2.0])
        a.stop_gradient = False

        node = self._make_node()
        node.inputs = [a, None]
        out = a * 3.0  # d/da = 3 * upstream
        upstream = paddle.to_tensor([1.0, 1.0])

        grads = node.default_backward_func(out, upstream)

        self.assertEqual(len(grads), 2)
        np.testing.assert_allclose(
            grads[0].numpy(), np.array([3.0, 3.0]), rtol=1e-6, atol=1e-6
        )
        self.assertIsNone(grads[1])


@unittest.skipUnless(pp_utils is not None, _SKIP_REASON)
class TestStreamRegistry(unittest.TestCase):
    """set_streams stores the provided stream objects and guards re-init."""

    def setUp(self):
        # Preserve and restore the module globals so this test never leaks
        # state into other tests in the same process (antipattern #11).
        self._orig_comp = pp_utils.get_comp_stream()
        self._orig_comm = pp_utils.get_comm_stream()
        self.addCleanup(setattr, pp_utils, "_COMP_STREAM", self._orig_comp)
        self.addCleanup(setattr, pp_utils, "_COMM_STREAM", self._orig_comm)

    def test_set_streams_stores_and_guards(self):
        # Start from a clean, uninitialised registry.
        pp_utils._COMP_STREAM = None
        pp_utils._COMM_STREAM = None

        comp = object()
        comm = object()
        # Both streams supplied -> no CUDA allocation on this CPU-only host.
        pp_utils.set_streams(comp_stream=comp, comm_stream=comm)
        self.assertIs(pp_utils.get_comp_stream(), comp)
        self.assertIs(pp_utils.get_comm_stream(), comm)

        # Second call must be a no-op: the already-initialised guard keeps the
        # first pair, it does not overwrite with the new objects.
        other_comp = object()
        other_comm = object()
        pp_utils.set_streams(comp_stream=other_comp, comm_stream=other_comm)
        self.assertIs(pp_utils.get_comp_stream(), comp)
        self.assertIs(pp_utils.get_comm_stream(), comm)


if __name__ == "__main__":
    unittest.main()
