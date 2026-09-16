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

"""CPU-only behavior tests for pure logic in
``paddlefleet.pipeline_parallel.utils``.

This module targets a slice that is DISTINCT from the sibling
``test_forward_backward_overlap_utils_*`` tests (those cover the separate
``pp_utils.forward_backward_overlap_utils`` module). Here we exercise only the
device-independent, single-process logic of ``pipeline_parallel/utils.py``:

* ``is_vp_first_stage`` / ``is_vp_last_stage`` -- pure integer/branch logic,
  including the assertion contract that rejects a non-zero ``vp_stage`` when
  ``vp_size`` is ``None`` or ``<= 1``.
* ``get_pp_first_rank`` / ``get_pp_last_rank`` -- pure index selection over the
  group's ``ranks()`` list (first vs. last element, not min/max/sorted).
* ``NoopScheduleNode`` -- identity pass-through in both directions.
* ``stream_acquire_context`` -- ordering contract (wait before body, record
  after) and the ``finally`` guarantee that ``record`` runs even on exception.
* ``ScheduleNode.default_backward_func`` -- real eager autograd on CPU with
  hand-derived gradients (no mocked kernels, no CUDA streams touched).

The stage/peer helpers that depend on a real process group
(``is_pp_first_stage``, ``is_pp_last_stage``, ``get_pp_next_rank``,
``get_pp_prev_rank``) degenerate on a single CPU process (``get_pg_rank``
forces rank 0 when distributed is not initialized), so asserting their peer
arithmetic here would only prove a degenerate path; they are deliberately left
to real multi-card tests. Likewise the stream-wrapped ``ScheduleNode._forward``
/ ``_backward`` require CUDA streams and are out of scope for CPU.

Every expected gradient is derived by hand from the closed form of the forward
op (e.g. d(2x)/dx = 2), never by re-running the production code as its oracle.
"""

import unittest

try:
    import numpy as np
    import paddle

    from paddlefleet.pipeline_parallel.utils import (
        NoopScheduleNode,
        ScheduleNode,
        get_pp_first_rank,
        get_pp_last_rank,
        is_vp_first_stage,
        is_vp_last_stage,
        stream_acquire_context,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # honest capability probe
    _IMPORT_ERROR = exc

PADDLE_AVAILABLE = _IMPORT_ERROR is None
SKIP_REASON = (
    "paddle / paddlefleet pipeline_parallel.utils import failed: "
    f"{_IMPORT_ERROR!r}"
    if _IMPORT_ERROR is not None
    else ""
)


class _FakeGroup:
    """Minimal stand-in for a process group exposing only ``ranks()``.

    ``get_pp_first_rank`` / ``get_pp_last_rank`` read ``pp_group.ranks()``
    directly and select index 0 / -1; they do not perform communication, so a
    plain list is a faithful, non-degenerate input here.
    """

    def __init__(self, ranks):
        self._ranks = list(ranks)

    def ranks(self):
        return self._ranks


class _RecordingEvent:
    """Records the ordered (op, stream) calls made by a context manager."""

    def __init__(self, log):
        self._log = log

    def wait(self, stream):
        self._log.append(("wait", stream))

    def record(self, stream):
        self._log.append(("record", stream))


@unittest.skipUnless(PADDLE_AVAILABLE, SKIP_REASON)
class TestVirtualPipelineStagePredicates(unittest.TestCase):
    """is_vp_first_stage / is_vp_last_stage pure branch + assertion logic."""

    def test_none_or_degenerate_vp_size_is_both_ends(self):
        # vp_size None or <= 1 => the single stage is both first and last.
        for vp_size in (None, 1):
            self.assertTrue(is_vp_first_stage(None, vp_size))
            self.assertTrue(is_vp_first_stage(0, vp_size))
            self.assertTrue(is_vp_last_stage(None, vp_size))
            self.assertTrue(is_vp_last_stage(0, vp_size))

    def test_multi_vp_first_and_last_are_distinguished(self):
        # vp_size = 4 separates the two predicates: stage 0 is first-only,
        # stage 3 is last-only, stages in between are neither.
        self.assertTrue(is_vp_first_stage(0, 4))
        self.assertFalse(is_vp_first_stage(1, 4))
        self.assertFalse(is_vp_first_stage(3, 4))

        self.assertTrue(is_vp_last_stage(3, 4))
        self.assertFalse(is_vp_last_stage(0, 4))
        self.assertFalse(is_vp_last_stage(2, 4))

    def test_nonzero_stage_with_degenerate_size_asserts(self):
        # Contract: a non-zero vp_stage is illegal when vp_size is None/<=1.
        for vp_size in (None, 1):
            with self.assertRaises(AssertionError):
                is_vp_first_stage(2, vp_size)
            with self.assertRaises(AssertionError):
                is_vp_last_stage(2, vp_size)


@unittest.skipUnless(PADDLE_AVAILABLE, SKIP_REASON)
class TestPipelineFirstLastRankSelection(unittest.TestCase):
    """get_pp_first_rank / get_pp_last_rank index selection over ranks()."""

    def test_selects_actual_first_and_last_not_sorted(self):
        # Distinguishable, deliberately unsorted rank list: first element is
        # not the min and last element is not the max, so a min/max/sort bug
        # would be caught.
        group = _FakeGroup([3, 7, 2, 5])
        self.assertEqual(get_pp_first_rank(group), 3)
        self.assertEqual(get_pp_last_rank(group), 5)

    def test_single_rank_group(self):
        group = _FakeGroup([9])
        self.assertEqual(get_pp_first_rank(group), 9)
        self.assertEqual(get_pp_last_rank(group), 9)


@unittest.skipUnless(PADDLE_AVAILABLE, SKIP_REASON)
class TestNoopScheduleNode(unittest.TestCase):
    """NoopScheduleNode passes inputs/outgrads through unchanged (identity)."""

    def test_forward_returns_same_object(self):
        node = NoopScheduleNode()
        sentinel = ("payload", paddle.to_tensor([1.0, 2.0]))
        self.assertIs(node.forward(sentinel), sentinel)

    def test_backward_returns_same_object(self):
        node = NoopScheduleNode()
        grads = [paddle.to_tensor([3.0, 4.0]), None]
        self.assertIs(node.backward(grads), grads)


@unittest.skipUnless(PADDLE_AVAILABLE, SKIP_REASON)
class TestStreamAcquireContext(unittest.TestCase):
    """stream_acquire_context ordering and finally-on-exception contract."""

    def test_wait_before_body_record_after(self):
        log = []
        stream = object()
        event = _RecordingEvent(log)
        with stream_acquire_context(stream, event):
            log.append(("body", stream))
        self.assertEqual(
            log,
            [("wait", stream), ("body", stream), ("record", stream)],
        )

    def test_record_runs_even_when_body_raises(self):
        log = []
        stream = object()
        event = _RecordingEvent(log)
        with (
            self.assertRaises(ValueError),
            stream_acquire_context(stream, event),
        ):
            log.append(("body", stream))
            raise ValueError("boom")
        # record must still fire via the finally clause, after wait+body.
        self.assertEqual(
            log,
            [("wait", stream), ("body", stream), ("record", stream)],
        )


@unittest.skipUnless(PADDLE_AVAILABLE, SKIP_REASON)
class TestScheduleNodeDefaultBackward(unittest.TestCase):
    """ScheduleNode.default_backward_func real CPU autograd, hand-derived grads.

    default_backward_func touches no CUDA streams; it drives
    paddle.autograd.backward and reads .grad off self.inputs. We construct the
    node, seed self.inputs with real leaf tensors, build the graph with plain
    ops whose derivatives are known by hand, then compare.
    """

    @staticmethod
    def _make_node():
        # stream/event are stored but unused by default_backward_func;
        # free_input must be False (asserted in __init__).
        return ScheduleNode(
            forward_func=lambda *a: a,
            stream=None,
            event=None,
        )

    def test_single_input_with_upstream_grad(self):
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        x.stop_gradient = False
        node = self._make_node()
        node.inputs = [x]
        out = x * 2.0  # d(2x)/dx = 2
        upstream = paddle.to_tensor([0.1, 0.2, 0.3])
        grad = node.default_backward_func(out, upstream)
        # single input => returns the tensor directly, not a 1-tuple.
        self.assertIsInstance(grad, paddle.Tensor)
        np.testing.assert_allclose(
            grad.numpy(), [0.2, 0.4, 0.6], rtol=1e-6, atol=1e-6
        )
        # states are reset after the backward.
        self.assertIsNone(node.inputs)

    def test_single_input_none_upstream_uses_scalar_backward(self):
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        x.stop_gradient = False
        node = self._make_node()
        node.inputs = [x]
        out = (x * 2.0).sum()  # scalar; d(sum 2x)/dx = 2 everywhere
        grad = node.default_backward_func(out, None)
        np.testing.assert_allclose(
            grad.numpy(), [2.0, 2.0, 2.0], rtol=1e-6, atol=1e-6
        )

    def test_multiple_inputs_return_grad_per_input(self):
        x = paddle.to_tensor([1.0, 2.0])
        y = paddle.to_tensor([3.0, 4.0])
        x.stop_gradient = False
        y.stop_gradient = False
        node = self._make_node()
        node.inputs = [x, y]
        out = x * 3.0 + y * 5.0  # dx = 3*g, dy = 5*g
        upstream = paddle.to_tensor([1.0, 1.0])
        grad = node.default_backward_func(out, upstream)
        self.assertIsInstance(grad, tuple)
        self.assertEqual(len(grad), 2)
        np.testing.assert_allclose(
            grad[0].numpy(), [3.0, 3.0], rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            grad[1].numpy(), [5.0, 5.0], rtol=1e-6, atol=1e-6
        )

    def test_none_input_yields_none_grad_slot(self):
        x = paddle.to_tensor([1.0, 2.0])
        x.stop_gradient = False
        node = self._make_node()
        node.inputs = [x, None]  # second input absent
        out = x * 4.0  # dx = 4*g
        upstream = paddle.to_tensor([1.0, 1.0])
        grad = node.default_backward_func(out, upstream)
        self.assertIsInstance(grad, tuple)
        self.assertEqual(len(grad), 2)
        np.testing.assert_allclose(
            grad[0].numpy(), [4.0, 4.0], rtol=1e-6, atol=1e-6
        )
        self.assertIsNone(grad[1])


if __name__ == "__main__":
    unittest.main()
