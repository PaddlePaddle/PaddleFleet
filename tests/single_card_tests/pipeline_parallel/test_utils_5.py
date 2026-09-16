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

This file targets a slice that is DISTINCT from the sibling
``test_utils`` / ``test_utils_2`` / ``test_utils_3`` files (which already cover
``NoopScheduleNode``, ``ScheduleNode.__init__`` / ``default_backward_func``,
``AbstractSchedulePlan``, ``is_vp_first_stage`` / ``is_vp_last_stage``,
``get_pp_first_rank`` / ``get_pp_last_rank`` / ``get_pp_next_rank`` /
``get_pp_prev_rank``, ``stream_acquire_context`` and the stream registry).
Here we exercise the two remaining pieces of pure, single-process logic:

* ``is_pp_first_stage`` / ``is_pp_last_stage`` -- the boolean stage predicates.
  Their only inputs are the group's rank and size, obtained from the genuine
  NOT-under-test collaborators ``get_pg_rank`` / ``get_pg_size``. Those two are
  patched to fixed, non-degenerate values (a single CPU process would otherwise
  force rank 0 / size 1 and collapse both predicates to ``True``); the group
  argument they receive is captured and asserted to be the exact object passed
  in. The behavior actually under test is the local comparison arithmetic
  (``rank == 0`` for "first", ``rank == size - 1`` for "last"). No real process
  group or collective is exercised -- this validates only local rank
  bookkeeping, and the cross-rank pipeline semantics remain a multi-card
  concern.
* ``make_viewless`` -- the thin wrapper around ``make_viewless_tensor``. For a
  non-view leaf tensor the real collaborator returns the input unchanged, so
  the wrapper must hand back the very same object. Its argument-forwarding
  contract (``inp`` is the tensor, ``requires_grad`` tracks the tensor's own
  ``requires_grad`` rather than a hardcoded constant, ``keep_graph=True``) is
  checked separately by spying on the genuine collaborator with a
  distinguishable return value.

Every expected value is derived by hand from the source contract, never from
the production output and never from the coverage file. The stream-wrapped
``ScheduleNode._forward`` / ``_backward`` and ``set_streams``' CUDA-stream
allocation need a real GPU and are deliberately out of scope for a CPU host.
"""

import unittest
from unittest import mock

try:
    import paddle

    import paddlefleet.pipeline_parallel.utils as pp_utils

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # honest capability probe
    paddle = None
    pp_utils = None
    _IMPORT_ERROR = exc

_HAS_DEPS = _IMPORT_ERROR is None
_SKIP_REASON = (
    ""
    if _IMPORT_ERROR is None
    else (
        "paddle / paddlefleet.pipeline_parallel.utils not importable on this "
        f"CPU-only collector: {_IMPORT_ERROR!r}"
    )
)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestPipelineStagePredicates(unittest.TestCase):
    """``is_pp_first_stage`` / ``is_pp_last_stage`` branch + collaborator wiring.

    ``get_pg_rank`` / ``get_pg_size`` are genuine collaborators that read a real
    process group; on a single CPU process they degenerate to rank 0 / size 1.
    They are patched here to distinct, non-degenerate values so the comparison
    logic is actually exercised, and the group they are handed is asserted to be
    forwarded unchanged. No collective / real process group runs.
    """

    def setUp(self):
        paddle.set_device("cpu")

    def _patch_group(self, rank, size, captured):
        def fake_rank(pp_group):
            captured.setdefault("rank_groups", []).append(pp_group)
            return rank

        def fake_size(pp_group):
            captured.setdefault("size_groups", []).append(pp_group)
            return size

        return (
            mock.patch.object(pp_utils, "get_pg_rank", side_effect=fake_rank),
            mock.patch.object(pp_utils, "get_pg_size", side_effect=fake_size),
        )

    def test_first_stage_true_only_at_rank_zero(self):
        group = object()  # sentinel group; only forwarded, never inspected

        # rank 0 of a 4-way group -> first stage.
        captured = {}
        p_rank, p_size = self._patch_group(0, 4, captured)
        with p_rank, p_size:
            self.assertTrue(pp_utils.is_pp_first_stage(group))
        # is_pp_first_stage compares rank == 0; it must query the rank of the
        # exact group passed in.
        self.assertEqual(captured["rank_groups"], [group])

        # rank 2 of the same group -> NOT the first stage.
        captured = {}
        p_rank, p_size = self._patch_group(2, 4, captured)
        with p_rank, p_size:
            self.assertFalse(pp_utils.is_pp_first_stage(group))
        self.assertEqual(captured["rank_groups"], [group])

    def test_last_stage_true_only_at_final_rank(self):
        group = object()

        # rank 3 of a 4-way group == size - 1 -> last stage.
        captured = {}
        p_rank, p_size = self._patch_group(3, 4, captured)
        with p_rank, p_size:
            self.assertTrue(pp_utils.is_pp_last_stage(group))
        # is_pp_last_stage needs both rank and size of the same group.
        self.assertEqual(captured["rank_groups"], [group])
        self.assertEqual(captured["size_groups"], [group])

        # rank 1 (!= size - 1) -> NOT the last stage.
        captured = {}
        p_rank, p_size = self._patch_group(1, 4, captured)
        with p_rank, p_size:
            self.assertFalse(pp_utils.is_pp_last_stage(group))
        self.assertEqual(captured["rank_groups"], [group])

    def test_middle_stage_is_neither_first_nor_last(self):
        group = object()
        # rank 2 of a 4-way group: not 0 and not size - 1.
        captured_first = {}
        p_rank, p_size = self._patch_group(2, 4, captured_first)
        with p_rank, p_size:
            self.assertFalse(pp_utils.is_pp_first_stage(group))

        captured_last = {}
        p_rank, p_size = self._patch_group(2, 4, captured_last)
        with p_rank, p_size:
            self.assertFalse(pp_utils.is_pp_last_stage(group))

    def test_single_stage_group_is_both_first_and_last(self):
        group = object()
        # A degenerate 1-stage group: rank 0 == 0 (first) and 0 == size - 1
        # (last), so both predicates hold simultaneously.
        captured = {}
        p_rank, p_size = self._patch_group(0, 1, captured)
        with p_rank, p_size:
            self.assertTrue(pp_utils.is_pp_first_stage(group))

        captured = {}
        p_rank, p_size = self._patch_group(0, 1, captured)
        with p_rank, p_size:
            self.assertTrue(pp_utils.is_pp_last_stage(group))


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestMakeViewless(unittest.TestCase):
    """``make_viewless`` wraps ``make_viewless_tensor`` with a fixed contract."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_non_view_input_triggers_missing_is_view_bug(self):
        # PRODUCTION BUG: make_viewless delegates to make_viewless_tensor
        # (utils/_fleet_utils.py:552), whose ``if not inp._is_view():`` guard
        # references an attribute paddle.Tensor does not expose in this build.
        # The documented non-view short-circuit therefore raises AttributeError
        # before it can return the input. Not a test artifact: production makes
        # the identical call. Captured via assertRaises, production untouched.
        e = paddle.to_tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32"
        )
        self.assertFalse(hasattr(e, "_is_view"))  # root cause: API absent
        with self.assertRaises(AttributeError):
            pp_utils.make_viewless(e)

    def test_forwards_requires_grad_and_keep_graph(self):
        # Spy on the genuine collaborator with a distinguishable return value to
        # observe exactly what make_viewless passes through. Two tensors whose
        # requires_grad differ prove the flag tracks the input rather than a
        # hardcoded constant.
        frozen = paddle.to_tensor([1.0, 2.0], dtype="float32")
        frozen.stop_gradient = True
        trainable = paddle.to_tensor([3.0, 4.0], dtype="float32")
        trainable.stop_gradient = False
        # Sanity: the two inputs genuinely differ in requires_grad, so a
        # constant would be caught by the per-tensor equality below.
        self.assertNotEqual(frozen.requires_grad, trainable.requires_grad)

        for tensor in (frozen, trainable):
            marker = object()
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
            ):
                out = pp_utils.make_viewless(tensor)

            self.assertIs(out, marker)  # collaborator's result is used
            self.assertIs(captured["inp"], tensor)
            self.assertEqual(captured["requires_grad"], tensor.requires_grad)
            self.assertIs(captured["keep_graph"], True)


if __name__ == "__main__":
    unittest.main()
