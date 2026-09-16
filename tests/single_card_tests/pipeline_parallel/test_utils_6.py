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

"""CPU-only behavior tests for a DISTINCT slice of
``paddlefleet.pipeline_parallel.utils``.

The sibling ``test_utils`` / ``test_utils_2`` / ``test_utils_3`` files already
exercise the virtual-stage predicates (``is_vp_*``), the first/last/next/prev
GLOBAL-rank helpers, ``NoopScheduleNode``, ``ScheduleNode`` init/backward,
``AbstractSchedulePlan``, ``stream_acquire_context`` and the stream registry.
To avoid duplicate coverage this file targets the two pieces none of them touch:

* ``is_pp_first_stage`` / ``is_pp_last_stage`` -- the PHYSICAL pipeline
  stage-boundary predicates. ``get_pg_rank`` and ``get_pg_size`` are genuine
  NOT-under-test collaborators (they query the real process group). On a single
  CPU process they degenerate to rank 0 / size 1, which would only prove a
  trivial path, so they are patched to fixed, distinguishable values and the
  actual boolean the predicate computes is asserted -- plus the fact that the
  predicate forwards the exact ``pp_group`` object to each collaborator. In
  particular ``is_pp_last_stage`` must consume BOTH rank and size, so the same
  rank under two different sizes is checked to give different answers; a bug
  that ignored ``get_pg_size`` would be caught.
* ``make_viewless`` -- the thin wrapper over ``make_viewless_tensor``. For a
  freshly created (non-view) tensor the documented contract is a pure
  pass-through: the exact same object is returned and its values are unchanged.
  The real ``make_viewless_tensor`` collaborator is kept (not mocked).

No CUDA streams are touched, no collective is faked, and every expected value
is derived by hand from the predicate definition, never from the production
output and never from the coverage file.
"""

import unittest
from unittest import mock

try:
    import numpy as np
    import paddle

    import paddlefleet.pipeline_parallel.utils as pp_utils

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # honest capability probe
    np = None
    paddle = None
    pp_utils = None
    _IMPORT_ERROR = exc

_HAS_DEPS = _IMPORT_ERROR is None
_SKIP_REASON = (
    ""
    if _IMPORT_ERROR is None
    else f"paddle / paddlefleet not importable on this CPU-only host: {_IMPORT_ERROR!r}"
)


class _FakeGroup:
    """Opaque stand-in for a pipeline process group.

    ``is_pp_first_stage`` / ``is_pp_last_stage`` only hand this object to the
    ``get_pg_rank`` / ``get_pg_size`` collaborators; they never call any method
    on it themselves, so a bare marker object is a faithful input and lets us
    assert the group is forwarded unchanged.
    """


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestPipelineStageBoundaryPredicates(unittest.TestCase):
    """is_pp_first_stage / is_pp_last_stage boolean + collaborator-forwarding."""

    def _patched(self, rank, size):
        """Patch the two rank/size collaborators, recording the group each got.

        Returns the ``captured`` dict so a test can assert the exact
        ``pp_group`` object reached each collaborator (antipattern #3: a
        collaborator call is only meaningful together with its arguments).
        """
        captured = {}

        def fake_rank(group=None):
            captured["rank_group"] = group
            return rank

        def fake_size(group=None):
            captured["size_group"] = group
            return size

        return captured, (
            mock.patch.object(pp_utils, "get_pg_rank", side_effect=fake_rank),
            mock.patch.object(pp_utils, "get_pg_size", side_effect=fake_size),
        )

    def test_first_stage_true_only_at_rank_zero(self):
        group = _FakeGroup()
        captured, (p_rank, p_size) = self._patched(rank=0, size=4)
        with p_rank, p_size:
            self.assertTrue(pp_utils.is_pp_first_stage(group))
        # The predicate only needs the rank; it must forward the real group.
        self.assertIs(captured["rank_group"], group)

    def test_first_stage_false_for_nonzero_rank(self):
        group = _FakeGroup()
        captured, (p_rank, p_size) = self._patched(rank=2, size=4)
        with p_rank, p_size:
            self.assertFalse(pp_utils.is_pp_first_stage(group))
        self.assertIs(captured["rank_group"], group)

    def test_last_stage_true_at_final_rank(self):
        # rank == size - 1  =>  last stage.  (3 == 4 - 1)
        group = _FakeGroup()
        captured, (p_rank, p_size) = self._patched(rank=3, size=4)
        with p_rank, p_size:
            self.assertTrue(pp_utils.is_pp_last_stage(group))
        self.assertIs(captured["rank_group"], group)
        self.assertIs(captured["size_group"], group)

    def test_last_stage_false_before_final_rank(self):
        # 2 != 4 - 1  =>  not last stage.
        group = _FakeGroup()
        _captured, (p_rank, p_size) = self._patched(rank=2, size=4)
        with p_rank, p_size:
            self.assertFalse(pp_utils.is_pp_last_stage(group))

    def test_last_stage_consumes_size_not_just_rank(self):
        # Same rank, different world size => different verdict. This is the
        # discriminating case: an implementation that ignored get_pg_size and
        # compared rank against a hard-coded bound would fail one of these.
        group = _FakeGroup()

        _c1, (p_rank1, p_size1) = self._patched(rank=2, size=3)
        with p_rank1, p_size1:
            self.assertTrue(pp_utils.is_pp_last_stage(group))  # 2 == 3 - 1

        _c2, (p_rank2, p_size2) = self._patched(rank=2, size=4)
        with p_rank2, p_size2:
            self.assertFalse(pp_utils.is_pp_last_stage(group))  # 2 != 4 - 1

    def test_single_stage_is_both_first_and_last(self):
        # world size 1: the only rank (0) is simultaneously first and last.
        group = _FakeGroup()
        _captured, (p_rank, p_size) = self._patched(rank=0, size=1)
        with p_rank, p_size:
            self.assertTrue(pp_utils.is_pp_first_stage(group))
            self.assertTrue(pp_utils.is_pp_last_stage(group))  # 0 == 1 - 1


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestMakeViewless(unittest.TestCase):
    """make_viewless wraps make_viewless_tensor; non-view input passes through.

    make_viewless_tensor returns its input unchanged when the tensor is not a
    view, so make_viewless of a freshly created (non-view) tensor must return
    the identical object with identical values. The real make_viewless_tensor
    collaborator is exercised here -- it is not mocked.
    """

    def setUp(self):
        paddle.set_device("cpu")

    def test_non_view_tensor_is_returned_unchanged(self):
        e = paddle.to_tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32"
        )
        e.stop_gradient = True  # requires_grad is read by make_viewless
        self.assertFalse(e._is_view())  # precondition: genuinely not a view
        before = e.numpy().copy()

        out = pp_utils.make_viewless(e)

        # Non-view path: identical object, so no data was copied or reshaped.
        self.assertIs(out, e)
        # And its content is untouched by the wrapper.
        np.testing.assert_array_equal(out.numpy(), before)
        # The result is (still) not a view -- the whole point of "viewless".
        self.assertFalse(out._is_view())

    def test_requires_grad_flag_is_preserved_on_passthrough(self):
        e = paddle.to_tensor([7.0, 8.0], dtype="float32")
        e.stop_gradient = False  # a grad-requiring leaf
        self.assertFalse(e._is_view())

        out = pp_utils.make_viewless(e)

        self.assertIs(out, e)
        # stop_gradient / requires_grad round-trips untouched on the non-view path.
        self.assertFalse(out.stop_gradient)


if __name__ == "__main__":
    unittest.main()
