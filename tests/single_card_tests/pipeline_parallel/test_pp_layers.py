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

"""CPU-only behavior tests for the device-independent pipeline-parallel
stage/rank-mapping logic in ``paddlefleet.pipeline_parallel.utils``.

Scope covered here (pure, non-communicating logic):
- ``is_vp_first_stage`` / ``is_vp_last_stage``: virtual-pipeline stage boundary
  rules, including the ``vp_size <= 1`` collapse and its guard assertion.
- ``is_pp_first_stage`` / ``is_pp_last_stage``: first/last stage predicates.
- ``get_pp_first_rank`` / ``get_pp_last_rank``: endpoint global-rank selection
  from the group's rank list.
- ``get_pp_next_rank`` / ``get_pp_prev_rank``: neighbour global-rank selection
  and the None-at-boundary contract.

Every expected value is derived by hand from the boundary rule and plain index
arithmetic, never from the production output. ``get_pg_rank`` / ``get_pg_size``
are genuine not-under-test collaborators (they read a real process group's rank
and world size); here they are patched to inject a chosen topology so the pure
index/boundary logic can run on CPU, and the tests assert the exact resulting
global rank -- not merely that they were called. The process group itself is a
lightweight stand-in exposing only ``ranks()``.

No real process group or collective communication is executed. Anything whose
correctness depends on cross-rank exchange (actual P2P send/recv, meta exchange,
scheduling) is intentionally out of scope: faking it here would prove nothing
about multi-rank semantics.
"""

import unittest
from unittest import mock

try:
    from paddlefleet.pipeline_parallel import utils as pputils

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # honest dependency probe
    pputils = None
    _IMPORT_ERROR = exc

_HAS_DEPS = _IMPORT_ERROR is None
_SKIP_REASON = (
    "paddle / paddlefleet.pipeline_parallel.utils not importable on this CPU "
    f"host: {_IMPORT_ERROR!r}"
)


class _FakeGroup:
    """Minimal stand-in for a pipeline process group.

    ``paddlefleet.pipeline_parallel.utils`` only ever calls ``pp_group.ranks()``
    on the group, so a plain object exposing that method is sufficient to drive
    the endpoint/neighbour selection logic without a real process group.
    """

    def __init__(self, rank_list):
        self._rank_list = list(rank_list)

    def ranks(self):
        return self._rank_list


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestVpStageBoundaries(unittest.TestCase):
    """``is_vp_first_stage`` / ``is_vp_last_stage`` boundary rules.

    Contract: when ``vp_size`` is None or ``<= 1`` the pipeline is not
    interleaved, so both predicates return True and ``vp_stage`` must be None or
    0 (else an AssertionError). Otherwise first == (vp_stage == 0) and
    last == (vp_stage == vp_size - 1).
    """

    def test_interleaved_first_stage(self):
        # vp_size=4: only vp_stage 0 is the first virtual stage.
        self.assertTrue(pputils.is_vp_first_stage(0, 4))
        self.assertFalse(pputils.is_vp_first_stage(1, 4))
        self.assertFalse(pputils.is_vp_first_stage(3, 4))

    def test_interleaved_last_stage(self):
        # vp_size=4: only vp_stage 3 (== vp_size - 1) is the last virtual stage.
        self.assertTrue(pputils.is_vp_last_stage(3, 4))
        self.assertFalse(pputils.is_vp_last_stage(0, 4))
        self.assertFalse(pputils.is_vp_last_stage(2, 4))

    def test_first_and_last_are_distinct_middle(self):
        # A middle stage is neither first nor last; guards against the two
        # predicates being swapped or aliased.
        self.assertFalse(pputils.is_vp_first_stage(2, 4))
        self.assertFalse(pputils.is_vp_last_stage(2, 4))
        # The two ends are mirror images of each other.
        self.assertTrue(pputils.is_vp_first_stage(0, 4))
        self.assertTrue(pputils.is_vp_last_stage(3, 4))

    def test_non_interleaved_collapses_to_true(self):
        # vp_size None or <= 1: the single stage is both first and last.
        for vp_size in (None, 1):
            self.assertTrue(pputils.is_vp_first_stage(0, vp_size))
            self.assertTrue(pputils.is_vp_last_stage(0, vp_size))
            self.assertTrue(pputils.is_vp_first_stage(None, vp_size))
            self.assertTrue(pputils.is_vp_last_stage(None, vp_size))

    def test_non_interleaved_rejects_nonzero_stage(self):
        # vp_size <= 1 with a non-zero vp_stage violates the documented guard.
        for func in (pputils.is_vp_first_stage, pputils.is_vp_last_stage):
            with self.assertRaises(AssertionError):
                func(2, 1)
            with self.assertRaises(AssertionError):
                func(1, None)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestPpEndpointRanks(unittest.TestCase):
    """``get_pp_first_rank`` / ``get_pp_last_rank`` endpoint selection.

    The group holds global ranks [10, 11, 12, 13]; using values distinct from
    their positions ensures the functions return the rank *value*, not an index.
    """

    def test_first_and_last_endpoint_rank(self):
        group = _FakeGroup([10, 11, 12, 13])
        self.assertEqual(pputils.get_pp_first_rank(group), 10)
        self.assertEqual(pputils.get_pp_last_rank(group), 13)

    def test_single_rank_group_endpoints_coincide(self):
        group = _FakeGroup([7])
        self.assertEqual(pputils.get_pp_first_rank(group), 7)
        self.assertEqual(pputils.get_pp_last_rank(group), 7)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestPpStagePredicates(unittest.TestCase):
    """``is_pp_first_stage`` / ``is_pp_last_stage`` with an injected topology.

    ``get_pg_rank`` / ``get_pg_size`` are patched to supply the in-group rank and
    world size; the predicates' own comparison logic is what is exercised.
    """

    def test_first_stage_only_at_rank_zero(self):
        group = _FakeGroup([10, 11, 12, 13])
        with mock.patch.object(pputils, "get_pg_rank", return_value=0):
            self.assertTrue(pputils.is_pp_first_stage(group))
        with mock.patch.object(pputils, "get_pg_rank", return_value=2):
            self.assertFalse(pputils.is_pp_first_stage(group))

    def test_last_stage_only_at_final_rank(self):
        group = _FakeGroup([10, 11, 12, 13])
        with (
            mock.patch.object(pputils, "get_pg_rank", return_value=3),
            mock.patch.object(pputils, "get_pg_size", return_value=4),
        ):
            self.assertTrue(pputils.is_pp_last_stage(group))
        with (
            mock.patch.object(pputils, "get_pg_rank", return_value=1),
            mock.patch.object(pputils, "get_pg_size", return_value=4),
        ):
            self.assertFalse(pputils.is_pp_last_stage(group))


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestPpNeighbourRanks(unittest.TestCase):
    """``get_pp_next_rank`` / ``get_pp_prev_rank`` neighbour selection.

    Group global ranks are [10, 11, 12, 13]. The neighbour is the global rank
    adjacent to the current in-group position; the first stage has no previous
    and the last stage has no next.
    """

    def test_next_rank_is_following_global_rank(self):
        group = _FakeGroup([10, 11, 12, 13])
        # In-group rank 1 of world size 4 is not the last stage; the next
        # global rank is ranks[2] == 12.
        with (
            mock.patch.object(pputils, "get_pg_rank", return_value=1),
            mock.patch.object(pputils, "get_pg_size", return_value=4),
        ):
            self.assertEqual(pputils.get_pp_next_rank(group), 12)

    def test_next_rank_none_on_last_stage(self):
        group = _FakeGroup([10, 11, 12, 13])
        with (
            mock.patch.object(pputils, "get_pg_rank", return_value=3),
            mock.patch.object(pputils, "get_pg_size", return_value=4),
        ):
            self.assertIsNone(pputils.get_pp_next_rank(group))

    def test_prev_rank_is_preceding_global_rank(self):
        group = _FakeGroup([10, 11, 12, 13])
        # In-group rank 2 is not the first stage; the previous global rank is
        # ranks[1] == 11.
        with (
            mock.patch.object(pputils, "get_pg_rank", return_value=2),
            mock.patch.object(pputils, "get_pg_size", return_value=4),
        ):
            self.assertEqual(pputils.get_pp_prev_rank(group), 11)

    def test_prev_rank_none_on_first_stage(self):
        group = _FakeGroup([10, 11, 12, 13])
        with (
            mock.patch.object(pputils, "get_pg_rank", return_value=0),
            mock.patch.object(pputils, "get_pg_size", return_value=4),
        ):
            self.assertIsNone(pputils.get_pp_prev_rank(group))


if __name__ == "__main__":
    unittest.main()
