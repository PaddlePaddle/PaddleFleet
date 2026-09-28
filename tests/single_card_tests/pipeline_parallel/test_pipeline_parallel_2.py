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

Covered here are only the non-communicating helpers whose correctness is a
matter of local branch selection / index arithmetic:

* ``is_vp_first_stage`` / ``is_vp_last_stage`` - virtual-pipeline stage
  classification and the assertion contract for degenerate ``vp_size``.
* ``get_pp_first_rank`` / ``get_pp_last_rank`` - endpoint selection from the
  group's global-rank list.
* ``get_pp_next_rank`` / ``get_pp_prev_rank`` - the *global rank* of the
  neighbouring pipeline stage. This is pure arithmetic on the rank list and
  the current in-group rank; it performs NO collective communication. The
  rank/size readers (``get_pg_rank`` / ``get_pg_size``) are genuine external
  collaborators that report the runtime's placement, so they are stubbed to
  position this process at a known stage while the real neighbour-selection
  logic runs. Real cross-rank send/recv is NOT exercised by this file.

Expected values are hand-derived from the documented stage semantics and are
never taken from the production functions' own output.
"""

import unittest
from unittest import mock

try:
    from paddlefleet.pipeline_parallel import utils as pp_utils
    from paddlefleet.pipeline_parallel.utils import (
        get_pp_first_rank,
        get_pp_last_rank,
        get_pp_next_rank,
        get_pp_prev_rank,
        is_vp_first_stage,
        is_vp_last_stage,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # honest dependency probe
    pp_utils = None
    get_pp_first_rank = None
    get_pp_last_rank = None
    get_pp_next_rank = None
    get_pp_prev_rank = None
    is_vp_first_stage = None
    is_vp_last_stage = None
    _IMPORT_ERROR = exc

_HAS_DEPS = _IMPORT_ERROR is None
_SKIP_REASON = (
    f"paddle / paddlefleet not importable on this CPU host: {_IMPORT_ERROR!r}"
)


class _FakePPGroup:
    """Deterministic stand-in for a pipeline process group.

    Only ``ranks()`` is consulted by the endpoint / neighbour helpers; it
    returns the ordered list of *global* ranks that make up this PP group.
    """

    def __init__(self, ranks):
        self._ranks = list(ranks)

    def ranks(self):
        return list(self._ranks)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestVirtualPipelineStageClassification(unittest.TestCase):
    """is_vp_first_stage / is_vp_last_stage over a hand-built truth table."""

    def test_no_virtual_pipeline_is_both_first_and_last(self):
        # vp_size None or <= 1 means "no virtual pipeline": the sole/absent VP
        # stage is trivially first and last. vp_stage must then be None or 0.
        for vp_size in (None, 0, 1):
            for vp_stage in (None, 0):
                self.assertTrue(is_vp_first_stage(vp_stage, vp_size))
                self.assertTrue(is_vp_last_stage(vp_stage, vp_size))

    def test_multi_stage_first_only_at_stage_zero(self):
        # vp_size == 4 -> first iff stage == 0, last iff stage == 3.
        expected_first = {0: True, 1: False, 2: False, 3: False}
        expected_last = {0: False, 1: False, 2: False, 3: True}
        for stage in range(4):
            self.assertEqual(is_vp_first_stage(stage, 4), expected_first[stage])
            self.assertEqual(is_vp_last_stage(stage, 4), expected_last[stage])

    def test_two_stage_boundaries(self):
        self.assertTrue(is_vp_first_stage(0, 2))
        self.assertFalse(is_vp_first_stage(1, 2))
        self.assertFalse(is_vp_last_stage(0, 2))
        self.assertTrue(is_vp_last_stage(1, 2))

    def test_degenerate_vp_size_rejects_nonzero_stage(self):
        # When there is no virtual pipeline, a non-zero stage index is an
        # illegal caller mistake and must trip the guard, not silently pass.
        for fn in (is_vp_first_stage, is_vp_last_stage):
            with self.assertRaises(AssertionError):
                fn(2, 1)
            with self.assertRaises(AssertionError):
                fn(3, None)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestPipelineRankEndpoints(unittest.TestCase):
    """get_pp_first_rank / get_pp_last_rank pick the list endpoints."""

    def test_endpoints_use_global_rank_list_order(self):
        group = _FakePPGroup([4, 5, 6, 7])
        # Endpoints are read straight from the ordered global-rank list; the
        # list is intentionally not [0..n) so a wrong index can be caught.
        self.assertEqual(get_pp_first_rank(group), 4)
        self.assertEqual(get_pp_last_rank(group), 7)

    def test_single_rank_group_endpoints_coincide(self):
        group = _FakePPGroup([9])
        self.assertEqual(get_pp_first_rank(group), 9)
        self.assertEqual(get_pp_last_rank(group), 9)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestPipelineNeighbourRankSelection(unittest.TestCase):
    """get_pp_next_rank / get_pp_prev_rank map an in-group position to the
    neighbouring stage's *global* rank (pure arithmetic, no collective)."""

    def _run_at(self, ranks, in_group_rank):
        group = _FakePPGroup(ranks)
        with (
            mock.patch.object(
                pp_utils, "get_pg_rank", return_value=in_group_rank
            ),
            mock.patch.object(pp_utils, "get_pg_size", return_value=len(ranks)),
        ):
            return get_pp_next_rank(group), get_pp_prev_rank(group)

    def test_neighbours_across_all_positions(self):
        ranks = [4, 5, 6, 7]
        # Hand-derived: first stage has no prev, last stage has no next; the
        # interior stages step one entry along the global-rank list.
        expected = {
            0: (5, None),
            1: (6, 4),
            2: (7, 5),
            3: (None, 6),
        }
        for pos, (exp_next, exp_prev) in expected.items():
            nxt, prev = self._run_at(ranks, pos)
            self.assertEqual(nxt, exp_next, f"next rank wrong at stage {pos}")
            self.assertEqual(prev, exp_prev, f"prev rank wrong at stage {pos}")

    def test_single_rank_group_has_no_neighbours(self):
        # world-size-1 local path: the sole stage is both first and last.
        nxt, prev = self._run_at([9], 0)
        self.assertIsNone(nxt)
        self.assertIsNone(prev)


if __name__ == "__main__":
    unittest.main()
