# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""grad_reduce: which process groups may be summed, and the check that proves it.

Regression home for the ``cp>1`` bug found in end-to-end review: on a
``sharding_first`` topology with ``data_parallel_size=1`` the samples live on the
sharding dimension, so the data-parallel group spans ``cp``. Chaining a ``cp``
reduction into a ``dp`` reduction then counted the context-parallel contribution
twice and ``norm_global`` came out ``sqrt(cp)`` high with nothing in the numbers
to show it.

No cluster needed: the planner works off group sizes and rank sets, so stubs pin
the combination logic exactly.
"""

import importlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

gr = importlib.import_module(
    "paddlefleet.internal_medicine.backends.paddlefleet.grad_reduce"
)


def info(name, ranks):
    return gr.GroupInfo(name, object(), len(ranks), ranks)


class OrthogonalityTest(unittest.TestCase):
    def test_groups_sharing_only_this_rank_are_orthogonal(self):
        rows = info("dp", [0, 8, 16, 24])
        cols = info("cp", [0, 1])
        self.assertTrue(gr.is_orthogonal(cols, [rows]))

    def test_a_group_nested_in_a_chosen_one_is_rejected(self):
        """The bug: dp spans cp, so summing both double counts cp."""
        spanning_dp = info("dp", list(range(32)))
        cp = info("cp", [0, 1])
        self.assertFalse(gr.is_orthogonal(cp, [spanning_dp]))

    def test_a_group_without_membership_is_treated_as_unsafe(self):
        """An unverifiable layout must degrade to a partial cover, not a wrong sum."""
        self.assertFalse(gr.is_orthogonal(info("dp", []), []))


class PlanTest(unittest.TestCase):
    def test_the_reviewed_sharding_first_layout_drops_cp(self):
        """world=32, cp=2, dp spans all 32 ranks -> dp alone, no cp factor."""
        dp = info("dp", list(range(32)))
        cp = info("cp", [0, 1])
        chosen, product = gr.plan_total_groups([dp, cp], target=32)
        self.assertEqual([g.name for g in chosen], ["dp"])
        self.assertEqual(product, 32)

    def test_a_genuinely_orthogonal_layout_uses_both(self):
        dp = info("dp", [0, 2, 4, 6, 8, 10, 12, 14])
        cp = info("cp", [0, 1])
        chosen, product = gr.plan_total_groups([dp, cp], target=16)
        self.assertEqual(sorted(g.name for g in chosen), ["cp", "dp"])
        self.assertEqual(product, 16)

    def test_largest_first_so_a_full_span_wins_over_two_partial_ones(self):
        """Order matters: taking cp first would leave dp unable to fit."""
        dp = info("dp", list(range(16)))
        cp = info("cp", [0, 1])
        ep = info("ep", [0, 4, 8, 12])
        chosen, product = gr.plan_total_groups([cp, ep, dp], target=16)
        self.assertEqual([g.name for g in chosen], ["dp"])
        self.assertEqual(product, 16)

    def test_a_partial_cover_is_reported_rather_than_padded(self):
        """The caller suppresses norm_global on this; it must not silently pass."""
        cp = info("cp", [0, 1])
        chosen, product = gr.plan_total_groups([cp], target=32)
        self.assertEqual([g.name for g in chosen], ["cp"])
        self.assertLess(product, 32)

    def test_singleton_groups_are_skipped(self):
        chosen, product = gr.plan_total_groups(
            [info("dp", [0]), info("cp", [0])], target=1
        )
        self.assertEqual(chosen, [])
        self.assertEqual(product, 1)

    def test_nothing_oversteps_the_target(self):
        dp = info("dp", list(range(8)))
        chosen, product = gr.plan_total_groups([dp], target=4)
        self.assertEqual(chosen, [])
        self.assertEqual(product, 1)


class DescribeTest(unittest.TestCase):
    def test_the_plan_is_readable_in_a_log_line(self):
        self.assertEqual(
            gr.describe([info("dp", [0, 1]), info("cp", [0, 2])]), "dp=2 x cp=2"
        )
        self.assertEqual(gr.describe([]), "nothing")


if __name__ == "__main__":
    unittest.main()
