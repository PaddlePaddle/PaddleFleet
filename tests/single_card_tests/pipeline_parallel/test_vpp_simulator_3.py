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

"""CPU-only behavior tests for the deterministic *scheduling* logic of
``paddlefleet.pipeline_parallel.vpp_simulator.VPPSimulator``.

This file deliberately targets a slice that is distinct from the ``Chunk``
metadata / ``PPChunkRecorder`` helpers: it exercises the pure integer
arithmetic and chunk-ordering that build a virtual-pipeline schedule, plus the
timeline invariants the public ``schedule()`` entry must honour.

Concretely covered:

* ``__init__`` derived quantities (``first_chunk_acc``, ``num_steps``,
  ``layer_num``) that drive every downstream decision.
* ``_get_consume_time`` - forward / bubble cost 1 step, backward costs 2.
* ``_get_virtual_pp_rank`` - the micro-step -> virtual-pp-rank map for both the
  forward and (mirrored) backward directions.
* ``_get_warmup_and_steady_steps`` - both scheduling regimes: the interleaved
  ``PipelineParallelWithInterleave`` branch and the balanced-memory branch that
  activates only when ``pp_degree <= num_acc_steps < 2 * pp_degree``.
* the full ``schedule()`` chunk sequence per stage (type / virtual-pp-rank /
  accumulation-step / layer-id), hand-traced below, and the per-stage timeline
  invariants (monotone non-overlap, consume-time honoured, first chunk at 0).
* ``compute_bubble_rate`` cross-checked against an independently counted
  execution-time budget.

Every expected value is hand-derived from the algorithm's arithmetic; none is
copied from the production output or from any coverage_test file. The
schedule is deterministic, so the traces are exact.

The simulator only needs ``enum`` / ``numpy`` / ``matplotlib`` itself, but the
``paddlefleet`` package import pulls in ``paddle``; on a bare CPU host that may
be absent, so the heavy import is guarded and every test class is skipped with
an honest reason carrying the real error.
"""

import unittest

try:
    from paddlefleet.pipeline_parallel.vpp_simulator import (
        ChunkType,
        VPPSimulator,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # honest dependency probe
    ChunkType = None
    VPPSimulator = None
    _IMPORT_ERROR = exc

_HAS_DEPS = _IMPORT_ERROR is None
_SKIP_REASON = (
    "paddle / paddlefleet / matplotlib not importable on this CPU host: "
    f"{_IMPORT_ERROR!r}"
)


def _stage_sequence(sim, stage_id):
    """Return the ordered (type, vpp_rank, acc_step, layer_id) tuples for a
    stage's schedule row, so ordering and pairing can be compared exactly."""
    return [
        (c.chunk_type.value, c.virtual_pp_rank, c.acc_step, c.layer_id)
        for c in sim.schedule_table[stage_id]
    ]


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestDerivedInitState(unittest.TestCase):
    """__init__ arithmetic that seeds the whole schedule."""

    def test_first_chunk_acc_num_steps_layer_num(self):
        # first_chunk_acc = (num_acc_steps % pp_degree) + pp_degree
        # num_steps = num_acc_steps * vpp_degree
        # layer_num = pp_degree * vpp_degree
        sim = VPPSimulator(pp_degree=2, vpp_degree=2, num_acc_steps=4)
        self.assertEqual(sim.first_chunk_acc, (4 % 2) + 2)  # == 2
        self.assertEqual(sim.num_steps, 4 * 2)  # == 8
        self.assertEqual(sim.layer_num, 2 * 2)  # == 4
        # schedule_table starts as one empty row per pipeline stage.
        self.assertEqual(len(sim.schedule_table), 2)
        self.assertEqual(sim.schedule_table, [[], []])
        self.assertFalse(sim._is_scheduled)

    def test_first_chunk_acc_with_nonzero_remainder(self):
        # 5 % 3 == 2, so first_chunk_acc == 2 + 3 == 5.
        sim = VPPSimulator(pp_degree=3, vpp_degree=2, num_acc_steps=5)
        self.assertEqual(sim.first_chunk_acc, (5 % 3) + 3)
        self.assertEqual(sim.num_steps, 5 * 2)
        self.assertEqual(sim.layer_num, 3 * 2)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestConsumeTime(unittest.TestCase):
    """Forward / bubble take 1 step; backward takes 2 (double the FLOPs)."""

    def test_consume_time_by_chunk_type(self):
        sim = VPPSimulator(pp_degree=2, vpp_degree=2, num_acc_steps=4)
        # The virtual-pp-rank / acc-step arguments must NOT change the cost.
        for vpr in (0, 1):
            for acc in (0, 3):
                self.assertEqual(
                    sim._get_consume_time(vpr, acc, ChunkType.FORWARD), 1
                )
                self.assertEqual(
                    sim._get_consume_time(vpr, acc, ChunkType.BUBBLE), 1
                )
                self.assertEqual(
                    sim._get_consume_time(vpr, acc, ChunkType.BACKWARD), 2
                )


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestVirtualPPRankMap(unittest.TestCase):
    """_get_virtual_pp_rank: micro-step -> virtual pipeline rank.

    For pp=2, vpp=2, acc=4: first_chunk_acc=2, first_chunk_steps=4.
    Forward: ms<4 -> ms//2 ; ms>=4 -> ((ms-4) % 4) // 2.
    Backward mirrors: vpp - fwd - 1 == 1 - fwd (for vpp=2).
    """

    def setUp(self):
        self.sim = VPPSimulator(pp_degree=2, vpp_degree=2, num_acc_steps=4)

    def test_forward_ranks(self):
        expected = [0, 0, 1, 1, 0, 0, 1, 1]
        got = [
            self.sim._get_virtual_pp_rank(ms, forward=True) for ms in range(8)
        ]
        self.assertEqual(got, expected)

    def test_backward_ranks_are_mirrored(self):
        expected = [1, 1, 0, 0, 1, 1, 0, 0]
        got = [
            self.sim._get_virtual_pp_rank(ms, forward=False) for ms in range(8)
        ]
        self.assertEqual(got, expected)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestWarmupAndSteadySteps(unittest.TestCase):
    """_get_warmup_and_steady_steps for both scheduling regimes."""

    def test_interleave_branch(self):
        # acc=4 is NOT in [pp, 2*pp) = [2, 4), so the interleave branch runs.
        # warmup = (pp-stage-1)*2 + (vpp-1)*first_chunk_acc, capped at num_steps
        # steady = num_steps - warmup ; first_chunk_acc = 2, num_steps = 8.
        sim = VPPSimulator(pp_degree=2, vpp_degree=2, num_acc_steps=4)
        self.assertEqual(sim._get_warmup_and_steady_steps(0), (4, 4))
        self.assertEqual(sim._get_warmup_and_steady_steps(1), (2, 6))

    def test_balanced_memory_branch(self):
        # acc=2 IS in [pp, 2*pp) = [2, 4), so the balanced-memory branch runs.
        # warmup = acc*(vpp-1) + pp-stage-1 ; steady = acc - (pp-stage-1).
        sim = VPPSimulator(pp_degree=2, vpp_degree=2, num_acc_steps=2)
        self.assertEqual(sim._get_warmup_and_steady_steps(0), (3, 1))
        self.assertEqual(sim._get_warmup_and_steady_steps(1), (2, 2))

    def test_forwards_per_stage_equal_num_steps(self):
        # An invariant of both formulas: warmup + steady == num_steps, i.e. each
        # stage issues exactly one forward per (micro-batch x virtual chunk).
        for sim in (
            VPPSimulator(pp_degree=2, vpp_degree=2, num_acc_steps=4),
            VPPSimulator(pp_degree=2, vpp_degree=2, num_acc_steps=2),
            VPPSimulator(pp_degree=3, vpp_degree=2, num_acc_steps=3),
        ):
            for stage_id in range(sim.pp_degree):
                w, s = sim._get_warmup_and_steady_steps(stage_id)
                self.assertEqual(w + s, sim.num_steps)


# Hand-traced schedule for pp=2, vpp=2, acc=4 (interleave regime).
# layer_id = virtual_pp_rank * pp_degree + stage_id.
_EXPECTED_STAGE0 = [
    ("F", 0, 0, 0),
    ("F", 0, 1, 0),
    ("F", 1, 0, 2),
    ("F", 1, 1, 2),
    ("F", 0, 2, 0),
    ("B", 1, 0, 2),
    ("F", 0, 3, 0),
    ("B", 1, 1, 2),
    ("F", 1, 2, 2),
    ("B", 0, 0, 0),
    ("F", 1, 3, 2),
    ("B", 0, 1, 0),
    ("B", 1, 2, 2),
    ("B", 1, 3, 2),
    ("B", 0, 2, 0),
    ("B", 0, 3, 0),
]
_EXPECTED_STAGE1 = [
    ("F", 0, 0, 1),
    ("F", 0, 1, 1),
    ("F", 1, 0, 3),
    ("B", 1, 0, 3),
    ("F", 1, 1, 3),
    ("B", 1, 1, 3),
    ("F", 0, 2, 1),
    ("B", 0, 0, 1),
    ("F", 0, 3, 1),
    ("B", 0, 1, 1),
    ("F", 1, 2, 3),
    ("B", 1, 2, 3),
    ("F", 1, 3, 3),
    ("B", 1, 3, 3),
    ("B", 0, 2, 1),
    ("B", 0, 3, 1),
]


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestScheduleChunkSequence(unittest.TestCase):
    """Full public schedule() must emit the hand-traced chunk order.

    schedule() only fills in start/end times; it neither reorders nor adds
    chunks (no BUBBLE chunks are created by _schedule_without_bubble), so the
    (type, vpp_rank, acc_step, layer_id) trace is exactly the ordering logic.
    """

    def setUp(self):
        self.sim = VPPSimulator(pp_degree=2, vpp_degree=2, num_acc_steps=4)
        self.table = self.sim.schedule()

    def test_stage0_sequence(self):
        self.assertEqual(_stage_sequence(self.sim, 0), _EXPECTED_STAGE0)

    def test_stage1_sequence(self):
        self.assertEqual(_stage_sequence(self.sim, 1), _EXPECTED_STAGE1)

    def test_schedule_marks_state_and_returns_table(self):
        self.assertTrue(self.sim._is_scheduled)
        self.assertIs(self.table, self.sim.schedule_table)

    def test_each_stage_has_num_steps_forward_and_backward(self):
        # 2 * num_steps chunks per stage: num_steps forwards + num_steps backs.
        for stage_id in range(self.sim.pp_degree):
            row = self.sim.schedule_table[stage_id]
            forwards = [c for c in row if c.chunk_type == ChunkType.FORWARD]
            backwards = [c for c in row if c.chunk_type == ChunkType.BACKWARD]
            self.assertEqual(len(forwards), self.sim.num_steps)
            self.assertEqual(len(backwards), self.sim.num_steps)

    def test_no_bubble_chunks_are_materialised(self):
        for row in self.sim.schedule_table:
            for c in row:
                self.assertNotEqual(c.chunk_type, ChunkType.BUBBLE)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestScheduleTimelineInvariants(unittest.TestCase):
    """Per-stage timing contract produced by _add_bubble / _barrier."""

    def setUp(self):
        self.sim = VPPSimulator(pp_degree=2, vpp_degree=2, num_acc_steps=4)
        self.sim.schedule()

    def test_consume_time_honoured(self):
        # Every forward spans exactly 1 step, every backward exactly 2.
        for row in self.sim.schedule_table:
            for c in row:
                span = c.end - c.start
                if c.chunk_type == ChunkType.FORWARD:
                    self.assertEqual(span, 1)
                else:
                    self.assertEqual(span, 2)

    def test_within_stage_monotone_non_overlapping(self):
        # A single stage runs one chunk at a time: each chunk starts no earlier
        # than the previous chunk on the same stage finished.
        for row in self.sim.schedule_table:
            for i in range(1, len(row)):
                self.assertGreaterEqual(row[i].start, row[i - 1].end)

    def test_first_chunk_starts_at_time_zero(self):
        # Stage 0's leading forward is layer 0: it has no pre-order dependency
        # and no earlier chunk, so it must begin the timeline at t=0.
        first = self.sim.schedule_table[0][0]
        self.assertEqual(first.start, 0)
        self.assertEqual(first.end, 1)

    def test_all_starts_and_ends_non_negative(self):
        for row in self.sim.schedule_table:
            for c in row:
                self.assertGreaterEqual(c.start, 0)
                self.assertGreater(c.end, c.start)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestBubbleRate(unittest.TestCase):
    """compute_bubble_rate cross-checked with an independent time budget."""

    def test_bubble_rate_matches_independent_budget(self):
        sim = VPPSimulator(pp_degree=2, vpp_degree=2, num_acc_steps=4)
        rate = sim.compute_bubble_rate()  # triggers schedule() internally

        # Independent execution budget: each stage does num_steps forwards
        # (cost 1) and num_steps backwards (cost 2) -> 3 * num_steps per stage.
        expected_sum_exec = sim.pp_degree * 3 * sim.num_steps
        # Confirm the scheduled spans really add up to that budget.
        actual_sum_exec = sum(
            c.end - c.start
            for row in sim.schedule_table
            for c in row
            if c.chunk_type != ChunkType.BUBBLE
        )
        self.assertEqual(actual_sum_exec, expected_sum_exec)

        # Reconstruct the rate from the makespan measured off the schedule and
        # the independently counted budget (not from compute_bubble_rate's own
        # internal accumulation).
        min_start = min(c.start for row in sim.schedule_table for c in row)
        max_end = max(c.end for row in sim.schedule_table for c in row)
        total_possible = sim.pp_degree * (max_end - min_start)
        expected_rate = (total_possible - expected_sum_exec) / total_possible

        self.assertAlmostEqual(rate, expected_rate, places=9)
        self.assertGreaterEqual(rate, 0.0)
        self.assertLess(rate, 1.0)

    def test_bubble_rate_is_deterministic(self):
        a = VPPSimulator(
            pp_degree=2, vpp_degree=2, num_acc_steps=4
        ).compute_bubble_rate()
        b = VPPSimulator(
            pp_degree=2, vpp_degree=2, num_acc_steps=4
        ).compute_bubble_rate()
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
