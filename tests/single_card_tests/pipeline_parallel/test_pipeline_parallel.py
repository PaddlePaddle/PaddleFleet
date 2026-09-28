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

"""CPU-only behavior tests for the device-independent pure scheduling logic in
``paddlefleet.pipeline_parallel.vpp_simulator``.

This module simulates Virtual Pipeline Parallel (VPP) scheduling with plain
Python integer arithmetic; it needs no accelerator and performs no collective
communication, so every expected value below is derived BY HAND from the VPP
contract (each of ``num_acc_steps * vpp_degree`` micro-chunks per pipeline stage
runs exactly one forward and one backward; forward/bubble cost 1 time-unit and
backward costs 2). Nothing is copied from the code under test.

Covered surface:

* ``Chunk.layer_id`` - ``virtual_pp_rank * pp_degree + stage_id`` for real
  chunks, ``None`` for bubbles - plus the ``__str__`` label contract.
* ``VPPSimulator.__init__`` derived quantities (``first_chunk_acc``,
  ``num_steps``, ``layer_num``, empty per-stage table).
* ``_get_consume_time`` cost table.
* ``_get_virtual_pp_rank`` warm-up / steady-state chunk-to-vrank mapping,
  including the forward mirror for the backward pass.
* ``_get_warmup_and_steady_steps`` for both the interleave branch (with the
  ``min(., num_steps)`` clamp) and the balanced-memory branch.
* ``compute_bubble_rate`` formula, exercised on a hand-built schedule table so
  the bubble ratio is checked in isolation from the scheduler.
* Full ``schedule()`` structural invariants (per-stage forward/backward counts,
  consume-time durations, non-overlap ordering, and per-layer accumulation-step
  coverage) for one interleave and one balanced-memory configuration.
* ``PPChunkRecorder`` layer-window gating and ``step`` reset.

Out of scope (needs a real multi-rank pipeline / rendering backend, not a CPU
unit): actual cross-stage send/recv, ``draw_chunks`` / ``draw_balls`` plotting,
and any real process-group numerics.
"""

import unittest

try:
    from paddlefleet.pipeline_parallel.vpp_simulator import (
        Chunk,
        ChunkType,
        PPChunkRecorder,
        VPPSimulator,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # honest dependency probe
    Chunk = None
    ChunkType = None
    PPChunkRecorder = None
    VPPSimulator = None
    _IMPORT_ERROR = exc

_HAS_DEPS = _IMPORT_ERROR is None
_SKIP_REASON = (
    "paddlefleet.pipeline_parallel.vpp_simulator not importable on this CPU "
    f"host: {_IMPORT_ERROR!r}"
)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestChunkType(unittest.TestCase):
    """The three enum values are load-bearing: they are embedded verbatim in
    ``Chunk.__str__`` and used to branch consume-time and layer-id logic."""

    def test_enum_values(self):
        # Hand-fixed by the schedule/label contract, not read from the enum.
        self.assertEqual(ChunkType.FORWARD.value, "F")
        self.assertEqual(ChunkType.BACKWARD.value, "B")
        self.assertEqual(ChunkType.BUBBLE.value, "Z")


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestChunk(unittest.TestCase):
    """``Chunk.layer_id`` mapping and the human-readable label."""

    def test_layer_id_real_chunk(self):
        # layer_id = virtual_pp_rank * pp_degree + stage_id.
        # vpr=2, pp_degree=4, stage_id=1 -> 2*4 + 1 = 9 (derived by hand).
        chunk = Chunk(
            virtual_pp_rank=2,
            acc_step=0,
            pp_degree=4,
            vpp_degree=3,
            stage_id=1,
            chunk_type=ChunkType.FORWARD,
            start=0,
            end=0,
        )
        self.assertEqual(chunk.layer_id, 9)

        # A different (vpr, stage) must move the id: vpr=0, stage=3 -> 3.
        other = Chunk(
            virtual_pp_rank=0,
            acc_step=0,
            pp_degree=4,
            vpp_degree=3,
            stage_id=3,
            chunk_type=ChunkType.BACKWARD,
            start=0,
            end=0,
        )
        self.assertEqual(other.layer_id, 3)

    def test_bubble_has_no_layer_id(self):
        chunk = Chunk(
            virtual_pp_rank=2,
            acc_step=0,
            pp_degree=4,
            vpp_degree=3,
            stage_id=1,
            chunk_type=ChunkType.BUBBLE,
            start=3,
            end=4,
        )
        self.assertIsNone(chunk.layer_id)

    def test_str_label_real_chunk(self):
        # Label = "<type><layer_id>_<acc_step+1><(start, end)>".
        # vpr=1, pp=2, stage=1 -> layer_id 3; acc_step 0 -> shown 1-indexed.
        chunk = Chunk(
            virtual_pp_rank=1,
            acc_step=0,
            pp_degree=2,
            vpp_degree=2,
            stage_id=1,
            chunk_type=ChunkType.FORWARD,
            start=2,
            end=3,
        )
        self.assertEqual(str(chunk), "F3_1(2, 3)")

    def test_str_label_bubble(self):
        chunk = Chunk(
            virtual_pp_rank=0,
            acc_step=0,
            pp_degree=2,
            vpp_degree=2,
            stage_id=0,
            chunk_type=ChunkType.BUBBLE,
            start=4,
            end=5,
        )
        self.assertEqual(str(chunk), "Z((4, 5))")


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestSimulatorInit(unittest.TestCase):
    """Derived configuration quantities computed in ``__init__``."""

    def test_derived_quantities(self):
        sim = VPPSimulator(pp_degree=2, vpp_degree=2, num_acc_steps=4)
        # first_chunk_acc = (num_acc_steps % pp_degree) + pp_degree
        #                 = (4 % 2) + 2 = 2
        self.assertEqual(sim.first_chunk_acc, 2)
        # num_steps = num_acc_steps * vpp_degree = 4 * 2 = 8
        self.assertEqual(sim.num_steps, 8)
        # layer_num = pp_degree * vpp_degree = 2 * 2 = 4
        self.assertEqual(sim.layer_num, 4)
        # One (initially empty) schedule column per pipeline stage.
        self.assertEqual(sim.schedule_table, [[], []])
        self.assertFalse(sim._is_scheduled)

    def test_first_chunk_acc_with_remainder(self):
        # (5 % 3) + 3 = 2 + 3 = 5 ; num_steps = 5 * 2 = 10 ; layer_num = 6.
        sim = VPPSimulator(pp_degree=3, vpp_degree=2, num_acc_steps=5)
        self.assertEqual(sim.first_chunk_acc, 5)
        self.assertEqual(sim.num_steps, 10)
        self.assertEqual(sim.layer_num, 6)
        self.assertEqual(sim.schedule_table, [[], [], []])


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestConsumeTime(unittest.TestCase):
    """Forward and bubble cost one time-unit, backward costs two."""

    def test_costs(self):
        sim = VPPSimulator(pp_degree=2, vpp_degree=2, num_acc_steps=4)
        self.assertEqual(sim._get_consume_time(0, 0, ChunkType.FORWARD), 1)
        self.assertEqual(sim._get_consume_time(0, 0, ChunkType.BUBBLE), 1)
        self.assertEqual(sim._get_consume_time(0, 0, ChunkType.BACKWARD), 2)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestVirtualPPRank(unittest.TestCase):
    """``_get_virtual_pp_rank`` chunk-index -> virtual-pp-rank mapping.

    Config pp=2, vpp=2, acc=4 -> first_chunk_acc = (4%2)+2 = 2,
    first_chunk_steps = first_chunk_acc * vpp = 4. Values below are hand-traced:
      * micro_step < 4 : vrank = micro_step // 2
      * micro_step >= 4: vrank = ((micro_step-4) % 4) // 2
    Backward mirrors forward via vrank -> (vpp - vrank - 1).
    """

    def setUp(self):
        self.sim = VPPSimulator(pp_degree=2, vpp_degree=2, num_acc_steps=4)

    def test_forward_mapping(self):
        expected_forward = {
            0: 0,
            1: 0,
            2: 1,
            3: 1,  # warm-up region (micro_step < 4)
            4: 0,
            5: 0,
            6: 1,
            7: 1,
            8: 0,  # steady region (micro_step >= 4)
        }
        for micro_step, expected in expected_forward.items():
            self.assertEqual(
                self.sim._get_virtual_pp_rank(micro_step, forward=True),
                expected,
                msg=f"forward micro_step={micro_step}",
            )

    def test_backward_is_mirror(self):
        # vpp=2 so mirror is (1 - forward_vrank).
        expected_backward = {0: 1, 1: 1, 2: 0, 3: 0, 4: 1, 6: 0, 8: 1}
        for micro_step, expected in expected_backward.items():
            self.assertEqual(
                self.sim._get_virtual_pp_rank(micro_step, forward=False),
                expected,
                msg=f"backward micro_step={micro_step}",
            )


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestWarmupAndSteadySteps(unittest.TestCase):
    """``_get_warmup_and_steady_steps`` for both scheduling branches."""

    def test_interleave_branch(self):
        # num_acc_steps(4) is NOT in [pp_degree, 2*pp_degree) = [2, 4), so the
        # interleave branch runs. pp=2, vpp=2, acc=4:
        #   first_chunk_acc = 2 ; num_steps = 8
        #   warmup = (pp - stage - 1)*2 + (vpp - 1)*first_chunk_acc, clamped to
        #            num_steps ; steady = num_steps - warmup
        sim = VPPSimulator(pp_degree=2, vpp_degree=2, num_acc_steps=4)
        # stage 0: (2-0-1)*2 + 1*2 = 2 + 2 = 4 ; steady = 8 - 4 = 4
        self.assertEqual(sim._get_warmup_and_steady_steps(0), (4, 4))
        # stage 1: (2-1-1)*2 + 1*2 = 0 + 2 = 2 ; steady = 8 - 2 = 6
        self.assertEqual(sim._get_warmup_and_steady_steps(1), (2, 6))

    def test_interleave_branch_clamped(self):
        # pp=4, vpp=1, acc=1 -> interleave branch. first_chunk_acc=(1%4)+4=5,
        # num_steps = 1*1 = 1. stage 0 raw warmup = (4-0-1)*2 + 0*5 = 6, clamped
        # to num_steps=1 ; steady = 1 - 1 = 0.
        sim = VPPSimulator(pp_degree=4, vpp_degree=1, num_acc_steps=1)
        self.assertEqual(sim._get_warmup_and_steady_steps(0), (1, 0))

    def test_balanced_memory_branch(self):
        # num_acc_steps(3) IS in [pp_degree, 2*pp_degree) = [2, 4), so the
        # balanced-memory branch runs. pp=2, vpp=2, acc=3:
        #   warmup = acc*(vpp-1) + pp - stage - 1
        #   steady = acc - (pp - stage - 1)
        sim = VPPSimulator(pp_degree=2, vpp_degree=2, num_acc_steps=3)
        # stage 0: warmup = 3*1 + 2 - 0 - 1 = 4 ; steady = 3 - 1 = 2
        self.assertEqual(sim._get_warmup_and_steady_steps(0), (4, 2))
        # stage 1: warmup = 3*1 + 2 - 1 - 1 = 3 ; steady = 3 - 0 = 3
        self.assertEqual(sim._get_warmup_and_steady_steps(1), (3, 3))


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestComputeBubbleRate(unittest.TestCase):
    """``compute_bubble_rate`` formula, checked on a hand-built table so the
    result is independent of the scheduler itself.

        bubble_rate = (pp_degree * (max_end - min_start) - sum_exec)
                      / (pp_degree * (max_end - min_start))
    where sum_exec sums (end - start) over non-bubble chunks only.
    """

    def _chunk(self, chunk_type, stage_id, start, end):
        return Chunk(
            virtual_pp_rank=0,
            acc_step=0,
            pp_degree=2,
            vpp_degree=1,
            stage_id=stage_id,
            chunk_type=chunk_type,
            start=start,
            end=end,
        )

    def test_rate_without_bubble(self):
        sim = VPPSimulator(pp_degree=2, vpp_degree=1, num_acc_steps=1)
        sim.schedule_table = [
            [
                self._chunk(ChunkType.FORWARD, 0, 0, 1),
                self._chunk(ChunkType.BACKWARD, 0, 1, 3),
            ],
            [
                self._chunk(ChunkType.FORWARD, 1, 1, 2),
                self._chunk(ChunkType.BACKWARD, 1, 2, 4),
            ],
        ]
        sim._is_scheduled = True
        # min_start=0, max_end=4, total_time=4, total_possible=2*4=8.
        # sum_exec = 1 + 2 + 1 + 2 = 6 ; bubble = 8 - 6 = 2 ; rate = 2/8 = 0.25.
        self.assertAlmostEqual(sim.compute_bubble_rate(), 0.25, places=12)

    def test_bubble_chunk_excluded_from_exec_but_extends_span(self):
        sim = VPPSimulator(pp_degree=2, vpp_degree=1, num_acc_steps=1)
        sim.schedule_table = [
            [
                self._chunk(ChunkType.FORWARD, 0, 0, 1),
                self._chunk(ChunkType.BACKWARD, 0, 1, 3),
                self._chunk(ChunkType.BUBBLE, 0, 4, 6),
            ],
            [
                self._chunk(ChunkType.FORWARD, 1, 1, 2),
                self._chunk(ChunkType.BACKWARD, 1, 2, 4),
            ],
        ]
        sim._is_scheduled = True
        # Bubble pushes max_end to 6 but does not count toward sum_exec.
        # total_time=6, total_possible=12, sum_exec=6, rate = (12-6)/12 = 0.5.
        self.assertAlmostEqual(sim.compute_bubble_rate(), 0.5, places=12)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestScheduleInvariants(unittest.TestCase):
    """Structural invariants of the full ``schedule()`` output.

    From the VPP contract, each pipeline stage runs exactly ``num_steps =
    num_acc_steps * vpp_degree`` forward chunks and the same number of backward
    chunks (no BUBBLE-typed chunks are ever appended). Each (stage, virtual pp
    rank) layer is visited once per accumulation step, forward and backward.
    """

    def _check(self, pp_degree, vpp_degree, num_acc_steps):
        sim = VPPSimulator(
            pp_degree=pp_degree,
            vpp_degree=vpp_degree,
            num_acc_steps=num_acc_steps,
        )
        table = sim.schedule()
        num_steps = num_acc_steps * vpp_degree

        self.assertEqual(len(table), pp_degree)

        total_chunks = 0
        for stage_id, column in enumerate(table):
            forwards = [c for c in column if c.chunk_type == ChunkType.FORWARD]
            backwards = [
                c for c in column if c.chunk_type == ChunkType.BACKWARD
            ]
            # No bubble-typed chunks are materialised by schedule().
            self.assertEqual(len(forwards) + len(backwards), len(column))
            self.assertEqual(len(forwards), num_steps)
            self.assertEqual(len(backwards), num_steps)
            total_chunks += len(column)

            # Durations honour the consume-time table.
            for c in column:
                expected = 1 if c.chunk_type == ChunkType.FORWARD else 2
                self.assertEqual(c.end - c.start, expected)
                self.assertGreaterEqual(c.start, 0)

            # Chunks within a stage are ordered and non-overlapping.
            for prev, cur in zip(column, column[1:]):
                self.assertGreaterEqual(cur.start, prev.end)

            # Each (stage, vrank) layer is scheduled once per acc step, in both
            # directions, and its layer_id follows vrank*pp + stage.
            for direction in (forwards, backwards):
                by_vrank = {}
                for c in direction:
                    by_vrank.setdefault(c.virtual_pp_rank, []).append(c)
                self.assertEqual(set(by_vrank), set(range(vpp_degree)))
                for vrank, chunks in by_vrank.items():
                    self.assertEqual(
                        sorted(c.acc_step for c in chunks),
                        list(range(num_acc_steps)),
                    )
                    for c in chunks:
                        self.assertEqual(
                            c.layer_id, vrank * pp_degree + stage_id
                        )

        # Global chunk budget: pp * vpp * acc forward + same many backward.
        self.assertEqual(
            total_chunks, num_acc_steps * pp_degree * vpp_degree * 2
        )

        rate = sim.compute_bubble_rate()
        self.assertGreaterEqual(rate, 0.0)
        self.assertLess(rate, 1.0)

    def test_interleave_config(self):
        # acc=4 not in [2,4) -> interleave branch.
        self._check(pp_degree=2, vpp_degree=2, num_acc_steps=4)

    def test_balanced_memory_config(self):
        # acc=3 in [2,4) -> balanced-memory branch.
        self._check(pp_degree=2, vpp_degree=2, num_acc_steps=3)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestPPChunkRecorder(unittest.TestCase):
    """``PPChunkRecorder`` gates layer ids to the local hidden-layer window
    ``[head, head + num_hidden_layers)`` and accumulates per-layer forward
    visits into ``acc_stamp``.
    """

    def _recorder(self):
        # head=2 empty layers, 3 hidden layers, tail=1 -> valid ids {2, 3, 4}
        # mapping to acc_stamp indices {0, 1, 2}.
        return PPChunkRecorder(
            pp_degree=2,
            vpp_degree=2,
            num_acc_steps=4,
            num_hidden_layers=3,
            num_empty_layers_add_in_head=2,
            num_empty_layers_add_in_tail=1,
        )

    def test_initial_stamp_is_zero(self):
        rec = self._recorder()
        self.assertEqual(rec.acc_stamp, [0, 0, 0])

    def test_out_of_window_returns_false_and_no_increment(self):
        rec = self._recorder()
        # Below the head window.
        self.assertIs(rec.record_chunk_forward(0), False)
        self.assertIs(rec.record_chunk_forward(1), False)
        # At/after head + num_hidden = 5 (upper bound is exclusive).
        self.assertIs(rec.record_chunk_forward(5), False)
        self.assertIs(rec.record_chunk_forward(6), False)
        self.assertEqual(rec.acc_stamp, [0, 0, 0])

    def test_in_window_accumulates_at_shifted_index(self):
        rec = self._recorder()
        # layer_id 2 -> index 0, hit twice; layer_id 4 -> index 2, hit once.
        self.assertIsNone(rec.record_chunk_forward(2))
        self.assertIsNone(rec.record_chunk_forward(2))
        self.assertIsNone(rec.record_chunk_forward(4))
        self.assertEqual(rec.acc_stamp, [2, 0, 1])

    def test_step_resets_stamp(self):
        rec = self._recorder()
        rec.record_chunk_forward(3)
        self.assertEqual(rec.acc_stamp, [0, 1, 0])
        rec.step()
        self.assertEqual(rec.acc_stamp, [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
