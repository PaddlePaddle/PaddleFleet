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

"""CPU-only behavior tests for the device-independent interleaved (VPP)
pipeline schedule logic in ``paddlefleet.pipeline_parallel.vpp_simulator``.

The interleaved / virtual-pipeline schedule ordering is pure Python index
math (no collectives, no device kernels), so it is fully exercisable on a CPU
host. Every expected value below is derived BY HAND from the interleaved 1F1B
contract and written as a literal, never obtained by calling the code under
test:

* ``VPPSimulator.__init__`` derived quantities -- ``first_chunk_acc``,
  ``num_steps`` and ``layer_num`` follow closed-form formulas.
* ``Chunk.layer_id`` / ``Chunk.__str__`` -- ``layer_id == vpp_rank*pp + stage``
  for compute chunks, ``None`` for bubbles; the string encodes layer id, the
  1-indexed acc step and the (start, end) window.
* ``_get_virtual_pp_rank`` -- the interleave index math that maps a micro step
  to the virtual chunk it belongs to, including the first-chunk warmup region,
  the cyclic steady region, and the forward/backward chunk reversal.
* ``_get_warmup_and_steady_steps`` -- both scheduler regimes (the plain
  interleave branch and the ``pp <= acc < 2*pp`` balanced-memory branch) plus
  the ``min(warmup, num_steps)`` clamp when ``acc < pp``.
* ``_get_consume_time`` -- forward/bubble cost 1, backward costs 2.
* ``schedule()`` end to end -- per stage every virtual chunk consumes each of
  the ``num_acc_steps`` micro-batches exactly once in both directions, chunk
  durations equal the consume times, chunks within a stage are non-overlapping
  and monotonically ordered, and every layer id is consistent.

This file does NOT cover the real cross-rank P2P send/recv, gradient overlap,
or any process-group behavior of the runtime ``PipelineParallelWithInterleave``
engine: those require a real multi-card pipeline and are out of scope for a CPU
unit test (asserting them here would only fake collectives). The plotting
helpers (``draw_chunks``/``draw_balls``) are also excluded -- they are I/O side
effects, not schedule logic.
"""

import unittest

try:
    from paddlefleet.pipeline_parallel.vpp_simulator import (
        Chunk,
        ChunkType,
        VPPSimulator,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # honest dependency probe
    Chunk = None
    ChunkType = None
    VPPSimulator = None
    _IMPORT_ERROR = exc

_HAS_DEPS = _IMPORT_ERROR is None
_SKIP_REASON = (
    "paddlefleet.pipeline_parallel.vpp_simulator not importable on this "
    f"host: {_IMPORT_ERROR!r}"
)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestVPPSimulatorInit(unittest.TestCase):
    """Closed-form derived quantities computed in ``__init__``."""

    def test_derived_quantities_plain_interleave(self):
        # pp=4, vpp=2, acc=8: first_chunk_acc = (8 % 4) + 4 = 4,
        # num_steps = acc * vpp = 16, layer_num = pp * vpp = 8.
        sim = VPPSimulator(pp_degree=4, vpp_degree=2, num_acc_steps=8)
        self.assertEqual(sim.first_chunk_acc, 4)
        self.assertEqual(sim.num_steps, 16)
        self.assertEqual(sim.layer_num, 8)
        self.assertEqual(len(sim.schedule_table), 4)
        self.assertEqual(sim.schedule_table, [[], [], [], []])

    def test_derived_quantities_non_multiple_acc(self):
        # pp=4, vpp=2, acc=6: first_chunk_acc = (6 % 4) + 4 = 6,
        # num_steps = 12, layer_num = 8.
        sim = VPPSimulator(pp_degree=4, vpp_degree=2, num_acc_steps=6)
        self.assertEqual(sim.first_chunk_acc, 6)
        self.assertEqual(sim.num_steps, 12)
        self.assertEqual(sim.layer_num, 8)

    def test_derived_quantities_three_chunks(self):
        # pp=2, vpp=3, acc=4: first_chunk_acc = (4 % 2) + 2 = 2,
        # num_steps = 12, layer_num = 6.
        sim = VPPSimulator(pp_degree=2, vpp_degree=3, num_acc_steps=4)
        self.assertEqual(sim.first_chunk_acc, 2)
        self.assertEqual(sim.num_steps, 12)
        self.assertEqual(sim.layer_num, 6)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestChunk(unittest.TestCase):
    """``Chunk.layer_id`` math and string encoding."""

    def test_compute_chunk_layer_id(self):
        # layer_id = virtual_pp_rank * pp_degree + stage_id = 1*4 + 2 = 6.
        chunk = Chunk(
            virtual_pp_rank=1,
            acc_step=0,
            pp_degree=4,
            vpp_degree=2,
            stage_id=2,
            chunk_type=ChunkType.FORWARD,
            start=3,
            end=4,
        )
        self.assertEqual(chunk.layer_id, 6)

    def test_layer_id_varies_with_rank_and_stage(self):
        # A different (rank, stage) must land on a different layer:
        # 0*4 + 3 = 3, distinct from the 6 above.
        chunk = Chunk(
            virtual_pp_rank=0,
            acc_step=2,
            pp_degree=4,
            vpp_degree=2,
            stage_id=3,
            chunk_type=ChunkType.BACKWARD,
            start=5,
            end=7,
        )
        self.assertEqual(chunk.layer_id, 3)

    def test_bubble_chunk_has_no_layer_id(self):
        chunk = Chunk(
            virtual_pp_rank=0,
            acc_step=0,
            pp_degree=4,
            vpp_degree=2,
            stage_id=0,
            chunk_type=ChunkType.BUBBLE,
            start=2,
            end=3,
        )
        self.assertIsNone(chunk.layer_id)

    def test_str_encodes_layer_acc_and_window(self):
        # Forward chunk: "F" + layer_id(6) + "_" + (acc_step+1=1) + (start,end).
        chunk = Chunk(
            virtual_pp_rank=1,
            acc_step=0,
            pp_degree=4,
            vpp_degree=2,
            stage_id=2,
            chunk_type=ChunkType.FORWARD,
            start=3,
            end=4,
        )
        self.assertEqual(str(chunk), "F6_1(3, 4)")

    def test_str_uses_one_indexed_acc_step(self):
        # acc_step=2 renders as "_3" (1-indexed); backward layer_id 0*2+0 = 0.
        chunk = Chunk(
            virtual_pp_rank=0,
            acc_step=2,
            pp_degree=2,
            vpp_degree=3,
            stage_id=0,
            chunk_type=ChunkType.BACKWARD,
            start=5,
            end=7,
        )
        self.assertEqual(str(chunk), "B0_3(5, 7)")


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestVirtualPPRank(unittest.TestCase):
    """``_get_virtual_pp_rank`` interleave index math.

    Expected sequences are enumerated by hand from the formula:
    ``first_chunk_steps = first_chunk_acc * vpp``; below it the rank is
    ``micro_step // first_chunk_acc``; at/above it the rank cycles as
    ``((micro_step - first_chunk_steps) % (pp*vpp)) // pp``. Backward reverses
    the rank via ``vpp - rank - 1``.
    """

    def test_forward_sequence_pp4_vpp2_acc8(self):
        sim = VPPSimulator(pp_degree=4, vpp_degree=2, num_acc_steps=8)
        # first_chunk_acc=4, first_chunk_steps=8.
        expected = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
        actual = [
            sim._get_virtual_pp_rank(step, forward=True) for step in range(16)
        ]
        self.assertEqual(actual, expected)

    def test_backward_sequence_pp4_vpp2_acc8(self):
        sim = VPPSimulator(pp_degree=4, vpp_degree=2, num_acc_steps=8)
        # Backward reverses each forward rank via 1 - rank.
        expected = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
        actual = [
            sim._get_virtual_pp_rank(step, forward=False) for step in range(16)
        ]
        self.assertEqual(actual, expected)

    def test_forward_sequence_pp2_vpp3_acc4(self):
        sim = VPPSimulator(pp_degree=2, vpp_degree=3, num_acc_steps=4)
        # first_chunk_acc=2, first_chunk_steps=6.
        expected = [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2]
        actual = [
            sim._get_virtual_pp_rank(step, forward=True) for step in range(12)
        ]
        self.assertEqual(actual, expected)

    def test_backward_sequence_pp2_vpp3_acc4(self):
        sim = VPPSimulator(pp_degree=2, vpp_degree=3, num_acc_steps=4)
        # Backward reverses each forward rank via 2 - rank.
        expected = [2, 2, 1, 1, 0, 0, 2, 2, 1, 1, 0, 0]
        actual = [
            sim._get_virtual_pp_rank(step, forward=False) for step in range(12)
        ]
        self.assertEqual(actual, expected)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestWarmupAndSteadySteps(unittest.TestCase):
    """``_get_warmup_and_steady_steps`` for both regimes and the clamp."""

    def test_plain_interleave_branch(self):
        # pp=4, vpp=2, acc=8 -> NOT (pp <= acc < 2*pp) since 8 == 2*pp.
        # warmup = (pp - stage - 1)*2 + (vpp-1)*first_chunk_acc, fca=4.
        # steady = num_steps - warmup, num_steps=16.
        sim = VPPSimulator(pp_degree=4, vpp_degree=2, num_acc_steps=8)
        expected = [(10, 6), (8, 8), (6, 10), (4, 12)]
        actual = [sim._get_warmup_and_steady_steps(s) for s in range(4)]
        self.assertEqual(actual, expected)

    def test_balanced_memory_branch(self):
        # pp=4, vpp=2, acc=6 -> pp <= acc < 2*pp holds (4 <= 6 < 8).
        # warmup = acc*(vpp-1) + pp - stage - 1 = 6 + 3 - stage.
        # steady = acc - (pp - stage - 1) = 3 + stage.
        sim = VPPSimulator(pp_degree=4, vpp_degree=2, num_acc_steps=6)
        expected = [(9, 3), (8, 4), (7, 5), (6, 6)]
        actual = [sim._get_warmup_and_steady_steps(s) for s in range(4)]
        self.assertEqual(actual, expected)

    def test_warmup_clamped_to_num_steps(self):
        # pp=4, vpp=3, acc=2 -> plain branch (2 < pp). first_chunk_acc=6,
        # num_steps=6. Raw warmup = (3-stage)*2 + 2*6 = far above 6, so it is
        # clamped to num_steps and steady collapses to 0 for every stage.
        sim = VPPSimulator(pp_degree=4, vpp_degree=3, num_acc_steps=2)
        expected = [(6, 0), (6, 0), (6, 0), (6, 0)]
        actual = [sim._get_warmup_and_steady_steps(s) for s in range(4)]
        self.assertEqual(actual, expected)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestConsumeTime(unittest.TestCase):
    """``_get_consume_time``: forward/bubble cost 1, backward costs 2."""

    def test_consume_times_by_chunk_type(self):
        sim = VPPSimulator(pp_degree=4, vpp_degree=2, num_acc_steps=8)
        self.assertEqual(sim._get_consume_time(0, 0, ChunkType.FORWARD), 1)
        self.assertEqual(sim._get_consume_time(0, 0, ChunkType.BUBBLE), 1)
        self.assertEqual(sim._get_consume_time(0, 0, ChunkType.BACKWARD), 2)

    def test_consume_time_independent_of_rank_and_step(self):
        # The cost depends only on the chunk type; rank/acc_step do not change
        # it. Distinct (rank, acc_step) inputs must return the same value.
        sim = VPPSimulator(pp_degree=4, vpp_degree=2, num_acc_steps=8)
        self.assertEqual(
            sim._get_consume_time(1, 5, ChunkType.BACKWARD),
            sim._get_consume_time(0, 0, ChunkType.BACKWARD),
        )


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestScheduleInvariants(unittest.TestCase):
    """End-to-end ``schedule()`` content, coverage and ordering invariants.

    These invariants come from the interleaved 1F1B contract and are
    independent of the scheduler's own bookkeeping: every one of the
    ``vpp_degree`` virtual chunks on a stage must run each of the
    ``num_acc_steps`` micro-batches exactly once, forward and backward; chunk
    durations equal the consume times; and chunks within a stage never overlap
    and are emitted in non-decreasing time order.
    """

    def _assert_schedule_contract(self, pp, vpp, acc):
        sim = VPPSimulator(pp_degree=pp, vpp_degree=vpp, num_acc_steps=acc)
        table = sim.schedule()
        num_steps = sim.num_steps  # acc * vpp

        self.assertEqual(len(table), pp)

        for stage in range(pp):
            chunks = table[stage]
            forwards = [c for c in chunks if c.chunk_type == ChunkType.FORWARD]
            backwards = [
                c for c in chunks if c.chunk_type == ChunkType.BACKWARD
            ]

            # No bubble chunks are materialised into the table; only forward
            # and backward, num_steps of each.
            self.assertEqual(len(chunks), 2 * num_steps)
            self.assertEqual(len(forwards), num_steps)
            self.assertEqual(len(backwards), num_steps)

            # Each virtual chunk processes every micro-batch exactly once, in
            # both directions.
            for v in range(vpp):
                fwd_acc = sorted(
                    c.acc_step for c in forwards if c.virtual_pp_rank == v
                )
                bwd_acc = sorted(
                    c.acc_step for c in backwards if c.virtual_pp_rank == v
                )
                self.assertEqual(fwd_acc, list(range(acc)))
                self.assertEqual(bwd_acc, list(range(acc)))

            for c in chunks:
                # layer_id is consistent with (rank, stage).
                self.assertEqual(
                    c.layer_id, c.virtual_pp_rank * pp + c.stage_id
                )
                # Duration equals the consume time: forward 1, backward 2.
                duration = c.end - c.start
                expected_duration = (
                    1 if c.chunk_type == ChunkType.FORWARD else 2
                )
                self.assertEqual(duration, expected_duration)
                self.assertGreaterEqual(c.start, 0)

            # Within a stage chunks are serialised: no overlap, ordered.
            for i in range(1, len(chunks)):
                self.assertGreaterEqual(chunks[i].start, chunks[i - 1].end)

    def test_schedule_pp4_vpp2_acc8(self):
        self._assert_schedule_contract(pp=4, vpp=2, acc=8)

    def test_schedule_pp2_vpp3_acc4(self):
        self._assert_schedule_contract(pp=2, vpp=3, acc=4)

    def test_schedule_pp4_vpp2_acc6_balanced_memory(self):
        self._assert_schedule_contract(pp=4, vpp=2, acc=6)


if __name__ == "__main__":
    unittest.main()
