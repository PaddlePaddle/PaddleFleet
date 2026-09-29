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

"""CPU-only behavior tests for the F-then-B interleaved (VPP) pipeline
*schedule ordering* implemented by
``paddlefleet.pipeline_parallel.vpp_simulator``.

The simulator is device-independent pure Python: it produces the ordering of
FORWARD / BACKWARD chunks each pipeline stage executes under the
``VPPFhenBInBalancedMemory`` and standard-interleave schedules, plus a bubble
rate metric. None of this needs a GPU or a process group, so it is exercised
here on CPU.

Every expected value below is derived BY HAND from the schedule definition
(virtual-pp-rank interleaving, warmup / steady / cooldown phase sizes, and the
requirement that each virtual chunk processes every micro-batch exactly once)
and written as explicit literals -- never read back from the code under test.

Out of scope (needs a real multi-card pipeline / process group, not faked
here): actual tensor send/recv, cross-rank gradient reduction, and the wall
clock timing of the barrier phase. Only the local ordering contract is checked.
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
    "paddlefleet.pipeline_parallel.vpp_simulator not importable on this CPU "
    f"host (its package __init__ pulls in paddle): {_IMPORT_ERROR!r}"
)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestChunkIdentity(unittest.TestCase):
    """A ``Chunk`` knows which model layer it maps to and prints itself."""

    def test_forward_chunk_layer_id_and_repr(self):
        # layer_id == virtual_pp_rank * pp_degree + stage_id for non-bubble.
        # vpr=1, pp=4, stage=3 -> 1*4+3 = 7. acc_step=2 -> label uses acc+1=3.
        c = Chunk(1, 2, 4, 2, 3, ChunkType.FORWARD, 5, 7)
        self.assertEqual(c.layer_id, 7)
        self.assertEqual(str(c), "F7_3(5, 7)")

    def test_backward_chunk_layer_id_and_repr(self):
        # vpr=0, pp=4, stage=3 -> 0*4+3 = 3. acc_step=0 -> label 1.
        c = Chunk(0, 0, 4, 2, 3, ChunkType.BACKWARD, 1, 4)
        self.assertEqual(c.layer_id, 3)
        self.assertEqual(str(c), "B3_1(1, 4)")

    def test_bubble_chunk_has_no_layer_id(self):
        # A bubble maps to no layer; repr shows only the (start, end) window.
        c = Chunk(0, 0, 4, 2, 0, ChunkType.BUBBLE, 3, 5)
        self.assertIsNone(c.layer_id)
        self.assertEqual(str(c), "Z((3, 5))")

    def test_chunk_type_marker_letters(self):
        self.assertEqual(ChunkType.FORWARD.value, "F")
        self.assertEqual(ChunkType.BACKWARD.value, "B")
        self.assertEqual(ChunkType.BUBBLE.value, "Z")


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestDerivedConfig(unittest.TestCase):
    """Constructor derives the schedule sizing constants."""

    def test_standard_interleave_constants(self):
        # pp=4, vpp=2, acc=8: first_chunk_acc = (8 % 4) + 4 = 4,
        # num_steps = acc*vpp = 16, layer_num = pp*vpp = 8.
        s = VPPSimulator(4, 2, 8)
        self.assertEqual(s.first_chunk_acc, 4)
        self.assertEqual(s.num_steps, 16)
        self.assertEqual(s.layer_num, 8)

    def test_fthenb_balanced_memory_constants(self):
        # pp=4, vpp=2, acc=6: first_chunk_acc = (6 % 4) + 4 = 6,
        # num_steps = 12, layer_num = 8.
        s = VPPSimulator(4, 2, 6)
        self.assertEqual(s.first_chunk_acc, 6)
        self.assertEqual(s.num_steps, 12)
        self.assertEqual(s.layer_num, 8)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestVirtualPPRankInterleaving(unittest.TestCase):
    """``_get_virtual_pp_rank`` maps a global micro-step to a model chunk.

    For pp=4, vpp=2, acc=8: first_chunk_acc = 4, so first_chunk_steps =
    first_chunk_acc * vpp = 8. For micro_step < 8 the rank is
    ``micro_step // 4``; from 8 onward it is ``((micro_step - 8) % 8) // 4``.
    Backward flips the rank via ``vpp - rank - 1``.
    """

    def test_forward_rank_sequence(self):
        s = VPPSimulator(4, 2, 8)
        got = [s._get_virtual_pp_rank(i, forward=True) for i in range(17)]
        # Hand-derived: first block of 4 -> chunk 0, next 4 -> chunk 1, then
        # the steady modulo pattern repeats the 4+4 grouping.
        expected = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0]
        self.assertEqual(got, expected)

    def test_backward_rank_is_mirror_of_forward(self):
        s = VPPSimulator(4, 2, 8)
        got = [s._get_virtual_pp_rank(i, forward=False) for i in range(17)]
        # Backward rank = vpp - forward_rank - 1 = 1 - forward_rank.
        expected = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1]
        self.assertEqual(got, expected)

    def test_forward_and_backward_are_complementary(self):
        # For vpp=2 the two directions must partition the chunk set at every
        # step: forward_rank + backward_rank == vpp - 1 == 1.
        s = VPPSimulator(4, 2, 8)
        for i in range(17):
            f = s._get_virtual_pp_rank(i, forward=True)
            b = s._get_virtual_pp_rank(i, forward=False)
            self.assertEqual(f + b, 1, f"step {i}: {f} + {b} != 1")


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestWarmupSteadySteps(unittest.TestCase):
    """``_get_warmup_and_steady_steps`` sizes each stage's schedule phases."""

    def test_standard_interleave_branch(self):
        # pp=4, vpp=2, acc=8 -> acc >= 2*pp, so the standard interleave branch:
        # warmup = (pp - stage - 1)*2 + (vpp-1)*first_chunk_acc, capped at
        # num_steps=16; steady = num_steps - warmup. first_chunk_acc = 4.
        #   stage0: (3)*2 + 4 = 10 -> steady 6
        #   stage1: (2)*2 + 4 = 8  -> steady 8
        #   stage2: (1)*2 + 4 = 6  -> steady 10
        #   stage3: (0)*2 + 4 = 4  -> steady 12
        s = VPPSimulator(4, 2, 8)
        got = [s._get_warmup_and_steady_steps(st) for st in range(4)]
        self.assertEqual(got, [(10, 6), (8, 8), (6, 10), (4, 12)])

    def test_fthenb_balanced_memory_branch(self):
        # pp=4, vpp=2, acc=6 -> pp <= acc < 2*pp, so the FhenB balanced-memory
        # branch: warmup = acc*(vpp-1) + pp - stage - 1 = 6 + 3 - stage;
        # steady = acc - (pp - stage - 1) = 3 + stage.
        #   stage0: (9, 3)  stage1: (8, 4)  stage2: (7, 5)  stage3: (6, 6)
        s = VPPSimulator(4, 2, 6)
        got = [s._get_warmup_and_steady_steps(st) for st in range(4)]
        self.assertEqual(got, [(9, 3), (8, 4), (7, 5), (6, 6)])

    def test_phase_sizes_sum_to_num_steps(self):
        # In both branches warmup + steady == num_steps for every stage: this
        # is what makes forward count == num_steps per stage below.
        for pp, vpp, acc in [(4, 2, 8), (4, 2, 6)]:
            s = VPPSimulator(pp, vpp, acc)
            for st in range(pp):
                w, st_steady = s._get_warmup_and_steady_steps(st)
                self.assertEqual(w + st_steady, s.num_steps)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestScheduleOrdering(unittest.TestCase):
    """``_schedule_without_bubble`` lays out F/B chunks per stage.

    Independent contract of any correct interleaved 1F1B schedule: every
    stage runs ``num_steps`` forwards and ``num_steps`` backwards, and each
    virtual chunk performs exactly one forward and one backward for every one
    of the ``acc`` micro-batches (so the acc_step labels are a permutation of
    ``range(acc)``). None of these expectations are read from the simulator.
    """

    def _collect(self, pp, vpp, acc):
        s = VPPSimulator(pp, vpp, acc)
        s._schedule_without_bubble()
        return s

    def test_per_stage_forward_backward_counts(self):
        for pp, vpp, acc in [(4, 2, 8), (4, 2, 6)]:
            s = self._collect(pp, vpp, acc)
            for st in range(pp):
                table = s.schedule_table[st]
                fwd = [c for c in table if c.chunk_type == ChunkType.FORWARD]
                bwd = [c for c in table if c.chunk_type == ChunkType.BACKWARD]
                self.assertEqual(len(fwd), s.num_steps)
                self.assertEqual(len(bwd), s.num_steps)

    def test_each_chunk_processes_every_microbatch_once(self):
        for pp, vpp, acc in [(4, 2, 8), (4, 2, 6)]:
            s = self._collect(pp, vpp, acc)
            for st in range(pp):
                table = s.schedule_table[st]
                for vpr in range(vpp):
                    fwd_acc = sorted(
                        c.acc_step
                        for c in table
                        if c.chunk_type == ChunkType.FORWARD
                        and c.virtual_pp_rank == vpr
                    )
                    bwd_acc = sorted(
                        c.acc_step
                        for c in table
                        if c.chunk_type == ChunkType.BACKWARD
                        and c.virtual_pp_rank == vpr
                    )
                    self.assertEqual(fwd_acc, list(range(acc)))
                    self.assertEqual(bwd_acc, list(range(acc)))

    def test_layer_id_follows_rank_and_stage(self):
        # layer_id == virtual_pp_rank * pp + stage_id for every scheduled chunk.
        s = self._collect(4, 2, 8)
        for st in range(4):
            for c in s.schedule_table[st]:
                self.assertEqual(c.layer_id, c.virtual_pp_rank * 4 + st)

    def test_phase_structure_warmup_steady_cooldown(self):
        # Stage 0 of pp=4/vpp=2/acc=8: warmup=10 (all F), steady=6 (F,B
        # pairs), cooldown = num_steps - steady = 10 (all B). Total 32.
        s = self._collect(4, 2, 8)
        table = s.schedule_table[0]
        letters = "".join(c.chunk_type.value for c in table)
        expected = "F" * 10 + "FB" * 6 + "B" * 10
        self.assertEqual(letters, expected)
        self.assertEqual(len(table), 32)

    def test_warmup_forward_rank_and_accstep_sequence(self):
        # The first warmup_steps=10 chunks of stage 0 are forwards whose ranks
        # follow the interleave pattern [0,0,0,0,1,1,1,1,0,0]; the acc_step of
        # each is the running per-rank forward counter -> chunk 0 acc 0..3,
        # chunk 1 acc 0..3, chunk 0 acc 4..5.
        s = self._collect(4, 2, 8)
        warm = s.schedule_table[0][:10]
        self.assertTrue(all(c.chunk_type == ChunkType.FORWARD for c in warm))
        self.assertEqual(
            [c.virtual_pp_rank for c in warm],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
        )
        self.assertEqual(
            [c.acc_step for c in warm],
            [0, 1, 2, 3, 0, 1, 2, 3, 4, 5],
        )

    def test_first_steady_pair_is_forward_then_backward(self):
        # First steady iteration on stage 0: a forward on chunk 0 (its 7th
        # forward -> acc_step 6) immediately followed by a backward on chunk 1
        # (its 1st backward -> acc_step 0).
        s = self._collect(4, 2, 8)
        f_chunk, b_chunk = s.schedule_table[0][10], s.schedule_table[0][11]
        self.assertEqual(f_chunk.chunk_type, ChunkType.FORWARD)
        self.assertEqual((f_chunk.virtual_pp_rank, f_chunk.acc_step), (0, 6))
        self.assertEqual(b_chunk.chunk_type, ChunkType.BACKWARD)
        self.assertEqual((b_chunk.virtual_pp_rank, b_chunk.acc_step), (1, 0))


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestBubbleRate(unittest.TestCase):
    """``compute_bubble_rate`` reports pipeline idle fraction in [0, 1)."""

    def test_rate_is_a_fraction(self):
        rate = VPPSimulator(4, 2, 8).compute_bubble_rate()
        self.assertIsInstance(rate, float)
        self.assertGreater(rate, 0.0)
        self.assertLess(rate, 1.0)

    def test_more_accumulation_amortizes_the_bubble(self):
        # Independently known property: warmup + cooldown bubbles are fixed by
        # the topology, so spreading them over more micro-batches strictly
        # lowers the bubble fraction. Rates must be strictly decreasing.
        rates = [
            VPPSimulator(4, 2, acc).compute_bubble_rate()
            for acc in (4, 8, 16, 32)
        ]
        for earlier, later in zip(rates, rates[1:]):
            self.assertGreater(earlier, later)


if __name__ == "__main__":
    unittest.main()
