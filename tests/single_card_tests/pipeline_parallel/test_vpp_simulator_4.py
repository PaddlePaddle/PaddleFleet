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

"""CPU-only behaviour tests for the deterministic scheduling logic in
``paddlefleet.pipeline_parallel.vpp_simulator``.

This file targets a slice that is DISTINCT from the existing
``tests/single_card_tests/test_vpp_simulator.py`` (which covers
``VPPSimulator.compute_bubble_rate`` and ``PPChunkRecorder``). Here we pin the
pure, device-independent building blocks that decide *which* chunk runs *where*
in the interleaved (VPP) schedule, with every expected value hand-derived and
hardcoded (never taken from the production output):

* ``_get_virtual_pp_rank`` - the forward/backward virtual-pp-rank selector.
* ``_get_warmup_and_steady_steps`` - both the standard interleave branch and
  the balanced-memory branch (``pp <= num_acc_steps < 2*pp``).
* ``_get_consume_time`` - the FORWARD/BUBBLE=1, BACKWARD=2 cost contract.
* ``Chunk.layer_id`` / ``Chunk.__str__`` - identity math and rendering.
* ``_schedule_without_bubble`` - the full per-stage chunk ORDER and identity
  (chunk type, virtual-pp-rank, acc step, layer id) for pp=2/vpp=2/acc=4,
  compared against a chunk-by-chunk hand trace.

Timing assignment (``_add_bubble`` / ``_barrier``) and the matplotlib drawing
helpers are out of scope here.
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
    f"CPU host: {_IMPORT_ERROR!r}"
)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestVirtualPpRankSelection(unittest.TestCase):
    """``_get_virtual_pp_rank`` maps a micro step to a virtual pp rank.

    For pp=2, vpp=2, acc=4: ``first_chunk_acc = (4 % 2) + 2 = 2`` and
    ``first_chunk_steps = first_chunk_acc * vpp = 4``. For a micro step below
    first_chunk_steps the rank is ``micro_step // first_chunk_acc``; above it,
    ``((micro_step - 4) % 4) // 2``. The backward flag mirrors the rank via
    ``vpp - rank - 1``. Both tables below are hand traced, not read back from
    the implementation.
    """

    def setUp(self):
        self.sim = VPPSimulator(pp_degree=2, vpp_degree=2, num_acc_steps=4)

    def test_config_derived_constants(self):
        self.assertEqual(self.sim.first_chunk_acc, 2)
        self.assertEqual(self.sim.num_steps, 8)
        self.assertEqual(self.sim.layer_num, 4)

    def test_forward_ranks_hand_traced(self):
        # micro_step:  0  1  2  3 | 4  5  6  7 | 8  9 10
        expected = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]
        got = [
            self.sim._get_virtual_pp_rank(ms, forward=True)
            for ms in range(len(expected))
        ]
        self.assertEqual(got, expected)

    def test_backward_ranks_are_mirrored(self):
        # backward rank = vpp_degree - forward_rank - 1 = 1 - forward_rank
        expected = [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0]
        got = [
            self.sim._get_virtual_pp_rank(ms, forward=False)
            for ms in range(len(expected))
        ]
        self.assertEqual(got, expected)

    def test_balanced_memory_config_ranks(self):
        # pp=4, vpp=2, acc=5: first_chunk_acc = (5 % 4) + 4 = 5,
        # first_chunk_steps = 10. Below 10: ms // 5; above: ((ms-10) % 8) // 4.
        sim = VPPSimulator(pp_degree=4, vpp_degree=2, num_acc_steps=5)
        self.assertEqual(sim.first_chunk_acc, 5)
        forward = {0: 0, 4: 0, 5: 1, 9: 1, 10: 0, 13: 0, 14: 1}
        for ms, rank in forward.items():
            self.assertEqual(
                sim._get_virtual_pp_rank(ms, forward=True), rank, msg=f"ms={ms}"
            )
        # Backward mirrors: rank 0 -> 1, rank 1 -> 0.
        self.assertEqual(sim._get_virtual_pp_rank(0, forward=False), 1)
        self.assertEqual(sim._get_virtual_pp_rank(5, forward=False), 0)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestWarmupAndSteadySteps(unittest.TestCase):
    """``_get_warmup_and_steady_steps`` splits each stage into warmup / steady.

    Two branches exist. The balanced-memory branch triggers only when
    ``pp <= num_acc_steps < 2*pp``; otherwise the standard interleave branch
    runs. All expected pairs are computed by hand from the documented formulas.
    """

    def test_standard_interleave_branch(self):
        # pp=2, vpp=2, acc=4 (4 not in [2, 4) -> standard branch).
        # num_steps = acc * vpp = 8, first_chunk_acc = 2.
        # warmup = (pp - stage - 1) * 2 + (vpp - 1) * first_chunk_acc, capped
        #          at num_steps; steady = num_steps - warmup.
        sim = VPPSimulator(pp_degree=2, vpp_degree=2, num_acc_steps=4)
        # stage 0: (2-0-1)*2 + 1*2 = 4 -> steady 4
        self.assertEqual(sim._get_warmup_and_steady_steps(0), (4, 4))
        # stage 1: (2-1-1)*2 + 1*2 = 2 -> steady 6
        self.assertEqual(sim._get_warmup_and_steady_steps(1), (2, 6))

    def test_balanced_memory_branch(self):
        # pp=4, vpp=2, acc=5 (5 in [4, 8) -> balanced-memory branch).
        # warmup = acc*(vpp-1) + pp - stage - 1 = 5 + (3 - stage)
        # steady = acc - (pp - stage - 1) = 5 - (3 - stage)
        sim = VPPSimulator(pp_degree=4, vpp_degree=2, num_acc_steps=5)
        self.assertEqual(sim._get_warmup_and_steady_steps(0), (8, 2))
        self.assertEqual(sim._get_warmup_and_steady_steps(1), (7, 3))
        self.assertEqual(sim._get_warmup_and_steady_steps(2), (6, 4))
        self.assertEqual(sim._get_warmup_and_steady_steps(3), (5, 5))

    def test_standard_branch_warmup_strictly_decreases_with_stage(self):
        # Distinct fixture where the standard branch is guaranteed (acc large).
        sim = VPPSimulator(pp_degree=4, vpp_degree=2, num_acc_steps=8)
        warmups = [sim._get_warmup_and_steady_steps(s)[0] for s in range(4)]
        self.assertEqual(warmups, sorted(warmups, reverse=True))
        self.assertEqual(len(set(warmups)), 4)  # all distinct, none collapsed


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestConsumeTime(unittest.TestCase):
    """Forward/bubble chunks cost 1 tick, backward chunks cost 2."""

    def test_cost_by_chunk_type(self):
        sim = VPPSimulator(pp_degree=2, vpp_degree=2, num_acc_steps=4)
        self.assertEqual(sim._get_consume_time(0, 0, ChunkType.FORWARD), 1)
        self.assertEqual(sim._get_consume_time(0, 0, ChunkType.BUBBLE), 1)
        self.assertEqual(sim._get_consume_time(0, 0, ChunkType.BACKWARD), 2)

    def test_cost_ignores_rank_and_step(self):
        sim = VPPSimulator(pp_degree=2, vpp_degree=2, num_acc_steps=4)
        # Backward is always 2 regardless of virtual_pp_rank / acc_step.
        for vpr in range(2):
            for acc in range(4):
                self.assertEqual(
                    sim._get_consume_time(vpr, acc, ChunkType.BACKWARD), 2
                )
                self.assertEqual(
                    sim._get_consume_time(vpr, acc, ChunkType.FORWARD), 1
                )


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestChunkIdentityAndRepr(unittest.TestCase):
    """``Chunk.layer_id`` and ``Chunk.__str__`` rendering.

    layer_id = virtual_pp_rank * pp_degree + stage_id for compute chunks, and
    None for bubbles. The string form embeds the type letter, layer id and a
    1-based acc step for compute chunks; bubbles render only their (start, end)
    span wrapped in literal parentheses.
    """

    def test_compute_chunk_layer_id(self):
        # virtual_pp_rank=1, pp_degree=2, stage_id=0 -> 1*2 + 0 = 2
        chunk = Chunk(
            virtual_pp_rank=1,
            acc_step=0,
            pp_degree=2,
            vpp_degree=2,
            stage_id=0,
            chunk_type=ChunkType.FORWARD,
            start=0,
            end=1,
        )
        self.assertEqual(chunk.layer_id, 2)
        # "F" + layer_id(2) + "_" + (acc_step+1=1) + tuple(start, end)
        self.assertEqual(str(chunk), "F2_1(0, 1)")

    def test_compute_chunk_layer_id_uses_stage_offset(self):
        # Same rank, stage_id=1 -> 1*2 + 1 = 3; acc_step=2 -> shown as 3.
        chunk = Chunk(
            virtual_pp_rank=1,
            acc_step=2,
            pp_degree=2,
            vpp_degree=2,
            stage_id=1,
            chunk_type=ChunkType.BACKWARD,
            start=4,
            end=6,
        )
        self.assertEqual(chunk.layer_id, 3)
        self.assertEqual(str(chunk), "B3_3(4, 6)")

    def test_bubble_chunk_has_no_layer_id(self):
        chunk = Chunk(
            virtual_pp_rank=0,
            acc_step=0,
            pp_degree=2,
            vpp_degree=2,
            stage_id=0,
            chunk_type=ChunkType.BUBBLE,
            start=3,
            end=5,
        )
        self.assertIsNone(chunk.layer_id)
        self.assertEqual(str(chunk), "Z((3, 5))")


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestScheduleWithoutBubbleOrder(unittest.TestCase):
    """Full per-stage chunk ORDER produced by ``_schedule_without_bubble``.

    The expected sequences below are traced chunk-by-chunk for pp=2, vpp=2,
    acc=4 by replaying the warmup / steady / cooldown loops of the interleaved
    schedule; each entry is
    ``(chunk_type, virtual_pp_rank, acc_step, layer_id)``. This is a hand trace,
    not a copy of the simulator's own emission.
    """

    # stage 0: warmup=4, steady=4, cooldown=4 (16 chunks total).
    STAGE0 = [
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
    # stage 1: warmup=2, steady=6, cooldown=2 (16 chunks total).
    STAGE1 = [
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

    def _actual_stage(self, sim, stage_id):
        return [
            (c.chunk_type.value, c.virtual_pp_rank, c.acc_step, c.layer_id)
            for c in sim.schedule_table[stage_id]
        ]

    def test_stage_order_matches_hand_trace(self):
        sim = VPPSimulator(pp_degree=2, vpp_degree=2, num_acc_steps=4)
        sim._schedule_without_bubble()
        self.assertEqual(len(sim.schedule_table), 2)
        self.assertEqual(self._actual_stage(sim, 0), self.STAGE0)
        self.assertEqual(self._actual_stage(sim, 1), self.STAGE1)

    def test_each_layer_runs_every_acc_step_exactly_once(self):
        # Independent multiset invariant: across all stages, every layer id must
        # own exactly one FORWARD and one BACKWARD chunk per acc step (0..3),
        # each acc step appearing exactly once. This catches dropped/duplicated
        # micro steps that a prefix-only check would miss.
        sim = VPPSimulator(pp_degree=2, vpp_degree=2, num_acc_steps=4)
        sim._schedule_without_bubble()
        forward = {}
        backward = {}
        for stage in sim.schedule_table:
            for c in stage:
                bucket = (
                    forward if c.chunk_type is ChunkType.FORWARD else backward
                )
                bucket.setdefault(c.layer_id, []).append(c.acc_step)
        self.assertEqual(set(forward), {0, 1, 2, 3})
        self.assertEqual(set(backward), {0, 1, 2, 3})
        for layer_id in range(4):
            self.assertEqual(sorted(forward[layer_id]), [0, 1, 2, 3])
            self.assertEqual(sorted(backward[layer_id]), [0, 1, 2, 3])

    def test_public_schedule_preserves_order(self):
        # The public entry runs _schedule_without_bubble then bubble/timing
        # passes; those passes must not reorder or add/drop compute chunks.
        sim = VPPSimulator(pp_degree=2, vpp_degree=2, num_acc_steps=4)
        sim.schedule()
        self.assertTrue(sim._is_scheduled)
        self.assertEqual(self._actual_stage(sim, 0), self.STAGE0)
        self.assertEqual(self._actual_stage(sim, 1), self.STAGE1)


if __name__ == "__main__":
    unittest.main()
