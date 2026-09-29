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

"""Multi-card (PP=4) schedule-math tests for the real paddlefleet VPPSimulator.

VPPSimulator is a pure-Python pipeline *schedule* simulator: given pp_degree,
vpp_degree and num_acc_steps it derives per-stage warmup/steady step counts,
builds a per-stage chunk table (layer_id / acc_step / chunk_type) and assigns
start/end times. It performs no collective communication, so every rank of the
launched PP=4 world runs the identical schedule math. The real 4-way pipeline
world is still initialized and its degree asserted, so the file executes as a
genuine multi-card job under ``paddle.distributed.launch`` (topology PP=4,
MP=1). No claim is made here about cross-rank collective numerics -- the
surface under test is pure schedule math run identically on all four ranks.

All expected values are derived BY HAND from the VPP scheduling definition,
never from the simulator or from the coverage-test source.
"""

import os

# The simulator's module imports matplotlib (used only by the draw_* helpers).
# Force a headless backend so importing it and exercising the broken draw path
# does not require a display. matplotlib is not the surface under test.
os.environ.setdefault("MPLBACKEND", "Agg")

import unittest

import matplotlib.pyplot as plt
from paddle.distributed import fleet

from paddlefleet.pipeline_parallel.vpp_simulator import (
    Chunk,
    ChunkType,
    PPChunkRecorder,
    VPPSimulator,
)

# Topology under test: PP=4, MP=1 (num_gpus=4).
PP_DEGREE = 4
MP_DEGREE = 1


def _init_pipeline_parallel():
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": MP_DEGREE,
        "pp_degree": PP_DEGREE,
    }
    fleet.init(is_collective=True, strategy=strategy)


def setUpModule():
    """Bring up the real 4-way pipeline world once for the module."""
    _init_pipeline_parallel()


class TestLaunchedPipelineWorld(unittest.TestCase):
    """The file must run as a real 4-rank pipeline job, not single process."""

    def test_pipeline_world_is_degree_four(self):
        hcg = fleet.get_hybrid_communicate_group()
        # Ties the simulator's pp_degree=4 to the actually launched topology;
        # a 1-rank / wrong-degree launch would fail here instead of silently
        # passing pure-python math on a single process.
        self.assertEqual(hcg.get_pipe_parallel_world_size(), PP_DEGREE)
        self.assertEqual(hcg.get_model_parallel_world_size(), MP_DEGREE)


class TestChunkLayerId(unittest.TestCase):
    """Chunk.layer_id = virtual_pp_rank * pp_degree + stage_id (non-bubble).

    layer_id drives the whole dependency graph (_get_preorder_chunk), so a
    wrong offset or a swapped rank/stage would reorder the schedule. Bubble
    chunks carry no layer_id.
    """

    def test_layer_id_for_compute_chunks(self):
        # virtual_pp_rank=1, stage_id=2, pp=4 -> 1*4 + 2 = 6.
        fwd = Chunk(
            virtual_pp_rank=1,
            acc_step=0,
            pp_degree=PP_DEGREE,
            vpp_degree=2,
            stage_id=2,
            chunk_type=ChunkType.FORWARD,
            start=0,
            end=0,
        )
        self.assertEqual(fwd.layer_id, 6)
        # virtual_pp_rank=0, stage_id=3 -> 0*4 + 3 = 3.
        bwd = Chunk(
            virtual_pp_rank=0,
            acc_step=1,
            pp_degree=PP_DEGREE,
            vpp_degree=2,
            stage_id=3,
            chunk_type=ChunkType.BACKWARD,
            start=0,
            end=0,
        )
        self.assertEqual(bwd.layer_id, 3)

    def test_bubble_chunk_has_no_layer_id(self):
        bubble = Chunk(
            virtual_pp_rank=1,
            acc_step=0,
            pp_degree=PP_DEGREE,
            vpp_degree=2,
            stage_id=2,
            chunk_type=ChunkType.BUBBLE,
            start=0,
            end=0,
        )
        self.assertIsNone(bubble.layer_id)


class TestWarmupSteadySteps(unittest.TestCase):
    """_get_warmup_and_steady_steps: exact per-stage counts, hand derived.

    Two distinct branches are selected purely by num_acc_steps vs pp_degree,
    so degenerate single-config checks would miss a broken branch selector.
    """

    def test_interleave_branch(self):
        # num_acc_steps=8 >= 2*pp: PipelineParallelWithInterleave branch.
        # num_steps = 8*2 = 16; first_chunk_acc = 8%4 + 4 = 4.
        # warmup(s) = (pp-s-1)*2 + (vpp-1)*first_chunk_acc = (3-s)*2 + 4.
        # steady(s) = num_steps - warmup(s).
        sim = VPPSimulator(pp_degree=PP_DEGREE, vpp_degree=2, num_acc_steps=8)
        expected_warmup = [10, 8, 6, 4]
        expected_steady = [6, 8, 10, 12]
        for stage_id in range(PP_DEGREE):
            warmup, steady = sim._get_warmup_and_steady_steps(stage_id)
            self.assertEqual(warmup, expected_warmup[stage_id])
            self.assertEqual(steady, expected_steady[stage_id])
            # Every stage does the full num_steps of forward+backward work.
            self.assertEqual(warmup + steady, sim.num_steps)

    def test_fthenb_balanced_memory_branch(self):
        # pp <= num_acc_steps < 2*pp (4 <= 4 < 8): VPPFhenBInBalancedMemory.
        # warmup(s) = num_acc*(vpp-1) + pp - s - 1 = 4 + 3 - s = 7 - s.
        # steady(s) = num_acc - (pp - s - 1) = 4 - (3 - s) = 1 + s.
        sim = VPPSimulator(pp_degree=PP_DEGREE, vpp_degree=2, num_acc_steps=4)
        expected_warmup = [7, 6, 5, 4]
        expected_steady = [1, 2, 3, 4]
        for stage_id in range(PP_DEGREE):
            warmup, steady = sim._get_warmup_and_steady_steps(stage_id)
            self.assertEqual(warmup, expected_warmup[stage_id])
            self.assertEqual(steady, expected_steady[stage_id])
            self.assertEqual(warmup + steady, sim.num_steps)


class TestScheduleTableContent(unittest.TestCase):
    """schedule() builds a complete, balanced per-stage chunk table.

    Working config pp=4, vpp=2, num_acc=8 (>= pp, interleave branch). Stage s
    hosts exactly the virtual layers {s, s+pp}; each such layer must be both
    forwarded and backwarded once per accumulation step (acc_steps 0..num_acc-1
    with no loss or duplication). This rejects mis-assigned virtual ranks and
    dropped/duplicated micro-steps that a length-only check would pass.
    """

    def test_schedule_table_is_complete_and_balanced(self):
        pp_degree, vpp_degree, num_acc_steps = PP_DEGREE, 2, 8
        sim = VPPSimulator(
            pp_degree=pp_degree,
            vpp_degree=vpp_degree,
            num_acc_steps=num_acc_steps,
        )
        table = sim.schedule()

        self.assertEqual(len(table), pp_degree)
        self.assertTrue(sim._is_scheduled)

        num_steps = num_acc_steps * vpp_degree  # 16 forwards + 16 backwards
        for stage_id, stage in enumerate(table):
            # _add_bubble adjusts timing but inserts no BUBBLE chunks here.
            self.assertTrue(
                all(c.chunk_type != ChunkType.BUBBLE for c in stage)
            )
            self.assertEqual(len(stage), 2 * num_steps)

            expected_layers = {stage_id, stage_id + pp_degree}
            for chunk_type in (ChunkType.FORWARD, ChunkType.BACKWARD):
                by_layer = {}
                for chunk in stage:
                    if chunk.chunk_type == chunk_type:
                        by_layer.setdefault(chunk.layer_id, []).append(
                            chunk.acc_step
                        )
                self.assertEqual(set(by_layer), expected_layers)
                for layer_id, acc_steps in by_layer.items():
                    self.assertEqual(
                        sorted(acc_steps), list(range(num_acc_steps))
                    )


class TestPPChunkRecorder(unittest.TestCase):
    """PPChunkRecorder maps a raw layer_id onto a real-layer acc counter.

    Empty head/tail layers are out of range and must be rejected; real layers
    accumulate a per-layer count (not a boolean), offset by the head padding;
    step() clears the counters.
    """

    def _recorder(self, head=0, tail=0, hidden=8):
        return PPChunkRecorder(
            pp_degree=PP_DEGREE,
            vpp_degree=2,
            num_acc_steps=4,
            num_hidden_layers=hidden,
            num_empty_layers_add_in_head=head,
            num_empty_layers_add_in_tail=tail,
        )

    def test_initial_acc_stamp_length_and_zeros(self):
        recorder = self._recorder(hidden=8)
        self.assertEqual(recorder.acc_stamp, [0] * 8)

    def test_head_offset_and_range_bounds(self):
        # head=2, hidden=8 -> valid raw layer ids are 2..9 -> acc_stamp[0..7].
        recorder = self._recorder(head=2, tail=1, hidden=8)
        self.assertFalse(recorder.record_chunk_forward(0))
        self.assertFalse(recorder.record_chunk_forward(1))
        self.assertIsNone(recorder.record_chunk_forward(2))  # -> stamp[0]
        self.assertIsNone(recorder.record_chunk_forward(9))  # -> stamp[7]
        # Raw layer 10 >= hidden + head (== 10) is out of range.
        self.assertFalse(recorder.record_chunk_forward(10))
        expected = [0] * 8
        expected[0] = 1
        expected[7] = 1
        self.assertEqual(recorder.acc_stamp, expected)

    def test_forward_accumulates_per_layer(self):
        recorder = self._recorder(head=0, hidden=8)
        recorder.record_chunk_forward(3)
        recorder.record_chunk_forward(3)
        recorder.record_chunk_forward(3)
        self.assertEqual(recorder.acc_stamp[3], 3)
        self.assertFalse(recorder.record_chunk_forward(8))  # out of range
        self.assertEqual(recorder.acc_stamp[3], 3)

    def test_step_resets_counters(self):
        recorder = self._recorder(head=0, hidden=8)
        recorder.record_chunk_forward(0)
        recorder.record_chunk_forward(5)
        self.assertEqual(recorder.acc_stamp[0], 1)
        self.assertEqual(recorder.acc_stamp[5], 1)
        recorder.step()
        self.assertEqual(recorder.acc_stamp, [0] * 8)


class TestKnownScheduleBugs(unittest.TestCase):
    """Locks on two confirmed production defects (production is NOT edited).

    When either defect is fixed the corresponding assertRaises will stop
    firing and the test fails, surfacing that the lock must be removed.
    """

    def test_warmup_clamp_collapses_steady_and_breaks_barrier(self):
        # num_acc_steps=2 < pp_degree=4. The interleave branch computes
        # warmup = (3-s)*2 + (vpp-1)*first_chunk_acc, then clamps it to
        # num_steps = 2*2 = 4, collapsing steady_steps to 0 for every stage.
        # In _barrier the cooldown phase then indexes a BACKWARD chunk where a
        # FORWARD is expected, so _barrier_two_chunk's chunk-type assertion
        # fails. CORRECT behaviour would be a complete schedule for any
        # num_acc_steps >= 1; this currently raises instead.
        sim = VPPSimulator(pp_degree=PP_DEGREE, vpp_degree=2, num_acc_steps=2)
        for stage_id in range(PP_DEGREE):
            _, steady = sim._get_warmup_and_steady_steps(stage_id)
            self.assertEqual(steady, 0)  # documents the clamp collapse
        with self.assertRaises(AssertionError):
            sim.schedule()

    def test_draw_chunks_is_broken(self):
        # draw_chunks() first reaches `plt.cm.get_cmap` (removed in newer
        # matplotlib -> AttributeError) and, where that still resolves, then
        # evaluates `range(self.pp_degree + 1) * stage_height` at the
        # set_yticks call, multiplying a range by a float -> TypeError. Either
        # way the visualization path is dead. Schedule math above is unaffected.
        self.addCleanup(plt.close, "all")
        sim = VPPSimulator(pp_degree=PP_DEGREE, vpp_degree=2, num_acc_steps=8)
        with self.assertRaises((TypeError, AttributeError)):
            sim.draw_chunks()


if __name__ == "__main__":
    unittest.main()
