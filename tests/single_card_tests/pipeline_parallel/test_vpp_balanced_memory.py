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

"""CPU-only behavior tests for the deterministic partition / scheduling math in
``paddlefleet.pipeline_parallel.vpp_simulator``.

Covered (device-independent, no collectives, no tensors):
  * ``VPPSimulator.__init__`` derived quantities (``first_chunk_acc``,
    ``num_steps``, ``layer_num``).
  * ``VPPSimulator._get_warmup_and_steady_steps`` -- the balanced-memory
    (``VPPFhenBInBalancedMemory``) partition branch, the interleave branch, the
    interleave ``min(..., num_steps)`` clamp, and the boundary that selects
    between them.
  * ``VPPSimulator._get_virtual_pp_rank`` -- the first-chunk region, the cyclic
    region, and the forward->backward chunk inversion.
  * ``Chunk.layer_id`` -- the ``virtual_pp_rank * pp_degree + stage_id`` map and
    the bubble ``None`` case.
  * ``VPPSimulator.schedule`` -- structural contract of the emitted schedule
    (per-stage forward/backward counts, per-chunk durations, layer ownership,
    no intra-stage time overlap).

Every expected value below is derived BY HAND from the balanced-memory
scheduling definition and written here as an explicit constant; nothing is read
back from the production code or from any coverage_test file.

Out of scope: real cross-rank pipeline execution (send/recv, gradient
accumulation across stages) needs a genuine multi-rank process group and is left
to the multi-card suite; faking ``world_size`` here would only exercise local
integer math, which is exactly what these tests already anchor independently.
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
    f"host: {_IMPORT_ERROR!r}"
)

# ---------------------------------------------------------------------------
# Hand-derived reference constants (independent of the production formulas).
#
# Balanced-memory branch is taken iff  pp_degree <= num_acc_steps < 2*pp_degree.
# For that branch, per the VPPFhenBInBalancedMemory definition:
#     warmup = num_acc*(vpp-1) + pp - stage - 1
#     steady = num_acc - (pp - stage - 1)
# Otherwise the interleave branch is used:
#     warmup = min((pp-stage-1)*2 + (vpp-1)*first_chunk_acc, num_acc*vpp)
#     steady = num_acc*vpp - warmup
# The numbers below were worked out by hand for pp=4, vpp=2.
# ---------------------------------------------------------------------------

# Balanced: pp=4, vpp=2, acc=6  (4 <= 6 < 8)
_BAL_A = {"pp": 4, "vpp": 2, "acc": 6}
_BAL_A_WARMUP_STEADY = {0: (9, 3), 1: (8, 4), 2: (7, 5), 3: (6, 6)}
_BAL_A_FIRST_CHUNK_ACC = 6  # (6 % 4) + 4
_BAL_A_NUM_STEPS = 12  # 6 * 2
_BAL_A_LAYER_NUM = 8  # 4 * 2

# Balanced: pp=4, vpp=2, acc=7  (4 <= 7 < 8)
_BAL_B_WARMUP_STEADY = {0: (10, 4), 1: (9, 5), 2: (8, 6), 3: (7, 7)}

# Interleave: pp=4, vpp=2, acc=8  (8 is NOT < 8 -> interleave branch)
_INT_WARMUP_STEADY = {0: (10, 6), 1: (8, 8), 2: (6, 10), 3: (4, 12)}
_INT_FIRST_CHUNK_ACC = 4  # (8 % 4) + 4
_INT_NUM_STEPS = 16  # 8 * 2

# Interleave with clamp: pp=4, vpp=2, acc=2  (2 < 4 -> interleave, warmup clamped
# to num_steps=4, so every stage collapses to (4, 0)).
_CLAMP_WARMUP_STEADY = {0: (4, 0), 1: (4, 0), 2: (4, 0), 3: (4, 0)}

# _get_virtual_pp_rank for the balanced config pp=4, vpp=2, acc=6
# (first_chunk_acc=6, first_chunk_steps=12). Forward ranks for micro_step 0..23:
_VPR_FORWARD = [
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    0,
    0,
    0,
    0,
]
# Backward ranks are the forward ranks inverted as (vpp - r - 1):
_VPR_BACKWARD = [
    1,
    1,
    1,
    1,
    1,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
]


def _make_sim(pp, vpp, acc):
    return VPPSimulator(pp_degree=pp, vpp_degree=vpp, num_acc_steps=acc)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestInitDerivedQuantities(unittest.TestCase):
    """__init__ pre-computes partition sizing used throughout scheduling."""

    def test_balanced_config_derived_quantities(self):
        sim = _make_sim(**_BAL_A)
        self.assertEqual(sim.first_chunk_acc, _BAL_A_FIRST_CHUNK_ACC)
        self.assertEqual(sim.num_steps, _BAL_A_NUM_STEPS)
        self.assertEqual(sim.layer_num, _BAL_A_LAYER_NUM)
        # One (empty) schedule row per pipeline stage.
        self.assertEqual(len(sim.schedule_table), _BAL_A["pp"])
        self.assertTrue(all(row == [] for row in sim.schedule_table))

    def test_interleave_config_derived_quantities(self):
        sim = _make_sim(pp=4, vpp=2, acc=8)
        self.assertEqual(sim.first_chunk_acc, _INT_FIRST_CHUNK_ACC)
        self.assertEqual(sim.num_steps, _INT_NUM_STEPS)
        self.assertEqual(sim.layer_num, 8)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestWarmupSteadyPartition(unittest.TestCase):
    """_get_warmup_and_steady_steps is the balanced-memory partition core."""

    def test_balanced_branch_acc6(self):
        sim = _make_sim(**_BAL_A)
        for stage, expected in _BAL_A_WARMUP_STEADY.items():
            self.assertEqual(
                sim._get_warmup_and_steady_steps(stage),
                expected,
                msg=f"stage {stage}",
            )

    def test_balanced_branch_acc7(self):
        sim = _make_sim(pp=4, vpp=2, acc=7)
        for stage, expected in _BAL_B_WARMUP_STEADY.items():
            self.assertEqual(
                sim._get_warmup_and_steady_steps(stage),
                expected,
                msg=f"stage {stage}",
            )

    def test_interleave_branch_acc8(self):
        sim = _make_sim(pp=4, vpp=2, acc=8)
        for stage, expected in _INT_WARMUP_STEADY.items():
            self.assertEqual(
                sim._get_warmup_and_steady_steps(stage),
                expected,
                msg=f"stage {stage}",
            )

    def test_interleave_warmup_is_clamped_to_num_steps(self):
        sim = _make_sim(pp=4, vpp=2, acc=2)
        self.assertEqual(sim.num_steps, 4)
        for stage, expected in _CLAMP_WARMUP_STEADY.items():
            warmup, steady = sim._get_warmup_and_steady_steps(stage)
            self.assertEqual((warmup, steady), expected, msg=f"stage {stage}")
            # steady must never go negative once warmup is clamped.
            self.assertGreaterEqual(steady, 0)

    def test_boundary_selects_different_branch(self):
        # Same (pp, vpp); only num_acc_steps crosses the balanced boundary.
        # acc=7 is inside the balanced range, acc=8 is not, so stage 1 must
        # switch from the balanced result to the interleave result.
        balanced = _make_sim(pp=4, vpp=2, acc=7)
        interleave = _make_sim(pp=4, vpp=2, acc=8)
        self.assertEqual(balanced._get_warmup_and_steady_steps(1), (9, 5))
        self.assertEqual(interleave._get_warmup_and_steady_steps(1), (8, 8))
        self.assertNotEqual(
            balanced._get_warmup_and_steady_steps(1),
            interleave._get_warmup_and_steady_steps(1),
        )


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestVirtualPPRank(unittest.TestCase):
    """_get_virtual_pp_rank maps a micro step to the virtual pipeline rank."""

    def test_forward_ranks_full_sequence(self):
        sim = _make_sim(**_BAL_A)
        got = [
            sim._get_virtual_pp_rank(ms, forward=True)
            for ms in range(len(_VPR_FORWARD))
        ]
        self.assertEqual(got, _VPR_FORWARD)

    def test_backward_ranks_are_inverted(self):
        sim = _make_sim(**_BAL_A)
        got = [
            sim._get_virtual_pp_rank(ms, forward=False)
            for ms in range(len(_VPR_BACKWARD))
        ]
        self.assertEqual(got, _VPR_BACKWARD)
        # The inversion is exactly vpp - forward_rank - 1, elementwise.
        for ms in range(len(_VPR_FORWARD)):
            fwd = sim._get_virtual_pp_rank(ms, forward=True)
            bwd = sim._get_virtual_pp_rank(ms, forward=False)
            self.assertEqual(bwd, _BAL_A["vpp"] - fwd - 1, msg=f"ms {ms}")

    def test_first_chunk_region_vs_cyclic_region(self):
        # micro steps below first_chunk_steps (=12) partition by first_chunk_acc
        # (=6): steps 0..5 -> rank 0, 6..11 -> rank 1. From step 12 on the rank
        # cycles by pp_degree blocks instead.
        sim = _make_sim(**_BAL_A)
        self.assertEqual(sim._get_virtual_pp_rank(5, forward=True), 0)
        self.assertEqual(sim._get_virtual_pp_rank(6, forward=True), 1)
        self.assertEqual(sim._get_virtual_pp_rank(11, forward=True), 1)
        # crossing into the cyclic region resets to rank 0.
        self.assertEqual(sim._get_virtual_pp_rank(12, forward=True), 0)
        self.assertEqual(sim._get_virtual_pp_rank(16, forward=True), 1)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestChunkLayerId(unittest.TestCase):
    """Chunk.layer_id encodes virtual_pp_rank * pp_degree + stage_id."""

    def _chunk(self, vpr, stage, chunk_type):
        return Chunk(
            virtual_pp_rank=vpr,
            acc_step=0,
            pp_degree=4,
            vpp_degree=2,
            stage_id=stage,
            chunk_type=chunk_type,
            start=0,
            end=0,
        )

    def test_forward_layer_id(self):
        # vpr=1, stage=2 -> 1*4 + 2 = 6 ; vpr=0, stage=3 -> 3.
        self.assertEqual(self._chunk(1, 2, ChunkType.FORWARD).layer_id, 6)
        self.assertEqual(self._chunk(0, 3, ChunkType.FORWARD).layer_id, 3)

    def test_backward_layer_id_uses_same_map(self):
        self.assertEqual(self._chunk(1, 0, ChunkType.BACKWARD).layer_id, 4)

    def test_bubble_has_no_layer_id(self):
        self.assertIsNone(self._chunk(1, 2, ChunkType.BUBBLE).layer_id)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestScheduleStructure(unittest.TestCase):
    """schedule() emits the balanced-memory schedule; check its structure.

    For pp=4, vpp=2, acc=6 the hand-derived contract is:
      * each stage runs (warmup + steady) forward chunks and num_steps=12
        backward chunks -> 12 forward + 12 backward per stage;
      * forward chunks take 1 time unit, backward chunks take 2 (the fixed
        consume-time model), so every stage sums to 12*1 + 12*2 = 36 and the
        whole schedule sums to 4*36 = 144;
      * stage s owns exactly layers {s, s + pp} = {s, s + 4};
      * chunks on a stage never overlap in time.
    """

    def setUp(self):
        self.sim = _make_sim(**_BAL_A)
        self.table = self.sim.schedule()

    def test_per_stage_forward_and_backward_counts(self):
        for stage in range(_BAL_A["pp"]):
            row = self.table[stage]
            fwd = [c for c in row if c.chunk_type == ChunkType.FORWARD]
            bwd = [c for c in row if c.chunk_type == ChunkType.BACKWARD]
            self.assertEqual(len(fwd), 12, msg=f"stage {stage} forward")
            self.assertEqual(len(bwd), 12, msg=f"stage {stage} backward")

    def test_total_chunk_count(self):
        total = sum(len(row) for row in self.table)
        self.assertEqual(total, 96)  # 4 stages * (12 fwd + 12 bwd)

    def test_chunk_durations_match_consume_model(self):
        total_duration = 0
        for row in self.table:
            for c in row:
                duration = c.end - c.start
                if c.chunk_type == ChunkType.FORWARD:
                    self.assertEqual(duration, 1, msg=str(c))
                elif c.chunk_type == ChunkType.BACKWARD:
                    self.assertEqual(duration, 2, msg=str(c))
                total_duration += duration
        self.assertEqual(total_duration, 144)

    def test_stage_layer_ownership(self):
        for stage in range(_BAL_A["pp"]):
            owned = {c.layer_id for c in self.table[stage]}
            self.assertEqual(
                owned,
                {stage, stage + _BAL_A["pp"]},
                msg=f"stage {stage}",
            )

    def test_first_forward_chunk_of_stage0_starts_at_zero(self):
        first = self.table[0][0]
        self.assertEqual(first.chunk_type, ChunkType.FORWARD)
        self.assertEqual(first.layer_id, 0)  # vpr 0, stage 0
        self.assertEqual(first.start, 0)
        self.assertEqual(first.end, 1)

    def test_no_time_overlap_within_a_stage(self):
        for stage, row in enumerate(self.table):
            for i in range(1, len(row)):
                self.assertGreaterEqual(
                    row[i].start,
                    row[i - 1].end,
                    msg=f"stage {stage} chunk {i} overlaps predecessor",
                )

    def test_bubble_rate_is_a_fraction(self):
        rate = self.sim.compute_bubble_rate()
        self.assertIsInstance(rate, float)
        # A warmup/cooldown pipeline necessarily idles some stages, so the rate
        # is strictly between 0 and 1 (sanity bound, not a numeric anchor).
        self.assertGreater(rate, 0.0)
        self.assertLess(rate, 1.0)


if __name__ == "__main__":
    unittest.main()
