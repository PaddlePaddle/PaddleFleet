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

"""CPU-only behavior tests for the *assembled schedule* produced by
``paddlefleet.pipeline_parallel.vpp_simulator``.

The simulator is device-independent pure Python. This file deliberately covers
a slice DISTINCT from the sibling
``test_pipeline_parallel_withinterleave_fthenb.py`` (which validates the
per-step ``_get_virtual_pp_rank`` interleaving and ``_get_warmup_and_steady_steps``
sizing helpers in isolation). Here we exercise the pieces that consume those
helpers and turn them into a concrete schedule:

* ``_get_consume_time`` - the FORWARD/BUBBLE=1, BACKWARD=2 cost model.
* ``_barrier_two_chunk`` - start-time max plus the same-type guard.
* ``_get_preorder_chunk`` / ``_find_preorder_chunk_from_stage`` - the cross-stage
  data dependency lookup, including the ValueError when no predecessor exists.
* ``_schedule_without_bubble`` - the full ordered (type, layer_id, acc_step)
  sequence each stage executes, hand-traced below from the warmup/steady/cooldown
  phase definition.
* ``schedule`` - structural timeline invariants (per-chunk duration matches the
  cost model; chunks within a stage are monotonic and non-overlapping).
* ``compute_bubble_rate`` - the exec-time accounting.
* ``PPChunkRecorder`` and the ``set/get_global_pp_chunk_recorder`` module state.

Every expected value is hand-derived from the algorithm and written as an
explicit literal; nothing is read back from the code under test. The makespan
used only inside the bubble-rate formula-consistency check is taken from the
schedule as a shared input (documented at the call site) - the load-bearing
compute total there is the hand-derived constant.

Out of scope (needs a real multi-card pipeline / process group, not faked
here): actual tensor send/recv, cross-rank gradient reduction, and the wall
clock timing of the barrier phase. Only local ordering / accounting is checked.
"""

import unittest

try:
    from paddlefleet.pipeline_parallel.vpp_simulator import (
        Chunk,
        ChunkType,
        PPChunkRecorder,
        VPPSimulator,
        get_global_pp_recorder,
        set_global_pp_chunk_recorder,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # honest dependency probe
    Chunk = None
    ChunkType = None
    PPChunkRecorder = None
    VPPSimulator = None
    get_global_pp_recorder = None
    set_global_pp_chunk_recorder = None
    _IMPORT_ERROR = exc

_HAS_DEPS = _IMPORT_ERROR is None
_SKIP_REASON = (
    "paddlefleet.pipeline_parallel.vpp_simulator not importable on this CPU "
    f"host (its module pulls in matplotlib/numpy and the package __init__ "
    f"pulls in paddle): {_IMPORT_ERROR!r}"
)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestConsumeTime(unittest.TestCase):
    """The cost model charges 1 tick for FORWARD/BUBBLE and 2 for BACKWARD."""

    def test_forward_and_bubble_cost_one_backward_costs_two(self):
        s = VPPSimulator(2, 2, 4)
        # Cost depends only on chunk_type; vpp_rank / acc_step are ignored.
        self.assertEqual(s._get_consume_time(0, 0, ChunkType.FORWARD), 1)
        self.assertEqual(s._get_consume_time(1, 3, ChunkType.FORWARD), 1)
        self.assertEqual(s._get_consume_time(0, 0, ChunkType.BUBBLE), 1)
        self.assertEqual(s._get_consume_time(0, 0, ChunkType.BACKWARD), 2)
        self.assertEqual(s._get_consume_time(1, 2, ChunkType.BACKWARD), 2)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestBarrierTwoChunk(unittest.TestCase):
    """``_barrier_two_chunk`` lifts c1.start to max(c1.start, c2.start)."""

    def test_earlier_chunk_start_is_pulled_forward(self):
        s = VPPSimulator(2, 2, 4)
        c1 = Chunk(0, 0, 2, 2, 0, ChunkType.FORWARD, start=1, end=2)
        c2 = Chunk(0, 0, 2, 2, 1, ChunkType.FORWARD, start=5, end=6)
        s._barrier_two_chunk(c1, c2)
        self.assertEqual(c1.start, 5)  # raised to c2.start
        self.assertEqual(c1.end, 2)  # end is NOT recomputed by this method
        self.assertEqual(c2.start, 5)  # c2 is left untouched
        self.assertEqual(c2.end, 6)

    def test_already_later_start_is_kept(self):
        s = VPPSimulator(2, 2, 4)
        c1 = Chunk(0, 0, 2, 2, 0, ChunkType.BACKWARD, start=9, end=11)
        c2 = Chunk(0, 0, 2, 2, 1, ChunkType.BACKWARD, start=3, end=5)
        s._barrier_two_chunk(c1, c2)
        self.assertEqual(c1.start, 9)  # max keeps the larger existing value

    def test_mismatched_chunk_types_are_rejected(self):
        s = VPPSimulator(2, 2, 4)
        c1 = Chunk(0, 0, 2, 2, 0, ChunkType.FORWARD, start=1, end=2)
        c2 = Chunk(0, 0, 2, 2, 1, ChunkType.BACKWARD, start=5, end=7)
        with self.assertRaises(AssertionError):
            s._barrier_two_chunk(c1, c2)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestPreorderDependency(unittest.TestCase):
    """``_get_preorder_chunk`` resolves the cross-stage data dependency.

    For pp=2, vpp=2 layer_id == vpp_rank*2 + stage_id, so layer_num = 4 and the
    last layer index is 3. A FORWARD chunk depends on the (layer_id-1) forward
    on the previous stage; a BACKWARD chunk depends on the (layer_id+1) backward
    on the next stage. The first forward layer (0) and last backward layer (3)
    have no predecessor.
    """

    def test_bubble_has_no_preorder(self):
        s = VPPSimulator(2, 2, 4)
        bubble = Chunk(0, 0, 2, 2, 0, ChunkType.BUBBLE, 0, 1)
        self.assertIsNone(s._get_preorder_chunk(bubble))

    def test_forward_layer_zero_has_no_preorder(self):
        s = VPPSimulator(2, 2, 4)
        # vpr=0, stage=0 -> layer_id 0, the pipeline entry point.
        c = Chunk(0, 0, 2, 2, 0, ChunkType.FORWARD, 0, 0)
        self.assertEqual(c.layer_id, 0)
        self.assertIsNone(s._get_preorder_chunk(c))

    def test_backward_last_layer_has_no_preorder(self):
        s = VPPSimulator(2, 2, 4)
        # vpr=1, stage=1 -> layer_id 3 == layer_num - 1, the pipeline exit.
        c = Chunk(1, 0, 2, 2, 1, ChunkType.BACKWARD, 0, 0)
        self.assertEqual(c.layer_id, 3)
        self.assertIsNone(s._get_preorder_chunk(c))

    def test_forward_finds_previous_layer_on_previous_stage(self):
        s = VPPSimulator(2, 2, 4)
        # Target: forward layer 1 (vpr0,stage1), acc 0. Predecessor is the
        # forward at layer 0 (vpr0,stage0) with the same acc, placed on
        # stage (1-1)%2 == 0.
        pred = Chunk(0, 0, 2, 2, 0, ChunkType.FORWARD, 0, 0)
        s.schedule_table[0].append(pred)
        # A decoy with a different acc_step must be ignored.
        decoy = Chunk(0, 1, 2, 2, 0, ChunkType.FORWARD, 0, 0)
        s.schedule_table[0].append(decoy)
        target = Chunk(0, 0, 2, 2, 1, ChunkType.FORWARD, 0, 0)
        self.assertEqual(target.layer_id, 1)
        self.assertIs(s._get_preorder_chunk(target), pred)

    def test_backward_finds_next_layer_on_next_stage(self):
        s = VPPSimulator(2, 2, 4)
        # Target: backward layer 0 (vpr0,stage0), acc 0. Predecessor is the
        # backward at layer 1 (vpr0,stage1) with the same acc, on stage
        # (0+1)%2 == 1.
        pred = Chunk(0, 0, 2, 2, 1, ChunkType.BACKWARD, 0, 0)
        s.schedule_table[1].append(pred)
        target = Chunk(0, 0, 2, 2, 0, ChunkType.BACKWARD, 0, 0)
        self.assertEqual(target.layer_id, 0)
        self.assertIs(s._get_preorder_chunk(target), pred)

    def test_missing_predecessor_raises(self):
        s = VPPSimulator(2, 2, 4)
        # Forward layer 1 needs a layer-0 forward on stage 0; table is empty.
        target = Chunk(0, 0, 2, 2, 1, ChunkType.FORWARD, 0, 0)
        with self.assertRaises(ValueError):
            s._get_preorder_chunk(target)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestScheduleWithoutBubbleOrdering(unittest.TestCase):
    """The order of chunks emitted per stage is fully determined by the
    warmup / steady / cooldown phase construction.

    Hand-trace for pp=2, vpp=2, acc=4 (standard-interleave branch, since
    acc == 2*pp): first_chunk_acc = (4 % 2) + 2 = 2, num_steps = 8.

    forward virtual-pp-rank per global micro-step 0..7 -> [0,0,1,1,0,0,1,1];
    backward rank = 1 - forward rank.
    Phase sizes: stage0 (warmup 4, steady 4), stage1 (warmup 2, steady 6).
    layer_id = vpp_rank*2 + stage_id.

    Each stage schedules exactly num_steps=8 forwards and 8 backwards, and every
    (layer, acc_step) pair for that stage's two layers appears once per
    direction. The lists below are transcribed from that trace.
    """

    def _emitted(self, sim, stage_id):
        return [
            (c.chunk_type.value, c.layer_id, c.acc_step)
            for c in sim.schedule_table[stage_id]
        ]

    def test_stage0_full_order(self):
        sim = VPPSimulator(2, 2, 4)
        sim._schedule_without_bubble()
        expected = [
            ("F", 0, 0),
            ("F", 0, 1),
            ("F", 2, 0),
            ("F", 2, 1),
            ("F", 0, 2),
            ("B", 2, 0),
            ("F", 0, 3),
            ("B", 2, 1),
            ("F", 2, 2),
            ("B", 0, 0),
            ("F", 2, 3),
            ("B", 0, 1),
            ("B", 2, 2),
            ("B", 2, 3),
            ("B", 0, 2),
            ("B", 0, 3),
        ]
        self.assertEqual(self._emitted(sim, 0), expected)

    def test_stage1_full_order(self):
        sim = VPPSimulator(2, 2, 4)
        sim._schedule_without_bubble()
        expected = [
            ("F", 1, 0),
            ("F", 1, 1),
            ("F", 3, 0),
            ("B", 3, 0),
            ("F", 3, 1),
            ("B", 3, 1),
            ("F", 1, 2),
            ("B", 1, 0),
            ("F", 1, 3),
            ("B", 1, 1),
            ("F", 3, 2),
            ("B", 3, 2),
            ("F", 3, 3),
            ("B", 3, 3),
            ("B", 1, 2),
            ("B", 1, 3),
        ]
        self.assertEqual(self._emitted(sim, 1), expected)

    def test_each_stage_covers_every_layer_acc_pair_once_per_direction(self):
        sim = VPPSimulator(2, 2, 4)
        sim._schedule_without_bubble()
        for stage_id, layers in ((0, {0, 2}), (1, {1, 3})):
            forwards = set()
            backwards = set()
            for c in sim.schedule_table[stage_id]:
                key = (c.layer_id, c.acc_step)
                bucket = (
                    forwards if c.chunk_type == ChunkType.FORWARD else backwards
                )
                self.assertNotIn(key, bucket, f"duplicate {c}")
                bucket.add(key)
            expected_pairs = {
                (layer, acc) for layer in layers for acc in range(4)
            }
            self.assertEqual(forwards, expected_pairs)
            self.assertEqual(backwards, expected_pairs)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestScheduledTimeline(unittest.TestCase):
    """``schedule`` fills start/end times without changing chunk order.

    Independent invariants: the ordering is identical to
    ``_schedule_without_bubble``; each chunk's duration equals the cost model
    (FORWARD 1, BACKWARD 2); within a stage chunks never overlap and advance
    monotonically (chunk[i].start >= chunk[i-1].end). This code path never
    emits BUBBLE chunks, so every chunk is FORWARD or BACKWARD.
    """

    def test_order_is_preserved_by_scheduling(self):
        ordered = VPPSimulator(2, 2, 4)
        ordered._schedule_without_bubble()
        expected = [
            [(c.chunk_type.value, c.layer_id, c.acc_step) for c in stage]
            for stage in ordered.schedule_table
        ]

        scheduled = VPPSimulator(2, 2, 4)
        table = scheduled.schedule()
        got = [
            [(c.chunk_type.value, c.layer_id, c.acc_step) for c in stage]
            for stage in table
        ]
        self.assertEqual(got, expected)
        self.assertTrue(scheduled._is_scheduled)

    def test_durations_match_cost_model(self):
        sim = VPPSimulator(2, 2, 4)
        for stage in sim.schedule():
            for c in stage:
                self.assertIn(
                    c.chunk_type, (ChunkType.FORWARD, ChunkType.BACKWARD)
                )
                expected_dur = 1 if c.chunk_type == ChunkType.FORWARD else 2
                self.assertEqual(c.end - c.start, expected_dur, f"{c}")

    def test_chunks_are_monotonic_and_non_overlapping_per_stage(self):
        sim = VPPSimulator(2, 2, 4)
        for stage in sim.schedule():
            for prev, cur in zip(stage, stage[1:]):
                self.assertGreaterEqual(
                    cur.start,
                    prev.end,
                    f"{cur} starts before {prev} finishes",
                )


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestBubbleRateAccounting(unittest.TestCase):
    """``compute_bubble_rate`` credits real compute against the makespan.

    For pp=2, vpp=2, acc=4 each stage runs num_steps=8 forwards (1 tick each)
    and 8 backwards (2 ticks each) = 24 ticks; across pp=2 stages the total
    useful compute is a hand-derived 48 ticks. The makespan (max_end-min_start)
    is a scheduling outcome we take from the table as a shared input; the
    load-bearing quantity checked here is the 48-tick compute total baked into
    the rate.
    """

    def test_bubble_rate_uses_hand_derived_compute_total(self):
        sim = VPPSimulator(2, 2, 4)
        table = sim.schedule()

        # Independent compute total: 2 stages * (8*1 + 8*2) == 48.
        useful = sum(
            c.end - c.start
            for stage in table
            for c in stage
            if c.chunk_type != ChunkType.BUBBLE
        )
        self.assertEqual(useful, 48)

        # Makespan taken from the schedule (not independently hand-traced).
        min_start = min(c.start for stage in table for c in stage)
        max_end = max(c.end for stage in table for c in stage)
        total_time = max_end - min_start
        self.assertGreater(total_time, 0)

        expected_rate = 1.0 - useful / (sim.pp_degree * total_time)
        rate = sim.compute_bubble_rate()
        self.assertGreater(rate, 0.0)  # a real schedule has some bubble
        self.assertLess(rate, 1.0)
        self.assertAlmostEqual(rate, expected_rate, places=9)

    def test_compute_bubble_rate_schedules_lazily(self):
        sim = VPPSimulator(2, 2, 4)
        self.assertFalse(sim._is_scheduled)
        sim.compute_bubble_rate()
        self.assertTrue(sim._is_scheduled)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestPPChunkRecorder(unittest.TestCase):
    """``PPChunkRecorder`` tallies how often each *hidden* layer runs forward.

    Layers are laid out as [head empty | hidden | tail empty]. A layer_id is
    recorded only when head <= layer_id < head + num_hidden; it then bumps
    acc_stamp[layer_id - head]. The tail count is stored but the range check
    never consults it, because head + num_hidden already excludes the tail
    block. record_chunk_forward returns False when it rejects a layer and None
    (implicitly) when it records one; ``step`` clears the tally.
    """

    def _recorder(self, head=2, tail=0, hidden=8):
        return PPChunkRecorder(
            pp_degree=2,
            vpp_degree=2,
            num_acc_steps=4,
            num_hidden_layers=hidden,
            num_empty_layers_add_in_head=head,
            num_empty_layers_add_in_tail=tail,
        )

    def test_init_state(self):
        rec = self._recorder(hidden=8)
        self.assertEqual(rec.num_hidden_layers, 8)
        self.assertEqual(rec.num_empty_layers_add_in_head, 2)
        self.assertEqual(rec.acc_stamp, [0] * 8)

    def test_in_range_layer_increments_shifted_slot(self):
        rec = self._recorder(head=2, hidden=8)
        result = rec.record_chunk_forward(4)  # 4 - head(2) -> slot 2
        self.assertIsNone(result)
        expected = [0] * 8
        expected[2] = 1
        self.assertEqual(rec.acc_stamp, expected)

    def test_last_hidden_layer_is_in_range(self):
        # head=2, hidden=8 -> valid ids [2, 10); id 9 is the last hidden layer
        # and lands in slot 7 (tail param is irrelevant to the bound).
        rec = self._recorder(head=2, tail=2, hidden=8)
        result = rec.record_chunk_forward(9)
        self.assertIsNone(result)
        expected = [0] * 8
        expected[7] = 1
        self.assertEqual(rec.acc_stamp, expected)

    def test_head_empty_layer_is_rejected(self):
        rec = self._recorder(head=2, hidden=8)
        result = rec.record_chunk_forward(1)  # 1 < head(2)
        self.assertIs(result, False)
        self.assertEqual(rec.acc_stamp, [0] * 8)

    def test_layer_at_or_beyond_upper_bound_is_rejected(self):
        rec = self._recorder(head=2, hidden=8)  # upper bound head+hidden == 10
        result = rec.record_chunk_forward(10)
        self.assertIs(result, False)
        self.assertEqual(rec.acc_stamp, [0] * 8)

    def test_repeated_records_accumulate(self):
        rec = self._recorder(head=2, hidden=8)
        rec.record_chunk_forward(4)
        rec.record_chunk_forward(4)
        rec.record_chunk_forward(5)
        expected = [0] * 8
        expected[2] = 2  # id 4 twice
        expected[3] = 1  # id 5 once
        self.assertEqual(rec.acc_stamp, expected)

    def test_step_resets_the_tally(self):
        rec = self._recorder(head=2, hidden=8)
        rec.record_chunk_forward(4)
        rec.record_chunk_forward(9)
        self.assertNotEqual(rec.acc_stamp, [0] * 8)
        rec.step()
        self.assertEqual(rec.acc_stamp, [0] * 8)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestGlobalRecorder(unittest.TestCase):
    """The module-level recorder setter/getter round-trip an object identity."""

    def setUp(self):
        try:
            self._orig = get_global_pp_recorder()
            self._had_orig = True
        except NameError:
            self._had_orig = False

        def _restore():
            if self._had_orig:
                set_global_pp_chunk_recorder(self._orig)

        self.addCleanup(_restore)

    def test_set_then_get_returns_same_object(self):
        sentinel = object()
        set_global_pp_chunk_recorder(sentinel)
        self.assertIs(get_global_pp_recorder(), sentinel)

    def test_set_overwrites_previous_value(self):
        first = object()
        second = object()
        set_global_pp_chunk_recorder(first)
        self.assertIs(get_global_pp_recorder(), first)
        set_global_pp_chunk_recorder(second)
        self.assertIs(get_global_pp_recorder(), second)


if __name__ == "__main__":
    unittest.main()
