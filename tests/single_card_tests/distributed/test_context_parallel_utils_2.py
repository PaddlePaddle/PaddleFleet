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

"""Behavior tests for the balanced (DualChunkSwap) sharding helpers in
``paddlefleet.context_parallel_utils``: ``scatter_balance`` and
``reduce_scatter_any_axis``.

Scope and environment. ``scatter_balance`` performs NO collective communication:
for a given ``(nranks, rank)`` it selects, purely locally, which two slices of
the sequence this rank should keep (one chunk taken from the front, one mirrored
chunk taken from the back). That per-rank shard-selection index math is a genuine
single-process / world-size-independent code path, so it is exercised here with a
plain data-carrying group object and hand-derived expected row indices. These
tests deliberately do NOT claim to verify cross-rank communication, gathering, or
the reduce-scatter numerics -- those require a real multi-rank process group and
belong in the multi-card suite. What is verified: the balanced split keeps the
correct front/back slices per rank, honours the requested axis, and together the
per-rank shards partition the whole sequence with no drops or overlaps; the
single-rank fast path returns an independent clone with identical values; and the
divisibility precondition (which runs before any collective) rejects bad input.
"""

import unittest

_SKIP_REASON = None
try:
    import numpy as np
    import paddle

    from paddlefleet.context_parallel_utils import (
        reduce_scatter_any_axis,
        scatter_balance,
    )
except ImportError as exc:  # local envs may lack a compiled paddle build
    np = None
    paddle = None
    scatter_balance = None
    reduce_scatter_any_axis = None
    _SKIP_REASON = f"paddle / paddlefleet not importable: {exc}"


class _Group:
    """Minimal stand-in for a paddle communication group.

    ``scatter_balance`` and ``reduce_scatter_any_axis`` read only ``nranks`` and
    ``rank`` off the group before deciding what local slice to keep; neither issues
    a collective when the guarded/local paths below are hit. This is a genuine
    data-carrying collaborator, not a mock of the code under test.
    """

    def __init__(self, nranks, rank=0):
        self.nranks = nranks
        self.rank = rank


@unittest.skipUnless(_SKIP_REASON is None, str(_SKIP_REASON))
class TestScatterBalanceLocalShardSelection(unittest.TestCase):
    """scatter_balance selects the correct front+back slices for each rank."""

    def _rows(self, seq_len, width):
        # Distinguishable content: row i is [i*width, ..., i*width + width - 1],
        # so a wrong slice start, swapped chunk, or wrong axis is visible.
        return paddle.arange(seq_len * width, dtype="float32").reshape(
            [seq_len, width]
        )

    def test_two_ranks_partition_sequence_axis0(self):
        # seq_len=8, nranks=2 -> interval = 8 // 2 // 2 = 2.
        # rank 0: front rows [0,1] + back rows [6,7]
        # rank 1: front rows [2,3] + back rows [4,5]
        x = self._rows(8, 3)
        rank0 = scatter_balance(x, group=_Group(nranks=2, rank=0), axis=0)
        rank1 = scatter_balance(x, group=_Group(nranks=2, rank=1), axis=0)

        np.testing.assert_array_equal(rank0.numpy(), x.numpy()[[0, 1, 6, 7]])
        np.testing.assert_array_equal(rank1.numpy(), x.numpy()[[2, 3, 4, 5]])
        # The union of the two shards must reconstruct the whole sequence with
        # no dropped or duplicated rows (the balancing contract).
        union = np.concatenate([rank0.numpy(), rank1.numpy()], axis=0)
        np.testing.assert_array_equal(
            np.sort(union[:, 0]), np.arange(8, dtype="float32") * 3
        )

    def test_four_ranks_each_takes_mirrored_pair(self):
        # seq_len=8, nranks=4 -> interval = 8 // 4 // 2 = 1.
        # rank r keeps front row r and back row (7 - r).
        x = self._rows(8, 2)
        expected = {0: [0, 7], 1: [1, 6], 2: [2, 5], 3: [3, 4]}
        seen = []
        for r, rows in expected.items():
            shard = scatter_balance(x, group=_Group(nranks=4, rank=r), axis=0)
            np.testing.assert_array_equal(shard.numpy(), x.numpy()[rows])
            seen.extend(rows)
        # All four shards jointly cover every row exactly once.
        self.assertEqual(sorted(seen), list(range(8)))

    def test_scatter_honours_non_default_axis(self):
        # Same balancing along axis=1: seq (columns) = 8, nranks=2, interval=2.
        # rank 0 keeps columns [0,1] (front) + [6,7] (back).
        x = paddle.arange(3 * 8, dtype="float32").reshape([3, 8])
        rank0 = scatter_balance(x, group=_Group(nranks=2, rank=0), axis=1)
        np.testing.assert_array_equal(rank0.numpy(), x.numpy()[:, [0, 1, 6, 7]])

    def test_undivisible_sequence_raises_before_any_collective(self):
        # 7 is not divisible by nranks*2 == 4; the guard must reject it.
        x = self._rows(7, 4)
        with self.assertRaises(AssertionError) as ctx:
            scatter_balance(x, group=_Group(nranks=2, rank=0), axis=0)
        self.assertIn("divided exactly", str(ctx.exception))


@unittest.skipUnless(_SKIP_REASON is None, str(_SKIP_REASON))
class TestScatterBalanceSingleRank(unittest.TestCase):
    """The world-size-1 fast path returns an independent copy of the input."""

    def test_returns_independent_clone_with_identical_values(self):
        x = paddle.arange(8 * 3, dtype="float32").reshape([8, 3])
        result = scatter_balance(x, group=_Group(nranks=1), axis=0)

        np.testing.assert_array_equal(result.numpy(), x.numpy())
        self.assertIsNot(result, x)
        # A clone must own independent storage: mutating the source afterward
        # must not leak into the returned tensor.
        before = result.numpy().copy()
        x[0, 0] = -999.0
        np.testing.assert_array_equal(result.numpy(), before)


@unittest.skipUnless(_SKIP_REASON is None, str(_SKIP_REASON))
class TestReduceScatterAnyAxisLocalPaths(unittest.TestCase):
    """reduce_scatter_any_axis single-rank clone and the divisibility guard.

    The multi-rank numeric path issues a real reduce_scatter / alltoall and is
    intentionally not exercised here (needs a genuine process group). Only the
    world-size-1 clone and the precondition check -- both of which run before any
    collective -- are verified.
    """

    def test_single_rank_returns_independent_clone(self):
        x = paddle.arange(6 * 4, dtype="float32").reshape([6, 4])
        result = reduce_scatter_any_axis(x, axis=0, group=_Group(nranks=1))

        np.testing.assert_array_equal(result.numpy(), x.numpy())
        self.assertIsNot(result, x)
        before = result.numpy().copy()
        x[1, 1] = -777.0
        np.testing.assert_array_equal(result.numpy(), before)

    def test_undivisible_axis_raises_before_collective(self):
        # shape[axis]=7 not divisible by nranks=2; guard runs before any comm.
        x = paddle.arange(7 * 4, dtype="float32").reshape([7, 4])
        with self.assertRaises(AssertionError):
            reduce_scatter_any_axis(x, axis=0, group=_Group(nranks=2))

    @unittest.expectedFailure
    def test_guard_message_should_be_a_readable_string(self):
        # BUG (context_parallel_utils.py:293-296): the assert message is written
        # as a *tuple*  ("... can't be ", f"divided exactly ... {parallelism}")
        # because of the stray comma, instead of one concatenated string. The
        # condition still fires correctly, but AssertionError.args[0] is a tuple,
        # so the surfaced message is an unreadable tuple repr rather than a
        # sentence. Asserting the correct contract (a str message) fails today;
        # marked expectedFailure and left for a production fix -- not edited here.
        x = paddle.arange(7 * 4, dtype="float32").reshape([7, 4])
        with self.assertRaises(AssertionError) as ctx:
            reduce_scatter_any_axis(x, axis=0, group=_Group(nranks=2))
        self.assertIsInstance(ctx.exception.args[0], str)


if __name__ == "__main__":
    unittest.main()
