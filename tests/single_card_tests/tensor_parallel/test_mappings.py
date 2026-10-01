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

"""CPU behavior tests for ``paddlefleet.tensor_parallel.mappings``.

Repository module: "分布式训练" (tensor parallel). This file targets only the
CPU-observable control flow of the TP mapping primitives -- the decisions that
do NOT require a real >1-rank process group:

  * the ``world_size == 1`` fast paths (genuine single-rank pass-through);
  * the ``group is None`` / non-divisibility assertion guards;
  * the local split index math of ``_split_along_last_dim`` /
    ``_split_along_first_dim`` (pure slicing, no collective); and
  * the autograd ``Function`` forward dispatch (``group is None`` pass-through
    and routing into the local split), asserted on hand-derived slices.

The genuine collective paths (all_reduce / all_gather / reduce_scatter with
world_size > 1) are intentionally NOT exercised here: they require a real
multi-rank process group and belong in ``tests/multi_card_tests``. Faking
world_size plus mocking the collective to assert only "was called" would prove
nothing about split sizes, peer selection, or cross-rank reduction, so it is
avoided. Paddle is not installed in the no-card env, hence the honest skip.
"""

import unittest

try:
    import paddle

    from paddlefleet.tensor_parallel.mappings import (
        _CopyToModelParallelRegion,
        _gather_along_first_dim,
        _gather_along_last_dim,
        _GatherFromModelParallelRegion,
        _GatherFromSequenceParallelRegion,
        _reduce,
        _reduce_scatter_along_first_dim,
        _reduce_scatter_along_last_dim,
        _ReduceFromModelParallelRegion,
        _ReduceScatterToSequenceParallelRegion,
        _ScatterToModelParallelRegion,
        _ScatterToSequenceParallelRegion,
        _split_along_first_dim,
        _split_along_last_dim,
    )

    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False

_SKIP_REASON = "paddle is not installed in this environment"


class _Group:
    """Minimal real stand-in for a process group value object.

    The tested single-rank / index-math paths only *read* topology scalars
    (``world_size``, ``nranks``, ``rank``, ``ranks``); they never invoke a
    collective on it. This is a plain data holder, not a mocked collective.
    """

    def __init__(self, world_size, rank=0):
        self.world_size = world_size
        self.nranks = world_size
        self.rank = rank
        self.ranks = list(range(world_size))


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestReduce(unittest.TestCase):
    """_reduce: guard + single-rank pass-through (no collective here)."""

    def test_none_group_asserts(self):
        with self.assertRaises(AssertionError):
            _reduce(paddle.zeros([2, 4], dtype="float32"), None)

    def test_world_size_one_returns_same_object(self):
        x = paddle.arange(8, dtype="float32").reshape([2, 4])
        # world_size == 1 must bypass all_reduce and return the input as-is.
        self.assertIs(_reduce(x, _Group(1)), x)


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestSplitAlongLastDim(unittest.TestCase):
    """_split_along_last_dim: guard, bypass, and local per-rank slice math."""

    def test_none_group_asserts(self):
        with self.assertRaises(AssertionError):
            _split_along_last_dim(paddle.zeros([2, 4], dtype="float32"), None)

    def test_world_size_one_returns_same_object(self):
        x = paddle.arange(8, dtype="float32").reshape([2, 4])
        self.assertIs(_split_along_last_dim(x, _Group(1)), x)

    def test_each_rank_gets_its_own_last_dim_chunk(self):
        # x = [[0,1,2,3],[4,5,6,7]]; ws=2 splits last dim into cols {0,1}|{2,3}.
        x = paddle.arange(8, dtype="float32").reshape([2, 4])
        rank0 = _split_along_last_dim(x, _Group(2, rank=0))
        rank1 = _split_along_last_dim(x, _Group(2, rank=1))
        self.assertEqual(rank0.tolist(), [[0.0, 1.0], [4.0, 5.0]])
        self.assertEqual(rank1.tolist(), [[2.0, 3.0], [6.0, 7.0]])


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestSplitAlongFirstDim(unittest.TestCase):
    """_split_along_first_dim: guard, bypass, divisibility, per-rank rows."""

    def test_none_group_asserts(self):
        with self.assertRaises(AssertionError):
            _split_along_first_dim(paddle.zeros([4, 3], dtype="float32"), None)

    def test_world_size_one_returns_same_object(self):
        x = paddle.arange(12, dtype="float32").reshape([4, 3])
        self.assertIs(_split_along_first_dim(x, _Group(1)), x)

    def test_non_divisible_first_dim_asserts(self):
        x = paddle.arange(12, dtype="float32").reshape([4, 3])
        with self.assertRaises(AssertionError):
            _split_along_first_dim(x, _Group(3))

    def test_each_rank_gets_its_own_row_block(self):
        # rows 0..3; ws=2 -> rank0 keeps rows {0,1}, rank1 keeps rows {2,3}.
        x = paddle.arange(12, dtype="float32").reshape([4, 3])
        rank0 = _split_along_first_dim(x, _Group(2, rank=0))
        rank1 = _split_along_first_dim(x, _Group(2, rank=1))
        self.assertEqual(rank0.tolist(), [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
        self.assertEqual(rank1.tolist(), [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]])


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestGatherBypassAndGuards(unittest.TestCase):
    """world_size==1 bypass and None guards for the gather / RS helpers."""

    def test_gather_last_dim_world_size_one_passthrough(self):
        x = paddle.arange(8, dtype="float32").reshape([2, 4])
        self.assertIs(_gather_along_last_dim(x, _Group(1)), x)

    def test_gather_first_dim_none_group_asserts(self):
        with self.assertRaises(AssertionError):
            _gather_along_first_dim(paddle.zeros([2, 4], dtype="float32"), None)

    def test_gather_first_dim_world_size_one_passthrough(self):
        x = paddle.arange(8, dtype="float32").reshape([2, 4])
        self.assertIs(_gather_along_first_dim(x, _Group(1)), x)

    def test_reduce_scatter_first_dim_none_group_asserts(self):
        with self.assertRaises(AssertionError):
            _reduce_scatter_along_first_dim(
                paddle.zeros([4, 8], dtype="float32"), None
            )

    def test_reduce_scatter_first_dim_world_size_one_passthrough(self):
        x = paddle.arange(32, dtype="float32").reshape([4, 8])
        self.assertIs(_reduce_scatter_along_first_dim(x, _Group(1)), x)

    def test_reduce_scatter_first_dim_non_divisible_asserts(self):
        # dim0 == 4 is not divisible by world_size 3 -> guard fires before
        # any collective is attempted.
        x = paddle.arange(32, dtype="float32").reshape([4, 8])
        with self.assertRaises(AssertionError):
            _reduce_scatter_along_first_dim(x, _Group(3))

    def test_reduce_scatter_last_dim_non_divisible_asserts(self):
        # last dim 4 is not divisible by world_size 3 -> guard fires first.
        x = paddle.arange(8, dtype="float32").reshape([2, 4])
        with self.assertRaises(AssertionError):
            _reduce_scatter_along_last_dim(x, _Group(3))


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestReduceScatterLastDimSingleRankBug(unittest.TestCase):
    """Documents a real production bug via an independent expected contract."""

    @unittest.expectedFailure
    def test_world_size_one_should_be_identity(self):
        # Correct behaviour: reduce-scatter across a single rank is a no-op,
        # so the output must equal the input content (as every sibling
        # world_size==1 fast path does). _reduce_scatter_along_last_dim has
        # no such fast path and instead calls
        #   paddle.split(input_, split_size_or_sections=..., dim=1)
        #   paddle.concat(split_tensors, dim=0)
        # Paddle's real signatures are paddle.split(x, num_or_sections,
        # axis=0) and paddle.concat(x, axis=0); the torch-style
        # ``split_size_or_sections`` / ``dim`` kwargs raise TypeError even
        # for world_size == 1. Asserting the correct contract here; marked
        # expectedFailure so a future fix surfaces without editing production.
        x = paddle.arange(8, dtype="float32").reshape([2, 4])
        out = _reduce_scatter_along_last_dim(x, _Group(1))
        self.assertEqual(out.tolist(), x.tolist())


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestAutogradForwardDispatch(unittest.TestCase):
    """autograd Function.forward routing observable without a collective."""

    def test_copy_forward_is_content_passthrough(self):
        # _CopyToModelParallelRegion.forward always returns the input content
        # (the all-reduce lives in backward only).
        x = paddle.arange(8, dtype="float32").reshape([2, 4])
        out = _CopyToModelParallelRegion.apply(x, _Group(1))
        self.assertEqual(out.tolist(), x.tolist())

    def test_reduce_from_region_bypasses_when_nranks_leq_one(self):
        # forward: `if group is None or group.nranks <= 1: return input_`.
        x = paddle.arange(8, dtype="float32").reshape([2, 4])
        out = _ReduceFromModelParallelRegion.apply(x, _Group(1))
        self.assertEqual(out.tolist(), x.tolist())

    def test_scatter_to_mp_none_group_passthrough(self):
        x = paddle.arange(8, dtype="float32").reshape([2, 4])
        out = _ScatterToModelParallelRegion.apply(x, None)
        self.assertEqual(out.tolist(), x.tolist())

    def test_scatter_to_mp_routes_into_last_dim_split(self):
        # With a real 2-rank group value, forward must return this rank's
        # last-dim chunk -- the same local slice as _split_along_last_dim.
        x = paddle.arange(8, dtype="float32").reshape([2, 4])
        out = _ScatterToModelParallelRegion.apply(x, _Group(2, rank=1))
        self.assertEqual(out.tolist(), [[2.0, 3.0], [6.0, 7.0]])

    def test_gather_from_mp_none_group_passthrough(self):
        x = paddle.arange(8, dtype="float32").reshape([2, 4])
        out = _GatherFromModelParallelRegion.apply(x, None)
        self.assertEqual(out.tolist(), x.tolist())

    def test_scatter_to_sp_none_group_passthrough(self):
        x = paddle.arange(12, dtype="float32").reshape([4, 3])
        out = _ScatterToSequenceParallelRegion.apply(x, None)
        self.assertEqual(out.tolist(), x.tolist())

    def test_scatter_to_sp_routes_into_first_dim_split(self):
        # forward routes into _split_along_first_dim for a real 2-rank group.
        x = paddle.arange(12, dtype="float32").reshape([4, 3])
        out = _ScatterToSequenceParallelRegion.apply(x, _Group(2, rank=1))
        self.assertEqual(out.tolist(), [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]])

    def test_gather_from_sp_none_group_passthrough(self):
        x = paddle.arange(8, dtype="float32").reshape([2, 4])
        out = _GatherFromSequenceParallelRegion.apply(
            x, None, True, None, False
        )
        self.assertEqual(out.tolist(), x.tolist())

    def test_reduce_scatter_to_sp_none_group_passthrough(self):
        x = paddle.arange(8, dtype="float32").reshape([2, 4])
        out = _ReduceScatterToSequenceParallelRegion.apply(x, None)
        self.assertEqual(out.tolist(), x.tolist())


if __name__ == "__main__":
    unittest.main()
