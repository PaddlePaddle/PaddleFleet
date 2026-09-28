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

"""CPU behavior tests for ``paddlefleet.tensor_parallel.mappings`` (part 7).

Repository module: "分布式训练" (tensor parallel). This file deliberately
covers the all-to-all / all-gather-last-dim / reduce-scatter-last-dim entry
points that the base ``test_mappings.py`` does NOT touch, and only their
CPU-observable control flow (no real >1-rank process group required):

  * ``all_to_all`` wrapper ``group is not None`` guard;
  * ``_AllToAll.forward`` ``world_size == 1`` fast path -- including that the
    split-size arguments are ignored before the bypass returns the input;
  * ``all_to_all`` delegating into that single-rank bypass;
  * ``_AllGatherFromTensorParallelRegion.forward`` ``group is None``
    pass-through vs. routing into ``_gather_along_last_dim`` (whose own
    single-rank fast path is a no-op); and
  * ``_ReduceScatterToTensorParallelRegion.forward`` ``group is None``
    pass-through.

A real all-to-all / all-gather / reduce-scatter with ``world_size > 1`` needs a
genuine multi-rank process group and belongs in ``tests/multi_card_tests``.
Faking ``world_size`` plus mocking the collective to assert only "was called"
would prove nothing about split sizes, peer selection or cross-rank ordering,
so it is avoided (antipattern #13). Inputs are distinguishable ``arange`` data
so a pass-through that zeroed / reordered / reshaped content would be rejected.

One genuine production bug is surfaced via ``expectedFailure``:
``all_to_all_sp2hp`` has an inverted divisibility assertion. Paddle is not
installed in the no-card env, hence the honest skip.
"""

import unittest
from unittest import mock

try:
    import paddle

    from paddlefleet.tensor_parallel.mappings import (
        _AllGatherFromTensorParallelRegion,
        _AllToAll,
        _ReduceScatterToTensorParallelRegion,
        all_to_all,
        all_to_all_sp2hp,
    )

    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False

_SKIP_REASON = "paddle is not installed in this environment"
_GROUP_RESOLVER = (
    "paddlefleet.tensor_parallel.mappings."
    "get_tensor_model_parallel_group_if_none"
)


class _Group:
    """Minimal real stand-in for a process-group value object.

    The tested single-rank / guard paths only *read* topology scalars
    (``world_size``, ``nranks``, ``rank``, ``ranks``); they never invoke a
    collective on it. This is a plain data holder, not a mocked collective.
    """

    def __init__(self, world_size, rank=0):
        self.world_size = world_size
        self.nranks = world_size
        self.rank = rank
        self.ranks = list(range(world_size))


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestAllToAllForward(unittest.TestCase):
    """all_to_all wrapper guard + _AllToAll.forward single-rank bypass."""

    def test_wrapper_rejects_none_group(self):
        # all_to_all: `assert group is not None` fires before any dispatch.
        with self.assertRaises(AssertionError):
            all_to_all(None, paddle.arange(32, dtype="float32").reshape([4, 8]))

    def test_forward_world_size_one_returns_input_content(self):
        # forward: `if world_size == 1: return input` -> exact content back.
        x = paddle.arange(32, dtype="float32").reshape([4, 8])
        out = _AllToAll.apply(_Group(1), x, None, None)
        self.assertEqual(out.tolist(), x.tolist())

    def test_forward_ignores_split_sizes_at_single_rank(self):
        # The world_size == 1 bypass returns the input *before* the branch that
        # consumes output/input split sizes, so non-trivial split args must not
        # change the result. This distinguishes the bypass branch from the
        # split-consuming branch without a real collective.
        x = paddle.arange(32, dtype="float32").reshape([4, 8])
        out = _AllToAll.apply(_Group(1), x, [4], [4])
        self.assertEqual(out.tolist(), x.tolist())

    def test_wrapper_delegates_into_single_rank_bypass(self):
        # all_to_all(group, x) -> _AllToAll.apply(...) -> ws==1 bypass.
        x = paddle.arange(32, dtype="float32").reshape([4, 8])
        out = all_to_all(_Group(1), x)
        self.assertEqual(out.tolist(), x.tolist())


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestAllGatherFromTPRegionForward(unittest.TestCase):
    """_AllGatherFromTensorParallelRegion.forward routing branches."""

    def test_none_group_returns_input_content(self):
        # forward: `if group is None: return input_`.
        x = paddle.arange(8, dtype="float32").reshape([2, 4])
        out = _AllGatherFromTensorParallelRegion.apply(x, None)
        self.assertEqual(out.tolist(), x.tolist())

    def test_single_rank_group_routes_through_gather_noop(self):
        # group is not None -> forward routes into _gather_along_last_dim,
        # whose own `world_size == 1` fast path returns the input unchanged.
        # This exercises the else-branch (a distinct code path from the
        # None-guard) while remaining honest that no cross-rank gather ran.
        x = paddle.arange(8, dtype="float32").reshape([2, 4])
        out = _AllGatherFromTensorParallelRegion.apply(x, _Group(1))
        self.assertEqual(out.tolist(), x.tolist())


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestReduceScatterToTPRegionForward(unittest.TestCase):
    """_ReduceScatterToTensorParallelRegion.forward group-None branch."""

    def test_none_group_returns_input_content(self):
        # forward: `if group is None: return input_`. The world_size == 1
        # else-branch would route into _reduce_scatter_along_last_dim, which
        # carries a separate known Paddle-signature bug already documented in
        # the base test file; it is not re-exercised here.
        x = paddle.arange(8, dtype="float32").reshape([2, 4])
        out = _ReduceScatterToTensorParallelRegion.apply(x, None)
        self.assertEqual(out.tolist(), x.tolist())


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestAllToAllSp2HpInvertedAssertBug(unittest.TestCase):
    """Documents a real production bug in all_to_all_sp2hp."""

    @unittest.expectedFailure
    def test_divisible_last_dim_must_be_accepted(self):
        # Correct contract: a hidden dim that IS divisible by world_size is the
        # VALID case and must be accepted. Production line reads
        #     assert input_.shape[-1] % world_size, (...)
        # which is inverted: `8 % 1 == 0` (and any exact multiple) is falsy, so
        # the assertion fires precisely on the valid, divisible input. At TP=1
        # sp2hp is a shape/content identity, so the expected output equals the
        # input. Marked expectedFailure so a future fix surfaces without
        # editing production. Only the group resolver (a genuine not-under-test
        # collaborator reading global TP state) is mocked; the real sp2hp guard
        # logic runs.
        x = paddle.arange(16, dtype="float32").reshape([2, 8])
        with mock.patch(_GROUP_RESOLVER, return_value=_Group(1)):
            out = all_to_all_sp2hp(x)
        self.assertEqual(out.reshape([2, 8]).tolist(), x.tolist())


if __name__ == "__main__":
    unittest.main()
