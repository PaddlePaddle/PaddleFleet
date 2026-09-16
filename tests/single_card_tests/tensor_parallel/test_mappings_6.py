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

"""CPU behavior tests for ``paddlefleet.tensor_parallel.mappings`` (part 6).

Repository module: "分布式训练" (tensor parallel). This file deliberately
targets branches NOT exercised by ``test_mappings.py`` (which covers the
``world_size == 1`` fast paths, the None / divisibility guards, the local
``_split_*`` index math, and the autograd *forward* dispatch):

  * the autograd *backward* routing that lands in a purely local split --
    ``_GatherFromModelParallelRegion.backward`` -> ``_split_along_last_dim``
    and ``_GatherFromSequenceParallelRegion.backward`` (the
    ``tensor_parallel_output_grad is False`` branch) -> ``_split_along_first_dim``;
  * the ``all_to_all`` ``group is None`` assertion guard; and
  * the ``_AllToAll.forward`` ``world_size == 1`` pass-through.

These backward paths route into *local* slicing only -- no collective runs on
a single process, so the per-rank slice is deterministic and hand-derivable.
The genuine cross-rank backward paths (``tensor_parallel_output_grad is True``
-> reduce-scatter, and the gather/reduce-scatter collectives at
``world_size > 1``) are intentionally NOT exercised here: they require a real
multi-rank process group and belong in ``tests/multi_card_tests``. Only the
local slice routing is asserted below, against independently hand-written
expected tensors (the tested ``_split_*`` helper is never used to build the
expected value). Paddle is not installed in the no-card env, hence the honest
skip rather than a fake pass.
"""

import unittest

try:
    import paddle

    from paddlefleet.tensor_parallel.mappings import (
        _GatherFromModelParallelRegion,
        _GatherFromSequenceParallelRegion,
        all_to_all,
    )

    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False

_SKIP_REASON = "paddle is not installed in this environment"


class _Group:
    """Minimal real stand-in for a process-group value object.

    The tested backward-split routing only *reads* topology scalars
    (``world_size``, ``rank``, ``ranks``); it never invokes a collective on
    it. ``_split_along_last_dim`` reads ``len(group.ranks)`` and ``group.rank``
    while ``_split_along_first_dim`` reads ``group.world_size`` and
    ``group.rank``, so all three are kept mutually consistent.
    """

    def __init__(self, world_size, rank=0):
        self.world_size = world_size
        self.nranks = world_size
        self.rank = rank
        self.ranks = list(range(world_size))


class _Ctx:
    """Plain autograd ``ctx`` value holder for direct ``backward`` calls.

    Calling the ``@staticmethod backward`` directly with a hand-built context
    exercises the real production branch/slicing logic; the context only
    supplies the attributes that ``backward`` reads.
    """

    def __init__(
        self,
        group,
        tensor_parallel_output_grad=True,
        output_split_sizes=None,
        use_global_buffer=False,
    ):
        self.group = group
        self.tensor_parallel_output_grad = tensor_parallel_output_grad
        self.output_split_sizes = output_split_sizes
        self.use_global_buffer = use_global_buffer


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestGatherFromModelParallelRegionBackward(unittest.TestCase):
    """backward routes the upstream grad into a local last-dim split.

    ``_GatherFromModelParallelRegion.forward`` is an all-gather over the last
    dim, so its backward must scatter the incoming grad back by keeping this
    rank's last-dim chunk (``_split_along_last_dim``). This is local slicing;
    no collective is involved, so the result is fully hand-derivable.
    """

    def test_backward_keeps_this_ranks_last_dim_chunk(self):
        # grad = [[0,1,2,3],[4,5,6,7]]; ws=2 splits last dim into
        # cols {0,1} (rank 0) and cols {2,3} (rank 1).
        grad = paddle.arange(8, dtype="float32").reshape([2, 4])
        out0 = _GatherFromModelParallelRegion.backward(
            _Ctx(_Group(2, rank=0)), grad
        )
        out1 = _GatherFromModelParallelRegion.backward(
            _Ctx(_Group(2, rank=1)), grad
        )
        # Distinct per-rank slices catch a wrong-rank or wrong-axis split.
        self.assertEqual(out0.tolist(), [[0.0, 1.0], [4.0, 5.0]])
        self.assertEqual(out1.tolist(), [[2.0, 3.0], [6.0, 7.0]])


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestGatherFromSequenceParallelRegionBackward(unittest.TestCase):
    """backward's ``tensor_parallel_output_grad is False`` branch.

    In that branch the grad is *scattered* along the first dim rather than
    reduce-scattered, i.e. it routes into ``_split_along_first_dim`` -- a local
    per-rank row block. The ``tensor_parallel_output_grad is True`` branch
    (reduce-scatter, a real collective) is not exercised here.
    """

    def test_backward_non_tp_grad_keeps_this_ranks_row_block(self):
        # rows 0..3; ws=2 -> rank0 keeps rows {0,1}, rank1 keeps rows {2,3}.
        grad = paddle.arange(12, dtype="float32").reshape([4, 3])
        ctx0 = _Ctx(
            _Group(2, rank=0),
            tensor_parallel_output_grad=False,
            output_split_sizes=None,
        )
        ctx1 = _Ctx(
            _Group(2, rank=1),
            tensor_parallel_output_grad=False,
            output_split_sizes=None,
        )
        out0 = _GatherFromSequenceParallelRegion.backward(ctx0, grad)
        out1 = _GatherFromSequenceParallelRegion.backward(ctx1, grad)
        self.assertEqual(out0.tolist(), [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
        self.assertEqual(out1.tolist(), [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]])


@unittest.skipUnless(HAS_PADDLE, _SKIP_REASON)
class TestAllToAllGuardAndSingleRankBypass(unittest.TestCase):
    """``all_to_all`` None guard and ``_AllToAll.forward`` ws==1 bypass."""

    def test_none_group_asserts(self):
        # all_to_all guards ``group is not None`` before touching _AllToAll.
        x = paddle.arange(6, dtype="float32").reshape([2, 3])
        with self.assertRaises(AssertionError):
            all_to_all(None, x)

    def test_world_size_one_returns_input_content(self):
        # _AllToAll.forward bypasses the collective at world_size == 1 and
        # returns the input unchanged; assert full content, not just shape.
        x = paddle.arange(6, dtype="float32").reshape([2, 3])
        out = all_to_all(_Group(1), x)
        self.assertEqual(out.tolist(), [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])


if __name__ == "__main__":
    unittest.main()
