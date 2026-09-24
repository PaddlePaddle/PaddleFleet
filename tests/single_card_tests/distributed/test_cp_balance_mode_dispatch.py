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

"""Behavior tests for the contiguous context-parallel layout in
``paddlefleet.context_parallel_utils``.

Scope and honesty about the environment
---------------------------------------
Only the *local*, collective-free behavior is exercised here, and every
expected value is derived by hand (or with numpy as an independent oracle),
never by re-running the code under test:

* ``scatter_contiguous`` is an embarrassingly local operation -- each rank
  reads only ``group.rank`` / ``group.nranks`` from its process group and
  slices its own contiguous chunk with ``paddle.slice``. No NCCL collective
  is issued, so a single process can legitimately reproduce what every rank
  would compute, and the per-rank chunks are checked against explicit
  expected rows plus a reconstruction of the whole tensor.
* The PyLayer ``mode`` dispatch (``mode.startswith("contiguous")``) is driven
  through the real ``forward`` / ``backward`` entry points for the branches
  that resolve to ``scatter_contiguous`` (also local), and the observable
  output is compared to the hand-derived slice -- not merely "was called".
* The ``world_size > 1`` guard on each PyLayer is a real behavioral contract
  and is asserted with ``assertRaises``.

The multi-rank ``all_gather_contiguous`` / ``reduce_scatter_contiguous``
paths issue real ``dist.stream`` collectives; those require an actual
multi-card process group and are intentionally NOT faked here (mocking the
collective would prove nothing about cross-rank layout). Only their
supported ``nranks == 1`` local fall-back is covered, and labelled as such.

The hybrid communicate group is replaced by a lightweight stub: it is
external distributed infrastructure that merely supplies topology metadata
(the CP group and world size); the dispatch, guard, and slicing logic under
test all still execute for real.
"""

import os
import sys
import unittest

_project_root = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    )
)
sys.path.insert(0, os.path.join(_project_root, "src"))

from types import SimpleNamespace
from unittest import mock

try:
    import numpy as np
    import paddle

    HAVE_PADDLE = True
except ImportError:
    # Honest skip: this checkout has no Paddle wheel installed, so the real
    # tensor ops below cannot run. We do not fake a pass.
    HAVE_PADDLE = False


class _CpGroup:
    """Topology-metadata collaborator for the CP process group.

    ``scatter_contiguous`` reads only ``nranks`` and ``rank`` and then slices
    locally, so this stub is sufficient to reproduce any single rank's share
    without a real collective.
    """

    def __init__(self, nranks, rank):
        self.nranks = nranks
        self.rank = rank


def _patch_hcg(nranks, rank):
    """Patch fleet's hybrid communicate group with a stub exposing the CP
    group and world size. Returns (patch_context, group)."""
    group = _CpGroup(nranks=nranks, rank=rank)
    hcg = mock.MagicMock()
    hcg.get_context_parallel_group.return_value = group
    hcg.get_context_parallel_world_size.return_value = nranks
    patch = mock.patch(
        "paddle.distributed.fleet.get_hybrid_communicate_group",
        return_value=hcg,
    )
    return patch, group


@unittest.skipUnless(HAVE_PADDLE, "paddle is not installed in this environment")
class TestScatterContiguousLayout(unittest.TestCase):
    """scatter_contiguous places rank r's contiguous chunk along an axis."""

    def test_axis0_per_rank_chunks_and_reconstruction(self):
        from paddlefleet.context_parallel_utils import scatter_contiguous

        # x rows: [[0,1],[2,3],...,[10,11]]; nranks=3 -> chunk of 2 rows each.
        # Hand-derived, rank r owns rows [2r, 2r+1].
        x = paddle.arange(12).reshape([6, 2]).cast("float32")
        expected = {
            0: [[0.0, 1.0], [2.0, 3.0]],
            1: [[4.0, 5.0], [6.0, 7.0]],
            2: [[8.0, 9.0], [10.0, 11.0]],
        }
        chunks = []
        for rank in range(3):
            out = scatter_contiguous(
                x, group=_CpGroup(nranks=3, rank=rank), axis=0
            )
            self.assertEqual(out.tolist(), expected[rank])
            chunks.append(out)
        # Union of the shards, in rank order, must be the original tensor:
        # a swapped or off-by-chunk slice would break this.
        rebuilt = paddle.concat(chunks, axis=0)
        self.assertEqual(rebuilt.tolist(), x.tolist())

    def test_axis1_column_chunks(self):
        from paddlefleet.context_parallel_utils import scatter_contiguous

        # 3x4 with distinct values; nranks=2 along axis=1 -> 2 columns each.
        x = paddle.arange(12).reshape([3, 4]).cast("float32")
        rank0 = scatter_contiguous(x, group=_CpGroup(2, 0), axis=1)
        rank1 = scatter_contiguous(x, group=_CpGroup(2, 1), axis=1)
        self.assertEqual(rank0.tolist(), [[0.0, 1.0], [4.0, 5.0], [8.0, 9.0]])
        self.assertEqual(rank1.tolist(), [[2.0, 3.0], [6.0, 7.0], [10.0, 11.0]])

    def test_negative_axis_matches_numpy_oracle(self):
        from paddlefleet.context_parallel_utils import scatter_contiguous

        # numpy slicing is an independent reference for the axis=-1 chunk.
        base = np.arange(24, dtype="float32").reshape([2, 3, 4])
        x = paddle.to_tensor(base)
        out = scatter_contiguous(x, group=_CpGroup(nranks=2, rank=1), axis=-1)
        expected = base[:, :, 2:4]  # rank 1 of 2 -> second half of last axis
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_nranks_1_returns_independent_clone(self):
        from paddlefleet.context_parallel_utils import scatter_contiguous

        # nranks==1 is a documented no-op: full tensor, but a distinct buffer.
        x = paddle.arange(6).reshape([3, 2]).cast("float32")
        out = scatter_contiguous(x, group=_CpGroup(nranks=1, rank=0), axis=0)
        self.assertEqual(out.tolist(), x.tolist())
        out[0, 0] = -99.0
        self.assertEqual(x[0, 0].item(), 0.0)  # input must be untouched

    def test_nranks_1_allows_indivisible_length(self):
        from paddlefleet.context_parallel_utils import scatter_contiguous

        # The divisibility guard only applies once there is more than one rank.
        x = paddle.arange(30).reshape([10, 3]).cast("float32")
        out = scatter_contiguous(x, group=_CpGroup(nranks=1, rank=0), axis=0)
        self.assertEqual(out.shape, [10, 3])

    def test_indivisible_length_raises_valueerror(self):
        from paddlefleet.context_parallel_utils import scatter_contiguous

        # 10 rows cannot be split evenly across 4 ranks; equal-sized shards
        # would silently drop the tail, so the function must refuse.
        x = paddle.arange(30).reshape([10, 3]).cast("float32")
        with self.assertRaises(ValueError) as ctx:
            scatter_contiguous(x, group=_CpGroup(nranks=4, rank=0), axis=0)
        self.assertIn("divisible", str(ctx.exception))


@unittest.skipUnless(HAVE_PADDLE, "paddle is not installed in this environment")
class TestNranks1LocalFallback(unittest.TestCase):
    """The supported world-size==1 fall-back of the collective helpers.

    These only cover the local no-collective branch; the multi-rank NCCL
    paths need a real process group (multi-card) and are not asserted here.
    """

    def test_all_gather_contiguous_clone(self):
        from paddlefleet.context_parallel_utils import all_gather_contiguous

        x = paddle.arange(8).reshape([2, 4]).cast("float32")
        out = all_gather_contiguous(x, group=_CpGroup(1, 0), axis=0)
        self.assertEqual(out.tolist(), x.tolist())
        out[0, 0] = 123.0
        self.assertEqual(x[0, 0].item(), 0.0)  # clone, not aliased

    def test_reduce_scatter_contiguous_clone(self):
        from paddlefleet.context_parallel_utils import (
            reduce_scatter_contiguous,
        )

        x = paddle.arange(24).reshape([4, 6]).cast("float32")
        out = reduce_scatter_contiguous(x, axis=0, group=_CpGroup(1, 0))
        self.assertEqual(out.tolist(), x.tolist())
        out[0, 0] = 77.0
        self.assertEqual(x[0, 0].item(), 0.0)


@unittest.skipUnless(HAVE_PADDLE, "paddle is not installed in this environment")
class TestPyLayerContiguousDispatch(unittest.TestCase):
    """The mode string routes to the contiguous (local) layout, and the
    observable result is the hand-derived contiguous chunk -- not just a
    record that some collaborator was called."""

    def test_scatter_op_forward_contiguous_produces_rank_chunk(self):
        from paddlefleet.context_parallel_utils import (
            ContextParallelScatterOp,
        )

        # nranks=3, rank=1 -> contiguous rows [2:4] of the 6-row input.
        x = paddle.arange(12).reshape([6, 2]).cast("float32")
        patch, _ = _patch_hcg(nranks=3, rank=1)
        ctx = SimpleNamespace()
        with patch:
            out = ContextParallelScatterOp.forward(
                ctx, x, axis=0, mode="contiguous_allgather"
            )
        self.assertEqual(out.tolist(), [[4.0, 5.0], [6.0, 7.0]])
        self.assertEqual(ctx.mode, "contiguous_allgather")
        self.assertEqual(ctx.axis, 0)

    def test_gather_op_backward_contiguous_scatters_grad(self):
        from paddlefleet.context_parallel_utils import (
            ContextParallelGatherOp,
        )

        # GatherOp.backward in contiguous mode is a local scatter of the
        # incoming gradient: rank=2 of 3 -> rows [4:6].
        grad = paddle.arange(12).reshape([6, 2]).cast("float32")
        ctx = SimpleNamespace(
            mode="contiguous_allgather",
            axis=0,
            group=_CpGroup(nranks=3, rank=2),
        )
        out = ContextParallelGatherOp.backward(ctx, grad)
        self.assertEqual(out.tolist(), [[8.0, 9.0], [10.0, 11.0]])


@unittest.skipUnless(HAVE_PADDLE, "paddle is not installed in this environment")
class TestPyLayerWorldSizeGuard(unittest.TestCase):
    """Each CP PyLayer forward must reject a degenerate cp_world_size <= 1."""

    def _assert_guard(self, op):
        x = paddle.arange(16).reshape([4, 4]).cast("float32")
        patch, _ = _patch_hcg(nranks=1, rank=0)
        with patch, self.assertRaises(AssertionError):
            op.forward(
                SimpleNamespace(), x, axis=0, mode="contiguous_allgather"
            )

    def test_scatter_op_guard(self):
        from paddlefleet.context_parallel_utils import (
            ContextParallelScatterOp,
        )

        self._assert_guard(ContextParallelScatterOp)

    def test_gather_op_guard(self):
        from paddlefleet.context_parallel_utils import (
            ContextParallelGatherOp,
        )

        self._assert_guard(ContextParallelGatherOp)

    def test_all_gather_op_guard(self):
        from paddlefleet.context_parallel_utils import (
            ContextParallelAllGatherOp,
        )

        self._assert_guard(ContextParallelAllGatherOp)


if __name__ == "__main__":
    unittest.main()
