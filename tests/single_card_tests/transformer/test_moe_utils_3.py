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

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

try:
    import paddle

    from paddlefleet.transformer.moe.moe_utils import (
        AllGatherGroupOp,
        _AllToAll,
        all_gather_group,
        barrier_ep,
        reduce_scatter_group,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # CPU box without paddle
    paddle = None
    _IMPORT_ERROR = exc

_SKIP_REASON = f"paddle / paddlefleet import unavailable: {_IMPORT_ERROR!r}"


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class MoeUtilsCollectiveBase(unittest.TestCase):
    """Shared CPU setup for the EP collective helpers in ``moe_utils``.

    All cases below exercise the ``world_size <= 1`` / ``nranks == 1``
    fallback branches, which are pure-local data paths and therefore valid
    to verify single-process. Real cross-rank all-to-all / all-gather /
    reduce-scatter numerics require a genuine process group and are NOT
    claimed here.
    """

    def setUp(self):
        self._orig_device = paddle.get_device()
        paddle.set_device("cpu")
        self.addCleanup(paddle.set_device, self._orig_device)


class TestBarrierEp(MoeUtilsCollectiveBase):
    def test_barrier_ep_forwards_group_and_returns_none(self):
        # barrier_ep is a thin wrapper: it must hand the *same* group object
        # to paddle.distributed.barrier and return nothing of its own.
        group = SimpleNamespace(name="ep_group")
        with patch("paddle.distributed.barrier") as mock_barrier:
            result = barrier_ep(group)
        mock_barrier.assert_called_once_with(group)
        self.assertIs(mock_barrier.call_args.args[0], group)
        self.assertIsNone(result)


class TestAllToAllSingleRank(MoeUtilsCollectiveBase):
    def test_forward_returns_input_and_skips_communication(self):
        # world_size <= 1 must short-circuit to the input verbatim, touching
        # neither the barrier nor the all-to-all collective.
        x = paddle.arange(32, dtype="float32").reshape([4, 8])
        group = SimpleNamespace()
        with (
            patch("paddle.distributed.get_world_size", return_value=1),
            patch("paddle.distributed.barrier") as mock_barrier,
            patch("paddle.distributed.alltoall_single") as mock_a2a,
        ):
            out = _AllToAll.apply([4, 8], x, group=group)
        np.testing.assert_array_equal(out.numpy(), x.numpy())
        self.assertEqual(list(out.shape), [4, 8])
        self.assertEqual(out.dtype, x.dtype)
        mock_barrier.assert_not_called()
        mock_a2a.assert_not_called()


class TestReduceScatterGroupSingleRank(MoeUtilsCollectiveBase):
    def test_returns_independent_clone_without_communication(self):
        # nranks == 1 => return input.clone(): same values, but a *distinct*
        # storage, and no reduce_scatter collective is issued.
        x = paddle.arange(32, dtype="float32").reshape([4, 8])
        group = SimpleNamespace(nranks=1)
        with patch("paddle.distributed.stream.reduce_scatter") as mock_rs:
            out = reduce_scatter_group(x, group=group)
        np.testing.assert_array_equal(out.numpy(), x.numpy())
        self.assertEqual(list(out.shape), [4, 8])
        self.assertIsNot(out, x)
        mock_rs.assert_not_called()
        # Independent storage: mutating the clone leaves the source intact.
        out[0, 0] = -999.0
        self.assertEqual(float(x[0, 0]), 0.0)


class TestAllGatherGroupSingleRank(MoeUtilsCollectiveBase):
    def test_all_gather_group_returns_independent_clone(self):
        # nranks == 1 => return input.clone() before any all_gather call.
        x = paddle.arange(24, dtype="float32").reshape([3, 8])
        group = SimpleNamespace(nranks=1)
        with patch("paddle.distributed.stream.all_gather") as mock_ag:
            out = all_gather_group(x, group=group)
        np.testing.assert_array_equal(out.numpy(), x.numpy())
        self.assertEqual(list(out.shape), [3, 8])
        self.assertIsNot(out, x)
        mock_ag.assert_not_called()
        out[0, 0] = -1.0
        self.assertEqual(float(x[0, 0]), 0.0)


class TestAllGatherGroupOpSingleRank(MoeUtilsCollectiveBase):
    def test_forward_barriers_group_then_returns_clone(self):
        # Forward always barriers on the group, then (nranks == 1) returns a
        # clone without invoking the all_gather collective.
        x = paddle.arange(24, dtype="float32").reshape([3, 8])
        group = SimpleNamespace(nranks=1)
        with (
            patch("paddle.distributed.barrier") as mock_barrier,
            patch("paddle.distributed.stream.all_gather") as mock_ag,
        ):
            out = AllGatherGroupOp.apply(x, group=group)
        mock_barrier.assert_called_once()
        self.assertIs(mock_barrier.call_args.args[0], group)
        mock_ag.assert_not_called()
        np.testing.assert_array_equal(out.numpy(), x.numpy())
        self.assertEqual(list(out.shape), [3, 8])

    def test_backward_passes_gradient_through_as_identity(self):
        # Backward reduce-scatters the incoming grad; at nranks == 1 that is a
        # clone, so the input gradient must equal the (non-uniform) upstream
        # grad exactly. No real collective is issued on this local path.
        x = paddle.arange(24, dtype="float32").reshape([3, 8])
        x.stop_gradient = False
        group = SimpleNamespace(nranks=1)
        upstream = paddle.arange(24, dtype="float32").reshape([3, 8]) + 100.0
        with (
            patch("paddle.distributed.barrier"),
            patch("paddle.distributed.stream.all_gather") as mock_ag,
            patch("paddle.distributed.stream.reduce_scatter") as mock_rs,
        ):
            out = AllGatherGroupOp.apply(x, group=group)
            out.backward(upstream)
        mock_ag.assert_not_called()
        mock_rs.assert_not_called()
        self.assertIsNotNone(x.grad)
        np.testing.assert_array_equal(x.grad.numpy(), upstream.numpy())


if __name__ == "__main__":
    unittest.main()
