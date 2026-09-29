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

"""No-card behavior tests for paddlefleet.nn.moe.all_gather.

Scope (无卡 / distributed module rules):
  These tests exercise ONLY the world-size==1 / local-copy code paths and the
  pure input-validation guards of the AllGather-based MoE dispatcher. Those
  branches are real production code that runs on CPU without launching any
  collective, so verifying them here is legitimate single-path evidence.

  The genuine cross-rank numerics -- all_gather rank ordering, reduce-scatter
  SUM reduction, and the AlltoAll expert dispatch/permute/unpermute with
  num_local_experts>1 spread over several ranks -- depend on multiple ranks
  each holding DIFFERENT data and exchanging it through a real process group.
  They cannot be validated in a single CPU process. Faking `world_size` and
  mocking `dist.stream.all_gather` / `alltoall_single` (as the coverage_test
  did) only proves a call happened; it cannot detect a wrong peer, a swapped
  shard, a dropped reduction term, or a mis-permuted expert. Those cases are
  therefore explicitly skipped below with a pointer to the multi-card suite,
  never faked.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

try:
    import paddle

    from paddlefleet.nn.moe.all_gather import (
        AlltoAllSmart,
        allgather_async,
        reduce_scatter_async,
    )

    _HAS_PADDLE = True
except ImportError:
    _HAS_PADDLE = False


def _single_rank_group():
    """A lightweight stand-in for a process group with a single member.

    `allgather_async` / `reduce_scatter_async` only read `.nranks`; when it is 1
    they short-circuit before touching `paddle.distributed`, so no real group is
    needed and no collective is invoked.
    """
    return SimpleNamespace(nranks=1)


@unittest.skipUnless(_HAS_PADDLE, "paddle / paddlefleet not importable")
class TestAllgatherAsyncSingleRank(unittest.TestCase):
    """world_size==1 clone contract of allgather_async (no collective)."""

    def test_returns_independent_clone_without_task(self):
        x = paddle.to_tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32"
        )
        out, task = allgather_async(x, group=_single_rank_group())

        # No async work is scheduled on the degenerate single-rank path.
        self.assertIsNone(task)
        # Content is preserved exactly.
        np.testing.assert_array_equal(out.numpy(), x.numpy())
        # It is a real clone: mutating the output must not alias the input.
        out[0, 0] = -99.0
        self.assertEqual(float(x[0, 0]), 1.0)

    def test_preserves_multi_dim_shape_and_dtype(self):
        x = paddle.arange(2 * 3 * 4, dtype="int64").reshape([2, 3, 4])
        out, task = allgather_async(x, group=_single_rank_group())
        self.assertIsNone(task)
        self.assertEqual(out.shape, [2, 3, 4])
        self.assertEqual(out.dtype, x.dtype)
        np.testing.assert_array_equal(out.numpy(), x.numpy())


@unittest.skipUnless(_HAS_PADDLE, "paddle / paddlefleet not importable")
class TestReduceScatterAsyncSingleRank(unittest.TestCase):
    """world_size==1 clone contract + divisibility guard (no collective)."""

    def test_returns_independent_clone_without_task(self):
        x = paddle.to_tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype="float32"
        )
        out, task = reduce_scatter_async(x, group=_single_rank_group())

        self.assertIsNone(task)
        np.testing.assert_array_equal(out.numpy(), x.numpy())
        out[1, 1] = 123.0
        self.assertEqual(float(x[1, 1]), 4.0)

    def test_rejects_length_not_divisible_by_parallelism(self):
        # nranks=3 does not divide a leading dim of 10. The assertion fires
        # BEFORE any collective is reached, so this guard is pure CPU logic.
        x = paddle.zeros([10, 4], dtype="float32")
        group = SimpleNamespace(nranks=3)
        with self.assertRaises(AssertionError) as cm:
            reduce_scatter_async(x, group=group)
        # The message must report the offending length and parallelism, not a
        # generic failure -- catches a guard that checks the wrong axis.
        msg = str(cm.exception)
        self.assertIn("10", msg)
        self.assertIn("3", msg)

    def test_accepts_divisible_length_at_guard(self):
        # A divisible leading dim must pass the divisibility assertion. With a
        # real group absent, execution then reaches the actual collective and
        # raises a *different* (non-Assertion) error; catching AssertionError
        # here would mean the guard wrongly rejected valid input.
        x = paddle.zeros([8, 4], dtype="float32")
        group = SimpleNamespace(nranks=4)
        try:
            reduce_scatter_async(x, group=group)
        except AssertionError:
            self.fail("divisibility guard rejected a divisible leading dim")
        except Exception:
            # Expected: no real process group in this single CPU process.
            pass


@unittest.skipUnless(_HAS_PADDLE, "paddle / paddlefleet not importable")
class TestAlltoAllSmartSingleRankRouting(unittest.TestCase):
    """world_size==1 local-copy dispatch of AlltoAllSmart (no collective).

    With group.nranks<=1 the forward takes the local-copy branch
    (`output_local_expert[:] = input_local_expert[:]`) instead of
    `dist.stream.alltoall_single`, so the per-expert token packing runs for
    real on CPU. `dist.get_rank` / `dist.get_world_size` are topology queries
    (not collectives); they are pinned to the single-rank values 0 / 1 so the
    supported world-size==1 path can execute. This validates ONLY that local
    path; cross-rank dispatch is covered by the skipped multi-card case below.
    """

    def _run_single_rank_forward(self):
        num_local_experts = 2
        # capacity is recomputed inside forward as
        #   len(send_rank_global) // world_size // num_local_experts
        # world_size==1 -> every routed slot belongs to this rank.
        send_rank_global = paddle.zeros([4], dtype="int64")
        recv_rank_global = paddle.zeros([4], dtype="int64")
        local_expert_id = paddle.to_tensor([0, 0, 1, 1], dtype="int64")

        # Distinguishable per-token payloads so a wrong slice offset, a swapped
        # expert block, or a mis-sized copy changes the observed output.
        x0 = paddle.to_tensor(
            [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]], dtype="float32"
        )
        x1 = paddle.to_tensor(
            [[20.0, 21.0, 22.0], [23.0, 24.0, 25.0]], dtype="float32"
        )
        router = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0]], dtype="float32")

        # Expert i sends and receives its own 2 tokens (single rank).
        send_counts = np.array([[2], [2]], dtype="int64")
        recv_counts = np.array([[2], [2]], dtype="int64")
        send_counts_num = np.array([2, 2], dtype="int64")
        recv_counts_num = np.array([2, 2], dtype="int64")

        with (
            mock.patch(
                "paddlefleet.nn.moe.all_gather.dist.get_rank", return_value=0
            ),
            mock.patch(
                "paddlefleet.nn.moe.all_gather.dist.get_world_size",
                return_value=1,
            ),
        ):
            output, router_loss, scatter_idx = AlltoAllSmart.apply(
                x0,
                x1,
                router,
                router_loss_fn=lambda r: r,  # identity: router loss == input
                forward_func_dict=None,  # no expert compute, pure dispatch
                local_expert_id=local_expert_id,
                send_rank_global=send_rank_global,
                recv_rank_global=recv_rank_global,
                num_local_experts=num_local_experts,
                capacity=None,
                group=SimpleNamespace(nranks=1),
                recv_size=4,
                send_counts=send_counts,
                recv_counts=recv_counts,
                send_counts_num=send_counts_num,
                recv_counts_num=recv_counts_num,
                is_first_fwd=True,
            )
        return output, router_loss, scatter_idx, x0, x1, router

    def test_output_packs_experts_in_order_with_correct_tokens(self):
        output, router_loss, scatter_idx, x0, x1, router = (
            self._run_single_rank_forward()
        )

        # The dispatch buffer must be expert-0's tokens followed by expert-1's,
        # each copied verbatim into its [output_ptr : output_ptr+recv] slot.
        expected = np.concatenate([x0.numpy(), x1.numpy()], axis=0)
        self.assertEqual(list(output.shape), [4, 3])
        np.testing.assert_array_equal(output.numpy(), expected)

        # router_loss_fn output is really consumed and returned (not dropped).
        np.testing.assert_array_equal(router_loss.numpy(), router.numpy())

    def test_scatter_index_is_identity_permutation_for_single_rank(self):
        _, _, scatter_idx, _, _, _ = self._run_single_rank_forward()

        # Independent reference: for world_size==1 every recv_mask entry is 1,
        # so `maximum(cumsum(mask) - 1, 0)` collapses to arange, and the
        # expert/capacity transpose is its own inverse when there is one rank.
        # A non-arange result would signal a broken reshape/transpose.
        expected_idx = np.arange(4, dtype=scatter_idx.numpy().dtype)
        np.testing.assert_array_equal(scatter_idx.numpy(), expected_idx)


@unittest.skipUnless(_HAS_PADDLE, "paddle / paddlefleet not importable")
class TestAllGatherCrossRankNumerics(unittest.TestCase):
    """Cross-rank numerics require a real process group -> multi-card suite."""

    @unittest.skip(
        "Cross-rank AllGather ordering, reduce-scatter SUM, and the "
        "num_local_experts>1 AlltoAll expert permute/unpermute need multiple "
        "ranks holding distinct data over a real process group. Faking "
        "world_size and mocking the collective cannot verify peer, shard "
        "order, reduction terms, or expert ownership. Run under "
        "tests/multi_card_tests/ with paddle.distributed.launch."
    )
    def test_multi_rank_gather_and_dispatch(self):
        raise AssertionError("must run on real multi-rank process group")


if __name__ == "__main__":
    unittest.main()
