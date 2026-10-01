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

"""Behaviour tests for paddlefleet.nn.moe_deepep.moe_communication.

Scope: distributed-training MoE communication wrappers, no-card (CPU).

What is verified on CPU (real production entry, independent expected):
  * The ``expert_model_parallel_size <= 1`` short-circuit in both
    ``AllToAllMoECommunication.forward`` and ``DeepEPMoECommunication.forward``
    returns the input hidden states untouched (identity), without touching the
    token dispatcher / collectives.  This is a genuine world-size==1 local
    fallback path, so single-process verification is appropriate.
  * ``DeepEPMoECommunication.expert_forward`` control logic: how the gathered
    per-rank buffer is split by ``tokens_per_expert`` into chunks, which global
    expert each chunk is routed to (the ``i + moe_rank * num_experts_per_device``
    ownership mapping), and the order in which expert outputs are concatenated.
    Experts are replaced by distinguishable marker layers and the expected
    output is built by independent NumPy slicing (never by the tested split).

What is deliberately NOT verified here (skipped with reason):
  * The ``expert_model_parallel_size > 1`` numeric paths of ``forward`` route
    through ``token_dispatcher.token_permutation`` / ``_AllToAll`` collectives
    and require a real multi-rank process group.  Faking ``world_size`` and
    mocking the collective to claim cross-rank numerics is an anti-pattern
    (single-card cannot prove dispatch-by-expert-ownership across ranks); this
    is left to the multi-card suite.
"""

import unittest

import numpy as np
import paddle
from paddle import nn

from paddlefleet.nn.moe_deepep.moe_communication import (
    AllToAllMoECommunication,
    DeepEPMoECommunication,
    MoECommunicationInterface,
)


class _MarkerExpert(nn.Layer):
    """Expert whose forward is identity plus a per-expert additive tag.

    Keeping the input rows intact makes token *ordering* observable, while the
    unique ``tag`` makes expert *ownership* observable: if a chunk is routed to
    the wrong global expert, the added constant will not match the reference.
    """

    def __init__(self, tag):
        super().__init__()
        self._tag = float(tag)

    def forward(self, x):
        return x + self._tag


def _global_tag(global_expert_idx):
    # Distinguishable, off-by-one sensitive tag for global expert index.
    return (global_expert_idx + 1) * 1000.0


def _make_experts(num_global_experts):
    return nn.LayerList(
        [_MarkerExpert(_global_tag(g)) for g in range(num_global_experts)]
    )


class TestInterfaceContract(unittest.TestCase):
    """The ABC is the declared contract both wrappers must satisfy."""

    def test_wrappers_are_interface_and_layer(self):
        from abc import ABC

        self.assertTrue(issubclass(MoECommunicationInterface, ABC))
        for cls in (AllToAllMoECommunication, DeepEPMoECommunication):
            self.assertTrue(issubclass(cls, MoECommunicationInterface))
            self.assertTrue(issubclass(cls, nn.Layer))


class TestDeepEPExpertForward(unittest.TestCase):
    """CPU-testable control logic of DeepEPMoECommunication.expert_forward.

    Verifies chunk boundaries, expert ownership (moe_rank offset) and the
    concatenation order against an independent NumPy reference. No collective
    is involved: this method runs entirely on the local rank's gathered buffer.
    """

    def _run(
        self,
        dispatched_np,
        tokens_per_expert,
        moe_rank,
        num_per_device,
        num_global,
    ):
        comm = DeepEPMoECommunication()
        experts = _make_experts(num_global)
        dispatched = paddle.to_tensor(dispatched_np, dtype="float32")
        out = comm.expert_forward(
            dispatched,
            tokens_per_expert,
            experts,
            moe_rank=moe_rank,
            num_experts_per_device=num_per_device,
        )
        return out

    def _independent_expected(
        self, dispatched_np, sizes, moe_rank, num_per_device
    ):
        # Rebuild the expected output WITHOUT the tested paddle.split: slice the
        # rows at the cumulative boundaries and add the tag of the global expert
        # that local chunk i (-> global i + moe_rank*num_per_device) owns.
        rows = []
        start = 0
        for i, n in enumerate(sizes):
            g = i + moe_rank * num_per_device
            block = dispatched_np[start : start + n] + _global_tag(g)
            rows.append(block)
            start += n
        return np.concatenate(rows, axis=0)

    def test_basic_dispatch_rank0(self):
        # 5 rows, distinguishable content; local experts 0 and 1 own 2 and 3 rows.
        dispatched = np.arange(5 * 4, dtype=np.float32).reshape([5, 4])
        sizes = [2, 3]
        out = self._run(
            dispatched, sizes, moe_rank=0, num_per_device=2, num_global=2
        )
        expected = self._independent_expected(dispatched, sizes, 0, 2)
        self.assertEqual(list(out.shape), [5, 4])
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_moe_rank_offset_selects_global_experts(self):
        # moe_rank=1, 2 experts/device -> local chunks route to GLOBAL experts
        # 2 and 3, not 0 and 1. A dropped moe_rank offset would pick tags
        # 1000/2000 instead of 3000/4000 and fail.
        dispatched = np.arange(4 * 3, dtype=np.float32).reshape([4, 3])
        sizes = [2, 2]
        out = self._run(
            dispatched, sizes, moe_rank=1, num_per_device=2, num_global=4
        )
        expected = self._independent_expected(dispatched, sizes, 1, 2)
        np.testing.assert_array_equal(out.numpy(), expected)
        # Cross-check: the wrong-ownership output (rank 0 tags) must differ.
        wrong = self._independent_expected(dispatched, sizes, 0, 2)
        self.assertFalse(np.array_equal(out.numpy(), wrong))

    def test_tensor_tokens_per_expert_is_converted(self):
        # tokens_per_expert given as a paddle Tensor exercises the .tolist()
        # branch; uneven split [1, 3] also pins the chunk boundary.
        dispatched = np.arange(4 * 2, dtype=np.float32).reshape([4, 2])
        sizes = [1, 3]
        out = self._run(
            dispatched,
            paddle.to_tensor(sizes, dtype="int64"),
            moe_rank=0,
            num_per_device=2,
            num_global=2,
        )
        expected = self._independent_expected(dispatched, sizes, 0, 2)
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_concatenation_order_preserves_rows(self):
        # Distinct rows + distinct tags: any reordering of chunks or rows shows.
        dispatched = np.arange(6 * 2, dtype=np.float32).reshape([6, 2]) + 0.5
        sizes = [1, 2, 3]
        out = self._run(
            dispatched, sizes, moe_rank=0, num_per_device=3, num_global=3
        )
        expected = self._independent_expected(dispatched, sizes, 0, 3)
        self.assertEqual(list(out.shape), [6, 2])
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_zero_token_expert_boundary(self):
        # Boundary: a single local expert with zero tokens. The gathered buffer
        # has 0 rows; the result stays an empty [0, d_model] tensor. Content is
        # trivially empty here, so this only pins the degenerate shape/dtype.
        comm = DeepEPMoECommunication()
        experts = _make_experts(1)
        dispatched = paddle.zeros([0, 4], dtype="float32")
        out = comm.expert_forward(
            dispatched, [0], experts, moe_rank=0, num_experts_per_device=1
        )
        self.assertEqual(list(out.shape), [0, 4])
        self.assertEqual(out.dtype, dispatched.dtype)


class TestSingleRankFallback(unittest.TestCase):
    """expert_model_parallel_size <= 1 must return input untouched.

    token_dispatcher is None and experts is empty on purpose: if the code did
    not short-circuit, it would raise on ``None.token_permutation`` / indexing,
    so the identity assertion also proves the collective path is not entered.
    """

    def _call(self, comm):
        x = paddle.arange(4 * 8, dtype="float32").reshape([4, 8])
        result = comm.forward(
            hidden_states=x,
            topk_indices=paddle.to_tensor([[0, 1]]),
            topk_weights=paddle.ones([4, 2]),
            gates_masked=paddle.ones([4, 8]),
            mask=paddle.ones([4, 8]),
            priorities=paddle.ones([4, 2]),
            expert_model_parallel_size=1,
            moe_group=None,
            experts=nn.LayerList([]),
            moe_rank=0,
            num_experts_per_device=8,
            num_experts=8,
            topk=2,
            token_dispatcher=None,
        )
        return x, result

    def test_alltoall_ep1_returns_same_tensor(self):
        x, result = self._call(AllToAllMoECommunication())
        self.assertIs(result, x)
        np.testing.assert_array_equal(result.numpy(), x.numpy())

    def test_deepep_ep1_returns_same_tensor(self):
        x, result = self._call(DeepEPMoECommunication())
        self.assertIs(result, x)
        np.testing.assert_array_equal(result.numpy(), x.numpy())


class TestMultiRankNumericsSkipped(unittest.TestCase):
    """Cross-rank comm numerics are out of scope for a no-card test."""

    @unittest.skip(
        "EP>1 forward routes through token_dispatcher.token_permutation and "
        "_AllToAll collectives, requiring a real multi-rank process group. "
        "Faking world_size + mocking the collective cannot prove "
        "dispatch-by-expert-ownership or cross-rank reassembly; verified in "
        "the multi-card suite instead."
    )
    def test_ep_gt_1_requires_real_process_group(self):
        pass


if __name__ == "__main__":
    unittest.main()
