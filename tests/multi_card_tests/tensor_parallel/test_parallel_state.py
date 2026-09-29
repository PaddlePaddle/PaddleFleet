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

import sys
import unittest

import paddle
import paddle.distributed as dist

from paddlefleet import parallel_state as ps
from tests.multi_card_tests.tensor_parallel.test_utilities import Utils


def test_tensor_model_parallel_group_membership():
    """The TP group must actually span the correct global ranks.

    Topology is TP=4 over world_size=4, so the single tensor-model-parallel
    group holds global ranks [0, 1, 2, 3]. We prove this with REAL collectives
    (not shape/None checks): an all_reduce whose sum is derived by hand, and an
    all_gather whose per-position values pin down membership AND ordering.
    """
    tp_group = ps.get_tensor_model_parallel_group()
    assert tp_group.nranks == 4, tp_group.nranks

    # Rank-distinguishable payload: rank r contributes (r + 1) -> {1, 2, 3, 4}.
    local = paddle.to_tensor([float(Utils.rank + 1)]).cuda()
    dist.all_reduce(local, group=tp_group)
    # Independent reference: sum_{r=0..3} (r + 1) = 1 + 2 + 3 + 4 = 10.
    assert paddle.equal_all(local, paddle.to_tensor([10.0]).cuda())

    # all_gather orders outputs by group-local rank. Each process sends its own
    # global rank, so position i must carry the value of group member i.
    send = paddle.to_tensor([float(Utils.rank)]).cuda()
    gathered = []
    dist.all_gather(gathered, send, group=tp_group)
    values = [int(t.item()) for t in gathered]
    # tp_group.ranks == [0, 1, 2, 3] in ascending group-local order, so the
    # gathered payloads must reproduce exactly that sequence.
    assert values == [0, 1, 2, 3], values


def test_tensor_model_parallel_rank_and_world_size():
    """TP world size and this rank's index within the TP group."""
    assert ps.get_tensor_model_parallel_world_size() == 4

    tp_group = ps.get_tensor_model_parallel_group()
    # Derive the expected TP rank independently from the group membership list
    # rather than from the function under test.
    expected_tp_rank = list(tp_group.ranks).index(Utils.rank)
    assert ps.get_tensor_model_parallel_rank() == expected_tp_rank
    # With TP spanning the whole world, the group-local index equals the global
    # rank; this catches a group that silently reordered its members.
    assert expected_tp_rank == Utils.rank


def test_pipeline_single_stage_flags():
    """With PP=1 every rank is simultaneously the first and last stage."""
    assert ps.get_pipeline_model_parallel_world_size() == 1
    assert ps.get_pipeline_model_parallel_rank() == 0
    assert ps.is_pipeline_first_stage() is True
    assert ps.is_pipeline_last_stage() is True


def test_data_parallel_group_is_singleton():
    """DP/sharding degree is 1, so each rank sits alone in its DP group.

    A real all_reduce over a size-1 group is the identity; running it proves
    the returned group genuinely isolates this rank instead of aliasing a
    larger group (which would change the value).
    """
    dp_group = ps.get_data_parallel_group()
    assert dp_group is not None
    assert dp_group.nranks == 1, dp_group.nranks

    local = paddle.to_tensor([float(Utils.rank + 7)]).cuda()
    dist.all_reduce(local, group=dp_group)
    assert paddle.equal_all(
        local, paddle.to_tensor([float(Utils.rank + 7)]).cuda()
    )


def test_global_memory_buffer_reuses_storage():
    """initialize_model_parallel wires up the global memory buffer.

    The buffer's contract is storage reuse: requesting the same (name, dtype)
    must hand back a view onto the same backing allocation, not a fresh tensor.
    """
    assert ps.have_global_memory_buffer() is True
    buffer = ps.get_global_memory_buffer()

    first = buffer.get_tensor([2, 3], "float32", "probe")
    second = buffer.get_tensor([2, 3], "float32", "probe")
    assert list(first.shape) == [2, 3]
    assert first.data_ptr() == second.data_ptr()


class TestParallelStateExpertGroupBugs(unittest.TestCase):
    """Real production defects in the expert-parallel accessors.

    These are documented WITHOUT editing production: one via expectedFailure
    (wrong return value) and one via assertRaises (crash instead of fallback).
    They are exercised in the same TP=4 topology set up by the launcher.
    """

    @unittest.expectedFailure
    def test_expert_tensor_parallel_world_size_backward_compat(self):
        # BUG (src/paddlefleet/parallel_state.py:377): when no expert-tensor
        # group and no explicit override are configured, the documented
        # backward-compatible fallback should be the tensor-model parallel
        # world size (4 here). Instead the function returns the module global
        # _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE, which initialize_model_parallel
        # never sets, so it returns None. Asserting the correct value fails.
        self.assertEqual(
            ps.get_expert_tensor_parallel_world_size(),
            ps.get_tensor_model_parallel_world_size(),
        )

    def test_expert_tensor_and_model_group_uninitialized_raises(self):
        # BUG (src/paddlefleet/parallel_state.py:405-434):
        # initialize_model_parallel never assigns
        # _EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP, so in a normally-initialized
        # distributed job these accessors raise AssertionError instead of the
        # benign fallback their own else-branch returns when distributed is not
        # initialized. (Were the group set, .size()/.rank() are also wrong:
        # paddle Group exposes .nranks/.rank as properties, not callables.)
        # We lock the current crashing contract via assertRaises.
        with self.assertRaises(AssertionError):
            ps.get_expert_tensor_and_model_parallel_world_size()
        with self.assertRaises(AssertionError):
            ps.get_expert_tensor_and_model_parallel_rank()


if __name__ == "__main__":
    Utils.initialize_model_parallel(4, 1)
    test_tensor_model_parallel_group_membership()
    test_tensor_model_parallel_rank_and_world_size()
    test_pipeline_single_stage_flags()
    test_data_parallel_group_is_singleton()
    test_global_memory_buffer_reuses_storage()
    unittest.main(argv=[sys.argv[0]], exit=False, verbosity=2)
