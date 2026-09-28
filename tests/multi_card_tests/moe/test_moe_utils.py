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
"""Real multi-card behaviour tests for ``transformer.moe.moe_utils``.

Topology (Pattern D, matching the module fixture the tests target): TP=4,
sharding=2, EP=1 -> world_size 8. The tensor model-parallel group therefore
has 4 ranks and drives the real group collectives (``all_gather_group`` /
``reduce_scatter_group``, their PyLayer wrappers and ``_AllToAll``) with
distinguishable per-rank content, so a wrong rank order, a dropped rank, a
wrong reduce op or a reversed all-to-all direction changes the exact result.
The local permute / unpermute maths and the aux-loss trick are checked
numerically against independent hand derivations.

Run with:
  python -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 \\
      tests/multi_card_tests/moe/test_moe_utils.py
"""

import numpy as np
import paddle

from paddlefleet.transformer.moe import moe_utils
from paddlefleet.utils import get_tensor_model_parallel_group_if_none
from tests.multi_card_tests.tensor_parallel.test_utilities import Utils


def test_permute_groups_tokens_by_expert():
    # Local MoE permute (no collective). routing_map is top-1:
    #   token 0 -> expert 0, token 1 -> expert 1,
    #   token 2 -> expert 0, token 3 -> expert 1.
    # permute groups rows by expert in ascending (expert, token) order, so the
    # gathered order is tokens [0, 2] (expert 0) then [1, 3] (expert 1).
    tokens = paddle.to_tensor(
        [[t * 10 + c for c in range(3)] for t in range(4)], dtype="float32"
    ).cuda()
    routing_map = paddle.to_tensor(
        [[1, 0], [0, 1], [1, 0], [0, 1]], dtype="int64"
    ).cuda()

    permuted, sorted_indices = moe_utils.permute(tokens, routing_map)

    # Independent hand derivation of the expert-major gather order.
    expected_indices = [0, 2, 1, 3]
    expected_permuted = np.array(
        [[0, 1, 2], [20, 21, 22], [10, 11, 12], [30, 31, 32]],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(sorted_indices.numpy(), expected_indices)
    np.testing.assert_array_equal(permuted.numpy(), expected_permuted)


def test_unpermute_scatter_add_sums_expert_copies():
    # Build the permuted buffer and index map BY HAND (independent of permute)
    # to pin unpermute's scatter-ADD semantics. sorted_indices places two
    # copies of token 0 (routed to experts 0 and 1) back onto output row 0, so
    # the restored row 0 must be the SUM of both copies; rows 1 and 2 receive a
    # single copy each. An overwrite (last-wins) instead of add would drop the
    # second contribution and change row 0.
    tok0 = [1.0, 2.0, 3.0]
    tok1 = [4.0, 5.0, 6.0]
    tok2 = [7.0, 8.0, 9.0]
    permuted = paddle.to_tensor(
        [tok0, tok1, tok0, tok2], dtype="float32"
    ).cuda()
    sorted_indices = paddle.to_tensor([0, 1, 0, 2], dtype="int64").cuda()

    unpermuted = moe_utils.unpermute(permuted, sorted_indices, [3, 3])

    expected = np.array(
        [[2.0, 4.0, 6.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32
    )
    np.testing.assert_array_equal(unpermuted.numpy(), expected)


def test_permute_unpermute_roundtrip_top1():
    # Top-1 routing: every token goes to exactly one expert, so the scatter-add
    # in unpermute has no colliding indices and must reconstruct the input
    # exactly. A wrong gather order or a dropped row would change the values.
    tokens = paddle.to_tensor(
        [[t * 10 + c for c in range(4)] for t in range(6)], dtype="float32"
    ).cuda()
    routing_map = paddle.to_tensor(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        dtype="int64",
    ).cuda()

    permuted, sorted_indices = moe_utils.permute(tokens, routing_map)
    restored = moe_utils.unpermute(permuted, sorted_indices, list(tokens.shape))

    np.testing.assert_array_equal(restored.numpy(), tokens.numpy())


def test_add_auxiliary_loss_injects_unit_gradient():
    # AddAuxiliaryLoss forward is an identity clone; its backward injects a
    # gradient of exactly 1 into the aux-loss scalar (when it requires grad)
    # while passing the upstream gradient through to x unchanged. Verify both
    # the identity forward and the two independent backward contributions.
    x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32").cuda()
    x.stop_gradient = False
    aux = paddle.to_tensor([5.0], dtype="float32").cuda()
    aux.stop_gradient = False

    out = moe_utils.AddAuxiliaryLoss.apply(x, aux)
    np.testing.assert_array_equal(out.numpy(), x.numpy())

    upstream = paddle.to_tensor(
        [[10.0, 20.0], [30.0, 40.0]], dtype="float32"
    ).cuda()
    paddle.autograd.backward([out], [upstream])

    np.testing.assert_array_equal(x.grad.numpy(), upstream.numpy())
    np.testing.assert_array_equal(
        aux.grad.numpy(), np.array([1.0], dtype=np.float32)
    )


def test_all_to_all_equal_split():
    # Real all-to-all across the 4-rank tensor-parallel group. Rank r sends
    # its j-th row to rank j; after the exchange rank r's i-th row is the r-th
    # row that rank i sent. With input row value (rank*10 + j), rank r's output
    # row i is (i*10 + rank). A reversed direction or wrong peer changes these.
    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
    rank = tp_group.rank
    world_size = tp_group.world_size

    input_data = (
        (paddle.arange(world_size, dtype="float32") + rank * 10)
        .reshape([world_size, 1])
        .cuda()
    )
    input_data.stop_gradient = False

    output_data = moe_utils._AllToAll.apply(
        [world_size, 1], input_data, None, None, tp_group
    )
    expected = np.array(
        [[i * 10 + rank] for i in range(world_size)], dtype=np.float32
    )
    np.testing.assert_array_equal(output_data.numpy(), expected)

    # Backward is another all-to-all. With upstream grad row value
    # (rank*100 + i), the gradient landing at rank r row j is (j*100 + rank).
    grad_seed = paddle.to_tensor(
        [[rank * 100 + i] for i in range(world_size)], dtype="float32"
    ).cuda()
    paddle.autograd.backward([output_data], [grad_seed])
    expected_grad = np.array(
        [[j * 100 + rank] for j in range(world_size)], dtype=np.float32
    )
    np.testing.assert_array_equal(input_data.grad.numpy(), expected_grad)


def test_all_gather_group_rank_order():
    # Real all-gather along axis 0 concatenates every rank's row in rank order.
    # Rank r contributes row (r*10 + c); the gathered result on every rank is
    # rows 0..W-1 with value (i*10 + c). A wrong rank order changes the layout.
    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
    rank = tp_group.rank
    world_size = tp_group.world_size

    local = paddle.to_tensor(
        [[rank * 10 + c for c in range(4)]], dtype="float32"
    ).cuda()
    gathered = moe_utils.all_gather_group(local, group=tp_group)

    expected = np.array(
        [[i * 10 + c for c in range(4)] for i in range(world_size)],
        dtype=np.float32,
    )
    assert list(gathered.shape) == [world_size, 4]
    np.testing.assert_array_equal(gathered.numpy(), expected)


def test_reduce_scatter_group_sums_then_scatters():
    # Real reduce-scatter: sum the [W, 1] inputs across all ranks, then rank r
    # keeps row r of the sum. Input row value is (rank*100 + j); summing over
    # ranks 0..W-1 gives row j = 100*S + W*j with S = W*(W-1)/2, so rank r
    # receives (100*S + W*r). A missing reduction term or wrong scatter offset
    # changes this value.
    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
    rank = tp_group.rank
    world_size = tp_group.world_size

    local = paddle.to_tensor(
        [[rank * 100 + j] for j in range(world_size)], dtype="float32"
    ).cuda()
    scattered = moe_utils.reduce_scatter_group(local, group=tp_group)

    s = world_size * (world_size - 1) // 2
    expected = np.array([[100.0 * s + world_size * rank]], dtype=np.float32)
    assert list(scattered.shape) == [1, 1]
    np.testing.assert_array_equal(scattered.numpy(), expected)


def test_all_gather_group_op_forward_backward():
    # AllGatherGroupOp: forward all-gather (axis 0), backward reduce-scatter.
    # Forward: rank r feeds a constant row (r+1); gathered row i is (i+1).
    # Backward: upstream grad row i, col c at rank r is (r*100 + i*10 + c);
    # reduce-scatter sums over ranks -> row i,c = 100*S + W*(i*10 + c), and
    # rank r keeps row r -> (100*S + 10*W*r + W*c), with S = W*(W-1)/2.
    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
    rank = tp_group.rank
    world_size = tp_group.world_size

    inp = (paddle.ones([1, 4], dtype="float32") * (rank + 1)).cuda()
    inp.stop_gradient = False

    gathered = moe_utils.AllGatherGroupOp.apply(inp, tp_group)
    expected_fwd = np.array(
        [[i + 1] * 4 for i in range(world_size)], dtype=np.float32
    )
    assert list(gathered.shape) == [world_size, 4]
    np.testing.assert_array_equal(gathered.numpy(), expected_fwd)

    grad_seed = paddle.to_tensor(
        [
            [rank * 100 + i * 10 + c for c in range(4)]
            for i in range(world_size)
        ],
        dtype="float32",
    ).cuda()
    paddle.autograd.backward([gathered], [grad_seed])

    s = world_size * (world_size - 1) // 2
    expected_grad = np.array(
        [
            [
                100.0 * s + 10 * world_size * rank + world_size * c
                for c in range(4)
            ]
        ],
        dtype=np.float32,
    )
    assert list(inp.grad.shape) == [1, 4]
    np.testing.assert_array_equal(inp.grad.numpy(), expected_grad)


def test_reduce_scatter_group_op_forward_backward():
    # ReduceScatterGroupOp: forward reduce-scatter, backward all-gather.
    # Forward: input row value (rank*100 + j) -> rank r keeps (100*S + W*r).
    # Backward: upstream grad at rank r is scalar (rank*7 + 3); the all-gather
    # backward replicates every rank's grad in rank order, so every rank's
    # input grad row i is (i*7 + 3) with NO cross-rank reduction.
    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
    rank = tp_group.rank
    world_size = tp_group.world_size

    inp = paddle.to_tensor(
        [[rank * 100 + j] for j in range(world_size)], dtype="float32"
    ).cuda()
    inp.stop_gradient = False

    scattered = moe_utils.ReduceScatterGroupOp.apply(inp, tp_group)
    s = world_size * (world_size - 1) // 2
    expected_fwd = np.array([[100.0 * s + world_size * rank]], dtype=np.float32)
    assert list(scattered.shape) == [1, 1]
    np.testing.assert_array_equal(scattered.numpy(), expected_fwd)

    grad_seed = paddle.to_tensor([[rank * 7 + 3]], dtype="float32").cuda()
    paddle.autograd.backward([scattered], [grad_seed])
    expected_grad = np.array(
        [[i * 7 + 3] for i in range(world_size)], dtype=np.float32
    )
    assert list(inp.grad.shape) == [world_size, 1]
    np.testing.assert_array_equal(inp.grad.numpy(), expected_grad)


if __name__ == "__main__":
    Utils.initialize_model_parallel(
        tensor_parallel_size=4, sharding_parallel_size=2
    )
    test_permute_groups_tokens_by_expert()
    test_unpermute_scatter_add_sums_expert_copies()
    test_permute_unpermute_roundtrip_top1()
    test_add_auxiliary_loss_injects_unit_gradient()
    test_all_to_all_equal_split()
    test_all_gather_group_rank_order()
    test_reduce_scatter_group_sums_then_scatters()
    test_all_gather_group_op_forward_backward()
    test_reduce_scatter_group_op_forward_backward()
