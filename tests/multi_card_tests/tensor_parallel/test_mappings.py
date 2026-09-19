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

# Referred to NVIDIA Megatron-LM https://github.com/NVIDIA/Megatron-LM.git
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import paddle

from paddlefleet.tensor_parallel import mappings
from paddlefleet.utils import get_tensor_model_parallel_group_if_none
from tests.multi_card_tests.tensor_parallel.test_utilities import Utils


def test_CopyToModelParallelRegion():
    input_data = paddle.ones(1).cuda() * Utils.rank

    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)

    class Ctx:
        group = tp_group

    output_data = mappings._CopyToModelParallelRegion.backward(
        Ctx(), input_data
    )
    result = paddle.ones([1]).cuda()
    result = result * 22 if Utils.rank >= 4 else result * 6
    assert paddle.equal_all(output_data, result)
    assert paddle.equal_all(
        input_data, mappings.copy_to_tensor_model_parallel_region(input_data)
    )
    assert paddle.equal_all(
        input_data,
        mappings._CopyToModelParallelRegion.symbolic(
            None, input_data, tp_group
        ),
    )


def test_ReduceFromModelParallelRegion():
    input_data = paddle.ones(1).cuda() * Utils.rank

    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
    output_data = mappings._ReduceFromModelParallelRegion.symbolic(
        None, input_data, tp_group
    )

    result = paddle.ones(1).cuda()
    result = result * 22 if Utils.rank >= 4 else result * 6
    assert paddle.equal_all(output_data, result)

    input_data = paddle.ones(1).cuda() * Utils.rank
    assert paddle.equal_all(
        mappings.reduce_from_tensor_model_parallel_region(input_data), result
    )

    class Ctx:
        group = tp_group

    output_data = mappings._ReduceFromModelParallelRegion.backward(
        Ctx(), input_data
    )
    assert paddle.equal_all(input_data, output_data)


def test_ScatterToModelParallelRegion():
    input_data = paddle.rand((8, 4)).cuda()

    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
    output_data = mappings.scatter_to_tensor_model_parallel_region(input_data)

    req_dim = int(Utils.rank)
    assert paddle.equal_all(output_data, input_data[:, req_dim].reshape((8, 1)))
    output_data = mappings._ScatterToModelParallelRegion.symbolic(
        None, input_data, tp_group
    )
    assert paddle.equal_all(output_data, input_data[:, req_dim].reshape((8, 1)))

    input_data = paddle.ones([8]).cuda() * Utils.rank

    class Ctx:
        group = tp_group

    actual_output_data = mappings._ScatterToModelParallelRegion.backward(
        Ctx(), input_data
    )
    expected_output = paddle.cat(
        (
            paddle.ones([8]) * 0,
            paddle.ones([8]) * 1,
            paddle.ones([8]) * 2,
            paddle.ones([8]) * 3,
        )
    ).cuda()
    if Utils.rank >= 4:
        expected_output = expected_output + 4
    assert paddle.equal_all(actual_output_data, expected_output)


def test_GatherFromModelParallelRegion():
    input_data = paddle.rand((8, 4)).cuda()

    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
    req_dim = Utils.rank

    class Ctx:
        group = tp_group

    output_data = mappings._GatherFromModelParallelRegion.backward(
        Ctx(), input_data
    )
    assert paddle.equal_all(output_data, input_data[:, req_dim].reshape((8, 1)))

    input_data = paddle.ones([8]).cuda() * Utils.rank
    actual_output_data = mappings.gather_from_tensor_model_parallel_region(
        input_data
    )
    expected_output = paddle.cat(
        (
            paddle.ones([8]) * 0,
            paddle.ones([8]) * 1,
            paddle.ones([8]) * 2,
            paddle.ones([8]) * 3,
        )
    ).cuda()
    if Utils.rank >= 4:
        expected_output = expected_output + 4
    assert paddle.equal_all(actual_output_data, expected_output)
    assert paddle.equal_all(
        mappings._GatherFromModelParallelRegion.symbolic(
            None, input_data, tp_group
        ),
        expected_output,
    )


def test_ScatterToSequenceParallelRegion():
    input_data = paddle.rand((8, 4)).cuda()

    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
    req_dim = Utils.rank * 2
    output_data = mappings._ScatterToSequenceParallelRegion.symbolic(
        None, input_data, tp_group
    )
    assert paddle.equal_all(output_data, input_data[req_dim : req_dim + 2, :])
    output_data = mappings.scatter_to_sequence_parallel_region(input_data)
    assert paddle.equal_all(output_data, input_data[req_dim : req_dim + 2, :])

    input_data = paddle.ones([4]).cuda() * Utils.rank

    class Ctx:
        group = tp_group

    output_data = mappings._ScatterToModelParallelRegion.backward(
        Ctx(), input_data
    )
    expected_output = paddle.concat(
        (
            paddle.ones([4]) * 0,
            paddle.ones([4]) * 1,
            paddle.ones([4]) * 2,
            paddle.ones([4]) * 3,
        )
    ).cuda()
    if Utils.rank >= 4:
        expected_output = expected_output + 4
    assert paddle.equal_all(output_data, expected_output)


def test_GatherFromSequenceParallelRegion():
    input_data = paddle.ones([4]).cuda() * Utils.rank

    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
    output_data = mappings.gather_from_sequence_parallel_region(input_data)
    expected_output = paddle.concat(
        (
            paddle.ones([4]) * 0,
            paddle.ones([4]) * 1,
            paddle.ones([4]) * 2,
            paddle.ones([4]) * 3,
        )
    ).cuda()
    if Utils.rank >= 4:
        expected_output = expected_output + 4
    assert paddle.equal_all(output_data, expected_output)
    assert paddle.equal_all(
        mappings._GatherFromSequenceParallelRegion.symbolic(
            None, input_data, tp_group
        ),
        expected_output,
    )
    input_data = paddle.vstack(
        (
            paddle.ones([4]) * 0,
            paddle.ones([4]) * 1,
            paddle.ones([4]) * 2,
            paddle.ones([4]) * 3,
        )
    ).cuda()

    class Ctx:
        tensor_parallel_output_grad = True
        output_split_sizes = None
        group = tp_group
        use_global_buffer = False

    output_data = mappings._GatherFromSequenceParallelRegion.backward(
        Ctx(), input_data
    )
    expected_output = paddle.ones((1, 4)).cuda() * 4 * int(Utils.rank % 4)
    assert paddle.equal_all(output_data, expected_output)


def test_ReduceScatterToSequenceParallelRegion():
    input_data = paddle.vstack(
        (
            paddle.ones([4]) * 0,
            paddle.ones([4]) * 1,
            paddle.ones([4]) * 2,
            paddle.ones([4]) * 3,
        )
    ).cuda()

    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
    output_data = mappings.reduce_scatter_to_sequence_parallel_region(
        input_data
    )
    expected_output = paddle.ones([1, 4]).cuda() * 4 * int(Utils.rank % 4)
    assert paddle.equal_all(output_data, expected_output)
    assert paddle.equal_all(
        mappings._ReduceScatterToSequenceParallelRegion.symbolic(
            None, input_data, tp_group
        ),
        expected_output.reshape((1, 4)),
    )
    input_data = paddle.ones([4]).cuda() * Utils.rank

    class Ctx:
        input_split_sizes = None
        group = tp_group
        use_global_buffer = False

    output_data = mappings._ReduceScatterToSequenceParallelRegion.backward(
        Ctx(), input_data
    )
    expected_output = paddle.concat(
        (
            paddle.ones([4]) * 0,
            paddle.ones([4]) * 1,
            paddle.ones([4]) * 2,
            paddle.ones([4]) * 3,
        )
    ).cuda()
    if Utils.rank >= 4:
        expected_output = expected_output + 4
    assert paddle.equal_all(output_data, expected_output)


def test_AllGatherFromTensorParallelRegion():
    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
    local_rank = tp_group.rank

    # Forward: all-gather along last dim concatenates every rank's local
    # column in rank order. Each rank contributes a distinct constant so a
    # wrong gather order or a dropped rank changes the exact column layout.
    input_data = (
        paddle.ones([2, 1], dtype="float32") * (local_rank + 1)
    ).cuda()
    output_data = mappings.all_gather_last_dim_from_tensor_parallel_region(
        input_data
    )
    expected = paddle.to_tensor(
        [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]], dtype="float32"
    ).cuda()
    assert paddle.equal_all(output_data, expected)
    assert paddle.equal_all(
        mappings._AllGatherFromTensorParallelRegion.symbolic(
            None, input_data, tp_group
        ),
        expected,
    )

    # Backward: reduce-scatter along last dim. Rank r keeps the sum over all
    # ranks of the r-th last-dim chunk. With row value (r*100 + col) the sum
    # over ranks 0..3 of a two-wide chunk starting at column 2r is
    # [600 + 8r, 604 + 8r] on both rows.
    row = paddle.arange(8, dtype="float32") + local_rank * 100
    grad_input = paddle.stack([row, row], axis=0).cuda()

    class Ctx:
        group = tp_group

    grad_output = mappings._AllGatherFromTensorParallelRegion.backward(
        Ctx(), grad_input
    )
    c0 = 600.0 + 8 * local_rank
    c1 = 604.0 + 8 * local_rank
    expected_grad = paddle.to_tensor(
        [[c0, c1], [c0, c1]], dtype="float32"
    ).cuda()
    assert paddle.equal_all(grad_output, expected_grad)


def test_ReduceScatterToTensorParallelRegion():
    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
    local_rank = tp_group.rank

    # Forward: reduce-scatter along last dim. Rank r receives the summed
    # r-th last-dim chunk across all ranks. Input row value is (r*100 + col),
    # so rank r's slice is [600 + 8r, 604 + 8r] on both rows.
    row = paddle.arange(8, dtype="float32") + local_rank * 100
    input_data = paddle.stack([row, row], axis=0).cuda()
    output_data = mappings.reduce_scatter_last_dim_to_tensor_parallel_region(
        input_data
    )
    c0 = 600.0 + 8 * local_rank
    c1 = 604.0 + 8 * local_rank
    expected = paddle.to_tensor([[c0, c1], [c0, c1]], dtype="float32").cuda()
    assert paddle.equal_all(output_data, expected)
    assert paddle.equal_all(
        mappings._ReduceScatterToTensorParallelRegion.symbolic(
            None, input_data, tp_group
        ),
        expected,
    )

    # Backward: all-gather along last dim. Each rank feeds a distinct constant
    # column; the gathered result is [1, 2, 3, 4] in rank order on both rows.
    grad_input = (
        paddle.ones([2, 1], dtype="float32") * (local_rank + 1)
    ).cuda()

    class Ctx:
        group = tp_group

    grad_output = mappings._ReduceScatterToTensorParallelRegion.backward(
        Ctx(), grad_input
    )
    expected_grad = paddle.to_tensor(
        [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]], dtype="float32"
    ).cuda()
    assert paddle.equal_all(grad_output, expected_grad)


def test_AllToAll():
    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
    local_rank = tp_group.rank

    # Equal-split all-to-all: rank r sends its j-th row block to rank j and
    # receives, as its i-th row, the r-th block from rank i. With input row
    # value (r*10 + j), rank r's output is [r, 10+r, 20+r, 30+r]. A reversed
    # direction or wrong peer selection changes these exact values.
    input_data = (
        (paddle.arange(4, dtype="float32") + local_rank * 10)
        .reshape([4, 1])
        .cuda()
    )
    output_data = mappings.all_to_all(tp_group, input_data)
    expected = paddle.to_tensor(
        [
            [float(local_rank)],
            [10.0 + local_rank],
            [20.0 + local_rank],
            [30.0 + local_rank],
        ],
        dtype="float32",
    ).cuda()
    assert paddle.equal_all(output_data, expected)


def test_all_to_all_hp2sp():
    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
    rank = tp_group.rank
    world_size = tp_group.world_size

    # Input holds [num_tokens, H/TP]; here 4 tokens x 2 local-hidden channels.
    # Encode each element as rank*100 + row*10 + col so every rank, token and
    # channel is distinguishable: a wrong all-to-all peer, a wrong split axis
    # or a wrong concat order all change the exact reconstructed row.
    input_data = paddle.to_tensor(
        [[rank * 100 + row * 10 + col for col in range(2)] for row in range(4)],
        dtype="float32",
    ).cuda()

    output_data = mappings.all_to_all_hp2sp(input_data)

    # Independent hand derivation. hp2sp maps [T, H/TP] -> [T/TP, H]. With
    # T=4, TP=4, H/TP=2 each rank ends up owning token `rank` with the full
    # hidden regathered from every rank's local slice, giving one row [1, 8]:
    # element (s, c) = source-rank s's local channel c for this token, i.e.
    #   s*100 + rank*10 + c.
    expected = paddle.to_tensor(
        [
            [
                s * 100 + rank * 10 + c
                for s in range(world_size)
                for c in range(2)
            ]
        ],
        dtype="float32",
    ).cuda()
    assert list(output_data.shape) == [1, world_size * 2]
    assert paddle.equal_all(output_data, expected)


def test_all_to_all_sp2hp_divisibility_guard_is_inverted():
    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
    world_size = tp_group.world_size

    # A valid sp2hp input has a hidden size divisible by TP. Production guards
    # it with `assert input_.shape[-1] % world_size` (missing `== 0`): for the
    # CORRECT divisible case `H % world_size == 0`, which is falsy and raises.
    # Lock the real bug -- a valid divisible input must currently raise
    # AssertionError. When the guard is fixed to `== 0` this stops raising and
    # the test fails, surfacing the regression. Do not edit production here.
    input_data = paddle.to_tensor(
        [[float(c) for c in range(world_size)] for _ in range(2)],
        dtype="float32",
    ).cuda()

    raised = False
    try:
        mappings.all_to_all_sp2hp(input_data)
    except AssertionError:
        raised = True
    assert raised, (
        "expected all_to_all_sp2hp to raise AssertionError on valid divisible "
        "input due to the inverted divisibility guard (missing '== 0')"
    )


if __name__ == "__main__":
    Utils.initialize_model_parallel(4, 1)
    test_CopyToModelParallelRegion()
    test_ReduceFromModelParallelRegion()
    test_ScatterToModelParallelRegion()
    test_GatherFromModelParallelRegion()
    test_ReduceScatterToSequenceParallelRegion()
    test_GatherFromSequenceParallelRegion()
    test_AllGatherFromTensorParallelRegion()
    test_ReduceScatterToTensorParallelRegion()
    test_AllToAll()
    test_all_to_all_hp2sp()
    test_all_to_all_sp2hp_divisibility_guard_is_inverted()
