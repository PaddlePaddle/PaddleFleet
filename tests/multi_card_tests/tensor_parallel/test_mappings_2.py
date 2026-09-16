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

import numpy as np
import paddle

from paddlefleet.tensor_parallel import mappings
from paddlefleet.utils import get_tensor_model_parallel_group_if_none
from tests.multi_card_tests.tensor_parallel.test_utilities import Utils


def test_all_gather_last_dim_forward_and_backward():
    # Covers _AllGatherFromTensorParallelRegion (forward all-gather on the last
    # dim, backward reduce-scatter on the last dim). Not exercised by the
    # sibling test_mappings.py, which stops at the sequence/model-parallel ops.
    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
    lr = tp_group.rank
    ws = tp_group.world_size
    assert ws == 4

    # Forward: every rank owns a distinguishable [2, 2] block; the region
    # concatenates them along the LAST dim in rank order.
    local_block = np.arange(4).reshape(2, 2).astype("float32") + lr * 100.0
    input_data = paddle.to_tensor(local_block).cuda()
    output_data = mappings.all_gather_last_dim_from_tensor_parallel_region(
        input_data
    )
    expected_fwd = np.concatenate(
        [
            np.arange(4).reshape(2, 2).astype("float32") + r * 100.0
            for r in range(ws)
        ],
        axis=-1,
    )
    assert paddle.equal_all(output_data, paddle.to_tensor(expected_fwd).cuda())
    assert paddle.equal_all(
        mappings._AllGatherFromTensorParallelRegion.symbolic(
            None, input_data, tp_group
        ),
        paddle.to_tensor(expected_fwd).cuda(),
    )

    # Backward: reduce-scatter along the last dim. Column block lr is summed
    # element-wise across every rank and this rank keeps block lr.
    grad_base = np.arange(16).reshape(2, 8).astype("float32")
    grad_output = paddle.to_tensor(grad_base + lr * 1000.0).cuda()

    class Ctx:
        group = tp_group

    actual_bwd = mappings._AllGatherFromTensorParallelRegion.backward(
        Ctx(), grad_output
    )
    expected_bwd = ws * grad_base[:, 2 * lr : 2 * lr + 2] + 1000.0 * sum(
        range(ws)
    )
    assert paddle.equal_all(actual_bwd, paddle.to_tensor(expected_bwd).cuda())


def test_reduce_scatter_last_dim_forward_and_backward():
    # Covers _ReduceScatterToTensorParallelRegion (forward reduce-scatter on the
    # last dim, backward all-gather on the last dim).
    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
    lr = tp_group.rank
    ws = tp_group.world_size
    assert ws == 4

    # Forward: reduce-scatter along the last dim. Column block lr is summed
    # across ranks; this rank keeps block lr.
    fwd_base = np.arange(16).reshape(2, 8).astype("float32")
    input_data = paddle.to_tensor(fwd_base + lr * 1000.0).cuda()
    output_data = mappings.reduce_scatter_last_dim_to_tensor_parallel_region(
        input_data
    )
    expected_fwd = ws * fwd_base[:, 2 * lr : 2 * lr + 2] + 1000.0 * sum(
        range(ws)
    )
    assert paddle.equal_all(output_data, paddle.to_tensor(expected_fwd).cuda())
    assert paddle.equal_all(
        mappings._ReduceScatterToTensorParallelRegion.symbolic(
            None, input_data, tp_group
        ),
        paddle.to_tensor(expected_fwd).cuda(),
    )

    # Backward: all-gather along the last dim, concatenating rank blocks in
    # order.
    grad_base = np.arange(4).reshape(2, 2).astype("float32")
    grad_output = paddle.to_tensor(grad_base + lr * 10.0).cuda()

    class Ctx:
        group = tp_group

    actual_bwd = mappings._ReduceScatterToTensorParallelRegion.backward(
        Ctx(), grad_output
    )
    expected_bwd = np.concatenate(
        [
            np.arange(4).reshape(2, 2).astype("float32") + r * 10.0
            for r in range(ws)
        ],
        axis=-1,
    )
    assert paddle.equal_all(actual_bwd, paddle.to_tensor(expected_bwd).cuda())


def test_all_to_all_forward_and_backward():
    # Covers _AllToAll / all_to_all: equal-split all-to-all and its inverse
    # routing in backward.
    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
    lr = tp_group.rank
    ws = tp_group.world_size
    assert ws == 4

    # Forward: rank i receives, as its row r, row i of rank r's input.
    local_in = np.array(
        [[lr * 100 + j * 10, lr * 100 + j * 10 + 1] for j in range(ws)],
        dtype="float32",
    )
    input_data = paddle.to_tensor(local_in).cuda()
    output_data = mappings.all_to_all(tp_group, input_data)
    expected_fwd = np.array(
        [[r * 100 + lr * 10, r * 100 + lr * 10 + 1] for r in range(ws)],
        dtype="float32",
    )
    assert paddle.equal_all(output_data, paddle.to_tensor(expected_fwd).cuda())

    # Backward: gradient routing is the inverse all-to-all. grad wrt input on
    # rank lr, row r, equals upstream row lr on rank r.
    grad_local = np.array(
        [[lr * 1000 + j * 10, lr * 1000 + j * 10 + 1] for j in range(ws)],
        dtype="float32",
    )
    grad_output = paddle.to_tensor(grad_local).cuda()

    class Ctx:
        group = tp_group
        output_split_sizes = None
        input_split_sizes = None

    grads = mappings._AllToAll.backward(Ctx(), grad_output)
    grad_input = grads[1]
    expected_bwd = np.array(
        [[r * 1000 + lr * 10, r * 1000 + lr * 10 + 1] for r in range(ws)],
        dtype="float32",
    )
    assert paddle.equal_all(grad_input, paddle.to_tensor(expected_bwd).cuda())


def test_all_to_all_hp2sp_forward():
    # Covers all_to_all_hp2sp: [num_tokens, H/TP] -> [num_tokens/TP, H].
    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
    lr = tp_group.rank
    ws = tp_group.world_size
    assert ws == 4

    # num_tokens=4, H/TP=2 -> output [1, 8].
    local_in = np.array(
        [[lr * 100 + j * 10, lr * 100 + j * 10 + 1] for j in range(ws)],
        dtype="float32",
    )
    input_data = paddle.to_tensor(local_in).cuda()
    output_data = mappings.all_to_all_hp2sp(input_data)
    expected = np.concatenate(
        [
            np.array(
                [[r * 100 + lr * 10, r * 100 + lr * 10 + 1]], dtype="float32"
            )
            for r in range(ws)
        ],
        axis=-1,
    )
    assert paddle.equal_all(output_data, paddle.to_tensor(expected).cuda())


def test_all_to_all_sp2hp_divisibility_guard_is_inverted():
    # REAL BUG: all_to_all_sp2hp guards with `assert input_.shape[-1] %
    # world_size`, which is truthy only when the hidden dim is NOT divisible by
    # world size, yet the message states it must BE divisible. A valid,
    # divisible hidden dim therefore raises AssertionError instead of running.
    # See src/paddlefleet/tensor_parallel/mappings.py:620.
    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
    ws = tp_group.world_size
    assert ws == 4

    hidden = 2 * ws  # divisible by world size -> the intended valid case
    input_data = paddle.ones([4, hidden]).cuda()
    raised = False
    try:
        mappings.all_to_all_sp2hp(input_data)
    except AssertionError:
        raised = True
    assert raised, (
        "expected AssertionError from the inverted divisibility check in "
        "all_to_all_sp2hp"
    )


if __name__ == "__main__":
    Utils.initialize_model_parallel(4, 1)
    test_all_gather_last_dim_forward_and_backward()
    test_reduce_scatter_last_dim_forward_and_backward()
    test_all_to_all_forward_and_backward()
    test_all_to_all_hp2sp_forward()
    test_all_to_all_sp2hp_divisibility_guard_is_inverted()
