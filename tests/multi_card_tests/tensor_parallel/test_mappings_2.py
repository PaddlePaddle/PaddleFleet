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

# Real TP=4 multi-card behavior tests for paddlefleet.tensor_parallel.mappings.
# These complement the sibling test_mappings.py (which drives the last-dim
# all-gather / reduce-scatter / all-to-all ops) by covering the model-parallel
# and sequence-parallel regions end-to-end through their public autograd
# wrappers: each case runs on real GPUs in the real Fleet process group, uses
# .cuda() tensors, and executes a real collective in the forward or backward
# pass. Expected values are derived by hand from each rank's distinct input,
# never from the function under test.

import paddle

from paddlefleet.tensor_parallel import mappings
from paddlefleet.utils import get_tensor_model_parallel_group_if_none
from tests.multi_card_tests.tensor_parallel.test_utilities import Utils

TP_SIZE = 4


def test_copy_forward_identity_backward_all_reduce():
    # _CopyToModelParallelRegion: forward is identity, backward all-reduces
    # (sums) the upstream gradient across the TP group.
    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
    rank = tp_group.rank
    ws = tp_group.world_size
    assert ws == TP_SIZE

    # Each rank feeds a distinct constant (rank + 1); identity keeps it.
    local = (paddle.ones([3], dtype="float32") * (rank + 1)).cuda()
    local.stop_gradient = False
    out = mappings.copy_to_tensor_model_parallel_region(local)
    expected_fwd = (paddle.ones([3], dtype="float32") * (rank + 1)).cuda()
    assert paddle.equal_all(out, expected_fwd)

    # Backward sums the rank-distinct upstream grads: 1 + 2 + 3 + 4 = 10.
    upstream = (paddle.ones([3], dtype="float32") * (rank + 1)).cuda()
    out.backward(upstream)
    reduced = float(sum(range(1, ws + 1)))
    expected_grad = (paddle.ones([3], dtype="float32") * reduced).cuda()
    assert paddle.equal_all(local.grad, expected_grad)


def test_reduce_forward_all_reduce_backward_identity():
    # _ReduceFromModelParallelRegion: forward all-reduces (sums) across the TP
    # group, backward is identity (upstream grad passes through unchanged).
    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
    rank = tp_group.rank
    ws = tp_group.world_size
    assert ws == TP_SIZE

    local = (paddle.ones([2], dtype="float32") * (rank + 1)).cuda()
    local.stop_gradient = False
    out = mappings.reduce_from_tensor_model_parallel_region(local)
    reduced = float(sum(range(1, ws + 1)))  # 1 + 2 + 3 + 4 = 10
    expected_fwd = (paddle.ones([2], dtype="float32") * reduced).cuda()
    assert paddle.equal_all(out, expected_fwd)

    # Backward is identity: rank r keeps its own distinct upstream grad.
    upstream = (paddle.ones([2], dtype="float32") * (rank + 1) * 5).cuda()
    out.backward(upstream)
    expected_grad = (paddle.ones([2], dtype="float32") * (rank + 1) * 5).cuda()
    assert paddle.equal_all(local.grad, expected_grad)


def test_scatter_forward_keeps_rank_column_backward_gathers():
    # _ScatterToModelParallelRegion: forward keeps this rank's last-dim column,
    # backward all-gathers the per-rank column grads back along the last dim.
    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
    rank = tp_group.rank
    ws = tp_group.world_size
    assert ws == TP_SIZE

    # Deterministic full tensor, identical on every rank; last dim == ws so the
    # split produces width-1 columns and rank r keeps column r.
    full = paddle.arange(8 * ws, dtype="float32").reshape([8, ws]).cuda()
    full.stop_gradient = False
    out = mappings.scatter_to_tensor_model_parallel_region(full)
    expected_fwd = full.detach()[:, rank : rank + 1]
    assert paddle.equal_all(out, expected_fwd)

    # Backward all-gathers: rank r contributes constant (r + 1), so the
    # reconstructed grad has column c filled with (c + 1).
    upstream = (paddle.ones([8, 1], dtype="float32") * (rank + 1)).cuda()
    out.backward(upstream)
    expected_grad = paddle.concat(
        [paddle.ones([8, 1], dtype="float32") * (c + 1) for c in range(ws)],
        axis=-1,
    ).cuda()
    assert paddle.equal_all(full.grad, expected_grad)


def test_gather_forward_concats_columns_backward_splits():
    # _GatherFromModelParallelRegion: forward all-gathers each rank's column
    # along the last dim, backward keeps only this rank's column of the grad.
    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
    rank = tp_group.rank
    ws = tp_group.world_size
    assert ws == TP_SIZE

    local = (paddle.ones([2, 1], dtype="float32") * (rank + 1)).cuda()
    local.stop_gradient = False
    out = mappings.gather_from_tensor_model_parallel_region(local)
    expected_fwd = paddle.concat(
        [paddle.ones([2, 1], dtype="float32") * (c + 1) for c in range(ws)],
        axis=-1,
    ).cuda()
    assert paddle.equal_all(out, expected_fwd)

    # Backward splits along the last dim; identical upstream on every rank, so
    # rank r's grad is exactly column r of that upstream tensor.
    upstream = paddle.arange(2 * ws, dtype="float32").reshape([2, ws]).cuda()
    out.backward(upstream)
    expected_grad = upstream[:, rank : rank + 1]
    assert paddle.equal_all(local.grad, expected_grad)


def test_reduce_scatter_sequence_forward_and_backward():
    # _ReduceScatterToSequenceParallelRegion: forward reduce-scatters along the
    # first dim (sum across ranks, then keep this rank's row block); backward
    # all-gathers the per-rank grads back along the first dim.
    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
    rank = tp_group.rank
    ws = tp_group.world_size
    assert ws == TP_SIZE

    # Rank r's row i holds (r + 1) * 10 + i on both columns.
    col = (rank + 1) * 10.0 + paddle.arange(ws, dtype="float32").reshape(
        [ws, 1]
    )
    local = paddle.concat([col, col], axis=-1).cuda()
    local.stop_gradient = False
    out = mappings.reduce_scatter_to_sequence_parallel_region(local)

    # Summed row i = sum_r((r + 1) * 10 + i) = base + ws * i, base = 100 for
    # ws = 4; rank r keeps row r, so the [1, 2] output is base + ws * rank.
    base = float(sum((k + 1) * 10 for k in range(ws)))
    expected_fwd = (
        paddle.ones([1, 2], dtype="float32") * (base + ws * rank)
    ).cuda()
    assert paddle.equal_all(out, expected_fwd)

    # Backward all-gathers: rank r feeds constant (r + 1), so the reconstructed
    # grad has row r filled with (r + 1).
    upstream = (paddle.ones([1, 2], dtype="float32") * (rank + 1)).cuda()
    out.backward(upstream)
    expected_grad = paddle.concat(
        [paddle.ones([1, 2], dtype="float32") * (r + 1) for r in range(ws)],
        axis=0,
    ).cuda()
    assert paddle.equal_all(local.grad, expected_grad)


def test_gather_sequence_forward_and_backward():
    # _GatherFromSequenceParallelRegion (tensor_parallel_output_grad=True):
    # forward all-gathers along the first dim; backward reduce-scatters along
    # the first dim (sum across ranks, then keep this rank's row block).
    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
    rank = tp_group.rank
    ws = tp_group.world_size
    assert ws == TP_SIZE

    local = (paddle.ones([1, 2], dtype="float32") * (rank + 1)).cuda()
    local.stop_gradient = False
    out = mappings.gather_from_sequence_parallel_region(local)
    expected_fwd = paddle.concat(
        [paddle.ones([1, 2], dtype="float32") * (r + 1) for r in range(ws)],
        axis=0,
    ).cuda()
    assert paddle.equal_all(out, expected_fwd)

    # Backward reduce-scatters. Every rank feeds the SAME upstream (rows
    # 1..ws), so the reduce sums ws identical copies (x ws) before scattering;
    # rank r keeps row r, i.e. ws * (rank + 1).
    upstream = paddle.concat(
        [paddle.ones([1, 2], dtype="float32") * (r + 1) for r in range(ws)],
        axis=0,
    ).cuda()
    out.backward(upstream)
    expected_grad = (
        paddle.ones([1, 2], dtype="float32") * (ws * (rank + 1))
    ).cuda()
    assert paddle.equal_all(local.grad, expected_grad)


def test_all_to_all_sp2hp_divisibility_guard_is_inverted():
    # REAL BUG: all_to_all_sp2hp guards with
    #     assert input_.shape[-1] % world_size
    # which is truthy only when the hidden dim is NOT divisible by world size,
    # yet the message states it must BE divisible. A valid, divisible hidden
    # dim therefore raises AssertionError instead of running. Production is not
    # edited; this pins the current wrong behavior.
    # See src/paddlefleet/tensor_parallel/mappings.py:620.
    tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
    ws = tp_group.world_size
    assert ws == TP_SIZE

    hidden = 2 * ws  # divisible by world size -> the intended valid case
    input_data = paddle.ones([4, hidden], dtype="float32").cuda()
    raised = False
    try:
        mappings.all_to_all_sp2hp(input_data)
    except AssertionError:
        raised = True
    assert raised, (
        "expected AssertionError from the inverted divisibility check in "
        "all_to_all_sp2hp (mappings.py:620)"
    )


if __name__ == "__main__":
    Utils.initialize_model_parallel(4, 1)
    test_copy_forward_identity_backward_all_reduce()
    test_reduce_forward_all_reduce_backward_identity()
    test_scatter_forward_keeps_rank_column_backward_gathers()
    test_gather_forward_concats_columns_backward_splits()
    test_reduce_scatter_sequence_forward_and_backward()
    test_gather_sequence_forward_and_backward()
    test_all_to_all_sp2hp_divisibility_guard_is_inverted()
