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

"""Real multi-card (PP=2) behavior tests for pipeline-parallel p2p.

These tests run under ``paddle.distributed.launch`` with 2 GPUs; each rank is a
distinct pipeline stage that exchanges rank-distinguishable payloads through the
REAL production path
(``paddlefleet.pipeline_parallel.pp_utils.p2p_communication``). This file
targets paths NOT covered by ``test_p2p_communication.py``: the multi-tensor
(tuple) forward transfer and the dynamic-shape forward transfer across two
rounds. Expected received tensors are derived by hand on the receiving side, so
a reversed direction, a wrong peer, a wrong per-tensor ordering, or a dropped
message all produce a value mismatch (an empty recv buffer equals neither).
"""

import paddle
from paddle.distributed import fleet

from paddlefleet.pipeline_parallel.pp_utils.p2p_communication import (
    P2pHelper,
    initialize_p2p_groups,
)

PP_DEGREE = 2


def _init_pipeline_parallel():
    """Bring up a real PP=2 hybrid group and wire the p2p module to its HCG."""
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": PP_DEGREE,
    }
    fleet.init(is_collective=True, strategy=strategy)
    hcg = fleet.get_hybrid_communicate_group()
    # mp_degree == 1, so keep partial send/recv off: each tensor travels as a
    # single message and the received content is fully comparable end to end.
    initialize_p2p_groups(hcg, enable_partial_send_recv=False)
    return hcg


def test_send_forward_recv_forward_transfers_tuple_of_tensors():
    """A tuple activation crosses stages preserving per-tensor identity.

    Stage 0 sends a tuple of two distinguishable tensors with DIFFERENT shapes
    (so a swapped order or wrong shape is visible) forward to stage 1. Stage 1
    receives the tuple and checks each element's shape and exact content against
    hand-written expectations. This exercises the tuple meta encode/decode plus
    the multi-tensor allocation branch of ``_p2p_helper`` over real collectives.
    """
    hcg = fleet.get_hybrid_communicate_group()
    stage_id = hcg.get_stage_id()
    p2p = P2pHelper(use_cache=True, dynamic_shape=False)

    first = (paddle.arange(6, dtype="float32").reshape([2, 3]) + 10.0).cuda()
    first.stop_gradient = False
    second = (paddle.arange(4, dtype="float32") + 50.0).cuda()
    second.stop_gradient = False
    payload = (first, second)

    if stage_id == 0:
        # First stage: emit the two-tensor activation downstream.
        p2p.send_forward(payload, pp_last_stage=False, batch_p2p_comm=False)
    else:
        # Last stage: receive the tuple and verify each tensor element-wise.
        received = p2p.recv_forward(pp_first_stage=False, batch_p2p_comm=False)
        assert isinstance(received, tuple), type(received)
        assert len(received) == 2, len(received)
        # Hand-derived: exactly what stage 0 built for each tuple slot.
        expected_first = (
            paddle.arange(6, dtype="float32").reshape([2, 3]) + 10.0
        ).cuda()
        expected_second = (paddle.arange(4, dtype="float32") + 50.0).cuda()
        assert list(received[0].shape) == [2, 3], received[0].shape
        assert received[0].dtype == paddle.float32, received[0].dtype
        assert paddle.equal_all(received[0], expected_first)
        assert list(received[1].shape) == [4], received[1].shape
        assert received[1].dtype == paddle.float32, received[1].dtype
        assert paddle.equal_all(received[1], expected_second)


def test_dynamic_shape_forward_transfers_changing_shapes():
    """Dynamic-shape mode carries a different shape on each successive round.

    With ``dynamic_shape=True`` each ``send_forward`` transmits fresh metadata,
    so stage 0 can send two activations of DIFFERENT shapes back to back and
    stage 1 must reconstruct each round's shape and content exactly. A meta
    cache that leaked round 0's shape into round 1, or a dropped round, changes
    the decoded result. Expected tensors are hand-written on the receiver.
    """
    hcg = fleet.get_hybrid_communicate_group()
    stage_id = hcg.get_stage_id()
    p2p = P2pHelper(use_cache=True, dynamic_shape=True)

    round0 = (paddle.arange(6, dtype="float32").reshape([2, 3]) + 200.0).cuda()
    round0.stop_gradient = False
    round1 = (paddle.arange(5, dtype="float32") + 700.0).cuda()
    round1.stop_gradient = False

    if stage_id == 0:
        # First stage: two forward emissions with distinct shapes.
        p2p.send_forward(round0, pp_last_stage=False, batch_p2p_comm=False)
        p2p.send_forward(round1, pp_last_stage=False, batch_p2p_comm=False)
    else:
        # Last stage: reconstruct each round independently.
        got0 = p2p.recv_forward(pp_first_stage=False, batch_p2p_comm=False)
        got1 = p2p.recv_forward(pp_first_stage=False, batch_p2p_comm=False)
        expected0 = (
            paddle.arange(6, dtype="float32").reshape([2, 3]) + 200.0
        ).cuda()
        expected1 = (paddle.arange(5, dtype="float32") + 700.0).cuda()
        assert list(got0.shape) == [2, 3], got0.shape
        assert paddle.equal_all(got0, expected0)
        assert list(got1.shape) == [5], got1.shape
        assert paddle.equal_all(got1, expected1)


if __name__ == "__main__":
    _init_pipeline_parallel()
    test_send_forward_recv_forward_transfers_tuple_of_tensors()
    test_dynamic_shape_forward_transfers_changing_shapes()
