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

import os

os.environ["PADDLE_USE_FOUR_DIRECTIONS_P2P"] = "True"

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet

from paddlefleet.pipeline_parallel.pp_utils.four_directions_p2p_communication import (
    _is_valid_send_recv_partial,
    initialize_p2p_groups,
    recv_partial,
    send_partial,
)
from paddlefleet.training.initialize import initialize_fleet

# Topology under test: PP=4, MP=1, DP=1 (num_gpus=4). The four pipeline stages
# form a real ring. With mp_degree==1 the partial-split predicate is False, so
# send_partial / recv_partial fall through to whole-tensor isend / recv over the
# four direction p2p groups -- the real multi-card collectives, not a mock.
PP_DEGREE = 4
MP_DEGREE = 1


def _init_pipeline_parallel():
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": MP_DEGREE,
        "pp_degree": PP_DEGREE,
        "sharding_degree": 1,
        "sep_degree": 1,
        "cp_degree": 1,
        "ep_degree": 1,
        "moe_sharding_degree": 1,
        "order": [
            "sharding",
            "moe_sharding",
            "pp",
            "sep",
            "cp",
            "dp",
            "ep",
            "mp",
        ],
    }
    initialize_fleet(strategy)


def _setup():
    _init_pipeline_parallel()
    hcg = fleet.get_hybrid_communicate_group()
    # Registers the module-level _hcg and the four direction p2p groups used by
    # send_partial / recv_partial, and enables the partial send/recv branch.
    initialize_p2p_groups(
        hcg, enable_partial_send_recv=True, enable_timer=False
    )
    return hcg


def _ramp():
    # [[0, 1, 2, 3], [4, 5, 6, 7]] -- position-distinguishable base content so a
    # transposed, rolled or partially dropped buffer changes the exact values.
    return np.arange(8, dtype="float32").reshape([2, 4])


def test_is_valid_send_recv_partial_contract():
    """Per-rank local contract of the partial-send predicate.

    initialize_p2p_groups(enable_partial_send_recv=True) has set the module
    flag, so the early `not _enable_partial_send_recv` return is skipped and the
    real `mp_degree > 1 and numel % mp_degree == 0` rule decides the result.
    Expectations are hand-derived from the divisibility, not shape-only.
    """
    # numel = 2 * 4 = 8; 8 % 2 == 0 and 2 > 1 -> True.
    assert _is_valid_send_recv_partial(paddle.ones([2, 4], "float32"), 2)
    # numel 8; 8 % 4 == 0 and 4 > 1 -> True.
    assert _is_valid_send_recv_partial(paddle.ones([2, 4], "float32"), 4)
    # numel = 2 * 3 = 6; 6 % 4 == 2 != 0 -> False.
    assert not _is_valid_send_recv_partial(paddle.ones([2, 3], "float32"), 4)
    # mp_degree 1 is not > 1 -> False regardless of divisibility.
    assert not _is_valid_send_recv_partial(paddle.ones([2, 4], "float32"), 1)


def test_is_valid_send_recv_partial_zero_element_raises():
    """Lock the real zero-element guard: numel == 0 must trip the production
    `assert tensor_numel != 0`. No collective is issued; this is a per-rank
    local contract, so it is asserted on every rank."""
    zero_tensor = paddle.zeros([0, 4], dtype="float32")
    raised = False
    try:
        _is_valid_send_recv_partial(zero_tensor, 2)
    except AssertionError:
        raised = True
    assert raised, "expected AssertionError for a zero-element partial tensor"


def test_forward_ring_send_next_recv_prev():
    """Real forward ring over all 4 stages: send to the next stage, receive
    from the previous one. Each stage tags its payload with stage_id * 100, so
    the received buffer must equal the previous stage's hand-derived payload; a
    reversed direction, a wrong peer or a dropped rank changes these values."""
    hcg = fleet.get_hybrid_communicate_group()
    stage_id = hcg.get_stage_id()
    mp_rank = hcg.get_model_parallel_rank()
    (
        send_next_group,
        _send_prev_group,
        _recv_next_group,
        recv_prev_group,
    ) = hcg.get_p2p_groups()

    base = paddle.to_tensor(_ramp(), dtype="float32").cuda()
    send_tensor = base + stage_id * 100.0
    recv_tensor = paddle.empty([2, 4], dtype="float32").cuda()

    # Post the async isend to the next stage first, then the blocking recv from
    # the previous stage. With mp_degree==1 send_partial issues a non-blocking
    # whole-tensor isend, so the full 4-stage ring cannot deadlock.
    send_task = send_partial(
        send_tensor,
        dst=1,
        nranks=MP_DEGREE,
        rank_id=mp_rank,
        group=send_next_group,
        use_calc_stream=False,
    )
    recv_partial(
        recv_tensor,
        src=0,
        nranks=MP_DEGREE,
        rank_id=mp_rank,
        group=recv_prev_group,
        use_calc_stream=True,
    )
    if send_task is not None:
        send_task.wait()
    dist.barrier()

    prev_stage = (stage_id - 1) % PP_DEGREE
    expected = _ramp() + prev_stage * 100.0
    np.testing.assert_array_equal(recv_tensor.numpy(), expected)


def test_backward_ring_send_prev_recv_next():
    """Real backward ring: send to the previous stage, receive from the next
    one. The payload tag adds a distinct +7 offset so a swapped forward/backward
    direction or a stale forward buffer cannot pass as a correct result."""
    hcg = fleet.get_hybrid_communicate_group()
    stage_id = hcg.get_stage_id()
    mp_rank = hcg.get_model_parallel_rank()
    (
        _send_next_group,
        send_prev_group,
        recv_next_group,
        _recv_prev_group,
    ) = hcg.get_p2p_groups()

    base = paddle.to_tensor(_ramp(), dtype="float32").cuda()
    send_tensor = base + stage_id * 100.0 + 7.0
    recv_tensor = paddle.empty([2, 4], dtype="float32").cuda()

    send_task = send_partial(
        send_tensor,
        dst=0,
        nranks=MP_DEGREE,
        rank_id=mp_rank,
        group=send_prev_group,
        use_calc_stream=False,
    )
    recv_partial(
        recv_tensor,
        src=1,
        nranks=MP_DEGREE,
        rank_id=mp_rank,
        group=recv_next_group,
        use_calc_stream=True,
    )
    if send_task is not None:
        send_task.wait()
    dist.barrier()

    next_stage = (stage_id + 1) % PP_DEGREE
    expected = _ramp() + next_stage * 100.0 + 7.0
    np.testing.assert_array_equal(recv_tensor.numpy(), expected)


if __name__ == "__main__":
    _setup()
    test_is_valid_send_recv_partial_contract()
    test_is_valid_send_recv_partial_zero_element_raises()
    test_forward_ring_send_next_recv_prev()
    test_backward_ring_send_prev_recv_next()
