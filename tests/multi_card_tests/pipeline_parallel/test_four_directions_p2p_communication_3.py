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
import unittest

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

# Topology under test: PP=4, MP=1 (num_gpus=4). With mp_degree==1 the partial
# split path is inactive, so send_partial/recv_partial exercise the real whole
# tensor isend/recv collectives across the four pipeline p2p groups.
PP_DEGREE = 4
MP_DEGREE = 1


def _init_pipeline_parallel():
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": MP_DEGREE,
        "pp_degree": PP_DEGREE,
    }
    fleet.init(is_collective=True, strategy=strategy)


def setUpModule():
    _init_pipeline_parallel()
    hcg = fleet.get_hybrid_communicate_group()
    # Registers the module level _hcg and the four p2p groups used by
    # send_partial / recv_partial, and enables the partial send/recv branch.
    initialize_p2p_groups(
        hcg, enable_partial_send_recv=True, enable_timer=False
    )


class TestIsValidSendRecvPartial(unittest.TestCase):
    """Real production guard logic; expectations derived by hand.

    initialize_p2p_groups(enable_partial_send_recv=True) has set the module
    flag, so the early `not _enable_partial_send_recv` return is not taken and
    the divisibility contract `mp_degree > 1 and numel % mp_degree == 0` is
    what actually decides the result.
    """

    def test_divisible_numel_with_mp_gt_one_is_valid(self):
        # numel = 2 * 4 = 8; 8 % 2 == 0 and 2 > 1 -> True.
        tensor = paddle.ones([2, 4], dtype="float32")
        self.assertTrue(_is_valid_send_recv_partial(tensor, 2))
        # same numel 8; 8 % 4 == 0 and 4 > 1 -> True.
        self.assertTrue(_is_valid_send_recv_partial(tensor, 4))

    def test_indivisible_numel_is_invalid(self):
        # numel = 2 * 3 = 6; 6 % 4 == 2 != 0 -> False.
        tensor = paddle.ones([2, 3], dtype="float32")
        self.assertFalse(_is_valid_send_recv_partial(tensor, 4))

    def test_mp_degree_one_is_invalid(self):
        # mp_degree 1 is not > 1 -> False regardless of divisibility.
        tensor = paddle.ones([2, 4], dtype="float32")
        self.assertFalse(_is_valid_send_recv_partial(tensor, 1))

    def test_zero_element_tensor_raises(self):
        # numel == 0 must trip the production `assert tensor_numel != 0`.
        tensor = paddle.zeros([0, 4], dtype="float32")
        with self.assertRaises(AssertionError):
            _is_valid_send_recv_partial(tensor, 2)


class TestFourDirectionsP2P(unittest.TestCase):
    """Real 4-rank pipeline p2p over the four direction groups.

    Each rank builds a rank-distinguishable payload (a fixed [2, 4] ramp
    shifted by stage_id * 100) and rotates it around the 4-stage ring. The
    received buffer is compared against the hand-derived payload of the peer
    stage, so a reversed direction, a wrong peer, or a dropped rank changes the
    exact received values.
    """

    @staticmethod
    def _ramp_numpy():
        # [[0, 1, 2, 3], [4, 5, 6, 7]] -- position-distinguishable content.
        return np.arange(8, dtype="float32").reshape([2, 4])

    @unittest.skipIf(
        not (paddle.is_compiled_with_cuda() and dist.is_initialized()),
        "four-directions pipeline p2p requires the 4-GPU launcher",
    )
    def test_send_next_recv_prev_ring(self):
        hcg = fleet.get_hybrid_communicate_group()
        stage_id = hcg.get_stage_id()
        mp_rank = hcg.get_model_parallel_rank()
        (
            send_next_group,
            _send_prev_group,
            _recv_next_group,
            recv_prev_group,
        ) = hcg.get_p2p_groups()

        base = paddle.arange(8, dtype="float32").reshape([2, 4]).cuda()
        send_tensor = base + stage_id * 100.0
        recv_tensor = paddle.empty([2, 4], dtype="float32").cuda()

        # Forward ring: send to the next stage, receive from the previous one.
        # isend is asynchronous while recv blocks, so posting all sends first
        # then all receives keeps the full ring deadlock free.
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
        expected = self._ramp_numpy() + prev_stage * 100.0
        np.testing.assert_array_equal(recv_tensor.numpy(), expected)

    @unittest.skipIf(
        not (paddle.is_compiled_with_cuda() and dist.is_initialized()),
        "four-directions pipeline p2p requires the 4-GPU launcher",
    )
    def test_send_prev_recv_next_ring(self):
        hcg = fleet.get_hybrid_communicate_group()
        stage_id = hcg.get_stage_id()
        mp_rank = hcg.get_model_parallel_rank()
        (
            _send_next_group,
            send_prev_group,
            recv_next_group,
            _recv_prev_group,
        ) = hcg.get_p2p_groups()

        base = paddle.arange(8, dtype="float32").reshape([2, 4]).cuda()
        send_tensor = base + stage_id * 100.0
        recv_tensor = paddle.empty([2, 4], dtype="float32").cuda()

        # Backward ring: send to the previous stage, receive from the next one.
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
        expected = self._ramp_numpy() + next_stage * 100.0
        np.testing.assert_array_equal(recv_tensor.numpy(), expected)


if __name__ == "__main__":
    unittest.main()
