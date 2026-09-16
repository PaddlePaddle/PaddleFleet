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

# Exercise the synchronous-send ordering branch of ``_p2p_helper``
# (recv_prev -> send_next -> recv_next -> send_prev). This flag is read at
# import time by the production module, so it must be set before the import.
os.environ["PADDLE_P2P_SYNC_SEND"] = "1"

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet

from paddlefleet.pipeline_parallel.pp_utils.four_directions_p2p_communication import (
    SendRecvMeta,
    _p2p_helper,
    initialize_p2p_groups,
)
from paddlefleet.pipeline_parallel.pp_utils.utils import paddle_2_number
from paddlefleet.training.initialize import initialize_fleet

PP_DEGREE = 2
FP32 = paddle_2_number(paddle.float32)


def _init_pp():
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
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


def setUpModule():
    _init_pp()
    hcg = fleet.get_hybrid_communicate_group()
    initialize_p2p_groups(hcg, enable_partial_send_recv=True)


def _linear_payload(base, shape):
    """Build a fully deterministic, position-distinguishable GPU tensor.

    Element at flat index ``k`` equals ``base + k``. ``base`` tags the logical
    sender so that a wrong peer, a reversed direction or a swapped recv slot
    yields values that differ from the hand-written expectation below. The
    expected tensors in each test are written independently as the same known
    constants; they are never produced by ``_p2p_helper``.
    """
    numel = 1
    for dim in shape:
        numel *= dim
    values = [float(base + k) for k in range(numel)]
    return paddle.to_tensor(values, dtype="float32").reshape(shape).cuda()


def _single_meta():
    meta = SendRecvMeta()
    meta.send_shape_message = [2, 4]
    meta.send_dtype_message = FP32
    meta.recv_shape_message = [2, 4]
    meta.recv_dtype_message = FP32
    meta.recv_stop_gradient = False
    return meta


def _tuple_meta():
    meta = SendRecvMeta()
    meta.send_shape_message = ([2, 4], [3, 5])
    meta.send_dtype_message = (FP32, FP32)
    meta.recv_shape_message = ([2, 4], [3, 5])
    meta.recv_dtype_message = (FP32, FP32)
    meta.recv_stop_gradient = (False, False)
    return meta


class TestSyncSendFourDirectionsP2P(unittest.TestCase):
    """Real 2-GPU (PP=2) point-to-point transfers through ``_p2p_helper`` in
    ``PADDLE_P2P_SYNC_SEND`` mode. Every payload is rank/position
    distinguishable and each receiving rank checks the exact transmitted
    values against a hand-written constant, so a reversed direction, a wrong
    peer or a swapped recv slot is rejected -- not merely ``assertIsNotNone``.
    """

    def _assert_equal(self, actual, expected):
        self.assertIsNotNone(actual)
        self.assertEqual(list(actual.shape), list(expected.shape))
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertTrue(
            bool(paddle.equal_all(actual, expected)),
            f"received {actual.numpy().tolist()} != "
            f"expected {expected.numpy().tolist()}",
        )

    def test_forward_single_tensor_content(self):
        """Stage 0 sends to next; stage 1 recv_prev must hold the exact bytes."""
        pp_rank = fleet.get_hybrid_communicate_group().get_stage_id()
        meta = _single_meta()

        if pp_rank == 0:
            payload = _linear_payload(1000, [2, 4])
            recv_prev, recv_next = _p2p_helper(
                tensor_send_next=payload,
                tensor_send_prev=None,
                recv_prev=False,
                recv_next=False,
                sync_recv=True,
                send_recv_meta=meta,
            )
            self.assertIsNone(recv_prev)
            self.assertIsNone(recv_next)
        else:
            recv_prev, recv_next = _p2p_helper(
                tensor_send_next=None,
                tensor_send_prev=None,
                recv_prev=True,
                recv_next=False,
                sync_recv=True,
                send_recv_meta=meta,
            )
            expected = paddle.to_tensor(
                [
                    [1000.0, 1001.0, 1002.0, 1003.0],
                    [1004.0, 1005.0, 1006.0, 1007.0],
                ],
                dtype="float32",
            ).cuda()
            self._assert_equal(recv_prev, expected)
            self.assertIsNone(recv_next)

        dist.barrier()

    def test_backward_single_tensor_content(self):
        """Stage 1 sends to prev; stage 0 recv_next must hold the exact bytes."""
        pp_rank = fleet.get_hybrid_communicate_group().get_stage_id()
        meta = _single_meta()

        if pp_rank == 1:
            payload = _linear_payload(2000, [2, 4])
            recv_prev, recv_next = _p2p_helper(
                tensor_send_next=None,
                tensor_send_prev=payload,
                recv_prev=False,
                recv_next=False,
                sync_recv=True,
                send_recv_meta=meta,
            )
            self.assertIsNone(recv_prev)
            self.assertIsNone(recv_next)
        else:
            recv_prev, recv_next = _p2p_helper(
                tensor_send_next=None,
                tensor_send_prev=None,
                recv_prev=False,
                recv_next=True,
                sync_recv=True,
                send_recv_meta=meta,
            )
            expected = paddle.to_tensor(
                [
                    [2000.0, 2001.0, 2002.0, 2003.0],
                    [2004.0, 2005.0, 2006.0, 2007.0],
                ],
                dtype="float32",
            ).cuda()
            self._assert_equal(recv_next, expected)
            self.assertIsNone(recv_prev)

        dist.barrier()

    def test_forward_tuple_content(self):
        """Stage 0 sends a 2-tuple next; stage 1 recv_prev preserves order
        and per-tensor content, not just tuple-ness."""
        pp_rank = fleet.get_hybrid_communicate_group().get_stage_id()
        meta = _tuple_meta()

        if pp_rank == 0:
            t1 = _linear_payload(3000, [2, 4])
            t2 = _linear_payload(4000, [3, 5])
            recv_prev, recv_next = _p2p_helper(
                tensor_send_next=(t1, t2),
                tensor_send_prev=None,
                recv_prev=False,
                recv_next=False,
                sync_recv=True,
                send_recv_meta=meta,
            )
            self.assertIsNone(recv_prev)
            self.assertIsNone(recv_next)
        else:
            recv_prev, recv_next = _p2p_helper(
                tensor_send_next=None,
                tensor_send_prev=None,
                recv_prev=True,
                recv_next=False,
                sync_recv=True,
                send_recv_meta=meta,
            )
            self.assertIsInstance(recv_prev, tuple)
            self.assertEqual(len(recv_prev), 2)
            expected_1 = paddle.to_tensor(
                [
                    [3000.0, 3001.0, 3002.0, 3003.0],
                    [3004.0, 3005.0, 3006.0, 3007.0],
                ],
                dtype="float32",
            ).cuda()
            expected_2 = paddle.to_tensor(
                [
                    [4000.0, 4001.0, 4002.0, 4003.0, 4004.0],
                    [4005.0, 4006.0, 4007.0, 4008.0, 4009.0],
                    [4010.0, 4011.0, 4012.0, 4013.0, 4014.0],
                ],
                dtype="float32",
            ).cuda()
            self._assert_equal(recv_prev[0], expected_1)
            self._assert_equal(recv_prev[1], expected_2)
            self.assertIsNone(recv_next)

        dist.barrier()

    def test_backward_tuple_content(self):
        """Stage 1 sends a 2-tuple prev; stage 0 recv_next preserves order
        and per-tensor content."""
        pp_rank = fleet.get_hybrid_communicate_group().get_stage_id()
        meta = _tuple_meta()

        if pp_rank == 1:
            t1 = _linear_payload(5000, [2, 4])
            t2 = _linear_payload(6000, [3, 5])
            recv_prev, recv_next = _p2p_helper(
                tensor_send_next=None,
                tensor_send_prev=(t1, t2),
                recv_prev=False,
                recv_next=False,
                sync_recv=True,
                send_recv_meta=meta,
            )
            self.assertIsNone(recv_prev)
            self.assertIsNone(recv_next)
        else:
            recv_prev, recv_next = _p2p_helper(
                tensor_send_next=None,
                tensor_send_prev=None,
                recv_prev=False,
                recv_next=True,
                sync_recv=True,
                send_recv_meta=meta,
            )
            self.assertIsInstance(recv_next, tuple)
            self.assertEqual(len(recv_next), 2)
            expected_1 = paddle.to_tensor(
                [
                    [5000.0, 5001.0, 5002.0, 5003.0],
                    [5004.0, 5005.0, 5006.0, 5007.0],
                ],
                dtype="float32",
            ).cuda()
            expected_2 = paddle.to_tensor(
                [
                    [6000.0, 6001.0, 6002.0, 6003.0, 6004.0],
                    [6005.0, 6006.0, 6007.0, 6008.0, 6009.0],
                    [6010.0, 6011.0, 6012.0, 6013.0, 6014.0],
                ],
                dtype="float32",
            ).cuda()
            self._assert_equal(recv_next[0], expected_1)
            self._assert_equal(recv_next[1], expected_2)
            self.assertIsNone(recv_prev)

        dist.barrier()

    def test_forward_single_tensor_non_blocking_content(self):
        """sync_recv=False queues irecv tasks and waits them before returning;
        the completed buffer on stage 1 must still equal the sent bytes."""
        pp_rank = fleet.get_hybrid_communicate_group().get_stage_id()
        meta = _single_meta()

        if pp_rank == 0:
            payload = _linear_payload(7000, [2, 4])
            recv_prev, recv_next = _p2p_helper(
                tensor_send_next=payload,
                tensor_send_prev=None,
                recv_prev=False,
                recv_next=False,
                sync_recv=False,
                send_recv_meta=meta,
            )
            self.assertIsNone(recv_prev)
            self.assertIsNone(recv_next)
        else:
            recv_prev, recv_next = _p2p_helper(
                tensor_send_next=None,
                tensor_send_prev=None,
                recv_prev=True,
                recv_next=False,
                sync_recv=False,
                send_recv_meta=meta,
            )
            expected = paddle.to_tensor(
                [
                    [7000.0, 7001.0, 7002.0, 7003.0],
                    [7004.0, 7005.0, 7006.0, 7007.0],
                ],
                dtype="float32",
            ).cuda()
            self._assert_equal(recv_prev, expected)
            self.assertIsNone(recv_next)

        dist.barrier()


if __name__ == "__main__":
    unittest.main()
