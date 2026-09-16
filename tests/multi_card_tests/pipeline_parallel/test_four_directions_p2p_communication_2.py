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

import unittest

import paddle
from paddle.distributed import fleet

from paddlefleet.pipeline_parallel.pp_utils.four_directions_p2p_communication import (
    P2pHelper,
    initialize_p2p_groups,
)

# PP=2 topology. Rank 0 is pipeline stage 0 (first stage, sends activations
# forward), rank 1 is pipeline stage 1 (last stage, sends gradients backward).
# Every payload below is built from paddle.arange plus a per-direction offset
# so the exact element values are unique per position; the receiver compares
# against an independently written literal, never against anything produced by
# the module under test. A reversed direction, a wrong peer, or a dropped
# element would change the exact received values and fail the comparison.
_PP_DEGREE = 2


def _forward_activation():
    # Stage-0 -> stage-1 activation. Values [[100..102],[103..105]].
    return (paddle.arange(6, dtype="float32").reshape([2, 3]) + 100.0).cuda()


def _expected_forward_activation():
    return paddle.to_tensor(
        [[100.0, 101.0, 102.0], [103.0, 104.0, 105.0]], dtype="float32"
    ).cuda()


def _backward_grad():
    # Stage-1 -> stage-0 gradient. Values [[200..202],[203..205]].
    return (paddle.arange(6, dtype="float32").reshape([2, 3]) + 200.0).cuda()


def _expected_backward_grad():
    return paddle.to_tensor(
        [[200.0, 201.0, 202.0], [203.0, 204.0, 205.0]], dtype="float32"
    ).cuda()


class TestFourDirectionsP2P(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": _PP_DEGREE,
        }
        fleet.init(is_collective=True, strategy=strategy)
        cls.hcg = fleet.get_hybrid_communicate_group()
        # Wire the real p2p groups into the module under test so that
        # P2pHelper sends over the actual NCCL peer groups of this launch.
        initialize_p2p_groups(cls.hcg)
        cls.stage_id = cls.hcg.get_stage_id()

    def test_forward_delivers_exact_activation(self):
        # Stage 0 sends a distinguishable activation; stage 1 must receive the
        # exact tensor (meta + payload) over the real forward p2p group.
        helper = P2pHelper(use_cache=True)
        if self.stage_id == 0:
            helper.send_forward(_forward_activation(), pp_last_stage=False)
        else:
            received = helper.recv_forward(pp_first_stage=False)
            expected = _expected_forward_activation()
            self.assertEqual(list(received.shape), [2, 3])
            self.assertEqual(received.dtype, paddle.float32)
            self.assertTrue(paddle.equal_all(received, expected))
            # Meta really crossed the wire: shape/dtype recorded on recv side.
            self.assertEqual(
                list(helper._send_recv_meta.recv_shape_message), [2, 3]
            )

    def test_full_forward_backward_cycle(self):
        # One helper drives a full 1F1B leg: forward activation stage0->stage1,
        # then gradient stage1->stage0. The backward recv buffer on stage 0 is
        # sized from the send meta established during the forward pass, so this
        # also exercises meta reuse across directions.
        helper = P2pHelper(use_cache=True)
        if self.stage_id == 0:
            helper.send_forward(_forward_activation(), pp_last_stage=False)
            received_grad = helper.recv_backward(pp_last_stage=False)
            expected_grad = _expected_backward_grad()
            self.assertEqual(list(received_grad.shape), [2, 3])
            self.assertEqual(received_grad.dtype, paddle.float32)
            self.assertTrue(paddle.equal_all(received_grad, expected_grad))
        else:
            received_act = helper.recv_forward(pp_first_stage=False)
            self.assertTrue(
                paddle.equal_all(received_act, _expected_forward_activation())
            )
            helper.send_backward(_backward_grad(), pp_first_stage=False)

    def test_interleaved_send_recv_step(self):
        # Four-direction interleaved op: after a warmup forward establishes
        # meta on both stages, stage 0 sends the next activation while
        # receiving a gradient, and stage 1 sends a gradient while receiving
        # the next activation. Both received tensors are checked against
        # independent literals so a swapped send/recv direction is rejected.
        helper = P2pHelper(use_cache=True)
        step_act = (
            paddle.arange(6, dtype="float32").reshape([2, 3]) + 300.0
        ).cuda()
        expected_step_act = paddle.to_tensor(
            [[300.0, 301.0, 302.0], [303.0, 304.0, 305.0]], dtype="float32"
        ).cuda()
        step_grad = (
            paddle.arange(6, dtype="float32").reshape([2, 3]) + 400.0
        ).cuda()
        expected_step_grad = paddle.to_tensor(
            [[400.0, 401.0, 402.0], [403.0, 404.0, 405.0]], dtype="float32"
        ).cuda()

        if self.stage_id == 0:
            # Warmup forward to publish send meta to the next stage.
            helper.send_forward(_forward_activation(), pp_last_stage=False)
            received_grad = helper.send_forward_recv_backward(
                step_act, pp_last_stage=False
            )
            self.assertEqual(list(received_grad.shape), [2, 3])
            self.assertTrue(paddle.equal_all(received_grad, expected_step_grad))
        else:
            # Warmup recv to learn recv meta from the previous stage.
            helper.recv_forward(pp_first_stage=False)
            received_act = helper.send_backward_recv_forward(
                step_grad, pp_first_stage=False
            )
            self.assertEqual(list(received_act.shape), [2, 3])
            self.assertTrue(paddle.equal_all(received_act, expected_step_act))

    def test_tuple_forward_delivers_each_element(self):
        # Forward a tuple of two distinct tensors so the tuple meta path and
        # per-element buffer allocation are exercised. Each element is checked
        # against its own literal; a wrong concat/order or a dropped element
        # changes the exact per-element values.
        helper = P2pHelper(use_cache=True)
        first = (
            paddle.arange(4, dtype="float32").reshape([2, 2]) + 10.0
        ).cuda()
        second = (paddle.arange(3, dtype="float32") + 50.0).cuda()
        first.stop_gradient = False
        second.stop_gradient = False
        if self.stage_id == 0:
            helper.send_forward((first, second), pp_last_stage=False)
        else:
            received = helper.recv_forward(pp_first_stage=False)
            self.assertIsInstance(received, tuple)
            self.assertEqual(len(received), 2)
            self.assertTrue(
                paddle.equal_all(
                    received[0],
                    paddle.to_tensor(
                        [[10.0, 11.0], [12.0, 13.0]], dtype="float32"
                    ).cuda(),
                )
            )
            self.assertTrue(
                paddle.equal_all(
                    received[1],
                    paddle.to_tensor(
                        [50.0, 51.0, 52.0], dtype="float32"
                    ).cuda(),
                )
            )


if __name__ == "__main__":
    unittest.main()
