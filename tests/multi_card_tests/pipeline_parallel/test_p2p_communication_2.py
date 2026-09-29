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

"""Real multi-card (PP=2) behavior tests for pipeline-parallel p2p (part 2).

These tests are launched by ``paddle.distributed.launch`` with 2 GPUs; each
rank is a distinct pipeline stage and exchanges rank-distinguishable payloads
through the REAL production p2p path
(``paddlefleet.pipeline_parallel.pp_utils.p2p_communication``).

This file focuses on code paths NOT exercised by the sibling
``test_p2p_communication.py``:

* the combined ``send_forward_recv_forward`` 1F1B op, and
* the multi-tensor (tuple) forward + backward round trip, which drives the
  ``tensor_type == 1`` branch of ``send_meta``/``recv_meta`` and the tuple
  branch of ``_p2p_helper``.

Every expected received tensor is derived by hand on the receiving side (never
read back from the sender object or produced by the function under test), and
each element is distinct, so a reversed direction, a wrong peer, a swapped
tuple member, or a dropped message all produce a value mismatch.
"""

import paddle
from paddle.distributed import fleet

from paddlefleet.pipeline_parallel.pp_utils.p2p_communication import (
    P2pHelper,
    initialize_p2p_groups,
)
from tests.multi_card_tests.tensor_parallel.test_utilities import Utils

PP_DEGREE = 2


def _wire_p2p_module():
    """Point the p2p module's HCG at the live PP=2 hybrid group.

    ``Utils.initialize_model_parallel`` brings up fleet and the hybrid topology
    but does not populate ``p2p_communication._hcg``; the production p2p path
    reads that module global, so it must be initialized explicitly. mp_degree
    is 1 here, so partial send/recv stays off and each tensor travels as one
    whole message whose content is fully comparable on the receiver.
    """
    hcg = fleet.get_hybrid_communicate_group()
    initialize_p2p_groups(hcg, enable_partial_send_recv=False)
    return hcg


def test_send_forward_recv_forward_delivers_activation():
    """Combined ``send_forward_recv_forward`` op moves an activation forward.

    Stage 0 (first stage) has an activation to emit and nothing to receive from
    a previous stage (``recv_prev=False``); stage 1 (last stage) emits nothing
    and receives from its previous stage (``recv_prev=True``). The metadata is
    transmitted for real via ``send_meta``/``recv_meta`` (the receiver does not
    hard-code the shape) and the activation is delivered over a real p2p
    send/recv. The expected tensor is hand-built on stage 1, so a wrong peer or
    a dropped payload yields a mismatch.
    """
    hcg = fleet.get_hybrid_communicate_group()
    stage_id = hcg.get_stage_id()
    p2p = P2pHelper(use_cache=True, dynamic_shape=False)
    shape = [2, 3]

    # Per-element distinct payload: a transpose/reshape error is observable.
    activation = (
        paddle.arange(6, dtype="float32").reshape(shape) + 100.0
    ).cuda()

    if stage_id == 0:
        # First stage: send forward, receive nothing from a previous stage.
        received = p2p.send_forward_recv_forward(
            activation, recv_prev=False, batch_p2p_comm=False
        )
        assert received is None, received
    else:
        # Last stage: send nothing forward, receive the activation.
        received = p2p.send_forward_recv_forward(
            None, recv_prev=True, batch_p2p_comm=False
        )
        assert received is not None
        assert list(received.shape) == shape, received.shape
        assert received.dtype == paddle.float32, received.dtype
        # Hand-derived: [[100, 101, 102], [103, 104, 105]] -- exactly what
        # stage 0 constructed, reconstructed independently here.
        expected = (
            paddle.arange(6, dtype="float32").reshape(shape) + 100.0
        ).cuda()
        assert paddle.equal_all(received, expected)


def test_tuple_forward_and_backward_roundtrip():
    """Multi-tensor (tuple) activation forward + gradient backward round trip.

    Stage 0 sends a two-tensor tuple with distinct shapes and content forward;
    stage 1 receives the tuple, checks each member element-for-element, then
    sends back a distinct two-tensor gradient tuple; stage 0 receives it and
    checks each member. This drives the ``tensor_type == 1`` encode/decode path
    of ``send_meta``/``recv_meta`` and the tuple branch of ``_p2p_helper``.

    The two members have different shapes ([2, 2] vs [3]) and disjoint value
    ranges, so a swapped tuple order, a wrong peer, a reversed direction, or a
    dropped member all produce a mismatch. Every member requires grad
    (``stop_gradient = False``) so the send-message tuple and the transmitted
    meta agree on member count. Expected tensors are hand-built per stage.
    """
    hcg = fleet.get_hybrid_communicate_group()
    stage_id = hcg.get_stage_id()
    p2p = P2pHelper(use_cache=True, dynamic_shape=False)

    def _make(base, numel, out_shape):
        t = paddle.arange(numel, dtype="float32").reshape(out_shape) + base
        t = t.cuda()
        # Tuple members are filtered out of set_send_message when
        # stop_gradient is True; keep grad on so meta member count matches.
        t.stop_gradient = False
        return t

    # Forward payload (stage 0 -> stage 1).
    fwd_a = _make(200.0, 4, [2, 2])  # [[200, 201], [202, 203]]
    fwd_b = _make(500.0, 3, [3])  # [500, 501, 502]
    # Backward payload (stage 1 -> stage 0).
    bwd_a = _make(700.0, 4, [2, 2])  # [[700, 701], [702, 703]]
    bwd_b = _make(800.0, 3, [3])  # [800, 801, 802]

    if stage_id == 0:
        # First stage: emit the tuple forward, then receive the grad tuple.
        p2p.send_forward(
            (fwd_a, fwd_b), pp_last_stage=False, batch_p2p_comm=False
        )
        grad = p2p.recv_backward(pp_last_stage=False, batch_p2p_comm=False)
        assert isinstance(grad, tuple), type(grad)
        assert len(grad) == 2, len(grad)
        assert list(grad[0].shape) == [2, 2], grad[0].shape
        assert list(grad[1].shape) == [3], grad[1].shape
        # Hand-derived: exactly the tuple stage 1 constructed as (bwd_a, bwd_b).
        assert paddle.equal_all(grad[0], bwd_a)
        assert paddle.equal_all(grad[1], bwd_b)
    else:
        # Last stage: receive the tuple, verify each member, then send grads.
        acts = p2p.recv_forward(pp_first_stage=False, batch_p2p_comm=False)
        assert isinstance(acts, tuple), type(acts)
        assert len(acts) == 2, len(acts)
        assert list(acts[0].shape) == [2, 2], acts[0].shape
        assert list(acts[1].shape) == [3], acts[1].shape
        # Hand-derived: exactly the tuple stage 0 constructed as (fwd_a, fwd_b).
        assert paddle.equal_all(acts[0], fwd_a)
        assert paddle.equal_all(acts[1], fwd_b)
        p2p.send_backward(
            (bwd_a, bwd_b), pp_first_stage=False, batch_p2p_comm=False
        )


if __name__ == "__main__":
    Utils.initialize_model_parallel(
        tensor_parallel_size=1, pipeline_parallel_size=PP_DEGREE
    )
    _wire_p2p_module()
    test_send_forward_recv_forward_delivers_activation()
    test_tuple_forward_and_backward_roundtrip()
