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

# Four-directions p2p requires this env var: paddle's topology reads the
# ``_use_four_directions`` flag at HybridCommunicateGroup creation time (during
# fleet init below), so it must be set at import time -- before initialize_fleet
# runs -- exactly as a real four-directions p2p launcher would set it.
os.environ["PADDLE_USE_FOUR_DIRECTIONS_P2P"] = "True"

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet

from paddlefleet.pipeline_parallel.pp_utils.four_directions_p2p_communication import (
    SendRecvMeta,
    _is_valid_send_recv_partial,
    _p2p_helper,
    initialize_p2p_groups,
    recv_partial,
    send_partial,
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


def _setup():
    _init_pp()
    hcg = fleet.get_hybrid_communicate_group()
    initialize_p2p_groups(
        hcg, enable_partial_send_recv=True, enable_timer=False
    )
    return hcg


def _tagged_payload(tag, shape):
    """Deterministic, sender-tagged GPU tensor.

    Element at flat index ``k`` is ``tag + k``. ``tag`` identifies the logical
    sender so the receiver can assert it obtained the *other* rank's data. A
    wrong peer, a reversed direction or a corrupted buffer changes these exact
    values. Expected tensors below are written by hand with the same closed
    form and are never produced by the functions under test.
    """
    numel = 1
    for dim in shape:
        numel *= dim
    values = [float(tag + k) for k in range(numel)]
    return paddle.to_tensor(values, dtype="float32").reshape(shape).cuda()


def test_send_partial_recv_partial_forward():
    """Real forward-direction P2P: rank r sends to its next peer, receives
    from its prev peer. With PP=2 the two stages form a ring, so both ranks
    exchange in one shot. Each rank must receive the *other* rank's tagged
    payload byte-for-byte."""
    hcg = fleet.get_hybrid_communicate_group()
    pp_rank = hcg.get_stage_id()
    peer = 1 - pp_rank

    shape = [2, 4]
    send_tensor = _tagged_payload(pp_rank * 1000, shape)

    # Both ranks send and receive in the same step (a 2-stage ring), so both
    # the send (dst=1 -> _get_p2p_next_rank) and the recv (src=0 ->
    # _get_p2p_prev_rank) must be posted non-blocking *before* either is waited.
    # ``use_calc_stream=False`` makes send_partial issue an isend and
    # recv_partial an irecv; the NCCL kernels are only driven forward on
    # ``.wait()``. If the recv were blocking instead, every rank would enter it
    # before any rank reached ``send_task.wait()``, so no send would ever launch
    # and all recvs would block forever -- a real deadlock. Posting both async
    # ops first lets NCCL match the send/recv pair on every rank.
    send_task = send_partial(
        send_tensor, dst=1, nranks=1, use_calc_stream=False
    )
    recv_tensor = paddle.empty(shape, dtype="float32").cuda()
    recv_task = recv_partial(
        recv_tensor, src=0, nranks=1, use_calc_stream=False
    )
    if recv_task is not None:
        recv_task.wait()
    if send_task is not None:
        send_task.wait()

    # Hand-derived expectation: prev peer of rank r in a 2-stage ring is the
    # other rank, so we must have received the peer's tag (peer*1000 + k).
    expected = _tagged_payload(peer * 1000, shape)
    assert paddle.equal_all(recv_tensor, expected), (
        f"rank {pp_rank} forward recv mismatch: got {recv_tensor.numpy()!r}, "
        f"expected {expected.numpy()!r}"
    )
    dist.barrier()


def test_send_partial_recv_partial_backward():
    """Real backward-direction P2P: rank r sends to its prev peer, receives
    from its next peer (dst=0 / src=1). Uses a distinct shape and tag base so a
    swapped forward/backward direction or wrong recv slot is caught."""
    hcg = fleet.get_hybrid_communicate_group()
    pp_rank = hcg.get_stage_id()
    peer = 1 - pp_rank

    shape = [3, 2]
    send_tensor = _tagged_payload(pp_rank * 1000 + 7, shape)

    # Same simultaneous-exchange constraint as the forward case: post the isend
    # (dst=0 -> _get_p2p_prev_rank) and the irecv (src=1 -> _get_p2p_next_rank)
    # before waiting either, so both NCCL kernels are in flight and can match.
    send_task = send_partial(
        send_tensor, dst=0, nranks=1, use_calc_stream=False
    )
    recv_tensor = paddle.empty(shape, dtype="float32").cuda()
    recv_task = recv_partial(
        recv_tensor, src=1, nranks=1, use_calc_stream=False
    )
    if recv_task is not None:
        recv_task.wait()
    if send_task is not None:
        send_task.wait()

    expected = _tagged_payload(peer * 1000 + 7, shape)
    assert paddle.equal_all(recv_tensor, expected), (
        f"rank {pp_rank} backward recv mismatch: got {recv_tensor.numpy()!r}, "
        f"expected {expected.numpy()!r}"
    )
    dist.barrier()


def test_send_recv_meta_roundtrip():
    """Real metadata handshake over the pipe group. Two ordered phases keep it
    deadlock-free: rank 0 sends meta -> rank 1 recvs; then rank 1 sends ->
    rank 0 recvs. The receiver must reconstruct the exact shape, dtype code and
    stop_gradient flag the sender described. Shapes and the stop_gradient flags
    are hand-chosen and distinct per phase, so a truncated or misrouted
    handshake changes them."""
    hcg = fleet.get_hybrid_communicate_group()
    pp_rank = hcg.get_stage_id()
    pp_group = hcg.get_pipe_parallel_group()

    # Phase A: rank 0 describes a rank-3 fp32 trainable tensor.
    tensor_a = paddle.randn([3, 5, 7], dtype="float32").cuda()
    tensor_a.stop_gradient = False
    if pp_rank == 0:
        SendRecvMeta().send_meta(tensor_a, pp_group)
    else:
        meta = SendRecvMeta()
        meta.recv_meta(pp_group)
        assert meta.recv_shape_message == [3, 5, 7], meta.recv_shape_message
        assert meta.recv_dtype_message == FP32, meta.recv_dtype_message
        assert meta.recv_stop_gradient is False, meta.recv_stop_gradient
    dist.barrier()

    # Phase B: rank 1 describes a rank-2 fp16 frozen tensor (distinct on every
    # axis so a stale Phase-A result would fail here).
    tensor_b = paddle.randn([2, 6], dtype="float16").cuda()
    tensor_b.stop_gradient = True
    fp16 = paddle_2_number(paddle.float16)
    if pp_rank == 1:
        SendRecvMeta().send_meta(tensor_b, pp_group)
    else:
        meta = SendRecvMeta()
        meta.recv_meta(pp_group)
        assert meta.recv_shape_message == [2, 6], meta.recv_shape_message
        assert meta.recv_dtype_message == fp16, meta.recv_dtype_message
        assert meta.recv_stop_gradient is True, meta.recv_stop_gradient
    dist.barrier()


def test_p2p_helper_forward_activation_backward_grad():
    """Exercise all four directions of ``_p2p_helper`` in one coordinated step,
    mirroring a real pipeline micro-step:

      * stage 0 sends its activation to the next stage and receives the
        gradient coming back from the next stage (send_next + recv_next);
      * stage 1 receives that activation from the prev stage and sends its
        gradient back to the prev stage (recv_prev + send_prev).

    Both the delivered activation and the returned gradient are compared to
    hand-written constants, so a wrong direction, peer or recv slot is caught."""
    hcg = fleet.get_hybrid_communicate_group()
    pp_rank = hcg.get_stage_id()

    shape = [2, 4]
    activation_tag = 10  # stage 0 -> stage 1
    grad_tag = 500  # stage 1 -> stage 0

    meta = SendRecvMeta()
    meta.send_shape_message = shape
    meta.send_dtype_message = FP32
    meta.recv_shape_message = shape
    meta.recv_dtype_message = FP32
    meta.recv_stop_gradient = False

    if pp_rank == 0:
        activation = _tagged_payload(activation_tag, shape)
        recv_prev, recv_next = _p2p_helper(
            tensor_send_next=activation,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=True,
            sync_recv=True,
            send_recv_meta=meta,
        )
        assert recv_prev is None
        expected_grad = _tagged_payload(grad_tag, shape)
        assert paddle.equal_all(recv_next, expected_grad), (
            f"stage 0 recv_next grad mismatch: got {recv_next.numpy()!r}, "
            f"expected {expected_grad.numpy()!r}"
        )
    else:
        grad = _tagged_payload(grad_tag, shape)
        recv_prev, recv_next = _p2p_helper(
            tensor_send_next=None,
            tensor_send_prev=grad,
            recv_prev=True,
            recv_next=False,
            sync_recv=True,
            send_recv_meta=meta,
        )
        assert recv_next is None
        expected_activation = _tagged_payload(activation_tag, shape)
        assert paddle.equal_all(recv_prev, expected_activation), (
            f"stage 1 recv_prev activation mismatch: got "
            f"{recv_prev.numpy()!r}, expected {expected_activation.numpy()!r}"
        )
    dist.barrier()


def test_is_valid_send_recv_partial_zero_element_guard():
    """Lock the real zero-element guard contract of the partial-send predicate:
    a zero-element tensor must raise AssertionError rather than silently
    proceeding. No collective is issued; this is a per-rank local contract."""
    zero_tensor = paddle.empty([0], dtype="float32").cuda()
    raised = False
    try:
        _is_valid_send_recv_partial(zero_tensor, 2)
    except AssertionError:
        raised = True
    assert raised, "expected AssertionError for a zero-element partial tensor"


if __name__ == "__main__":
    _setup()
    test_send_partial_recv_partial_forward()
    test_send_partial_recv_partial_backward()
    test_send_recv_meta_roundtrip()
    test_p2p_helper_forward_activation_backward_grad()
    test_is_valid_send_recv_partial_zero_element_guard()
