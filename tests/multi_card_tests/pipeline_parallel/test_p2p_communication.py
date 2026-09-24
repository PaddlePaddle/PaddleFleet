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

These tests run under ``paddle.distributed.launch`` with 2 GPUs. Each rank is
a distinct pipeline stage and exchanges rank-distinguishable payloads through
the REAL production p2p path
(``paddlefleet.pipeline_parallel.pp_utils.p2p_communication``): real
``send_meta``/``recv_meta`` and real forward/backward ``send``/``recv`` between
pipeline peers. Expected received tensors are derived by hand so a reversed
direction, a wrong peer, or a dropped message produces a mismatch.
"""

import paddle
from paddle.distributed import fleet

from paddlefleet.pipeline_parallel.pp_utils.p2p_communication import (
    P2pHelper,
    SendRecvMeta,
    batch_send_recv_on_calc_stream,
    initialize_p2p_groups,
)
from paddlefleet.pipeline_parallel.pp_utils.utils import paddle_2_number

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
    # mp_degree == 1 here, so keep partial send/recv off: the whole tensor
    # travels as one message and the received content is fully comparable.
    initialize_p2p_groups(hcg, enable_partial_send_recv=False)
    return hcg


def test_send_recv_meta_transmits_shape_and_dtype():
    """SendRecvMeta.send_meta -> recv_meta carries shape+dtype across ranks.

    Stage 0 encodes the metadata of a [5, 7] float32 tensor and sends it to
    its pipeline-next peer; stage 1 decodes it from its pipeline-prev peer.
    The transmitted shape/dtype IS the contract of this API, and the expected
    values are hand-written (not read back from the sender object), so a wrong
    peer or a corrupted encode/decode changes the decoded result.
    """
    hcg = fleet.get_hybrid_communicate_group()
    pp_group = hcg.get_pipe_parallel_group()
    stage_id = hcg.get_stage_id()

    if stage_id == 0:
        meta = SendRecvMeta()
        tensor = paddle.zeros([5, 7], dtype="float32").cuda()
        meta.set_send_message(tensor)
        meta.send_meta(tensor, pp_group)
    else:
        meta = SendRecvMeta()
        meta.recv_meta(pp_group)
        # Hand-derived: stage 0 sent a single tensor (tensor_type 0) of shape
        # [5, 7] with float32 dtype -> recv_* fields hold the scalar (not
        # tuple) shape list and the float32 dtype number.
        assert meta.recv_shape_message == [5, 7], meta.recv_shape_message
        assert meta.recv_dtype_message == paddle_2_number(paddle.float32), (
            meta.recv_dtype_message
        )
        assert meta.recv_stop_gradient is True, meta.recv_stop_gradient
        assert meta.recv_key_message is None, meta.recv_key_message


def test_send_forward_and_recv_backward_roundtrip():
    """Full forward activation + backward gradient p2p between PP stages.

    Stage 0 (first stage) sends a distinguishable activation forward to stage 1
    and later receives the gradient stage 1 sends back. Stage 1 (last stage)
    receives the activation, checks it element-for-element, then sends its own
    distinguishable gradient back. Forward payload is arange+100, backward
    payload is arange+900; both expected tensors are derived by hand on the
    receiving side. A reversed direction, a wrong peer, or a dropped message
    yields a value mismatch (an empty recv buffer would not equal either).
    """
    hcg = fleet.get_hybrid_communicate_group()
    stage_id = hcg.get_stage_id()
    p2p = P2pHelper(use_cache=True, dynamic_shape=False)
    shape = [2, 3]

    fwd_payload = (
        paddle.arange(6, dtype="float32").reshape(shape) + 100.0
    ).cuda()
    bwd_payload = (
        paddle.arange(6, dtype="float32").reshape(shape) + 900.0
    ).cuda()

    if stage_id == 0:
        # First stage: emit the activation forward, then receive the gradient.
        p2p.send_forward(fwd_payload, pp_last_stage=False, batch_p2p_comm=False)
        grad = p2p.recv_backward(pp_last_stage=False, batch_p2p_comm=False)
        assert grad is not None
        assert list(grad.shape) == shape, grad.shape
        # Hand-derived: exactly what stage 1 constructed as bwd_payload.
        assert paddle.equal_all(grad, bwd_payload)
    else:
        # Last stage: receive the activation, verify it, then send gradient.
        activation = p2p.recv_forward(
            pp_first_stage=False, batch_p2p_comm=False
        )
        assert activation is not None
        assert list(activation.shape) == shape, activation.shape
        # Hand-derived: exactly what stage 0 constructed as fwd_payload.
        assert paddle.equal_all(activation, fwd_payload)
        p2p.send_backward(
            bwd_payload, pp_first_stage=False, batch_p2p_comm=False
        )


def test_batch_send_recv_on_calc_stream_empty_raises():
    """Lock a real defect: batch_send_recv_on_calc_stream crashes on [].

    ``batch_send_recv_on_calc_stream`` reads ``p2p_op_list[0].group`` before
    checking for emptiness, so an empty op list raises IndexError instead of
    being a no-op. This is a rank-local behavior (no collective is reached),
    so every stage asserts it independently. Production is NOT edited; if the
    function is later hardened to accept an empty list this assertion fails and
    surfaces the change. See src/paddlefleet/pipeline_parallel/pp_utils/
    p2p_communication.py:357.
    """
    raised = False
    try:
        batch_send_recv_on_calc_stream([])
    except IndexError:
        raised = True
    assert raised, (
        "expected batch_send_recv_on_calc_stream([]) to raise IndexError due "
        "to the unguarded p2p_op_list[0] access"
    )


if __name__ == "__main__":
    _init_pipeline_parallel()
    test_send_recv_meta_transmits_shape_and_dtype()
    test_send_forward_and_recv_backward_roundtrip()
    test_batch_send_recv_on_calc_stream_empty_raises()
