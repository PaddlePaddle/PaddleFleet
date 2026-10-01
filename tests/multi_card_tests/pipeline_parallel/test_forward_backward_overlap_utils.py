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

"""Real PP=4 behavior tests for the pipeline forward/backward-overlap utils.

Run through ``paddle.distributed.launch`` on 4 GPUs (PP=4). The centerpiece is
``test_four_stage_pipeline_forward_backward``: a genuine 4-stage pipeline where
stage i (rank i) owns a distinct weight ``Wi``, the forward activation is moved
stage->stage with a real ``paddle.distributed.send``/``recv`` and the backward
activation gradient is moved back stage<-stage the same way -- exactly how a
``ScheduleNode`` is consumed inside the pipeline scheduler. Every expected value
(delivered activation, delivered gradient, per-stage weight gradient) is derived
BY HAND from the fixed constants in ``_STAGE_WEIGHTS`` / ``_X`` / ``_G3`` below,
never from the functions under test.

The remaining ``test_*`` functions pin the *local* (per-rank, non-collective by
construction) numeric contracts of ``detach_and_requires_grad``, ``FakeClone``,
``clone_and_clear_dataptr``, ``ScheduleNode`` (both the plain and the recompute
path) and ``ScheduleChunk`` with distinguishable, rank-dependent inputs and
independent hand-derived expectations. ``FakeClone.forward`` returns
``empty_like`` by design (it avoids a DtoD copy), so only its shape/dtype and
its load-bearing identity backward are asserted; its uninitialised forward
content is deliberately not compared.
"""

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.fleet.recompute import custom_state_manager

from paddlefleet.pipeline_parallel.pp_utils.forward_backward_overlap_utils import (
    FakeClone,
    ScheduleChunk,
    ScheduleNode,
    clone_and_clear_dataptr,
    detach_and_requires_grad,
)

PP_DEGREE = 4

# Fixed pipeline constants (batch=1, hidden=2). Weights are chosen so every
# stage transforms the activation differently, making a mis-routed stage,
# a reversed matmul or a dropped transpose observable in the exact numbers.
_X = np.array([[1.0, 2.0]], dtype="float32")
_STAGE_WEIGHTS = {
    0: np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32"),
    1: np.array([[1.0, 0.0], [1.0, 1.0]], dtype="float32"),
    2: np.array([[2.0, 0.0], [0.0, 2.0]], dtype="float32"),
    3: np.array([[1.0, 1.0], [0.0, 1.0]], dtype="float32"),
}
# Non-uniform upstream gradient on the final stage output.
_G3 = np.array([[1.0, 2.0]], dtype="float32")


def _init_pp():
    """Initialise a real PP=4 fleet process group (launcher entry point)."""
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": PP_DEGREE,
    }
    fleet.init(is_collective=True, strategy=strategy)


def _param(arr):
    """Create a trainable parameter holding exactly ``arr`` (float32)."""
    return paddle.create_parameter(
        shape=list(arr.shape),
        dtype="float32",
        default_initializer=paddle.nn.initializer.Assign(arr),
    )


def _matmul_fwd(weight):
    def fwd_func(inputs, is_first_fwd=False):
        return paddle.matmul(inputs, weight)

    return fwd_func


def test_detach_and_requires_grad_preserves_value_and_breaks_graph():
    # Rank-dependent, distinguishable content so a swapped tensor is visible.
    rank = float(dist.get_rank())
    base = (paddle.arange(6, dtype="float32").reshape([2, 3])).cuda() + rank
    a = base.clone()
    a.stop_gradient = False
    b = a * 2.0

    d = detach_and_requires_grad(b)

    # Value is preserved exactly and stop_gradient is copied from the source
    # (b required grad -> d does too).
    np.testing.assert_array_equal(d.numpy(), base.numpy() * 2.0)
    assert d.stop_gradient is False

    # The detach severs the graph: a function of d must not push grad into a,
    # while d itself is a fresh leaf that accumulates a hand-known gradient.
    (d * 3.0).sum().backward()
    assert a.grad is None
    np.testing.assert_array_equal(
        d.grad.numpy(), np.full([2, 3], 3.0, dtype="float32")
    )


def test_detach_and_requires_grad_stop_gradient_and_containers():
    rank = float(dist.get_rank())
    # A stop_gradient=True input must stay True (the helper copies the flag
    # rather than forcing requires-grad, despite its name).
    frozen = (paddle.arange(3, dtype="float32") + rank).cuda()
    frozen.stop_gradient = True
    out = detach_and_requires_grad(frozen)
    assert out.stop_gradient is True
    np.testing.assert_array_equal(out.numpy(), frozen.numpy())

    # Tuple with nested list and a non-tensor sentinel: structure, per-element
    # stop_gradient and the pass-through of the non-tensor are all preserved.
    t1 = paddle.to_tensor([1.0, 2.0]).cuda()
    t1.stop_gradient = False
    t2 = paddle.to_tensor([3.0, 4.0]).cuda()
    t2.stop_gradient = True
    nested = [paddle.to_tensor([5.0]).cuda()]
    tup = detach_and_requires_grad((t1, t2, nested, 7))
    assert isinstance(tup, tuple) and len(tup) == 4
    assert tup[0].stop_gradient is False
    assert tup[1].stop_gradient is True
    assert isinstance(tup[2], list)
    np.testing.assert_array_equal(tup[0].numpy(), [1.0, 2.0])
    np.testing.assert_array_equal(tup[1].numpy(), [3.0, 4.0])
    np.testing.assert_array_equal(tup[2][0].numpy(), [5.0])
    assert tup[3] == 7

    # Dict keeps keys and passes a None value straight through.
    td = paddle.to_tensor([1.0, 2.0]).cuda()
    td.stop_gradient = False
    dct = detach_and_requires_grad({"a": td, "b": None})
    assert isinstance(dct, dict) and set(dct) == {"a", "b"}
    assert dct["a"].stop_gradient is False
    np.testing.assert_array_equal(dct["a"].numpy(), [1.0, 2.0])
    assert dct["b"] is None


def test_fake_clone_shape_and_identity_backward():
    rank = float(dist.get_rank())
    x = (paddle.arange(6, dtype="float32").reshape([2, 3])).cuda() + rank
    x.stop_gradient = False

    out = FakeClone.apply(x)

    # forward is empty_like: same shape/dtype, fresh storage. Content is
    # uninitialised by design, so it is intentionally NOT compared; the
    # load-bearing contract is the identity backward below.
    assert list(out.shape) == [2, 3]
    assert out.dtype == x.dtype
    assert out is not x
    assert out.stop_gradient is False

    upstream = (
        paddle.arange(6, dtype="float32").reshape([2, 3]) * 0.5 + 1.0
    ).cuda()
    paddle.autograd.backward([out], [upstream])
    np.testing.assert_array_equal(x.grad.numpy(), upstream.numpy())


def test_clone_and_clear_dataptr_filters_and_passes_grad():
    rank = float(dist.get_rank())
    t1 = (paddle.arange(4, dtype="float32") + rank).cuda()
    t2 = (paddle.arange(4, dtype="float32") + 10.0).reshape([2, 2]).cuda()

    # None and the plain int are dropped; only the two tensors survive, each
    # keeping shape/dtype (content is empty_like, so not compared). A broken
    # filter that kept None would yield length 4 instead of 2.
    ret = clone_and_clear_dataptr([t1, None, t2, 5])
    assert isinstance(ret, list) and len(ret) == 2
    assert list(ret[0].shape) == [4]
    assert list(ret[1].shape) == [2, 2]
    assert ret[0].dtype == t1.dtype

    # Dict variant drops the None value and keeps the remaining key.
    dct = clone_and_clear_dataptr({"x": t1, "y": None})
    assert isinstance(dct, dict) and set(dct) == {"x"}
    assert list(dct["x"].shape) == [4]

    # Single-tensor wrapper retains the gradient path even though the forward
    # data is dropped: grad must flow straight through unchanged.
    x = (paddle.arange(6, dtype="float32").reshape([2, 3])).cuda() + rank
    x.stop_gradient = False
    wrapped = clone_and_clear_dataptr(x)
    assert list(wrapped.shape) == [2, 3]
    upstream = (paddle.ones([2, 3]) * 2.0).cuda()
    paddle.autograd.backward([wrapped], [upstream])
    np.testing.assert_array_equal(x.grad.numpy(), upstream.numpy())


def _local_linear_inputs():
    rank = float(dist.get_rank())
    w_np = np.array([[1.0, 2.0], [0.0, 1.0], [3.0, -1.0]], dtype="float32")
    w = paddle.to_tensor(w_np).cuda()
    w.stop_gradient = True
    x_np = np.arange(6, dtype="float32").reshape(2, 3) + rank
    x = paddle.to_tensor(x_np).cuda()
    x.stop_gradient = False
    return w, w_np, x, x_np


def test_schedule_node_forward_backward():
    w, w_np, x, x_np = _local_linear_inputs()
    node = ScheduleNode(_matmul_fwd(w), name="lin")

    out = node.forward(x)
    np.testing.assert_allclose(out.numpy(), x_np @ w_np, rtol=1e-5, atol=1e-5)

    g_np = np.arange(4, dtype="float32").reshape(2, 2) + 1.0
    grad = node.backward(paddle.to_tensor(g_np).cuda())

    # dL/dx = g @ W^T; the weight is frozen so exactly one input grad returns.
    assert isinstance(grad, tuple) and len(grad) == 1
    np.testing.assert_allclose(
        grad[0].numpy(), g_np @ w_np.T, rtol=1e-5, atol=1e-5
    )
    # backward must clear the retained forward state.
    assert node.inputs is None and node.outputs is None


def test_schedule_node_recompute_forward_backward():
    w, w_np, x, x_np = _local_linear_inputs()
    node = ScheduleNode(_matmul_fwd(w), name="lin_rc")

    # ScheduleNode.first_forward snapshots a "custom" recompute state via
    # custom_state_manager.custom_get_state_func, which is None until fleet
    # registers it. Provide the documented no-op default (identical to the
    # fallback in tensor_parallel/random.py) so the real recompute path runs;
    # restore the prior hooks afterward. This configures a real global
    # collaborator, it does not mock ScheduleNode.
    _orig_get = custom_state_manager.custom_get_state_func
    _orig_set = custom_state_manager.custom_set_state_func
    if _orig_get is None:
        custom_state_manager.custom_get_state_func = lambda *a, **k: None
        custom_state_manager.custom_set_state_func = lambda *a, **k: None
    try:
        # first_forward runs under no_grad and snapshots RNG/AMP state; the
        # second forward recomputes under a real dygraph guard. The recomputed
        # numbers must still match the independent numpy reference.
        node.first_forward(x)
        out = node.forward(x)
        np.testing.assert_allclose(
            out.numpy(), x_np @ w_np, rtol=1e-5, atol=1e-5
        )

        g_np = np.arange(4, dtype="float32").reshape(2, 2) + 1.0
        grad = node.backward(paddle.to_tensor(g_np).cuda())
        assert len(grad) == 1
        np.testing.assert_allclose(
            grad[0].numpy(), g_np @ w_np.T, rtol=1e-5, atol=1e-5
        )
    finally:
        custom_state_manager.custom_get_state_func = _orig_get
        custom_state_manager.custom_set_state_func = _orig_set


def test_schedule_chunk_local_chain():
    rank = float(dist.get_rank())
    w1_np = np.array([[1.0, 2.0], [0.0, 1.0], [3.0, -1.0]], dtype="float32")
    w2_np = np.array(
        [[1.0, 0.0, 2.0, -1.0], [0.5, 1.0, -2.0, 3.0]], dtype="float32"
    )
    w1 = paddle.to_tensor(w1_np).cuda()
    w1.stop_gradient = True
    w2 = paddle.to_tensor(w2_np).cuda()
    w2.stop_gradient = True

    n1 = ScheduleNode(_matmul_fwd(w1), name="n1")
    n2 = ScheduleNode(_matmul_fwd(w2), name="n2")
    chunk = ScheduleChunk([n1, n2])
    assert len(chunk.nodes) == 2

    x_np = np.arange(6, dtype="float32").reshape(2, 3) + rank
    x = paddle.to_tensor(x_np).cuda()
    x.stop_gradient = False

    out = chunk.forward(x)
    np.testing.assert_allclose(
        out.numpy(), (x_np @ w1_np) @ w2_np, rtol=1e-5, atol=1e-5
    )

    g_np = np.arange(8, dtype="float32").reshape(2, 4) + 1.0
    grad = chunk.backward(paddle.to_tensor(g_np).cuda())
    # Manual chain rule: dx = ((g @ W2^T) @ W1^T).
    expected = (g_np @ w2_np.T) @ w1_np.T
    assert len(grad) == 1
    np.testing.assert_allclose(grad[0].numpy(), expected, rtol=1e-5, atol=1e-5)


def test_schedule_chunk_rejects_invalid_node():
    # _check_nodes_valid must reject a member that is not a ScheduleNode/Chunk.
    raised = False
    try:
        ScheduleChunk([object()])
    except AssertionError:
        raised = True
    assert raised, "ScheduleChunk must reject a non-ScheduleNode member"


def test_four_stage_pipeline_forward_backward():
    hcg = fleet.get_hybrid_communicate_group()
    pp_group = hcg.get_pipe_parallel_group()
    stage_id = hcg.get_stage_id()
    next_rank = hcg._get_p2p_next_rank()
    prev_rank = hcg._get_p2p_prev_rank()

    # Independent hand derivation from the fixed constants (see module doc):
    #   h0 = x  @ W0 = [[ 7, 10]]      h1 = h0 @ W1 = [[17, 10]]
    #   h2 = h1 @ W2 = [[34, 20]]      y  = h2 @ W3 = [[34, 54]]
    #   g3 = [[1, 2]]
    #   dh2 = g3 @ W3^T = [[3, 2]]     dh1 = dh2 @ W2^T = [[6, 4]]
    #   dh0 = dh1 @ W1^T = [[6, 10]]
    #   dW3 = h2^T @ g3  = [[34, 68], [20, 40]]
    #   dW2 = h1^T @ dh2 = [[51, 34], [30, 20]]
    #   dW1 = h0^T @ dh1 = [[42, 28], [60, 40]]
    #   dW0 = x^T  @ dh0 = [[ 6, 10], [12, 20]]
    act_in_ref = {
        1: np.array([[7.0, 10.0]], dtype="float32"),
        2: np.array([[17.0, 10.0]], dtype="float32"),
        3: np.array([[34.0, 20.0]], dtype="float32"),
    }
    grad_in_ref = {
        3: np.array([[3.0, 2.0]], dtype="float32"),
        2: np.array([[6.0, 4.0]], dtype="float32"),
        1: np.array([[6.0, 10.0]], dtype="float32"),
    }
    y_ref = np.array([[34.0, 54.0]], dtype="float32")
    dw_ref = {
        0: np.array([[6.0, 10.0], [12.0, 20.0]], dtype="float32"),
        1: np.array([[42.0, 28.0], [60.0, 40.0]], dtype="float32"),
        2: np.array([[51.0, 34.0], [30.0, 20.0]], dtype="float32"),
        3: np.array([[34.0, 68.0], [20.0, 40.0]], dtype="float32"),
    }

    weight = _param(_STAGE_WEIGHTS[stage_id])
    node = ScheduleNode(_matmul_fwd(weight), name=f"stage{stage_id}")

    # --- forward: activation flows 0 -> 1 -> 2 -> 3 via real send/recv ---
    if stage_id == 0:
        act = paddle.to_tensor(_X).cuda()
        act.stop_gradient = True
    else:
        act = paddle.zeros([1, 2], dtype="float32").cuda()
        dist.recv(act, src=prev_rank, group=pp_group)
        # The activation actually delivered across ranks must match the
        # independent hand derivation of the upstream stage's output.
        np.testing.assert_allclose(
            act.numpy(), act_in_ref[stage_id], rtol=1e-5, atol=1e-5
        )
        act.stop_gradient = False

    out = node.forward(act)
    if stage_id != PP_DEGREE - 1:
        dist.send(out, dst=next_rank, group=pp_group)
    else:
        # Final stage output is the fully composed pipeline result.
        np.testing.assert_allclose(out.numpy(), y_ref, rtol=1e-5, atol=1e-5)

    # --- backward: gradient flows 3 -> 2 -> 1 -> 0 via real send/recv ---
    if stage_id == PP_DEGREE - 1:
        upstream = paddle.to_tensor(_G3).cuda()
    else:
        upstream = paddle.zeros([1, 2], dtype="float32").cuda()
        dist.recv(upstream, src=next_rank, group=pp_group)
        # The gradient actually delivered across ranks must match the hand
        # derivation of the downstream stage's input gradient.
        np.testing.assert_allclose(
            upstream.numpy(), grad_in_ref[stage_id + 1], rtol=1e-5, atol=1e-5
        )

    grad = node.backward(upstream)

    if stage_id == 0:
        # x is frozen, so ScheduleNode returns no input gradient here.
        assert isinstance(grad, tuple) and len(grad) == 0
    else:
        assert len(grad) == 1
        np.testing.assert_allclose(
            grad[0].numpy(), grad_in_ref[stage_id], rtol=1e-5, atol=1e-5
        )
        dist.send(grad[0], dst=prev_rank, group=pp_group)

    # Each stage's own weight gradient is checked against its hand-derived
    # value: a mis-routed activation or gradient would corrupt these numbers.
    assert weight.grad is not None
    np.testing.assert_allclose(
        weight.grad.numpy(), dw_ref[stage_id], rtol=1e-5, atol=1e-5
    )


if __name__ == "__main__":
    _init_pp()
    test_detach_and_requires_grad_preserves_value_and_breaks_graph()
    test_detach_and_requires_grad_stop_gradient_and_containers()
    test_fake_clone_shape_and_identity_backward()
    test_clone_and_clear_dataptr_filters_and_passes_grad()
    test_schedule_node_forward_backward()
    test_schedule_node_recompute_forward_backward()
    test_schedule_chunk_local_chain()
    test_schedule_chunk_rejects_invalid_node()
    test_four_stage_pipeline_forward_backward()
