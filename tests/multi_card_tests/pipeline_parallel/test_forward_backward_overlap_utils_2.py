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

"""Multi-card (PP=2) tests for pipeline forward/backward overlap utilities.

The centerpiece is ``TestTwoStagePipelineOverlapUtils``: it builds a real
two-stage pipeline where stage 0 (rank 0) and stage 1 (rank 1) each own a
distinct weight, drives forward with a real ``paddle.distributed.send``/``recv``
of the activation across ranks, and drives backward with a real send/recv of
the activation gradient back across ranks -- exactly how ``ScheduleNode`` is
consumed inside the pipeline scheduler. All expected values are derived by
hand from the fixed constants below, never from the functions under test.

The remaining test cases exercise the *local* numeric contracts of the schedule
primitives (they are per-rank, non-collective helpers by construction) with
distinguishable content and hand-derived expectations.
"""

import unittest

import numpy as np
import paddle
from paddle.distributed import fleet

from paddlefleet.pipeline_parallel.pp_utils.forward_backward_overlap_utils import (
    FakeClone,
    ScheduleChunk,
    ScheduleNode,
    clone_and_clear_dataptr,
    detach_and_requires_grad,
)

PP_DEGREE = 2


def setUpModule():
    """Initialize fleet once for all tests in this module (PP=2)."""
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": PP_DEGREE,
    }
    fleet.init(is_collective=True, strategy=strategy)
    np.random.seed(2024)
    paddle.seed(2024)


def _param(value):
    """Create a parameter holding exactly ``value`` (row-major float32)."""
    arr = np.array(value, dtype="float32")
    return paddle.create_parameter(
        shape=list(arr.shape),
        dtype="float32",
        default_initializer=paddle.nn.initializer.Assign(arr),
    )


class TestTwoStagePipelineOverlapUtils(unittest.TestCase):
    """Real PP=2 pipeline driven by ScheduleNode + real p2p collectives.

    Fixed constants (batch=1, hidden=2):
        x  = [[1, 2]]                 on stage 0
        W0 = [[1, 2], [3, 4]]         on stage 0
        W1 = [[5, 6], [7, 8]]         on stage 1
        g  = [[1, 3]]  (upstream grad on stage 1's output, non-uniform)

    Hand derivation (independent of the module under test):
        h      = x @ W0            = [[7, 10]]
        y      = h @ W1            = [[105, 122]]
        dL/dh  = g @ W1^T          = [[23, 31]]
        dL/dW1 = h^T @ g           = [[7, 21], [10, 30]]
        dL/dW0 = x^T @ (dL/dh)     = [[23, 31], [46, 62]]
    """

    def test_two_stage_pipeline_forward_backward(self):
        hcg = fleet.get_hybrid_communicate_group()
        pp_group = hcg.get_pipe_parallel_group()
        stage_id = hcg.get_stage_id()
        next_rank = hcg._get_p2p_next_rank()
        prev_rank = hcg._get_p2p_prev_rank()

        h_ref = np.array([[7.0, 10.0]], dtype="float32")
        grad_h_ref = np.array([[23.0, 31.0]], dtype="float32")
        w0_grad_ref = np.array([[23.0, 31.0], [46.0, 62.0]], dtype="float32")
        w1_grad_ref = np.array([[7.0, 21.0], [10.0, 30.0]], dtype="float32")

        if stage_id == 0:
            w0 = _param([[1.0, 2.0], [3.0, 4.0]])
            x = paddle.to_tensor([[1.0, 2.0]], dtype="float32").cuda()
            x.stop_gradient = True

            def fwd0(inputs, **kwargs):
                return paddle.matmul(inputs, w0)

            node0 = ScheduleNode(fwd_func=fwd0, name="stage0")
            h = node0.forward(x)
            np.testing.assert_allclose(h.numpy(), h_ref, rtol=1e-5, atol=1e-6)

            # Forward: real activation transfer to the next stage.
            paddle.distributed.send(h, dst=next_rank, group=pp_group)

            # Backward: real gradient transfer from the next stage.
            grad_h = paddle.zeros([1, 2], dtype="float32").cuda()
            paddle.distributed.recv(grad_h, src=next_rank, group=pp_group)
            np.testing.assert_allclose(
                grad_h.numpy(), grad_h_ref, rtol=1e-5, atol=1e-6
            )

            node0.backward(grad_h)
            self.assertIsNotNone(w0.grad)
            np.testing.assert_allclose(
                w0.grad.numpy(), w0_grad_ref, rtol=1e-5, atol=1e-6
            )
        else:
            w1 = _param([[5.0, 6.0], [7.0, 8.0]])

            recv_h = paddle.zeros([1, 2], dtype="float32").cuda()
            paddle.distributed.recv(recv_h, src=prev_rank, group=pp_group)
            # The activation actually delivered across ranks must match the
            # independent hand derivation of x @ W0.
            np.testing.assert_allclose(
                recv_h.numpy(), h_ref, rtol=1e-5, atol=1e-6
            )
            recv_h.stop_gradient = False

            def fwd1(inputs, **kwargs):
                return paddle.matmul(inputs, w1)

            node1 = ScheduleNode(fwd_func=fwd1, name="stage1")
            y = node1.forward(recv_h)
            np.testing.assert_allclose(
                y.numpy(), [[105.0, 122.0]], rtol=1e-5, atol=1e-6
            )

            upstream = paddle.to_tensor([[1.0, 3.0]], dtype="float32").cuda()
            grads = node1.backward(upstream)
            self.assertEqual(len(grads), 1)
            np.testing.assert_allclose(
                grads[0].numpy(), grad_h_ref, rtol=1e-5, atol=1e-6
            )
            self.assertIsNotNone(w1.grad)
            np.testing.assert_allclose(
                w1.grad.numpy(), w1_grad_ref, rtol=1e-5, atol=1e-6
            )

            # Send the input gradient back to the previous stage.
            paddle.distributed.send(grads[0], dst=prev_rank, group=pp_group)


class TestScheduleNodeNumeric(unittest.TestCase):
    """Local numeric contract of a single ScheduleNode (per-rank)."""

    def test_forward_backward_and_reset(self):
        w = _param([[1.0, 2.0], [3.0, 4.0]])
        x = paddle.to_tensor([[1.0, 2.0]], dtype="float32").cuda()
        x.stop_gradient = False

        def fwd(inputs, **kwargs):
            return paddle.matmul(inputs, w)

        node = ScheduleNode(fwd_func=fwd, name="linear")
        out = node.forward(x)
        # out = x @ W = [[7, 10]]
        np.testing.assert_allclose(
            out.numpy(), [[7.0, 10.0]], rtol=1e-5, atol=1e-6
        )
        self.assertIsNotNone(node.inputs)
        self.assertIsNotNone(node.outputs)

        upstream = paddle.to_tensor([[1.0, 3.0]], dtype="float32").cuda()
        grads = node.backward(upstream)
        # dL/dx = g @ W^T = [[7, 15]]; dL/dW = x^T @ g = [[1, 3], [2, 6]]
        self.assertEqual(len(grads), 1)
        np.testing.assert_allclose(
            grads[0].numpy(), [[7.0, 15.0]], rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            w.grad.numpy(), [[1.0, 3.0], [2.0, 6.0]], rtol=1e-5, atol=1e-6
        )
        # backward must reset the retained forward state.
        self.assertIsNone(node.inputs)
        self.assertIsNone(node.outputs)


class TestScheduleChunkNumeric(unittest.TestCase):
    """Local numeric contract of ScheduleChunk chaining (per-rank)."""

    def test_chunk_chains_forward_and_backward(self):
        wa = _param([[1.0, 2.0], [3.0, 4.0]])
        wb = _param([[5.0, 6.0], [7.0, 8.0]])
        x = paddle.to_tensor([[1.0, 2.0]], dtype="float32").cuda()
        x.stop_gradient = False

        def fwd_a(inputs, **kwargs):
            return paddle.matmul(inputs, wa)

        def fwd_b(inputs, **kwargs):
            return paddle.matmul(inputs, wb)

        chunk = ScheduleChunk(
            [
                ScheduleNode(fwd_func=fwd_a, name="a"),
                ScheduleNode(fwd_func=fwd_b, name="b"),
            ]
        )
        # forward: x @ Wa @ Wb = [[7, 10]] @ Wb = [[105, 122]]
        y = chunk.forward(x)
        np.testing.assert_allclose(
            y.numpy(), [[105.0, 122.0]], rtol=1e-5, atol=1e-6
        )

        upstream = paddle.to_tensor([[1.0, 3.0]], dtype="float32").cuda()
        grad_x = chunk.backward(upstream)
        # dL/dh_a = g @ Wb^T = [[23, 31]]; dL/dx = [[23,31]] @ Wa^T = [[85, 193]]
        self.assertEqual(len(grad_x), 1)
        np.testing.assert_allclose(
            grad_x[0].numpy(), [[85.0, 193.0]], rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            wb.grad.numpy(), [[7.0, 21.0], [10.0, 30.0]], rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            wa.grad.numpy(), [[23.0, 31.0], [46.0, 62.0]], rtol=1e-5, atol=1e-6
        )


class TestScheduleChunkValidation(unittest.TestCase):
    """ScheduleChunk must reject members that are not ScheduleNode/Chunk."""

    def test_rejects_non_node_members(self):
        with self.assertRaises(AssertionError):
            ScheduleChunk([lambda inputs: inputs])


class TestFakeCloneGradientPassthrough(unittest.TestCase):
    """FakeClone yields a same-shape/dtype tensor and passes grad through."""

    def test_shape_dtype_and_grad_passthrough(self):
        x = paddle.to_tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32"
        ).cuda()
        x.stop_gradient = False
        cloned = FakeClone.apply(x)
        # forward returns empty_like: shape/dtype match, storage differs.
        # Content is intentionally uninitialized, so it is NOT asserted here.
        self.assertEqual(cloned.shape, x.shape)
        self.assertEqual(cloned.dtype, x.dtype)
        self.assertFalse(cloned is x)

        grad_out = paddle.to_tensor(
            [[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]], dtype="float32"
        ).cuda()
        paddle.autograd.backward([cloned], [grad_out])
        # FakeClone.backward returns grad_output unchanged.
        np.testing.assert_allclose(
            x.grad.numpy(), grad_out.numpy(), rtol=1e-6, atol=1e-6
        )


class TestCloneAndClearDataptr(unittest.TestCase):
    """clone_and_clear_dataptr structure/keys contract (per-rank)."""

    def test_tuple_filters_none_and_keeps_order(self):
        t1 = paddle.ones([2, 4], dtype="float32").cuda()
        t2 = paddle.ones([3, 6], dtype="float32").cuda()
        # A None element must be dropped: a broken impl that kept it would
        # yield length 3 instead of 2.
        result = clone_and_clear_dataptr((t1, None, t2))
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].shape, t1.shape)
        self.assertEqual(result[1].shape, t2.shape)

    def test_dict_preserves_keys(self):
        t1 = paddle.ones([2, 4], dtype="float32").cuda()
        t2 = paddle.ones([3, 6], dtype="float32").cuda()
        result = clone_and_clear_dataptr({"a": t1, "b": t2})
        self.assertIsInstance(result, dict)
        self.assertEqual(set(result.keys()), {"a", "b"})
        self.assertEqual(result["a"].shape, t1.shape)
        self.assertEqual(result["b"].shape, t2.shape)


class TestDetachAndRequiresGrad(unittest.TestCase):
    """detach_and_requires_grad preserves values/stop_gradient and detaches."""

    def test_tuple_preserves_values_and_stop_gradient(self):
        t1 = paddle.to_tensor([1.0, 2.0], dtype="float32").cuda()
        t1.stop_gradient = False
        t2 = paddle.to_tensor([3.0, 4.0], dtype="float32").cuda()
        t2.stop_gradient = True
        result = detach_and_requires_grad((t1, t2))
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        np.testing.assert_allclose(result[0].numpy(), [1.0, 2.0], rtol=1e-6)
        np.testing.assert_allclose(result[1].numpy(), [3.0, 4.0], rtol=1e-6)
        # stop_gradient flag of each input is preserved.
        self.assertFalse(result[0].stop_gradient)
        self.assertTrue(result[1].stop_gradient)
        # The result is genuinely detached: gradient must not reach t1.
        (result[0] * 2.0).sum().backward()
        self.assertIsNone(t1.grad)

    def test_nested_tuple_recursion(self):
        t1 = paddle.to_tensor([1.0], dtype="float32").cuda()
        t1.stop_gradient = False
        t2 = paddle.to_tensor([2.0], dtype="float32").cuda()
        t2.stop_gradient = False
        t3 = paddle.to_tensor([3.0], dtype="float32").cuda()
        t3.stop_gradient = False
        result = detach_and_requires_grad(((t1, t2), t3))
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], tuple)
        self.assertEqual(len(result[0]), 2)
        np.testing.assert_allclose(result[0][0].numpy(), [1.0], rtol=1e-6)
        np.testing.assert_allclose(result[0][1].numpy(), [2.0], rtol=1e-6)
        np.testing.assert_allclose(result[1].numpy(), [3.0], rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
