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

"""Behavior tests for pipeline forward/backward-overlap scheduling primitives.

These utilities (``detach_and_requires_grad``, ``FakeClone``,
``clone_and_clear_dataptr``, ``ScheduleNode`` and ``ScheduleChunk``) are the
rank-local building blocks of the pipeline-overlap scheduler. They run on real
``.cuda()`` tensors inside a genuine PP=4 fleet process group (launched with
``paddle.distributed.launch``), and every expected value is derived by hand /
with an independent NumPy reference -- never from the function under test.

The primitives themselves do not issue pipeline collectives; the cross-rank
send/recv lives in ``p2p_communication``. Accordingly this file makes no
cross-rank numeric claim: each of the 4 PP ranks verifies the local forward and
backward contract independently on distinguishable, rank-dependent inputs.
"""

import unittest

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet

from paddlefleet.pipeline_parallel.pp_utils.forward_backward_overlap_utils import (
    FakeClone,
    ScheduleChunk,
    ScheduleNode,
    clone_and_clear_dataptr,
    detach_and_requires_grad,
)
from paddlefleet.training.initialize import initialize_fleet

PP_DEGREE = 4


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


class TestDetachAndRequiresGrad(unittest.TestCase):
    def test_plain_tensor_preserves_value_and_breaks_graph(self):
        # Rank-dependent, distinguishable content so a mixed-up tensor shows up.
        base = (paddle.arange(6, dtype="float32").reshape([2, 3])).cuda()
        base = base + float(dist.get_rank())
        a = base.clone()
        a.stop_gradient = False
        b = a * 2.0

        d = detach_and_requires_grad(b)

        # Value is preserved exactly by the detach.
        np.testing.assert_array_equal(d.numpy(), base.numpy() * 2.0)
        # stop_gradient is copied from the source (b required grad -> d does).
        self.assertFalse(d.stop_gradient)

        # The detach must sever the graph: gradients from a function of d must
        # not reach a, while d itself is a fresh leaf that accumulates grad.
        loss = (d * 3.0).sum()
        loss.backward()
        self.assertIsNone(a.grad)
        np.testing.assert_array_equal(
            d.grad.numpy(), np.full([2, 3], 3.0, dtype="float32")
        )

    def test_preserves_stop_gradient_true(self):
        a = (paddle.arange(3, dtype="float32") + dist.get_rank()).cuda()
        a.stop_gradient = True

        d = detach_and_requires_grad(a)

        # Despite the name, the helper copies stop_gradient rather than forcing
        # it False; a True input must stay True.
        self.assertTrue(d.stop_gradient)
        np.testing.assert_array_equal(d.numpy(), a.numpy())

    def test_tuple_nested_and_nontensor_passthrough(self):
        t1 = paddle.to_tensor([1.0, 2.0]).cuda()
        t1.stop_gradient = False
        t2 = paddle.to_tensor([3.0, 4.0]).cuda()
        t2.stop_gradient = True
        nested = [paddle.to_tensor([5.0]).cuda()]
        sentinel = 7  # non-tensor element must pass through unchanged

        out = detach_and_requires_grad((t1, t2, nested, sentinel))

        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 4)
        self.assertFalse(out[0].stop_gradient)
        self.assertTrue(out[1].stop_gradient)
        self.assertIsInstance(out[2], list)
        np.testing.assert_array_equal(out[0].numpy(), [1.0, 2.0])
        np.testing.assert_array_equal(out[1].numpy(), [3.0, 4.0])
        np.testing.assert_array_equal(out[2][0].numpy(), [5.0])
        self.assertEqual(out[3], 7)

    def test_dict_preserves_keys_and_none(self):
        t = paddle.to_tensor([1.0, 2.0]).cuda()
        t.stop_gradient = False

        out = detach_and_requires_grad({"a": t, "b": None})

        self.assertIsInstance(out, dict)
        self.assertEqual(set(out), {"a", "b"})
        self.assertFalse(out["a"].stop_gradient)
        np.testing.assert_array_equal(out["a"].numpy(), [1.0, 2.0])
        self.assertIsNone(out["b"])


class TestFakeCloneAndCloneClear(unittest.TestCase):
    def test_fakeclone_shape_dtype_and_identity_backward(self):
        x = (paddle.arange(6, dtype="float32").reshape([2, 3])).cuda()
        x = x + float(dist.get_rank())
        x.stop_gradient = False

        out = FakeClone.apply(x)

        # Forward returns empty_like: same shape/dtype, content is uninitialized
        # by design (avoids the DtoD copy), so content is deliberately NOT
        # compared. The load-bearing contract is the identity backward.
        self.assertEqual(list(out.shape), [2, 3])
        self.assertEqual(out.dtype, x.dtype)
        self.assertFalse(out.stop_gradient)

        upstream = (
            paddle.arange(6, dtype="float32").reshape([2, 3]) * 0.5 + 1.0
        ).cuda()
        paddle.autograd.backward([out], [upstream])
        np.testing.assert_array_equal(x.grad.numpy(), upstream.numpy())

    def test_clone_and_clear_list_drops_none_and_nontensor(self):
        t1 = (paddle.arange(4, dtype="float32") + dist.get_rank()).cuda()
        t2 = (paddle.arange(4, dtype="float32") + 10.0).reshape([2, 2]).cuda()

        ret = clone_and_clear_dataptr([t1, None, t2, 5])

        # None and the plain int are filtered out; only the two tensors remain,
        # each keeping shape/dtype (content is FakeClone empty_like, not checked).
        self.assertIsInstance(ret, list)
        self.assertEqual(len(ret), 2)
        self.assertEqual(list(ret[0].shape), [4])
        self.assertEqual(list(ret[1].shape), [2, 2])
        self.assertEqual(ret[0].dtype, t1.dtype)

    def test_clone_and_clear_dict_drops_none_value(self):
        t = (paddle.arange(4, dtype="float32") + dist.get_rank()).cuda()

        ret = clone_and_clear_dataptr({"x": t, "y": None})

        self.assertIsInstance(ret, dict)
        self.assertEqual(set(ret), {"x"})
        self.assertEqual(list(ret["x"].shape), [4])

    def test_clone_and_clear_single_tensor_identity_backward(self):
        x = (paddle.arange(6, dtype="float32").reshape([2, 3])).cuda()
        x = x + float(dist.get_rank())
        x.stop_gradient = False

        wrapped = clone_and_clear_dataptr(x)

        self.assertEqual(list(wrapped.shape), [2, 3])
        # Purpose of the wrapper is to retain the gradient path even when the
        # forward data is dropped: grad must flow straight through unchanged.
        upstream = (paddle.ones([2, 3]) * 2.0).cuda()
        paddle.autograd.backward([wrapped], [upstream])
        np.testing.assert_array_equal(x.grad.numpy(), upstream.numpy())


def _linear_fwd(weight):
    def fwd_func(inputs, is_first_fwd=False):
        return paddle.matmul(inputs, weight)

    return fwd_func


class TestScheduleNodeForwardBackward(unittest.TestCase):
    def _weight(self):
        w = paddle.to_tensor(
            [[1.0, 2.0], [0.0, 1.0], [3.0, -1.0]], dtype="float32"
        ).cuda()
        w.stop_gradient = True
        return w

    def _inputs(self):
        x_np = np.arange(6, dtype="float32").reshape(2, 3) + dist.get_rank()
        x = paddle.to_tensor(x_np).cuda()
        x.stop_gradient = False
        return x, x_np

    def test_non_recompute_forward_and_backward(self):
        w = self._weight()
        w_np = w.numpy()
        node = ScheduleNode(_linear_fwd(w), name="lin")
        x, x_np = self._inputs()

        out = node.forward(x)
        np.testing.assert_allclose(
            out.numpy(), x_np @ w_np, rtol=1e-5, atol=1e-5
        )

        g_np = np.arange(4, dtype="float32").reshape(2, 2) + 1.0
        grad = node.backward(paddle.to_tensor(g_np).cuda())

        self.assertIsInstance(grad, tuple)
        self.assertEqual(len(grad), 1)
        np.testing.assert_allclose(
            grad[0].numpy(), g_np @ w_np.T, rtol=1e-5, atol=1e-5
        )

    def test_recompute_forward_and_backward(self):
        w = self._weight()
        w_np = w.numpy()
        node = ScheduleNode(_linear_fwd(w), name="lin_rc")
        x, x_np = self._inputs()

        # first_forward runs under no_grad and captures RNG/AMP state; the
        # subsequent forward recomputes under a real dygraph guard.
        node.first_forward(x)
        out = node.forward(x)
        np.testing.assert_allclose(
            out.numpy(), x_np @ w_np, rtol=1e-5, atol=1e-5
        )

        g_np = np.arange(4, dtype="float32").reshape(2, 2) + 1.0
        grad = node.backward(paddle.to_tensor(g_np).cuda())
        self.assertEqual(len(grad), 1)
        np.testing.assert_allclose(
            grad[0].numpy(), g_np @ w_np.T, rtol=1e-5, atol=1e-5
        )


class TestScheduleChunkComposition(unittest.TestCase):
    def test_chunk_chains_forward_and_backward(self):
        w1 = paddle.to_tensor(
            [[1.0, 2.0], [0.0, 1.0], [3.0, -1.0]], dtype="float32"
        ).cuda()
        w1.stop_gradient = True
        w2 = paddle.to_tensor(
            [[1.0, 0.0, 2.0, -1.0], [0.5, 1.0, -2.0, 3.0]], dtype="float32"
        ).cuda()
        w2.stop_gradient = True
        w1_np, w2_np = w1.numpy(), w2.numpy()

        n1 = ScheduleNode(_linear_fwd(w1), name="n1")
        n2 = ScheduleNode(_linear_fwd(w2), name="n2")
        chunk = ScheduleChunk([n1, n2])
        self.assertEqual(len(chunk.nodes), 2)

        x_np = np.arange(6, dtype="float32").reshape(2, 3) + dist.get_rank()
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
        self.assertEqual(len(grad), 1)
        np.testing.assert_allclose(
            grad[0].numpy(), expected, rtol=1e-5, atol=1e-5
        )

    def test_chunk_rejects_invalid_node(self):
        with self.assertRaises(AssertionError):
            ScheduleChunk([object()])


if __name__ == "__main__":
    unittest.main()
