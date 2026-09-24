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

"""Behavior tests for paddlefleet.nn.moe.all_to_all.

Scope and environment (无卡 / CPU only):
    ``AlltoAll`` and ``AlltoAllAsync`` are expert-parallel collectives. Their
    cross-rank numerics (token dispatch/combine, split sizes, peer ordering)
    can only be proven with a real >=2 rank process group where each rank
    holds distinguishable data. Those cases are explicitly skipped below with
    reasons, per the distributed-training rules: faking ``get_world_size`` and
    mocking ``stream.alltoall_single`` only proves parameter passing, never the
    real reorder/reduction.

    What IS genuinely CPU-testable is the ``world_size <= 1`` local path that
    both PyLayers take: the all-to-all collapses to identity, and
    ``AlltoAllAsync`` still runs the real overlapped ``fn`` through
    ``manual_backward``. We verify value identity (not shape-only), forward
    inverse (gradient identity) for the sync op, and real ``fn`` execution /
    output composition for the async op against independently computed
    expectations.
"""

import unittest

import numpy as np
import paddle
from paddle.distributed.communication.group import Group

from paddlefleet.nn.moe.all_to_all import AlltoAll, AlltoAllAsync


def _single_rank_group():
    """Construct a real single-rank group (nranks == 1).

    This mirrors the "dummy" group built by
    ``paddlefleet.nn.moe.utils._parse_moe_group`` and drives the genuine
    ``world_size <= 1`` local code path -- no collective is mocked.
    """
    return Group(0, None, [0])


class TestAlltoAllSingleRank(unittest.TestCase):
    """world_size == 1: the sync all-to-all must be an identity map."""

    def setUp(self):
        paddle.set_device("cpu")
        self.group = _single_rank_group()
        # Precondition: we are genuinely on the local (world_size<=1) path.
        self.assertEqual(paddle.distributed.get_world_size(self.group), 1)

    def test_forward_returns_exact_input_values(self):
        # Distinguishable content so a reorder/zeroing would be caught.
        x = paddle.arange(4 * 8, dtype="float32").reshape([4, 8])
        out = AlltoAll.apply(x, group=self.group, sync_op=True)
        # Identity path: values must be preserved element-for-element,
        # not merely the same shape.
        np.testing.assert_array_equal(out.numpy(), x.numpy())

    def test_forward_identity_for_non_c_order_content(self):
        # A second distinguishable pattern (descending) rules out an
        # accidental "return sorted / return zeros_like" implementation.
        base = np.arange(6 * 5, dtype="float32")[::-1].reshape([6, 5]).copy()
        x = paddle.to_tensor(base)
        out = AlltoAll.apply(x, group=self.group, sync_op=True)
        np.testing.assert_array_equal(out.numpy(), base)

    def test_backward_is_gradient_identity(self):
        # Drive a real autograd graph through a scaling node so the gradient
        # that reaches the leaf exercises the single-rank inverse of AlltoAll.
        leaf = paddle.arange(4 * 8, dtype="float32").reshape([4, 8])
        leaf.stop_gradient = False
        x = leaf * 2.0  # non-leaf feeding the collective
        out = AlltoAll.apply(x, group=self.group, sync_op=True)

        # Non-uniform upstream grad exposes any reorder in the backward map.
        upstream = paddle.arange(4 * 8, dtype="float32").reshape([4, 8]) - 5.0
        out.backward(upstream)

        # AlltoAll is self-inverse and identity at world_size==1, so the grad
        # arriving at ``leaf`` is exactly 2 * upstream (from the scaling node).
        expected = 2.0 * (
            np.arange(4 * 8, dtype="float32").reshape([4, 8]) - 5.0
        )
        self.assertIsNotNone(leaf.grad)
        np.testing.assert_allclose(
            leaf.grad.numpy(), expected, rtol=1e-6, atol=0
        )


class TestAlltoAllAsyncSingleRank(unittest.TestCase):
    """world_size == 1: async op returns (x,) + fn(*fn_args), fn run for real."""

    def setUp(self):
        paddle.set_device("cpu")
        self.group = _single_rank_group()
        self.assertEqual(paddle.distributed.get_world_size(self.group), 1)

    def test_fn_none_raises_assertion(self):
        # Real contract: fn is mandatory for the async op.
        x = paddle.arange(4 * 8, dtype="float32").reshape([4, 8])
        with self.assertRaises(AssertionError):
            AlltoAllAsync.apply(x, group=self.group, fn=None, is_first_fwd=True)

    def test_forward_composes_identity_payload_with_real_fn_output(self):
        # x is the all-to-all payload (identity at world_size==1); ``a`` is the
        # separate argument the overlapped fn actually consumes.
        x_np = np.arange(4 * 8, dtype="float32").reshape([4, 8])
        a_np = (np.arange(4 * 8, dtype="float32").reshape([4, 8]) - 3.0) * 0.5
        x = paddle.to_tensor(x_np)
        a = paddle.to_tensor(a_np)

        def fn(t):
            return t * 3.0 + 1.0

        out = AlltoAllAsync.apply(
            x, a, group=self.group, fn=fn, is_first_fwd=True
        )

        # Structure: (payload,) + (fn_out,)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        # Payload passes through untouched (identity), value-for-value.
        np.testing.assert_array_equal(out[0].numpy(), x_np)
        # fn is genuinely executed on the real argument; compare to an
        # independently computed expectation, not a shape.
        np.testing.assert_allclose(
            out[1].numpy(), a_np * 3.0 + 1.0, rtol=1e-6, atol=0
        )

    def test_forward_consumes_each_fn_arg(self):
        # Two distinguishable args combined non-symmetrically: swapping or
        # dropping an arg changes the numeric result.
        a_np = np.arange(4 * 8, dtype="float32").reshape([4, 8])
        b_np = np.ones([4, 8], dtype="float32") * 10.0
        x = paddle.zeros([4, 8], dtype="float32")
        a = paddle.to_tensor(a_np)
        b = paddle.to_tensor(b_np)

        def fn(t_a, t_b):
            return t_a - 2.0 * t_b

        out = AlltoAllAsync.apply(
            x, a, b, group=self.group, fn=fn, is_first_fwd=True
        )
        self.assertEqual(len(out), 2)
        np.testing.assert_allclose(
            out[1].numpy(), a_np - 2.0 * b_np, rtol=1e-6, atol=0
        )

    def test_forward_multi_output_fn_flattened(self):
        # fn returning a tuple -> (x,) + (o1, o2): length 3, each value checked.
        a_np = np.arange(4 * 8, dtype="float32").reshape([4, 8])
        x = paddle.arange(4 * 8, dtype="float32").reshape([4, 8])
        a = paddle.to_tensor(a_np)

        def fn(t):
            return t + 100.0, t * -1.0

        out = AlltoAllAsync.apply(
            x, a, group=self.group, fn=fn, is_first_fwd=True
        )
        self.assertEqual(len(out), 3)
        np.testing.assert_array_equal(out[0].numpy(), x.numpy())
        np.testing.assert_allclose(out[1].numpy(), a_np + 100.0, rtol=1e-6)
        np.testing.assert_allclose(out[2].numpy(), a_np * -1.0, rtol=1e-6)

    def test_forward_list_output_converted_to_tuple(self):
        # A list return from fn is normalized to tuple entries.
        a_np = np.arange(4 * 8, dtype="float32").reshape([4, 8])
        x = paddle.arange(4 * 8, dtype="float32").reshape([4, 8])
        a = paddle.to_tensor(a_np)

        def fn(t):
            return [t + 7.0]

        out = AlltoAllAsync.apply(
            x, a, group=self.group, fn=fn, is_first_fwd=True
        )
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        np.testing.assert_allclose(out[1].numpy(), a_np + 7.0, rtol=1e-6)


class TestCrossRankBehaviorNotVerified(unittest.TestCase):
    """Explicit skips: multi-rank numerics cannot be proven on one CPU rank."""

    def test_cross_rank_alltoall_dispatch_combine_skipped(self):
        self.skipTest(
            "Real all-to-all dispatch/combine (per-rank token ownership, "
            "split sizes, peer ordering, backward reverse-alltoall) requires a "
            ">=2 rank process group with distinguishable per-rank data. Faking "
            "world_size and mocking stream.alltoall_single would only assert "
            "call/args, not the reorder/reduction (antipattern #13). Belongs in "
            "tests/multi_card_tests/moe/ under a real launcher."
        )

    def test_alltoall_smart_permute_index_math_skipped(self):
        self.skipTest(
            "AlltoAllSmart's distributed_input->alltoall_out permutation "
            "(recv_mask cumsum/transpose split index math in "
            "paddlefleet.nn.moe.all_gather) collapses to the identity map when "
            "world_size==1 (all tokens received locally), so a single rank "
            "cannot discriminate a wrong permutation. Discriminating cases need "
            ">=2 real ranks with differing expert ownership."
        )

    def test_async_manual_backward_pipeline_skipped(self):
        self.skipTest(
            "AlltoAllAsync's is_first_fwd=False branch caches fn outputs and "
            "frees their data ptr for the two-phase manual_backward pipeline; "
            "driving its gradient path in isolation tests framework internals "
            "rather than the collective. Its end-to-end backward is covered by "
            "the real MoE layer forward/backward under a multi-card launcher."
        )


if __name__ == "__main__":
    unittest.main()
