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

"""No-card behavior tests for paddlefleet.nn.moe.utils.

Scope (无卡 / distributed + model-layer rules):
  moe/utils.py is a collection of distributed/autograd helpers. This suite
  verifies the parts whose *behavior* is real on a single CPU process:

    * scatter_axis        -- pure local index math: each rank slices its own
                             contiguous chunk out of the SAME full tensor. No
                             collective is involved, so the exact slice content
                             (and the partition -> reassembly invariant) is
                             genuine CPU evidence, not a faked collective.
    * detach_and_requires_grad_ / FakeClone / manual_backward
                          -- pure autograd helpers; forward content and the
                             hand-derived gradient values are checked.
    * all_gather_group / reduce_scatter_group / ScatterOp /
      ReduceScatterGroupOp / AllGatherGroupOp
                          -- ONLY the parallelism==1 local-copy branch and the
                             pure divisibility guard (which fires before any
                             communication). These are real production paths
                             that run on CPU.
    * _parse_moe_group    -- the string -> group dispatch table, the "dummy"
                             group construction, single-rank / exception
                             fallbacks and the validation guard.

  Deliberately NOT faked (see skip tests): the genuine cross-rank numerics of
  all_gather / reduce_scatter (parallelism>1) and the PyLayer backward that
  reduce-scatters / all-gathers across ranks. Those depend on multiple ranks
  each holding DIFFERENT data exchanged through a real process group. Faking
  `world_size` and mocking `dist.stream.*` (as the coverage_test did) only
  proves a call happened; it cannot detect a wrong peer, swapped shard or a
  dropped reduction term. They belong to the multi-card suite.

  fleet HCG and the global process group are mocked ONLY where they are an
  unavoidable external dependency of the branch under test (_parse_moe_group);
  no collective is simulated -- we observe the real branch selection / fallback.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

try:
    import paddle

    from paddlefleet.nn.moe import utils as moe_utils
    from paddlefleet.nn.moe.utils import (
        AllGatherGroupOp,
        FakeClone,
        ReduceScatterGroupOp,
        ScatterOp,
        all_gather_group,
        detach_and_requires_grad_,
        manual_backward,
        reduce_scatter_group,
        scatter_axis,
        _parse_moe_group,
    )

    _HAS_PADDLE = True
except ImportError:
    _HAS_PADDLE = False


def _group(nranks=1, rank=0):
    """Lightweight config carrier for a process group.

    scatter_axis / all_gather_group / reduce_scatter_group read only `.nranks`
    (and scatter_axis also `.rank`). When nranks==1 they short-circuit before
    touching paddle.distributed; the nranks>1 slice math in scatter_axis is
    purely local. This is a plain namespace, NOT a mock collective.
    """
    return SimpleNamespace(nranks=nranks, rank=rank)


@unittest.skipUnless(_HAS_PADDLE, "paddle / paddlefleet not importable")
class TestScatterAxis(unittest.TestCase):
    """scatter_axis is pure local slicing (no collective): verify exact content."""

    def test_parallelism_1_returns_independent_copy(self):
        x = paddle.arange(12, dtype="float32").reshape([4, 3])
        out = scatter_axis(x, group=_group(nranks=1), axis=0)
        # nranks==1 short-circuits to input.clone(): same values, distinct tensor.
        np.testing.assert_array_equal(out.numpy(), x.numpy())
        self.assertFalse(out is x)

    def test_axis0_each_rank_gets_its_own_contiguous_rows(self):
        x = paddle.arange(12, dtype="float32").reshape([4, 3])
        r0 = scatter_axis(x, group=_group(nranks=2, rank=0), axis=0)
        r1 = scatter_axis(x, group=_group(nranks=2, rank=1), axis=0)
        # interval = 4 // 2 = 2 -> rank0 rows [0:2], rank1 rows [2:4].
        np.testing.assert_array_equal(r0.numpy(), x.numpy()[0:2])
        np.testing.assert_array_equal(r1.numpy(), x.numpy()[2:4])
        # Partition invariant: shards are disjoint and reassemble the whole input.
        np.testing.assert_array_equal(
            np.concatenate([r0.numpy(), r1.numpy()], axis=0), x.numpy()
        )

    def test_axis1_slices_columns_not_rows(self):
        x = paddle.arange(12, dtype="float32").reshape([3, 4])
        c0 = scatter_axis(x, group=_group(nranks=2, rank=0), axis=1)
        c1 = scatter_axis(x, group=_group(nranks=2, rank=1), axis=1)
        np.testing.assert_array_equal(c0.numpy(), x.numpy()[:, 0:2])
        np.testing.assert_array_equal(c1.numpy(), x.numpy()[:, 2:4])
        np.testing.assert_array_equal(
            np.concatenate([c0.numpy(), c1.numpy()], axis=1), x.numpy()
        )

    def test_rank3_of_4_picks_the_correct_middle_chunk(self):
        # Guards against off-by-one in `interval * rank : interval * (rank + 1)`.
        x = paddle.arange(24, dtype="float32").reshape([8, 3])
        out = scatter_axis(x, group=_group(nranks=4, rank=3), axis=0)
        np.testing.assert_array_equal(out.numpy(), x.numpy()[6:8])

    def test_non_divisible_length_raises(self):
        x = paddle.arange(21, dtype="float32").reshape([7, 3])
        with self.assertRaises(AssertionError):
            scatter_axis(x, group=_group(nranks=2, rank=0), axis=0)


@unittest.skipUnless(_HAS_PADDLE, "paddle / paddlefleet not importable")
class TestDetachAndRequiresGrad(unittest.TestCase):
    """detach_and_requires_grad_ must preserve values, order, stop_gradient, None."""

    def test_preserves_value_order_and_stop_gradient_flags(self):
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        x.stop_gradient = False
        y = paddle.to_tensor([4.0, 5.0])
        y.stop_gradient = True
        out = detach_and_requires_grad_(x, y, None)

        self.assertEqual(len(out), 3)
        self.assertIsNone(out[2])
        # stop_gradient copied per-arg from the corresponding source.
        self.assertFalse(out[0].stop_gradient)
        self.assertTrue(out[1].stop_gradient)
        # Values preserved and kept in order (not swapped).
        np.testing.assert_array_equal(out[0].numpy(), [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(out[1].numpy(), [4.0, 5.0])
        # Returned tensors are detached copies, not the original objects.
        self.assertFalse(out[0] is x)
        self.assertFalse(out[1] is y)


@unittest.skipUnless(_HAS_PADDLE, "paddle / paddlefleet not importable")
class TestFakeClone(unittest.TestCase):
    """FakeClone: forward returns identical values; backward is the identity."""

    def test_contiguous_forward_value_and_identity_backward(self):
        x = paddle.arange(16, dtype="float32").reshape([4, 4])
        x.stop_gradient = False
        out = FakeClone.apply(x)
        np.testing.assert_array_equal(out.numpy(), x.numpy())

        # Non-uniform upstream so a scaled / permuted grad would be caught.
        upstream = paddle.to_tensor(
            np.arange(1, 17, dtype="float32").reshape([4, 4]) * 0.5
        )
        paddle.autograd.backward([out], [upstream])
        self.assertIsNotNone(x.grad)
        np.testing.assert_array_equal(x.grad.numpy(), upstream.numpy())

    def test_non_contiguous_forward_preserves_strided_content(self):
        base = paddle.arange(32, dtype="float32").reshape([4, 8])
        x = base[:, ::2]  # non-contiguous view -> forward takes the clone path
        out = FakeClone.apply(x)
        expected = np.arange(32, dtype="float32").reshape([4, 8])[:, ::2]
        np.testing.assert_array_equal(out.numpy(), expected)


@unittest.skipUnless(_HAS_PADDLE, "paddle / paddlefleet not importable")
class TestManualBackward(unittest.TestCase):
    """manual_backward: real forward values + hand-derived backward gradients."""

    def test_first_fwd_returns_no_backward_and_correct_output(self):
        x = paddle.arange(4, dtype="float32")
        x.stop_gradient = False
        bwd, out = manual_backward(lambda a: a * 2, True, x)
        # is_first_fwd=True: no backward closure, output still computed.
        self.assertIsNone(bwd)
        self.assertEqual(len(out), 1)
        np.testing.assert_array_equal(out[0].numpy(), x.numpy() * 2)

    def test_single_arg_backward_matches_analytic_gradient(self):
        x = paddle.arange(4, dtype="float32")
        x.stop_gradient = False
        bwd, out = manual_backward(lambda a: a * 2, False, x)
        np.testing.assert_array_equal(out[0].numpy(), x.numpy() * 2)

        upstream = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
        grads = bwd(upstream)
        # d(2a)/da = 2  ->  grad = 2 * upstream (element-wise, non-uniform).
        self.assertEqual(len(grads), 1)
        np.testing.assert_array_equal(grads[0].numpy(), upstream.numpy() * 2)

    def test_multi_arg_backward_routes_gradient_to_each_input(self):
        x = paddle.arange(4, dtype="float32")
        x.stop_gradient = False
        y = paddle.arange(4, dtype="float32") + 10.0
        y.stop_gradient = False
        bwd, out = manual_backward(lambda a, b: a + b, False, x, y)
        np.testing.assert_array_equal(out[0].numpy(), (x + y).numpy())

        upstream = paddle.to_tensor([2.0, 3.0, 5.0, 7.0])
        grads = bwd(upstream)
        # d(a+b)/da = d(a+b)/db = 1  ->  both grads equal upstream.
        self.assertEqual(len(grads), 2)
        np.testing.assert_array_equal(grads[0].numpy(), upstream.numpy())
        np.testing.assert_array_equal(grads[1].numpy(), upstream.numpy())


@unittest.skipUnless(_HAS_PADDLE, "paddle / paddlefleet not importable")
class TestLocalCollectivePaths(unittest.TestCase):
    """parallelism==1 copy branch + pure divisibility guard (no communication)."""

    def test_all_gather_group_nranks1_is_value_preserving_copy(self):
        x = paddle.arange(12, dtype="float32").reshape([4, 3])
        out = all_gather_group(x, group=_group(nranks=1), axis=0)
        np.testing.assert_array_equal(out.numpy(), x.numpy())
        self.assertFalse(out is x)

    def test_reduce_scatter_group_nranks1_is_value_preserving_copy(self):
        x = paddle.arange(12, dtype="float32").reshape([4, 3])
        out = reduce_scatter_group(x, group=_group(nranks=1))
        np.testing.assert_array_equal(out.numpy(), x.numpy())
        self.assertFalse(out is x)

    def test_reduce_scatter_group_non_divisible_raises_before_comm(self):
        # The divisibility assert fires before any collective is attempted,
        # so it is exercisable on CPU with a plain nranks=2 config carrier.
        x = paddle.arange(21, dtype="float32").reshape([7, 3])
        with self.assertRaises(AssertionError):
            reduce_scatter_group(x, group=_group(nranks=2))

    def test_scatter_op_nranks1_forward_copy_and_identity_backward(self):
        x = paddle.arange(12, dtype="float32").reshape([4, 3])
        x.stop_gradient = False
        out = ScatterOp.apply(x, axis=0, group=_group(nranks=1))
        np.testing.assert_array_equal(out.numpy(), x.numpy())

        upstream = paddle.to_tensor(
            np.arange(1, 13, dtype="float32").reshape([4, 3])
        )
        paddle.autograd.backward([out], [upstream])
        # nranks==1: backward all_gather_group is a copy -> gradient identity.
        np.testing.assert_array_equal(x.grad.numpy(), upstream.numpy())

    def test_scatter_op_forward_slices_locally_for_nranks2(self):
        # Forward only calls scatter_axis (local slice, no collective).
        x = paddle.arange(12, dtype="float32").reshape([4, 3])
        out = ScatterOp.apply(x, axis=0, group=_group(nranks=2, rank=1))
        np.testing.assert_array_equal(out.numpy(), x.numpy()[2:4])

    def test_reduce_scatter_op_nranks1_forward_and_backward_identity(self):
        x = paddle.arange(12, dtype="float32").reshape([4, 3])
        x.stop_gradient = False
        out = ReduceScatterGroupOp.apply(x, _group(nranks=1))
        np.testing.assert_array_equal(out.numpy(), x.numpy())
        upstream = paddle.to_tensor(
            np.arange(1, 13, dtype="float32").reshape([4, 3])
        )
        paddle.autograd.backward([out], [upstream])
        np.testing.assert_array_equal(x.grad.numpy(), upstream.numpy())

    def test_all_gather_op_nranks1_forward_and_backward_identity(self):
        x = paddle.arange(12, dtype="float32").reshape([4, 3])
        x.stop_gradient = False
        out = AllGatherGroupOp.apply(x, _group(nranks=1))
        np.testing.assert_array_equal(out.numpy(), x.numpy())
        upstream = paddle.to_tensor(
            np.arange(1, 13, dtype="float32").reshape([4, 3])
        )
        paddle.autograd.backward([out], [upstream])
        np.testing.assert_array_equal(x.grad.numpy(), upstream.numpy())

    @unittest.skip(
        "Cross-rank all_gather ordering / reduce_scatter SUM reduction and the "
        "PyLayer backwards that gather/scatter across ranks need multiple ranks "
        "with DIFFERENT data over a real process group. Faking world_size and "
        "mocking dist.stream.* only proves a call happened, not the peer / shard "
        "/ reduction correctness. See the multi-card MoE suite."
    )
    def test_cross_rank_collective_numerics(self):
        pass


@unittest.skipUnless(_HAS_PADDLE, "paddle / paddlefleet not importable")
class TestParseMoeGroup(unittest.TestCase):
    """String -> process-group dispatch, construction, fallbacks and validation.

    fleet HCG / global group are mocked because they require a real distributed
    init; they are external dependencies, not the logic under test. No collective
    is simulated -- we observe which branch is taken and what group is returned.
    """

    _HCG = "paddlefleet.nn.moe.utils.fleet.get_hybrid_communicate_group"
    _GLOBAL = "paddlefleet.nn.moe.utils._get_global_group"

    def test_invalid_group_name_raises(self):
        with self.assertRaises(AssertionError):
            _parse_moe_group("not_a_real_group")

    def test_dummy_builds_single_member_group(self):
        out = _parse_moe_group("dummy")
        self.assertEqual(out.nranks, 1)

    def test_name_is_case_insensitive(self):
        # "DUMMY" is lowercased then matched -> same single-member group.
        out = _parse_moe_group("DUMMY")
        self.assertEqual(out.nranks, 1)

    def test_dp_and_data_return_the_data_parallel_group(self):
        for name in ("dp", "data"):
            sentinel = object()
            hcg = mock.MagicMock()
            hcg.get_data_parallel_group.return_value = sentinel
            with mock.patch(self._HCG, return_value=hcg):
                out = _parse_moe_group(name)
            self.assertIs(out, sentinel, name)

    def test_mp_aliases_return_model_parallel_group_when_multi_rank(self):
        for name in ("mp", "model", "tp"):
            sentinel = SimpleNamespace(nranks=4)
            hcg = mock.MagicMock()
            hcg.get_model_parallel_group.return_value = sentinel
            with mock.patch(self._HCG, return_value=hcg):
                out = _parse_moe_group(name)
            self.assertIs(out, sentinel, name)

    def test_mp_single_rank_falls_back_to_fresh_dummy(self):
        sentinel = SimpleNamespace(nranks=1)
        hcg = mock.MagicMock()
        hcg.get_model_parallel_group.return_value = sentinel
        with mock.patch(self._HCG, return_value=hcg):
            out = _parse_moe_group("mp")
        # nranks<=1 -> a brand new single-member group, NOT the mp group itself.
        self.assertIsNot(out, sentinel)
        self.assertEqual(out.nranks, 1)

    def test_mp_falls_back_to_dummy_when_hcg_raises(self):
        hcg = mock.MagicMock()
        hcg.get_model_parallel_group.side_effect = RuntimeError("no mp group")
        with mock.patch(self._HCG, return_value=hcg):
            out = _parse_moe_group("mp")
        self.assertEqual(out.nranks, 1)

    def test_world_none_all_return_the_global_group(self):
        for name in ("world", "none", "all"):
            sentinel = object()
            with mock.patch(self._GLOBAL, return_value=sentinel):
                out = _parse_moe_group(name)
            self.assertIs(out, sentinel, name)


# Intentionally not covered here, with reason:
#   * get_hcg / hack_offload_wait -- one-line delegations to fleet / task with no
#     branching logic to verify on CPU.
#   * get_async_loader -- device async-load plumbing bound to a module global and
#     fleet internals; its only real logic is "create once", and exercising it
#     would require mocking create_async_load (the collaborator being cached)
#     plus fleet.fleet, which yields no meaningful CPU behavior evidence.


if __name__ == "__main__":
    unittest.main()
