# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""Single-card unit tests for the AllGather token-dispatcher PyLayers.

These tests exercise the **single-rank fallback paths** — where the
collective ops would be a no-op — without spinning up multi-card NCCL.
They cover:

* ``ReduceScatterGroupOp`` forward (clone) + backward (clone) on a
  single-rank group.
* ``_RouterAllGather`` (in token_dispatcher.py) forward+backward with
  ``group=None`` and on a single-rank group, including the explicit
  shape-restoration path on backward (handles the 1-D grad case from
  sonic-moe).
* ``_AllGatherCombineAsync`` forward+backward for nranks==1; verifies
  that the captured ``fn`` graph runs and its outputs/gradients flow.
* ``_PreAllGatherResult`` against a manually-built single-rank handle.
* ``AllGatherTokenDispatcher.pre_allgather`` early return when the
  group is single-rank.
* ``AllGatherTokenDispatcher.token_combine`` with
  ``combine_overlap_handle=None`` — passthrough path.
* ``GroupedMLPExpert.intermediate_size_per_partition`` constructor
  override.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import paddle
import paddle.distributed as dist
import paddlefleet_ops

# Temporarily disable sonicmoe Python imports during module load.
# The sonicmoe ecosystem ops are already loaded; re-importing the
# Python wrapper triggers custom-op re-registration crashes.
_original_sonic_moe_available = paddlefleet_ops.is_sonic_moe_available
paddlefleet_ops.is_sonic_moe_available = lambda: False

from paddlefleet.transformer.moe.moe_expert import GroupedMLPExpert
from paddlefleet.transformer.moe.moe_layer import MoELayer
from paddlefleet.transformer.moe.moe_utils import ReduceScatterGroupOp
from paddlefleet.transformer.moe.token_dispatcher import (
    AllGatherTokenDispatcher,
    MoEFlexTokenDispatcher,
    _AllGatherCombineAsync,
    _PreAllGatherResult,
    _RouterAllGather,
)

# Restore the real value so concurrent / subsequent test files see it.
paddlefleet_ops.is_sonic_moe_available = _original_sonic_moe_available


def _single_rank_group():
    return dist.new_group([dist.get_rank()])


class TestReduceScatterGroupOp(unittest.TestCase):
    """Forward/backward of ReduceScatterGroupOp on a single-rank group.

    nranks==1 is the ``input.clone()`` shortcut inside
    ``reduce_scatter_group`` / ``all_gather_group`` so this exercises
    the PyLayer plumbing (barrier + ctx.group) without real NCCL traffic.
    """

    def test_forward_single_rank_returns_clone(self):
        g = _single_rank_group()
        x = paddle.randn([4, 8], dtype="float32")
        x.stop_gradient = False
        y = ReduceScatterGroupOp.apply(x, g)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(paddle.allclose(y, x).item())

    def test_backward_single_rank_returns_clone(self):
        g = _single_rank_group()
        x = paddle.randn([4, 8], dtype="float32")
        x.stop_gradient = False
        y = ReduceScatterGroupOp.apply(x, g)
        y.sum().backward()
        # On a single-rank group AllGather of grad-of-ones is just clone:
        # gradient should be ones with same shape as x.
        self.assertEqual(x.grad.shape, x.shape)
        self.assertTrue(
            paddle.allclose(x.grad, paddle.ones_like(x.grad)).item()
        )


class TestRouterAllGather(unittest.TestCase):
    """``_RouterAllGather`` is the router-local AllGather: forward
    concatenates across EP, backward *slices* (no reduction). The
    single-rank path returns clone on forward and reshapes-only on
    backward.
    """

    def test_group_none_forward_returns_clone(self):
        x = paddle.randn([6, 4], dtype="float32")
        x.stop_gradient = False
        y = _RouterAllGather.apply(x, None)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(paddle.allclose(y, x).item())

    def test_group_none_backward_passes_through(self):
        x = paddle.randn([6, 4], dtype="float32")
        x.stop_gradient = False
        y = _RouterAllGather.apply(x, None)
        y.sum().backward()
        self.assertEqual(x.grad.shape, x.shape)

    def test_group_none_backward_reshapes_1d_grad(self):
        """Sonic-moe's ``_DownProjection`` may flatten ``topk_scores``
        gradient to 1-D ``[T*K]``; ``_RouterAllGather.backward`` must
        restore the original 2-D shape so upstream broadcast checks
        pass. Simulate this by manually calling backward via a chain
        that flattens.
        """
        x = paddle.randn([6, 4], dtype="float32")
        x.stop_gradient = False
        y = _RouterAllGather.apply(x, None)
        # Reshape consumer to a 1-D view, force grad to flow as 1-D.
        z = y.reshape([-1])
        z.sum().backward()
        self.assertEqual(list(x.grad.shape), [6, 4])

    def test_single_rank_group_forward(self):
        g = _single_rank_group()
        x = paddle.randn([5, 3], dtype="float32")
        x.stop_gradient = False
        y = _RouterAllGather.apply(x, g)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(paddle.allclose(y, x).item())


class TestRouterAllGatherMultiRank(unittest.TestCase):
    """Cover the nranks > 1 forward / backward paths of
    ``_RouterAllGather`` by mocking ``paddle.distributed.stream.all_gather``
    and feeding a fake group. No real NCCL traffic; verifies shape
    arithmetic + per-rank slice semantics (no cross-rank reduction).
    """

    def test_multi_rank_forward_concatenates(self):
        x = paddle.randn([3, 5], dtype="float32")
        x.stop_gradient = False
        fake_group = SimpleNamespace(nranks=2, rank=0)

        def fake_ag(out, inp, **kw):
            stacked = paddle.concat([inp, inp + 100.0], axis=0)
            paddle.assign(stacked, out)

        with mock.patch(
            "paddle.distributed.stream.all_gather", side_effect=fake_ag
        ):
            y = _RouterAllGather.apply(x, fake_group)
        self.assertEqual(list(y.shape), [6, 5])
        self.assertTrue(paddle.allclose(y[:3], x).item())
        self.assertTrue(paddle.allclose(y[3:], x + 100.0).item())

    def test_multi_rank_backward_slices_rank_segment(self):
        x = paddle.randn([3, 5], dtype="float32")
        x.stop_gradient = False
        fake_group = SimpleNamespace(nranks=2, rank=0)

        def fake_ag(out, inp, **kw):
            paddle.assign(paddle.concat([inp, inp], axis=0), out)

        with mock.patch(
            "paddle.distributed.stream.all_gather", side_effect=fake_ag
        ):
            y = _RouterAllGather.apply(x, fake_group)
        # Distinguishable per-rank grad: chunk0 = 1.0, chunk1 = 2.0.
        grad = paddle.concat(
            [paddle.full([3, 5], 1.0), paddle.full([3, 5], 2.0)], axis=0
        )
        y.backward(grad)
        # rank=0 must receive only its own chunk (1.0), no reduction.
        self.assertTrue(
            paddle.allclose(x.grad, paddle.full_like(x, 1.0)).item()
        )

    def test_multi_rank_backward_reshapes_1d_grad(self):
        """If the upstream grad arrives as 1-D, backward must reshape
        it to global shape before splitting."""
        x = paddle.randn([3, 4], dtype="float32")
        x.stop_gradient = False
        fake_group = SimpleNamespace(nranks=2, rank=1)

        def fake_ag(out, inp, **kw):
            paddle.assign(paddle.concat([inp, inp], axis=0), out)

        with mock.patch(
            "paddle.distributed.stream.all_gather", side_effect=fake_ag
        ):
            y = _RouterAllGather.apply(x, fake_group)
        # Reshape consumer to a 1-D view to force grad to flow as 1-D.
        z = y.reshape([-1])
        flat_grad = paddle.concat(
            [paddle.full([12], 1.0), paddle.full([12], 2.0)]
        )
        z.backward(flat_grad)
        # rank=1 must receive the second chunk (2.0).
        self.assertTrue(
            paddle.allclose(x.grad, paddle.full_like(x, 2.0)).item()
        )


class TestAllGatherCombineAsync(unittest.TestCase):
    """``_AllGatherCombineAsync`` fuses the ReduceScatter combine with a
    captured ``fn`` graph. With ``group=None`` the collective collapses
    to a clone but ``fn`` still runs and its grad must propagate back.
    """

    def test_group_none_runs_fn_and_propagates_gradients(self):
        x = paddle.randn([2, 8], dtype="float32")
        x.stop_gradient = False
        a = paddle.randn([2, 8], dtype="float32")
        a.stop_gradient = False

        def fn(t):
            return (t * 3.0,)

        combined, fn_out = _AllGatherCombineAsync.apply(
            x, None, a, fn=fn, is_first_fwd=False
        )
        # Forward: combined ~ x, fn_out ~ 3*a
        self.assertEqual(combined.shape, x.shape)
        self.assertEqual(fn_out.shape, a.shape)
        self.assertTrue(paddle.allclose(fn_out, a * 3.0).item())

        # Backward through both outputs.
        loss = combined.sum() + fn_out.sum()
        loss.backward()
        # x grad = ones (combine is clone with sum loss).
        self.assertTrue(paddle.allclose(x.grad, paddle.ones_like(x)).item())
        # a grad = 3 (sum loss * d(3a)/da).
        self.assertTrue(
            paddle.allclose(a.grad, paddle.full_like(a, 3.0)).item()
        )

    def test_group_none_first_fwd_backward_returns_zeros(self):
        """is_first_fwd=True sets ctx.bwf=None; backward must not crash
        and should return zero grads for fn_args (no real backward graph)."""
        x = paddle.randn([2, 8], dtype="float32")
        x.stop_gradient = False
        a = paddle.randn([2, 8], dtype="float32")
        a.stop_gradient = False

        def fn(t):
            return (t * 3.0,)

        combined, fn_out = _AllGatherCombineAsync.apply(
            x, None, a, fn=fn, is_first_fwd=True
        )
        # Forward still produces the same outputs.
        self.assertTrue(paddle.allclose(fn_out, a * 3.0).item())

        loss = combined.sum() + fn_out.sum()
        loss.backward()
        # x grad = ones (combined is clone).
        self.assertTrue(paddle.allclose(x.grad, paddle.ones_like(x)).item())
        # a grad = zeros (bwf is None, so fn backward is skipped).
        self.assertTrue(paddle.allclose(a.grad, paddle.zeros_like(a)).item())


class TestPreAllGatherResult(unittest.TestCase):
    """``_PreAllGatherResult`` consumes a pre-issued async AllGather
    handle. We build a fake handle for the single-rank case where the
    AllGather "result" is just a copy of the input.
    """

    def test_consumes_fake_handle(self):
        g = _single_rank_group()
        x = paddle.randn([3, 5], dtype="float32")
        x.stop_gradient = False
        # Build a fake handle: dummy task with .wait(), output==input.
        out_buf = x.clone().detach()

        class _DummyTask:
            def wait(self):
                return None

        handle = {"output": out_buf, "task": _DummyTask(), "group": g}
        y = _PreAllGatherResult.apply(x, handle)
        self.assertTrue(paddle.allclose(y, x).item())
        # Backward must call ReduceScatterGroupOp on the single-rank group
        # → clone path, so x.grad = ones.
        y.sum().backward()
        self.assertTrue(paddle.allclose(x.grad, paddle.ones_like(x)).item())


class TestAllGatherTokenDispatcherSingleRank(unittest.TestCase):
    """Constructor + early-return paths of ``AllGatherTokenDispatcher``."""

    def test_pre_allgather_single_rank_no_handle(self):
        g = _single_rank_group()
        d = AllGatherTokenDispatcher(
            moe_group=g,
            expert_model_parallel_size=1,
            num_experts=8,
        )
        d.pre_allgather(paddle.randn([4, 16]))
        self.assertIsNone(d._pre_ag_handle)

    def test_pre_allgather_with_fp8_dispatch_no_handle(self):
        g = _single_rank_group()
        # fp8_dispatch was removed from AllGatherTokenDispatcher; all
        # paths use the plain AllGather path. This test verifies that
        # pre_allgather works correctly regardless.
        d = AllGatherTokenDispatcher(
            moe_group=g,
            expert_model_parallel_size=2,
            num_experts=8,
        )
        d.pre_allgather(paddle.randn([4, 16]))
        # group.nranks == 1: pre_allgather sets handle to None.
        self.assertIsNone(d._pre_ag_handle)

    def test_token_combine_no_overlap_passthrough(self):
        g = _single_rank_group()
        d = AllGatherTokenDispatcher(
            moe_group=g,
            expert_model_parallel_size=1,
            num_experts=8,
        )
        x = paddle.randn([4, 16], dtype="float32")
        out = d.token_combine(x, combine_overlap_handle=None)
        # Pure passthrough: the ReduceScatter happens later in
        # combine_postprocess, not here.
        self.assertTrue(paddle.equal_all(out, x).item())
        self.assertIsNone(d._overlap_combined)

    def test_token_dispatch_passthrough(self):
        g = _single_rank_group()
        d = AllGatherTokenDispatcher(
            moe_group=g,
            expert_model_parallel_size=1,
            num_experts=8,
        )
        x = paddle.randn([6, 16], dtype="float32")
        out, extra = d.token_dispatch(x)
        self.assertTrue(paddle.equal_all(out, x).item())
        self.assertIsNone(extra)

    def test_dispatch_postprocess_returns_tokens_per_expert(self):
        g = _single_rank_group()
        d = AllGatherTokenDispatcher(
            moe_group=g,
            expert_model_parallel_size=1,
            num_experts=8,
        )
        d.tokens_per_expert = None
        x = paddle.randn([4, 16], dtype="float32")
        out, tpe = d.dispatch_postprocess(x)
        self.assertTrue(paddle.equal_all(out, x).item())
        self.assertIsNone(tpe)

    def test_combine_preprocess_passthrough(self):
        g = _single_rank_group()
        d = AllGatherTokenDispatcher(
            moe_group=g,
            expert_model_parallel_size=1,
            num_experts=8,
        )
        x = paddle.randn([4, 16], dtype="float32")
        out = d.combine_preprocess(x)
        self.assertIs(out, x)

    def test_combine_postprocess_overlap_cache_path(self):
        """When ``token_combine`` cached the result via the overlap path,
        ``combine_postprocess`` returns the cached output and clears the
        cache.
        """
        g = _single_rank_group()
        d = AllGatherTokenDispatcher(
            moe_group=g,
            expert_model_parallel_size=1,
            num_experts=8,
        )
        cached = paddle.randn([4, 16], dtype="float32")
        d._overlap_combined = cached
        out = d.combine_postprocess(paddle.zeros([4, 16]))
        self.assertIs(out, cached)
        self.assertIsNone(d._overlap_combined)

    def test_constructor_records_state(self):
        g = _single_rank_group()
        d = AllGatherTokenDispatcher(
            moe_group=g,
            expert_model_parallel_size=4,
            num_experts=32,
        )
        self.assertIs(d.moe_group, g)
        self.assertEqual(d.ep_size, 4)
        self.assertEqual(d.num_experts, 32)
        # In allgather mode every rank holds every expert.
        self.assertEqual(d.num_local_experts, 32)
        self.assertIsNone(d._pre_ag_handle)


class _FakeTask:
    def wait(self):
        pass


def _fake_all_gather(output, input, **kw):
    stacked = paddle.concat([input, input], axis=0)
    paddle.assign(stacked, output)
    return _FakeTask()


def _fake_reduce_scatter(output, input, **kw):
    local_T = input.shape[0] // 2
    paddle.assign(input[:local_T], output)
    return _FakeTask()


class TestAllGatherCombineAsyncMultiRank(unittest.TestCase):
    """Cover the nranks > 1 forward / backward paths of
    ``_AllGatherCombineAsync`` by mocking the collective primitives.
    """

    def test_multi_rank_forward_reduce_scatters(self):
        x = paddle.randn([4, 8], dtype="float32")
        x.stop_gradient = False
        a = paddle.randn([4, 8], dtype="float32")
        a.stop_gradient = False
        fake_group = SimpleNamespace(nranks=2, rank=0)

        def fn(t):
            return (t * 3.0,)

        with mock.patch(
            "paddle.distributed.stream.reduce_scatter",
            side_effect=_fake_reduce_scatter,
        ):
            combined, fn_out = _AllGatherCombineAsync.apply(
                x, fake_group, a, fn=fn, is_first_fwd=False
            )

        # reduce_scatter halves the token dim.
        self.assertEqual(list(combined.shape), [2, 8])
        self.assertTrue(paddle.allclose(combined, x[:2]).item())
        self.assertTrue(paddle.allclose(fn_out, a * 3.0).item())

    def test_multi_rank_backward_all_gathers(self):
        x = paddle.randn([4, 8], dtype="float32")
        x.stop_gradient = False
        a = paddle.randn([4, 8], dtype="float32")
        a.stop_gradient = False
        fake_group = SimpleNamespace(nranks=2, rank=0)

        def fn(t):
            return (t * 3.0,)

        with mock.patch(
            "paddle.distributed.stream.reduce_scatter",
            side_effect=_fake_reduce_scatter,
        ):
            combined, fn_out = _AllGatherCombineAsync.apply(
                x, fake_group, a, fn=fn, is_first_fwd=False
            )

        loss = combined.sum() + fn_out.sum()
        with (
            mock.patch(
                "paddle.distributed.stream.all_gather",
                side_effect=_fake_all_gather,
            ),
            mock.patch(
                "paddle.distributed.stream.reduce_scatter",
                side_effect=_fake_reduce_scatter,
            ),
        ):
            loss.backward()

        # grad of reduce_scatter is all_gather: shape restored to [4, 8].
        self.assertEqual(list(x.grad.shape), [4, 8])
        self.assertTrue(
            paddle.allclose(a.grad, paddle.full_like(a, 3.0)).item()
        )


class TestPreAllGatherResultMultiRank(unittest.TestCase):
    """Cover the multi-rank backward of ``_PreAllGatherResult`` where
    ``ReduceScatterGroupOp`` invokes the real reduce-scatter primitive.
    """

    def test_multi_rank_forward_backward(self):
        fake_group = SimpleNamespace(nranks=2, rank=0)
        x = paddle.randn([4, 8], dtype="float32")
        x.stop_gradient = False

        out_buf = paddle.concat([x, x + 1.0], axis=0)

        class _DummyTask:
            def wait(self):
                pass

        handle = {"output": out_buf, "task": _DummyTask(), "group": fake_group}

        y = _PreAllGatherResult.apply(x, handle)
        self.assertEqual(list(y.shape), [8, 8])
        self.assertTrue(paddle.allclose(y[:4], x).item())
        self.assertTrue(paddle.allclose(y[4:], x + 1.0).item())

        with mock.patch(
            "paddle.distributed.stream.reduce_scatter",
            side_effect=_fake_reduce_scatter,
        ):
            y.sum().backward()

        self.assertEqual(list(x.grad.shape), [4, 8])
        self.assertTrue(paddle.allclose(x.grad, paddle.ones_like(x)).item())


class TestAllGatherTokenDispatcherMultiRank(unittest.TestCase):
    """Cover the multi-rank paths inside ``AllGatherTokenDispatcher`` that
    exercise async AllGather, pre-ag-handle cleanup, overlap combine, and
    the ReduceScatter combine-postprocess branch.
    """

    def test_pre_allgather_creates_async_handle(self):
        fake_group = SimpleNamespace(nranks=2, rank=0)
        d = AllGatherTokenDispatcher(
            moe_group=fake_group, expert_model_parallel_size=2, num_experts=8
        )
        x = paddle.randn([4, 16], dtype="float32")
        with mock.patch(
            "paddle.distributed.stream.all_gather", side_effect=_fake_all_gather
        ):
            d.pre_allgather(x)
        self.assertIsNotNone(d._pre_ag_handle)
        self.assertEqual(list(d._pre_ag_handle["output"].shape), [8, 16])

    def test_pre_allgather_cleans_leftover_handle(self):
        fake_group = SimpleNamespace(nranks=2, rank=0)
        d = AllGatherTokenDispatcher(
            moe_group=fake_group, expert_model_parallel_size=2, num_experts=8
        )

        class _DummyTask:
            def wait(self):
                pass

        d._pre_ag_handle = {
            "output": None,
            "task": _DummyTask(),
            "group": fake_group,
        }
        x = paddle.randn([4, 16], dtype="float32")
        with mock.patch(
            "paddle.distributed.stream.all_gather", side_effect=_fake_all_gather
        ):
            d.pre_allgather(x)
        self.assertIsNotNone(d._pre_ag_handle)
        # New handle should have been created after clearing the old one.
        self.assertEqual(list(d._pre_ag_handle["output"].shape), [8, 16])

    def test_pre_allgather_warns_on_leftover_wait_failure(self):
        fake_group = SimpleNamespace(nranks=2, rank=0)
        d = AllGatherTokenDispatcher(
            moe_group=fake_group, expert_model_parallel_size=2, num_experts=8
        )

        class _FailingTask:
            def wait(self):
                raise RuntimeError("boom")

        d._pre_ag_handle = {
            "output": None,
            "task": _FailingTask(),
            "group": fake_group,
        }
        x = paddle.randn([4, 16], dtype="float32")
        with (
            mock.patch(
                "paddle.distributed.stream.all_gather",
                side_effect=_fake_all_gather,
            ),
            mock.patch(
                "paddlefleet.transformer.moe.token_dispatcher.logger.warning"
            ) as mock_warn,
        ):
            d.pre_allgather(x)
        self.assertTrue(mock_warn.called)
        self.assertIsNotNone(d._pre_ag_handle)

    def test_dispatch_preprocess_uses_pre_ag_handle(self):
        fake_group = SimpleNamespace(nranks=2, rank=0)
        d = AllGatherTokenDispatcher(
            moe_group=fake_group, expert_model_parallel_size=2, num_experts=8
        )
        x = paddle.randn([4, 16], dtype="float32")
        probs = paddle.randn([4, 8])
        topk_weights = paddle.randn([4, 2])
        topk_indices = paddle.randint(0, 8, [4, 2])
        mask = paddle.ones([4, 8], dtype="bool")

        out_buf = paddle.concat(
            [x.reshape([-1, 16]), x.reshape([-1, 16])], axis=0
        )

        class _DummyTask:
            def wait(self):
                pass

        d._pre_ag_handle = {
            "output": out_buf,
            "task": _DummyTask(),
            "group": fake_group,
        }

        with (
            mock.patch("paddle.distributed.barrier"),
            mock.patch(
                "paddle.distributed.stream.all_gather",
                side_effect=_fake_all_gather,
            ),
        ):
            result = d.dispatch_preprocess(
                x, probs, mask, topk_weights, topk_indices
            )

        self.assertEqual(list(result.shape), [8, 16])
        self.assertIsNone(d._pre_ag_handle)

    def test_dispatch_preprocess_fallback_all_gather_group_op(self):
        fake_group = SimpleNamespace(nranks=2, rank=0)
        d = AllGatherTokenDispatcher(
            moe_group=fake_group, expert_model_parallel_size=2, num_experts=8
        )
        x = paddle.randn([4, 16], dtype="float32")
        probs = paddle.randn([4, 8])
        topk_weights = paddle.randn([4, 2])
        topk_indices = paddle.randint(0, 8, [4, 2])
        mask = paddle.ones([4, 8], dtype="bool")

        with (
            mock.patch("paddle.distributed.barrier"),
            mock.patch(
                "paddle.distributed.stream.all_gather",
                side_effect=_fake_all_gather,
            ),
        ):
            result = d.dispatch_preprocess(
                x, probs, mask, topk_weights, topk_indices
            )

        self.assertEqual(list(result.shape), [8, 16])

    def test_token_combine_overlap_path(self):
        fake_group = SimpleNamespace(nranks=2, rank=0)
        d = AllGatherTokenDispatcher(
            moe_group=fake_group, expert_model_parallel_size=2, num_experts=8
        )
        hidden = paddle.randn([4, 16], dtype="float32")

        def fn(t):
            return (t * 2.0,)

        handle = {"fn": fn, "fn_args": (hidden.clone(),)}

        with mock.patch(
            "paddle.distributed.stream.reduce_scatter",
            side_effect=_fake_reduce_scatter,
        ):
            combined = d.token_combine(hidden, combine_overlap_handle=handle)

        self.assertEqual(list(combined.shape), [2, 16])
        self.assertIn("fn_out", handle)
        self.assertIsNotNone(d._overlap_combined)

    def test_combine_postprocess_reduce_scatter_path(self):
        fake_group = SimpleNamespace(nranks=2, rank=0)
        d = AllGatherTokenDispatcher(
            moe_group=fake_group, expert_model_parallel_size=2, num_experts=8
        )
        hidden = paddle.randn([4, 16], dtype="float32")
        hidden.stop_gradient = False

        with mock.patch(
            "paddle.distributed.stream.reduce_scatter",
            side_effect=_fake_reduce_scatter,
        ):
            out = d.combine_postprocess(hidden)

        self.assertEqual(list(out.shape), [2, 16])


class TestGroupedMLPExpertShardedStateDict(unittest.TestCase):
    """Cover the intermediate-sharded branch of
    ``GroupedMLPExpert.sharded_state_dict``.

    The branch is hit when ``intermediate_size_per_partition`` differs
    from ``config.moe_intermediate_size`` (i.e. the AllGather dispatcher
    layout). Verifies:

    * ``shard_weight`` is invoked on the right axes (3 for w1, 1 for w2)
    * ``weight1`` is reshaped to ``[E, H, 2, I_local]`` to disentangle
      gate/up before EP-sharding the intermediate dim
    * ``weight2`` is passed through as ``[E, I_local, H]``
    * ``grouped_gemm_param`` is set on both sharded results
    * Validation errors are raised for missing ep_group, non-gated MLP,
      and dimension mismatches
    """

    def _make_self(
        self,
        E=4,
        H=8,
        I_full=16,
        ep_size=2,
        ep_group=None,
        gated_linear_unit=True,
    ):
        I_local = I_full // ep_size
        if ep_group is None:
            ep_group = SimpleNamespace(nranks=ep_size)
        weight1 = paddle.randn([E, H, 2 * I_local], dtype="float32")
        weight1.name = "w1_param"
        weight2 = paddle.randn([E, I_local, H], dtype="float32")
        weight2.name = "w2_param"
        config = SimpleNamespace(
            moe_intermediate_size=I_full,
            hidden_size=H,
            gated_linear_unit=gated_linear_unit,
            moe_token_dispatcher_type="allgather",
        )
        mock_self = SimpleNamespace(
            num_local_experts=E,
            intermediate_size_per_partition=I_local,
            ep_group=ep_group,
            weight1=weight1,
            weight2=weight2,
            config=config,
        )
        # Make .state_dict() return tensors with the parameter shapes;
        # sharded_state_dict only consumes weight1/weight2.
        mock_self.state_dict = lambda structured_name_prefix="": {
            "weight1": weight1.clone().detach(),
            "weight2": weight2.clone().detach(),
        }
        return mock_self, E, H, I_local

    def test_intermediate_sharded_branch_calls_shard_weight(self):
        mock_self, E, H, I_local = self._make_self()
        captured = []

        def fake_shard_weight(key, weight, axis, group):
            captured.append(
                {
                    "key": key,
                    "weight_shape": list(weight.shape),
                    "axis": axis,
                    "group": group,
                }
            )
            return weight  # a writable object so .grouped_gemm_param sticks

        with mock.patch(
            "paddlefleet.transformer.moe.moe_expert.shard_weight",
            side_effect=fake_shard_weight,
        ):
            out = GroupedMLPExpert.sharded_state_dict(
                mock_self, structured_name_prefix="prefix."
            )

        keys = sorted(out.keys())
        self.assertEqual(keys, ["prefix.weight1", "prefix.weight2"])
        # Both results must carry grouped_gemm_param=True.
        self.assertTrue(out["prefix.weight1"].grouped_gemm_param)
        self.assertTrue(out["prefix.weight2"].grouped_gemm_param)

        by_key = {c["key"]: c for c in captured}
        # weight1 sharded along axis=3 (I_local), reshape disentangles
        # gate/up onto axis=2.
        self.assertEqual(by_key["prefix.weight1"]["axis"], 3)
        self.assertEqual(
            by_key["prefix.weight1"]["weight_shape"], [E, H, 2, I_local]
        )
        # weight2 sharded along axis=1 (intermediate dim).
        self.assertEqual(by_key["prefix.weight2"]["axis"], 1)
        self.assertEqual(
            by_key["prefix.weight2"]["weight_shape"], [E, I_local, H]
        )
        # ep_group threaded through.
        self.assertIs(by_key["prefix.weight1"]["group"], mock_self.ep_group)
        self.assertIs(by_key["prefix.weight2"]["group"], mock_self.ep_group)

    def test_intermediate_sharded_requires_ep_group(self):
        mock_self, *_ = self._make_self(ep_group=False)
        mock_self.ep_group = None
        with self.assertRaisesRegex(ValueError, "EP group"):
            GroupedMLPExpert.sharded_state_dict(mock_self)

    def test_intermediate_sharded_requires_gated_linear_unit(self):
        mock_self, *_ = self._make_self(gated_linear_unit=False)
        with (
            self.assertRaisesRegex(ValueError, "gated"),
            mock.patch("paddlefleet.transformer.moe.moe_expert.shard_weight"),
        ):
            GroupedMLPExpert.sharded_state_dict(mock_self)

    def test_intermediate_sharded_rejects_size_mismatch(self):
        # ep_size * I_local != moe_intermediate_size
        mock_self, *_ = self._make_self(ep_size=2, I_full=16)
        # Force a wrong ep_size via the group nranks.
        mock_self.ep_group = SimpleNamespace(nranks=4)
        with (
            self.assertRaisesRegex(ValueError, "inconsistency"),
            mock.patch("paddlefleet.transformer.moe.moe_expert.shard_weight"),
        ):
            GroupedMLPExpert.sharded_state_dict(mock_self)

    def test_non_intermediate_sharded_skips_branch(self):
        """If intermediate_size_per_partition == moe_intermediate_size
        the new branch must be bypassed (else axis=3 would be wrong)."""
        # ep_size=1 keeps I_local == I_full, so is_intermediate_sharded=False.
        mock_self, *_ = self._make_self(ep_size=1, I_full=16)
        # Original path then enters model_type/ep_group path. We only
        # need to confirm shard_weight is NOT called on the 4-D w1.
        with (
            mock.patch(
                "paddlefleet.transformer.moe.moe_expert.shard_weight"
            ) as sw,
            mock.patch(
                "paddlefleet.transformer.moe.moe_expert.build_sharded_state_dict",
                return_value={},
            ),
        ):
            GroupedMLPExpert.sharded_state_dict(mock_self)
        # Either branch (model_type fall-through) might still call
        # shard_weight, but never with axis=3.
        for call in sw.call_args_list:
            self.assertNotEqual(call.kwargs.get("axis"), 3)

    def test_intermediate_sharded_rejects_bad_weight1_shape(self):
        """weight1 last dim must equal 2 * I_local."""
        mock_self, E, H, I_local = self._make_self()
        # Corrupt weight1 so its last dim is wrong.
        bad_w1 = paddle.randn([E, H, 2 * I_local + 1], dtype="float32")
        bad_w1.name = "w1_param"
        mock_self.weight1 = bad_w1
        mock_self.state_dict = lambda structured_name_prefix="": {
            "weight1": bad_w1.clone().detach(),
            "weight2": mock_self.weight2.clone().detach(),
        }
        with (
            self.assertRaisesRegex(ValueError, "weight1 last dim"),
            mock.patch("paddlefleet.transformer.moe.moe_expert.shard_weight"),
        ):
            GroupedMLPExpert.sharded_state_dict(mock_self)


class TestGroupedMLPExpertForwardEdgeCases(unittest.TestCase):
    """Cover edge-case branches in GroupedMLPExpert.forward."""

    def _make_expert(self, E=2, H=8, I=16):
        config = SimpleNamespace(
            hidden_size=H,
            moe_intermediate_size=I,
            gated_linear_unit=True,
            activation_func=paddle.nn.functional.gelu,
            init_method=lambda t: None,
            output_layer_init_method=lambda t: None,
            using_sonic_moe=False,
            use_bias=False,
            fp8=None,
            recompute_granularity=None,
            recompute_modules=None,
        )
        expert = GroupedMLPExpert(
            num_local_experts=E,
            config=config,
            moe_deep_gemm=False,
        )
        return expert

    def test_empty_tokens_with_nonzero_tokens_per_expert_raises(self):
        """RuntimeError when numel==0 but tokens_per_expert is not all-zero."""
        expert = self._make_expert()
        empty = paddle.randn([0, 8], dtype="float32")
        bad_tpe = paddle.to_tensor([1, 0], dtype="int32")
        with self.assertRaisesRegex(RuntimeError, "all-zero tokens_per_expert"):
            expert.forward(empty, bad_tpe)


class TestSonicMoEExpertValidation(unittest.TestCase):
    """Cover validation paths in SonicMoEExpert.__init__."""

    def test_gated_linear_unit_required(self):
        """SonicMoEExpert requires gated_linear_unit=True."""
        config = SimpleNamespace(
            hidden_size=8,
            moe_intermediate_size=16,
            gated_linear_unit=False,
            activation_func=paddle.nn.functional.gelu,
            init_method=lambda t: None,
            output_layer_init_method=lambda t: None,
        )
        with self.assertRaisesRegex(ValueError, "SwiGLU"):
            from paddlefleet.transformer.moe.moe_expert import SonicMoEExpert

            SonicMoEExpert(
                num_local_experts=2,
                topk=1,
                config=config,
            )


class TestRouterAllGatherBackwardReshape(unittest.TestCase):
    """Cover the defensive shape-restoration branches inside
    ``_RouterAllGather.backward`` by invoking the static method directly
    with a hand-built context object. Paddle's autograd reshapes the
    upstream gradient before calling ``PyLayer.backward``, so these
    branches are otherwise unreachable from a Python-level test.
    """

    def test_group_none_reshapes_grad_when_shape_mismatch(self):
        """When group is None and the grad arrives with a non-matching
        shape, backward must reshape it to the original input shape.
        """

        class _Ctx:
            group = None
            input_shape = [6, 4]

        # 1-D grad with same numel as input_shape.
        out = _RouterAllGather.backward(_Ctx(), paddle.ones([24]))
        self.assertEqual(list(out.shape), [6, 4])

    def test_multi_rank_reshapes_grad_to_global_shape(self):
        """When the upstream grad is 1-D, backward must reshape it to
        ``[T_global, *trailing]`` before splitting per rank.
        """

        class _Ctx:
            group = SimpleNamespace(nranks=2, rank=1)
            input_shape = [3, 4]

        flat_grad = paddle.concat(
            [paddle.full([12], 1.0), paddle.full([12], 2.0)]
        )
        out = _RouterAllGather.backward(_Ctx(), flat_grad)
        # rank=1 owns the second chunk (2.0).
        self.assertEqual(list(out.shape), [3, 4])
        self.assertTrue(paddle.allclose(out, paddle.full([3, 4], 2.0)).item())

    def test_multi_rank_reshapes_chunk_to_local_shape(self):
        """When the per-rank split chunk shape does not match
        ``local_shape`` (e.g. trailing dims are differently grouped),
        backward must reshape it.
        """

        class _Ctx:
            group = SimpleNamespace(nranks=2, rank=0)
            # local_shape with multi-D trailing dims.
            input_shape = [3, 2, 2]

        # Provide global-shaped grad as flat to also exercise the
        # global-shape reshape path; the chunk after split is [3, 2, 2]
        # so chunk-shape branch only triggers if trailing dims merge.
        # Build grad already in [6, 2, 2] then split chunk shape == local.
        # To force a chunk reshape, give grad shape [6, 4] (numel=24)
        # which matches global_shape [6, 2, 2] only after reshape; chunk
        # of grad [6,4] split axis=0 gives [3,4] != local_shape [3,2,2].
        grad = paddle.arange(24, dtype="float32").reshape([6, 4])
        out = _RouterAllGather.backward(_Ctx(), grad)
        self.assertEqual(list(out.shape), [3, 2, 2])


class TestAllGatherCombineAsyncErrors(unittest.TestCase):
    """Cover validation paths in ``_AllGatherCombineAsync.forward``."""

    def test_forward_rejects_non_divisible_token_dim(self):
        """token dim must be divisible by EP size."""
        x = paddle.randn([5, 8], dtype="float32")
        x.stop_gradient = False
        a = paddle.randn([5, 8], dtype="float32")
        a.stop_gradient = False
        fake_group = SimpleNamespace(nranks=2, rank=0)

        def fn(t):
            return (t,)

        with self.assertRaisesRegex(ValueError, "not divisible"):
            _AllGatherCombineAsync.apply(
                x, fake_group, a, fn=fn, is_first_fwd=False
            )


class TestAllGatherTokenDispatcher3DInputs(unittest.TestCase):
    """Cover the 3-D ``hidden_states`` branches of ``pre_allgather`` and
    ``dispatch_preprocess``.
    """

    def test_pre_allgather_3d_hidden_states(self):
        fake_group = SimpleNamespace(nranks=2, rank=0)
        d = AllGatherTokenDispatcher(
            moe_group=fake_group, expert_model_parallel_size=2, num_experts=8
        )
        x = paddle.randn([2, 3, 16], dtype="float32")
        with mock.patch(
            "paddle.distributed.stream.all_gather", side_effect=_fake_all_gather
        ):
            d.pre_allgather(x)
        self.assertIsNotNone(d._pre_ag_handle)

    def test_dispatch_preprocess_3d_hidden_states(self):
        fake_group = SimpleNamespace(nranks=2, rank=0)
        d = AllGatherTokenDispatcher(
            moe_group=fake_group, expert_model_parallel_size=2, num_experts=8
        )
        x = paddle.randn([2, 3, 16], dtype="float32")
        probs = paddle.randn([6, 8])
        topk_weights = paddle.randn([6, 2])
        topk_indices = paddle.randint(0, 8, [6, 2])
        mask = paddle.ones([6, 8], dtype="bool")
        with (
            mock.patch("paddle.distributed.barrier"),
            mock.patch(
                "paddle.distributed.stream.all_gather",
                side_effect=_fake_all_gather,
            ),
        ):
            result = d.dispatch_preprocess(
                x, probs, mask, topk_weights, topk_indices
            )
        # 6 (= 2*3) tokens locally, *2 ranks = 12 globally.
        self.assertEqual(list(result.shape), [12, 16])


class TestAllGatherTokenDispatcherValidation(unittest.TestCase):
    """Cover validation paths in ``AllGatherTokenDispatcher``."""

    def _make_dispatcher(self):
        fake_group = SimpleNamespace(nranks=2, rank=0)
        return AllGatherTokenDispatcher(
            moe_group=fake_group, expert_model_parallel_size=2, num_experts=8
        )

    def test_dispatch_preprocess_requires_topk_indices_and_weights(self):
        d = self._make_dispatcher()
        x = paddle.randn([4, 16], dtype="float32")
        probs = paddle.randn([4, 8])
        mask = paddle.ones([4, 8], dtype="bool")
        with (
            mock.patch("paddle.distributed.barrier"),
            mock.patch(
                "paddle.distributed.stream.all_gather",
                side_effect=_fake_all_gather,
            ),
            self.assertRaisesRegex(ValueError, "topk_indices"),
        ):
            d.dispatch_preprocess(x, probs, mask, None, None)

    def test_token_combine_rejects_non_dict_handle(self):
        d = self._make_dispatcher()
        x = paddle.randn([4, 16], dtype="float32")
        with self.assertRaisesRegex(TypeError, "must be a dict"):
            d.token_combine(x, combine_overlap_handle="not-a-dict")

    def test_token_combine_rejects_handle_missing_keys(self):
        d = self._make_dispatcher()
        x = paddle.randn([4, 16], dtype="float32")
        with self.assertRaisesRegex(ValueError, "'fn' and 'fn_args'"):
            d.token_combine(x, combine_overlap_handle={"fn": lambda t: (t,)})

    def test_token_combine_rejects_non_tuple_fn_args(self):
        d = self._make_dispatcher()
        x = paddle.randn([4, 16], dtype="float32")
        with self.assertRaisesRegex(TypeError, "must be a tuple"):
            d.token_combine(
                x,
                combine_overlap_handle={
                    "fn": lambda t: (t,),
                    "fn_args": [x],  # list, not tuple
                },
            )


class TestMoEFlexTokenDispatcherCombineGuard(unittest.TestCase):
    """Cover ``MoEFlexTokenDispatcher.token_combine`` overlap-handle
    rejection.
    """

    def test_token_combine_rejects_overlap_handle(self):
        # Bypass __init__ since it constructs a real comm manager.
        d = MoEFlexTokenDispatcher.__new__(MoEFlexTokenDispatcher)
        d._comm_manager = mock.MagicMock()
        x = paddle.randn([4, 16], dtype="float32")
        with self.assertRaisesRegex(ValueError, "does not support"):
            d.token_combine(x, combine_overlap_handle={"fn": lambda t: t})


class TestMoELayerDispatchPreprocessGuard(unittest.TestCase):
    """Cover ``MoELayer.dispatch_preprocess`` type-check on the
    token_dispatcher (rejects non-MoEFlex backends).
    """

    def test_rejects_non_moeflex_dispatcher(self):
        mock_self = SimpleNamespace(
            use_latent_moe=False,
            token_dispatcher=mock.MagicMock(spec=AllGatherTokenDispatcher),
        )
        x = paddle.randn([4, 16], dtype="float32")
        probs = paddle.randn([4, 8])
        indices = paddle.randint(0, 8, [4, 2])
        with self.assertRaisesRegex(TypeError, "MoEFlexTokenDispatcher"):
            MoELayer.dispatch_preprocess(mock_self, (x, probs, indices))


class TestMoELayerForwardPreAllGatherBranches(unittest.TestCase):
    """Cover the ``forward()`` pre_allgather gate-overlap branches:

    * ``use_latent_moe=True``: project to latent space and pre_allgather it.
    * ``use_latent_moe=False``: pre_allgather raw hidden_states.

    To keep the test single-card and self-contained we stub ``self.gate``
    to raise a sentinel so the rest of forward is short-circuited; we only
    verify the pre_allgather call.
    """

    class _Sentinel(Exception):
        pass

    def _make_self(self, use_latent_moe, fc1_out=None):
        latent_proj = mock.MagicMock(return_value=fc1_out)
        dispatcher = mock.MagicMock(spec=AllGatherTokenDispatcher)
        config = SimpleNamespace(moe_allgather_gate_overlap=True)

        def gate_stub(*args, **kwargs):
            raise self._Sentinel

        return SimpleNamespace(
            expert_model_parallel_size=2,
            sequence_parallel=False,
            moe_token_dispatcher_type="allgather",
            config=config,
            use_latent_moe=use_latent_moe,
            fc1_latent_proj=latent_proj,
            token_dispatcher=dispatcher,
            gate=gate_stub,
            layer_number=0,
        )

    def test_pre_allgather_with_latent_moe(self):
        latent_out = paddle.randn([4, 16], dtype="float32")
        mock_self = self._make_self(use_latent_moe=True, fc1_out=latent_out)
        x = paddle.randn([4, 8], dtype="float32")
        with self.assertRaises(self._Sentinel):
            MoELayer.forward(mock_self, x)
        # fc1_latent_proj projection cached; pre_allgather called on it.
        self.assertIs(mock_self._latent_hidden, latent_out)
        mock_self.token_dispatcher.pre_allgather.assert_called_once_with(
            latent_out
        )

    def test_pre_allgather_without_latent_moe(self):
        mock_self = self._make_self(use_latent_moe=False)
        x = paddle.randn([4, 8], dtype="float32")
        with self.assertRaises(self._Sentinel):
            MoELayer.forward(mock_self, x)
        self.assertIsNone(mock_self._latent_hidden)
        mock_self.token_dispatcher.pre_allgather.assert_called_once_with(x)


class TestMoELayerCustomForwardBranches(unittest.TestCase):
    """Cover branches inside ``MoELayer.custom_forward``:

    * latent_moe with a cached ``_latent_hidden`` (the AllGather-overlap
      path stashes the projected latent tensor in ``forward()`` so
      ``custom_forward`` must consume it instead of re-projecting).
    * ``log_moe_balance`` allgather branch (re-derives per-rank tokens
      from ``_global_topk_indices`` since AllGather has no comm_manager).
    * ``log_moe_balance`` non-allgather branch (reads from comm_manager).
    """

    def _make_self(self, dispatcher_type, use_latent_moe, latent_cached=None):
        dispatcher = mock.MagicMock()
        if dispatcher_type == "allgather":
            dispatcher._global_topk_indices = paddle.to_tensor(
                [[0, 1], [2, 3], [0, 2], [1, 3]], dtype="int64"
            )
        else:
            dispatcher._comm_manager.tokens_per_expert = paddle.to_tensor(
                [1, 2, 3, 4], dtype="int64"
            )
        # dispatch returns a (hidden_states, _) tuple.
        token_seq = paddle.randn([4, 8], dtype="float32")
        return SimpleNamespace(
            use_latent_moe=use_latent_moe,
            _latent_hidden=latent_cached,
            fc1_latent_proj=mock.MagicMock(side_effect=lambda x: x * 2),
            fc2_latent_proj=mock.MagicMock(side_effect=lambda x: x + 1),
            dispatch=mock.MagicMock(return_value=(token_seq, None)),
            routed_experts_compute=mock.MagicMock(return_value=token_seq),
            combine=mock.MagicMock(return_value=token_seq),
            moe_token_dispatcher_type=dispatcher_type,
            token_dispatcher=dispatcher,
            num_experts=4,
            expert_model_parallel_size=2,
            num_experts_per_tok=2,
            moe_group=SimpleNamespace(rank=0),
            moe_rank=0,
            layer_number=0,
        )

    def _patch_balance_enabled(self, enabled):
        return mock.patch(
            "paddlefleet.transformer.moe.moe_layer.global_moe_balance_training_logs_enabled",
            return_value=enabled,
        )

    def _patch_has_grad(self, has_grad):
        tracer = SimpleNamespace(_has_grad=has_grad)
        return mock.patch(
            "paddlefleet.transformer.moe.moe_layer.framework._dygraph_tracer",
            return_value=tracer,
        )

    def test_latent_moe_with_cached_hidden_consumes_cache(self):
        cached = paddle.randn([4, 8], dtype="float32")
        mock_self = self._make_self(
            dispatcher_type="alltoall",
            use_latent_moe=True,
            latent_cached=cached,
        )
        x = paddle.randn([4, 8], dtype="float32")
        with self._patch_has_grad(False):
            out = MoELayer.custom_forward(
                mock_self,
                x,
                probs=paddle.randn([4, 4]),
                routing_map=paddle.ones([4, 4], dtype="bool"),
            )
        # fc1_latent_proj must NOT be called — cached path takes precedence.
        mock_self.fc1_latent_proj.assert_not_called()
        # cache cleared.
        self.assertIsNone(mock_self._latent_hidden)
        # fc2 projection applied to combined output.
        mock_self.fc2_latent_proj.assert_called_once()
        self.assertEqual(out.shape, [4, 8])

    def test_log_balance_allgather_branch(self):
        mock_self = self._make_self(
            dispatcher_type="allgather", use_latent_moe=False
        )
        x = paddle.randn([4, 8], dtype="float32")
        with (
            self._patch_has_grad(True),
            self._patch_balance_enabled(True),
            mock.patch(
                "paddlefleet.transformer.moe.moe_layer.log_moe_balance"
            ) as log_balance,
        ):
            MoELayer.custom_forward(
                mock_self,
                x,
                probs=paddle.randn([4, 4]),
                routing_map=paddle.ones([4, 4], dtype="bool"),
            )
        log_balance.assert_called_once()
        # tokens_per_expert was derived per-rank ([ep_size, num_experts/ep])
        # → has shape [num_experts/ep_size] = [2].
        _, _, _, tpe = log_balance.call_args[0]
        self.assertEqual(list(tpe.shape), [2])

    def test_log_balance_non_allgather_branch(self):
        mock_self = self._make_self(
            dispatcher_type="alltoall", use_latent_moe=False
        )
        x = paddle.randn([4, 8], dtype="float32")
        with (
            self._patch_has_grad(True),
            self._patch_balance_enabled(True),
            mock.patch(
                "paddlefleet.transformer.moe.moe_layer.log_moe_balance"
            ) as log_balance,
        ):
            MoELayer.custom_forward(
                mock_self,
                x,
                probs=paddle.randn([4, 4]),
                routing_map=paddle.ones([4, 4], dtype="bool"),
            )
        log_balance.assert_called_once()
        # tokens_per_expert came from comm_manager.tokens_per_expert.
        _, _, _, tpe = log_balance.call_args[0]
        self.assertTrue(
            paddle.allclose(
                tpe, paddle.to_tensor([1, 2, 3, 4], dtype="int64")
            ).item()
        )


class TestMoELayerFusionMoEForwardBranches(unittest.TestCase):
    """Cover branches inside ``MoELayer.fusion_moe_forward``:

    * AllGather dispatcher: token_combine + combine_postprocess split.
    * AllGather log_balance: re-derive tokens_per_expert from
      ``_global_topk_indices`` (no comm_manager).
    * Non-allgather log_balance: read from comm_manager.
    """

    def _make_self(self, dispatcher_type, has_balance_log):
        dispatcher = mock.MagicMock()
        if dispatcher_type == "allgather":
            dispatcher._global_topk_indices = paddle.to_tensor(
                [[0, 1], [2, 3], [0, 2], [1, 3]], dtype="int64"
            )
            dispatcher._global_topk_weights = paddle.randn([4, 2])
            dispatcher.token_combine = mock.MagicMock(
                return_value=paddle.randn([4, 8])
            )
            dispatcher.combine_postprocess = mock.MagicMock(
                return_value=paddle.randn([4, 8])
            )
        else:
            dispatcher._comm_manager.dispatched_indices = paddle.to_tensor(
                [[0, 1], [2, 3]], dtype="int64"
            )
            dispatcher._comm_manager.dispatched_probs = paddle.randn([2, 2])
            dispatcher._comm_manager.tokens_per_expert = paddle.to_tensor(
                [1, 2, 3, 4], dtype="int64"
            )
            dispatcher._comm_manager.combine = mock.MagicMock(
                return_value=paddle.randn([4, 8])
            )

        token_seq = paddle.randn([4, 8], dtype="float32")
        # Callable mock for grouped_gemm_experts (sonic-moe expert).
        grouped = mock.MagicMock(return_value=token_seq)
        return SimpleNamespace(
            use_latent_moe=False,
            _latent_hidden=None,
            fc1_latent_proj=mock.MagicMock(side_effect=lambda x: x),
            fc2_latent_proj=mock.MagicMock(side_effect=lambda x: x),
            dispatch=mock.MagicMock(return_value=(token_seq, None)),
            _use_hybrid_ep_fusion=mock.MagicMock(return_value=False),
            using_sonic_moe=True,
            grouped_gemm_experts=grouped,
            fp8=None,
            moe_token_dispatcher_type=dispatcher_type,
            token_dispatcher=dispatcher,
            num_experts=4,
            expert_model_parallel_size=2,
            num_experts_per_tok=2,
            moe_group=SimpleNamespace(rank=0),
            moe_rank=0,
            layer_number=0,
            use_rr_deepep_combine=False,
            config=SimpleNamespace(activation_func_clamp_value=0.0),
        )

    def _patch_balance(self, enabled):
        return mock.patch(
            "paddlefleet.transformer.moe.moe_layer.global_moe_balance_training_logs_enabled",
            return_value=enabled,
        )

    def _patch_has_grad(self, has_grad):
        tracer = SimpleNamespace(_has_grad=has_grad)
        return mock.patch(
            "paddlefleet.transformer.moe.moe_layer.framework._dygraph_tracer",
            return_value=tracer,
        )

    def _run_fusion_forward(self, mock_self):
        x = paddle.randn([4, 8], dtype="float32")
        return MoELayer.fusion_moe_forward(
            mock_self,
            x,
            probs=paddle.randn([4, 4]),
            routing_map=paddle.ones([4, 4], dtype="bool"),
            combine_overlap_handle=None,
            topk_weights=paddle.randn([4, 2]),
            topk_indices=paddle.randint(0, 4, [4, 2]),
        )

    def test_allgather_combine_path(self):
        mock_self = self._make_self(
            dispatcher_type="allgather", has_balance_log=False
        )
        with (
            self._patch_has_grad(False),
            self._patch_balance(False),
        ):
            self._run_fusion_forward(mock_self)
        mock_self.token_dispatcher.token_combine.assert_called_once()
        mock_self.token_dispatcher.combine_postprocess.assert_called_once()

    def test_allgather_log_balance(self):
        mock_self = self._make_self(
            dispatcher_type="allgather", has_balance_log=True
        )
        with (
            self._patch_has_grad(True),
            self._patch_balance(True),
            mock.patch(
                "paddlefleet.transformer.moe.moe_layer.log_moe_balance"
            ) as log_balance,
        ):
            self._run_fusion_forward(mock_self)
        log_balance.assert_called_once()
        _, _, _, tpe = log_balance.call_args[0]
        # ep_size=2, num_experts=4 → per-rank tpe has shape [2].
        self.assertEqual(list(tpe.shape), [2])

    def test_non_allgather_log_balance(self):
        mock_self = self._make_self(
            dispatcher_type="alltoall", has_balance_log=True
        )
        with (
            self._patch_has_grad(True),
            self._patch_balance(True),
            mock.patch(
                "paddlefleet.transformer.moe.moe_layer.log_moe_balance"
            ) as log_balance,
        ):
            self._run_fusion_forward(mock_self)
        log_balance.assert_called_once()
        _, _, _, tpe = log_balance.call_args[0]
        self.assertTrue(
            paddle.allclose(
                tpe, paddle.to_tensor([1, 2, 3, 4], dtype="int64")
            ).item()
        )


if __name__ == "__main__":
    unittest.main()
