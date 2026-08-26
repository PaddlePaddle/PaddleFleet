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
"""Multi-card (EP>1) tests for RingMoETokenDispatcher.

Covers both ring topologies on two cards: G=2/N=1 (intra only) and G=1/N=2
(inter only), selected through NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN.

Run with:
  python -m paddle.distributed.launch --gpus=0,1 \
      tests/multi_card_tests/moe/test_ring_dispatcher_ep.py
"""

import os
import random
import unittest
from unittest import mock

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet

from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddlefleet.training.initialize import initialize_fleet
from paddlefleet.transformer.moe import moe_layer

_fleet_initialised = False
_pg_collection = None


def _ensure_fleet():
    global _fleet_initialised, _pg_collection
    if _fleet_initialised:
        return _pg_collection
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": 1,
        "sharding_degree": 2,
        "sep_degree": 1,
        "cp_degree": 1,
        "ep_degree": 2,
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
    initialize_fleet(strategy=strategy)
    _pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    _fleet_initialised = True
    return _pg_collection


def _make_dispatcher(gpus_per_node, ep_group, num_experts=4):
    """Build a dispatcher with the ring topology forced to a known G.

    ``_detect_gpus_per_node`` reads the env at construction time, so this is how
    a two-card job gets to exercise both G=2/N=1 and G=1/N=2. Sub-group creation
    is collective and cached per (ep_ranks, G), so every rank must call this in
    the same order -- which unittest guarantees within a single test.
    """
    from paddlefleet.transformer.moe.token_dispatcher import (
        RingMoETokenDispatcher,
    )

    prev = os.environ.get("NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN")
    os.environ["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] = str(gpus_per_node)
    try:
        return RingMoETokenDispatcher(
            ep_group, ep_group.nranks, num_experts=num_experts
        )
    finally:
        if prev is None:
            os.environ.pop("NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN", None)
        else:
            os.environ["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] = prev


def _scale_expert_fn(scale):
    """Stand-in for the SonicMoE grouped GEMM with the real call signature.

    Linear in the tokens and independent of the routing, so a test can predict
    the ring's output and gradients without pulling in the fused kernels.
    """

    def expert_fn(
        g_tok,
        g_idx,
        g_w,
        use_fp8,
        tokens_per_expert=None,
        fp8_scale=None,
        recompute_moe_gate_up=False,
        fp8_combine_grad_handle=None,
    ):
        return g_tok * scale

    return expert_fn


class _RingTestBase(unittest.TestCase):
    """Initialises fleet once, seeds every test, exposes the EP group."""

    @classmethod
    def setUpClass(cls):
        cls.pg_collection = _ensure_fleet()

    def setUp(self):
        self.seed = 42
        random.seed(self.seed)
        np.random.seed(self.seed)
        paddle.seed(self.seed)
        model_parallel_cuda_manual_seed(self.seed)
        self.ep_group = self.__class__.pg_collection.ep
        self.ep_size = self.ep_group.nranks
        self.rank = dist.get_rank(self.ep_group)
        self.num_experts = 4
        self.d_latent = 8
        self.T_local = 4

    def _tokens(self, rows=None, requires_grad=True):
        x = paddle.randn([rows or self.T_local, self.d_latent])
        x = x * (self.rank + 1)
        x.stop_gradient = not requires_grad
        return x

    def _routing(self, rows=None):
        rows = rows or self.T_local
        idx = paddle.to_tensor(
            [
                [i % self.num_experts, (i + 1) % self.num_experts]
                for i in range(rows)
            ],
            dtype="int32",
        )
        w = paddle.randn([rows, 2]).abs()
        return idx, w / w.sum(axis=1, keepdim=True)


class TestRingTopology(_RingTestBase):
    def test_detect_gpus_per_node_env_priority(self):
        from paddlefleet.transformer.moe import token_dispatcher as td

        keys = (
            "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN",
            "PADDLE_LOCAL_SIZE",
            "CUDA_VISIBLE_DEVICES",
        )
        saved = {k: os.environ.get(k) for k in keys}
        try:
            for k in keys:
                os.environ.pop(k, None)
            self.assertEqual(td._detect_gpus_per_node(), 1)
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
            self.assertEqual(td._detect_gpus_per_node(), 3)
            os.environ["PADDLE_LOCAL_SIZE"] = "4"
            self.assertEqual(td._detect_gpus_per_node(), 4)
            os.environ["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] = "8"
            self.assertEqual(td._detect_gpus_per_node(), 8)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_intra_only_topology(self):
        disp = _make_dispatcher(2, self.ep_group)
        self.assertEqual((disp.G, disp.N), (2, 1))
        self.assertIsNotNone(disp.intra_group)
        self.assertIsNone(disp.inter_group)

    def test_inter_only_topology(self):
        disp = _make_dispatcher(1, self.ep_group)
        self.assertEqual((disp.G, disp.N), (1, 2))
        self.assertIsNone(disp.intra_group)
        self.assertIsNotNone(disp.inter_group)

    def test_subgroups_are_cached(self):
        a = _make_dispatcher(2, self.ep_group)
        b = _make_dispatcher(2, self.ep_group)
        self.assertIs(a.intra_group, b.intra_group)

    def test_ep_not_divisible_by_gpus_per_node_raises(self):
        from paddlefleet.transformer.moe import token_dispatcher as td

        class _FakeGroup:
            ranks = [0, 1, 2]

        with self.assertRaises(ValueError):
            td._build_ring_subgroups(_FakeGroup(), 2)

    def test_no_ep_group_collapses_to_passthrough(self):
        from paddlefleet.transformer.moe.token_dispatcher import (
            RingMoETokenDispatcher,
        )

        # EP==1: no sub-groups to build, and the ring degenerates to the
        # expert GEMM alone. Not collective, so it needs no rank agreement.
        disp = RingMoETokenDispatcher(None, 1, num_experts=self.num_experts)
        self.assertEqual((disp.G, disp.N), (1, 1))
        self.assertIsNone(disp.intra_group)
        self.assertIsNone(disp.inter_group)
        x = self._tokens()
        idx, w = self._routing()
        out = disp.ring_forward(x, w, idx, _scale_expert_fn(2.0), w.dtype)
        np.testing.assert_allclose(
            out.numpy(), x.numpy() * 2.0, rtol=1e-6, atol=1e-6
        )

    def test_fp8_dispatch_not_implemented(self):
        from paddlefleet.transformer.moe.token_dispatcher import (
            RingMoETokenDispatcher,
        )

        with self.assertRaises(NotImplementedError):
            RingMoETokenDispatcher(
                self.ep_group,
                self.ep_size,
                num_experts=self.num_experts,
                fp8_dispatch=True,
            )


class TestRingCollectives(_RingTestBase):
    def test_ring_all_gather_forward_and_backward(self):
        from paddlefleet.transformer.moe.token_dispatcher import _RingAllGather

        x = self._tokens()
        out = _RingAllGather.apply(x, self.ep_group)
        self.assertEqual(
            out.shape, [self.T_local * self.ep_size, self.d_latent]
        )
        peers = [paddle.empty_like(out) for _ in range(self.ep_size)]
        dist.all_gather(peers, out, group=self.ep_group)
        for peer in peers:
            np.testing.assert_array_equal(out.numpy(), peer.numpy())
        out.sum().backward()
        self.assertEqual(x.grad.shape, [self.T_local, self.d_latent])

    def test_inter_ring_shift_rotates_rows(self):
        from paddlefleet.transformer.moe.token_dispatcher import _InterRingShift

        disp = _make_dispatcher(1, self.ep_group)
        group = disp.inter_group
        r = group.rank
        dst, src = (r + 1) % group.nranks, (r - 1) % group.nranks
        x = paddle.full([self.T_local, self.d_latent], float(r + 1))
        x.stop_gradient = False
        handle = {}
        out = _InterRingShift.apply(x, group, dst, src, handle)
        handle["task"].wait()
        # Rows come from ``src``, whose fill value is src + 1.
        np.testing.assert_allclose(
            out.numpy(), np.full(out.shape, float(src + 1)), rtol=1e-6
        )
        out.sum().backward()
        self.assertEqual(x.grad.shape, [self.T_local, self.d_latent])

    def test_drain_async_handle(self):
        from paddlefleet.transformer.moe import token_dispatcher as td

        self.assertIsNone(td._drain_async_handle(None, "test"))

        waited = []

        class _OkTask:
            def wait(self):
                waited.append(True)

        self.assertIsNone(td._drain_async_handle({"task": _OkTask()}, "test"))
        self.assertEqual(waited, [True])

        class _BadTask:
            def wait(self):
                raise RuntimeError("nccl said no")

        # A failed wait must be swallowed: there is nothing left to recover and
        # raising would mask whatever aborted the previous forward.
        self.assertIsNone(td._drain_async_handle({"task": _BadTask()}, "test"))

    def test_degenerate_helpers_are_passthrough(self):
        disp = _make_dispatcher(2, self.ep_group)
        x = self._tokens()
        idx, w = self._routing()
        self.assertIs(disp._ag(x, None), x)
        self.assertIs(disp._rs(x, None), x)
        self.assertIs(disp._ag_indices(idx, None), idx)
        self.assertIs(disp._ag_router(w, None), w)

    def test_ag_indices_and_router_gather(self):
        disp = _make_dispatcher(2, self.ep_group)
        idx, w = self._routing()
        g_idx = disp._ag_indices(idx, disp.intra_group)
        g_w = disp._ag_router(w, disp.intra_group)
        self.assertEqual(g_idx.shape[0], self.T_local * disp.G)
        self.assertEqual(g_w.shape[0], self.T_local * disp.G)


class TestRingPrefetch(_RingTestBase):
    def test_prefetch_matches_inline_all_gather(self):
        from paddlefleet.transformer.moe.token_dispatcher import (
            _RingAllGather,
            _RingPreAllGatherResult,
        )

        disp = _make_dispatcher(2, self.ep_group)
        base_in = self._tokens()

        inline_in = base_in.clone().detach()
        inline_in.stop_gradient = False
        inline = _RingAllGather.apply(inline_in, disp.intra_group)
        (inline.astype("float32") ** 2).sum().backward()

        pre_in = base_in.clone().detach()
        pre_in.stop_gradient = False
        disp.pre_intra_allgather(pre_in)
        handle = disp._take_pre_intra_ag(pre_in)
        self.assertIsNotNone(handle)
        pre = _RingPreAllGatherResult.apply(pre_in, handle)
        (pre.astype("float32") ** 2).sum().backward()

        np.testing.assert_array_equal(inline.numpy(), pre.numpy())
        np.testing.assert_array_equal(
            inline_in.grad.numpy(), pre_in.grad.numpy()
        )

    def test_take_prefetch_is_idempotent(self):
        disp = _make_dispatcher(2, self.ep_group)
        x = self._tokens()
        disp.pre_intra_allgather(x)
        self.assertIsNotNone(disp._take_pre_intra_ag(x))
        # Handle is consumed; a second round must issue its own AllGather.
        self.assertIsNone(disp._take_pre_intra_ag(x))

    def test_flat_path_prefetch_also_drains_leftovers(self):
        # pre_allgather on the inherited flat path shares _drain_async_handle;
        # the ring never calls it, so cover the refactored line here.
        from paddlefleet.transformer.moe.token_dispatcher import (
            AllGatherTokenDispatcher,
        )

        flat = AllGatherTokenDispatcher(
            self.ep_group, self.ep_size, num_experts=self.num_experts
        )
        x1, x2 = self._tokens(), self._tokens()
        flat.pre_allgather(x1)
        first = flat._pre_ag_handle
        flat.pre_allgather(x2)
        self.assertIsNot(flat._pre_ag_handle, first)
        flat._pre_ag_handle["task"].wait()

    def test_shape_mismatch_raises_instead_of_falling_back(self):
        disp = _make_dispatcher(2, self.ep_group)
        x = self._tokens()
        disp.pre_intra_allgather(x)
        with self.assertRaises(RuntimeError):
            disp._take_pre_intra_ag(self._tokens(rows=self.T_local // 2))

    def test_prefetch_noop_without_intra_group(self):
        disp = _make_dispatcher(1, self.ep_group)  # G=1 -> intra_group is None
        disp.pre_intra_allgather(self._tokens())
        self.assertIsNone(disp._pre_intra_ag)
        self.assertIsNone(disp._take_pre_intra_ag(self._tokens()))

    def test_leftover_prefetch_is_drained(self):
        disp = _make_dispatcher(2, self.ep_group)
        # Keep both inputs alive: the collective reads them asynchronously.
        x1, x2 = self._tokens(), self._tokens()
        disp.pre_intra_allgather(x1)
        first = disp._pre_intra_ag
        # A forward that never reached the ring leaves the handle behind; the
        # next prefetch must wait it out rather than race it.
        disp.pre_intra_allgather(x2)
        self.assertIsNotNone(disp._pre_intra_ag)
        self.assertIsNot(disp._pre_intra_ag, first)


class TestRingForward(_RingTestBase):
    """``ring_forward`` end to end, with the grouped GEMM stubbed out.

    With ``expert_fn = x * scale`` the whole ring is linear, so the expected
    output is the local tokens scaled by ``scale * G`` (the intra
    ReduceScatter-SUM adds the G identical copies the AllGather produced) on
    the intra-only topology, and by ``scale`` per node on the inter-only one.
    """

    def _run(self, disp, x, idx, w, **kwargs):
        return disp.ring_forward(
            x, w, idx, _scale_expert_fn(2.0), w.dtype, **kwargs
        )

    def test_intra_only_forward_and_backward(self):
        disp = _make_dispatcher(2, self.ep_group)
        x = self._tokens()
        idx, w = self._routing()
        out = self._run(disp, x, idx, w)
        self.assertEqual(out.shape, [self.T_local, self.d_latent])
        np.testing.assert_allclose(
            out.numpy(), x.numpy() * 2.0 * disp.G, rtol=1e-5, atol=1e-5
        )
        out.sum().backward()
        self.assertEqual(x.grad.shape, x.shape)

    def test_inter_only_forward_and_backward(self):
        disp = _make_dispatcher(1, self.ep_group)
        x = self._tokens()
        idx, w = self._routing()
        out = self._run(disp, x, idx, w)
        self.assertEqual(out.shape, [self.T_local, self.d_latent])
        # Every node contributes ``x * scale`` for the rows it is handed, and
        # the inter ReduceScatter sums the N contributions back home.
        np.testing.assert_allclose(
            out.numpy(), x.numpy() * 2.0 * disp.N, rtol=1e-5, atol=1e-5
        )
        out.sum().backward()
        self.assertEqual(x.grad.shape, x.shape)

    def test_forward_accepts_3d_input(self):
        disp = _make_dispatcher(2, self.ep_group)
        x = self._tokens()
        idx, w = self._routing()
        flat = self._run(disp, x, idx, w)
        x3 = x.reshape([2, self.T_local // 2, self.d_latent])
        out = self._run(disp, x3, idx, w)
        np.testing.assert_allclose(out.numpy(), flat.numpy(), rtol=1e-6)

    def test_prefetched_round0_matches_inline(self):
        disp = _make_dispatcher(2, self.ep_group)
        idx, w = self._routing()
        x = self._tokens()
        baseline = self._run(disp, x, idx, w)

        x_pre = x.clone().detach()
        x_pre.stop_gradient = False
        disp.pre_intra_allgather(x_pre)
        pre = self._run(disp, x_pre, idx, w)
        np.testing.assert_array_equal(baseline.numpy(), pre.numpy())
        self.assertIsNone(disp._pre_intra_ag)

    def test_check_equal_tokens(self):
        disp = _make_dispatcher(1, self.ep_group)
        # Same count on both ranks: passes and returns nothing.
        self.assertIsNone(disp._check_equal_tokens(self.T_local))
        # Rank 0 disagrees, so every rank must raise -- the all_gather inside
        # makes the check itself collective.
        with self.assertRaises(ValueError):
            disp._check_equal_tokens(
                self.T_local + (1 if self.rank == 0 else 0)
            )

    def test_global_tokens_per_expert_sums_over_ep(self):
        disp = _make_dispatcher(2, self.ep_group)
        idx, _ = self._routing()
        counts = disp.global_tokens_per_expert(idx)
        self.assertEqual(counts.shape, [self.num_experts])
        # Each rank routes T_local * topk assignments; EP ranks are summed.
        self.assertEqual(
            int(counts.sum()), self.T_local * idx.shape[1] * self.ep_size
        )


class TestRingCombineOverlap(_RingTestBase):
    """The shared-expert subgraph carried through ``_inter_combine``."""

    def _handle(self, residual):
        def fn(res):
            return (res * 3.0,)

        return {"fn": fn, "fn_args": (residual,)}

    def test_inter_combine_without_handle(self):
        disp = _make_dispatcher(2, self.ep_group)
        x = self._tokens()
        # group=None means the rows are already final: pure passthrough.
        self.assertIs(disp._inter_combine(x, None, None), x)
        rs = disp._inter_combine(x, disp.intra_group, None)
        self.assertEqual(rs.shape, [self.T_local // disp.G, self.d_latent])

    def test_overlapped_combine_matches_serial(self):
        disp = _make_dispatcher(1, self.ep_group)
        idx, w = self._routing()
        x = self._tokens()
        serial = disp.ring_forward(x, w, idx, _scale_expert_fn(2.0), w.dtype)

        x2 = x.clone().detach()
        x2.stop_gradient = False
        residual = self._tokens()
        handle = self._handle(residual)
        overlapped = disp.ring_forward(
            x2,
            w,
            idx,
            _scale_expert_fn(2.0),
            w.dtype,
            combine_overlap_handle=handle,
        )
        np.testing.assert_allclose(
            overlapped.numpy(), serial.numpy(), rtol=1e-5, atol=1e-5
        )
        # The subgraph ran and its result was handed back for MoELayer to add.
        np.testing.assert_allclose(
            handle["fn_out"][0].numpy(),
            residual.numpy() * 3.0,
            rtol=1e-5,
        )

    def test_overlapped_combine_on_single_node_ring(self):
        # N==1 takes the group=None branch of _AllGatherCombineAsync, which
        # still has to populate fn_out or MoELayer would read a missing key.
        disp = _make_dispatcher(2, self.ep_group)
        idx, w = self._routing()
        residual = self._tokens()
        handle = self._handle(residual)
        out = disp.ring_forward(
            self._tokens(),
            w,
            idx,
            _scale_expert_fn(2.0),
            w.dtype,
            combine_overlap_handle=handle,
        )
        self.assertEqual(out.shape, [self.T_local, self.d_latent])
        np.testing.assert_allclose(
            handle["fn_out"][0].numpy(), residual.numpy() * 3.0, rtol=1e-5
        )

    def test_overlapped_combine_backward(self):
        disp = _make_dispatcher(1, self.ep_group)
        idx, w = self._routing()
        x = self._tokens()
        residual = self._tokens()
        handle = self._handle(residual)
        out = disp.ring_forward(
            x,
            w,
            idx,
            _scale_expert_fn(2.0),
            w.dtype,
            combine_overlap_handle=handle,
        )
        (out.sum() + handle["fn_out"][0].sum()).backward()
        self.assertEqual(x.grad.shape, x.shape)
        np.testing.assert_allclose(
            residual.grad.numpy(), np.full(residual.shape, 3.0), rtol=1e-5
        )


class TestIntermediateShardingPredicate(unittest.TestCase):
    """``ringmoe`` must shard experts exactly like ``allgather``."""

    def _sharded(self, dispatcher_type, expert_parallel=True):
        from paddlefleet.transformer.moe.moe_expert import GroupedMLPExpert

        class _Cfg:
            moe_token_dispatcher_type = dispatcher_type

        class _Expert:
            config = _Cfg()

        obj = _Expert()
        obj.expert_parallel = expert_parallel
        return GroupedMLPExpert.intermediate_ep_sharded.fget(obj)

    def test_ringmoe_shards_like_allgather(self):
        self.assertTrue(self._sharded("ringmoe"))
        self.assertTrue(self._sharded("allgather"))

    def test_other_dispatchers_and_no_ep(self):
        self.assertFalse(self._sharded("alltoall"))
        self.assertFalse(self._sharded("ringmoe", expert_parallel=False))


class TestMoELayerRingBranches(unittest.TestCase):
    """``MoELayer``'s ringmoe branches, driven as unbound methods.

    Same approach as ``TestMoELayerCombineEP`` in the allgather test: the
    branches under test only read a handful of attributes, so a stand-in object
    avoids building a full TransformerConfig.
    """

    def _layer(self, **attrs):
        from types import SimpleNamespace

        base = {
            "expert_model_parallel_size": 2,
            "moe_allgather_gate_overlap": True,
            "moe_token_dispatcher_type": "ringmoe",
            "use_ring_moe": True,
            "use_latent_moe": False,
            "_latent_hidden": None,
            "token_dispatcher": mock.MagicMock(),
            "_supports_three_path_clone": lambda: True,
        }
        base.update(attrs)
        return SimpleNamespace(**base)

    # -- _maybe_pre_allgather_overlap -------------------------------------
    def _pre_overlap(self, layer, hs, dispatcher_hs=None):
        from paddlefleet.transformer.moe.moe_layer import MoELayer

        return MoELayer._maybe_pre_allgather_overlap(layer, hs, dispatcher_hs)

    def test_ring_prefetch_uses_dispatcher_path_tensor(self):
        layer = self._layer()
        hs, disp_hs = paddle.randn([4, 8]), paddle.randn([4, 8])
        self._pre_overlap(layer, hs, disp_hs)
        layer.token_dispatcher.pre_intra_allgather.assert_called_once_with(
            disp_hs
        )
        self.assertIsNone(layer._latent_hidden)

    def test_ring_prefetch_falls_back_to_hidden_states(self):
        layer = self._layer()
        hs = paddle.randn([4, 8])
        self._pre_overlap(layer, hs)
        layer.token_dispatcher.pre_intra_allgather.assert_called_once_with(hs)

    def test_ring_prefetch_hoists_latent_projection(self):
        layer = self._layer(
            use_latent_moe=True, fc1_latent_proj=lambda t: t * 2.0
        )
        hs = paddle.randn([4, 8])
        self._pre_overlap(layer, hs)
        np.testing.assert_allclose(
            layer._latent_hidden.numpy(), hs.numpy() * 2.0, rtol=1e-6
        )
        layer.token_dispatcher.pre_intra_allgather.assert_called_once_with(
            layer._latent_hidden
        )

    def test_ring_prefetch_skipped_for_custom_expert_input(self):
        layer = self._layer(_supports_three_path_clone=lambda: False)
        self._pre_overlap(layer, paddle.randn([4, 8]))
        layer.token_dispatcher.pre_intra_allgather.assert_not_called()
        self.assertIsNone(layer._latent_hidden)

    def test_prefetch_disabled_without_ep_or_flag(self):
        for attrs in (
            {"expert_model_parallel_size": 1},
            {"moe_allgather_gate_overlap": False},
        ):
            layer = self._layer(**attrs)
            self._pre_overlap(layer, paddle.randn([4, 8]))
            layer.token_dispatcher.pre_intra_allgather.assert_not_called()
            layer.token_dispatcher.pre_allgather.assert_not_called()

    def test_allgather_prefetch_path_unchanged(self):
        layer = self._layer(
            moe_token_dispatcher_type="allgather", use_ring_moe=False
        )
        hs = paddle.randn([4, 8])
        self._pre_overlap(layer, hs, paddle.randn([4, 8]))
        # The flat path keeps prefetching hidden_states, not the dispatcher path.
        layer.token_dispatcher.pre_allgather.assert_called_once_with(hs)

    def test_other_dispatchers_only_clear_latent_cache(self):
        layer = self._layer(
            moe_token_dispatcher_type="alltoall",
            use_ring_moe=False,
            _latent_hidden=paddle.randn([4, 8]),
        )
        self._pre_overlap(layer, paddle.randn([4, 8]))
        self.assertIsNone(layer._latent_hidden)
        layer.token_dispatcher.pre_allgather.assert_not_called()

    # -- config validation ------------------------------------------------
    def _validate(self, **attrs):
        from paddlefleet.transformer.moe.moe_layer import MoELayer

        base = {
            "using_sonic_moe": True,
            "moe_use_fusion_node": True,
            "moe_expert_fusion": True,
            "moe_deep_gemm": False,
            "moe_intermediate_size": 256,
            "fp8": False,
        }
        base.update(attrs)
        layer = self._layer(**base)
        MoELayer._validate_intermediate_ep_sharding_config(layer)
        return layer

    def test_validation_names_the_configured_dispatcher(self):
        with self.assertRaises(ValueError) as ctx:
            self._validate(using_sonic_moe=False)
        self.assertIn("'ringmoe'", str(ctx.exception))

    def test_validation_force_corrects_incompatible_flags(self):
        layer = self._validate(
            moe_use_fusion_node=False,
            moe_expert_fusion=False,
            moe_deep_gemm=True,
        )
        self.assertTrue(layer.moe_use_fusion_node)
        self.assertTrue(layer.moe_expert_fusion)
        self.assertFalse(layer.moe_deep_gemm)

    def test_validation_rejects_indivisible_intermediate_size(self):
        with self.assertRaises(ValueError):
            self._validate(moe_intermediate_size=255)

    # -- combine() routing ------------------------------------------------
    def test_combine_routes_ringmoe_to_token_combine(self):
        from paddlefleet.transformer.moe.moe_layer import MoELayer

        layer = mock.MagicMock()
        layer.moe_token_dispatcher_type = "ringmoe"
        MoELayer.combine(
            layer, paddle.randn([4, 8]), combine_overlap_handle=None
        )
        layer.token_dispatcher.token_combine.assert_called_once()
        layer.token_dispatcher.combine_postprocess.assert_called_once()

    # -- ringmoe_forward --------------------------------------------------
    def _forward_layer(self, **attrs):
        base = {
            "using_sonic_moe": True,
            "_project_to_latent": lambda t: t,
            "layer_number": 0,
            "moe_group": None,
            "num_experts_per_tok": 2,
            "is_mtp_layer": False,
            "recompute_moe_gate_up": False,
            "grouped_gemm_experts": object(),
            "use_latent_moe": False,
            "latent_norm": None,
            "fc2_latent_proj": lambda t: t * 3.0,
        }
        base.update(attrs)
        layer = self._layer(**base)
        layer.token_dispatcher.ring_forward.return_value = paddle.ones([4, 8])
        return layer

    def _forward(self, layer, **kwargs):
        from paddlefleet.transformer.moe.moe_layer import MoELayer

        x = paddle.randn([4, 8])
        return MoELayer.ringmoe_forward(
            layer,
            x,
            paddle.randn([4, 4]),
            None,
            topk_weights=paddle.randn([4, 2]),
            topk_indices=paddle.zeros([4, 2], dtype="int32"),
            **kwargs,
        )

    def test_ringmoe_forward_requires_sonic_moe(self):
        with self.assertRaises(ValueError):
            self._forward(self._forward_layer(using_sonic_moe=False))

    def test_ringmoe_forward_calls_ring_forward(self):
        layer = self._forward_layer()
        handle = {"fn": None, "fn_args": ()}
        out = self._forward(layer, combine_overlap_handle=handle)
        np.testing.assert_allclose(out.numpy(), np.ones([4, 8]), rtol=1e-6)
        kwargs = layer.token_dispatcher.ring_forward.call_args.kwargs
        self.assertIs(kwargs["combine_overlap_handle"], handle)

    def test_ringmoe_forward_projects_back_from_latent(self):
        layer = self._forward_layer(
            use_latent_moe=True, latent_norm=lambda t: t + 1.0
        )
        out = self._forward(layer)
        # (ring output 1 + norm 1) * fc2 3
        np.testing.assert_allclose(out.numpy(), np.full([4, 8], 6.0), rtol=1e-6)

    def test_ringmoe_forward_logs_balance_from_dispatcher(self):
        layer = self._forward_layer()
        layer.token_dispatcher.global_tokens_per_expert.return_value = (
            paddle.zeros([4], dtype="int32")
        )
        with (
            mock.patch.object(
                moe_layer,
                "global_moe_balance_training_logs_enabled",
                return_value=True,
            ),
            mock.patch.object(moe_layer, "log_moe_balance") as logged,
            paddle.enable_grad(),
        ):
            self._forward(layer)
        logged.assert_called_once()
        layer.token_dispatcher.global_tokens_per_expert.assert_called_once()


if __name__ == "__main__":
    unittest.main()
