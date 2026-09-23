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

Two cards can only reach the inter-only ring (G=1, N=2), selected by patching
``_RING_GPUS_PER_NODE`` to 1: the dispatcher requires N>1, so a single-node ring
(G=EP, N=1) is a rejected configuration and is asserted as such here. The
two-level topology (G>1 AND N>1) needs four cards and lives in
``test_ring_dispatcher_two_level.py``.

Run with:
  python -m paddle.distributed.launch --gpus=0,1 \
      tests/multi_card_tests/moe/test_ring_dispatcher_ep.py
"""

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
from paddlefleet.transformer.moe.token_dispatcher import (
    quantize_activation_blockscaled_fast as _sonic_fp8_quant,
)

# Tests that actually invoke SonicMoE's fp8 block-scaled quantizer only work on a
# build/GPU where it is available (e.g. Blackwell); skip (not error) elsewhere.
# fp8-flag-only tests below do NOT need it and stay unguarded.
_needs_sonicmoe_fp8 = unittest.skipUnless(
    _sonic_fp8_quant is not None,
    "SonicMoE fp8 quantizer (quantize_activation_blockscaled_fast) unavailable",
)

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
    # fleet is process-global, and this module does not always own the process:
    # a runner can batch several test files together, or load this file twice
    # (as ``__main__`` and as a package module), in which case the guard above
    # sits in one module copy while fleet is already up. A second
    # initialize_fleet then trips ``args is already initialized`` in
    # set_global_variables. Probe the hybrid-communicate-group singleton (unset
    # before fleet.init) rather than catching a broad exception, so a genuine
    # comm-group failure still surfaces. Same approach as test_router.py.
    if getattr(fleet.fleet, "_hcg", None) is None:
        initialize_fleet(strategy=strategy)
    _pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    ep = _pg_collection.ep
    if ep is None or ep.nranks != 2:
        # Reusing someone else's topology would silently test the wrong ring
        # (G/N assertions assume EP==2), so fail loudly instead of skipping --
        # a skipped ring suite would go unnoticed.
        raise RuntimeError(
            "RingMoE EP tests need ep_degree=2, but the process was already "
            f"initialised with ep={None if ep is None else ep.nranks}. Launch "
            "this file on its own: python -m paddle.distributed.launch "
            "--gpus=0,1 tests/multi_card_tests/moe/test_ring_dispatcher_ep.py"
        )
    _fleet_initialised = True
    return _pg_collection


def _make_dispatcher(
    gpus_per_node, ep_group, num_experts=4, fp8_dispatch=False, use_ue8m0=False
):
    """Build a dispatcher with the ring topology forced to a known G.

    Production always splits on ``_RING_GPUS_PER_NODE``; patching it to 1 is how
    a two-card job reaches the inter-only ring, the only topology EP==2 can form
    now that N==1 is rejected. Sub-group creation is collective and cached per
    (ep_ranks, G), so every rank must call this in the same order -- which
    unittest guarantees within a single test.
    """
    from paddlefleet.transformer.moe import token_dispatcher as td

    with mock.patch.object(td, "_RING_GPUS_PER_NODE", gpus_per_node):
        return td.RingMoETokenDispatcher(
            ep_group,
            ep_group.nranks,
            num_experts=num_experts,
            fp8_dispatch=fp8_dispatch,
            use_ue8m0=use_ue8m0,
        )


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
        sync_free_sizing=False,
    ):
        return g_tok * scale

    return expert_fn


class _Fp8StraightThrough(paddle.autograd.PyLayer):
    """Stand-in for the fp8 expert GEMM: e4m3 in, bf16 out, bf16 grad back.

    The real SonicMoE GEMM consumes the gathered e4m3 bytes and hands back a
    bf16 ``dx``; the gather PyLayers rely on that (their ReduceScatter would
    otherwise trip NCCL's "float8 not supported for reductions"). Reproducing
    the dtype contract here is what makes the gather tests exercise the real
    straight-through path instead of a mocked quantizer.
    """

    @staticmethod
    def forward(ctx, tok_fp8):
        ctx.set_grad_in_dtype_consistent(False)
        ctx.set_materialize_grads(False)
        return tok_fp8.astype("bfloat16")

    @staticmethod
    def backward(ctx, grad):
        return grad


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

    def _tokens(self, rows=None, requires_grad=True, dtype="float32"):
        x = paddle.randn([rows or self.T_local, self.d_latent])
        x = (x * (self.rank + 1)).astype(dtype)
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
    def test_gpus_per_node_is_fixed_at_eight(self):
        from paddlefleet.transformer.moe import token_dispatcher as td

        self.assertEqual(td._RING_GPUS_PER_NODE, 8)

    def test_single_node_ring_is_rejected(self):
        """EP inside one machine gives N==1, which the ring refuses.

        With N==1 the two-level ring degenerates to a single intra AllGather,
        i.e. exactly the flat 'allgather' dispatcher, so accepting it would only
        hide a misconfiguration behind a slower code path. The sub-group build
        runs first and is a world collective, so every rank reaches the assert.
        """
        with self.assertRaisesRegex(AssertionError, "EP to span >1 node"):
            _make_dispatcher(2, self.ep_group)

    def test_inter_only_topology(self):
        disp = _make_dispatcher(1, self.ep_group)
        self.assertEqual((disp.G, disp.N), (1, 2))
        self.assertIsNone(disp.intra_group)
        self.assertIsNotNone(disp.inter_group)

    def test_subgroups_are_cached(self):
        a = _make_dispatcher(1, self.ep_group)
        b = _make_dispatcher(1, self.ep_group)
        self.assertIs(a.inter_group, b.inter_group)

    def test_ep_not_divisible_by_gpus_per_node_raises(self):
        from paddlefleet.transformer.moe import token_dispatcher as td

        class _FakeGroup:
            ranks = [0, 1, 2]

        with self.assertRaises(ValueError):
            td._build_ring_subgroups(_FakeGroup(), 2)

    def test_no_ep_group_is_rejected(self):
        """EP==1 has no ring at all; the dispatcher must say so, not degenerate.

        Purely local, so this needs no rank agreement.
        """
        from paddlefleet.transformer.moe.token_dispatcher import (
            RingMoETokenDispatcher,
        )

        with self.assertRaisesRegex(AssertionError, "real EP group"):
            RingMoETokenDispatcher(None, 1, num_experts=self.num_experts)

    def test_fp8_dispatch_is_accepted(self):
        """fp8 dispatch is supported now (was NotImplementedError before).

        The blocker was never the ring: MoELayer already validates the fp8
        shard alignment generically across the intermediate-sharded
        dispatchers, and the ring quantizes per round via _RingFP8AllGather.
        """
        disp = _make_dispatcher(1, self.ep_group, fp8_dispatch=True)
        self.assertTrue(disp.fp8_dispatch)

    @_needs_sonicmoe_fp8
    def test_degenerate_fp8_gather_preserves_input_gradient(self):
        """A degenerate group still quantizes; only the collective is skipped."""
        disp = _make_dispatcher(1, self.ep_group, fp8_dispatch=True)
        x = self._tokens(dtype="bfloat16")
        tok = x * 1.0
        gathered, scale = disp._ag_tokens(tok, None)
        self.assertEqual(gathered.shape, [self.T_local, self.d_latent])
        self.assertEqual(str(gathered.dtype), "paddle.float8_e4m3fn")
        # fp8 mode must always produce a scale, collective or not.
        self.assertIsNotNone(scale)
        self.assertEqual(scale.shape[0], self.T_local)
        _Fp8StraightThrough.apply(gathered).sum().backward()
        self.assertIsNotNone(x.grad)
        np.testing.assert_allclose(
            x.grad.astype("float32").numpy(), np.ones_like(x.numpy())
        )

    def test_fp8_dispatch_accepts_ue8m0(self):
        """``use_ue8m0`` must not gate fp8 dispatch.

        AllGatherTokenDispatcher stores the flag and never reads it -- fp8
        dispatch quantization always uses the int32-scale helper -- so the ring
        must accept the same combination instead of blocking a config the flat
        path runs fine. Production sets use_ue8m0=true.
        """
        disp = _make_dispatcher(
            1, self.ep_group, fp8_dispatch=True, use_ue8m0=True
        )
        self.assertTrue(disp.fp8_dispatch)

    def test_calls_per_micro_batch_delays_fp8_weight_release(self):
        """The ring calls the expert N times per micro batch.

        SonicMoEExpert counts raw forward() calls, so without
        _calls_per_micro_batch the fp8 weights would be released after the first
        ring round of the last micro batch instead of the last round. Exercise
        the property arithmetic directly -- building a real expert needs a full
        TransformerConfig and the fused kernels.
        """
        from paddlefleet.transformer.moe.moe_expert import SonicMoEExpert

        e = object.__new__(SonicMoEExpert)
        e._num_micro_batches = 4
        e._calls_per_micro_batch = 1
        e._forward_counter = 3
        self.assertTrue(e._is_last_micro_batch)  # flat path: 4th of 4 calls

        # Ring with N=2: 8 calls total, so call #3 must NOT release.
        e._calls_per_micro_batch = 2
        self.assertFalse(e._is_last_micro_batch)
        e._forward_counter = 6
        self.assertFalse(e._is_last_micro_batch)
        e._forward_counter = 7
        self.assertTrue(e._is_last_micro_batch)

    def test_set_calls_per_micro_batch_rejects_zero(self):
        from paddlefleet.transformer.moe.moe_expert import SonicMoEExpert

        e = object.__new__(SonicMoEExpert)
        e._calls_per_micro_batch = 1
        with self.assertRaises(ValueError):
            SonicMoEExpert.set_calls_per_micro_batch(e, 0)

    def test_bf16_path_returns_no_scale(self):
        """``_ag_tokens`` must keep returning ``scale=None`` in bf16 mode."""
        disp = _make_dispatcher(1, self.ep_group, self.num_experts)
        x = self._tokens()
        g_tok, g_scale = disp._ag_tokens(x, disp.inter_group)
        self.assertIsNone(g_scale)
        self.assertEqual(g_tok.shape[0], self.T_local * disp.N)


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

    @_needs_sonicmoe_fp8
    def test_fp8_all_gather_forward_and_backward(self):
        """FP8 gather keeps its collective and ReduceScatter gradient dual.

        Data and scale travel as two separate plain AllGathers, so both come
        back per-rank-concatenated; the gradient goes the other way through one
        ReduceScatter-SUM, which is why each local row sees ``N``.
        """
        disp = _make_dispatcher(1, self.ep_group, fp8_dispatch=True)
        x = self._tokens(dtype="bfloat16")
        tok = x * 1.0
        gathered, scale = disp._ag_tokens(tok, disp.inter_group)
        self.assertEqual(gathered.shape, [self.T_local * disp.N, self.d_latent])
        self.assertEqual(str(gathered.dtype), "paddle.float8_e4m3fn")
        self.assertEqual(scale.shape[0], self.T_local * disp.N)
        _Fp8StraightThrough.apply(gathered).sum().backward()
        self.assertIsNotNone(x.grad)
        np.testing.assert_allclose(
            x.grad.astype("float32").numpy(),
            np.full_like(x.numpy(), disp.N),
            rtol=1e-6,
            atol=1e-6,
        )

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
        disp = _make_dispatcher(1, self.ep_group)
        x = self._tokens()
        idx, w = self._routing()
        self.assertIs(disp._ag(x, None), x)
        self.assertIs(disp._rs(x, None), x)
        self.assertIs(disp._ag_indices(idx, None), idx)
        self.assertIs(disp._ag_router(w, None), w)

    def test_ag_indices_and_router_gather(self):
        disp = _make_dispatcher(1, self.ep_group)
        idx, w = self._routing()
        g_idx = disp._ag_indices(idx, disp.inter_group)
        g_w = disp._ag_router(w, disp.inter_group)
        self.assertEqual(g_idx.shape[0], self.T_local * disp.N)
        self.assertEqual(g_w.shape[0], self.T_local * disp.N)

    def test_flat_path_prefetch_also_drains_leftovers(self):
        # pre_allgather belongs to the inherited flat EP path; the ring never
        # prefetches, so cover _drain_async_handle's caller here.
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


class TestRingForward(_RingTestBase):
    """``ring_forward`` end to end, with the grouped GEMM stubbed out.

    With ``expert_fn = x * scale`` the whole ring is linear, so on the
    inter-only topology reachable here the expected output is the local tokens
    scaled by ``scale * N``: every node contributes ``x * scale`` for the rows it
    is handed and the inter ReduceScatter sums the N contributions back home.
    """

    def _run(self, disp, x, idx, w, **kwargs):
        return disp.ring_forward(
            x, w, idx, _scale_expert_fn(2.0), w.dtype, **kwargs
        )

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
        disp = _make_dispatcher(1, self.ep_group)
        x = self._tokens()
        idx, w = self._routing()
        flat = self._run(disp, x, idx, w)
        x3 = x.reshape([2, self.T_local // 2, self.d_latent])
        out = self._run(disp, x3, idx, w)
        np.testing.assert_allclose(out.numpy(), flat.numpy(), rtol=1e-6)

    def test_fp8_rejects_unaligned_hidden_width(self):
        disp = _make_dispatcher(1, self.ep_group, fp8_dispatch=True)
        x = self._tokens()
        idx, w = self._routing()
        with self.assertRaisesRegex(ValueError, "multiple of 128"):
            self._run(disp, x, idx, w)

    def test_equal_token_check_runs_once_per_dispatcher(self):
        disp = _make_dispatcher(1, self.ep_group)
        idx, w = self._routing()
        self.assertFalse(disp._equal_tokens_checked)
        self._run(disp, self._tokens(), idx, w)
        self.assertTrue(disp._equal_tokens_checked)
        with mock.patch.object(disp, "_check_equal_tokens") as checked:
            self._run(disp, self._tokens(), idx, w)
        checked.assert_not_called()

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
        disp = _make_dispatcher(1, self.ep_group)
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
        disp = _make_dispatcher(1, self.ep_group)
        x = self._tokens()
        # group=None means the rows are already final: pure passthrough. The
        # partials arrive as a per-node list, so a single-entry list is the
        # degenerate case.
        self.assertIs(disp._inter_combine([x], None, None), x)
        rs = disp._inter_combine([x] * disp.N, disp.inter_group, None)
        self.assertEqual(rs.shape, [self.T_local, self.d_latent])

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
            # Real MoELayer always carries its config; the latent projection
            # path passes it to deferrable_linear_bare even when no dW point is
            # selected.
            "config": SimpleNamespace(p2p_overlap_dw_calc=None),
            "token_dispatcher": mock.MagicMock(),
            "_supports_three_path_clone": lambda: True,
        }
        base.update(attrs)
        return SimpleNamespace(**base)

    # -- _maybe_pre_allgather_overlap -------------------------------------
    def _pre_overlap(self, layer, hs):
        from paddlefleet.transformer.moe.moe_layer import MoELayer

        return MoELayer._maybe_pre_allgather_overlap(layer, hs)

    def test_ringmoe_never_prefetches(self):
        # The ring used to pre-issue round 0's intra AllGather here. It was
        # removed: the only thing it could hide was the gate, and the early
        # collective cost more in contention than the gate was worth.
        layer = self._layer(_latent_hidden=paddle.randn([4, 8]))
        self._pre_overlap(layer, paddle.randn([4, 8]))
        layer.token_dispatcher.pre_allgather.assert_not_called()
        layer.token_dispatcher.pre_intra_allgather.assert_not_called()
        self.assertIsNone(layer._latent_hidden)

    def test_prefetch_disabled_without_ep_or_flag(self):
        for attrs in (
            {"expert_model_parallel_size": 1},
            {"moe_allgather_gate_overlap": False},
        ):
            layer = self._layer(
                moe_token_dispatcher_type="allgather",
                use_ring_moe=False,
                **attrs,
            )
            self._pre_overlap(layer, paddle.randn([4, 8]))
            layer.token_dispatcher.pre_allgather.assert_not_called()

    def test_allgather_prefetch_path_unchanged(self):
        layer = self._layer(
            moe_token_dispatcher_type="allgather", use_ring_moe=False
        )
        hs = paddle.randn([4, 8])
        self._pre_overlap(layer, hs)
        layer.token_dispatcher.pre_allgather.assert_called_once_with(hs)

    def test_allgather_prefetch_hoists_latent_projection(self):
        layer = self._layer(
            moe_token_dispatcher_type="allgather",
            use_ring_moe=False,
            use_latent_moe=True,
            fc1_latent_proj=lambda t: t * 2.0,
        )
        hs = paddle.randn([4, 8])
        self._pre_overlap(layer, hs)
        np.testing.assert_allclose(
            layer._latent_hidden.numpy(), hs.numpy() * 2.0, rtol=1e-6
        )
        layer.token_dispatcher.pre_allgather.assert_called_once_with(
            layer._latent_hidden
        )

    def test_other_dispatchers_only_clear_latent_cache(self):
        layer = self._layer(
            moe_token_dispatcher_type="alltoall",
            use_ring_moe=False,
            _latent_hidden=paddle.randn([4, 8]),
        )
        self._pre_overlap(layer, paddle.randn([4, 8]))
        self.assertIsNone(layer._latent_hidden)
        layer.token_dispatcher.pre_allgather.assert_not_called()

    def test_ringmoe_gate_overlap_hoists_latent_and_pregates(self):
        # New ringmoe gate-overlap branch: hoist the latent projection, then
        # pre-issue round 0 via pre_gate_token_ag (not pre_allgather).
        layer = self._layer(
            moe_token_dispatcher_type="ringmoe",
            use_ring_moe=True,
            use_latent_moe=True,
            fc1_latent_proj=lambda t: t * 2.0,
        )
        hs = paddle.randn([4, 8])
        self._pre_overlap(layer, hs)
        np.testing.assert_allclose(
            layer._latent_hidden.numpy(), hs.numpy() * 2.0, rtol=1e-6
        )
        layer.token_dispatcher.pre_gate_token_ag.assert_called_once_with(
            layer._latent_hidden
        )
        layer.token_dispatcher.pre_allgather.assert_not_called()

    def test_ringmoe_gate_overlap_no_latent_pregates_hidden(self):
        layer = self._layer(
            moe_token_dispatcher_type="ringmoe",
            use_ring_moe=True,
            use_latent_moe=False,
        )
        hs = paddle.randn([4, 8])
        self._pre_overlap(layer, hs)
        self.assertIsNone(layer._latent_hidden)
        layer.token_dispatcher.pre_gate_token_ag.assert_called_once_with(hs)

    def _validate(self, **attrs):
        from paddlefleet.transformer.moe.moe_layer import MoELayer

        base = {
            "using_sonic_moe": True,
            "moe_use_fusion_node": True,
            "moe_expert_fusion": True,
            "moe_deep_gemm": False,
            "moe_intermediate_size": 256,
            "fp8": False,
            # The real MoELayer always sets these two. The production
            # allgather+fp8 baseline uses fp8_wgrad=False.
            "fp8_wgrad": False,
            "fp8_dispatch_bwd": False,
        }
        base.update(attrs)
        layer = self._layer(**base)
        MoELayer._validate_intermediate_ep_sharding_config(layer)
        return layer

    def test_validation_accepts_fp8_regardless_of_wgrad(self):
        """Both fp8_wgrad settings must pass validation.

        The production allgather+fp8 baseline runs with fp8_wgrad=False, so the
        ring must not reject it either. (The "dz storage freed but no bwd
        prequant" failure seen on ringmoe+fp8 is caused by SonicMoE's global
        single-slot prequant handoff versus the ring's N expert calls per micro
        batch, not by this flag.)
        """
        for wgrad in (False, True):
            layer = self._validate(fp8=True, fp8_wgrad=wgrad)
            self.assertEqual(layer.fp8_wgrad, wgrad)

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

    def test_validation_fp8_requires_128_aligned_shard(self):
        # fp8 block-scale tiles have a fixed width, so mid/EP must be a multiple
        # of it -- the larger intermediate size divides cleanly, the smaller one
        # does not.
        self._validate(fp8=True, moe_intermediate_size=256)
        with self.assertRaises(ValueError) as ctx:
            self._validate(fp8=True, moe_intermediate_size=192)
        # The message names the configured dispatcher, not "allgather".
        self.assertIn("ringmoe + fp8", str(ctx.exception))

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


class TestRingSubgroupInit(_RingTestBase):
    """``init_ring_subgroups`` is the one place allowed to build the groups."""

    def test_init_populates_the_cache_dispatchers_then_reuse(self):
        from paddlefleet.transformer.moe import token_dispatcher as td

        # G is part of the cache key, so the pre-create and the dispatcher have
        # to agree on it -- patch both to the one topology EP==2 can form.
        with mock.patch.object(td, "_RING_GPUS_PER_NODE", 1):
            groups = td.init_ring_subgroups(self.ep_group)
            self.assertEqual(len(groups), 7)
            _, _, intra, inter, rs_g, ag_g, rt_g = groups
            disp = td.RingMoETokenDispatcher(
                self.ep_group, self.ep_size, num_experts=self.num_experts
            )
        # Same objects: the dispatcher hit the cache, so it ran no collective.
        self.assertIs(disp.intra_group, intra)
        self.assertIs(disp.inter_group, inter)
        # The three extra intra comms exist for the RS and the two
        # gather-aheads; G==1 here, so they are degenerate and stay None.
        self.assertIs(disp.intra_rs_group, rs_g)
        self.assertIs(disp.intra_ag_group, ag_g)
        self.assertIs(disp.intra_rt_group, rt_g)

    def test_init_defaults_to_the_parallel_state_ep_group(self):
        from paddlefleet.transformer.moe import token_dispatcher as td

        self.assertIsNotNone(td.init_ring_subgroups())

    def test_init_is_a_noop_when_ep_is_degenerate(self):
        from types import SimpleNamespace

        from paddlefleet.transformer.moe import token_dispatcher as td

        # EP==1 builds nothing, so it must not enter the world collective --
        # otherwise an EP-less rank would block the ranks that do build groups.
        self.assertIsNone(
            td.init_ring_subgroups(SimpleNamespace(nranks=1, ranks=[0]))
        )


class TestBuilderInitialisesRingSubgroups(unittest.TestCase):
    """Regression: PP stages without MoE layers must still create the groups.

    Sub-group creation is a *world* collective while a dispatcher only exists on
    stages that own a MoE layer, so ``pipeline_model_parallel_size > 1`` plus a
    sparse ``moe_layer_freq`` (or a dense prefix) used to hang: the MoE stages
    waited in ``all_gather_object`` for ranks that never arrived. The builder
    runs on every rank, so the call has to live there.
    """

    def _build(self, dispatcher_type):
        import paddlefleet.gpt_builders as gb

        config = mock.MagicMock()
        config.moe_token_dispatcher_type = dispatcher_type
        config.num_empty_layers_add_in_head = 0
        config.num_empty_layers_add_in_tail = 0
        config.separate_mtp_headloss = False
        with (
            mock.patch.object(gb, "init_ring_subgroups") as init,
            mock.patch.object(
                gb, "get_gpt_decoder_layers_spec", return_value=[]
            ),
            mock.patch.object(gb, "_get_effective_mtp_layers", return_value=0),
            mock.patch.object(gb, "get_gpt_spec"),
            mock.patch.object(gb, "build_spec_layer"),
            mock.patch.object(gb, "LanguageLoss"),
        ):
            gb.gpt_builder(config)
        return init

    def test_ringmoe_build_initialises_subgroups(self):
        self._build("ringmoe").assert_called_once_with()

    def test_other_dispatchers_do_not_create_ring_groups(self):
        self._build("allgather").assert_not_called()


@_needs_sonicmoe_fp8
class TestFp8FusedGatherHelpers(_RingTestBase):
    """The fused (data ++ scale) fp8 AllGather helpers used by the combine
    backward and the flat pre_allgather. 128-aligned hidden for the block tile."""

    def test_fused_pack_gather_split_roundtrip(self):
        from paddlefleet.transformer.moe import token_dispatcher as td

        x = self._tokens(requires_grad=False, dtype="bfloat16")
        x = paddle.concat([x] * 16, axis=-1)  # widen 8 -> 128
        fused_local, H, nsb, sdt = td._quantize_and_pack_fp8(x)
        self.assertEqual(H, 128)
        self.assertEqual(fused_local.shape[0], self.T_local)
        fg, H2, nsb2, sdt2, task = td._fused_fp8_all_gather_async(
            x, self.ep_group
        )
        task.wait()
        data, scale = td._split_fused_fp8_gather(fg, H2, nsb2, sdt2)
        self.assertEqual(str(data.dtype), "paddle.float8_e4m3fn")
        self.assertEqual(data.shape, [self.T_local * self.ep_size, 128])
        self.assertEqual(scale.shape[0], self.T_local * self.ep_size)


class TestAllGatherCombineNoOverlapGrad(_RingTestBase):
    """`_AllGatherCombineNoOverlap` combine PyLayer: forward ReduceScatter, and
    a backward that must populate the fp8 grad handle (or plain-AllGather bf16)."""

    def _widen(self, x):
        return paddle.concat([x] * 16, axis=-1)  # 8 -> 128 for fp8 tiles

    def test_bf16_backward_all_gathers_grad(self):
        from paddlefleet.transformer.moe import token_dispatcher as td

        x = paddle.randn([self.T_local * self.ep_size, self.d_latent])
        x.stop_gradient = False
        out = td._AllGatherCombineNoOverlap.apply(x, self.ep_group)
        self.assertEqual(out.shape[0], self.T_local)
        out.sum().backward()
        self.assertEqual(x.grad.shape, x.shape)

    @_needs_sonicmoe_fp8
    def test_fp8_backward_populates_handle(self):
        from paddlefleet.transformer.moe import token_dispatcher as td

        x = self._widen(
            paddle.randn([self.T_local * self.ep_size, self.d_latent])
        )
        x = x.astype("bfloat16")
        x.stop_gradient = False
        handle = {}
        out = td._AllGatherCombineNoOverlap.apply(x, self.ep_group, handle)
        _Fp8StraightThrough.apply(out).sum().backward()
        self.assertIn("data", handle)
        self.assertIn("scale", handle)
        self.assertEqual(str(handle["data"].dtype), "paddle.float8_e4m3fn")

    def test_degenerate_group_is_clone(self):
        from paddlefleet.transformer.moe import token_dispatcher as td

        x = paddle.randn([self.T_local, self.d_latent])
        x.stop_gradient = False
        out = td._AllGatherCombineNoOverlap.apply(x, None)
        out.sum().backward()
        np.testing.assert_array_equal(x.grad.numpy(), np.ones_like(x.numpy()))


class TestSonicMoEExpertReleasePredicate(unittest.TestCase):
    """`_release_fp8_weight_after_fwd` gate, driven on a bare instance."""

    def test_predicate_false_off_1x32(self):
        from types import SimpleNamespace

        from paddlefleet.transformer.moe.moe_expert import SonicMoEExpert

        e = object.__new__(SonicMoEExpert)
        e._num_micro_batches = 1
        e._calls_per_micro_batch = 1
        e._forward_counter = 0  # last micro batch
        e.config = SimpleNamespace(
            fp8_weight_quant_format="bf16", recompute_granularity=None
        )
        e.weight1 = SimpleNamespace()
        e.weight2 = SimpleNamespace()
        # quant format is not "1x32", so the release predicate is False.
        self.assertFalse(SonicMoEExpert._release_fp8_weight_after_fwd(e, False))


class TestRingDegenerateHelpers(_RingTestBase):
    """Degenerate / bail-out branches that the two-level ring never reaches on a
    single-node (G=1) EP group: pure-Python guards, no collective."""

    def test_order_after_reports_missing_stream(self):
        from paddlefleet.transformer.moe import token_dispatcher as td

        class _NoStreamPG:
            def get_stream(self, place):
                raise RuntimeError("no comm stream yet")

        class _G:
            process_group = _NoStreamPG()

        td._ORDER_AFTER_WARNED = False
        # get_stream raises -> treated as "nothing to order against yet".
        self.assertFalse(td._order_after(_G(), _G()))
        self.assertTrue(td._ORDER_AFTER_WARNED)

    def test_calc_stream_all_gather_degenerate_is_clone(self):
        from paddlefleet.transformer.moe import token_dispatcher as td

        x = self._tokens(requires_grad=False)
        out = td._calc_stream_all_gather(x, None)
        np.testing.assert_array_equal(out.numpy(), x.numpy())

    def test_prefetch_tok_ag_fp8_degenerate_returns_none(self):
        from paddlefleet.transformer.moe import token_dispatcher as td

        self.assertIsNone(td._prefetch_tok_ag_fp8({}, None))

    def test_rs_degenerate_is_identity(self):
        disp = _make_dispatcher(1, self.ep_group)
        x = self._tokens(requires_grad=False)
        self.assertIs(disp._rs(x, None), x)

    def test_pre_gate_token_ag_noop_on_inter_only(self):
        # G=1 inter-only ring has no intra group, so pre_gate_token_ag returns
        # early (nothing packed) and ring_forward falls back to inline gather.
        disp = _make_dispatcher(1, self.ep_group)
        disp.pre_gate_token_ag(self._tokens(requires_grad=False))
        self.assertIsNone(disp._gate_pack)


if __name__ == "__main__":
    unittest.main()
