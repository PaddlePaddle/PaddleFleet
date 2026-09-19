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
"""Component tests for ``OffloadManager``, driven without a pipeline runtime.

The manager only needs paddle streams and a pinned pool, so these tests build it
directly and play the schedule callbacks by hand
(``begin_forward_group`` / ``prefetch_next_group`` / ``end_iteration``). That
keeps them free of ``fleet.init`` and fast enough to cover every mode.

Offloading only changes *where* an activation lives between forward and backward,
so the bar for every mode is bit-exact loss and gradients against a run with the
feature off. Timing-dependent counters (``hit`` versus ``late``) are deliberately
not asserted on: which one a copy lands in depends on how fast the device is, and
pinning that down would make the suite flaky rather than strict.
"""

from __future__ import annotations

import math
import unittest
from unittest import mock

import paddle

from paddlefleet.activation_offload import (
    OffloadManager,
    current_offload_manager,
    manager_from_config,
    offload_enabled,
    offload_groups,
    offload_kwargs_from_config,
    offload_region,
    reset_offload_manager,
)

MB = 1 << 20
_REQUIRE_GPU = unittest.skipUnless(
    paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0,
    "Requires a CUDA device",
)


class _TwoLinears(paddle.nn.Layer):
    """Saves a handful of megabyte-scale activations per forward."""

    def __init__(self, hidden=512, inter=1024):
        super().__init__()
        self.fc1 = paddle.nn.Linear(hidden, inter)
        self.act = paddle.nn.GELU()
        self.fc2 = paddle.nn.Linear(inter, hidden)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class _SharedInput(paddle.nn.Layer):
    """Feeds one activation to three operators, so pack fires on it repeatedly."""

    def __init__(self, hidden=512):
        super().__init__()
        self.a = paddle.nn.Linear(hidden, hidden)
        self.b = paddle.nn.Linear(hidden, hidden)
        self.c = paddle.nn.Linear(hidden, hidden)

    def forward(self, x):
        return self.a(x) + self.b(x) + self.c(x)


class _StubConfig:
    """Minimal stand-in for TransformerConfig, for the config readers."""

    _OFFLOAD_MODULE_NAMES = frozenset({"core_attn", "qkv_linear", "mlp_norm"})

    def __init__(self, **kw):
        self.fine_grained_activation_offloading = False
        self.offload_modules = None
        for k, v in kw.items():
            setattr(self, k, v)


@_REQUIRE_GPU
class _OffloadTestCase(unittest.TestCase):
    """Shared fixture: one network and input reused across every mode.

    Reusing the same weights and input is what makes the bit-exactness checks
    meaningful without seeding: the reference run and the run under test see
    identical numbers by construction.
    """

    HIDDEN = 512
    NET_CLS = _TwoLinears

    def setUp(self):
        reset_offload_manager()
        paddle.seed(46)
        self.net = self.NET_CLS(self.HIDDEN)
        self.x = paddle.randn([4, 512, self.HIDDEN])

    def tearDown(self):
        reset_offload_manager()

    def _manager(self, **kw):
        kw.setdefault("min_offload_numel", 1024)
        kw.setdefault("min_offload_bytes", 1)
        kw.setdefault("numa_bind", False)
        return OffloadManager(**kw)

    def _iteration(self, mgr, chunk=0):
        """One forward/backward through a single group. Returns (loss, grads)."""
        self.net.clear_gradients()
        mgr.begin_forward_group(chunk)
        with mgr.scope("main"):
            out = self.net(self.x)
        mgr.clear_current_group()
        mgr.prefetch_next_group(chunk)
        loss = out.sum()
        loss.backward()
        mgr.end_iteration()
        grads = {
            name: p.grad.numpy().copy()
            for name, p in self.net.named_parameters()
            if p.grad is not None
        }
        return float(loss), grads

    def _reference(self):
        """The same iteration with offloading switched off."""
        return self._iteration(self._manager(enabled=False))

    def _assert_bit_exact(self, ref, got, tag=""):
        ref_loss, ref_grads = ref
        got_loss, got_grads = got
        self.assertEqual(ref_loss, got_loss, f"{tag}: loss differs")
        self.assertEqual(set(ref_grads), set(got_grads), f"{tag}: param set")
        for name, want in ref_grads.items():
            self.assertEqual(
                want.tolist(), got_grads[name].tolist(), f"{tag}: grad {name}"
            )


class TestPrefetchModes(_OffloadTestCase):
    def test_budget_mode_prefetches_every_offloaded_tensor(self):
        mgr = self._manager(prefetch_budget_bytes=64 * MB)
        ref = self._reference()
        got = self._iteration(mgr)
        self._assert_bit_exact(ref, got, "budget")
        s = mgr.stats
        self.assertGreater(s["packed"], 0, "nothing was offloaded")
        self.assertEqual(s["unpacked"], s["packed"])
        self.assertEqual(s["exposed"], 0, "every tensor should be prefetched")
        self.assertEqual(s["prefetched"], s["packed"])
        # Whether a copy landed in hit or late is a timing question; that it was
        # accounted exactly once is not.
        self.assertEqual(s["hit"] + s["late"], s["packed"])
        self.assertEqual(s["not_consumed"], 0)

    def test_tight_budget_stalls_but_stays_exact(self):
        # One tensor's worth of budget forces the queue to drain and refill on
        # every consumption, which is the path that decrements _inflight_bytes.
        ref = self._reference()
        mgr = self._manager(prefetch_budget_bytes=1)
        got = self._iteration(mgr)
        self._assert_bit_exact(ref, got, "tight budget")
        self.assertGreater(mgr.stats["prefetched"], 0)

    def test_zero_budget_disables_prefetch(self):
        ref = self._reference()
        mgr = self._manager(prefetch_budget_bytes=0)
        got = self._iteration(mgr)
        self._assert_bit_exact(ref, got, "budget=0")
        self.assertEqual(mgr.stats["prefetched"], 0)
        self.assertEqual(
            mgr.stats["exposed"],
            mgr.stats["packed"],
            "every unpack should fall back to a lazy reload",
        )

    def test_whole_group_mode(self):
        ref = self._reference()
        mgr = self._manager(prefetch_budget_bytes=math.inf)
        got = self._iteration(mgr)
        self._assert_bit_exact(ref, got, "whole group")
        self.assertEqual(mgr.stats["prefetched"], mgr.stats["packed"])
        self.assertEqual(mgr.stats["exposed"], 0)

    def test_sync_mode_is_exact_and_skips_prefetch(self):
        ref = self._reference()
        mgr = self._manager(sync_mode=True, prefetch_budget_bytes=64 * MB)
        got = self._iteration(mgr)
        self._assert_bit_exact(ref, got, "sync_mode")
        self.assertGreater(mgr.stats["packed"], 0)
        self.assertEqual(mgr.stats["prefetched"], 0)

    def test_d2h_on_the_main_stream_is_exact(self):
        ref = self._reference()
        mgr = self._manager(prefetch_budget_bytes=64 * MB)
        mgr.d2h_on_main = True
        got = self._iteration(mgr)
        self._assert_bit_exact(ref, got, "d2h_on_main")
        self.assertGreater(mgr.stats["packed"], 0)


class TestSeparateChannelsAndDegradation(_OffloadTestCase):
    def test_large_tensors_bypass_the_pool_and_the_budget(self):
        # A single very large activation must not be able to consume the whole
        # prefetch budget, so it gets its own allocation and copy.
        ref = self._reference()
        mgr = self._manager(big_tensor_bytes=1, prefetch_budget_bytes=64 * MB)
        got = self._iteration(mgr)
        self._assert_bit_exact(ref, got, "big tensor channel")
        self.assertEqual(mgr.stats["big_packed"], mgr.stats["packed"])
        self.assertEqual(mgr.pool.n_alloc, 0, "pool must be bypassed")

    def test_full_pool_leaves_activations_on_the_device(self):
        # Degrading is the point: a full host pool must not raise.
        ref = self._reference()
        mgr = self._manager(pool_capacity_bytes=1)
        got = self._iteration(mgr)
        self._assert_bit_exact(ref, got, "pool_oom")
        self.assertEqual(mgr.stats["packed"], 0)
        self.assertGreater(mgr.stats["pool_oom"], 0)

    def test_byte_threshold_skips_small_activations(self):
        mgr = self._manager(min_offload_bytes=1 << 30)
        self._iteration(mgr)
        self.assertEqual(mgr.stats["packed"], 0)
        self.assertGreater(mgr.stats["skipped"], 0)

    def test_element_threshold_skips_small_activations(self):
        mgr = self._manager(min_offload_numel=1 << 30, min_offload_bytes=None)
        self._iteration(mgr)
        self.assertEqual(mgr.stats["packed"], 0)

    def test_disabled_manager_passes_tensors_through(self):
        mgr = self._manager(enabled=False)
        self._iteration(mgr)
        self.assertEqual(mgr.stats["packed"], 0)
        self.assertGreater(mgr.stats["skipped"], 0)

    def test_offset_views_are_not_offloaded(self):
        # unpack reapplies the original metadata, offset included, to a fresh
        # buffer, so a non-zero offset would read past the holder.
        mgr = self._manager()
        base = paddle.randn([4096, 64])
        view = base[1:]
        self.assertNotEqual(view._offset(), 0)
        self.assertIs(mgr._pack_impl(view, "main"), view)
        self.assertEqual(mgr.stats["packed"], 0)

    def test_non_contiguous_views_are_not_offloaded(self):
        mgr = self._manager()
        t = paddle.randn([256, 256]).t()
        self.assertFalse(t.is_contiguous())
        self.assertIs(mgr._pack_impl(t, "main"), t)
        self.assertEqual(mgr.stats["packed"], 0)

    def test_parameters_are_not_offloaded(self):
        mgr = self._manager()
        weight = self.net.fc1.weight
        self.assertIs(mgr._pack_impl(weight, "main"), weight)
        self.assertEqual(mgr.stats["packed"], 0)

    def test_non_tensors_pass_straight_through(self):
        mgr = self._manager()
        self.assertIsNone(mgr._pack_impl(None, "main"))
        self.assertEqual(mgr._pack_impl(7, "main"), 7)

    def test_reload_of_an_empty_batch_is_a_no_op(self):
        mgr = self._manager()
        mgr._reload_batch([])
        self.assertEqual(mgr.stats["h2d_batches"], 0)


class TestDedup(_OffloadTestCase):
    NET_CLS = _SharedInput

    def test_one_activation_saved_by_three_operators_is_copied_once(self):
        ref = self._reference()
        mgr = self._manager(prefetch_budget_bytes=64 * MB)
        got = self._iteration(mgr)
        self._assert_bit_exact(ref, got, "dedup")
        # Three linears share the input, so pack fires on it three times and two
        # of those must collapse onto the first record.
        self.assertGreaterEqual(mgr.stats["dedup_hits"], 2)
        self.assertGreater(mgr.stats["dedup_bytes_saved"], 0)

    def test_deduplicated_records_are_only_released_once(self):
        # The pinned buffer is shared by every slot, so returning it before the
        # last unpack would hand a live buffer to the next allocation.
        mgr = self._manager(prefetch_budget_bytes=64 * MB)
        self._iteration(mgr)
        self.assertEqual(mgr.stats["not_consumed"], 0)
        self.assertEqual(mgr.pool.in_use_bytes, 0)
        self.assertEqual(mgr.pool.n_free_dup, 0)


class TestLearningAndPolicy(_OffloadTestCase):
    def test_fraction_keeps_later_boundaries_resident(self):
        # The skip set can only be computed once a forward has been observed, so
        # the first iteration offloads everything and only measures.
        mgr = self._manager(fraction=0.5, prefetch_budget_bytes=64 * MB)
        self._iteration(mgr)
        self.assertEqual(mgr.stats["skipped_policy"], 0, "learning iteration")
        self.assertGreater(mgr.stats["learned_boundaries"], 0)
        self.assertGreater(mgr.stats["skipped_boundaries"], 0)
        learned = mgr.stats["packed"]

        mgr.reset_stats()
        self._iteration(mgr)
        self.assertGreater(mgr.stats["skipped_policy"], 0)
        self.assertLess(mgr.stats["packed"], learned)

    def test_fraction_zero_offloads_nothing_after_learning(self):
        mgr = self._manager(fraction=0.0, prefetch_budget_bytes=64 * MB)
        self._iteration(mgr)
        mgr.reset_stats()
        self._iteration(mgr)
        self.assertEqual(mgr.stats["packed"], 0)

    def test_pp_delta_offloads_less_on_higher_ranks(self):
        def learned_packed(pp_rank):
            mgr = self._manager(
                delta_offload_bytes_across_pp_ranks=4 * MB,
                pp_rank=pp_rank,
                prefetch_budget_bytes=64 * MB,
            )
            self._iteration(mgr)  # learning
            mgr.reset_stats()
            self._iteration(mgr)
            return mgr.stats["packed"]

        self.assertLess(learned_packed(1), learned_packed(0))

    def test_auto_budget_starts_from_the_group_size_and_grows(self):
        mgr = self._manager(prefetch_budget_bytes=None)
        self.assertTrue(mgr.auto_budget)
        self._iteration(mgr)  # learning iteration picks the initial budget
        initial = mgr.prefetch_budget_bytes
        self.assertIsNotNone(initial)
        self.assertGreater(initial, 0)
        budgets = [initial]
        for _ in range(4):
            self._iteration(mgr)
            budgets.append(mgr.prefetch_budget_bytes)
        self.assertEqual(budgets, sorted(budgets), "budget must not shrink")
        self.assertGreater(mgr.stats["budget_raises"], 0)
        # It never grows past a whole group: beyond that there is nothing left
        # to prefetch.
        self.assertLessEqual(mgr.prefetch_budget_bytes, mgr._group_bytes)

    def test_auto_budget_never_grows_past_a_whole_group(self):
        mgr = self._manager(prefetch_budget_bytes=None)
        self._iteration(mgr)  # learning iteration fixes _group_bytes
        mgr.prefetch_budget_bytes = mgr._group_bytes
        raises = mgr.stats["budget_raises"]
        self._iteration(mgr)
        self.assertEqual(mgr.prefetch_budget_bytes, mgr._group_bytes)
        self.assertEqual(mgr.stats["budget_raises"], raises)

    def test_pp_delta_stops_once_its_allowance_is_spent(self):
        # A delta smaller than one activation is exhausted by the first boundary
        # it keeps resident; the rest are still offloaded.
        mgr = self._manager(
            delta_offload_bytes_across_pp_ranks=1,
            pp_rank=1,
            prefetch_budget_bytes=64 * MB,
        )
        self._iteration(mgr)
        self.assertEqual(mgr.stats["skipped_boundaries"], 1)
        mgr.reset_stats()
        self._iteration(mgr)
        self.assertEqual(mgr.stats["skipped_policy"], 1)
        self.assertGreater(mgr.stats["packed"], 0)

    def test_pp_delta_skips_boundaries_fraction_already_took(self):
        # Both knobs keep the later boundaries resident, so they overlap; the
        # delta must not count a boundary fraction already dropped.
        mgr = self._manager(
            fraction=0.5,
            delta_offload_bytes_across_pp_ranks=1,
            pp_rank=1,
            prefetch_budget_bytes=64 * MB,
        )
        self._iteration(mgr)
        learned = mgr.stats["learned_boundaries"]
        skipped = mgr.stats["skipped_boundaries"]
        self.assertGreater(learned, 1)
        self.assertGreater(skipped, 0)
        self.assertLessEqual(skipped, learned)

    def test_budget_tuning_respects_the_group_ceiling(self):
        # Driven directly: reaching the ceiling through real iterations depends
        # on how many stalls the device happens to produce.
        mgr = self._manager(prefetch_budget_bytes=None)
        mgr._group_bytes = 1000
        mgr.prefetch_budget_bytes = 1000
        mgr.stats["late"] = 5  # stalls grew since the last iteration
        mgr._tune_budget()
        self.assertEqual(mgr.prefetch_budget_bytes, 1000)
        self.assertEqual(mgr.stats["budget_raises"], 0)

    def test_budget_tuning_doubles_towards_the_ceiling(self):
        mgr = self._manager(prefetch_budget_bytes=None)
        mgr._group_bytes = 1000
        mgr.prefetch_budget_bytes = 300
        mgr.stats["late"] = 5
        mgr._tune_budget()
        self.assertEqual(mgr.prefetch_budget_bytes, 600)
        mgr.stats["late"] = 9
        mgr._tune_budget()
        self.assertEqual(mgr.prefetch_budget_bytes, 1000, "clamped to a group")
        self.assertEqual(mgr.stats["budget_raises"], 2)


class TestGrouping(_OffloadTestCase):
    def _two_group_iteration(self, mgr):
        """Forward two groups, then run their backwards in order.

        This is what the pipeline schedule does, played by hand: both chunks run
        forward before either runs backward, so ``prefetch_next_group_head`` has
        a complete group to queue.
        """
        self.net.clear_gradients()
        outs = []
        for chunk in (0, 1):
            mgr.begin_forward_group(chunk)
            with mgr.scope("main"):
                outs.append(self.net(self.x))
            mgr.clear_current_group()
        for chunk, out in enumerate(outs):
            mgr.prefetch_next_group(chunk)
            out.sum().backward()
            mgr.prefetch_next_group_head(chunk)
        mgr.end_iteration()

    def test_backward_order_is_recorded_then_used(self):
        mgr = self._manager(prefetch_budget_bytes=64 * MB)
        self.assertEqual(mgr._bwd_next, {})
        self._two_group_iteration(mgr)
        # The order is a function of the schedule, so one observed iteration is
        # enough and it is frozen at the iteration boundary.
        self.assertEqual(mgr._bwd_next, {(0, 0): (1, 0)})
        self.assertEqual(
            mgr.stats["head_prefetch"], 0, "no cross-group pull while learning"
        )

        mgr.reset_stats()
        self._two_group_iteration(mgr)
        self.assertGreater(
            mgr.stats["head_prefetch"], 0, "next group should be queued"
        )

    def test_a_single_group_records_no_successor(self):
        # With one group there is nothing to pull forward, so no table is built.
        mgr = self._manager(prefetch_budget_bytes=64 * MB)
        self._iteration(mgr)
        self.assertEqual(mgr._bwd_next, {})

    def test_cross_group_prefetch_can_be_switched_off(self):
        mgr = self._manager(
            prefetch_budget_bytes=64 * MB, cross_group_prefetch=False
        )
        self._two_group_iteration(mgr)
        mgr.reset_stats()
        self._two_group_iteration(mgr)
        self.assertEqual(mgr.stats["head_prefetch"], 0)

    def test_head_prefetch_is_inert_outside_budget_mode(self):
        mgr = self._manager(prefetch_budget_bytes=math.inf)
        self._two_group_iteration(mgr)
        mgr.reset_stats()
        self._two_group_iteration(mgr)
        self.assertEqual(mgr.stats["head_prefetch"], 0)

    def test_queueing_a_group_is_idempotent(self):
        mgr = self._manager(prefetch_budget_bytes=64 * MB)
        mgr.begin_forward_group(0)
        with mgr.scope("main"):
            out = self.net(self.x)
        mgr.clear_current_group()
        self.assertTrue(mgr._enqueue_group(0, 0))
        queued = list(mgr._pending[(0, 0)])
        self.assertTrue(mgr._enqueue_group(0, 0), "already queued")
        self.assertEqual(list(mgr._pending[(0, 0)]), queued)
        # A group whose forward has not been seen cannot be queued.
        self.assertFalse(mgr._enqueue_group(9, 9))
        out.sum().backward()
        mgr.end_iteration()

    def test_state_is_cleared_between_iterations(self):
        mgr = self._manager(prefetch_budget_bytes=64 * MB)
        self._iteration(mgr)
        self.assertEqual(mgr._records, {})
        self.assertEqual(mgr._groups, {})
        self.assertEqual(mgr._pending, {})
        self.assertEqual(mgr._inflight_bytes, 0)
        self.assertIsNone(mgr._active_bwd_gk)

    def test_pinned_water_mark_is_flat_across_iterations(self):
        # A pool that grows every step is the failure this bucketing prevents.
        mgr = self._manager(prefetch_budget_bytes=64 * MB)
        self._iteration(mgr)
        self._iteration(mgr)
        after_second = mgr.pool.total_bytes
        self._iteration(mgr)
        self.assertEqual(mgr.pool.total_bytes, after_second)

    def test_the_group_being_consumed_is_served_first(self):
        """Cross-group prefetch must not starve the group running backward.

        ``_pending`` is insertion-ordered, so a group queued ahead of the active
        one would be served first and spend the budget on tensors that are
        several backward steps away.
        """
        mgr = self._manager(prefetch_budget_bytes=64 * MB)
        self.net.clear_gradients()
        outs = []
        for chunk in (0, 1):
            mgr.begin_forward_group(chunk)
            with mgr.scope("main"):
                outs.append(self.net(self.x))
            mgr.clear_current_group()

        # Queue the future group first, then make the other one active.
        mgr._enqueue_group(1, 0)
        mgr._active_bwd_gk = (0, 0)
        mgr._enqueue_group(0, 0)
        self.assertEqual(
            next(iter(mgr._pending)), (1, 0), "future group is first"
        )

        # Only enough budget for one tensor, so whichever group wins is visible.
        smallest = min(mgr._records[pid].nbytes for pid in mgr._groups[(0, 0)])
        mgr.prefetch_budget_bytes = smallest
        mgr._fill_budget()

        served = [
            pid
            for pid in mgr._groups[(0, 0)]
            if mgr._records[pid].gpu is not None
        ]
        starved = [
            pid
            for pid in mgr._groups[(1, 0)]
            if mgr._records[pid].gpu is not None
        ]
        self.assertTrue(served, "the active group should have been served")
        self.assertFalse(starved, "the future group should have waited")

        mgr.prefetch_budget_bytes = 64 * MB
        for out in outs:
            out.sum().backward()
        mgr.end_iteration()


class TestDiagnostics(_OffloadTestCase):
    def test_profile_records_host_time(self):
        mgr = self._manager(prefetch_budget_bytes=1)
        mgr.profile = True
        self._iteration(mgr)
        self.assertGreater(mgr.stats["pack_ms"], 0.0)
        self.assertGreater(mgr.stats["unpack_ms"], 0.0)

    def test_unpack_flushes_a_still_batched_copy(self):
        """A tensor whose D2H has not been issued yet must be flushed on read.

        With batching enabled the copy is deferred, so unpack has to force it out
        before reading the pinned buffer -- otherwise it would read uninitialised
        host memory.
        """
        mgr = self._manager(prefetch_budget_bytes=0)
        mgr.max_pending_bytes = 1 << 40  # never auto-flush
        src = paddle.randn([1024, 1024])
        cpu = mgr._pack_impl(src, "main")
        self.assertIsNot(cpu, src, "the tensor should have been offloaded")
        self.assertTrue(mgr._pending_packs, "the copy is still batched")

        back = mgr._unpack_impl(cpu)
        self.assertFalse(mgr._pending_packs, "unpack must flush first")
        paddle.device.synchronize()
        self.assertTrue(bool((back == src).all()), "reloaded data must match")

    def test_periodic_stats_line(self):
        mgr = self._manager(prefetch_budget_bytes=64 * MB)
        mgr._log_every = 1
        with self.assertLogs(
            "paddlefleet.activation_offload.manager", level="INFO"
        ) as logs:
            self._iteration(mgr)
        self.assertGreater(
            sum("activation-offload" in m for m in logs.output),
            1,
            "the first pack plus at least one periodic line",
        )

    def test_unconsumed_registrations_are_reported(self):
        # Offloading in forward and never running backward leaks records; the
        # counter is the only sign, so it has to reach the log.
        mgr = self._manager(prefetch_budget_bytes=64 * MB)
        mgr.begin_forward_group(0)
        with mgr.scope("main"):
            self.net(self.x)
        mgr.clear_current_group()
        with self.assertLogs(
            "paddlefleet.activation_offload.manager", level="WARNING"
        ) as logs:
            mgr.end_iteration()
        self.assertGreater(mgr.stats["not_consumed"], 0)
        self.assertTrue(any("never consumed" in m for m in logs.output))

    def test_full_pool_is_reported(self):
        mgr = self._manager(pool_capacity_bytes=1)
        with self.assertLogs(
            "paddlefleet.activation_offload.manager", level="WARNING"
        ) as logs:
            self._iteration(mgr)
        self.assertTrue(any("pinned pool" in m for m in logs.output))

    def test_stalls_are_reported_only_when_the_budget_is_pinned(self):
        # With an adaptive budget a stall is a normal transient the controller
        # will act on, so warning about it would be noise.
        mgr = self._manager(prefetch_budget_bytes=0)
        with self.assertLogs(
            "paddlefleet.activation_offload.manager", level="WARNING"
        ) as logs:
            self._iteration(mgr)
        self.assertTrue(any("not prefetched" in m for m in logs.output))

    def test_each_degradation_is_warned_about_once(self):
        mgr = self._manager(pool_capacity_bytes=1)
        with self.assertLogs(
            "paddlefleet.activation_offload.manager", level="WARNING"
        ):
            self._iteration(mgr)
        self.assertIn("pool_oom", mgr._warned)
        # A second iteration hits the same degradation and must stay quiet.
        with mock.patch.object(mgr, "_warn_once", wraps=mgr._warn_once) as warn:
            self._iteration(mgr)
        for call in warn.call_args_list:
            self.assertIn(call.args[0], mgr._warned)

    def test_format_stats_reports_the_counters_it_documents(self):
        mgr = self._manager(prefetch_budget_bytes=64 * MB)
        self._iteration(mgr)
        line = mgr.format_stats()
        for field in (
            "packed=",
            "prefetched=",
            "hit=",
            "late=",
            "exposed=",
            "head_pf=",
            "dedup=",
            "budget=",
            "pool_oom=",
            "not_consumed=",
            "pinned=",
        ):
            self.assertIn(field, line)
        self.assertIn("by group main=", line)

    def test_format_stats_on_an_untouched_manager(self):
        mgr = self._manager(enabled=False)
        self.assertIn("packed=0", mgr.format_stats())
        self.assertEqual(mgr._by_group_str(), "")

    def test_format_stats_survives_an_unavailable_allocator(self):
        # Logging must never be the thing that breaks a run.
        mgr = self._manager(enabled=False)
        with mock.patch(
            "paddle.device.cuda.memory_allocated", side_effect=RuntimeError
        ):
            line = mgr.format_stats()
        self.assertIn("packed=0", line)
        self.assertNotIn("mem cur=", line)

    def test_skipped_reporting_survives_an_odd_tensor(self):
        mgr = self._manager()
        broken = mock.Mock()
        broken.element_size.side_effect = RuntimeError
        mgr._note_skipped(broken, 1024)  # must not raise
        self.assertEqual(mgr._max_skipped_bytes, 0)

    def test_largest_skipped_activation_is_reported(self):
        # The element threshold, not the byte one: the report exists to name the
        # activation that a predicate left on the device.
        mgr = self._manager(min_offload_numel=1 << 30)
        with self.assertLogs(
            "paddlefleet.activation_offload.manager", level="INFO"
        ) as logs:
            self._iteration(mgr)
        self.assertTrue(
            any("largest activation skipped" in m for m in logs.output)
        )

    def test_parameters_are_not_reported_as_skipped_activations(self):
        """A weight is never a candidate, so it must not enter that report.

        Weights reach pack like anything else a backward needs, and they are
        usually the largest tensors around. Naming one would send the reader
        hunting for memory that offloading could never have saved -- and this
        fires on a plain Linear, so it is the common case, not a corner one.
        """
        mgr = self._manager()
        weight = self.net.fc1.weight
        mgr._pack_impl(weight, "main")
        self.assertEqual(mgr.stats["skipped"], 1)
        self.assertEqual(mgr._max_skipped_bytes, 0)
        self.assertEqual(mgr._max_skipped_desc, "")


class TestConfigReaders(unittest.TestCase):
    """These read a config object only; no device work involved."""

    def test_disabled_config_offloads_no_regions(self):
        cfg = _StubConfig(fine_grained_activation_offloading=False)
        self.assertFalse(offload_enabled(cfg))
        self.assertEqual(offload_groups(cfg), frozenset())

    def test_explicit_module_list_is_honoured(self):
        cfg = _StubConfig(
            fine_grained_activation_offloading=True,
            offload_modules=["core_attn", "mlp_norm"],
        )
        self.assertEqual(
            offload_groups(cfg), frozenset({"core_attn", "mlp_norm"})
        )

    def test_enabled_without_a_list_offloads_every_supported_region(self):
        # "Enabled but unspecified" means all of them.
        cfg = _StubConfig(
            fine_grained_activation_offloading=True, offload_modules=None
        )
        self.assertEqual(offload_groups(cfg), _StubConfig._OFFLOAD_MODULE_NAMES)

    def test_empty_list_also_falls_back_to_every_region(self):
        cfg = _StubConfig(
            fine_grained_activation_offloading=True, offload_modules=[]
        )
        self.assertEqual(offload_groups(cfg), _StubConfig._OFFLOAD_MODULE_NAMES)

    def test_byte_threshold_disables_the_element_threshold(self):
        # The two are combined with AND, so leaving the element default in place
        # would override any byte threshold the config asks for.
        cfg = _StubConfig(
            fine_grained_activation_offloading=True,
            min_offloaded_tensor_bytes=4096,
        )
        kwargs = offload_kwargs_from_config(cfg)
        self.assertEqual(kwargs["min_offload_bytes"], 4096)
        self.assertEqual(kwargs["min_offload_numel"], 0)

    def test_element_threshold_kept_when_no_byte_threshold_is_given(self):
        cfg = _StubConfig(fine_grained_activation_offloading=True)
        kwargs = offload_kwargs_from_config(cfg)
        self.assertIsNone(kwargs["min_offload_bytes"])
        self.assertEqual(kwargs["min_offload_numel"], 256 * 1024)

    def test_knobs_are_forwarded_with_their_defaults(self):
        cfg = _StubConfig(fine_grained_activation_offloading=True)
        kwargs = offload_kwargs_from_config(cfg)
        self.assertEqual(kwargs["fraction"], 1.0)
        self.assertEqual(kwargs["delta_offload_bytes_across_pp_ranks"], 0)
        self.assertEqual(kwargs["pp_rank"], 0)
        self.assertIsNone(kwargs["prefetch_budget_bytes"])
        self.assertIsNone(kwargs["pool_capacity_bytes"])
        self.assertTrue(kwargs["numa_bind"])
        self.assertTrue(kwargs["cross_group_prefetch"])

    def test_explicit_pp_rank_wins_over_the_config(self):
        cfg = _StubConfig(
            fine_grained_activation_offloading=True,
            pipeline_model_parallel_rank=2,
        )
        self.assertEqual(offload_kwargs_from_config(cfg)["pp_rank"], 2)
        self.assertEqual(
            offload_kwargs_from_config(cfg, pp_rank=5)["pp_rank"], 5
        )

    def test_manager_from_config_builds_the_singleton(self):
        reset_offload_manager()
        try:
            cfg = _StubConfig(
                fine_grained_activation_offloading=True,
                min_offloaded_tensor_bytes=4096,
                activation_offload_numa_bind=False,
            )
            mgr = manager_from_config(cfg)
            self.assertIs(current_offload_manager(), mgr)
            self.assertTrue(mgr.enabled)
            self.assertEqual(mgr.min_offload_bytes, 4096)
            self.assertEqual(mgr.min_offload_numel, 0)
        finally:
            reset_offload_manager()


@_REQUIRE_GPU
class TestSingletonAndRegion(unittest.TestCase):
    def setUp(self):
        reset_offload_manager()

    def tearDown(self):
        reset_offload_manager()

    def test_region_is_inert_when_disabled(self):
        with offload_region(False, "core_attn"):
            pass  # must not need a manager at all

    def test_region_is_inert_before_a_manager_exists(self):
        # Model code should not have to care about construction order.
        self.assertIsNone(current_offload_manager())
        with offload_region(True, "core_attn"):
            pass

    def test_region_uses_the_singleton_once_it_exists(self):
        from paddlefleet.activation_offload import get_offload_manager

        mgr = get_offload_manager(
            min_offload_numel=1024, min_offload_bytes=1, numa_bind=False
        )
        self.assertIs(current_offload_manager(), mgr)
        x = paddle.randn([4, 512, 512])
        net = paddle.nn.Linear(512, 512)
        mgr.begin_forward_group(0)
        with offload_region(True, "core_attn"):
            out = net(x)
        mgr.clear_current_group()
        mgr.prefetch_next_group(0)
        out.sum().backward()
        mgr.end_iteration()
        self.assertGreater(mgr.stats["packed"], 0)
        self.assertIn("core_attn", mgr._bytes_by_group)

    def test_singleton_keeps_the_first_kwargs(self):
        from paddlefleet.activation_offload import get_offload_manager

        first = get_offload_manager(numa_bind=False, min_offload_numel=7)
        second = get_offload_manager(numa_bind=False, min_offload_numel=99)
        self.assertIs(first, second)
        self.assertEqual(second.min_offload_numel, 7)

    def test_reset_lets_a_new_configuration_take_effect(self):
        from paddlefleet.activation_offload import get_offload_manager

        first = get_offload_manager(numa_bind=False, min_offload_numel=7)
        reset_offload_manager()
        self.assertIsNone(current_offload_manager())
        second = get_offload_manager(numa_bind=False, min_offload_numel=99)
        self.assertIsNot(first, second)
        self.assertEqual(second.min_offload_numel, 99)

    def test_numa_bind_runs_before_any_pinned_allocation(self):
        # Patched out rather than exercised: binding for real would restrict
        # every later test in this process to one node's cores.
        with mock.patch(
            "paddlefleet.activation_offload.manager._numa_bind"
        ) as bind:
            OffloadManager(numa_bind=True)
        bind.assert_called_once_with()

    def test_numa_bind_is_skipped_when_disabled(self):
        with mock.patch(
            "paddlefleet.activation_offload.manager._numa_bind"
        ) as bind:
            OffloadManager(numa_bind=False)
        bind.assert_not_called()


if __name__ == "__main__":
    unittest.main()
