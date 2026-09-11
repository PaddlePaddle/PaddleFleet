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

"""Activation offloading core: saved_tensors_hooks pack/unpack and group prefetch.

Properties the implementation relies on:

- pack only fires inside a forward ``scope()``. It returns a pinned tensor and
  keeps the metadata in a side table keyed by ``id(pinned)``, so unpack finds its
  record by object identity and cannot mismatch.
- Groups are keyed by ``(chunk_id, micro-batch index within the chunk)``. The
  pipeline runtime's ``step_id`` cannot pair a forward group with its backward,
  because forward and backward step ids are numbered independently; each chunk
  keeps its own counters, cleared by ``end_iteration()``.
- Prefetch issues H2D in reverse save order, since the last tensor saved is the
  first one backward consumes.
- Within a region, tensors are deduplicated by ``data_ptr()``: an activation saved
  by several operators makes pack fire once per operator.
- ``hit`` and ``late`` are counted unconditionally, because ``exposed`` only sees
  tensors that were never prefetched and cannot observe a stall caused by too
  small a budget.
- A reload destination must be allocated from the consumer stream's allocator
  pool. Allocating inside the H2D stream guard draws from a pool whose blocks can
  have different alignment, which lets cuBLAS pick a different reduction order and
  shifts results by a rounding step -- enough to break a bitwise comparison.
- Sliced views with a non-zero ``_offset()`` are not offloaded: unpack reapplies
  the original metadata, offset included, to a fresh buffer, and offset plus numel
  then exceeds the holder. Non-contiguous views are not offloaded either, since
  the layout would not survive the round trip.
- ``sync_mode=True`` makes every copy synchronous for debugging, and disables
  prefetch so it cannot race with those copies.
"""

from __future__ import annotations

import logging
import math
import os
import time
from contextlib import nullcontext

import paddle
from paddle.base.framework import EagerParamBase

from .numa_bind import bind as _numa_bind
from .pinned_pool import PinnedPool
from .pylayer_shim import install as _install_pylayer_shim

logger = logging.getLogger(__name__)


def offload_region(enabled, name):
    """Wrap a module's forward so that everything it saves is offloaded.

    A region name denotes an execution span, not a particular tensor::

        with offload_region(self.offload_qkv_linear, "qkv_linear"):
            mixed_qkv, _ = self.qkv_proj(hidden_states)

    Wrapping the module is what makes the offloaded set equal to the activations
    its backward needs. Tagging the output instead would be wrong: that tensor is
    the *next* module's input and belongs to the next region.

    Returns ``nullcontext()`` when disabled or before a manager exists, so model
    code need not care about construction order.
    """
    if not enabled:
        return nullcontext()
    mgr = current_offload_manager()
    if mgr is None:
        return nullcontext()
    return mgr.scope(name)


class _Record:
    __slots__ = (
        "cpu",
        "shape",
        "dtype",
        "stop_gradient",
        "off_event",
        "gpu",
        "reload_event",
        "group_key",
        "big",
        "nbytes",
        "counted",
        "batch_id",
        "refs",
        "probed",
    )

    def __init__(self, **kw):
        self.gpu = None
        self.reload_event = None
        self.big = False
        self.counted = False  # already counted towards _inflight_bytes
        # Reload batch this record belongs to; the main stream waits once per batch.
        self.batch_id = -1
        # One activation may be saved by several operators, so pack fires once per
        # operator. After dedup those slots share a single record, and the pinned
        # buffer can only be returned once every one of them has been unpacked.
        self.refs = 1
        self.probed = False  # hit/late are counted on first consumption only
        for k, v in kw.items():
            setattr(self, k, v)


class OffloadManager:
    """Process-wide singleton, obtained through ``get_offload_manager()``.

    Args:
        min_offload_numel: tensors with fewer elements than this are left alone.
        enabled: master switch; when false pack passes tensors through.
        sync_mode: synchronous debugging mode (see the module docstring).
        big_tensor_bytes: tensors above this size get their own pinned allocation
            and copy, bypassing the pool and the prefetch budget, so that a single
            very large activation cannot consume the whole budget at once. ``None``
            (the default) sends everything through the pool.
        pool_capacity_bytes: host memory limit for the pinned pool. A tensor that
            does not fit is left on the device instead of raising.
        prefetch_budget_bytes: byte ceiling on activations reloaded but not yet
            consumed. Peak device memory is roughly this plus the lazy-path floor,
            which makes it the main memory/latency knob. ``None`` (the default)
            picks a value at the end of the first iteration from the observed group
            size; ``0`` disables prefetch so every unpack waits for its own H2D;
            ``math.inf`` reloads a whole group at a time. Whole-group reload costs
            more device memory at every shape measured, hence not the default.
        min_offload_bytes: byte threshold, superseding the element threshold. The
            host cost of pack is per tensor while the benefit scales with bytes,
            and an element threshold would skip half the traffic for low-precision
            activations.
        fraction: offload only the first N% of offloadable tensors in forward
            order (0 to 1), counting tensors rather than bytes. For when bandwidth
            rather than memory is the constraint.
        delta_offload_bytes_across_pp_ranks / pp_rank: offload ``pp_rank * delta``
            fewer bytes per pipeline rank, since higher ranks hold fewer in-flight
            micro-batches and need the memory less.
        numa_bind: bind to the cores local to this GPU before allocating pinned
            memory (see the ``numa_bind`` module).
        cross_group_prefetch: at ``BACKWARD_END``, also queue the group that runs
            backward next (see ``prefetch_next_group_head``). On by default:
            otherwise the first tensor a group consumes has no preceding
            computation to hide its copy behind. Still capped by
            ``prefetch_budget_bytes``, so it cannot raise peak memory. Only
            meaningful in budget mode.
    """

    def __init__(
        self,
        min_offload_numel=256 * 1024,
        enabled=True,
        sync_mode=False,
        big_tensor_bytes=None,
        pool_capacity_bytes=None,
        prefetch_budget_bytes=None,
        min_offload_bytes=None,
        fraction=1.0,
        delta_offload_bytes_across_pp_ranks=0,
        pp_rank=0,
        numa_bind=True,
        cross_group_prefetch=True,
    ):
        _install_pylayer_shim()
        if numa_bind:
            # Must precede the first pin_memory. It does not have to precede CUDA
            # context creation, since page ownership follows the affinity in
            # effect when the pages are allocated.
            _numa_bind()
        self.d2h_stream = paddle.device.Stream()
        self.h2d_stream = paddle.device.Stream()
        self.pool = PinnedPool(capacity_bytes=pool_capacity_bytes)
        self.min_offload_numel = min_offload_numel
        self.min_offload_bytes = min_offload_bytes
        self.fraction = fraction
        self.pp_delta_bytes = delta_offload_bytes_across_pp_ranks
        self.pp_rank = pp_rank
        self.enabled = enabled
        self.sync_mode = sync_mode
        self.big_tensor_bytes = big_tensor_bytes
        self.prefetch_budget_bytes = prefetch_budget_bytes
        self.cross_group_prefetch = cross_group_prefetch
        # Measure pack/unpack host time and how long a late reload blocks.
        self.profile = False
        self.unsafe_no_alloc_fence = False  # diagnostic only; see _reload_batch
        # Issuing D2H on the main stream needs no cross-stream synchronization but
        # occupies the main stream and cannot overlap with compute; measured
        # slower than the side stream, so it stays off. Kept as an escape hatch.
        self.d2h_on_main = False
        # Log a stats line every N packed tensors (0 = only the first one).
        self._log_every = int(os.environ.get("ACT_OFFLOAD_LOG_EVERY", "200"))
        # A large tensor filtered out by the thresholds is the easiest way to end
        # up paying for offloading without saving anything, so remember the
        # largest one skipped and report it once per iteration.
        self._max_skipped_bytes = 0
        self._max_skipped_desc = ""
        self._bytes_by_group: dict = {}  # region name -> bytes offloaded
        self._records: dict = {}  # id(pinned) -> _Record
        # Per-region dedup table: (data_ptr, shape, dtype) -> (_Record, source).
        # Valid inside one region only; _ScopeGuard swaps it on entry and exit.
        self._dedup: dict = {}
        self._groups: dict = {}  # (chunk, seq) -> [id(pinned)...] in save order
        self._pending: dict = {}  # (chunk, seq) -> [id...] awaiting prefetch
        self._pending_packs: list = []  # [(rec, src)] batched D2H not yet issued
        self._pending_bytes = 0
        # 0 = issue D2H as soon as each tensor is packed. Batching further saves
        # cross-stream synchronization but holds the source tensors longer, which
        # costs more than it saves; >0 accumulates to that many bytes first.
        self.max_pending_bytes = 0
        # Reload batches the main stream has already waited on.
        self._waited_batches: set = set()
        self._batch_seq = 0  # reload batch id counter
        self._inflight_bytes = 0  # reloaded but not yet consumed
        self._active_bwd_gk = None  # group currently running backward
        # Cross-group prefetch needs to know the globally next group to run
        # backward. Under VPP that is not the next micro-batch of this chunk, so
        # the order is recorded during the first iteration and looked up after
        # (see prefetch_next_group_head).
        self._bwd_order: list = []  # backward group order, while learning
        self._bwd_next: dict = {}  # group -> next group; non-empty once recorded
        self._cur_group_key = None
        self._fwd_seq: dict = {}  # chunk -> micro-batches started in forward
        self._bwd_seq: dict = {}  # chunk -> micro-batches started in backward
        # fraction and pp_delta need to know how many boundaries a micro-batch has
        # and how large each one is, which is only known once forward has run. The
        # first iteration therefore offloads everything and only observes; at the
        # end of it the set of boundary indices to skip is computed and reused,
        # since the boundary sequence is stable in steady state.
        self._boundary_seq = 0  # boundary index within this micro-batch
        self._learn_bytes: dict = {}  # boundary index -> bytes, while learning
        self._learned = False
        self._skip_boundaries: frozenset = frozenset()
        self._group_bytes = 0  # bytes one micro-batch group actually offloads
        # prefetch_budget_bytes=None means "let _tune_budget pick it"; any explicit
        # value, 0 and math.inf included, is honoured as given and never adjusted.
        self.auto_budget = prefetch_budget_bytes is None
        # Stalls seen last iteration (late + exposed).
        self._budget_probe_stalls = 0
        self._warned: set = set()  # degradation warnings already emitted
        self.reset_stats()

    # ---------------- pipeline schedule callbacks ----------------

    def begin_forward_group(self, chunk_id):
        """FORWARD_BEGIN: open a group keyed by (chunk, micro-batch index)."""
        seq = self._fwd_seq.get(chunk_id, 0)
        self._fwd_seq[chunk_id] = seq + 1
        self._cur_group_key = (chunk_id, seq)
        # Boundary indices restart per micro-batch: the fraction and pp_delta
        # decisions only carry across iterations because the same micro-batch sees
        # the same boundary sequence.
        self._boundary_seq = 0

    def clear_current_group(self):
        """FORWARD_END: close the current group."""
        self._cur_group_key = None

    def prefetch_next_group(self, chunk_id):
        """BACKWARD_BEGIN: prefetch this chunk's next micro-batch group.

        Backward within a chunk runs micro-batches strictly in order.
        """
        seq = self._bwd_seq.get(chunk_id, 0)
        self._bwd_seq[chunk_id] = seq + 1
        # Record the global backward group order during the first iteration, for
        # cross-group prefetch (see prefetch_next_group_head). Under VPP the next
        # group to run backward is not this chunk's next micro-batch, so it has to
        # be observed rather than derived.
        if not self._bwd_next:
            self._bwd_order.append((chunk_id, seq))
        self.prefetch_group(chunk_id, seq)

    def _in_budget_mode(self) -> bool:
        """Whether bounded-budget prefetch is active.

        Whole-group mode (``None`` before a budget has been learned, or an
        explicit ``inf``) reloads a group at a time, so there is nothing to pull
        forward; ``0`` disables prefetch outright; ``sync_mode`` would make
        prefetch race with the synchronous copies in unpack. ``prefetch_group``
        and ``prefetch_next_group_head`` share this test so the ``0`` / ``None`` /
        ``inf`` semantics cannot drift apart.
        """
        if self.sync_mode:
            return False
        b = self.prefetch_budget_bytes
        return b not in (0, None) and b != math.inf

    def prefetch_group(self, chunk_id, seq):
        """Start reloading a group.

        In whole-group mode the group is reloaded in reverse save order in one
        idempotent batch. In budget mode it is queued in consumption order and
        issued in instalments that fit the budget.

        Whole-group mode costs more device memory at every shape measured, so
        ``None`` only means "no budget learned yet" and ``_finish_learning``
        replaces it with a bounded budget at the end of the first iteration. Pass
        ``math.inf`` explicitly to keep whole-group behavior.
        """
        if self._in_budget_mode():
            # Consumption order is reverse save order.
            self._active_bwd_gk = (chunk_id, seq)
            self._enqueue_group(chunk_id, seq)
            self._fill_budget()
            return
        if self.sync_mode or self.prefetch_budget_bytes == 0:
            return  # everything falls back to lazy reload in unpack
        recs = [
            self._records.get(pid)
            for pid in reversed(self._groups.get((chunk_id, seq), []))
        ]
        recs = [r for r in recs if r is not None and r.gpu is None]
        if recs:
            self._reload_batch(recs)  # one batch, one shared event
            self.stats["prefetched"] += len(recs)

    def _enqueue_group(self, chunk_id, seq) -> bool:
        """Queue a group in consumption order. Idempotent.

        Returns False if the group is absent from ``_groups``, which is the same as
        "forward has not finished it": forward and backward micro-steps run
        serially on one Python thread, so a present key always means a complete
        group. Idempotence keeps a re-queue from pushing already-consumed ids back
        onto the queue.
        """
        gk = (chunk_id, seq)
        if gk in self._pending:
            return True
        ids = self._groups.get(gk)
        if not ids:
            return False
        self._pending[gk] = list(reversed(ids))
        return True

    def prefetch_next_group_head(self, chunk_id=None):
        """BACKWARD_END: also queue the next group to run backward.

        ``BACKWARD_BEGIN`` is the only other anchor, so without this the first
        tensor each group consumes has no preceding computation to hide its copy
        behind. Queuing the next group early lets its leading tensors go out during
        the tail of this group's backward.

        The next group cannot be derived as ``(chunk_id, _bwd_seq[chunk_id])``:
        that means "this chunk's next micro-batch", which under VPP is not the
        globally next group. Backward runs
        ``(1,0)(1,1)(1,2)(1,3) (0,0)(0,1)(0,2)(0,3) (1,4)...``, so after ``(1,3)``
        the guess says ``(1,4)`` while the successor is ``(0,0)``. Guessing wrong is
        not a numerical problem, since lazy reload backs it up, but it spends the
        budget on a group that is several backward steps away and starves the one
        being consumed -- exactly the stall this is meant to remove.

        The order is therefore recorded during the first iteration (``_bwd_order``
        into ``_bwd_next``) and looked up afterwards. It is constant across
        iterations, being a function of stage count, chunk count, accumulation steps
        and micro-step. The learning iteration does no cross-group prefetch, which
        is equivalent to having it off.

        The amount issued is still governed by ``prefetch_budget_bytes``, so this
        cannot raise peak memory; the same allowance is spent earlier.
        """
        if not self.cross_group_prefetch or not self._in_budget_mode():
            return
        nxt = self._bwd_next.get(self._active_bwd_gk)
        if nxt is None:
            return  # still learning, or this was the last group of the iteration
        if self._enqueue_group(*nxt):
            self.stats["head_prefetch"] += 1
            self._fill_budget()

    def _fill_budget(self):
        """Issue H2D in consumption order while the budget allows, one batch."""
        budget = self.prefetch_budget_bytes
        batch = []
        keys = list(self._pending.keys())
        # The group currently running backward has to go first. Cross-group
        # prefetch puts the next group into _pending as well, and dict order is
        # insertion order, so without an explicit priority a future group could be
        # served while the group being consumed waits -- a priority inversion that
        # is most visible when several chunks interleave under VPP.
        if (
            self._active_bwd_gk in self._pending
            and keys[0] != self._active_bwd_gk
        ):
            keys.remove(self._active_bwd_gk)
            keys.insert(0, self._active_bwd_gk)
        for gk in keys:
            q = self._pending[gk]
            while q:
                rec = self._records.get(q[0])
                if rec is None or rec.gpu is not None:  # consumed, or on device
                    q.pop(0)
                    continue
                if (
                    self._inflight_bytes
                    and self._inflight_bytes + rec.nbytes > budget
                ):
                    q = None  # budget full; refill once something is consumed
                    break
                q.pop(0)
                rec.counted = True
                self._inflight_bytes += rec.nbytes
                batch.append(rec)
            if q is None:
                break
            del self._pending[gk]
        if batch:
            self._reload_batch(batch)
            self.stats["prefetched"] += len(batch)

    def end_iteration(self):
        """Call after every accumulation step to clear per-iteration state."""
        self.flush_packs()  # nothing should be pending; issue it if it is
        if self._max_skipped_bytes:
            logger.info(
                "[activation_offload] largest activation skipped inside a "
                "region this step: %s. If offloading did not save memory, check "
                "this first: anything larger than the tensors actually being "
                "offloaded is outside every region.",
                self._max_skipped_desc,
            )
            self._max_skipped_bytes = 0
            self._max_skipped_desc = ""
        leaked = len(self._records)
        self.stats["not_consumed"] += leaked
        self._records.clear()
        self._dedup.clear()  # normally swapped by _ScopeGuard; this is a backstop
        self._groups.clear()
        self._pending.clear()
        self._waited_batches.clear()
        self._inflight_bytes = 0
        self._active_bwd_gk = None
        # Freeze the backward group order recorded during the first iteration.
        # Only build the table with two or more groups: a single group has no
        # successor to pull forward.
        if not self._bwd_next and len(self._bwd_order) >= 2:
            self._bwd_next = dict(zip(self._bwd_order, self._bwd_order[1:]))
            logger.info(
                "[activation_offload] backward group order recorded (%d "
                "groups); cross-group prefetch is now active: %s",
                len(self._bwd_order),
                self._bwd_order[:8],
            )
        self._bwd_order = []
        self._fwd_seq.clear()
        self._bwd_seq.clear()
        self._boundary_seq = 0
        if not self._learned and self._learn_bytes:
            self._finish_learning()
        elif self.auto_budget:
            self._tune_budget()
        self._warn_on_degradation(leaked)

    def _warn_on_degradation(self, leaked):
        """Surface the three silent slowdowns as logs, once each.

        ``exposed`` (backward waiting on an un-prefetched H2D), ``pool_oom`` (the
        pinned pool was full so the activation stayed on the device) and
        ``not_consumed`` (offloaded in forward but never consumed in backward) are
        only counters; in production they show up as unexplained slowness or as
        memory that does not go down, and nobody reads ``mgr.stats``.
        """
        if leaked:
            self._warn_once(
                "not_consumed",
                "%d offloaded activations were never consumed by backward. "
                "Either the graph was discarded, or end_iteration() is being "
                "called at the wrong point; pinned buffers are being recycled "
                "late as a result.",
                leaked,
            )
        oom = self.stats["pool_oom"]
        if oom:
            self._warn_once(
                "pool_oom",
                "pinned pool hit its capacity %d time(s); those activations "
                "silently stayed on GPU. Raise "
                "activation_offload_pool_capacity_bytes or offload fewer "
                "modules.",
                oom,
            )
        # late/exposed are a normal transient while the budget is being tuned, so
        # only warn when the user pinned the budget and the controller will not
        # act on it.
        if not self.auto_budget and (
            self.stats["exposed"] or self.stats["late"]
        ):
            self._warn_once(
                "stalled",
                "%d activation reloads were not prefetched at all and %d more "
                "were still in flight when backward needed them; both blocked "
                "the main stream. Raise "
                "activation_offload_prefetch_budget_bytes, or leave it None to "
                "let it tune itself.",
                self.stats["exposed"],
                self.stats["late"],
            )

    def _warn_once(self, key, msg, *args):
        if key in self._warned:
            return
        self._warned.add(key)
        logger.warning("[activation_offload] " + msg, *args)

    def _tune_budget(self):
        """Adaptive budget: climb towards the smallest budget with no stalls.

        The feedback signal must be ``late`` (the H2D was still in flight when
        backward needed the tensor), not ``exposed``. ``exposed`` only counts
        tensors that were never prefetched, and in budget mode every consumption
        issues the next one, so the next tensor is always "issued but unfinished",
        ``rec.gpu`` is set and ``exposed`` never increments -- the controller would
        never see the stalls it is causing.

        The smallest stall-free budget is also the best point for both memory and
        time, so start small and double while last iteration added new stalls. No
        fixed fraction of the group size: that ratio differs by an order of
        magnitude between shapes.
        """
        stalls = self.stats["late"] + self.stats["exposed"]
        grew = stalls > self._budget_probe_stalls
        self._budget_probe_stalls = stalls
        if not grew:
            return  # copies are hidden; no need to grow
        cap = self._group_bytes
        if cap and self.prefetch_budget_bytes >= cap:
            return  # growing past a whole group is pointless
        target = max(1, self.prefetch_budget_bytes) * 2
        self.prefetch_budget_bytes = min(target, cap) if cap else target
        self.stats["budget_raises"] += 1

    def _finish_learning(self):
        """End of the learning iteration: pick which boundaries to skip.

        Both fraction and pp_delta keep the *later* boundaries on the device.
        Backward consumes in reverse forward order, so the later boundaries are
        needed first, and leaving them resident is exactly what keeps their reload
        from blocking the main stream.
        """
        self._learned = True
        order = sorted(self._learn_bytes)  # boundary index is forward order
        skip: set = set()

        # fraction: offload only the first N% of boundaries, keep the rest resident
        keep_from = int(len(order) * max(0.0, min(1.0, self.fraction)))
        skip.update(order[keep_from:])

        # pp_delta: higher pipeline ranks keep pp_rank * delta more bytes resident
        budget = self.pp_rank * self.pp_delta_bytes
        if budget > 0:
            for idx in reversed(order):
                if budget <= 0:
                    break
                if idx in skip:
                    continue
                skip.add(idx)
                budget -= self._learn_bytes[idx]

        self._skip_boundaries = frozenset(skip)
        # Bytes one micro-batch group will actually offload; the adaptive budget
        # uses it as an upper bound.
        self._group_bytes = sum(
            self._learn_bytes[i] for i in order if i not in skip
        )
        if self.auto_budget:
            # Start at a thirty-second of the group and let _tune_budget double
            # towards the optimum. Starting low only costs a few stalls in the
            # first iterations.
            self.prefetch_budget_bytes = max(1 << 20, self._group_bytes // 32)
        self.stats["learned_boundaries"] = len(order)
        self.stats["skipped_boundaries"] = len(skip)
        self._learn_bytes.clear()

    # ---------------- pack / unpack ----------------

    def _pack(self, t, scope_name):
        if not self.profile:
            return self._pack_impl(t, scope_name)
        _t = time.perf_counter()
        try:
            return self._pack_impl(t, scope_name)
        finally:
            self.stats["pack_ms"] += (time.perf_counter() - _t) * 1e3

    def _unpack(self, packed):
        if not self.profile:
            return self._unpack_impl(packed)
        _t = time.perf_counter()
        try:
            return self._unpack_impl(packed)
        finally:
            self.stats["unpack_ms"] += (time.perf_counter() - _t) * 1e3

    def _note_skipped(self, t, numel):
        """Remember the largest tensor skipped, for one report per iteration.

        When offloading does not reduce memory, the first thing to rule out is
        that the biggest activation never entered a region or was filtered by a
        predicate.
        """
        try:
            nbytes = numel * (
                t.element_size() if hasattr(t, "element_size") else 2
            )
        except Exception:
            return
        if nbytes > self._max_skipped_bytes:
            self._max_skipped_bytes = nbytes
            self._max_skipped_desc = (
                f"{list(t.shape)} {t.dtype} {nbytes / 1048576.0:.1f}MB "
                f"contiguous={t.is_contiguous()} offset={t._offset()} "
                f"place={t.place}"
            )

    def _pack_impl(self, t, scope_name):
        # Top-level varargs may carry None and non-Tensor values into pack; pass
        # them straight through.
        if t is None or not isinstance(t, paddle.Tensor):
            return t
        # The element count must be computed from the Python-side shape.
        # ``int(t.numel())`` builds a 0-D device tensor and forces a
        # device-to-host synchronization, which dominates the per-tensor cost of
        # the hook; multiplying out ``t.shape`` is free by comparison.
        numel = 1
        for s in t.shape:
            numel *= s
        is_param = isinstance(t, EagerParamBase)
        if (
            not self.enabled
            or is_param  # parameters are not activations
            or not t.place.is_gpu_place()
            or numel < self.min_offload_numel
            or not t.is_contiguous()
            or t._offset() != 0  # see the module docstring
        ):
            self.stats["skipped"] += 1
            if not is_param:
                # A parameter was never a candidate, so reporting one as the
                # largest activation left behind only sends the reader looking
                # for memory that offloading could not have saved anyway.
                self._note_skipped(t, numel)
            return t

        nbytes = numel * (t.element_size() if hasattr(t, "element_size") else 2)

        if (
            self.min_offload_bytes is not None
            and nbytes < self.min_offload_bytes
        ):
            self.stats["skipped"] += 1
            return t

        # Dedup within the region. When several operators save the same activation
        # pack fires once per operator, with identical ``data_ptr()`` but a
        # different Python wrapper each time. ``data_ptr`` is a valid identity only
        # because this table lives inside a single region and its second element
        # holds a strong reference to the source tensor: while that reference
        # exists no other tensor can occupy the address. Across regions the
        # argument fails, which is why _ScopeGuard swaps the table on entry and
        # exit.
        dkey = (t.data_ptr(), tuple(t.shape), str(t.dtype))
        dup = self._dedup.get(dkey)
        if dup is not None:
            rec0 = dup[0]
            rec0.refs += 1
            self.stats["dedup_hits"] += 1
            self.stats["dedup_bytes_saved"] += nbytes
            return rec0.cpu

        # fraction and pp_delta decide what to skip by a boundary's index in
        # forward order. The index must advance here rather than only when a tensor
        # is really offloaded, or the numbering would differ between the learning
        # iteration and steady state.
        boundary = self._boundary_seq
        self._boundary_seq += 1
        if not self._learned:
            self._learn_bytes[boundary] = nbytes
        elif boundary in self._skip_boundaries:
            self.stats["skipped_policy"] += 1
            return t

        # Separate path for very large tensors: no pool, no budget, its own pinned
        # allocation and copy, discarded after unpack.
        if self.big_tensor_bytes is not None and nbytes > self.big_tensor_bytes:
            self.stats["big_packed"] += 1
            cpu = t.pin_memory()
        else:
            cpu = self.pool.alloc(t.shape, t.dtype)
            if cpu is None:  # pool at capacity: leave this tensor on the device
                self.stats["pool_oom"] += 1
                return t

        if self.sync_mode:
            cpu.copy_(t, False)
            off_event = None
        elif self.d2h_on_main:
            # Issued on the main stream: ordered against the producing kernel by
            # construction, so no wait_stream or record_stream is needed, and the
            # source block is reclaimed normally once the copy completes.
            cpu.copy_(t, False)
            off_event = None
        else:
            off_event = None  # deferred to flush_packs

        rec = _Record(
            cpu=cpu,
            shape=t.shape,
            dtype=t.dtype,
            stop_gradient=t.stop_gradient,
            off_event=off_event,
            group_key=self._cur_group_key,
            nbytes=nbytes,
            big=(nbytes > self.big_tensor_bytes)
            if self.big_tensor_bytes is not None
            else False,
        )
        self._records[id(cpu)] = rec
        # The second element is a strong reference to the source tensor, which is
        # what keeps this address from being reused within the region.
        self._dedup[dkey] = (rec, t)
        if self._cur_group_key is not None:
            self._groups.setdefault(self._cur_group_key, []).append(id(cpu))
        self.stats["packed"] += 1
        self.stats["packed_bytes"] += nbytes
        # Per-region accounting: needed to tell where the offloaded bytes went and
        # which regions are not covered.
        self._bytes_by_group[scope_name] = (
            self._bytes_by_group.get(scope_name, 0) + nbytes
        )
        # Out-of-memory failures often happen before the first optimizer step, so
        # logging driven by the training loop never gets a chance to print. Log the
        # first packed tensor unconditionally -- it proves a region was entered and
        # shows how large that tensor was -- then a full stats line every
        # ACT_OFFLOAD_LOG_EVERY tensors.
        if self.stats["packed"] == 1:
            logger.info(
                "[activation_offload] first pack: group=%s shape=%s dtype=%s "
                "%.1fMB | %s",
                scope_name,
                list(t.shape),
                t.dtype,
                nbytes / 1048576.0,
                self.format_stats(),
            )
        elif self._log_every and self.stats["packed"] % self._log_every == 0:
            logger.info(self.format_stats())
        if not self.sync_mode and not self.d2h_on_main:
            # Batched: one wait_stream and one event shared by the batch.
            self._pending_packs.append((rec, t))
            self._pending_bytes += nbytes
            if self._pending_bytes >= self.max_pending_bytes:
                self.flush_packs()
        return cpu

    def flush_packs(self):
        """Issue the batched D2H: one wait_stream, N copies, one event.

        By the time flush runs, the producing kernels of every source tensor in the
        batch have been issued on the main stream, so a single ``wait_stream``
        covers all of them and the copies are ordered within the D2H stream.
        Synchronising per tensor instead costs an order of magnitude more than the
        copies themselves.

        Cost: the source tensors stay referenced until flush, delaying their
        release. ``max_pending_bytes`` bounds that, and the scope exit flushes.
        """
        if not self._pending_packs:
            return
        batch = self._pending_packs
        self._pending_packs = []
        self._pending_bytes = 0
        self.d2h_stream.wait_stream(paddle.device.current_stream())
        with paddle.device.stream_guard(self.d2h_stream):
            for rec, src in batch:
                rec.cpu.copy_(src, False)
                src._record_stream()  # keep the block alive until the D2H reads it
        off_event = self.d2h_stream.record_event()
        for rec, _ in batch:
            rec.off_event = off_event
        self.stats["d2h_batches"] += 1

    def _reload_async(self, rec):
        self._reload_batch([rec])

    def _reload_batch(self, recs):
        """Reload a batch: one allocator fence, one wait per distinct D2H event,
        N copies and one shared reload event. The batch becomes ready together."""
        recs = [r for r in recs if r.gpu is None]
        if not recs:
            return
        # Destinations must come from the main stream's allocator pool; see the
        # module docstring for why this is a numerical requirement.
        gpus = [paddle.empty(r.shape, dtype=r.dtype) for r in recs]
        if not self.unsafe_no_alloc_fence:
            # These blocks may have just been reclaimed from the main stream, so
            # writing them has to be ordered after their previous users. One fence
            # covers the batch, since everything already issued on the main stream
            # includes every block's previous user.
            self.h2d_stream.wait_event(
                paddle.device.current_stream().record_event()
            )
        seen = set()
        for r in recs:  # wait once per distinct off_event
            ev = r.off_event
            if ev is not None and id(ev) not in seen:
                seen.add(id(ev))
                self.h2d_stream.wait_event(ev)
        with paddle.device.stream_guard(self.h2d_stream):
            for r, g in zip(recs, gpus):
                g.copy_(r.cpu, False)
        ev = self.h2d_stream.record_event()
        bid = self._batch_seq
        self._batch_seq += 1
        for r, g in zip(recs, gpus):
            r.gpu = g
            r.reload_event = ev
            r.batch_id = bid
        self.stats["h2d_batches"] += 1

    def _unpack_impl(self, packed):
        rec = (
            self._records.get(id(packed))
            if isinstance(packed, paddle.Tensor)
            else None
        )
        if rec is None or rec.cpu is not packed:
            return packed
        # After dedup, several slots share one record: only the last consumption
        # deregisters it, returns the pinned buffer and frees its budget.
        rec.refs -= 1
        last = rec.refs <= 0
        if last:
            self._records.pop(id(packed))
            gk = rec.group_key
            if gk is not None and id(packed) in self._groups.get(gk, []):
                self._groups[gk].remove(id(packed))

        if self.sync_mode:
            gpu = paddle.empty(rec.shape, dtype=rec.dtype)
            gpu.copy_(rec.cpu, False)
            if last and not rec.big:
                self.pool.free(rec.cpu)
        else:
            if rec.off_event is None and not self.d2h_on_main:
                self.flush_packs()  # still batched; issue it now
            # Never prefetched: fall back to a lazy reload and count it exposed.
            if rec.gpu is None:
                self._reload_async(rec)
                self.stats["exposed"] += 1
            gpu = rec.gpu
            if rec.reload_event is not None and not rec.probed:
                # Counted unconditionally: query() is a non-blocking test, not a
                # synchronization point. These are the only way to tell whether
                # prefetch actually hid the copy -- ``exposed`` cannot, see
                # _tune_budget.
                rec.probed = True
                if rec.reload_event.query():
                    self.stats["hit"] += 1
                else:
                    self.stats["late"] += 1
                    if self.profile:
                        # Only measure the wait in diagnostic mode; the
                        # synchronize is itself the expensive part.
                        _t = time.perf_counter()
                        rec.reload_event.synchronize()
                        self.stats["late_wait_ms"] += (
                            time.perf_counter() - _t
                        ) * 1e3
            # A batch shares one reload event, so the main stream waits once per
            # batch and the rest is ordered within the stream. The batch id, not
            # ``id(event)``, identifies it: event objects get collected and their
            # addresses reused, so deduplicating by id would skip a wait that was
            # actually needed and corrupt results.
            if rec.batch_id not in self._waited_batches:
                self._waited_batches.add(rec.batch_id)
                paddle.device.current_stream().wait_event(rec.reload_event)
                self.stats["main_waits"] += 1
            gpu._record_stream()
            if last:
                if not rec.big:
                    self.pool.free(rec.cpu, rec.reload_event)
                if rec.counted:  # budget mode: release the allowance and refill
                    self._inflight_bytes -= rec.nbytes
                    self._fill_budget()

        gpu.stop_gradient = rec.stop_gradient
        self.stats["unpacked"] += 1
        return gpu

    # ---------------- public API ----------------

    def scope(self, name="default"):
        """Context manager: activations saved inside it go through pack/unpack.

        Usage::

            with get_offload_manager().scope("moe"):
                out = expert_layer(x)
        """
        return _ScopeGuard(self, name)

    def format_stats(self, prefix="activation-offload"):
        """One readable line of state plus memory, for the training loop to log.

        How to read it:
          ``packed`` = 0            no region was entered: the config did not take
                                    effect, or this path has no boundaries
          ``prefetched`` << packed  prefetch anchors are not wired up
                                    (``enable_fleet_prefetch`` was not called)
          ``late`` > 0              prefetch was issued but unfinished, so the main
                                    stream waited: the budget is too small
          ``exposed`` > 0           never prefetched (no anchors, or budget 0)
          ``pool_oom`` > 0          pinned pool full; those tensors stayed resident
          ``not_consumed`` > 0      offloaded in forward, never consumed in backward
          ``dedup`` > 0             an activation was saved by several operators and
                                    merged, which is a pure saving
          rising ``pinned`` with pool ``hit`` < 1  the pool is leaking because its
                                    bucket key started tracking shapes again; in
                                    steady state ``hit`` should be close to 1
        """
        s = self.stats
        mb = 1024.0 * 1024.0
        try:
            # All three come from the paddle allocator, not from the driver:
            # NCCL buffers, cuBLAS workspaces and other library allocations are
            # not counted, so reserved reads lower than the device's actual usage.
            cur = paddle.device.cuda.memory_allocated() / mb
            peak = paddle.device.cuda.max_memory_allocated() / mb
            reserved = paddle.device.cuda.memory_reserved() / mb
            mem = f" | mem cur={cur:.0f}MB peak={peak:.0f}MB reserved={reserved:.0f}MB"
        except Exception:
            mem = ""
        return (
            f"[{prefix}] packed={s['packed']} ({s['packed_bytes'] / mb:.0f}MB) "
            f"unpacked={s['unpacked']} skipped={s['skipped']} "
            f"prefetched={s['prefetched']} hit={s['hit']} late={s['late']} "
            f"exposed={s['exposed']} head_pf={s['head_prefetch']} "
            f"dedup={s['dedup_hits']} ({s['dedup_bytes_saved'] / mb:.0f}MB saved) "
            # None means budget mode is not active, which is not the same as a
            # budget of zero; it prints as 0 only to keep the format simple. Use
            # exposed and late to tell the two apart.
            f"budget={(self.prefetch_budget_bytes or 0) / mb:.0f}MB"
            f"(+{s['budget_raises']}) "
            f"pool_oom={s['pool_oom']} not_consumed={s['not_consumed']} "
            # pinned is a host-side figure and does not overlap with the device
            # numbers above; they must not be added together.
            f"{self.pool.stats_line()}" + mem + self._by_group_str()
        )

    def _by_group_str(self):
        """Per-region byte accounting, largest first."""
        if not self._bytes_by_group:
            return ""
        items = sorted(self._bytes_by_group.items(), key=lambda kv: -kv[1])
        return " | by group " + " ".join(
            f"{name}={nbytes / 1048576.0:.0f}MB" for name, nbytes in items
        )

    def reset_stats(self):
        self.stats = {
            "packed": 0,
            "unpacked": 0,
            "skipped": 0,
            "prefetched": 0,
            "exposed": 0,
            "not_consumed": 0,
            "packed_bytes": 0,
            "big_packed": 0,
            "pool_oom": 0,
            "skipped_policy": 0,  # skipped by fraction / pp_delta
            "dedup_hits": 0,  # packs of an activation already packed in this region
            "dedup_bytes_saved": 0,  # one-way bytes dedup avoided
            "head_prefetch": 0,  # times cross-group prefetch queued the next group
            "learned_boundaries": 0,  # boundaries seen during the learning pass
            "skipped_boundaries": 0,  # of those, skipped by policy
            "budget_raises": 0,  # times the adaptive budget doubled
            "d2h_batches": 0,  # D2H flushes, i.e. cross-stream synchronizations
            "h2d_batches": 0,  # batched reloads
            "main_waits": 0,  # times the main stream waited on a reload event
            # Filled only when profile=True:
            "pack_ms": 0.0,  # host time inside the pack hook
            "unpack_ms": 0.0,  # host time inside the unpack hook
            "late_wait_ms": 0.0,  # total time spent waiting on late reloads
            # Always counted; see _unpack_impl and _tune_budget
            "hit": 0,  # H2D had completed by the time it was consumed
            "late": 0,  # H2D was still in flight, so the main stream waited
        }


class _ScopeGuard:
    """Context manager returned by ``scope()``.

    Installs and removes ``saved_tensors_hooks``, and calls ``flush_packs()`` on
    exit -- at that point every source tensor's producing kernel has been issued,
    so one ``wait_stream`` covers the whole batch.

    It also owns the dedup table's lifetime, swapping in an empty table on entry
    and restoring the previous one on exit, since an address is only a unique
    identity while the table still references the source tensor (see
    ``_pack_impl``). Regions cannot currently nest -- the inner hooks would shadow
    the outer ones -- but saving and restoring means nothing here assumes that.
    """

    __slots__ = ("_mgr", "_name", "_hooks", "_prev_dedup")

    def __init__(self, mgr, name):
        self._mgr = mgr
        self._name = name
        self._hooks = None
        self._prev_dedup = None

    def __enter__(self):
        mgr, name = self._mgr, self._name
        self._prev_dedup = mgr._dedup
        mgr._dedup = {}
        self._hooks = paddle.autograd.saved_tensors_hooks(
            lambda t: mgr._pack(t, name), mgr._unpack
        )
        self._hooks.__enter__()
        return self

    def __exit__(self, *exc):
        try:
            return self._hooks.__exit__(*exc)
        finally:
            # Drop the source references the dedup table holds before flushing.
            # Flush takes its own references to this batch's sources, so no tensor
            # loses its last reference in between.
            self._mgr._dedup = self._prev_dedup
            self._prev_dedup = None
            self._mgr.flush_packs()


_manager: OffloadManager | None = None


def get_offload_manager(**kw) -> OffloadManager:
    """Return the process-wide singleton. Only the first call's kwargs apply."""
    global _manager
    if _manager is None:
        _manager = OffloadManager(**kw)
    return _manager


def current_offload_manager() -> OffloadManager | None:
    """The singleton if one exists, else None.

    ``offload_region`` uses this to degrade to a no-op before the manager has been
    built, so model code need not care about construction order.
    """
    return _manager


def reset_offload_manager():
    """Drop the singleton so the next ``get_offload_manager`` rebuilds it.

    For cases that need several configurations in one process, such as tests. The
    production path does not need it: the singleton is built with the model and
    does not change for the rest of training.
    """
    global _manager
    _manager = None


def offload_enabled(config) -> bool:
    """Whether this config enables fine-grained activation offloading."""
    return bool(getattr(config, "fine_grained_activation_offloading", False))


def offload_groups(config) -> frozenset:
    """Region names this config offloads; empty when offloading is off.

    Model code turns this into a per-module boolean in ``__init__``, following the
    same pattern as the recompute flags, so the forward path only reads an
    attribute::

        self.offload_core_attn = "core_attn" in offload_groups(config)
        ...
        with offload_region(self.offload_core_attn, "core_attn"):
            core_attn_out = self.core_attention(...)

    With the master switch on and ``offload_modules`` unset, every supported region
    is returned: enabled but unspecified means all of them.
    """
    if not offload_enabled(config):
        return frozenset()
    names = getattr(config, "offload_modules", None)
    if names:
        return frozenset(names)
    return getattr(type(config), "_OFFLOAD_MODULE_NAMES", frozenset())


def offload_kwargs_from_config(config, pp_rank=None) -> dict:
    """Translate a ``TransformerConfig`` into ``OffloadManager`` kwargs.

    Kept separate from ``manager_from_config`` so that callers which cannot use the
    process-wide singleton -- tests building one manager per configuration in a
    single process -- share the same mapping.
    """
    if pp_rank is None:
        pp_rank = getattr(config, "pipeline_model_parallel_rank", 0) or 0
    min_bytes = getattr(config, "min_offloaded_tensor_bytes", None)
    return {
        "enabled": offload_enabled(config),
        "min_offload_bytes": min_bytes,
        # The byte threshold is the only one the config exposes, so the element
        # threshold has to step aside: the two are combined with AND, and leaving
        # the constructor default in place would override any byte threshold the
        # config asks for.
        "min_offload_numel": 0 if min_bytes is not None else 256 * 1024,
        "fraction": getattr(config, "activation_offload_fraction", 1.0),
        "delta_offload_bytes_across_pp_ranks": getattr(
            config, "delta_offload_bytes_across_pp_ranks", 0
        ),
        "pp_rank": pp_rank,
        "prefetch_budget_bytes": getattr(
            config, "activation_offload_prefetch_budget_bytes", None
        ),
        "pool_capacity_bytes": getattr(
            config, "activation_offload_pool_capacity_bytes", None
        ),
        "numa_bind": getattr(config, "activation_offload_numa_bind", True),
        "cross_group_prefetch": getattr(
            config, "activation_offload_cross_group_prefetch", True
        ),
    }


def manager_from_config(config, pp_rank=None) -> OffloadManager:
    """Build or fetch the singleton from a ``TransformerConfig``.

    Only the first call takes effect, so this must run before the first forward
    pass. Doing so also guarantees the PyLayer shim and the NUMA bind are in place
    before any pinned memory is allocated.
    """
    return get_offload_manager(**offload_kwargs_from_config(config, pp_rank))
