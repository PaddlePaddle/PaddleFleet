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

"""CPU-only behavior tests for the pipeline forward/backward hook machinery in
``paddlefleet.transformer.indexer_loss_overlap`` -- the module that defers the
DSA indexer-loss branch into the pipeline's forward send/recv window and
registers the drain on Paddle's micro-step hook locations.

Scope: this file exercises only the device-independent pure logic, whose
correctness is a matter of local queue bookkeeping, latch/state transitions and
config gating:

* ``enabled`` / ``_has_dsa_indexer`` / ``_recomputes_core_attn`` -- config
  predicates.
* ``validate_config`` -- the raise/pass gating for each unsupported shape.
* ``enqueue`` / ``pending`` / ``stats`` -- queue accounting and the copy
  semantics of ``stats``.
* ``drain`` / ``drain_all`` -- LIFO drain order, the ``on_done`` finally-callback
  (including its reset and that failures are NOT swallowed).
* ``defer_discard`` -- the owner/already-set gating and the captured callback.
* ``_p2p_issued_hook`` -- the ``_P2P_WINDOW_SEEN`` latch (set even on an empty
  queue) and the two per-window counters.
* ``_forward_end_hook`` -- the correctness fallback: disarmed once the latch is
  set, and the once-only permanent-fallback warning on the second drain.
* ``register_pipeline_hooks`` -- only the idempotency guard and the
  not-a-pipeline-model early return; both avoid touching Paddle's global hook
  registry.

Out of scope (documented, not faked): the real ``register_pipeline_hooks``
registration path mutates Paddle's process-global micro-step hook registry and
depends on the ``PipelineParallelMicroStepLocations`` enum of the installed
Paddle; asserting it would require a real ``PipelineParallel`` model and would
leave global side effects that cannot be cleanly restored, so only its two
side-effect-free early returns are covered here. The actual overlap with a live
pipeline send/recv is a multi-rank property and belongs in a multi-card test.

Every expected value below is derived by hand from the module contract, never
from the production output or from the coverage file.
"""

import unittest
from types import SimpleNamespace

try:
    from paddlefleet.transformer import indexer_loss_overlap

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # honest dependency probe
    indexer_loss_overlap = None
    _IMPORT_ERROR = exc

_HAS_DEPS = _IMPORT_ERROR is None
_SKIP_REASON = (
    "paddlefleet.transformer.indexer_loss_overlap not importable on this "
    f"CPU host: {_IMPORT_ERROR!r}"
)


def _make_work(owner, seqlen=0, on_done=None):
    """Build a ``_PendingWork`` with placeholder tensor fields.

    ``drain`` / ``defer_discard`` only ever touch ``owner``, ``on_done`` and (for
    identification here) ``seqlen``; the tensor slots are held as opaque
    references and never inspected by the module, so real Paddle tensors are not
    needed to exercise this pure queue/state logic.
    """
    return indexer_loss_overlap._PendingWork(
        owner=owner,
        query=None,
        kv=None,
        lse_indexer=None,
        topk_indices=None,
        topk_scores=None,
        index_q=None,
        weights=None,
        index_k=None,
        input_ids=None,
        batch=1,
        seqlen=seqlen,
        on_done=on_done,
    )


class _RecordingOwner:
    """Stands in for the ``MQALatentAttention`` that owns the loss branch.

    Records, in call order, the ``seqlen`` of each work whose branch is run, so
    tests can assert the drain order rather than merely that draining happened.
    """

    def __init__(self, calls):
        self._calls = calls

    def _run_indexer_loss_branch(self, work):
        self._calls.append(work.seqlen)


class _RaisingOwner:
    """Owner whose branch fails, to prove ``drain`` does not swallow errors."""

    def _run_indexer_loss_branch(self, work):
        raise RuntimeError("branch blew up")


class _ModuleStateMixin:
    """Snapshot every module-global before each test and restore it after.

    The module keeps a process-level queue, latch, counters and stats dict; a
    test that mutated them without restoring would silently contaminate every
    later test in the same process (antipattern #11).
    """

    def setUp(self):
        m = indexer_loss_overlap
        self._saved = {
            "queue": list(m._QUEUE),
            "hooks_registered": m._HOOKS_REGISTERED,
            "p2p_seen": m._P2P_WINDOW_SEEN,
            "fe_drains": m._FORWARD_END_DRAINS,
            "warned": m._FALLBACK_WARNED,
            "stats": dict(m._STATS),
        }
        self.addCleanup(self._restore_module_state)
        m._QUEUE.clear()
        m._HOOKS_REGISTERED = False
        m._P2P_WINDOW_SEEN = False
        m._FORWARD_END_DRAINS = 0
        m._FALLBACK_WARNED = False
        for key in m._STATS:
            m._STATS[key] = 0

    def _restore_module_state(self):
        m = indexer_loss_overlap
        m._QUEUE.clear()
        m._QUEUE.extend(self._saved["queue"])
        m._HOOKS_REGISTERED = self._saved["hooks_registered"]
        m._P2P_WINDOW_SEEN = self._saved["p2p_seen"]
        m._FORWARD_END_DRAINS = self._saved["fe_drains"]
        m._FALLBACK_WARNED = self._saved["warned"]
        m._STATS.clear()
        m._STATS.update(self._saved["stats"])


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestConfigPredicates(unittest.TestCase):
    """`enabled`, `_has_dsa_indexer`, `_recomputes_core_attn` gating rules."""

    def test_enabled_reads_flag_and_coerces_to_bool(self):
        self.assertIs(
            indexer_loss_overlap.enabled(
                SimpleNamespace(dsa_indexer_loss_bwd_p2p_overlap=True)
            ),
            True,
        )
        # Truthy non-bool must be coerced to a real bool, not passed through.
        self.assertIs(
            indexer_loss_overlap.enabled(
                SimpleNamespace(dsa_indexer_loss_bwd_p2p_overlap=1)
            ),
            True,
        )
        self.assertIs(
            indexer_loss_overlap.enabled(
                SimpleNamespace(dsa_indexer_loss_bwd_p2p_overlap=False)
            ),
            False,
        )
        # Missing attribute defaults to disabled.
        self.assertIs(indexer_loss_overlap.enabled(SimpleNamespace()), False)

    def test_has_dsa_indexer_requires_all_three_conditions(self):
        has = indexer_loss_overlap._has_dsa_indexer
        # All three met, and -2 is present among other ratios -> True.
        self.assertTrue(
            has(
                SimpleNamespace(
                    experimental_attention_variant="dsv4_hybrid",
                    hybrid_mla_attention="mqa_dsa",
                    csa_compress_ratios=[1, -2, 4],
                )
            )
        )
        # No -2 in the ratios -> False (this is the row-selecting condition).
        self.assertFalse(
            has(
                SimpleNamespace(
                    experimental_attention_variant="dsv4_hybrid",
                    hybrid_mla_attention="mqa_dsa",
                    csa_compress_ratios=[1, 2, 4],
                )
            )
        )
        # Wrong attention variant short-circuits to False.
        self.assertFalse(
            has(
                SimpleNamespace(
                    experimental_attention_variant="vanilla",
                    hybrid_mla_attention="mqa_dsa",
                    csa_compress_ratios=[-2],
                )
            )
        )
        # Wrong hybrid mode short-circuits to False.
        self.assertFalse(
            has(
                SimpleNamespace(
                    experimental_attention_variant="dsv4_hybrid",
                    hybrid_mla_attention="mla",
                    csa_compress_ratios=[-2],
                )
            )
        )
        # Empty / missing ratios -> False.
        self.assertFalse(
            has(
                SimpleNamespace(
                    experimental_attention_variant="dsv4_hybrid",
                    hybrid_mla_attention="mqa_dsa",
                    csa_compress_ratios=None,
                )
            )
        )

    def test_recomputes_core_attn_handles_list_and_dict_modules(self):
        rc = indexer_loss_overlap._recomputes_core_attn
        # selective + core_attn in a list -> True.
        self.assertTrue(
            rc(
                SimpleNamespace(
                    recompute_granularity="selective",
                    recompute_modules=["mlp", "core_attn"],
                )
            )
        )
        # selective + core_attn as a dict key -> True (both spellings covered).
        self.assertTrue(
            rc(
                SimpleNamespace(
                    recompute_granularity="selective",
                    recompute_modules={"core_attn": 2},
                )
            )
        )
        # core_attn absent -> False.
        self.assertFalse(
            rc(
                SimpleNamespace(
                    recompute_granularity="selective",
                    recompute_modules=["mlp"],
                )
            )
        )
        # Non-selective granularity short-circuits before the module check.
        self.assertFalse(
            rc(
                SimpleNamespace(
                    recompute_granularity="full",
                    recompute_modules=["core_attn"],
                )
            )
        )


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestValidateConfig(unittest.TestCase):
    """`validate_config` raises on dead/unsupported shapes, passes when valid."""

    @staticmethod
    def _valid_config(**overrides):
        # A fully supported shape: enabled, real pipeline, a DSA indexer present,
        # a positive coeff, sparse loss on, no core_attn recompute, async p2p.
        base = {
            "dsa_indexer_loss_bwd_p2p_overlap": True,
            "pipeline_model_parallel_size": 2,
            "experimental_attention_variant": "dsv4_hybrid",
            "hybrid_mla_attention": "mqa_dsa",
            "csa_compress_ratios": [-2],
            "dsa_indexer_loss_coeff": 1.0,
            "dsa_indexer_use_sparse_loss": True,
            "recompute_granularity": None,
            "recompute_modules": None,
            "overlap_p2p_comm": True,
            "batch_p2p_comm": None,
        }
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_disabled_config_is_never_validated(self):
        # Flag off: the function must return before any of the raising checks,
        # even though pp_size == 1 would otherwise be rejected.
        cfg = self._valid_config(
            dsa_indexer_loss_bwd_p2p_overlap=False,
            pipeline_model_parallel_size=1,
        )
        self.assertIsNone(indexer_loss_overlap.validate_config(cfg))

    def test_valid_config_passes(self):
        self.assertIsNone(
            indexer_loss_overlap.validate_config(self._valid_config())
        )

    def test_pipeline_size_one_is_rejected(self):
        cfg = self._valid_config(pipeline_model_parallel_size=1)
        with self.assertRaisesRegex(ValueError, "requires pipeline"):
            indexer_loss_overlap.validate_config(cfg)

    def test_missing_dsa_indexer_is_rejected(self):
        # pp_size is valid, so this must be the DSAIndexer branch specifically.
        cfg = self._valid_config(experimental_attention_variant="vanilla")
        with self.assertRaisesRegex(ValueError, "builds no"):
            indexer_loss_overlap.validate_config(cfg)

    def test_non_positive_coeff_is_rejected(self):
        cfg = self._valid_config(dsa_indexer_loss_coeff=0.0)
        with self.assertRaisesRegex(ValueError, "no layer"):
            indexer_loss_overlap.validate_config(cfg)

    def test_dense_loss_is_rejected(self):
        cfg = self._valid_config(dsa_indexer_use_sparse_loss=False)
        with self.assertRaisesRegex(ValueError, "sparse"):
            indexer_loss_overlap.validate_config(cfg)

    def test_core_attn_recompute_warns_but_does_not_raise(self):
        cfg = self._valid_config(
            recompute_granularity="selective",
            recompute_modules=["core_attn"],
        )
        with self.assertLogs(
            indexer_loss_overlap.logger, level="WARNING"
        ) as cm:
            self.assertIsNone(indexer_loss_overlap.validate_config(cfg))
        self.assertTrue(
            any("core_attn" in msg for msg in cm.output),
            cm.output,
        )

    def test_batched_p2p_warns_but_does_not_raise(self):
        # overlap_p2p_comm off selects the batched ops -> correct but no overlap.
        cfg = self._valid_config(overlap_p2p_comm=False)
        with self.assertLogs(
            indexer_loss_overlap.logger, level="WARNING"
        ) as cm:
            self.assertIsNone(indexer_loss_overlap.validate_config(cfg))
        self.assertTrue(
            any("overlap_p2p_comm" in msg for msg in cm.output),
            cm.output,
        )


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestQueueAccounting(_ModuleStateMixin, unittest.TestCase):
    """`enqueue`, `pending` and `stats` bookkeeping."""

    def test_enqueue_grows_queue_and_counts(self):
        m = indexer_loss_overlap
        self.assertEqual(m.pending(), 0)
        self.assertEqual(m.stats()["enqueued"], 0)
        m.enqueue(_make_work(_RecordingOwner([]), seqlen=7))
        self.assertEqual(m.pending(), 1)
        self.assertEqual(m.stats()["enqueued"], 1)
        m.enqueue(_make_work(_RecordingOwner([]), seqlen=8))
        self.assertEqual(m.pending(), 2)
        self.assertEqual(m.stats()["enqueued"], 2)

    def test_stats_returns_defensive_copy(self):
        m = indexer_loss_overlap
        snapshot = m.stats()
        snapshot["enqueued"] = 999
        # Mutating the returned dict must not reach the module counter.
        self.assertEqual(m.stats()["enqueued"], 0)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestDrain(_ModuleStateMixin, unittest.TestCase):
    """`drain` / `drain_all` order, callbacks and failure propagation."""

    def test_empty_queue_drains_nothing(self):
        m = indexer_loss_overlap
        self.assertEqual(m.drain(), 0)
        self.assertEqual(m.stats()["drained"], 0)

    def test_drain_runs_branches_in_lifo_order(self):
        m = indexer_loss_overlap
        calls = []
        owner = _RecordingOwner(calls)
        m.enqueue(_make_work(owner, seqlen=10))
        m.enqueue(_make_work(owner, seqlen=20))
        ran = m.drain()
        self.assertEqual(ran, 2)
        # pop() from the tail -> the last enqueued (20) runs first.
        self.assertEqual(calls, [20, 10])
        self.assertEqual(m.pending(), 0)
        self.assertEqual(m.stats()["drained"], 2)

    def test_on_done_runs_after_branch_and_is_cleared(self):
        m = indexer_loss_overlap
        events = []
        owner = _RecordingOwner(events)  # appends seqlen (=1) when branch runs

        def _on_done():
            events.append("done")

        work = _make_work(owner, seqlen=1, on_done=_on_done)
        m.enqueue(work)
        m.drain()
        # Branch first (records 1), then the finally-callback.
        self.assertEqual(events, [1, "done"])
        # drain() nulls the callback reference after running it.
        self.assertIsNone(work.on_done)

    def test_branch_failure_propagates_but_still_runs_on_done(self):
        m = indexer_loss_overlap
        done_calls = []
        work = _make_work(
            _RaisingOwner(), seqlen=1, on_done=lambda: done_calls.append(1)
        )
        m.enqueue(work)
        # The module deliberately does not swallow a failing branch...
        with self.assertRaises(RuntimeError):
            m.drain()
        # ...but the finally-clause still fires the recompute-arming callback.
        self.assertEqual(done_calls, [1])
        # The failing item was popped before raising, leaving an empty queue.
        self.assertEqual(m.pending(), 0)

    def test_drain_all_delegates_to_drain(self):
        m = indexer_loss_overlap
        calls = []
        m.enqueue(_make_work(_RecordingOwner(calls), seqlen=42))
        self.assertEqual(m.drain_all(), 1)
        self.assertEqual(calls, [42])
        self.assertEqual(m.pending(), 0)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestDeferDiscard(_ModuleStateMixin, unittest.TestCase):
    """`defer_discard` owner/already-set gating and the captured callback."""

    def test_empty_queue_returns_false(self):
        m = indexer_loss_overlap
        span = SimpleNamespace(
            discard_output_and_register_recompute=lambda t: None
        )
        self.assertFalse(m.defer_discard(object(), span, object()))

    def test_matching_owner_installs_callback_binding_hook_tensor(self):
        m = indexer_loss_overlap
        owner = object()
        work = _make_work(owner, seqlen=1)
        m.enqueue(work)

        captured = []
        span = SimpleNamespace(
            discard_output_and_register_recompute=captured.append
        )
        hook_tensor = object()
        self.assertTrue(m.defer_discard(owner, span, hook_tensor))
        # Callback is installed but not yet fired.
        self.assertIsNotNone(work.on_done)
        self.assertEqual(captured, [])
        # Firing it forwards exactly the hook_tensor it was created with.
        work.on_done()
        self.assertEqual(len(captured), 1)
        self.assertIs(captured[0], hook_tensor)

    def test_different_owner_is_rejected(self):
        m = indexer_loss_overlap
        work = _make_work(object(), seqlen=1)
        m.enqueue(work)
        span = SimpleNamespace(
            discard_output_and_register_recompute=lambda t: None
        )
        self.assertFalse(m.defer_discard(object(), span, object()))
        self.assertIsNone(work.on_done)

    def test_already_deferred_is_not_overwritten(self):
        m = indexer_loss_overlap
        owner = object()

        def sentinel():  # identity marker: must survive defer_discard untouched
            return None

        work = _make_work(owner, seqlen=1, on_done=sentinel)
        m.enqueue(work)
        span = SimpleNamespace(
            discard_output_and_register_recompute=lambda t: None
        )
        self.assertFalse(m.defer_discard(owner, span, object()))
        self.assertIs(work.on_done, sentinel)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestP2pIssuedHook(_ModuleStateMixin, unittest.TestCase):
    """`_p2p_issued_hook` sets the latch and drains into the overlap window."""

    def test_drains_queue_and_counts_into_p2p_window(self):
        m = indexer_loss_overlap
        calls = []
        m.enqueue(_make_work(_RecordingOwner(calls), seqlen=5))
        # Extra kwargs mirror what the schedule passes and must be absorbed.
        m._p2p_issued_hook(output_tensor=object(), step_id=3)
        self.assertTrue(m._P2P_WINDOW_SEEN)
        self.assertEqual(calls, [5])
        self.assertEqual(m.pending(), 0)
        stats = m.stats()
        self.assertEqual(stats["drained"], 1)
        self.assertEqual(stats["drained_in_hook"], 1)
        self.assertEqual(stats["drained_in_p2p_window"], 1)

    def test_latch_is_set_even_on_empty_queue(self):
        m = indexer_loss_overlap
        m._p2p_issued_hook()
        # The latch records that this schedule provides the window, regardless
        # of whether this micro-step had work; the counters stay at zero.
        self.assertTrue(m._P2P_WINDOW_SEEN)
        stats = m.stats()
        self.assertEqual(stats["drained"], 0)
        self.assertEqual(stats["drained_in_hook"], 0)
        self.assertEqual(stats["drained_in_p2p_window"], 0)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestForwardEndHook(_ModuleStateMixin, unittest.TestCase):
    """`_forward_end_hook` fallback drain, latch disarm and permanent warning."""

    def test_disarmed_once_p2p_window_seen(self):
        m = indexer_loss_overlap
        m._P2P_WINDOW_SEEN = True
        calls = []
        m.enqueue(_make_work(_RecordingOwner(calls), seqlen=1))
        m._forward_end_hook(output_tensor=object(), step_id=0)
        # The latch stands the fallback down: nothing drains here.
        self.assertEqual(calls, [])
        self.assertEqual(m.pending(), 1)
        self.assertEqual(m._FORWARD_END_DRAINS, 0)
        self.assertEqual(m.stats()["drained"], 0)

    def test_fallback_drains_when_window_unseen(self):
        m = indexer_loss_overlap
        calls = []
        m.enqueue(_make_work(_RecordingOwner(calls), seqlen=9))
        m._forward_end_hook()
        self.assertEqual(calls, [9])
        self.assertEqual(m._FORWARD_END_DRAINS, 1)
        self.assertFalse(m._FALLBACK_WARNED)  # first drain is expected, no warn
        stats = m.stats()
        self.assertEqual(stats["drained"], 1)
        self.assertEqual(stats["drained_in_hook"], 1)
        # The fallback does not touch the p2p-window counter.
        self.assertEqual(stats["drained_in_p2p_window"], 0)

    def test_empty_queue_does_not_advance_fallback_counter(self):
        m = indexer_loss_overlap
        m._forward_end_hook()
        self.assertEqual(m._FORWARD_END_DRAINS, 0)
        self.assertFalse(m._FALLBACK_WARNED)

    def test_second_fallback_drain_warns_once(self):
        m = indexer_loss_overlap
        owner = _RecordingOwner([])
        # First fallback drain: expected, silent.
        m.enqueue(_make_work(owner, seqlen=1))
        m._forward_end_hook()
        self.assertEqual(m._FORWARD_END_DRAINS, 1)
        self.assertFalse(m._FALLBACK_WARNED)
        # Second fallback drain: proves P2P_ISSUED never armed -> warn once.
        m.enqueue(_make_work(owner, seqlen=2))
        with self.assertLogs(
            indexer_loss_overlap.logger, level="WARNING"
        ) as cm:
            m._forward_end_hook()
        self.assertEqual(m._FORWARD_END_DRAINS, 2)
        self.assertTrue(m._FALLBACK_WARNED)
        self.assertTrue(
            any("P2P_ISSUED" in msg for msg in cm.output), cm.output
        )


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestRegisterPipelineHooks(_ModuleStateMixin, unittest.TestCase):
    """`register_pipeline_hooks` side-effect-free early returns only.

    The real registration path mutates Paddle's process-global micro-step hook
    registry; that is out of scope here (see the module docstring). These two
    branches return before any registry mutation, so they are safe on CPU.
    """

    def test_idempotent_when_already_registered(self):
        m = indexer_loss_overlap
        m._HOOKS_REGISTERED = True
        # Early guard returns True without importing or touching Paddle at all.
        self.assertTrue(m.register_pipeline_hooks(pp_model=object()))

    def test_non_pipeline_model_returns_false_without_registering(self):
        m = indexer_loss_overlap
        # A plain object has no ``_forward_step``; whether or not Paddle exposes
        # the micro-step locations, the function must decline and stay unarmed.
        self.assertFalse(m.register_pipeline_hooks(pp_model=object()))
        self.assertFalse(m._HOOKS_REGISTERED)


if __name__ == "__main__":
    unittest.main()
