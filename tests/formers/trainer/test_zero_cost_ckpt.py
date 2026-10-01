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

"""Behavior tests for ``trainer/utils/zero_cost_checkpoint.py`` (no-card / CPU).

Oracles are derived independently of the production implementation:

* ``md5``: the expected digest is built from raw IEEE-754 little/big-endian bytes
  produced by ``struct.pack`` (not from ``tensor.numpy().tobytes()``), so a change
  to how the tensor is serialized is observable.
* ``_unwrap_opt_for_fused_states``: hand-built ``_inner_opt`` chains whose class
  names are known; the expected stop point is derived from the documented rule
  (stop at the first sharding-optimizer class name), not from the function.
* ``_zcc_lookup_local_shape``: expected ``local_shape`` is the one whose element
  product equals the requested ``numel``, computed by hand.
* ``sharded_state_dict_compatibility``: expected is that the wrapped callable sees
  plain local tensors (never ``ShardedWeight``) and that re-wrapping restores the
  original ``ShardedWeight`` objects (identity) carrying the callable's output.
* Manager barrier (``sync_offload_status`` / ``maybe_sync_offload_status``): the
  async D2H offload is only "done" once the worker publishes ``global_step.value``
  equal to the manager's step; the barrier must keep polling (read happens strictly
  after completion) and must watch ``current_worker`` specifically, never accept a
  stale echo from another worker. Completion timing is injected via a patched
  ``time.sleep`` that advances the worker after a controlled number of polls.
* Callback ``on_step_end``: manager.global_step must be refreshed to the current
  step on every offloading branch and left untouched on idle steps; PREPARE must be
  queued before the OFFLOAD hook is dispatched.
"""

import hashlib
import struct
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import paddle

    from paddlefleet.trainer.utils.zero_cost_checkpoint import (
        ZCCTaskType,
        ZCCWorkerStatus,
        ZeroCostCheckpointCallback,
        ZeroCostCheckpointManager,
        _unwrap_opt_for_fused_states,
        _zcc_lookup_local_shape,
        md5,
        sharded_state_dict_compatibility,
    )

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    _IMPORT_ERROR = str(exc)

requires_zcc = unittest.skipUnless(
    _IMPORT_ERROR is None,
    f"paddle/paddlefleet unavailable: {_IMPORT_ERROR}",
)


class _Holder:
    """Stand-in for a ``multiprocessing.Value``; only ``.value`` is read/written."""

    def __init__(self, value):
        self.value = value


class _RecordingQueue:
    """Records every ``put`` so the queued ZCC task can be inspected by value."""

    def __init__(self):
        self.puts = []

    def put(self, task):
        self.puts.append(task)


class _Worker:
    """Minimal ZCC worker double exposing only the fields the manager reads."""

    def __init__(self, worker_id=0, status=0, global_step=0):
        self.worker_id = worker_id
        self.status = _Holder(status)
        self.global_step = _Holder(global_step)
        self.task_queue = _RecordingQueue()


def _make_manager(**attrs):
    """Build a manager without spawning worker processes.

    ``__init__`` would spawn subprocesses and require an initialized fleet. The
    barrier / selection methods under test only touch the plain attributes set
    here, and the assertions observe what those *real* methods do (poll, select,
    queue, transition), never merely read back the attributes we assigned.
    """
    m = ZeroCostCheckpointManager.__new__(ZeroCostCheckpointManager)
    m.workers = attrs.get("workers", [])
    m.current_worker = attrs.get("current_worker", None)
    m.global_step = attrs.get("global_step", 0)
    m.pipeline_hooks_steps = attrs.get("pipeline_hooks_steps", 1)
    m.current_pipeline_hook_step = attrs.get("current_pipeline_hook_step", 1)
    m.ready_to_save = attrs.get("ready_to_save", True)
    return m


class _RecordingManager:
    """Recorder standing in for the manager collaborator of the callback.

    Records the ordered sequence of manager calls and the PREPARE payloads so the
    callback's bookkeeping (global_step refresh) and dispatch order can be checked.
    """

    _SENTINEL = object()

    def __init__(self):
        self.global_step = self._SENTINEL
        self.calls = []
        self.idle_payloads = []

    def get_idle_worker_for_saving(self, payload):
        self.calls.append("idle")
        self.idle_payloads.append(payload)

    def zcc_pipeline_hook(self, hook_id):
        self.calls.append(("hook", hook_id))


def _make_callback(manager, ema_interval=2):
    """Construct the callback via its real ``__init__`` then stub only the heavy
    ``maybe_update_zcc_worker`` collaborator (it touches fleet / IPC). The branch
    logic, ``_get_save_infos_based_on_steps`` and ``get_rng_states`` stay real."""
    args = SimpleNamespace(zcc_ema_interval=ema_interval)
    timer = SimpleNamespace(
        start=lambda *a, **k: None, stop=lambda *a, **k: None
    )
    cb = ZeroCostCheckpointCallback(args, manager, timer, None)
    cb.maybe_update_zcc_worker = lambda *a, **k: None
    return cb


_SLEEP = "paddlefleet.trainer.utils.zero_cost_checkpoint.time.sleep"


@requires_zcc
class TestMd5(unittest.TestCase):
    """md5(tensor) hashes the tensor's raw bytes."""

    def _byte_digest(self, floats):
        prefix = "<" if sys.byteorder == "little" else ">"
        raw = struct.pack(f"{prefix}{len(floats)}f", *floats)
        return hashlib.md5(raw).hexdigest()

    def test_matches_independent_byte_derivation(self):
        vals = [1.0, 2.0, -3.5]
        tensor = paddle.to_tensor(vals, dtype="float32")
        # Oracle: digest of IEEE-754 bytes built by struct, not tensor.tobytes().
        self.assertEqual(md5(tensor), self._byte_digest(vals))

    def test_distinguishes_content(self):
        a = paddle.to_tensor([1.0, 2.0, 3.0], dtype="float32")
        b = paddle.to_tensor([1.0, 2.0, 3.5], dtype="float32")
        self.assertEqual(md5(a), md5(a.clone()))  # same content -> same digest
        self.assertNotEqual(md5(a), md5(b))  # one differing element observable

    def test_dtype_changes_bytes(self):
        # Same nominal values, different dtype -> different serialized bytes.
        as_f32 = paddle.to_tensor([1.0, 2.0, 3.0], dtype="float32")
        as_i32 = paddle.to_tensor([1, 2, 3], dtype="int32")
        self.assertNotEqual(md5(as_f32), md5(as_i32))


class _NoName:
    """Leaf optimizer double with no ``_inner_opt`` attribute."""


class DygraphShardingOptimizer:  # name is what the production rule matches on
    def __init__(self, inner=None):
        if inner is not None:
            self._inner_opt = inner


class _PlainWrap:
    """Non-sharding wrapper whose class name is not a stop point."""

    def __init__(self, inner):
        self._inner_opt = inner


@requires_zcc
class TestUnwrapOptForFusedStates(unittest.TestCase):
    """_unwrap_opt_for_fused_states descends ``_inner_opt`` until a sharding
    optimizer class name is hit, else returns the innermost optimizer."""

    def test_no_inner_returns_self(self):
        opt = _NoName()
        self.assertIs(_unwrap_opt_for_fused_states(opt), opt)

    def test_descends_to_innermost_when_no_match(self):
        leaf = _NoName()
        opt = _PlainWrap(_PlainWrap(leaf))
        # None of the wrapper names match -> loop exits at the leaf w/o _inner_opt.
        self.assertIs(_unwrap_opt_for_fused_states(opt), leaf)

    def test_stops_at_sharding_name_without_descending_further(self):
        deeper = _NoName()
        sharding = DygraphShardingOptimizer(
            inner=deeper
        )  # has its own _inner_opt
        opt = _PlainWrap(sharding)
        result = _unwrap_opt_for_fused_states(opt)
        # Must return the sharding optimizer itself, NOT its inner ``deeper``.
        self.assertIs(result, sharding)
        self.assertIsNot(result, deeper)


@requires_zcc
class TestZccLookupLocalShape(unittest.TestCase):
    """_zcc_lookup_local_shape returns the recorded local_shape whose element
    product equals ``numel``; else None."""

    def _meta(self, mapping):
        return SimpleNamespace(state_dict_metadata=mapping)

    def test_returns_matching_shape(self):
        # product 2*3*2*4 == 48
        entry = SimpleNamespace(local_shape=[2, 3, 2, 4])
        shape = _zcc_lookup_local_shape(self._meta({"w": entry}), "w", 48)
        self.assertEqual(shape, [2, 3, 2, 4])

    def test_numel_mismatch_returns_none(self):
        entry = SimpleNamespace(local_shape=[2, 3, 2, 4])  # product 48
        self.assertIsNone(
            _zcc_lookup_local_shape(self._meta({"w": entry}), "w", 47)
        )

    def test_missing_key_returns_none(self):
        entry = SimpleNamespace(local_shape=[4])
        self.assertIsNone(
            _zcc_lookup_local_shape(self._meta({"w": entry}), "other", 4)
        )

    def test_absent_metadata_returns_none(self):
        self.assertIsNone(_zcc_lookup_local_shape(SimpleNamespace(), "w", 4))

    def test_picks_matching_candidate_from_list(self):
        wrong = SimpleNamespace(local_shape=[5, 5])  # product 25
        right = SimpleNamespace(local_shape=[3, 4])  # product 12
        shape = _zcc_lookup_local_shape(
            self._meta({"w": [wrong, right]}), "w", 12
        )
        # Must select the candidate matching numel, not simply the first one.
        self.assertEqual(shape, [3, 4])


@requires_zcc
class TestShardedStateDictCompatibility(unittest.TestCase):
    """The decorator unwraps ShardedWeight args into plain local tensors before
    calling the wrapped function, leaves non-sharded dicts alone, and (optionally)
    re-wraps the result back into the original ShardedWeight objects."""

    def _sharded_weight_cls(self):
        try:
            from paddle.distributed.flex_checkpoint.dcp.sharded_weight import (
                ShardedWeight,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"ShardedWeight unavailable: {exc}")
        return ShardedWeight

    def _make_sharded(self, cls, tensor):
        # ShardedWeight is passive data here (not under test); bypass its
        # constructor so the decorator's real isinstance / .local_tensor logic runs
        # without depending on the constructor signature.
        sw = cls.__new__(cls)
        sw.local_tensor = tensor
        return sw

    def test_converts_sharded_to_local_tensor(self):
        cls = self._sharded_weight_cls()
        t1 = paddle.to_tensor([1.0, 2.0])
        t2 = paddle.to_tensor([3.0, 4.0])
        sw1, sw2 = self._make_sharded(cls, t1), self._make_sharded(cls, t2)
        captured = {}

        @sharded_state_dict_compatibility
        def fn(state_dict):
            captured["sd"] = state_dict
            return state_dict

        fn({"w": sw1, "b": sw2})
        # The wrapped function must see the raw local tensors, never ShardedWeight.
        self.assertIs(captured["sd"]["w"], t1)
        self.assertIs(captured["sd"]["b"], t2)
        self.assertNotIsInstance(captured["sd"]["w"], cls)
        self.assertEqual(fn.__name__, "fn")  # functools.wraps preserved

    def test_normal_dict_passed_through_unchanged(self):
        t = paddle.to_tensor([5.0, 6.0])
        captured = {}

        @sharded_state_dict_compatibility
        def fn(state_dict):
            captured["sd"] = state_dict
            return state_dict

        fn({"w": t})
        self.assertIs(captured["sd"]["w"], t)  # identity preserved, no rewrap

    def test_mixed_dict_not_converted(self):
        cls = self._sharded_weight_cls()
        sw = self._make_sharded(cls, paddle.to_tensor([1.0]))
        plain = paddle.to_tensor([2.0])
        captured = {}

        @sharded_state_dict_compatibility
        def fn(state_dict):
            captured["sd"] = state_dict
            return state_dict

        fn({"a": sw, "b": plain})
        # any-but-not-all sharded -> left untouched, ShardedWeight still present.
        self.assertIs(captured["sd"]["a"], sw)
        self.assertIsInstance(captured["sd"]["a"], cls)
        self.assertIs(captured["sd"]["b"], plain)

    def test_return_rewraps_into_original_sharded_weight(self):
        cls = self._sharded_weight_cls()
        sw = self._make_sharded(cls, paddle.to_tensor([1.0, 2.0]))

        def fn(state_dict):
            # produce a distinguishable updated local tensor
            return {"w": state_dict["w"] + 10.0}

        wrapped = sharded_state_dict_compatibility(
            fn, return_sharded_state_dict=True
        )
        out = wrapped({"w": sw})
        # Same ShardedWeight object is returned (ownership preserved) ...
        self.assertIs(out["w"], sw)
        # ... now carrying the function's output as its local tensor.
        self.assertIs(out["w"].local_tensor, sw.local_tensor)
        import numpy as np

        np.testing.assert_array_equal(
            out["w"].local_tensor.numpy(),
            np.array([11.0, 12.0], dtype="float32"),
        )


@requires_zcc
class TestGetIdleWorkerForSaving(unittest.TestCase):
    """Selection claims the first IDLE worker, skips busy ones, queues a PREPARE
    task carrying the payload, and propagates a worker error."""

    def test_selects_idle_and_queues_prepare_payload(self):
        idle = _Worker(worker_id=0, status=ZCCWorkerStatus.IDLE.value)
        m = _make_manager(workers=[idle], current_worker=None)
        payload = (("flash", "persistent"), ("lr", "state", "rng"))
        m.get_idle_worker_for_saving(payload)
        self.assertIs(m.current_worker, idle)
        self.assertEqual(len(idle.task_queue.puts), 1)
        task_type, task_payload = idle.task_queue.puts[0]
        self.assertIs(task_type, ZCCTaskType.PREPARE)
        self.assertIs(task_payload, payload)  # exact payload forwarded

    def test_skips_busy_worker(self):
        busy = _Worker(worker_id=0, status=ZCCWorkerStatus.DUMPING.value)
        idle = _Worker(worker_id=1, status=ZCCWorkerStatus.IDLE.value)
        m = _make_manager(workers=[busy, idle], current_worker=None)
        m.get_idle_worker_for_saving(None)
        self.assertIs(m.current_worker, idle)  # busy worker not claimed
        self.assertEqual(len(busy.task_queue.puts), 0)
        self.assertEqual(len(idle.task_queue.puts), 1)

    def test_error_worker_propagates(self):
        bad = _Worker(worker_id=0, status=ZCCWorkerStatus.ERROR.value)
        m = _make_manager(workers=[bad], current_worker=None)
        with self.assertRaises(RuntimeError):
            m.get_idle_worker_for_saving(None)


@requires_zcc
class TestSyncOffloadStatus(unittest.TestCase):
    """The barrier reads only after the worker signals completion (its published
    global_step equals the manager's step) and watches ``current_worker`` alone."""

    def test_polls_until_worker_step_catches_up(self):
        worker = _Worker(global_step=4)  # stale echo of a previous step
        m = _make_manager(
            workers=[worker], current_worker=worker, global_step=5
        )

        def advance(_seconds):
            advance.n += 1
            if advance.n >= 2:
                worker.global_step.value = 5  # D2H finishes on the 2nd poll

        advance.n = 0
        with patch(_SLEEP, side_effect=advance) as slept:
            m.sync_offload_status()

        self.assertGreaterEqual(slept.call_count, 2)  # did not accept step 4
        self.assertIsNone(m.current_worker)  # released only after match
        self.assertEqual(m.current_pipeline_hook_step, 0)

    def test_returns_immediately_when_step_matches(self):
        worker = _Worker(global_step=7)
        m = _make_manager(
            workers=[worker], current_worker=worker, global_step=7
        )
        with patch(_SLEEP) as slept:
            m.sync_offload_status()
        slept.assert_not_called()
        self.assertIsNone(m.current_worker)
        self.assertEqual(m.current_pipeline_hook_step, 0)

    def test_ignores_stale_echo_from_other_worker(self):
        target = _Worker(
            worker_id=0, global_step=4
        )  # current, still offloading
        decoy = _Worker(worker_id=1, global_step=0)
        m = _make_manager(
            workers=[target, decoy], current_worker=target, global_step=5
        )

        def advance(_seconds):
            advance.n += 1
            if advance.n == 1:
                decoy.global_step.value = (
                    5  # another worker echoes current step
                )
                self.assertIs(m.current_worker, target)  # must not release
            else:
                target.global_step.value = 5  # real completion

        advance.n = 0
        with patch(_SLEEP, side_effect=advance) as slept:
            m.sync_offload_status()

        self.assertGreaterEqual(slept.call_count, 2)
        self.assertIsNone(m.current_worker)
        self.assertEqual(target.global_step.value, 5)


@requires_zcc
class TestMaybeSyncOffloadStatus(unittest.TestCase):
    """Step-begin variant: sync only when the whole offload was already dispatched
    (current_pipeline_hook_step == pipeline_hooks_steps); otherwise decline so the
    trainer does not block on chunks that this step has yet to emit."""

    def test_no_current_worker_is_noop(self):
        m = _make_manager(current_worker=None, current_pipeline_hook_step=1)
        with patch(_SLEEP) as slept:
            m.maybe_sync_offload_status()
        slept.assert_not_called()
        self.assertIsNone(m.current_worker)
        self.assertEqual(m.current_pipeline_hook_step, 1)  # untouched

    def test_fully_dispatched_syncs_and_releases(self):
        worker = _Worker(global_step=5)  # already completed
        m = _make_manager(
            workers=[worker],
            current_worker=worker,
            global_step=5,
            pipeline_hooks_steps=1,
            current_pipeline_hook_step=1,
        )
        with patch(_SLEEP) as slept:
            m.maybe_sync_offload_status()
        slept.assert_not_called()
        self.assertIsNone(m.current_worker)  # sync ran -> released
        self.assertEqual(m.current_pipeline_hook_step, 0)

    def test_partially_dispatched_declines_without_blocking(self):
        # Worker step lags; had sync been (wrongly) entered it would poll forever.
        worker = _Worker(global_step=4)
        m = _make_manager(
            workers=[worker],
            current_worker=worker,
            global_step=5,
            pipeline_hooks_steps=4,
            current_pipeline_hook_step=1,
        )

        def fail_if_slept(_seconds):
            raise AssertionError(
                "barrier must not block when not fully dispatched"
            )

        with patch(_SLEEP, side_effect=fail_if_slept):
            m.maybe_sync_offload_status()
        self.assertIs(m.current_worker, worker)  # unchanged
        self.assertEqual(m.current_pipeline_hook_step, 1)


@requires_zcc
class TestCallbackOnStepEnd(unittest.TestCase):
    """on_step_end refreshes manager.global_step on every offloading branch, leaves
    it untouched on idle steps, and dispatches PREPARE before the OFFLOAD hook."""

    def _run(self, cb, *, should_save, ema_coef, global_step):
        args = SimpleNamespace(
            zcc_save_ema_coef=ema_coef,
            pipeline_model_parallel_size=1,
            save_rng_states=False,  # keep real get_rng_states -> None
            flash_device_save_steps=0,  # keep real _get_save_infos -> (None, None)
            save_steps=0,
            output_dir="/tmp/zcc-test-out",
        )
        cb.on_step_end(
            args,
            SimpleNamespace(global_step=global_step),
            SimpleNamespace(should_save=should_save),
            model=object(),  # not a PipelineLayer -> non-PP dispatch path
            lr_scheduler=SimpleNamespace(state_dict=lambda: {"lr": 0.1}),
            optimizer=object(),
        )

    def test_save_branch_refreshes_step_and_orders_dispatch(self):
        mgr = _RecordingManager()
        cb = _make_callback(mgr)
        self._run(cb, should_save=True, ema_coef=None, global_step=7)
        self.assertEqual(mgr.global_step, 7)
        self.assertIn("idle", mgr.calls)
        self.assertIn(("hook", 0), mgr.calls)
        # PREPARE (idle pick) must precede the OFFLOAD hook.
        self.assertLess(mgr.calls.index("idle"), mgr.calls.index(("hook", 0)))
        save_infos, non_cached = mgr.idle_payloads[0]
        self.assertEqual(save_infos, (None, None))
        self.assertEqual(non_cached[0], {"lr": 0.1})  # lr scheduler state
        self.assertIsNone(non_cached[2])  # rng states disabled

    def test_ema_branch_refreshes_step_with_prepare_only_payload(self):
        mgr = _RecordingManager()
        cb = _make_callback(mgr, ema_interval=2)
        self._run(cb, should_save=False, ema_coef=0.999, global_step=8)
        self.assertEqual(mgr.global_step, 8)  # 8 % 2 == 0 -> offloads
        save_infos, non_cached = mgr.idle_payloads[0]
        self.assertEqual(save_infos, (None, None))  # prepare-only, no dump dirs
        self.assertIsNone(non_cached[0])
        self.assertEqual(
            non_cached[1].global_step, 8
        )  # trainer state forwarded
        self.assertIsNone(non_cached[2])
        self.assertIn(("hook", 0), mgr.calls)

    def test_off_interval_step_leaves_manager_step_untouched(self):
        mgr = _RecordingManager()
        cb = _make_callback(mgr, ema_interval=2)
        self._run(cb, should_save=False, ema_coef=0.999, global_step=7)
        self.assertIs(
            mgr.global_step, _RecordingManager._SENTINEL
        )  # not refreshed
        self.assertNotIn("idle", mgr.calls)  # no worker claimed
        self.assertIn(
            ("hook", 0), mgr.calls
        )  # hook still fired (guarded in mgr)

    def test_ema_disabled_step_is_idle(self):
        mgr = _RecordingManager()
        cb = _make_callback(mgr, ema_interval=2)
        self._run(cb, should_save=False, ema_coef=None, global_step=8)
        self.assertIs(mgr.global_step, _RecordingManager._SENTINEL)
        self.assertNotIn("idle", mgr.calls)
        self.assertIn(("hook", 0), mgr.calls)


if __name__ == "__main__":
    unittest.main()
