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

"""Behavior tests for ``trainer/utils/offload_optimizer.py``.

Module under test: paddlefleet.trainer.utils.offload_optimizer.

Independent oracle
------------------
Every expectation below is hand-derived by reading the offload wrapper's
control flow, NOT by re-running the wrappers or copying any other test:

* ``offload(t)`` selects a place from the compile flags (CUDA -> CUDAPinnedPlace,
  XPU -> XPUPinnedPlace, else CPUPlace), calls ``to_device(t, place)``, asserts
  the migration is in-place (returns the same object) and returns ``None``.
* ``reload(t)`` calls ``to_device(t)`` with no place, so the target is the
  default device (``get_env_device()``); on a CPU build that is CPUPlace.
* The ``_add_accumulator`` wrapper calls the original, then offloads the freshly
  created accumulator -- i.e. state offload happens *at accumulator creation*.
* The ``adam_``/``adamw_`` wrapper reloads *every* tensor argument before the
  op and offloads only arguments at index >= 2 (state; never param/grad),
  gated by ``getattr(args[0], "is_offload_opt", <default>)``. The default is
  ``True`` for ``hack_offload_optimizer`` but ``False`` for
  ``hack_offload_optimizer_eb5`` -- an observable behavioral difference.
* The ``_insert_sync`` wrapper reloads the sync var, runs the original, then
  moves it back to its original place when ``is_offload_opt``.

These are CPU/no-GPU observations of the wrapper's *dispatch/lifecycle* control
logic only. Actual device migration, pinned-memory residency and real memory
release require a GPU and are NOT claimed here; the ``to_device`` primitive is
spied (an external collaborator from sharding_io) purely to observe which
tensors the wrapper decides to reload/offload and in what order.
"""

import contextlib
import unittest
from unittest import mock

try:
    import numpy as np
    import paddle
    from paddle import _C_ops
    from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.hybrid_parallel_optimizer import (
        HybridParallelOptimizer,
    )
    from paddle.optimizer import Optimizer

    from paddlefleet.trainer.utils import offload_optimizer as oo

    HAS_PADDLE = True
    _CUDA = paddle.is_compiled_with_cuda()
    _XPU = paddle.is_compiled_with_xpu()
except ImportError:
    HAS_PADDLE = False
    _CUDA = False
    _XPU = False

_MISSING = object()


class _ToDeviceSpy:
    """Records ``to_device`` invocations and returns the tensor unchanged.

    Returning the same tensor satisfies the in-place ``assert new is old``
    checks inside ``offload``/``reload``. ``reload`` calls ``to_device(t)`` so
    ``place is None``; ``offload`` and the ``_insert_sync`` move-back call
    ``to_device(t, place)`` with a concrete place. That lets us classify each
    recorded call as a reload (place is None) or an offload/move-back.
    """

    def __init__(self):
        self.calls = []

    def __call__(self, tensor, place=None):
        self.calls.append((tensor, place))
        return tensor

    @property
    def reloaded(self):
        return [t for (t, p) in self.calls if p is None]

    @property
    def offloaded(self):
        return [t for (t, p) in self.calls if p is not None]

    @property
    def offload_places(self):
        return [p for (t, p) in self.calls if p is not None]


@contextlib.contextmanager
def _force_cpu():
    """Force the CPU branch of ``offload`` so its target place is CPUPlace."""
    with (
        mock.patch("paddle.is_compiled_with_cuda", return_value=False),
        mock.patch("paddle.is_compiled_with_xpu", return_value=False),
    ):
        yield


def _tensor(values):
    return paddle.to_tensor(values, dtype="float32")


def _ids(tensors):
    return [id(t) for t in tensors]


@unittest.skipUnless(HAS_PADDLE, "paddle / paddlefleet not importable")
class OffloadReloadRealTest(unittest.TestCase):
    """Real (unmocked) offload/reload on a CPU build: identity + content."""

    def test_offload_selects_cpu_place_and_is_inplace_on_cpu_build(self):
        x = _tensor([[1.0, 2.0], [3.0, 4.0]])
        before = x.numpy().copy()
        with _force_cpu():
            ret = oo.offload(x)
        # No return value; migration is in-place so x itself is the result.
        self.assertIsNone(ret)
        self.assertTrue(x.place.is_cpu_place())
        np.testing.assert_array_equal(x.numpy(), before)

    def test_reload_targets_default_device_and_is_inplace_on_cpu_build(self):
        x = _tensor([9.0, 8.0, 7.0])
        before = x.numpy().copy()
        with (
            mock.patch("paddle.is_compiled_with_cuda", return_value=False),
            mock.patch("paddle.is_compiled_with_xpu", return_value=False),
            mock.patch("paddle.is_compiled_with_rocm", return_value=False),
        ):
            ret = oo.reload(x)
        self.assertIsNone(ret)
        self.assertTrue(x.place.is_cpu_place())
        np.testing.assert_array_equal(x.numpy(), before)

    def test_reload_passes_no_place_so_target_is_default_device(self):
        # Oracle: reload calls to_device(t) with no place argument.
        x = _tensor([1.0])
        spy = _ToDeviceSpy()
        with mock.patch.object(oo, "to_device", spy):
            oo.reload(x)
        self.assertEqual(len(spy.calls), 1)
        t, place = spy.calls[0]
        self.assertIs(t, x)
        self.assertIsNone(place)

    @unittest.skipUnless(
        _CUDA or _XPU, "pinned-place branch requires a CUDA/XPU build"
    )
    def test_offload_selects_pinned_place_on_accelerator_build(self):
        # Observes only the place-selection branch; no real migration.
        x = _tensor([1.0, 2.0])
        spy = _ToDeviceSpy()
        with mock.patch.object(oo, "to_device", spy):
            oo.offload(x)
        self.assertEqual(len(spy.calls), 1)
        t, place = spy.calls[0]
        self.assertIs(t, x)
        if _CUDA:
            self.assertIsInstance(place, paddle.CUDAPinnedPlace)
        else:
            self.assertIsInstance(place, paddle.XPUPinnedPlace)


@unittest.skipUnless(HAS_PADDLE, "paddle / paddlefleet not importable")
class HackOffloadWrapperTest(unittest.TestCase):
    """Wrapper dispatch/lifecycle for hack_offload_optimizer[_eb5]."""

    def _snapshot_hack_targets(self):
        """Save every attribute the hack patches; restore on cleanup.

        The hacks mutate global/class state (``_C_ops``, ``Optimizer``,
        ``HybridParallelOptimizer`` and optionally ``Muon``). Snapshot the
        *true* originals first, then register cleanup so any locally installed
        stubs and hack patches are undone even if an assertion fails.
        """
        saved = {}
        saved[(_C_ops, "adam_")] = getattr(_C_ops, "adam_", _MISSING)
        saved[(_C_ops, "adamw_")] = getattr(_C_ops, "adamw_", _MISSING)
        saved[(Optimizer, "_add_accumulator")] = Optimizer._add_accumulator
        saved[(HybridParallelOptimizer, "_insert_sync")] = (
            HybridParallelOptimizer._insert_sync
        )
        try:
            from paddle.optimizer.muon import Muon

            for attr in (
                "_muon_update_group",
                "_muon_update",
                "_apply_optimize",
            ):
                if hasattr(Muon, attr):
                    saved[(Muon, attr)] = getattr(Muon, attr)
        except ImportError:
            pass

        def restore():
            for (obj, name), val in saved.items():
                if val is _MISSING:
                    if hasattr(obj, name):
                        delattr(obj, name)
                else:
                    setattr(obj, name, val)

        self.addCleanup(restore)

    def _install_op_stubs(self):
        """Replace adam_/adamw_ with labeled stubs (the wrapped collaborators)."""
        labels = []

        def stub_adam(*args):
            labels.append("adam")
            return ("adam_ret", args)

        def stub_adamw(*args):
            labels.append("adamw")
            return ("adamw_ret", args)

        _C_ops.adam_ = stub_adam
        _C_ops.adamw_ = stub_adamw
        return labels

    def _probe_adamw_offloaded(self, hack_call):
        """Run a hack, drive the patched adamw_, return (spy, ret, tensors)."""
        self._snapshot_hack_targets()
        self._install_op_stubs()
        param, grad = _tensor([1.0]), _tensor([2.0])
        m1, m2 = _tensor([3.0]), _tensor([4.0])
        spy = _ToDeviceSpy()
        with _force_cpu(), mock.patch.object(oo, "to_device", spy):
            hack_call()
            ret = _C_ops.adamw_(param, grad, m1, m2)
        return spy, ret, (param, grad, m1, m2)

    def test_adamw_reloads_all_and_offloads_only_state_by_default(self):
        spy, ret, (param, grad, m1, m2) = self._probe_adamw_offloaded(
            oo.hack_offload_optimizer
        )
        # Reload happens for every tensor arg, in order, before the op.
        self.assertEqual(_ids(spy.reloaded), _ids([param, grad, m1, m2]))
        # Only state (index >= 2) is offloaded; param/grad are never offloaded.
        self.assertEqual(_ids(spy.offloaded), _ids([m1, m2]))
        # Offload target is CPUPlace on a CPU build.
        for place in spy.offload_places:
            self.assertTrue(place.is_cpu_place())
        # Call-through: the wrapped op ran and its return propagates.
        self.assertEqual(ret[0], "adamw_ret")

    def test_adamw_eb5_does_not_offload_state_by_default(self):
        spy, ret, (param, grad, m1, m2) = self._probe_adamw_offloaded(
            oo.hack_offload_optimizer_eb5
        )
        # eb5 default is_offload_opt=False: still reloads, but offloads nothing.
        self.assertEqual(_ids(spy.reloaded), _ids([param, grad, m1, m2]))
        self.assertEqual(spy.offloaded, [])
        self.assertEqual(ret[0], "adamw_ret")

    def test_mode_eb5_delegates_to_eb5_variant(self):
        # Observed via the eb5-specific default (no offload), not just "patched".
        spy, ret, _ = self._probe_adamw_offloaded(
            lambda: oo.hack_offload_optimizer(mode="eb5")
        )
        self.assertEqual(spy.offloaded, [])
        self.assertEqual(ret[0], "adamw_ret")

    @unittest.expectedFailure
    def test_adam_wrapper_should_dispatch_to_adam_not_adamw(self):
        """Known production bug: late-binding closure over ``origin_op``.

        Step 2 defines ``new_opt_op`` inside a ``for name in [...]`` loop that
        rebinds ``origin_op`` each iteration. Both installed closures share that
        one cell, so after the loop both dispatch to the *last* original
        (adamw_). Calling the patched ``adam_`` therefore runs adamw_'s original.
        This asserts the correct contract (adam_ -> adam_ original) and is
        expected to fail until the closure captures ``origin_op`` per name.
        """
        self._snapshot_hack_targets()
        labels = self._install_op_stubs()
        param, grad = _tensor([1.0]), _tensor([2.0])
        m1, m2 = _tensor([3.0]), _tensor([4.0])
        spy = _ToDeviceSpy()
        with _force_cpu(), mock.patch.object(oo, "to_device", spy):
            oo.hack_offload_optimizer()
            _C_ops.adam_(param, grad, m1, m2)
        self.assertEqual(labels, ["adam"])

    def test_add_accumulator_offloads_new_state_at_creation(self):
        self._snapshot_hack_targets()
        acc = _tensor([7.0, 8.0])
        recorded = {}

        def stub_add(self_opt, *args, **kwargs):
            recorded["args"] = args
            recorded["kwargs"] = kwargs
            return acc

        Optimizer._add_accumulator = stub_add
        spy = _ToDeviceSpy()
        dummy = object()
        with _force_cpu(), mock.patch.object(oo, "to_device", spy):
            oo.hack_offload_optimizer()
            ret = Optimizer._add_accumulator(dummy, "moment", fill_value=0.0)
        # Call-through returns the freshly created accumulator unchanged.
        self.assertIs(ret, acc)
        self.assertEqual(recorded["args"], ("moment",))
        self.assertEqual(recorded["kwargs"], {"fill_value": 0.0})
        # The new state is offloaded exactly once, at creation; nothing reloaded.
        self.assertEqual(_ids(spy.offloaded), _ids([acc]))
        self.assertEqual(spy.reloaded, [])
        for place in spy.offload_places:
            self.assertTrue(place.is_cpu_place())

    def test_insert_sync_reloads_then_moves_back_by_default(self):
        self._snapshot_hack_targets()
        sync_var = _tensor([5.0])
        marker = object()
        recorded = {}

        def stub_insert(self_opt, sv, *args, **kwargs):
            recorded["sync_var"] = sv
            recorded["args"] = args
            return marker

        HybridParallelOptimizer._insert_sync = stub_insert
        spy = _ToDeviceSpy()
        dummy = object()
        with _force_cpu(), mock.patch.object(oo, "to_device", spy):
            oo.hack_offload_optimizer()
            ret = HybridParallelOptimizer._insert_sync(dummy, sync_var, "extra")
        self.assertIs(ret, marker)
        self.assertIs(recorded["sync_var"], sync_var)
        self.assertEqual(recorded["args"], ("extra",))
        # Lifecycle: reload before op, then move back to the original place.
        self.assertEqual(_ids(spy.reloaded), _ids([sync_var]))
        self.assertEqual(_ids(spy.offloaded), _ids([sync_var]))
        self.assertTrue(spy.offload_places[0].is_cpu_place())

    def test_insert_sync_eb5_skips_move_back(self):
        self._snapshot_hack_targets()
        sync_var = _tensor([5.0])

        def stub_insert(self_opt, sv, *args, **kwargs):
            return "ok"

        HybridParallelOptimizer._insert_sync = stub_insert
        spy = _ToDeviceSpy()
        dummy = object()
        with _force_cpu(), mock.patch.object(oo, "to_device", spy):
            oo.hack_offload_optimizer_eb5()
            HybridParallelOptimizer._insert_sync(dummy, sync_var, "extra")
        # eb5 default is_offload_opt=False: reload happens, move-back does not.
        self.assertEqual(_ids(spy.reloaded), _ids([sync_var]))
        self.assertEqual(spy.offloaded, [])


if __name__ == "__main__":
    unittest.main()
