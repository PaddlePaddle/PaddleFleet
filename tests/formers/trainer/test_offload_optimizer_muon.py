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

"""Unit tests for Muon master-weight offload in trainer/utils/offload_optimizer.py.

The optimizer hack wraps Muon's Python-side momentum update so that both the
momentum buffer *and* the master weight are reloaded to device before the real
update and offloaded back afterwards, at per-update-group granularity. This
replaces the coarse ``_apply_optimize`` wrapper that used to load every master
weight at once.

These tests drive the wrapped entry points directly with light stubs and record
the ordering of ``reload`` / ``offload`` calls, asserting:
  1. master weight is reloaded *before* the wrapped update runs;
  2. master weight is offloaded *after* the update, but only when the owning
     param has ``is_offload_opt`` truthy;
  3. the same contract holds for the ``_muon_update`` per-param fallback used on
     older Paddle, and for the ``_eb5`` variant of the hack.
"""

import unittest
from unittest.mock import patch

import paddle
from paddle import _C_ops
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.hybrid_parallel_optimizer import (
    HybridParallelOptimizer,
)
from paddle.optimizer import Optimizer

import paddlefleet.trainer.utils.offload_optimizer as offload_mod
from paddlefleet.trainer.utils.offload_optimizer import (
    hack_offload_optimizer,
    hack_offload_optimizer_eb5,
)

try:
    from paddle.optimizer.muon import Muon

    _HAS_MUON = True
except ImportError:  # pragma: no cover - Muon missing on very old Paddle
    Muon = None
    _HAS_MUON = False


class _Param:
    """Stand-in Muon param exposing only ``name`` and the offload gate."""

    def __init__(self, name, is_offload_opt):
        self.name = name
        self.is_offload_opt = is_offload_opt


class _MuonSelf:
    """Minimal ``self`` for the wrapped Muon update entry points."""

    _moment_acc_str = "moment1"

    def __init__(self, master_weights, momentums):
        self._master_weights = master_weights
        self._momentums = momentums

    def _get_accumulator(self, acc_str, param):
        return self._momentums[param.name]


@unittest.skipUnless(_HAS_MUON, "Muon optimizer unavailable")
class TestMuonMasterWeightOffload(unittest.TestCase):
    """Verify the reload-before / offload-after master-weight lifecycle."""

    def setUp(self):
        # Snapshot every global the hack rebinds so the process stays hermetic.
        self._saved = {
            "add_acc": Optimizer._add_accumulator,
            "adam": _C_ops.adam_,
            "adamw": _C_ops.adamw_,
            "sync": HybridParallelOptimizer._insert_sync,
            "group": getattr(Muon, "_muon_update_group", None),
            "update": getattr(Muon, "_muon_update", None),
        }

    def tearDown(self):
        Optimizer._add_accumulator = self._saved["add_acc"]
        _C_ops.adam_ = self._saved["adam"]
        _C_ops.adamw_ = self._saved["adamw"]
        HybridParallelOptimizer._insert_sync = self._saved["sync"]
        self._restore(Muon, "_muon_update_group", self._saved["group"])
        self._restore(Muon, "_muon_update", self._saved["update"])

    @staticmethod
    def _restore(cls, name, original):
        if original is not None:
            setattr(cls, name, original)
        elif hasattr(cls, name):
            delattr(cls, name)

    def _record(self, events):
        """Patch module-level reload/offload to append tagged (op, id) events."""
        return (
            patch.object(
                offload_mod,
                "reload",
                lambda t: events.append(("reload", id(t))),
            ),
            patch.object(
                offload_mod,
                "offload",
                lambda t: events.append(("offload", id(t))),
            ),
        )

    def _drive_group(self, stub, params_grads, hack=hack_offload_optimizer):
        """Install a recording original _muon_update_group, apply the hack, and
        run the resulting wrapper. Returns the ordered event log."""
        events = []
        Muon._muon_update_group = lambda self, pg, *a, **k: events.append(
            ("update",)
        )
        hack()
        wrapper = Muon._muon_update_group
        r, o = self._record(events)
        with r, o:
            wrapper(stub, params_grads)
        return events

    def _drive_update(self, stub, param, momentum, hack=hack_offload_optimizer):
        """Force the per-param _muon_update fallback (no _muon_update_group),
        apply the hack, and run the resulting wrapper. Returns the event log."""
        events = []
        if hasattr(Muon, "_muon_update_group"):
            delattr(Muon, "_muon_update_group")
        Muon._muon_update = lambda self, *a, **k: events.append(("update",))
        hack()
        wrapper = Muon._muon_update
        r, o = self._record(events)
        with r, o:
            wrapper(
                stub,
                param,
                paddle.zeros([2, 2]),
                0.1,
                momentum,
                0.95,
                5,
                True,
                1e-9,
                0.0,
                3,
            )
        return events

    # ----- _muon_update_group (batched, newer Paddle) --------------------

    def test_group_reload_before_offload_after_when_enabled(self):
        mw, mom = paddle.zeros([2, 2]), paddle.zeros([2, 2])
        param = _Param("w0", is_offload_opt=True)
        stub = _MuonSelf({"w0": mw}, {"w0": mom})

        events = self._drive_group(stub, [(param, paddle.zeros([2, 2]))])

        i = events.index(("update",))
        self.assertIn(("reload", id(mw)), events[:i])
        self.assertIn(("reload", id(mom)), events[:i])
        self.assertIn(("offload", id(mw)), events[i + 1 :])
        self.assertIn(("offload", id(mom)), events[i + 1 :])

    def test_group_reloads_but_never_offloads_when_disabled(self):
        mw, mom = paddle.zeros([2, 2]), paddle.zeros([2, 2])
        param = _Param("w0", is_offload_opt=False)
        stub = _MuonSelf({"w0": mw}, {"w0": mom})

        events = self._drive_group(stub, [(param, paddle.zeros([2, 2]))])

        i = events.index(("update",))
        self.assertIn(("reload", id(mw)), events[:i])
        self.assertNotIn(("offload", id(mw)), events)
        self.assertNotIn(("offload", id(mom)), events)

    def test_group_offloads_only_enabled_param_in_mixed_group(self):
        mw_on, mw_off = paddle.zeros([2, 2]), paddle.zeros([2, 2])
        mom_on, mom_off = paddle.zeros([2, 2]), paddle.zeros([2, 2])
        p_on = _Param("on", is_offload_opt=True)
        p_off = _Param("off", is_offload_opt=False)
        stub = _MuonSelf(
            {"on": mw_on, "off": mw_off}, {"on": mom_on, "off": mom_off}
        )

        events = self._drive_group(
            stub, [(p_on, paddle.zeros([2, 2])), (p_off, paddle.zeros([2, 2]))]
        )

        i = events.index(("update",))
        self.assertIn(("reload", id(mw_on)), events[:i])
        self.assertIn(("reload", id(mw_off)), events[:i])
        self.assertIn(("offload", id(mw_on)), events[i + 1 :])
        self.assertNotIn(("offload", id(mw_off)), events)
        self.assertNotIn(("offload", id(mom_off)), events)

    def test_group_no_master_weights_only_moves_momentum(self):
        mom = paddle.zeros([2, 2])
        param = _Param("w0", is_offload_opt=True)
        stub = _MuonSelf(None, {"w0": mom})  # _master_weights is falsy

        events = self._drive_group(stub, [(param, paddle.zeros([2, 2]))])

        i = events.index(("update",))
        self.assertIn(("reload", id(mom)), events[:i])
        self.assertIn(("offload", id(mom)), events[i + 1 :])
        # Only momentum is ever touched; there is nothing else to move.
        self.assertEqual(
            [e for e in events if e[0] in ("reload", "offload")],
            [("reload", id(mom)), ("offload", id(mom))],
        )

    # ----- _muon_update (per-param fallback, older Paddle) ---------------

    def test_update_reload_before_offload_after_when_enabled(self):
        mw, mom = paddle.zeros([2, 2]), paddle.zeros([2, 2])
        param = _Param("w0", is_offload_opt=True)
        stub = _MuonSelf({"w0": mw}, {"w0": mom})

        events = self._drive_update(stub, param, mom)

        i = events.index(("update",))
        self.assertIn(("reload", id(mw)), events[:i])
        self.assertIn(("reload", id(mom)), events[:i])
        self.assertIn(("offload", id(mw)), events[i + 1 :])
        self.assertIn(("offload", id(mom)), events[i + 1 :])

    def test_update_reloads_but_never_offloads_when_disabled(self):
        mw, mom = paddle.zeros([2, 2]), paddle.zeros([2, 2])
        param = _Param("w0", is_offload_opt=False)
        stub = _MuonSelf({"w0": mw}, {"w0": mom})

        events = self._drive_update(stub, param, mom)

        i = events.index(("update",))
        self.assertIn(("reload", id(mw)), events[:i])
        self.assertNotIn(("offload", id(mw)), events)
        self.assertNotIn(("offload", id(mom)), events)

    # ----- _eb5 variant shares the same contract -------------------------

    def test_eb5_group_reload_before_offload_after_when_enabled(self):
        mw, mom = paddle.zeros([2, 2]), paddle.zeros([2, 2])
        param = _Param("w0", is_offload_opt=True)
        stub = _MuonSelf({"w0": mw}, {"w0": mom})

        events = self._drive_group(
            stub,
            [(param, paddle.zeros([2, 2]))],
            hack=hack_offload_optimizer_eb5,
        )

        i = events.index(("update",))
        self.assertIn(("reload", id(mw)), events[:i])
        self.assertIn(("offload", id(mw)), events[i + 1 :])

    def test_eb5_update_reloads_but_never_offloads_when_disabled(self):
        mw, mom = paddle.zeros([2, 2]), paddle.zeros([2, 2])
        param = _Param("w0", is_offload_opt=False)
        stub = _MuonSelf({"w0": mw}, {"w0": mom})

        events = self._drive_update(
            stub, param, mom, hack=hack_offload_optimizer_eb5
        )

        i = events.index(("update",))
        self.assertIn(("reload", id(mw)), events[:i])
        self.assertNotIn(("offload", id(mw)), events)


if __name__ == "__main__":
    unittest.main()
