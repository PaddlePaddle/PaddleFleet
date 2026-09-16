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

"""Tests for the Paddle runtime *install* patch family of
``paddlefleet.accuracy_compatible_patch``.

Every function exercised here MUTATES global paddle/fleet classes and module
attributes (the tensor-fusion helpers, ``DygraphShardingOptimizerV2``,
``paddle.optimizer.AdamW``, the module install flag and the torch-import
globals). Each test therefore saves the originals in ``setUp`` and restores
them in ``tearDown`` so the shared pytest process is left exactly as it was
found; restoration is asserted where it matters.

Single card only: no distributed process group is initialised, so the
collective bodies of the installed ``comm_grads`` (and the static-graph-only
inner body of the AdamW append hook) are intentionally left uncovered - see
the module-level notes on each test.
"""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import paddle

from paddlefleet import accuracy_compatible_patch as acp


def setUpModule():
    if paddle.is_compiled_with_cuda():
        paddle.set_device("gpu")


_CONSUMERS = (
    "paddle.distributed.fleet.meta_optimizers.dygraph_optimizer."
    "dygraph_sharding_optimizer",
    "paddle.distributed.fleet.meta_optimizers.muon_sharding_optimizer",
    "paddle.distributed.fleet.meta_parallel.pipeline_parallel",
)
_MISSING = object()


class TestGroupIndices(unittest.TestCase):
    """``gptlm_head`` params are split into their own fusion group.

    Pure function - drives the pending-flush (75-77), single-index append (78)
    and trailing-pending append (81-82) paths with a fake helper/params.
    """

    def test_lm_head_in_middle_flushes_pending_and_splits(self):
        params = [
            SimpleNamespace(name="embedding", shape=[2, 2]),
            SimpleNamespace(name="block.gptlm_head.weight", shape=[2, 2]),
            SimpleNamespace(name="linear.a", shape=[2, 2]),
            SimpleNamespace(name="linear.b", shape=[2, 2]),
            SimpleNamespace(name="tail.a", shape=[2, 2]),
            SimpleNamespace(name="tail.b", shape=[2, 2]),
        ]
        helper = SimpleNamespace(
            core=SimpleNamespace(
                eager_assign_group_by_size=(
                    lambda parameters, flags, sizes: [[0, 1, 2, 3], [4, 5]]
                )
            )
        )

        groups = acp._group_indices(params, 1024, helper)

        # group0: pending [0] flushed by the lm_head at 1, lm_head alone as
        # [1], then [2, 3] trails; group1 has no lm_head so it stays [4, 5].
        self.assertEqual(groups, [[0], [1], [2, 3], [4, 5]])


class TestInstallFusionPatch(unittest.TestCase):
    """``_install_fusion_patch`` swaps the tensor-fusion helpers in place."""

    def setUp(self):
        from paddle.distributed.fleet.utils import (
            tensor_fusion_helper as helper,
        )

        self.helper = helper
        self._saved_assign = helper.assign_group_by_size
        self._saved_get = helper.get_group_size
        self._saved_comm = helper.FusedCommBuffer._comm_grads
        self._saved_consumers = {}
        for name in _CONSUMERS:
            try:
                module = __import__(name, fromlist=["_"])
            except ImportError:
                continue
            self._saved_consumers[name] = (
                getattr(module, "assign_group_by_size", _MISSING),
                getattr(module, "get_group_size", _MISSING),
            )

    def tearDown(self):
        helper = self.helper
        helper.assign_group_by_size = self._saved_assign
        helper.get_group_size = self._saved_get
        helper.FusedCommBuffer._comm_grads = self._saved_comm
        for name, (assign, get) in self._saved_consumers.items():
            module = __import__(name, fromlist=["_"])
            if assign is _MISSING:
                if hasattr(module, "assign_group_by_size"):
                    delattr(module, "assign_group_by_size")
            else:
                module.assign_group_by_size = assign
            if get is _MISSING:
                if hasattr(module, "get_group_size"):
                    delattr(module, "get_group_size")
            else:
                module.get_group_size = get
        # Verify restoration so a later test cannot inherit our mutations.
        self.assertIs(helper.assign_group_by_size, self._saved_assign)
        self.assertIs(helper.get_group_size, self._saved_get)
        self.assertIs(helper.FusedCommBuffer._comm_grads, self._saved_comm)

    def test_install_replaces_helpers_and_updates_consumers(self):
        helper = self.helper

        acp._install_fusion_patch()

        self.assertIsNot(helper.assign_group_by_size, self._saved_assign)
        self.assertIsNot(helper.get_group_size, self._saved_get)
        self.assertIsNot(helper.FusedCommBuffer._comm_grads, self._saved_comm)
        for name, (assign, get) in self._saved_consumers.items():
            module = __import__(name, fromlist=["_"])
            if assign is not _MISSING:
                self.assertIs(
                    module.assign_group_by_size, helper.assign_group_by_size
                )
            if get is not _MISSING:
                self.assertIs(module.get_group_size, helper.get_group_size)

    def test_installed_assign_and_get_group_size_run(self):
        helper = self.helper
        acp._install_fusion_patch()

        p1 = paddle.create_parameter(shape=[4, 4], dtype="float32")
        p2 = paddle.create_parameter(shape=[4, 4], dtype="float32")

        var_groups = helper.assign_group_by_size([p1, p2])
        self.assertEqual(sum(len(v) for v in var_groups.values()), 2)

        sizes = helper.get_group_size([p1, p2])
        self.assertTrue(all(size >= 0 for size in sizes))

    def test_installed_comm_grads_early_returns_without_sync(self):
        # Only the ``need_reduce_scale_sync() is False`` early-return (118-119)
        # is single-card reachable; the collective body (121-183) needs a real
        # distributed process group and is intentionally not exercised here.
        helper = self.helper
        acp._install_fusion_patch()

        fake_self = SimpleNamespace(need_reduce_scale_sync=lambda: False)
        self.assertIsNone(helper.FusedCommBuffer._comm_grads(fake_self))


class TestInstallShardingShapePatch(unittest.TestCase):
    """``_install_sharding_shape_patch`` wraps ``_create_slice_param`` once."""

    def setUp(self):
        from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer import (
            dygraph_sharding_optimizer as sharding,
        )

        self.cls = sharding.DygraphShardingOptimizerV2
        self._saved = self.cls._create_slice_param

    def tearDown(self):
        self.cls._create_slice_param = self._saved
        self.assertIs(self.cls._create_slice_param, self._saved)

    def test_install_marks_the_wrapper_and_is_idempotent(self):
        self.assertFalse(
            getattr(self._saved, "_fleet_accuracy_compatible", False)
        )

        acp._install_sharding_shape_patch()
        patched = self.cls._create_slice_param
        self.assertTrue(getattr(patched, "_fleet_accuracy_compatible", False))

        # Second call takes the idempotent early-return (216-217).
        acp._install_sharding_shape_patch()
        self.assertIs(self.cls._create_slice_param, patched)


class TestInstallAdamwPatch(unittest.TestCase):
    """``_install_adamw_patch`` wraps ``AdamW._append_optimize_op`` once."""

    def setUp(self):
        from paddle.optimizer import AdamW

        self.AdamW = AdamW
        self._saved = AdamW._append_optimize_op

    def tearDown(self):
        self.AdamW._append_optimize_op = self._saved
        self.assertIs(self.AdamW._append_optimize_op, self._saved)

    def test_install_marks_the_wrapper_and_is_idempotent(self):
        # The nested ``append_optimize_op`` body (235-260) only runs through
        # the static-graph op-append path; in single-card eager mode AdamW
        # never calls ``_append_optimize_op`` (the fused C++ kernel is used
        # instead), so that inner body stays uncovered by design.
        self.assertFalse(
            getattr(self._saved, "_fleet_accuracy_compatible", False)
        )

        acp._install_adamw_patch()
        patched = self.AdamW._append_optimize_op
        self.assertTrue(getattr(patched, "_fleet_accuracy_compatible", False))

        # Second call takes the idempotent early-return (232-233).
        acp._install_adamw_patch()
        self.assertIs(self.AdamW._append_optimize_op, patched)


class TestInstallAccuracyCompatiblePaddlePatches(unittest.TestCase):
    """The install entry point gates on the flag and runs each installer once.

    Also covers ``_accuracy_compatible_enabled`` (line 61).
    """

    def setUp(self):
        self._saved_flag = os.environ.get("FLAGS_use_dsv4_accuracy")
        self._saved_patched = acp._PADDLE_RUNTIME_PATCHED

    def tearDown(self):
        acp._PADDLE_RUNTIME_PATCHED = self._saved_patched
        if self._saved_flag is None:
            os.environ.pop("FLAGS_use_dsv4_accuracy", None)
        else:
            os.environ["FLAGS_use_dsv4_accuracy"] = self._saved_flag

    def test_disabled_path_returns_false(self):
        os.environ.pop("FLAGS_use_dsv4_accuracy", None)
        self.assertFalse(acp.install_accuracy_compatible_paddle_patches())

    def test_enabled_path_runs_each_installer_once(self):
        os.environ["FLAGS_use_dsv4_accuracy"] = "1"
        acp._PADDLE_RUNTIME_PATCHED = False

        # Spy on the installers so globals are not re-mutated here.
        with (
            patch.object(acp, "_install_fusion_patch") as fusion,
            patch.object(acp, "_install_sharding_shape_patch") as sharding,
            patch.object(acp, "_install_adamw_patch") as adamw,
        ):
            self.assertTrue(acp.install_accuracy_compatible_paddle_patches())
            self.assertTrue(acp._PADDLE_RUNTIME_PATCHED)
            # Second call short-circuits (already patched).
            self.assertTrue(acp.install_accuracy_compatible_paddle_patches())

        fusion.assert_called_once()
        sharding.assert_called_once()
        adamw.assert_called_once()


class TestImportTorchMegatronBranch(unittest.TestCase):
    """``_import_torch`` prepends ``_MEGATRON_SITE_PACKAGES`` to ``sys.path``."""

    def setUp(self):
        self._saved_torch = acp._TORCH
        self._saved_pkgs = acp._MEGATRON_SITE_PACKAGES
        self._fake_path = "/tmp/__fake_megatron_site_packages_for_test__"

    def tearDown(self):
        acp._TORCH = self._saved_torch
        acp._MEGATRON_SITE_PACKAGES = self._saved_pkgs
        while self._fake_path in sys.path:
            sys.path.remove(self._fake_path)

    def test_inserts_megatron_path_and_returns_torch(self):
        while self._fake_path in sys.path:
            sys.path.remove(self._fake_path)
        acp._MEGATRON_SITE_PACKAGES = self._fake_path
        acp._TORCH = None

        torch = acp._import_torch()

        self.assertIsNotNone(torch)
        self.assertIn(self._fake_path, sys.path)
        self.assertEqual(sys.path[0], self._fake_path)


if __name__ == "__main__":
    unittest.main()
