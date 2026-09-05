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
"""Unit tests for the NUMA binding helper of activation offloading.

Nothing here changes the real CPU affinity of the test process: ``bind()`` is
only ever reached with ``os.sched_setaffinity`` patched out. Binding for real
would restrict every later test in the same process to one node's cores.
"""

from __future__ import annotations

import unittest
from unittest import mock

from paddlefleet.activation_offload import numa_bind as nb

_MOD = "paddlefleet.activation_offload.numa_bind"


class TestCpusOfNumaNode(unittest.TestCase):
    def test_parses_ranges_and_singletons(self):
        with mock.patch(
            f"{_MOD}.open", mock.mock_open(read_data="0-3,8,10-11\n")
        ):
            self.assertEqual(nb.cpus_of_numa_node(0), [0, 1, 2, 3, 8, 10, 11])

    def test_parses_single_cpu(self):
        with mock.patch(f"{_MOD}.open", mock.mock_open(read_data="7")):
            self.assertEqual(nb.cpus_of_numa_node(3), [7])

    def test_tolerates_trailing_separator(self):
        with mock.patch(f"{_MOD}.open", mock.mock_open(read_data="0-1,,4")):
            self.assertEqual(nb.cpus_of_numa_node(0), [0, 1, 4])

    def test_missing_node_returns_empty(self):
        # A node with no cpulist in sysfs must not raise: the caller degrades.
        with mock.patch(f"{_MOD}.open", side_effect=OSError):
            self.assertEqual(nb.cpus_of_numa_node(99), [])


class TestGpusPerNuma(unittest.TestCase):
    def test_default(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(nb._gpus_per_numa(), 2)

    def test_override(self):
        with mock.patch.dict("os.environ", {nb.GPUS_PER_NUMA_ENV: "4"}):
            self.assertEqual(nb._gpus_per_numa(), 4)

    def test_zero_is_clamped_to_one(self):
        with mock.patch.dict("os.environ", {nb.GPUS_PER_NUMA_ENV: "0"}):
            self.assertEqual(nb._gpus_per_numa(), 1)

    def test_garbage_falls_back_to_default(self):
        with mock.patch.dict("os.environ", {nb.GPUS_PER_NUMA_ENV: "abc"}):
            self.assertEqual(nb._gpus_per_numa(), 2)


class TestCurrentGpu(unittest.TestCase):
    def test_selected_gpus_wins_over_visible_devices(self):
        env = {"FLAGS_selected_gpus": "3", "CUDA_VISIBLE_DEVICES": "7"}
        with mock.patch.dict("os.environ", env):
            self.assertEqual(nb.current_gpu(), 3)

    def test_falls_back_to_first_visible_device(self):
        with mock.patch.dict(
            "os.environ", {"CUDA_VISIBLE_DEVICES": "5,6,7"}, clear=True
        ):
            self.assertEqual(nb.current_gpu(), 5)

    def test_non_numeric_is_ignored(self):
        # A UUID-style CUDA_VISIBLE_DEVICES carries no ordinal to bind against.
        env = {"CUDA_VISIBLE_DEVICES": "GPU-abcdef", "FLAGS_selected_gpus": ""}
        with mock.patch.dict("os.environ", env):
            self.assertIsNone(nb.current_gpu())

    def test_unset_returns_none(self):
        env = {"FLAGS_selected_gpus": "", "CUDA_VISIBLE_DEVICES": ""}
        with mock.patch.dict("os.environ", env):
            self.assertIsNone(nb.current_gpu())


class TestBind(unittest.TestCase):
    """``os.sched_setaffinity`` is patched in every case that reaches it."""

    def test_unknown_gpu_is_a_quiet_no_op(self):
        # A failed topology probe must not stop training from starting.
        with (
            mock.patch(f"{_MOD}.current_gpu", return_value=None),
            mock.patch("os.sched_setaffinity") as setaff,
        ):
            self.assertFalse(nb.bind())
        setaff.assert_not_called()

    def test_node_without_cpulist_is_a_quiet_no_op(self):
        with (
            mock.patch(f"{_MOD}.cpus_of_numa_node", return_value=[]),
            mock.patch("os.sched_setaffinity") as setaff,
        ):
            self.assertFalse(nb.bind(gpu=0))
        setaff.assert_not_called()

    def test_binds_to_the_node_local_to_the_gpu(self):
        cpus = [0, 1, 2, 3]
        with (
            mock.patch.dict("os.environ", {nb.GPUS_PER_NUMA_ENV: "2"}),
            mock.patch(
                f"{_MOD}.cpus_of_numa_node", return_value=cpus
            ) as cpus_of,
            mock.patch("os.sched_setaffinity") as setaff,
        ):
            self.assertTrue(nb.bind(gpu=3))
        # gpu 3 with two GPUs per node lives on node 1
        cpus_of.assert_called_once_with(1)
        setaff.assert_called_once_with(0, set(cpus))

    def test_permission_error_is_reported_not_raised(self):
        # Containers may not allow sched_setaffinity; that must not be fatal.
        with (
            mock.patch(f"{_MOD}.cpus_of_numa_node", return_value=[0, 1]),
            mock.patch("os.sched_setaffinity", side_effect=OSError("denied")),
        ):
            self.assertFalse(nb.bind(gpu=0))

    def test_real_affinity_is_untouched_by_this_test_file(self):
        import os

        before = os.sched_getaffinity(0)
        with mock.patch("os.sched_setaffinity") as setaff:
            nb.bind(gpu=0)
        setaff.assert_called()  # the success path really was exercised
        self.assertEqual(os.sched_getaffinity(0), before)


if __name__ == "__main__":
    unittest.main()
