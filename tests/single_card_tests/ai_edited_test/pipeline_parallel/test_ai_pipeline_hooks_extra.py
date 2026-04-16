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
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

import unittest
from collections import defaultdict
from unittest.mock import MagicMock


class TestPipelineHookInit(unittest.TestCase):
    """Tests for PipelineHook initialization."""

    def test_init_default(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_hooks import (
            PipelineHook,
        )

        hook = PipelineHook()
        self.assertIsInstance(hook.hooks, dict)
        self.assertIsInstance(hook.hooks, defaultdict)
        self.assertEqual(hook._current_id, 0)
        self.assertEqual(hook._hooks_capacity, 0)

    def test_init_hooks_empty(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_hooks import (
            PipelineHook,
        )

        hook = PipelineHook()
        for key in hook.hooks:
            self.assertEqual(len(hook.hooks[key]), 0)


class TestPipelineHookResetCurrentId(unittest.TestCase):
    """Tests for PipelineHook.reset_current_id."""

    def test_reset_after_running(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_hooks import (
            PipelineHook,
        )

        hook = PipelineHook()
        hook._hooks_capacity = 2
        hook._current_id = 2
        hook.reset_current_id()
        self.assertEqual(hook._current_id, 0)

    def test_reset_multiple_times(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_hooks import (
            PipelineHook,
        )

        hook = PipelineHook()
        hook._current_id = 5
        hook.reset_current_id()
        hook.reset_current_id()
        self.assertEqual(hook._current_id, 0)


class TestPipelineHookSetHooksCapacity(unittest.TestCase):
    """Tests for PipelineHook.set_hooks_capacity."""

    def test_set_capacity(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_hooks import (
            PipelineHook,
        )

        hook = PipelineHook()
        hook.set_hooks_capacity(10)
        self.assertEqual(hook.hooks_capacity, 10)

    def test_set_capacity_multiple(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_hooks import (
            PipelineHook,
        )

        hook = PipelineHook()
        hook.set_hooks_capacity(5)
        self.assertEqual(hook.hooks_capacity, 5)
        hook.set_hooks_capacity(20)
        self.assertEqual(hook.hooks_capacity, 20)


class TestPipelineHookRegisterHook(unittest.TestCase):
    """Tests for PipelineHook.register_hook."""

    def test_register_single(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_hooks import (
            PipelineHook,
        )

        hook = PipelineHook()
        fn = MagicMock()
        hook.set_hooks_capacity(5)
        hook.register_hook(2, fn)
        self.assertEqual(len(hook.hooks[2]), 1)
        self.assertEqual(hook.hooks[2][0], fn)

    def test_register_multiple_same_id(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_hooks import (
            PipelineHook,
        )

        hook = PipelineHook()
        fn1 = MagicMock()
        fn2 = MagicMock()
        hook.set_hooks_capacity(5)
        hook.register_hook(1, fn1)
        hook.register_hook(1, fn2)
        self.assertEqual(len(hook.hooks[1]), 2)

    def test_register_out_of_range(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_hooks import (
            PipelineHook,
        )

        hook = PipelineHook()
        hook.set_hooks_capacity(3)
        with self.assertRaises(AssertionError):
            hook.register_hook(5, MagicMock())

    def test_register_different_ids(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_hooks import (
            PipelineHook,
        )

        hook = PipelineHook()
        fn1 = MagicMock()
        fn2 = MagicMock()
        hook.set_hooks_capacity(10)
        hook.register_hook(0, fn1)
        hook.register_hook(9, fn2)
        self.assertEqual(len(hook.hooks[0]), 1)
        self.assertEqual(len(hook.hooks[9]), 1)


class TestPipelineHookRunHook(unittest.TestCase):
    """Tests for PipelineHook.run_hook."""

    def test_run_hook_calls_function(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_hooks import (
            PipelineHook,
        )

        hook = PipelineHook()
        fn = MagicMock()
        hook.set_hooks_capacity(3)
        hook.register_hook(0, fn)
        hook.run_hook()
        fn.assert_called_once_with(0)

    def test_run_hook_increments_id(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_hooks import (
            PipelineHook,
        )

        hook = PipelineHook()
        hook.set_hooks_capacity(3)
        hook.register_hook(0, lambda x: None)
        hook.run_hook()
        self.assertEqual(hook.current_id, 1)

    def test_run_hook_calls_all_at_id(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_hooks import (
            PipelineHook,
        )

        hook = PipelineHook()
        fn1 = MagicMock()
        fn2 = MagicMock()
        hook.set_hooks_capacity(3)
        hook.register_hook(1, fn1)
        hook.register_hook(1, fn2)
        hook.register_hook(0, lambda x: None)
        hook.run_hook()
        hook.run_hook()
        fn1.assert_called_once_with(1)
        fn2.assert_called_once_with(1)

    def test_run_hook_out_of_range(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_hooks import (
            PipelineHook,
        )

        hook = PipelineHook()
        hook.set_hooks_capacity(1)
        hook.register_hook(0, lambda x: None)
        hook.run_hook()
        with self.assertRaises(AssertionError):
            hook.run_hook()


class TestPipelineHookCurrentId(unittest.TestCase):
    """Tests for PipelineHook.current_id property."""

    def test_current_id_initial(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_hooks import (
            PipelineHook,
        )

        hook = PipelineHook()
        self.assertEqual(hook.current_id, 0)

    def test_current_id_after_register(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_hooks import (
            PipelineHook,
        )

        hook = PipelineHook()
        hook.set_hooks_capacity(5)
        hook.register_hook(0, lambda x: None)
        self.assertEqual(hook.current_id, 0)


class TestPipelineHookHooksCapacity(unittest.TestCase):
    """Tests for PipelineHook.hooks_capacity property."""

    def test_hooks_capacity_default(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_hooks import (
            PipelineHook,
        )

        hook = PipelineHook()
        self.assertEqual(hook.hooks_capacity, 0)

    def test_hooks_capacity_after_set(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_hooks import (
            PipelineHook,
        )

        hook = PipelineHook()
        hook.set_hooks_capacity(7)
        self.assertEqual(hook.hooks_capacity, 7)


class TestPipelineHookFullCycle(unittest.TestCase):
    """Full cycle tests for PipelineHook."""

    def test_full_cycle_register_and_run(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_hooks import (
            PipelineHook,
        )

        hook = PipelineHook()
        results = []
        hook.set_hooks_capacity(4)
        for i in range(4):
            hook.register_hook(i, lambda idx, r=results, val=i: r.append(val))
        for i in range(4):
            hook.run_hook()
        self.assertEqual(results, [0, 1, 2, 3])
        self.assertEqual(hook.current_id, 4)

    def test_full_cycle_reset_and_rerun(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_hooks import (
            PipelineHook,
        )

        hook = PipelineHook()
        counter = {"val": 0}
        hook.set_hooks_capacity(2)
        hook.register_hook(
            0, lambda x: counter.update({"val": counter["val"] + 1})
        )
        hook.register_hook(
            1, lambda x: counter.update({"val": counter["val"] + 10})
        )
        hook.run_hook()
        hook.run_hook()
        self.assertEqual(counter["val"], 11)
        hook.reset_current_id()
        hook.run_hook()
        self.assertEqual(counter["val"], 12)

    def test_hooks_defaultdict_behavior(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_hooks import (
            PipelineHook,
        )

        hook = PipelineHook()
        # Accessing a non-existent key should return empty list
        self.assertEqual(hook.hooks[999], [])


if __name__ == "__main__":
    unittest.main()
