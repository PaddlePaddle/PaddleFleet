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
from unittest.mock import MagicMock


class TestPipelineParallelMicroStepCallbackInit(unittest.TestCase):
    """Tests for PipelineParallelMicroStepCallback initialization."""

    def test_init_default(self):
        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            PipelineParallelMicroStepCallback,
            PipelineParallelMicroStepLocations,
        )

        cb = PipelineParallelMicroStepCallback()
        self.assertIn(
            PipelineParallelMicroStepLocations.FORWARD_BEGIN, cb.hooks
        )
        self.assertIn(PipelineParallelMicroStepLocations.FORWARD_END, cb.hooks)
        self.assertIn(
            PipelineParallelMicroStepLocations.BACKWARD_BEGIN, cb.hooks
        )
        self.assertIn(PipelineParallelMicroStepLocations.BACKWARD_END, cb.hooks)
        for loc in PipelineParallelMicroStepLocations:
            self.assertEqual(len(cb.hooks[loc]), 0)

    def test_global_instance(self):
        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            PipelineParallelMicroStepCallback,
            pipeline_parallel_callbacks_,
        )

        self.assertIsInstance(
            pipeline_parallel_callbacks_, PipelineParallelMicroStepCallback
        )


class TestPipelineParallelMicroStepCallbackRegister(unittest.TestCase):
    """Tests for PipelineParallelMicroStepCallback.register_hook."""

    def test_register_single_hook(self):
        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            PipelineParallelMicroStepCallback,
            PipelineParallelMicroStepLocations,
        )

        cb = PipelineParallelMicroStepCallback()
        hook = MagicMock()
        cb.register_hook(PipelineParallelMicroStepLocations.FORWARD_BEGIN, hook)
        self.assertEqual(
            len(cb.hooks[PipelineParallelMicroStepLocations.FORWARD_BEGIN]), 1
        )

    def test_register_multiple_hooks(self):
        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            PipelineParallelMicroStepCallback,
            PipelineParallelMicroStepLocations,
        )

        cb = PipelineParallelMicroStepCallback()
        hook1 = MagicMock()
        hook2 = MagicMock()
        cb.register_hook(PipelineParallelMicroStepLocations.FORWARD_END, hook1)
        cb.register_hook(PipelineParallelMicroStepLocations.FORWARD_END, hook2)
        self.assertEqual(
            len(cb.hooks[PipelineParallelMicroStepLocations.FORWARD_END]), 2
        )

    def test_register_invalid_location(self):
        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            PipelineParallelMicroStepCallback,
        )

        cb = PipelineParallelMicroStepCallback()
        with self.assertRaises(AssertionError):
            cb.register_hook("invalid_location", MagicMock())


class TestPipelineParallelMicroStepCallbackOnLocation(unittest.TestCase):
    """Tests for PipelineParallelMicroStepCallback.on_location."""

    def test_on_location_forward_begin(self):
        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            PipelineParallelMicroStepCallback,
            PipelineParallelMicroStepLocations,
        )

        cb = PipelineParallelMicroStepCallback()
        hook = MagicMock()
        cb.register_hook(PipelineParallelMicroStepLocations.FORWARD_BEGIN, hook)
        cb.on_location(
            PipelineParallelMicroStepLocations.FORWARD_BEGIN,
            input_tensor="fake_tensor",
        )
        hook.assert_called_once_with(input_tensor="fake_tensor")

    def test_on_location_backward_end(self):
        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            PipelineParallelMicroStepCallback,
            PipelineParallelMicroStepLocations,
        )

        cb = PipelineParallelMicroStepCallback()
        hook = MagicMock()
        cb.register_hook(PipelineParallelMicroStepLocations.BACKWARD_END, hook)
        cb.on_location(
            PipelineParallelMicroStepLocations.BACKWARD_END,
            step_id=0,
        )
        hook.assert_called_once_with(step_id=0)

    def test_on_location_no_hooks(self):
        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            PipelineParallelMicroStepCallback,
            PipelineParallelMicroStepLocations,
        )

        cb = PipelineParallelMicroStepCallback()
        # Should not raise even if no hooks registered
        cb.on_location(PipelineParallelMicroStepLocations.FORWARD_BEGIN)

    def test_on_location_invalid_location(self):
        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            PipelineParallelMicroStepCallback,
        )

        cb = PipelineParallelMicroStepCallback()
        with self.assertRaises(AssertionError):
            cb.on_location("invalid_location")

    def test_on_location_multiple_hooks(self):
        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            PipelineParallelMicroStepCallback,
            PipelineParallelMicroStepLocations,
        )

        cb = PipelineParallelMicroStepCallback()
        hook1 = MagicMock()
        hook2 = MagicMock()
        cb.register_hook(
            PipelineParallelMicroStepLocations.BACKWARD_BEGIN, hook1
        )
        cb.register_hook(
            PipelineParallelMicroStepLocations.BACKWARD_BEGIN, hook2
        )
        cb.on_location(
            PipelineParallelMicroStepLocations.BACKWARD_BEGIN,
            input_tensor_grad="grad",
        )
        hook1.assert_called_once_with(input_tensor_grad="grad")
        hook2.assert_called_once_with(input_tensor_grad="grad")


class TestPipelineParallelMicroStepLocations(unittest.TestCase):
    """Tests for PipelineParallelMicroStepLocations enum values."""

    def test_enum_members(self):
        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            PipelineParallelMicroStepLocations,
        )

        members = list(PipelineParallelMicroStepLocations)
        self.assertEqual(len(members), 4)
        self.assertEqual(
            PipelineParallelMicroStepLocations.FORWARD_BEGIN.value,
            "forward_begin",
        )
        self.assertEqual(
            PipelineParallelMicroStepLocations.FORWARD_END.value,
            "forward_end",
        )
        self.assertEqual(
            PipelineParallelMicroStepLocations.BACKWARD_BEGIN.value,
            "backward_begin",
        )
        self.assertEqual(
            PipelineParallelMicroStepLocations.BACKWARD_END.value,
            "backward_end",
        )


class TestCallbackAllLocations(unittest.TestCase):
    """Integration tests registering hooks at all locations."""

    def test_all_locations_hooks(self):
        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            PipelineParallelMicroStepCallback,
            PipelineParallelMicroStepLocations,
        )

        cb = PipelineParallelMicroStepCallback()
        hooks = {}
        for loc in PipelineParallelMicroStepLocations:
            h = MagicMock()
            hooks[loc] = h
            cb.register_hook(loc, h)

        for loc in PipelineParallelMicroStepLocations:
            cb.on_location(loc, test_key="value")
            hooks[loc].assert_called_once_with(test_key="value")

    def test_callback_with_various_kwargs(self):
        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            PipelineParallelMicroStepCallback,
            PipelineParallelMicroStepLocations,
        )

        cb = PipelineParallelMicroStepCallback()
        hook = MagicMock()

        cb.register_hook(PipelineParallelMicroStepLocations.FORWARD_BEGIN, hook)
        cb.on_location(
            PipelineParallelMicroStepLocations.FORWARD_BEGIN,
            input_tensor="t",
            output_tensor="o",
            step_id=3,
        )
        hook.assert_called_once_with(
            input_tensor="t", output_tensor="o", step_id=3
        )


if __name__ == "__main__":
    unittest.main()
