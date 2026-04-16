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
from dataclasses import fields
from unittest.mock import MagicMock


class TestP2PAsyncHandleInit(unittest.TestCase):
    """Tests for P2PAsyncHandle initialization."""

    def test_init_default_values(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            P2PAsyncHandle,
        )

        fwd_wait_fn = MagicMock()
        fwd_async_fn = MagicMock()
        bwd_wait_fn = MagicMock()
        bwd_async_fn = MagicMock()

        handle = P2PAsyncHandle(
            forward_handle_wait_fn=fwd_wait_fn,
            forward_async_comm_fn=fwd_async_fn,
            backward_handle_wait_fn=bwd_wait_fn,
            backward_async_comm_fn=bwd_async_fn,
        )

        self.assertEqual(handle.forward_handle_wait_fn, fwd_wait_fn)
        self.assertEqual(handle.forward_async_comm_fn, fwd_async_fn)
        self.assertEqual(handle.backward_handle_wait_fn, bwd_wait_fn)
        self.assertEqual(handle.backward_async_comm_fn, bwd_async_fn)
        # Check default None values
        self.assertIsNone(handle.next_forward_virtual_pp_rank)
        self.assertIsNone(handle.input_tensor)
        self.assertIsNone(handle.out_fwd_wait_handles)
        self.assertIsNone(handle.next_backward_virtual_pp_rank)
        self.assertIsNone(handle.output_tensor_grad)
        self.assertIsNone(handle.recv_next)

    def test_init_with_all_fields(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            P2PAsyncHandle,
        )

        fwd_wait_fn = MagicMock()
        fwd_async_fn = MagicMock()
        bwd_wait_fn = MagicMock()
        bwd_async_fn = MagicMock()

        handle = P2PAsyncHandle(
            forward_handle_wait_fn=fwd_wait_fn,
            forward_async_comm_fn=fwd_async_fn,
            backward_handle_wait_fn=bwd_wait_fn,
            backward_async_comm_fn=bwd_async_fn,
        )
        handle.next_forward_virtual_pp_rank = 1
        handle.input_tensor = "fake_input"
        handle.out_fwd_wait_handles = ["handle1", "handle2"]
        handle.next_backward_virtual_pp_rank = 0
        handle.output_tensor_grad = "fake_grad"
        handle.recv_next = True

        self.assertEqual(handle.next_forward_virtual_pp_rank, 1)
        self.assertEqual(handle.input_tensor, "fake_input")
        self.assertEqual(len(handle.out_fwd_wait_handles), 2)
        self.assertEqual(handle.next_backward_virtual_pp_rank, 0)
        self.assertEqual(handle.output_tensor_grad, "fake_grad")
        self.assertTrue(handle.recv_next)


class TestP2PAsyncHandleDataclassFields(unittest.TestCase):
    """Tests for P2PAsyncHandle dataclass field inspection."""

    def test_field_names(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            P2PAsyncHandle,
        )

        field_names = [f.name for f in fields(P2PAsyncHandle)]
        # Only the 4 callable fields are dataclass fields
        self.assertIn("forward_handle_wait_fn", field_names)
        self.assertIn("forward_async_comm_fn", field_names)
        self.assertIn("backward_handle_wait_fn", field_names)
        self.assertIn("backward_async_comm_fn", field_names)
        # The remaining attributes are class-level defaults, not dataclass fields
        self.assertEqual(len(field_names), 4)

    def test_callable_fields(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            P2PAsyncHandle,
        )

        fn = MagicMock()
        handle = P2PAsyncHandle(
            forward_handle_wait_fn=fn,
            forward_async_comm_fn=fn,
            backward_handle_wait_fn=fn,
            backward_async_comm_fn=fn,
        )
        # Verify callable fields can be invoked
        handle.forward_handle_wait_fn()
        handle.forward_async_comm_fn()
        handle.backward_handle_wait_fn()
        handle.backward_async_comm_fn()
        self.assertEqual(fn.call_count, 4)


class TestP2PAsyncHandleAttributeAssignment(unittest.TestCase):
    """Tests for P2PAsyncHandle attribute assignment patterns."""

    def test_assign_forward_attributes(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            P2PAsyncHandle,
        )

        fn = MagicMock()
        handle = P2PAsyncHandle(fn, fn, fn, fn)
        handle.next_forward_virtual_pp_rank = 2
        handle.input_tensor = MagicMock()
        handle.out_fwd_wait_handles = [MagicMock()]

        self.assertEqual(handle.next_forward_virtual_pp_rank, 2)
        self.assertIsNotNone(handle.input_tensor)

    def test_assign_backward_attributes(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            P2PAsyncHandle,
        )

        fn = MagicMock()
        handle = P2PAsyncHandle(fn, fn, fn, fn)
        handle.next_backward_virtual_pp_rank = 1
        handle.output_tensor_grad = MagicMock()
        handle.recv_next = False

        self.assertEqual(handle.next_backward_virtual_pp_rank, 1)
        self.assertFalse(handle.recv_next)


class TestP2PAsyncHandleMultipleInstances(unittest.TestCase):
    """Tests creating multiple P2PAsyncHandle instances."""

    def test_two_independent_handles(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            P2PAsyncHandle,
        )

        fn1 = MagicMock()
        fn2 = MagicMock()
        h1 = P2PAsyncHandle(fn1, fn1, fn1, fn1)
        h2 = P2PAsyncHandle(fn2, fn2, fn2, fn2)

        h1.next_forward_virtual_pp_rank = 0
        h2.next_forward_virtual_pp_rank = 1
        h1.input_tensor = "input_0"
        h2.input_tensor = "input_1"

        self.assertEqual(h1.next_forward_virtual_pp_rank, 0)
        self.assertEqual(h2.next_forward_virtual_pp_rank, 1)
        self.assertEqual(h1.input_tensor, "input_0")
        self.assertEqual(h2.input_tensor, "input_1")

    def test_handle_reassignment(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            P2PAsyncHandle,
        )

        fn = MagicMock()
        handle = P2PAsyncHandle(fn, fn, fn, fn)
        handle.input_tensor = "first"
        self.assertEqual(handle.input_tensor, "first")
        handle.input_tensor = "second"
        self.assertEqual(handle.input_tensor, "second")


class TestP2PAsyncHandleCallableTypes(unittest.TestCase):
    """Tests for P2PAsyncHandle callable field types."""

    def test_lambda_callables(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            P2PAsyncHandle,
        )

        handle = P2PAsyncHandle(
            forward_handle_wait_fn=lambda: "fwd_wait",
            forward_async_comm_fn=lambda: "fwd_async",
            backward_handle_wait_fn=lambda: "bwd_wait",
            backward_async_comm_fn=lambda: "bwd_async",
        )

        self.assertEqual(handle.forward_handle_wait_fn(), "fwd_wait")
        self.assertEqual(handle.forward_async_comm_fn(), "fwd_async")
        self.assertEqual(handle.backward_handle_wait_fn(), "bwd_wait")
        self.assertEqual(handle.backward_async_comm_fn(), "bwd_async")


if __name__ == "__main__":
    unittest.main()
