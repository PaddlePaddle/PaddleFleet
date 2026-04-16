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

import queue
import unittest
from unittest.mock import MagicMock, patch


class TestOffloadQueueInit(unittest.TestCase):
    """Tests for OffloadQueue initialization."""

    def test_init_offload_false(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            OffloadQueue,
        )

        q = OffloadQueue(offload=False)
        self.assertFalse(q.offload)

    def test_init_offload_true(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            OffloadQueue,
        )

        q = OffloadQueue(offload=True)
        self.assertTrue(q.offload)

    def test_init_with_maxsize(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            OffloadQueue,
        )

        q = OffloadQueue(offload=False, maxsize=5)
        self.assertFalse(q.offload)
        self.assertEqual(q.maxsize, 5)

    def test_is_subclass_of_queue(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            OffloadQueue,
        )

        self.assertTrue(issubclass(OffloadQueue, queue.Queue))


class TestOffloadQueuePutGetNoOffload(unittest.TestCase):
    """Tests for OffloadQueue put/get without offload."""

    def test_put_get_value(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            OffloadQueue,
        )

        q = OffloadQueue(offload=False)
        q.put("test_value")
        self.assertEqual(q.get(), "test_value")

    def test_put_get_int(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            OffloadQueue,
        )

        q = OffloadQueue(offload=False)
        q.put(42)
        self.assertEqual(q.get(), 42)

    def test_put_get_none(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            OffloadQueue,
        )

        q = OffloadQueue(offload=False)
        q.put(None)
        self.assertIsNone(q.get())

    def test_multiple_put_get(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            OffloadQueue,
        )

        q = OffloadQueue(offload=False)
        for i in range(10):
            q.put(i)
        for i in range(10):
            self.assertEqual(q.get(), i)

    def test_qsize(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            OffloadQueue,
        )

        q = OffloadQueue(offload=False)
        q.put(1)
        q.put(2)
        self.assertEqual(q.qsize(), 2)
        q.get()
        self.assertEqual(q.qsize(), 1)

    def test_empty(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            OffloadQueue,
        )

        q = OffloadQueue(offload=False)
        self.assertTrue(q.empty())
        q.put(1)
        self.assertFalse(q.empty())


class TestOffloadQueuePutGetNonTensor(unittest.TestCase):
    """Tests for OffloadQueue with non-tensor values when offload=True."""

    def test_put_get_string_with_offload(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            OffloadQueue,
        )

        q = OffloadQueue(offload=True)
        q.put("string_value")
        self.assertEqual(q.get(), "string_value")

    def test_put_get_int_with_offload(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            OffloadQueue,
        )

        q = OffloadQueue(offload=True)
        q.put(123)
        self.assertEqual(q.get(), 123)

    def test_put_get_dict_with_offload(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            OffloadQueue,
        )

        q = OffloadQueue(offload=True)
        data = {"key": "value"}
        q.put(data)
        self.assertEqual(q.get(), data)


class TestOffloadQueuePutGetTensorOffload(unittest.TestCase):
    """Tests for OffloadQueue with tensor offload enabled."""

    def test_put_get_tensor_offload(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            OffloadQueue,
        )

        q = OffloadQueue(offload=True)
        tensor = paddle.randn([4, 4])
        with patch.object(
            tensor, "pin_memory", return_value=MagicMock()
        ) as mock_pin:
            mock_pin.return_value._share_buffer_to = MagicMock()
            q.put(tensor)
            mock_pin.assert_called_once()

    def test_put_get_tuple_of_tensors_offload(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            OffloadQueue,
        )

        q = OffloadQueue(offload=True)
        t1 = paddle.randn([2, 3])
        t2 = paddle.randn([2, 3])
        with (
            patch.object(
                t1, "pin_memory", return_value=MagicMock()
            ) as mock_pin1,
            patch.object(
                t2, "pin_memory", return_value=MagicMock()
            ) as mock_pin2,
        ):
            mock_pin1.return_value._share_buffer_to = MagicMock()
            mock_pin2.return_value._share_buffer_to = MagicMock()
            q.put((t1, t2))

    def test_put_non_tensor_tuple_offload(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            OffloadQueue,
        )

        q = OffloadQueue(offload=True)
        q.put((1, 2, 3))
        result = q.get()
        self.assertEqual(result, (1, 2, 3))


class TestOffloadQueueNoOffloadTensor(unittest.TestCase):
    """Tests for OffloadQueue with tensor when offload=False."""

    def test_put_get_tensor_no_offload(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            OffloadQueue,
        )

        q = OffloadQueue(offload=False)
        tensor = paddle.randn([4, 4])
        q.put(tensor)
        result = q.get()
        self.assertTrue(paddle.allclose(tensor, result))

    def test_put_get_tuple_tensor_no_offload(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            OffloadQueue,
        )

        q = OffloadQueue(offload=False)
        t1 = paddle.randn([2, 3])
        t2 = paddle.randn([2, 3])
        q.put((t1, t2))
        result = q.get()
        self.assertEqual(len(result), 2)


class TestVPPFhenBInBalancedMemorySchedulerName(unittest.TestCase):
    """Tests for VPPFhenBInBalancedMemory scheduler name."""

    def test_scheduler_name(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            VPPFhenBInBalancedMemory,
        )

        pp = VPPFhenBInBalancedMemory.__new__(VPPFhenBInBalancedMemory)
        self.assertEqual(pp._get_scheduler_name(), "VPPFhenBInBalancedMemory")


if __name__ == "__main__":
    unittest.main()
