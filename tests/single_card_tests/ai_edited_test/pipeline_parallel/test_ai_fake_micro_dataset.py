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


class TestFakeMicroDatasetBasicIteration(unittest.TestCase):
    """Tests for FakeMicroDataset basic iteration."""

    def test_iteration_first_stage(self):
        import paddle

        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            FakeMicroDataset,
        )

        data = (paddle.randn([8, 4]), paddle.randn([8, 1]))
        ds = FakeMicroDataset(
            data,
            is_first_stage=True,
            is_last_stage=True,
            acc_steps=2,
            micro_batch_size=4,
        )
        results = list(ds)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0].shape, [4, 4])
        self.assertEqual(results[1][0].shape, [4, 4])

    def test_iteration_stop(self):
        import paddle

        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            FakeMicroDataset,
        )

        data = (paddle.randn([8, 4]), paddle.randn([8, 1]))
        ds = FakeMicroDataset(
            data,
            is_first_stage=True,
            is_last_stage=True,
            acc_steps=2,
            micro_batch_size=4,
        )
        it = iter(ds)
        next(it)
        next(it)
        with self.assertRaises(StopIteration):
            next(it)

    def test_iteration_only_last_stage(self):
        import paddle

        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            FakeMicroDataset,
        )

        data = (None, paddle.randn([8, 1]))
        ds = FakeMicroDataset(
            data,
            is_first_stage=False,
            is_last_stage=True,
            acc_steps=2,
            micro_batch_size=4,
        )
        results = list(ds)
        self.assertEqual(len(results), 2)
        self.assertIsNone(results[0][0])
        self.assertEqual(results[0][1].shape, [4, 1])


class TestFakeMicroDatasetTupleInputs(unittest.TestCase):
    """Tests for FakeMicroDataset with tuple inputs."""

    def test_tuple_of_tensors(self):
        import paddle

        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            FakeMicroDataset,
        )

        data = (
            (paddle.randn([8, 4]), paddle.randn([8, 4])),
            (paddle.randn([8, 1]),),
        )
        ds = FakeMicroDataset(
            data,
            is_first_stage=True,
            is_last_stage=True,
            acc_steps=2,
            micro_batch_size=4,
        )
        results = list(ds)
        self.assertEqual(len(results), 2)
        self.assertEqual(len(results[0][0]), 2)
        self.assertEqual(results[0][0][0].shape, [4, 4])

    def test_tuple_of_lists(self):
        import paddle

        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            FakeMicroDataset,
        )

        data_list = [paddle.randn([4, 2]) for _ in range(2)]
        label_list = [paddle.randn([4, 1]) for _ in range(2)]
        data = (data_list, label_list)
        ds = FakeMicroDataset(
            data,
            is_first_stage=True,
            is_last_stage=True,
            acc_steps=2,
            micro_batch_size=4,
        )
        results = list(ds)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0].shape, [4, 2])


class TestFakeMicroDatasetDictInputs(unittest.TestCase):
    """Tests for FakeMicroDataset with dict inputs."""

    def test_dict_inputs(self):
        import paddle

        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            FakeMicroDataset,
        )

        data = ({"input_ids": paddle.randn([8, 4])}, paddle.randn([8, 1]))
        ds = FakeMicroDataset(
            data,
            is_first_stage=True,
            is_last_stage=True,
            acc_steps=2,
            micro_batch_size=4,
        )
        results = list(ds)
        self.assertEqual(len(results), 2)
        self.assertIn("input_ids", results[0][0])
        self.assertEqual(results[0][0]["input_ids"].shape, [4, 4])

    def test_dict_list_values(self):
        import paddle

        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            FakeMicroDataset,
        )

        data_list = [paddle.randn([4, 2]) for _ in range(2)]
        label_list = [paddle.randn([4, 1]) for _ in range(2)]
        data = ({"input_ids": data_list}, label_list)
        ds = FakeMicroDataset(
            data,
            is_first_stage=True,
            is_last_stage=True,
            acc_steps=2,
            micro_batch_size=4,
        )
        results = list(ds)
        self.assertEqual(len(results), 2)


class TestFakeMicroDatasetListInputs(unittest.TestCase):
    """Tests for FakeMicroDataset with list inputs."""

    def test_list_inputs(self):
        import paddle

        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            FakeMicroDataset,
        )

        data_list = [paddle.randn([4, 2]) for _ in range(2)]
        label_list = [paddle.randn([4, 1]) for _ in range(2)]
        data = (data_list, label_list)
        ds = FakeMicroDataset(
            data,
            is_first_stage=True,
            is_last_stage=True,
            acc_steps=2,
            micro_batch_size=4,
        )
        results = list(ds)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0].shape, [4, 2])

    def test_list_of_list_inputs(self):
        import paddle

        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            FakeMicroDataset,
        )

        data_list = [
            [paddle.randn([4, 2]), paddle.randn([4, 3])] for _ in range(2)
        ]
        label_list = [paddle.randn([4, 1]) for _ in range(2)]
        data = (data_list, label_list)
        ds = FakeMicroDataset(
            data,
            is_first_stage=True,
            is_last_stage=True,
            acc_steps=2,
            micro_batch_size=4,
        )
        results = list(ds)
        self.assertEqual(len(results), 2)
        self.assertEqual(len(results[0][0]), 2)


class TestFakeMicroDatasetValidation(unittest.TestCase):
    """Tests for FakeMicroDataset validation logic."""

    def test_check_data_valid_success(self):
        import paddle

        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            FakeMicroDataset,
        )

        ds = FakeMicroDataset.__new__(FakeMicroDataset)
        ds._micro_batch_size = 4
        ds._acc_steps = 2
        data = paddle.randn([8, 4])
        # Should not raise
        ds._check_data_valid(data)

    def test_check_data_valid_failure(self):
        import paddle

        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            FakeMicroDataset,
        )

        ds = FakeMicroDataset.__new__(FakeMicroDataset)
        ds._micro_batch_size = 3
        ds._acc_steps = 2
        data = paddle.randn([8, 4])
        with self.assertRaises(AssertionError):
            ds._check_data_valid(data)

    def test_assert_stage_condition(self):
        import paddle

        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            FakeMicroDataset,
        )

        data = (paddle.randn([8, 4]), paddle.randn([8, 1]))
        ds = FakeMicroDataset(
            data,
            is_first_stage=False,
            is_last_stage=False,
            acc_steps=2,
            micro_batch_size=4,
        )
        it = iter(ds)
        with self.assertRaises(AssertionError):
            next(it)


class TestFakeMicroDatasetNoneData(unittest.TestCase):
    """Tests for FakeMicroDataset with None data values."""

    def test_none_in_tuple(self):
        import paddle

        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            FakeMicroDataset,
        )

        data = ((None, paddle.randn([8, 4])), paddle.randn([8, 1]))
        ds = FakeMicroDataset(
            data,
            is_first_stage=True,
            is_last_stage=True,
            acc_steps=2,
            micro_batch_size=4,
        )
        results = list(ds)
        self.assertEqual(len(results), 2)
        self.assertIsNone(results[0][0][0])
        self.assertIsNotNone(results[0][0][1])

    def test_none_single_tensor(self):
        import paddle

        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            FakeMicroDataset,
        )

        data = (None, paddle.randn([8, 1]))
        ds = FakeMicroDataset(
            data,
            is_first_stage=True,
            is_last_stage=True,
            acc_steps=2,
            micro_batch_size=4,
        )
        results = list(ds)
        self.assertEqual(len(results), 2)
        self.assertIsNone(results[0][0])


if __name__ == "__main__":
    unittest.main()
