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
from unittest.mock import MagicMock, patch


class TestGetAction(unittest.TestCase):
    """Tests for get_action function."""

    @patch(
        "paddle.distributed.fleet.meta_parallel.pipeline_parallel.HOOK_ACTION"
    )
    def test_get_action_is_dp(self, mock_action):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            get_action,
        )

        mock_action.ALL_REDUCE = "all_reduce"
        mock_action.REDUCE_SCATTER = "reduce_scatter"
        mock_action.REDUCE = "reduce"
        self.assertEqual(get_action(is_dp=True), "all_reduce")

    @patch(
        "paddle.distributed.fleet.meta_parallel.pipeline_parallel.HOOK_ACTION"
    )
    def test_get_action_shard_split(self, mock_action):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            get_action,
        )

        mock_action.ALL_REDUCE = "all_reduce"
        mock_action.REDUCE_SCATTER = "reduce_scatter"
        mock_action.REDUCE = "reduce"
        self.assertEqual(
            get_action(is_dp=False, shard_split_param=True), "reduce_scatter"
        )

    @patch(
        "paddle.distributed.fleet.meta_parallel.pipeline_parallel.HOOK_ACTION"
    )
    def test_get_action_default(self, mock_action):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            get_action,
        )

        mock_action.ALL_REDUCE = "all_reduce"
        mock_action.REDUCE_SCATTER = "reduce_scatter"
        mock_action.REDUCE = "reduce"
        self.assertEqual(get_action(is_dp=False), "reduce")


class TestNoPipelineParallelExtra(unittest.TestCase):
    """Additional NoPipelineParallel tests."""

    def test_no_pipeline_parallel_is_abstract_base(self):
        from paddle.distributed.fleet.meta_parallel import (
            MetaParallelBase,
            NoPipelineParallel,
        )

        self.assertTrue(issubclass(NoPipelineParallel, MetaParallelBase))

    def test_no_pipeline_is_last_stage_always_true(self):
        from paddle.distributed.fleet.meta_parallel import NoPipelineParallel

        pp = NoPipelineParallel.__new__(NoPipelineParallel)
        self.assertTrue(pp.is_pipeline_last_stage())
        self.assertTrue(pp.is_pipeline_last_stage(ignore_virtual=False))

    def test_no_pipeline_total_loss_initialized_none(self):
        from paddle.distributed.fleet.meta_parallel import NoPipelineParallel

        pp = NoPipelineParallel.__new__(NoPipelineParallel)
        pp.total_loss = None
        self.assertIsNone(pp.total_loss)

    def test_no_pipeline_loss_fn_idx_default(self):
        from paddle.distributed.fleet.meta_parallel import NoPipelineParallel

        pp = NoPipelineParallel.__new__(NoPipelineParallel)
        pp.loss_fn_idx = 0
        self.assertEqual(pp.loss_fn_idx, 0)


class TestFakeMicroDatasetExtra(unittest.TestCase):
    """Additional FakeMicroDataset tests."""

    def test_fake_micro_dataset_single_tensor_input(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            FakeMicroDataset,
        )

        data = (paddle.randn([8, 4]), paddle.randn([8, 1]))
        ds = FakeMicroDataset(
            data,
            is_first_stage=True,
            is_last_stage=True,
            acc_steps=4,
            micro_batch_size=2,
        )
        count = 0
        for item in ds:
            count += 1
            self.assertIsNotNone(item[0])
            self.assertIsNotNone(item[1])
        self.assertEqual(count, 4)

    def test_fake_micro_dataset_none_in_list(self):
        import paddle
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            FakeMicroDataset,
        )

        # Create list of tensors and one None at valid position
        t1 = paddle.randn([4, 2])
        t2 = paddle.randn([4, 2])
        data_list = [t1, t2]
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
        self.assertIsNotNone(results[0][0])


class TestPipelineParallelMicroStepCallbackExtra(unittest.TestCase):
    """Additional PipelineParallelMicroStepCallback tests."""

    def test_register_at_all_four_locations(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            PipelineParallelMicroStepCallback,
            PipelineParallelMicroStepLocations,
        )

        cb = PipelineParallelMicroStepCallback()
        count = 0
        for loc in PipelineParallelMicroStepLocations:
            cb.register_hook(loc, lambda x: None)
            count += 1
        self.assertEqual(count, 4)

    def test_callback_kwargs_forward(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            PipelineParallelMicroStepCallback,
            PipelineParallelMicroStepLocations,
        )

        cb = PipelineParallelMicroStepCallback()
        received = {}

        def hook(**kwargs):
            received.update(kwargs)

        cb.register_hook(PipelineParallelMicroStepLocations.FORWARD_END, hook)
        cb.on_location(
            PipelineParallelMicroStepLocations.FORWARD_END,
            output_tensor="tensor_out",
            step_id=5,
        )
        self.assertEqual(received["output_tensor"], "tensor_out")
        self.assertEqual(received["step_id"], 5)


class TestPipelineDatasetPreprocessor(unittest.TestCase):
    """Tests for PipelineDatasetPreprocessor."""

    def test_pipeline_dataset_preprocessor_call(self):
        from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
            PipelineDatasetPreprocessor,
        )

        func = MagicMock(return_value="data_result")
        preprocessor = PipelineDatasetPreprocessor(func)
        result = preprocessor()
        func.assert_called_once()
        self.assertEqual(result, "data_result")


if __name__ == "__main__":
    unittest.main()
