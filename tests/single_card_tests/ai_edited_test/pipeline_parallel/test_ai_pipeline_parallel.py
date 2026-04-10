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

    def test_get_action_dp(self):
        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            HOOK_ACTION,
            get_action,
        )

        result = get_action(is_dp=True)
        self.assertEqual(result, HOOK_ACTION.ALL_REDUCE)

    def test_get_action_shard_split(self):
        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            HOOK_ACTION,
            get_action,
        )

        result = get_action(is_dp=False, shard_split_param=True)
        self.assertEqual(result, HOOK_ACTION.REDUCE_SCATTER)

    def test_get_action_reduce(self):
        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            HOOK_ACTION,
            get_action,
        )

        result = get_action(is_dp=False, shard_split_param=False)
        self.assertEqual(result, HOOK_ACTION.REDUCE)


class TestPipelineDatasetPreprocessor(unittest.TestCase):
    """Tests for PipelineDatasetPreprocessor."""

    def test_call(self):
        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            PipelineDatasetPreprocessor,
        )

        fn = MagicMock(return_value="data")
        preprocessor = PipelineDatasetPreprocessor(fn)
        result = preprocessor()
        self.assertEqual(result, "data")


class TestPipelineParallelMicroStepLocations(unittest.TestCase):
    """Tests for PipelineParallelMicroStepLocations enum."""

    def test_enum_values(self):
        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            PipelineParallelMicroStepLocations,
        )

        self.assertEqual(
            PipelineParallelMicroStepLocations.FORWARD_BEGIN.value,
            "forward_begin",
        )
        self.assertEqual(
            PipelineParallelMicroStepLocations.FORWARD_END.value, "forward_end"
        )
        self.assertEqual(
            PipelineParallelMicroStepLocations.BACKWARD_BEGIN.value,
            "backward_begin",
        )
        self.assertEqual(
            PipelineParallelMicroStepLocations.BACKWARD_END.value,
            "backward_end",
        )


class TestPipelineParallelMicroStepCallback(unittest.TestCase):
    """Tests for PipelineParallelMicroStepCallback."""

    def test_init(self):
        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            PipelineParallelMicroStepCallback,
            PipelineParallelMicroStepLocations,
        )

        callback = PipelineParallelMicroStepCallback()
        self.assertEqual(
            len(
                callback.hooks[PipelineParallelMicroStepLocations.FORWARD_BEGIN]
            ),
            0,
        )

    def test_register_hook_valid(self):
        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            PipelineParallelMicroStepCallback,
            PipelineParallelMicroStepLocations,
        )

        callback = PipelineParallelMicroStepCallback()
        fn = MagicMock()
        callback.register_hook(
            PipelineParallelMicroStepLocations.FORWARD_BEGIN, fn
        )
        self.assertEqual(
            len(
                callback.hooks[PipelineParallelMicroStepLocations.FORWARD_BEGIN]
            ),
            1,
        )

    def test_register_hook_invalid(self):
        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            PipelineParallelMicroStepCallback,
        )

        callback = PipelineParallelMicroStepCallback()
        fn = MagicMock()
        with self.assertRaises(AssertionError):
            callback.register_hook("invalid_location", fn)

    def test_on_location(self):
        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            PipelineParallelMicroStepCallback,
            PipelineParallelMicroStepLocations,
        )

        callback = PipelineParallelMicroStepCallback()
        fn = MagicMock()
        callback.register_hook(
            PipelineParallelMicroStepLocations.FORWARD_END, fn
        )
        callback.on_location(
            PipelineParallelMicroStepLocations.FORWARD_END, step_id=0
        )
        fn.assert_called_once_with(step_id=0)

    def test_on_location_invalid(self):
        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            PipelineParallelMicroStepCallback,
        )

        callback = PipelineParallelMicroStepCallback()
        with self.assertRaises(AssertionError):
            callback.on_location("invalid")


class TestFakeMicroDataset(unittest.TestCase):
    """Tests for FakeMicroDataset."""

    def _create_mock_data(self):
        import paddle

        data = paddle.randn([4, 3])
        label = paddle.randn([4, 1])
        return (data, label)

    def test_iter_first_and_last_stage(self):
        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            FakeMicroDataset,
        )

        data, label = self._create_mock_data()
        dataset = FakeMicroDataset(
            data=(data, label),
            is_first_stage=True,
            is_last_stage=True,
            acc_steps=2,
            micro_batch_size=2,
        )
        results = list(dataset)
        self.assertEqual(len(results), 2)

    def test_iter_first_stage_only(self):
        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            FakeMicroDataset,
        )

        data, label = self._create_mock_data()
        dataset = FakeMicroDataset(
            data=(data, label),
            is_first_stage=True,
            is_last_stage=False,
            acc_steps=2,
            micro_batch_size=2,
        )
        results = list(dataset)
        self.assertEqual(len(results), 2)

    def test_iter_last_stage_only(self):
        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            FakeMicroDataset,
        )

        data, label = self._create_mock_data()
        dataset = FakeMicroDataset(
            data=(data, label),
            is_first_stage=False,
            is_last_stage=True,
            acc_steps=2,
            micro_batch_size=2,
        )
        results = list(dataset)
        self.assertEqual(len(results), 2)

    def test_load_micro_batch_tuple_data(self):
        import paddle

        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            FakeMicroDataset,
        )

        data = paddle.randn([4, 3])
        label = paddle.randn([4, 1])
        dataset = FakeMicroDataset(
            data=((data,), label),
            is_first_stage=True,
            is_last_stage=True,
            acc_steps=2,
            micro_batch_size=2,
        )
        result = dataset._load_micro_batch(0)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_load_micro_batch_list_data(self):
        import paddle

        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            FakeMicroDataset,
        )

        data_list = [paddle.randn([2, 3]), paddle.randn([2, 3])]
        label_list = [paddle.randn([2, 1]), paddle.randn([2, 1])]
        dataset = FakeMicroDataset(
            data=(data_list, label_list),
            is_first_stage=True,
            is_last_stage=True,
            acc_steps=2,
            micro_batch_size=2,
        )
        result = dataset._load_micro_batch(0)
        self.assertIsNotNone(result)

    def test_load_micro_batch_dict_data(self):
        import paddle

        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            FakeMicroDataset,
        )

        data = paddle.randn([4, 3])
        label = paddle.randn([4, 1])
        dataset = FakeMicroDataset(
            data=({"input": data}, label),
            is_first_stage=True,
            is_last_stage=True,
            acc_steps=2,
            micro_batch_size=2,
        )
        result = dataset._load_micro_batch(0)
        self.assertIsNotNone(result)

    def test_load_micro_batch_none_data(self):
        import paddle

        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            FakeMicroDataset,
        )

        data = paddle.randn([4, 3])
        label = None
        dataset = FakeMicroDataset(
            data=(data, label),
            is_first_stage=True,
            is_last_stage=False,
            acc_steps=2,
            micro_batch_size=2,
        )
        result = dataset._load_micro_batch(0)
        self.assertIsNotNone(result)

    def test_check_data_valid(self):
        import paddle

        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            FakeMicroDataset,
        )

        data = paddle.randn([4, 3])
        label = paddle.randn([4, 1])
        dataset = FakeMicroDataset(
            data=(data, label),
            is_first_stage=True,
            is_last_stage=True,
            acc_steps=2,
            micro_batch_size=2,
        )
        dataset._check_data_valid(data)

    def test_check_data_valid_invalid(self):
        import paddle

        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            FakeMicroDataset,
        )

        data = paddle.randn([5, 3])  # 5 not divisible by 2*2=4
        label = paddle.randn([5, 1])
        dataset = FakeMicroDataset(
            data=(data, label),
            is_first_stage=True,
            is_last_stage=True,
            acc_steps=2,
            micro_batch_size=2,
        )
        with self.assertRaises(AssertionError):
            dataset._check_data_valid(data)


class TestNoPipelineParallel(unittest.TestCase):
    """Tests for NoPipelineParallel class."""

    def test_is_pipeline_last_stage(self):
        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            NoPipelineParallel,
        )

        mock_layers = MagicMock()
        mock_strategy = MagicMock()
        npp = NoPipelineParallel.__new__(NoPipelineParallel)
        npp._layers = mock_layers
        npp._strategy = mock_strategy
        npp._hcg = None
        self.assertTrue(npp.is_pipeline_last_stage())

    def test_check_micro_batch_data_valid_tuple(self):
        import paddle

        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            NoPipelineParallel,
        )

        npp = NoPipelineParallel.__new__(NoPipelineParallel)
        t = paddle.randn([2, 3])
        npp._check_micro_batch_data_valid((t, t))

    def test_check_micro_batch_data_valid_dict(self):
        import paddle

        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            NoPipelineParallel,
        )

        npp = NoPipelineParallel.__new__(NoPipelineParallel)
        t = paddle.randn([2, 3])
        npp._check_micro_batch_data_valid({"a": t, "b": t})

    def test_check_micro_batch_data_valid_none(self):
        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            NoPipelineParallel,
        )

        npp = NoPipelineParallel.__new__(NoPipelineParallel)
        npp._check_micro_batch_data_valid(None)

    def test_check_micro_batch_data_valid_invalid(self):
        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            NoPipelineParallel,
        )

        npp = NoPipelineParallel.__new__(NoPipelineParallel)
        with self.assertRaises(AssertionError):
            npp._check_micro_batch_data_valid("not_a_tensor")


class TestGetAlignModeScale(unittest.TestCase):
    """Tests for _get_align_mode_scale function."""

    def test_basic(self):
        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            _get_align_mode_scale,
        )

        mock_hcg = MagicMock()
        mock_hcg.get_data_parallel_world_size.return_value = 2
        mock_hcg.get_sharding_parallel_world_size.return_value = 1
        with patch(
            "paddle.distributed.fleet.get_hybrid_communicate_group",
            return_value=mock_hcg,
        ):
            result = _get_align_mode_scale()
            self.assertEqual(result, 2)

    def test_sharding_dp(self):
        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            _get_align_mode_scale,
        )

        mock_hcg = MagicMock()
        mock_hcg.get_data_parallel_world_size.return_value = 2
        mock_hcg.get_sharding_parallel_world_size.return_value = 4
        with patch(
            "paddle.distributed.fleet.get_hybrid_communicate_group",
            return_value=mock_hcg,
        ):
            result = _get_align_mode_scale()
            self.assertEqual(result, 8)


class TestParallelBase(unittest.TestCase):
    """Tests for ParallelBase abstract class."""

    def test_cannot_instantiate(self):
        from paddlefleet.pipeline_parallel.pipeline_parallel import ParallelBase

        with self.assertRaises(TypeError):
            ParallelBase()


if __name__ == "__main__":
    unittest.main()
