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

from paddlefleet.pipeline_parallel import PipelineLayer


class TestDistributedModelSingleProcess(unittest.TestCase):
    """Tests for distributed_model function in single-process mode."""

    @patch("paddle.distributed.get_world_size", return_value=1)
    @patch("paddle.distributed.fleet.fleet")
    def test_single_process_returns_no_pipeline(self, mock_fleet, mock_ws):
        from paddlefleet.distributed.model import distributed_model

        mock_strategy = MagicMock()
        mock_strategy.amp = False
        mock_strategy.pipeline_configs = {
            "micro_batch_size": 1,
            "accumulate_steps": 4,
        }
        mock_fleet._user_defined_strategy = mock_strategy

        model = MagicMock()
        with patch(
            "paddlefleet.distributed.model.NoPipelineParallel"
        ) as mock_nopp:
            mock_nopp.return_value = model
            result = distributed_model(model)
            # With world_size <= 1, should wrap in NoPipelineParallel
            mock_nopp.assert_called_once()

    @patch("paddle.distributed.get_world_size", return_value=1)
    @patch("paddle.distributed.fleet.fleet")
    def test_none_model_raises(self, mock_fleet, mock_ws):
        from paddlefleet.distributed.model import distributed_model

        mock_strategy = MagicMock()
        mock_strategy.pipeline_configs = {
            "micro_batch_size": 1,
            "accumulate_steps": 4,
        }
        mock_fleet._user_defined_strategy = mock_strategy

        with self.assertRaises(AssertionError):
            distributed_model(None)


class TestDistributedModelAMP(unittest.TestCase):
    """Tests for distributed_model with AMP settings."""


class TestDistributedModelPipelineParallel(unittest.TestCase):
    """Tests for distributed_model with pipeline parallel."""

    @patch("paddle.distributed.get_world_size", return_value=4)
    @patch("paddle.distributed.fleet.fleet")
    def test_not_pipeline_layer_raises(self, mock_fleet, mock_ws):
        from paddlefleet.distributed.model import distributed_model

        mock_hcg = MagicMock()
        mock_hcg.get_parallel_mode.return_value = MagicMock()
        mock_hcg.get_pipe_parallel_world_size.return_value = 2

        mock_strategy = MagicMock()
        mock_strategy.amp = False
        mock_strategy.pipeline_configs = {
            "micro_batch_size": 1,
            "accumulate_steps": 4,
        }
        mock_strategy.hybrid_configs = {
            "pp_configs": MagicMock(use_dualpipev=False)
        }
        mock_fleet._user_defined_strategy = mock_strategy
        mock_fleet._hcg = mock_hcg

        model = MagicMock()
        with (  # noqa: SIM117
            patch("paddlefleet.distributed.model.PipelineLayer", PipelineLayer),
            patch.object(
                PipelineLayer, "__instancecheck__", return_value=False
            ),
        ):
            # model is not PipelineLayer, should raise
            with self.assertRaises(AssertionError):
                distributed_model(model)


class TestDistributedModelInterleave(unittest.TestCase):
    """Tests for distributed_model interleave pipeline selection."""


class TestDistributedInitModule(unittest.TestCase):
    """Tests for the distributed __init__ module."""

    def test_import_distributed(self):
        import paddlefleet.distributed

        self.assertIsNotNone(paddlefleet.distributed)


if __name__ == "__main__":
    unittest.main()
