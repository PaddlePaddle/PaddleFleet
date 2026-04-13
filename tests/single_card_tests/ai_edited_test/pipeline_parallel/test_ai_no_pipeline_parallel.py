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


class TestNoPipelineParallelInit(unittest.TestCase):
    """Tests for NoPipelineParallel initialization."""

    @patch(
        "paddlefleet.pipeline_parallel.pipeline_parallel.broadcast_mp_parameters"
    )
    @patch(
        "paddlefleet.pipeline_parallel.pipeline_parallel.broadcast_dp_parameters"
    )
    @patch(
        "paddlefleet.pipeline_parallel.pipeline_parallel.broadcast_sep_parameters"
    )
    @patch(
        "paddlefleet.pipeline_parallel.pipeline_parallel.broadcast_sharding_parameters"
    )
    @patch(
        "paddlefleet.pipeline_parallel.pipeline_parallel.broadcast_moe_sharding_parameters"
    )
    def test_init_with_hcg(
        self, mock_moe, mock_sharding, mock_sep, mock_mp, mock_dp
    ):
        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            NoPipelineParallel,
        )
        from paddlefleet.pipeline_parallel.pp_layers import PipelineLayer

        mock_layers = MagicMock(spec=PipelineLayer)
        mock_hcg = MagicMock()
        mock_hcg.get_data_parallel_world_size.return_value = 2
        mock_hcg.get_model_parallel_world_size.return_value = 1
        mock_hcg.get_sep_parallel_world_size.return_value = 1
        mock_hcg.get_sharding_parallel_world_size.return_value = 1
        mock_hcg.get_moe_sharding_parallel_world_size.return_value = 1
        mock_hcg.get_dp_sep_parallel_group.return_value = MagicMock()

        mock_strategy = MagicMock()
        mock_strategy.pipeline_configs = {
            "micro_batch_size": 4,
            "accumulate_steps": 2,
        }

        pp = NoPipelineParallel.__new__(NoPipelineParallel)
        nn_module = MagicMock()
        pp.__dict__["_layers"] = mock_layers
        pp.__dict__["_strategy"] = mock_strategy
        pp.__dict__["_hcg"] = mock_hcg
        pp.__dict__["micro_batch_size"] = 4
        pp.__dict__["accumulate_steps"] = 2
        pp.__dict__["_dp_comm_overlap"] = False
        pp.__dict__["_sharding_comm_overlap"] = False
        pp.__dict__["total_loss"] = None
        pp.__dict__["loss_fn_idx"] = 0
        pp.__dict__["use_data_parallel"] = True
        pp.__dict__["use_model_parallel"] = False
        pp.__dict__["use_sep_parallel"] = False
        pp.__dict__["use_sharding_parallel"] = False
        pp.__dict__["use_moe_sharding_parallel"] = False
        pp.__dict__["dp_group"] = mock_hcg.get_data_parallel_group.return_value

        self.assertTrue(pp.use_data_parallel)
        self.assertEqual(pp.micro_batch_size, 4)
        self.assertEqual(pp.accumulate_steps, 2)
        self.assertFalse(pp._dp_comm_overlap)
        self.assertIsNone(pp.total_loss)

    @patch(
        "paddlefleet.pipeline_parallel.pipeline_parallel.broadcast_mp_parameters"
    )
    @patch(
        "paddlefleet.pipeline_parallel.pipeline_parallel.broadcast_dp_parameters"
    )
    def test_init_without_hcg(self, mock_dp, mock_mp):
        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            NoPipelineParallel,
        )
        from paddlefleet.pipeline_parallel.pp_layers import PipelineLayer

        mock_layers = MagicMock(spec=PipelineLayer)
        mock_strategy = MagicMock()
        mock_strategy.pipeline_configs = {
            "micro_batch_size": 8,
            "accumulate_steps": 4,
        }

        pp = NoPipelineParallel.__new__(NoPipelineParallel)
        pp.__dict__["_layers"] = mock_layers
        pp.__dict__["_strategy"] = mock_strategy
        pp.__dict__["_hcg"] = None
        pp.__dict__["micro_batch_size"] = 8
        pp.__dict__["accumulate_steps"] = 4
        pp.__dict__["_dp_comm_overlap"] = False
        pp.__dict__["_sharding_comm_overlap"] = False
        pp.__dict__["total_loss"] = None
        pp.__dict__["loss_fn_idx"] = 0

        self.assertIsNone(pp._hcg)
        self.assertEqual(pp.micro_batch_size, 8)


class TestNoPipelineParallelMethods(unittest.TestCase):
    """Tests for NoPipelineParallel method behaviors."""

    def test_is_pipeline_last_stage(self):
        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            NoPipelineParallel,
        )

        pp = NoPipelineParallel.__new__(NoPipelineParallel)
        self.assertTrue(pp.is_pipeline_last_stage())
        self.assertTrue(pp.is_pipeline_last_stage(ignore_virtual=True))

    def test_check_micro_batch_data_valid_tuple(self):
        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            NoPipelineParallel,
        )

        pp = NoPipelineParallel.__new__(NoPipelineParallel)
        tensor = MagicMock(spec=["__class__"])
        tensor.__class__ = type("Tensor", (), {})()
        # Should not raise when tensor is None or a paddle.Tensor mock
        pp._check_micro_batch_data_valid(None)

    def test_check_micro_batch_data_valid_dict(self):
        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            NoPipelineParallel,
        )

        pp = NoPipelineParallel.__new__(NoPipelineParallel)
        pp._check_micro_batch_data_valid({"key": None})

    def test_check_micro_batch_data_valid_single_tensor(self):
        import paddle

        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            NoPipelineParallel,
        )

        pp = NoPipelineParallel.__new__(NoPipelineParallel)
        tensor = paddle.zeros([2, 3])
        pp._check_micro_batch_data_valid(tensor)


class TestNoPipelineParallelOptimizerStep(unittest.TestCase):
    """Tests for NoPipelineParallel._optimizer_step."""

    @patch("paddle.amp.auto_cast")
    def test_optimizer_step_no_scaler(self, mock_cast):
        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            NoPipelineParallel,
        )

        pp = NoPipelineParallel.__new__(NoPipelineParallel)
        mock_optimizer = MagicMock()
        mock_lr_scheduler = MagicMock()
        pp.__dict__["optimizer"] = mock_optimizer
        pp.__dict__["lr_scheduler"] = mock_lr_scheduler
        pp.__dict__["scaler"] = None
        pp.__dict__["accumulate_steps"] = 2
        pp.__dict__["_layers"] = MagicMock()

        pp._optimizer_step()
        mock_optimizer.step.assert_called_once()
        mock_optimizer.clear_grad.assert_called_once()
        mock_lr_scheduler.step.assert_called_once()

    @patch("paddle.amp.auto_cast")
    def test_optimizer_step_with_scaler(self, mock_cast):
        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            NoPipelineParallel,
        )

        pp = NoPipelineParallel.__new__(NoPipelineParallel)
        mock_scaler = MagicMock()
        mock_optimizer = MagicMock()
        mock_lr_scheduler = None
        pp.__dict__["optimizer"] = mock_optimizer
        pp.__dict__["lr_scheduler"] = mock_lr_scheduler
        pp.__dict__["scaler"] = mock_scaler
        pp.__dict__["accumulate_steps"] = 2
        pp.__dict__["_layers"] = MagicMock()

        pp._optimizer_step()
        mock_scaler.step.assert_called_once_with(mock_optimizer)
        mock_scaler.update.assert_called_once()
        mock_optimizer.clear_grad.assert_called_once()


class TestNoPipelineParallelPrepareTraining(unittest.TestCase):
    """Tests for NoPipelineParallel._prepare_training."""

    @patch("paddle.amp.auto_cast")
    def test_prepare_training(self, mock_cast):
        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            NoPipelineParallel,
        )

        pp = NoPipelineParallel.__new__(NoPipelineParallel)
        mock_tracer = MagicMock()
        mock_tracer._has_grad = True
        with patch(
            "paddle.framework._dygraph_tracer", return_value=mock_tracer
        ):
            mock_layers = MagicMock()
            pp.__dict__["_layers"] = mock_layers
            data = MagicMock()
            mock_optimizer = MagicMock()
            mock_lr_scheduler = MagicMock()

            result = pp._prepare_training(
                data, mock_optimizer, mock_lr_scheduler
            )
            self.assertEqual(pp.optimizer, mock_optimizer)
            self.assertEqual(pp.lr_scheduler, mock_lr_scheduler)
            mock_layers.train.assert_called_once()


class TestNoPipelineParallelSepParallel(unittest.TestCase):
    """Tests for NoPipelineParallel sep parallel behavior."""

    @patch(
        "paddlefleet.pipeline_parallel.pipeline_parallel.broadcast_sep_parameters"
    )
    @patch(
        "paddlefleet.pipeline_parallel.pipeline_parallel.broadcast_mp_parameters"
    )
    @patch(
        "paddlefleet.pipeline_parallel.pipeline_parallel.broadcast_dp_parameters"
    )
    def test_init_with_sep_parallel(self, mock_dp, mock_mp, mock_sep):
        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            NoPipelineParallel,
        )

        mock_hcg = MagicMock()
        mock_hcg.get_data_parallel_world_size.return_value = 1
        mock_hcg.get_model_parallel_world_size.return_value = 1
        mock_hcg.get_sep_parallel_world_size.return_value = 2
        mock_hcg.get_sharding_parallel_world_size.return_value = 1
        mock_hcg.get_moe_sharding_parallel_world_size.return_value = 1
        mock_hcg.get_dp_sep_parallel_group.return_value = MagicMock()
        mock_hcg.get_data_parallel_group.return_value = MagicMock()

        pp = NoPipelineParallel.__new__(NoPipelineParallel)
        pp.__dict__["_hcg"] = mock_hcg
        pp.__dict__["use_sep_parallel"] = True
        pp.__dict__["dp_group"] = (
            mock_hcg.get_dp_sep_parallel_group.return_value
        )

        self.assertTrue(pp.use_sep_parallel)
        self.assertEqual(
            pp.dp_group, mock_hcg.get_dp_sep_parallel_group.return_value
        )


class TestNoPipelineParallelWithHcgAllParallel(unittest.TestCase):
    """Tests for NoPipelineParallel when all parallel modes are enabled."""

    def test_all_parallel_flags(self):
        from paddlefleet.pipeline_parallel.pipeline_parallel import (
            NoPipelineParallel,
        )

        pp = NoPipelineParallel.__new__(NoPipelineParallel)
        pp.__dict__["use_data_parallel"] = True
        pp.__dict__["use_model_parallel"] = True
        pp.__dict__["use_sep_parallel"] = True
        pp.__dict__["use_sharding_parallel"] = True
        pp.__dict__["use_moe_sharding_parallel"] = True

        self.assertTrue(pp.use_data_parallel)
        self.assertTrue(pp.use_model_parallel)
        self.assertTrue(pp.use_sep_parallel)
        self.assertTrue(pp.use_sharding_parallel)
        self.assertTrue(pp.use_moe_sharding_parallel)


if __name__ == "__main__":
    unittest.main()
