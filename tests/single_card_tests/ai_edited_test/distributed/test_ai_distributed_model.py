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


# Tests for src/paddlefleet/distributed/model.py
# Test distributed_model function

import unittest
from unittest import mock

import paddle
from paddle.distributed.fleet.meta_parallel import PipelineLayer


class TestDistributedModelSingleGPU(unittest.TestCase):
    """Tests for distributed_model in single GPU mode."""

    def test_distributed_model_world_size_one(self):
        """Test distributed_model returns NoPipelineParallel when world_size <= 1."""
        from paddle.distributed.fleet import distributed_model

        mock_model = mock.MagicMock()
        mock_fleet = mock.MagicMock()
        mock_strategy = mock.MagicMock()
        mock_strategy.amp = False
        mock_fleet._user_defined_strategy = mock_strategy
        mock_fleet.fleet = mock_fleet

        with mock.patch("paddle.distributed.get_world_size", return_value=1):  # noqa: SIM117
            with mock.patch("paddle.distributed.fleet.fleet", mock_fleet):
                with mock.patch(
                    "paddle.distributed.fleet.model.NoPipelineParallel",
                    return_value=mock_model,
                ) as mock_nopp:
                    result = distributed_model(mock_model)
                    mock_nopp.assert_called_once()

    def test_distributed_model_none_raises(self):
        """Test distributed_model raises when model is None."""
        from paddle.distributed.fleet import distributed_model

        mock_fleet = mock.MagicMock()
        mock_strategy = mock.MagicMock()
        mock_fleet._user_defined_strategy = mock_strategy
        mock_fleet.fleet = mock_fleet

        with mock.patch("paddle.distributed.get_world_size", return_value=1):  # noqa: SIM117
            with mock.patch("paddle.distributed.fleet.fleet", mock_fleet):
                with self.assertRaises(AssertionError):
                    distributed_model(None)


def _make_mock_pipeline_model():
    """Create a mock model that is an instance of PipelineLayer."""
    mock_model = mock.MagicMock(spec=PipelineLayer)
    mock_model.get_num_virtual_stages = mock.MagicMock(return_value=1)
    return mock_model


class TestDistributedModelAMP(unittest.TestCase):
    """Tests for distributed_model with AMP enabled."""

    def test_amp_o2_fp16(self):
        """Test AMP O2 level with pure fp16."""
        from paddle.distributed.fleet import distributed_model

        mock_model = _make_mock_pipeline_model()
        mock_fleet = mock.MagicMock()
        mock_strategy = mock.MagicMock()
        mock_strategy.amp = True
        mock_strategy.amp_configs = {
            "use_pure_fp16": True,
            "use_pure_bf16": False,
            "init_loss_scaling": 1.0,
            "incr_ratio": 2.0,
            "decr_ratio": 0.5,
            "incr_every_n_steps": 2000,
            "decr_every_n_nan_or_inf": 2000,
            "use_dynamic_loss_scaling": True,
        }
        mock_hcg = mock.MagicMock()
        mock_hcg.get_parallel_mode.return_value = "DATA_PARALLEL"
        mock_fleet._user_defined_strategy = mock_strategy
        mock_fleet._hcg = mock_hcg
        mock_fleet.fleet = mock_fleet

        with mock.patch("paddle.distributed.get_world_size", return_value=4):  # noqa: SIM117
            with mock.patch("paddle.distributed.fleet.fleet", mock_fleet):
                with mock.patch(
                    "paddle.distributed.fleet.model.NoPipelineParallel",
                    return_value=mock_model,
                ) as mock_nopp:
                    with mock.patch(
                        "paddle.amp.decorate", return_value=mock_model
                    ) as mock_dec:
                        with mock.patch("paddle.amp.GradScaler"):
                            distributed_model(mock_model)
                            mock_dec.assert_called_once()
                            call_kwargs = mock_dec.call_args[1]
                            self.assertEqual(call_kwargs["level"], "O2")
                            self.assertEqual(call_kwargs["dtype"], "float16")

    def test_amp_o2_bf16(self):
        """Test AMP O2 level with pure bf16."""
        from paddle.distributed.fleet import distributed_model

        mock_model = _make_mock_pipeline_model()
        mock_fleet = mock.MagicMock()
        mock_strategy = mock.MagicMock()
        mock_strategy.amp = True
        mock_strategy.amp_configs = {
            "use_pure_fp16": False,
            "use_pure_bf16": True,
            "init_loss_scaling": 1.0,
            "incr_ratio": 2.0,
            "decr_ratio": 0.5,
            "incr_every_n_steps": 2000,
            "decr_every_n_nan_or_inf": 2000,
            "use_dynamic_loss_scaling": True,
        }
        mock_hcg = mock.MagicMock()
        mock_hcg.get_parallel_mode.return_value = "DATA_PARALLEL"
        mock_fleet._user_defined_strategy = mock_strategy
        mock_fleet._hcg = mock_hcg
        mock_fleet.fleet = mock_fleet

        with mock.patch("paddle.distributed.get_world_size", return_value=4):  # noqa: SIM117
            with mock.patch("paddle.distributed.fleet.fleet", mock_fleet):
                with mock.patch(
                    "paddle.distributed.fleet.model.NoPipelineParallel",
                    return_value=mock_model,
                ):
                    with mock.patch(
                        "paddle.amp.decorate", return_value=mock_model
                    ) as mock_dec:
                        with mock.patch("paddle.amp.GradScaler"):
                            distributed_model(mock_model)
                            call_kwargs = mock_dec.call_args[1]
                            self.assertEqual(call_kwargs["dtype"], "bfloat16")

    def test_amp_o1(self):
        """Test AMP O1 level (not pure fp16 or bf16)."""
        from paddle.distributed.fleet import distributed_model

        mock_model = _make_mock_pipeline_model()
        mock_fleet = mock.MagicMock()
        mock_strategy = mock.MagicMock()
        mock_strategy.amp = True
        mock_strategy.amp_configs = {
            "use_pure_fp16": False,
            "use_pure_bf16": False,
            "init_loss_scaling": 1.0,
            "incr_ratio": 2.0,
            "decr_ratio": 0.5,
            "incr_every_n_steps": 2000,
            "decr_every_n_nan_or_inf": 2000,
            "use_dynamic_loss_scaling": True,
        }
        mock_hcg = mock.MagicMock()
        mock_hcg.get_parallel_mode.return_value = "DATA_PARALLEL"
        mock_fleet._user_defined_strategy = mock_strategy
        mock_fleet._hcg = mock_hcg
        mock_fleet.fleet = mock_fleet

        with mock.patch("paddle.distributed.get_world_size", return_value=4):  # noqa: SIM117
            with mock.patch("paddle.distributed.fleet.fleet", mock_fleet):
                with mock.patch(
                    "paddle.distributed.fleet.model.NoPipelineParallel",
                    return_value=mock_model,
                ):
                    with mock.patch(
                        "paddle.amp.decorate", return_value=mock_model
                    ) as mock_dec:
                        with mock.patch("paddle.amp.GradScaler"):
                            distributed_model(mock_model)
                            # O1 should not call amp.decorate
                            mock_dec.assert_not_called()

    def test_grad_scaler_creation(self):
        """Test GradScaler is created when AMP is enabled."""
        from paddle.distributed.fleet import distributed_model

        mock_model = _make_mock_pipeline_model()
        mock_fleet = mock.MagicMock()
        mock_strategy = mock.MagicMock()
        mock_strategy.amp = True
        mock_strategy.amp_configs = {
            "use_pure_fp16": True,
            "use_pure_bf16": False,
            "init_loss_scaling": 2.0,
            "incr_ratio": 3.0,
            "decr_ratio": 0.25,
            "incr_every_n_steps": 1000,
            "decr_every_n_nan_or_inf": 500,
            "use_dynamic_loss_scaling": True,
        }
        mock_hcg = mock.MagicMock()
        mock_hcg.get_parallel_mode.return_value = "DATA_PARALLEL"
        mock_fleet._user_defined_strategy = mock_strategy
        mock_fleet._hcg = mock_hcg
        mock_fleet.fleet = mock_fleet

        with mock.patch("paddle.distributed.get_world_size", return_value=4):  # noqa: SIM117
            with mock.patch("paddle.distributed.fleet.fleet", mock_fleet):
                with mock.patch(
                    "paddle.distributed.fleet.model.NoPipelineParallel",
                    return_value=mock_model,
                ):
                    with mock.patch(
                        "paddle.amp.decorate", return_value=mock_model
                    ):
                        with mock.patch("paddle.amp.GradScaler") as mock_gs:
                            distributed_model(mock_model)
                            mock_gs.assert_called_once()
                            call_kwargs = mock_gs.call_args[1]
                            self.assertEqual(
                                call_kwargs["init_loss_scaling"], 2.0
                            )
                            self.assertEqual(call_kwargs["incr_ratio"], 3.0)


class TestDistributedModelPipeline(unittest.TestCase):
    """Tests for distributed_model with pipeline parallelism."""

    def test_not_pipeline_layer_raises(self):
        """Test non-PipelineLayer model raises in multi-GPU mode."""
        from paddle.distributed.fleet import distributed_model
        from paddle.distributed.fleet.base.topology import ParallelMode

        mock_model = mock.MagicMock(spec=paddle.nn.Layer)
        mock_fleet = mock.MagicMock()
        mock_strategy = mock.MagicMock()
        mock_strategy.amp = False
        mock_hcg = mock.MagicMock()
        mock_hcg.get_parallel_mode.return_value = ParallelMode.PIPELINE_PARALLEL
        mock_fleet._user_defined_strategy = mock_strategy
        mock_fleet._hcg = mock_hcg
        mock_fleet.fleet = mock_fleet

        with mock.patch("paddle.distributed.get_world_size", return_value=4):  # noqa: SIM117
            with mock.patch("paddle.distributed.fleet.fleet", mock_fleet):
                with self.assertRaises(AssertionError):
                    distributed_model(mock_model)

    # TODO(hushenwei2000): enable this test after migrate to paddle pp
    # Paddle has implemented DualPipeVParallel, so it no longer raises ValueError.
    # def test_dualpipev_raises(self):
    #     """Test dualpipev raises ValueError."""
    #     from paddle.distributed.fleet.base.topology import ParallelMode
    #
    #     from paddle.distributed.fleet import distributed_model
    #
    #     mock_model = _make_mock_pipeline_model()
    #     mock_fleet = mock.MagicMock()
    #     mock_strategy = mock.MagicMock()
    #     mock_strategy.amp = False
    #     mock_strategy.hybrid_configs = {
    #         "pp_configs": mock.MagicMock(use_dualpipev=True)
    #     }
    #     mock_hcg = mock.MagicMock()
    #     mock_hcg.get_parallel_mode.return_value = ParallelMode.PIPELINE_PARALLEL
    #     mock_fleet._user_defined_strategy = mock_strategy
    #     mock_fleet._hcg = mock_hcg
    #     mock_fleet.fleet = mock_fleet
    #
    #     with mock.patch("paddle.distributed.get_world_size", return_value=4):
    #         with mock.patch("paddle.distributed.fleet.fleet", mock_fleet):
    #             with self.assertRaises(ValueError) as ctx:
    #                 distributed_model(mock_model)
    #             self.assertIn("dualpipev", str(ctx.exception))

    def test_1f1b_pipeline(self):
        """Test 1f1b pipeline when virtual stages == 1."""
        from paddle.distributed.fleet import distributed_model
        from paddle.distributed.fleet.base.topology import ParallelMode

        mock_model = _make_mock_pipeline_model()
        mock_fleet = mock.MagicMock()
        mock_strategy = mock.MagicMock()
        mock_strategy.amp = False
        mock_strategy.hybrid_configs = {
            "pp_configs": mock.MagicMock(use_dualpipev=False)
        }
        mock_hcg = mock.MagicMock()
        mock_hcg.get_parallel_mode.return_value = ParallelMode.PIPELINE_PARALLEL
        mock_fleet._user_defined_strategy = mock_strategy
        mock_fleet._hcg = mock_hcg
        mock_fleet.fleet = mock_fleet

        with mock.patch("paddle.distributed.get_world_size", return_value=4):  # noqa: SIM117
            with mock.patch("paddle.distributed.fleet.fleet", mock_fleet):
                with mock.patch(
                    "paddle.distributed.fleet.model.PipelineParallel",
                    return_value=mock_model,
                ) as mock_pp:
                    result = distributed_model(mock_model)
                    mock_pp.assert_called_once()


class TestDistributedModelNonPipeline(unittest.TestCase):
    """Tests for distributed_model when not in pipeline parallel mode."""

    def test_non_pipeline_mode_uses_nopp_with_hcg(self):
        """Test non-pipeline mode uses NoPipelineParallel with hcg."""
        from paddle.distributed.fleet import distributed_model

        mock_model = _make_mock_pipeline_model()
        mock_result = mock.MagicMock()
        mock_fleet = mock.MagicMock()
        mock_strategy = mock.MagicMock()
        mock_strategy.amp = False
        mock_hcg = mock.MagicMock()
        mock_hcg.get_parallel_mode.return_value = "DATA_PARALLEL"
        mock_fleet._user_defined_strategy = mock_strategy
        mock_fleet._hcg = mock_hcg
        mock_fleet.fleet = mock_fleet

        with mock.patch("paddle.distributed.get_world_size", return_value=4):  # noqa: SIM117
            with mock.patch("paddle.distributed.fleet.fleet", mock_fleet):
                with mock.patch(
                    "paddle.distributed.fleet.model.NoPipelineParallel",
                    return_value=mock_result,
                ) as mock_nopp:
                    result = distributed_model(mock_model)
                    # Should be called with hcg when world_size > 1
                    call_kwargs = mock_nopp.call_args[1]
                    self.assertIn("hcg", call_kwargs)


if __name__ == "__main__":
    unittest.main()
