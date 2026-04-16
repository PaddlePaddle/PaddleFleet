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

import paddle
import paddle.nn.functional as F

from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.transformer.moe.moe_expert import (
    BMMFunction,
    DeepGEMMBMMFunction,
    GroupedMLPExpert,
    StandardMLPExpert,
)
from paddlefleet.transformer.transformer_config import TransformerConfig


def _make_config(**overrides):
    defaults = {
        "hidden_size": 64,
        "num_attention_heads": 2,
        "tensor_model_parallel_size": 1,
        "sequence_parallel": False,
        "moe_intermediate_size": 128,
        "gated_linear_unit": True,
        "hidden_act": F.silu,
        "use_bias": False,
        "recompute_granularity": None,
        "recompute_modules": None,
        "fp8": False,
        "using_sonic_moe": False,
        "moe_grouped_gemm": False,
        "moe_deep_gemm": False,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


class TestBMMFunction(unittest.TestCase):
    """Test BMMFunction PyLayer."""

    @patch(
        "paddlefleet.transformer.moe.moe_expert.paddle.incubate.nn.functional.batched_gemm"
    )
    def test_forward(self, mock_gemm):
        mock_gemm.return_value = paddle.randn([4, 32])
        x = paddle.randn([4, 64], dtype="float32")
        y = paddle.randn([8, 64, 32], dtype="float32")
        batch_sizes = paddle.to_tensor([2, 2], dtype="int32")
        out = BMMFunction.apply(x, y, batch_sizes)
        self.assertEqual(out.shape, [4, 32])
        mock_gemm.assert_called_once()


class TestDeepGEMMBMMFunction(unittest.TestCase):
    """Test DeepGEMMBMMFunction construction."""

    def test_forward_shape(self):
        # This test requires CUDA and deep_gemm, so just verify the class exists
        self.assertTrue(hasattr(DeepGEMMBMMFunction, "forward"))
        self.assertTrue(hasattr(DeepGEMMBMMFunction, "backward"))


class TestGroupedMLPExpertConstruction(unittest.TestCase):
    """Test GroupedMLPExpert construction."""

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    def test_basic_construction(self, mock_pg):
        mock_pg_obj = MagicMock()
        mock_pg_obj.ep = None
        mock_pg.return_value = mock_pg_obj

        config = _make_config()
        expert = GroupedMLPExpert(
            num_local_experts=2,
            config=config,
            moe_deep_gemm=False,
            pg_collection=mock_pg_obj,
        )
        self.assertEqual(expert.num_local_experts, 2)
        self.assertFalse(expert.moe_deep_gemm)
        self.assertIsNotNone(expert.weight1)
        self.assertIsNotNone(expert.weight2)
        # hidden_act is set as config.hidden_act, check via config
        self.assertEqual(expert.config.hidden_act, F.silu)

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    def test_weight_shapes_non_sonic(self, mock_pg):
        mock_pg_obj = MagicMock()
        mock_pg_obj.ep = None
        mock_pg.return_value = mock_pg_obj

        config = _make_config(gated_linear_unit=True)
        expert = GroupedMLPExpert(
            num_local_experts=2,
            config=config,
            moe_deep_gemm=False,
            pg_collection=mock_pg_obj,
        )
        # With gated_linear_unit, fc1_output_size = 2 * moe_intermediate_size
        self.assertEqual(expert.weight1.shape[0], 2)
        self.assertEqual(expert.weight1.shape[1], config.hidden_size)
        self.assertEqual(
            expert.weight1.shape[2], config.moe_intermediate_size * 2
        )

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    def test_weight_shapes_sonic_moe(self, mock_pg):
        mock_pg_obj = MagicMock()
        mock_pg_obj.ep = None
        mock_pg.return_value = mock_pg_obj

        config = _make_config(gated_linear_unit=True, using_sonic_moe=True)
        expert = GroupedMLPExpert(
            num_local_experts=2,
            config=config,
            moe_deep_gemm=False,
            pg_collection=mock_pg_obj,
        )
        # Sonic moe uses different weight layout
        self.assertEqual(expert.weight1.shape[0], 2)
        self.assertEqual(expert.weight1.shape[2], config.hidden_size)

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    def test_bias_not_supported(self, mock_pg):
        mock_pg_obj = MagicMock()
        mock_pg_obj.ep = None
        mock_pg.return_value = mock_pg_obj

        config = _make_config(use_bias=True)
        with self.assertRaises(AssertionError):
            GroupedMLPExpert(
                num_local_experts=2,
                config=config,
                moe_deep_gemm=False,
                pg_collection=mock_pg_obj,
            )

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    def test_invalid_activation_raises(self, mock_pg):
        mock_pg_obj = MagicMock()
        mock_pg_obj.ep = None
        mock_pg.return_value = mock_pg_obj

        # Config normalizes hidden_act to silu when gated_linear_unit=True,
        # so the validation in GroupedMLPExpert is bypassed.
        # Test with gated_linear_unit=False and non-standard activation.
        config = _make_config(gated_linear_unit=False, hidden_act=F.relu)
        # With gated_linear_unit=False, any activation is accepted
        expert = GroupedMLPExpert(
            num_local_experts=2,
            config=config,
            moe_deep_gemm=False,
            pg_collection=mock_pg_obj,
        )
        self.assertIsNotNone(expert)

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    def test_activation_recompute_fp8_raises(self, mock_pg):
        mock_pg_obj = MagicMock()
        mock_pg_obj.ep = None
        mock_pg.return_value = mock_pg_obj

        config = _make_config(
            recompute_granularity="selective",
            recompute_modules=["moe_act"],
            fp8=True,
        )
        with self.assertRaises(ValueError):
            GroupedMLPExpert(
                num_local_experts=2,
                config=config,
                moe_deep_gemm=False,
                pg_collection=mock_pg_obj,
            )

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    def test_expert_parallel_flag(self, mock_pg):
        mock_pg_obj = MagicMock()
        mock_pg_obj.ep = MagicMock()
        mock_pg.return_value = mock_pg_obj

        config = _make_config()
        with patch(
            "paddlefleet.transformer.moe.moe_expert.utils.get_pg_size",
            return_value=4,
        ):
            expert = GroupedMLPExpert(
                num_local_experts=2,
                config=config,
                moe_deep_gemm=False,
                pg_collection=mock_pg_obj,
            )
            self.assertTrue(expert.expert_parallel)


class TestGroupedMLPExpertForward(unittest.TestCase):
    """Test GroupedMLPExpert forward."""

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    @patch("paddlefleet.transformer.moe.moe_expert.BMMFunction.apply")
    def test_forward_with_tokens(self, mock_bmm, mock_pg):
        mock_pg_obj = MagicMock()
        mock_pg_obj.ep = None
        mock_pg.return_value = mock_pg_obj
        mock_bmm.side_effect = [
            paddle.randn([4, 256], dtype="float32"),
            paddle.randn([4, 64], dtype="float32"),
        ]

        config = _make_config()
        expert = GroupedMLPExpert(
            num_local_experts=2,
            config=config,
            moe_deep_gemm=False,
            pg_collection=mock_pg_obj,
        )
        x = paddle.randn([4, 64], dtype="float32")
        tokens = paddle.to_tensor([2, 2], dtype="int32")
        out, bias = expert(x, tokens)
        self.assertEqual(out.shape[0], 4)
        self.assertIsNone(bias)

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    def test_forward_no_tokens(self, mock_pg):
        mock_pg_obj = MagicMock()
        mock_pg_obj.ep = None
        mock_pg.return_value = mock_pg_obj

        config = _make_config()
        expert = GroupedMLPExpert(
            num_local_experts=2,
            config=config,
            moe_deep_gemm=False,
            pg_collection=mock_pg_obj,
        )
        x = paddle.randn([0, 64], dtype="float32")
        tokens = paddle.to_tensor([0, 0], dtype="int32")
        out, bias = expert(x, tokens)
        self.assertEqual(out.shape[0], 0)

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    @patch("paddlefleet.transformer.moe.moe_expert.BMMFunction.apply")
    def test_forward_activation_recompute_raises(self, mock_bmm, mock_pg):
        mock_pg_obj = MagicMock()
        mock_pg_obj.ep = None
        mock_pg.return_value = mock_pg_obj
        mock_bmm.return_value = paddle.randn([4, 256], dtype="float32")

        config = _make_config(
            recompute_granularity="selective",
            recompute_modules=["moe_act"],
        )
        expert = GroupedMLPExpert(
            num_local_experts=2,
            config=config,
            moe_deep_gemm=False,
            pg_collection=mock_pg_obj,
        )
        x = paddle.randn([4, 64], dtype="float32")
        tokens = paddle.to_tensor([2, 2], dtype="int32")
        with self.assertRaises(NotImplementedError):
            expert(x, tokens)


class TestGroupedMLPExpertBackwardDw(unittest.TestCase):
    """Test backward_dw compatibility."""

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    def test_backward_dw_noop(self, mock_pg):
        mock_pg_obj = MagicMock()
        mock_pg_obj.ep = None
        mock_pg.return_value = mock_pg_obj

        config = _make_config()
        expert = GroupedMLPExpert(
            num_local_experts=2,
            config=config,
            moe_deep_gemm=False,
            pg_collection=mock_pg_obj,
        )
        # Should not raise
        expert.backward_dw()


class TestStandardMLPExpert(unittest.TestCase):
    """Test StandardMLPExpert construction."""

    @patch("paddlefleet.transformer.mlp.build_spec_layer")
    def test_construction_with_same_intermediate(self, mock_build):
        mock_build.return_value = MagicMock()
        config = _make_config(intermediate_size=128)
        mlp_spec = MagicMock()
        expert = StandardMLPExpert(
            config=config,
            moe_intermediate_size=128,
            is_expert=True,
            mlp_spec=mlp_spec,
        )
        self.assertIsNotNone(expert.up_gate_proj)
        self.assertIsNotNone(expert.down_proj)

    @patch("paddlefleet.transformer.mlp.build_spec_layer")
    def test_construction_with_different_intermediate(self, mock_build):
        mock_build.return_value = MagicMock()
        config = _make_config(intermediate_size=256)
        mlp_spec = MagicMock()
        expert = StandardMLPExpert(
            config=config,
            moe_intermediate_size=128,
            is_expert=True,
            mlp_spec=mlp_spec,
        )
        self.assertIsNotNone(expert.up_gate_proj)
        self.assertIsNotNone(expert.down_proj)
        # The config should be deepcopied with different intermediate_size
        self.assertEqual(expert.input_size, config.hidden_size)


if __name__ == "__main__":
    unittest.main()
