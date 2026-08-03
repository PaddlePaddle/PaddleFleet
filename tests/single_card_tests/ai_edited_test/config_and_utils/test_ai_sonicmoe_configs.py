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
"""Tests for the rename of sonicmoe_quant_format to fp8_weight_quant_format."""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)


class TestFp8WeightQuantFormat(unittest.TestCase):
    def _make_config(self, **overrides):
        from paddlefleet.transformer.transformer_config import (
            TransformerConfig,
        )

        return TransformerConfig(
            hidden_size=64, num_attention_heads=4, **overrides
        )

    def _make_expert_stub(self, quant_format):
        """Minimal stand-in for a SonicMoEExpert instance."""
        stub = mock.MagicMock()
        stub.config = self._make_config(fp8_weight_quant_format=quant_format)
        stub._is_last_micro_batch = True
        stub.weight1 = mock.Mock(fp8=None)
        stub.weight2 = mock.Mock(fp8=None)
        return stub

    def test_field_renamed(self):
        config = self._make_config()
        self.assertEqual(config.fp8_weight_quant_format, "32x32")
        self.assertFalse(hasattr(config, "sonicmoe_quant_format"))
        self.assertFalse(hasattr(config, "sonicmoe_save_upgate_out_in_fp8"))

    def test_deprecated_name_rejected(self):
        config = self._make_config()
        with self.assertRaises(ValueError) as ctx:
            config._process_attribute("sonicmoe_quant_format", "1x32")
        self.assertIn("fp8_weight_quant_format", str(ctx.exception))

    def test_release_fp8_weight_after_fwd_reads_new_field(self):
        from paddlefleet.transformer.moe import moe_expert

        with mock.patch.object(
            moe_expert, "g_shard_bypass_dygraph_optimizer", 0
        ):
            release = moe_expert.SonicMoEExpert._release_fp8_weight_after_fwd
            self.assertTrue(release(self._make_expert_stub("1x32"), False))
            self.assertFalse(release(self._make_expert_stub("32x32"), False))

    def test_quant_weight_rejects_unsupported_format(self):
        from paddlefleet.transformer.moe import moe_expert

        with self.assertRaises(AssertionError) as ctx:
            moe_expert.SonicMoEExpert.quant_weight(
                self._make_expert_stub("16x16")
            )
        self.assertIn("fp8_weight_quant_format", str(ctx.exception))

    @mock.patch("paddlefleet.transformer.moe.moe_layer.StandardMLPExpert")
    @mock.patch("paddlefleet.transformer.moe.moe_layer.paddle.version")
    @mock.patch("paddlefleet.transformer.moe.moe_layer.paddlefleet_ops")
    @mock.patch("paddlefleet.transformer.moe.moe_layer.utils")
    def test_moe_layer_warns_when_fp8_without_sonic_moe(
        self, mock_utils, mock_ops, mock_version, mock_expert
    ):
        import paddle

        from paddlefleet.tensor_parallel import (
            ColumnParallelLinear,
            RowParallelLinear,
        )
        from paddlefleet.transformer.mlp import MLPSublayersSpec
        from paddlefleet.transformer.moe import moe_layer

        mock_utils.get_pg_size.return_value = 1
        mock_utils.get_pg_rank.return_value = 0
        mock_ops.is_sonic_moe_available.return_value = False
        mock_version.cuda.return_value = "12.2"
        mock_expert.return_value = paddle.nn.Layer()

        config = self._make_config(
            intermediate_size=256,
            moe_intermediate_size=128,
            n_routed_experts=4,
            num_experts_per_tok=2,
            n_shared_experts=0,
            gated_linear_unit=True,
            fp8="hybrid",
            moe_use_fusion_node=True,
            moe_expert_fusion=True,
            moe_deep_gemm=False,
            using_sonic_moe=False,
        )
        sublayers = moe_layer.MoESublayers(
            mlp_spec=MLPSublayersSpec(
                up_gate_proj=ColumnParallelLinear,
                hidden_act=None,
                down_proj=RowParallelLinear,
            )
        )
        with mock.patch.object(moe_layer, "logger") as mock_logger:
            moe_layer.MoELayer(
                config,
                sublayers=sublayers,
                pg_collection=mock.MagicMock(),
            )
        self.assertTrue(
            any(
                "fp8_weight_quant_format" in str(call)
                for call in mock_logger.warning.call_args_list
            )
        )


if __name__ == "__main__":
    unittest.main()
