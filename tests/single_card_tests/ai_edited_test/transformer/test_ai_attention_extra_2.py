# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless distributed on the License is distributed on an "AS IS" BASIS,
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

from paddlefleet.transformer.attention import (
    Attention,
    CrossAttention,
    CrossAttentionSublayersSpec,
    SelfAttention,
    SelfAttentionSublayersSpec,
)
from paddlefleet.transformer.enums import AttnMaskType


class TestAttentionInitAttributes(unittest.TestCase):
    """Tests for Attention.__init__ attribute setup."""

    @patch("paddlefleet.transformer.attention.build_spec_layer")
    @patch("paddlefleet.transformer.attention.get_pg_size", return_value=1)
    @patch(
        "paddlefleet.transformer.attention.ProcessGroupCollection.use_mpu_process_groups"
    )
    def test_attention_sets_layer_number(self, mock_pg, mock_size, mock_build):
        """Attention should set layer_number from constructor."""
        mock_pg.return_value = MagicMock(
            tp=MagicMock(world_size=1, rank=0),
            cp=MagicMock(world_size=1, rank=0),
        )
        config = MagicMock()
        config.head_dim = 64
        config.num_attention_heads = 8
        config.num_key_value_heads = 8
        config.softmax_scale = None
        config.use_bias = False
        config.output_layer_init_method = MagicMock()
        config.recompute_granularity = None
        config.recompute_modules = None
        config.tensor_model_parallel_size = 1

        spec = SelfAttentionSublayersSpec()
        attn = Attention(
            config=config,
            sublayers_spec=spec,
            layer_number=3,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        self.assertEqual(attn.layer_number, 3)

    @patch("paddlefleet.transformer.attention.build_spec_layer")
    @patch("paddlefleet.transformer.attention.get_pg_size", return_value=1)
    @patch(
        "paddlefleet.transformer.attention.ProcessGroupCollection.use_mpu_process_groups"
    )
    def test_attention_sets_attention_type(
        self, mock_pg, mock_size, mock_build
    ):
        """Attention should set attention_type from constructor."""
        mock_pg.return_value = MagicMock(
            tp=MagicMock(world_size=1, rank=0),
            cp=MagicMock(world_size=1, rank=0),
        )
        config = MagicMock()
        config.head_dim = 64
        config.num_attention_heads = 8
        config.num_key_value_heads = 8
        config.softmax_scale = None
        config.use_bias = False
        config.output_layer_init_method = MagicMock()
        config.recompute_granularity = None
        config.recompute_modules = None
        config.tensor_model_parallel_size = 1

        spec = SelfAttentionSublayersSpec()
        attn = Attention(
            config=config,
            sublayers_spec=spec,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="cross",
        )
        self.assertEqual(attn.attention_type, "cross")


class TestAttentionSetForRecomputeInputLayernorm(unittest.TestCase):
    """Tests for Attention.set_for_recompute_input_layernorm."""

    def test_raises_not_implemented(self):
        """set_for_recompute_input_layernorm should raise NotImplementedError."""
        with patch.object(Attention, "__init__", lambda self, *a, **kw: None):
            attn = Attention.__new__(Attention)
            with self.assertRaises(NotImplementedError):
                attn.set_for_recompute_input_layernorm()


class TestCrossAttentionSublayersSpecDefaults(unittest.TestCase):
    """Tests for CrossAttentionSublayersSpec default values."""

    def test_default_values_are_none(self):
        """All fields of CrossAttentionSublayersSpec should default to None."""
        spec = CrossAttentionSublayersSpec()
        self.assertIsNone(spec.linear_q)
        self.assertIsNone(spec.linear_kv)
        self.assertIsNone(spec.core_attention)
        self.assertIsNone(spec.o_proj)


class TestCrossAttentionInitValidation(unittest.TestCase):
    """Tests for CrossAttention initialization validation."""

    @patch("paddlefleet.transformer.attention.build_spec_layer")
    @patch("paddlefleet.transformer.attention.get_pg_size", return_value=1)
    @patch(
        "paddlefleet.transformer.attention.ProcessGroupCollection.use_mpu_process_groups"
    )
    def test_cross_attention_rejects_gqa(self, mock_pg, mock_size, mock_build):
        """CrossAttention should reject group query attention."""
        mock_pg.return_value = MagicMock(
            tp=MagicMock(world_size=1, rank=0),
            cp=MagicMock(world_size=1, rank=0),
        )
        config = MagicMock()
        config.head_dim = 64
        config.num_attention_heads = 8
        config.num_key_value_heads = (
            4  # GQA: different from num_attention_heads
        )
        config.softmax_scale = None
        config.use_bias = False
        config.output_layer_init_method = MagicMock()
        config.recompute_granularity = None
        config.recompute_modules = None
        config.tensor_model_parallel_size = 1

        spec = CrossAttentionSublayersSpec()
        with self.assertRaises(ValueError):
            CrossAttention(
                config=config,
                sublayers_spec=spec,
                layer_number=1,
                attn_mask_type=AttnMaskType.padding,
            )


class TestSelfAttentionBackwardDW(unittest.TestCase):
    """Tests for SelfAttention.backward_dw."""

    def test_backward_dw_calls_qkv_and_output(self):
        """backward_dw should call _backward_qkv_proj and _backward_output_proj."""
        with patch.object(
            SelfAttention, "__init__", lambda self, *a, **kw: None
        ):
            attn = SelfAttention.__new__(SelfAttention)
            attn.qkv_proj = MagicMock()
            attn.o_proj = MagicMock()
            attn.backward_dw()
            attn.qkv_proj.backward_dw.assert_called_once()
            attn.o_proj.backward_dw.assert_called_once()


class TestSelfAttentionGetQKVSplitQKVFalse(unittest.TestCase):
    """Tests for SelfAttention.get_query_key_value_tensors with split_qkv=False."""

    @patch("paddlefleet.transformer.attention.build_spec_layer")
    @patch("paddlefleet.transformer.attention.get_pg_size", return_value=1)
    @patch(
        "paddlefleet.transformer.attention.ProcessGroupCollection.use_mpu_process_groups"
    )
    def test_returns_unsplit_mixed_qkv(self, mock_pg, mock_size, mock_build):
        """get_query_key_value_tensors with split_qkv=False returns unsplit tensor."""
        mock_pg.return_value = MagicMock(
            tp=MagicMock(world_size=1, rank=0),
            cp=MagicMock(world_size=1, rank=0),
        )
        config = MagicMock()
        config.head_dim = 64
        config.num_attention_heads = 8
        config.num_key_value_heads = 8
        config.softmax_scale = None
        config.use_bias = False
        config.output_layer_init_method = MagicMock()
        config.recompute_granularity = None
        config.recompute_modules = None
        config.tensor_model_parallel_size = 1
        config.gated_attention = False
        config.qk_norm_type = "per_head"
        config.rms_norm_eps = 1e-5

        spec = SelfAttentionSublayersSpec()
        attn = SelfAttention(
            config=config,
            sublayers_spec=spec,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )
        # Mock qkv_proj to return a tensor of the right shape
        b, s, h = 2, 4, 64
        total_dim = (8 + 2 * 8) * 64  # q + 2*kv
        mixed_qkv = paddle.randn([b, s, total_dim])
        attn.qkv_proj = MagicMock(return_value=(mixed_qkv, None))
        attn.num_attention_heads_per_partition = 8
        attn.num_query_groups_per_partition = 8
        attn.hidden_size_per_attention_head = 64
        attn.gated_attention = False
        attn.q_norm = None
        attn.k_norm = None
        attn.pg_collection = MagicMock(tp=MagicMock(world_size=1, rank=0))

        result = attn.get_query_key_value_tensors(
            paddle.randn([b, s, h]), split_qkv=False
        )
        self.assertEqual(len(result), 2)  # (mixed_qkv, split_arg_list)


if __name__ == "__main__":
    unittest.main()
