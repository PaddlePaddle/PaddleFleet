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

from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.transformer.attention import (
    CrossAttention,
    CrossAttentionSublayersSpec,
    SelfAttention,
    SelfAttentionSublayersSpec,
)
from paddlefleet.transformer.transformer_config import TransformerConfig


class FakeLinear(paddle.nn.Layer):
    def __init__(self, in_f, out_f, **kwargs):
        super().__init__()
        self.linear = paddle.nn.Linear(in_f, out_f)
        self.gather_output = kwargs.get("gather_output", False)
        self.input_is_parallel = kwargs.get("input_is_parallel", False)
        self.skip_bias_add = kwargs.get("skip_bias_add", False)
        self.is_expert = kwargs.get("is_expert", False)
        self.tp_group = kwargs.get("tp_group", None)

    def forward(self, x):
        return self.linear(x), self.linear.bias

    def backward_dw(self):
        pass


def _make_config(**overrides):
    defaults = {
        "hidden_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 32,
        "use_bias": False,
        "attention_bias": False,
        "sequence_parallel": False,
        "recompute_granularity": None,
        "recompute_modules": None,
        "apply_rope_fusion": False,
        "gated_attention": False,
        "use_qk_norm": False,
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


class TestSelfAttentionGatedAttention(unittest.TestCase):
    """Test SelfAttention with gated_attention enabled."""

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    @patch("paddlefleet.transformer.attention.build_spec_layer")
    def test_gated_attention_projection_size(self, mock_build, mock_pg):
        mock_pg_obj = MagicMock()
        mock_pg_obj.tp = MagicMock()
        mock_pg_obj.cp = MagicMock()
        mock_pg_obj.tp.world_size = 1
        mock_pg.return_value = mock_pg_obj
        mock_build.return_value = MagicMock()

        config = _make_config(gated_attention=True)
        spec = SelfAttentionSublayersSpec(qkv_proj=FakeLinear)
        attn = SelfAttention(
            config, spec, layer_number=1, pg_collection=mock_pg_obj
        )
        self.assertTrue(attn.gated_attention)

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    @patch("paddlefleet.transformer.attention.build_spec_layer")
    def test_get_qkv_returns_four_tensors_when_gated(self, mock_build, mock_pg):
        mock_pg_obj = MagicMock()
        mock_pg_obj.tp = MagicMock()
        mock_pg_obj.cp = MagicMock()
        mock_pg_obj.tp.world_size = 1
        mock_pg.return_value = mock_pg_obj

        config = _make_config(gated_attention=True)
        spec = SelfAttentionSublayersSpec(qkv_proj=FakeLinear)
        attn = SelfAttention(
            config, spec, layer_number=1, pg_collection=mock_pg_obj
        )
        self.assertIsNotNone(attn.qkv_proj)


class TestSelfAttentionQKNorm(unittest.TestCase):
    """Test QK normalization paths."""

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    @patch("paddlefleet.transformer.attention.build_spec_layer")
    def test_per_head_norm_with_tp(self, mock_build, mock_pg):
        mock_pg_obj = MagicMock()
        mock_pg_obj.tp = MagicMock()
        mock_pg_obj.cp = MagicMock()
        mock_pg_obj.tp.world_size = 2
        mock_pg.return_value = mock_pg_obj
        mock_build.return_value = MagicMock()

        config = _make_config(
            use_qk_norm=True,
            qk_norm_type="per_head",
            tensor_model_parallel_size=2,
        )
        spec = SelfAttentionSublayersSpec(
            qkv_proj=FakeLinear,
            q_norm=MagicMock(),
            k_norm=MagicMock(),
        )
        attn = SelfAttention(
            config, spec, layer_number=1, pg_collection=mock_pg_obj
        )
        self.assertIsNotNone(attn.q_norm)
        self.assertIsNotNone(attn.k_norm)

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    @patch("paddlefleet.transformer.attention.build_spec_layer")
    def test_per_layer_norm_hidden_size(self, mock_build, mock_pg):
        mock_pg_obj = MagicMock()
        mock_pg_obj.tp = MagicMock()
        mock_pg_obj.cp = MagicMock()
        mock_pg_obj.tp.world_size = 2
        mock_pg.return_value = mock_pg_obj
        mock_build.return_value = MagicMock()

        config = _make_config(
            use_qk_norm=True,
            qk_norm_type="per_layer",
            tensor_model_parallel_size=2,
        )
        spec = SelfAttentionSublayersSpec(
            qkv_proj=FakeLinear,
            q_norm=MagicMock(),
            k_norm=MagicMock(),
        )
        attn = SelfAttention(
            config, spec, layer_number=1, pg_collection=mock_pg_obj
        )
        # per_layer norm size should be full dimension
        self.assertIsNotNone(attn.q_norm)
        self.assertIsNotNone(attn.k_norm)


class TestCrossAttention(unittest.TestCase):
    """Test CrossAttention construction."""

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    @patch("paddlefleet.transformer.attention.build_spec_layer")
    def test_construction(self, mock_build, mock_pg):
        mock_pg_obj = MagicMock()
        mock_pg_obj.tp = MagicMock()
        mock_pg_obj.cp = MagicMock()
        mock_pg_obj.tp.world_size = 1
        mock_pg.return_value = mock_pg_obj
        mock_build.return_value = MagicMock()

        config = _make_config()
        spec = CrossAttentionSublayersSpec(
            linear_q=FakeLinear,
            linear_kv=FakeLinear,
        )
        attn = CrossAttention(
            config, spec, layer_number=1, pg_collection=mock_pg_obj
        )
        self.assertIsNotNone(attn.linear_q)
        self.assertIsNotNone(attn.linear_kv)

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    @patch("paddlefleet.transformer.attention.build_spec_layer")
    def test_gqa_raises(self, mock_build, mock_pg):
        mock_pg_obj = MagicMock()
        mock_pg_obj.tp = MagicMock()
        mock_pg_obj.cp = MagicMock()
        mock_pg_obj.tp.world_size = 1
        mock_pg.return_value = mock_pg_obj

        config = _make_config(num_key_value_heads=2)
        spec = CrossAttentionSublayersSpec(
            linear_q=FakeLinear,
            linear_kv=FakeLinear,
        )
        with self.assertRaises(ValueError):
            CrossAttention(
                config, spec, layer_number=1, pg_collection=mock_pg_obj
            )


class TestAttentionRecomputeFlags(unittest.TestCase):
    """Test recompute configuration."""

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    @patch("paddlefleet.transformer.attention.build_spec_layer")
    def test_selective_core_attn(self, mock_build, mock_pg):
        mock_pg_obj = MagicMock()
        mock_pg_obj.tp = MagicMock()
        mock_pg_obj.cp = MagicMock()
        mock_pg_obj.tp.world_size = 1
        mock_pg.return_value = mock_pg_obj
        mock_build.return_value = MagicMock()

        config = _make_config(
            recompute_granularity="selective",
            recompute_modules=["core_attn"],
        )
        spec = SelfAttentionSublayersSpec(qkv_proj=FakeLinear)
        attn = SelfAttention(
            config, spec, layer_number=1, pg_collection=mock_pg_obj
        )
        self.assertTrue(attn.recompute_core_attention)

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    @patch("paddlefleet.transformer.attention.build_spec_layer")
    def test_flash_attn_rr_flag(self, mock_build, mock_pg):
        mock_pg_obj = MagicMock()
        mock_pg_obj.tp = MagicMock()
        mock_pg_obj.cp = MagicMock()
        mock_pg_obj.tp.world_size = 1
        mock_pg.return_value = mock_pg_obj
        mock_build.return_value = MagicMock()

        config = _make_config(
            recompute_granularity="selective",
            recompute_modules=["flash_attn"],
        )
        spec = SelfAttentionSublayersSpec(qkv_proj=FakeLinear)
        attn = SelfAttention(
            config, spec, layer_number=1, pg_collection=mock_pg_obj
        )
        self.assertTrue(attn.use_rr_flash_attention)

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    @patch("paddlefleet.transformer.attention.build_spec_layer")
    def test_no_recompute(self, mock_build, mock_pg):
        mock_pg_obj = MagicMock()
        mock_pg_obj.tp = MagicMock()
        mock_pg_obj.cp = MagicMock()
        mock_pg_obj.tp.world_size = 1
        mock_pg.return_value = mock_pg_obj
        mock_build.return_value = MagicMock()

        config = _make_config()
        spec = SelfAttentionSublayersSpec(qkv_proj=FakeLinear)
        attn = SelfAttention(
            config, spec, layer_number=1, pg_collection=mock_pg_obj
        )
        self.assertFalse(attn.recompute_core_attention)
        self.assertFalse(attn.use_rr_flash_attention)


class TestAttentionSetForRecompute(unittest.TestCase):
    """Test set_for_recompute_input_layernorm."""

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    @patch("paddlefleet.transformer.attention.build_spec_layer")
    def test_raises_not_implemented(self, mock_build, mock_pg):
        mock_pg_obj = MagicMock()
        mock_pg_obj.tp = MagicMock()
        mock_pg_obj.cp = MagicMock()
        mock_pg_obj.tp.world_size = 1
        mock_pg.return_value = mock_pg_obj
        mock_build.return_value = MagicMock()

        config = _make_config()
        spec = SelfAttentionSublayersSpec(qkv_proj=FakeLinear)
        attn = SelfAttention(
            config, spec, layer_number=1, pg_collection=mock_pg_obj
        )
        with self.assertRaises(NotImplementedError):
            attn.set_for_recompute_input_layernorm()


class TestSelfAttentionProjectionSizes(unittest.TestCase):
    """Test projection size calculations."""

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    @patch("paddlefleet.transformer.attention.build_spec_layer")
    def test_query_projection_size(self, mock_build, mock_pg):
        mock_pg_obj = MagicMock()
        mock_pg_obj.tp = MagicMock()
        mock_pg_obj.cp = MagicMock()
        mock_pg_obj.tp.world_size = 1
        mock_pg.return_value = mock_pg_obj
        mock_build.return_value = MagicMock()

        config = _make_config(
            hidden_size=256, num_attention_heads=8, head_dim=32
        )
        spec = SelfAttentionSublayersSpec(qkv_proj=FakeLinear)
        attn = SelfAttention(
            config, spec, layer_number=1, pg_collection=mock_pg_obj
        )
        self.assertEqual(attn.query_projection_size, 256)

    @patch.object(ProcessGroupCollection, "use_mpu_process_groups")
    @patch("paddlefleet.transformer.attention.build_spec_layer")
    def test_kv_projection_size_gqa(self, mock_build, mock_pg):
        mock_pg_obj = MagicMock()
        mock_pg_obj.tp = MagicMock()
        mock_pg_obj.cp = MagicMock()
        mock_pg_obj.tp.world_size = 2
        mock_pg.return_value = mock_pg_obj
        mock_build.return_value = MagicMock()

        config = _make_config(
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=32,
            tensor_model_parallel_size=2,
        )
        spec = SelfAttentionSublayersSpec(qkv_proj=FakeLinear)
        attn = SelfAttention(
            config, spec, layer_number=1, pg_collection=mock_pg_obj
        )
        self.assertEqual(attn.kv_projection_size, 128)


if __name__ == "__main__":
    unittest.main()
