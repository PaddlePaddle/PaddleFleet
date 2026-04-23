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

import paddle


class TestFP8Utils(unittest.TestCase):
    """Unit tests for fp8_utils module."""

    def test_has_config_with_valid_key(self):
        """Test has_config returns True when key exists and value is truthy."""
        from paddlefleet.transformer.moe.fp8_utils import has_config

        result = has_config({"key1": True, "key2": False}, "key1")
        self.assertTrue(result)

    def test_has_config_with_false_value(self):
        """Test has_config returns False when value is falsy."""
        from paddlefleet.transformer.moe.fp8_utils import has_config

        result = has_config({"key1": False}, "key1")
        self.assertFalse(result)

    def test_has_config_with_missing_key(self):
        """Test has_config returns False when key is missing."""
        from paddlefleet.transformer.moe.fp8_utils import has_config

        result = has_config({"key1": True}, "key2")
        self.assertFalse(result)

    def test_has_config_with_none_config(self):
        """Test has_config returns False when config is None."""
        from paddlefleet.transformer.moe.fp8_utils import has_config

        result = has_config(None, "key1")
        self.assertFalse(result)

    def test_fused_stack_quant_precomputed_fp8_weight(self):
        """Test fused_stack_quant with precomputed fp8 weight_stacked."""
        from paddlefleet.transformer.moe.fp8_utils import fused_stack_quant

        w1 = paddle.randn([64, 128], dtype=paddle.bfloat16)
        w1.fp8_weight_stacked = paddle.zeros(
            [128, 64], dtype=paddle.float8_e4m3fn
        )
        w1.fp8_scale_stacked = paddle.ones([1, 8], dtype=paddle.float32)
        w2 = paddle.randn([64, 128], dtype=paddle.bfloat16)
        weight_list = [w1, w2]
        w, scale = fused_stack_quant(weight_list, transpose=False)
        self.assertIsNotNone(w)
        self.assertIsNotNone(scale)

    def test_fused_stack_quant_precomputed_transpose(self):
        """Test fused_stack_quant cache-hit with precomputed transpose.

        fused_stack_quant enters cache path via hasattr(w[0], 'fp8_weight_stacked'),
        then _get_fp8_weight_and_scale checks fp8_weight_stacked_transpose for
        transpose=True. So both attributes must be set.
        """
        from paddlefleet.transformer.moe.fp8_utils import fused_stack_quant

        w1 = paddle.randn([64, 128], dtype=paddle.bfloat16)
        # fp8_weight_stacked is required to enter cache path
        w1.fp8_weight_stacked = paddle.zeros(
            [128, 64], dtype=paddle.float8_e4m3fn
        )
        w1.fp8_scale_stacked = paddle.ones([1, 8], dtype=paddle.float32)
        # fp8_weight_stacked_transpose is the actual transpose cache
        w1.fp8_weight_stacked_transpose = paddle.zeros(
            [128, 64], dtype=paddle.float8_e4m3fn
        )
        w1.fp8_scale_stacked_transpose = paddle.ones(
            [1, 8], dtype=paddle.float32
        )
        w2 = paddle.randn([64, 128], dtype=paddle.bfloat16)
        weight_list = [w1, w2]
        w, scale = fused_stack_quant(weight_list, transpose=True)
        # Should return the precomputed transpose cache directly
        self.assertIs(w, w1.fp8_weight_stacked_transpose)
        self.assertIs(scale, w1.fp8_scale_stacked_transpose)

    def test_get_fp8_weight_and_scale(self):
        """Test _get_fp8_weight_and_scale helper."""
        from paddlefleet.transformer.moe.fp8_utils import (
            _get_fp8_weight_and_scale,
        )

        weight = MagicMock()
        weight.fp8_weight_stacked = "w"
        weight.fp8_scale_stacked = "s"
        weight.fp8_weight_stacked_transpose = "wt"
        weight.fp8_scale_stacked_transpose = "st"

        w, s = _get_fp8_weight_and_scale(weight, transpose=False)
        self.assertEqual(w, "w")
        self.assertEqual(s, "s")

        w, s = _get_fp8_weight_and_scale(weight, transpose=True)
        self.assertEqual(w, "wt")
        self.assertEqual(s, "st")

    @patch(
        "paddlefleet.transformer.moe.fp8_utils.paddle.incubate.nn.functional.fp8_gemm_blockwise"
    )
    def test_kitchen_gemm_zero_input(self, mock_gemm):
        """Test kitchen_gemm with zero-sized input."""
        from paddlefleet.transformer.moe.fp8_utils import kitchen_gemm

        x_fp8 = paddle.zeros([0, 64], dtype=paddle.float8_e4m3fn)
        x_scale = paddle.ones([1, 8], dtype=paddle.float32)
        w_fp8 = paddle.zeros([128, 64], dtype=paddle.float8_e4m3fn)
        w_scale = paddle.ones([1, 8], dtype=paddle.float32)
        out = kitchen_gemm(
            x_fp8,
            x_scale,
            w_fp8,
            w_scale,
            is_a_1d_scaled=True,
            is_b_1d_scaled=True,
        )
        self.assertEqual(out.shape[0], 0)

    def test_fp8_align_constant(self):
        """Test FP8_ALIGN constant value."""
        from paddlefleet.transformer.moe.fp8_utils import FP8_ALIGN

        self.assertEqual(FP8_ALIGN, 128)

    def test_experts_group_gemm_node_init_basic(self):
        """Test ExpertsGroupGemmContiguousNode initialization."""
        from paddlefleet.transformer.moe.fp8_utils import (
            ExpertsGroupGemmContiguousNode,
        )

        custom_map = MagicMock()
        experts = [MagicMock() for _ in range(2)]
        for e in experts:
            e.up_gate_proj = MagicMock()
            e.up_gate_proj.weight = paddle.randn(
                [128, 64], dtype=paddle.bfloat16
            )
            e.down_proj = MagicMock()
            e.down_proj.weight = paddle.randn([64, 128], dtype=paddle.bfloat16)
        custom_map.experts = experts

        node = ExpertsGroupGemmContiguousNode(
            custom_map,
            use_fp8_mlp=False,
            moe_grouped_gemm=False,
        )
        self.assertIsNotNone(node)
        self.assertFalse(node.use_fp8_mlp)
        self.assertFalse(node.moe_grouped_gemm)

    def test_experts_group_gemm_node_cached_tensors(self):
        """Test cached_tensors returns correct list."""
        from paddlefleet.transformer.moe.fp8_utils import (
            ExpertsGroupGemmContiguousNode,
        )

        custom_map = MagicMock()
        custom_map.experts = [MagicMock(), MagicMock()]
        node = ExpertsGroupGemmContiguousNode(
            custom_map,
            use_fp8_mlp=False,
            moe_grouped_gemm=False,
        )
        cached = node.cached_tensors()
        self.assertEqual(len(cached), 6)
        for c in cached:
            self.assertIsNone(c)

    def test_experts_group_gemm_node_set_cached_tensors(self):
        """Test set_cached_tensors correctly stores values."""
        from paddlefleet.transformer.moe.fp8_utils import (
            ExpertsGroupGemmContiguousNode,
        )

        custom_map = MagicMock()
        custom_map.experts = [MagicMock(), MagicMock()]
        node = ExpertsGroupGemmContiguousNode(
            custom_map,
            use_fp8_mlp=False,
            moe_grouped_gemm=False,
        )
        values = [paddle.ones([2]) if i < 2 else None for i in range(6)]
        node.set_cached_tensors(values)
        self.assertIsNotNone(node.tokens_per_expert)
        self.assertIsNotNone(node.m_indices)
        self.assertIsNone(node.input)

    def test_experts_group_gemm_node_clear_cached_tensors(self):
        """Test clear_cached_tensors sets all to None."""
        from paddlefleet.transformer.moe.fp8_utils import (
            ExpertsGroupGemmContiguousNode,
        )

        custom_map = MagicMock()
        custom_map.experts = [MagicMock(), MagicMock()]
        node = ExpertsGroupGemmContiguousNode(
            custom_map,
            use_fp8_mlp=False,
            moe_grouped_gemm=False,
        )
        node.set_cached_tensors([paddle.ones([2])] * 6)
        node.clear_cached_tensors()
        cached = node.cached_tensors()
        for c in cached:
            self.assertIsNone(c)

    def test_experts_group_gemm_node_reset_state(self):
        """Test reset_state clears tokens_per_expert and m_indices."""
        from paddlefleet.transformer.moe.fp8_utils import (
            ExpertsGroupGemmContiguousNode,
        )

        custom_map = MagicMock()
        custom_map.experts = [MagicMock(), MagicMock()]
        node = ExpertsGroupGemmContiguousNode(
            custom_map,
            use_fp8_mlp=False,
            moe_grouped_gemm=False,
        )
        node.tokens_per_expert = [1, 2]
        node.m_indices = paddle.ones([3], dtype="int32")
        node.reset_state()
        self.assertIsNone(node.tokens_per_expert)
        self.assertIsNone(node.m_indices)

    def test_experts_group_gemm_node_gen_m_indices(self):
        """Test gen_m_indices generates correct indices."""
        from paddlefleet.transformer.moe.fp8_utils import (
            ExpertsGroupGemmContiguousNode,
        )

        custom_map = MagicMock()
        custom_map.experts = [MagicMock()] * 3
        node = ExpertsGroupGemmContiguousNode(
            custom_map,
            use_fp8_mlp=False,
            moe_grouped_gemm=False,
        )
        tokens_per_expert = [2, 0, 3]
        indices = node.gen_m_indices(tokens_per_expert)
        expected = paddle.to_tensor([0, 0, 2, 2, 2], dtype="int32")
        self.assertTrue(paddle.allclose(indices, expected))

    def test_experts_group_gemm_node_gen_m_indices_empty(self):
        """Test gen_m_indices with all zero tokens."""
        from paddlefleet.transformer.moe.fp8_utils import (
            ExpertsGroupGemmContiguousNode,
        )

        custom_map = MagicMock()
        custom_map.experts = [MagicMock()] * 2
        node = ExpertsGroupGemmContiguousNode(
            custom_map,
            use_fp8_mlp=False,
            moe_grouped_gemm=False,
        )
        indices = node.gen_m_indices([0, 0])
        self.assertEqual(indices.shape[0], 0)

    def test_experts_group_gemm_node_clear_activation_tensors(self):
        """Test clear_activation_tensors resets input/output references."""
        from paddlefleet.transformer.moe.fp8_utils import (
            ExpertsGroupGemmContiguousNode,
        )

        custom_map = MagicMock()
        custom_map.experts = [MagicMock(), MagicMock()]
        node = ExpertsGroupGemmContiguousNode(
            custom_map,
            use_fp8_mlp=False,
            moe_grouped_gemm=False,
        )
        node.input = paddle.ones([4, 8])
        node.input_fp8 = paddle.zeros([4, 8], dtype=paddle.float8_e4m3fn)
        node.input_scale = paddle.ones([1, 1], dtype=paddle.float32)
        node.o1 = paddle.ones([4, 16])
        node.clear_activation_tensors()
        self.assertIsNone(node.input)
        self.assertIsNone(node.input_fp8)
        self.assertIsNone(node.input_scale)
        self.assertIsNone(node.o1)

    def test_experts_group_gemm_node_expert_id_init(self):
        """Test node init with specific expert_id."""
        from paddlefleet.transformer.moe.fp8_utils import (
            ExpertsGroupGemmContiguousNode,
        )

        custom_map = MagicMock()
        custom_map.experts = [MagicMock(), MagicMock(), MagicMock()]
        node = ExpertsGroupGemmContiguousNode(
            custom_map,
            use_fp8_mlp=False,
            moe_grouped_gemm=False,
            expert_id=1,
        )
        self.assertEqual(len(node.experts), 1)

    def test_experts_group_gemm_node_subbatch_assertion(self):
        """Test subbatch token num must be positive and aligned."""
        from paddlefleet.transformer.moe.fp8_utils import (
            ExpertsGroupGemmContiguousNode,
        )

        custom_map = MagicMock()
        custom_map.experts = [MagicMock()]
        with self.assertRaises(AssertionError):
            ExpertsGroupGemmContiguousNode(
                custom_map,
                moe_subbatch_token_num_after_dispatch=-1,
                use_fp8_mlp=False,
                moe_grouped_gemm=False,
            )
        with self.assertRaises(AssertionError):
            ExpertsGroupGemmContiguousNode(
                custom_map,
                moe_subbatch_token_num_after_dispatch=127,
                use_fp8_mlp=False,
                moe_grouped_gemm=False,
            )

    def test_swiglu_fallback(self):
        """Test swiglu fallback function with split."""
        from paddlefleet.transformer.moe.fp8_utils import swiglu

        x = paddle.randn([4, 8], dtype=paddle.float32)
        result = swiglu(x, y=None)
        self.assertEqual(result.shape, [4, 4])

    def test_swiglu_fallback_with_y(self):
        """Test swiglu fallback function with y provided."""
        from paddlefleet.transformer.moe.fp8_utils import swiglu

        x = paddle.randn([4, 4], dtype=paddle.float32)
        y = paddle.randn([4, 4], dtype=paddle.float32)
        result = swiglu(x, y=y)
        self.assertEqual(result.shape, [4, 4])


if __name__ == "__main__":
    unittest.main()
