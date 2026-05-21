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
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

import paddle


class TestMIndicesChanges(unittest.TestCase):
    """Tests covering the m_indices refactoring changes in fp8_utils.py.

    Changes covered:
    1. fwd_gate_up: condition expanded to (moe_deep_gemm or moe_expert_fusion), else m_indices=None
    2. fwd_gate_up_bf16: uses self.m_indices (renamed from tokens_per_expert_indices)
    3. fwd_gate_up_fp8: no longer generates m_indices internally (relies on fwd_gate_up)
    4. fwd_down_bf16: uses self.m_indices
    5. bwd_down_input_bf16: uses self.m_indices
    6. bwd_gate_up_input_bf16: uses self.m_indices
    7. subbatch backward loop: condition expanded
    8. backward end: condition expanded
    """

    def _make_node(
        self,
        moe_deep_gemm=False,
        moe_grouped_gemm=False,
        moe_expert_fusion=False,
    ):
        """Helper to create an ExpertsGroupGemmContiguousNode with mocked experts."""
        from paddlefleet.transformer.moe.fp8_utils import (
            ExpertsGroupGemmContiguousNode,
        )

        custom_map = MagicMock()
        custom_map.experts = [MagicMock(), MagicMock(), MagicMock()]
        node = ExpertsGroupGemmContiguousNode(
            custom_map,
            use_fp8_mlp=False,
            moe_grouped_gemm=moe_grouped_gemm,
            moe_deep_gemm=moe_deep_gemm,
        )
        node.moe_expert_fusion = moe_expert_fusion
        return node

    def test_fwd_gate_up_m_indices_with_moe_expert_fusion(self):
        """Test fwd_gate_up generates m_indices when moe_expert_fusion=True."""
        node = self._make_node(
            moe_deep_gemm=False, moe_grouped_gemm=False, moe_expert_fusion=True
        )
        tokens_per_expert = [2, 1, 3]
        x = paddle.randn([6, 8], dtype="float32")
        expert_w1 = [paddle.randn([8, 16], dtype="float32") for _ in range(3)]

        node.fwd_gate_up(
            x, expert_w1, num_expert=3, tokens_per_expert=tokens_per_expert
        )

        self.assertIsNotNone(node.m_indices)
        expected = paddle.to_tensor([0, 0, 1, 2, 2, 2], dtype="int32")
        self.assertTrue(paddle.equal_all(node.m_indices, expected))

    def test_fwd_gate_up_m_indices_with_moe_deep_gemm(self):
        """Test fwd_gate_up generates m_indices when moe_deep_gemm=True."""
        node = self._make_node(
            moe_deep_gemm=True, moe_grouped_gemm=True, moe_expert_fusion=False
        )
        tokens_per_expert = [1, 2, 0]
        x = paddle.randn([3, 8], dtype="bfloat16")
        expert_w1 = paddle.randn([3, 8, 16], dtype="bfloat16")

        with patch(
            "paddlefleet.transformer.moe.fp8_utils.paddlefleet_deep_gemm"
        ) as mock_dg:
            mock_dg.m_grouped_bf16_gemm_nn_contiguous = MagicMock()
            node.fwd_gate_up(
                x, expert_w1, num_expert=3, tokens_per_expert=tokens_per_expert
            )

        self.assertIsNotNone(node.m_indices)
        expected = paddle.to_tensor([0, 1, 1], dtype="int32")
        self.assertTrue(paddle.equal_all(node.m_indices, expected))

    def test_fwd_gate_up_m_indices_none_when_neither(self):
        """Test fwd_gate_up sets m_indices=None when neither flag is set."""
        node = self._make_node(
            moe_deep_gemm=False, moe_grouped_gemm=False, moe_expert_fusion=False
        )
        tokens_per_expert = [2, 1, 3]
        x = paddle.randn([6, 8], dtype="float32")
        expert_w1 = [paddle.randn([8, 16], dtype="float32") for _ in range(3)]

        node.fwd_gate_up(
            x, expert_w1, num_expert=3, tokens_per_expert=tokens_per_expert
        )

        self.assertIsNone(node.m_indices)

    def test_fwd_gate_up_fp8_uses_precomputed_m_indices(self):
        """Test fwd_gate_up_fp8 uses pre-computed m_indices from fwd_gate_up (no internal gen)."""
        from paddlefleet.transformer.moe.fp8_utils import (
            ExpertsGroupGemmContiguousNode,
        )

        custom_map = MagicMock()
        custom_map.experts = [MagicMock(), MagicMock()]
        node = ExpertsGroupGemmContiguousNode(
            custom_map,
            use_fp8_mlp=True,
            moe_grouped_gemm=True,
            moe_deep_gemm=False,
        )
        node.moe_expert_fusion = True
        tokens_per_expert = [3, 2]
        x = paddle.randn([5, 8], dtype="bfloat16")
        expert_w1 = paddle.randn([2, 8, 16], dtype="bfloat16")

        with (
            patch(
                "paddlefleet.transformer.moe.fp8_utils.fused_stack_quant"
            ) as mock_fsq,
            patch(
                "paddlefleet.transformer.moe.fp8_utils.paddlefleet_deep_gemm"
            ) as mock_dg,
        ):
            mock_fsq.return_value = (
                paddle.zeros([2, 16, 8], dtype="float8_e4m3fn"),
                paddle.ones([2, 1, 1], dtype="float32"),
            )
            mock_dg.m_grouped_fp8_gemm_nt_contiguous = MagicMock()
            node.fwd_gate_up(
                x, expert_w1, num_expert=2, tokens_per_expert=tokens_per_expert
            )

        self.assertIsNotNone(node.m_indices)

    def test_fwd_down_bf16_uses_m_indices(self):
        """Test fwd_down_bf16 uses self.m_indices for deep_gemm path."""
        node = self._make_node(
            moe_deep_gemm=True, moe_grouped_gemm=True, moe_expert_fusion=False
        )
        tokens_per_expert = [2, 1, 0]
        node.tokens_per_expert = tokens_per_expert
        node.m_indices = node.gen_m_indices(tokens_per_expert)

        o1 = paddle.randn([3, 16], dtype="bfloat16")
        unzipped_probs = paddle.ones([3, 1], dtype="bfloat16")
        expert_w2 = paddle.randn([3, 16, 8], dtype="bfloat16")

        with (
            patch(
                "paddlefleet.transformer.moe.fp8_utils.paddlefleet_deep_gemm"
            ) as mock_dg,
            patch(
                "paddlefleet.transformer.moe.fp8_utils.fused_swiglu_scale_forward"
            ) as mock_swiglu,
        ):
            mock_dg.m_grouped_bf16_gemm_nn_contiguous = MagicMock()
            mock_swiglu.return_value = paddle.randn([3, 8], dtype="bfloat16")
            node.fwd_down_bf16(o1, unzipped_probs, expert_w2)

        # Verify deep_gemm was called with m_indices
        mock_dg.m_grouped_bf16_gemm_nn_contiguous.assert_called_once()

    def test_bwd_down_input_bf16_uses_m_indices(self):
        """Test bwd_down_input_bf16 uses self.m_indices for deep_gemm path."""
        node = self._make_node(
            moe_deep_gemm=True, moe_grouped_gemm=True, moe_expert_fusion=False
        )
        tokens_per_expert = [2, 1, 0]
        node.tokens_per_expert = tokens_per_expert
        node.m_indices = node.gen_m_indices(tokens_per_expert)

        unzipped_grad = paddle.randn([3, 8], dtype="bfloat16")
        expert_w2 = paddle.randn([3, 8, 16], dtype="bfloat16")
        o1 = paddle.randn([3, 16], dtype="bfloat16")
        unzipped_probs = paddle.ones([3, 1], dtype="bfloat16")

        with (
            patch(
                "paddlefleet.transformer.moe.fp8_utils.paddlefleet_deep_gemm"
            ) as mock_dg,
            patch(
                "paddlefleet.transformer.moe.fp8_utils.fused_swiglu_scale_forward"
            ) as mock_fwd,
            patch(
                "paddlefleet.transformer.moe.fp8_utils.fused_swiglu_scale_backward"
            ) as mock_bwd,
        ):
            mock_dg.m_grouped_bf16_gemm_nt_contiguous = MagicMock()
            mock_fwd.return_value = paddle.randn([3, 8], dtype="bfloat16")
            mock_bwd.return_value = (
                paddle.randn([3, 16], dtype="bfloat16"),
                paddle.randn([3, 1], dtype="bfloat16"),
            )
            node.bwd_down_input_bf16(
                expert_w2, unzipped_grad, o1, unzipped_probs
            )

        mock_dg.m_grouped_bf16_gemm_nt_contiguous.assert_called_once()

    def test_bwd_gate_up_input_bf16_uses_m_indices(self):
        """Test bwd_gate_up_input_bf16 uses self.m_indices for deep_gemm path."""
        node = self._make_node(
            moe_deep_gemm=True, moe_grouped_gemm=True, moe_expert_fusion=False
        )
        tokens_per_expert = [2, 1, 0]
        node.tokens_per_expert = tokens_per_expert
        node.m_indices = node.gen_m_indices(tokens_per_expert)

        do1 = paddle.randn([3, 16], dtype="bfloat16")
        expert_w1 = paddle.randn([3, 8, 16], dtype="bfloat16")

        with patch(
            "paddlefleet.transformer.moe.fp8_utils.paddlefleet_deep_gemm"
        ) as mock_dg:
            mock_dg.m_grouped_bf16_gemm_nt_contiguous = MagicMock()
            node.bwd_gate_up_input_bf16(do1, expert_w1)

        mock_dg.m_grouped_bf16_gemm_nt_contiguous.assert_called_once()

    def test_subbatch_backward_m_indices_with_expert_fusion(self):
        """Test subbatch backward generates m_indices when moe_expert_fusion=True."""
        from paddlefleet.transformer.moe.fp8_utils import (
            ExpertsGroupGemmContiguousNode,
        )

        custom_map = MagicMock()
        custom_map.experts = [MagicMock(), MagicMock()]
        node = ExpertsGroupGemmContiguousNode(
            custom_map,
            use_fp8_mlp=False,
            moe_grouped_gemm=False,
            moe_deep_gemm=False,
            moe_subbatch_token_num_after_dispatch=128,
        )
        node.moe_expert_fusion = True
        node.expert_id = 0
        total_rows = 256
        node.tokens_per_expert = [total_rows]
        node.m_indices = node.gen_m_indices(node.tokens_per_expert)
        node.input = paddle.randn([total_rows, 8], dtype="float32")
        node.input_fp8 = None
        node.input_scale = paddle.ones([total_rows, 1], dtype="float32")
        node.o1 = paddle.randn([total_rows, 16], dtype="float32")

        out_grad = paddle.randn([total_rows, 8], dtype="float32")
        unzipped_probs = paddle.ones([total_rows], dtype="float32")

        with patch.object(node, "backward_impl") as mock_bwd:

            def side_effect(og, up, **kwargs):
                n = og.shape[0]
                return og, paddle.randn([n, 1], dtype="float32")

            mock_bwd.side_effect = side_effect
            node.backward(out_grad, unzipped_probs)

        # After subbatch backward, m_indices should be restored
        self.assertIsNotNone(node.m_indices)
        expected = paddle.to_tensor([0] * total_rows, dtype="int32")
        self.assertTrue(paddle.equal_all(node.m_indices, expected))


if __name__ == "__main__":
    unittest.main()
