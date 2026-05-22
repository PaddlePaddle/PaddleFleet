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
import unittest
from unittest.mock import MagicMock, patch

import paddle


class TestMIndicesChanges(unittest.TestCase):
    """Tests covering the m_indices / tokens_per_expert_indices changes in fp8_utils.py.

    Changes covered:
    1. fwd_gate_up: generates tokens_per_expert_indices when moe_deep_gemm=True
    2. fwd_gate_up_bf16: uses tokens_per_expert_indices in deep_gemm path (moe_expert_fusion+moe_deep_gemm)
    3. fwd_gate_up_fp8: generates m_indices when moe_expert_fusion=True
    4. fwd_down_bf16: uses tokens_per_expert_indices in deep_gemm path
    5. bwd_down_input_bf16: uses tokens_per_expert_indices in deep_gemm path
    6. bwd_gate_up_input_bf16: uses tokens_per_expert_indices in deep_gemm path
    7. subbatch backward: generates tokens_per_expert_indices when moe_deep_gemm=True
    """

    def _make_node(
        self, moe_deep_gemm=False, moe_expert_fusion=False, use_fp8_mlp=False
    ):
        """Helper to create an ExpertsGroupGemmContiguousNode."""
        from paddlefleet.transformer.moe.fp8_utils import (
            ExpertsGroupGemmContiguousNode,
        )

        custom_map = MagicMock()
        custom_map.experts = [MagicMock(), MagicMock(), MagicMock()]
        custom_map.grouped_gemm_experts = MagicMock()
        node = ExpertsGroupGemmContiguousNode(
            custom_map,
            use_fp8_mlp=use_fp8_mlp,
            moe_deep_gemm=moe_deep_gemm,
            moe_expert_fusion=moe_expert_fusion,
        )
        return node

    def test_fwd_gate_up_generates_indices_with_deep_gemm(self):
        """Test fwd_gate_up generates tokens_per_expert_indices when moe_deep_gemm=True."""
        node = self._make_node(moe_deep_gemm=True, moe_expert_fusion=True)
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

        self.assertIsNotNone(node.tokens_per_expert_indices)
        expected = paddle.to_tensor([0, 1, 1], dtype="int32")
        self.assertTrue(
            paddle.equal_all(node.tokens_per_expert_indices, expected)
        )

    def test_fwd_gate_up_no_indices_without_deep_gemm(self):
        """Test fwd_gate_up does not generate tokens_per_expert_indices when moe_deep_gemm=False."""
        node = self._make_node(moe_deep_gemm=False, moe_expert_fusion=True)
        tokens_per_expert = [2, 1, 0]
        x = paddle.randn([3, 8], dtype="bfloat16")
        expert_w1 = paddle.randn([3, 8, 16], dtype="bfloat16")

        node.fwd_gate_up(
            x, expert_w1, num_expert=3, tokens_per_expert=tokens_per_expert
        )

        self.assertFalse(
            hasattr(node, "tokens_per_expert_indices")
            and node.tokens_per_expert_indices is not None
        )

    def test_fwd_gate_up_bf16_deep_gemm_path(self):
        """Test fwd_gate_up_bf16 calls deep_gemm with tokens_per_expert_indices."""
        node = self._make_node(moe_deep_gemm=True, moe_expert_fusion=True)
        tokens_per_expert = [2, 1, 0]
        x = paddle.randn([3, 8], dtype="bfloat16")
        expert_w1 = paddle.randn([3, 8, 16], dtype="bfloat16")

        with patch(
            "paddlefleet.transformer.moe.fp8_utils.paddlefleet_deep_gemm"
        ) as mock_dg:
            mock_dg.m_grouped_bf16_gemm_nn_contiguous = MagicMock()
            node.fwd_gate_up(
                x, expert_w1, num_expert=3, tokens_per_expert=tokens_per_expert
            )

        mock_dg.m_grouped_bf16_gemm_nn_contiguous.assert_called_once()

    def test_fwd_gate_up_bf16_split_path(self):
        """Test fwd_gate_up_bf16 uses per-expert F.linear when moe_expert_fusion=False."""
        node = self._make_node(moe_deep_gemm=False, moe_expert_fusion=False)
        tokens_per_expert = [2, 1, 3]
        x = paddle.randn([6, 8], dtype="float32")
        # Split path: expert_w1 is a list, F.linear does x @ weight so weight shape [in, out]
        expert_w1 = [paddle.randn([8, 16], dtype="float32") for _ in range(3)]

        node.fwd_gate_up(
            x, expert_w1, num_expert=3, tokens_per_expert=tokens_per_expert
        )

        # Should complete without error, m_indices/tokens_per_expert_indices not set
        self.assertIsNone(node.m_indices)

    def test_fwd_gate_up_fp8_generates_m_indices(self):
        """Test fwd_gate_up_fp8 generates m_indices when moe_expert_fusion=True."""
        from paddlefleet.transformer.moe.fp8_utils import (
            ExpertsGroupGemmContiguousNode,
        )

        custom_map = MagicMock()
        custom_map.grouped_gemm_experts = MagicMock()
        custom_map.grouped_gemm_experts.weight1 = MagicMock(spec=[])
        node = ExpertsGroupGemmContiguousNode(
            custom_map,
            use_fp8_mlp=True,
            moe_deep_gemm=True,
            moe_expert_fusion=True,
        )
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
        expected = paddle.to_tensor([0, 0, 0, 1, 1], dtype="int32")
        self.assertTrue(paddle.equal_all(node.m_indices, expected))

    def test_fwd_down_bf16_deep_gemm_path(self):
        """Test fwd_down_bf16 calls deep_gemm with tokens_per_expert_indices."""
        node = self._make_node(moe_deep_gemm=True, moe_expert_fusion=True)
        tokens_per_expert = [2, 1, 0]
        node.tokens_per_expert = tokens_per_expert
        node.tokens_per_expert_indices = node.gen_m_indices(tokens_per_expert)

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

        mock_dg.m_grouped_bf16_gemm_nn_contiguous.assert_called_once()

    def test_bwd_down_input_bf16_deep_gemm_path(self):
        """Test bwd_down_input_bf16 calls deep_gemm with tokens_per_expert_indices."""
        node = self._make_node(moe_deep_gemm=True, moe_expert_fusion=True)
        tokens_per_expert = [2, 1, 0]
        node.tokens_per_expert = tokens_per_expert
        node.tokens_per_expert_indices = node.gen_m_indices(tokens_per_expert)

        unzipped_grad = paddle.randn([3, 8], dtype="bfloat16")
        expert_w2 = paddle.randn([3, 16, 8], dtype="bfloat16")
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
            mock_fwd.return_value = paddle.randn([3, 4], dtype="bfloat16")
            mock_bwd.return_value = (
                paddle.randn([3, 16], dtype="bfloat16"),
                paddle.randn([3, 1], dtype="bfloat16"),
            )
            node.bwd_down_input_bf16(
                expert_w2, unzipped_grad, o1, unzipped_probs
            )

        mock_dg.m_grouped_bf16_gemm_nt_contiguous.assert_called_once()

    def test_bwd_gate_up_input_bf16_deep_gemm_path(self):
        """Test bwd_gate_up_input_bf16 calls deep_gemm with tokens_per_expert_indices."""
        node = self._make_node(moe_deep_gemm=True, moe_expert_fusion=True)
        tokens_per_expert = [2, 1, 0]
        node.tokens_per_expert = tokens_per_expert
        node.tokens_per_expert_indices = node.gen_m_indices(tokens_per_expert)

        do1 = paddle.randn([3, 16], dtype="bfloat16")
        expert_w1 = paddle.randn([3, 8, 16], dtype="bfloat16")

        with patch(
            "paddlefleet.transformer.moe.fp8_utils.paddlefleet_deep_gemm"
        ) as mock_dg:
            mock_dg.m_grouped_bf16_gemm_nt_contiguous = MagicMock()
            node.bwd_gate_up_input_bf16(do1, expert_w1)

        mock_dg.m_grouped_bf16_gemm_nt_contiguous.assert_called_once()

    def test_subbatch_backward_generates_indices(self):
        """Test subbatch backward generates tokens_per_expert_indices when moe_deep_gemm=True."""
        from paddlefleet.transformer.moe.fp8_utils import (
            ExpertsGroupGemmContiguousNode,
        )

        custom_map = MagicMock()
        custom_map.grouped_gemm_experts = MagicMock()
        node = ExpertsGroupGemmContiguousNode(
            custom_map,
            use_fp8_mlp=False,
            moe_deep_gemm=True,
            moe_subbatch_token_num_after_dispatch=128,
            moe_expert_fusion=True,
        )
        node.expert_id = 0
        total_rows = 256
        node.tokens_per_expert = [total_rows]
        node.tokens_per_expert_indices = node.gen_m_indices(
            node.tokens_per_expert
        )
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

        # After subbatch backward, tokens_per_expert_indices should be restored
        self.assertIsNotNone(node.tokens_per_expert_indices)
        expected = paddle.to_tensor([0] * total_rows, dtype="int32")
        self.assertTrue(
            paddle.equal_all(node.tokens_per_expert_indices, expected)
        )


if __name__ == "__main__":
    unittest.main()
