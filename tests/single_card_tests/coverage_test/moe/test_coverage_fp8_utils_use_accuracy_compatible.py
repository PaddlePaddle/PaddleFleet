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

import numpy as np
import paddle


def _make_node(use_accuracy_compatible):
    from paddlefleet.transformer.moe.fp8_utils import (
        ExpertsGroupGemmContiguousNode,
    )

    custom_map = MagicMock()
    custom_map.experts = [MagicMock()]
    return ExpertsGroupGemmContiguousNode(
        custom_map,
        use_fp8_mlp=False,
        moe_expert_fusion=False,
        use_accuracy_compatible=use_accuracy_compatible,
    )


class TestExpertsGroupGemmUseAccuracyCompatibleInit(unittest.TestCase):
    """The flag must be stored on the node and default to False."""

    def test_flag_stored_true(self):
        node = _make_node(True)
        self.assertTrue(node.use_accuracy_compatible)

    def test_flag_stored_false(self):
        node = _make_node(False)
        self.assertFalse(node.use_accuracy_compatible)

    def test_flag_default_false(self):
        from paddlefleet.transformer.moe.fp8_utils import (
            ExpertsGroupGemmContiguousNode,
        )

        custom_map = MagicMock()
        custom_map.experts = [MagicMock()]
        node = ExpertsGroupGemmContiguousNode(
            custom_map, use_fp8_mlp=False, moe_expert_fusion=False
        )
        self.assertFalse(node.use_accuracy_compatible)


class TestBwdGateUpInputBf16UseAccuracyCompatible(unittest.TestCase):
    """Cover the per-expert matmul vs F.linear branch in
    bwd_gate_up_input_bf16."""

    def test_branches_match(self):
        """matmul (compatible) and F.linear (default) must give identical
        dx for the split-group bf16 backward."""
        paddle.seed(0)
        do1 = paddle.randn([3, 4], dtype=paddle.bfloat16)
        # expert_w1[i] is [K, N]; the code uses expert_w1[i].T -> [N, K]
        expert_w1 = [
            paddle.randn([5, 4], dtype=paddle.bfloat16),
            paddle.randn([5, 4], dtype=paddle.bfloat16),
        ]

        node_compat = _make_node(True)
        node_compat.tokens_per_expert = [2, 1]
        node_default = _make_node(False)
        node_default.tokens_per_expert = [2, 1]

        dx_compat = node_compat.bwd_gate_up_input_bf16(do1, expert_w1)
        dx_default = node_default.bwd_gate_up_input_bf16(do1, expert_w1)

        self.assertEqual(dx_compat.shape, [3, 5])
        np.testing.assert_array_equal(
            dx_compat.astype("float32").numpy(),
            dx_default.astype("float32").numpy(),
        )

    def test_compatible_matches_reference_matmul(self):
        """The compatible branch must equal a manual per-expert matmul."""
        paddle.seed(1)
        do1 = paddle.randn([3, 4], dtype=paddle.bfloat16)
        expert_w1 = [
            paddle.randn([5, 4], dtype=paddle.bfloat16),
            paddle.randn([5, 4], dtype=paddle.bfloat16),
        ]

        node = _make_node(True)
        node.tokens_per_expert = [2, 1]
        dx = node.bwd_gate_up_input_bf16(do1, expert_w1)

        ref0 = paddle.matmul(do1[:2], expert_w1[0].T.contiguous())
        ref1 = paddle.matmul(do1[2:], expert_w1[1].T.contiguous())
        ref = paddle.concat([ref0, ref1], axis=0)
        np.testing.assert_array_equal(
            dx.astype("float32").numpy(), ref.astype("float32").numpy()
        )

    def test_skips_zero_token_expert(self):
        """An expert with zero tokens is skipped; remaining tokens still
        produce the correct dx in the compatible branch."""
        paddle.seed(2)
        do1 = paddle.randn([2, 4], dtype=paddle.bfloat16)
        expert_w1 = [
            paddle.randn([5, 4], dtype=paddle.bfloat16),
            paddle.randn([5, 4], dtype=paddle.bfloat16),
        ]

        node = _make_node(True)
        node.tokens_per_expert = [0, 2]
        dx = node.bwd_gate_up_input_bf16(do1, expert_w1)

        ref = paddle.matmul(do1, expert_w1[1].T.contiguous())
        np.testing.assert_array_equal(
            dx.astype("float32").numpy(), ref.astype("float32").numpy()
        )


class TestBwdDownInputBf16UseAccuracyCompatible(unittest.TestCase):
    """Cover the per-expert matmul vs F.linear branch in
    bwd_down_input_bf16. The downstream fused swiglu ops are mocked so we can
    isolate and assert the ``do2_s`` produced by the branch under test."""

    def _inputs(self):
        paddle.seed(3)
        total, hidden, inter = 3, 4, 2
        unzipped_grad = paddle.randn([total, hidden], dtype=paddle.bfloat16)
        # expert_w2[i] is [inter, hidden]; code uses expert_w2[i].T -> [hidden, inter]
        expert_w2 = [
            paddle.randn([inter, hidden], dtype=paddle.bfloat16),
            paddle.randn([inter, hidden], dtype=paddle.bfloat16),
        ]
        o1 = paddle.randn([total, 2 * inter], dtype=paddle.bfloat16)
        unzipped_probs = paddle.ones([total, 1], dtype=paddle.bfloat16)
        return unzipped_grad, expert_w2, o1, unzipped_probs

    def _run_capture_do2s(self, flag, unzipped_grad, expert_w2, o1, probs):
        """Run bwd_down_input_bf16 with the swiglu ops mocked, returning the
        ``do2_s`` tensor handed to fused_swiglu_scale_backward."""
        node = _make_node(flag)
        node.tokens_per_expert = [2, 1]

        captured = {}

        def fake_backward(x, scale, out_grad):
            captured["do2_s"] = out_grad
            return (
                paddle.zeros_like(o1),
                paddle.zeros_like(probs),
            )

        with (
            patch(
                "paddlefleet.transformer.moe.fp8_utils.fused_swiglu_scale_forward",
                MagicMock(return_value=paddle.zeros([o1.shape[0], 2])),
            ),
            patch(
                "paddlefleet.transformer.moe.fp8_utils.fused_swiglu_scale_backward",
                MagicMock(side_effect=fake_backward),
            ),
        ):
            node.bwd_down_input_bf16(expert_w2, unzipped_grad, o1, probs)
        return captured["do2_s"]

    def test_do2s_branches_match(self):
        """matmul (compatible) and F.linear (default) must give identical
        do2_s for the split-group bf16 backward."""
        unzipped_grad, expert_w2, o1, probs = self._inputs()

        do2s_compat = self._run_capture_do2s(
            True, unzipped_grad, expert_w2, o1, probs
        )
        do2s_default = self._run_capture_do2s(
            False, unzipped_grad, expert_w2, o1, probs
        )

        self.assertEqual(do2s_compat.shape, [3, 2])
        np.testing.assert_array_equal(
            do2s_compat.astype("float32").numpy(),
            do2s_default.astype("float32").numpy(),
        )

    def test_do2s_matches_reference_matmul(self):
        """The compatible branch must equal a manual per-expert matmul."""
        unzipped_grad, expert_w2, o1, probs = self._inputs()

        do2s = self._run_capture_do2s(True, unzipped_grad, expert_w2, o1, probs)

        ref0 = paddle.matmul(unzipped_grad[:2], expert_w2[0].T.contiguous())
        ref1 = paddle.matmul(unzipped_grad[2:], expert_w2[1].T.contiguous())
        ref = paddle.concat([ref0, ref1], axis=0)
        np.testing.assert_array_equal(
            do2s.astype("float32").numpy(), ref.astype("float32").numpy()
        )


if __name__ == "__main__":
    unittest.main()
