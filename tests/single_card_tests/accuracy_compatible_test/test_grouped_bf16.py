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

"""Check the scoped expert implementation on native BF16 tensors."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import paddle
import paddle.nn.functional as F

from paddlefleet.transformer.moe.fp8_utils import (
    ExpertsGroupGemmContiguousNode,
    moe_token_padding_alignment,
    use_sequential_bf16_experts,
)
from paddlefleet.transformer.moe.fusion_layer_utils import UnZipNode
from tests.single_card_tests.accuracy_compatible_test._assertions import (
    assert_bitwise_equal,
)


class TestAccuracyCompatibleGroupedBF16(unittest.TestCase):
    def node(self, counts):
        return SimpleNamespace(
            sequential_bf16_experts=True,
            tokens_per_expert=counts,
            use_fp8_mlp=False,
            moe_expert_fusion=True,
            moe_deep_gemm=False,
            use_accuracy_compatible=True,
            activation_type="swiglu",
            clamp_value=None,
            is_split_group_gemm=False,
        )

    def equal(self, a, b):
        assert_bitwise_equal(a, b)

    def test_gate_excludes_other_modes_and_padding_is_opt_in(self):
        options = {
            "use_accuracy_compatible": True,
            "moe_expert_fusion": True,
            "use_fp8_mlp": False,
            "moe_deep_gemm": False,
            "activation_type": "swiglu",
            "clamp_value": None,
        }
        gate = use_sequential_bf16_experts
        self.assertTrue(gate(**options))
        for key, value in [
            ("use_accuracy_compatible", False),
            ("moe_expert_fusion", False),
            ("use_fp8_mlp", True),
            ("moe_deep_gemm", True),
            ("activation_type", "geglu"),
            ("activation_type", "situ"),
            ("clamp_value", 0.0),
            ("clamp_value", 7.0),
        ]:
            with self.subTest(key=key, value=value):
                self.assertFalse(gate(**(options | {key: value})))
        alignment = moe_token_padding_alignment
        base = {
            "use_accuracy_compatible": True,
            "use_fp8_mlp": False,
            "moe_grouped_gemm": True,
        }
        self.assertEqual(alignment(**base), 128)
        self.assertEqual(alignment(**base, sequential_bf16_experts=True), 1)
        self.assertEqual(
            alignment(
                **(base | {"use_fp8_mlp": True}), sequential_bf16_experts=True
            ),
            128,
        )

    def test_forward_and_both_input_gradients_match_sequential_autograd(self):
        paddle.seed(73)
        node = self.node([2, 0, 3])
        x = paddle.randn([5, 32], dtype="float32").cast("bfloat16")
        w1 = paddle.randn([3, 32, 64], dtype="float32").cast("bfloat16")
        w2 = paddle.randn([3, 32, 32], dtype="float32").cast("bfloat16")
        probs = paddle.to_tensor(
            [0.123, 0.827, 0.343, 0.657, 0.721], dtype="float32"
        )
        dy = paddle.randn([5, 32], dtype="float32").cast("bfloat16")
        with paddle.no_grad():
            o1 = ExpertsGroupGemmContiguousNode.fwd_gate_up_bf16(node, x, w1)
            output = ExpertsGroupGemmContiguousNode.fwd_down_bf16(
                node, o1, probs, w2
            )
            do1, o2, dp = ExpertsGroupGemmContiguousNode.bwd_down_input_bf16(
                node, w2, dy, o1, probs
            )
            dx = ExpertsGroupGemmContiguousNode.bwd_gate_up_input_bf16(
                node, do1, w1
            )
        expected_y, expected_dx, expected_dp, expected_o2 = [], [], [], []
        for expert, lo, hi in [(0, 0, 2), (2, 2, 5)]:
            xi = x[lo:hi].detach()
            pi = probs[lo:hi].detach()
            xi.stop_gradient = False
            pi.stop_gradient = False
            hidden = paddle.matmul(
                xi, w1[expert].T.contiguous(), transpose_y=True
            )
            gate, up = paddle.chunk(hidden, 2, axis=-1)
            activated = (F.silu(gate) * (up + paddle.zeros_like(up))).cast(
                "bfloat16"
            )
            scaled = (activated.cast("float32") * pi.unsqueeze(-1)).cast(
                "bfloat16"
            )
            yi = paddle.matmul(
                scaled, w2[expert].T.contiguous(), transpose_y=True
            )
            yi.backward(dy[lo:hi])
            expected_y.append(yi.detach())
            expected_dx.append(xi.grad)
            expected_dp.append(pi.grad)
            expected_o2.append(scaled.detach())
        for got, parts in [
            (output, expected_y),
            (dx, expected_dx),
            (dp, expected_dp),
            (o2, expected_o2),
        ]:
            self.equal(got, paddle.concat(parts, axis=0))

    def test_disabled_path_calls_existing_batched_gemm(self):
        node = self.node([2, 0, 3])
        node.sequential_bf16_experts = False
        x = paddle.ones([5, 32], dtype="bfloat16")
        w1 = paddle.ones([3, 32, 64], dtype="bfloat16")
        sentinel = paddle.zeros([5, 64], dtype="bfloat16")
        with patch.object(
            paddle.incubate.nn.functional,
            "batched_gemm",
            Mock(return_value=sentinel),
        ) as grouped:
            self.assertIs(
                ExpertsGroupGemmContiguousNode.fwd_gate_up_bf16(node, x, w1),
                sentinel,
            )
            grouped.assert_called_once_with(x, w1, node.tokens_per_expert)

    def test_unzip_rebuilds_expert_order_and_retains_disabled_output(self):
        hidden = paddle.to_tensor([[1, 2], [3, 4], [5, 6]], dtype="bfloat16")
        rowmap = paddle.to_tensor([[-1, 2], [0, -1], [1, 3]], dtype="int32")
        for enabled, fill, rows in [
            (True, True, 4),
            (True, True, 6),
            (False, True, 4),
            (True, False, 4),
        ]:
            node = UnZipNode(None, sequential_bf16_experts=enabled)
            raw = paddle.full([rows, 2], -1, dtype="bfloat16")
            probs = paddle.ones([rows], dtype="float32")
            with patch.object(
                F,
                "moe_permute",
                Mock(return_value=(raw, rowmap, probs, paddle.empty([0]))),
            ):
                result = node.forward(
                    hidden,
                    None,
                    None,
                    2,
                    2,
                    [2, 2],
                    fill_output=fill,
                    padding_alignment=1,
                )
            self.assertIs(result[1], rowmap)
            self.assertIs(result[2], probs)
            if enabled and fill:
                expected = [[3, 4], [5, 6], [1, 2], [5, 6]] + [[0, 0]] * (
                    rows - 4
                )
                self.equal(
                    result[0], paddle.to_tensor(expected, dtype="bfloat16")
                )
            else:
                self.assertIs(result[0], raw)
        node = UnZipNode(None, sequential_bf16_experts=True)
        with (
            patch.object(
                F,
                "moe_permute",
                Mock(
                    return_value=(
                        paddle.empty([3, 2], dtype="bfloat16"),
                        rowmap,
                        probs,
                        paddle.empty([0]),
                    )
                ),
            ),
            self.assertRaisesRegex(RuntimeError, "valid rows"),
        ):
            node.forward(hidden, None, None, 2, 2, [2, 2], padding_alignment=1)

    def test_empty_forward_and_input_gradient_keep_shapes(self):
        node = self.node([0, 0, 0])
        x = paddle.empty([0, 32], dtype="bfloat16")
        w1 = paddle.ones([3, 32, 64], dtype="bfloat16")
        o1 = ExpertsGroupGemmContiguousNode.fwd_gate_up_bf16(node, x, w1)
        self.assertEqual(o1.shape, [0, 64])
        dx = ExpertsGroupGemmContiguousNode.bwd_gate_up_input_bf16(node, o1, w1)
        self.assertEqual(dx.shape, [0, 32])
        self.assertEqual(dx.dtype, paddle.bfloat16)


if __name__ == "__main__":
    paddle.set_device("gpu:0")
    unittest.main()
