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

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy
import paddle
import paddle.nn.functional as F


class TestIEEEGroupedBF16(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = (
            Path(__file__).resolve().parents[3]
            / "src/paddlefleet/transformer/moe/fp8_utils.py"
        )
        tree = ast.parse(source.read_text())
        node = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef)
            and n.name == "ExpertsGroupGemmContiguousNode"
        )
        names = {
            "fwd_gate_up_bf16",
            "fwd_down_bf16",
            "bwd_down_input_bf16",
            "bwd_gate_up_input_bf16",
        }
        methods = [
            n
            for n in node.body
            if isinstance(n, ast.FunctionDef) and n.name in names
        ]
        helpers = [
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef)
            and n.name
            in {
                "use_sequential_bf16_experts",
                "_sequential_expert_matmul",
                "moe_token_padding_alignment",
            }
        ]
        cls.ns = {
            "paddle": paddle,
            "F": F,
            "numpy": numpy,
            "FP8_ALIGN": 128,
            "ieee_kernel_enabled": lambda: True,
        }
        exec(
            compile(
                ast.Module(body=helpers + methods, type_ignores=[]),
                str(source),
                "exec",
            ),
            cls.ns,
        )

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
        self.assertEqual(a.shape, b.shape)
        self.assertEqual(a.dtype, b.dtype)
        self.assertEqual(
            a.cast("float32").numpy().tobytes(),
            b.cast("float32").numpy().tobytes(),
        )

    def test_gate_excludes_other_modes_and_padding_is_opt_in(self):
        options = {
            "use_accuracy_compatible": True,
            "moe_expert_fusion": True,
            "use_fp8_mlp": False,
            "moe_deep_gemm": False,
            "activation_type": "swiglu",
            "clamp_value": None,
        }
        gate = self.ns["use_sequential_bf16_experts"]
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
        with patch.dict(self.ns, ieee_kernel_enabled=lambda: False):
            self.assertTrue(gate(**options))
        alignment = self.ns["moe_token_padding_alignment"]
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
            o1 = self.ns["fwd_gate_up_bf16"](node, x, w1)
            output = self.ns["fwd_down_bf16"](node, o1, probs, w2)
            do1, o2, dp = self.ns["bwd_down_input_bf16"](
                node, w2, dy, o1, probs
            )
            dx = self.ns["bwd_gate_up_input_bf16"](node, do1, w1)
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
            self.assertIs(self.ns["fwd_gate_up_bf16"](node, x, w1), sentinel)
            grouped.assert_called_once_with(x, w1, node.tokens_per_expert)

    def test_unzip_rebuilds_expert_order_and_retains_disabled_output(self):
        source = (
            Path(__file__).resolve().parents[3]
            / "src/paddlefleet/transformer/moe/fusion_layer_utils.py"
        )
        tree = ast.parse(source.read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "UnZipNode"
        )
        namespace = {"paddle": paddle, "FP8_ALIGN": 128}
        exec(
            compile(
                ast.Module(body=[cls], type_ignores=[]), str(source), "exec"
            ),
            namespace,
        )
        hidden = paddle.to_tensor([[1, 2], [3, 4], [5, 6]], dtype="bfloat16")
        rowmap = paddle.to_tensor([[-1, 2], [0, -1], [1, 3]], dtype="int32")
        for enabled, fill, rows in [
            (True, True, 4),
            (True, True, 6),
            (False, True, 4),
            (True, False, 4),
        ]:
            node = namespace["UnZipNode"](None, sequential_bf16_experts=enabled)
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
        node = namespace["UnZipNode"](None, sequential_bf16_experts=True)
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
        o1 = self.ns["fwd_gate_up_bf16"](node, x, w1)
        self.assertEqual(o1.shape, [0, 64])
        dx = self.ns["bwd_gate_up_input_bf16"](node, o1, w1)
        self.assertEqual(dx.shape, [0, 32])
        self.assertEqual(dx.dtype, paddle.bfloat16)


if __name__ == "__main__":
    paddle.set_device("gpu:0")
    unittest.main()
