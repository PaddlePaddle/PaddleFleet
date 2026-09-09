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

"""Execute production blocks with tiny CPU adapters; no GPU accuracy claim."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def production_function(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if name == "backward":
        layer = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "FusedGateDetachMatmul"
        )
        functions = [
            node
            for node in layer.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        ]
    assert len(functions) == 1
    functions[0].decorator_list = []
    exec(
        compile(
            ast.Module(body=functions, type_ignores=[]),
            str(ROOT / path),
            "exec",
        ),
        namespace,
    )
    return namespace[name]


class Tensor:
    def __init__(self, data, dtype="float32", casts=()):
        self.data = np.asarray(data, dtype=np.float32)
        self.dtype = dtype
        self.shape = self.data.shape
        self.stop_gradient = False
        self.casts = casts

    def cast(self, dtype):
        # Record conversion order; BF16 device rounding is tested by real runs.
        return Tensor(self.data, dtype, (*self.casts, dtype))

    def _slice(self, start, end):
        return Tensor(self.data[start:end], self.dtype)

    def transpose(self, axes):
        return Tensor(self.data.transpose(axes), self.dtype, self.casts)

    def contiguous(self):
        return self


class TestGLM52WeightGradientContracts(unittest.TestCase):
    def adapter(self):
        calls = []

        def matmul(left, right, transpose_x=False):
            calls.append((left.dtype, right.dtype, transpose_x))
            return Tensor(
                (left.data.T if transpose_x else left.data) @ right.data
            )

        return SimpleNamespace(
            float32="float32", bfloat16="bfloat16", matmul=matmul
        ), calls

    def test_aligned_gate_roundtrip_matches_weight_storage_dtype(self):
        for enabled, dtype, expected in (
            (True, "float32", ("bfloat16", "float32", "float32")),
            (True, "bfloat16", ("bfloat16",)),
        ):
            with self.subTest(enabled=enabled, dtype=dtype):
                paddle, _ = self.adapter()
                x, weight, dy = (
                    Tensor([[1, 2]]),
                    Tensor([[3, 4]], dtype),
                    Tensor([[5]]),
                )
                ctx = SimpleNamespace(
                    saved_tensor=lambda: (x, weight, weight),
                    sequence_shards=1,
                    dtype="float32",
                    defer_dw=False,
                    use_accuracy_compatible=enabled,
                )
                backward = production_function(
                    "src/paddlefleet/transformer/moe/moe_router.py",
                    "backward",
                    {"paddle": paddle, "ieee_kernel_enabled": lambda: enabled},
                )
                _, gradient = backward(ctx, dy)
                self.assertEqual(gradient.casts, expected)

    def test_expert_fp32_orientation_and_empty_expert_cursor(self):
        paddle, calls = self.adapter()
        paddle.zeros = lambda shape, dtype: Tensor(np.zeros(shape), dtype)
        paddle.stack = lambda values, axis: Tensor(
            np.stack([value.data for value in values], axis=axis)
        )
        namespace = {
            "paddle": paddle,
            "ieee_kernel_enabled": lambda: True,
            "self": SimpleNamespace(
                use_accuracy_compatible=True,
                use_fp8_mlp=False,
                tokens_per_expert=[2, 0, 1],
            ),
            "x": Tensor([[1, 2], [3, 4], [5, 6]], "bfloat16"),
            "dy": Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], "bfloat16"),
            "weights": SimpleNamespace(shape=[3, 2, 3]),
        }
        gradient = production_function(
            "src/paddlefleet/transformer/moe/fp8_utils.py",
            "_batched_weight_grad",
            namespace,
        )()
        np.testing.assert_array_equal(
            gradient.data,
            [
                [[13, 17, 21], [18, 24, 30]],
                [[0, 0, 0], [0, 0, 0]],
                [[35, 40, 45], [42, 48, 54]],
            ],
        )
        self.assertEqual(calls, [("float32", "float32", True)] * 2)

    def test_expert_non_ieee_or_non_uac_keeps_native_batched_gemm(self):
        for compatible, enabled in ((False, True), (False, False)):
            with self.subTest(compatible=compatible, enabled=enabled):
                calls = []
                fallback = lambda *args, **kwargs: (
                    calls.append((args, kwargs)) or "native"
                )
                paddle = SimpleNamespace(
                    incubate=SimpleNamespace(
                        nn=SimpleNamespace(
                            functional=SimpleNamespace(batched_gemm=fallback)
                        )
                    )
                )
                namespace = {
                    "paddle": paddle,
                    "ieee_kernel_enabled": lambda: enabled,
                    "self": SimpleNamespace(
                        use_accuracy_compatible=compatible,
                        use_fp8_mlp=False,
                        tokens_per_expert=[1],
                    ),
                    "x": "x",
                    "dy": "dy",
                    "weights": None,
                }
                result = production_function(
                    "src/paddlefleet/transformer/moe/fp8_utils.py",
                    "_batched_weight_grad",
                    namespace,
                )()
                self.assertEqual(result, "native")
                self.assertEqual(
                    calls, [(("x", "dy", [1]), {"trans_lhs": True})]
                )


if __name__ == "__main__":
    unittest.main()
