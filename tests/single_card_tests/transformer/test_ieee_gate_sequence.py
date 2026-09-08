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

"""Exercise the production gate PyLayer against independent local graphs."""

import ast
import unittest
from pathlib import Path
from unittest.mock import patch

import paddle
import paddle.nn.functional as F


class TestIEEEGateSequence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = (
            Path(__file__).resolve().parents[3]
            / "src/paddlefleet/transformer/moe/moe_router.py"
        )
        tree = ast.parse(source.read_text())
        node = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "FusedGateDetachMatmul"
        )
        cls.ns = {"paddle": paddle, "F": F, "ieee_kernel_enabled": lambda: True}
        exec(
            compile(
                ast.Module(body=[node], type_ignores=[]), str(source), "exec"
            ),
            cls.ns,
        )
        cls.gate = cls.ns["FusedGateDetachMatmul"]

    def equal(self, a, b):
        self.assertEqual(a.shape, b.shape)
        self.assertEqual(
            a.cast("float32").numpy().tobytes(),
            b.cast("float32").numpy().tobytes(),
        )

    def test_local_graphs_preserve_forward_dx_and_local_wgrad_rounding(self):
        for shards in (1, 2):
            with self.subTest(shards=shards):
                self.check_local_graphs(shards)

    def check_local_graphs(self, shards):
        paddle.seed(512)
        x = paddle.randn([84, 32], dtype="float32").cast("bfloat16")
        w = (
            paddle.randn([16, 32], dtype="float32")
            .cast("bfloat16")
            .cast("float32")
        )
        dy = paddle.randn([84, 16], dtype="float32")
        x.stop_gradient = False
        w.stop_gradient = False
        y = self.gate.apply(x, w, False, True, shards)
        y.backward(dy)
        ys, dxs, dws = [], [], []
        size = 84 // shards
        for lo in range(0, 84, size):
            hi = lo + size
            xi = x[lo:hi].detach()
            wi = w.detach().cast("bfloat16")
            xi.stop_gradient = False
            wi.stop_gradient = False
            yi = F.linear(xi.cast("float32"), wi.cast("float32").T)
            yi.backward(dy[lo:hi])
            ys.append(yi.detach())
            dxs.append(xi.grad)
            dws.append(wi.grad.cast("float32"))
        self.equal(y, paddle.concat(ys, axis=0))
        self.equal(x.grad, paddle.concat(dxs, axis=0))
        self.equal(w.grad, dws[0] if shards == 1 else dws[0] + dws[1])
        union = (
            paddle.matmul(dy, x.detach().cast("float32"), transpose_x=True)
            .cast("bfloat16")
            .cast("float32")
        )
        if shards > 1:
            self.assertNotEqual(
                w.grad.numpy().tobytes(), union.numpy().tobytes()
            )

    def test_existing_union_path_when_ieee_or_uac_is_off(self):
        x = paddle.ones([6, 8], dtype="bfloat16")
        w = paddle.ones([3, 8], dtype="float32")
        for ieee, uac in [(False, True), (True, False)]:
            with (
                patch.dict(self.ns, ieee_kernel_enabled=lambda: ieee),
                patch.object(F, "linear", wraps=F.linear) as linear,
            ):
                self.gate.apply(x, w, False, uac, 2)
                self.assertEqual(linear.call_count, 1)

    def test_nondivisible_rows_use_existing_union_path(self):
        x = paddle.ones([5, 8], dtype="bfloat16")
        w = paddle.ones([3, 8], dtype="float32")
        with patch.object(F, "linear", wraps=F.linear) as linear:
            self.gate.apply(x, w, False, True, 2)
            self.assertEqual(linear.call_count, 1)

    def test_deferred_weight_gradient_retains_existing_forward(self):
        x = paddle.ones([6, 8], dtype="bfloat16")
        w = paddle.ones([3, 8], dtype="float32")
        with patch.object(F, "linear", wraps=F.linear) as linear:
            self.gate.apply(x, w, True, True, 2)
            self.assertEqual(linear.call_count, 1)


if __name__ == "__main__":
    paddle.set_device("gpu:0")
    unittest.main()
