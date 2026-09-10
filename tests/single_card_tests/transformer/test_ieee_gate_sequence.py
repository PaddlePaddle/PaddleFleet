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

import unittest
from unittest.mock import patch

import paddle
import paddle.nn.functional as F

from paddlefleet.transformer.moe.moe_router import (
    FusedGateDetachMatmul,
    StandardMoERouter,
)
from paddlefleet.transformer.transformer_config import TransformerConfig


class TestIEEEGateSequence(unittest.TestCase):
    gate = FusedGateDetachMatmul

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
        y = self.gate.apply(x, w, False, True, shards, True)
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

    def test_master_and_moments_match_restore_round_lifecycle(self):
        for shards in (1, 2):
            with self.subTest(shards=shards):
                paddle.seed(513)
                initial = (
                    paddle.randn([16, 32]).cast("bfloat16").cast("float32")
                )
                master = paddle.create_parameter([16, 32], "float32")
                live = paddle.create_parameter([16, 32], "float32")
                master.set_value(initial)
                live.set_value(initial)
                saved_master = initial.clone()
                opts = [
                    paddle.optimizer.AdamW(
                        learning_rate=0.003,
                        parameters=[weight],
                        beta1=0.9,
                        beta2=0.95,
                        epsilon=1e-8,
                        weight_decay=0.1,
                    )
                    for weight in (master, live)
                ]
                for step in range(3):
                    x = paddle.randn([84, 32]).cast("bfloat16")
                    x.stop_gradient = False
                    dy = paddle.randn([84, 16])
                    y = self.gate.apply(x, master, False, True, shards, True)
                    y.backward(dy)
                    ys, dxs, dws = [], [], []
                    size = 84 // shards
                    for lo in range(0, 84, size):
                        xi = x[lo : lo + size].detach()
                        wi = live.detach().cast("bfloat16")
                        xi.stop_gradient = False
                        wi.stop_gradient = False
                        yi = F.linear(xi.cast("float32"), wi.cast("float32").T)
                        yi.backward(dy[lo : lo + size])
                        ys.append(yi.detach())
                        dxs.append(xi.grad)
                        dws.append(wi.grad.cast("float32"))
                    reference_grad = dws[0] if shards == 1 else dws[0] + dws[1]
                    self.equal(y, paddle.concat(ys))
                    self.equal(x.grad, paddle.concat(dxs))
                    self.equal(master.grad, reference_grad)
                    # Old callback restores the master immediately before AdamW.
                    live.set_value(saved_master)
                    live.grad = reference_grad
                    for opt in opts:
                        opt.step()
                        opt.clear_grad()
                    self.equal(master, live)
                    for key in (
                        "moment1",
                        "moment2",
                        "beta1_pow_acc",
                        "beta2_pow_acc",
                    ):
                        states = []
                        for opt in opts:
                            values = [
                                v
                                for k, v in opt.state_dict().items()
                                if key in k
                            ]
                            self.assertEqual(len(values), 1)
                            states.append(values[0])
                        self.equal(*states)
                    saved_master = live.detach().clone()
                    live.set_value(live.cast("bfloat16").cast("float32"))
                    self.equal(master.cast("bfloat16").cast("float32"), live)
                    if step == 0:
                        self.assertNotEqual(
                            master.numpy().tobytes(), live.numpy().tobytes()
                        )

    def test_existing_union_path_when_accuracy_compatible_is_off(self):
        x = paddle.ones([6, 8], dtype="bfloat16")
        w = paddle.ones([3, 8], dtype="float32")
        for use_fp32_master in (False, True):
            with (
                self.subTest(use_fp32_master=use_fp32_master),
                patch.object(F, "linear", wraps=F.linear) as linear,
            ):
                self.gate.apply(x, w, False, False, 2, use_fp32_master)
                self.assertEqual(linear.call_count, 1)

    def test_nondivisible_rows_use_existing_union_path(self):
        x = paddle.ones([5, 8], dtype="bfloat16")
        w = paddle.ones([3, 8], dtype="float32")
        with patch.object(F, "linear", wraps=F.linear) as linear:
            self.gate.apply(x, w, False, True, 2, True)
            self.assertEqual(linear.call_count, 1)

    def test_deferred_weight_gradient_retains_existing_forward(self):
        x = paddle.ones([6, 8], dtype="bfloat16")
        w = paddle.ones([3, 8], dtype="float32")
        with patch.object(F, "linear", wraps=F.linear) as linear:
            self.gate.apply(x, w, True, True, 2, True)
            self.assertEqual(linear.call_count, 1)

    def test_default_fp32_weight_is_not_rounded_or_sequence_split(self):
        paddle.seed(514)
        x = paddle.randn([12, 8]).cast("bfloat16")
        w = paddle.randn([4, 8])
        dy = paddle.randn([12, 4])
        x.stop_gradient = w.stop_gradient = False
        # The pre-existing public call must retain its FP32 parameter semantics,
        # even when sequence_shards is supplied by a newer caller.
        y = self.gate.apply(x, w, False, True, 2)
        y.backward(dy)
        self.equal(y, F.linear(x.detach().cast("float32"), w.detach().T))
        self.equal(x.grad, paddle.matmul(dy, w.detach()).cast("bfloat16"))
        self.equal(
            w.grad,
            paddle.matmul(dy, x.detach().cast("float32"), transpose_x=True),
        )
        self.assertNotEqual(
            w.numpy().tobytes(),
            w.cast("bfloat16").cast("float32").numpy().tobytes(),
        )

    def test_router_storage_policy_preserves_legacy_checkpoint_dtype(self):
        for enabled, master, expected in (
            (True, False, paddle.bfloat16),
            (True, True, paddle.float32),
            (False, False, paddle.float32),
            (False, True, paddle.float32),
        ):
            with self.subTest(enabled=enabled, master=master):
                config = TransformerConfig(
                    num_hidden_layers=1,
                    hidden_size=8,
                    num_attention_heads=1,
                    n_routed_experts=4,
                    num_experts_per_tok=2,
                    params_dtype="bfloat16",
                    use_accuracy_compatible=enabled,
                    moe_router_use_fp32_master=master,
                )
                router = StandardMoERouter(config)
                self.assertEqual(router.weight.dtype, expected)
                checkpoint_weight = (
                    paddle.arange(32).reshape([4, 8]).cast(expected)
                )
                router.weight.set_value(checkpoint_weight)
                self.equal(router.weight, checkpoint_weight)


if __name__ == "__main__":
    paddle.set_device("gpu:0")
    unittest.main()
