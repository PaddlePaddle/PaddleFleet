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
"""Tests for the ``"hf"`` router pieces in ``transformer/moe/moe_router``.

Two places where the Megatron and HuggingFace references disagree:

* **Where the gate projection is promoted to FP32.** Megatron runs the projection
  itself in ``router_dtype=fp32``; ``Qwen3_5MoeTopKRouter`` keeps it in the
  activation dtype and only upcasts inside the softmax. The backward differs to
  match: the reference rounds the incoming gradient back to the activation dtype
  before its two GEMMs, because on its graph the FP32 softmax input is a *cast* of
  the activation-dtype logits.
* **The softmax itself.** ``HFBitexactSoftmax`` pins both halves -- an explicit
  max-subtract / exp / normalize forward, and the reference's
  ``grad*p - (grad*p).sum()*p`` epilogue -- because paddle's fused softmax and its
  grad each differ from torch by 1 ULP on a large fraction of elements.

``targets_hf`` is the only discriminator, and the target arrives as a parameter,
so the tests also check that ``True``/``"megatron"`` stay on the Megatron branch.
"""

import os
import sys
import unittest
from unittest.mock import patch

import numpy as np
import paddle

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

from paddlefleet.transformer.moe.moe_router import (
    FusedGateDetachMatmul,
    HFBitexactSoftmax,
)


class TestHFBitexactSoftmaxForward(unittest.TestCase):
    """The forward is a specific max-subtract / exp / normalize sequence."""

    def setUp(self):
        paddle.seed(20260908)
        self.logits = paddle.randn([8, 16], dtype=paddle.float32)

    def test_matches_explicit_sequence(self):
        """Bit-for-bit equal to the transcription, not to ``F.softmax``."""
        out = HFBitexactSoftmax.apply(self.logits)
        with paddle.amp.auto_cast(False):
            x = self.logits.astype(paddle.float32)
            exp = paddle.exp(x - x.max(axis=-1, keepdim=True))
            expected = exp / exp.sum(axis=-1, keepdim=True)
        np.testing.assert_array_equal(out.numpy(), expected.numpy())

    def test_rows_sum_to_one(self):
        out = HFBitexactSoftmax.apply(self.logits)
        np.testing.assert_allclose(
            out.sum(axis=-1).numpy(),
            np.ones(8, dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_output_is_fp32_for_bf16_logits(self):
        """The reference upcasts inside the softmax, so the probs are FP32."""
        out = HFBitexactSoftmax.apply(self.logits.astype(paddle.bfloat16))
        self.assertEqual(out.dtype, paddle.float32)

    def test_shift_invariance_from_the_max_subtract(self):
        """Adding a per-row constant must not change the probabilities."""
        shifted = self.logits + 50.0
        a = HFBitexactSoftmax.apply(self.logits)
        b = HFBitexactSoftmax.apply(shifted)
        # Mathematically exact, but the shift itself is an FP32 rounding, so
        # a couple of elements land one FP32 ULP apart.
        np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-5, atol=1e-6)

    def test_finite_on_large_logits(self):
        """The max-subtract is what keeps ``exp`` from overflowing."""
        big = paddle.to_tensor([[200.0, 100.0, -300.0, 0.0]], dtype="float32")
        out = HFBitexactSoftmax.apply(big)
        self.assertTrue(bool(paddle.all(paddle.isfinite(out))))


class TestHFBitexactSoftmaxBackward(unittest.TestCase):
    """The backward is the reference epilogue, not paddle's softmax_grad."""

    def setUp(self):
        paddle.seed(4)
        self.logits = paddle.randn([6, 12], dtype=paddle.float32)

    def test_matches_reference_epilogue(self):
        x = self.logits.detach()
        x.stop_gradient = False
        probs = HFBitexactSoftmax.apply(x)
        g = paddle.randn(probs.shape, dtype=paddle.float32)
        (gx,) = paddle.grad([probs], [x], grad_outputs=[g])
        with paddle.amp.auto_cast(False):
            p = probs.detach()
            inner = (g * p).sum(axis=-1, keepdim=True)
            expected = (g * p - inner * p).astype(paddle.float32)
        np.testing.assert_array_equal(gx.numpy(), expected.numpy())

    def test_grad_is_cast_back_to_logits_dtype(self):
        x = self.logits.astype(paddle.bfloat16)
        x.stop_gradient = False
        probs = HFBitexactSoftmax.apply(x)
        g = paddle.randn(probs.shape, dtype=paddle.float32)
        (gx,) = paddle.grad([probs], [x], grad_outputs=[g])
        self.assertEqual(gx.dtype, paddle.bfloat16)

    def test_uniform_upstream_grad_gives_zero(self):
        """A constant grad is orthogonal to the simplex, so it cancels."""
        x = self.logits.detach()
        x.stop_gradient = False
        probs = HFBitexactSoftmax.apply(x)
        g = paddle.ones(probs.shape, dtype=paddle.float32)
        (gx,) = paddle.grad([probs], [x], grad_outputs=[g])
        np.testing.assert_allclose(
            gx.numpy(), np.zeros_like(gx.numpy()), rtol=0, atol=1e-7
        )


class TestFusedGateDetachMatmulTarget(unittest.TestCase):
    """Where the gate projection is promoted, per reference."""

    def setUp(self):
        paddle.seed(20260908)
        # w is [E, D]; forward transposes internally, output is [B, E].
        self.x = paddle.randn([8, 32], dtype=paddle.bfloat16)
        self.w = paddle.randn([4, 32], dtype=paddle.bfloat16)

    def test_hf_forward_keeps_projection_in_activation_dtype(self):
        """``F.linear(x, w.cast(x.dtype)).cast(fp32)`` -- cast after the GEMM."""
        out = FusedGateDetachMatmul.apply(self.x, self.w, False, "hf")
        expected = paddle.nn.functional.linear(
            self.x, self.w.T.cast(self.x.dtype)
        ).cast(paddle.float32)
        np.testing.assert_array_equal(out.numpy(), expected.numpy())

    def test_megatron_forward_promotes_both_operands_first(self):
        """``F.linear(x.cast(fp32), w.cast(fp32))`` -- cast before the GEMM."""
        for target in (True, "megatron"):
            with self.subTest(target=target):
                out = FusedGateDetachMatmul.apply(self.x, self.w, False, target)
                expected = paddle.nn.functional.linear(
                    self.x.cast(paddle.float32), self.w.T.cast(paddle.float32)
                )
                np.testing.assert_array_equal(out.numpy(), expected.numpy())

    def test_hf_and_megatron_forwards_differ_in_bf16(self):
        """Guard against the target being ignored on this path."""
        hf = FusedGateDetachMatmul.apply(self.x, self.w, False, "hf").numpy()
        mg = FusedGateDetachMatmul.apply(
            self.x, self.w, False, "megatron"
        ).numpy()
        np.testing.assert_allclose(hf, mg, rtol=0, atol=0.5)
        self.assertFalse(np.array_equal(hf, mg))

    def test_both_targets_return_fp32(self):
        for target in ("hf", "megatron", False):
            with self.subTest(target=target):
                out = FusedGateDetachMatmul.apply(self.x, self.w, False, target)
                self.assertEqual(out.dtype, paddle.float32)

    def test_hf_backward_rounds_grad_to_activation_dtype_first(self):
        """The FP32 softmax input is a cast, so its grad comes back rounded."""
        x = self.x.detach()
        x.stop_gradient = False
        w = self.w.detach()
        w.stop_gradient = False
        out = FusedGateDetachMatmul.apply(x, w, False, "hf")
        g = paddle.randn(out.shape, dtype=paddle.float32)
        gx, gw = paddle.grad([out], [x, w], grad_outputs=[g])
        gr = g.cast(self.x.dtype)
        np.testing.assert_array_equal(
            gx.numpy(),
            paddle.matmul(gr, self.w.cast(self.x.dtype))
            .cast(self.x.dtype)
            .numpy(),
        )
        np.testing.assert_array_equal(
            gw.numpy(),
            paddle.matmul(gr, self.x, transpose_x=True)
            .cast(self.w.dtype)
            .numpy(),
        )

    def test_hf_and_megatron_backwards_differ(self):
        """The two epilogues are different arithmetic, not aliases."""
        g = None
        grads = []
        for target in ("hf", "megatron"):
            xi = self.x.detach()
            xi.stop_gradient = False
            wi = self.w.detach()
            wi.stop_gradient = False
            out = FusedGateDetachMatmul.apply(xi, wi, False, target)
            if g is None:
                # Drawn once: paddle.randn advances the RNG, so drawing per
                # iteration would compare two different problems.
                g = paddle.randn(out.shape, dtype=paddle.float32)
            _, gw = paddle.grad([out], [xi, wi], grad_outputs=[g])
            grads.append(gw.astype("float32").numpy().copy())
        np.testing.assert_allclose(grads[0], grads[1], rtol=0, atol=1.0)
        self.assertFalse(np.array_equal(grads[0], grads[1]))


_CP_PATCH = (
    "paddlefleet.transformer.moe.moe_router.get_context_parallel_world_size"
)


def _router_config(**overrides):
    from paddlefleet.transformer.transformer_config import TransformerConfig

    defaults = {
        "hidden_size": 64,
        "num_attention_heads": 2,
        "intermediate_size": 256,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "sequence_parallel": False,
        "tensor_model_parallel_size": 1,
        "topk_method": "greedy",
        "norm_topk_prob": True,
        "scoring_func": "softmax",
        "n_group": 1,
        "topk_group": 1,
        "routed_scaling_factor": 1.0,
        "routed_scaling_factor_learnable": False,
        "moe_router_force_load_balancing": False,
        "moe_router_load_balancing_type": "aux_loss",
        "moe_deep_gemm": False,
        "router_aux_loss_coef": 0.01,
        "router_z_loss_coef": None,
        "moe_n_hash_layers": 0,
        "use_accuracy_compatible": "hf",
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)


class TestRouterHFScoreAndNormalization(unittest.TestCase):
    """``gate_score_func`` and the top-k renormalization under ``"hf"``."""

    @patch(_CP_PATCH, return_value=1)
    def test_softmax_scoring_uses_the_bitexact_pylayer(self, _cp):
        from paddlefleet.transformer.moe.moe_router import StandardMoERouter

        router = StandardMoERouter(_router_config())
        paddle.seed(51)
        logits = paddle.randn([6, 4], dtype=paddle.float32)
        out = router.gate_score_func(logits)
        np.testing.assert_array_equal(
            out.numpy(), HFBitexactSoftmax.apply(logits).numpy()
        )

    @patch(_CP_PATCH, return_value=1)
    def test_non_softmax_scoring_skips_the_pylayer(self, _cp):
        """The override is only defined for ``scoring_func == "softmax"``."""
        from paddlefleet.transformer.moe.moe_router import StandardMoERouter

        router = StandardMoERouter(_router_config(scoring_func="sigmoid"))
        paddle.seed(52)
        logits = paddle.randn([6, 4], dtype=paddle.float32)
        out = router.gate_score_func(logits)
        np.testing.assert_allclose(
            out.numpy(),
            paddle.nn.functional.sigmoid(logits.cast("float32")).numpy(),
            rtol=1e-6,
            atol=1e-6,
        )

    @patch(_CP_PATCH, return_value=1)
    def test_non_hf_targets_use_paddles_softmax(self, _cp):
        from paddlefleet.transformer.moe.moe_router import StandardMoERouter

        paddle.seed(53)
        logits = paddle.randn([6, 4], dtype=paddle.float32)
        for target in (True, "megatron"):
            with self.subTest(target=target):
                router = StandardMoERouter(
                    _router_config(use_accuracy_compatible=target)
                )
                out = router.gate_score_func(logits)
                np.testing.assert_allclose(
                    out.numpy(),
                    paddle.nn.functional.softmax(
                        logits.cast("float32"), axis=-1
                    ).numpy(),
                    rtol=1e-6,
                    atol=1e-6,
                )

    def _topk_router(self, **overrides):
        """``TopKRouter`` is where the renormalization lives, not the standard one."""
        from paddlefleet.transformer.moe.moe_router import TopKRouter

        router = TopKRouter(config=_router_config(**overrides))
        router.set_layer_number(0)
        return router

    @patch(_CP_PATCH, return_value=1)
    def test_topk_weights_renormalize_to_one(self, _cp):
        """HF divides the FP32 top-k by their plain sum, then rounds once."""
        router = self._topk_router(norm_topk_prob=True)
        paddle.seed(54)
        hidden = paddle.randn([1, 6, 64], dtype=paddle.float32)
        out = router(hidden, input_ids=None)
        top_gate = out[1]
        sums = top_gate.astype("float32").sum(axis=-1).numpy()
        np.testing.assert_allclose(
            sums, np.ones_like(sums), rtol=1e-5, atol=1e-5
        )

    @patch(_CP_PATCH, return_value=1)
    def test_hf_and_megatron_topk_weights_agree_numerically(self, _cp):
        """The fp32-then-round and fp64-sum branches differ only in rounding."""
        gates = []
        for target in ("hf", "megatron"):
            paddle.seed(55)
            router = self._topk_router(use_accuracy_compatible=target)
            hidden = paddle.randn([1, 6, 64], dtype=paddle.float32)
            gates.append(
                router(hidden, input_ids=None)[1].astype("float32").numpy()
            )
        np.testing.assert_allclose(gates[0], gates[1], rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
