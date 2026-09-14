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
"""Tests for ``Qwen3_5RMSNorm``'s ``"hf"`` weight-gradient reduction order.

``_HFBroadcastScale`` exists because ``weight``'s gradient is a column reduction
of ``normed * grad_out``, and float addition is not associative, so the *row
order* of that reduction is part of the answer. torch reduces in the operand's
**physical** layout; for the per-head q/k norms the incoming gradient is a
``[b, s, h, d]`` view that is physically ``[b, h, s, d]``, so the reference sums
head-by-head while a naive logical reduction interleaves heads.

Tests cover: forward equals the plain expression, ``grad_normed`` is unaffected,
``head_major`` really changes the reduction order (and matches an explicit
head-major reference), the dtype contract, and the config gating that selects
this layer only under the ``"hf"`` target.
"""

import os
import sys
import unittest

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

from paddlefleet.models.qwen3_5.qwen3_5_model import (
    _HFBroadcastScale,
)


class _Cfg:
    """Minimal stand-in for the two config fields the norm reads."""

    def __init__(self, use_accuracy_compatible=False, rms_norm_eps=1e-6):
        self.use_accuracy_compatible = use_accuracy_compatible
        self.rms_norm_eps = rms_norm_eps


class TestHFBroadcastScaleForward(unittest.TestCase):
    def setUp(self):
        paddle.seed(20260908)
        self.normed = paddle.randn([2, 3, 8], dtype=paddle.float32)
        self.weight = paddle.randn([8], dtype=paddle.float32)
        self.weight.stop_gradient = False

    def test_forward_is_one_centered_scale(self):
        """Forward must be exactly ``normed * (1 + weight)`` in FP32."""
        out = _HFBroadcastScale.apply(self.normed, self.weight)
        expected = self.normed * (1.0 + self.weight.astype("float32"))
        np.testing.assert_array_equal(out.numpy(), expected.numpy())

    def test_forward_shape_and_dtype(self):
        out = _HFBroadcastScale.apply(self.normed, self.weight)
        self.assertEqual(out.shape, self.normed.shape)
        self.assertEqual(out.dtype, paddle.float32)

    def test_zero_weight_is_identity(self):
        """Weight is initialized to 0, so step 0 must pass ``normed`` through."""
        w = paddle.zeros([8], dtype=paddle.float32)
        w.stop_gradient = False
        out = _HFBroadcastScale.apply(self.normed, w)
        np.testing.assert_array_equal(out.numpy(), self.normed.numpy())


class TestHFBroadcastScaleBackward(unittest.TestCase):
    def setUp(self):
        paddle.seed(11)
        self.normed = paddle.randn([2, 4, 3, 8], dtype=paddle.float32)
        self.normed.stop_gradient = False

    def _weight(self):
        w = paddle.randn([8], dtype=paddle.float32)
        w.stop_gradient = False
        return w

    def test_grad_normed_ignores_row_order(self):
        """``grad_normed`` is elementwise, so ``head_major`` must not touch it."""
        for head_major in (False, True):
            with self.subTest(head_major=head_major):
                w = self._weight()
                x = self.normed.detach()
                x.stop_gradient = False
                out = _HFBroadcastScale.apply(x, w, head_major)
                g = paddle.randn(out.shape, dtype=paddle.float32)
                gx, _ = paddle.grad([out], [x, w], grad_outputs=[g])
                np.testing.assert_allclose(
                    gx.numpy(),
                    (g * (1.0 + w.astype("float32"))).numpy(),
                    rtol=0,
                    atol=0,
                )

    def test_head_major_matches_explicit_transposed_reduction(self):
        """The head-major branch equals summing a ``[b, h, s, d]`` view."""
        w = self._weight()
        x = self.normed.detach()
        x.stop_gradient = False
        out = _HFBroadcastScale.apply(x, w, True)
        g = paddle.randn(out.shape, dtype=paddle.float32)
        _, gw = paddle.grad([out], [x, w], grad_outputs=[g])
        product = (g * x).transpose([0, 2, 1, 3])
        expected = product.reshape([-1, 8]).sum(axis=0, dtype="float32")
        np.testing.assert_array_equal(gw.numpy(), expected.numpy())

    def test_logical_order_matches_untransposed_reduction(self):
        """With ``head_major=False`` the reduction keeps the logical order."""
        w = self._weight()
        x = self.normed.detach()
        x.stop_gradient = False
        out = _HFBroadcastScale.apply(x, w, False)
        g = paddle.randn(out.shape, dtype=paddle.float32)
        _, gw = paddle.grad([out], [x, w], grad_outputs=[g])
        expected = (g * x).reshape([-1, 8]).sum(axis=0, dtype="float32")
        np.testing.assert_array_equal(gw.numpy(), expected.numpy())

    def test_head_major_only_applies_to_4d(self):
        """A 3-D product has no head axis, so the flag must be a no-op there."""
        x = paddle.randn([2, 5, 8], dtype=paddle.float32)
        x.stop_gradient = False
        grads = []
        for head_major in (False, True):
            xi = x.detach()
            xi.stop_gradient = False
            w = paddle.zeros([8], dtype=paddle.float32)
            w.stop_gradient = False
            out = _HFBroadcastScale.apply(xi, w, head_major)
            g = paddle.ones(out.shape, dtype=paddle.float32)
            # Request both inputs: Paddle's PyLayer contract wants position 0
            # to be None only when that input is absent from the graph, and
            # production backward always has both.
            grads.append(
                paddle.grad([out], [xi, w], grad_outputs=[g])[1].numpy()
            )
        np.testing.assert_array_equal(grads[0], grads[1])

    def test_grad_weight_is_cast_back_to_param_dtype(self):
        """Accumulate in FP32, round once at the end, like the reference."""
        w = paddle.zeros([8], dtype=paddle.bfloat16)
        w.stop_gradient = False
        x = self.normed.detach()
        x.stop_gradient = False
        out = _HFBroadcastScale.apply(x, w, True)
        g = paddle.randn(out.shape, dtype=paddle.float32)
        _, gw = paddle.grad([out], [x, w], grad_outputs=[g])
        self.assertEqual(gw.dtype, paddle.bfloat16)

    def test_reduction_orders_actually_differ(self):
        """Guard against the flag being wired to something inert."""
        w = self._weight()
        outs = []
        g = paddle.randn(self.normed.shape, dtype=paddle.float32) * 1e3
        for head_major in (False, True):
            xi = self.normed.detach()
            xi.stop_gradient = False
            wi = w.detach()
            wi.stop_gradient = False
            out = _HFBroadcastScale.apply(xi, wi, head_major)
            outs.append(
                paddle.grad([out], [xi, wi], grad_outputs=[g])[1].numpy().copy()
            )
        # Same mathematical value, different summation order.
        np.testing.assert_allclose(outs[0], outs[1], rtol=1e-4, atol=1e-3)
        self.assertFalse(np.array_equal(outs[0], outs[1]))


class TestQwen3_5RMSNormForward(unittest.TestCase):
    """The layer routes through ``_HFBroadcastScale`` only under ``"hf"``."""

    def _make(self, target, head_major_grad=False):
        from paddlefleet.models.qwen3_5.qwen3_5_model import Qwen3_5RMSNorm
        from paddlefleet.transformer.transformer_config import (
            TransformerConfig,
        )

        config = TransformerConfig(
            num_hidden_layers=2,
            hidden_size=8,
            intermediate_size=32,
            num_attention_heads=2,
            rms_norm_eps=1e-6,
            use_accuracy_compatible=target,
        )
        return Qwen3_5RMSNorm(
            config, hidden_size=8, head_major_grad=head_major_grad
        )

    def test_weight_starts_at_zero_so_step_zero_is_plain_rmsnorm(self):
        """1-centered parameterization: ``(1 + 0) * normed``."""
        paddle.seed(41)
        norm = self._make("hf")
        x = paddle.randn([2, 3, 8], dtype=paddle.float32)
        out = norm(x)
        xf = x.astype("float32")
        expected = xf * paddle.rsqrt(
            xf.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon
        )
        np.testing.assert_allclose(
            out.numpy(), expected.numpy(), rtol=1e-6, atol=1e-6
        )

    def test_all_targets_agree_on_the_forward(self):
        """Only the weight-gradient reduction differs, never the forward."""
        outs = []
        for target in ("hf", "megatron", False):
            paddle.seed(42)
            norm = self._make(target)
            x = paddle.randn([2, 3, 8], dtype=paddle.float32)
            outs.append(norm(x).numpy().copy())
        np.testing.assert_allclose(outs[0], outs[1], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(outs[0], outs[2], rtol=1e-6, atol=1e-6)

    def test_head_major_grad_is_stored(self):
        self.assertIs(self._make("hf", True).head_major_grad, True)
        self.assertIs(self._make("hf", False).head_major_grad, False)

    def test_hf_backward_reaches_the_weight(self):
        """Exercises the ``_HFBroadcastScale`` branch end to end."""
        paddle.seed(43)
        norm = self._make("hf", head_major_grad=False)
        x = paddle.randn([2, 3, 8], dtype=paddle.float32)
        x.stop_gradient = False
        norm(x).sum().backward()
        self.assertIsNotNone(norm.weight.grad)
        self.assertIsNotNone(x.grad)

    def test_frozen_weight_takes_the_plain_expression(self):
        """The branch also requires the weight to be trainable."""
        paddle.seed(44)
        norm = self._make("hf")
        norm.weight.stop_gradient = True
        x = paddle.randn([2, 3, 8], dtype=paddle.float32)
        out = norm(x)
        xf = x.astype("float32")
        normed = xf * paddle.rsqrt(
            xf.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon
        )
        expected = normed * (1.0 + norm.weight.astype("float32"))
        np.testing.assert_array_equal(out.numpy(), expected.numpy())

    def test_output_dtype_follows_the_input(self):
        norm = self._make("hf")
        x = paddle.randn([2, 3, 8], dtype=paddle.bfloat16)
        self.assertEqual(norm(x).dtype, paddle.bfloat16)


if __name__ == "__main__":
    unittest.main()
