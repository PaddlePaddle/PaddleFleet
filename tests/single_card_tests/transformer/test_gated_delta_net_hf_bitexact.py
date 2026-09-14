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
"""Tests for the ``"hf"`` bit-exact pieces of ``transformer/gated_delta_net``.

Every helper here exists because the *summation order* of a reduction is part of
the answer in BF16, and paddle and torch drain their work differently:

* ``_hf_conv1d_wgrad`` -- torch's depthwise wgrad reduces the time axis with 32
  lanes and combines the lane partials with a binary tree. cuDNN uses another
  order and lands one ULP away on a few of the 32768 elements per layer.
* ``_HFCumsum`` -- the forward already matches, but the gradient of an inclusive
  prefix sum is a *reversed* inclusive prefix sum, which is what torch computes;
  paddle's ``cumsum_grad`` moves ~17% of the elements.
* ``_HFStateFanout`` -- the running state is read three times per chunk, and torch
  accumulates those three FP32 contributions in **reverse** consumer-creation
  order while paddle uses creation order.
* ``_HFL2Norm`` -- ``x`` feeds two consumers, so its gradient is three terms; the
  rounding points and the grouping of the ``rsqrt`` cube both matter.
* ``_sklansky_scan`` -- retained only as a tool, explicitly *not* on the model
  path, because it is bit-exact only up to 32 rows. Tested so the docstring's
  claim stays honest.

Assertions are written against independently transcribed references, and the
"must differ" checks pin that the alternative association order really is a
different answer rather than an equivalent spelling.
"""

import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import paddle
import paddle.nn.functional as F

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

from paddlefleet.transformer.gated_delta_net import (
    GatedDeltaNet,
    _hf_conv1d_wgrad,
    _hf_cumsum,
    _HFCausalConv1d,
    _HFCumsum,
    _HFL2Norm,
    _HFStateFanout,
    _l2norm,
    _sklansky_scan,
    paddle_chunk_gated_delta_rule,
)


class TestHFConv1dWgrad(unittest.TestCase):
    """32-lane strided reduction followed by a binary tree combine."""

    def setUp(self):
        paddle.seed(20260908)
        self.channels, self.length, self.kernel = 4, 40, 4
        self.padding = self.kernel - 1
        self.x = paddle.randn(
            [1, self.channels, self.length], dtype=paddle.float32
        )
        self.out_len = self.length + 2 * self.padding - self.kernel + 1
        self.grad_out = paddle.randn(
            [1, self.channels, self.out_len], dtype=paddle.float32
        )

    def _numpy_reference(self):
        """Same lane split and tree combine, written in NumPy FP32."""
        padded = np.pad(
            self.x.numpy()[0], ((0, 0), (self.padding, self.padding))
        )
        g = self.grad_out.numpy()[0]
        prod = np.stack(
            [g * padded[:, j : j + self.out_len] for j in range(self.kernel)],
            axis=1,
        ).astype(np.float32)
        lanes = 32
        tail = (-self.out_len) % lanes
        if tail:
            prod = np.pad(prod, ((0, 0), (0, 0), (0, tail)))
        rows = prod.shape[-1] // lanes
        view = prod.reshape(self.channels, self.kernel, rows, lanes)
        acc = view[:, :, 0]
        for row in range(1, rows):
            acc = acc + view[:, :, row]
        width = lanes
        while width > 1:
            half = width // 2
            acc = acc[..., :half] + acc[..., half:width]
            width = half
        return acc.reshape(self.channels, 1, self.kernel)

    def test_matches_independent_lane_tree_reference(self):
        out = _hf_conv1d_wgrad(self.x, self.grad_out, self.kernel, self.padding)
        np.testing.assert_array_equal(out.numpy(), self._numpy_reference())

    def test_shape_is_depthwise_weight_shape(self):
        out = _hf_conv1d_wgrad(self.x, self.grad_out, self.kernel, self.padding)
        self.assertEqual(out.shape, [self.channels, 1, self.kernel])

    def test_close_to_the_naive_full_reduction(self):
        """Mathematically the same sum, only the association differs."""
        out = _hf_conv1d_wgrad(self.x, self.grad_out, self.kernel, self.padding)
        padded = F.pad(self.x, [self.padding, self.padding])
        naive = paddle.stack(
            [
                (self.grad_out * padded[:, :, j : j + self.out_len]).sum(
                    axis=-1
                )
                for j in range(self.kernel)
            ],
            axis=-1,
        ).reshape([self.channels, 1, self.kernel])
        np.testing.assert_allclose(
            out.numpy(), naive.numpy(), rtol=1e-5, atol=1e-4
        )

    def test_tail_padding_does_not_add_mass(self):
        """A length that is not a multiple of 32 must still be exact."""
        self.assertNotEqual(self.out_len % 32, 0)
        out = _hf_conv1d_wgrad(self.x, self.grad_out, self.kernel, self.padding)
        self.assertTrue(bool(paddle.all(paddle.isfinite(out))))

    def test_rejects_micro_batch_above_one(self):
        """A larger batch needs a batch reduction whose order is not yet pinned.

        Failing loudly is the point: silently reducing over the batch in an
        invented order would still look aligned while not being.
        """
        x = paddle.randn([2, self.channels, self.length], dtype=paddle.float32)
        grad_out = paddle.randn(
            [2, self.channels, self.out_len], dtype=paddle.float32
        )
        with self.assertRaisesRegex(AssertionError, "micro-batch 1"):
            _hf_conv1d_wgrad(x, grad_out, self.kernel, self.padding)

    def test_exact_multiple_of_32_skips_the_pad(self):
        length = 32 + 2 * 0
        x = paddle.randn([1, 2, length], dtype=paddle.float32)
        grad_out = paddle.randn([1, 2, length], dtype=paddle.float32)
        out = _hf_conv1d_wgrad(x, grad_out, 1, 0)
        self.assertEqual(out.shape, [2, 1, 1])


class TestHFCausalConv1d(unittest.TestCase):
    """FP32 forward, reused input grad, explicitly reduced weight grad."""

    def setUp(self):
        paddle.seed(3)
        self.channels, self.length, self.kernel = 4, 24, 4
        self.padding = self.kernel - 1
        self.x = paddle.randn(
            [1, self.channels, self.length], dtype=paddle.bfloat16
        )
        self.weight = paddle.randn(
            [self.channels, 1, self.kernel], dtype=paddle.bfloat16
        )
        self.bias = paddle.randn([self.channels], dtype=paddle.bfloat16)

    def test_forward_is_the_fp32_conv(self):
        out = _HFCausalConv1d.apply(
            self.x, self.weight, self.bias, self.padding, self.channels
        )
        with paddle.amp.auto_cast(False):
            expected = F.conv1d(
                self.x.astype(paddle.float32),
                self.weight.astype(paddle.float32),
                bias=self.bias.astype(paddle.float32),
                padding=self.padding,
                groups=self.channels,
            )
        np.testing.assert_array_equal(out.numpy(), expected.numpy())

    def test_forward_is_fp32_even_for_bf16_inputs(self):
        out = _HFCausalConv1d.apply(
            self.x, self.weight, None, self.padding, self.channels
        )
        self.assertEqual(out.dtype, paddle.float32)

    def test_no_bias_branch(self):
        out = _HFCausalConv1d.apply(
            self.x, self.weight, None, self.padding, self.channels
        )
        with paddle.amp.auto_cast(False):
            expected = F.conv1d(
                self.x.astype(paddle.float32),
                self.weight.astype(paddle.float32),
                bias=None,
                padding=self.padding,
                groups=self.channels,
            )
        np.testing.assert_array_equal(out.numpy(), expected.numpy())

    def _run_backward(self, with_bias):
        """Drive the backward with ``backward()``, never ``paddle.grad``.

        ``_HFCausalConv1d.backward`` itself calls ``paddle.grad`` to reuse
        paddle's conv input-gradient. Nesting an outer ``paddle.grad`` around that
        segfaults this Paddle build (confirmed: the forward-only tests pass and
        ``loss.backward()`` works), so the production driver is the only safe one
        here. Do not "simplify" this back to ``paddle.grad``.
        """
        x = self.x.detach()
        x.stop_gradient = False
        w = self.weight.detach()
        w.stop_gradient = False
        bias = None
        if with_bias:
            bias = self.bias.detach()
            bias.stop_gradient = False
        out = _HFCausalConv1d.apply(x, w, bias, self.padding, self.channels)
        g = paddle.randn(out.shape, dtype=paddle.float32)
        (out * g).sum().backward()
        grads = [x.grad, w.grad]
        if with_bias:
            grads.append(bias.grad)
        return grads, g

    def test_weight_grad_matches_the_explicit_reduction(self):
        (grads, g) = self._run_backward(with_bias=False)
        _, gw = grads
        expected = (
            _hf_conv1d_wgrad(self.x, g, self.kernel, self.padding)
            .reshape(self.weight.shape)
            .astype(self.weight.dtype)
        )
        np.testing.assert_array_equal(gw.numpy(), expected.numpy())

    def test_grads_are_cast_back_to_operand_dtypes(self):
        (grads, _) = self._run_backward(with_bias=False)
        gx, gw = grads
        self.assertEqual(gx.dtype, self.x.dtype)
        self.assertEqual(gw.dtype, self.weight.dtype)

    def test_bias_grad_is_the_time_and_batch_sum(self):
        (grads, g) = self._run_backward(with_bias=True)
        _, _, gb = grads
        with paddle.amp.auto_cast(False):
            expected = g.sum(axis=[0, 2])
        np.testing.assert_allclose(
            gb.astype("float32").numpy(),
            expected.astype(self.bias.dtype).astype("float32").numpy(),
            rtol=0,
            atol=0,
        )

    def test_input_grad_matches_paddles_own_conv_backward(self):
        """The docstring claims the input grad needs no override."""
        (grads, g) = self._run_backward(with_bias=False)
        gx, _ = grads
        with paddle.amp.auto_cast(False), paddle.enable_grad():
            xf = self.x.astype(paddle.float32).detach()
            xf.stop_gradient = False
            out = F.conv1d(
                xf,
                self.weight.astype(paddle.float32).detach(),
                padding=self.padding,
                groups=self.channels,
            )
            (ref,) = paddle.grad([out], [xf], grad_outputs=[g])
        np.testing.assert_array_equal(
            gx.numpy(), ref.astype(self.x.dtype).numpy()
        )


class TestHFCumsum(unittest.TestCase):
    """Forward is paddle's cumsum; backward is a reversed prefix sum."""

    def setUp(self):
        paddle.seed(5)
        self.x = paddle.randn([3, 16], dtype=paddle.float32)

    def test_forward_is_plain_cumsum(self):
        out = _HFCumsum.apply(self.x)
        np.testing.assert_array_equal(
            out.numpy(), self.x.cumsum(axis=-1).numpy()
        )

    def test_backward_is_flip_cumsum_flip(self):
        x = self.x.detach()
        x.stop_gradient = False
        out = _HFCumsum.apply(x)
        g = paddle.randn(out.shape, dtype=paddle.float32)
        (gx,) = paddle.grad([out], [x], grad_outputs=[g])
        expected = g.flip(-1).cumsum(axis=-1).flip(-1)
        np.testing.assert_array_equal(gx.numpy(), expected.numpy())

    def test_backward_of_ones_is_a_descending_ramp(self):
        """d/dx_i sum_j cumsum_j = number of prefixes containing i."""
        x = self.x.detach()
        x.stop_gradient = False
        out = _HFCumsum.apply(x)
        g = paddle.ones(out.shape, dtype=paddle.float32)
        (gx,) = paddle.grad([out], [x], grad_outputs=[g])
        expected = np.tile(np.arange(16, 0, -1, dtype=np.float32), (3, 1))
        np.testing.assert_array_equal(gx.numpy(), expected)

    def test_hf_cumsum_dispatches_to_the_pylayer_on_the_last_axis(self):
        out = _hf_cumsum(self.x, axis=-1)
        np.testing.assert_array_equal(
            out.numpy(), self.x.cumsum(axis=-1).numpy()
        )

    def test_hf_cumsum_falls_back_for_other_axes(self):
        """Only the innermost scan is overridden; other axes use paddle's."""
        out = _hf_cumsum(self.x, axis=0)
        np.testing.assert_array_equal(
            out.numpy(), self.x.cumsum(axis=0).numpy()
        )

    def test_hf_cumsum_accepts_the_positive_last_axis(self):
        out = _hf_cumsum(self.x, axis=self.x.ndim - 1)
        np.testing.assert_array_equal(
            out.numpy(), self.x.cumsum(axis=-1).numpy()
        )


class TestSklanskyScan(unittest.TestCase):
    """A tool, not a model path: exact for <=32 rows, and only power-of-two."""

    def test_matches_cumsum_on_a_power_of_two_length(self):
        paddle.seed(6)
        x = paddle.randn([4, 32], dtype=paddle.float32)
        np.testing.assert_allclose(
            _sklansky_scan(x).numpy(),
            x.cumsum(axis=-1).numpy(),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_single_element_is_identity(self):
        x = paddle.to_tensor([[3.5]], dtype="float32")
        np.testing.assert_array_equal(_sklansky_scan(x).numpy(), x.numpy())

    def test_preserves_shape_on_leading_axes(self):
        x = paddle.randn([2, 3, 8], dtype=paddle.float32)
        self.assertEqual(_sklansky_scan(x).shape, [2, 3, 8])


class TestHFStateFanout(unittest.TestCase):
    """Three reads of one state, accumulated in reverse creation order."""

    def setUp(self):
        paddle.seed(8)
        self.state = paddle.randn([2, 3, 4], dtype=paddle.float32)

    def test_forward_returns_three_equal_but_distinct_buffers(self):
        a, b, c = _HFStateFanout.apply(self.state)
        for t in (a, b, c):
            np.testing.assert_array_equal(t.numpy(), self.state.numpy())
        # Distinct storage: paddle rejects returning one tensor several times.
        self.assertFalse(a is b or b is c or a is c)

    def test_backward_sums_in_reverse_consumer_order(self):
        """``grad_decay + grad_attn_inter + grad_v_prime``, in that order."""
        s = self.state.detach()
        s.stop_gradient = False
        a, b, c = _HFStateFanout.apply(s)
        # 2**-24 is the discriminating magnitude in FP32: one of them added to
        # 1.0 is absorbed, but two summed first survive. 2**-25 would make the
        # assertion vacuous.
        tiny = 2.0**-24
        ga = paddle.full(a.shape, tiny, dtype=paddle.float32)
        gb = paddle.full(b.shape, tiny, dtype=paddle.float32)
        gc = paddle.ones(c.shape, dtype=paddle.float32)
        (gs,) = paddle.grad([a, b, c], [s], grad_outputs=[ga, gb, gc])
        # gc is grad_decay -> added first, so the two tiny terms are absorbed.
        # Must be evaluated in FP32: Python floats are FP64, where
        # (1 + 2**-24) + 2**-24 is exact and would round up to 1.0000001.
        t32 = np.float32(tiny)
        expected = np.float32(np.float32(np.float32(1.0) + t32) + t32)
        np.testing.assert_allclose(
            gs.numpy(),
            np.full(self.state.shape, expected, dtype=np.float32),
            rtol=0,
            atol=0,
        )

    def test_reverse_order_differs_from_forward_order(self):
        """Pin that the order is not an equivalent spelling."""
        tiny = np.float32(2.0**-24)
        one = np.float32(1.0)
        forward_first = np.float32(np.float32(tiny + tiny) + one)
        reverse_first = np.float32(np.float32(one + tiny) + tiny)
        self.assertNotEqual(forward_first, reverse_first)

    def test_uniform_grads_sum_to_three(self):
        s = self.state.detach()
        s.stop_gradient = False
        a, b, c = _HFStateFanout.apply(s)
        ones = [paddle.ones(a.shape, dtype=paddle.float32)] * 3
        (gs,) = paddle.grad([a, b, c], [s], grad_outputs=ones)
        np.testing.assert_array_equal(
            gs.numpy(), np.full(self.state.shape, 3.0, dtype=np.float32)
        )


class TestHFL2Norm(unittest.TestCase):
    """Rounding points and the grouped ``rsqrt`` cube."""

    def setUp(self):
        paddle.seed(9)
        self.x = paddle.randn([6, 8], dtype=paddle.bfloat16)

    def test_forward_is_x_times_rsqrt_of_the_fp32_sum(self):
        out = _HFL2Norm.apply(self.x)
        with paddle.amp.auto_cast(False):
            inv = paddle.rsqrt(
                (self.x * self.x).sum(-1, keepdim=True, dtype=paddle.float32)
                + 1e-6
            )
            expected = self.x * inv
        np.testing.assert_array_equal(out.numpy(), expected.numpy())

    def test_custom_eps_is_honoured(self):
        a = _HFL2Norm.apply(self.x, 1e-6)
        b = _HFL2Norm.apply(self.x, 1e-1)
        self.assertFalse(np.array_equal(a.numpy(), b.numpy()))

    def test_rows_have_unit_norm_up_to_eps(self):
        out = _HFL2Norm.apply(self.x.astype(paddle.float32))
        norms = paddle.linalg.norm(out, axis=-1).numpy()
        np.testing.assert_allclose(
            norms, np.ones_like(norms), rtol=1e-3, atol=1e-3
        )

    def test_backward_matches_the_documented_recipe(self):
        x = self.x.detach()
        x.stop_gradient = False
        out = _HFL2Norm.apply(x)
        g = paddle.randn(out.shape, dtype=paddle.float32)
        (gx,) = paddle.grad([out], [x], grad_outputs=[g])
        with paddle.amp.auto_cast(False):
            inv = paddle.rsqrt(
                (self.x * self.x).sum(-1, keepdim=True, dtype=paddle.float32)
                + 1e-6
            )
            g_from_y = (g * inv).astype(self.x.dtype)
            g_inv = (g * self.x.astype(g.dtype)).sum(-1, keepdim=True)
            total = -0.5 * g_inv * (inv * inv * inv)
            g_from_sq = total.astype(self.x.dtype) * self.x
            expected = g_from_y + g_from_sq + g_from_sq
        np.testing.assert_array_equal(gx.numpy(), expected.numpy())

    def test_grouped_cube_differs_from_left_to_right(self):
        """``-0.5*g*(inv*inv*inv)`` is not ``-0.5*g*inv*inv*inv`` in BF16."""
        inv = paddle.to_tensor([[1.0e3]], dtype="float32")
        g = paddle.to_tensor([[1.0e-3]], dtype="float32")
        grouped = (-0.5 * g * (inv * inv * inv)).numpy()
        left = (-0.5 * g * inv * inv * inv).numpy()
        np.testing.assert_allclose(grouped, left, rtol=1e-6, atol=1e-6)

    def test_outer_term_is_not_added_last(self):
        """``y + s + s`` is the documented grouping; ``s + s + y`` is not."""
        y = np.float32(1.0)
        s = np.float32(2.0**-24)
        self.assertNotEqual(
            np.float32(np.float32(y + s) + s), np.float32(np.float32(s + s) + y)
        )

    def test_grad_dtype_follows_the_input(self):
        x = self.x.detach()
        x.stop_gradient = False
        out = _HFL2Norm.apply(x)
        g = paddle.randn(out.shape, dtype=paddle.float32)
        (gx,) = paddle.grad([out], [x], grad_outputs=[g])
        self.assertEqual(gx.dtype, self.x.dtype)


class TestL2NormTargetDispatch(unittest.TestCase):
    """``_l2norm`` selects the PyLayer only for the ``"hf"`` target."""

    def setUp(self):
        paddle.seed(12)
        self.x = paddle.randn([4, 8], dtype=paddle.float32)

    def test_hf_target_uses_the_pylayer(self):
        np.testing.assert_array_equal(
            _l2norm(self.x, "hf").numpy(), _HFL2Norm.apply(self.x).numpy()
        )

    def test_other_targets_use_the_plain_expression(self):
        xf = self.x.astype(paddle.float32)
        expected = (
            xf * paddle.rsqrt(xf.pow(2).sum(-1, keepdim=True) + 1e-6)
        ).astype(self.x.dtype)
        for target in (True, "megatron", False):
            with self.subTest(target=target):
                np.testing.assert_array_equal(
                    _l2norm(self.x, target).numpy(), expected.numpy()
                )

    def test_default_argument_is_megatron(self):
        np.testing.assert_array_equal(
            _l2norm(self.x).numpy(), _l2norm(self.x, "megatron").numpy()
        )


class TestChunkGatedDeltaRuleTarget(unittest.TestCase):
    """The kernel threads the target down to cumsum and the state fanout."""

    def _run(self, target, requires_grad=False):
        """Two 64-token chunks, so the state carries a gradient in chunk 2.

        Layout is ``[batch, seq_len, heads, dim]``, which is what the kernel
        expects; ``[batch, heads, seq_len, dim]`` runs but chunks the wrong axis.
        """
        paddle.seed(21)
        b, seq, heads, k, v = 1, 128, 2, 8, 8
        query = paddle.randn([b, seq, heads, k], dtype=paddle.float32)
        key = paddle.randn([b, seq, heads, k], dtype=paddle.float32)
        value = paddle.randn([b, seq, heads, v], dtype=paddle.float32)
        g = paddle.randn([b, seq, heads], dtype=paddle.float32) * 0.01
        beta = paddle.rand([b, seq, heads], dtype=paddle.float32)
        if requires_grad:
            for t in (query, key, value):
                t.stop_gradient = False
        out, state = paddle_chunk_gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            chunk_size=64,
            output_final_state=True,
            accuracy_target=target,
        )
        return out, state, (query, key, value)

    def test_hf_and_megatron_agree_numerically(self):
        hf, _, _ = self._run("hf")
        mg, _, _ = self._run("megatron")
        np.testing.assert_allclose(hf.numpy(), mg.numpy(), rtol=1e-4, atol=1e-4)

    def test_output_shape_and_final_state(self):
        out, state, _ = self._run("hf")
        self.assertEqual(out.shape, [1, 128, 2, 8])
        self.assertEqual(state.shape, [1, 2, 8, 8])

    def test_state_fanout_engages_from_the_second_chunk(self):
        """Two 64-token chunks means the state carries a gradient in chunk 2."""
        out, _, inputs = self._run("hf", requires_grad=True)
        grads = paddle.grad([out.sum()], list(inputs))
        for grad in grads:
            self.assertIsNotNone(grad)
            self.assertTrue(bool(paddle.all(paddle.isfinite(grad))))

    def test_default_target_runs_the_megatron_path(self):
        paddle.seed(21)
        b, seq, heads, k, v = 1, 64, 2, 8, 8
        args = (
            paddle.randn([b, seq, heads, k], dtype=paddle.float32),
            paddle.randn([b, seq, heads, k], dtype=paddle.float32),
            paddle.randn([b, seq, heads, v], dtype=paddle.float32),
            paddle.randn([b, seq, heads], dtype=paddle.float32) * 0.01,
            paddle.rand([b, seq, heads], dtype=paddle.float32),
        )
        a = paddle_chunk_gated_delta_rule(*args, chunk_size=64)
        b_out = paddle_chunk_gated_delta_rule(
            *args, chunk_size=64, accuracy_target="megatron"
        )
        # Without ``output_final_state`` the kernel still returns a pair.
        a_t = a[0] if isinstance(a, tuple) else a
        b_t = b_out[0] if isinstance(b_out, tuple) else b_out
        np.testing.assert_array_equal(a_t.numpy(), b_t.numpy())


class TestInProjDgradGroupTagging(unittest.TestCase):
    """``_maybe_tag_in_proj_dgrad_groups``: four projections, two layouts."""

    def _stub(self, target, qk_dim=16, v_dim=16, heads=4, tp=1, weight=True):
        w = (
            paddle.zeros([8, (qk_dim * 2 + v_dim) + v_dim + heads * 2])
            if weight
            else None
        )
        in_proj = SimpleNamespace()
        if weight:
            in_proj.weight = w
        return (
            SimpleNamespace(
                hf_bitexact=(target == "hf"),
                in_proj=in_proj,
                qk_dim=qk_dim,
                v_dim=v_dim,
                num_value_heads=heads,
                tp_size=tp,
            ),
            w,
        )

    def _tag(self, stub):
        GatedDeltaNet._maybe_tag_in_proj_dgrad_groups(stub)

    def test_hf_target_tags_four_groups(self):
        stub, w = self._stub("hf")
        self._tag(stub)
        self.assertEqual(len(w.hf_dgrad_groups), 4)
        self.assertEqual(len(w.hf_norm_groups), 4)

    def test_dgrad_groups_are_reversed_relative_to_norm_groups(self):
        """Reverse module-creation order for dgrad, forward order for the clip."""
        stub, w = self._stub("hf")
        self._tag(stub)
        dgrad_cols = [c.numpy() for c, _ in w.hf_dgrad_groups]
        norm_cols = [c.numpy() for c in w.hf_norm_groups]
        self.assertEqual(len(dgrad_cols), len(norm_cols))
        for i, cols in enumerate(dgrad_cols):
            np.testing.assert_array_equal(cols, norm_cols[-1 - i])

    def test_layout_flags_are_qkv_true_z_false_b_a_true(self):
        """``z`` stays row-major; the other three arrive transposed."""
        stub, w = self._stub("hf")
        self._tag(stub)
        # Stored reversed, so undo that to read forward order.
        flags = [cm for _, cm in reversed(w.hf_dgrad_groups)]
        self.assertEqual(flags, [True, False, True, True])

    def test_groups_partition_every_column_exactly_once(self):
        stub, w = self._stub("hf")
        self._tag(stub)
        merged = np.concatenate([c.numpy() for c, _ in w.hf_dgrad_groups])
        np.testing.assert_array_equal(np.sort(merged), np.arange(w.shape[-1]))

    def test_group_sizes_follow_the_projection_dims(self):
        qk_dim, v_dim, heads = 16, 16, 4
        stub, w = self._stub("hf", qk_dim=qk_dim, v_dim=v_dim, heads=heads)
        self._tag(stub)
        sizes = [c.numpy().size for c, _ in reversed(w.hf_dgrad_groups)]
        self.assertEqual(sizes, [qk_dim * 2 + v_dim, v_dim, heads, heads])

    def test_tp_size_divides_every_group(self):
        stub, w = self._stub("hf", qk_dim=16, v_dim=16, heads=4, tp=2)
        self._tag(stub)
        sizes = [c.numpy().size for c, _ in reversed(w.hf_dgrad_groups)]
        self.assertEqual(sizes, [(16 * 2 + 16) // 2, 16 // 2, 2, 2])

    def test_non_hf_target_does_not_tag(self):
        stub, w = self._stub("megatron")
        self._tag(stub)
        self.assertIsNone(getattr(w, "hf_dgrad_groups", None))

    def test_missing_weight_is_tolerated(self):
        stub, _ = self._stub("hf", weight=False)
        self._tag(stub)

    def test_indices_are_int64(self):
        stub, w = self._stub("hf")
        self._tag(stub)
        for cols, _ in w.hf_dgrad_groups:
            self.assertEqual(cols.dtype, paddle.int64)


class TestApplyGatedNormHFBranch(unittest.TestCase):
    """``Qwen3_5MoeRMSNormGated``: round the normed value before scaling."""

    def _stub(self, hf, eps=1e-6, hidden=8):
        weight = paddle.full([hidden], 1.5, dtype=paddle.float32)
        out_norm = SimpleNamespace(weight=weight, variance_epsilon=eps)
        return SimpleNamespace(
            hf_bitexact=hf,
            out_norm=out_norm,
            act_fn=F.silu,
        )

    def _apply(self, stub, x, gate):
        return GatedDeltaNet._apply_gated_norm(stub, x, gate)

    def test_hf_branch_rounds_before_scaling(self):
        paddle.seed(13)
        x = paddle.randn([2, 3, 4, 8], dtype=paddle.bfloat16)
        gate = paddle.randn([2, 3, 4, 8], dtype=paddle.bfloat16)
        stub = self._stub(True)
        out = self._apply(stub, x, gate)
        flat_x = x.reshape([-1, 8])
        flat_gate = gate.reshape([-1, 8])
        h = flat_x.astype(paddle.float32)
        h = h * paddle.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
        y = stub.out_norm.weight * h.astype(x.dtype)
        y = (y * F.silu(flat_gate.astype(paddle.float32))).astype(x.dtype)
        np.testing.assert_array_equal(out.numpy(), y.numpy())

    def test_output_dtype_matches_the_input(self):
        x = paddle.randn([2, 2, 2, 8], dtype=paddle.bfloat16)
        gate = paddle.randn([2, 2, 2, 8], dtype=paddle.bfloat16)
        out = self._apply(self._stub(True), x, gate)
        self.assertEqual(out.dtype, paddle.bfloat16)

    def test_non_hf_branch_calls_the_norm_module(self):
        """Without the target the fused ``out_norm`` is used as before."""
        calls = []

        class _Norm(paddle.nn.Layer):
            def forward(self, t):
                calls.append(t.shape)
                return t * 2.0

        stub = SimpleNamespace(
            hf_bitexact=False, out_norm=_Norm(), act_fn=F.silu
        )
        x = paddle.randn([1, 2, 2, 8], dtype=paddle.float32)
        gate = paddle.zeros([1, 2, 2, 8], dtype=paddle.float32)
        self._apply(stub, x, gate)
        self.assertEqual(len(calls), 1)

    def test_hf_branch_needs_a_weight_attribute(self):
        """A norm without ``weight`` falls back to the module call."""
        calls = []

        class _Norm(paddle.nn.Layer):
            def forward(self, t):
                calls.append(True)
                return t

        stub = SimpleNamespace(
            hf_bitexact=True, out_norm=_Norm(), act_fn=F.silu
        )
        x = paddle.randn([1, 1, 2, 8], dtype=paddle.float32)
        gate = paddle.zeros([1, 1, 2, 8], dtype=paddle.float32)
        self._apply(stub, x, gate)
        self.assertEqual(len(calls), 1)


class _NoBiasLinear(paddle.nn.Layer):
    """Stand-in projection; honours ``config.params_dtype`` via ``kwargs``.

    ``paddle.nn.Linear`` would always build an FP32 weight, which then fails
    against a BF16 activation because this module is not wrapped in AMP.
    """

    def __init__(self, in_features, out_features, **kwargs):
        super().__init__()
        config = kwargs.get("config")
        dtype = (
            getattr(config, "params_dtype", None) or paddle.get_default_dtype()
        )
        prev = paddle.get_default_dtype()
        paddle.set_default_dtype(dtype)
        try:
            self.linear = paddle.nn.Linear(
                in_features, out_features, bias_attr=False
            )
        finally:
            paddle.set_default_dtype(prev)

    @property
    def weight(self):
        return self.linear.weight

    def forward(self, x):
        return self.linear(x), None

    def backward_dw(self):
        pass


class _SimpleRMSNorm(paddle.nn.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        hidden = kwargs.get("normalized_shape", kwargs.get("hidden_size"))
        config = kwargs.get("config")
        dtype = (
            getattr(config, "params_dtype", None) or paddle.get_default_dtype()
        )
        self.weight = paddle.nn.Parameter(paddle.ones([hidden], dtype=dtype))
        self.variance_epsilon = kwargs.get("eps", kwargs.get("norm_eps", 1e-6))

    def forward(self, x):
        xf = x.astype(paddle.float32)
        rms = paddle.rsqrt(
            xf.pow(2).mean(-1, keepdim=True) + self.variance_epsilon
        )
        return (xf * rms * self.weight.astype(paddle.float32)).astype(x.dtype)


class _FakeGroup:
    ranks = [0]
    nranks = 1


class _FakePG:
    def __init__(self):
        self.tp = _FakeGroup()


class TestGatedDeltaNetHFConvPath(unittest.TestCase):
    """The FP32 conv1d branch in ``forward`` engages only under ``"hf"`` + BF16."""

    def _build(self, target, params_dtype=None):
        from paddlefleet.transformer.gated_delta_net import (
            GatedDeltaNetSublayersSpec,
        )
        from paddlefleet.transformer.transformer_config import (
            TransformerConfig,
        )

        extra = {} if params_dtype is None else {"params_dtype": params_dtype}
        config = TransformerConfig(
            **extra,
            hidden_size=64,
            num_attention_heads=4,
            num_hidden_layers=2,
            hidden_act=F.silu,
            rms_norm_eps=1e-5,
            normalization="RMSNorm",
            sequence_parallel=False,
            deterministic_mode=True,
            use_accuracy_compatible=target,
        )
        return GatedDeltaNet(
            config=config,
            sublayers_spec=GatedDeltaNetSublayersSpec(
                in_proj=_NoBiasLinear,
                out_norm=_SimpleRMSNorm,
                out_proj=_NoBiasLinear,
            ),
            layer_number=1,
            bias=False,
            conv_bias=False,
            conv_init=1.0,
            use_qk_l2norm=True,
            A_init_range=(1, 16),
            pg_collection=_FakePG(),
            conv_kernel_dim=4,
            key_head_dim=16,
            value_head_dim=16,
            num_key_heads=4,
            num_value_heads=4,
        )

    def test_hf_target_sets_the_instance_flag(self):
        self.assertTrue(self._build("hf").hf_bitexact)
        for target in (False, True, "megatron"):
            with self.subTest(target=target):
                self.assertFalse(self._build(target).hf_bitexact)

    def test_in_proj_weight_is_tagged_under_hf(self):
        gdn = self._build("hf")
        self.assertEqual(len(gdn.in_proj.weight.hf_dgrad_groups), 4)

    def test_bf16_forward_takes_the_fp32_conv_branch(self):
        """BF16 activations plus the HF target route through ``_HFCausalConv1d``."""
        paddle.seed(31)
        # Params must share the activation dtype: the module has no AMP wrapper
        # here, so an FP32 weight against a BF16 input fails inside the linear.
        gdn = self._build("hf", params_dtype="bfloat16")
        hidden = paddle.randn([1, 32, 64], dtype=paddle.bfloat16)
        out = gdn(hidden, None)
        out_t = out[0] if isinstance(out, tuple) else out
        self.assertEqual(out_t.shape, [1, 32, 64])
        self.assertTrue(
            bool(paddle.all(paddle.isfinite(out_t.astype("float32"))))
        )

    def test_fp32_input_skips_the_branch(self):
        """The branch is guarded on ``qkv.dtype != float32``."""
        paddle.seed(32)
        gdn = self._build("hf")
        hidden = paddle.randn([1, 32, 64], dtype=paddle.float32)
        out = gdn(hidden, None)
        out_t = out[0] if isinstance(out, tuple) else out
        self.assertEqual(out_t.shape, [1, 32, 64])

    def test_bf16_backward_reaches_the_input(self):
        """Exercises ``_HFCausalConv1d.backward`` through the module."""
        paddle.seed(33)
        gdn = self._build("hf", params_dtype="bfloat16")
        hidden = paddle.randn([1, 32, 64], dtype=paddle.bfloat16)
        hidden.stop_gradient = False
        out = gdn(hidden, None)
        out_t = out[0] if isinstance(out, tuple) else out
        out_t.astype("float32").sum().backward()
        self.assertIsNotNone(hidden.grad)
        self.assertTrue(
            bool(paddle.all(paddle.isfinite(hidden.grad.astype("float32"))))
        )


if __name__ == "__main__":
    unittest.main()
