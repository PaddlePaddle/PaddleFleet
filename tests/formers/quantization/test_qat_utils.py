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

"""Behavior tests for paddlefleet.quantization.qat_utils.

These tests exercise the REAL simulated-quantization QAT primitives and compare
against INDEPENDENTLY hand-derived expected values (numpy re-implementation of
the quantization formula and the straight-through-estimator gradient). None of
the expected values are produced by calling the function under test.

Environment note (see unit-test-rules.md, 计算优化):
- ``quantize`` / ``dequantize`` / ``int8_backward`` only use float ops
  (max/abs/round/clip/matmul) and run on 无卡 CPU; these classes force CPU.
- ``int8_forward`` performs a real int8 @ int8 GEMM whose numeric backend is
  GPU-only, so its numeric class is skipped unless CUDA is available. We assert
  the CORRECT dequantized-matmul contract there; running it requires single-card.
"""

import os
import types
import unittest

import numpy as np
import paddle

from paddlefleet.quantization.qat_utils import (
    QMIN_QMAX_MAPPING,
    dequantize,
    int8_backward,
    int8_forward,
    quantize,
)


def _round_half_away(arr):
    """Round half away from zero, matching paddle.round semantics."""
    arr = np.asarray(arr, dtype=np.float64)
    return np.sign(arr) * np.floor(np.abs(arr) + 0.5)


class _Config:
    """Minimal real quantization config consumed by qat_utils.

    Holds only the attributes the int8 code path actually reads. This is a data
    container, not a stand-in for the code under test.
    """

    def __init__(
        self,
        scale_epsilon=1e-5,
        apply_hadamard=False,
        hadamard_block_size=1,
        apply_online_actscale_step=0,
        actscale_moving_rate=0.1,
        quant_input_grad=False,
        quant_weight_grad=False,
    ):
        self.scale_epsilon = scale_epsilon
        self.apply_hadamard = apply_hadamard
        self.hadamard_block_size = hadamard_block_size
        self.apply_online_actscale_step = apply_online_actscale_step
        self.actscale_moving_rate = actscale_moving_rate
        self.quant_input_grad = quant_input_grad
        self.quant_weight_grad = quant_weight_grad
        self.fp8_format = {
            "activation": paddle.float8_e4m3fn,
            "weight": paddle.float8_e4m3fn,
            "grad_output": paddle.float8_e5m2,
        }


# ---- independent numpy references for the simulated-quant forward ----


def _ref_quantize_activation(x_np, qmin, qmax, eps):
    """Per-tensor activation quant: scale = max|x|/qmax + eps."""
    scale = np.abs(x_np).max() / qmax + eps
    q = np.clip(_round_half_away(x_np / scale), qmin, qmax).astype(np.int8)
    return q, np.array([scale], dtype=np.float32)


def _ref_quantize_weight(x_np, qmin, qmax, eps):
    """Channel-wise (axis=0) weight quant, scale per output channel."""
    scale = np.abs(x_np).max(axis=0, keepdims=True) / qmax + eps  # [1, N]
    q = np.clip(_round_half_away(x_np / scale), qmin, qmax).astype(np.int8)
    return q, scale.squeeze(0).astype(np.float32)  # hadamard_scale == 1.0


QMIN_QMAX_A8W8 = (-128, 127)
QMIN_QMAX_A8W4_W = (-8, 7)
EPS = 1e-5


class TestQminQmaxMapping(unittest.TestCase):
    """The range table is load-bearing: it drives scale and saturation."""

    def test_integer_ranges_match_signed_bit_widths(self):
        # int8 symmetric-ish range and int4 weight range, derived from bit width.
        self.assertEqual(
            QMIN_QMAX_MAPPING["a8w8linear_activation"], (-128, 127)
        )
        self.assertEqual(QMIN_QMAX_MAPPING["a8w8linear_weight"], (-128, 127))
        self.assertEqual(
            QMIN_QMAX_MAPPING["a8w4linear_activation"], (-128, 127)
        )
        self.assertEqual(QMIN_QMAX_MAPPING["a8w4linear_weight"], (-8, 7))

    def test_fp8_ranges_present(self):
        # fp8 amax bounds keyed by dtype (used by the fp8 path's qmax).
        self.assertEqual(QMIN_QMAX_MAPPING["float8_e4m3fn"], (-488, 488))
        self.assertEqual(QMIN_QMAX_MAPPING["float8_e5m2"], (-57344, 57344))


class TestQuantizeSimulatedForward(unittest.TestCase):
    """Real ``quantize`` vs independent numpy quant formula (CPU)."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_activation_a8w8_matches_hand_derived(self):
        cfg = _Config(scale_epsilon=EPS)
        x_np = np.array([[-1.0, 0.4, 2.0, -1.6]], dtype=np.float32)
        q_ref, s_ref = _ref_quantize_activation(x_np, -128, 127, EPS)

        x = paddle.to_tensor(x_np)
        q, s = quantize(x, "a8w8linear", "activation", cfg)

        self.assertEqual(q.dtype, paddle.int8)
        self.assertEqual(list(s.shape), [1])
        self.assertTrue(s.stop_gradient)  # scale must not carry gradient
        np.testing.assert_array_equal(q.numpy(), q_ref)
        np.testing.assert_allclose(s.numpy(), s_ref, rtol=1e-6, atol=1e-8)

    def test_activation_saturates_at_qmax_with_fixed_scale(self):
        # activation_scale provided + training=False -> scale used verbatim,
        # so oversized values must clip to qmax/qmin (not wrap around).
        cfg = _Config(scale_epsilon=EPS)
        fixed_scale = paddle.to_tensor([0.01], dtype="float32")
        fixed_scale.stop_gradient = True
        x_np = np.array([[2.0, -3.0, 0.006, -0.004]], dtype=np.float32)
        # x / 0.01 = [200, -300, 0.6, -0.4] -> clip to [127, -128, 1, 0].
        # 0.006 (not 0.005) is used on purpose: 0.005/0.01 == 0.5 is an exact
        # rounding tie where paddle.round (half-to-even -> 0) and numpy
        # round-half-away (-> 1) disagree; 0.6 rounds to 1 under both.
        q_ref = np.clip(_round_half_away(x_np / 0.01), -128, 127).astype(
            np.int8
        )

        x = paddle.to_tensor(x_np)
        q, s = quantize(
            x, "a8w8linear", "activation", cfg, activation_scale=fixed_scale
        )
        np.testing.assert_array_equal(q.numpy(), q_ref)
        self.assertEqual(q.numpy()[0, 0], 127)
        self.assertEqual(q.numpy()[0, 1], -128)
        np.testing.assert_allclose(s.numpy(), [0.01], rtol=1e-6)

    def test_weight_a8w8_channelwise_matches_hand_derived(self):
        cfg = _Config(scale_epsilon=EPS)
        x_np = np.array([[1.0, -2.0, 0.5], [-0.3, 0.8, 2.0]], dtype=np.float32)
        q_ref, s_ref = _ref_quantize_weight(x_np, -128, 127, EPS)

        x = paddle.to_tensor(x_np)
        q, s = quantize(x, "a8w8linear", "weight", cfg)

        self.assertEqual(q.dtype, paddle.int8)
        self.assertEqual(list(s.shape), [x_np.shape[1]])  # per output channel
        np.testing.assert_array_equal(q.numpy(), q_ref)
        np.testing.assert_allclose(s.numpy(), s_ref, rtol=1e-6, atol=1e-8)

    def test_weight_a8w4_uses_int4_range(self):
        cfg = _Config(scale_epsilon=EPS)
        x_np = np.array([[1.0, -2.0, 0.5], [-0.3, 0.8, 2.0]], dtype=np.float32)
        q_ref, s_ref = _ref_quantize_weight(x_np, -8, 7, EPS)

        x = paddle.to_tensor(x_np)
        q, s = quantize(x, "a8w4linear", "weight", cfg)

        # int4 payload is still stored in an int8 container.
        self.assertEqual(q.dtype, paddle.int8)
        self.assertTrue((q.numpy() >= -8).all() and (q.numpy() <= 7).all())
        np.testing.assert_array_equal(q.numpy(), q_ref)
        np.testing.assert_allclose(s.numpy(), s_ref, rtol=1e-6, atol=1e-8)

    def test_unknown_algo_raises_keyerror(self):
        cfg = _Config()
        x = paddle.to_tensor([[1.0, 2.0]], dtype="float32")
        with self.assertRaises(KeyError):
            quantize(x, "bogus_algo", "activation", cfg)


class TestDequantizeWeight(unittest.TestCase):
    """Real ``dequantize`` recovers ``q * scale`` per channel (CPU)."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_weight_dequant_applies_channel_scale(self):
        cfg = _Config()
        # Distinguishable payload so a transposed/mis-broadcast scale is caught.
        q_np = np.array([[1, -2, 3], [4, 5, -6]], dtype=np.int8)  # [K=2, N=3]
        scale_np = np.array([0.5, 2.0, 10.0], dtype=np.float32)  # per channel N
        expected = q_np.astype(np.float32) * scale_np  # broadcast over N

        q = paddle.to_tensor(q_np)
        scale = paddle.to_tensor(scale_np)
        out = dequantize(q, scale, "weight", "a8w8linear", cfg)

        self.assertEqual(out.dtype, scale.dtype)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6, atol=1e-8)

    def test_qdq_roundtrip_is_bounded_fake_quant(self):
        # quantize then dequantize == independent fake-quant value, and the
        # approximation error is bounded by scale/2 (quant approximation, not
        # an exact reconstruction).
        cfg = _Config(scale_epsilon=EPS)
        w_np = np.array([[1.0, -2.0, 0.5], [-0.3, 0.8, 2.0]], dtype=np.float32)
        scale_ref = np.abs(w_np).max(axis=0, keepdims=True) / 127 + EPS  # [1,N]
        qdq_ref = (
            np.clip(_round_half_away(w_np / scale_ref), -128, 127) * scale_ref
        )

        w = paddle.to_tensor(w_np)
        q, s = quantize(w, "a8w8linear", "weight", cfg)
        deq = dequantize(q, s, "weight", "a8w8linear", cfg)

        np.testing.assert_allclose(deq.numpy(), qdq_ref, rtol=1e-5, atol=1e-6)
        err = np.abs(deq.numpy() - w_np).max()
        self.assertLessEqual(err, scale_ref.max() / 2 + 1e-6)

    def test_unknown_algo_raises_notimplemented(self):
        cfg = _Config()
        q = paddle.to_tensor([[1, 2, 3]], dtype="int8")
        scale = paddle.ones([3])
        with self.assertRaises(NotImplementedError):
            dequantize(q, scale, "weight", "bogus_algo", cfg)

    def test_unknown_tensor_type_raises_notimplemented(self):
        cfg = _Config()
        q = paddle.to_tensor([[1, 2, 3]], dtype="int8")
        scale = paddle.ones([3])
        with self.assertRaises(NotImplementedError):
            dequantize(q, scale, "activation", "a8w8linear", cfg)


class TestInt8BackwardSTE(unittest.TestCase):
    """Real ``int8_backward`` implements the straight-through estimator (CPU).

    STE contract: the round/clip of the forward quantization is treated as
    identity in the backward pass, so
        dx = grad_output @ dequant(quant_w, scale_w).T
        dw = x.T @ grad_output               (full-precision x, no requant)
    """

    def setUp(self):
        paddle.set_device("cpu")

    def _ctx(self, cfg, x_stop=False, w_stop=False, algo="a8w8linear"):
        return types.SimpleNamespace(
            x_stop_gradient=x_stop,
            w_stop_gradient=w_stop,
            weight_quantize_algo=algo,
            quantization_config=cfg,
        )

    def test_ste_gradients_match_hand_derived(self):
        cfg = _Config(apply_hadamard=False)
        # Non-uniform, distinct M/K/N to catch transpose / orientation errors.
        x_np = np.array([[1.0, -2.0, 0.5], [3.0, 0.25, -1.0]], dtype=np.float32)
        grad_np = np.array(
            [[0.5, -1.0, 2.0, 0.25], [-0.5, 1.5, -2.0, 1.0]], dtype=np.float32
        )  # [M=2, N=4]
        qw_np = np.array(
            [[1, -2, 3, -4], [5, 6, -7, 8], [-1, 2, -3, 4]], dtype=np.int8
        )  # [K=3, N=4]
        scale_w_np = np.array([0.5, 1.0, 2.0, 0.25], dtype=np.float32)  # [N]

        qdq_ref = qw_np.astype(np.float32) * scale_w_np  # [K, N]
        dx_ref = grad_np @ qdq_ref.T  # [M, K]
        dw_ref = x_np.T @ grad_np  # [K, N]

        ctx = self._ctx(cfg)
        dx, dw = int8_backward(
            ctx,
            paddle.to_tensor(x_np),
            paddle.to_tensor(grad_np),
            paddle.to_tensor(qw_np),
            paddle.to_tensor(scale_w_np),
            None,
            None,
        )

        self.assertIsNotNone(dx)
        self.assertIsNotNone(dw)
        self.assertEqual(list(dx.shape), [2, 3])
        self.assertEqual(list(dw.shape), [3, 4])
        np.testing.assert_allclose(dx.numpy(), dx_ref, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(dw.numpy(), dw_ref, rtol=1e-5, atol=1e-5)

        # Guard the STE property: dw is built from full-precision x, so a wrong
        # (e.g. doubled) input would give a materially different gradient.
        self.assertGreater(np.abs(dw_ref).max(), 1e-3)
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(
                dw.numpy(), (2.0 * x_np).T @ grad_np, rtol=1e-5, atol=1e-5
            )

    def test_stop_gradient_flags_yield_none(self):
        cfg = _Config(apply_hadamard=False)
        x = paddle.to_tensor([[1.0, -2.0, 0.5]], dtype="float32")
        grad = paddle.to_tensor([[0.5, -1.0, 2.0, 0.25]], dtype="float32")
        qw = paddle.to_tensor(
            [[1, -2, 3, -4], [5, 6, -7, 8], [-1, 2, -3, 4]], dtype="int8"
        )
        scale_w = paddle.to_tensor([0.5, 1.0, 2.0, 0.25], dtype="float32")

        # x frozen -> no input grad, weight grad still produced.
        dx, dw = int8_backward(
            self._ctx(cfg, x_stop=True, w_stop=False),
            x,
            grad,
            qw,
            scale_w,
            None,
            None,
        )
        self.assertIsNone(dx)
        self.assertIsNotNone(dw)

        # weight frozen -> no weight grad, input grad still produced.
        dx, dw = int8_backward(
            self._ctx(cfg, x_stop=False, w_stop=True),
            x,
            grad,
            qw,
            scale_w,
            None,
            None,
        )
        self.assertIsNotNone(dx)
        self.assertIsNone(dw)


@unittest.skipUnless(
    paddle.is_compiled_with_cuda(),
    "int8_forward runs a real int8 @ int8 GEMM whose backend is GPU-only; "
    "single-card required to verify these numerics.",
)
class TestInt8Forward(unittest.TestCase):
    """Real ``int8_forward`` vs independent dequantized-matmul reference."""

    def setUp(self):
        # int8_forward runs on a real GPU. A launcher/fleet exports
        # FLAGS_selected_gpus when it selects a card; the CI runner leaves it
        # as an empty string, so paddle.set_device("gpu") (via ParallelEnv ->
        # int(FLAGS_selected_gpus[0])) would raise ValueError. Pin an explicit
        # single card as a launcher would, and restore afterwards. Skip
        # honestly on a CUDA build that has no usable device.
        if paddle.device.cuda.device_count() == 0:
            self.skipTest(
                "int8_forward needs a usable CUDA device; this build has none"
            )
        self._orig_device = paddle.get_device()
        self._orig_selected_gpus = os.environ.get("FLAGS_selected_gpus")
        os.environ["FLAGS_selected_gpus"] = "0"
        paddle.set_device("gpu:0")
        self.addCleanup(self._restore_gpu_env)

    def _restore_gpu_env(self):
        paddle.set_device(self._orig_device)
        if self._orig_selected_gpus is None:
            os.environ.pop("FLAGS_selected_gpus", None)
        else:
            os.environ["FLAGS_selected_gpus"] = self._orig_selected_gpus

    def test_forward_matches_quantized_matmul(self):
        cfg = _Config(scale_epsilon=EPS, apply_hadamard=False)
        x_np = np.array(
            [[-1.0, 0.4, 2.0, -1.6], [0.5, -0.5, 1.0, 0.2]], dtype=np.float32
        )  # [M=2, K=4]
        qw_np = np.array(
            [[1, -2, 3], [4, 5, -6], [-7, 8, 9], [10, -11, 12]], dtype=np.int8
        )  # [K=4, N=3]
        scale_w_np = np.array([0.5, 1.0, 2.0], dtype=np.float32)  # [N]
        bias_np = np.array([0.1, -0.2, 0.3], dtype=np.float32)

        qx_ref, sx_ref = _ref_quantize_activation(x_np, -128, 127, EPS)
        expected = (qx_ref.astype(np.float32) @ qw_np.astype(np.float32)) * (
            sx_ref * scale_w_np
        ) + bias_np

        out, quant_x, scale_x = int8_forward(
            x=paddle.to_tensor(x_np),
            quant_w=paddle.to_tensor(qw_np),
            scale_w=paddle.to_tensor(scale_w_np),
            weight_quantize_algo="a8w8linear",
            bias=paddle.to_tensor(bias_np),
            quantization_config=cfg,
        )

        self.assertEqual(quant_x.dtype, paddle.int8)
        self.assertEqual(list(scale_x.shape), [1])
        self.assertEqual(list(out.shape), [2, 3])
        np.testing.assert_array_equal(quant_x.numpy(), qx_ref)
        np.testing.assert_allclose(
            out.numpy().astype(np.float32), expected, rtol=1e-4, atol=1e-4
        )


if __name__ == "__main__":
    unittest.main()
