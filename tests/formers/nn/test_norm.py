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

"""Behavior tests for paddlefleet.nn.norm (LayerNorm, RMSNorm, Norm factory).

Scope: 无卡 (CPU-only). These tests exercise the real production layers and
compare their output against an *independent* NumPy reference derived from the
RMSNorm / LayerNorm definitions (computed in float64, first principles). They
verify the actual normalization numerics, per-channel weight/bias application
and eps consumption -- not just output shape or `isinstance`.

Two facts about the production code shape these tests:
  * `RMSNorm.forward` calls `detect_device()`; on CPU that helper returns
    "gpu" (paddle.get_device() == "cpu" falls through to the else branch).
    With the default `fuse_rms_norm=True` the forward would therefore dispatch
    to the GPU-only `fused_rms_norm_ext` kernel. To exercise the numeric path
    on CPU we must set `fuse_rms_norm=False`; the fused/GPU path is covered by
    a separate test that skips unless a CUDA device is present.
  * `Norm.create` reads `has_bias` from `config.get("use_bias", ...)` but never
    forwards an effect downstream (LayerNorm always carries a bias, RMSNorm
    never does). `has_bias` is effectively a dead parameter, so no meaningful
    behavioral assertion is made on it here.
"""

import unittest

import numpy as np
import paddle

from paddlefleet.nn.norm import LayerNorm, Norm, RMSNorm
from paddlefleet.transformers import LlamaConfig

# Fixed, fully distinct input: distinct rows (catches cross-row leakage / wrong
# reduction axis) and distinct values per position (catches transposition).
_HID = 4
_X = np.array(
    [[1.0, 2.0, 3.0, 4.0], [-2.0, 0.5, -1.5, 3.0]],
    dtype=np.float32,
)
# Distinct per-channel weight: catches weight applied along the wrong axis or
# in the wrong order.
_W = np.array([1.0, 2.0, 0.5, 3.0], dtype=np.float32)


def _rms_norm_reference(x, weight, eps):
    """Independent RMSNorm: y_i = x_i / sqrt(mean_j(x_j^2) + eps) * w_i."""
    x = np.asarray(x, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    mean_square = np.mean(x * x, axis=-1, keepdims=True)
    inv_rms = 1.0 / np.sqrt(mean_square + eps)
    return x * inv_rms * weight


def _layer_norm_reference(x, weight, bias, eps):
    """Independent LayerNorm over the last dim with biased variance."""
    x = np.asarray(x, dtype=np.float64)
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    normed = (x - mean) / np.sqrt(var + eps)
    return normed * np.asarray(weight, dtype=np.float64) + np.asarray(
        bias, dtype=np.float64
    )


def _make_config(hidden_size=16, rms_norm_eps=1e-6, fuse_rms_norm=False, **kw):
    config = LlamaConfig()
    config.hidden_size = hidden_size
    config.rms_norm_eps = rms_norm_eps
    # Force the manual (CPU-executable) RMSNorm path; see module docstring.
    config.fuse_rms_norm = fuse_rms_norm
    for key, value in kw.items():
        setattr(config, key, value)
    return config


def _set_param(param, values):
    param.set_value(paddle.to_tensor(np.asarray(values), dtype=param.dtype))


class TestRMSNorm(unittest.TestCase):
    """Numeric behavior of the real RMSNorm manual path on CPU."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_matches_hand_derived_reference(self):
        norm = RMSNorm(_make_config(), hidden_size=_HID, norm_eps=1e-6)
        _set_param(norm.weight, _W)
        out = norm(paddle.to_tensor(_X)).numpy()
        expected = _rms_norm_reference(_X, _W, 1e-6)
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)
        # Distinct weight must actually shape the output: it cannot equal the
        # unit-weight normalization.
        unit = _rms_norm_reference(_X, np.ones(_HID), 1e-6)
        self.assertGreater(np.abs(expected - unit).max(), 1e-2)

    def test_unit_rms_invariant(self):
        # With weight == 1 and negligible eps, each row of the output has
        # root-mean-square 1. Fails if mean-normalization is used instead.
        norm = RMSNorm(_make_config(), hidden_size=_HID, norm_eps=1e-8)
        out = norm(paddle.to_tensor(_X)).numpy().astype(np.float64)
        mean_square = np.mean(out * out, axis=-1)
        np.testing.assert_allclose(
            mean_square, np.ones_like(mean_square), rtol=1e-4, atol=1e-4
        )

    def test_eps_is_consumed(self):
        x = paddle.to_tensor(_X)
        big = RMSNorm(_make_config(), hidden_size=_HID, norm_eps=10.0)
        small = RMSNorm(_make_config(), hidden_size=_HID, norm_eps=1e-6)
        out_big = big(x).numpy()
        np.testing.assert_allclose(
            out_big,
            _rms_norm_reference(_X, np.ones(_HID), 10.0),
            rtol=1e-5,
            atol=1e-6,
        )
        # A large eps must materially damp the output vs a tiny eps.
        self.assertGreater(np.abs(out_big - small(x).numpy()).max(), 1e-2)

    def test_constructor_overrides(self):
        norm = RMSNorm(
            _make_config(hidden_size=16), hidden_size=8, norm_eps=1e-3
        )
        self.assertEqual(norm.hidden_size, 8)
        self.assertAlmostEqual(norm.variance_epsilon, 1e-3)
        self.assertEqual(list(norm.weight.shape), [8])
        # Without overrides, hidden_size / eps come from the config.
        default = RMSNorm(_make_config(hidden_size=16, rms_norm_eps=2e-6))
        self.assertEqual(default.hidden_size, 16)
        self.assertAlmostEqual(default.variance_epsilon, 2e-6)
        self.assertEqual(list(default.weight.shape), [16])

    def test_fused_path_matches_reference_on_gpu(self):
        # The fused kernel is GPU-only; on CPU detect_device() reports "gpu"
        # so the default fuse_rms_norm=True path would call it and fail. Only
        # meaningful with a real CUDA device.
        if paddle.device.cuda.device_count() == 0:
            self.skipTest(
                "fused_rms_norm_ext is a GPU kernel; not executable on CPU"
            )
        paddle.set_device("gpu")
        norm = RMSNorm(_make_config(fuse_rms_norm=True), hidden_size=_HID)
        _set_param(norm.weight, _W)
        out = norm(paddle.to_tensor(_X)).numpy()
        np.testing.assert_allclose(
            out,
            _rms_norm_reference(_X, _W, norm.variance_epsilon),
            rtol=1e-3,
            atol=1e-3,
        )


class TestLayerNorm(unittest.TestCase):
    """Numeric behavior of the real LayerNorm on CPU."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_matches_hand_derived_reference(self):
        # Default gamma == 1, beta == 0.
        norm = LayerNorm(_make_config(), hidden_size=_HID, norm_eps=1e-5)
        out = norm(paddle.to_tensor(_X)).numpy()
        expected = _layer_norm_reference(
            _X, np.ones(_HID), np.zeros(_HID), 1e-5
        )
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)

    def test_weight_and_bias_applied(self):
        gamma = np.array([1.0, 2.0, 0.5, 3.0], dtype=np.float32)
        beta = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
        norm = LayerNorm(_make_config(), hidden_size=_HID, norm_eps=1e-5)
        _set_param(norm.weight, gamma)
        _set_param(norm.bias, beta)
        out = norm(paddle.to_tensor(_X)).numpy()
        expected = _layer_norm_reference(_X, gamma, beta, 1e-5)
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)
        # The affine transform must actually change the result.
        plain = _layer_norm_reference(_X, np.ones(_HID), np.zeros(_HID), 1e-5)
        self.assertGreater(np.abs(expected - plain).max(), 1e-2)

    def test_eps_is_consumed(self):
        x = paddle.to_tensor(_X)
        big = LayerNorm(_make_config(), hidden_size=_HID, norm_eps=5.0)
        small = LayerNorm(_make_config(), hidden_size=_HID, norm_eps=1e-5)
        out_big = big(x).numpy()
        np.testing.assert_allclose(
            out_big,
            _layer_norm_reference(_X, np.ones(_HID), np.zeros(_HID), 5.0),
            rtol=1e-5,
            atol=1e-5,
        )
        self.assertGreater(np.abs(out_big - small(x).numpy()).max(), 1e-2)

    def test_constructor_overrides(self):
        norm = LayerNorm(
            _make_config(hidden_size=16), hidden_size=32, norm_eps=1e-8
        )
        self.assertEqual(norm.hidden_size, 32)
        self.assertAlmostEqual(norm.norm_eps, 1e-8)
        self.assertEqual(list(norm.weight.shape), [32])
        # Without overrides, hidden_size / eps come from config.get(...).
        cfg = _make_config(hidden_size=16)
        cfg.norm_eps = 1e-4
        default = LayerNorm(cfg)
        self.assertEqual(default.hidden_size, 16)
        self.assertAlmostEqual(default.norm_eps, 1e-4)


class TestNormFactory(unittest.TestCase):
    """Norm.create dispatch, parameter forwarding and error contract."""

    def setUp(self):
        paddle.set_device("cpu")

    def test_default_is_rms_and_computes_rms(self):
        norm = Norm.create(_make_config(), hidden_size=_HID, norm_eps=1e-6)
        self.assertIsInstance(norm, RMSNorm)
        _set_param(norm.weight, _W)
        out = norm(paddle.to_tensor(_X)).numpy()
        np.testing.assert_allclose(
            out, _rms_norm_reference(_X, _W, 1e-6), rtol=1e-5, atol=1e-6
        )

    def test_layer_norm_computes_layer_norm(self):
        norm = Norm.create(
            _make_config(),
            hidden_size=_HID,
            norm_eps=1e-5,
            norm_type="layer_norm",
        )
        self.assertIsInstance(norm, LayerNorm)
        out = norm(paddle.to_tensor(_X)).numpy()
        np.testing.assert_allclose(
            out,
            _layer_norm_reference(_X, np.ones(_HID), np.zeros(_HID), 1e-5),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_forwards_eps_to_computation(self):
        x = paddle.to_tensor(_X)
        norm = Norm.create(_make_config(), hidden_size=_HID, norm_eps=7.0)
        out = norm(x).numpy()
        np.testing.assert_allclose(
            out,
            _rms_norm_reference(_X, np.ones(_HID), 7.0),
            rtol=1e-5,
            atol=1e-6,
        )
        tiny = Norm.create(_make_config(), hidden_size=_HID, norm_eps=1e-6)
        self.assertGreater(np.abs(out - tiny(x).numpy()).max(), 1e-2)

    def test_forwards_hidden_size(self):
        norm = Norm.create(_make_config(), hidden_size=8)
        self.assertEqual(norm.hidden_size, 8)
        self.assertEqual(list(norm.weight.shape), [8])

    def test_invalid_norm_type_raises(self):
        with self.assertRaises(KeyError):
            Norm.create(_make_config(), norm_type="does_not_exist")


if __name__ == "__main__":
    unittest.main()
