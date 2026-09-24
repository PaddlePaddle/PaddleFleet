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

"""Behavior tests for paddlefleet.utils.initializer.

These tests exercise the real initializer APIs on CPU and check the actual
statistical / deterministic properties the weight-init routines promise:
exact constant fills, hand-derived fan/gain arithmetic, empirical mean and
std of the produced tensors against independently derived targets, uniform
range bounds, and seed reproducibility. Expected values are derived by hand
from the initialization math, never by calling the function under test.
"""

import math
import unittest

import numpy as np
import paddle
from paddle import nn

from paddlefleet.utils.initializer import (
    _calculate_correct_fan,
    _calculate_fan_in_and_fan_out,
    _calculate_gain,
    _no_grad_fill_,
    _no_grad_normal_,
    _no_grad_uniform_,
    bias_init_with_prob,
    constant_,
    conv_init_,
    kaiming_normal_,
    kaiming_uniform_,
    linear_init_,
    normal_,
    ones_,
    reset_initialized_parameter,
    uniform_,
    vector_,
    xavier_normal_,
    xavier_uniform_,
    zeros_,
)

SEED = 20240117


def _np(tensor):
    return np.asarray(tensor.numpy(), dtype=np.float64)


def _mean(tensor):
    return float(_np(tensor).mean())


def _std(tensor):
    return float(_np(tensor).std())


class _StatMixin:
    def assertStdClose(self, tensor, target, rtol=0.06):
        emp = _std(tensor)
        self.assertAlmostEqual(
            emp,
            target,
            delta=max(rtol * target, 5e-3),
            msg=f"empirical std {emp:.6f} vs target {target:.6f}",
        )


class TestConstantFills(unittest.TestCase):
    def test_no_grad_fill_exact(self):
        t = paddle.zeros([7])
        out = _no_grad_fill_(t, 3.14)
        # Every element must equal the fill value, not merely be non-zero.
        np.testing.assert_allclose(_np(out), np.full(7, 3.14), atol=1e-6)

    def test_constant_exact(self):
        t = paddle.zeros([3, 4])
        out = constant_(t, 42.0)
        np.testing.assert_array_equal(_np(out), np.full((3, 4), 42.0))

    def test_ones_exact(self):
        t = paddle.zeros([5])
        out = ones_(t)
        np.testing.assert_array_equal(_np(out), np.ones(5))

    def test_zeros_exact(self):
        t = paddle.full([5], 9.0)
        out = zeros_(t)
        np.testing.assert_array_equal(_np(out), np.zeros(5))

    def test_vector_sets_exact_values_in_order(self):
        t = paddle.zeros([4])
        out = vector_(t, [1.0, 2.0, 3.0, 4.0])
        # Order matters: catches reversed / scattered assignment.
        np.testing.assert_array_equal(_np(out), np.array([1.0, 2.0, 3.0, 4.0]))


class TestBiasInitWithProb(unittest.TestCase):
    def test_default_prob(self):
        # p=0.01 -> (1-p)/p = 0.99/0.01 = 99 -> -ln(99), derived by hand.
        self.assertAlmostEqual(
            bias_init_with_prob(), -math.log(99.0), places=10
        )

    def test_prob_half_is_zero(self):
        # p=0.5 -> odds 1 -> -ln(1) = 0.
        self.assertAlmostEqual(bias_init_with_prob(0.5), 0.0, places=12)

    def test_prob_tenth(self):
        # p=0.1 -> 0.9/0.1 = 9 -> -ln(9).
        self.assertAlmostEqual(
            bias_init_with_prob(0.1), -math.log(9.0), places=10
        )


class TestFanCalculations(unittest.TestCase):
    def test_2d_default_order(self):
        t = paddle.zeros([64, 128])
        self.assertEqual(_calculate_fan_in_and_fan_out(t), (128, 64))

    def test_2d_reverse_order(self):
        t = paddle.zeros([64, 128])
        self.assertEqual(
            _calculate_fan_in_and_fan_out(t, reverse=True), (64, 128)
        )

    def test_4d_receptive_field(self):
        t = paddle.zeros([64, 128, 3, 3])
        # receptive field 3*3=9: fan_in=128*9, fan_out=64*9.
        self.assertEqual(_calculate_fan_in_and_fan_out(t), (1152, 576))

    def test_3d_receptive_field(self):
        t = paddle.zeros([8, 4, 5])
        # receptive field 5: fan_in=shape[1]*5=20, fan_out=shape[0]*5=40.
        self.assertEqual(_calculate_fan_in_and_fan_out(t), (20, 40))

    def test_1d_raises(self):
        with self.assertRaises(ValueError):
            _calculate_fan_in_and_fan_out(paddle.zeros([10]))

    def test_correct_fan_in(self):
        self.assertEqual(
            _calculate_correct_fan(paddle.zeros([64, 128]), "fan_in"), 128
        )

    def test_correct_fan_out(self):
        self.assertEqual(
            _calculate_correct_fan(paddle.zeros([64, 128]), "fan_out"), 64
        )

    def test_correct_fan_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            _calculate_correct_fan(paddle.zeros([64, 128]), "diagonal")


class TestCalculateGain(unittest.TestCase):
    def test_linear_family_and_sigmoid(self):
        for name in ("linear", "conv1d", "conv2d", "conv3d", "sigmoid"):
            self.assertEqual(_calculate_gain(name), 1)

    def test_tanh(self):
        self.assertAlmostEqual(_calculate_gain("tanh"), 5.0 / 3.0, places=12)

    def test_relu(self):
        self.assertAlmostEqual(
            _calculate_gain("relu"), math.sqrt(2.0), places=12
        )

    def test_leaky_relu_default_slope(self):
        # default negative_slope=0.01 -> sqrt(2/(1+0.01^2)).
        expected = math.sqrt(2.0 / (1.0 + 0.0001))
        self.assertAlmostEqual(
            _calculate_gain("leaky_relu"), expected, places=12
        )

    def test_leaky_relu_custom_slope(self):
        expected = math.sqrt(2.0 / (1.0 + 0.04))
        self.assertAlmostEqual(
            _calculate_gain("leaky_relu", 0.2), expected, places=12
        )

    def test_selu(self):
        self.assertAlmostEqual(_calculate_gain("selu"), 0.75, places=12)

    def test_unsupported_raises(self):
        with self.assertRaises(ValueError):
            _calculate_gain("gelu")

    def test_leaky_relu_bool_param_rejected(self):
        # bool is excluded from the numeric branch and must raise.
        with self.assertRaises(ValueError):
            _calculate_gain("leaky_relu", True)


class TestUniformDistribution(_StatMixin, unittest.TestCase):
    def setUp(self):
        paddle.seed(SEED)

    def test_range_mean_and_std(self):
        a, b = -2.0, 2.0
        t = paddle.zeros([100000])
        out = uniform_(t, a, b)
        arr = _np(out)
        self.assertGreaterEqual(arr.min(), a - 1e-6)
        self.assertLessEqual(arr.max(), b + 1e-6)
        # Uniform[a,b]: mean=(a+b)/2=0, std=(b-a)/sqrt(12).
        self.assertAlmostEqual(_mean(out), 0.0, delta=0.05)
        self.assertStdClose(out, (b - a) / math.sqrt(12.0))

    def test_no_grad_uniform_matches_uniform(self):
        a, b = 0.0, 1.0
        t = paddle.zeros([100000])
        out = _no_grad_uniform_(t, a, b)
        arr = _np(out)
        self.assertGreaterEqual(arr.min(), a - 1e-6)
        self.assertLessEqual(arr.max(), b + 1e-6)
        self.assertAlmostEqual(_mean(out), 0.5, delta=0.02)
        self.assertStdClose(out, (b - a) / math.sqrt(12.0))

    def test_same_seed_reproduces(self):
        paddle.seed(777)
        first = _np(uniform_(paddle.zeros([256]), -1.0, 1.0)).copy()
        paddle.seed(777)
        second = _np(uniform_(paddle.zeros([256]), -1.0, 1.0)).copy()
        np.testing.assert_array_equal(first, second)


class TestNormalDistribution(_StatMixin, unittest.TestCase):
    def setUp(self):
        paddle.seed(SEED)

    def test_mean_and_std(self):
        t = paddle.zeros([100000])
        normal_(t, mean=5.0, std=2.0)
        self.assertAlmostEqual(_mean(t), 5.0, delta=0.05)
        self.assertStdClose(t, 2.0)

    def test_no_grad_normal_zero_mean(self):
        t = paddle.zeros([100000])
        _no_grad_normal_(t, mean=0.0, std=0.5)
        self.assertAlmostEqual(_mean(t), 0.0, delta=0.02)
        self.assertStdClose(t, 0.5)

    def test_same_seed_reproduces(self):
        paddle.seed(555)
        first = _np(_no_grad_normal_(paddle.zeros([256]), 0.0, 1.0)).copy()
        paddle.seed(555)
        second = _np(_no_grad_normal_(paddle.zeros([256]), 0.0, 1.0)).copy()
        np.testing.assert_array_equal(first, second)


class TestXavierInit(_StatMixin, unittest.TestCase):
    def setUp(self):
        paddle.seed(SEED)

    def test_xavier_uniform_bound_and_std(self):
        # shape [200,300]: fan_in=300, fan_out=200 (default order).
        fan_in, fan_out = 300, 200
        std_target = math.sqrt(2.0 / (fan_in + fan_out))
        k = math.sqrt(3.0) * std_target
        out = xavier_uniform_(paddle.zeros([200, 300]))
        arr = _np(out)
        self.assertLessEqual(arr.max(), k + 1e-6)
        self.assertGreaterEqual(arr.min(), -k - 1e-6)
        # Uniform[-k,k] has std = k/sqrt(3) = std_target.
        self.assertStdClose(out, std_target)

    def test_xavier_normal_std(self):
        fan_in, fan_out = 300, 200
        std_target = math.sqrt(2.0 / (fan_in + fan_out))
        out = xavier_normal_(paddle.zeros([200, 300]))
        self.assertAlmostEqual(_mean(out), 0.0, delta=0.01)
        self.assertStdClose(out, std_target)

    def test_gain_scales_std(self):
        fan_in, fan_out = 300, 200
        std_target = 2.0 * math.sqrt(2.0 / (fan_in + fan_out))
        out = xavier_normal_(paddle.zeros([200, 300]), gain=2.0)
        self.assertStdClose(out, std_target)


class TestKaimingInit(_StatMixin, unittest.TestCase):
    def setUp(self):
        paddle.seed(SEED)

    def test_kaiming_uniform_default(self):
        # default: mode=fan_in, nonlinearity=leaky_relu, a=0. The production
        # _calculate_gain passes a=0 as the negative_slope, so the slope is 0
        # (not the 0.01 leaky default) and gain = sqrt(2 / (1 + 0**2)) = sqrt(2).
        fan_in = 400
        gain = math.sqrt(2.0)
        std_target = gain / math.sqrt(fan_in)
        k = math.sqrt(3.0) * std_target
        out = kaiming_uniform_(paddle.zeros([256, 400]))
        arr = _np(out)
        self.assertLessEqual(arr.max(), k + 1e-6)
        self.assertGreaterEqual(arr.min(), -k - 1e-6)
        self.assertStdClose(out, std_target)

    def test_kaiming_normal_relu(self):
        fan_in = 400
        gain = math.sqrt(2.0)  # relu
        std_target = gain / math.sqrt(fan_in)
        out = kaiming_normal_(paddle.zeros([256, 400]), nonlinearity="relu")
        self.assertAlmostEqual(_mean(out), 0.0, delta=0.01)
        self.assertStdClose(out, std_target)


class TestLinearConvInit(_StatMixin, unittest.TestCase):
    def setUp(self):
        paddle.seed(SEED)

    def test_linear_init_bound_and_distribution(self):
        # paddle Linear weight is [in_features, out_features]; bound uses
        # shape[0] = in_features = 128 -> 1/sqrt(128).
        linear = nn.Linear(128, 64)
        linear_init_(linear)
        bound = 1.0 / math.sqrt(128)
        w = _np(linear.weight)
        self.assertLessEqual(w.max(), bound + 1e-6)
        self.assertGreaterEqual(w.min(), -bound - 1e-6)
        # Uniform[-bound,bound] std = bound/sqrt(3).
        self.assertStdClose(linear.weight, bound / math.sqrt(3.0))
        self.assertAlmostEqual(_mean(linear.weight), 0.0, delta=0.02)
        b = _np(linear.bias)
        self.assertLessEqual(b.max(), bound + 1e-6)
        self.assertGreaterEqual(b.min(), -bound - 1e-6)

    def test_conv_init_bound_and_distribution(self):
        conv = nn.Conv2D(4, 16, kernel_size=3)
        conv_init_(conv)
        # weight shape [16,4,3,3]; bound = 1/sqrt(prod(shape[1:])) = 1/sqrt(36).
        bound = 1.0 / math.sqrt(4 * 3 * 3)
        w = _np(conv.weight)
        self.assertLessEqual(w.max(), bound + 1e-6)
        self.assertGreaterEqual(w.min(), -bound - 1e-6)
        self.assertStdClose(conv.weight, bound / math.sqrt(3.0))

    def test_conv_init_bias_when_present(self):
        conv = nn.Conv2D(4, 16, kernel_size=3, bias_attr=True)
        conv_init_(conv)
        bound = 1.0 / math.sqrt(4 * 3 * 3)
        self.assertIsNotNone(conv.bias)
        b = _np(conv.bias)
        self.assertLessEqual(b.max(), bound + 1e-6)
        self.assertGreaterEqual(b.min(), -bound - 1e-6)


class TestResetInitializedParameter(_StatMixin, unittest.TestCase):
    def setUp(self):
        paddle.seed(SEED)

    def test_layernorm_reset_to_constants(self):
        ln = nn.LayerNorm(32)
        # Corrupt params so the reset must actively overwrite them.
        zeros_(ln.weight)
        ones_(ln.bias)
        reset_initialized_parameter(ln, include_self=True)
        np.testing.assert_array_equal(_np(ln.weight), np.ones(32))
        np.testing.assert_array_equal(_np(ln.bias), np.zeros(32))

    def test_batchnorm_reset_to_constants(self):
        bn = nn.BatchNorm2D(16)
        zeros_(bn.weight)
        ones_(bn.bias)
        reset_initialized_parameter(bn, include_self=True)
        np.testing.assert_array_equal(_np(bn.weight), np.ones(16))
        np.testing.assert_array_equal(_np(bn.bias), np.zeros(16))

    def test_linear_reset_bound(self):
        linear = nn.Linear(50, 20)
        constant_(linear.weight, 5.0)
        reset_initialized_parameter(linear, include_self=True)
        # k = sqrt(1/shape[0]) = sqrt(1/50).
        k = math.sqrt(1.0 / 50.0)
        w = _np(linear.weight)
        self.assertLessEqual(w.max(), k + 1e-6)
        self.assertGreaterEqual(w.min(), -k - 1e-6)
        # No longer the corrupted constant, and matches uniform std.
        self.assertNotAlmostEqual(w.max(), 5.0, places=3)
        self.assertStdClose(linear.weight, k / math.sqrt(3.0))

    def test_conv2d_reset_bound(self):
        conv = nn.Conv2D(4, 16, kernel_size=3)
        constant_(conv.weight, 5.0)
        reset_initialized_parameter(conv, include_self=True)
        # k = sqrt(groups / (in * kh * kw)) = sqrt(1/36).
        k = math.sqrt(1.0 / (4 * 3 * 3))
        w = _np(conv.weight)
        self.assertLessEqual(w.max(), k + 1e-6)
        self.assertGreaterEqual(w.min(), -k - 1e-6)
        self.assertStdClose(conv.weight, k / math.sqrt(3.0))

    def test_embedding_reset_normal(self):
        emb = nn.Embedding(1000, 64)
        zeros_(emb.weight)
        reset_initialized_parameter(emb, include_self=True)
        # Embedding reset uses normal(mean=0, std=1).
        self.assertAlmostEqual(_mean(emb.weight), 0.0, delta=0.03)
        self.assertStdClose(emb.weight, 1.0)


if __name__ == "__main__":
    unittest.main()
