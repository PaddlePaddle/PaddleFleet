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

"""Unit tests for HyperEncoderRMSNorm."""

import types
import unittest

import paddle

from paddlefleet.models.hyperencoder.norm import (
    HyperEncoderRMSNorm,
)


def _cfg(**over):
    base = {
        "hidden_size": 8,
        "rms_norm_eps": 1e-6,
        "normalization": "RMSNorm",
        "layernorm_zero_centered_gamma": False,
        "persist_layer_norm": False,
        "params_dtype": paddle.float32,
    }
    base.update(over)
    return types.SimpleNamespace(**base)


class TestConstruction(unittest.TestCase):
    def test_defaults_from_config(self):
        norm = HyperEncoderRMSNorm(_cfg())
        self.assertEqual(norm.normalized_shape, 8)
        self.assertAlmostEqual(norm.variance_epsilon, 1e-6)
        self.assertEqual(norm.weight.shape, [8])

    def test_explicit_shape_and_eps(self):
        norm = HyperEncoderRMSNorm(_cfg(), normalized_shape=4, norm_eps=1e-5)
        self.assertEqual(norm.normalized_shape, 4)
        self.assertAlmostEqual(norm.variance_epsilon, 1e-5)

    def test_params_dtype_none_falls_back_to_default(self):
        # params_dtype None -> weight dtype defaults to paddle's default dtype.
        norm = HyperEncoderRMSNorm(_cfg(params_dtype=None))
        self.assertEqual(norm.weight.shape, [8])

    def test_non_rmsnorm_raises(self):
        with self.assertRaises(ValueError):
            HyperEncoderRMSNorm(_cfg(normalization="LayerNorm"))

    def test_zero_centered_gamma_raises(self):
        with self.assertRaises(ValueError):
            HyperEncoderRMSNorm(_cfg(layernorm_zero_centered_gamma=True))

    def test_persist_layer_norm_raises(self):
        with self.assertRaises(ValueError):
            HyperEncoderRMSNorm(_cfg(persist_layer_norm=True))

    def test_input_is_parallel_marks_weight(self):
        # Should not raise even when the SP helper is a no-op fallback.
        norm = HyperEncoderRMSNorm(_cfg(), input_is_parallel=True)
        self.assertIsNotNone(norm.weight)


class TestForwardBackward(unittest.TestCase):
    def test_forward_matches_reference(self):
        norm = HyperEncoderRMSNorm(_cfg())
        x = paddle.randn([2, 3, 8], dtype="float32")
        out = norm(x)
        ref = x * paddle.rsqrt(paddle.mean(x * x, axis=-1, keepdim=True) + 1e-6)
        self.assertTrue(paddle.allclose(out, ref, atol=1e-5))
        self.assertEqual(out.dtype, x.dtype)

    def test_backward_produces_grads(self):
        norm = HyperEncoderRMSNorm(_cfg())
        x = paddle.randn([4, 8], dtype="float32")
        x.stop_gradient = False
        out = norm(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(norm.weight.grad)
        self.assertEqual(x.grad.shape, x.shape)
        self.assertEqual(norm.weight.grad.shape, norm.weight.shape)

    def test_gradient_numeric_close(self):
        eps = 1e-6
        norm = HyperEncoderRMSNorm(_cfg(), norm_eps=eps)
        norm.weight.set_value(paddle.randn([8], dtype="float32"))
        x = paddle.randn([3, 8], dtype="float32")
        x.stop_gradient = False
        norm(x).sum().backward()
        # Cross-check against the same expression built from plain paddle ops.
        x2 = x.detach()
        x2.stop_gradient = False
        w2 = norm.weight.detach()
        w2.stop_gradient = False
        t1 = x2.astype("float32")
        t5 = paddle.rsqrt(paddle.mean(t1 * t1, axis=-1, keepdim=True) + eps)
        ref = (t1 * t5) * w2.astype("float32")
        ref.sum().backward()
        self.assertTrue(paddle.allclose(x.grad, x2.grad, atol=1e-4))
        self.assertTrue(paddle.allclose(norm.weight.grad, w2.grad, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
