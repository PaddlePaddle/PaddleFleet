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

"""Behavior tests for paddlefleet.quantization.quantization_linear.

These tests exercise the real production entries (``QuantMapping``,
``QuantizationLinear.__init__``, ``quant_weight_forward``, ``dequant_weight``
and ``QuantizationLinear.forward``) rather than re-deriving their logic inside
the test.

The numerical assertions do NOT call the function under test to build the
expected value. Instead they build an *independent* reference from the
full-precision weight:

    reference = x @ W_fp32.T (+ bias)

and require the quantized linear forward to match it within a *hand-derived
quantization error bound*. For symmetric per-output-channel int8 weight-only
quantization the dequantized weight satisfies

    |dequant(W)[o, k] - W[o, k]| <= step[o] / 2,   step[o] = max_k|W[o, k]| / 127

so the per-output-element forward error is bounded by

    |y_q[..., o] - y_ref[..., o]| <= (step[o] / 2) * sum_k |x[..., k]|

The step is computed by hand directly from the float weight, so the bound is
independent of how Paddle stores its scale tensor.

``weight_only_linear`` / ``weight_quantize`` / ``weight_dequantize`` are fused
CUDA kernels that are only available on GPU. On a CPU-only (无卡) box the
numerical tests are skipped with an explicit reason; the pure-Python
construction / mapping contracts still run on CPU.
"""

import unittest

import numpy as np
import paddle

from paddlefleet.quantization.quantization_config import QuantizationConfig
from paddlefleet.quantization.quantization_linear import (
    QuantizationLinear,
    QuantMapping,
    dequant_weight,
    quant_weight_forward,
)

GPU_AVAILABLE = (
    paddle.device.is_compiled_with_cuda()
    and paddle.device.cuda.device_count() > 0
)
GPU_SKIP_REASON = (
    "weight_only_linear / weight_quantize / weight_dequantize are GPU-only "
    "CUDA kernels; the numerical quant-linear behavior cannot be validated on "
    "a CPU-only (无卡) machine and must not be faked with a CPU stand-in."
)


class TestQuantMapping(unittest.TestCase):
    """The algo -> (quant_dtype, bit) contract consumed by QuantizationLinear."""

    def test_exact_dtype_and_bit_per_algo(self):
        # QuantizationLinear.__init__ unpacks (quant_dtype, quant_weight_bit)
        # from this table and uses the bit width to size the int4-packed weight
        # ([out // 2, in]) vs the int8 weight ([out, in]). Assert the exact
        # values, not merely that the keys exist.
        self.assertEqual(QuantMapping["weight_only_int8"], ("int8", 8))
        self.assertEqual(QuantMapping["weight_only_int4"], ("int4", 4))
        self.assertEqual(QuantMapping["llm.int8"], ("int8", 8))
        self.assertEqual(QuantMapping["fp4"], ("fp4", 4))
        self.assertEqual(QuantMapping["nf4"], ("nf4", 4))
        self.assertEqual(QuantMapping["a8w8linear"], ("int8", 8))
        self.assertEqual(QuantMapping["a8w4linear"], ("int8", 8))
        self.assertEqual(QuantMapping["fp8linear"], ("fp8", 8))

    def test_bit_width_matches_dtype_family(self):
        # Guard against a swap that leaves the tuple shape intact but breaks
        # the packing math (e.g. an int4 entry silently carrying bit=8).
        for algo, (dtype_str, bit) in QuantMapping.items():
            self.assertIsInstance(dtype_str, str, algo)
            self.assertIn(bit, (4, 8), algo)
            if "int4" in algo or dtype_str == "int4" or dtype_str == "fp4":
                self.assertEqual(bit, 4, algo)


def _weight_only_config():
    return QuantizationConfig(
        weight_quantize_algo="weight_only_int8", group_size=-1
    )


class TestQuantizationLinearConstruction(unittest.TestCase):
    """QuantizationLinear.__init__ must size parameters from the real config.

    Runs on CPU. Construction is wrapped in ``paddle.LazyGuard()`` because that
    is exactly how the real loader builds these layers
    (``model_utils.from_pretrained`` -> ``ContextManagers([no_init_weights,
    LazyGuard]) -> replace_with_quantization_linear``). Outside LazyGuard the
    ``create_parameter(dtype="int8")`` call eagerly runs Paddle's default
    XavierUniform initializer, whose ``uniform`` kernel is not registered for
    int8 and raises at construction time -- a path production never takes. Under
    LazyGuard the parameter metadata (shape/dtype) is still materialized, so the
    shape-derivation logic that distinguishes int8 from int4 packing is
    exercised genuinely (the constructor is never mocked).
    """

    def test_int8_parameter_layout(self):
        with paddle.LazyGuard():
            layer = QuantizationLinear(
                in_features=8,
                out_features=4,
                quantization_config=_weight_only_config(),
                weight_quantize_algo="weight_only_int8",
                dtype="float16",
            )
        # int8 weight is stored transposed as [out, in].
        self.assertEqual(layer.quant_weight.shape, [4, 8])
        self.assertEqual(layer.quant_weight.dtype, paddle.int8)
        # per-output-channel scale, kept in the compute dtype and frozen.
        self.assertEqual(layer.weight_scale.shape, [4])
        self.assertEqual(layer.weight_scale.dtype, paddle.float16)
        self.assertTrue(layer.weight_scale.stop_gradient)
        # default bias present, one per output feature, in compute dtype.
        self.assertIsNotNone(layer.bias)
        self.assertEqual(layer.bias.shape, [4])
        self.assertEqual(layer.bias.dtype, paddle.float16)
        self.assertEqual(layer.quant_dtype, "int8")
        self.assertEqual(layer.quant_weight_bit, 8)
        # the algo is stamped onto the weight for downstream dispatch.
        self.assertEqual(
            layer.quant_weight.weight_quantize_algo, "weight_only_int8"
        )

    def test_int4_weight_is_packed_two_per_byte(self):
        config = QuantizationConfig(
            weight_quantize_algo="weight_only_int4", group_size=-1
        )
        with paddle.LazyGuard():
            layer = QuantizationLinear(
                in_features=8,
                out_features=4,
                quantization_config=config,
                weight_quantize_algo="weight_only_int4",
                dtype="float16",
            )
        # int4 packs two values into one int8 -> first dim is out // 2.
        self.assertEqual(layer.quant_weight.shape, [2, 8])
        self.assertEqual(layer.quant_weight.dtype, paddle.int8)
        # scale is still per full output channel, not per packed row.
        self.assertEqual(layer.weight_scale.shape, [4])
        self.assertEqual(layer.quant_weight_bit, 4)

    def test_bias_attr_false_disables_bias(self):
        with paddle.LazyGuard():
            layer = QuantizationLinear(
                in_features=8,
                out_features=4,
                quantization_config=_weight_only_config(),
                weight_quantize_algo="weight_only_int8",
                dtype="float16",
                bias_attr=False,
            )
        self.assertIsNone(layer.bias)

    def test_groupwise_weightonly_is_rejected(self):
        # group_size != -1 hits an explicit NotImplementedError guard; verify
        # the real guard fires rather than swallowing it.
        config = QuantizationConfig(
            weight_quantize_algo="weight_only_int8", group_size=64
        )
        with self.assertRaises(NotImplementedError), paddle.LazyGuard():
            QuantizationLinear(
                in_features=8,
                out_features=4,
                quantization_config=config,
                weight_quantize_algo="weight_only_int8",
                dtype="float16",
            )

    def test_unknown_algo_is_rejected(self):
        config = _weight_only_config()
        with self.assertRaises(KeyError):
            # QuantMapping[...] lookup fails before any parameter is created.
            QuantizationLinear(
                in_features=8,
                out_features=4,
                quantization_config=config,
                weight_quantize_algo="not_a_real_algo",
                dtype="float16",
            )


def _quant_step_from_weight(w_fp32):
    """Hand-derived symmetric per-output-channel int8 step: max|W| / 127."""
    max_abs = np.max(np.abs(w_fp32), axis=1)  # [out]
    return max_abs / 127.0


def _forward_error_bound(x_fp32, step, safety=2.0):
    """Per-output-element quant error bound for y = x @ W.T.

    bound[m, o] = (step[o] / 2) * sum_k |x[m, k]|, times a small safety factor
    that absorbs ties-away rounding and fp16 accumulation noise.
    """
    abs_x_row_sum = np.sum(np.abs(x_fp32), axis=1, keepdims=True)  # [M, 1]
    return safety * (step[None, :] / 2.0) * abs_x_row_sum  # [M, out]


@unittest.skipUnless(GPU_AVAILABLE, GPU_SKIP_REASON)
class TestQuantWeightForwardNumeric(unittest.TestCase):
    """quant_weight_forward (weight_only_int8) vs full-precision reference."""

    def _run(self, with_bias):
        paddle.seed(2024)
        in_features, out_features, tokens = 128, 64, 8
        # weight_quantize consumes an [in, out] weight (its first dim must be
        # divisible by 64) and returns the transposed [out, in] int8 payload
        # that weight_only_linear expects; keep w in that [in, out] layout so
        # the whole pipeline reconstructs y = x @ w.
        w = paddle.randn([in_features, out_features], dtype="float32")
        x = paddle.randn([tokens, in_features], dtype="float32")
        bias = (
            paddle.randn([out_features], dtype="float32") if with_bias else None
        )

        w_np = w.numpy()
        x_np = x.numpy()
        bias_np = bias.numpy() if with_bias else 0.0

        # Independent reference: full-precision matmul (hand-computed in fp32).
        ref = x_np @ w_np + bias_np  # [tokens, out]

        # Prepare the quantized weight with the Paddle utility (NOT the code
        # under test) so the kernel receives its expected packed layout.
        q_weight, w_scale = paddle.nn.quant.weight_quantize(
            w.cast("float16"), algo="weight_only_int8"
        )

        out = quant_weight_forward(
            x=x.cast("float16"),
            quant_weight=q_weight,
            bias=bias.cast("float16") if with_bias else None,
            weight_scale=w_scale,
            quant_state=None,
            quant_dtype="int8",
            quantization_config=_weight_only_config(),
            weight_quantize_algo="weight_only_int8",
            dtype="float16",
        )
        actual = out.astype("float32").numpy()

        # step is per output channel: max over the input axis of the [in, out]
        # weight, i.e. axis=1 of its [out, in] transpose.
        step = _quant_step_from_weight(w_np.T)
        bound = _forward_error_bound(x_np, step)
        allowed = bound + 2e-2 * np.abs(ref) + 5e-3

        diff = np.abs(actual - ref)
        self.assertTrue(
            np.all(diff <= allowed),
            f"max excess={np.max(diff - allowed):.4g}, "
            f"max diff={diff.max():.4g}",
        )
        # Negative control: the bound is tight enough to reject a zeroed output
        # (the reference is non-trivial), proving the assertion has teeth.
        self.assertFalse(np.all(np.abs(0.0 - ref) <= allowed))

    def test_forward_without_bias(self):
        self._run(with_bias=False)

    def test_forward_with_bias(self):
        self._run(with_bias=True)


@unittest.skipUnless(GPU_AVAILABLE, GPU_SKIP_REASON)
class TestDequantWeightNumeric(unittest.TestCase):
    """dequant_weight round-trips the int8 weight back near the fp32 weight."""

    def test_dequant_matches_float_weight_within_step(self):
        paddle.seed(7)
        in_features, out_features = 128, 32
        # weight_quantize wants an [in, out] weight (first dim divisible by 64);
        # weight_dequantize is its inverse and reconstructs the SAME [in, out]
        # orientation, so dq is compared directly against the original w.
        w = paddle.randn([in_features, out_features], dtype="float32")
        w_np = w.numpy()

        q_weight, w_scale = paddle.nn.quant.weight_quantize(
            w.cast("float16"), algo="weight_only_int8"
        )
        dq = dequant_weight(
            quant_weight=q_weight,
            quantization_config=_weight_only_config(),
            weight_quantize_algo="weight_only_int8",
            dtype="float16",
            weight_scale=w_scale,
            quant_state=None,
            input_shape=[in_features, out_features],
        )
        self.assertEqual(dq.dtype, paddle.float16)
        self.assertEqual(dq.shape, [in_features, out_features])

        dq_np = dq.astype("float32").numpy()
        # Independent reference is the ORIGINAL float weight; each element must
        # sit within half a quant step (max|W|/127 over the input axis, i.e. per
        # output channel) plus fp16 slack. step is per output channel so it
        # broadcasts along the [in, out] weight's output axis.
        step = _quant_step_from_weight(w_np.T)  # [out]
        allowed = (step[None, :] / 2.0) + 3e-3 + 1e-2 * np.abs(w_np)
        diff = np.abs(dq_np - w_np)
        self.assertTrue(
            np.all(diff <= allowed),
            f"max excess={np.max(diff - allowed):.4g}",
        )


@unittest.skipUnless(GPU_AVAILABLE, GPU_SKIP_REASON)
class TestQuantizationLinearForwardEndToEnd(unittest.TestCase):
    """QuantizationLinear.forward end-to-end vs full-precision reference."""

    def test_layer_forward_within_quant_bound(self):
        paddle.seed(123)
        in_features, out_features, tokens = 128, 64, 4
        # weight_quantize wants an [in, out] weight (first dim divisible by 64);
        # its transposed [out, in] payload matches layer.quant_weight's [out, in]
        # shape, and the layer reconstructs y = x @ w for this [in, out] w.
        w = paddle.randn([in_features, out_features], dtype="float32")
        x = paddle.randn([tokens, in_features], dtype="float32")

        w_np = w.numpy()
        x_np = x.numpy()

        # Built under LazyGuard exactly as the real loader does; otherwise the
        # int8 create_parameter eagerly runs XavierUniform, whose uniform kernel
        # is not registered for int8. set_value then materializes the lazy param
        # just like checkpoint loading.
        with paddle.LazyGuard():
            layer = QuantizationLinear(
                in_features=in_features,
                out_features=out_features,
                quantization_config=_weight_only_config(),
                weight_quantize_algo="weight_only_int8",
                dtype="float16",
                bias_attr=False,
            )
        q_weight, w_scale = paddle.nn.quant.weight_quantize(
            w.cast("float16"), algo="weight_only_int8"
        )
        layer.quant_weight.set_value(q_weight)
        layer.weight_scale.set_value(w_scale)

        with paddle.no_grad():
            out = layer(x.cast("float16"))
        actual = out.astype("float32").numpy()

        ref = x_np @ w_np  # no bias; w is [in, out] so y = x @ w
        step = _quant_step_from_weight(w_np.T)
        allowed = _forward_error_bound(x_np, step) + 2e-2 * np.abs(ref) + 5e-3

        diff = np.abs(actual - ref)
        self.assertTrue(
            np.all(diff <= allowed),
            f"max excess={np.max(diff - allowed):.4g}",
        )
        self.assertFalse(np.all(np.abs(2.0 * actual - ref) <= allowed))


if __name__ == "__main__":
    unittest.main()
