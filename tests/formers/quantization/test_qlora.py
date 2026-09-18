# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

"""Behavior tests for paddlefleet.quantization.qlora.

These exercise the REAL quant/dequant/linear entry points and compare against
an INDEPENDENT hand-derived NF4 reference (level table + per-block absmax
nearest-level rounding). No production function is used to build the expected
values.

Environment note: the NF4 blockwise quant/dequant kernels come from the
``paddleslim_ops`` custom ops (quant_blockwise / dequant_blockwise) and the
fp8 double-quant path from ``paddleslim``. These kernels are GPU-only, so on a
无卡 (CPU-only) box or without paddleslim installed every case below is skipped
with an explicit reason rather than faked.
"""

import unittest

import numpy as np

try:
    import paddle

    HAS_PADDLE = True
except ImportError:  # paddle itself missing
    HAS_PADDLE = False

try:
    import paddleslim  # noqa: F401
    import paddleslim_ops  # noqa: F401

    HAS_PADDLESLIM = True
except ImportError:
    HAS_PADDLESLIM = False

_CUDA = HAS_PADDLE and paddle.is_compiled_with_cuda()
_SKIP = (not HAS_PADDLE) or (not HAS_PADDLESLIM) or (not _CUDA)
_SKIP_REASON = (
    "qlora NF4/fp8 blockwise kernels require paddle + paddleslim + "
    "paddleslim_ops on a CUDA device (GPU-only custom ops)"
)


# Standard NormalFloat4 (NF4) code values from the QLoRA definition, ascending.
# Used only to build an independent reference; production is never consulted.
_NF4_LEVELS = np.array(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ],
    dtype=np.float64,
)

# Half of the largest gap between adjacent NF4 levels: the worst-case
# normalized round-trip error of a correct nearest-level NF4 quantizer.
_NF4_MAX_NORM_ERR = float(np.diff(_NF4_LEVELS).max()) / 2.0


def _nf4_reference_dequantize(weight_np, block_size):
    """Independent NF4 round-trip on a numpy weight.

    Row-major flatten, split into blocks of ``block_size``, per-block absmax
    scale, map each normalized value to its NEAREST NF4 level, then rescale.
    Returns (dequantized_like_input, per_element_absmax).
    """
    flat = weight_np.astype(np.float64).reshape(-1)
    n = flat.shape[0]
    deq = np.empty_like(flat)
    absmax_per_elem = np.empty_like(flat)
    for start in range(0, n, block_size):
        block = flat[start : start + block_size]
        absmax = np.abs(block).max()
        if absmax == 0.0:
            deq[start : start + block_size] = 0.0
            absmax_per_elem[start : start + block_size] = 0.0
            continue
        normalized = block / absmax
        idx = np.abs(normalized[:, None] - _NF4_LEVELS[None, :]).argmin(axis=1)
        deq[start : start + block_size] = _NF4_LEVELS[idx] * absmax
        absmax_per_elem[start : start + block_size] = absmax
    return deq.reshape(weight_np.shape), absmax_per_elem.reshape(
        weight_np.shape
    )


@unittest.skipIf(_SKIP, _SKIP_REASON)
class TestQloraQuantizeDequantizeNumeric(unittest.TestCase):
    """quantize_dequantize must land on the NF4 grid, per independent ref."""

    def test_single_block_matches_nf4_grid(self):
        from paddlefleet.quantization.qlora import (
            qlora_weight_quantize_dequantize,
        )

        # 64 fp16-exact values (multiples of 1/16) -> exactly one block of 64.
        # absmax == 2.0; values form a UNIFORM grid so NF4 (non-uniform) must
        # actually move them: an identity quantizer would fail this test.
        base = (np.arange(64) - 32).astype(np.float32) * 0.0625
        weight_np = base.reshape(8, 8)
        weight = paddle.to_tensor(weight_np, dtype=paddle.float16).cuda()

        qdq = qlora_weight_quantize_dequantize(
            weight, quant_algo="nf4", block_size=64
        )
        got = qdq.astype("float32").numpy()

        ref, absmax = _nf4_reference_dequantize(weight_np, block_size=64)
        # fp16 storage of level*absmax is the only allowed discrepancy.
        np.testing.assert_allclose(got, ref, atol=8e-3, rtol=0.0)

        # Guard against a no-op quantizer: NF4 must introduce real error.
        max_err = np.abs(got - weight_np).max()
        self.assertGreater(max_err, 1e-2)

    def test_multiblock_error_within_nf4_bound(self):
        from paddlefleet.quantization.qlora import (
            qlora_weight_quantize_dequantize,
        )

        # 256 elements -> 4 blocks of 64 at the default block_size. fp16-exact
        # values (multiples of 1/8) with block-varying magnitude.
        weight_np = ((np.arange(256).reshape(16, 16) % 17) - 8).astype(
            np.float32
        ) * 0.125
        weight = paddle.to_tensor(weight_np, dtype=paddle.float16).cuda()

        qdq = qlora_weight_quantize_dequantize(
            weight, quant_algo="nf4", block_size=64
        )
        got = qdq.astype("float32").numpy()

        # Bound holds regardless of the kernel's internal block flatten order:
        # every element's error <= (max NF4 gap / 2) * (its block absmax),
        # and every block absmax <= global absmax.
        _, absmax = _nf4_reference_dequantize(weight_np, block_size=64)
        per_elem_bound = _NF4_MAX_NORM_ERR * absmax + 8e-3
        err = np.abs(got - weight_np)
        self.assertTrue(
            np.all(err <= per_elem_bound),
            f"max normalized error {(err / np.maximum(absmax, 1e-9)).max()} "
            f"exceeds NF4 bound {_NF4_MAX_NORM_ERR}",
        )
        # Real quantization happened (not an identity pass-through).
        self.assertGreater(err.max(), 1e-2)


@unittest.skipIf(_SKIP, _SKIP_REASON)
class TestQloraWeightLinear(unittest.TestCase):
    """qlora_weight_linear == x @ dequant(W) [+ bias], vs independent ref."""

    def _fixed_inputs(self):
        weight_np = (np.arange(64) - 32).astype(np.float32).reshape(
            8, 8
        ) * 0.0625
        x_np = ((np.arange(16).reshape(2, 8) % 5) - 2).astype(np.float32) * 0.25
        return weight_np, x_np

    def test_linear_matches_dequant_matmul(self):
        from paddlefleet.quantization.qlora import (
            qlora_weight_linear,
            qlora_weight_quantize,
        )

        weight_np, x_np = self._fixed_inputs()
        weight = paddle.to_tensor(weight_np, dtype=paddle.float16).cuda()
        x = paddle.to_tensor(x_np, dtype=paddle.float16).cuda()

        quant_weight, state = qlora_weight_quantize(
            weight=weight,
            quant_algo="nf4",
            double_quant=False,
            block_size=64,
            return_dict=False,
        )
        out = qlora_weight_linear(
            x=x,
            quant_weight=quant_weight,
            dtype=paddle.float16,
            state=state,
            quant_algo="nf4",
            block_size=64,
        )

        ref_w, _ = _nf4_reference_dequantize(weight_np, block_size=64)
        ref = x_np @ ref_w  # weight is [in_features, out_features]
        np.testing.assert_allclose(
            out.astype("float32").numpy(), ref, atol=5e-2, rtol=1e-2
        )

    def test_linear_bias_is_added(self):
        from paddlefleet.quantization.qlora import (
            qlora_weight_linear,
            qlora_weight_quantize,
        )

        weight_np, x_np = self._fixed_inputs()
        bias_np = (np.arange(8).astype(np.float32) - 4) * 0.5
        weight = paddle.to_tensor(weight_np, dtype=paddle.float16).cuda()
        x = paddle.to_tensor(x_np, dtype=paddle.float16).cuda()
        bias = paddle.to_tensor(bias_np, dtype=paddle.float16).cuda()

        quant_weight, state = qlora_weight_quantize(
            weight=weight,
            quant_algo="nf4",
            double_quant=False,
            block_size=64,
            return_dict=False,
        )
        out = qlora_weight_linear(
            x=x,
            quant_weight=quant_weight,
            dtype=paddle.float16,
            state=state,
            quant_algo="nf4",
            block_size=64,
            bias=bias,
        )

        ref_w, _ = _nf4_reference_dequantize(weight_np, block_size=64)
        ref = x_np @ ref_w + bias_np
        np.testing.assert_allclose(
            out.astype("float32").numpy(), ref, atol=5e-2, rtol=1e-2
        )
        # The bias term must be the difference between the two paths.
        out_nobias = qlora_weight_linear(
            x=x,
            quant_weight=quant_weight,
            dtype=paddle.float16,
            state=state,
            quant_algo="nf4",
            block_size=64,
        )
        delta = (out - out_nobias).astype("float32").numpy()
        np.testing.assert_allclose(
            delta,
            np.broadcast_to(bias_np, delta.shape),
            atol=1e-2,
            rtol=0.0,
        )


@unittest.skipIf(_SKIP, _SKIP_REASON)
class TestQloraStateDictNaming(unittest.TestCase):
    """return_dict key naming and linear_name prefixing are contract."""

    def _weight(self):
        weight_np = (np.arange(64) - 32).astype(np.float32).reshape(
            8, 8
        ) * 0.0625
        return paddle.to_tensor(weight_np, dtype=paddle.float16).cuda()

    def test_single_quant_keys_with_and_without_prefix(self):
        from paddlefleet.quantization.qlora import qlora_weight_quantize

        named = qlora_weight_quantize(
            weight=self._weight(),
            quant_algo="nf4",
            double_quant=False,
            block_size=64,
            return_dict=True,
            linear_name="dec.mlp",
        )
        self.assertEqual(
            set(named), {"dec.mlp.weight_scale", "dec.mlp.quant_weight"}
        )

        anon = qlora_weight_quantize(
            weight=self._weight(),
            quant_algo="nf4",
            double_quant=False,
            block_size=64,
            return_dict=True,
        )
        self.assertEqual(set(anon), {"weight_scale", "quant_weight"})

    def test_double_quant_keys_with_prefix(self):
        from paddlefleet.quantization.qlora import qlora_weight_quantize

        named = qlora_weight_quantize(
            weight=self._weight(),
            quant_algo="nf4",
            double_quant=True,
            block_size=64,
            double_quant_block_size=256,
            return_dict=True,
            linear_name="L0",
        )
        self.assertEqual(
            set(named),
            {
                "L0.qweight_scale",
                "L0.double_weight_scale",
                "L0.weight_scale_offset",
                "L0.quant_weight",
            },
        )


@unittest.skipIf(_SKIP, _SKIP_REASON)
class TestQloraDoubleQuantRoundTrip(unittest.TestCase):
    """Double-quant (fp8-quantized scales) still round-trips within bound."""

    def test_double_quant_dequant_within_bound(self):
        from paddlefleet.quantization.qlora import (
            qlora_weight_quantize_dequantize,
        )

        weight_np = ((np.arange(256).reshape(16, 16) % 13) - 6).astype(
            np.float32
        ) * 0.125
        weight = paddle.to_tensor(weight_np, dtype=paddle.float16).cuda()

        qdq = qlora_weight_quantize_dequantize(
            weight,
            quant_algo="nf4",
            double_quant=True,
            block_size=64,
            double_quant_block_size=256,
        )
        got = qdq.astype("float32").numpy()

        # NF4 level error PLUS the extra relative error from fp8-quantizing the
        # per-block scales. Allow a modest multiplicative slack on the scale.
        _, absmax = _nf4_reference_dequantize(weight_np, block_size=64)
        global_absmax = float(np.abs(weight_np).max())
        bound = _NF4_MAX_NORM_ERR * absmax * 1.15 + 0.05 * global_absmax
        err = np.abs(got - weight_np)
        self.assertTrue(
            np.all(err <= bound),
            f"double-quant error {err.max()} exceeds bound {bound.max()}",
        )
        self.assertGreater(err.max(), 1e-2)


if __name__ == "__main__":
    unittest.main()
