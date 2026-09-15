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

"""Behavior tests for peft/lora/lora_quantization_layers.py.

These tests run in the no-card (CPU) environment. The only GPU-bound
collaborator is ``quant_weight_linear`` (the quantized base matmul, which
lives in paddlefleet.quantization and is NOT the code under test); it is
replaced by a distinguishable marker so that the LoRA-wrapper logic under
test -- scaling, adapter delta application, disable/merge state semantics
and the Fleet bias split -- runs for real on CPU and is compared against an
independent NumPy reference.
"""

import math
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import paddle

from paddlefleet.peft.lora import LoRAConfig
from paddlefleet.peft.lora.lora_quantization_layers import (
    FleetQuantizationLoRALinear,
    QuantizationLoRABaseLinear,
    QuantizationLoRALinear,
)

_MODULE = "paddlefleet.peft.lora.lora_quantization_layers"


def _make_quant_config(double_quant=False):
    """Minimal stand-in for a QuantizationConfig; only the attributes read
    by the wrapper / kernel signature are populated."""
    return SimpleNamespace(
        qlora_weight_double_quant=double_quant,
        group_size=-1,
        llm_int8_threshold=6.0,
        qlora_weight_blocksize=64,
        qlora_weight_double_quant_block_size=256,
    )


def _make_base_layer(
    in_features=4,
    out_features=3,
    algo="weight_only_int8",
    dtype="float32",
    double_quant=False,
    bias=None,
    weight_scale=None,
):
    """Container mimicking the already-quantized base linear whose public
    attributes the LoRA wrapper consumes. This is a fixture, not the code
    under test; the tests assert the wrapper forwards these objects verbatim.
    """
    layer = SimpleNamespace()
    layer.in_features = in_features
    layer.out_features = out_features
    layer.weight_quantize_algo = algo
    layer._dtype = dtype
    layer.quant_dtype = "int8"
    # A plain tensor carries no mp_moe / truthy is_distributed, so the wrapper
    # takes the single-device parameter path.
    layer.quant_weight = paddle.zeros([out_features, in_features], dtype="int8")
    layer.bias = bias
    layer.quantization_config = _make_quant_config(double_quant)
    if double_quant:
        layer.qweight_scale = paddle.to_tensor([1.0, 2.0], dtype=dtype)
        layer.double_weight_scale = paddle.to_tensor([3.0], dtype=dtype)
        layer.weight_scale_offset = paddle.to_tensor([0.5], dtype=dtype)
    else:
        if weight_scale is None:
            weight_scale = paddle.to_tensor(
                [float(i + 1) for i in range(out_features)], dtype=dtype
            )
        layer.weight_scale = weight_scale
    return layer


def _set_lora_weights(layer, a_np, b_np):
    layer.lora_A.set_value(paddle.to_tensor(a_np, dtype=layer._dtype))
    layer.lora_B.set_value(paddle.to_tensor(b_np, dtype=layer._dtype))


class TestQuantizationLoRAInit(unittest.TestCase):
    """__init__ config consumption and validation contracts."""

    def test_scaling_uses_alpha_over_rank(self):
        # scaling is recomputed by the layer; compare against an independent
        # formula, and against a second rank to prove it is not a constant.
        base = _make_base_layer()
        layer = QuantizationLoRABaseLinear(
            base, LoRAConfig(r=4, lora_alpha=8, lora_dropout=0.0, rslora=False)
        )
        self.assertAlmostEqual(layer.scaling, 8.0 / 4.0)
        self.assertFalse(layer.disable_lora)

        layer2 = QuantizationLoRABaseLinear(
            _make_base_layer(),
            LoRAConfig(r=8, lora_alpha=8, lora_dropout=0.0, rslora=False),
        )
        self.assertAlmostEqual(layer2.scaling, 8.0 / 8.0)
        self.assertNotAlmostEqual(layer.scaling, layer2.scaling)

    def test_rslora_scaling_uses_sqrt_rank(self):
        layer = QuantizationLoRABaseLinear(
            _make_base_layer(),
            LoRAConfig(r=4, lora_alpha=8, lora_dropout=0.0, rslora=True),
        )
        self.assertAlmostEqual(layer.scaling, 8.0 / math.sqrt(4))
        # rslora must differ from the plain alpha/r branch at this rank.
        self.assertNotAlmostEqual(layer.scaling, 8.0 / 4.0)

    def test_non_positive_or_non_int_rank_rejected(self):
        for bad_r in (0, -1, 4.0):
            with self.assertRaises(ValueError):
                QuantizationLoRABaseLinear(
                    _make_base_layer(), LoRAConfig(r=bad_r, lora_alpha=8)
                )

    def test_llm_int8_not_supported(self):
        base = _make_base_layer(algo="llm.int8")
        with self.assertRaises(NotImplementedError):
            QuantizationLoRABaseLinear(base, LoRAConfig(r=4, lora_alpha=8))

    def test_single_quant_selects_weight_scale(self):
        base = _make_base_layer(algo="weight_only_int8")
        layer = QuantizationLoRABaseLinear(base, LoRAConfig(r=4, lora_alpha=8))
        # The non-double-quant branch wires weight_scale and none of the
        # double-quant state tensors.
        self.assertIs(layer.weight_scale, base.weight_scale)
        self.assertFalse(hasattr(layer, "qweight_scale"))
        self.assertFalse(hasattr(layer, "double_weight_scale"))
        self.assertFalse(hasattr(layer, "weight_scale_offset"))

    def test_double_quant_selects_quant_state_tensors(self):
        base = _make_base_layer(algo="nf4", double_quant=True)
        layer = QuantizationLoRABaseLinear(base, LoRAConfig(r=4, lora_alpha=8))
        # The double-quant branch wires the three quant-state tensors by
        # identity and must NOT set a plain weight_scale.
        self.assertIs(layer.qweight_scale, base.qweight_scale)
        self.assertIs(layer.double_weight_scale, base.double_weight_scale)
        self.assertIs(layer.weight_scale_offset, base.weight_scale_offset)
        self.assertFalse(hasattr(layer, "weight_scale"))


class TestQuantizationLoRALinearForward(unittest.TestCase):
    """forward numerics and disable/merge state semantics."""

    def _build(self, algo="weight_only_int8", bias=None):
        base = _make_base_layer(
            in_features=4, out_features=3, algo=algo, bias=bias
        )
        layer = QuantizationLoRALinear(base, LoRAConfig(r=4, lora_alpha=8))
        a_np = (np.arange(16).reshape(4, 4).astype(np.float32) - 8.0) * 0.1
        b_np = (np.arange(12).reshape(4, 3).astype(np.float32) - 6.0) * 0.1
        _set_lora_weights(layer, a_np, b_np)
        return base, layer, a_np, b_np

    def test_forward_adds_scaled_lora_delta(self):
        base, layer, a_np, b_np = self._build()
        x_np = np.array(
            [[0.5, -1.0, 2.0, 0.0], [1.0, 0.0, -0.5, 3.0]], dtype=np.float32
        )
        x = paddle.to_tensor(x_np)
        marker_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], np.float32)

        captured = {}

        def fake_kernel(**kwargs):
            captured.update(kwargs)
            return paddle.to_tensor(marker_np)

        with mock.patch(
            f"{_MODULE}.quant_weight_linear", side_effect=fake_kernel
        ):
            with paddle.no_grad():
                out = layer(x)

        # Independent reference: base output (marker) + scaled adapter delta.
        expected = marker_np + (x_np @ a_np @ b_np) * (8.0 / 4.0)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-6)

        # The base kernel must receive the untouched input and the layer's own
        # quant metadata / scale / (absent) quant_state.
        np.testing.assert_array_equal(captured["x"].numpy(), x_np)
        self.assertIs(captured["weight_scale"], base.weight_scale)
        self.assertIs(captured["bias"], base.bias)
        self.assertEqual(captured["quant_dtype"], "int8")
        self.assertEqual(captured["weight_quantize_algo"], "weight_only_int8")
        self.assertEqual(captured["dtype"], "float32")
        self.assertIsNone(captured["quant_state"])

    def test_disable_lora_returns_base_output_only(self):
        _, layer, _, _ = self._build()
        layer.disable_lora = True
        x = paddle.to_tensor(
            [[0.5, -1.0, 2.0, 0.0], [1.0, 0.0, -0.5, 3.0]], dtype="float32"
        )
        marker_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], np.float32)

        with mock.patch(
            f"{_MODULE}.quant_weight_linear",
            side_effect=lambda **kw: paddle.to_tensor(marker_np),
        ):
            with paddle.no_grad():
                out = layer(x)
        # With adapter disabled the non-zero A/B must not contribute.
        np.testing.assert_allclose(out.numpy(), marker_np, rtol=1e-6, atol=1e-6)

    def test_merge_unmerge_are_noops_and_warn(self):
        _, layer, a_np, b_np = self._build()
        before_a = layer.lora_A.numpy().copy()
        before_b = layer.lora_B.numpy().copy()
        before_w = layer.quant_weight.numpy().copy()

        with mock.patch(f"{_MODULE}.logger") as mlog:
            layer.merge()
            layer.unmerge()

        self.assertEqual(mlog.warning.call_count, 2)
        self.assertIn("merge", mlog.warning.call_args_list[0].args[0])
        self.assertIn("unmerge", mlog.warning.call_args_list[1].args[0])
        # merge/unmerge are unsupported: no weights may change.
        np.testing.assert_array_equal(layer.lora_A.numpy(), before_a)
        np.testing.assert_array_equal(layer.lora_B.numpy(), before_b)
        np.testing.assert_array_equal(layer.quant_weight.numpy(), before_w)


class TestFleetQuantizationLoRALinear(unittest.TestCase):
    """The Fleet wrapper returns (output, out_bias) and honours skip_bias_add."""

    def _build(self, skip_bias_add):
        bias = paddle.to_tensor([0.1, -0.2, 0.3], dtype="float32")
        base = _make_base_layer(
            in_features=4, out_features=3, algo="weight_only_int8", bias=bias
        )
        layer = FleetQuantizationLoRALinear(
            base,
            skip_bias_add=skip_bias_add,
            lora_config=LoRAConfig(r=4, lora_alpha=8),
        )
        a_np = (np.arange(16).reshape(4, 4).astype(np.float32) - 8.0) * 0.1
        b_np = (np.arange(12).reshape(4, 3).astype(np.float32) - 6.0) * 0.1
        _set_lora_weights(layer, a_np, b_np)
        return bias, layer, a_np, b_np

    def test_skip_bias_add_splits_bias_out(self):
        bias, layer, a_np, b_np = self._build(skip_bias_add=True)
        self.assertTrue(layer.skip_bias_add)
        x_np = np.array(
            [[0.5, -1.0, 2.0, 0.0], [1.0, 0.0, -0.5, 3.0]], dtype=np.float32
        )
        marker_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], np.float32)
        captured = {}

        def fake_kernel(**kwargs):
            captured.update(kwargs)
            return paddle.to_tensor(marker_np)

        with mock.patch(
            f"{_MODULE}.quant_weight_linear", side_effect=fake_kernel
        ):
            with paddle.no_grad():
                output, out_bias = layer(paddle.to_tensor(x_np))

        # Bias is returned to the caller and withheld from the base matmul.
        self.assertIs(out_bias, bias)
        self.assertIsNone(layer.bias)
        self.assertIsNone(captured["bias"])
        expected = marker_np + (x_np @ a_np @ b_np) * (8.0 / 4.0)
        np.testing.assert_allclose(
            output.numpy(), expected, rtol=1e-5, atol=1e-6
        )

    def test_keep_bias_passes_bias_to_kernel(self):
        bias, layer, _, _ = self._build(skip_bias_add=False)
        x = paddle.to_tensor(
            [[0.5, -1.0, 2.0, 0.0], [1.0, 0.0, -0.5, 3.0]], dtype="float32"
        )
        marker_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], np.float32)
        captured = {}

        def fake_kernel(**kwargs):
            captured.update(kwargs)
            return paddle.to_tensor(marker_np)

        with mock.patch(
            f"{_MODULE}.quant_weight_linear", side_effect=fake_kernel
        ):
            with paddle.no_grad():
                output, out_bias = layer(x)

        # No split: caller gets no separate bias, kernel keeps the bias.
        self.assertIsNone(out_bias)
        self.assertIs(layer.bias, bias)
        self.assertIs(captured["bias"], bias)


if __name__ == "__main__":
    unittest.main()
