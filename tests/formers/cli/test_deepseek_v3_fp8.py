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

"""CPU-observable behavior tests for the DeepSeek-V3 FP8 linear module.

Target production code:
``src/paddlefleet/cli/train/deepseek_v3_pretrain/fp8_linear.py`` and the CPU
side of ``kernel.py``.

Scope (unit-test-rules.md "计算优化" / quantization):
  * block-wise scale *shape and axis mapping* derived by ``register_scale``
    (ceil-division per block, in-features -> rows, out-features -> cols);
  * the ``element_size() == 1`` gate that decides whether a scale parameter is
    registered at all;
  * the control-flow branch of ``fp8_linear`` that delegates a non-quantized
    (element_size > 1) weight to the pristine, pre-monkey-patch
    ``paddle.nn.functional.linear``, plus the module-level monkey-patch side
    effect;
  * the CPU-observable block-layout / rank preconditions asserted by
    ``kernel.act_quant`` and ``kernel.weight_dequant`` *before* any Triton GPU
    launch.

FP8 GEMM numerics themselves require Triton + a GPU and are NOT asserted here;
those paths are skipped with an explicit reason rather than faked. Every oracle
below is hand-derived and independent of the production implementation.
"""

import importlib.util
import os
import sys
import types
import unittest

import numpy as np

# --- Load the real production modules without triggering the package
# __init__ (which imports workflow.py -> AutoTokenizer and is unrelated to the
# FP8 linear logic under test). Only the parent package object is stubbed; the
# fp8_linear / kernel source that actually runs is the real production code,
# and the ``from paddlefleet.transformers.linear_utils import ...`` inside it
# resolves against the genuinely installed package.
_PKG = "paddlefleet.cli.train.deepseek_v3_pretrain"
_MODULE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "..",
        "src",
        "paddlefleet",
        "cli",
        "train",
        "deepseek_v3_pretrain",
    )
)


def _load_real_module(sub_name):
    """Execute the real ``<sub_name>.py`` from the production directory."""
    if _PKG not in sys.modules:
        stub = types.ModuleType(_PKG)
        stub.__path__ = [_MODULE_DIR]
        stub.__package__ = _PKG
        sys.modules[_PKG] = stub
    full_name = f"{_PKG}.{sub_name}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    spec = importlib.util.spec_from_file_location(
        full_name, os.path.join(_MODULE_DIR, f"{sub_name}.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


try:
    import paddle

    _fp8 = _load_real_module("fp8_linear")
    _PADDLE_OK = True
    _PADDLE_REASON = ""
except ImportError as exc:  # paddle / paddlefleet not installed in this env
    paddle = None
    _fp8 = None
    _PADDLE_OK = False
    _PADDLE_REASON = f"paddle/paddlefleet import unavailable: {exc}"

try:
    import triton  # noqa: F401

    _TRITON_OK = True
except ImportError:
    _TRITON_OK = False


def _weight_adapter(np_dtype, in_features, out_features):
    """A thin, attribute-settable stand-in for a Linear weight.

    ``shape`` and ``element_size()`` are taken verbatim from a *real* paddle
    tensor of the requested dtype, so the ``element_size() == 1`` gate and the
    ``[in, out]`` shape read inside ``register_scale`` exercise genuine values.
    The wrapper exists only so that ``weight._scale = ...`` (which production
    performs on an ``EagerParamBase``) has a place to land.
    """
    real = paddle.to_tensor(
        np.zeros((in_features, out_features), dtype=np_dtype)
    )

    class _WeightAdapter:
        def __init__(self):
            self.shape = list(real.shape)
            self._scale = None
            self._real = real

        def element_size(self):
            return self._real.element_size()

    return _WeightAdapter()


def _make_scale_host(weight, block_size):
    """Real ``paddle.nn.Layer`` host so ``create_parameter`` is the real API."""

    class _ScaleHost(paddle.nn.Layer):
        def __init__(self):
            super().__init__()
            self.weight = weight
            self.block_size = block_size
            self._weight_attr = None

    return _ScaleHost()


@unittest.skipUnless(_PADDLE_OK, _PADDLE_REASON)
class RegisterScaleTest(unittest.TestCase):
    """``register_scale`` block-wise scale shape / axis / gating."""

    def _run(self, np_dtype, in_features, out_features, block_size):
        host = _make_scale_host(
            _weight_adapter(np_dtype, in_features, out_features), block_size
        )
        _fp8.register_scale(host)
        return host

    def test_scale_shape_is_ceil_blocks_per_axis(self):
        # Non-divisible dims: ceil rounds up (floor would give [2, 3]).
        # in-features map to rows, out-features to cols (no axis swap).
        host = self._run(
            "int8", in_features=300, out_features=500, block_size=128
        )
        self.assertTrue(hasattr(host, "weight_scale_inv"))
        self.assertEqual(list(host.weight_scale_inv.shape), [3, 4])
        self.assertEqual(host.weight_scale_inv.dtype, paddle.float32)
        # scale is bound back onto the weight for the GEMM path to consume.
        self.assertIs(host.weight._scale, host.weight_scale_inv)

    def test_scale_shape_divisible_axis_mapping(self):
        # 256 -> 2 row-blocks, 512 -> 4 col-blocks; asymmetric counts catch a
        # swapped in/out axis assignment.
        host = self._run(
            "int8", in_features=256, out_features=512, block_size=128
        )
        self.assertEqual(list(host.weight_scale_inv.shape), [2, 4])

    def test_custom_block_size_changes_block_count(self):
        host = self._run(
            "int8", in_features=256, out_features=256, block_size=64
        )
        # ceil(256 / 64) == 4 on both axes.
        self.assertEqual(list(host.weight_scale_inv.shape), [4, 4])

    def test_non_quantized_weight_registers_no_scale(self):
        # element_size() == 4 (float32) fails the `== 1` gate: no scale param,
        # and the weight is left untouched.
        host = self._run(
            "float32", in_features=256, out_features=512, block_size=128
        )
        self.assertFalse(hasattr(host, "weight_scale_inv"))
        self.assertIsNone(host.weight._scale)


@unittest.skipUnless(_PADDLE_OK, _PADDLE_REASON)
class Fp8LinearControlFlowTest(unittest.TestCase):
    """``fp8_linear`` branch selection and the global monkey-patch."""

    def test_module_monkey_patches_functional_linear(self):
        # Importing the module rebinds paddle.nn.functional.linear; the saved
        # original must remain callable and distinct (else infinite recursion).
        self.assertIs(paddle.nn.functional.linear, _fp8.fp8_linear)
        self.assertTrue(callable(_fp8.original_linear))
        self.assertIsNot(_fp8.original_linear, _fp8.fp8_linear)

    def test_non_quantized_weight_delegates_to_original_linear(self):
        # float32 weight => element_size 4 > 1 => plain y = x @ w + b.
        x = paddle.to_tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32"
        )
        w = paddle.to_tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype="float32"
        )
        b = paddle.to_tensor([10.0, 20.0], dtype="float32")
        out = _fp8.fp8_linear(x, w, b)
        expected = np.array([[14.0, 25.0], [20.0, 31.0]], dtype=np.float32)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6, atol=1e-6)

    def test_non_quantized_weight_no_bias(self):
        x = paddle.to_tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32"
        )
        w = paddle.to_tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype="float32"
        )
        out = _fp8.fp8_linear(x, w, None)
        expected = np.array([[4.0, 5.0], [10.0, 11.0]], dtype=np.float32)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6, atol=1e-6)


@unittest.skipUnless(_PADDLE_OK, _PADDLE_REASON)
class Fp8LinearLayerTest(unittest.TestCase):
    """The real ``fp8_linear.Linear`` subclass (paddle.nn.Linear based)."""

    def test_default_block_size_and_no_scale_for_fp32_weight(self):
        layer = _fp8.Linear(16, 8)
        self.assertEqual(layer.block_size, 128)
        # float32 weight is not quantized -> register_scale is a no-op.
        self.assertFalse(hasattr(layer, "weight_scale_inv"))
        self.assertIsNone(getattr(layer.weight, "_scale", None))
        self.assertEqual(list(layer.weight.shape), [16, 8])

    @unittest.expectedFailure
    def test_block_size_kwarg_is_forwarded(self):
        # PRODUCTION BUG: __init__ forwards **kwargs (including block_size) to
        # paddle.nn.Linear.__init__, which rejects the unknown kwarg with a
        # TypeError. A caller can therefore never actually customise block_size
        # via the constructor; it is permanently pinned to 128. This documents
        # the intended contract that currently cannot hold. No production edit.
        layer = _fp8.Linear(16, 8, block_size=64)
        self.assertEqual(layer.block_size, 64)


@unittest.skipUnless(
    _PADDLE_OK and _TRITON_OK,
    "kernel.py requires both paddle and triton to import",
)
class KernelCpuPreconditionTest(unittest.TestCase):
    """CPU-observable block-layout / rank guards asserted before GPU launch."""

    @classmethod
    def setUpClass(cls):
        cls._kernel = _load_real_module("kernel")

    def test_act_quant_rejects_non_block_divisible_last_dim(self):
        # 130 % 128 != 0: the block-quantization layout precondition fires as an
        # AssertionError before any Triton kernel is dispatched.
        x = paddle.zeros([2, 130], dtype="float32")
        self.assertTrue(x.is_contiguous())
        with self.assertRaises(AssertionError):
            self._kernel.act_quant(x, block_size=128)

    def test_weight_dequant_rejects_non_2d_input(self):
        # weight_dequant requires rank-2 x and s; a rank-3 x must be rejected
        # (again before the GPU dequant kernel runs).
        x = paddle.zeros([2, 4, 8], dtype="float32")
        s = paddle.zeros([1, 1], dtype="float32")
        with self.assertRaises(AssertionError):
            self._kernel.weight_dequant(x, s, block_size=128)


if __name__ == "__main__":
    unittest.main()
