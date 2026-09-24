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
"""Behavior tests for paddlefleet.fp8 quantization/utils orchestration.

These target the CPU-observable control logic of ``get_quant_func`` (recipe
validation, block-method selection, flag propagation, the ``_mn_major``
scale-transpose, and the weight prequant cache short-circuit) and the
dtype/structure branching in ``is_fp8_tensor``. The underlying DeepGEMM
quant kernel needs a Hopper GPU, so the tests that exercise orchestration
replace ``fp8_quant_blockwise`` with a distinguishable marker and assert on
the exact args it receives and how its results are mapped -- they never
assert on real quantized numerics. ``is_fp8_tensor`` reads only ``.dtype``,
so it is driven with lightweight duck-typed carriers.

paddlefleet imports paddle at import time; when paddle is absent the whole
module is skipped with an honest reason.
"""

import types
import unittest
from unittest import mock

try:
    import paddle

    from paddlefleet.fp8.quantization import get_quant_func
    from paddlefleet.fp8.utils import is_fp8_tensor

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle / paddlefleet not installed locally
    paddle = None
    get_quant_func = None
    is_fp8_tensor = None
    _IMPORT_ERROR = exc

_HAS_PADDLE = _IMPORT_ERROR is None
_SKIP_REASON = (
    f"paddle/paddlefleet not importable: {_IMPORT_ERROR}"
    if _IMPORT_ERROR is not None
    else ""
)

# Fully-qualified target the production code binds at call time:
#   _quant = paddle.incubate.nn.functional.fp8_quant_blockwise
_QUANT_TARGET = "paddle.incubate.nn.functional.fp8_quant_blockwise"


@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON)
class TestGetQuantFuncRecipeGuard(unittest.TestCase):
    """get_quant_func must reject any recipe other than 'blockwise'."""

    def test_unsupported_recipe_raises_valueerror_naming_recipe(self):
        with self.assertRaises(ValueError) as ctx:
            get_quant_func("per_tensor")
        # The offending recipe name is echoed back in the message.
        self.assertIn("per_tensor", str(ctx.exception))

    def test_empty_recipe_raises_valueerror(self):
        with self.assertRaises(ValueError):
            get_quant_func("")

    def test_blockwise_recipe_returns_two_distinct_callables(self):
        inp_func, weight_func = get_quant_func("blockwise")
        self.assertTrue(callable(inp_func))
        self.assertTrue(callable(weight_func))
        self.assertIsNot(inp_func, weight_func)


@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON)
class TestWeightQuantCacheShortCircuit(unittest.TestCase):
    """weight_quant_func returns the cached 4-tuple without touching the kernel.

    _cached_weight_result reads x.fp8_weight_fwd / fp8_scale_fwd / fp8_scale_bwd
    and, on a hit, returns (None, scale_bwd, fp8_fwd, scale_fwd). The ordering
    is load-bearing: a swap between the fwd fp8 and either scale slot would
    corrupt dgrad reuse. Expected mapping is hand-derived from the source.
    """

    def test_cache_hit_returns_expected_slot_mapping(self):
        _, weight_func = get_quant_func("blockwise")
        sentinel_fwd = object()
        sentinel_scale_fwd = object()
        sentinel_scale_bwd = object()
        cached_input = types.SimpleNamespace(
            fp8_weight_fwd=sentinel_fwd,
            fp8_scale_fwd=sentinel_scale_fwd,
            fp8_scale_bwd=sentinel_scale_bwd,
        )
        # Kernel must NOT be called on a cache hit; make it explode if it is.
        with mock.patch(
            _QUANT_TARGET, side_effect=AssertionError("kernel called")
        ):
            result = weight_func(cached_input)
        self.assertEqual(len(result), 4)
        fp8_bwd, scale_bwd, fp8_fwd, scale_fwd = result
        self.assertIsNone(fp8_bwd)  # bwd fp8 derived later by caller
        self.assertIs(scale_bwd, sentinel_scale_bwd)
        self.assertIs(fp8_fwd, sentinel_fwd)
        self.assertIs(scale_fwd, sentinel_scale_fwd)

    def test_partial_cache_attrs_is_a_miss(self):
        """fp8_weight_fwd set but scales missing -> fall through to the kernel."""
        partial_input = types.SimpleNamespace(fp8_weight_fwd=object())
        marker = ("BWD", "SB", "FWD", "SF")
        # ``get_quant_func`` binds ``_quant`` to
        # ``paddle.incubate.nn.functional.fp8_quant_blockwise`` at call time,
        # so the patch must be active *before* the factory runs.
        with mock.patch(_QUANT_TARGET, return_value=marker) as fake_quant:
            _, weight_func = get_quant_func("blockwise")
            result = weight_func(partial_input)
        fake_quant.assert_called_once()
        # out_scale_trans defaults to False -> tuple passed through unchanged.
        self.assertEqual(result, marker)


@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON)
class TestWeightQuantKernelOrchestration(unittest.TestCase):
    """On a cache miss weight_quant_func drives the kernel with 128x128 blocks."""

    def test_weight_uses_128x128_and_forwards_flags(self):
        fresh_input = object()  # no cache attributes -> cache miss
        marker = ("FP8_BWD", "SCALE_BWD", "FP8_FWD", "SCALE_FWD")
        # Patch before the factory captures ``_quant`` (bound at call time).
        with mock.patch(_QUANT_TARGET, return_value=marker) as fake_quant:
            _, weight_func = get_quant_func(
                "blockwise",
                input_trans=True,
                out_scale_trans=False,
                pow2_scale=True,
            )
            result = weight_func(fresh_input)

        fake_quant.assert_called_once()
        args, kwargs = fake_quant.call_args
        self.assertIs(args[0], fresh_input)
        self.assertEqual(kwargs["quant_method"], "128x128")
        self.assertTrue(kwargs["input_transpose"])
        self.assertFalse(kwargs["output_scale_transpose"])
        self.assertTrue(kwargs["using_pow2_scale"])
        # out_scale_trans=False -> no _mn_major transpose applied.
        self.assertEqual(result, marker)

    def test_out_scale_trans_transposes_scales_but_not_fp8(self):
        scale_bwd = mock.MagicMock(name="scale_bwd")
        scale_fwd = mock.MagicMock(name="scale_fwd")
        marker = ("FP8_BWD", scale_bwd, "FP8_FWD", scale_fwd)
        # Patch before the factory captures ``_quant`` (bound at call time).
        with mock.patch(_QUANT_TARGET, return_value=marker):
            _, weight_func = get_quant_func(
                "blockwise",
                input_trans=True,
                out_scale_trans=True,
                pow2_scale=True,
            )
            fp8_bwd, out_sb, fp8_fwd, out_sf = weight_func(object())

        # fp8 payloads untouched; only the scales get the stride-only .T view.
        self.assertEqual(fp8_bwd, "FP8_BWD")
        self.assertEqual(fp8_fwd, "FP8_FWD")
        self.assertIs(out_sb, scale_bwd.T)
        self.assertIs(out_sf, scale_fwd.T)


@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON)
class TestInpQuantKernelOrchestration(unittest.TestCase):
    """inp_quant_func drives the kernel with 1x128 blocks and forwards flags."""

    def test_non_ue8m0_inp_uses_1x128_and_returns_kernel_result(self):
        # Patch before the factory captures ``_quant`` (bound at call time).
        with mock.patch(_QUANT_TARGET, return_value="INP_RESULT") as fake_quant:
            inp_func, _ = get_quant_func(
                "blockwise",
                input_trans=True,
                out_scale_trans=True,
                pow2_scale=False,
            )
            result = inp_func("XT")

        fake_quant.assert_called_once()
        args, kwargs = fake_quant.call_args
        self.assertEqual(args[0], "XT")
        self.assertEqual(kwargs["quant_method"], "1x128")
        self.assertTrue(kwargs["input_transpose"])
        self.assertTrue(kwargs["output_scale_transpose"])
        self.assertFalse(kwargs["using_pow2_scale"])
        # Non-UE8M0 input path is a functools.partial: result passes straight
        # through with no _mn_major and no using_ue8m0_scale flag.
        self.assertEqual(result, "INP_RESULT")
        self.assertNotIn("using_ue8m0_scale", kwargs)

    def test_ue8m0_inp_forces_transpose_flags_and_mn_major(self):
        scale = mock.MagicMock(name="scale")
        scale_t = mock.MagicMock(name="scale_t")
        marker = ("FP8", scale, "FP8_T", scale_t)
        # Patch before the factory captures ``_quant`` (bound at call time).
        with mock.patch(_QUANT_TARGET, return_value=marker) as fake_quant:
            inp_func, _ = get_quant_func(
                "blockwise",
                input_trans=True,
                pow2_scale=True,
                use_ue8m0=True,
            )
            fp8, out_scale, fp8_t, out_scale_t = inp_func("X")

        args, kwargs = fake_quant.call_args
        self.assertEqual(args[0], "X")
        self.assertEqual(kwargs["quant_method"], "1x128")
        self.assertTrue(kwargs["input_transpose"])
        # UE8M0 forces these regardless of out_scale_trans / pow2 defaults.
        self.assertTrue(kwargs["output_scale_transpose"])
        self.assertTrue(kwargs["using_pow2_scale"])
        self.assertTrue(kwargs["using_ue8m0_scale"])
        # Both scales get _mn_major (.T); fp8 payloads are untouched.
        self.assertEqual(fp8, "FP8")
        self.assertEqual(fp8_t, "FP8_T")
        self.assertIs(out_scale, scale.T)
        self.assertIs(out_scale_t, scale_t.T)


@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON)
class TestIsFp8Tensor(unittest.TestCase):
    """is_fp8_tensor branch logic. The function only reads .dtype, so inputs
    are duck-typed carriers with a dtype attribute -- no GPU fp8 tensor
    allocation is required. Expected values are hand-derived from the source.
    """

    @staticmethod
    def _carrier(dtype):
        return types.SimpleNamespace(dtype=dtype)

    def test_non_tuple_inputs_return_false(self):
        self.assertFalse(is_fp8_tensor("not a tuple"))
        self.assertFalse(is_fp8_tensor(42))
        self.assertFalse(is_fp8_tensor(None))
        self.assertFalse(is_fp8_tensor([1, 2]))  # list, not tuple

    def test_wrong_length_tuple_returns_false(self):
        self.assertFalse(is_fp8_tensor((1,)))
        self.assertFalse(is_fp8_tensor((1, 2, 3)))

    def test_e4m3_with_float32_scale_is_fp8(self):
        pair = (
            self._carrier(paddle.float8_e4m3fn),
            self._carrier(paddle.float32),
        )
        self.assertTrue(is_fp8_tensor(pair))

    def test_e4m3_with_int32_scale_is_fp8(self):
        # UE8M0-packed scales are int32 and must be accepted.
        pair = (
            self._carrier(paddle.float8_e4m3fn),
            self._carrier(paddle.int32),
        )
        self.assertTrue(is_fp8_tensor(pair))

    def test_e4m3_with_float16_scale_is_rejected(self):
        pair = (
            self._carrier(paddle.float8_e4m3fn),
            self._carrier(paddle.float16),
        )
        self.assertFalse(is_fp8_tensor(pair))

    def test_non_fp8_tensor_dtype_is_rejected(self):
        pair = (
            self._carrier(paddle.bfloat16),
            self._carrier(paddle.float32),
        )
        self.assertFalse(is_fp8_tensor(pair))

    def test_e5m2_tensor_triggers_assertion(self):
        # Documented "not supported yet" guard: e5m2 raises rather than
        # returning False.
        pair = (
            self._carrier(paddle.float8_e5m2),
            self._carrier(paddle.float32),
        )
        with self.assertRaises(AssertionError):
            is_fp8_tensor(pair)


if __name__ == "__main__":
    unittest.main()
