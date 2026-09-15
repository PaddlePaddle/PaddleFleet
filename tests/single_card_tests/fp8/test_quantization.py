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
"""Behavior tests for ``paddlefleet.fp8.quantization.get_quant_func``.

``get_quant_func`` is pure orchestration: it validates the recipe, builds
the ``inp_quant_func`` / ``weight_quant_func`` closures, threads the caller
flags into ``fp8_quant_blockwise`` invocations, short-circuits weight
quantization through a per-tensor cache, and applies the ``_mn_major``
(``.T``) scale-layout fixup. The underlying ``fp8_quant_blockwise`` kernel
is a GPU collaborator whose numerics are not under test here; it is
replaced with a distinguishable marker so the orchestration (argument
threading, tuple ordering, transpose targets, cache short-circuit) can be
observed on CPU. Kernel numerics remain unverified by this file.

paddlefleet imports paddle at import time and paddle is absent in some
environments, so the whole suite is honestly skipped when the import fails.
"""

from __future__ import annotations

import functools
import unittest
from unittest import mock

try:
    from paddlefleet.fp8.quantization import get_quant_func

    _IMPORT_ERROR: ImportError | None = None
except ImportError as exc:  # paddle (or paddlefleet) not installed
    get_quant_func = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc

_QUANT_TARGET = (
    "paddlefleet.fp8.quantization.paddle."
    "incubate.nn.functional.fp8_quant_blockwise"
)

_SKIP_REASON = (
    "paddlefleet.fp8.quantization import failed "
    f"(paddle not installed): {_IMPORT_ERROR!r}"
)


class _FakeTensor:
    """Marker with a ``.T`` that yields a new, name-tagged marker.

    ``_mn_major`` calls ``scale.T``; using a distinguishable tag lets the
    test prove which operands were transposed (scales) and which passed
    through untouched (fp8 payloads), without a real paddle tensor.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @property
    def T(self) -> _FakeTensor:
        return _FakeTensor(self.name + ".T")

    def __repr__(self) -> str:
        return f"_FakeTensor({self.name!r})"


def _patch_kernel(**mock_kwargs):
    """Patch the fp8_quant_blockwise collaborator with a MagicMock."""
    return mock.patch(_QUANT_TARGET, create=True, **mock_kwargs)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestRecipeValidation(unittest.TestCase):
    """The recipe guard runs before the kernel is ever referenced."""

    def test_unsupported_recipe_names_offender_and_supported_set(self):
        # Hand-chosen recipe string that is not "blockwise".
        with self.assertRaises(ValueError) as ctx:
            get_quant_func("rowwise")
        msg = str(ctx.exception)
        self.assertIn("rowwise", msg)
        self.assertIn("blockwise", msg)

    def test_empty_recipe_still_rejected(self):
        with self.assertRaises(ValueError):
            get_quant_func("")

    def test_validation_precedes_kernel_lookup(self):
        # side_effect fires only if the kernel attribute is *accessed and
        # called*; the ValueError path must return before touching it.
        with _patch_kernel(side_effect=AssertionError("kernel touched")):
            with self.assertRaises(ValueError):
                get_quant_func("not_blockwise")


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestInpQuantFuncPartial(unittest.TestCase):
    """Default (non-UE8M0) inp path is a partial with threaded flags."""

    def test_partial_binds_kernel_and_all_flags(self):
        with _patch_kernel() as kernel:
            inp_func, _ = get_quant_func(
                "blockwise",
                input_trans=True,
                out_scale_trans=True,
                pow2_scale=True,
            )
        # inp_quant_func is a functools.partial over the real kernel.
        self.assertIsInstance(inp_func, functools.partial)
        self.assertIs(inp_func.func, kernel)
        self.assertEqual(inp_func.args, ())
        # Every caller flag is threaded to its own kernel kwarg. A swap
        # (e.g. input_trans wired to output_scale_transpose) is rejected.
        self.assertEqual(
            inp_func.keywords,
            {
                "output_scale_transpose": True,
                "quant_method": "1x128",
                "input_transpose": True,
                "using_pow2_scale": True,
            },
        )

    def test_partial_defaults_are_all_false(self):
        with _patch_kernel():
            inp_func, _ = get_quant_func("blockwise")
        self.assertEqual(
            inp_func.keywords,
            {
                "output_scale_transpose": False,
                "quant_method": "1x128",
                "input_transpose": False,
                "using_pow2_scale": False,
            },
        )

    def test_flags_thread_independently(self):
        # Only out_scale_trans set: it must land on output_scale_transpose
        # while input_transpose / using_pow2_scale stay False.
        with _patch_kernel():
            inp_func, _ = get_quant_func("blockwise", out_scale_trans=True)
        self.assertTrue(inp_func.keywords["output_scale_transpose"])
        self.assertFalse(inp_func.keywords["input_transpose"])
        self.assertFalse(inp_func.keywords["using_pow2_scale"])


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestWeightQuantFuncCache(unittest.TestCase):
    """weight_quant_func short-circuits on pre-quantized tensors."""

    def test_cache_hit_returns_ordered_tuple_without_kernel(self):
        fp8_fwd = _FakeTensor("fp8_fwd")
        scale_fwd = _FakeTensor("scale_fwd")
        scale_bwd = _FakeTensor("scale_bwd")

        cached = mock.Mock()
        cached.fp8_weight_fwd = fp8_fwd
        cached.fp8_scale_fwd = scale_fwd
        cached.fp8_scale_bwd = scale_bwd

        # Kernel must NOT run on a cache hit.
        with _patch_kernel(side_effect=AssertionError("kernel called")) as k:
            _, weight_func = get_quant_func("blockwise")
            result = weight_func(cached)
        k.assert_not_called()

        # Contract order is (fp8_bwd, scale_bwd, fp8_fwd, scale_fwd) with
        # fp8_bwd deferred (None) and the rest returned by identity.
        self.assertEqual(len(result), 4)
        self.assertIsNone(result[0])
        self.assertIs(result[1], scale_bwd)
        self.assertIs(result[2], fp8_fwd)
        self.assertIs(result[3], scale_fwd)

    def test_missing_fwd_scale_falls_through_to_kernel(self):
        # fp8_weight_fwd present but fp8_scale_fwd absent -> incomplete
        # cache -> real quant path must run.
        payload = (
            _FakeTensor("k_bwd"),
            _FakeTensor("k_scale_bwd"),
            _FakeTensor("k_fwd"),
            _FakeTensor("k_scale_fwd"),
        )
        partial_cache = mock.Mock()
        partial_cache.fp8_weight_fwd = _FakeTensor("fwd")
        partial_cache.fp8_scale_fwd = None
        partial_cache.fp8_scale_bwd = _FakeTensor("bwd")

        with _patch_kernel(return_value=payload) as k:
            _, weight_func = get_quant_func("blockwise")
            result = weight_func(partial_cache)
        k.assert_called_once()
        # Fell through: kernel output passed through verbatim.
        self.assertEqual(result, payload)

    def test_no_cache_attrs_falls_through_to_kernel(self):
        payload = (
            _FakeTensor("k_bwd"),
            _FakeTensor("k_scale_bwd"),
            _FakeTensor("k_fwd"),
            _FakeTensor("k_scale_fwd"),
        )
        with _patch_kernel(return_value=payload) as k:
            _, weight_func = get_quant_func("blockwise")
            result = weight_func(object())  # no fp8_weight_fwd attribute
        k.assert_called_once()
        self.assertEqual(result, payload)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestWeightQuantFuncKernelCall(unittest.TestCase):
    """Non-UE8M0 weight path calls the kernel with fixed orientation."""

    def test_weight_kernel_uses_128x128_and_input_transpose(self):
        payload = (
            _FakeTensor("bwd"),
            _FakeTensor("scale_bwd"),
            _FakeTensor("fwd"),
            _FakeTensor("scale_fwd"),
        )
        x = _FakeTensor("weight")
        with _patch_kernel(return_value=payload) as k:
            _, weight_func = get_quant_func(
                "blockwise", out_scale_trans=False, pow2_scale=True
            )
            result = weight_func(x)

        k.assert_called_once()
        args, kwargs = k.call_args
        self.assertEqual(args, (x,))
        self.assertEqual(kwargs["quant_method"], "128x128")
        # Weight always emits both orientations in one launch.
        self.assertTrue(kwargs["input_transpose"])
        self.assertFalse(kwargs["output_scale_transpose"])
        self.assertTrue(kwargs["using_pow2_scale"])
        # out_scale_trans False -> no transpose fixup, verbatim passthrough.
        self.assertEqual(result, payload)

    def test_out_scale_trans_transposes_only_scales(self):
        fp8_bwd = _FakeTensor("fp8_bwd")
        scale_bwd = _FakeTensor("scale_bwd")
        fp8_fwd = _FakeTensor("fp8_fwd")
        scale_fwd = _FakeTensor("scale_fwd")
        payload = (fp8_bwd, scale_bwd, fp8_fwd, scale_fwd)

        with _patch_kernel(return_value=payload) as k:
            _, weight_func = get_quant_func("blockwise", out_scale_trans=True)
            r_bwd, r_scale_bwd, r_fwd, r_scale_fwd = weight_func(
                _FakeTensor("w")
            )

        self.assertTrue(k.call_args.kwargs["output_scale_transpose"])
        # fp8 payloads are untouched (identity)...
        self.assertIs(r_bwd, fp8_bwd)
        self.assertIs(r_fwd, fp8_fwd)
        # ...while both scales get the _mn_major .T fixup.
        self.assertEqual(r_scale_bwd.name, "scale_bwd.T")
        self.assertEqual(r_scale_fwd.name, "scale_fwd.T")


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestUE8M0Path(unittest.TestCase):
    """UE8M0 emits both orientations and .T-fixes every scale."""

    def test_inp_input_trans_flags_and_transposed_scales(self):
        fp8 = _FakeTensor("fp8")
        scale = _FakeTensor("scale")
        fp8_t = _FakeTensor("fp8_t")
        scale_t = _FakeTensor("scale_t")
        with _patch_kernel(return_value=(fp8, scale, fp8_t, scale_t)) as k:
            inp_func, _ = get_quant_func(
                "blockwise", input_trans=True, use_ue8m0=True
            )
            out = inp_func(_FakeTensor("x"))

        kwargs = k.call_args.kwargs
        self.assertEqual(kwargs["quant_method"], "1x128")
        self.assertTrue(kwargs["input_transpose"])
        self.assertTrue(kwargs["output_scale_transpose"])
        self.assertTrue(kwargs["using_pow2_scale"])
        self.assertTrue(kwargs["using_ue8m0_scale"])

        # Returns (fp8, scale.T, fp8_t, scale_t.T): payloads identity,
        # scales transposed.
        self.assertIs(out[0], fp8)
        self.assertEqual(out[1].name, "scale.T")
        self.assertIs(out[2], fp8_t)
        self.assertEqual(out[3].name, "scale_t.T")

    def test_inp_no_input_trans_returns_two_tuple(self):
        fp8 = _FakeTensor("fp8")
        scale = _FakeTensor("scale")
        # Kernel returns a 4-tuple; the no-transpose branch keeps [:2].
        extra = (_FakeTensor("fp8_t"), _FakeTensor("scale_t"))
        with _patch_kernel(return_value=(fp8, scale) + extra) as k:
            inp_func, _ = get_quant_func(
                "blockwise", input_trans=False, use_ue8m0=True
            )
            out = inp_func(_FakeTensor("x"))

        self.assertFalse(k.call_args.kwargs["input_transpose"])
        self.assertTrue(k.call_args.kwargs["using_ue8m0_scale"])
        self.assertEqual(len(out), 2)
        self.assertIs(out[0], fp8)
        self.assertEqual(out[1].name, "scale.T")

    def test_weight_uses_128x128_ue8m0_and_transposes_scales(self):
        fp8_bwd = _FakeTensor("fp8_bwd")
        scale_bwd = _FakeTensor("scale_bwd")
        fp8_fwd = _FakeTensor("fp8_fwd")
        scale_fwd = _FakeTensor("scale_fwd")
        payload = (fp8_bwd, scale_bwd, fp8_fwd, scale_fwd)
        with _patch_kernel(return_value=payload) as k:
            _, weight_func = get_quant_func("blockwise", use_ue8m0=True)
            r_bwd, r_scale_bwd, r_fwd, r_scale_fwd = weight_func(
                _FakeTensor("w")
            )

        kwargs = k.call_args.kwargs
        self.assertEqual(kwargs["quant_method"], "128x128")
        self.assertTrue(kwargs["input_transpose"])
        self.assertTrue(kwargs["using_ue8m0_scale"])
        self.assertTrue(kwargs["using_pow2_scale"])
        self.assertIs(r_bwd, fp8_bwd)
        self.assertIs(r_fwd, fp8_fwd)
        self.assertEqual(r_scale_bwd.name, "scale_bwd.T")
        self.assertEqual(r_scale_fwd.name, "scale_fwd.T")


if __name__ == "__main__":
    unittest.main()
