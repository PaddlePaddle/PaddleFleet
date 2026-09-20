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

What is verified here, and why it is CPU-observable:

* The recipe guard raises ``ValueError`` with an exact, hand-derived message
  before any Paddle kernel is touched (guard is at the top of the function).
* In the default (non-UE8M0) path ``inp_quant_func`` is a ``functools.partial``
  whose ``.func`` is the real ``fp8_quant_blockwise`` kernel and whose bound
  keywords thread every caller flag through unchanged. This is inspectable
  without ever invoking the kernel.
* ``weight_quant_func`` short-circuits through its pre-quantized-weight cache
  (``_cached_weight_result``) when the input carries ``fp8_weight_fwd`` /
  ``fp8_scale_fwd`` / ``fp8_scale_bwd`` attributes, returning the 4-tuple
  ``(None, scale_bwd, fp8_fwd, scale_fwd)`` with no kernel launch. Using
  distinct sentinel objects and identity assertions pins the exact slot
  mapping, so any reordering of the tuple would be caught.
* Selecting ``use_ue8m0=True`` swaps ``inp_quant_func`` from a partial to a
  closure; asserting the partial-vs-closure distinction pins the branch.

Independence: expected values are hand-derived from the recipe guard string,
the documented tuple layout, and the literal keyword bindings in the source.
The function-under-test is never called to compute its own expected values.

Environment note: importing ``paddlefleet.fp8.quantization`` imports Paddle at
module load time. The local no-card environment has no Paddle installed, so
these tests skip honestly with the captured ImportError rather than passing
vacuously. The cache-hit and partial-inspection assertions are true CPU logic
and would execute on any host where Paddle imports; they do not require a GPU.
"""

import functools
import unittest

try:
    import paddle

    from paddlefleet.fp8.quantization import get_quant_func

    _IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - depends on local env
    paddle = None
    get_quant_func = None
    _IMPORT_ERROR = exc

_SKIP_REASON = (
    "paddlefleet.fp8.quantization requires paddle at import time; "
    f"paddle is not importable in this environment: {_IMPORT_ERROR}"
)


class _PreQuantizedWeight:
    """Plain carrier object exercising the weight cache short-circuit.

    ``_cached_weight_result`` reads the attributes below via ``getattr``; a
    non-tensor object is sufficient because the cache-hit branch never calls
    the Paddle kernel.
    """


@unittest.skipUnless(get_quant_func is not None, _SKIP_REASON)
class TestGetQuantFuncRecipeGuard(unittest.TestCase):
    """The recipe guard runs before any kernel reference is resolved."""

    def test_unsupported_recipe_raises_exact_message(self):
        # Hand-derived from the f-string in the source guard.
        bad_recipe = "rowwise_v9"
        expected = (
            "fp8_recipe rowwise_v9 is not supported. "
            "Supported recipes are blockwise."
        )
        with self.assertRaises(ValueError) as ctx:
            get_quant_func(bad_recipe)
        self.assertEqual(str(ctx.exception), expected)

    def test_blockwise_recipe_is_accepted(self):
        # Should not raise; returns a 2-tuple of callables.
        inp_func, weight_func = get_quant_func("blockwise")
        self.assertTrue(callable(inp_func))
        self.assertTrue(callable(weight_func))


@unittest.skipUnless(get_quant_func is not None, _SKIP_REASON)
class TestInpQuantPartialBindings(unittest.TestCase):
    """The default path binds every flag onto the kernel via functools.partial."""

    def test_defaults_bind_expected_keywords(self):
        inp_func, _ = get_quant_func("blockwise")
        self.assertIsInstance(inp_func, functools.partial)
        # Identity: the partial wraps the real blockwise quant kernel.
        self.assertIs(
            inp_func.func,
            paddle.incubate.nn.functional.fp8_quant_blockwise,
        )
        # Full keyword set, hand-derived from the default arguments.
        self.assertEqual(
            inp_func.keywords,
            {
                "output_scale_transpose": False,
                "quant_method": "1x128",
                "input_transpose": False,
                "using_pow2_scale": False,
            },
        )

    def test_flags_thread_through_to_keywords(self):
        inp_func, _ = get_quant_func(
            "blockwise",
            input_trans=True,
            out_scale_trans=True,
            pow2_scale=True,
        )
        self.assertIsInstance(inp_func, functools.partial)
        self.assertEqual(
            inp_func.keywords,
            {
                "output_scale_transpose": True,
                "quant_method": "1x128",
                "input_transpose": True,
                "using_pow2_scale": True,
            },
        )
        # Non-UE8M0 partial must not smuggle in a ue8m0 flag.
        self.assertNotIn("using_ue8m0_scale", inp_func.keywords)


@unittest.skipUnless(get_quant_func is not None, _SKIP_REASON)
class TestWeightQuantCacheHit(unittest.TestCase):
    """The weight closure returns the cached 4-tuple without a kernel launch.

    Distinct sentinel objects plus assertIs pin the documented tuple layout
    (fp8_bwd, scale_bwd, fp8_fwd, scale_fwd); a swapped slot would fail.
    """

    def _make_cached_weight(self):
        x = _PreQuantizedWeight()
        # Distinct objects so any reordering is detectable.
        x.fp8_weight_fwd = object()
        x.fp8_scale_fwd = object()
        x.fp8_scale_bwd = object()
        return x

    def test_cache_hit_tuple_layout_default_path(self):
        _, weight_func = get_quant_func("blockwise")
        x = self._make_cached_weight()
        result = weight_func(x)

        self.assertEqual(len(result), 4)
        # Slot 0 (fp8_bwd) is intentionally None; caller derives it via .T.
        self.assertIsNone(result[0])
        self.assertIs(result[1], x.fp8_scale_bwd)
        self.assertIs(result[2], x.fp8_weight_fwd)
        self.assertIs(result[3], x.fp8_scale_fwd)

    def test_cache_hit_tuple_layout_ue8m0_path(self):
        # The cache short-circuit precedes the UE8M0 branch, so the layout
        # is identical regardless of use_ue8m0.
        _, weight_func = get_quant_func("blockwise", use_ue8m0=True)
        x = self._make_cached_weight()
        result = weight_func(x)

        self.assertEqual(len(result), 4)
        self.assertIsNone(result[0])
        self.assertIs(result[1], x.fp8_scale_bwd)
        self.assertIs(result[2], x.fp8_weight_fwd)
        self.assertIs(result[3], x.fp8_scale_fwd)


@unittest.skipUnless(get_quant_func is not None, _SKIP_REASON)
class TestUe8m0BranchSelection(unittest.TestCase):
    """UE8M0 selection changes the inp callable from a partial to a closure."""

    def test_ue8m0_inp_is_closure_not_partial(self):
        inp_default, _ = get_quant_func("blockwise")
        inp_ue8m0, _ = get_quant_func("blockwise", use_ue8m0=True)

        # Default path is a partial; UE8M0 path is a plain closure.
        self.assertIsInstance(inp_default, functools.partial)
        self.assertNotIsInstance(inp_ue8m0, functools.partial)
        self.assertTrue(callable(inp_ue8m0))


if __name__ == "__main__":
    unittest.main()
