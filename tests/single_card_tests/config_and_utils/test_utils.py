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

"""Behavior tests for the pure-Python helpers in paddlefleet.utils.

This is the base slice of the paddlefleet.utils coverage; sibling files
test_utils_2/_3/_4/_5.py concurrently cover the remaining helpers. This file
deliberately takes the "obvious first third": the simplest pure-Python helpers
with clear I/O contracts plus the two small container/buffer classes.

Slice covered here:
  * WrappedTensor           - single-use unwrap semantics (object identity)
  * GlobalMemoryBuffer      - allocate-vs-reuse decision and grow logic,
                              observed through a real mem_alloc_context whose
                              enter count reveals every (re)allocation
  * ensure_divisibility     - divisible passes, indivisible raises w/ message
  * divide                  - integer quotient and the indivisible guard
  * deprecate_inference_params - context/params precedence and the warning
  * get_attr_wrapped_model  - .module unwrapping, allow_none semantics,
                              return_model_obj, and the two RuntimeError paths
  * get_model_type / get_model_xattn / get_model_config - the thin wrappers

Left to siblings: init_method_normal / scaled_init_method_normal /
truncated_init_method_normal / get_magic_init_method, get_pg_size /
get_pg_rank, log_single_rank, get_tensor_model_parallel_group_if_none,
prepare_input_tensors_for_wgrad_compute, get_paddle_version /
is_paddle_min_version, get_batch_on_this_cp_rank, the NVTX helpers, and the
make_viewless_tensor / MakeViewlessTensor family.

Every expected value below is hand-derived. Collaborators are genuine (plain
Python holder objects, a real recording context manager); the code under test
is never mocked.

paddlefleet.utils.__init__ lazily loads _fleet_utils, which imports paddle
unconditionally at module top, so nothing here is reachable without paddle.
Only a genuine missing dependency (ImportError) is allowed to skip; any other
error surfaces as a real failure rather than a fake pass.
"""

import os
import sys
import unittest
import warnings

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "src"))
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    import paddle

    from paddlefleet.utils import (
        GlobalMemoryBuffer,
        WrappedTensor,
        deprecate_inference_params,
        divide,
        ensure_divisibility,
        get_attr_wrapped_model,
        get_model_config,
        get_model_type,
        get_model_xattn,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - depends on environment
    _IMPORT_ERROR = exc

_SKIP_REASON = (
    None
    if _IMPORT_ERROR is None
    else f"paddle / paddlefleet.utils not importable: {_IMPORT_ERROR!r}"
)


class _RecordingAllocContext:
    """A genuine context-manager factory handed to GlobalMemoryBuffer.

    get_tensor enters this context exactly once per real (re)allocation and
    never on a cache hit, so the recorded enter count is a direct, non-mocking
    witness of the allocate-vs-reuse decision under test.
    """

    def __init__(self):
        self.enter_count = 0
        self.exit_count = 0

    def __call__(self):
        return self

    def __enter__(self):
        self.enter_count += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit_count += 1
        return False


class _Holder:
    """Plain attribute bag used as a genuine collaborator for the model
    getters; unlike MagicMock it does not synthesize arbitrary attributes,
    so hasattr/getattr reflect exactly what the test wired up."""


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestWrappedTensor(unittest.TestCase):
    """WrappedTensor stores one indirect reference and yields it exactly once."""

    def test_unwrap_returns_the_exact_wrapped_object(self):
        # __init__ only stores the argument in a list, so any sentinel works
        # and identity (not equality) is the contract.
        sentinel = object()
        wrapped = WrappedTensor(sentinel)
        self.assertIs(wrapped.unwrap(), sentinel)

    def test_second_unwrap_raises_runtime_error(self):
        wrapped = WrappedTensor(object())
        wrapped.unwrap()
        with self.assertRaises(RuntimeError):
            wrapped.unwrap()


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestGlobalMemoryBuffer(unittest.TestCase):
    """Allocation is keyed by (name, dtype) and reused while the cached flat
    buffer holds at least the requested number of elements."""

    def test_returns_requested_shape_and_dtype(self):
        buf = GlobalMemoryBuffer()
        t = buf.get_tensor([3, 4], paddle.float32, "a")
        self.assertEqual(list(t.shape), [3, 4])
        self.assertEqual(t.dtype, paddle.float32)

    def test_same_then_smaller_request_reuses_single_allocation(self):
        # First [4, 4] -> 16 elements allocated once. A repeat [4, 4] and a
        # smaller [2, 2] -> 4 <= 16 must both hit the cache: no new enter.
        buf = GlobalMemoryBuffer()
        ctx = _RecordingAllocContext()
        first = buf.get_tensor([4, 4], paddle.float32, "reuse", ctx)
        self.assertEqual(ctx.enter_count, 1)
        again = buf.get_tensor([4, 4], paddle.float32, "reuse", ctx)
        self.assertEqual(ctx.enter_count, 1)
        smaller = buf.get_tensor([2, 2], paddle.float32, "reuse", ctx)
        self.assertEqual(ctx.enter_count, 1)
        # The views still carry the freshly requested shapes.
        self.assertEqual(list(first.shape), [4, 4])
        self.assertEqual(list(again.shape), [4, 4])
        self.assertEqual(list(smaller.shape), [2, 2])

    def test_larger_request_triggers_reallocation(self):
        # [2, 2] -> 4 elements, then [4, 4] -> 16 > 4 forces a second alloc.
        buf = GlobalMemoryBuffer()
        ctx = _RecordingAllocContext()
        buf.get_tensor([2, 2], paddle.float32, "grow", ctx)
        self.assertEqual(ctx.enter_count, 1)
        grown = buf.get_tensor([4, 4], paddle.float32, "grow", ctx)
        self.assertEqual(ctx.enter_count, 2)
        self.assertEqual(list(grown.shape), [4, 4])

    def test_distinct_names_allocate_independently(self):
        # Same shape/dtype but different name -> different key -> two allocs.
        buf = GlobalMemoryBuffer()
        ctx = _RecordingAllocContext()
        ta = buf.get_tensor([2, 2], paddle.float32, "name_a", ctx)
        tb = buf.get_tensor([2, 2], paddle.float32, "name_b", ctx)
        self.assertEqual(ctx.enter_count, 2)
        self.assertEqual(list(ta.shape), [2, 2])
        self.assertEqual(list(tb.shape), [2, 2])


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestEnsureDivisibility(unittest.TestCase):
    def test_divisible_values_do_not_raise(self):
        # 12 % 4 == 0, 7 % 7 == 0, 0 % 5 == 0 -> all return None silently.
        self.assertIsNone(ensure_divisibility(12, 4))
        self.assertIsNone(ensure_divisibility(7, 7))
        self.assertIsNone(ensure_divisibility(0, 5))

    def test_indivisible_raises_with_message(self):
        # 10 % 3 == 1 -> AssertionError carrying the two operands.
        with self.assertRaises(AssertionError) as cm:
            ensure_divisibility(10, 3)
        self.assertIn("10 is not divisible by 3", str(cm.exception))


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestDivide(unittest.TestCase):
    def test_returns_integer_quotient(self):
        self.assertEqual(divide(12, 4), 3)
        self.assertEqual(divide(20, 5), 4)
        self.assertEqual(divide(7, 1), 7)

    def test_indivisible_raises(self):
        with self.assertRaises(AssertionError):
            divide(10, 3)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestDeprecateInferenceParams(unittest.TestCase):
    """context wins whenever it is provided; only the (None, params) case
    both warns and forwards the legacy argument."""

    def test_both_none_returns_none(self):
        self.assertIsNone(deprecate_inference_params(None, None))

    def test_context_takes_precedence_over_params(self):
        context = object()
        params = object()
        # context present -> returned regardless of params; identity checked.
        self.assertIs(deprecate_inference_params(context, params), context)
        # context present, params absent -> still returns context.
        self.assertIs(deprecate_inference_params(context, None), context)

    def test_only_params_warns_and_forwards_params(self):
        params = object()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = deprecate_inference_params(None, params)
        self.assertIs(result, params)
        self.assertEqual(len(caught), 1)
        self.assertIn("inference_params", str(caught[0].message))


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestGetAttrWrappedModel(unittest.TestCase):
    """Walks the .module chain according to allow_none, then returns either
    the attribute value or the owning object."""

    def test_direct_attribute_returned(self):
        model = _Holder()
        model.tag = "value"
        self.assertEqual(get_attr_wrapped_model(model, "tag"), "value")

    def test_unwraps_module_chain_when_allow_none_false(self):
        # outer.tag is None so, with allow_none=False, the search descends
        # into outer.module (inner) whose tag is the first non-None value.
        inner = _Holder()
        inner.tag = "deep"
        outer = _Holder()
        outer.tag = None
        outer.module = inner
        self.assertEqual(
            get_attr_wrapped_model(outer, "tag", allow_none=False), "deep"
        )
        # return_model_obj yields the object that actually owns the attribute.
        self.assertIs(
            get_attr_wrapped_model(
                outer, "tag", allow_none=False, return_model_obj=True
            ),
            inner,
        )

    def test_list_model_raises(self):
        with self.assertRaises(RuntimeError):
            get_attr_wrapped_model([_Holder()], "tag")

    def test_missing_attribute_without_module_raises(self):
        # allow_none default True: attribute absent and no .module to descend.
        with self.assertRaises(RuntimeError):
            get_attr_wrapped_model(_Holder(), "nonexistent")


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestModelGetters(unittest.TestCase):
    """Thin wrappers over get_attr_wrapped_model with fixed attribute names."""

    def test_get_model_type_reads_model_type(self):
        model = _Holder()
        model.model_type = "gpt-mini"
        self.assertEqual(get_model_type(model), "gpt-mini")

    def test_get_model_xattn_present_and_absent(self):
        present = _Holder()
        present.xattn_needed = True
        self.assertIs(get_model_xattn(present), True)
        # Absent attribute with no .module raises inside the wrapper, which
        # swallows the RuntimeError and reports False.
        self.assertIs(get_model_xattn(_Holder()), False)

    def test_get_model_config_unwraps_with_allow_none_false(self):
        config = object()
        inner = _Holder()
        inner.config = config
        outer = _Holder()
        outer.config = None
        outer.module = inner
        self.assertIs(get_model_config(outer), config)


if __name__ == "__main__":
    unittest.main()
