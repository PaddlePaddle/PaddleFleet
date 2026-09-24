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

"""Behavior tests for helpers in ``paddlefleet.utils._fleet_utils``.

The production module (re-exported through ``paddlefleet.utils``) collects small
distributed-training utilities. These tests drive the *real* functions and
compare against expected values derived by hand from each function's documented
contract -- never from the implementation itself:

* ``divide`` / ``ensure_divisibility`` -- integer division guarded by a
  divisibility assertion.
* ``init_method_normal`` / ``scaled_init_method_normal`` -- build a
  ``functools.partial`` around ``paddle.nn.init.normal_`` with a standard
  deviation computed as ``sigma / sqrt(multiplier * num_layers)``. The tests
  recompute that ``std`` independently and check the wired keyword, i.e. the
  parameter actually consumed downstream, not merely that a partial is returned.
* ``get_pg_size`` / ``get_pg_rank`` -- collapse to ``1`` / ``0`` when the process
  group is trivial or distributed is uninitialised, else read ``nranks`` /
  ``rank`` off the group. Distributed initialisation is a collaborator and is
  mocked; the branching logic under test is executed for real.
* ``prepare_input_tensors_for_wgrad_compute`` -- flattens leading dims of a 3D
  tensor to 2D; the tests assert the full reshaped *contents*, not just shape.
* ``is_paddle_min_version`` -- version comparison with an ``check_equality``
  toggle and an ``ImportError`` contract when ``packaging`` is absent.
* ``deprecate_inference_params`` -- forwards / warns following its precedence.
* ``get_attr_wrapped_model`` and friends -- unwrap the ``.module`` chain.
* ``GlobalMemoryBuffer.get_tensor`` -- reuse-or-reallocate flat storage.
* ``WrappedTensor`` -- single-shot unwrap.
* ``nvtx_decorator`` / ``nvtx_range_pop`` -- disabled-path identity and the
  empty-stack guard.

The module imports ``paddle`` at import time, so the whole suite is skipped with
an honest reason when Paddle / paddlefleet is unavailable rather than reporting a
hollow pass. Nothing here fakes a ``world_size`` to stand in for real multi-card
numerics; every assertion concerns locally observable, single-process behaviour.
"""

import math
import os
import sys
import types
import unittest
import warnings
from unittest import mock

# Make the in-tree ``src/paddlefleet`` importable when the package has not been
# pip-installed (repo_root/src is 4 levels up from this file).
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    import numpy as np
    import paddle

    import paddlefleet.utils._fleet_utils as fu

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle / paddlefleet not installed on this host
    np = None
    fu = None
    _IMPORT_ERROR = exc


_skip_reason = f"paddle/paddlefleet unavailable: {_IMPORT_ERROR}"


@unittest.skipUnless(_IMPORT_ERROR is None, _skip_reason)
class TestDivide(unittest.TestCase):
    """``divide`` performs a divisibility-checked integer division."""

    def test_exact_division_values(self):
        # Hand-derived: exact quotients.
        self.assertEqual(fu.divide(12, 4), 3)
        self.assertEqual(fu.divide(100, 10), 10)
        self.assertEqual(fu.divide(7, 1), 7)
        # Result is a Python int (floor division), not a float.
        self.assertIsInstance(fu.divide(12, 4), int)

    def test_non_divisible_raises_assertion(self):
        with self.assertRaises(AssertionError):
            fu.divide(7, 3)

    def test_ensure_divisibility_contract(self):
        # Returns None (no raise) when divisible; raises otherwise.
        self.assertIsNone(fu.ensure_divisibility(10, 5))
        self.assertIsNone(fu.ensure_divisibility(0, 1))
        with self.assertRaises(AssertionError):
            fu.ensure_divisibility(5, 3)


@unittest.skipUnless(_IMPORT_ERROR is None, _skip_reason)
class TestInitMethods(unittest.TestCase):
    """Init factories wire ``normal_`` with an independently computed ``std``."""

    def test_init_method_normal_wiring(self):
        fn = fu.init_method_normal(0.02)
        # The factory must target paddle's in-place normal initializer with the
        # given sigma as std and a zero mean.
        self.assertIs(fn.func, paddle.nn.init.normal_)
        self.assertEqual(fn.keywords["mean"], 0.0)
        self.assertEqual(fn.keywords["std"], 0.02)

    def test_scaled_std_is_sigma_over_sqrt_mult_layers(self):
        # std = sigma / sqrt(multiplier * num_layers). Two distinguishable
        # anchors computed by hand.
        fn = fu.scaled_init_method_normal(
            0.02, num_layers=8
        )  # default mult=2.0
        self.assertIs(fn.func, paddle.nn.init.normal_)
        self.assertEqual(fn.keywords["mean"], 0.0)
        self.assertAlmostEqual(
            fn.keywords["std"], 0.02 / math.sqrt(2.0 * 8), places=12
        )
        # sqrt(2*8)=4 -> 0.005 exactly.
        self.assertAlmostEqual(fn.keywords["std"], 0.005, places=12)

        fn2 = fu.scaled_init_method_normal(0.02, num_layers=50)
        # sqrt(2*50)=10 -> 0.002 exactly; differs from the first anchor.
        self.assertAlmostEqual(fn2.keywords["std"], 0.002, places=12)
        self.assertNotAlmostEqual(
            fn.keywords["std"], fn2.keywords["std"], places=6
        )

    def test_scaled_std_consumes_multiplier(self):
        # Changing only the multiplier must change std by 1/sqrt(mult) ratio.
        base = fu.scaled_init_method_normal(0.1, num_layers=4, multiplier=1.0)
        quad = fu.scaled_init_method_normal(0.1, num_layers=4, multiplier=4.0)
        self.assertAlmostEqual(
            base.keywords["std"], 0.1 / math.sqrt(4.0), places=12
        )
        self.assertAlmostEqual(
            quad.keywords["std"], 0.1 / math.sqrt(16.0), places=12
        )
        # multiplier x4 -> std halved.
        self.assertAlmostEqual(
            quad.keywords["std"], base.keywords["std"] / 2.0, places=12
        )


@unittest.skipUnless(_IMPORT_ERROR is None, _skip_reason)
class TestProcessGroupSizeAndRank(unittest.TestCase):
    """``get_pg_size`` / ``get_pg_rank`` collapse trivial groups then read the group."""

    def test_uninitialized_defaults_are_size1_rank0(self):
        # Real path: distributed is not initialised in this single process, so
        # a ``None`` group must yield world size 1 and rank 0.
        self.assertFalse(paddle.distributed.is_initialized())
        self.assertEqual(fu.get_pg_size(group=None), 1)
        self.assertEqual(fu.get_pg_rank(group=None), 0)

    def test_none_group_short_circuits_even_when_initialized(self):
        with mock.patch("paddle.distributed.is_initialized", return_value=True):
            self.assertEqual(fu.get_pg_size(group=None), 1)
            self.assertEqual(fu.get_pg_rank(group=None), 0)

    def test_single_rank_group_size_is_one(self):
        group = types.SimpleNamespace(ranks=[0], nranks=1)
        with mock.patch("paddle.distributed.is_initialized", return_value=True):
            # len(ranks) == 1 branch: size 1 regardless of nranks.
            self.assertEqual(fu.get_pg_size(group=group), 1)

    def test_multi_rank_group_reads_nranks_and_rank(self):
        group = types.SimpleNamespace(ranks=[0, 1, 2, 3], nranks=4, rank=2)
        with mock.patch("paddle.distributed.is_initialized", return_value=True):
            self.assertEqual(fu.get_pg_size(group=group), 4)
            self.assertEqual(fu.get_pg_rank(group=group), 2)


@unittest.skipUnless(_IMPORT_ERROR is None, _skip_reason)
class TestPrepareInputTensorsForWgrad(unittest.TestCase):
    """3D activations are flattened to 2D preserving row-major contents."""

    def test_2d_input_unchanged_shape_and_values(self):
        grad = paddle.arange(8 * 16, dtype="float32").reshape([8, 16])
        inp = (paddle.arange(8 * 16, dtype="float32") + 1000.0).reshape([8, 16])
        out_grad, out_inp = fu.prepare_input_tensors_for_wgrad_compute(
            grad, inp
        )
        self.assertEqual(list(out_grad.shape), [8, 16])
        self.assertEqual(list(out_inp.shape), [8, 16])
        np.testing.assert_array_equal(
            out_grad.numpy(), np.arange(128, dtype=np.float32).reshape(8, 16)
        )
        np.testing.assert_array_equal(
            out_inp.numpy(),
            (np.arange(128, dtype=np.float32) + 1000.0).reshape(8, 16),
        )

    def test_3d_flattened_to_2d_keeps_row_major_order(self):
        grad = paddle.arange(2 * 4 * 16, dtype="float32").reshape([2, 4, 16])
        inp = (paddle.arange(2 * 4 * 16, dtype="float32") + 7.0).reshape(
            [2, 4, 16]
        )
        out_grad, out_inp = fu.prepare_input_tensors_for_wgrad_compute(
            grad, inp
        )
        # Leading dims collapse: [2,4,16] -> [8,16], contents unchanged.
        self.assertEqual(list(out_grad.shape), [8, 16])
        self.assertEqual(list(out_inp.shape), [8, 16])
        np.testing.assert_array_equal(
            out_grad.numpy(), np.arange(128, dtype=np.float32).reshape(8, 16)
        )
        np.testing.assert_array_equal(
            out_inp.numpy(),
            (np.arange(128, dtype=np.float32) + 7.0).reshape(8, 16),
        )


@unittest.skipUnless(_IMPORT_ERROR is None, _skip_reason)
class TestIsPaddleMinVersion(unittest.TestCase):
    """Version comparison honours the ``check_equality`` toggle and dep guard."""

    def test_equal_version_respects_equality_toggle(self):
        # Comparing the installed version against itself: >= is True, > is False.
        # Expected values follow from ordering semantics, independent of impl.
        current = paddle.__version__
        self.assertTrue(fu.is_paddle_min_version(current, check_equality=True))
        self.assertFalse(
            fu.is_paddle_min_version(current, check_equality=False)
        )

    def test_missing_packaging_raises_import_error(self):
        # When the packaging dependency is unavailable the function must raise
        # ImportError (a precise contract), not silently return a bool.
        with mock.patch.object(fu, "HAVE_PACKAGING", False):  # noqa: SIM117
            with self.assertRaises(ImportError):
                fu.is_paddle_min_version("3.0.0")


@unittest.skipUnless(_IMPORT_ERROR is None, _skip_reason)
class TestDeprecateInferenceParams(unittest.TestCase):
    """Precedence and deprecation warning for the renamed argument."""

    def test_both_none_returns_none_without_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertIsNone(fu.deprecate_inference_params(None, None))
        self.assertEqual(len(caught), 0)

    def test_context_present_takes_precedence_no_warning(self):
        context = object()
        params = object()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = fu.deprecate_inference_params(context, params)
        # Given context wins; params ignored; no deprecation warning.
        self.assertIs(result, context)
        self.assertEqual(len(caught), 0)

    def test_legacy_params_only_warns_and_forwards(self):
        params = object()
        with self.assertWarns(UserWarning):
            result = fu.deprecate_inference_params(None, params)
        # The legacy object is forwarded as the new context.
        self.assertIs(result, params)


@unittest.skipUnless(_IMPORT_ERROR is None, _skip_reason)
class TestGetAttrWrappedModel(unittest.TestCase):
    """Unwrap the ``.module`` chain to locate an attribute."""

    def test_direct_attribute_returned(self):
        model = types.SimpleNamespace(model_type="gpt")
        self.assertEqual(fu.get_attr_wrapped_model(model, "model_type"), "gpt")

    def test_unwraps_nested_module_and_can_return_owner(self):
        inner = types.SimpleNamespace(foo=42)
        outer = types.SimpleNamespace(module=inner)  # no ``foo`` on outer
        self.assertEqual(fu.get_attr_wrapped_model(outer, "foo"), 42)
        self.assertIs(
            fu.get_attr_wrapped_model(outer, "foo", return_model_obj=True),
            inner,
        )

    def test_allow_none_flag_changes_stop_condition(self):
        inner = types.SimpleNamespace(cfg="C")
        outer = types.SimpleNamespace(cfg=None, module=inner)
        # allow_none=True stops at the first object that *has* the attribute,
        # even if its value is None.
        self.assertIsNone(fu.get_attr_wrapped_model(outer, "cfg"))
        # allow_none=False keeps unwrapping past a None-valued attribute.
        self.assertEqual(
            fu.get_attr_wrapped_model(outer, "cfg", allow_none=False), "C"
        )

    def test_list_input_raises(self):
        with self.assertRaises(RuntimeError):
            fu.get_attr_wrapped_model([types.SimpleNamespace()], "x")

    def test_missing_attribute_without_module_raises(self):
        model = types.SimpleNamespace(x=1)
        with self.assertRaises(RuntimeError):
            fu.get_attr_wrapped_model(model, "y")

    def test_get_model_type_and_xattn_behaviour(self):
        self.assertEqual(
            fu.get_model_type(types.SimpleNamespace(model_type="llama")),
            "llama",
        )
        # xattn present -> returned; absent (RuntimeError swallowed) -> False.
        self.assertTrue(
            fu.get_model_xattn(types.SimpleNamespace(xattn_needed=True))
        )
        self.assertFalse(fu.get_model_xattn(types.SimpleNamespace(other=1)))


@unittest.skipUnless(_IMPORT_ERROR is None, _skip_reason)
class TestGlobalMemoryBuffer(unittest.TestCase):
    """Flat storage is reused when large enough and reallocated when it grows."""

    def test_returns_view_over_buffer_contents_in_row_major(self):
        buf = fu.GlobalMemoryBuffer()
        buf.get_tensor([3, 3], "float32", "a")  # allocates flat length 9
        flat = buf.buffer[("a", "float32")]
        self.assertEqual(list(flat.shape), [9])
        # Fill the underlying flat buffer with a known ramp.
        paddle.assign(paddle.arange(9, dtype="float32"), output=flat)
        # A smaller request reuses the same storage; the returned tensor is a
        # row-major view over the first 6 elements.
        view = buf.get_tensor([2, 3], "float32", "a")
        self.assertEqual(list(view.shape), [2, 3])
        np.testing.assert_array_equal(
            view.numpy(), np.arange(6, dtype=np.float32).reshape(2, 3)
        )

    def test_reuse_then_reallocate_on_growth(self):
        buf = fu.GlobalMemoryBuffer()
        buf.get_tensor([3, 3], "float32", "a")  # len 9
        first = buf.buffer[("a", "float32")]
        buf.get_tensor([2, 2], "float32", "a")  # 4 <= 9 -> reuse
        self.assertIs(buf.buffer[("a", "float32")], first)
        buf.get_tensor([4, 4], "float32", "a")  # 16 > 9 -> reallocate
        grown = buf.buffer[("a", "float32")]
        self.assertIsNot(grown, first)
        self.assertEqual(list(grown.shape), [16])

    def test_distinct_names_and_dtypes_are_separate_buffers(self):
        buf = fu.GlobalMemoryBuffer()
        buf.get_tensor([2, 2], "float32", "x")
        buf.get_tensor([2, 2], "float32", "y")
        buf.get_tensor([2, 2], "float16", "x")
        self.assertIn(("x", "float32"), buf.buffer)
        self.assertIn(("y", "float32"), buf.buffer)
        self.assertIn(("x", "float16"), buf.buffer)
        self.assertIsNot(
            buf.buffer[("x", "float32")], buf.buffer[("y", "float32")]
        )
        self.assertIsNot(
            buf.buffer[("x", "float32")], buf.buffer[("x", "float16")]
        )


@unittest.skipUnless(_IMPORT_ERROR is None, _skip_reason)
class TestWrappedTensor(unittest.TestCase):
    """``unwrap`` yields the stored object exactly once."""

    def test_unwrap_returns_same_object(self):
        sentinel = object()
        wrapped = fu.WrappedTensor(sentinel)
        self.assertIs(wrapped.unwrap(), sentinel)

    def test_second_unwrap_raises_runtime_error(self):
        wrapped = fu.WrappedTensor(object())
        wrapped.unwrap()
        with self.assertRaises(RuntimeError):
            wrapped.unwrap()


@unittest.skipUnless(_IMPORT_ERROR is None, _skip_reason)
class TestNvtxRanges(unittest.TestCase):
    """Disabled decorator is a no-op; popping an empty range stack raises."""

    def test_decorator_disabled_returns_original_function(self):
        original = fu._nvtx_enabled
        self.addCleanup(setattr, fu, "_nvtx_enabled", original)
        fu._nvtx_enabled = False

        def my_func():
            return 123

        wrapped = fu.nvtx_decorator()(my_func)
        # No wrapping when profiling is disabled: same function object, and it
        # still returns its real result.
        self.assertIs(wrapped, my_func)
        self.assertEqual(wrapped(), 123)

    def test_pop_from_empty_stack_raises(self):
        orig_enabled = fu._nvtx_enabled
        orig_messages = fu._nvtx_range_messages
        self.addCleanup(setattr, fu, "_nvtx_enabled", orig_enabled)
        self.addCleanup(setattr, fu, "_nvtx_range_messages", orig_messages)
        fu._nvtx_enabled = True
        fu._nvtx_range_messages = []
        # Pass an explicit msg so the empty-stack guard is what triggers, not
        # the caller-path frame introspection.
        with self.assertRaises(RuntimeError):
            fu.nvtx_range_pop(msg="probe")


if __name__ == "__main__":
    unittest.main()
