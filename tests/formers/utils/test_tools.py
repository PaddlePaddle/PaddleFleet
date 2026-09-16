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

"""Behavior tests for paddlefleet.utils.tools.

These exercise the real helper functions in ``src/paddlefleet/utils/tools.py``.
Every expected value is hand-derived from the documented contract and the
control flow of the function, never produced by calling the function under
test. For ``get_env_device`` the four probing paddle APIs are patched so each
branch (and its priority order) can be asserted to return the exact device
string. Environment / paddle-API patches are always scoped with ``with`` blocks
so state is restored on exit.

No GPU is required (no-card CPU); the numeric helpers are pure Python/NumPy and
``get_env_device`` / ``device_guard`` observe branch selection through patched
collaborators rather than real device numerics.
"""

import contextlib
import unittest
from unittest.mock import patch

from paddlefleet.utils.tools import (
    PaddleDeviceWrapper,
    TimeCostAverage,
    compare_version,
    device_guard,
    dispatch_to,
    get_bool_ids_greater_than,
    get_env_device,
    get_span,
)


class TestCompareVersion(unittest.TestCase):
    """compare_version -> 1 if version > pair, 0 if equal, -1 if less.

    Segments are compared numerically left-to-right; a non-numeric segment on
    the ``version`` side yields -1, on the ``pair_version`` side yields 1.
    """

    def test_equal_returns_zero(self):
        self.assertEqual(compare_version("2.2.0", "2.2.0"), 0)

    def test_greater_patch_component(self):
        self.assertEqual(compare_version("2.2.1", "2.2.0"), 1)

    def test_less_minor_component(self):
        self.assertEqual(compare_version("2.1.0", "2.2.0"), -1)

    def test_numeric_not_lexical(self):
        # 10 > 9 numerically; a lexical string compare would wrongly give -1.
        self.assertEqual(compare_version("2.10.0", "2.9.0"), 1)

    def test_rc_suffix_is_treated_as_less(self):
        # "0-rc0" is non-numeric on the version side -> -1, regardless of value.
        self.assertEqual(compare_version("2.2.0-rc0", "2.2.0"), -1)

    def test_earlier_numeric_segment_wins_over_rc(self):
        # Minor 3 > 2 is decided before the rc segment is ever inspected.
        self.assertEqual(compare_version("2.3.0-rc0", "2.2.0"), 1)

    def test_non_numeric_on_pair_side_returns_one(self):
        self.assertEqual(compare_version("2.2.0", "2.2.x"), 1)

    def test_whitespace_is_stripped(self):
        self.assertEqual(compare_version("  2.2.0  ", "2.2.0"), 0)

    def test_shorter_prefix_equal_via_zip_truncation(self):
        # zip stops at the shorter list; "2.2" and "2.2.0" compare equal.
        self.assertEqual(compare_version("2.2", "2.2.0"), 0)


class TestGetBoolIdsGreaterThan(unittest.TestCase):
    """get_bool_ids_greater_than -> indices whose prob is strictly > limit.

    With return_prob the elements become (index, prob) tuples; a >1-D input
    recurses per row and returns a nested list preserving row structure.
    """

    def test_simple_indices_exact(self):
        probs = [0.1, 0.6, 0.8, 0.3]
        # Only 0.6 (idx 1) and 0.8 (idx 2) exceed 0.5.
        self.assertEqual(get_bool_ids_greater_than(probs, limit=0.5), [1, 2])

    def test_strict_boundary_excludes_equal(self):
        # 0.5 is not > 0.5; only 0.51 (idx 1) qualifies.
        self.assertEqual(get_bool_ids_greater_than([0.5, 0.51], limit=0.5), [1])

    def test_return_prob_pairs_index_and_value(self):
        probs = [0.1, 0.6, 0.8, 0.3]
        result = get_bool_ids_greater_than(probs, limit=0.5, return_prob=True)
        self.assertEqual([idx for idx, _ in result], [1, 2])
        self.assertAlmostEqual(float(result[0][1]), 0.6, places=7)
        self.assertAlmostEqual(float(result[1][1]), 0.8, places=7)

    def test_two_dimensional_recurses_per_row(self):
        probs = [[0.1, 0.6], [0.8, 0.3]]
        # Row 0: 0.6 at idx 1 -> [1]; Row 1: 0.8 at idx 0 -> [0].
        self.assertEqual(
            get_bool_ids_greater_than(probs, limit=0.5), [[1], [0]]
        )

    def test_default_limit_is_half(self):
        # Default limit 0.5: 0.9 (idx 1) qualifies, 0.2 (idx 0) does not.
        self.assertEqual(get_bool_ids_greater_than([0.2, 0.9]), [1])


class TestGetSpan(unittest.TestCase):
    """get_span -> set of (start, end) couples with no index reused.

    A two-pointer walk over sorted starts/ends pairs each end with the latest
    unused start <= that end. With with_prob each element is a (index, prob)
    tuple sorted/compared on the index.
    """

    def test_simple_pairs_exact_set(self):
        # starts [0,2], ends [1,3] -> couples {(0,1),(2,3)}.
        result = get_span([0, 2], [1, 3])
        self.assertEqual(result, {(0, 1), (2, 3)})

    def test_equal_start_end_is_paired(self):
        # starts [1,2], ends [1,3]: 1==1 pairs, then 2<3 pairs.
        result = get_span([1, 2], [1, 3])
        self.assertEqual(result, {(1, 1), (2, 3)})

    def test_with_prob_keeps_tuples(self):
        start_ids = [(0, 0.9), (2, 0.8)]
        end_ids = [(1, 0.7), (3, 0.6)]
        result = get_span(start_ids, end_ids, with_prob=True)
        self.assertEqual(
            result,
            {((0, 0.9), (1, 0.7)), ((2, 0.8), (3, 0.6))},
        )

    def test_unsorted_input_is_sorted_first(self):
        # Same couples as the sorted case even though input order differs.
        result = get_span([2, 0], [3, 1])
        self.assertEqual(result, {(0, 1), (2, 3)})


class TestDispatchTo(unittest.TestCase):
    """dispatch_to(dispatch_fn, cond) -> decorator.

    The wrapper calls dispatch_fn when cond(*args, **kwargs) is truthy, else the
    original fn. The original callable is preserved on wrapper.__original_fn__.
    """

    def test_default_cond_always_dispatches_and_forwards_args(self):
        seen = {}

        def dispatch_fn(*args, **kwargs):
            seen["args"] = args
            seen["kwargs"] = kwargs
            return "dispatched"

        def original_fn(*args, **kwargs):
            return "original"

        wrapped = dispatch_to(dispatch_fn)(original_fn)
        self.assertEqual(wrapped("x", 1, k=2), "dispatched")
        self.assertEqual(seen["args"], ("x", 1))
        self.assertEqual(seen["kwargs"], {"k": 2})
        self.assertIs(wrapped.__original_fn__, original_fn)

    def test_cond_false_calls_original(self):
        wrapped = dispatch_to(
            lambda *a, **k: "dispatched", cond=lambda *a, **k: False
        )(lambda *a, **k: "original")
        self.assertEqual(wrapped("x"), "original")

    def test_cond_depends_on_argument_value(self):
        def dispatch_fn(x):
            return "big"

        def original_fn(x):
            return "small"

        wrapped = dispatch_to(dispatch_fn, cond=lambda x: x > 10)(original_fn)
        self.assertEqual(wrapped(50), "big")
        self.assertEqual(wrapped(5), "small")


class TestTimeCostAverage(unittest.TestCase):
    """TimeCostAverage accumulates recorded times and averages them."""

    def test_empty_average_is_zero(self):
        self.assertEqual(TimeCostAverage().get_average(), 0)

    def test_average_of_records(self):
        avg = TimeCostAverage()
        avg.record(2.0)
        avg.record(4.0)
        # total 6.0 over 2 records -> 3.0
        self.assertAlmostEqual(avg.get_average(), 3.0, places=7)

    def test_reset_clears_state(self):
        avg = TimeCostAverage()
        avg.record(10.0)
        avg.reset()
        self.assertEqual(avg.get_average(), 0)
        avg.record(5.0)
        self.assertAlmostEqual(avg.get_average(), 5.0, places=7)


@contextlib.contextmanager
def _patched_backends(cuda, rocm, xpu, custom):
    """Patch the four paddle probes get_env_device consults, in one scope."""
    with (
        patch("paddle.is_compiled_with_cuda", return_value=cuda),
        patch("paddle.is_compiled_with_rocm", return_value=rocm),
        patch("paddle.is_compiled_with_xpu", return_value=xpu),
        patch("paddle.device.get_all_custom_device_type", return_value=custom),
    ):
        yield


class TestGetEnvDevice(unittest.TestCase):
    """get_env_device -> device string, checked in a fixed priority order:

    cuda -> "gpu", rocm -> "rocm", xpu -> "xpu", first custom device, else
    "cpu". Each branch is asserted with the exact expected string, including
    the priority: when several probes are True the earliest one wins.
    """

    def test_cuda_wins_even_when_all_true(self):
        # cuda checked first; must return "gpu" despite rocm/xpu/custom set.
        with _patched_backends(True, True, True, ["npu"]):
            self.assertEqual(get_env_device(), "gpu")

    def test_rocm_when_not_cuda(self):
        with _patched_backends(False, True, True, ["npu"]):
            self.assertEqual(get_env_device(), "rocm")

    def test_xpu_when_not_cuda_or_rocm(self):
        with _patched_backends(False, False, True, ["npu"]):
            self.assertEqual(get_env_device(), "xpu")

    def test_first_custom_device_when_no_builtin(self):
        # Returns custom_devices[0], not any later entry.
        with _patched_backends(False, False, False, ["npu", "mlu"]):
            self.assertEqual(get_env_device(), "npu")

    def test_cpu_when_nothing_available(self):
        with _patched_backends(False, False, False, []):
            self.assertEqual(get_env_device(), "cpu")


class TestDeviceGuard(unittest.TestCase):
    """device_guard sets the requested device on enter and restores on exit.

    paddle.set_device / paddle.device.get_device are collaborators here; the
    branch logic under test (which device string is chosen, and that the
    original is restored) is observed through the recorded set_device calls.
    """

    def test_cpu_sets_cpu_then_restores_origin(self):
        calls = []
        with (
            patch(
                "paddlefleet.utils.tools.is_paddle_available", return_value=True
            ),
            patch("paddle.device.get_device", return_value="gpu:3"),
            patch("paddle.set_device", side_effect=lambda d: calls.append(d)),
        ):
            with device_guard("cpu"):
                self.assertEqual(calls, ["cpu"])
            self.assertEqual(calls, ["cpu", "gpu:3"])

    def test_gpu_uses_dev_id_then_restores(self):
        calls = []
        with (
            patch(
                "paddlefleet.utils.tools.is_paddle_available", return_value=True
            ),
            patch("paddle.device.get_device", return_value="cpu"),
            patch("paddle.set_device", side_effect=lambda d: calls.append(d)),
        ):
            with device_guard("gpu", dev_id=2):
                self.assertEqual(calls, ["gpu:2"])
            self.assertEqual(calls, ["gpu:2", "cpu"])

    def test_raises_when_paddle_unavailable(self):
        with patch(
            "paddlefleet.utils.tools.is_paddle_available", return_value=False
        ):
            guard = device_guard("cpu")
            with self.assertRaises(ImportError), guard:
                pass


class TestPaddleDeviceWrapperGetNestedAttr(unittest.TestCase):
    """get_nested_attr walks a dotted path, returning default on any miss.

    Called unbound (self is unused by the method) so no paddle-dependent
    __init__ is triggered; the traversal logic is the behavior under test.
    """

    def _ns(self):
        from types import SimpleNamespace

        return SimpleNamespace(a=SimpleNamespace(b=SimpleNamespace(c=42)))

    def test_resolves_full_path(self):
        self.assertEqual(
            PaddleDeviceWrapper.get_nested_attr(None, self._ns(), "a.b.c"), 42
        )

    def test_missing_leaf_returns_default(self):
        self.assertEqual(
            PaddleDeviceWrapper.get_nested_attr(
                None, self._ns(), "a.b.x", default="dflt"
            ),
            "dflt",
        )

    def test_missing_intermediate_returns_default(self):
        self.assertIsNone(
            PaddleDeviceWrapper.get_nested_attr(None, self._ns(), "a.missing.c")
        )


if __name__ == "__main__":
    unittest.main()
