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

"""Behavior tests for the CPU-observable index dispatch of
``paddlefleet_ops._extensions.flashmask.rr_attn_estimate_triton_op._extract_raw_ptrs``.

``_extract_raw_ptrs(startend_row_indices, causal)`` takes a
``[B, HIDS, seqlen_q, mode]`` tensor and, purely on the host, decides which
last-axis *column* feeds each of the four token-level bound pointers
(``lt_start``, ``lt_end``, ``ut_start``, ``ut_end``) that later drive the GPU
Triton kernel. The wiring depends on both ``mode in {1, 2, 4}`` and ``causal``:

    mode=1                -> lt_start=col0; lt_end/ut_start/ut_end alias col0
    mode=2, causal=True   -> lt_start=col0, lt_end=col1; ut_* alias col0
    mode=2, causal=False  -> lt_start=col0, ut_end=col1; lt_end/ut_start alias col0
    mode=4                -> lt_start=col0, lt_end=col1, ut_start=col2, ut_end=col3
    mode not in {1,2,4}   -> ValueError (via ``_require``)

This is host-side orchestration/index math and runs on CPU-only paddle; no CUDA
device and no kernel launch are involved. Every expected column mapping below is
hand-derived from the routing table above -- we never call ``_extract_raw_ptrs``
(or any production helper) to produce its own expected values. Actual Triton
kernel numerics are a GPU concern and are explicitly NOT asserted here.
"""

import unittest

# The production module imports ``paddle`` (and Triton) at import time; the local
# CI image for this file may lack a working ``paddle`` install. Guard the import
# so the suite skips honestly instead of erroring at collection. Only a genuinely
# missing dependency (ImportError / ModuleNotFoundError) is treated as skip-able;
# other errors are left to surface as real failures.
try:
    import paddle
    from paddlefleet_ops._extensions.flashmask.rr_attn_estimate_triton_op import (
        RawPtrs,
        _extract_raw_ptrs,
    )

    _IMPORT_ERROR = None
except ImportError as exc:
    paddle = None
    RawPtrs = None
    _extract_raw_ptrs = None
    _IMPORT_ERROR = repr(exc)

_MODULE_AVAILABLE = _IMPORT_ERROR is None

# Distinct per-column constants so every expected value below is a hand-written
# literal (never derived from the function under test). Column c carries
# _COL_VALUES[c] == (c + 1) * 11; a mis-routed column shows up as the wrong
# constant. Defined by the same (c + 1) * 11 rule used in _make_indices so the
# two stay in lock-step for any column index.
_COL_VALUES = (11, 22, 33, 44)

# Small, fully-enumerable shape. Contents beyond the per-column constant are
# irrelevant to the routing logic under test.
_B, _HIDS, _S = 2, 2, 3


@unittest.skipUnless(
    _MODULE_AVAILABLE,
    f"paddlefleet_ops.rr_attn_estimate_triton_op not importable ({_IMPORT_ERROR})",
)
class TestExtractRawPtrsColumnDispatch(unittest.TestCase):
    """CPU-observable column routing of ``_extract_raw_ptrs``.

    Runs on CPU-only paddle: the function performs only tensor slicing, so no
    GPU device is required. Kernel numeric behavior is out of scope.
    """

    def _make_indices(self, mode):
        """Build a [B, HIDS, S, mode] int32 tensor whose column c is the
        constant _COL_VALUES[c]. mode is exactly the last-axis size."""
        paddle.set_device("cpu")
        cols = [
            paddle.full([_B, _HIDS, _S], (c + 1) * 11, dtype="int32")
            for c in range(mode)
        ]
        return paddle.stack(cols, axis=-1)

    def _assert_carries_column(self, field, col):
        """field must be a [B, HIDS, S] tensor whose every element equals the
        hand-derived constant for source column ``col``."""
        expected_value = _COL_VALUES[col]
        self.assertEqual(
            field.shape,
            [_B, _HIDS, _S],
            f"field should drop the mode axis, got shape {field.shape}",
        )
        self.assertTrue(
            bool((field == expected_value).all()),
            f"expected every element == {expected_value} (column {col}), got min={int(field.min())} max={int(field.max())}",
        )

    def test_mode1_all_bounds_take_column0(self):
        """mode=1 exposes only lt_start; the other three bounds alias col0.

        Hand-derived mapping: lt_start=col0, lt_end=col0, ut_start=col0,
        ut_end=col0. causal is irrelevant for mode=1, so both settings must
        produce identical column routing.
        """
        for causal in (True, False):
            with self.subTest(causal=causal):
                x = self._make_indices(mode=1)
                mode, raw = _extract_raw_ptrs(x, causal=causal)
                self.assertEqual(mode, 1)
                self.assertIsInstance(raw, RawPtrs)
                self._assert_carries_column(raw.lt_start, 0)
                self._assert_carries_column(raw.lt_end, 0)
                self._assert_carries_column(raw.ut_start, 0)
                self._assert_carries_column(raw.ut_end, 0)

    def test_mode2_causal_routes_second_column_to_lt_end(self):
        """mode=2 + causal=True: col1 is the lower-triangular END bound.

        Hand-derived: lt_start=col0, lt_end=col1, ut_start=col0, ut_end=col0.
        A wrong branch that sent col1 to ut_end instead would fail here.
        """
        x = self._make_indices(mode=2)
        mode, raw = _extract_raw_ptrs(x, causal=True)
        self.assertEqual(mode, 2)
        self._assert_carries_column(raw.lt_start, 0)
        self._assert_carries_column(raw.lt_end, 1)
        self._assert_carries_column(raw.ut_start, 0)
        self._assert_carries_column(raw.ut_end, 0)

    def test_mode2_noncausal_routes_second_column_to_ut_end(self):
        """mode=2 + causal=False: col1 is the upper-triangular END bound.

        Hand-derived: lt_start=col0, lt_end=col0, ut_start=col0, ut_end=col1.
        This is the mirror image of the causal case and exercises the ``causal``
        parameter actually changing where col1 lands.
        """
        x = self._make_indices(mode=2)
        mode, raw = _extract_raw_ptrs(x, causal=False)
        self.assertEqual(mode, 2)
        self._assert_carries_column(raw.lt_start, 0)
        self._assert_carries_column(raw.lt_end, 0)
        self._assert_carries_column(raw.ut_start, 0)
        self._assert_carries_column(raw.ut_end, 1)

    def test_causal_flag_flips_only_target_of_second_column(self):
        """Directly contrast the two mode=2 routings to pin the causal effect.

        The second column must reach lt_end under causal=True and ut_end under
        causal=False; the two outputs must therefore differ at exactly those
        slots. A function that ignored ``causal`` would make these equal.
        """
        x = self._make_indices(mode=2)
        _, raw_causal = _extract_raw_ptrs(x, causal=True)
        _, raw_noncausal = _extract_raw_ptrs(x, causal=False)

        # causal=True puts col1 (22) in lt_end; causal=False keeps col0 (11).
        self.assertTrue(bool((raw_causal.lt_end == _COL_VALUES[1]).all()))
        self.assertTrue(bool((raw_noncausal.lt_end == _COL_VALUES[0]).all()))
        # ...and symmetrically for ut_end.
        self.assertTrue(bool((raw_causal.ut_end == _COL_VALUES[0]).all()))
        self.assertTrue(bool((raw_noncausal.ut_end == _COL_VALUES[1]).all()))

    def test_mode4_routes_four_distinct_columns_in_order(self):
        """mode=4: each bound takes its own column, in order 0,1,2,3.

        Hand-derived: lt_start=col0, lt_end=col1, ut_start=col2, ut_end=col3.
        Distinct constants catch any swap of ut_start/ut_end (a plausible bug
        the coverage-only sibling could never detect). causal is irrelevant for
        mode=4 -> both settings must match.
        """
        for causal in (True, False):
            with self.subTest(causal=causal):
                x = self._make_indices(mode=4)
                mode, raw = _extract_raw_ptrs(x, causal=causal)
                self.assertEqual(mode, 4)
                self._assert_carries_column(raw.lt_start, 0)
                self._assert_carries_column(raw.lt_end, 1)
                self._assert_carries_column(raw.ut_start, 2)
                self._assert_carries_column(raw.ut_end, 3)

    def test_default_bounds_alias_lt_start_object(self):
        """Bounds not explicitly reassigned must be the *same object* as
        lt_start, not independent copies.

        This is a real allocation/identity contract: aliasing avoids redundant
        [B, HIDS, S] buffers, and the kernel receives one shared pointer. Only
        the columns the routing table reassigns get a fresh contiguous tensor.
        """
        # mode=1: all three defaults alias lt_start.
        _, raw1 = _extract_raw_ptrs(self._make_indices(1), causal=True)
        self.assertIs(raw1.lt_end, raw1.lt_start)
        self.assertIs(raw1.ut_start, raw1.lt_start)
        self.assertIs(raw1.ut_end, raw1.lt_start)

        # mode=2 causal: only lt_end is fresh; ut_start/ut_end alias lt_start.
        _, raw2c = _extract_raw_ptrs(self._make_indices(2), causal=True)
        self.assertIsNot(raw2c.lt_end, raw2c.lt_start)
        self.assertIs(raw2c.ut_start, raw2c.lt_start)
        self.assertIs(raw2c.ut_end, raw2c.lt_start)

        # mode=2 non-causal: only ut_end is fresh; lt_end/ut_start alias.
        _, raw2n = _extract_raw_ptrs(self._make_indices(2), causal=False)
        self.assertIs(raw2n.lt_end, raw2n.lt_start)
        self.assertIs(raw2n.ut_start, raw2n.lt_start)
        self.assertIsNot(raw2n.ut_end, raw2n.lt_start)

    def test_returned_mode_equals_last_axis_size(self):
        """The first return value is exactly the tensor's last-axis size."""
        for mode in (1, 2, 4):
            with self.subTest(mode=mode):
                x = self._make_indices(mode)
                returned_mode, _ = _extract_raw_ptrs(x, causal=True)
                self.assertEqual(returned_mode, mode)
                self.assertEqual(x.shape[-1], mode)

    def test_unsupported_mode_raises_valueerror(self):
        """A last-axis size outside {1, 2, 4} must be rejected via _require.

        mode=3 and mode=5 are the nearest invalid neighbors; both must raise
        ValueError rather than silently mis-slicing.
        """
        for bad_mode in (3, 5):
            with self.subTest(bad_mode=bad_mode):
                x = self._make_indices(bad_mode)
                with self.assertRaises(ValueError):
                    _extract_raw_ptrs(x, causal=True)


if __name__ == "__main__":
    unittest.main()
