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

"""CPU-observable index-math tests for
``paddlefleet_ops._extensions.flashmask.rr_attn_estimate_triton_op._extract_raw_ptrs``.

``_extract_raw_ptrs`` is pure, GPU-free orchestration: it reads ``mode`` from
the trailing dimension of ``startend_row_indices`` ([B, HIDS, seqlen_q, mode]),
validates it, then slices out the per-column boundary tensors and fills the
unused boundaries with an *alias* of ``lt_start`` as a placeholder:

  * mode=1              -> lt_start=col0; lt_end/ut_start/ut_end alias lt_start
  * mode=2, causal=True -> lt_end=col1; ut_start/ut_end alias lt_start
  * mode=2, causal=False-> ut_end=col1; lt_end/ut_start alias lt_start
  * mode=4              -> lt_end=col1, ut_start=col2, ut_end=col3 (all distinct)
  * mode not in {1,2,4} -> ValueError

All expected values below are hand-written literals derived from a
distinguishable ``arange`` input; we never call ``_extract_raw_ptrs`` (or slice
the input the way it does) to compute its own expected output. The Triton
kernel numerics are a separate GPU concern and are NOT exercised here.
"""

import unittest

import numpy as np

# paddlefleet_ops imports ``paddle`` (and Triton) at import time; the local
# environment for this file may lack a working ``paddle`` install. Guard the
# import so the suite skips honestly instead of erroring at collection. Only a
# genuinely missing dependency is treated as skip -- not an implementation bug.
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


@unittest.skipUnless(
    _MODULE_AVAILABLE,
    "paddlefleet_ops.rr_attn_estimate_triton_op not importable "
    f"({_IMPORT_ERROR})",
)
class TestExtractRawPtrs(unittest.TestCase):
    """Mode dispatch, column selection and placeholder aliasing.

    Runs on CPU-only paddle: ``_extract_raw_ptrs`` performs only
    contiguous/slice operations, so no CUDA device is required. Kernel
    numeric correctness is explicitly NOT claimed here.
    """

    def setUp(self):
        paddle.set_device("cpu")

    def _indices(self, mode):
        """Distinguishable [2, 1, 3, mode] int32 tensor via arange.

        With value(b,h,q,m) = flat index, each trailing-dim column carries a
        unique, easily hand-derivable set of numbers.
        """
        n = 2 * 1 * 3 * mode
        return paddle.arange(n, dtype="int32").reshape([2, 1, 3, mode])

    def test_mode1_only_lt_start_rest_alias(self):
        """mode=1: lt_start=col0; the other three are the SAME object.

        Input arange(6).reshape([2,1,3,1]); col0 = [[[0,1,2]],[[3,4,5]]].
        """
        x = self._indices(1)
        mode, raw = _extract_raw_ptrs(x, causal=True)

        self.assertEqual(mode, 1)
        self.assertIsInstance(raw, RawPtrs)

        expected_lt_start = np.array([[[0, 1, 2]], [[3, 4, 5]]], dtype=np.int32)
        np.testing.assert_array_equal(raw.lt_start.numpy(), expected_lt_start)
        # shape drops the trailing mode axis -> [B, HIDS, seqlen_q].
        self.assertEqual(raw.lt_start.shape, [2, 1, 3])

        # Unused boundaries are aliased to lt_start (placeholder reuse), not
        # independent copies. Identity distinguishes aliasing from an
        # accidental extra allocation or a wrong-column slice.
        self.assertIs(raw.lt_end, raw.lt_start)
        self.assertIs(raw.ut_start, raw.lt_start)
        self.assertIs(raw.ut_end, raw.lt_start)

    def test_mode2_causal_true_uses_col1_as_lt_end(self):
        """mode=2, causal=True: lt_end=col1; ut_start/ut_end alias lt_start.

        Input arange(12).reshape([2,1,3,2]);
        col0 = [[[0,2,4]],[[6,8,10]]], col1 = [[[1,3,5]],[[7,9,11]]].
        """
        x = self._indices(2)
        mode, raw = _extract_raw_ptrs(x, causal=True)

        self.assertEqual(mode, 2)

        expected_lt_start = np.array(
            [[[0, 2, 4]], [[6, 8, 10]]], dtype=np.int32
        )
        expected_lt_end = np.array([[[1, 3, 5]], [[7, 9, 11]]], dtype=np.int32)
        np.testing.assert_array_equal(raw.lt_start.numpy(), expected_lt_start)
        np.testing.assert_array_equal(raw.lt_end.numpy(), expected_lt_end)

        # lt_end is a genuinely different tensor (col1), not the lt_start alias.
        self.assertIsNot(raw.lt_end, raw.lt_start)
        # ut_* are unused for causal mode-2 and alias lt_start (col0), NOT col1.
        self.assertIs(raw.ut_start, raw.lt_start)
        self.assertIs(raw.ut_end, raw.lt_start)
        np.testing.assert_array_equal(raw.ut_end.numpy(), expected_lt_start)

    def test_mode2_causal_false_uses_col1_as_ut_end(self):
        """mode=2, causal=False: col1 goes to ut_end, not lt_end.

        This is the branch that swaps which boundary the second column feeds;
        a bug that ignored ``causal`` would put col1 into lt_end instead.
        """
        x = self._indices(2)
        mode, raw = _extract_raw_ptrs(x, causal=False)

        self.assertEqual(mode, 2)

        expected_lt_start = np.array(
            [[[0, 2, 4]], [[6, 8, 10]]], dtype=np.int32
        )
        expected_ut_end = np.array([[[1, 3, 5]], [[7, 9, 11]]], dtype=np.int32)
        np.testing.assert_array_equal(raw.lt_start.numpy(), expected_lt_start)
        np.testing.assert_array_equal(raw.ut_end.numpy(), expected_ut_end)

        # ut_end holds col1; lt_end and ut_start remain the lt_start alias.
        self.assertIsNot(raw.ut_end, raw.lt_start)
        self.assertIs(raw.lt_end, raw.lt_start)
        self.assertIs(raw.ut_start, raw.lt_start)
        np.testing.assert_array_equal(raw.lt_end.numpy(), expected_lt_start)

    def test_mode4_maps_four_distinct_columns(self):
        """mode=4: cols 0..3 -> lt_start, lt_end, ut_start, ut_end (distinct).

        Input arange(24).reshape([2,1,3,4]);
          col0=[[[0,4,8]],[[12,16,20]]]   col1=[[[1,5,9]],[[13,17,21]]]
          col2=[[[2,6,10]],[[14,18,22]]]  col3=[[[3,7,11]],[[15,19,23]]]
        """
        x = self._indices(4)
        mode, raw = _extract_raw_ptrs(x, causal=True)

        self.assertEqual(mode, 4)

        expected = {
            "lt_start": np.array([[[0, 4, 8]], [[12, 16, 20]]], dtype=np.int32),
            "lt_end": np.array([[[1, 5, 9]], [[13, 17, 21]]], dtype=np.int32),
            "ut_start": np.array(
                [[[2, 6, 10]], [[14, 18, 22]]], dtype=np.int32
            ),
            "ut_end": np.array([[[3, 7, 11]], [[15, 19, 23]]], dtype=np.int32),
        }
        np.testing.assert_array_equal(
            raw.lt_start.numpy(), expected["lt_start"]
        )
        np.testing.assert_array_equal(raw.lt_end.numpy(), expected["lt_end"])
        np.testing.assert_array_equal(
            raw.ut_start.numpy(), expected["ut_start"]
        )
        np.testing.assert_array_equal(raw.ut_end.numpy(), expected["ut_end"])

        # For mode=4 every boundary is its own column -> no aliasing at all.
        ptrs = [raw.lt_start, raw.lt_end, raw.ut_start, raw.ut_end]
        for i in range(len(ptrs)):
            for j in range(i + 1, len(ptrs)):
                self.assertIsNot(ptrs[i], ptrs[j])

    def test_mode4_is_independent_of_causal_flag(self):
        """mode=4 ignores ``causal``: column mapping is identical either way."""
        x = self._indices(4)
        _, raw_causal = _extract_raw_ptrs(x, causal=True)
        _, raw_noncausal = _extract_raw_ptrs(x, causal=False)
        for name in ("lt_start", "lt_end", "ut_start", "ut_end"):
            np.testing.assert_array_equal(
                getattr(raw_causal, name).numpy(),
                getattr(raw_noncausal, name).numpy(),
            )

    def test_unsupported_mode_raises_value_error(self):
        """Trailing dim outside {1,2,4} is rejected with ValueError.

        mode is read from ``shape[-1]``; a size-3 trailing dim must raise.
        """
        x = self._indices(3)  # trailing dim = 3 -> unsupported
        with self.assertRaises(ValueError):
            _extract_raw_ptrs(x, causal=True)


if __name__ == "__main__":
    unittest.main()
