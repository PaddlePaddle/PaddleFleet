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

"""Behavior tests for the top-p (nucleus) block-selection facet of
``paddlefleet_ops._extensions.flashmask.block_mask_utils.find_blocks_topp``.

This is the public entry point that consumes the module's bitonic argsort
(``bitonic_argsort_device`` / ``_bitonic_merge`` / ``_compare_and_swap``) and
the ``top_p_kernel``. The selection contract implemented by the kernel is:

    given a row of *unnormalized* non-negative weights, sort descending, and
    keep every element whose *strictly-larger* neighbours' running sum is below
    ``row_sum * p`` -- i.e. keep the smallest prefix (by descending value)
    whose exclusive cumulative weight is still under the cutoff. A zero row
    keeps nothing.

The kernel runs only on a GPU, so these tests run the real entry point on a
CUDA device and compare the returned boolean mask against an INDEPENDENT numpy
top-p reference (``_topp_reference``) that never calls the production code. We
assert the exact kept/dropped positions (identity, not just counts), so a
scatter/index error, a wrong cutoff, or an off-by-one in the exclusive prefix
would be caught.
"""

import unittest

import numpy as np

# paddlefleet_ops imports ``paddle`` (and Triton) at import time; the local
# environment for this file has no working ``paddle`` install. Guard the
# import so the suite skips honestly instead of erroring at collection.
# Only a genuinely missing dependency is swallowed here -- never a real bug.
try:
    import paddle
    from paddlefleet_ops._extensions.flashmask.block_mask_utils import (
        find_blocks_topp,
    )

    _IMPORT_ERROR = None
except ImportError as exc:
    paddle = None
    find_blocks_topp = None
    _IMPORT_ERROR = repr(exc)

_MODULE_AVAILABLE = _IMPORT_ERROR is None


def _gpu_available():
    """True only when a real CUDA device is present to launch the kernel."""
    if not _MODULE_AVAILABLE:
        return False
    try:
        return paddle.device.cuda.device_count() > 0
    except Exception:
        return False


def _topp_reference(row, p):
    """Independent, hand-derived top-p selection over one row.

    Mirrors the documented contract WITHOUT calling the production kernel:
    keep each element while the running sum of strictly-larger elements is
    below ``sum(row) * p``. A zero row keeps nothing. Uses numpy's own
    (descending, stable) argsort as the reference ordering.
    """
    row = np.asarray(row, dtype=np.float64)
    n = row.shape[0]
    keep = np.zeros(n, dtype=bool)
    total = float(row.sum())
    if total == 0.0:
        return keep
    cutoff = total * p
    order = np.argsort(-row, kind="stable")  # descending by value
    exclusive_prefix = 0.0
    for pos in order:
        if exclusive_prefix < cutoff:
            keep[pos] = True
        exclusive_prefix += float(row[pos])
    return keep


@unittest.skipUnless(
    _MODULE_AVAILABLE,
    f"paddle / paddlefleet_ops not importable in this environment "
    f"(no paddle installed): {_IMPORT_ERROR}",
)
class TestFindBlocksTopp(unittest.TestCase):
    """Numeric behavior of find_blocks_topp against an independent reference."""

    def setUp(self):
        if not _gpu_available():
            self.skipTest(
                "find_blocks_topp launches a Triton GPU kernel; no CUDA "
                "device is available to execute it."
            )

    def _run(self, rows, p):
        """Run find_blocks_topp on rows shaped [1, 1, len(rows), n]."""
        arr = np.asarray(rows, dtype=np.float32)
        x = paddle.to_tensor(arr.reshape(1, 1, arr.shape[0], arr.shape[1]))
        out = find_blocks_topp(x, p)
        mask = np.asarray(out.numpy()).reshape(arr.shape[0], arr.shape[1])
        return mask.astype(bool)

    def test_selects_nucleus_positions(self):
        # Integer weights -> exact sums, no float-boundary ambiguity.
        # row=[1,4,2,3], sum=10, p=0.5 -> cutoff=5.
        # desc order values [4,3,2,1] at indices [1,3,2,0];
        # exclusive prefixes [0,4,7,9] -> keep [T,T,F,F] -> positions {1,3}.
        row = [1.0, 4.0, 2.0, 3.0]
        expected = np.array([False, True, False, True])
        np.testing.assert_array_equal(expected, _topp_reference(row, 0.5))
        got = self._run([row], 0.5)[0]
        np.testing.assert_array_equal(got, expected)

    def test_small_p_keeps_only_the_single_largest(self):
        # cutoff = 0.5 < the top value's exclusive prefix (0) only for rank 0,
        # so exactly the maximum survives -- the "always keep >= 1" boundary.
        row = [1.0, 4.0, 2.0, 3.0]
        expected = np.array([False, True, False, False])
        np.testing.assert_array_equal(expected, _topp_reference(row, 0.05))
        got = self._run([row], 0.05)[0]
        np.testing.assert_array_equal(got, expected)

    def test_large_p_keeps_all_but_the_smallest(self):
        # sum=10, p=0.75 -> cutoff=7.5; exclusive prefixes [0,4,7,9]
        # -> keep top three -> drop only the smallest (index 0).
        row = [1.0, 4.0, 2.0, 3.0]
        expected = np.array([False, True, True, True])
        np.testing.assert_array_equal(expected, _topp_reference(row, 0.75))
        got = self._run([row], 0.75)[0]
        np.testing.assert_array_equal(got, expected)

    def test_zero_row_keeps_nothing(self):
        row = [0.0, 0.0, 0.0, 0.0]
        expected = np.array([False, False, False, False])
        np.testing.assert_array_equal(expected, _topp_reference(row, 0.5))
        got = self._run([row], 0.5)[0]
        np.testing.assert_array_equal(got, expected)

    def test_rows_are_selected_independently(self):
        # Two rows with different mass distributions must not cross-contaminate
        # (checks per-row stride handling). row1's tied zeros are all dropped,
        # so the tie-break order is irrelevant here.
        rows = [
            [1.0, 4.0, 2.0, 3.0],  # sum 10, p=0.5 -> {1,3}
            [3.0, 0.0, 0.0, 0.0],  # sum 3,  p=0.5 -> cutoff 1.5 -> {0}
        ]
        expected = np.array(
            [[False, True, False, True], [True, False, False, False]]
        )
        for i, r in enumerate(rows):
            np.testing.assert_array_equal(expected[i], _topp_reference(r, 0.5))
        got = self._run(rows, 0.5)
        np.testing.assert_array_equal(got, expected)

    def test_returned_mask_shape_and_dtype_are_preserved(self):
        # Content is checked elsewhere; here confirm the wrapper restores the
        # original [b, h, m, n] shape and returns a boolean mask.
        arr = np.array([[1.0, 4.0, 2.0, 3.0]], dtype=np.float32)
        x = paddle.to_tensor(arr.reshape(1, 1, 1, 4))
        out = find_blocks_topp(x, 0.5)
        self.assertEqual(list(out.shape), [1, 1, 1, 4])
        self.assertEqual(out.dtype, paddle.bool)


if __name__ == "__main__":
    unittest.main()
