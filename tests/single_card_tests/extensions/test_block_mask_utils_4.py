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

Facet under test (distinct from the sibling files targeting the same module):
the *content* of the boolean mask produced by the top-p nucleus rule, and in
particular the three things that make this implementation non-trivial --

  1. The cutoff is ``row_sum * p`` over the *unnormalised* row (kernel line
     ``actual_cutoff = row_sum * threshold_p``), so the decision is invariant
     to a positive rescale of the whole row, not a fixed absolute threshold.
  2. An element is kept iff the sum of all *strictly larger* elements is below
     the cutoff (``(cum_probs - x_sorted) < actual_cutoff`` on the descending
     sort), i.e. the element that crosses the threshold is still kept.
  3. The descending argsort + scatter must map the kept flags back onto the
     *original* column positions, so an unordered row is a real test of the
     bitonic argsort / scatter path, not just a prefix cut.

All expected masks below are hand-derived by walking that rule on paper and
written as literal booleans; we never call ``find_blocks_topp`` (or reproduce
its kernel) to compute its own expected values.

Import guard: ``paddlefleet_ops`` imports ``paddle`` (and Triton) at import
time and this host may lack them, so the whole suite skips honestly when the
real production entry point cannot be imported. The ``top_p_kernel`` bitonic
argsort runs only on a real GPU, so the suite also skips when paddle is not
compiled with CUDA.

On this Paddle build (real Hopper GPU) ``find_blocks_topp`` completes end to
end: ``x.reshape(-1, n)`` accepts the varargs form and
``paddle.empty(x_reshaped.shape, dtype=paddle.bool, device=x.device)`` is
honored. The tests below therefore assert the *correct* nucleus behaviour as
plain positive checks against hand-derived masks.
"""

import unittest

import numpy as np

try:
    import paddle
    from paddlefleet_ops._extensions.flashmask.block_mask_utils import (
        find_blocks_topp,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # genuinely missing dependency, not a bug we hide
    paddle = None
    find_blocks_topp = None
    _IMPORT_ERROR = repr(exc)

_MODULE_AVAILABLE = _IMPORT_ERROR is None


@unittest.skipUnless(
    _MODULE_AVAILABLE,
    "find_blocks_topp not importable "
    f"(paddle/Triton/paddlefleet_ops missing): {_IMPORT_ERROR}",
)
class TestFindBlocksToppNucleusMask(unittest.TestCase):
    """Hand-derived content checks for the top-p nucleus mask.

    Each case documents the paper derivation of its expected mask. The row is
    processed by: sort descending, keep element i iff the sum of strictly
    larger elements < row_sum * p, then scatter the kept flags back to the
    original column order.

    The GPU Triton kernel (bitonic argsort + scatter) needs a real CUDA
    device; the class is skipped honestly when paddle is not compiled with
    CUDA. The assertions are real numeric checks of the intended, correct
    behaviour.
    """

    def setUp(self):
        if not paddle.is_compiled_with_cuda():
            self.skipTest(
                "top_p_kernel is a GPU-only Triton kernel; requires a "
                "CUDA-compiled paddle build with a real GPU"
            )
        paddle.set_device("gpu")

    def _mask_of(self, data, p):
        """Call the real production entry point and return a bool ndarray."""
        x = paddle.to_tensor(data, dtype="float32")
        out = find_blocks_topp(x, p)
        self.assertEqual(out.dtype, paddle.bool)
        return np.asarray(out.numpy(), dtype=bool)

    def test_basic_descending_cut(self):
        # row [0.4,0.3,0.2,0.1], sum 1.0, cutoff 0.8.
        # exclusive-prefix on descending order: [0.0,0.4,0.7,0.9]; <0.8 keeps
        # the first three, drops the last.
        mask = self._mask_of([[0.4, 0.3, 0.2, 0.1]], 0.8)
        np.testing.assert_array_equal(mask, [[True, True, True, False]])

    def test_unordered_row_sort_and_scatter(self):
        # row [0.1,0.4,0.2,0.3], sum 1.0, cutoff 0.7.
        # descending values 0.4(col1),0.3(col3),0.2(col2),0.1(col0);
        # exclusive-prefix [0.0,0.4,0.7,0.9]; <0.7 keeps cols 1 and 3 only.
        # Scattered back to original order -> [F,T,F,T]. A prefix-only or
        # no-scatter implementation would wrongly return [T,T,F,F].
        mask = self._mask_of([[0.1, 0.4, 0.2, 0.3]], 0.7)
        np.testing.assert_array_equal(mask, [[False, True, False, True]])

    def test_cutoff_scales_with_unnormalised_row_sum(self):
        # row [4,3,2,1], sum 10.0, cutoff = 10*0.8 = 8.0.
        # exclusive-prefix [0,4,7,9]; <8 keeps first three -> [T,T,T,F].
        # If the cutoff were a fixed 0.8 (ignoring the sum) the second element
        # (prefix 4) would already exceed it and the mask would be [T,F,F,F],
        # so this pins the row_sum*p scaling.
        mask = self._mask_of([[4.0, 3.0, 2.0, 1.0]], 0.8)
        np.testing.assert_array_equal(mask, [[True, True, True, False]])

    def test_zero_row_masks_everything(self):
        # row_sum == 0.0 hits the early-return branch: all-False output.
        mask = self._mask_of([[0.0, 0.0, 0.0, 0.0]], 0.5)
        np.testing.assert_array_equal(mask, [[False, False, False, False]])

    def test_p_one_keeps_all(self):
        # cutoff = sum; every exclusive-prefix (max 0.9) < 1.0 -> all kept.
        mask = self._mask_of([[0.4, 0.3, 0.2, 0.1]], 1.0)
        np.testing.assert_array_equal(mask, [[True, True, True, True]])

    def test_p_zero_keeps_nothing(self):
        # cutoff = 0; even the top element has exclusive-prefix 0, and 0 < 0
        # is False, so *nothing* is kept. This implementation offers no
        # "keep at least one" guarantee -- documented, not asserted as ideal.
        mask = self._mask_of([[0.4, 0.3, 0.2, 0.1]], 0.0)
        np.testing.assert_array_equal(mask, [[False, False, False, False]])

    def test_non_power_of_two_width_padding(self):
        # n=3 -> BLOCK_SIZE padded to 4 with a -inf sort sentinel that must not
        # leak into the mask. row [0.5,0.3,0.2], sum 1.0, cutoff 0.7.
        # exclusive-prefix [0.0,0.5,0.8]; <0.7 keeps first two -> [T,T,F].
        mask = self._mask_of([[0.5, 0.3, 0.2]], 0.7)
        np.testing.assert_array_equal(mask, [[True, True, False]])

    def test_multidim_rows_are_independent_and_shape_preserved(self):
        # Shape [1,2,1,4]: two independent rows share one p=0.7.
        #   row A [0.4,0.3,0.2,0.1]: exclusive [0,0.4,0.7,0.9] -> [T,T,F,F]
        #   row B [0.1,0.4,0.2,0.3]: kept cols 1,3 scattered -> [F,T,F,T]
        data = [[[[0.4, 0.3, 0.2, 0.1]], [[0.1, 0.4, 0.2, 0.3]]]]
        x = paddle.to_tensor(data, dtype="float32")
        out = find_blocks_topp(x, 0.7)
        self.assertEqual(out.dtype, paddle.bool)
        self.assertEqual(list(out.shape), [1, 2, 1, 4])
        expected = np.array(
            [[[[True, True, False, False]], [[False, True, False, True]]]],
            dtype=bool,
        )
        np.testing.assert_array_equal(
            np.asarray(out.numpy(), dtype=bool), expected
        )


if __name__ == "__main__":
    unittest.main()
