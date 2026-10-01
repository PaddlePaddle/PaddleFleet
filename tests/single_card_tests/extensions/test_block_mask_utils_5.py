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

"""Behavior tests for the top-p block-selection scan in block_mask_utils.

Module under test:
    packages/paddlefleet_ops/src/paddlefleet_ops/_extensions/flashmask/
    block_mask_utils.py :: find_blocks_topp

Facet: the per-row nucleus/top-p selection. find_blocks_topp reshapes the
input to rows, and for each row the triton kernel (1) sums the unnormalized
row, (2) sorts descending, (3) runs a cumulative-sum scan, and keeps every
element whose *strictly preceding* prefix sum is below ``row_sum * p``
(condition ``(cumsum - self) < cutoff``), then scatters the kept mask back to
the original column order. A zero-sum row short-circuits to an all-False row.

Environment (see unit-test-rules.md, "计算优化" / 单卡): find_blocks_topp
dispatches a triton GPU kernel, so the numeric assertions require a real CUDA
device (Fleet single-card). Expected masks below are hand-derived from the
selection rule above, never by calling find_blocks_topp to compute its own
reference. The local environment has no paddle/triton, so the suite skips with
an honest reason instead of a fake pass.
"""

import unittest

try:
    import numpy as np
    import paddle
    from paddlefleet_ops._extensions.flashmask.block_mask_utils import (
        find_blocks_topp,
    )

    _IMPORT_OK = True
    _IMPORT_REASON = ""
except ImportError as exc:  # honest: dependency/import not available locally
    _IMPORT_OK = False
    _IMPORT_REASON = f"paddle/paddlefleet_ops import failed: {exc}"


def _cuda_available():
    return _IMPORT_OK and paddle.is_compiled_with_cuda()


@unittest.skipUnless(_IMPORT_OK, _IMPORT_REASON)
class TestFindBlocksTopp(unittest.TestCase):
    """Numeric behavior of find_blocks_topp (top-p nucleus block selection)."""

    @unittest.skipUnless(
        _cuda_available(),
        "find_blocks_topp runs a triton GPU kernel; needs CUDA",
    )
    def test_topp_selects_hand_derived_nucleus(self):
        # Two rows, distinct values (no sort ties), n=4 == power of 2 so no
        # padding branch is exercised here; p=0.6.
        #
        # row0 = [1, 3, 2, 4], sum=10, cutoff = 10*0.6 = 6.0
        #   sorted desc  : [4, 3, 2, 1]  (orig ids [3, 1, 2, 0])
        #   prefix-before: [0, 4, 7, 9]
        #   keep(<6)     : [T, T, F, F]  -> keep orig ids {3, 1}
        #   mask (orig)  : [F, T, F, T]
        # row1 = [8, 1, 0.6, 0.4], sum=10, cutoff = 6.0
        #   sorted desc  : [8, 1, 0.6, 0.4] (orig ids [0, 1, 2, 3])
        #   prefix-before: [0, 8, 9, 9.6]
        #   keep(<6)     : [T, F, F, F]  -> keep orig id {0}
        #   mask (orig)  : [T, F, F, F]
        x = paddle.to_tensor(
            [[[[1.0, 3.0, 2.0, 4.0], [8.0, 1.0, 0.6, 0.4]]]],
            dtype="float32",
        )
        out = find_blocks_topp(x, 0.6)

        # Shape/layout must survive the internal reshape round-trip.
        self.assertEqual(list(out.shape), [1, 1, 2, 4])
        self.assertEqual(out.dtype, paddle.bool)

        expected = np.array(
            [[[[False, True, False, True], [True, False, False, False]]]],
            dtype=bool,
        )
        np.testing.assert_array_equal(out.numpy(), expected)

    @unittest.skipUnless(
        _cuda_available(),
        "find_blocks_topp runs a triton GPU kernel; needs CUDA",
    )
    def test_topp_threshold_shifts_selection(self):
        # Same row, larger p must keep a strict superset (monotonic nucleus).
        # row = [1, 3, 2, 4], sum=10.
        #   p=0.45 -> cutoff 4.5: prefix-before [0,4,7,9] keep(<4.5)=[T,T,F,F]
        #             -> ids {3,1} -> mask [F, T, F, T]
        #   p=0.80 -> cutoff 8.0: keep(<8)=[T,T,T,F]
        #             -> ids {3,1,2} -> mask [F, T, T, T]
        x = paddle.to_tensor([[[[1.0, 3.0, 2.0, 4.0]]]], dtype="float32")

        out_small = find_blocks_topp(x, 0.45).numpy()
        out_large = find_blocks_topp(x, 0.80).numpy()

        np.testing.assert_array_equal(
            out_small, np.array([[[[False, True, False, True]]]], dtype=bool)
        )
        np.testing.assert_array_equal(
            out_large, np.array([[[[False, True, True, True]]]], dtype=bool)
        )
        # The larger threshold must not drop any block the smaller one kept.
        self.assertTrue(bool(np.all(out_large >= out_small)))

    @unittest.skipUnless(
        _cuda_available(),
        "find_blocks_topp runs a triton GPU kernel; needs CUDA",
    )
    def test_zero_sum_row_is_all_false(self):
        # row_sum == 0 short-circuits the kernel to an all-zero (all-False) row.
        x = paddle.zeros([1, 1, 1, 4], dtype="float32")
        out = find_blocks_topp(x, 0.9)

        self.assertEqual(out.dtype, paddle.bool)
        np.testing.assert_array_equal(
            out.numpy(), np.zeros([1, 1, 1, 4], dtype=bool)
        )


if __name__ == "__main__":
    unittest.main()
