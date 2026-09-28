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

"""Behavior tests for the top-p (nucleus) block selection in block_mask_utils.

Module under test:
    ``paddlefleet_ops._extensions.flashmask.block_mask_utils.find_blocks_topp``

Facet (distinct from sibling files, which cover shape/reshape, jit-callable
and structural checks): the *content and identity* of the nucleus selection.
For each row of unnormalized probabilities ``x`` and threshold ``p`` the kernel
keeps column ``j`` iff the cumulative probability of all columns strictly
"heavier" than ``j`` (i.e. the exclusive descending prefix sum) is below
``p * sum(x)``. Rows whose sum is exactly zero yield an all-False mask.

Environment: ``find_blocks_topp`` launches a Triton GPU kernel, so it is a
single-card (GPU) test. When paddle / the extension is not importable, or no
CUDA device is present, the tests skip with an honest reason -- they are NOT
counted as passed. Expected masks below are hand-derived from the algorithm,
never produced by calling the function under test.
"""

import unittest

try:
    import numpy as np
    import paddle
    from paddlefleet_ops._extensions.flashmask.block_mask_utils import (
        find_blocks_topp,
    )

    _IMPORT_OK = True
    _IMPORT_ERR = ""
except (ImportError, ModuleNotFoundError) as exc:  # honest dependency probe
    np = None
    paddle = None
    find_blocks_topp = None
    _IMPORT_OK = False
    _IMPORT_ERR = repr(exc)


def _cuda_ready():
    if not _IMPORT_OK:
        return False
    if not paddle.is_compiled_with_cuda():
        return False
    try:
        return paddle.device.cuda.device_count() > 0
    except (RuntimeError, ValueError):
        return False


_CUDA_OK = _cuda_ready()
_RUN = _IMPORT_OK and _CUDA_OK

if not _IMPORT_OK:
    _SKIP_REASON = (
        "paddlefleet_ops flashmask block_mask_utils not importable "
        f"({_IMPORT_ERR}); requires paddle + triton"
    )
elif not _CUDA_OK:
    _SKIP_REASON = (
        "find_blocks_topp launches a Triton GPU kernel; "
        "no CUDA device available in this environment"
    )
else:
    _SKIP_REASON = ""


@unittest.skipUnless(_RUN, _SKIP_REASON)
class TestFindBlocksToppNucleus(unittest.TestCase):
    """Real GPU-observed content of the top-p nucleus mask."""

    def _run(self, data_4d, p):
        x = paddle.to_tensor(data_4d, dtype="float32")
        x = x.cuda()
        out = find_blocks_topp(x, p)
        arr = out.numpy()
        # The contract promises a bool mask with the input's shape preserved.
        self.assertEqual(list(out.shape), list(np.asarray(data_4d).shape))
        self.assertEqual(arr.dtype, np.bool_)
        return arr.astype(bool)

    def test_nucleus_selection_identity_p_half(self):
        # Row = [0.1, 0.4, 0.2, 0.3], sum = 1.0, p = 0.5 -> cutoff = 0.5.
        # Descending order: 0.4(id1) 0.3(id3) 0.2(id2) 0.1(id0).
        # Exclusive prefix:  0.0     0.4     0.7     0.9.
        # keep = prefix < 0.5 -> [T, T, F, F] over sorted ids {1, 3}.
        # Scatter back to original columns -> [F, T, F, T].
        mask = self._run([[[[0.1, 0.4, 0.2, 0.3]]]], 0.5)
        expected = np.array([[[[False, True, False, True]]]], dtype=bool)
        np.testing.assert_array_equal(mask, expected)

    def test_threshold_boundary_is_exclusive_prefix(self):
        # Row = [0.5, 0.3, 0.15, 0.05], sum = 1.0, already descending by id.
        # Exclusive prefix: 0.0, 0.5, 0.8, 0.95.
        row = [[[[0.5, 0.3, 0.15, 0.05]]]]

        # p = 0.4 -> cutoff 0.4: only prefix 0.0 < 0.4 -> keep top-1 only.
        low = self._run(row, 0.4)
        np.testing.assert_array_equal(
            low, np.array([[[[True, False, False, False]]]], dtype=bool)
        )

        # p = 0.6 -> cutoff 0.6: prefixes 0.0 and 0.5 < 0.6 -> keep top-2.
        # 0.8 is NOT < 0.6, so the third column stays dropped. This pins the
        # boundary to the *exclusive* prefix (an inclusive-prefix bug would
        # instead drop the second column at p = 0.6).
        high = self._run(row, 0.6)
        np.testing.assert_array_equal(
            high, np.array([[[[True, True, False, False]]]], dtype=bool)
        )

    def test_zero_row_all_false_and_per_row_independence(self):
        # Batch of two rows sharing p = 0.5.
        #   row0 = [0, 0, 0, 0]      -> sum == 0 -> all-False branch.
        #   row1 = [0.1, 0.4, 0.2, 0.3] -> [F, T, F, T] (see first test).
        # Proves the zero-sum short circuit fires per row and does not leak
        # into / corrupt the neighbouring row's selection.
        data = [[[[0.0, 0.0, 0.0, 0.0], [0.1, 0.4, 0.2, 0.3]]]]
        mask = self._run(data, 0.5)
        expected = np.array(
            [[[[False, False, False, False], [False, True, False, True]]]],
            dtype=bool,
        )
        np.testing.assert_array_equal(mask, expected)

    def test_padding_columns_excluded_with_non_power_of_two_n(self):
        # n = 6 forces BLOCK_SIZE = next_power_of_2(6) = 8, so columns 6 and 7
        # are padding and must never be reported as kept.
        # Row = [0.30, 0.05, 0.20, 0.12, 0.25, 0.08], sum = 1.00, p = 0.8.
        # Descending: 0.30(0) 0.25(4) 0.20(2) 0.12(3) 0.08(5) 0.05(1).
        # Exclusive prefix: 0.0 0.30 0.55 0.75 0.87 0.95.
        # keep = prefix < 0.8 -> first four sorted ids {0, 4, 2, 3}.
        # Scatter back -> [T, F, T, T, T, F].
        mask = self._run([[[[0.30, 0.05, 0.20, 0.12, 0.25, 0.08]]]], 0.8)
        expected = np.array(
            [[[[True, False, True, True, True, False]]]], dtype=bool
        )
        np.testing.assert_array_equal(mask, expected)


if __name__ == "__main__":
    unittest.main()
