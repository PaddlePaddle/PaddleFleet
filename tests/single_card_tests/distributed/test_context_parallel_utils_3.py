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

"""Behavior tests for the index-preprocessing helpers in
``paddlefleet.context_parallel_utils``.

Covered production functions:
  * ``preprocess_index`` -- shift a single chunk's start/end row indices by
    ``chunk_id * seq_blocksize`` and clip into ``[0, max_seqlen_q]``.
  * ``preprocess_index_dual_chunks`` -- the DualChunkSwap variant that shifts
    the same indices by two different chunk origins, offsets the (non-zero)
    second chunk by ``max_seqlen_q`` so it occupies a disjoint coordinate
    region, and merges the two via element-wise maximum.

These are pure tensor arithmetic (subtraction / clip / where / maximum) and run
on CPU. Every expected value below is derived by hand from the documented
formula, independently of the implementation -- no call to the function under
test is used to build the expected output. Distinguishable, non-degenerate
inputs are chosen so a wrong shift direction, a missing clip bound, a dropped
second-chunk offset, or a swapped merge would produce a different tensor and
fail the assertion.

Paddle is an optional/heavy dependency; when it is unavailable the whole module
skips with an honest reason rather than silently passing.
"""

import os
import sys
import unittest

# The repository uses a ``src`` layout (see pyproject ``where = ["src"]``); make
# ``paddlefleet`` importable when running the file directly from the tree.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    import paddle

    from paddlefleet.context_parallel_utils import (
        preprocess_index,
        preprocess_index_dual_chunks,
    )

    # These helpers only need CPU tensor ops; pin the device so the file runs
    # in a no-accelerator environment.
    paddle.set_device("cpu")
    _PADDLE_AVAILABLE = True
    _SKIP_REASON = ""
except ImportError as exc:  # pragma: no cover - exercised only without paddle
    _PADDLE_AVAILABLE = False
    _SKIP_REASON = f"paddle / paddlefleet not importable: {exc}"


def _int32(rows):
    return paddle.to_tensor(rows, dtype="int32")


@unittest.skipUnless(_PADDLE_AVAILABLE, _SKIP_REASON)
class TestPreprocessIndex(unittest.TestCase):
    """``preprocess_index`` == clip(indices - chunk_id*blk, 0, max)."""

    def test_shift_and_upper_clip(self):
        # chunk_id=1, blk=8 -> subtract 8; max=16 clips the upper end.
        # [[10,20],[30,40]] - 8 = [[2,12],[22,32]] -> clip[0,16] = [[2,12],[16,16]]
        result = preprocess_index(
            _int32([[10, 20], [30, 40]]),
            chunk_id=1,
            seq_blocksize=8,
            max_seqlen_q=16,
        )
        self.assertEqual(result.tolist(), [[2, 12], [16, 16]])

    def test_chunk_zero_is_pure_clip(self):
        # chunk_id=0 -> no shift; values already inside [0,16] pass through,
        # including the exact boundary 0 and 16.
        result = preprocess_index(
            _int32([[5, 10], [0, 16]]),
            chunk_id=0,
            seq_blocksize=8,
            max_seqlen_q=16,
        )
        self.assertEqual(result.tolist(), [[5, 10], [0, 16]])

    def test_negative_clamped_to_zero(self):
        # chunk_id=1, blk=8 -> subtract 8.
        # [[3,5],[9,25]] - 8 = [[-5,-3],[1,17]] -> clip[0,16] = [[0,0],[1,16]]
        result = preprocess_index(
            _int32([[3, 5], [9, 25]]),
            chunk_id=1,
            seq_blocksize=8,
            max_seqlen_q=16,
        )
        self.assertEqual(result.tolist(), [[0, 0], [1, 16]])


@unittest.skipUnless(_PADDLE_AVAILABLE, _SKIP_REASON)
class TestPreprocessIndexDualChunks(unittest.TestCase):
    """``preprocess_index_dual_chunks``:

    first  = clip(indices - cf*blk, 0, max)
    second = clip(indices - cs*blk, 0, max)
    second = where(second != 0, second + max, second)
    out    = maximum(first, second)
    """

    def test_second_chunk_offset_dominates(self):
        # cf=1, cs=2, blk=8, max=16 on [[20,30]].
        # first  = clip([12,22],0,16) = [12,16]
        # second = clip([ 4,14],0,16) = [ 4,14]; non-zero -> +16 = [20,30]
        # out    = max([12,16],[20,30]) = [20,30]
        result = preprocess_index_dual_chunks(
            _int32([[20, 30]]),
            chunk_id_first=1,
            chunk_id_second=2,
            seq_blocksize=8,
            max_seqlen_q=16,
        )
        self.assertEqual(result.tolist(), [[20, 30]])

    def test_zero_entry_keeps_zero_nonzero_gets_offset(self):
        # cf=0, cs=1, blk=8, max=16 on [[5,20]].
        # first  = clip([5,20],0,16) = [5,16]
        # second = clip([-3,12],0,16) = [0,12]
        #          -> where !=0: [0, 12+16=28]  (the 0 stays 0, no offset)
        # out    = max([5,16],[0,28]) = [5,28]
        result = preprocess_index_dual_chunks(
            _int32([[5, 20]]),
            chunk_id_first=0,
            chunk_id_second=1,
            seq_blocksize=8,
            max_seqlen_q=16,
        )
        self.assertEqual(result.tolist(), [[5, 28]])

    def test_equal_chunk_ids_still_offset_second(self):
        # Even with cf == cs == 0 the second (non-zero) copy is pushed into the
        # [max, 2*max] region, so the output is NOT equal to the input.
        # first  = clip([10,20],0,16) = [10,16]
        # second = clip([10,20],0,16) = [10,16]; non-zero -> +16 = [26,32]
        # out    = max([10,16],[26,32]) = [26,32]
        result = preprocess_index_dual_chunks(
            _int32([[10, 20]]),
            chunk_id_first=0,
            chunk_id_second=0,
            seq_blocksize=8,
            max_seqlen_q=16,
        )
        self.assertEqual(result.tolist(), [[26, 32]])


if __name__ == "__main__":
    unittest.main()
