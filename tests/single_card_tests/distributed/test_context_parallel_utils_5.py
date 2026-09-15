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

"""Behavior tests for the FlashMask index preprocessing helpers in
``paddlefleet.context_parallel_utils``.

``preprocess_index`` (contiguous strategy) and
``preprocess_index_dual_chunks`` (DualChunkSwap balanced strategy) rewrite the
FlashMask ``startend_row_indices`` so that a rank's local query chunk(s) point
at the right rows of the *gathered* key/value sequence. They are pure integer
tensor ops (clip / offset / elementwise-maximum) and run on CPU, so they can be
verified with small, position-distinguishable inputs whose expected outputs are
derived by hand from the DualChunkSwap definition -- independent of the
implementation. No collective communication is exercised here; the cross-rank
gather/scatter behavior of this module still needs a real process group and is
out of scope for a single-card test.
"""

import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src"
    ),
)

try:
    import paddle

    from paddlefleet.context_parallel_utils import (
        preprocess_index,
        preprocess_index_dual_chunks,
    )

    HAS_PADDLE = True
    IMPORT_ERROR = ""
except ImportError as exc:  # paddle / flashmask kernels not installed locally
    HAS_PADDLE = False
    IMPORT_ERROR = repr(exc)


def _int32(values):
    """Build an int32 paddle tensor from a nested python list."""
    return paddle.to_tensor(values, dtype="int32")


@unittest.skipUnless(
    HAS_PADDLE,
    "paddle (and paddlefleet.context_parallel_utils) is not importable in "
    "this environment: " + IMPORT_ERROR,
)
class TestPreprocessIndex(unittest.TestCase):
    """``preprocess_index`` = clip(indices - chunk_id*seq_blocksize, 0, max)."""

    def test_shift_and_clip_both_bounds(self):
        # chunk_id=2, seq_blocksize=4 -> subtract rows_min = 8, then clip[0, 4].
        # input:      [ 3,  8, 10, 13, 20]
        # - rows_min: [-5,  0,  2,  5, 12]
        # clip[0, 4]: [ 0,  0,  2,  4,  4]
        out = preprocess_index(
            _int32([3, 8, 10, 13, 20]),
            chunk_id=2,
            seq_blocksize=4,
            max_seqlen_q=4,
        )
        self.assertEqual(out.numpy().tolist(), [0, 0, 2, 4, 4])

    def test_chunk_zero_is_pure_clip(self):
        # chunk_id=0 -> rows_min=0, so the result is just clip[0, 6].
        # negatives floor to 0, values above the cap saturate at 6.
        out = preprocess_index(
            _int32([-4, -1, 0, 3, 6, 9]),
            chunk_id=0,
            seq_blocksize=6,
            max_seqlen_q=6,
        )
        self.assertEqual(out.numpy().tolist(), [0, 0, 0, 3, 6, 6])

    def test_two_dimensional_indices(self):
        # Batched startend_row_indices: shift is applied elementwise.
        # rows_min = 1*5 = 5, clip[0, 5].
        # row0: [ 5,  7, 12] -> [ 0,  2,  5]
        # row1: [ 4, 10,  6] -> [ 0,  5,  1]
        out = preprocess_index(
            _int32([[5, 7, 12], [4, 10, 6]]),
            chunk_id=1,
            seq_blocksize=5,
            max_seqlen_q=5,
        )
        self.assertEqual(out.numpy().tolist(), [[0, 2, 5], [0, 5, 1]])


@unittest.skipUnless(
    HAS_PADDLE,
    "paddle (and paddlefleet.context_parallel_utils) is not importable in "
    "this environment: " + IMPORT_ERROR,
)
class TestPreprocessIndexDualChunks(unittest.TestCase):
    """DualChunkSwap: a rank holds one chunk from each end of the sequence.

    first  = clip(indices - id_first *blk, 0, max)
    second = clip(indices - id_second*blk, 0, max);
             then every *non-zero* second-chunk row is pushed into the upper
             half of the reconstructed window by adding ``max_seqlen_q`` (a zero
             stays zero -- it is the "no visible row" sentinel).
    combined = maximum(first, second)
    """

    def test_dual_chunk_offset_and_combine(self):
        # id_first=0 (rows_min=0), id_second=3 (rows_min=12), blk=max=4.
        # input:              [ 2,  5, 13, 16, 20]
        # first  = clip(x,0,4)=[ 2,  4,  4,  4,  4]
        # second raw:
        #   x-12 =            [-10, -7,  1,  4,  8]
        #   clip =            [  0,  0,  1,  4,  4]
        #   +4 where !=0 =    [  0,  0,  5,  8,  8]
        # max(first, second) =[  2,  4,  5,  8,  8]
        out = preprocess_index_dual_chunks(
            _int32([2, 5, 13, 16, 20]),
            chunk_id_first=0,
            chunk_id_second=3,
            seq_blocksize=4,
            max_seqlen_q=4,
        )
        self.assertEqual(out.numpy().tolist(), [2, 4, 5, 8, 8])

    def test_offset_is_actually_applied(self):
        # Guard against the upper-half offset being dropped: if the
        # ``+max_seqlen_q`` step were removed, second-chunk rows would collapse
        # into the lower half and the combined result would differ. The
        # hand-derived "no-offset" combined is maximum([2,4,4,4,4],
        # [0,0,1,4,4]) = [2,4,4,4,4]; the real function must NOT return that.
        out = (
            preprocess_index_dual_chunks(
                _int32([2, 5, 13, 16, 20]),
                chunk_id_first=0,
                chunk_id_second=3,
                seq_blocksize=4,
                max_seqlen_q=4,
            )
            .numpy()
            .tolist()
        )
        self.assertNotEqual(out, [2, 4, 4, 4, 4])
        # Concretely, row index 13 must land in the upper half (5), not 4.
        self.assertEqual(out[2], 5)

    def test_zero_is_sentinel_not_offset(self):
        # A value landing exactly on the second chunk's start clips to 0, and 0
        # must stay 0 (it means "no visible row"), so it is never bumped to
        # max_seqlen_q. id_second=3, blk=4 -> rows_min_second=12.
        # input = [12]: first=clip(12,0,4)=4, second raw=clip(0,0,4)=0 -> 0
        # combined = max(4, 0) = 4.
        out = preprocess_index_dual_chunks(
            _int32([12]),
            chunk_id_first=0,
            chunk_id_second=3,
            seq_blocksize=4,
            max_seqlen_q=4,
        )
        self.assertEqual(out.numpy().tolist(), [4])

    def test_second_chunk_dominates_when_first_saturates_low(self):
        # Choose indices that are absent from the first chunk (clip to 0) but
        # present in the second, so the maximum is driven by the offset second
        # chunk. id_first=1 (rows_min=4), id_second=2 (rows_min=8), blk=max=4.
        # input:                 [ 4,  9, 11]
        # first  = clip(x-4,0,4)= [ 0,  4,  4]
        # second raw:
        #   x-8 =                 [-4,  1,  3]
        #   clip =                [ 0,  1,  3]
        #   +4 where !=0 =        [ 0,  5,  7]
        # max(first, second) =    [ 0,  5,  7]
        out = preprocess_index_dual_chunks(
            _int32([4, 9, 11]),
            chunk_id_first=1,
            chunk_id_second=2,
            seq_blocksize=4,
            max_seqlen_q=4,
        )
        self.assertEqual(out.numpy().tolist(), [0, 5, 7])


if __name__ == "__main__":
    unittest.main()
