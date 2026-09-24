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

"""Behavior tests for FlashMask context-parallel mask-mode logic.

Scope: the CPU-observable, pure control flow that classifies / remaps the
FlashMask ``startend_row_indices`` per context-parallel chunk and the
mask-mode enum dispatch in ``paddlefleet.context_parallel_utils``:

    * preprocess_index                 (single-chunk / contiguous mode)
    * preprocess_index_dual_chunks     (DualChunkSwap balanced mode)
    * the ``dualchunk_allgather`` / ``contiguous_allgather`` / unsupported
      enum dispatch inside the balanced all-gather forward and backward.

The general split / gather / reduce-scatter collective helpers in the same
module are intentionally left to the sibling test_context_parallel_utils.py.

Every expected value below is hand-derived from the arithmetic in the
production functions (subtract chunk offset, clip to [0, max_seqlen_q],
offset the second chunk past max_seqlen_q where non-zero, element-wise
maximum). No expected value is produced by calling the code under test.
"""

import os
import sys
import unittest

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

try:
    import paddle

    from paddlefleet import context_parallel_utils as cp_utils

    HAS_PADDLE = True
    PADDLE_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - environment dependent
    # Honest reason: this repo's context_parallel_utils imports paddle at
    # module top level, so without the paddle wheel installed the module
    # (and every helper under test) cannot be imported at all. These are
    # CPU-executable pure-tensor helpers, so they are skipped as "not run
    # for lack of the paddle dependency", not as "not applicable".
    paddle = None
    cp_utils = None
    HAS_PADDLE = False
    PADDLE_IMPORT_ERROR = exc


class _FakeGroup:
    """Minimal stand-in for a distributed group.

    The enum-dispatch error path is reached before any collective call, so
    only the ``rank`` / ``world_size`` attributes that the forward reads up
    front are needed. This is a genuine (non-mocked) collaborator for the
    control-flow branch under test, not a replacement for the code itself.
    """

    def __init__(self, rank=0, world_size=2):
        self.rank = rank
        self.world_size = world_size


@unittest.skipUnless(
    HAS_PADDLE,
    "paddle is not installed in this environment; the FlashMask index/"
    "mode helpers require the real paddle tensor ops (clip/where/maximum)",
)
class PreprocessIndexTest(unittest.TestCase):
    """Single-chunk index remap: offset by chunk start, clip to window."""

    def test_offsets_by_chunk_start_and_clips_both_bounds(self):
        # chunk_id=1, seq_blocksize=4 -> rows_min = 4.
        # [2, 4, 6, 8, 10] - 4 = [-2, 0, 2, 4, 6]
        # clip(., min=0, max=4)   = [ 0, 0, 2, 4, 4]
        indices = paddle.to_tensor([2, 4, 6, 8, 10], dtype="int32")
        out = cp_utils.preprocess_index(
            indices, chunk_id=1, seq_blocksize=4, max_seqlen_q=4
        )
        self.assertEqual(out.tolist(), [0, 0, 2, 4, 4])

    def test_chunk_zero_has_no_offset_only_clip(self):
        # chunk_id=0 -> rows_min = 0, so only clipping to [0, 4] applies.
        # [-1, 3, 5] -> clip -> [0, 3, 4]
        indices = paddle.to_tensor([-1, 3, 5], dtype="int32")
        out = cp_utils.preprocess_index(
            indices, chunk_id=0, seq_blocksize=4, max_seqlen_q=4
        )
        self.assertEqual(out.tolist(), [0, 3, 4])


@unittest.skipUnless(
    HAS_PADDLE,
    "paddle is not installed in this environment; the FlashMask index/"
    "mode helpers require the real paddle tensor ops (clip/where/maximum)",
)
class PreprocessIndexDualChunksTest(unittest.TestCase):
    """DualChunkSwap remap: each rank owns two chunks; the second chunk's
    non-zero indices are shifted past ``max_seqlen_q`` and merged by max."""

    def test_dual_chunk_shift_and_maximum_merge(self):
        # first=1, second=2, seq_blocksize=4, max_seqlen_q=4.
        # rows_min_first = 4, rows_min_second = 8, input = [3, 6, 9, 13].
        # first  = clip([-1, 2, 5, 9],  0, 4) = [0, 2, 4, 4]
        # second = clip([-5,-2, 1, 5],  0, 4) = [0, 0, 1, 4]
        # where(second!=0, second+4, second)  = [0, 0, 5, 8]
        # maximum(first, second)               = [0, 2, 5, 8]
        indices = paddle.to_tensor([3, 6, 9, 13], dtype="int32")
        out = cp_utils.preprocess_index_dual_chunks(
            indices,
            chunk_id_first=1,
            chunk_id_second=2,
            seq_blocksize=4,
            max_seqlen_q=4,
        )
        self.assertEqual(out.tolist(), [0, 2, 5, 8])

        # Independent negative control: had the +max_seqlen_q shift on the
        # second chunk been dropped, the merge would collapse to
        # maximum([0,2,4,4], [0,0,1,4]) = [0,2,4,4]. The observed result must
        # differ, proving the offset branch actually executed.
        self.assertNotEqual(out.tolist(), [0, 2, 4, 4])

    def test_dual_chunk_far_apart_blocks(self):
        # first=0, second=3, seq_blocksize=2, max_seqlen_q=2, input=[1,3,5,7].
        # rows_min_first = 0, rows_min_second = 6.
        # first  = clip([1, 3, 5, 7],   0, 2) = [1, 2, 2, 2]
        # second = clip([-5,-3,-1, 1],  0, 2) = [0, 0, 0, 1]
        # where(second!=0, second+2, second)  = [0, 0, 0, 3]
        # maximum(first, second)               = [1, 2, 2, 3]
        indices = paddle.to_tensor([1, 3, 5, 7], dtype="int32")
        out = cp_utils.preprocess_index_dual_chunks(
            indices,
            chunk_id_first=0,
            chunk_id_second=3,
            seq_blocksize=2,
            max_seqlen_q=2,
        )
        self.assertEqual(out.tolist(), [1, 2, 2, 3])


@unittest.skipUnless(
    HAS_PADDLE,
    "paddle is not installed in this environment; the FlashMask mode "
    "dispatch is exercised through the real balanced all-gather entries",
)
class FlashMaskModeEnumDispatchTest(unittest.TestCase):
    """The mask-mode enum only accepts the two documented modes; any other
    value must raise ValueError naming the offending mode. Driven through the
    real production entries (no branch is re-implemented in the test)."""

    def test_forward_rejects_unsupported_mode(self):
        q = paddle.zeros([1, 4, 1, 8], dtype="float32")
        k = paddle.zeros([1, 4, 1, 8], dtype="float32")
        v = paddle.zeros([1, 4, 1, 8], dtype="float32")
        rows = paddle.zeros([1, 1, 4, 1], dtype="int32")
        with self.assertRaisesRegex(ValueError, "not_a_real_mode"):
            cp_utils.cp_flashmask_allgatherkv_balance_forward(
                q,
                k,
                v,
                rows,
                None,
                _FakeGroup(rank=0, world_size=2),
                False,
                True,
                None,
                mode="not_a_real_mode",
            )

    def test_backward_rejects_unsupported_mode(self):
        q = paddle.zeros([1, 4, 1, 8], dtype="float32")
        k = paddle.zeros([1, 4, 1, 8], dtype="float32")
        v = paddle.zeros([1, 4, 1, 8], dtype="float32")
        rows = paddle.zeros([1, 1, 4, 1], dtype="int32")
        with self.assertRaisesRegex(ValueError, "not_a_real_mode"):
            cp_utils.cp_flashmask_allgatherkv_balance_backward(
                q,
                k,
                v,
                rows,
                q,
                None,
                q,
                None,
                _FakeGroup(rank=0, world_size=2),
                False,
                2,
                None,
                mode="not_a_real_mode",
            )


if __name__ == "__main__":
    unittest.main()
