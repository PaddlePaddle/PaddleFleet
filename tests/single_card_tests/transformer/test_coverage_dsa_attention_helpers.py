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

"""Unit tests for the module-level helpers of ``dsa_attention``.

Single card only: ``gather_from_sequence_parallel_region`` is patched.

Surface under test:
  * ``_normalize_dsa_mask`` / ``_align_dsa_indexer_mask`` layout alignment
  * ``_sparse_index_mask`` top-k scatter with invalid (-1) entries
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import paddle

from paddlefleet.transformer.dsa_attention import (
    _align_dsa_indexer_mask,
    _normalize_dsa_mask,
    _sparse_index_mask,
)

MODULE = "paddlefleet.transformer.dsa_attention"


def _fake_all_gather(tensor, *args, **kwargs):
    """Stand in for a two-rank all-gather along axis 0 on a single card."""
    return paddle.concat([tensor, tensor], axis=0)


def _true(tensor):
    """Unwrap a 0-d paddle bool tensor into a Python bool."""
    return bool(tensor)


def _equal_all(left, right):
    """Exact equality; bf16 is compared through its lossless fp32 cast.

    ``paddle.equal_all`` has no bfloat16 kernel, and widening bfloat16 to
    float32 is exact, so the comparison stays bit-for-bit.
    """
    if left.dtype in (paddle.bfloat16, paddle.float16):
        left = left.cast("float32")
        right = right.cast("float32")
    return paddle.equal_all(left, right)


class TestNormalizeAndAlignIndexerMask(unittest.TestCase):
    """Mask normalization and sequence-parallel last-dim alignment."""

    def test_normalize_squeezes_singleton_head_axis(self):
        self.assertIsNone(_normalize_dsa_mask(None))
        mask = paddle.zeros([2, 1, 4, 4], dtype="float32")
        self.assertEqual(list(_normalize_dsa_mask(mask).shape), [2, 4, 4])
        with self.assertRaises(AssertionError):
            _normalize_dsa_mask(paddle.zeros([2, 3, 4, 4], dtype="float32"))

    def test_none_mask_returns_none(self):
        self.assertIsNone(_align_dsa_indexer_mask(None, 8))

    def test_matching_last_dim_is_passed_through(self):
        mask = paddle.zeros([2, 4, 4], dtype="float32")
        self.assertIs(_align_dsa_indexer_mask(mask, 4), mask)

    def test_mismatch_without_sequence_parallel_is_rejected(self):
        mask = paddle.zeros([2, 4, 2], dtype="float32")
        with self.assertRaisesRegex(ValueError, "key length"):
            _align_dsa_indexer_mask(mask, 4)
        with self.assertRaisesRegex(ValueError, "key length"):
            _align_dsa_indexer_mask(
                mask,
                4,
                sequence_parallel=True,
                tp_group=SimpleNamespace(nranks=1),
            )
        with self.assertRaisesRegex(ValueError, "key length"):
            _align_dsa_indexer_mask(
                mask,
                4,
                sequence_parallel=False,
                tp_group=SimpleNamespace(nranks=2),
            )

    def test_unsupported_rank_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "rank 2 or 3"):
            _align_dsa_indexer_mask(paddle.zeros([2], "float32"), 4)

    def _gathered(self, mask, score_sk):
        with patch(
            MODULE + ".gather_from_sequence_parallel_region", _fake_all_gather
        ):
            return _align_dsa_indexer_mask(
                mask,
                score_sk,
                sequence_parallel=True,
                tp_group=SimpleNamespace(nranks=2),
            )

    def test_two_dim_mask_is_gathered_on_the_last_axis(self):
        mask = paddle.arange(8, dtype="float32").reshape([4, 2])

        aligned = self._gathered(mask, 4)

        self.assertEqual(list(aligned.shape), [4, 4])
        expected = paddle.concat([mask, mask], axis=-1)
        self.assertTrue(_true(_equal_all(aligned, expected)))

    def test_three_dim_mask_is_gathered_on_the_last_axis(self):
        mask = paddle.arange(12, dtype="float32").reshape([1, 6, 2])

        aligned = self._gathered(mask, 4)

        self.assertEqual(list(aligned.shape), [1, 6, 4])
        expected = paddle.concat([mask, mask], axis=-1)
        self.assertTrue(_true(_equal_all(aligned, expected)))


class TestSparseIndexMask(unittest.TestCase):
    """Top-k scatter into the sparse attention mask."""

    def _expected(self, rows, sk):
        expected = paddle.full([1, len(rows), sk], float("-inf"))
        for row, indices in enumerate(rows):
            for index in indices:
                if index >= 0:
                    expected[0, row, index] = 0.0
        return expected

    def test_invalid_entries_do_not_mask_a_valid_key_zero(self):
        # Causal rows with fewer visible keys than top-k pad with -1, and the
        # valid key 0 shares a row with those -1 entries.
        rows = [[0, -1, -1], [1, 0, -1], [2, 1, 0], [3, 2, 1]]
        topk = paddle.to_tensor([rows], dtype="int64")

        mask = _sparse_index_mask(topk, 4, 4)

        self.assertEqual(list(mask.shape), [1, 4, 4])
        self.assertTrue(_true(_equal_all(mask, self._expected(rows, 4))))

    def test_all_invalid_row_stays_fully_masked(self):
        topk = paddle.to_tensor([[[-1, -1], [0, 1]]], dtype="int64")

        mask = _sparse_index_mask(topk, 2, 3)

        self.assertTrue(
            _true(_equal_all(mask, self._expected([[-1, -1], [0, 1]], 3)))
        )


if __name__ == "__main__":
    unittest.main()
