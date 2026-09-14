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

"""Unit tests for the prefix-LM three-region mask representations."""

import unittest

import paddle

from paddlefleet.transformer.prefix_lm_mask import (
    build_dense_mask,
    build_flashmask_row_indices,
    build_prefix_lm_layout,
    expand_layout_to_dense,
    expand_row_indices_to_dense,
    prefix_lm_pad_len,
    prefix_lm_segment_starts,
)


class TestCheckLens(unittest.TestCase):
    def test_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            build_dense_mask([1, 2], [1])

    def test_empty(self):
        with self.assertRaises(ValueError):
            build_dense_mask([], [])

    def test_negative_prefix(self):
        with self.assertRaises(ValueError):
            build_dense_mask([-1], [2])

    def test_non_positive_suffix(self):
        with self.assertRaises(ValueError):
            build_dense_mask([2], [0])

    def test_negative_pad(self):
        with self.assertRaises(ValueError):
            build_dense_mask([2], [2], pad_len=-1)


class TestSegmentStarts(unittest.TestCase):
    def test_prefix_sum(self):
        self.assertEqual(prefix_lm_segment_starts([2, 3], [1, 1]), [0, 3])


class TestPadLen(unittest.TestCase):
    def test_sp_disabled_uses_align(self):
        # base = 1, align = 128 -> round up to 128
        self.assertEqual(
            prefix_lm_pad_len(100, tp_size=4, sequence_parallel=False), 28
        )

    def test_sp_enabled_uses_lcm(self):
        # base = tp_size = 4, lcm(4, 128) = 128
        self.assertEqual(
            prefix_lm_pad_len(100, tp_size=4, sequence_parallel=True), 28
        )

    def test_already_aligned(self):
        self.assertEqual(
            prefix_lm_pad_len(128, tp_size=1, sequence_parallel=False), 0
        )


class TestBuildDenseMask(unittest.TestCase):
    def test_shape_no_pad(self):
        m = build_dense_mask([2], [2])
        self.assertEqual(m.shape, [1, 1, 4, 4])
        self.assertEqual(m.dtype, paddle.bool)

    def test_zero_prefix_segment(self):
        # c == 0 skips the prefix-open branch but still opens causal suffix.
        m = build_dense_mask([0], [3])
        self.assertEqual(m.shape, [1, 1, 3, 3])

    def test_with_pad_self_loop(self):
        m = build_dense_mask([2], [2], pad_len=2)
        self.assertEqual(m.shape, [1, 1, 6, 6])
        # pad rows only see themselves (diagonal is False = allowed).
        self.assertFalse(bool(m[0, 0, 5, 5]))
        self.assertTrue(bool(m[0, 0, 5, 0]))


class TestFlashMaskRowIndices(unittest.TestCase):
    def test_shape_no_pad(self):
        ri = build_flashmask_row_indices([2], [2])
        self.assertEqual(ri.shape, [1, 1, 4, 4])

    def test_shape_with_pad(self):
        ri = build_flashmask_row_indices([2], [2], pad_len=2)
        self.assertEqual(ri.shape, [1, 1, 6, 4])


class TestLayout(unittest.TestCase):
    def test_fields(self):
        layout = build_prefix_lm_layout([2, 1], [2, 1], pad_len=2)
        self.assertEqual(layout["segment_starts"], (0, 4))
        self.assertEqual(layout["n_contexts"], (2, 1))
        self.assertEqual(layout["n_queries"], (2, 1))
        self.assertEqual(layout["pad_len"], 2)


class TestExpandRowIndices(unittest.TestCase):
    def test_bad_shape_raises(self):
        with self.assertRaises(ValueError):
            expand_row_indices_to_dense(
                paddle.zeros([1, 1, 4, 3], dtype="int32")
            )

    def test_expand_shape(self):
        ri = build_flashmask_row_indices([2], [2])
        dense = expand_row_indices_to_dense(ri)
        self.assertEqual(dense.shape, [1, 1, 4, 4])


class TestExpandLayout(unittest.TestCase):
    def test_no_pad(self):
        layout = build_prefix_lm_layout([2], [2])
        dense = expand_layout_to_dense(layout)
        self.assertEqual(dense.shape, [1, 1, 4, 4])

    def test_with_pad(self):
        layout = build_prefix_lm_layout([2], [2], pad_len=2)
        dense = expand_layout_to_dense(layout)
        self.assertEqual(dense.shape, [1, 1, 6, 6])


class TestThreeRepresentationsAgree(unittest.TestCase):
    """The three representations must expand to a bit-identical dense mask."""

    def _check(self, prefix_lens, suffix_lens, pad_len):
        dense = build_dense_mask(prefix_lens, suffix_lens, pad_len)
        via_ri = expand_row_indices_to_dense(
            build_flashmask_row_indices(prefix_lens, suffix_lens, pad_len)
        )
        via_layout = expand_layout_to_dense(
            build_prefix_lm_layout(prefix_lens, suffix_lens, pad_len)
        )
        self.assertTrue(bool((dense == via_ri).all()))
        self.assertTrue(bool((dense == via_layout).all()))

    def test_single_segment_no_pad(self):
        self._check([3], [2], 0)

    def test_multi_segment_with_pad(self):
        self._check([2, 3], [2, 1], 4)

    def test_zero_prefix(self):
        self._check([0, 2], [3, 2], 2)


class TestExplicitPlace(unittest.TestCase):
    """Exercise the optional ``place`` argument on the tensor builders."""

    def setUp(self):
        # Use whatever place the default tensor lives on, so the test is
        # agnostic to CPU/GPU execution.
        self.place = paddle.zeros([1]).place

    def test_dense_mask_with_place(self):
        m = build_dense_mask([2], [2], pad_len=2, place=self.place)
        self.assertEqual(m.shape, [1, 1, 6, 6])

    def test_row_indices_with_place(self):
        ri = build_flashmask_row_indices([2], [2], pad_len=2, place=self.place)
        self.assertEqual(ri.shape, [1, 1, 6, 4])


if __name__ == "__main__":
    unittest.main()
