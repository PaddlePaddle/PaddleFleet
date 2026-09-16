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

"""Behaviour tests for ``paddlefleet.transformer.utils``.

Every expected value here is derived by hand or with an independent numpy
construction that does not call the function under test:

* Causal / sliding-window masks are compared against literal boolean arrays
  worked out from the attend/no-attend semantics (True == masked), including a
  case where ``skv != sq`` so the key-offset logic is exercised, and negative
  controls that a real sliding window differs from a plain causal mask.
* ``attention_mask_func`` is checked for the exact filled values *and* that it
  mutates in place and returns the same object (it uses ``masked_fill_``).
* ``is_layer_window_attention`` is driven through every documented branch with
  distinguishable ``layer_number`` values so an off-by-one or a swapped branch
  is caught; the invalid-type contract is pinned with ``assertRaises``.
* ``get_doc_lens`` / ``get_doc_starts`` use packed-document fixtures whose
  lengths and starts are computed independently.

The module is CPU-only and needs Paddle. Imports are guarded so that a missing
dependency yields an honest skip carrying the error repr, never a fake pass.
"""

from __future__ import annotations

import unittest

import numpy as np

try:
    import paddle

    from paddlefleet.transformer import utils as tutils

    _IMPORT_ERROR: Exception | None = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    paddle = None
    tutils = None
    _IMPORT_ERROR = exc

_HAS_PADDLE = _IMPORT_ERROR is None
_SKIP_REASON = (
    f"paddle / paddlefleet.transformer.utils import failed: {_IMPORT_ERROR!r}"
)


@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON)
class TestCausalMask(unittest.TestCase):
    """get_default_causal_mask returns the strict upper-triangular mask."""

    def test_causal_mask_values(self):
        out = tutils.get_default_causal_mask(4)
        # True marks masked (future) positions: strictly above the diagonal.
        expected = np.triu(np.ones((4, 4), dtype=bool), k=1)
        self.assertEqual(out.dtype, paddle.bool)
        np.testing.assert_array_equal(out.numpy(), expected)
        # Diagonal (self) and past must be attendable (False).
        self.assertFalse(bool(out.numpy()[3, 0]))
        self.assertFalse(bool(out.numpy()[2, 2]))


@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON)
class TestSlidingWindowLeftSize(unittest.TestCase):
    """get_sliding_window_left_size picks the left window for both forms."""

    def test_int_form_returns_itself(self):
        self.assertEqual(tutils.get_sliding_window_left_size(5), 5)

    def test_tuple_form_returns_left_component(self):
        # (left, right): only the left element is the past window.
        self.assertEqual(tutils.get_sliding_window_left_size((3, 2)), 3)

    def test_negative_left_is_preserved(self):
        # -1 denotes an infinite past window; the caller interprets it.
        self.assertEqual(tutils.get_sliding_window_left_size((-1, 0)), -1)


@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON)
class TestSlidingWindowCausalMask(unittest.TestCase):
    """get_sliding_window_causal_mask: True == masked."""

    def test_int_causal_window(self):
        # left=2, right=0, sq=skv=4. Query i may attend keys [i-2 .. i].
        out = tutils.get_sliding_window_causal_mask(4, 4, 2)
        expected = np.array(
            [
                [False, True, True, True],
                [False, False, True, True],
                [False, False, False, True],
                [True, False, False, False],
            ]
        )
        np.testing.assert_array_equal(out.numpy(), expected)
        # A real window truncates: differs from a plain causal mask at [3, 0].
        causal = np.triu(np.ones((4, 4), dtype=bool), k=1)
        self.assertTrue(bool(out.numpy()[3, 0]))
        self.assertFalse(bool(causal[3, 0]))

    def test_tuple_two_sided_window(self):
        # (left=1, right=1): banded, so a future key can be visible.
        out = tutils.get_sliding_window_causal_mask(4, 4, (1, 1))
        expected = np.array(
            [
                [False, False, True, True],
                [False, False, False, True],
                [True, False, False, False],
                [True, True, False, False],
            ]
        )
        np.testing.assert_array_equal(out.numpy(), expected)
        # right=1 lets query 2 attend future key 3 (a causal mask would not).
        self.assertFalse(bool(out.numpy()[2, 3]))

    def test_infinite_window_is_plain_causal(self):
        # left=-1 => no left truncation => strict causal mask.
        out = tutils.get_sliding_window_causal_mask(4, 4, (-1, 0))
        expected = np.triu(np.ones((4, 4), dtype=bool), k=1)
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_offset_when_skv_gt_sq(self):
        # sq=2, skv=4, left=1, right=0. Diagonal offset is skv-sq=2.
        out = tutils.get_sliding_window_causal_mask(2, 4, 1)
        expected = np.array(
            [
                [True, False, False, True],
                [True, True, False, False],
            ]
        )
        np.testing.assert_array_equal(out.numpy(), expected)


@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON)
class TestAttentionMaskFunc(unittest.TestCase):
    """attention_mask_func fills masked entries in place with -10000.0."""

    def test_fills_and_mutates_in_place(self):
        scores = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
        mask = paddle.to_tensor([[False, True], [True, False]])
        result = tutils.attention_mask_func(scores, mask)
        expected = np.array([[1.0, -10000.0], [-10000.0, 4.0]], dtype="float32")
        np.testing.assert_array_equal(result.numpy(), expected)
        # masked_fill_ is in place: same object, and the source is mutated.
        self.assertIs(result, scores)
        np.testing.assert_array_equal(scores.numpy(), expected)


@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON)
class TestIsLayerWindowAttention(unittest.TestCase):
    """is_layer_window_attention branch behaviour."""

    def test_no_sliding_window_disables(self):
        self.assertFalse(tutils.is_layer_window_attention(None, 2, 0))
        self.assertFalse(tutils.is_layer_window_attention(0, None, 3))

    def test_none_freq_enables_every_layer(self):
        self.assertTrue(tutils.is_layer_window_attention(8, None, 0))
        self.assertTrue(tutils.is_layer_window_attention(8, None, 7))

    def test_int_freq_skips_multiples(self):
        # layer % freq != 0 -> window attention on.
        self.assertFalse(tutils.is_layer_window_attention(8, 2, 0))
        self.assertTrue(tutils.is_layer_window_attention(8, 2, 1))
        self.assertFalse(tutils.is_layer_window_attention(8, 2, 2))
        self.assertTrue(tutils.is_layer_window_attention(8, 2, 3))

    def test_list_freq_indexes_per_layer(self):
        freq = [0, 1, 0, 1]
        self.assertFalse(tutils.is_layer_window_attention(8, freq, 0))
        self.assertTrue(tutils.is_layer_window_attention(8, freq, 1))
        self.assertFalse(tutils.is_layer_window_attention(8, freq, 2))
        self.assertTrue(tutils.is_layer_window_attention(8, freq, 3))

    def test_invalid_freq_type_raises(self):
        with self.assertRaises(ValueError):
            tutils.is_layer_window_attention(8, 1.5, 0)


@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON)
class TestStartendAddSlidingWindow(unittest.TestCase):
    """Guard branches of startend_row_indices_add_sliding_window."""

    def _indices(self):
        return paddle.zeros([1, 2, 4, 1], dtype="int32")

    def test_none_window_returns_input_unchanged(self):
        x = self._indices()
        out = tutils.startend_row_indices_add_sliding_window(x, None, 0.5, 4)
        self.assertIs(out, x)

    def test_zero_window_returns_input_unchanged(self):
        x = self._indices()
        out = tutils.startend_row_indices_add_sliding_window(x, 0, 0.5, 4)
        self.assertIs(out, x)

    def test_negative_int_window_returns_input_unchanged(self):
        # window_size <= 0 (infinite/none) -> no truncation applied.
        x = self._indices()
        out = tutils.startend_row_indices_add_sliding_window(x, -1, 0.5, 4)
        self.assertIs(out, x)

    def test_negative_tuple_window_returns_input_unchanged(self):
        x = self._indices()
        out = tutils.startend_row_indices_add_sliding_window(x, (-1, 0), 0.5, 4)
        self.assertIs(out, x)

    def test_bad_num_vec_raises(self):
        # last dim must be 1 (LTS) or 2 (LTS & UTE).
        x = paddle.zeros([1, 2, 4, 3], dtype="int32")
        with self.assertRaises(ValueError):
            tutils.startend_row_indices_add_sliding_window(x, 2, 0.5, 4)


@unittest.skipUnless(_HAS_PADDLE, _SKIP_REASON)
class TestDocLensAndStarts(unittest.TestCase):
    """get_doc_lens / get_doc_starts on packed-document boundaries."""

    def test_doc_lens_two_docs(self):
        # doc0 ends at 3 (len 3), doc1 ends at 5 (len 2).
        idx = paddle.to_tensor([3, 3, 3, 5, 5], dtype="int32").reshape(
            [1, 1, 5, 1]
        )
        out = tutils.get_doc_lens(idx)
        np.testing.assert_array_equal(out.numpy(), np.array([3, 2]))

    def test_doc_lens_three_docs(self):
        # lengths 2, 1, 2 -> ends 2, 3, 5.
        idx = paddle.to_tensor([2, 2, 3, 5, 5], dtype="int32").reshape(
            [1, 1, 5, 1]
        )
        out = tutils.get_doc_lens(idx)
        np.testing.assert_array_equal(out.numpy(), np.array([2, 1, 2]))

    def test_doc_starts_multi(self):
        lens = paddle.to_tensor([2, 1, 2], dtype="int32")
        out = tutils.get_doc_starts(lens)
        np.testing.assert_array_equal(out.numpy(), np.array([0, 2, 3]))

    def test_doc_starts_single_doc_is_zero(self):
        lens = paddle.to_tensor([4], dtype="int32")
        out = tutils.get_doc_starts(lens)
        np.testing.assert_array_equal(out.numpy(), np.array([0]))


if __name__ == "__main__":
    unittest.main()
