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

"""Behavior tests for ``paddlefleet.transformer.utils`` document-mask helpers.

Every expected value here is derived independently of the production code:

* ``get_doc_lens`` / ``get_doc_starts`` are checked against hand-built packed
  layouts (contiguous, exclusive end-boundary encoding) and their inverse.
* ``get_sliding_window_causal_mask`` is compared to a band reference expressed
  purely as ``offset - left <= (j - i) <= offset + right`` -- it never touches
  the ``triu``/``tril`` path under test, so a wrong diagonal offset is caught.
* ``get_default_causal_mask`` is compared to ``np.triu(..., k=1)``.

Pure-CPU logic; requires only Paddle tensor ops (bool indexing, nonzero,
cumsum, triu/tril). If Paddle or the module cannot be imported the whole file
skips with the real import error, never a fake pass.
"""

from __future__ import annotations

import unittest

import numpy as np

try:
    import paddle

    from paddlefleet.transformer.utils import (
        get_default_causal_mask,
        get_doc_lens,
        get_doc_starts,
        get_sliding_window_causal_mask,
        get_sliding_window_left_size,
        is_layer_window_attention,
    )

    _IMPORT_ERROR: Exception | None = None
except (ImportError, ModuleNotFoundError) as exc:
    paddle = None
    _IMPORT_ERROR = exc

_HAS_DEPS = _IMPORT_ERROR is None
_SKIP_REASON = (
    f"paddle / paddlefleet.transformer.utils not importable: {_IMPORT_ERROR!r}"
)


def _encode_end_boundaries(doc_lengths: list[int]) -> list[int]:
    """Independent packed encoding: each position holds its doc's exclusive end.

    For docs of lengths ``[l0, l1, ...]`` the cumulative ends are
    ``[l0, l0+l1, ...]``; every position inside doc ``k`` stores end ``k``.
    This is the layout ``get_doc_lens`` must invert -- built here by hand so
    the test input never comes from the production helper.
    """
    values: list[int] = []
    cum = 0
    for length in doc_lengths:
        cum += length
        values.extend([cum] * length)
    return values


def _independent_swa_mask(sq, skv, sliding_window):
    """Band reference for the SWA mask (True == masked/blocked).

    Independent of the production triu/tril construction: a position ``(i, j)``
    is *allowed* iff ``offset - left <= (j - i) <= offset + right`` where
    ``offset = skv - sq``. ``left < 0`` means an infinite past window (no lower
    truncation), i.e. a plain causal mask.
    """
    if isinstance(sliding_window, int):
        left, right = sliding_window, 0
    else:
        left, right = sliding_window[0], sliding_window[1]
    offset = skv - sq
    masked = np.zeros((sq, skv), dtype=bool)
    for i in range(sq):
        for j in range(skv):
            d = j - i
            lower_ok = True if left < 0 else (d >= offset - left)
            upper_ok = d <= offset + right
            masked[i, j] = not (lower_ok and upper_ok)
    return masked


def _startend(values: list[int]):
    """Wrap a flat end-boundary list into the [b, h, seqlen, 1] input shape."""
    return paddle.to_tensor(values, dtype="int32").reshape(
        [1, 1, len(values), 1]
    )


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestGetDocLens(unittest.TestCase):
    """get_doc_lens must invert the packed end-boundary encoding exactly."""

    def test_single_document_spans_whole_sequence(self):
        out = get_doc_lens(_startend(_encode_end_boundaries([4])))
        self.assertEqual(out.numpy().tolist(), [4])
        self.assertEqual(out.dtype, paddle.int32)

    def test_equal_length_docs_split_at_the_right_boundary(self):
        # [2,2,4,4]: the change 2->4 at position 2 is the only interior break.
        out = get_doc_lens(_startend(_encode_end_boundaries([2, 2])))
        self.assertEqual(out.numpy().tolist(), [2, 2])

    def test_increasing_length_docs(self):
        out = get_doc_lens(_startend(_encode_end_boundaries([1, 2, 3])))
        self.assertEqual(out.numpy().tolist(), [1, 2, 3])

    def test_leading_short_then_long_doc(self):
        # [1,4,4,4]: a length-1 doc must not swallow the following doc.
        out = get_doc_lens(_startend(_encode_end_boundaries([1, 3])))
        self.assertEqual(out.numpy().tolist(), [1, 3])

    def test_long_then_trailing_short_doc(self):
        # [3,3,3,4]: the final single-token doc must be recovered, not dropped.
        out = get_doc_lens(_startend(_encode_end_boundaries([3, 1])))
        self.assertEqual(out.numpy().tolist(), [3, 1])

    def test_all_single_token_docs(self):
        out = get_doc_lens(_startend(_encode_end_boundaries([1, 1, 1, 1])))
        self.assertEqual(out.numpy().tolist(), [1, 1, 1, 1])

    def test_recovers_arbitrary_hand_built_layout(self):
        lengths = [2, 5, 1, 3, 4]
        out = get_doc_lens(_startend(_encode_end_boundaries(lengths)))
        self.assertEqual(out.numpy().tolist(), lengths)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestGetDocStarts(unittest.TestCase):
    """get_doc_starts is the exclusive prefix sum of document lengths."""

    def test_single_doc_starts_at_zero(self):
        out = get_doc_starts(paddle.to_tensor([5], dtype="int32"))
        self.assertEqual(out.numpy().tolist(), [0])

    @unittest.expectedFailure
    def test_single_doc_starts_dtype_is_documented_int32(self):
        # Production bug (documented, not fixed here): get_doc_starts promises
        # an int32 tensor in its docstring and explicitly does
        # ``doc_lens.flatten().cast("int32")`` on its input, but paddle.cumsum
        # promotes the accumulation to int64 and the result is never cast back
        # -- unlike get_doc_lens, which ends with ``.cast("int32")``. The
        # returned dtype is therefore int64, violating the documented contract.
        # Captured as an expected failure without modifying production code.
        out = get_doc_starts(paddle.to_tensor([5], dtype="int32"))
        self.assertEqual(out.dtype, paddle.int32)

    def test_multiple_docs_cumulative_offsets(self):
        out = get_doc_starts(paddle.to_tensor([2, 3, 4], dtype="int32"))
        # exclusive prefix sum: [0, 2, 2+3]
        self.assertEqual(out.numpy().tolist(), [0, 2, 5])

    def test_single_token_docs(self):
        out = get_doc_starts(paddle.to_tensor([1, 1, 1], dtype="int32"))
        self.assertEqual(out.numpy().tolist(), [0, 1, 2])

    def test_starts_match_independent_prefix_sum(self):
        lengths = [4, 1, 7, 2, 5]
        out = get_doc_starts(paddle.to_tensor(lengths, dtype="int32"))
        expected = np.concatenate([[0], np.cumsum(lengths)[:-1]]).tolist()
        self.assertEqual(out.numpy().tolist(), expected)

    def test_roundtrip_lens_then_starts(self):
        # Independent chain: hand encoding -> get_doc_lens -> get_doc_starts.
        lengths = [3, 2, 4]
        lens = get_doc_lens(_startend(_encode_end_boundaries(lengths)))
        starts = get_doc_starts(lens)
        self.assertEqual(starts.numpy().tolist(), [0, 3, 5])


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestGetDefaultCausalMask(unittest.TestCase):
    """True marks positions a query may NOT attend to (strict upper triangle)."""

    def test_matches_numpy_triu(self):
        for sq in (1, 3, 5):
            mask = get_default_causal_mask(sq)
            expected = np.triu(np.ones((sq, sq)), k=1).astype(bool)
            np.testing.assert_array_equal(mask.numpy(), expected)

    def test_diagonal_and_past_are_visible(self):
        mask = get_default_causal_mask(4).numpy()
        # nothing on or below the diagonal is masked
        self.assertFalse(mask[np.tril_indices(4)].any())
        # everything strictly above the diagonal is masked
        self.assertTrue(mask[np.triu_indices(4, k=1)].all())


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestGetSlidingWindowLeftSize(unittest.TestCase):
    def test_int_form_returns_itself(self):
        self.assertEqual(get_sliding_window_left_size(5), 5)

    def test_tuple_form_returns_left_only(self):
        self.assertEqual(get_sliding_window_left_size((3, 7)), 3)

    def test_negative_infinite_window_passthrough(self):
        self.assertEqual(get_sliding_window_left_size(-1), -1)
        self.assertEqual(get_sliding_window_left_size((-1, 2)), -1)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestGetSlidingWindowCausalMask(unittest.TestCase):
    """Full-matrix comparison against an independent band reference."""

    def test_square_window_two(self):
        mask = get_sliding_window_causal_mask(4, 4, 2)
        np.testing.assert_array_equal(
            mask.numpy(), _independent_swa_mask(4, 4, 2)
        )

    def test_non_square_two_sided_tuple(self):
        mask = get_sliding_window_causal_mask(3, 5, (1, 1))
        np.testing.assert_array_equal(
            mask.numpy(), _independent_swa_mask(3, 5, (1, 1))
        )

    def test_infinite_window_is_plain_causal(self):
        mask = get_sliding_window_causal_mask(4, 4, -1)
        causal = np.triu(np.ones((4, 4)), k=1).astype(bool)
        np.testing.assert_array_equal(mask.numpy(), causal)
        np.testing.assert_array_equal(
            mask.numpy(), _independent_swa_mask(4, 4, -1)
        )

    def test_int_form_matches_tuple_with_zero_right(self):
        # int W has one-sided (left=W, right=0) semantics == tuple (W, 0).
        as_int = get_sliding_window_causal_mask(5, 5, 3)
        as_tuple = get_sliding_window_causal_mask(5, 5, (3, 0))
        np.testing.assert_array_equal(as_int.numpy(), as_tuple.numpy())

    def test_window_actually_truncates_the_past(self):
        # A tighter window must mask strictly more than an infinite one; guards
        # against the left truncation silently being a no-op.
        tight = get_sliding_window_causal_mask(6, 6, 2).numpy()
        infinite = get_sliding_window_causal_mask(6, 6, -1).numpy()
        self.assertGreater(int(tight.sum()), int(infinite.sum()))
        # every position infinite-masks must also be tight-masked (superset)
        self.assertTrue(np.all(tight[infinite]))


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestIsLayerWindowAttention(unittest.TestCase):
    def test_disabled_when_no_sliding_window(self):
        self.assertFalse(is_layer_window_attention(None, 2, 0))
        self.assertFalse(is_layer_window_attention(0, None, 5))

    def test_none_freq_enables_every_layer(self):
        self.assertTrue(is_layer_window_attention(4, None, 0))
        self.assertTrue(is_layer_window_attention(4, None, 9))

    def test_int_freq_skips_multiples(self):
        # freq=2: layer % 2 != 0 -> odd layers windowed, even layers not.
        self.assertFalse(is_layer_window_attention(4, 2, 0))
        self.assertTrue(is_layer_window_attention(4, 2, 1))
        self.assertFalse(is_layer_window_attention(4, 2, 2))
        self.assertTrue(is_layer_window_attention(4, 2, 3))

    def test_list_freq_indexes_per_layer(self):
        freq = [0, 1, 1, 0]
        self.assertFalse(is_layer_window_attention(4, freq, 0))
        self.assertTrue(is_layer_window_attention(4, freq, 1))
        self.assertTrue(is_layer_window_attention(4, freq, 2))
        self.assertFalse(is_layer_window_attention(4, freq, 3))

    def test_invalid_freq_type_raises(self):
        with self.assertRaises(ValueError):
            is_layer_window_attention(4, 1.5, 0)


if __name__ == "__main__":
    unittest.main()
