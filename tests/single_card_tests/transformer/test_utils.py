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

Covers the attention-mask builders (causal / sliding-window), the shared
window-size helper, the in-place mask-fill, the per-layer window-attention
predicate, the sliding-window row-index clamp and the document-length /
document-start helpers.

Expected values are derived independently (hand-written literal masks and a
small pure-numpy banded-mask reference), never by calling the production
function under test. ``paddlefleet.transformer.utils`` imports ``paddle`` (and
``paddlefleet.training.global_vars``) at module load, so the whole import is
guarded; when paddle is unavailable the suite is honestly skipped with the
underlying error repr rather than faked green.
"""

import unittest

import numpy as np

_IMPORT_ERROR = None
try:
    import paddle

    from paddlefleet.transformer.utils import (
        attention_mask_func,
        get_default_causal_mask,
        get_doc_lens,
        get_doc_starts,
        get_sliding_window_causal_mask,
        get_sliding_window_left_size,
        is_layer_window_attention,
        startend_row_indices_add_sliding_window,
    )
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    paddle = None
    attention_mask_func = None
    get_default_causal_mask = None
    get_doc_lens = None
    get_doc_starts = None
    get_sliding_window_causal_mask = None
    get_sliding_window_left_size = None
    is_layer_window_attention = None
    startend_row_indices_add_sliding_window = None
    _IMPORT_ERROR = exc

_HAS_DEPS = _IMPORT_ERROR is None
_SKIP_REASON = (
    ""
    if _HAS_DEPS
    else f"paddlefleet.transformer.utils import failed (needs paddle): {_IMPORT_ERROR!r}"
)


# --- Independent references (no call into the module under test) -----------
def _reference_causal_mask(n):
    """True == masked. Upper triangle excluding the diagonal (future keys).

    numpy implementation, independent of paddle.triu.
    """
    return np.triu(np.ones((n, n), dtype=bool), k=1)


def _reference_swa_mask(sq, skv, left, right):
    """True == masked. Independent banded-window reference.

    A query row ``i`` may attend key col ``j`` iff
    ``(skv - sq) - left <= (j - i) <= (skv - sq) + right``.
    ``left < 0`` denotes an infinite past window (no lower bound), leaving a
    pure causal band bounded above by ``right``.
    """
    k = skv - sq
    mask = np.ones((sq, skv), dtype=bool)
    for i in range(sq):
        for j in range(skv):
            diff = j - i
            upper_ok = diff <= k + right
            lower_ok = True if left < 0 else diff >= k - left
            if upper_ok and lower_ok:
                mask[i, j] = False
    return mask


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestGetDefaultCausalMask(unittest.TestCase):
    def test_matches_independent_numpy_triu(self):
        for n in (1, 2, 5):
            mask = get_default_causal_mask(n)
            self.assertEqual(list(mask.shape), [n, n])
            self.assertEqual(mask.dtype, paddle.bool)
            np.testing.assert_array_equal(
                mask.numpy(), _reference_causal_mask(n)
            )

    def test_explicit_hand_written_mask(self):
        # Hand-written literal: True == masked future key (j > i).
        expected = np.array(
            [
                [False, True, True, True],
                [False, False, True, True],
                [False, False, False, True],
                [False, False, False, False],
            ]
        )
        np.testing.assert_array_equal(
            get_default_causal_mask(4).numpy(), expected
        )

    def test_lru_cache_returns_identical_object(self):
        # Caching identity is part of the contract (avoids rebuilding masks).
        self.assertIs(get_default_causal_mask(6), get_default_causal_mask(6))


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestGetSlidingWindowCausalMask(unittest.TestCase):
    def test_square_tuple_window_full_content(self):
        # (left=2, right=0) causal one-sided window, square 4x4.
        expected = np.array(
            [
                [False, True, True, True],
                [False, False, True, True],
                [False, False, False, True],
                [True, False, False, False],  # j=0 falls out of the past window
            ]
        )
        mask = get_sliding_window_causal_mask(4, 4, (2, 0))
        self.assertEqual(mask.dtype, paddle.bool)
        np.testing.assert_array_equal(mask.numpy(), expected)
        np.testing.assert_array_equal(
            mask.numpy(), _reference_swa_mask(4, 4, 2, 0)
        )

    def test_non_square_two_sided_window_full_content(self):
        expected = np.array(
            [
                [True, True, True, False, False, False, False, False],
                [True, True, True, True, False, False, False, False],
            ]
        )
        mask = get_sliding_window_causal_mask(2, 8, (3, 3))
        self.assertEqual(list(mask.shape), [2, 8])
        np.testing.assert_array_equal(mask.numpy(), expected)
        np.testing.assert_array_equal(
            mask.numpy(), _reference_swa_mask(2, 8, 3, 3)
        )

    def test_int_form_equals_tuple_left_right_zero_and_reference(self):
        # int W is documented as (left=W, right=0). Verify against the tuple
        # path AND an independent banded reference (not the causal helper).
        int_mask = get_sliding_window_causal_mask(6, 6, 2)
        tuple_mask = get_sliding_window_causal_mask(6, 6, (2, 0))
        ref = _reference_swa_mask(6, 6, 2, 0)
        np.testing.assert_array_equal(int_mask.numpy(), ref)
        np.testing.assert_array_equal(tuple_mask.numpy(), ref)
        np.testing.assert_array_equal(int_mask.numpy(), tuple_mask.numpy())

    def test_int_negative_is_causal_via_independent_reference(self):
        # -1 (infinite window) must degrade to a plain causal mask. Anchor to
        # the independent numpy causal reference, not to get_default_causal_mask.
        mask = get_sliding_window_causal_mask(4, 4, -1)
        np.testing.assert_array_equal(mask.numpy(), _reference_causal_mask(4))
        # Self-attention diagonal must stay visible (regression guard).
        self.assertFalse(mask.numpy()[0, 0])
        self.assertFalse(mask.numpy()[3, 3])

    def test_tuple_negative_left_is_causal(self):
        mask = get_sliding_window_causal_mask(4, 4, (-1, 0))
        np.testing.assert_array_equal(mask.numpy(), _reference_causal_mask(4))


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestGetSlidingWindowLeftSize(unittest.TestCase):
    def test_int_returned_as_is(self):
        self.assertEqual(get_sliding_window_left_size(512), 512)

    def test_tuple_returns_left_element(self):
        self.assertEqual(get_sliding_window_left_size((128, 0)), 128)
        self.assertEqual(get_sliding_window_left_size((256, 64)), 256)

    def test_negative_forms_preserved(self):
        self.assertEqual(get_sliding_window_left_size(-1), -1)
        self.assertEqual(get_sliding_window_left_size((-1, 0)), -1)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestAttentionMaskFunc(unittest.TestCase):
    def test_fills_masked_positions_in_place(self):
        scores = paddle.to_tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=paddle.float32
        )
        mask = paddle.to_tensor(
            [[False, True], [False, False]], dtype=paddle.bool
        )
        result = attention_mask_func(scores, mask)
        # Same object returned and mutated in place.
        self.assertIs(result, scores)
        expected = np.array([[1.0, -10000.0], [3.0, 4.0]], dtype=np.float32)
        np.testing.assert_array_equal(result.numpy(), expected)
        np.testing.assert_array_equal(scores.numpy(), expected)

    def test_all_masked(self):
        scores = paddle.ones([3, 3], dtype=paddle.float32)
        mask = paddle.ones([3, 3], dtype=paddle.bool)
        result = attention_mask_func(scores, mask)
        np.testing.assert_array_equal(
            result.numpy(), np.full((3, 3), -10000.0, dtype=np.float32)
        )

    def test_no_mask_leaves_values_untouched(self):
        scores = paddle.to_tensor(
            [[5.0, 6.0], [7.0, 8.0]], dtype=paddle.float32
        )
        mask = paddle.zeros([2, 2], dtype=paddle.bool)
        result = attention_mask_func(scores, mask)
        np.testing.assert_array_equal(
            result.numpy(),
            np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
        )


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestIsLayerWindowAttention(unittest.TestCase):
    def test_falsy_sliding_window_is_never_window(self):
        self.assertFalse(is_layer_window_attention(None, None, 1))
        self.assertFalse(is_layer_window_attention(0, None, 1))
        self.assertFalse(is_layer_window_attention((), 2, 1))

    def test_no_skip_freq_means_always_window(self):
        self.assertTrue(is_layer_window_attention((3, 3), None, 0))
        self.assertTrue(is_layer_window_attention((3, 3), None, 7))

    def test_int_skip_freq_skips_multiples_of_freq(self):
        # layer_number % freq == 0 -> skipped (returns False).
        self.assertFalse(is_layer_window_attention((3, 3), 2, 0))
        self.assertTrue(is_layer_window_attention((3, 3), 2, 1))
        self.assertFalse(is_layer_window_attention((3, 3), 2, 4))
        self.assertTrue(is_layer_window_attention((3, 3), 2, 3))

    def test_list_skip_freq_indexes_per_layer(self):
        flags = [1, 0, 1]
        self.assertTrue(is_layer_window_attention((3, 3), flags, 0))
        self.assertFalse(is_layer_window_attention((3, 3), flags, 1))
        self.assertTrue(is_layer_window_attention((3, 3), flags, 2))

    def test_list_skip_freq_boolean_entries(self):
        self.assertTrue(
            is_layer_window_attention((3, 3), [True, False, True], 0)
        )
        self.assertFalse(
            is_layer_window_attention((3, 3), [True, False, True], 1)
        )

    def test_list_skip_freq_boundary_index(self):
        flags = [1, 1, 1, 1, 0, 0]
        self.assertFalse(is_layer_window_attention((3, 3), flags, 5))
        self.assertTrue(is_layer_window_attention((3, 3), flags, 3))

    def test_invalid_skip_freq_type_raises(self):
        with self.assertRaises(ValueError):
            is_layer_window_attention((3, 3), {"invalid": 1}, 1)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestStartendRowIndicesAddSlidingWindow(unittest.TestCase):
    def _indices(self, values):
        # values: nested list shaped [bsz, heads, seq, num_vec].
        return paddle.to_tensor(np.array(values, dtype=np.int32))

    def test_no_sliding_window_passthrough_identity(self):
        idx = self._indices([[[[8], [8], [8]]]])
        out = startend_row_indices_add_sliding_window(idx, None, 0.0, 2)
        self.assertIs(out, idx)

    def test_zero_window_passthrough_identity(self):
        idx = self._indices([[[[8], [8], [8]]]])
        out = startend_row_indices_add_sliding_window(idx, 0, 0.0, 2)
        self.assertIs(out, idx)

    def test_negative_window_passthrough_identity(self):
        # -1 (infinite window) must not truncate the row indices.
        idx = self._indices([[[[8], [8], [8]]]])
        out = startend_row_indices_add_sliding_window(idx, -1, 0.0, 2)
        self.assertIs(out, idx)
        out_t = startend_row_indices_add_sliding_window(idx, (-1, 0), 0.0, 2)
        self.assertIs(out_t, idx)

    def test_clamps_each_position_to_window_plus_index(self):
        # seq=6, window=3 -> per-position cap is [3,4,5,6,7,8]; the LTS value is
        # clamped to min(original, window + position). heads==1 input expands to
        # kv_num_heads=2 (identical heads).
        orig = [0, 4, 4, 100, 3, 100]
        idx = self._indices([[[[v] for v in orig]]])  # [1, 1, 6, 1]
        out = startend_row_indices_add_sliding_window(idx, 3, 0.0, 2)
        self.assertEqual(list(out.shape), [1, 2, 6, 1])
        expected_pos = np.array([0, 4, 4, 6, 3, 8], dtype=np.int32)
        for head in range(2):
            np.testing.assert_array_equal(
                out.numpy()[0, head, :, 0], expected_pos
            )

    def test_int_and_tuple_forms_produce_same_clamp(self):
        orig = [10, 10, 10, 10]
        idx_int = self._indices([[[[v] for v in orig]]])
        idx_tuple = self._indices([[[[v] for v in orig]]])
        out_int = startend_row_indices_add_sliding_window(idx_int, 2, 0.0, 2)
        out_tuple = startend_row_indices_add_sliding_window(
            idx_tuple, (2, 0), 0.0, 2
        )
        # window=2, seq=4 -> cap [2,3,4,5]; min(10, cap) = [2,3,4,5].
        expected_pos = np.array([2, 3, 4, 5], dtype=np.int32)
        np.testing.assert_array_equal(out_int.numpy()[0, 0, :, 0], expected_pos)
        np.testing.assert_array_equal(out_int.numpy(), out_tuple.numpy())

    def test_head_wise_ratio_restores_non_swa_heads(self):
        # kv_num_heads=4, ratio=0.5 -> swa_head_num=2, so the first 2 (non-swa)
        # heads keep the ORIGINAL indices and the last 2 heads are clamped.
        orig = [100, 100, 100, 100]
        idx = self._indices([[[[v] for v in orig]]])  # heads==1 -> expands to 4
        out = startend_row_indices_add_sliding_window(idx, 2, 0.5, 4)
        self.assertEqual(list(out.shape), [1, 4, 4, 1])
        clamped = np.array([2, 3, 4, 5], dtype=np.int32)
        unchanged = np.array([100, 100, 100, 100], dtype=np.int32)
        np.testing.assert_array_equal(out.numpy()[0, 0, :, 0], unchanged)
        np.testing.assert_array_equal(out.numpy()[0, 1, :, 0], unchanged)
        np.testing.assert_array_equal(out.numpy()[0, 2, :, 0], clamped)
        np.testing.assert_array_equal(out.numpy()[0, 3, :, 0], clamped)

    def test_num_vec_two_clamps_lts_and_keeps_ute(self):
        # num_vec==2 (LTS & UTE): first channel clamped, second passes through.
        lts = np.array([100, 100, 100, 100], dtype=np.int32)
        ute = np.array([7, 7, 7, 7], dtype=np.int32)
        arr = np.stack([lts, ute], axis=-1).reshape(1, 1, 4, 2)
        idx = paddle.to_tensor(arr)
        out = startend_row_indices_add_sliding_window(idx, 2, 0.0, 2)
        self.assertEqual(list(out.shape), [1, 2, 4, 2])
        for head in range(2):
            np.testing.assert_array_equal(
                out.numpy()[0, head, :, 0],
                np.array([2, 3, 4, 5], dtype=np.int32),
            )
            np.testing.assert_array_equal(out.numpy()[0, head, :, 1], ute)

    def test_invalid_num_vec_raises(self):
        idx = self._indices([[[[1, 2, 3], [1, 2, 3]]]])  # num_vec == 3
        with self.assertRaises(ValueError):
            startend_row_indices_add_sliding_window(idx, 2, 0.0, 2)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestDocLenHelpers(unittest.TestCase):
    def test_get_doc_starts_is_exclusive_prefix_sum(self):
        lens = paddle.to_tensor(np.array([3, 2, 4], dtype=np.int32))
        starts = get_doc_starts(lens)
        np.testing.assert_array_equal(
            starts.numpy(), np.array([0, 3, 5], dtype=np.int32)
        )

    def test_get_doc_starts_single_element(self):
        lens = paddle.to_tensor(np.array([5], dtype=np.int32))
        starts = get_doc_starts(lens)
        np.testing.assert_array_equal(
            starts.numpy(), np.array([0], dtype=np.int32)
        )

    def test_get_doc_lens_recovers_two_docs(self):
        # positions 0..2 belong to doc ending at 3, positions 3..4 to doc
        # ending at 5 -> lengths [3, 2].
        startend = paddle.to_tensor(np.array([3, 3, 3, 5, 5], dtype=np.int32))
        lens = get_doc_lens(startend)
        np.testing.assert_array_equal(
            lens.numpy(), np.array([3, 2], dtype=np.int32)
        )

    def test_get_doc_lens_recovers_three_docs(self):
        startend = paddle.to_tensor(
            np.array([2, 2, 4, 4, 7, 7, 7], dtype=np.int32)
        )
        lens = get_doc_lens(startend)
        np.testing.assert_array_equal(
            lens.numpy(), np.array([2, 2, 3], dtype=np.int32)
        )

    def test_doc_lens_and_starts_roundtrip(self):
        startend = paddle.to_tensor(
            np.array([2, 2, 4, 4, 7, 7, 7], dtype=np.int32)
        )
        lens = get_doc_lens(startend)
        starts = get_doc_starts(lens)
        np.testing.assert_array_equal(
            starts.numpy(), np.array([0, 2, 4], dtype=np.int32)
        )


if __name__ == "__main__":
    unittest.main()
