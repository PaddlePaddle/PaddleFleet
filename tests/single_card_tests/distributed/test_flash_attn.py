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

"""Behavior tests for CPU-executable helpers in
``paddlefleet.refined_recompute.flash_attn``.

The three functions exercised here are pure tensor-shaping / dtype logic that
runs on CPU without any FlashAttention GPU kernel:

* ``flashattn_auto_cast`` -- conditionally casts each of q/k/v to a target
  dtype, leaving tensors that already match untouched (same object).
* ``slice_ulysses_mask_heads`` -- selects the FlashMask per-head index slice
  that a given Ulysses head-parallel rank owns.
* ``slice_ulysses_sink_heads`` -- selects the learnable-sink slice that a
  given rank owns.

Every expected value is hand-derived from the raw input (via numpy indexing
or an independently constructed dtype), never by calling the function under
test. ``cp_group`` is a tiny collaborator stub carrying the ``nranks``/``rank``
attributes the production code reads; the code under test is never mocked.

No paddle locally -> the import guard below reports an honest skip reason
rather than pretending the assertions ran.
"""

import unittest

try:
    import numpy as np
    import paddle

    from paddlefleet.refined_recompute.flash_attn import (
        flashattn_auto_cast,
        slice_ulysses_mask_heads,
        slice_ulysses_sink_heads,
    )

    _IMPORT_SKIP_REASON = None
except ImportError as exc:  # honest: only missing-dependency swallowed
    _IMPORT_SKIP_REASON = f"paddle / paddlefleet not importable: {exc}"


class _CpGroup:
    """Real collaborator stub mimicking the Ulysses CP process group.

    The production helpers only read ``.nranks`` and ``.rank`` off the group,
    so a plain value object is a faithful stand-in. This is a collaborator,
    not the code under test.
    """

    def __init__(self, nranks, rank):
        self.nranks = nranks
        self.rank = rank


@unittest.skipUnless(_IMPORT_SKIP_REASON is None, _IMPORT_SKIP_REASON or "")
class TestFlashattnAutoCast(unittest.TestCase):
    """flashattn_auto_cast casts only the tensors whose dtype differs."""

    def test_casts_all_when_dtype_differs(self):
        # float32 inputs, target float64 -> all three must be re-typed while
        # preserving values. Hand-derived expectation: identical numbers.
        base = np.array([[1.5, -2.25], [3.0, 4.75]], dtype=np.float32)
        q = paddle.to_tensor(base)
        k = paddle.to_tensor(base + 10.0)
        v = paddle.to_tensor(base + 20.0)

        qo, ko, vo = flashattn_auto_cast(q, k, v, dtype=paddle.float64)

        self.assertEqual(qo.dtype, paddle.float64)
        self.assertEqual(ko.dtype, paddle.float64)
        self.assertEqual(vo.dtype, paddle.float64)
        np.testing.assert_allclose(qo.numpy(), base.astype(np.float64))
        np.testing.assert_allclose(ko.numpy(), (base + 10.0).astype(np.float64))
        np.testing.assert_allclose(vo.numpy(), (base + 20.0).astype(np.float64))

    def test_no_copy_when_dtype_matches(self):
        # Target dtype already equals input dtype -> function must return the
        # very same objects (the `if dtype !=` guard must actually branch).
        q = paddle.to_tensor([1.0, 2.0], dtype=paddle.float32)
        k = paddle.to_tensor([3.0, 4.0], dtype=paddle.float32)
        v = paddle.to_tensor([5.0, 6.0], dtype=paddle.float32)

        qo, ko, vo = flashattn_auto_cast(q, k, v, dtype=paddle.float32)

        self.assertIs(qo, q)
        self.assertIs(ko, k)
        self.assertIs(vo, v)

    def test_each_tensor_checked_independently(self):
        # q already float64, k/v float32; target float64. Only k and v should
        # be cast; q must pass through as the same object.
        q = paddle.to_tensor([7.0, 8.0], dtype=paddle.float64)
        k = paddle.to_tensor([9.0, 10.0], dtype=paddle.float32)
        v = paddle.to_tensor([11.0, 12.0], dtype=paddle.float32)

        qo, ko, vo = flashattn_auto_cast(q, k, v, dtype=paddle.float64)

        self.assertIs(qo, q)  # untouched, no needless copy
        self.assertIsNot(ko, k)
        self.assertIsNot(vo, v)
        self.assertEqual(ko.dtype, paddle.float64)
        self.assertEqual(vo.dtype, paddle.float64)
        np.testing.assert_allclose(ko.numpy(), [9.0, 10.0])
        np.testing.assert_allclose(vo.numpy(), [11.0, 12.0])


@unittest.skipUnless(_IMPORT_SKIP_REASON is None, _IMPORT_SKIP_REASON or "")
class TestSliceUlyssesMaskHeads(unittest.TestCase):
    """slice_ulysses_mask_heads picks the local rank's per-head index block."""

    def test_single_mask_head_is_broadcast_untouched(self):
        # A single shared mask head is broadcast across all kv heads: the
        # function must return it verbatim (same object), regardless of rank.
        idx = paddle.arange(1 * 1 * 2 * 3, dtype="int64").reshape([1, 1, 2, 3])
        out = slice_ulysses_mask_heads(
            idx, num_k_heads=4, cp_group=_CpGroup(2, 1)
        )
        self.assertIs(out, idx)

    def test_selects_rank_head_block(self):
        # 4 mask heads, cp_size=2, rank=1 -> heads_per_rank=2, head_start=2,
        # so the local shard owns heads [2, 3]. Expected slice derived by
        # indexing the raw numpy array, independent of the function.
        raw = np.arange(1 * 4 * 2 * 3, dtype=np.int64).reshape([1, 4, 2, 3])
        idx = paddle.to_tensor(raw)

        out = slice_ulysses_mask_heads(
            idx, num_k_heads=4, cp_group=_CpGroup(2, 1)
        )

        expected = raw[:, 2:4, :, :]
        self.assertEqual(list(out.shape), [1, 2, 2, 3])
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_rank0_gets_leading_block(self):
        raw = np.arange(1 * 4 * 2 * 3, dtype=np.int64).reshape([1, 4, 2, 3])
        idx = paddle.to_tensor(raw)

        out = slice_ulysses_mask_heads(
            idx, num_k_heads=4, cp_group=_CpGroup(2, 0)
        )

        np.testing.assert_array_equal(out.numpy(), raw[:, 0:2, :, :])

    def test_invalid_mask_head_count_rejected(self):
        # 3 mask heads is neither 1 nor num_k_heads (4) -> contract violation.
        idx = paddle.arange(1 * 3 * 2 * 3, dtype="int64").reshape([1, 3, 2, 3])
        with self.assertRaises(AssertionError):
            slice_ulysses_mask_heads(
                idx, num_k_heads=4, cp_group=_CpGroup(2, 0)
            )

    def test_non_divisible_head_count_rejected(self):
        # 4 mask heads (== num_k_heads, passes first check) but cp_size=3
        # does not divide evenly -> must assert.
        idx = paddle.arange(1 * 4 * 2 * 3, dtype="int64").reshape([1, 4, 2, 3])
        with self.assertRaises(AssertionError):
            slice_ulysses_mask_heads(
                idx, num_k_heads=4, cp_group=_CpGroup(3, 0)
            )


@unittest.skipUnless(_IMPORT_SKIP_REASON is None, _IMPORT_SKIP_REASON or "")
class TestSliceUlyssesSinkHeads(unittest.TestCase):
    """slice_ulysses_sink_heads picks the local rank's learnable-sink block."""

    def test_none_passes_through(self):
        self.assertIsNone(
            slice_ulysses_sink_heads(
                None, num_q_heads=8, cp_group=_CpGroup(4, 0)
            )
        )

    def test_selects_rank_sink_block(self):
        # 8 query heads, cp_size=4, rank=2 -> heads_per_rank=2, head_start=4,
        # so this rank owns sink rows [4, 5]. Expected derived by raw indexing.
        raw = np.arange(8 * 2, dtype=np.float32).reshape([8, 2])
        sink = paddle.to_tensor(raw)

        out = slice_ulysses_sink_heads(
            sink, num_q_heads=8, cp_group=_CpGroup(4, 2)
        )

        expected = raw[4:6]
        self.assertEqual(list(out.shape), [2, 2])
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_last_rank_gets_trailing_block(self):
        raw = np.arange(8 * 2, dtype=np.float32).reshape([8, 2])
        sink = paddle.to_tensor(raw)

        out = slice_ulysses_sink_heads(
            sink, num_q_heads=8, cp_group=_CpGroup(4, 3)
        )

        np.testing.assert_array_equal(out.numpy(), raw[6:8])

    def test_head_count_mismatch_rejected(self):
        # learnable_sink has 4 rows but caller declares 8 query heads ->
        # ValueError with a precise contract, not a silent slice.
        sink = paddle.arange(4 * 2, dtype="float32").reshape([4, 2])
        with self.assertRaises(ValueError):
            slice_ulysses_sink_heads(
                sink, num_q_heads=8, cp_group=_CpGroup(4, 0)
            )


if __name__ == "__main__":
    unittest.main()
