# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Layout-aware CP slicing for the ``use_erndata`` MTP path.

That path never calls ``ContextParallelScatterOp``: it keeps its tensors
full-length on every CP rank and slices the local part itself. The slice must
reproduce the layout the rest of the model scatters with, or a rank holds
*other* ranks' tokens and the loss is silently wrong. So the parity checks call
``context_parallel_utils``' own scatter helpers rather than a hand-written
slice, which would keep passing while the real scatter drifted underneath:

  ``dualchunk_allgather``  -> ``scatter_balance``    (two zigzag chunks)
  ``contiguous_allgather`` -> ``scatter_contiguous`` (one rank-order chunk)

Both helpers read only ``group.nranks`` / ``group.rank``, so ``_FakeGroup``
below is enough to exercise them single-card with no distributed init.

Also covered: ``mode`` is required and keyword-only, unsupported modes raise
instead of defaulting to a layout, divisibility guards, and negative /
non-sequence axes.
"""

from __future__ import annotations

import unittest

import paddle

from paddlefleet.context_parallel_utils import (
    scatter_balance,
    scatter_contiguous,
)
from paddlefleet.transformer.multi_token_prediction import (
    extract_local_contiguous_chunk,
    extract_local_cp_chunks,
    extract_local_zigzag_chunks,
)


class _FakeGroup:
    """Minimal stand-in for a paddle CP group.

    ``scatter_balance`` and ``scatter_contiguous`` touch nothing but these two
    attributes, so this keeps the parity checks single-card.
    """

    def __init__(self, rank: int, nranks: int) -> None:
        self.rank = rank
        self.nranks = nranks


def _arange_bl(batch: int, length: int) -> paddle.Tensor:
    """[B, L] with globally unique values so slices are traceable."""
    return paddle.arange(batch * length, dtype="int64").reshape([batch, length])


class TestExtractLocalContiguousChunk(unittest.TestCase):
    def test_matches_scatter_contiguous_layout(self) -> None:
        # Parity against the real scatter, not a re-derivation of it.
        cp_size, length = 4, 16
        t = _arange_bl(2, length)
        for rank in range(cp_size):
            local = extract_local_contiguous_chunk(t, rank, cp_size, axis=1)
            expected = scatter_contiguous(
                t, group=_FakeGroup(rank, cp_size), axis=1
            )
            self.assertEqual(local.shape, expected.shape)
            self.assertTrue(bool((local == expected).all()))

    def test_ranks_tile_the_sequence_exactly_once(self) -> None:
        cp_size, length = 4, 16
        t = _arange_bl(1, length)
        rebuilt = paddle.concat(
            [
                extract_local_contiguous_chunk(t, r, cp_size, axis=1)
                for r in range(cp_size)
            ],
            axis=1,
        )
        self.assertTrue(bool((rebuilt == t).all()))

    def test_cp_size_one_is_identity(self) -> None:
        t = _arange_bl(2, 7)
        self.assertIs(extract_local_contiguous_chunk(t, 0, 1, axis=1), t)

    def test_indivisible_length_raises(self) -> None:
        # scatter_contiguous refuses an uneven split (it would drop the tail
        # while all_gather_contiguous still reports the shorter length), so the
        # local-slice twin must refuse it too.
        t = _arange_bl(1, 10)
        with self.assertRaisesRegex(ValueError, r"divisible by cp_size"):
            extract_local_contiguous_chunk(t, 0, 4, axis=1)

    def test_negative_axis(self) -> None:
        t = _arange_bl(2, 8)
        self.assertTrue(
            bool(
                (
                    extract_local_contiguous_chunk(t, 1, 2, axis=-1)
                    == extract_local_contiguous_chunk(t, 1, 2, axis=1)
                ).all()
            )
        )

    def test_slices_only_the_requested_axis(self) -> None:
        t = paddle.arange(2 * 8 * 3, dtype="int64").reshape([2, 8, 3])
        local = extract_local_contiguous_chunk(t, 1, 2, axis=1)
        self.assertEqual(local.shape, [2, 4, 3])
        self.assertTrue(bool((local == t[:, 4:8, :]).all()))


class TestExtractLocalCpChunksDispatch(unittest.TestCase):
    def test_dualchunk_matches_scatter_balance(self) -> None:
        # cp_size=4: with only two ranks the zigzag layout degenerates to
        # "first quarter + last quarter", which several wrong implementations
        # also produce. Four ranks pin the interleaving.
        cp_size, length = 4, 16
        t = _arange_bl(2, length)
        for rank in range(cp_size):
            got = extract_local_cp_chunks(
                t, rank, cp_size, axis=1, mode="dualchunk_allgather"
            )
            expected = scatter_balance(
                t, group=_FakeGroup(rank, cp_size), axis=1
            )
            self.assertEqual(got.shape, expected.shape)
            self.assertTrue(bool((got == expected).all()))

    def test_contiguous_matches_scatter_contiguous(self) -> None:
        cp_size, length = 4, 16
        t = _arange_bl(2, length)
        for rank in range(cp_size):
            got = extract_local_cp_chunks(
                t, rank, cp_size, axis=1, mode="contiguous_allgather"
            )
            expected = scatter_contiguous(
                t, group=_FakeGroup(rank, cp_size), axis=1
            )
            self.assertEqual(got.shape, expected.shape)
            self.assertTrue(bool((got == expected).all()))

    def test_dualchunk_delegates_to_zigzag(self) -> None:
        cp_size, length = 2, 16
        t = _arange_bl(2, length)
        for rank in range(cp_size):
            got = extract_local_cp_chunks(
                t, rank, cp_size, axis=1, mode="dualchunk_allgather"
            )
            expected = extract_local_zigzag_chunks(t, rank, cp_size, axis=1)
            self.assertTrue(bool((got == expected).all()))

    def test_contiguous_delegates_to_contiguous(self) -> None:
        cp_size, length = 2, 16
        t = _arange_bl(2, length)
        for rank in range(cp_size):
            got = extract_local_cp_chunks(
                t, rank, cp_size, axis=1, mode="contiguous_allgather"
            )
            expected = extract_local_contiguous_chunk(t, rank, cp_size, axis=1)
            self.assertTrue(bool((got == expected).all()))

    def test_the_two_layouts_actually_differ(self) -> None:
        # Guards the whole point of this change: if these were equal, routing
        # the erndata MTP path through the mode would be a no-op and the bug
        # (zigzag labels vs contiguous logits) would be invisible.
        t = _arange_bl(1, 16)
        zig = extract_local_cp_chunks(
            t, 0, 2, axis=1, mode="dualchunk_allgather"
        )
        con = extract_local_cp_chunks(
            t, 0, 2, axis=1, mode="contiguous_allgather"
        )
        self.assertEqual(zig.shape, con.shape)
        self.assertFalse(bool((zig == con).all()))

    def test_mode_is_required(self) -> None:
        # No default layout on purpose: the defect this helper fixes was a call
        # site that assumed zigzag instead of reading cp_balance_mode.
        t = _arange_bl(1, 16)
        with self.assertRaises(TypeError):
            extract_local_cp_chunks(t, 1, 2, axis=1)

    def test_mode_is_keyword_only(self) -> None:
        # Positional passing would let `axis` and `mode` be swapped silently.
        t = _arange_bl(1, 16)
        with self.assertRaises(TypeError):
            extract_local_cp_chunks(t, 1, 2, 1, "dualchunk_allgather")

    def test_cp_size_one_is_identity_for_any_mode(self) -> None:
        # Identity, not a copy -- unlike scatter_balance (clone) and
        # scatter_contiguous (paddle.assign), so an in-place write on the result
        # would reach the caller's full-length tensor. Pinned because callers
        # rely on it for the cheap CP=1 path.
        t = _arange_bl(2, 7)
        for mode in (
            "dualchunk_allgather",
            "contiguous_allgather",
            "contiguous_a2a",
            "nonsense",
        ):
            self.assertIs(
                extract_local_cp_chunks(t, 0, 1, axis=1, mode=mode), t
            )

    def test_contiguous_a2a_raises(self) -> None:
        t = _arange_bl(1, 16)
        with self.assertRaisesRegex(ValueError, r"unsupported cp_balance_mode"):
            extract_local_cp_chunks(t, 0, 2, axis=1, mode="contiguous_a2a")

    def test_unknown_mode_raises(self) -> None:
        t = _arange_bl(1, 16)
        with self.assertRaisesRegex(ValueError, r"unsupported cp_balance_mode"):
            extract_local_cp_chunks(t, 0, 2, axis=1, mode="zigzag")

    def test_float_tensors_pass_through_both_modes(self) -> None:
        # The embedding call sites slice [B, L, H] float tensors.
        t = paddle.randn([2, 8, 4], dtype="float32")
        for mode in ("dualchunk_allgather", "contiguous_allgather"):
            local = extract_local_cp_chunks(t, 1, 2, axis=1, mode=mode)
            self.assertEqual(local.shape, [2, 4, 4])
            self.assertEqual(local.dtype, t.dtype)


if __name__ == "__main__":
    unittest.main()
