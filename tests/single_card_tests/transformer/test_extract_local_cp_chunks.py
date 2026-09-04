# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Layout-aware CP slicing for the ``use_erndata`` MTP path.

The erndata MTP path never calls ``ContextParallelScatterOp``: it keeps its
tensors full-length on every CP rank and slices the local part itself. That
slice must reproduce, bit for bit, the layout the rest of the model scatters
with — otherwise the labels/embeddings a rank holds belong to *other* ranks'
tokens and the loss is silently wrong. So the parity checks here are against
``context_parallel_utils``' own scatter helpers:

  ``dualchunk_allgather``  -> ``scatter_balance``    (two zigzag chunks)
  ``contiguous_allgather`` -> ``scatter_contiguous`` (one rank-order chunk)

Covered:
  1. ``extract_local_contiguous_chunk`` == the slice ``scatter_contiguous``
     computes, for every rank, and the ranks tile the sequence exactly once.
  2. ``extract_local_cp_chunks`` dispatch: dualchunk -> zigzag helper,
     contiguous_allgather -> contiguous helper, ``cp_size == 1`` -> identity.
  3. Unsupported modes (``contiguous_a2a``, typos) raise ValueError rather than
     silently defaulting to a layout.
  4. Divisibility guards.
  5. Negative ``axis`` and non-sequence axes behave like the zigzag helper.
"""

from __future__ import annotations

import unittest

import paddle

from paddlefleet.transformer.multi_token_prediction import (
    extract_local_contiguous_chunk,
    extract_local_cp_chunks,
    extract_local_zigzag_chunks,
)


def _arange_bl(batch: int, length: int) -> paddle.Tensor:
    """[B, L] with globally unique values so slices are traceable."""
    return paddle.arange(batch * length, dtype="int64").reshape([batch, length])


class TestExtractLocalContiguousChunk(unittest.TestCase):
    def test_matches_scatter_contiguous_layout(self) -> None:
        # scatter_contiguous: rank r gets [r*chunk, (r+1)*chunk).
        cp_size, length = 4, 16
        t = _arange_bl(2, length)
        chunk = length // cp_size
        for rank in range(cp_size):
            local = extract_local_contiguous_chunk(t, rank, cp_size, axis=1)
            expected = t[:, rank * chunk : (rank + 1) * chunk]
            self.assertEqual(local.shape, [2, chunk])
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

    def test_default_mode_is_dualchunk(self) -> None:
        # Callers that predate the mode argument must keep the old layout.
        t = _arange_bl(1, 16)
        self.assertTrue(
            bool(
                (
                    extract_local_cp_chunks(t, 1, 2, axis=1)
                    == extract_local_zigzag_chunks(t, 1, 2, axis=1)
                ).all()
            )
        )

    def test_cp_size_one_is_identity_for_any_mode(self) -> None:
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
