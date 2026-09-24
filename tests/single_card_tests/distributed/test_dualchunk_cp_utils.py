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

"""Behavior tests for the dual-chunk (zigzag) CP helpers in ``cp_utils``.

Production under test: ``paddlefleet.transformer.cp_utils``
    - ``dualchunk_chunk_ids``   pure integer chunk-assignment arithmetic
    - ``dualchunk_partner``     pure integer peer selection (an involution)
    - ``dualchunk_swap``        local routing of one pairwise second-half swap

The expected values here are derived by hand from the documented contract
(``2*cp_size`` equal chunks, rank ``r`` keeps chunk ``2r`` and takes chunk
``2*cp_size-1-2r`` from rank ``cp_size-1-r``) and never by calling the function
under test. The chunk-id and partner tables are written out as literals.

Scope note for ``dualchunk_swap``: the real NCCL exchange is a multi-card
behavior and is NOT exercised here. This file pins only the single-process
routing decisions that ``dualchunk_swap`` makes before it hands off to the
collective -- which peer (as a *global* rank), which half of the tensor is
sent, the submitted send/recv order, and that the kept half survives in the
result. The ``batch_isend_irecv``/``P2POp`` collaborators are stubbed for that
purpose; because nothing is actually received, the returned second half is
undefined and is deliberately not asserted on.
"""

import unittest

try:
    import paddle

    from paddlefleet.transformer.cp_utils import (
        dualchunk_chunk_ids,
        dualchunk_partner,
        dualchunk_swap,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle (a hard dependency of cp_utils) is absent
    _IMPORT_ERROR = exc

_SKIP_REASON = (
    "paddlefleet.transformer.cp_utils requires paddle, which is not installed "
    f"in this environment: {_IMPORT_ERROR}"
)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestDualChunkChunkIds(unittest.TestCase):
    """``dualchunk_chunk_ids`` assigns two of ``2*cp_size`` chunks per rank."""

    # Hand-computed (lo, hi) per rank, independent of the implementation.
    # cp_size=1: chunks {0,1};  cp_size=2: {0..3};  cp_size=4: {0..7}.
    EXPECTED = {
        1: [(0, 1)],
        2: [(0, 3), (2, 1)],
        4: [(0, 7), (2, 5), (4, 3), (6, 1)],
    }

    def test_matches_hand_derived_table(self):
        for cp_size, per_rank in self.EXPECTED.items():
            for rank, expected in enumerate(per_rank):
                with self.subTest(cp_size=cp_size, rank=rank):
                    self.assertEqual(
                        dualchunk_chunk_ids(rank, cp_size), expected
                    )

    def test_ids_form_a_partition_of_all_chunks(self):
        # Every chunk 0..2*cp_size-1 must be owned by exactly one rank; a
        # duplicated or dropped id would silently corrupt the causal indexer.
        for cp_size in (1, 2, 3, 4, 5, 8):
            with self.subTest(cp_size=cp_size):
                owned = []
                for rank in range(cp_size):
                    lo, hi = dualchunk_chunk_ids(rank, cp_size)
                    owned.extend((lo, hi))
                self.assertEqual(sorted(owned), list(range(2 * cp_size)))

    def test_id_sum_is_constant_across_ranks(self):
        # Constant id sum == constant candidate count == balanced work.
        for cp_size in (1, 2, 3, 4, 5, 8):
            for rank in range(cp_size):
                lo, hi = dualchunk_chunk_ids(rank, cp_size)
                with self.subTest(cp_size=cp_size, rank=rank):
                    self.assertEqual(lo + hi, 2 * cp_size - 1)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestDualChunkPartner(unittest.TestCase):
    """``dualchunk_partner`` picks the swap peer, or -1 when nothing moves."""

    # Hand-computed partner per rank; -1 means "no swap".
    EXPECTED = {
        1: [-1],
        2: [1, 0],
        3: [2, -1, 0],  # middle rank of an odd group is self-paired
        4: [3, 2, 1, 0],
        5: [4, 3, -1, 1, 0],
    }

    def test_matches_hand_derived_table(self):
        for cp_size, per_rank in self.EXPECTED.items():
            for rank, expected in enumerate(per_rank):
                with self.subTest(cp_size=cp_size, rank=rank):
                    self.assertEqual(dualchunk_partner(rank, cp_size), expected)

    def test_is_an_involution_on_even_groups(self):
        # partner(partner(r)) == r so one function serves both directions.
        for cp_size in (2, 4, 6, 16):
            for rank in range(cp_size):
                p = dualchunk_partner(rank, cp_size)
                with self.subTest(cp_size=cp_size, rank=rank):
                    self.assertNotEqual(p, -1)
                    self.assertEqual(dualchunk_partner(p, cp_size), rank)

    def test_partner_holds_the_wanted_chunk(self):
        # The chunk this rank takes (its ``hi``) must be the contiguous odd
        # chunk (2*partner + 1) that the partner starts out holding.
        for cp_size in (2, 4, 5, 8):
            for rank in range(cp_size):
                p = dualchunk_partner(rank, cp_size)
                if p < 0:
                    continue
                _, hi = dualchunk_chunk_ids(rank, cp_size)
                with self.subTest(cp_size=cp_size, rank=rank):
                    self.assertEqual(hi, 2 * p + 1)


def _group(nranks, rank):
    """A minimal CP process-group stand-in.

    ``ranks`` maps the group-local index to a *global* rank and is offset by
    100 so a routing bug that returns a group-local index instead of the
    global rank cannot pass unnoticed.
    """
    import types

    return types.SimpleNamespace(
        nranks=nranks, rank=rank, ranks=[100 + i for i in range(nranks)]
    )


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestDualChunkSwapNoOp(unittest.TestCase):
    """Degenerate groups return the *same* tensor, never a copy-with-comm."""

    def _sample(self):
        return paddle.arange(8).reshape([1, 4, 2]).astype("float32")

    def test_none_group_returns_input_identity(self):
        x = self._sample()
        self.assertIs(dualchunk_swap(x, None, axis=1), x)

    def test_single_rank_group_returns_input_identity(self):
        x = self._sample()
        self.assertIs(dualchunk_swap(x, _group(1, 0), axis=1), x)

    def test_self_paired_middle_rank_returns_input_identity(self):
        # Odd group's middle rank (nranks=3, rank=1) has partner == -1.
        x = self._sample()
        self.assertIs(dualchunk_swap(x, _group(3, 1), axis=1), x)


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestDualChunkSwapEvenExtent(unittest.TestCase):
    """A two-chunks-per-rank swap needs an even extent on the swapped axis."""

    def test_odd_extent_raises_value_error(self):
        x = paddle.zeros([1, 5, 2], dtype="float32")
        # nranks=4, rank=1 -> partner 2, a real swap, so the check is reached.
        with self.assertRaisesRegex(ValueError, "even extent"):
            dualchunk_swap(x, _group(4, 1), axis=1)

    def test_even_extent_does_not_raise_for_extent_check(self):
        # Guard against the check firing on a valid even extent. The collective
        # is stubbed; we only care that no ValueError about extent is raised.
        from unittest import mock

        x = paddle.arange(16).reshape([1, 8, 2]).astype("float32")
        with (
            mock.patch(
                "paddle.distributed.batch_isend_irecv",
                return_value=[mock.MagicMock()],
            ),
            mock.patch("paddle.distributed.P2POp", return_value=object()),
        ):
            dualchunk_swap(x, _group(4, 1), axis=1)  # must not raise


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestDualChunkSwapRouting(unittest.TestCase):
    """Single-process routing contract of ``dualchunk_swap``.

    Real NCCL transfer is a multi-card behavior and is not run here; the
    collective is stubbed so the returned second half is undefined. What is
    pinned is entirely local: the peer's global rank, that the *second* half
    travels, the send/recv submission order, and that the first half is kept.
    """

    def _run_swap(self, x, group, axis=1):
        from unittest import mock

        captured = []

        def _p2p(op, tensor, peer, grp):
            # Record the routing decision; return a tagged sentinel so the
            # submitted op-list order can be inspected.
            captured.append((op, tensor, peer, grp))
            return (op, tensor, peer)

        with (
            mock.patch("paddle.distributed.P2POp", side_effect=_p2p),
            mock.patch(
                "paddle.distributed.batch_isend_irecv",
                return_value=[mock.MagicMock()],
            ) as batched,
        ):
            out = dualchunk_swap(x, group, axis=axis)

        return captured, batched, out

    def test_peer_is_partners_global_rank(self):
        cp_size = 4
        x = paddle.arange(16).reshape([1, 8, 2]).astype("float32")
        for rank in (1, 2):
            with self.subTest(rank=rank):
                group = _group(cp_size, rank)
                captured, batched, _ = self._run_swap(x, group)

                batched.assert_called_once()
                self.assertEqual(len(captured), 2)  # one send, one recv

                partner = dualchunk_partner(rank, cp_size)
                # Global rank == 100 + partner index (see _group). This fails
                # if the code sends to the group-local index instead.
                self.assertEqual(partner, cp_size - 1 - rank)
                peers = {c[2] for c in captured}
                self.assertEqual(peers, {100 + partner})

    def test_second_half_is_sent_and_first_half_kept(self):
        cp_size = 4
        x = paddle.arange(16).reshape([1, 8, 2]).astype("float32")
        group = _group(cp_size, 1)
        captured, _, out = self._run_swap(x, group)

        sent = [t for op, t, _, _ in captured if op is paddle.distributed.isend]
        self.assertEqual(len(sent), 1)
        # The odd (second) half x[:, 4:] is the buffer that travels.
        self.assertEqual(
            float((sent[0] - x[:, 4:]).abs().max()),
            0.0,
        )
        # The kept (first) half is preserved verbatim in the result; the
        # second half is undefined because the recv was stubbed.
        self.assertEqual(list(out.shape), list(x.shape))
        self.assertEqual(
            float((out[:, :4] - x[:, :4]).abs().max()),
            0.0,
        )

    def test_lower_rank_submits_send_first(self):
        cp_size = 4
        x = paddle.arange(16).reshape([1, 8, 2]).astype("float32")
        for rank in (1, 2):
            with self.subTest(rank=rank):
                group = _group(cp_size, rank)
                _, batched, _ = self._run_swap(x, group)

                (ops,) = batched.call_args[0]
                self.assertEqual(len(ops), 2)
                partner = dualchunk_partner(rank, cp_size)
                first_is_send = ops[0][0] is paddle.distributed.isend
                # Send-first exactly when this rank is the lower of the pair.
                self.assertEqual(first_is_send, rank < partner)

    def test_swap_honours_the_requested_axis(self):
        # Split happens along ``axis``; kept half of axis=2 must survive.
        cp_size = 4
        x = paddle.arange(24).reshape([1, 3, 4, 2]).astype("float32")
        group = _group(cp_size, 1)
        captured, _, out = self._run_swap(x, group, axis=2)

        sent = [t for op, t, _, _ in captured if op is paddle.distributed.isend]
        self.assertEqual(len(sent), 1)
        self.assertEqual(
            float((sent[0] - x[:, :, 2:, :]).abs().max()),
            0.0,
        )
        self.assertEqual(list(out.shape), list(x.shape))
        self.assertEqual(
            float((out[:, :, :2, :] - x[:, :, :2, :]).abs().max()),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
