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

"""Single-card tests for the dual-chunk CP row helpers in ``cp_utils``.

``tests/multi_card_tests/transformer/test_mqa_indexer_dualchunk_cp.py`` covers
the same helpers against real NCCL. This file pins the parts that are pure
arithmetic or pure routing, so the chunk assignment and the peer/op layout are
checked on every single-card run and a regression there does not need two GPUs
to surface.

The ``MQALatentAttention`` row-plumbing helpers are covered here too, with the
cuDNN indexer stubbed. The CP test exercises them for real but is gated on SM100
kernels, so on every other box the two ``seq_offset`` values, the ``valid_range``
slicing and the swap-back would otherwise go unchecked.
"""

import types
import unittest
from unittest import mock

import paddle

from paddlefleet.transformer.cp_utils import (
    dualchunk_chunk_ids,
    dualchunk_partner,
    dualchunk_swap,
)


def _make_mock_group(nranks=4, rank=1):
    """Mock process group; ``ranks`` maps group-local index to global rank."""
    group = mock.MagicMock()
    group.nranks = nranks
    group.rank = rank
    # Offset so a test cannot pass by confusing the two rank spaces.
    group.ranks = [100 + i for i in range(nranks)]
    return group


class TestDualChunkChunkIds(unittest.TestCase):
    def test_ids_partition_and_balance(self):
        """Every chunk has one owner and the per-rank id sum is constant."""
        for cp_size in (1, 2, 4, 8, 16):
            with self.subTest(cp_size=cp_size):
                owner = {}
                for rank in range(cp_size):
                    lo, hi = dualchunk_chunk_ids(rank, cp_size)
                    # Constant sum is the whole point: a causal row's candidate
                    # count grows linearly with its global position, so equal
                    # id sums mean equal work.
                    self.assertEqual(lo + hi, 2 * cp_size - 1)
                    # The kept chunk is the one contiguous CP already placed
                    # here, which is what makes the exchange one sendrecv.
                    self.assertEqual(lo, 2 * rank)
                    for c in (lo, hi):
                        self.assertNotIn(c, owner)
                        owner[c] = rank
                self.assertEqual(sorted(owner), list(range(2 * cp_size)))


class TestDualChunkPartner(unittest.TestCase):
    def test_involution_on_even_groups(self):
        """``partner(partner(r)) == r``, so one function serves both ways."""
        for cp_size in (2, 4, 16):
            for rank in range(cp_size):
                p = dualchunk_partner(rank, cp_size)
                self.assertEqual(dualchunk_partner(p, cp_size), rank)
                self.assertEqual(p, cp_size - 1 - rank)

    def test_no_partner_cases(self):
        """``-1`` means 'nothing to swap', not 'rank 0'."""
        self.assertEqual(dualchunk_partner(0, 1), -1)
        # Odd group: the middle rank is its own partner, so it keeps both
        # chunks rather than sending to itself.
        self.assertEqual(dualchunk_partner(1, 3), -1)
        self.assertEqual(dualchunk_partner(0, 3), 2)


class TestDualChunkSwap(unittest.TestCase):
    def test_no_op_without_a_peer(self):
        """Degenerate groups return the input untouched, not a copy-with-comm."""
        x = paddle.arange(8).reshape([1, 4, 2]).cast("float32")
        self.assertIs(dualchunk_swap(x, None, axis=1), x)
        self.assertIs(dualchunk_swap(x, _make_mock_group(1, 0), axis=1), x)
        # Odd group's middle rank has no partner.
        self.assertIs(dualchunk_swap(x, _make_mock_group(3, 1), axis=1), x)

    def test_odd_extent_rejected(self):
        """Two chunks per rank needs an even count on the swapped axis."""
        x = paddle.zeros([1, 5, 2])
        with self.assertRaisesRegex(ValueError, "even extent"):
            dualchunk_swap(x, _make_mock_group(4, 1), axis=1)

    def test_routing_and_op_order(self):
        """The peer, the halves and the isend/irecv order, without NCCL.

        ``batch_isend_irecv`` is stubbed: this pins the routing contract (send
        the odd chunk to ``cp_size-1-rank`` as a *global* rank, keep the even
        one, and order the ops by rank) which is what a wrong peer or a
        deadlock-prone ordering would break.
        """
        cp_size = 4
        x = paddle.arange(16).reshape([1, 8, 2]).cast("float32")
        for rank in (1, 2):
            with self.subTest(rank=rank):
                group = _make_mock_group(cp_size, rank)
                captured = []

                def _p2p_op(op, tensor, peer, grp):
                    captured.append((op, tensor, peer, grp))
                    return (op, tensor, peer)

                with (
                    mock.patch("paddle.distributed.P2POp", side_effect=_p2p_op),
                    mock.patch(
                        "paddle.distributed.batch_isend_irecv",
                        return_value=[mock.MagicMock()],
                    ) as batched,
                ):
                    out = dualchunk_swap(x, group, axis=1)

                batched.assert_called_once()
                (ops,) = batched.call_args[0]
                self.assertEqual(len(ops), 2)
                self.assertEqual(len(captured), 2)

                partner = dualchunk_partner(rank, cp_size)
                peers = {c[2] for c in captured}
                self.assertEqual(
                    peers,
                    {group.ranks[partner]},
                    "peer must be the partner's *global* rank",
                )
                # Lower rank sends first: harmless for NCCL's grouped p2p, but
                # it removes a hang mode if the ops ever degrade to blocking.
                # The submitted *list* order is what matters, not the order the
                # two P2POp objects happened to be constructed in.
                self.assertEqual(
                    ops[0][0] is paddle.distributed.isend,
                    rank < partner,
                    "ops must be submitted send-first only on the lower rank",
                )

                # The sent buffer is the second half; the first half survives
                # untouched in the result.
                sent = next(
                    t
                    for op, t, _, _ in captured
                    if op is paddle.distributed.isend
                )
                self.assertEqual(
                    float((sent - x[:, 4:]).abs().max()),
                    0.0,
                    "the odd chunk is the one that travels",
                )
                self.assertEqual(out.shape, x.shape)
                self.assertEqual(
                    float((out[:, :4] - x[:, :4]).abs().max()),
                    0.0,
                    "the kept chunk must not move",
                )


def _half_flip(x, group, axis=1):
    """Single-rank stand-in for ``dualchunk_swap``.

    An involution that leaves the first half alone and visibly reorders the
    second, which is the only property the row plumbing relies on. A real swap
    needs a peer rank; this one makes the swap-out/swap-back symmetry observable
    on one card.
    """
    m = int(x.shape[axis]) // 2
    keep, give = paddle.split(x, 2, axis=axis)
    return paddle.concat([keep, paddle.flip(give, axis)], axis=axis)


class TestDualChunkValidRange(unittest.TestCase):
    """``MQALatentAttention._dualchunk_valid_range`` reads two chunk offsets."""

    def test_reads_the_two_chunk_offsets_in_order(self):
        """Rows come from ``lo`` then ``hi``, each a length-``m`` global slice.

        The two ``(offset, length)`` pairs are what must line up with the two
        ``seq_offset`` values the kernel calls get; asking with the local offset,
        or in the wrong order, is silently wrong rather than a crash.
        """
        from paddlefleet.transformer.mqa_latent_attention import (
            MQALatentAttention,
        )

        asked = []

        def _chunk(
            meta, s_global, doc_start, doc_len, is_valid, offset, length
        ):
            asked.append((offset, length))
            return paddle.full([1, length, 2], float(offset))

        fake = types.SimpleNamespace(
            cp_rank=1, cp_size=4, _chunk_valid_range=_chunk
        )
        out = MQALatentAttention._dualchunk_valid_range(
            fake, "meta", 32, "doc_start", "doc_len", "is_valid", 8
        )

        # cp_rank=1 of 4 owns global chunks (2, 5); m = s // 2 = 4.
        self.assertEqual(asked, [(8, 4), (20, 4)])
        self.assertEqual(out.shape, [1, 8, 2])
        self.assertEqual(float(out[0, 0, 0]), 8.0)
        self.assertEqual(float(out[0, 4, 0]), 20.0)


class TestIndexerTopkDualChunk(unittest.TestCase):
    """``MQALatentAttention._indexer_topk_dualchunk`` with the kernel stubbed.

    The real path is covered by ``test_mqa_indexer_dualchunk_cp.py``, but its
    value tests need SM100 kernels. What is checked here is the plumbing that
    would be wrong on any box: two calls with the two global ``seq_offset``
    values, each fed its own half of the *already dual-chunk-ordered*
    ``valid_range``, and the results swapped back to contiguous rows.
    """

    CP_RANK, CP_SIZE, S, TOPK = 1, 4, 8, 4

    def _run(self, need_loss):
        from paddlefleet.transformer.mqa_latent_attention import (
            MQALatentAttention,
        )

        calls = []

        def _kernel(q, k, w, **kw):
            calls.append((q, w, kw))
            m = int(q.shape[1])
            # Rows tagged by their global position so the concat-then-swap-back
            # is checkable rather than symmetric-by-accident.
            rows = (
                kw["seq_offset"] + paddle.arange(m).cast("float32")
            ).reshape([1, m, 1])
            out = paddle.expand(rows, [1, m, self.TOPK])
            return (out, None, out) if kw["return_topk_scores"] else (out, None)

        q = paddle.arange(self.S).reshape([1, self.S, 1, 1]).cast("float32")
        w = paddle.arange(self.S).reshape([1, self.S, 1]).cast("float32")
        vr = paddle.arange(2 * self.S).reshape([1, self.S, 2]).cast("float32")
        fake = types.SimpleNamespace(
            cp_rank=self.CP_RANK, cp_size=self.CP_SIZE, cp_group="grp"
        )

        with (
            mock.patch(
                "paddlefleet.transformer.mqa_latent_attention.dualchunk_swap",
                side_effect=_half_flip,
            ) as swap,
            mock.patch(
                "paddlefleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn"
                ".cudnn_indexer_topk_fwd",
                side_effect=_kernel,
            ),
        ):
            selected, scores_out = MQALatentAttention._indexer_topk_dualchunk(
                fake,
                q,
                w,
                paddle.zeros([1, self.S, 1]),
                self.TOPK,
                "doc_lens",
                vr,
                need_loss,
            )
        return calls, swap, selected, scores_out, q, w, vr

    def test_two_calls_carry_the_chunk_offsets_and_their_own_rows(self):
        calls, swap, selected, scores_out, q, w, vr = self._run(False)

        self.assertEqual(len(calls), 2)
        # chunk ids (2, 5) with m = 4.
        self.assertEqual([c[2]["seq_offset"] for c in calls], [8, 20])

        q_zz, w_zz = _half_flip(q, "grp"), _half_flip(w, "grp")
        for i, sl in enumerate((slice(0, 4), slice(4, 8))):
            q_seen, w_seen, kw = calls[i]
            self.assertEqual(float((q_seen - q_zz[:, sl]).abs().max()), 0.0)
            self.assertEqual(float((w_seen - w_zz[:, sl]).abs().max()), 0.0)
            # ``vr_zz`` is built in dual-chunk order already, so it is sliced,
            # never swapped -- swapping it here would double-permute the rows.
            self.assertEqual(
                float((kw["valid_range"] - vr[:, sl]).abs().max()),
                0.0,
                "valid_range must be sliced, not swapped again",
            )
            self.assertFalse(kw["return_topk_scores"])

        # Swapped back: rows 8..11 stay, 20..23 come back reversed by the stub.
        self.assertEqual(
            [float(v) for v in selected[0, :, 0]],
            [8, 9, 10, 11, 23, 22, 21, 20],
        )
        # q, w out; selected back. Nothing else travels without the loss.
        self.assertEqual(swap.call_count, 3)
        self.assertEqual(scores_out, [])

    def test_scores_are_swapped_back_only_when_the_loss_needs_them(self):
        calls, swap, selected, scores_out, *_ = self._run(True)

        self.assertTrue(all(c[2]["return_topk_scores"] for c in calls))
        self.assertEqual(swap.call_count, 4)
        (scores,) = scores_out
        # Same layout as ``selected``: the KL sees unpermuted contiguous rows.
        self.assertEqual(
            [float(v) for v in scores[0, :, 0]],
            [8, 9, 10, 11, 23, 22, 21, 20],
        )


class TestChunkValidRange(unittest.TestCase):
    """``MQALatentAttention._chunk_valid_range`` picks one of two row sources.

    Both expose the same ``(offset, length)`` slice of one global table, which is
    what lets the dual-chunk layout ask for its two segments by chunk offset. The
    delegation is what can silently break -- passing the caller's ``window``
    instead of ``self.window_size``, or forgetting to unwrap the
    ``(valid_range, row_empty)`` pair -- so pin both paths here rather than only
    exercising the eager one from the CP test.
    """

    @staticmethod
    def _call(fake_self, meta, offset, length):
        from paddlefleet.transformer.mqa_latent_attention import (
            MQALatentAttention,
        )

        return MQALatentAttention._chunk_valid_range(
            fake_self,
            meta,
            512,
            "doc_start",
            "doc_len",
            "is_valid",
            offset,
            length,
        )

    def test_uses_meta_when_present(self):
        meta = mock.MagicMock()
        meta.indexer_valid_range.return_value = ("vr", "row_empty")
        fake = types.SimpleNamespace(window_size=128)
        self.assertEqual(self._call(fake, meta, 64, 32), "vr")
        meta.indexer_valid_range.assert_called_once_with(128, 64, 32)

    def test_falls_back_to_eager_build(self):
        eager = mock.MagicMock(return_value=("vr2", "row_empty"))
        fake = types.SimpleNamespace(
            window_size=128, _indexer_valid_range=eager
        )
        self.assertEqual(self._call(fake, None, 64, 32), "vr2")
        eager.assert_called_once_with(
            512, "doc_start", "doc_len", "is_valid", 64, 32
        )


if __name__ == "__main__":
    unittest.main()
