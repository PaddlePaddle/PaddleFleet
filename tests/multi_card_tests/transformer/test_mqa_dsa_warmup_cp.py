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

"""Context parallel for the DSA **warmup** phase, and for padded layouts.

Two gaps in ``test_mqa_dsa_cp.py`` / ``test_mla_cp_contiguous_allgather.py``:

1. ``hybrid_mla_attention="mqa_dsa"`` with ``dsa_indexer_use_sparse_loss=False``
   -- the phase-2 (DSA warmup) pairing, where attention consumes the full
   per-document causal set while the indexer is still being learned on the
   widened KL table. The existing CP suites run ``True`` everywhere except the
   one ``(masked=True, sparse=False)`` subtest of
   ``test_mqa_dsa_cp.py::test_7``, which only observes parameter gradients. The
   warmup mode takes a *different branch* of ``_forward_dsa``
   (``mqa_latent_attention.py:495`` and ``:594``) whose index table is built at
   ``s_global`` and row-sliced, so it needs its own CP evidence: that the
   attention output is the CP=1 reference, that the table really is the global
   one sliced, that the mode is bit-identical to ``"mqa_full_causal"`` under CP,
   and that the widened loss still normalises across the CP group on both the
   masked and the unmasked branch (``_indexer_loss_mask`` /
   ``loss_coeff / cp_size``).

2. A layout with genuine **row-validity pad rows**. ``_STRADDLE`` sums to
   exactly ``S_GLOBAL`` and ``U._row_end`` folds any trailing gap into one final
   document, so ``is_valid`` has been all-``True`` in every CP test so far; the
   pad-row path (all-``-1`` index row -> zero output, zero ``dq``) was only ever
   audited on one card. ``[475] @ s=512`` puts 37 pad rows on the last rank
   only, which is also the pad-imbalance the loss denominator has to survive.

Everything reuses ``test_mqa_dsa_cp``'s harness (fleet init, CP globals,
``run_core_cp``, ``_check``, ``_check_index_sets``). No ``if rank == X``
short-circuit exists in this file: every collective (``run_core_cp``'s
all-reduces, the all-gather inside the layer) is issued on all ranks and only
the assertions are rank-conditional.

Run (2 or 4 GPUs)::

    PYTHONPATH=./third_party/PaddleFleet/src:./third_party/PaddleFormers \
    python -m paddle.distributed.launch --devices 0,1 --nnodes 1 \
        --master 127.0.0.1:<port> \
        third_party/PaddleFleet/tests/multi_card_tests/transformer/\
test_mqa_dsa_warmup_cp.py
"""

import unittest

import paddle
import paddle.distributed as dist

# ``paddle.distributed.launch <thisfile>`` puts this directory on ``sys.path``,
# so the sibling harness imports as a top-level module (same as
# ``test_mla_cp_recompute`` importing ``test_mla_cp_contiguous_allgather``).
import test_mqa_dsa_cp as H

from paddlefleet.transformer.csa_attention import _derive_csa_doc_boundaries
from paddlefleet.transformer.mqa_latent_attention import MQALatentAttention

S_GLOBAL = H.S_GLOBAL
_STRADDLE = H._STRADDLE

# One document covering [0, 475) of a 512-long batch: rows 475..511 are pad
# rows. 37 is deliberately not a multiple of 512/cp_size, so the pad rows land
# on the last rank only for both CP=2 (256) and CP=4 (128). It also matches
# ``H._input_ids``' own ``n_pad=37``, so the loss row mask and the document
# validity agree.
_PAD_DOC_LEN = 475
_N_PAD = S_GLOBAL - _PAD_DOC_LEN

# The logged indexer loss is a bf16-fed fp32 KL reduction, so the per-rank sum
# lands a few 1e-4 off the single-rank value. A wrong denominator is off by
# ``cp_size`` (100% at CP=2), so this bound is three orders of magnitude away
# from the failure it has to catch.
LOSS_RTOL = 5e-3


def setUpModule():
    H.setUpModule()


def _pad_row_end(doc_len, s_global):
    """``[1, 1, s_global, 1]`` int32 mask whose tail rows are genuine pad rows.

    ``U._row_end`` cannot express this: it closes the trailing gap with one more
    document, which makes every row valid. Pointing the tail back at the
    previous document's end instead leaves ``pos_in_doc >= doc_len_per_pos``,
    which is exactly ``_derive_csa_doc_boundaries``' ``is_valid`` test
    (``csa_attention.py:136-138``).
    """
    return paddle.full([1, 1, s_global, 1], doc_len, dtype="int32")


def _doc_bounds(row_end, s_global):
    """``(doc_start, is_valid)`` as int lists, over the global sequence."""
    doc_start, _, is_valid, _, _ = _derive_csa_doc_boundaries(row_end, s_global)
    return doc_start.tolist(), [bool(v) for v in is_valid.tolist()]


def _maxabs(a, b):
    return float((a.cast("float32") - b.cast("float32")).abs().max())


def _local_slice():
    """``(row_offset, rows)`` of this CP rank's query rows."""
    rows = S_GLOBAL // H.CP_SIZE
    return H.CP_RANK * rows, rows


class _CPChecks(unittest.TestCase):
    """Shared assertions, borrowed from the harness class.

    ``_check`` (forward + every gradient against the CP=1 reference) and
    ``_check_index_sets`` (the sparse kernel is handed the reference's own
    columns) are the same contract here, so call the harness' own
    implementations rather than restating them.
    """

    def _check(self, res, tag):
        H.TestMQADSACP._check(self, res, tag)

    def _check_index_sets(self, res, tag):
        H.TestMQADSACP._check_index_sets(self, res, tag)

    def _assert_full_causal(self, idx, row_end, tag):
        """Every local row's selected set == its whole per-document causal set.

        This is what separates the warmup mode from phase 3/4: under
        ``window + top-k`` a row longer than ``window + index_topk`` selects a
        strict subset, so this assertion would fail there.
        """
        off, rows = _local_slice()
        doc_start, is_valid = _doc_bounds(row_end, S_GLOBAL)
        for r in range(rows):
            q = off + r
            got = {int(c) for c in idx[0][r] if c >= 0}
            want = set(range(doc_start[q], q + 1)) if is_valid[q] else set()
            self.assertEqual(
                got, want, f"{tag}: row {r} (global {q}) is not full-causal"
            )


class TestWarmupCP(_CPChecks):
    """``mqa_dsa`` + ``dsa_indexer_use_sparse_loss=False`` under CP."""

    @H.U._GPU
    def test_1_warmup_forward_equivalence(self):
        """CP=N == CP=1 on the warmup path, with and without a live loss.

        ``dsa_indexer_loss_coeff == 0`` takes the early branch that skips the
        indexer projections entirely (``mqa_latent_attention.py:495-507``);
        ``> 0`` takes the branch that still builds the wide top-k table for the
        KL but hands attention the full causal one (``:594-603``). Both must
        land on the same reference output, so both are checked.
        """
        for coeff in (0.0, 0.1):
            with self.subTest(loss_coeff=coeff):
                tag = f"warmup/coeff={coeff}"
                res = H.run_core_cp(
                    "mqa_dsa",
                    _STRADDLE,
                    loss_coeff=coeff,
                    with_input_ids=coeff > 0,
                    sparse_loss=False,
                )
                self._check(res, tag)
                self._check_index_sets(res, tag)
                self._assert_full_causal(res["idx_cp"], res["row_end"], tag)

    @H.U._GPU
    def test_2_warmup_token_indices_are_the_global_table_row_sliced(self):
        """The kernel's table == the global build, sliced -- bitwise.

        A per-rank build (``s_local`` rows, ``doc_start`` sliced first) clips
        each row at ``q - position_offset`` and drops the prefix owned by lower
        ranks. The control at the end constructs exactly that and asserts it
        differs, so this comparison cannot be vacuous on rank > 0.
        """
        off, rows = _local_slice()
        res = H.run_core_cp("mqa_dsa", _STRADDLE, sparse_loss=False)
        row_end = res["row_end"]
        doc_start, _, is_valid, _, _ = _derive_csa_doc_boundaries(
            row_end, S_GLOBAL
        )
        want = MQALatentAttention._build_full_causal_indices(
            1, S_GLOBAL, doc_start, is_valid
        )[:, off : off + rows]
        got = paddle.to_tensor(res["idx_cp"])
        self.assertEqual(list(got.shape), list(want.shape), "table shape")
        drift = int((got.cast("int64") != want.cast("int64")).sum())
        self.assertEqual(
            drift,
            0,
            f"{drift} of {int(got.numel())} slots differ from the global table",
        )

        # Control: the same builder driven at the local length, with the
        # document starts rebased (and clipped) into local coordinates -- the
        # shape a CP-unaware implementation would produce.
        local = MQALatentAttention._build_full_causal_indices(
            1,
            rows,
            paddle.clip(doc_start[off : off + rows] - off, min=0),
            is_valid[off : off + rows],
        )
        print(
            f"[warmup-cp{H.CP_SIZE} rank{H.CP_RANK}] token_indices drift vs "
            f"global table = {drift}",
            flush=True,
        )
        if H.CP_RANK == 0:
            return
        self.assertGreater(
            int((local.cast("int64") != want[:, :, :rows].cast("int64")).sum()),
            0,
            "a per-rank index build was indistinguishable from the global "
            "one, so this test is vacuous",
        )

    @H.U._GPU
    def test_3_warmup_equals_mqa_full_causal_under_cp(self):
        """Warmup output == ``hybrid_mla_attention="mqa_full_causal"`` output.

        Both modes call ``_build_full_causal_indices`` and then the same sparse
        kernel with the same inputs, so on each rank this must be *bitwise*
        equal, not merely close. The single-card claim (maxabs 0.0) is asserted
        here with the CP row-slicing in the path -- the two modes slice the
        table at different call sites (``forward`` vs ``_forward_dsa``), which
        is exactly what could drift apart.
        """
        ref = H.run_core_cp("mqa", _STRADDLE)
        for coeff in (0.0, 0.1):
            with self.subTest(loss_coeff=coeff):
                warm = H.run_core_cp(
                    "mqa_dsa",
                    _STRADDLE,
                    loss_coeff=coeff,
                    with_input_ids=coeff > 0,
                    sparse_loss=False,
                )
                delta = _maxabs(warm["out"], ref["out"])
                print(
                    f"[warmup-cp{H.CP_SIZE} rank{H.CP_RANK}] warmup vs "
                    f"mqa_full_causal (coeff={coeff}): maxabs={delta:.3e}",
                    flush=True,
                )
                self.assertEqual(
                    delta,
                    0.0,
                    f"warmup output is not bit-identical to mqa_full_causal "
                    f"(coeff={coeff}, maxabs={delta:.3e})",
                )

    @H.U._GPU
    def test_4_indexer_loss_cp_normalisation(self):
        """The logged indexer loss must sum to the CP=1 value across the group.

        Read straight out of ``DSAIndexerLossLoggingHelper``, so it observes the
        denominator itself rather than its shadow in the gradients:
        ``loss_mask is not None`` -> the global valid-row count (per-rank losses
        are partial sums), ``None`` -> a local mean scaled by ``1/cp_size``
        (``mqa_latent_attention.py:660-675``). ``sparse_loss`` is swept so that
        a failure can be attributed: warmup-only means the widened table broke
        it, both means the pre-existing normalisation is wrong.

        Note this layout does not exercise the *width* difference between the
        two ``sparse_loss`` values: with documents of 200/150/162 and a 128-wide
        forced window, no row has more than 72 non-local candidates, so the
        128-slot and the 512-slot table hold the same set (the rest is ``-1``).
        ``test_5`` covers the width itself.
        """
        for sparse in (False, True):
            for masked in (True, False):
                with self.subTest(sparse_loss=sparse, masked=masked):
                    tag = f"loss/sparse={sparse}/masked={masked}"
                    res = H.run_core_cp(
                        "mqa_dsa",
                        _STRADDLE,
                        loss_coeff=0.1,
                        with_input_ids=masked,
                        sparse_loss=sparse,
                    )
                    self.assertTrue(
                        any(n.startswith("indexer.") for n in res["param_err"]),
                        f"{tag}: the indexer received no gradient",
                    )
                    self._check(res, tag)
                    self._assert_loss_cp_sum(res, tag)

    @H.U._GPU
    def test_5_widened_warmup_loss_table_under_cp(self):
        """One 512-long document, so the widened KL table is genuinely wider.

        ``_indexer_valid_range`` leaves ``causal_len - window_size`` non-local
        candidates, which only exceeds ``index_topk`` (128 in this fixture) once
        a document is longer than 256. On this layout the warmup table really
        holds ``min(2048, s_global)`` = 512 slots against phase 3/4's 128, so
        the two logged losses must differ -- that is the non-vacuity control for
        ``test_4`` -- while each still normalises across the CP group.
        """
        logged = {}
        for sparse in (False, True):
            with self.subTest(sparse_loss=sparse):
                tag = f"wide/sparse={sparse}"
                res = H.run_core_cp(
                    "mqa_dsa",
                    [S_GLOBAL],
                    loss_coeff=0.1,
                    with_input_ids=True,
                    sparse_loss=sparse,
                )
                self._check(res, tag)
                logged[sparse] = self._assert_loss_cp_sum(res, tag)
        spread = abs(logged[False] - logged[True]) / abs(logged[True])
        print(
            f"[warmup-cp{H.CP_SIZE} rank{H.CP_RANK}] wide vs narrow KL: "
            f"{logged[False]:.6e} vs {logged[True]:.6e} rel={spread:.3e}",
            flush=True,
        )
        self.assertGreater(
            spread,
            1e-3,
            "the widened warmup KL table produced the same loss as the narrow "
            f"one ({logged[False]:.6e} vs {logged[True]:.6e}), so "
            "dsa_indexer_use_sparse_loss is not changing the table here and "
            "test_4 is width-blind",
        )

    def _assert_loss_cp_sum(self, res, tag):
        """Sum the per-rank logged loss and compare to the CP=1 value."""
        total = paddle.to_tensor([res["logged_cp"]], dtype="float64")
        dist.all_reduce(total, group=H.CP_GROUP)
        got, want = float(total[0]), res["logged_ref"]
        self.assertGreater(abs(want), 0.0, f"{tag}: reference logged no loss")
        rel = abs(got - want) / abs(want)
        print(
            f"[warmup-cp{H.CP_SIZE} rank{H.CP_RANK}] {tag}: "
            f"sum(loss)={got:.6e} ref={want:.6e} rel={rel:.3e}",
            flush=True,
        )
        self.assertLess(
            rel,
            LOSS_RTOL,
            f"{tag}: CP loss sum {got:.6e} != CP=1 {want:.6e} "
            f"(rel {rel:.3e}); a per-rank denominator would be off by "
            f"~{H.CP_SIZE}x",
        )
        return want


class TestPadRowsCP(_CPChecks):
    """``[475] @ s=512``: 37 real pad rows, all on the last CP rank.

    Every CP suite so far used layouts whose documents tile the sequence, so
    ``is_valid`` was all-``True`` and the pad-row path -- an all-``-1`` index
    row, which the kernel must turn into a zero output row with a zero ``dq``
    -- was never reached under CP. The rows also sit entirely on the last rank,
    which is the pad imbalance a per-rank loss denominator cannot survive.
    """

    def _run(self, mode, sparse_loss):
        row_end = _pad_row_end(_PAD_DOC_LEN, S_GLOBAL)
        return H.run_core_cp(
            mode,
            None,
            loss_coeff=0.1,
            with_input_ids=True,
            sparse_loss=sparse_loss,
            row_end=row_end,
        )

    def _check_pad(self, res, tag):
        off, rows = _local_slice()
        _, is_valid = _doc_bounds(res["row_end"], S_GLOBAL)
        local_pad = [r for r in range(rows) if not is_valid[off + r]]

        # The layout must actually produce pad rows *somewhere*: assert it on
        # the group, not on this rank, since only the last rank owns them.
        n_pad = paddle.to_tensor([len(local_pad)], dtype="int64")
        dist.all_reduce(n_pad, group=H.CP_GROUP)
        self.assertEqual(
            int(n_pad[0]),
            _N_PAD,
            f"{tag}: the layout produced {int(n_pad[0])} pad rows, expected "
            f"{_N_PAD} -- the fixture no longer tests what it claims",
        )

        out = res["out"].cast("float32")
        dq = res["dq_local"].cast("float32")
        for r in local_pad:
            self.assertEqual(
                float(out[0, r].abs().max()),
                0.0,
                f"{tag}: pad row {r} (global {off + r}) has a non-zero output",
            )
            self.assertEqual(
                float(dq[0, r].abs().max()),
                0.0,
                f"{tag}: pad row {r} (global {off + r}) has a non-zero dq",
            )
            cols = res["idx_cp"][0][r]
            self.assertEqual(
                int((cols >= 0).sum()),
                0,
                f"{tag}: pad row {r} (global {off + r}) selected columns",
            )
        print(
            f"[padrows-cp{H.CP_SIZE} rank{H.CP_RANK}] {tag}: "
            f"local_pad_rows={len(local_pad)} fwd={res['fwd']:.2e} "
            f"per_pos_max={max(res['per_pos']):.3e}",
            flush=True,
        )

    @H.U._GPU
    def test_1_pad_rows_mqa_full_causal(self):
        res = self._run("mqa", True)
        self._check(res, "pad/mqa_full_causal")
        self._check_index_sets(res, "pad/mqa_full_causal")
        self._check_pad(res, "pad/mqa_full_causal")

    @H.U._GPU
    def test_2_pad_rows_warmup(self):
        res = self._run("mqa_dsa", False)
        self._check(res, "pad/warmup")
        self._check_index_sets(res, "pad/warmup")
        self._assert_full_causal(res["idx_cp"], res["row_end"], "pad/warmup")
        self._check_pad(res, "pad/warmup")

    @H.U._GPU
    def test_3_pad_rows_sparse(self):
        """Same layout on the phase-3/4 (``window + top-k``) path.

        Included so a pad-row failure can be attributed: if it reproduces here
        it is not specific to the warmup branch this change introduced.
        """
        res = self._run("mqa_dsa", True)
        self._check(res, "pad/sparse")
        self._check_index_sets(res, "pad/sparse")
        self._check_pad(res, "pad/sparse")


if __name__ == "__main__":
    unittest.main(verbosity=2)
