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

Three gaps in ``test_mqa_dsa_cp.py`` / ``test_mla_cp_contiguous_allgather.py``:

1. ``hybrid_mla_attention="mqa_dsa"`` with ``dsa_indexer_use_sparse_loss=False``
   -- the phase-2 (DSA warmup) pairing, where attention consumes the full
   per-document causal set while the indexer is trained by a KL over that same
   full causal set (no top-k anywhere). The existing CP suites run ``True``
   everywhere except the one ``(masked=True, sparse=False)`` subtest of
   ``test_mqa_dsa_cp.py::test_7``, which only observes parameter gradients. The
   warmup mode takes its own branch (``mqa_latent_attention._forward_warmup``)
   whose index table and causal mask are both built at ``s_global`` and
   row-sliced, so it needs its own CP evidence: that the attention output is the
   CP=1 reference, that the table really is the global one sliced, that the mode
   is bit-identical to ``"mqa_full_causal"`` under CP, and that the full-causal
   KL normalises across the CP group on both the masked and the unmasked branch
   (``_indexer_loss_mask`` / the all-ones fallback, both of which divide by the
   **global** row count so the per-rank losses simply add up).

2. A layout with genuine **row-validity pad rows**. ``_STRADDLE`` sums to
   exactly ``S_GLOBAL`` and ``U._row_end`` folds any trailing gap into one final
   document, so ``is_valid`` has been all-``True`` in every CP test so far; the
   pad-row path (all-``-1`` index row -> zero output, zero ``dq``) was only ever
   audited on one card. ``[475] @ s=512`` puts 37 pad rows on the last rank
   only, which is also the pad-imbalance the loss denominator has to survive.

3. The way the **dense FA4** backend expresses the mask under CP. FA4 is the
   only backend the full-causal phases have, and its mask is an ``O(s)`` *row*
   bound rather than a column list, so localising it is a step the sparse phase-3
   path simply does not have: the kernel's own ``causal=True`` bottom-right-aligns
   the diagonal, which is wrong for every rank but the last, and the caller's
   bounds carry global row ids. ``MQALatentAttention._cp_row_bounds`` replaces the
   causal mode with an explicit second bound and shifts both into this rank's row
   space. Neither error raises, so ``TestDenseFA4CP`` carries the control that
   proves the checks can see an un-localised bound.

Everything reuses ``test_mqa_dsa_cp``'s harness (fleet init, CP globals -- which
includes its module-level ``FLAGS_flash_attn_version=4`` pin --, ``run_core_cp``,
``_check``, ``_check_index_sets``). No ``if rank == X`` short-circuit exists in
this file: every collective (``run_core_cp``'s all-reduces, the all-gather inside
the layer) is issued on all ranks and only the assertions are rank-conditional.

Note which observables exist on which phase. ``RecordingMQA`` captures a column
table for the phase-3 sparse kernel, and for the dense backend only at CP=1,
where it decodes the row bound it was handed; under CP the dense layer's bound is
localised and its values are local row ids, so no ``[b, s, s]`` table is an
honest reconstruction and ``idx_cp is None`` there. Dense CP claims are therefore
made against outputs, gradients and the bounds themselves.

Run (2 or 4 GPUs)::

    PYTHONPATH=./third_party/PaddleFleet/src:./third_party/PaddleFormers \
    python -m paddle.distributed.launch --devices 0,1 --nnodes 1 \
        --master 127.0.0.1:<port> \
        third_party/PaddleFleet/tests/multi_card_tests/transformer/\
test_mqa_dsa_warmup_cp.py
"""

import unittest
from unittest import mock

import numpy as np
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


def _bounds_column_sets(bounds, rows):
    """Per-local-row visible column set implied by a ``[LTS, UTE]`` pair.

    With ``causal=False`` and two bounds the kernel masks ``row >= LTS`` or
    ``row < UTE`` (``mask.py:513-518``), i.e. column ``j`` is visible exactly on
    rows ``[UTE_j, LTS_j)``. Inverting that per row is a decoding of the flashmask
    semantics, independent of how ``_cp_row_bounds`` computed the numbers, which
    is what makes it usable as the expectation's counterpart.
    """
    lts = bounds[0, 0, :, 0].tolist()
    ute = bounds[0, 0, :, 1].tolist()
    return [
        {j for j in range(len(lts)) if ute[j] <= r < lts[j]}
        for r in range(rows)
    ]


def _unlocalised_row_bounds(self, row_end, s_local):
    """``_cp_row_bounds`` without the ``preprocess_index`` shift -- the bug.

    Global ``[LTS, UTE]`` bounds compared against *local* row ids: the shapes
    still check out and the kernel still runs, which is what made the naive
    dense-under-CP call return silently wrong numbers. Used two ways -- decoded
    directly (``TestWarmupCP::test_2``) and patched over the real method
    (``TestDenseFA4CP::test_2``) -- to prove both levels of check can see it.
    """
    s_global = int(row_end.shape[2])
    causal_end = paddle.arange(s_global, dtype=row_end.dtype).reshape(
        [1, 1, s_global, 1]
    )
    return paddle.concat(
        [row_end, paddle.expand_as(causal_end, row_end)], axis=-1
    )


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

        This is what separates the warmup mode from phase 3: under
        ``window + top-k`` a row longer than ``window + index_topk`` selects a
        strict subset, so this assertion would fail there.

        ``idx`` is a ``[b, s_local, k]`` column table, which on the full-causal
        phases only the CP=1 reference produces (``RecordingMQA`` decodes the row
        bound it hands FA4); the CP layer's own set is checked through
        ``_cp_row_bounds`` instead (``TestWarmupCP::test_2``).
        """
        self.assertIsNotNone(
            idx, f"{tag}: no column table was captured, nothing to check"
        )
        off, rows = _local_slice()
        doc_start, is_valid = _doc_bounds(row_end, S_GLOBAL)
        for r in range(rows):
            q = off + r
            got = {int(c) for c in idx[0][r] if c >= 0}
            want = set(range(doc_start[q], q + 1)) if is_valid[q] else set()
            self.assertEqual(
                got, want, f"{tag}: row {r} (global {q}) is not full-causal"
            )

    def _check_dense_pad(self, res, tag):
        """Pad rows on the dense backend: zero output, zero ``dq``, no NaN.

        The column table expresses a pad row as an all-``-1`` row; dense has no
        table, so the same emptiness has to come out of the row bounds -- a pad
        row's ``LTS`` clips to its document end, which is at or below the row, so
        every column is masked. FA4 must then return an exactly zero row rather
        than a NaN from an empty softmax. Rank-conditional assertions only; the
        run itself is issued everywhere.
        """
        off, rows = _local_slice()
        _, is_valid = _doc_bounds(res["row_end"], S_GLOBAL)
        local_pad = [r for r in range(rows) if not is_valid[off + r]]
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
            worst = float(out[0, r].abs().max())
            self.assertEqual(
                worst,
                0.0,
                f"{tag}: pad row {r} (global {off + r}) output max|.|={worst}",
            )
            worst = float(dq[0, r].abs().max())
            self.assertEqual(
                worst,
                0.0,
                f"{tag}: pad row {r} (global {off + r}) dq max|.|={worst}",
            )
        print(
            f"[pad cp{H.CP_SIZE} rank{H.CP_RANK}] {tag}: "
            f"local_pad_rows={len(local_pad)} fwd={res['fwd']:.2e}",
            flush=True,
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
            f"this_rank={res['logged_cp']:.6e} sum(loss)={got:.6e} "
            f"ref={want:.6e} rel={rel:.3e}",
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


class TestWarmupCP(_CPChecks):
    """``mqa_dsa`` + ``dsa_indexer_use_sparse_loss=False`` under CP."""

    @H.U._GPU
    def test_1_warmup_forward_equivalence(self):
        """CP=N == CP=1 on the warmup path, with and without a live loss.

        ``dsa_indexer_loss_coeff == 0`` returns straight after the full-causal
        attention and never touches the indexer projections;
        ``> 0`` additionally runs the full-candidate indexer KL
        (``csa_indexer_topk_fwd`` with ``topk_effective=s_global``) over the
        whole causal set. Neither may perturb the attention output, so both are
        checked against the same reference.

        The attended column set is read off the CP=1 reference, which is the
        only side that can be decoded: dense FA4 is this phase's only backend and
        ``RecordingMQA`` only reconstructs its row bound at ``cp_size == 1``. The
        CP layer's own bound is checked in ``test_2``.
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
                self.assertIsNone(res["idx_cp"], f"{tag}: not dense")
                self._assert_full_causal(
                    res["idx_ref_slice"], res["row_end"], tag
                )

    @H.U._GPU
    def test_2_warmup_row_bounds_are_the_global_causal_set_localised(self):
        """``_cp_row_bounds`` selects exactly this rank's full causal set.

        The warmup's mask is the global per-document causal set restricted to
        this rank's rows, and on the dense backend it is carried by an ``O(s)``
        ``[LTS, UTE]`` pair rather than a column list. So decode the pair back
        into per-row column sets (``_bounds_column_sets``, which only knows the
        flashmask masking rule) and compare against the set the documents imply:
        for local row ``r``, global ``q = off + r``, that is
        ``[doc_start[q], q]`` -- and empty on a pad row.

        No collectives here: the bounds are a pure function of ``row_end``,
        ``cp_rank`` and ``s_local``, so the control can run unconditionally and
        the assertions need no rank gating beyond the one place localisation is
        provably a no-op.
        """
        off, rows = _local_slice()
        row_end = H.U._row_end(_STRADDLE, S_GLOBAL)
        doc_start, is_valid = _doc_bounds(row_end, S_GLOBAL)
        layer = H._build("mqa_dsa", H.CP_GROUP, sparse_loss=False)

        got = _bounds_column_sets(layer._cp_row_bounds(row_end, rows), rows)
        for r in range(rows):
            q = off + r
            want = set(range(doc_start[q], q + 1)) if is_valid[q] else set()
            self.assertEqual(
                got[r],
                want,
                f"local row {r} (global {q}) is visible over "
                f"{sorted(got[r])[:8]}..., expected {sorted(want)[:8]}...",
            )
        print(
            f"[warmup-cp{H.CP_SIZE} rank{H.CP_RANK}] row bounds decode to the "
            f"global causal set on all {rows} local rows",
            flush=True,
        )

        # Control: the same bounds without the ``preprocess_index`` shift, i.e.
        # global row ids compared against local ones. Rank 0's localisation is
        # ``clip(x, 0, s_local)``, and clipping an over-large bound masks the
        # same rows as leaving it over-large, so only rank > 0 can see this.
        if H.CP_RANK == 0:
            return
        bad = _bounds_column_sets(
            _unlocalised_row_bounds(layer, row_end, rows), rows
        )
        self.assertNotEqual(
            bad,
            got,
            "un-localised row bounds decoded to the same column sets, so this "
            "test cannot see a missing preprocess_index",
        )

    @H.U._GPU
    def test_3_warmup_equals_mqa_full_causal_under_cp(self):
        """Warmup output == ``hybrid_mla_attention="mqa_full_causal"`` output.

        Both modes reach the same ``_cp_row_bounds`` and then the same dense FA4
        kernel with the same inputs -- ``_forward_warmup`` reaches it *through*
        ``_forward_full_causal`` -- so on each rank this must be *bitwise* equal,
        not merely close. The single-card claim (maxabs 0.0) is asserted here
        with the CP row-slicing in the path, and it is also what pins the
        phase-2 attention set to the frozen backbone's phase-1 one.
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
        denominator itself rather than its shadow in the gradients. Phase 2
        (``sparse_loss=False``) has one formula on both branches: the row mask is
        always a real tensor -- ``_indexer_loss_mask``'s global-count mask when
        ``input_ids`` reached the layer, an all-ones mask over ``b * s_global``
        when it did not -- so every rank contributes its own rows to one global
        mean and the per-rank losses are partial sums. Phase 3
        (``sparse_loss=True``) still keeps the two-branch form (global count when
        masked, a local mean times ``1 / cp_size`` when not); both must land on
        the CP=1 value. ``sparse_loss`` is swept so a failure can be attributed:
        warmup-only means the full-causal KL broke it, both means the
        pre-existing normalisation is wrong.

        Note this layout does not exercise the *width* difference between the two
        ``sparse_loss`` values as sharply as it could: with documents of
        200/150/162 and a 128-wide forced window, most rows have few non-local
        candidates. ``test_5`` covers the width itself.
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
        """One 512-long document, so the two phases' KL supports differ widely.

        Phase 2's KL spans the whole per-document causal set (up to 512 columns
        on this layout) while phase 3's spans window(128) + top-k(128), so the
        two logged losses must differ -- that is the non-vacuity control for
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
        """``_check_dense_pad`` plus the column table's own emptiness.

        Phase 3 keeps a real ``[b, s, k]`` table, so a pad row must additionally
        show up there as an all-``-1`` row. The full-causal phases have no table
        (dense FA4 only), which is why the zero-output/zero-``dq`` half lives in
        ``_CPChecks`` on its own.
        """
        self._check_dense_pad(res, tag)
        off, rows = _local_slice()
        _, is_valid = _doc_bounds(res["row_end"], S_GLOBAL)
        for r in range(rows):
            if is_valid[off + r]:
                continue
            cols = res["idx_cp"][0][r]
            self.assertEqual(
                int((cols >= 0).sum()),
                0,
                f"{tag}: pad row {r} (global {off + r}) selected columns",
            )

    @H.U._GPU
    def test_1_pad_rows_mqa_full_causal(self):
        res = self._run("mqa", True)
        self._check(res, "pad/mqa_full_causal")
        self.assertIsNone(res["idx_cp"], "pad/mqa_full_causal: not dense")
        self._check_dense_pad(res, "pad/mqa_full_causal")

    @H.U._GPU
    def test_2_pad_rows_warmup(self):
        res = self._run("mqa_dsa", False)
        self._check(res, "pad/warmup")
        self.assertIsNone(res["idx_cp"], "pad/warmup: not dense")
        self._assert_full_causal(
            res["idx_ref_slice"], res["row_end"], "pad/warmup"
        )
        self._check_dense_pad(res, "pad/warmup")

    @H.U._GPU
    def test_3_pad_rows_sparse(self):
        """Same layout on the phase-3 (``window + top-k``) path.

        Included so a pad-row failure can be attributed: if it reproduces here
        it is not specific to the warmup branch this change introduced. This is
        also the only one of the three that still has a column table to check.
        """
        res = self._run("mqa_dsa", True)
        self._check(res, "pad/sparse")
        self._check_index_sets(res, "pad/sparse")
        self._check_pad(res, "pad/sparse")


class TestDenseFA4CP(_CPChecks):
    """The dense FA4 full-causal backend under CP -- the production path.

    Dense FA4 is the *only* backend the full-causal phases have, so every test
    in this file already runs it; what is left here is the part of the contract
    only this backend has. FA4's ``causal=True`` bottom-right-aligns the diagonal
    from the *shapes*, which under CP is right for the last rank alone, and the
    caller's row bounds carry global row ids. So ``_dense_attn`` turns the
    kernel's causal mode off and localises both bounds through
    ``preprocess_index`` (``MQALatentAttention._cp_row_bounds``). Neither error
    raises -- both return finite numbers -- hence ``test_2``'s control, which is
    the end-to-end counterpart of ``TestWarmupCP::test_2``'s decoding of the
    bounds themselves.
    """

    @H.U._GPU
    def test_1_dense_cp_equals_dense_cp1_with_a_sink(self):
        """The CP contract with the attention sink live, both full-causal phases.

        The sink is a per-head logit that FA4 takes as ``learnable_sink`` -- in
        bf16, which is why the fixture creates it in the module dtype -- and
        whose gradient is accumulated over this rank's rows only, so the group's
        SUM is what has to match (``param_err['softmax_offset']``). Sinkless
        equivalence on the same layout is ``TestWarmupCP::test_1`` and
        ``test_mqa_dsa_cp::test_2``.
        """
        sink = np.linspace(-1.0, 1.0, H.U.H)
        for mode, sparse_loss in (("mqa", True), ("mqa_dsa", False)):
            with self.subTest(mode=mode):
                tag = f"dense-sink/{mode}"
                res = H.run_core_cp(
                    mode,
                    _STRADDLE,
                    sparse_loss=sparse_loss,
                    sink=sink,
                    loss_coeff=0.1 if mode == "mqa_dsa" else 0.0,
                    with_input_ids=mode == "mqa_dsa",
                )
                self.assertIsNone(res["idx_cp"], f"{tag}: not dense")
                self._check(res, tag)
                self.assertIn(
                    "softmax_offset",
                    res["param_err"],
                    f"{tag}: the sink received no gradient",
                )

    @H.U._GPU
    def test_2_unlocalised_bounds_are_detectably_wrong(self):
        """Control: drop the ``preprocess_index`` shift and the checks must fail.

        Global bounds against local rows is the naive call, and it neither raises
        nor produces obviously broken output -- it is a plausible
        implementation. Every forward equivalence claim in this file is only
        worth something if it can see it.

        Observable on rank > 0 only: ``preprocess_index`` at ``chunk_id == 0`` is
        ``clip(x, 0, s_local)``, and clipping a bound to ``s_local`` masks
        exactly the same rows as leaving it above ``s_local``, so rank 0's
        localisation is a no-op by construction. The run itself happens on every
        rank -- it all-gathers the KV -- and only the assertion is
        rank-conditional.
        """
        good = H.run_core_cp("mqa", _STRADDLE)
        with mock.patch.object(
            MQALatentAttention, "_cp_row_bounds", _unlocalised_row_bounds
        ):
            bad = H.run_core_cp("mqa", _STRADDLE)
        print(
            f"[dense-cp{H.CP_SIZE} rank{H.CP_RANK}] control: localised "
            f"fwd={good['fwd']:.3e} unlocalised fwd={bad['fwd']:.3e}",
            flush=True,
        )
        if H.CP_RANK == 0:
            return
        self.assertGreater(
            bad["fwd"],
            100.0 * max(good["fwd"], 1e-6),
            "un-localised row bounds were indistinguishable from localised "
            f"ones (good={good['fwd']:.3e} bad={bad['fwd']:.3e}), so this "
            "file's forward equivalence cannot see a CP mask bug",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
